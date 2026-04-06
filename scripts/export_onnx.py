import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from scripts.model import Denoiser, n_actions, sample, snap

# Placeholder palette for the latency spike: 7 corners of the RGB cube
palette = [
    (0, 0, 0),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
]
sigma_aug_star = 0.0
frame_size = 64
opset = 18
reinit_std = 0.02
min_parity = 0.999
input_names = ["context", "actions", "noise"]


class FrameGraph(nn.Module):
    # Sampler settings (sigma_aug, sigma_max, sigma_min, rho) go to sample() and are baked into the graph
    def __init__(
        self,
        denoiser: Denoiser,
        steps: int,
        palette: torch.Tensor | None,
        **sampler: float,
    ) -> None:
        super().__init__()

        self.denoiser = denoiser
        self.steps = steps
        self.sampler = sampler
        self.register_buffer("palette", palette)

    def forward(
        self, context: torch.Tensor, actions: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        frame = sample(
            self.denoiser, context, actions, noise, self.steps, **self.sampler
        )  # shape: (1, 3, 64, 64)

        return frame if self.palette is None else snap(frame, self.palette)


def palette_tensor() -> torch.Tensor:
    return torch.tensor(palette, dtype=torch.float32) / 127.5 - 1  # shape: (colors, 3)


def randomize_zero_init(denoiser: Denoiser) -> None:
    # Random weights in the zero-initialized head and AdaGN layers make the output depend on the whole network
    for module in denoiser.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and not module.weight.any():
            nn.init.normal_(module.weight, std=reinit_std)
            nn.init.normal_(module.bias, std=reinit_std)


def random_inputs(
    context: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = (
        torch.rand(1, 3 * context, frame_size, frame_size, generator=generator) * 2 - 1
    )
    actions = torch.randint(0, n_actions, (1, context), generator=generator)
    noise = torch.randn(
        1, 3 * (context + 1), frame_size, frame_size, generator=generator
    )

    return frames, actions, noise


def export(
    graph: FrameGraph, inputs: tuple[torch.Tensor, ...], path: Path
) -> onnx.ModelProto:
    torch.onnx.export(
        graph,
        inputs,
        path,
        input_names=input_names,
        output_names=["frame"],
        opset_version=opset,
        dynamo=True,
        external_data=False,
        verbose=False,
    )

    # Drop the exporter's per-node debug metadata (stack traces), about 1 MB per denoiser call
    model = onnx.load(path)

    for node in model.graph.node:
        del node.metadata_props[:]

    if any(node.op_type == "GroupNormalization" for node in model.graph.node):
        raise RuntimeError(
            f"{path} contains GroupNormalization, which has no WebGPU kernel in onnxruntime-web"
        )

    onnx.save(model, path)
    return model


def run_onnx(path: Path, inputs: tuple[torch.Tensor, ...]) -> np.ndarray:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    return session.run(
        None,
        {
            name: tensor.numpy()
            for name, tensor in zip(input_names, inputs, strict=True)
        },
    )[0]


def count_gflops(denoiser: Denoiser, context: int) -> float:
    frames, actions, noise = random_inputs(context, torch.Generator().manual_seed(0))
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        denoiser(noise[:, :3], torch.ones(1), frames, actions, torch.zeros(1))

    return counter.get_total_flops() / 1e9


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[16, 24, 32],
        help="UNet base widths (default: 16 24 32).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 3],
        help="Euler step counts K (default: 1 3).",
    )
    parser.add_argument(
        "--context", type=int, default=4, help="Context frames N (default: 4)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("web/public/models/bench"),
        help="Output folder (default: web/public/models/bench).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for weights and test inputs (default: 0).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)

    manifest = []
    for width in args.widths:
        torch.manual_seed(args.seed)
        denoiser = Denoiser(width, args.context)
        randomize_zero_init(denoiser)
        denoiser.eval()
        params = sum(p.numel() for p in denoiser.parameters())
        gflops = count_gflops(denoiser, args.context)

        for steps in args.steps:
            name = f"w{width}_k{steps}"
            path = args.out_dir / f"{name}.onnx"
            graph = FrameGraph(
                denoiser, steps, palette_tensor(), sigma_aug=sigma_aug_star
            ).eval()
            inputs = random_inputs(args.context, generator)
            model = export(graph, inputs, path)

            ops = Counter(node.op_type for node in model.graph.node)
            print(
                f"{name}: {params:,} params, {gflops:.3f} GFLOPs per call, {path.stat().st_size / 1e6:.2f} MB"
            )
            print(
                f"  ops: {', '.join(f'{op} {count}' for op, count in ops.most_common())}"
            )

            # Fresh inputs, so parity is not measured on the tracing inputs
            inputs = random_inputs(args.context, generator)

            with torch.inference_mode():
                expected = graph(*inputs).numpy()

            actual = run_onnx(path, inputs)
            parity = (expected == actual).all(axis=1).mean()
            colors_used = np.unique(expected.reshape(3, -1), axis=1).shape[1]
            print(
                f"  parity: {parity:.6f} identical pixels, {colors_used} palette colors used"
            )

            if parity < min_parity:
                raise RuntimeError(
                    f"{path}: only {parity:.6f} of pixels match PyTorch, need {min_parity}"
                )

            manifest.append(
                {
                    "name": name,
                    "file": path.name,
                    "width": width,
                    "steps": steps,
                    "context": args.context,
                    "params": params,
                    "gflops": round(gflops, 4),
                }
            )

    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest)} graphs and manifest.json to {args.out_dir}")
