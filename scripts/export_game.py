import argparse
import base64
import importlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate import override_keys, sampler_keys, to_frames, to_model
from scripts.export_onnx import FrameGraph, export, min_parity, random_inputs, run_onnx
from scripts.games.common import action_names, event_restart, frame_size, load_chunk
from scripts.model import Denoiser

# KeyboardEvent.code to action, mirroring scripts/games/play.py
key_codes = {
    "ArrowUp": 1,
    "KeyW": 1,
    "ArrowDown": 2,
    "KeyS": 2,
    "ArrowLeft": 3,
    "KeyA": 3,
    "ArrowRight": 4,
    "KeyD": 4,
    "Space": 5,
}


def encode(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Run folder runs/<game>/<name> (required).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="last.pt",
        help="Checkpoint file in the run folder (default: last.pt).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Euler steps K, one graph each; the first is the default and gets the parity check (default: the run's sampler_steps).",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=None,
        help="Sampler sigma_max; the run's config value when unset (default: None).",
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=None,
        help="Sampler sigma_min; the run's config value when unset (default: None).",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Sampler rho; the run's config value when unset (default: None).",
    )
    parser.add_argument(
        "--starts",
        type=int,
        default=64,
        help="Starting contexts to ship (default: 64).",
    )
    parser.add_argument(
        "--parity-cases",
        type=int,
        default=4,
        help="Parity cases, each from a different start (default: 4).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("web/public/models"),
        help="Output folder (default: web/public/models).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for starts, parity actions and noise (default: 0).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace this game's existing files (default: False).",
    )
    args = parser.parse_args()

    config = json.loads((args.run / "config.json").read_text())
    name = config["game"]
    game = importlib.import_module(f"scripts.games.{name}")
    n = config["context"]
    steps = [config["sampler_steps"]] if args.steps is None else args.steps
    overrides = {
        key: vars(args)[key] for key in override_keys if vars(args)[key] is not None
    }
    sampler = {key: config[key] for key in sampler_keys} | overrides

    model_paths = [args.out_dir / f"{name}_k{k}.onnx" for k in steps]
    model_path = model_paths[0]
    meta_path = args.out_dir / f"{name}.json"
    starts_path = args.out_dir / f"{name}_starts.bin"
    parity_path = args.out_dir / f"{name}_parity.json"
    existing = [
        str(path)
        for path in (*model_paths, meta_path, starts_path, parity_path)
        if path.exists()
    ]
    if existing and not args.force:
        raise FileExistsError(
            f"{', '.join(existing)} already exist; pass --force to replace them"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint = torch.load(args.run / args.ckpt, map_location="cpu")
    denoiser = Denoiser(config["width"], n, config["sigma_data"])
    denoiser.load_state_dict(checkpoint["ema"])
    denoiser.eval()
    palette = (
        torch.tensor(game.palette, dtype=torch.float32) / 127.5 - 1
    )  # shape: (colors, 3)
    graphs = [FrameGraph(denoiser, k, palette, **sampler).eval() for k in steps]
    inputs = random_inputs(n, torch.Generator().manual_seed(args.seed))

    for graph, path in zip(graphs, model_paths, strict=True):
        model = export(graph, inputs, path)
        ops = Counter(node.op_type for node in model.graph.node)
        print(
            f"{path.name} ops: {', '.join(f'{op} {count}' for op, count in ops.most_common())}"
        )

    graph = graphs[0]

    # A start is a fresh game: frame t + 1 after a restart at step t, and no event on the N - 1 steps inside the context
    split = json.loads((args.run / "split.json").read_text())
    chunks = [load_chunk(Path(config["data_dir"]) / path) for path in split["heldout"]]
    candidates = [
        (c, t + 1)
        for c, chunk in enumerate(chunks)
        for t in np.flatnonzero(chunk["events"] & event_restart)
        if t + n < len(chunk["frames"]) and not chunk["events"][t + 1 : t + n].any()
    ]
    rng = np.random.default_rng(args.seed)
    starts = [
        candidates[i] for i in rng.choice(len(candidates), args.starts, replace=False)
    ]
    frames = np.stack(
        [chunks[c]["frames"][s : s + n] for c, s in starts]
    )  # shape: (starts, n, 64, 64, 3)
    actions = np.stack(
        [chunks[c]["actions"][s : s + n - 1] for c, s in starts]
    )  # shape: (starts, n - 1)

    for frame in frames.reshape(-1, frame_size, frame_size, 3):
        game.extract(frame)

    with open(starts_path, "wb") as file:
        file.write(frames.tobytes() + actions.tobytes())

    cases = []

    for i in range(args.parity_cases):
        case_actions = [*actions[i].tolist(), int(rng.choice(game.actions))]
        noise = rng.standard_normal(
            (3 * (n + 1), frame_size, frame_size), dtype=np.float32
        )
        inputs = (
            to_model(frames[i][None], torch.device("cpu")),
            torch.tensor([case_actions]),
            torch.from_numpy(noise)[None],
        )

        with torch.inference_mode():
            expected = to_frames(graph(*inputs))[0]  # shape: (64, 64, 3)

        actual = to_frames(torch.from_numpy(run_onnx(model_path, inputs)))[0]
        identical = np.all(actual == expected, axis=-1).mean()
        print(f"parity case {i}: {identical:.6f} identical pixels")

        if identical < min_parity:
            raise RuntimeError(
                f"{model_path} parity case {i}: only {identical:.6f} of pixels match PyTorch, need {min_parity}"
            )

        cases.append(
            {
                "frames": encode(frames[i]),
                "actions": case_actions,
                "noise": encode(noise.astype("<f4")),
                "expected": encode(expected),
            }
        )

    with open(parity_path, "w") as file:
        json.dump({"min_identical": min_parity, "cases": cases}, file)

    meta = {
        "game": name,
        "model": model_path.name,
        "tick_hz": game.tick_hz,
        "context": n,
        "actions": game.actions,
        "action_names": [action_names[action] for action in game.actions],
        "input_mode": game.input_mode,
        "keys": {
            code: action for code, action in key_codes.items() if action in game.actions
        },
        "palette": game.palette,
        "sampler": {"steps": steps[0], **sampler},
        "models": [
            {"steps": k, "file": path.name, "bytes": path.stat().st_size}
            for k, path in zip(steps, model_paths, strict=True)
        ],
        "starts": {"file": starts_path.name, "count": len(starts)},
        "parity": parity_path.name,
        "run": str(args.run),
        "step": checkpoint["step"],
    }
    with open(meta_path, "w") as file:
        json.dump(meta, file, indent=2)

    # Merge by game, so exporting one game keeps the others in their picker order
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    entry = {"game": name, "title": name.capitalize(), "meta": meta_path.name}
    games = [item["game"] for item in manifest]
    if name in games:
        manifest[games.index(name)] = entry
    else:
        manifest.append(entry)

    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2)

    for path in (*model_paths, meta_path, starts_path, parity_path, manifest_path):
        print(f"{path}: {path.stat().st_size / 1e6:.3f} MB")
