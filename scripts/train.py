import argparse
import copy
import importlib
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from scripts.games.common import frame_size, load_chunk
from scripts.model import Denoiser, edm_loss, sample, sigma_aug_max, snap

sources = ["random", "scripted", "human"]
heldout_share = 0.05
heldout_batches = 8
sampler_steps = 3
sigma_max = 5.0
sigma_min = 0.002
rho = 7.0
sigma_aug = 0.0
warmup_steps = 1000
betas = (0.9, 0.99)
weight_decay = 1e-2
grad_clip = 1.0
ema_decay = 0.999
log_every = 100
bucket_names = ["low", "mid", "high"]
bucket_edges = (0.1, 1.0)
rollout_steps = 200
rollout_ks = (1, 3)
gif_scale = 4
gif_gap = 2
gap_color = 255


def split_chunks(data: Path, seed: int) -> dict:
    rng = random.Random(seed)
    split = {"train": [], "heldout": []}

    for source in sources:
        paths = sorted((data / source).glob("*.npz"))
        n_heldout = max(round(heldout_share * len(paths)), 1) if len(paths) >= 2 else 0
        heldout = rng.sample(paths, n_heldout)
        split["train"] += [str(p.relative_to(data)) for p in paths if p not in heldout]
        split["heldout"] += [str(p.relative_to(data)) for p in sorted(heldout)]

    return split


def load_frames(paths: list[Path], context: int, device: torch.device) -> dict:
    chunks = [load_chunk(path) for path in paths]
    total = sum(len(chunk["frames"]) for chunk in chunks)
    frames = torch.empty(
        (total, frame_size, frame_size, 3), dtype=torch.uint8, device=device
    )

    # One padding entry per chunk lines actions and events up with frames, so one index addresses all three
    actions = torch.zeros(total, dtype=torch.uint8, device=device)
    events = torch.zeros(total, dtype=torch.uint8, device=device)
    starts = []
    histogram = np.zeros(256, dtype=np.int64)
    offset = 0

    for chunk in chunks:
        n_steps = len(chunk["actions"])
        end = offset + n_steps
        frames[offset : end + 1] = torch.from_numpy(chunk["frames"]).to(device)
        actions[offset:end] = torch.from_numpy(chunk["actions"]).to(device)
        events[offset:end] = torch.from_numpy(chunk["events"]).to(device)

        # Windows start at t - N + 1 for t in [N - 1, L - 1], so none crosses a chunk boundary
        starts.append(np.arange(offset, end - context + 1))
        histogram += np.bincount(chunk["frames"].ravel(), minlength=256)
        offset = end + 1

    return {
        "frames": frames,
        "actions": actions,
        "events": events,
        "starts": torch.from_numpy(np.concatenate(starts)).to(device),
        "histogram": histogram,
    }


# EDM assumes zero-mean data, but frames are mostly dark background, so the root mean square keeps the network target at unit variance
def pixel_rms(histogram: np.ndarray) -> float:
    values = np.arange(len(histogram)) / 127.5 - 1

    return float(np.sqrt((histogram * values**2).sum() / histogram.sum()))


def to_pixels(frames: torch.Tensor) -> torch.Tensor:
    # Frames stack along channels, oldest first, as in the ONNX graph contract
    return (
        frames.permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2).float() / 127.5
        - 1
    )  # shape: (batch, 3 * frames, 64, 64)


def get_batch(
    data: dict,
    windows: torch.Tensor,
    context: int,
    memory_format: torch.memory_format,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = data["starts"][windows, None] + torch.arange(
        context + 1, device=windows.device
    )  # shape: (batch, context + 1)
    pixels = to_pixels(
        data["frames"][index]
    )  # shape: (batch, 3 * (context + 1), 64, 64)
    actions = data["actions"][index[:, :context]].long()  # shape: (batch, context)

    return (
        pixels[:, : 3 * context].contiguous(memory_format=memory_format),
        pixels[:, 3 * context :].contiguous(memory_format=memory_format),
        actions,
    )


def bucket_stats(losses: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    low, high = bucket_edges
    masks = torch.stack(
        [sigma < low, (sigma >= low) & (sigma <= high), sigma > high]
    ).float()  # shape: (buckets, batch)

    # Loss sums per bucket, then sample counts per bucket
    return torch.cat([masks @ losses, masks.sum(dim=1)])  # shape: (2 * buckets,)


def bucket_means(stats: list[float], prefix: str = "") -> dict:
    sums, counts = stats[: len(bucket_names)], stats[len(bucket_names) :]
    means = {f"{prefix}loss": sum(sums) / sum(counts)}

    for name, total, count in zip(bucket_names, sums, counts, strict=True):
        means[f"{prefix}loss_{name}"] = total / count if count else None

    return means


@torch.no_grad()
def update_ema(ema: Denoiser, model: Denoiser) -> None:
    for ema_param, param in zip(ema.parameters(), model.parameters(), strict=True):
        ema_param.lerp_(param, 1 - ema_decay)


def heldout_loss(
    ema: Denoiser,
    heldout: dict,
    windows: torch.Tensor,
    config: dict,
    memory_format: torch.memory_format,
) -> dict:
    ema.eval()

    # The same seed every time draws the same noise, so held-out losses are comparable across steps
    generator = torch.Generator(device=windows.device).manual_seed(config["seed"])

    with torch.inference_mode():
        stats = torch.zeros(2 * len(bucket_names), device=windows.device)

        for batch_windows in windows.split(config["batch"]):
            context, target, actions = get_batch(
                heldout, batch_windows, config["context"], memory_format
            )
            stats += bucket_stats(*edm_loss(ema, target, context, actions, generator))

    return bucket_means(stats.tolist(), prefix="heldout_")


def rollout(
    denoiser: Denoiser,
    context: torch.Tensor,
    actions: torch.Tensor,
    noise: torch.Tensor,
    palette: torch.Tensor,
    config: dict,
    steps: int,
) -> np.ndarray:
    n = context.shape[1] // 3
    frames = []

    for i in range(len(noise)):
        frame = snap(
            sample(
                denoiser,
                context,
                actions[None, i : i + n],
                noise[i : i + 1],
                steps=steps,
                sigma_aug=config["sigma_aug"],
                sigma_max=config["sigma_max"],
                sigma_min=config["sigma_min"],
                rho=config["rho"],
            ),
            palette,
        )  # shape: (1, 3, 64, 64)
        frames.append(frame)
        context = torch.cat([context[:, 3:], frame], dim=1)

    pixels = ((torch.cat(frames) + 1) * 127.5).round().to(torch.uint8)

    return pixels.permute(0, 2, 3, 1).cpu().numpy()  # shape: (steps, 64, 64, 3)


def write_gif(path: Path, panels: list[np.ndarray], tick_hz: float) -> None:
    gap = np.full((*panels[0].shape[:2], gif_gap, 3), gap_color, dtype=np.uint8)
    columns = [panels[0]]

    for panel in panels[1:]:
        columns += [gap, panel]

    strips = (
        np.concatenate(columns, axis=2)
        .repeat(gif_scale, axis=1)
        .repeat(gif_scale, axis=2)
    )  # shape: (steps, 64 * scale, panel widths * scale, 3)
    images = [Image.fromarray(strip) for strip in strips]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / tick_hz),
        loop=0,
    )


def save_checkpoint(state: dict, paths: list[Path]) -> None:
    for path in paths:
        # Write then rename, so a job killed mid-save never leaves a truncated last.pt
        partial = path.with_name(f"{path.name}.partial")
        torch.save(state, partial)
        os.replace(partial, path)


def log_metrics(path: Path, record: dict) -> None:
    with open(path, "a") as file:
        file.write(f"{json.dumps(record)}\n")


def peak_memory(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9

    # MPS has no peak counter, so this is what the driver holds right now
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--game",
        type=str,
        default="snake",
        help="Game module in scripts/games (default: snake).",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run name; the run folder is <runs-dir>/<game>/<name> (default: none, required).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200_000,
        help="Total optimizer steps; with --resume, the new total (default: 200000).",
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="Batch size (default: 64)."
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="Peak learning rate (default: 0.0002)."
    )
    parser.add_argument(
        "--width", type=int, default=32, help="UNet base width (default: 32)."
    )
    parser.add_argument(
        "--w-event",
        type=float,
        default=4.0,
        help="Sampling weight of windows whose transition has an event flag (default: 4.0).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5000,
        help="Steps between held-out evaluations, checkpoints and sample GIFs (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the split, initialization and sampling (default: 0).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Chunk folder with random, scripted and human subfolders (default: data/<game>).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Folder that holds runs (default: runs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device; unset picks cuda, then mps, then cpu (default: None).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the loss and model with torch.compile (default: False).",
    )
    existing_run = parser.add_mutually_exclusive_group()
    existing_run.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last.pt; settings other than --steps come from the saved config (default: False).",
    )
    existing_run.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing run folder (default: False).",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    run_dir = args.runs_dir / args.game / args.name

    if args.resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location=device)
        config = checkpoint["config"] | {"train_steps": args.steps}
        with open(run_dir / "split.json") as file:
            split = json.load(file)
    else:
        if run_dir.exists() and not args.force:
            raise FileExistsError(
                f"{run_dir} already exists; pass --resume to continue it or --force to replace it"
            )
        if run_dir.exists():
            shutil.rmtree(run_dir)

        checkpoint = None
        data = args.data or Path("data") / args.game
        split = split_chunks(data, args.seed)
        config = {
            "game": args.game,
            "width": args.width,
            "context": importlib.import_module(f"scripts.games.{args.game}").context,
            "train_steps": args.steps,
            "batch": args.batch,
            "lr": args.lr,
            "w_event": args.w_event,
            "seed": args.seed,
            "data_dir": str(data),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "sampler_steps": sampler_steps,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "rho": rho,
            "sigma_aug_max": sigma_aug_max,
            "sigma_aug": sigma_aug,
        }

    if not split["heldout"]:
        raise ValueError(
            f"No held-out chunks in {config['data_dir']}: a source needs at least two chunks to hold one out"
        )

    module = importlib.import_module(f"scripts.games.{config['game']}")
    data = Path(config["data_dir"])
    n = config["context"]
    train = load_frames([data / path for path in split["train"]], n, device)
    heldout = load_frames([data / path for path in split["heldout"]], n, device)

    # Measured once from the training pixels; a resumed run keeps its value
    config.setdefault("sigma_data", pixel_rms(train["histogram"]))

    os.makedirs(run_dir / "samples", exist_ok=True)
    with open(run_dir / "config.json", "w") as file:
        json.dump(config, file, indent=2)
    with open(run_dir / "split.json", "w") as file:
        json.dump(split, file, indent=2)

    start_step = checkpoint["step"] if args.resume else 0

    # A resumed run draws a fresh data stream instead of replaying its first steps
    torch.manual_seed(config["seed"] + start_step)
    memory_format = (
        torch.channels_last if device.type == "cuda" else torch.contiguous_format
    )
    model = Denoiser(config["width"], n, config["sigma_data"]).to(
        device, memory_format=memory_format
    )
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], betas=betas, weight_decay=weight_decay
    )

    if args.resume:
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    loss_fn = torch.compile(edm_loss) if args.compile else edm_loss
    weights = torch.where(
        train["events"][train["starts"] + n - 1] != 0, config["w_event"], 1.0
    )  # shape: (windows,)

    # Fixed held-out windows and rollout noise keep evaluations comparable across steps
    generator = torch.Generator().manual_seed(config["seed"])
    heldout_windows = torch.randperm(len(heldout["starts"]), generator=generator)[
        : heldout_batches * config["batch"]
    ].to(device)

    # The last held-out chunk prefers human, then scripted play, where the snake eats and grows
    chunk = load_chunk(data / split["heldout"][-1])
    horizon = min(rollout_steps, len(chunk["actions"]) - n + 1)
    rollout_context = to_pixels(torch.from_numpy(chunk["frames"][None, :n])).to(
        device
    )  # shape: (1, 3 * context, 64, 64)
    rollout_actions = (
        torch.from_numpy(chunk["actions"][: horizon + n - 1]).long().to(device)
    )
    rollout_noise = torch.randn(
        (horizon, 3 * (n + 1), frame_size, frame_size), generator=generator
    ).to(device)
    real = chunk["frames"][n : n + horizon]
    palette = (torch.tensor(module.palette, dtype=torch.float32) / 127.5 - 1).to(
        device
    )  # shape: (colors, 3)

    metrics_path = run_dir / "metrics.jsonl"
    window_stats = torch.zeros(2 * len(bucket_names), device=device)
    last_step, last_time = start_step, time.perf_counter()
    progress_bar = tqdm(
        range(start_step + 1, config["train_steps"] + 1),
        desc="Training",
        initial=start_step,
        total=config["train_steps"],
    )
    model.train()

    for step in progress_bar:
        lr = config["lr"] * min(1.0, step / warmup_steps)

        for group in optimizer.param_groups:
            group["lr"] = lr

        windows = torch.multinomial(weights, config["batch"], replacement=True)
        context, target, actions = get_batch(train, windows, n, memory_format)

        optimizer.zero_grad()
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            losses, sigma = loss_fn(model, target, context, actions)
        loss = losses.mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        update_ema(ema, model)

        loss_value = loss.item()
        window_stats += bucket_stats(losses.detach(), sigma)
        progress_bar.set_postfix(
            loss=f"{loss_value:.6f}", lr=f"{lr:.2e}", refresh=False
        )

        if step % log_every == 0 or step == config["train_steps"]:
            now = time.perf_counter()
            log_metrics(
                metrics_path,
                {
                    "step": step,
                    **bucket_means(window_stats.tolist()),
                    "lr": lr,
                    "grad_norm": grad_norm.item(),
                    "steps_per_sec": (step - last_step) / (now - last_time),
                    "peak_memory_gb": peak_memory(device),
                },
            )
            window_stats.zero_()
            last_step, last_time = step, now

        if step % args.eval_every == 0 or step == config["train_steps"]:
            log_metrics(
                metrics_path,
                {
                    "step": step,
                    **heldout_loss(
                        ema, heldout, heldout_windows, config, memory_format
                    ),
                },
            )
            save_checkpoint(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                },
                [run_dir / f"ckpt_{step}.pt", run_dir / "last.pt"],
            )

            with torch.inference_mode():
                generated = [
                    rollout(
                        ema,
                        rollout_context,
                        rollout_actions,
                        rollout_noise,
                        palette,
                        config,
                        k,
                    )
                    for k in rollout_ks
                ]

            write_gif(
                run_dir / "samples" / f"step_{step}.gif",
                [real, *generated],
                module.tick_hz,
            )
            model.train()
            last_time = time.perf_counter()

    print(f"Finished {config['train_steps']} steps in {run_dir}")
