import argparse
import importlib
import json
import os
import random
import time
from collections import Counter, deque
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from scripts.games.common import event_random, frame_size, load_chunk
from scripts.model import Denoiser, sample, snap

batch_size = 256
horizons = [1, 10, 50, 100, 200, 500]
gif_rollouts = 4
gif_frames = 300
gif_scale = 8
hold_ms = 1000
marker_color = (255, 0, 0)
sampler_keys = ["sigma_aug", "sigma_max", "sigma_min", "rho"]
override_keys = ["sigma_max", "sigma_min", "rho"]


def pick_device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def to_model(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    # Frame k of the context fills channels 3k to 3k + 2, oldest first
    x = (
        torch.from_numpy(frames).to(device).permute(0, 1, 4, 2, 3)
    )  # shape: (batch, n, 3, 64, 64)

    return (
        x.flatten(start_dim=1, end_dim=2).float() / 127.5 - 1
    )  # shape: (batch, 3 * n, 64, 64)


def to_frames(x: torch.Tensor) -> np.ndarray:
    return (
        ((x + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    )  # shape: (batch, 64, 64, 3)


def next_frame(
    denoiser: Denoiser,
    context: torch.Tensor,
    actions: torch.Tensor,
    steps: int,
    sampler: dict,
    palette: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(
        (context.shape[0], context.shape[1] + 3, frame_size, frame_size),
        generator=generator,
        device=context.device,
    )  # shape: (batch, 3 * (n + 1), 64, 64)

    return snap(
        sample(denoiser, context, actions, noise, steps, **sampler), palette
    )  # shape: (batch, 3, 64, 64)


def match(generated: np.ndarray, real: np.ndarray) -> tuple[float, bool]:
    matches = np.all(generated == real, axis=-1)  # shape: (64, 64)

    return float(matches.mean()), bool(matches.all())


def one_step(
    generate: Callable,
    game: ModuleType,
    chunks: list,
    windows: list,
    n: int,
    device: torch.device,
) -> dict:
    pixels = []
    exact = []
    broken = Counter()
    passed = 0

    for start in tqdm(range(0, len(windows), batch_size), desc="One step"):
        batch = windows[start : start + batch_size]
        context = np.stack(
            [chunks[c]["frames"][t - n + 1 : t + 1] for c, t in batch]
        )  # shape: (batch, n, 64, 64, 3)
        actions = np.stack(
            [chunks[c]["actions"][t - n + 1 : t + 1] for c, t in batch]
        )  # shape: (batch, n)
        frames = to_frames(
            generate(
                to_model(context, device), torch.from_numpy(actions).long().to(device)
            )
        )

        for (c, t), real, action, frame in zip(
            batch, context, actions[:, -1], frames, strict=True
        ):
            # Random outcomes (food spawn, restart) have no single right frame
            if not chunks[c]["events"][t] & event_random:
                pixel, whole = match(frame, chunks[c]["frames"][t + 1])
                pixels.append(pixel)
                exact.append(whole)

            names, _ = game.check(list(map(game.extract, real)), int(action), frame)
            broken.update(names)
            passed += not names

    return {
        "transitions": len(windows),
        "deterministic": len(pixels),
        "pixel_accuracy": float(np.mean(pixels)),
        "frame_accuracy": float(np.mean(exact)),
        "rule_pass_rate": passed / len(windows),
        "broken": dict(broken),
    }


def open_loop(
    generate: Callable,
    chunks: list,
    starts: list,
    n: int,
    device: torch.device,
) -> dict:
    length = max(horizons)

    # A rollout ends where its chunk ends
    valid = np.array([min(length, len(chunks[c]["actions"]) - t) for c, t in starts])
    context = to_model(
        np.stack([chunks[c]["frames"][t - n + 1 : t + 1] for c, t in starts]), device
    )  # shape: (rollouts, 3 * n, 64, 64)
    pixels = np.full((len(starts), length), np.nan)
    exact = np.full((len(starts), length), np.nan)

    for i in tqdm(range(valid.max(initial=0)), desc="Open loop"):
        active = np.flatnonzero(valid > i)
        index = torch.from_numpy(active).to(device)
        actions = np.stack(
            [
                chunks[c]["actions"][t - n + 1 + i : t + 1 + i]
                for c, t in (starts[r] for r in active)
            ]
        )  # shape: (active, n)
        frame = generate(context[index], torch.from_numpy(actions).long().to(device))

        for j, (r, generated) in enumerate(zip(active, to_frames(frame), strict=True)):
            c, t = starts[r]
            real = chunks[c]["frames"][t + 1 + i]

            # A random draw has no single right outcome, so the real frame takes its place and the rollout goes on
            if chunks[c]["events"][t + i] & event_random:
                frame[j] = to_model(real[None, None], device)[0]
            else:
                pixels[r, i], exact[r, i] = match(generated, real)

        context[index] = torch.cat([context[index, 3:], frame], dim=1)

    results = {}

    for h in horizons:
        counted = ~np.isnan(pixels[:, h - 1])
        results[str(h)] = {
            "rollouts": int(counted.sum()),
            "pixel_accuracy": (
                float(pixels[counted, h - 1].mean()) if counted.any() else None
            ),
            "frame_accuracy": (
                float(exact[counted, h - 1].mean()) if counted.any() else None
            ),
        }

    return results


def save_gif(frames: np.ndarray, violation: bool, tick_hz: int, path: Path) -> None:
    # A 1 px border around every frame, red on the violation frame
    tiles = np.pad(
        frames, ((0, 0), (1, 1), (1, 1), (0, 0))
    )  # shape: (frames, 66, 66, 3)
    if violation:
        tiles[-1, [0, -1]] = marker_color
        tiles[-1, :, [0, -1]] = marker_color
    tiles = tiles.repeat(gif_scale, axis=1).repeat(gif_scale, axis=2)

    images = [Image.fromarray(tile) for tile in tiles]
    frame_ms = round(1000 / tick_hz)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=[frame_ms] * (len(images) - 1) + [hold_ms],
        loop=0,
    )


def dream(
    generate: Callable,
    game: ModuleType,
    chunks: list,
    starts: list,
    n: int,
    device: torch.device,
    horizon: int,
    epsilon: float,
    seed: int,
    gif_prefix: Path,
) -> dict:
    rng = random.Random(seed)
    real = np.stack(
        [chunks[c]["frames"][t - n + 1 : t + 1] for c, t in starts]
    )  # shape: (rollouts, n, 64, 64, 3)
    context = to_model(real, device)  # shape: (rollouts, 3 * n, 64, 64)

    # The last slot is overwritten by each step's new action
    actions = (
        torch.from_numpy(
            np.stack([chunks[c]["actions"][t - n + 1 : t + 1] for c, t in starts])
        )
        .long()
        .to(device)
    )  # shape: (rollouts, n)
    histories = [deque(map(game.extract, frames), maxlen=n) for frames in real]
    recordings = [deque(frames, maxlen=gif_frames) for frames in real[:gif_rollouts]]
    clean = np.full(len(starts), horizon)
    first = [[] for _ in starts]
    active = np.arange(len(starts))

    progress_bar = tqdm(range(horizon), desc="Dream play")

    for step in progress_bar:
        if not len(active):
            break

        new = [
            rng.choice(game.actions)
            if rng.random() < epsilon
            else game.policy(histories[r][-1], rng)
            for r in active
        ]
        index = torch.from_numpy(active).to(device)
        actions[index, -1] = torch.tensor(new, device=device)
        frame = generate(context[index], actions[index])
        context[index] = torch.cat([context[index, 3:], frame], dim=1)
        actions[index] = actions[index].roll(-1, dims=1)

        alive = []

        for r, action, generated in zip(active, new, to_frames(frame), strict=True):
            names, state = game.check(list(histories[r]), action, generated)
            if r < gif_rollouts:
                recordings[r].append(generated)
            if names:
                clean[r] = step
                first[r] = names
            else:
                histories[r].append(state)
                alive.append(r)

        active = np.array(alive, dtype=int)
        progress_bar.set_postfix(active=len(active))

    for r, frames in enumerate(recordings):
        path = Path(f"{gif_prefix}_{r}.gif")
        save_gif(np.stack(frames), bool(first[r]), game.tick_hz, path)

    return {
        "rollouts": len(starts),
        "horizon": horizon,
        "median": float(np.median(clean)),
        "p10": float(np.percentile(clean, 10)),
        "reached_horizon": float(np.mean(clean == horizon)),
        "first_broken": dict(Counter(name for names in first for name in names)),
        "clean_steps": clean.tolist(),
    }


def print_summary(metrics: dict) -> None:
    def cell(value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}"

    print(f"\nOne step\n{'K':>3} {'pixel':>8} {'frame':>8} {'rules':>8}  broken")

    for k, result in metrics["k"].items():
        row = result["one_step"]
        print(
            f"{k:>3} {cell(row['pixel_accuracy']):>8} {cell(row['frame_accuracy']):>8} "
            f"{cell(row['rule_pass_rate']):>8}  {row['broken']}"
        )

    print("\nOpen loop pixel accuracy at horizon (rollouts counted)")
    print(f"{'K':>3} " + " ".join(f"{h:>14}" for h in horizons))

    for k, result in metrics["k"].items():
        row = result["open_loop"]
        cells = [
            f"{cell(row[str(h)]['pixel_accuracy'])} ({row[str(h)]['rollouts']})"
            for h in horizons
        ]
        print(f"{k:>3} " + " ".join(f"{text:>14}" for text in cells))

    print(
        f"\nDream play, clean steps before the first violation\n"
        f"{'K':>3} {'median':>8} {'p10':>8} {'horizon':>8}  first broken"
    )

    for k, result in metrics["k"].items():
        row = result["dream"]
        print(
            f"{k:>3} {row['median']:>8.0f} {row['p10']:>8.0f} "
            f"{cell(row['reached_horizon']):>8}  {row['first_broken']}"
        )


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
        default=[1, 3],
        help="Euler step counts K to evaluate (default: 1 3).",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=2000,
        help="Held-out windows for the one-step rung (default: 2000).",
    )
    parser.add_argument(
        "--open-loop",
        type=int,
        default=50,
        help="Held-out starts for the open-loop rung (default: 50).",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=100,
        help="Parallel dream-play rollouts (default: 100).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5000,
        help="Maximum dream-play steps per rollout (default: 5000).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Chance of a random action in dream play (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for starts, actions and sampler noise (default: 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device; cuda, then mps, then cpu when unset (default: None).",
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
    args = parser.parse_args()

    started = time.time()
    device = pick_device(args.device)
    config = json.loads((args.run / "config.json").read_text())
    game = importlib.import_module(f"scripts.games.{config['game']}")
    n = config["context"]

    checkpoint = torch.load(args.run / args.ckpt, map_location=device)
    denoiser = Denoiser(config["width"], n, config["sigma_data"]).to(device)
    denoiser.load_state_dict(checkpoint["ema"])
    denoiser.eval()

    split = json.loads((args.run / "split.json").read_text())
    chunks = [load_chunk(Path(config["data_dir"]) / path) for path in split["heldout"]]
    windows = [
        (c, t)
        for c, chunk in enumerate(chunks)
        for t in range(n - 1, len(chunk["actions"]))
    ]
    rng = np.random.default_rng(args.seed)
    one_step_windows, open_loop_starts, dream_starts = [
        [
            windows[i]
            for i in rng.choice(len(windows), min(count, len(windows)), replace=False)
        ]
        for count in (args.windows, args.open_loop, args.rollouts)
    ]

    palette = (
        torch.tensor(game.palette, dtype=torch.float32, device=device) / 127.5 - 1
    )  # shape: (colors, 3)
    overrides = {
        key: vars(args)[key] for key in override_keys if vars(args)[key] is not None
    }
    sampler = {key: config[key] for key in sampler_keys} | overrides
    out_dir = args.run / f"eval_{checkpoint['step']}"

    # A tuned schedule gets its own folder, so the default-schedule results are never overwritten
    if overrides:
        out_dir = out_dir.with_name(
            f"{out_dir.name}_max{sampler['sigma_max']}_min{sampler['sigma_min']}_rho{sampler['rho']}"
        )
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
        "checkpoint": str(args.run / args.ckpt),
        "step": checkpoint["step"],
        "device": device.type,
        "args": vars(args),
        "sampler": sampler,
        "k": {},
    }

    with torch.inference_mode():
        for k in args.steps:
            # The same noise stream for every K
            generator = torch.Generator(device=device).manual_seed(args.seed)
            generate = partial(
                next_frame,
                denoiser,
                steps=k,
                sampler=sampler,
                palette=palette,
                generator=generator,
            )
            metrics["k"][str(k)] = {
                "one_step": one_step(
                    generate, game, chunks, one_step_windows, n, device
                ),
                "open_loop": open_loop(generate, chunks, open_loop_starts, n, device),
                "dream": dream(
                    generate,
                    game,
                    chunks,
                    dream_starts,
                    n,
                    device,
                    args.horizon,
                    args.epsilon,
                    args.seed,
                    out_dir / f"dream_k{k}",
                ),
            }

    metrics["seconds"] = time.time() - started
    with open(out_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2, default=str)

    print_summary(metrics)
    print(f"\n{metrics['seconds']:.1f} s, wrote {out_dir / 'metrics.json'}")
