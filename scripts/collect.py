import argparse
import importlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from scripts.games.common import (
    event_death,
    event_random,
    event_restart,
    event_score,
    frame_size,
    load_chunk,
    save_chunk,
)

# Share of generated chunks that are random: the plan's 25 / 65 frame mix, with human play as the other 10%
random_share = 0.28
sticky = 0.75
max_epsilon = 0.3
ramp_share = 0.5
ramp_epsilon = 0.5
min_ramp_score = 5
max_ramp_score = 60
preview_frames = 300
preview_scale = 8
sources = ["random", "scripted"]
event_names = {
    "score": event_score,
    "death": event_death,
    "restart": event_restart,
    "random": event_random,
}
snake_targets = {
    "max length": 40,
    "food events": 20_000,
    "wall deaths": 5_000,
    "self deaths": 1_000,
}


# Half the episodes turn reckless past a random score, so deaths also happen in long, late-game states
def episode_noise(rng: random.Random) -> tuple[float, float]:
    ramp_at = math.inf

    if rng.random() < ramp_share:
        ramp_at = rng.randint(min_ramp_score, max_ramp_score)

    return rng.uniform(0, max_epsilon), ramp_at


def collect_chunk(job: dict) -> dict:
    module = importlib.import_module(f"scripts.games.{job['game']}")
    game = module.Game(job["game_seed"])
    rng = random.Random(job["policy_seed"])
    steps = job["steps"]

    frames = np.empty((steps + 1, frame_size, frame_size, 3), dtype=np.uint8)
    actions = np.empty(steps, dtype=np.uint8)
    events = np.empty(steps, dtype=np.uint8)
    frames[0] = game.render()
    state = game.state()
    action = rng.choice(module.actions)
    epsilon, ramp_at = episode_noise(rng)
    causes = Counter()
    lengths = Counter()

    for t in range(steps):
        if job["source"] == "random":
            if rng.random() >= sticky:
                action = rng.choice(module.actions)
        elif rng.random() < (ramp_epsilon if state["score"] >= ramp_at else epsilon):
            action = rng.choice(module.actions)
        else:
            action = module.policy(state, rng)

        flags = game.step(action)
        state = game.state()
        frames[t + 1] = game.render()
        actions[t] = action
        events[t] = flags
        lengths[state["length"]] += 1

        if flags & event_death:
            causes[state["cause"]] += 1

        # Each episode gets its own noise level
        if flags & event_restart:
            epsilon, ramp_at = episode_noise(rng)

    save_chunk(job["path"], frames, actions, events, job["game_seed"])
    counts = {
        name: int(np.count_nonzero(events & flag)) for name, flag in event_names.items()
    }

    return {
        **job,
        "events": counts,
        "episodes": counts["restart"] + 1,
        "causes": causes,
        "lengths": lengths,
    }


def collect(
    game: str, steps: int, workers: int, chunk_steps: int, seed: int, out: Path
) -> list[dict]:
    n_chunks = math.ceil(steps / chunk_steps)
    n_random = round(n_chunks * random_share)
    seeds = np.random.SeedSequence(seed).generate_state(2 * n_chunks).reshape(-1, 2)

    jobs = []

    for i in range(n_chunks):
        source = sources[0] if i < n_random else sources[1]
        jobs.append(
            {
                "game": game,
                "source": source,
                "steps": min(chunk_steps, steps - i * chunk_steps),
                "game_seed": int(seeds[i, 0]),
                "policy_seed": int(seeds[i, 1]),
                "path": out / source / f"chunk_{i % workers}_{i // workers}.npz",
            }
        )

    results = []
    with (
        Pool(workers) as pool,
        tqdm(total=steps, desc="Collecting", unit="step") as progress_bar,
    ):
        for result in pool.imap(collect_chunk, jobs):
            results.append(result)
            progress_bar.update(result["steps"])

    return results


def summarize(results: list[dict]) -> dict:
    causes = sum((result["causes"] for result in results), Counter())
    lengths = sum((result["lengths"] for result in results), Counter())

    return {
        "steps": sum(result["steps"] for result in results),
        "episodes": sum(result["episodes"] for result in results),
        "events": {
            name: sum(result["events"][name] for result in results)
            for name in event_names
        },
        "deaths_by_cause": dict(causes),
        "max_length": max(lengths, default=0),
        "length_histogram": dict(sorted(lengths.items())),
    }


def write_gif(chunk_path: Path, start: int, path: Path, tick_hz: float) -> None:
    frames = load_chunk(chunk_path)["frames"][start : start + preview_frames]
    frames = frames.repeat(preview_scale, axis=1).repeat(
        preview_scale, axis=2
    )  # shape: (preview_frames, 512, 512, 3)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / tick_hz),
        loop=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--game",
        type=str,
        default="snake",
        help="Game module in scripts/games (default: snake).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500_000,
        help="Transitions to collect (default: 500000).",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Worker processes (default: 8)."
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=10_000,
        help="Transitions per chunk (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed that every chunk seed derives from (default: 0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/snake"),
        help="Output folder (default: data/snake).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=4,
        help="Preview GIFs to write (default: 4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing random and scripted chunks (default: False).",
    )
    args = parser.parse_args()

    existing = [
        path for source in sources for path in (args.out / source).glob("*.npz")
    ]
    if existing and not args.force:
        raise FileExistsError(
            f"{args.out} already holds {len(existing)} random or scripted chunks; pass --force to replace them"
        )

    for path in existing:
        path.unlink()

    module = importlib.import_module(f"scripts.games.{args.game}")
    start = time.perf_counter()
    results = collect(
        args.game, args.steps, args.workers, args.chunk_steps, args.seed, args.out
    )
    elapsed = time.perf_counter() - start
    print(
        f"Collected {args.steps} steps in {elapsed:.1f} s ({args.steps / elapsed:.0f} steps per second)"
    )

    coverage = {
        source: summarize([result for result in results if result["source"] == source])
        for source in sources
    }
    coverage["all"] = summarize(results)
    human_steps = sum(
        len(load_chunk(path)["actions"]) for path in (args.out / "human").glob("*.npz")
    )
    meta = {
        "game": args.game,
        "palette": module.palette,
        "tick_hz": module.tick_hz,
        "context": module.context,
        "seed": args.seed,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "steps": {
            **{source: coverage[source]["steps"] for source in sources},
            "human": human_steps,
        },
        "coverage": coverage,
        "chunks": [
            {
                "path": str(result["path"].relative_to(args.out)),
                "source": result["source"],
                "steps": result["steps"],
                "game_seed": result["game_seed"],
                "policy_seed": result["policy_seed"],
            }
            for result in results
        ],
    }
    with open(args.out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    os.makedirs(args.out / "preview", exist_ok=True)
    rng = random.Random(args.seed)

    for i in range(args.preview):
        result = rng.choice(results)
        offset = rng.randrange(max(result["steps"] + 1 - preview_frames, 0) + 1)
        write_gif(
            result["path"],
            offset,
            args.out / "preview" / f"preview_{i}.gif",
            module.tick_hz,
        )

    total = coverage["all"]
    values = {
        "max length": total["max_length"],
        "food events": total["events"]["score"],
        "wall deaths": total["deaths_by_cause"].get("wall", 0),
        "self deaths": total["deaths_by_cause"].get("self", 0),
    }
    print(
        f"Steps: {meta['steps']}, episodes: {total['episodes']}, events: {total['events']}, deaths by cause: {total['deaths_by_cause']}"
    )

    for name, target in snake_targets.items():
        status = "met" if values[name] >= target else "short"
        print(f"  {name}: {values[name]} (target {target}, {status})")

    print(
        f"Wrote {len(results)} chunks, meta.json and {args.preview} previews to {args.out}"
    )
