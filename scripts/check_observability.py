import argparse
import hashlib
import importlib
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from scripts.games.common import event_random, frame_size, load_chunk

max_strips = 5
strip_scale = 4
gap_width = 4
digest_size = 16


def find_collisions(paths: list[Path], context: int) -> dict:
    counts = Counter()
    all_random = {}
    # Context hash -> {next frame hash: (chunk index, t) of its first occurrence}
    outcomes = {}

    for chunk_index, path in enumerate(tqdm(paths, desc="Hashing chunks")):
        chunk = load_chunk(path)
        digests = [
            hashlib.blake2b(frame.tobytes(), digest_size=digest_size).digest()
            for frame in chunk["frames"]
        ]
        actions = chunk["actions"]
        randoms = chunk["events"] & event_random

        for t in range(context - 1, len(actions)):
            window = slice(t - context + 1, t + 1)
            key = b"".join(digests[window]) + actions[window].tobytes()
            counts[key] += 1
            all_random[key] = all_random.get(key, True) and bool(randoms[t])
            outcomes.setdefault(key, {}).setdefault(digests[t + 1], (chunk_index, t))

    repeated = [key for key, count in counts.items() if count > 1]
    conflicts = [key for key in repeated if len(outcomes[key]) > 1]

    return {
        "transitions": counts.total(),
        "contexts": len(counts),
        "repeated": len(repeated),
        "conflicts": len(conflicts),
        "unflagged": [
            list(outcomes[key].values()) for key in conflicts if not all_random[key]
        ],
    }


def save_strip(
    paths: list[Path], occurrences: list[tuple[int, int]], context: int, path: Path
) -> None:
    chunk_index, t = occurrences[0]
    tiles = list(load_chunk(paths[chunk_index])["frames"][t - context + 1 : t + 1])
    tiles.append(np.full((frame_size, gap_width, 3), 255, dtype=np.uint8))

    for chunk_index, t in occurrences:
        tiles.append(load_chunk(paths[chunk_index])["frames"][t + 1])

    strip = np.concatenate(tiles, axis=1)  # shape: (64, width, 3)
    strip = strip.repeat(strip_scale, axis=0).repeat(strip_scale, axis=1)
    Image.fromarray(strip).save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--game",
        type=str,
        default="snake",
        help="Game module in scripts/games (default: snake).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/snake"),
        help="Folder searched recursively for chunks (default: data/snake).",
    )
    args = parser.parse_args()

    context = importlib.import_module(f"scripts.games.{args.game}").context
    paths = sorted(args.data.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz chunks under {args.data}")

    report = find_collisions(paths, context)
    print(
        f"{report['transitions']} transitions, {report['contexts']} distinct contexts, "
        f"{report['repeated']} seen more than once, {report['conflicts']} with more than one next frame, "
        f"{len(report['unflagged'])} of those not flagged random"
    )

    out_dir = args.data / "observability"

    # Stale strips from an earlier failing run would look like current failures
    for path in out_dir.glob("collision_*.png"):
        path.unlink()

    if report["unflagged"]:
        os.makedirs(out_dir, exist_ok=True)

        for i, occurrences in enumerate(report["unflagged"][:max_strips]):
            save_strip(paths, occurrences, context, out_dir / f"collision_{i}.png")
            locations = ", ".join(f"{paths[c]} t={t}" for c, t in occurrences)
            print(f"  collision_{i}.png: {locations}")

        sys.exit(
            f"Hidden state: {len(report['unflagged'])} contexts lead to different next frames "
            f"without a random draw. Strips are in {out_dir}"
        )

    print("Observability check passed")
