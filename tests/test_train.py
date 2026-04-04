import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.games import snake
from scripts.games.common import save_chunk

repo = Path(__file__).parents[1]
chunks = [("random", 0), ("random", 1), ("random", 2), ("scripted", 0), ("scripted", 1)]
chunk_steps = 40
first_steps = 30
resume_steps = 40
width = 8
batch = 4


def write_chunks(data: Path) -> None:
    rng = random.Random(0)

    for seed, (source, i) in enumerate(chunks):
        game = snake.Game(seed)
        frames = [game.render()]
        actions = [rng.choice(snake.actions) for _ in range(chunk_steps)]
        events = []

        for action in actions:
            events.append(game.step(action))
            frames.append(game.render())

        save_chunk(
            data / source / f"chunk_{i}.npz",
            np.stack(frames),
            np.array(actions),
            np.array(events),
            seed,
        )


def run_train(data: Path, runs: Path, steps: int, *extra: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train",
            "--game=snake",
            "--name=test",
            f"--steps={steps}",
            f"--batch={batch}",
            f"--width={width}",
            f"--data={data}",
            f"--runs-dir={runs}",
            "--device=cpu",
            *extra,
        ],
        cwd=repo,
        check=True,
    )


def read_metrics(path: Path) -> list[dict]:
    with open(path) as file:
        return [json.loads(line) for line in file]


def test_train_writes_run_and_resumes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data, runs = Path(tmp) / "data", Path(tmp) / "runs"
        run_dir = runs / "snake" / "test"
        write_chunks(data)

        run_train(data, runs, first_steps)

        for name in ["config.json", "split.json", "last.pt", "metrics.jsonl"]:
            assert (run_dir / name).exists()

        assert (run_dir / "samples" / f"step_{first_steps}.gif").exists()
        with open(run_dir / "split.json") as file:
            split = json.load(file)
        assert len(split["heldout"]) == 2
        assert set(split["train"]).isdisjoint(split["heldout"])

        records = read_metrics(run_dir / "metrics.jsonl")
        assert all(math.isfinite(r["loss"]) for r in records if "loss" in r)
        assert all(
            math.isfinite(r["heldout_loss"]) for r in records if "heldout_loss" in r
        )

        run_train(data, runs, resume_steps, "--resume")
        checkpoint = torch.load(run_dir / "last.pt")
        records = read_metrics(run_dir / "metrics.jsonl")

        assert checkpoint["step"] == resume_steps
        assert (run_dir / f"ckpt_{first_steps}.pt").exists()
        assert [r["step"] for r in records if "loss" in r] == [
            first_steps,
            resume_steps,
        ]
