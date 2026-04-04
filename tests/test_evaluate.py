import json
import os
import subprocess
import sys
from itertools import pairwise, takewhile
from operator import itemgetter
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate import dream, horizons, one_step, open_loop, to_frames, to_model
from scripts.games import snake
from scripts.games.common import frame_size, save_chunk
from scripts.model import Denoiser
from tests.test_games import play

n = snake.context
seeds = [0, 1]
chunk_steps = 300
picks = 20
perfect_horizon = 200
width = 8
device = torch.device("cpu")


def make_chunk(seed: int) -> dict:
    frames, _, actions, events = play(seed)

    return {
        "frames": frames[: chunk_steps + 1],
        "actions": np.array(actions[:chunk_steps], dtype=np.uint8),
        "events": np.array(events[:chunk_steps], dtype=np.uint8),
    }


def oracle(context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    # A perfect world model: read the context back into a game and step it
    frames = to_frames(context.reshape(-1, 3, frame_size, frame_size)).reshape(
        len(context), n, frame_size, frame_size, 3
    )
    rendered = []

    for history, action in zip(frames, actions[:, -1].tolist(), strict=True):
        states = [snake.extract(frame) for frame in history]
        dead_frames = len(list(takewhile(itemgetter("dead"), reversed(states))))
        game = snake.Game(len(rendered))
        game.snake = states[-1]["snake"]
        game.direction = states[-1]["direction"]
        game.food = states[-1]["food"]
        game.score = states[-1]["score"]
        game.flash = snake.flash_ticks + 1 - dead_frames if dead_frames else 0
        game.step(action)
        rendered.append(game.render())

    return to_model(np.stack(rendered)[:, None], context.device)


def test_rules_pass_on_real_play() -> None:
    for seed in seeds:
        frames, true_states, actions, _ = play(seed)
        states = [snake.extract(frame) for frame in frames]

        for t, action in enumerate(actions):
            history = states[max(0, t - n + 1) : t + 1]
            names, state = snake.check(history, action, frames[t + 1])
            assert names == [], (seed, t, names)
            assert state == states[t + 1]

        # The streams include eating, wall and self deaths, and restarts
        causes = {state["cause"] for state in true_states}
        assert causes == {None, "wall", "self"}
        assert max(state["score"] for state in true_states) > 0


def test_rules_catch_planted_violations() -> None:
    # Going straight from the start hits the right wall on tick 8, then 3 dead frames and a restart on tick 11
    game = snake.Game(0)
    frames = [game.render()]

    for _ in range(11):
        game.step(0)
        frames.append(game.render())

    states = [snake.extract(frame) for frame in frames]
    assert [state["dead"] for state in states[7:]] == [False, True, True, True, False]

    def broken(t: int, frame: np.ndarray) -> list[str]:
        return snake.check(states[t - n + 1 : t + 1], 0, frame)[0]

    assert all(broken(t, frames[t + 1]) == [] for t in range(n - 1, 11))

    moved = [(col + 1, row) for col, row in states[4]["snake"]]
    assert "head" in broken(3, snake.draw({**states[4], "snake": moved}))

    second_food = frames[4].copy()
    second_food[snake.board_top : snake.board_top + snake.segment, : snake.segment] = (
        snake.food_color
    )
    assert "food" in broken(3, second_food)

    # The corner gap of a cell is never painted, and extract() never reads it
    stray = frames[4].copy()
    stray[snake.board_top + snake.segment, snake.segment] = snake.body_color
    assert broken(3, stray) == ["render"]

    assert "flash" in broken(9, frames[11])
    assert "restart" in broken(10, frames[10])
    assert broken(3, np.zeros_like(frames[4])) == ["unreadable"]


def test_rungs_score_a_perfect_model(tmp_path: Path) -> None:
    chunks = [make_chunk(seed) for seed in seeds]
    windows = [(c, t) for c in range(len(chunks)) for t in range(n - 1, chunk_steps)]
    starts = [windows[i] for i in np.random.default_rng(0).permutation(len(windows))]

    step = one_step(oracle, snake, chunks, starts[:picks], n, device)
    assert step["pixel_accuracy"] == step["frame_accuracy"] == 1
    assert step["rule_pass_rate"] == 1

    loop = open_loop(oracle, chunks, starts[:picks], n, device)
    assert loop["1"]["rollouts"] > 0 and loop["10"]["rollouts"] > 0
    assert all(row["pixel_accuracy"] in (None, 1) for row in loop.values())

    calls = []

    def recorded(context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        calls.append(actions.clone())

        return oracle(context, actions)

    played = dream(
        recorded,
        snake,
        chunks,
        starts[:4],
        n,
        device,
        perfect_horizon,
        0.1,
        0,
        tmp_path / "dream",
    )
    assert played["reached_horizon"] == 1 and played["first_broken"] == {}

    # The oracle only reads this tick's action, so check the older slots shift along
    assert all(
        torch.equal(now[:, :-1], before[:, 1:]) for before, now in pairwise(calls)
    )


def test_evaluate_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    for seed in seeds:
        chunk = make_chunk(seed)
        save_chunk(
            data_dir / f"chunk_{seed}.npz",
            chunk["frames"],
            chunk["actions"],
            chunk["events"],
            seed,
        )

    run = tmp_path / "run"
    os.makedirs(run)
    config = {
        "game": "snake",
        "width": width,
        "context": n,
        "sigma_data": 0.5,
        "data_dir": str(data_dir),
        "sampler_steps": 3,
        "sigma_max": 5.0,
        "sigma_min": 0.002,
        "rho": 7.0,
        "sigma_aug": 0.0,
    }
    (run / "config.json").write_text(json.dumps(config))
    split = {"train": [], "heldout": [f"chunk_{seed}.npz" for seed in seeds]}
    (run / "split.json").write_text(json.dumps(split))

    torch.manual_seed(0)
    weights = Denoiser(width, n, config["sigma_data"]).state_dict()
    checkpoint = {
        "step": 7,
        "model": weights,
        "ema": weights,
        "optimizer": {},
        "config": config,
    }
    torch.save(checkpoint, run / "last.pt")

    subprocess.run(
        [sys.executable, "-m", "scripts.evaluate", "--run", str(run)]
        + ["--windows", "8", "--open-loop", "3", "--rollouts", "3", "--horizon", "5"]
        + ["--device", "cpu"],
        check=True,
    )

    metrics = json.loads((run / "eval_7" / "metrics.json").read_text())
    assert set(metrics["k"]) == {"1", "3"}

    for result in metrics["k"].values():
        assert result["one_step"]["transitions"] == 8
        assert set(result["open_loop"]) == {str(h) for h in horizons}
        assert len(result["dream"]["clean_steps"]) == 3

    assert len(list((run / "eval_7").glob("dream_k*_*.gif"))) == 6
