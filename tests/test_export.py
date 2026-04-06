import base64
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate import to_frames, to_model
from scripts.export_onnx import min_parity, randomize_zero_init, run_onnx
from scripts.games import snake
from scripts.games.common import event_restart, frame_size, save_chunk
from scripts.model import Denoiser

repo = Path(__file__).parents[1]
n = snake.context
chunk_steps = 200
width = 8
step = 7
parity_cases = 2
other_game = {"game": "pong", "title": "Pong", "meta": "pong.json"}


def make_run(tmp_path: Path) -> tuple[Path, set]:
    # Random play dies often, so a short chunk holds several restarts
    rng = random.Random(0)
    game = snake.Game(0)
    frames = [game.render()]
    actions = [rng.choice(snake.actions) for _ in range(chunk_steps)]
    events = []

    for action in actions:
        events.append(game.step(action))
        frames.append(game.render())

    frames, actions, events = np.stack(frames), np.array(actions), np.array(events)
    save_chunk(tmp_path / "data" / "chunk_0.npz", frames, actions, events, 0)

    # Expected starts: N frames of a fresh game and the N - 1 actions between them, with no events
    windows = {
        frames[t + 1 : t + 1 + n].tobytes()
        + actions[t + 1 : t + n].astype(np.uint8).tobytes()
        for t in np.flatnonzero(events & event_restart)
        if t + n < len(frames) and not events[t + 1 : t + n].any()
    }

    run = tmp_path / "run"
    os.makedirs(run)
    config = {
        "game": "snake",
        "width": width,
        "context": n,
        "sigma_data": 0.5,
        "data_dir": str(tmp_path / "data"),
        "sampler_steps": 3,
        "sigma_max": 5.0,
        "sigma_min": 0.002,
        "rho": 7.0,
        "sigma_aug": 0.0,
    }
    (run / "config.json").write_text(json.dumps(config))
    (run / "split.json").write_text(
        json.dumps({"train": [], "heldout": ["chunk_0.npz"]})
    )

    torch.manual_seed(0)
    denoiser = Denoiser(width, n, config["sigma_data"])
    randomize_zero_init(denoiser)
    torch.save({"step": step, "ema": denoiser.state_dict()}, run / "last.pt")

    return run, windows


def run_export(run: Path, out: Path, starts: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.export_game", "--run", str(run)]
        + ["--out-dir", str(out), "--steps", "2", "--sigma-min", "0.3", "--rho", "3"]
        + ["--starts", str(starts), "--parity-cases", str(parity_cases)],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def decode(text: str, dtype: str, shape: tuple) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=dtype).reshape(shape)


def test_export_writes_browser_files(tmp_path: Path) -> None:
    run, windows = make_run(tmp_path)
    out = tmp_path / "models"
    os.makedirs(out)
    (out / "manifest.json").write_text(json.dumps([other_game]))

    result = run_export(run, out, len(windows))
    assert result.returncode == 0, result.stderr

    # All frames first, then all actions, as uint8 with no header
    count = len(windows)
    data = np.fromfile(out / "snake_starts.bin", dtype=np.uint8)
    frame_bytes = count * n * frame_size * frame_size * 3
    frames = data[:frame_bytes].reshape(count, n, frame_size, frame_size, 3)
    actions = data[frame_bytes:].reshape(count, n - 1)
    assert {f.tobytes() + a.tobytes() for f, a in zip(frames, actions)} == windows

    meta = json.loads((out / "snake.json").read_text())
    assert meta == {
        "game": "snake",
        "model": "snake.onnx",
        "tick_hz": snake.tick_hz,
        "context": n,
        "actions": snake.actions,
        "action_names": ["noop", "up", "down", "left", "right"],
        "input_mode": "queue",
        "keys": {
            "ArrowUp": 1,
            "KeyW": 1,
            "ArrowDown": 2,
            "KeyS": 2,
            "ArrowLeft": 3,
            "KeyA": 3,
            "ArrowRight": 4,
            "KeyD": 4,
        },
        "palette": [list(color) for color in snake.palette],
        "sampler": {
            "steps": 2,
            "sigma_aug": 0.0,
            "sigma_max": 5.0,
            "sigma_min": 0.3,
            "rho": 3.0,
        },
        "starts": {"file": "snake_starts.bin", "count": count},
        "parity": "snake_parity.json",
        "run": str(run),
        "step": step,
    }

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest == [
        other_game,
        {"game": "snake", "title": "Snake", "meta": "snake.json"},
    ]

    # The browser's parity check, run here through onnxruntime CPU
    parity = json.loads((out / "snake_parity.json").read_text())
    assert parity["min_identical"] == min_parity
    assert len(parity["cases"]) == parity_cases

    for case in parity["cases"]:
        case_frames = decode(case["frames"], "uint8", (n, frame_size, frame_size, 3))
        noise = decode(case["noise"], "<f4", (1, 3 * (n + 1), frame_size, frame_size))
        expected = decode(case["expected"], "uint8", (frame_size, frame_size, 3))
        assert any(np.array_equal(case_frames, start) for start in frames)

        inputs = (
            to_model(case_frames[None].copy(), torch.device("cpu")),
            torch.tensor([case["actions"]]),
            torch.from_numpy(noise.copy()),
        )
        actual = to_frames(torch.from_numpy(run_onnx(out / "snake.onnx", inputs)))[0]
        assert np.all(actual == expected, axis=-1).mean() >= min_parity

    result = run_export(run, out, len(windows))
    assert result.returncode != 0 and "FileExistsError" in result.stderr
