import random
import tempfile
from pathlib import Path

import numpy as np

from scripts.games import snake
from scripts.games.common import action_names, event_death, load_chunk, save_chunk

steps = 3000
epsilon = 0.3
policy_games = 50
min_survival = 500
min_length = 10
extract_keys = ["snake", "direction", "food", "score", "length", "dead"]


def play(seed: int) -> tuple[np.ndarray, list, list, list]:
    # Scripted play reaches long snakes; random actions from the full vocabulary add deaths
    game = snake.Game(seed)
    rng = random.Random(seed)
    frames = [game.render()]
    states = [game.state()]
    actions = []
    events = []

    for _ in range(steps):
        if rng.random() < epsilon:
            action = rng.randrange(len(action_names))
        else:
            action = snake.policy(states[-1], rng)
        events.append(game.step(action))
        actions.append(action)
        frames.append(game.render())
        states.append(game.state())

    return np.stack(frames), states, actions, events


def replay(seed: int, actions: list) -> np.ndarray:
    game = snake.Game(seed)
    frames = [game.render()]

    for action in actions:
        game.step(int(action))
        frames.append(game.render())

    return np.stack(frames)


def test_palette() -> None:
    frames = play(0)[0]
    matches = sum(np.all(frames == color, axis=-1) for color in snake.palette)
    assert np.all(matches == 1)


def test_determinism() -> None:
    rng = random.Random(0)
    actions = [rng.randrange(len(action_names)) for _ in range(steps)]
    assert np.array_equal(replay(1, actions), replay(1, actions))


def test_chunk_replay() -> None:
    frames, _, actions, events = play(2)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "chunk.npz"
        save_chunk(path, frames, np.array(actions), np.array(events), 2)
        chunk = load_chunk(path)

    assert chunk["frames"].dtype == np.uint8 and chunk["frames"].shape == frames.shape
    assert chunk["actions"].dtype == np.uint8 and chunk["events"].dtype == np.uint8
    assert np.array_equal(replay(chunk["seed"], chunk["actions"]), chunk["frames"])


def test_extract() -> None:
    frames, states, _, _ = play(3)

    for frame, state in zip(frames, states, strict=True):
        assert snake.extract(frame) == {key: state[key] for key in extract_keys}

    assert any(state["dead"] for state in states)


def test_policy() -> None:
    survival = []
    lengths = []

    for seed in range(policy_games):
        game = snake.Game(seed)
        rng = random.Random(seed)
        ticks = 1

        while not game.step(snake.policy(game.state(), rng)) & event_death:
            ticks += 1

        # The snake never shrinks, so its length at death is the longest it got
        survival.append(ticks)
        lengths.append(game.state()["length"])

    assert np.mean(survival) >= min_survival
    assert min(lengths) >= min_length
