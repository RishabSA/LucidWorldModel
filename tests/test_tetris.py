import random
import tempfile
from pathlib import Path

import numpy as np

from scripts.check_observability import find_collisions
from scripts.collect import collect, summarize
from scripts.games import tetris
from scripts.games.common import (
    action_names,
    event_death,
    event_score,
    load_chunk,
    save_chunk,
)

steps = 2000
epsilon = 0.3
n = tetris.context
policy_games = 3
policy_steps = 2000
min_lines = 60
collect_steps = 600
collect_chunk_steps = 300
collect_workers = 2


def play(seed: int, epsilon: float) -> tuple[np.ndarray, list, list, list]:
    # Random actions from the full vocabulary add top-outs and every kind of move
    game = tetris.Game(seed)
    rng = random.Random(seed)
    frames = [game.render()]
    states = [game.state()]
    actions = []
    events = []

    for _ in range(steps):
        if rng.random() < epsilon:
            action = rng.randrange(len(action_names))
        else:
            action = tetris.policy(states[-1], rng)
        events.append(game.step(action))
        actions.append(action)
        frames.append(game.render())
        states.append(game.state())

    return np.stack(frames), states, actions, events


def replay(seed: int, actions: list) -> np.ndarray:
    game = tetris.Game(seed)
    frames = [game.render()]

    for action in actions:
        game.step(int(action))
        frames.append(game.render())

    return np.stack(frames)


def test_palette() -> None:
    frames = np.concatenate([play(0, epsilon)[0], play(0, 1)[0]])
    matches = sum(np.all(frames == color, axis=-1) for color in tetris.palette)
    assert np.all(matches == 1)
    assert len(tetris.palette) == len(set(tetris.palette)) == 18


def test_determinism() -> None:
    rng = random.Random(0)
    actions = [rng.randrange(len(action_names)) for _ in range(steps)]
    assert np.array_equal(replay(1, actions), replay(1, actions))


def test_chunk_replay() -> None:
    frames, _, actions, events = play(2, epsilon)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "chunk.npz"
        save_chunk(path, frames, np.array(actions), np.array(events), 2)
        chunk = load_chunk(path)

    assert chunk["frames"].dtype == np.uint8 and chunk["frames"].shape == frames.shape
    assert chunk["actions"].dtype == np.uint8 and chunk["events"].dtype == np.uint8
    assert np.array_equal(replay(chunk["seed"], chunk["actions"]), chunk["frames"])


def test_extract() -> None:
    for seed, eps in [(3, epsilon), (3, 1)]:
        frames, states, _, _ = play(seed, eps)

        for frame, state in zip(frames, states, strict=True):
            assert tetris.extract(frame) == {
                key: state[key] for key in tetris.visible_keys
            }

        assert any(state["dead"] for state in states)


def chain(frames: np.ndarray, actions: list) -> list:
    # Histories of check() states, as in dream play, carry the gravity phase forward
    states = [tetris.extract(frames[0])]

    for t, action in enumerate(actions):
        states.append(
            tetris.check(states[max(0, t - n + 1) :], action, frames[t + 1])[1]
        )

    return states


def test_rules_pass_on_real_play() -> None:
    seen = []

    for seed, eps in [(4, epsilon), (5, 1)]:
        frames, true_states, actions, _ = play(seed, eps)
        seen += true_states
        extracted = [tetris.extract(frame) for frame in frames]
        chained = chain(frames, actions)

        for states in (extracted, chained):
            for t, action in enumerate(actions):
                history = states[max(0, t - n + 1) : t + 1]
                names, state = tetris.check(history, action, frames[t + 1])
                assert names == [], (seed, t, names)
                assert {key: state[key] for key in tetris.visible_keys} == extracted[
                    t + 1
                ]
                assert state["timer"] in (None, true_states[t + 1]["timer"]), (seed, t)

        # From the first gravity step on, dream play always knows the gravity phase
        assert all(
            state["timer"] is not None
            for state in chained[tetris.gravity_ticks :]
            if not state["dead"]
        )

    # The runs include line clears, top-outs and restarts
    assert any(state["cleared"] for state in seen)
    assert any(state["dead"] for state in seen)


def test_rules_catch_planted_violations() -> None:
    game = tetris.Game(0)
    frames = [game.render()]

    # Noops from the start: gravity every 6 ticks, so ticks 6 and 12 move the piece down
    for _ in range(13):
        game.step(0)
        frames.append(game.render())

    states = chain(frames, [0] * 13)
    rows = [state["piece"][3] for state in states]
    assert rows == [0] * 6 + [1] * 6 + [2] * 2

    def broken(t: int, frame: np.ndarray, action: int = 0) -> list[str]:
        return tetris.check(states[max(0, t - n + 1) : t + 1], action, frame)[0]

    assert all(broken(t, frames[t + 1]) == [] for t in range(13))

    # Gravity due on tick 11 but skipped, and gravity on tick 10 when it is not due
    assert broken(11, frames[11]) == ["gravity"]
    assert broken(10, frames[12]) == ["gravity"]

    name, rotation, col, row = states[8]["piece"]
    shifted = {**states[8], "piece": (name, rotation, col + 2, row)}
    assert "move" in broken(7, tetris.draw(shifted), tetris.right_action)

    extra = {**states[8], "board": states[8]["board"][:-1] + ("I" + "." * 9,)}
    assert "locked" in broken(7, tetris.draw(extra))

    counted = {**states[8], "score": 1}
    assert "score" in broken(7, tetris.draw(counted))

    swapped = {**states[8], "next": "O" if states[8]["next"] != "O" else "I"}
    assert "preview" in broken(7, tetris.draw(swapped))

    two = frames[8].copy()
    two[tetris.board_y : tetris.board_y + 3, tetris.board_x : tetris.board_x + 3] = (
        tetris.active_colors[name]
    )
    assert "piece" in broken(7, two)

    # The inside of a cell is never read by extract(), only checked by the render rule
    stray = frames[8].copy()
    stray[tetris.board_y + 1, tetris.board_x + 1] = tetris.locked_colors["I"]
    assert broken(7, stray) == ["render"]

    assert broken(7, np.zeros_like(frames[8])) == ["unreadable"]


def test_rules_catch_bad_locks() -> None:
    # A bottom row full except under the I piece's left end; a hard drop clears it
    game = tetris.Game(0)
    game.board = tetris.empty_board[:-1] + ("." + "Z" * 9,)
    game.piece = ("I", 1, 0, 2)
    game.next = "T"
    before = tetris.extract(game.render())
    flags = game.step(tetris.hard_drop_action)
    after = tetris.extract(game.render())
    assert flags & event_score and after["score"] == 1
    assert after["board"][-3:] == ("I" + "." * 9,) * 3

    def broken(frame: np.ndarray) -> list[str]:
        return tetris.check([before], tetris.hard_drop_action, frame)[0]

    assert broken(tetris.draw(after)) == []
    unclear = {
        **after,
        "board": tetris.empty_board[:-4] + ("I" + "." * 9,) * 3 + ("I" + "Z" * 9,),
    }
    assert "clear" in broken(tetris.draw(unclear))
    assert "spawn" in broken(tetris.draw({**after, "piece": tetris.spawn("O")}))
    assert "score" in broken(tetris.draw({**after, "score": 0}))


def test_rules_catch_bad_top_outs() -> None:
    # Hard drops at the spawn stack pieces up the middle until one cannot spawn
    game = tetris.Game(1)
    frames = [game.render()]
    actions = []
    flags = 0

    while not flags & event_death:
        flags = game.step(tetris.hard_drop_action)
        actions.append(tetris.hard_drop_action)
        frames.append(game.render())

    for _ in range(tetris.flash_ticks):
        game.step(0)
        actions.append(0)
        frames.append(game.render())

    states = [tetris.extract(frame) for frame in frames]
    end = len(states) - 1
    assert [state["dead"] for state in states[-5:]] == [False, True, True, True, False]

    def broken(t: int, frame: np.ndarray) -> list[str]:
        return tetris.check(states[max(0, t - n + 1) : t + 1], actions[t], frame)[0]

    assert all(broken(t, frames[t + 1]) == [] for t in range(end - 4, end))
    assert "death" in broken(end - 4, frames[end - 4])
    assert "flash" in broken(end - 2, frames[end])
    assert "restart" in broken(end - 1, frames[end - 1])


def test_policy() -> None:
    four_line_clears = 0

    for seed in range(policy_games):
        game = tetris.Game(seed)
        rng = random.Random(seed)
        lines = 0

        for _ in range(policy_steps):
            assert not game.step(tetris.policy(game.state(), rng)) & event_death
            lines += game.state()["cleared"]
            four_line_clears += game.state()["cleared"] == tetris.tetris_lines

        assert lines >= min_lines

    # The well rule leads to four-line clears
    assert four_line_clears > 0


def test_collect_replays_exactly_and_passes_observability() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        results = collect(
            "tetris",
            collect_steps,
            collect_workers,
            collect_chunk_steps,
            0,
            Path(tmp),
        )
        paths = [result["path"] for result in results]
        chunks = [load_chunk(path) for path in paths]
        report = find_collisions(paths, n)

    for chunk in chunks:
        game = tetris.Game(chunk["seed"])
        frames = [game.render()]
        events = []

        for action in chunk["actions"]:
            events.append(game.step(int(action)))
            frames.append(game.render())

        assert np.array_equal(np.stack(frames), chunk["frames"])
        assert np.array_equal(events, chunk["events"])

    assert report["unflagged"] == []
    assert summarize(results, tetris)["coverage"].keys() == tetris.targets.keys()
