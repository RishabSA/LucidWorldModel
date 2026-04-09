import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.check_observability import find_collisions
from scripts.collect import collect, summarize
from scripts.games import pong
from scripts.games.common import (
    action_names,
    event_death,
    event_restart,
    event_score,
    load_chunk,
)

n = pong.context
steps = 3000
epsilon = 0.3
collect_steps = 600
chunk_steps = 300
workers = 2
policy_seeds = [0, 1, 2]
min_return_rate = 0.75
extract_keys = ["ball", "cpu", "player", "scores", "dead"]


def play(seed: int, scripted: bool) -> tuple[np.ndarray, list, list, list]:
    # Random actions come from the full vocabulary, so unused ones must act as noop
    game = pong.Game(seed)
    rng = random.Random(seed)
    frames = [game.render()]
    states = [game.state()]
    actions = []
    events = []

    for _ in range(steps):
        if scripted and rng.random() >= epsilon:
            action = pong.policy(states[-1], rng)
        else:
            action = rng.randrange(len(action_names))
        events.append(game.step(action))
        actions.append(action)
        frames.append(game.render())
        states.append(game.state())

    return np.stack(frames), states, actions, events


def replay(seed: int, actions: list) -> tuple[np.ndarray, list]:
    game = pong.Game(seed)
    frames = [game.render()]
    events = []

    for action in actions:
        events.append(game.step(int(action)))
        frames.append(game.render())

    return np.stack(frames), events


def test_palette() -> None:
    for scripted in [False, True]:
        frames = play(0, scripted)[0]
        matches = sum(np.all(frames == color, axis=-1) for color in pong.palette)
        assert np.all(matches == 1)


def test_determinism() -> None:
    rng = random.Random(0)
    actions = [rng.randrange(len(action_names)) for _ in range(steps)]
    assert np.array_equal(replay(1, actions)[0], replay(1, actions)[0])


def test_collected_chunks_replay() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        results = collect("pong", collect_steps, workers, chunk_steps, 0, Path(tmp))
        paths = [result["path"] for result in results]
        chunks = [load_chunk(path) for path in paths]
        report = find_collisions(paths, n)

    assert [result["source"] for result in results] == ["random", "scripted"]

    for chunk in chunks:
        frames, events = replay(chunk["seed"], chunk["actions"])
        assert np.array_equal(frames, chunk["frames"])
        assert np.array_equal(events, chunk["events"])

    assert report["unflagged"] == []
    assert summarize(results, pong)["coverage"].keys() == pong.targets.keys()


def test_extract() -> None:
    for scripted in [False, True]:
        frames, states, _, _ = play(3, scripted)

        for frame, state in zip(frames, states, strict=True):
            assert pong.extract(frame) == {key: state[key] for key in extract_keys}

        assert any(state["dead"] for state in states)


def test_rules_pass_on_real_play() -> None:
    for scripted in [False, True]:
        frames, true_states, actions, events = play(4, scripted)
        states = [pong.extract(frame) for frame in frames]

        for t, action in enumerate(actions):
            history = states[max(0, t - n + 1) : t + 1]
            names, state = pong.check(history, action, frames[t + 1])
            assert names == [], (scripted, t, names)
            assert state == {
                **states[t + 1],
                "velocity": true_states[t + 1]["velocity"],
            }

        # The streams include hits on both paddles, points on both sides and match ends
        hits = {state["hit"][0] for state in true_states if state["hit"]}
        scorers = {
            s["scorer"]
            for s, e in zip(true_states[1:], events, strict=True)
            if e & event_score
        }
        assert hits == scorers == {"cpu", "player"}
        assert any(flags & event_death for flags in events)


def test_rules_catch_planted_violations() -> None:
    frames, true_states, actions, events = play(5, False)
    states = [pong.extract(frame) for frame in frames]

    def broken(t: int, frame: np.ndarray) -> list[str]:
        return pong.check(states[t - n + 1 : t + 1], actions[t], frame)[0]

    def first(condition: list) -> int:
        return next(t for t in range(n, steps) if condition[t])

    # A quiet tick with both paddles off the walls, a serve, a match end and the restart after it
    quiet = first(
        [
            not flags
            and not state["hit"]
            and pong.field_top < min(state["cpu"], state["player"])
            and max(state["cpu"], state["player"]) < pong.paddle_max_y
            for state, flags in zip(true_states[1:], events, strict=True)
        ]
    )
    hit = first([state["hit"] is not None for state in true_states[1:]])
    serve = first([flags & event_score and not flags & event_death for flags in events])
    end = first([flags & event_death for flags in events])
    restart = first([flags & event_restart for flags in events])
    assert restart == end + pong.flash_ticks

    after = states[quiet + 1]
    x, y = after["ball"]
    assert "ball" in broken(quiet, pong.draw({**after, "ball": (x, y + 1)}))
    assert "player" in broken(
        quiet, pong.draw({**after, "player": after["player"] + 1})
    )
    assert "cpu" in broken(quiet, pong.draw({**after, "cpu": after["cpu"] + 1}))
    scores = {**after["scores"], "player": after["scores"]["player"] + 1}
    assert "score" in broken(quiet, pong.draw({**after, "scores": scores}))

    # A paddle return whose vy does not match the hit zone
    bounced = states[hit + 1]
    x, y = bounced["ball"]
    assert "ball" in broken(hit, pong.draw({**bounced, "ball": (x, y + 1)}))

    # Row 0 above the digits is always background, and extract() reads only its first pixel
    stray = frames[quiet + 1].copy()
    stray[0, pong.line_x] = pong.wall_color
    assert broken(quiet, stray) == ["render"]

    assert "serve" in broken(
        serve, pong.draw({**states[serve + 1], "ball": pong.center})
    )
    assert "death" in broken(end, pong.draw({**states[end + 1], "dead": False}))
    assert "flash" in broken(end + 1, pong.draw({**states[end + 2], "dead": False}))
    assert "restart" in broken(restart, frames[restart])
    assert broken(quiet, np.zeros_like(frames[0])) == ["unreadable"]


def test_policy() -> None:
    counts = Counter()

    for seed in policy_seeds:
        game = pong.Game(seed)
        rng = random.Random(seed)

        for _ in range(steps):
            flags = game.step(pong.policy(game.state(), rng))
            counts.update(pong.tally(game.state(), flags))

    returns = sum(counts[("hit", "player", zone)] for zone in set(pong.zone_vy))
    misses = counts[("point", "cpu")]
    assert returns / (returns + misses) >= min_return_rate
    assert counts[("point", "player")] > misses > 0
