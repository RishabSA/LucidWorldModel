import tempfile
from pathlib import Path

import numpy as np

from scripts.check_observability import find_collisions
from scripts.collect import collect, summarize
from scripts.games import snake
from scripts.games.common import frame_size, load_chunk

steps = 600
chunk_steps = 300
workers = 2


def test_collect_replays_exactly_and_passes_observability() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        results = collect("snake", steps, workers, chunk_steps, 0, Path(tmp))
        paths = [result["path"] for result in results]
        chunks = [load_chunk(path) for path in paths]
        report = find_collisions(paths, snake.context)

    assert [result["source"] for result in results] == ["random", "scripted"]

    for chunk in chunks:
        assert chunk["frames"].shape == (chunk_steps + 1, frame_size, frame_size, 3)
        assert chunk["frames"].dtype == np.uint8
        assert chunk["actions"].shape == chunk["events"].shape == (chunk_steps,)
        assert chunk["actions"].dtype == chunk["events"].dtype == np.uint8

        game = snake.Game(chunk["seed"])
        frames = [game.render()]
        events = []

        for action in chunk["actions"]:
            events.append(game.step(int(action)))
            frames.append(game.render())

        assert np.array_equal(np.stack(frames), chunk["frames"])
        assert np.array_equal(events, chunk["events"])

    assert report["unflagged"] == []
    assert summarize(results, snake)["coverage"].keys() == snake.targets.keys()
