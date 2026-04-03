import argparse
import importlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from scripts.games.common import frame_size, save_chunk

data_dir = Path("data")
queue_depth = 3
save_every = 3000
key_actions = {
    pygame.K_UP: 1,
    pygame.K_w: 1,
    pygame.K_DOWN: 2,
    pygame.K_s: 2,
    pygame.K_LEFT: 3,
    pygame.K_a: 3,
    pygame.K_RIGHT: 4,
    pygame.K_d: 4,
    pygame.K_SPACE: 5,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a game at its tick rate.")
    parser.add_argument(
        "--game", type=str, default="snake", help="Game module (default: snake)."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save the session as a chunk under data/<game>/human/ (default: False).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Game seed (default: 0).")
    parser.add_argument(
        "--scale",
        type=int,
        default=8,
        help="Window pixels per frame pixel (default: 8).",
    )
    args = parser.parse_args()

    game_module = importlib.import_module(f"scripts.games.{args.game}")
    chunk_path = (
        data_dir
        / args.game
        / "human"
        / f"chunk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    )

    pygame.init()
    window = pygame.display.set_mode((frame_size * args.scale, frame_size * args.scale))
    pygame.display.set_caption(f"LucidWorldModel: {args.game}")
    clock = pygame.time.Clock()

    game = game_module.Game(args.seed)
    frames = [game.render()]
    actions = []
    events = []
    queued = []
    held = []
    running = True

    while running:
        frame_surface = pygame.surfarray.make_surface(frames[-1].transpose(1, 0, 2))
        window.blit(pygame.transform.scale(frame_surface, window.get_size()), (0, 0))
        pygame.display.flip()
        clock.tick(game_module.tick_hz)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key in key_actions:
                held.append(event.key)
                if len(queued) < queue_depth:
                    queued.append(key_actions[event.key])
            elif event.type == pygame.KEYUP and event.key in held:
                held.remove(event.key)

        if not running:
            break

        if game_module.input_mode == "queue":
            action = queued.pop(0) if queued else 0
        else:
            action = key_actions[held[-1]] if held else 0
        if action not in game_module.actions:
            action = 0

        events.append(game.step(action))
        actions.append(action)
        frames.append(game.render())

        if args.record and len(actions) % save_every == 0:
            save_chunk(chunk_path, np.stack(frames), actions, events, args.seed)

    pygame.quit()

    if args.record and actions:
        save_chunk(chunk_path, np.stack(frames), actions, events, args.seed)
        print(f"Saved {len(actions)} steps to {chunk_path}")
