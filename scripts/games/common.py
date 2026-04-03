import os
from pathlib import Path

import numpy as np
import pygame

frame_size = 64
action_names = [
    "noop",
    "up",
    "down",
    "left",
    "right",
    "space",
    "left_space",
    "right_space",
]

event_score = 1
event_death = 2
event_restart = 4
event_random = 8

digit_width = 3
digit_height = 5
digit_stride = digit_width + 1
digit_glyphs = [
    ["###", "#.#", "#.#", "#.#", "###"],
    [".#.", "##.", ".#.", ".#.", "###"],
    ["###", "..#", "###", "#..", "###"],
    ["###", "..#", "###", "..#", "###"],
    ["#.#", "#.#", "###", "..#", "..#"],
    ["###", "#..", "###", "..#", "###"],
    ["###", "#..", "###", "#.#", "###"],
    ["###", "..#", "..#", "..#", "..#"],
    ["###", "#.#", "###", "#.#", "###"],
    ["###", "#.#", "###", "..#", "###"],
]
digit_masks = np.array(
    [[[char == "#" for char in line] for line in glyph] for glyph in digit_glyphs]
)  # shape: (10, digit_height, digit_width)


def draw_number(
    surface: pygame.Surface, value: int, x: int, y: int, color: tuple
) -> None:
    for i, char in enumerate(str(value)):
        for row, col in np.argwhere(digit_masks[int(char)]):
            surface.fill(color, (x + i * digit_stride + col, y + row, 1, 1))


def read_number(
    frame: np.ndarray, x: int, y: int, color: tuple, max_digits: int
) -> int:
    digits = ""

    for i in range(max_digits):
        left = x + i * digit_stride
        mask = np.all(
            frame[y : y + digit_height, left : left + digit_width] == color, axis=-1
        )  # shape: (digit_height, digit_width)
        if not mask.any():
            break
        digits += str(np.flatnonzero(np.all(digit_masks == mask, axis=(1, 2)))[0])

    return int(digits)


def to_array(surface: pygame.Surface) -> np.ndarray:
    # surfarray is indexed (x, y, channel)
    return np.ascontiguousarray(
        pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    )  # shape: (frame_size, frame_size, 3)


def save_chunk(
    path: Path, frames: np.ndarray, actions: np.ndarray, events: np.ndarray, seed: int
) -> None:
    if len(frames) != len(actions) + 1 or len(events) != len(actions):
        raise ValueError(
            f"Chunk needs L + 1 frames and L actions and events, got {len(frames)} frames, {len(actions)} actions, {len(events)} events"
        )

    os.makedirs(path.parent, exist_ok=True)

    # Write then rename, so an interrupted save never leaves a truncated chunk
    partial = path.with_name(f"{path.name}.partial")
    with open(partial, "wb") as file:
        np.savez_compressed(
            file,
            frames=np.asarray(frames, dtype=np.uint8),
            actions=np.asarray(actions, dtype=np.uint8),
            events=np.asarray(events, dtype=np.uint8),
            seed=np.int64(seed),
        )
    os.replace(partial, path)


def load_chunk(path: Path) -> dict:
    with np.load(path) as chunk:
        return {
            "frames": chunk["frames"],
            "actions": chunk["actions"],
            "events": chunk["events"],
            "seed": int(chunk["seed"]),
        }
