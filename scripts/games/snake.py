import random
from collections import deque

import numpy as np
import pygame

from scripts.games.common import (
    action_names,
    draw_number,
    event_death,
    event_random,
    event_restart,
    event_score,
    frame_size,
    read_number,
    to_array,
)

tick_hz = 10
context = 4
actions = [0, 1, 2, 3, 4]
input_mode = "queue"

background_color = (16, 16, 24)
wall_color = (64, 72, 128)
body_color = (32, 200, 64)
head_color = (240, 232, 48)
food_color = (232, 32, 48)
dead_color = (168, 48, 232)
text_color = (240, 240, 240)
palette = [
    background_color,
    wall_color,
    body_color,
    head_color,
    food_color,
    dead_color,
    text_color,
]

cols = 16
rows = 14
cell = 4
segment = 3
board_top = 8
score_x = 1
score_y = 1
score_digits = 3
start_length = 3
start_cell = (8, 7)
flash_ticks = 3

moves = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
move_names = {move: name for name, move in moves.items()}
opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}


def in_bounds(col: int, row: int) -> bool:
    return 0 <= col < cols and 0 <= row < rows


class Game:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.surface = pygame.Surface((frame_size, frame_size))
        self.restart()

    def restart(self) -> None:
        self.snake = [(start_cell[0] - i, start_cell[1]) for i in range(start_length)]
        self.direction = "right"
        self.score = 0
        self.flash = 0
        self.cause = None
        self.food = self.rng.choice(self.empty_cells())

    def empty_cells(self) -> list:
        occupied = set(self.snake)
        return [
            (col, row)
            for row in range(rows)
            for col in range(cols)
            if (col, row) not in occupied
        ]

    def die(self, cause: str) -> int:
        self.flash = flash_ticks
        self.cause = cause
        return event_death

    def step(self, action: int) -> int:
        if self.flash:
            self.flash -= 1
            if self.flash:
                return 0
            self.restart()
            return event_restart | event_random

        direction = self.direction
        if action in actions[1:] and action_names[action] != opposite[direction]:
            direction = action_names[action]

        head_col, head_row = self.snake[0]
        dx, dy = moves[direction]
        head = (head_col + dx, head_row + dy)
        eating = head == self.food

        # The tail moves out of the way unless the snake grows this tick
        body = self.snake if eating else self.snake[:-1]
        if not in_bounds(*head):
            return self.die("wall")
        if head in body:
            return self.die("self")

        self.snake.insert(0, head)
        self.direction = direction
        if not eating:
            self.snake.pop()
            return 0

        self.score += 1
        empty = self.empty_cells()
        if not empty:
            self.food = None
            return event_score | self.die("full")
        self.food = self.rng.choice(empty)

        return event_score | event_random

    def render(self) -> np.ndarray:
        self.surface.fill(background_color)
        self.surface.fill(wall_color, (0, 0, frame_size, board_top))
        draw_number(self.surface, self.score, score_x, score_y, text_color)

        if self.food is not None:
            col, row = self.food
            self.surface.fill(
                food_color, (col * cell, board_top + row * cell, segment, segment)
            )

        # One rect per consecutive pair covers both segments and the bridge between them
        link_color = dead_color if self.flash else body_color

        for (col, row), (next_col, next_row) in zip(self.snake, self.snake[1:]):
            self.surface.fill(
                link_color,
                (
                    min(col, next_col) * cell,
                    board_top + min(row, next_row) * cell,
                    segment + abs(col - next_col) * cell,
                    segment + abs(row - next_row) * cell,
                ),
            )

        # The head keeps its color while dead, so a single frame still says which end is the head
        col, row = self.snake[0]
        self.surface.fill(
            head_color, (col * cell, board_top + row * cell, segment, segment)
        )

        return to_array(self.surface)

    def state(self) -> dict:
        return {
            "snake": list(self.snake),
            "direction": self.direction,
            "food": self.food,
            "score": self.score,
            "length": len(self.snake),
            "dead": self.flash > 0,
            "cause": self.cause,
            "flash": self.flash,
        }


def extract(frame: np.ndarray) -> dict:
    # Top-left pixel of every cell
    cells = frame[board_top::cell, ::cell]  # shape: (rows, cols, 3)
    dead = bool(np.all(cells == dead_color, axis=-1).any())
    link_color = dead_color if dead else body_color

    head_row, head_col = np.argwhere(np.all(cells == head_color, axis=-1))[0]
    snake = [(int(head_col), int(head_row))]
    growing = True

    # Follow the bridges from the head; each one sits in the gap of the upper or left cell
    while growing:
        col, row = snake[-1]
        growing = False

        for dx, dy in moves.values():
            neighbor = (col + dx, row + dy)
            x = min(col, neighbor[0]) * cell + (segment if dx else 0)
            y = board_top + min(row, neighbor[1]) * cell + (segment if dy else 0)
            if (
                in_bounds(*neighbor)
                and neighbor not in snake
                and np.all(frame[y, x] == link_color)
            ):
                snake.append(neighbor)
                growing = True
                break

    (head_col, head_row), (neck_col, neck_row) = snake[0], snake[1]
    food_cells = np.argwhere(np.all(cells == food_color, axis=-1))

    return {
        "snake": snake,
        "direction": move_names[(head_col - neck_col, head_row - neck_row)],
        "food": (
            (int(food_cells[0][1]), int(food_cells[0][0])) if len(food_cells) else None
        ),
        "score": read_number(frame, score_x, score_y, text_color, score_digits),
        "length": len(snake),
        "dead": dead,
    }


def distances(start: tuple, blocked: set) -> dict:
    distance = {start: 0}
    queue = deque([start])

    while queue:
        col, row = queue.popleft()

        for dx, dy in moves.values():
            neighbor = (col + dx, row + dy)
            if (
                in_bounds(*neighbor)
                and neighbor not in blocked
                and neighbor not in distance
            ):
                distance[neighbor] = distance[(col, row)] + 1
                queue.append(neighbor)

    return distance


def policy(state: dict, rng: random.Random) -> int:
    if state["dead"]:
        return 0

    snake = state["snake"]
    head_col, head_row = snake[0]

    # The tail cell is free because the tail moves away this tick
    blocked = set(snake[:-1])
    targets = {name: (head_col + dx, head_row + dy) for name, (dx, dy) in moves.items()}
    safe = [
        name
        for name, target in targets.items()
        if in_bounds(*target) and target not in blocked
    ]
    if not safe:
        return 0

    # Distance to the food from every free cell, then step to the closest neighbor
    to_food = distances(state["food"], blocked) if state["food"] else {}
    scores = {
        name: -to_food[targets[name]] for name in safe if targets[name] in to_food
    }
    if not scores:
        scores = {name: len(distances(targets[name], blocked)) for name in safe}
    best = max(scores.values())
    move = rng.choice([name for name in safe if scores.get(name) == best])

    return 0 if move == state["direction"] else action_names.index(move)
