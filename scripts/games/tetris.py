import random
from collections import Counter, deque
from functools import lru_cache
from itertools import pairwise, takewhile
from operator import itemgetter

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
context = 8
actions = [0, 1, 2, 3, 4, 5]
input_mode = "queue"

rotate_action = action_names.index("up")
soft_drop_action = action_names.index("down")
left_action = action_names.index("left")
right_action = action_names.index("right")
hard_drop_action = action_names.index("space")
# Action -> (rotation step, column step, row step)
moves = {
    rotate_action: (1, 0, 0),
    soft_drop_action: (0, 0, 1),
    left_action: (0, -1, 0),
    right_action: (0, 1, 0),
}

# Every color sits on the {0, 128, 255} grid, so any two are at least 127 apart
background_color = (0, 0, 0)
border_color = (128, 128, 255)
text_color = (255, 255, 255)
flash_color = (255, 128, 128)
active_colors = {
    "I": (0, 255, 255),
    "O": (255, 255, 0),
    "T": (128, 0, 255),
    "S": (0, 255, 0),
    "Z": (255, 0, 0),
    "J": (0, 128, 255),
    "L": (255, 128, 0),
}
locked_colors = {
    "I": (0, 128, 128),
    "O": (128, 128, 0),
    "T": (128, 0, 128),
    "S": (0, 128, 0),
    "Z": (128, 0, 0),
    "J": (0, 0, 128),
    "L": (128, 128, 128),
}
palette = [
    background_color,
    border_color,
    text_color,
    flash_color,
    *active_colors.values(),
    *locked_colors.values(),
]

cols = 10
rows = 20
cell = 3
board_x = 2
board_y = 2
preview_x = 40
preview_y = 2
preview_width = 16
preview_height = 10
counter_x = 42
counter_y = 15
counter_digits = 3
max_score = 10**counter_digits - 1
gravity_ticks = 6
flash_ticks = 3
piece_size = 4
spawn_col = 5
spawn_row = 0
empty_char = "."
flash_char = "#"
empty_line = empty_char * cols
empty_board = (empty_line,) * rows
visible_keys = ["board", "piece", "next", "score", "dead"]

# Spawn orientations as cell offsets from a fixed pivot, y pointing down
spawn_shapes = {
    "I": [(-2, 0), (-1, 0), (0, 0), (1, 0)],
    "O": [(-1, 0), (0, 0), (-1, 1), (0, 1)],
    "T": [(-1, 0), (0, 0), (1, 0), (0, 1)],
    "S": [(0, 0), (1, 0), (-1, 1), (0, 1)],
    "Z": [(-1, 0), (0, 0), (0, 1), (1, 1)],
    "J": [(-1, 0), (0, 0), (1, 0), (1, 1)],
    "L": [(-1, 0), (0, 0), (1, 0), (-1, 1)],
}
# Rotation states that look identical must behave identically, so I, S and Z only toggle
rotation_counts = {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}
piece_names = list(spawn_shapes)

# Scripted policy: a one-piece lookahead scoring Dellacherie's features with the El-Tetris weights
landing_weight = -4.500158825082766
lines_weight = 3.4181268101392694
row_transitions_weight = -3.2178882868487753
column_transitions_weight = -9.348695305445199
holes_weight = -7.899265427351652
wells_weight = -3.3855972247263626
# How the policy finishes a placement once in position; the rest of the time it waits for gravity
hard_drop_chance = 0.3
soft_drop_chance = 0.4
# Up to this stack height the policy keeps the right column open for four-line clears
stack_limit = 8
well_col = cols - 1
tetris_lines = 4
burn_penalty = 1000.0
evaluate_cache = 4096
# Rotations reach at most 2 rows above the pivot, so plans pad the board with this many free rows
headroom = 2


def rotations(offsets: list, count: int) -> list:
    states = [offsets]

    for _ in range(count - 1):
        states.append([(-dy, dx) for dx, dy in states[-1]])

    return states


shapes = {
    name: rotations(offsets, rotation_counts[name])
    for name, offsets in spawn_shapes.items()
}
cell_colors = {
    empty_char: background_color,
    flash_char: flash_color,
    **locked_colors,
    **{name.lower(): color for name, color in active_colors.items()},
}
cell_chars = {(r << 16) | (g << 8) | b: char for char, (r, g, b) in cell_colors.items()}
flash_table = str.maketrans(dict.fromkeys(piece_names, flash_char))


def width(offsets: list) -> int:
    return max(dx for dx, _ in offsets) - min(dx for dx, _ in offsets) + 1


# Coverage goals for the 1M-step dataset, checked by scripts/collect.py
targets = {
    "cleared lines": 20_000,
    "four-line clears": 400,
    "top-outs": 2_000,
    "placements": sum(
        cols - width(offsets) + 1 for states in shapes.values() for offsets in states
    ),
}


def preview_origin(name: str) -> tuple[int, int]:
    xs = [dx for dx, _ in spawn_shapes[name]]
    ys = [dy for _, dy in spawn_shapes[name]]
    x = preview_x + (preview_width - width(spawn_shapes[name]) * cell) // 2
    y = preview_y + (preview_height - (max(ys) - min(ys) + 1) * cell) // 2

    return x - min(xs) * cell, y - min(ys) * cell


preview_origins = {name: preview_origin(name) for name in piece_names}


def cells(piece: tuple) -> list:
    name, rotation, col, row = piece

    return [(col + dx, row + dy) for dx, dy in shapes[name][rotation]]


def spawn(name: str) -> tuple:
    return (name, 0, spawn_col, spawn_row)


def placement(piece: tuple) -> tuple:
    return (piece[0], piece[1], min(col for col, _ in cells(piece)))


def fits(board: tuple, piece: tuple) -> bool:
    return all(
        0 <= col < cols and 0 <= row < len(board) and board[row][col] == empty_char
        for col, row in cells(piece)
    )


# Blocked moves do nothing: no wall kicks
def move(board: tuple, piece: tuple, action: int) -> tuple:
    if action not in moves:
        return piece

    name, rotation, col, row = piece
    d_rotation, d_col, d_row = moves[action]
    moved = (
        name,
        (rotation + d_rotation) % len(shapes[name]),
        col + d_col,
        row + d_row,
    )

    return moved if fits(board, moved) else piece


def drop(board: tuple, piece: tuple) -> tuple:
    lower = move(board, piece, soft_drop_action)

    while lower != piece:
        piece, lower = lower, move(board, lower, soft_drop_action)

    return piece


# One tick on the active piece: the action first, then gravity; a blocked gravity step locks
def advance(board: tuple, piece: tuple, action: int, gravity: bool) -> tuple:
    if action == hard_drop_action:
        return drop(board, piece), True

    piece = move(board, piece, action)
    if not gravity:
        return piece, False
    lower = move(board, piece, soft_drop_action)

    return lower, lower == piece


def place(board: tuple, piece: tuple) -> tuple[tuple, int]:
    grid = list(board)

    for col, row in cells(piece):
        grid[row] = grid[row][:col] + piece[0] + grid[row][col + 1 :]

    kept = [line for line in grid if empty_char in line]
    cleared = rows - len(kept)

    return (empty_line,) * cleared + tuple(kept), cleared


# Lock, clear full rows and spawn the preview piece; the new piece is None on a top-out
def lock_piece(board: tuple, piece: tuple, next_name: str) -> tuple:
    board, cleared = place(board, piece)
    spawned = spawn(next_name)
    if fits(board, spawned):
        return board, cleared, spawned

    return tuple(line.translate(flash_table) for line in board), cleared, None


def draw(state: dict) -> np.ndarray:
    surface = pygame.Surface((frame_size, frame_size))
    surface.fill(background_color)
    pygame.draw.rect(
        surface,
        border_color,
        (board_x - 1, board_y - 1, cols * cell + 2, rows * cell + 2),
        width=1,
    )
    pygame.draw.rect(
        surface,
        border_color,
        (preview_x - 1, preview_y - 1, preview_width + 2, preview_height + 2),
        width=1,
    )

    for row, line in enumerate(state["board"]):
        for col, char in enumerate(line):
            if char != empty_char:
                surface.fill(
                    cell_colors[char],
                    (board_x + col * cell, board_y + row * cell, cell, cell),
                )

    if state["piece"] is not None:
        color = active_colors[state["piece"][0]]

        for col, row in cells(state["piece"]):
            surface.fill(
                color, (board_x + col * cell, board_y + row * cell, cell, cell)
            )

    x, y = preview_origins[state["next"]]

    for dx, dy in spawn_shapes[state["next"]]:
        surface.fill(
            active_colors[state["next"]], (x + dx * cell, y + dy * cell, cell, cell)
        )

    draw_number(surface, state["score"], counter_x, counter_y, text_color)

    return to_array(surface)


class Game:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.restart()

    def restart(self) -> None:
        self.board = empty_board
        self.score = 0
        self.flash = 0
        self.timer = 0
        self.placed = None
        self.cleared = 0
        self.piece = spawn(self.rng.choice(piece_names))
        self.next = self.rng.choice(piece_names)

    def step(self, action: int) -> int:
        self.placed = None
        self.cleared = 0
        if self.flash:
            self.flash -= 1
            if self.flash:
                return 0
            self.restart()
            return event_restart | event_random

        # The timer counts ticks since the last spawn or gravity step; soft drops leave it alone
        self.timer += 1
        gravity = self.timer == gravity_ticks
        if gravity:
            self.timer = 0
        self.piece, locked = advance(self.board, self.piece, action, gravity)
        if not locked:
            return 0

        self.placed = placement(self.piece)
        self.board, self.cleared, self.piece = lock_piece(
            self.board, self.piece, self.next
        )
        self.score = min(self.score + self.cleared, max_score)
        self.timer = 0
        flags = event_score if self.cleared else 0
        if self.piece is None:
            self.flash = flash_ticks
            return flags | event_death
        self.next = self.rng.choice(piece_names)

        return flags | event_random

    def render(self) -> np.ndarray:
        return draw(self.state())

    def state(self) -> dict:
        return {
            "board": self.board,
            "piece": self.piece,
            "next": self.next,
            "score": self.score,
            "dead": self.flash > 0,
            "flash": self.flash,
            "timer": None if self.flash else self.timer,
            "placed": self.placed,
            "cleared": self.cleared,
        }


def find_piece(name: str, active: list) -> tuple | None:
    left = min(col for col, _ in active)
    top = min(row for _, row in active)

    for rotation, offsets in enumerate(shapes[name]):
        piece = (
            name,
            rotation,
            left - min(dx for dx, _ in offsets),
            top - min(dy for _, dy in offsets),
        )
        if sorted(cells(piece)) == sorted(active):
            return piece

    return None


def extract(frame: np.ndarray) -> dict:
    # Top-left pixel of every cell
    samples = frame[
        board_y : board_y + rows * cell : cell, board_x : board_x + cols * cell : cell
    ].astype(np.int32)  # shape: (rows, cols, 3)
    codes = (samples[..., 0] << 16) | (samples[..., 1] << 8) | samples[..., 2]
    chars = [[cell_chars.get(code) for code in line] for line in codes.tolist()]
    if any(None in line for line in chars):
        raise ValueError("A board cell has a color outside the board palette")

    active = [
        (col, row)
        for row, line in enumerate(chars)
        for col, char in enumerate(line)
        if char.islower()
    ]
    names = {chars[row][col] for col, row in active}
    board = tuple(
        "".join(empty_char if char.islower() else char for char in line)
        for line in chars
    )

    preview = frame[
        preview_y : preview_y + preview_height, preview_x : preview_x + preview_width
    ]  # shape: (preview_height, preview_width, 3)
    shown = [
        name
        for name, color in active_colors.items()
        if np.all(preview == color, axis=-1).any()
    ]
    if len(shown) != 1:
        raise ValueError(f"The preview shows {len(shown)} piece colors, not 1")

    return {
        "board": board,
        # Anything but 4 cells of one color in a valid rotation is no piece, which check() flags
        "piece": (
            find_piece(names.pop().upper(), active)
            if len(names) == 1 and len(active) == piece_size
            else None
        ),
        "next": shown[0],
        "score": read_number(frame, counter_x, counter_y, text_color, counter_digits),
        "dead": any(flash_char in line for line in board),
    }


def spawned(before: dict, after: dict) -> bool:
    if after["dead"]:
        return False

    return before["dead"] or (
        after["piece"] == spawn(before["next"]) and after["board"] != before["board"]
    )


# Ticks since the last spawn or gravity step, or None if the history cannot tell
def known_timer(history: list) -> int | None:
    if history[-1].get("timer") is not None:
        return history[-1]["timer"]

    # Gravity runs every gravity_ticks after a spawn whatever the actions were, so a spawn in view fixes the phase
    for k in range(len(history) - 1, 0, -1):
        if spawned(history[k - 1], history[k]):
            return (len(history) - 1 - k) % gravity_ticks

    return None


# Rules broken if gravity did or did not run this tick, and whether the piece locked
def compare(prev: dict, action: int, gravity: bool, state: dict) -> tuple[dict, bool]:
    piece, locked = advance(prev["board"], prev["piece"], action, gravity)
    if not locked:
        return {
            "move": state["piece"] != piece,
            "locked": state["board"] != prev["board"],
            "score": state["score"] != prev["score"],
            "preview": state["next"] != prev["next"],
            "death": state["dead"],
        }, False

    board, cleared, spawned_piece = lock_piece(prev["board"], piece, prev["next"])
    score = min(prev["score"] + cleared, max_score)
    if spawned_piece is None:
        expected = {
            "board": board,
            "piece": None,
            "next": prev["next"],
            "score": score,
            "dead": True,
        }
        return {"death": any(state[key] != expected[key] for key in visible_keys)}, True

    # The new preview is a random draw, so any piece is fine there
    return {
        "clear" if cleared else "locked": state["board"] != board,
        "score": state["score"] != score,
        "spawn": state["piece"] != spawned_piece,
        "death": state["dead"],
    }, True


# History is the extracted states before this frame, oldest first; at least flash_ticks of them so the dead run is never cut off
def check(
    history: list, action: int, frame: np.ndarray
) -> tuple[list[str], dict | None]:
    try:
        state = extract(frame)
    except (IndexError, ValueError):
        return ["unreadable"], None

    prev = history[-1]
    broken = {
        "render": not np.array_equal(draw(state), frame),
        "piece": not state["dead"] and state["piece"] is None,
    }

    if prev["dead"]:
        dead_frames = len(list(takewhile(itemgetter("dead"), reversed(history))))
        if dead_frames < flash_ticks:
            broken["flash"] = any(state[key] != prev[key] for key in visible_keys)
            timer = None
        else:
            broken["restart"] = (
                state["dead"]
                or state["board"] != empty_board
                or state["score"] != 0
                or state["piece"] is None
                or state["piece"] != spawn(state["piece"][0])
            )
            timer = 0
    else:
        timer = known_timer(history)
        expected = [False, True] if timer is None else [timer + 1 == gravity_ticks]
        outcomes = {
            gravity: compare(prev, action, gravity, state) for gravity in (False, True)
        }
        gravity = next(
            (g for g in expected if not any(outcomes[g][0].values())), expected[0]
        )
        failed, locked = outcomes[gravity]

        # A frame that only fits the other gravity choice got the timer wrong
        if any(failed.values()) and not any(outcomes[not gravity][0].values()):
            failed = {"gravity": True}
        broken.update(failed)

        if state["dead"]:
            timer = None
        elif gravity or locked:
            timer = 0
        elif timer is not None:
            timer += 1

    return [name for name, hit in broken.items() if hit], {
        **state,
        "timer": timer,
    }


def heights(board: tuple) -> list:
    return [len("".join(column).lstrip(empty_char)) for column in zip(*board)]


# While stacking, the well column counts as a wall and filling it short of four lines is a burn
@lru_cache(maxsize=evaluate_cache)
def evaluate(board: tuple, piece: tuple, stacking: bool) -> float:
    after, cleared = place(board, piece)
    span = well_col if stacking else cols
    filled = [[char != empty_char for char in line[:span]] for line in after]
    landing = rows - sum(row for _, row in cells(piece)) / piece_size
    row_transitions = sum(
        sum(a != b for a, b in pairwise([True, *line, True]))
        for line in filled
        if any(line)
    )
    column_transitions = 0
    holes = 0
    wells = 0

    for col, column in enumerate(zip(*filled)):
        column_transitions += sum(a != b for a, b in pairwise([*column, True]))
        holes += column[column.index(True) :].count(False) if True in column else 0
        depth = 0

        # A well cell is empty with both sides filled; a well of depth d adds 1 + 2 + ... + d
        for row, full in enumerate(column):
            walled = (col == 0 or filled[row][col - 1]) and (
                col == span - 1 or filled[row][col + 1]
            )
            depth = depth + 1 if walled and not full else 0
            wells += depth

    burned = cleared < tetris_lines and any(col == well_col for col, _ in cells(piece))

    return (
        landing_weight * landing
        + lines_weight * cleared
        + row_transitions_weight * row_transitions
        + column_transitions_weight * column_transitions
        + holes_weight * holes
        + wells_weight * wells
        - (burn_penalty if stacking and burned else 0)
    )


# Best landing among the rotations and columns reachable from the piece's row, and the first move toward it
def plan(board: tuple, piece: tuple) -> tuple[tuple, int | None]:
    # Rows above the board count as free, so a rotation the top still blocks only waits
    padded = (empty_line,) * headroom + board
    name, rotation, col, row = piece
    start = (name, rotation, col, row + headroom)
    first = {start: None}
    queue = deque([start])

    while queue:
        current = queue.popleft()

        for action in (rotate_action, left_action, right_action):
            moved = move(padded, current, action)
            if moved not in first:
                first[moved] = first[current] or action
                queue.append(moved)

    stacking = max(heights(board)[:well_col]) <= stack_limit
    scores = {}

    for target in first:
        name, rotation, col, row = drop(padded, target)
        landing = (name, rotation, col, row - headroom)
        if all(row >= 0 for _, row in cells(landing)):
            scores[target] = evaluate(board, landing, stacking)

    best = max(scores, key=scores.get)

    return best, first[best]


def policy(state: dict, rng: random.Random) -> int:
    piece = state["piece"]
    if piece is None:
        return 0

    board = state["board"]
    target, action = plan(board, piece)

    # In place: finish with a mix of hard drops, soft drops and waiting for gravity
    if action is None:
        roll = rng.random()
        if roll < hard_drop_chance:
            return hard_drop_action
        return soft_drop_action if roll < hard_drop_chance + soft_drop_chance else 0
    if move(board, piece, action) != piece:
        return action

    # The top blocks this rotation for now: shift toward the target meanwhile, else move down
    shift = left_action if target[2] < piece[2] else right_action
    if target[2] != piece[2] and move(board, piece, shift) != piece:
        return shift

    return soft_drop_action


# Coverage keys for one transition; scripts/collect.py sums them over the data
def tally(state: dict, flags: int) -> list:
    keys = []

    if state["placed"] is not None:
        keys.append(("placed", *state["placed"]))
    if state["cleared"]:
        keys.append(("cleared", state["cleared"]))
    if flags & event_death:
        keys.append(("top-out",))

    return keys


def coverage(counts: Counter) -> dict:
    return {
        "cleared lines": sum(
            key[1] * count for key, count in counts.items() if key[0] == "cleared"
        ),
        "four-line clears": counts[("cleared", 4)],
        "top-outs": counts[("top-out",)],
        "placements": sum(1 for key in counts if key[0] == "placed"),
    }
