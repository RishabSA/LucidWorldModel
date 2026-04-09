import random
from collections import Counter
from itertools import takewhile
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

tick_hz = 30
context = 4
actions = [0, 1, 2]
input_mode = "hold"
# Coverage goals for collected data, checked by scripts/collect.py
targets = {
    "cpu points": 10_000,
    "player points": 10_000,
    "fewest zone hits": 1,
    "match ends": 500,
}

background_color = (16, 16, 24)
wall_color = (64, 72, 128)
paddle_color = (48, 160, 232)
ball_color = (240, 232, 48)
text_color = (240, 240, 240)
flash_color = (168, 48, 232)
palette = [
    background_color,
    wall_color,
    paddle_color,
    ball_color,
    text_color,
    flash_color,
]

top_wall = 7
bottom_wall = 63
field_top = top_wall + 1
ball_size = 2
paddle_width = 2
paddle_height = 10
ball_max_y = bottom_wall - ball_size
paddle_max_y = bottom_wall - paddle_height
cpu_x = 1
player_x = frame_size - 1 - paddle_width
# Ball columns that touch each paddle; the ball's x stays odd, so it always lands on them
cpu_contact = cpu_x + paddle_width
player_contact = player_x - ball_size
center = (31, 34)
paddle_start = 30
paddle_speed = 2
cpu_speed = 1
ball_speed = 2
serve_vy = [-1, 0, 1]
# New vy by the ball's center row relative to the paddle top: fifths of the paddle, the middle one gets the 11th row
zone_vy = [-2, -2, -1, -1, 0, 0, 0, 1, 1, 2, 2]
win_score = 9
flash_ticks = 3
score_xs = {"cpu": 24, "player": 37}
score_y = 1
score_digits = 1
line_x = 31
line_width = 2
dash_length = 4
up = action_names.index("up")
down = action_names.index("down")
paddle_moves = {up: -paddle_speed, down: paddle_speed}
# The serve frame shows the ball one step from the center, so its position gives away the drawn velocity
serves = [
    (center[0] + vx, center[1] + vy)
    for vx in (-ball_speed, ball_speed)
    for vy in serve_vy
]

# Scripted player: paddle tops it rests in, and the chance it acts on a tick; a lazy player misses far balls
home = (paddle_start - 1, paddle_start + 1)
focus = 0.4


def clamp(paddle: int) -> int:
    return max(field_top, min(paddle_max_y, paddle))


# Walls flip vy before the move, so the step between two frames is always the velocity
def bounce(y: int, vy: int) -> int:
    return vy if field_top <= y + vy <= ball_max_y else -vy


# One tick of everything but the serve, decided from the current frame alone
def advance(state: dict, action: int) -> dict:
    (x, y), (vx, vy) = state["ball"], state["velocity"]
    side, other = ("cpu", "player") if vx < 0 else ("player", "cpu")
    contact = cpu_contact if vx < 0 else player_contact
    offset = y + ball_size // 2 - state[side]
    hit = scorer = None

    if x == contact and 0 <= offset < len(zone_vy):
        hit = (side, zone_vy[offset])
        vx, vy = -vx, zone_vy[offset]
    elif x == contact:
        scorer = other
    vy = bounce(y, vy)

    # The CPU follows the ball while it approaches, otherwise drifts back to the middle
    approaching = state["velocity"][0] < 0
    target = (
        clamp(y + ball_size // 2 - paddle_height // 2) if approaching else paddle_start
    )
    cpu = state["cpu"] + max(-cpu_speed, min(cpu_speed, target - state["cpu"]))

    return {
        "ball": None if scorer else (x + vx, y + vy),
        "velocity": None if scorer else (vx, vy),
        "cpu": cpu,
        "player": clamp(state["player"] + paddle_moves.get(action, 0)),
        "hit": hit,
        "scorer": scorer,
    }


def draw(state: dict) -> np.ndarray:
    surface = pygame.Surface((frame_size, frame_size))
    surface.fill(flash_color if state["dead"] else background_color)
    surface.fill(wall_color, (0, top_wall, frame_size, 1))
    surface.fill(wall_color, (0, bottom_wall, frame_size, 1))

    for y in range(field_top + 1, bottom_wall, 2 * dash_length):
        surface.fill(wall_color, (line_x, y, line_width, dash_length))

    for side, x in score_xs.items():
        draw_number(surface, state["scores"][side], x, score_y, text_color)

    surface.fill(paddle_color, (cpu_x, state["cpu"], paddle_width, paddle_height))
    surface.fill(paddle_color, (player_x, state["player"], paddle_width, paddle_height))
    if state["ball"] is not None:
        surface.fill(ball_color, (*state["ball"], ball_size, ball_size))

    return to_array(surface)


class Game:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.restart()

    def restart(self) -> None:
        self.cpu = self.player = paddle_start
        self.scores = {"cpu": 0, "player": 0}
        self.flash = 0
        self.hit = None
        self.scorer = None
        self.serve()

    def serve(self) -> None:
        self.velocity = (
            self.rng.choice([-ball_speed, ball_speed]),
            self.rng.choice(serve_vy),
        )
        self.ball = (center[0] + self.velocity[0], center[1] + self.velocity[1])

    def step(self, action: int) -> int:
        if self.flash:
            self.flash -= 1
            if self.flash:
                return 0
            self.restart()
            return event_restart | event_random

        moved = advance(self.state(), action)
        self.ball, self.velocity = moved["ball"], moved["velocity"]
        self.cpu, self.player = moved["cpu"], moved["player"]
        self.hit, self.scorer = moved["hit"], moved["scorer"]
        if self.scorer is None:
            return 0

        self.scores[self.scorer] += 1
        if self.scores[self.scorer] == win_score:
            self.flash = flash_ticks
            return event_score | event_death
        self.serve()

        return event_score | event_random

    def render(self) -> np.ndarray:
        return draw(self.state())

    def state(self) -> dict:
        return {
            "ball": self.ball,
            "velocity": self.velocity,
            "cpu": self.cpu,
            "player": self.player,
            "scores": dict(self.scores),
            "score": sum(self.scores.values()),
            "dead": self.flash > 0,
            "flash": self.flash,
            "hit": self.hit,
            "scorer": self.scorer,
        }


def extract(frame: np.ndarray) -> dict:
    ball = np.argwhere(np.all(frame == ball_color, axis=-1))
    paddles = {
        side: int(np.flatnonzero(np.all(frame[:, x] == paddle_color, axis=-1))[0])
        for side, x in [("cpu", cpu_x), ("player", player_x)]
    }

    return {
        "ball": (int(ball[0][1]), int(ball[0][0])) if len(ball) else None,
        **paddles,
        "scores": {
            side: read_number(frame, x, score_y, text_color, score_digits)
            for side, x in score_xs.items()
        },
        "dead": bool(np.all(frame[0, 0] == flash_color)),
    }


# The step into this frame; a serve starts one step from the center
def velocity(before: dict, after: dict) -> tuple | None:
    if after["ball"] is None:
        return None

    serving = before["ball"] is None or before["scores"] != after["scores"]
    start = center if serving else before["ball"]

    return (after["ball"][0] - start[0], after["ball"][1] - start[1])


# History is the extracted states before this frame, oldest first; at least flash_ticks of them so the flash is never cut off
def check(
    history: list, action: int, frame: np.ndarray
) -> tuple[list[str], dict | None]:
    try:
        state = extract(frame)
    except (IndexError, ValueError):
        return ["unreadable"], None

    prev = history[-1]
    broken = {"render": not np.array_equal(draw(state), frame)}

    if prev["dead"]:
        dead_frames = len(list(takewhile(itemgetter("dead"), reversed(history))))
        if dead_frames < flash_ticks:
            broken["flash"] = state != {key: prev[key] for key in state}
        else:
            broken["restart"] = (
                state["dead"]
                or state["scores"] != {"cpu": 0, "player": 0}
                or state["cpu"] != paddle_start
                or state["player"] != paddle_start
                or state["ball"] not in serves
            )
    elif len(history) > 1:
        expected = advance({**prev, "velocity": velocity(history[-2], prev)}, action)
        scorer = expected["scorer"]
        scores = dict(prev["scores"])
        if scorer:
            scores[scorer] += 1
        ending = scorer is not None and scores[scorer] == win_score

        broken["cpu"] = state["cpu"] != expected["cpu"]
        broken["player"] = state["player"] != expected["player"]
        broken["score"] = state["scores"] != scores
        broken["death"] = state["dead"] != ending
        if scorer is None:
            broken["ball"] = state["ball"] != expected["ball"]
        elif ending:
            broken["ball"] = state["ball"] is not None
        else:
            broken["serve"] = state["ball"] not in serves

    # The policy needs the velocity, which one frame cannot show
    state["velocity"] = velocity(prev, state)

    return [name for name, failed in broken.items() if failed], state


def policy(state: dict, rng: random.Random) -> int:
    # No ball during the flash; extract() alone (the first dream frame) has no velocity
    if state.get("velocity") is None or rng.random() >= focus:
        return 0

    (x, y), (vx, vy) = state["ball"], state["velocity"]
    paddle = state["player"]
    low, high = home

    if vx > 0:
        # Where the ball meets the paddle
        while x < player_contact:
            vy = bounce(y, vy)
            x, y = x + vx, y + vy

        # Aim the paddle edge nearer the ball at it: the CPU misses those steep returns
        middle = y + ball_size // 2 - paddle_height // 2
        edge = max(zone_vy) if paddle < middle else min(zone_vy)
        tops = [
            y + ball_size // 2 - offset
            for offset, zone in enumerate(zone_vy)
            if zone == edge
        ]
        low, high = min(tops), max(tops)

    if paddle < low:
        return down
    if paddle > high:
        return up

    return 0


# Coverage keys for one transition; scripts/collect.py sums them over the data
def tally(state: dict, flags: int) -> list:
    keys = []

    if state["hit"] is not None:
        keys.append(("hit", *state["hit"]))
    if flags & event_score:
        keys.append(("point", state["scorer"]))
    if flags & event_death:
        keys.append(("match end",))

    return keys


def coverage(counts: Counter) -> dict:
    return {
        "cpu points": counts[("point", "cpu")],
        "player points": counts[("point", "player")],
        "fewest zone hits": min(
            counts[("hit", side, zone)] for side in score_xs for zone in set(zone_vy)
        ),
        "match ends": counts[("match end",)],
    }
