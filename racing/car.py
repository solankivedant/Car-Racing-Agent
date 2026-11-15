"""Car state, physics and sensors, vectorised over many cars.

State of every car (the proposal's "State"):
    position (x, y), direction (heading, rad; 0 = east, y axis points down),
    speed (px/s), velocity (vx, vy) = speed * (cos h, sin h), sensor distances (5 rays)

Actions (the proposal's "Actions"): every tick a car chooses a steering action and a
throttle action::

    steer    : LEFT (-1)       STRAIGHT (0)   RIGHT (+1)
    throttle : BRAKE (-1)      COAST (0)      ACCELERATE (+1)

Sensors: five distance rays w1..w5 at -90, -45, 0, +45, +90 degrees relative to
the heading (left ... right), as in the proposal sketch. A ray stops at the first
pixel that is not drivable (wall, grass or obstacle) or at ``RAY_MAX``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .track import Track

DT = 1 / 30            # simulation step (s)
V_MAX = 260.0          # px/s
ACCEL = 170.0          # px/s^2
BRAKE = 420.0          # px/s^2
DRAG = 0.25            # 1/s rolling + air drag
TURN = 2.8             # rad/s at low/medium speed
TURN_V0 = 40.0         # below this speed steering is weaker (no spinning on the spot)
CAR_L, CAR_W = 28.0, 14.0
RAY_ANGLES = np.radians([-90, -45, 0, 45, 90])
RAY_NAMES = ("w1_left", "w2_front_left", "w3_front", "w4_front_right", "w5_right")
RAY_MAX = 220.0
RAY_STEP = 3.0
_RAY_D = np.arange(RAY_STEP, RAY_MAX + RAY_STEP, RAY_STEP)

LEFT, STRAIGHT, RIGHT = -1, 0, 1
BRAKE_A, COAST, ACCELERATE = -1, 0, 1
STEER_NAMES = {LEFT: "LEFT", STRAIGHT: "STRAIGHT", RIGHT: "RIGHT"}
THROTTLE_NAMES = {BRAKE_A: "BRAKE", COAST: "COAST", ACCELERATE: "ACCELERATE"}
ACTIONS = [(s, t) for s in (LEFT, STRAIGHT, RIGHT) for t in (ACCELERATE, COAST, BRAKE_A)]


def action_name(steer: int, throttle: int) -> str:
    return f"{STEER_NAMES[int(steer)]}+{THROTTLE_NAMES[int(throttle)]}"


def turn_rate(speed):
    """Max yaw rate: weak when almost stopped, reduced at high speed (less grip)."""
    speed = np.asarray(speed, dtype=float)
    low = np.clip(speed / TURN_V0, 0.15, 1.0)
    high = 1.0 - 0.45 * np.clip(speed / V_MAX, 0, 1)
    return TURN * low * high


def step_physics(x, y, h, v, steer, throttle, dt=DT):
    """Advance arrays of cars by one tick. Pure function (returns new arrays)."""
    steer = np.asarray(steer, dtype=float)
    throttle = np.asarray(throttle, dtype=float)
    a = np.where(throttle > 0, ACCEL, np.where(throttle < 0, -BRAKE, 0.0)) - DRAG * v
    v = np.clip(v + a * dt, 0.0, V_MAX)
    h = h + steer * turn_rate(v) * dt
    x = x + np.cos(h) * v * dt
    y = y + np.sin(h) * v * dt
    return x, y, h, v


def corners(x, y, h):
    """(N, 5, 2) body sample points: centre + 4 corners."""
    x, y, h = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(h)
    c, s = np.cos(h), np.sin(h)
    out = [np.stack([x, y], -1)]
    for fl, fw in ((0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)):
        dx = fl * CAR_L * c - fw * CAR_W * s
        dy = fl * CAR_L * s + fw * CAR_W * c
        out.append(np.stack([x + dx, y + dy], -1))
    return np.stack(out, 1)


def collides(track: Track, x, y, h):
    p = corners(x, y, h)
    return ~track.inside(p[..., 0], p[..., 1]).all(axis=1)


def raycast(track: Track, x, y, h):
    """Distances (N, 5) of the five sensors. Fully vectorised."""
    x, y, h = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(h)
    ang = h[:, None] + RAY_ANGLES[None, :]                      # (N, 5)
    px = x[:, None, None] + np.cos(ang)[..., None] * _RAY_D     # (N, 5, K)
    py = y[:, None, None] + np.sin(ang)[..., None] * _RAY_D
    free = track.inside(px, py)
    hit = ~free
    first = np.where(hit.any(-1), hit.argmax(-1), len(_RAY_D))
    d = np.where(first == 0, 0.0, _RAY_D[np.maximum(first - 1, 0)])
    return np.where(first == len(_RAY_D), RAY_MAX, d)


def ray_endpoints(x, y, h, dists):
    ang = h + RAY_ANGLES
    return [(x + math.cos(a) * d, y + math.sin(a) * d) for a, d in zip(ang, dists)]


@dataclass
class CarState:
    """Read-only snapshot handed to an agent each tick (its percept)."""
    x: float
    y: float
    heading: float
    speed: float
    rays: np.ndarray          # (5,) w1..w5
    progress: float           # px driven along the lap (can be negative on the grid)
    lap_length: float
    t: float

    @property
    def vx(self):
        return self.speed * math.cos(self.heading)

    @property
    def vy(self):
        return self.speed * math.sin(self.heading)
