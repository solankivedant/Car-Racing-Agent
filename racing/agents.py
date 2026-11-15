"""The car agents.

=================  ==========================================================
ReflexAgent        simple reflex agent: percepts -> propositions -> rule base
PathAgent          A* path planning + path following, with reflex *safety*
                   rules that override the planner ("Rules: safety / follow path")
UtilityAgent       model-based utility agent: tries every action with a short
                   look-ahead and maximises U = progress - collision risk + speed
NeuroAgent         learning agent: small neural network whose weights are
                   learned by repeatedly running cars (genetic algorithm, see
                   learning.py)
=================  ==========================================================

Every agent implements ``reset(track)`` and ``act(state) -> (steer, throttle)``
and exposes ``self.info`` (the rule that fired, utilities, ...) for the HUD.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from . import car as C
from .car import (ACCELERATE, ACTIONS, BRAKE, BRAKE_A, COAST, LEFT, RIGHT, STRAIGHT, V_MAX,
                  CarState, action_name)
from .logic import RuleBase, rule
from .planner import ADMISSIBLE, HEURISTICS, SPACING, Grid, RacingLine, plan_lap
from .track import Track

AGENT_COLORS = {
    "reflex": (66, 135, 245), "utility": (245, 166, 35), "neuro": (46, 204, 113),
    "h0": (200, 200, 200), "h1": (231, 76, 60), "h2": (155, 89, 182),
    "h3": (241, 196, 15), "h4": (236, 64, 122),
}


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def stopping_distance(v: float, margin: float = 26.0) -> float:
    return v * v / (2 * BRAKE * 0.85) + margin


class Agent:
    kind = "agent"

    def __init__(self, name: str, color=None):
        self.name = name
        self.color = color or (220, 220, 220)
        self.info: Dict = {}

    def reset(self, track: Track):
        self.track = track
        self.info = {}

    def act(self, s: CarState):
        raise NotImplementedError


# ============================================================================ reflex
def percept_atoms(s: CarState) -> set:
    """Turn the raw sensor readings into propositions (the agent's facts)."""
    l, fl, f, fr, r = s.rays
    v = s.speed
    d_stop = stopping_distance(v)
    facts = {
        "FrontBlocked": f < d_stop,
        "FrontNear": f < d_stop + 70,
        "LeftMoreOpen": (l + 1.4 * fl) > (r + 1.4 * fr),
        "LeftTight": l < 15,          # thresholds scaled to the car body (14 px wide)
        "RightTight": r < 15,
        "DiagLeftTight": fl < 30,
        "DiagRightTight": fr < 30,
        "Slow": v < 45,
        "TooFast": v > math.sqrt(max(0.0, 2 * BRAKE * 0.7 * (f - 30))) + 10,
        "VeerLeft": fl - fr > 22,      # more room front-left: the road bends left
        "VeerRight": fr - fl > 22,
    }
    return {k for k, val in facts.items() if val}


def _toward_open(throttle):
    def fn(ctx):
        thr = throttle
        if throttle == COAST and "Slow" in ctx["facts"]:
            thr = ACCELERATE          # coasting at standstill would never move the car
        return (LEFT if "LeftMoreOpen" in ctx["facts"] else RIGHT, thr)
    return fn


def _steer_coast(steer):
    def fn(ctx):
        return (steer, ACCELERATE if "Slow" in ctx["facts"] else COAST)
    return fn


REFLEX_RULES = RuleBase([
    rule("R1", "turn to open side + BRAKE", "FrontBlocked & ~Slow", _toward_open(BRAKE_A)),
    rule("R2", "turn to open side + crawl", "FrontBlocked & Slow", _toward_open(ACCELERATE)),
    rule("R3", "RIGHT (left wall close)", "DiagLeftTight & ~DiagRightTight", _steer_coast(RIGHT)),
    rule("R4", "LEFT (right wall close)", "DiagRightTight & ~DiagLeftTight", _steer_coast(LEFT)),
    rule("R5", "RIGHT (side wall)", "LeftTight & ~RightTight", (RIGHT, ACCELERATE)),
    rule("R6", "LEFT (side wall)", "RightTight & ~LeftTight", (LEFT, ACCELERATE)),
    rule("R7", "bend ahead, too fast: steer + BRAKE", "FrontNear & TooFast", _toward_open(BRAKE_A)),
    rule("R8", "bend ahead: steer + COAST", "FrontNear & ~TooFast", _toward_open(COAST)),
    rule("R9", "follow bend LEFT", "VeerLeft", (LEFT, ACCELERATE)),
    rule("R10", "follow bend RIGHT", "VeerRight", (RIGHT, ACCELERATE)),
    rule("R11", "STRAIGHT + ACCELERATE", "", (STRAIGHT, ACCELERATE)),
])


class ReflexAgent(Agent):
    """Pure condition-action rules over the five ray sensors. No map, no plan."""
    kind = "reflex"

    def __init__(self, name="Reflex", rules: RuleBase = REFLEX_RULES):
        super().__init__(name, AGENT_COLORS["reflex"])
        self.rules = rules

    def act(self, s: CarState):
        facts = percept_atoms(s)
        a, r = self.rules.decide(facts, {"facts": facts, "state": s})
        self.info = {"rule": r.rid, "formula": r.formula, "facts": sorted(facts),
                     "action": action_name(*a)}
        return a


# ============================================================================ A* + safety
SAFETY_RULES = RuleBase([
    rule("S1", "SAFETY: emergency brake + evade", "FrontBlocked & ~Slow & OffLine",
         _toward_open(BRAKE_A)),
    rule("S2", "SAFETY: evade left wall", "LeftTight & ~RightTight", _steer_coast(RIGHT)),
    rule("S3", "SAFETY: evade right wall", "RightTight & ~LeftTight", _steer_coast(LEFT)),
    rule("P1", "FOLLOW PATH", "", None),
])

_PLAN_CACHE: Dict[tuple, tuple] = {}


def corner_speed(curv: np.ndarray, factor: float = 0.82) -> np.ndarray:
    """Fastest speed at which the car can still follow curvature ``curv``."""
    r = 1.0 / np.maximum(np.abs(curv), 1e-6)
    v = np.full_like(r, V_MAX)
    for _ in range(8):                         # fixed point of v = turn_rate(v) * r
        v = np.minimum(V_MAX, C.turn_rate(v) * r)
    return v * factor


class PathAgent(Agent):
    """Deliberative agent: A* plans a lap once, then a pure-pursuit controller
    follows the racing line with a speed profile. Reflex safety rules override it."""
    kind = "astar"

    def __init__(self, heuristic: str = "h4", name: Optional[str] = None):
        if heuristic not in HEURISTICS:
            raise ValueError(heuristic)
        super().__init__(name or f"A* {heuristic}", AGENT_COLORS[heuristic])
        self.heuristic = heuristic

    def reset(self, track: Track):
        super().reset(track)
        key = (track.file or track.name, track.length, self.heuristic)
        if key not in _PLAN_CACHE:
            grid, res, line = plan_lap(track, self.heuristic)
            _PLAN_CACHE[key] = (grid, res, line)
        self.grid, self.plan, self.line = _PLAN_CACHE[key]
        self.vc = corner_speed(self.line.curvature)
        self.idx = None

    def _speed_limit(self, i: int) -> float:
        n = len(self.line.pts)
        k = int(320 / SPACING)
        j = (i + np.arange(k)) % n
        dist = np.arange(k) * SPACING
        return float(np.min(np.sqrt(self.vc[j] ** 2 + 2 * BRAKE * 0.85 * dist)))

    def act(self, s: CarState):
        line = self.line
        n = len(line.pts)
        if self.idx is None:
            d = ((line.pts - (s.x, s.y)) ** 2).sum(1)
            # on the starting grid the car is behind the line: prefer the start of the path
            near_start = d[: n // 8]
            self.idx = int(np.argmin(near_start)) if near_start.min() < 60 ** 2 else int(np.argmin(d))
        self.idx = line.index_near(s.x, s.y, self.idx)
        off = math.sqrt(((line.pts[self.idx] - (s.x, s.y)) ** 2).sum())
        look = 26 + 0.16 * s.speed
        tgt = line.pts[(self.idx + int(look / SPACING)) % n]
        alpha = wrap(math.atan2(tgt[1] - s.y, tgt[0] - s.x) - s.heading)
        steer = RIGHT if alpha > 0.05 else LEFT if alpha < -0.05 else STRAIGHT
        v_lim = self._speed_limit(self.idx)
        throttle = ACCELERATE if s.speed < v_lim - 4 else BRAKE_A if s.speed > v_lim + 4 else COAST

        facts = percept_atoms(s)
        if off > 14:
            facts.add("OffLine")
        r = SAFETY_RULES.first_match(facts)
        if r.rid != "P1":
            a = r.resolve({"facts": facts})
        else:
            a = (steer, throttle)
        self.info = {"rule": r.rid, "formula": r.formula, "action": action_name(*a),
                     "v_limit": round(v_lim), "alpha": round(alpha, 2), "line_idx": self.idx,
                     "facts": sorted(facts)}
        return a


# ============================================================================ utility
class UtilityAgent(Agent):
    """Model-based utility agent.

    For each of the 9 (steer, throttle) actions it simulates the car ``horizon``
    ticks ahead with the physics model, then scores the outcome::

        U = W_PROGRESS * progress  +  W_SPEED * speed
            - W_COLLISION * sum_i w_i * exp(-ray_i / SIGMA)      (sensor weights w1..w5)
            - W_OVERSPEED * max(0, stopping_distance - front_ray)
            - CRASH  if the simulated car leaves the track

    and executes the action with the highest utility.
    """
    kind = "utility"
    SENSOR_WEIGHTS = np.array([0.6, 1.2, 2.0, 1.2, 0.6])     # w1..w5
    W_PROGRESS, W_SPEED, W_COLLISION, W_OVERSPEED = 1.0, 0.15, 6.0, 0.6
    SIGMA = 18.0
    CRASH = 1e4

    def __init__(self, name="Utility", horizon: int = 10):
        super().__init__(name, AGENT_COLORS["utility"])
        self.horizon = horizon

    def utilities(self, s: CarState):
        tr = self.track
        n = len(ACTIONS)
        x = np.full(n, s.x); y = np.full(n, s.y); h = np.full(n, s.heading); v = np.full(n, s.speed)
        st = np.array([a[0] for a in ACTIONS], float)
        th = np.array([a[1] for a in ACTIONS], float)
        crashed = np.zeros(n, bool)
        a0 = tr.arc_at(x, y).astype(float)
        prog = np.zeros(n)
        prev = a0.copy()
        for _ in range(self.horizon):
            x, y, h, v = C.step_physics(x, y, h, v, st, th)
            crashed |= C.collides(tr, x, y, h)
            a = tr.arc_at(x, y).astype(float)
            d = a - prev
            d = np.where(d < -tr.length / 2, d + tr.length, np.where(d > tr.length / 2, d - tr.length, d))
            prog += d
            prev = a
        rays = C.raycast(tr, x, y, h)
        risk = (np.exp(-rays / self.SIGMA) * self.SENSOR_WEIGHTS).sum(1)
        over = np.maximum(0, np.array([stopping_distance(vv) for vv in v]) - rays[:, 2])
        U = (self.W_PROGRESS * prog + self.W_SPEED * v - self.W_COLLISION * risk
             - self.W_OVERSPEED * over - self.CRASH * crashed)
        return U

    def act(self, s: CarState):
        U = self.utilities(s)
        k = int(np.argmax(U))
        a = ACTIONS[k]
        self.info = {"rule": "argmax U", "action": action_name(*a),
                     "utilities": {action_name(*ACTIONS[i]): round(float(U[i]), 1) for i in range(len(U))}}
        return a


# ============================================================================ learned
N_IN, N_HIDDEN, N_OUT = 7, 10, 2


def n_params() -> int:
    return N_IN * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUT + N_OUT


def nn_inputs(rays: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """rays (N,5), speed (N,) -> network input (N, 7)."""
    rays = np.atleast_2d(rays) / C.RAY_MAX
    sp = np.atleast_1d(speed)[:, None] / V_MAX
    return np.hstack([rays, sp, np.ones_like(sp)])


def nn_forward(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Batched forward pass. W: (P, n_params) or (n_params,), X: (P, 7) -> (P, 2)."""
    W = np.atleast_2d(W)
    P = W.shape[0]
    i = 0
    W1 = W[:, i:i + N_IN * N_HIDDEN].reshape(P, N_IN, N_HIDDEN); i += N_IN * N_HIDDEN
    b1 = W[:, i:i + N_HIDDEN]; i += N_HIDDEN
    W2 = W[:, i:i + N_HIDDEN * N_OUT].reshape(P, N_HIDDEN, N_OUT); i += N_HIDDEN * N_OUT
    b2 = W[:, i:i + N_OUT]
    hdn = np.tanh(np.einsum("pi,pij->pj", X, W1) + b1)
    return np.tanh(np.einsum("pi,pij->pj", hdn, W2) + b2)


def nn_to_actions(out: np.ndarray):
    steer = np.where(out[:, 0] > 0.25, RIGHT, np.where(out[:, 0] < -0.25, LEFT, STRAIGHT))
    thr = np.where(out[:, 1] > 0.0, ACCELERATE, np.where(out[:, 1] < -0.5, BRAKE_A, COAST))
    return steer, thr


class NeuroAgent(Agent):
    """Learning agent: 7-10-2 tanh network (5 rays + speed + bias -> steer, throttle).
    Its weights come from the genetic algorithm in learning.py."""
    kind = "neuro"

    def __init__(self, weights: np.ndarray, name="Learned (GA)"):
        super().__init__(name, AGENT_COLORS["neuro"])
        self.W = np.asarray(weights, float)

    def act(self, s: CarState):
        out = nn_forward(self.W, nn_inputs(s.rays[None, :], np.array([s.speed])))
        st, th = nn_to_actions(out)
        a = (int(st[0]), int(th[0]))
        self.info = {"rule": "neural net", "action": action_name(*a),
                     "net_out": [round(float(o), 2) for o in out[0]]}
        return a


def default_lineup(neuro_weights: Optional[np.ndarray] = None,
                   heuristics=("h1", "h2", "h3", "h4")) -> List[Agent]:
    cars: List[Agent] = [PathAgent(h) for h in heuristics]
    cars += [UtilityAgent(), ReflexAgent()]
    if neuro_weights is not None:
        cars.append(NeuroAgent(neuro_weights))
    return cars


_ = (ADMISSIBLE, Grid, RacingLine)
