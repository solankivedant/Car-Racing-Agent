"""Race simulation: many cars, one track, sense-act loop at 30 Hz.

Cars are "ghosts" (time-trial style): they cannot hit each other, only the walls
and obstacles, so every agent is judged on its own driving. A car that touches
a wall is out (CRASHED). A car that makes no progress for ``STUCK_S`` seconds is
out (STUCK). A car finishes when it has driven ``laps * track.length`` px along
the lap and crosses the finish line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import car as C
from .agents import Agent
from .track import Track

RACING, FINISHED, CRASHED, STUCK, TIMEOUT = "RACING", "FINISHED", "CRASHED", "STUCK", "TIMEOUT"
STUCK_S = 6.0


@dataclass
class CarRecord:
    agent: Agent
    x: float
    y: float
    heading: float
    speed: float = 0.0
    progress: float = 0.0
    last_arc: float = 0.0
    status: str = RACING
    finish_time: Optional[float] = None
    end_time: Optional[float] = None
    best_progress: float = 0.0
    best_t: float = 0.0
    distance: float = 0.0
    steer: int = 0
    throttle: int = 0
    rays: np.ndarray = field(default_factory=lambda: np.zeros(5))
    trail: List[tuple] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    actions: List[tuple] = field(default_factory=list)
    rules: dict = field(default_factory=dict)
    start_progress: float = 0.0

    @property
    def name(self):
        return self.agent.name

    @property
    def alive(self):
        return self.status == RACING


class Race:
    def __init__(self, track: Track, agents: List[Agent], max_time: float = 120.0,
                 record: bool = True):
        self.track = track
        self.agents = agents
        self.max_time = max_time
        self.record = record
        self.t = 0.0
        self.ticks = 0
        self.cars: List[CarRecord] = []
        slots = track.grid_slots(len(agents))
        for ag, (x, y, h, s0) in zip(agents, slots):
            ag.reset(track)
            arc = float(track.arc_at(x, y))
            c = CarRecord(ag, x, y, h, progress=s0, last_arc=arc, start_progress=s0)
            c.best_progress = s0
            c.rays = C.raycast(track, x, y, h)[0]
            c.trail.append((x, y))
            self.cars.append(c)
        self.goal = track.length * track.laps

    def state_of(self, c: CarRecord) -> C.CarState:
        return C.CarState(c.x, c.y, c.heading, c.speed, c.rays, c.progress, self.track.length, self.t)

    @property
    def done(self) -> bool:
        return all(not c.alive for c in self.cars) or self.t >= self.max_time

    def step(self):
        if self.done:
            return
        tr = self.track
        alive = [c for c in self.cars if c.alive]
        acts = []
        for c in alive:
            a = c.agent.act(self.state_of(c))
            acts.append(a)
            rid = c.agent.info.get("rule")
            if rid:
                c.rules[rid] = c.rules.get(rid, 0) + 1
        if alive:
            x = np.array([c.x for c in alive]); y = np.array([c.y for c in alive])
            h = np.array([c.heading for c in alive]); v = np.array([c.speed for c in alive])
            st = np.array([a[0] for a in acts]); th = np.array([a[1] for a in acts])
            x2, y2, h2, v2 = C.step_physics(x, y, h, v, st, th)
            crash = C.collides(tr, x2, y2, h2)
            arcs = tr.arc_at(x2, y2).astype(float)
            rays = C.raycast(tr, x2, y2, h2)
        self.t += C.DT
        self.ticks += 1
        for k, c in enumerate(alive):
            c.steer, c.throttle = acts[k]
            c.distance += float(np.hypot(x2[k] - c.x, y2[k] - c.y))
            c.x, c.y, c.heading, c.speed = float(x2[k]), float(y2[k]), float(h2[k]), float(v2[k])
            d = arcs[k] - c.last_arc
            if d < -tr.length / 2:
                d += tr.length
            elif d > tr.length / 2:
                d -= tr.length
            c.progress += d
            c.last_arc = arcs[k]
            c.rays = rays[k]
            if self.record:
                c.trail.append((c.x, c.y))
                c.speeds.append(c.speed)
                c.actions.append(acts[k])
            if c.progress > c.best_progress + 1:
                c.best_progress, c.best_t = c.progress, self.t
            if crash[k]:
                c.status, c.speed, c.end_time = CRASHED, 0.0, self.t
            elif c.progress >= self.goal:
                c.status, c.finish_time, c.end_time = FINISHED, self.t, self.t
            elif self.t - c.best_t > STUCK_S:
                c.status, c.end_time = STUCK, self.t
        if self.t >= self.max_time:
            for c in self.cars:
                if c.alive:
                    c.status, c.end_time = TIMEOUT, self.t

    def run(self):
        while not self.done:
            self.step()
        return self.results()

    def standings(self) -> List[CarRecord]:
        def key(c):
            if c.status == FINISHED:
                return (0, c.finish_time)
            return (1, -c.progress)               # race order = position on the lap
        return sorted(self.cars, key=key)

    def results(self) -> List[dict]:
        out = []
        for pos, c in enumerate(self.standings(), 1):
            driven = c.progress - c.start_progress
            out.append(dict(pos=pos, car=c.name, kind=c.agent.kind, status=c.status,
                            lap_time=round(c.finish_time, 2) if c.finish_time else None,
                            progress_pct=round(100 * min(1.0, max(0.0, c.progress) / self.goal), 1),
                            distance=round(c.distance), avg_speed=round(float(np.mean(c.speeds)), 1)
                            if c.speeds else 0.0, top_speed=round(max(c.speeds), 1) if c.speeds else 0.0,
                            end_time=round(c.end_time or self.t, 2), driven=round(driven)))
        return out


def run_race(track: Track, agents: List[Agent], **kw):
    r = Race(track, agents, **kw)
    r.run()
    return r
