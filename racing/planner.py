"""Deliberative layer: A* path planning on an occupancy grid of the track.

Grid
    The world is cut into ``CELL`` x ``CELL`` px cells. A cell is free if the car
    (radius ~ ``CAR_RADIUS``) fits there, i.e. the clearance to the nearest wall at
    the cell centre is at least that radius (configuration space). The grid is
    8-connected.

Cost of a move a -> b (in px)
    length(a, b) * (penalty(a) + penalty(b)) / 2
    penalty(c) = 1 + K * (1 - clearance(c) / (track_width / 2)),  clipped to >= 1
So driving near a wall is more expensive (a safety margin), and the cost of a move
is never below its straight-line length. Every geometric heuristic below is
therefore a valid lower bound. The cost is symmetric (undirected graph).

Lap planning
    Start and goal are both at the start/finish line. A one-cell "barrier" across the
    line stops A* from just stepping backwards over it, so the cheapest path from just
    after the line to just before it is a full lap.

Heuristics (``h(n)`` = estimate of the cost from n to the goal)
    h0  zero (Dijkstra baseline)
    h1  Euclidean distance            admissible, consistent
    h2  Manhattan distance            NOT admissible on an 8-connected grid
    h3  Octile distance               admissible, consistent, exact on an empty grid
    h4  ALT landmarks                 admissible, consistent; knows the track shape
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as cs_dijkstra

from .track import Track

CELL = 8
CAR_RADIUS = 16.0
WALL_PENALTY = 1.5
SQ2 = math.sqrt(2)
NEIGH = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]

HEURISTICS = {
    "h0": "Dijkstra (h = 0, baseline)",
    "h1": "Euclidean distance",
    "h2": "Manhattan distance",
    "h3": "Octile distance",
    "h4": "ALT landmarks",
}
ADMISSIBLE = {"h0", "h1", "h3", "h4"}

Cell = Tuple[int, int]


@dataclass
class PlanResult:
    cells: Optional[List[Cell]]
    cost: float
    expanded: int
    generated: int
    max_open: int
    ms: float
    heuristic: str
    closed: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def found(self):
        return self.cells is not None


class Grid:
    def __init__(self, track: Track, cell: int = CELL, car_radius: float = CAR_RADIUS):
        self.track, self.cell = track, cell
        R, C = track.height_px // cell, track.width_px // cell
        self.R, self.C = R, C
        cy = (np.arange(R) * cell + cell // 2)
        cx = (np.arange(C) * cell + cell // 2)
        clear = track.clearance[np.ix_(cy, cx)]
        self.clear = clear
        self.free = clear >= car_radius
        half = track.track_width / 2
        self.penalty = 1 + WALL_PENALTY * np.clip(1 - clear / half, 0, 1)
        self.arc = track.arc[np.ix_(cy, cx)]
        self.blocked = np.zeros_like(self.free)
        self._graph = None
        self._landmarks: Dict[int, tuple] = {}

    # ------------------------------------------------------------ helpers
    def to_cell(self, x: float, y: float) -> Cell:
        return int(y // self.cell), int(x // self.cell)

    def to_xy(self, rc: Cell) -> Tuple[float, float]:
        return rc[1] * self.cell + self.cell / 2, rc[0] * self.cell + self.cell / 2

    def passable(self, r, c) -> bool:
        return 0 <= r < self.R and 0 <= c < self.C and self.free[r, c] and not self.blocked[r, c]

    def nearest_free(self, rc: Cell) -> Cell:
        r0, c0 = rc
        for rad in range(0, 12):
            best = None
            for r in range(r0 - rad, r0 + rad + 1):
                for c in range(c0 - rad, c0 + rad + 1):
                    if self.passable(r, c):
                        d = (r - r0) ** 2 + (c - c0) ** 2
                        if best is None or d < best[0]:
                            best = (d, (r, c))
            if best:
                return best[1]
        raise ValueError(f"no free cell near {rc}")

    def step_cost(self, a: Cell, b: Cell) -> float:
        dl = SQ2 if (a[0] != b[0] and a[1] != b[1]) else 1.0
        return dl * self.cell * (self.penalty[a] + self.penalty[b]) / 2

    def neighbors(self, rc: Cell):
        r, c = rc
        for dr, dc, dl in NEIGH:
            q = (r + dr, c + dc)
            if self.passable(*q):
                # no corner cutting through walls
                if dr and dc and not (self.passable(r + dr, c) and self.passable(r, c + dc)):
                    continue
                yield q, dl * self.cell * (self.penalty[rc] + self.penalty[q]) / 2

    def path_cost(self, cells: List[Cell]) -> float:
        return sum(self.step_cost(a, b) for a, b in zip(cells, cells[1:]))

    # ------------------------------------------------------------ lap setup
    def lap_endpoints(self, gap_cells: float = 2.0):
        """Block a band across the start line; return (start_cell, goal_cell)."""
        sx, sy = self.track.start_pos
        h = self.track.start_heading
        dx, dy = math.cos(h), math.sin(h)
        ys = np.arange(self.R)[:, None] * self.cell + self.cell / 2
        xs = np.arange(self.C)[None, :] * self.cell + self.cell / 2
        along = (xs - sx) * dx + (ys - sy) * dy          # distance along the driving direction
        across = -(xs - sx) * dy + (ys - sy) * dx        # lateral distance from the start line centre
        band = (np.abs(along) < self.cell * 1.1) & (np.abs(across) < self.track.track_width / 2 + 2 * self.cell)
        self.blocked = self.free & band
        self._graph = None
        self._landmarks = {}
        s = self.nearest_free(self.to_cell(*self.track.point_at(self.cell * (gap_cells + 1))))
        g = self.nearest_free(self.to_cell(*self.track.point_at(-self.cell * (gap_cells + 1))))
        return s, g

    def clear_barrier(self):
        self.blocked[:] = False
        self._graph = None
        self._landmarks = {}

    # ------------------------------------------------------------ graph (for Dijkstra/ALT)
    def graph(self):
        if self._graph is None:
            R, C = self.R, self.C
            ok = self.free & ~self.blocked
            rows, cols, w = [], [], []
            idx = np.arange(R * C).reshape(R, C)
            for dr, dc, dl in NEIGH:
                r0, r1 = max(0, -dr), R - max(0, dr)
                c0, c1 = max(0, -dc), C - max(0, dc)
                a = ok[r0:r1, c0:c1] & ok[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
                if dr and dc:
                    a &= ok[r0 + dr:r1 + dr, c0:c1] & ok[r0:r1, c0 + dc:c1 + dc]
                ia = idx[r0:r1, c0:c1][a]
                ib = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc][a]
                pa = self.penalty[r0:r1, c0:c1][a]
                pb = self.penalty[r0 + dr:r1 + dr, c0 + dc:c1 + dc][a]
                rows.append(ia); cols.append(ib); w.append(dl * self.cell * (pa + pb) / 2)
            self._graph = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                                     shape=(R * C, R * C))
        return self._graph

    def dijkstra_from(self, cells: List[Cell]) -> np.ndarray:
        """Exact costs (len(cells), R, C) using scipy's C Dijkstra."""
        idx = [r * self.C + c for r, c in cells]
        d = cs_dijkstra(self.graph(), directed=True, indices=idx)
        return d.reshape(len(cells), self.R, self.C)

    def landmarks(self, k: int = 8):
        """Farthest-point landmark selection + their distance tables (cached)."""
        if k in self._landmarks:
            return self._landmarks[k]
        t0 = time.perf_counter()
        ok = self.free & ~self.blocked
        seed = tuple(np.argwhere(ok)[0])
        d0 = self.dijkstra_from([seed])[0]
        d0[~np.isfinite(d0)] = -1
        first = tuple(np.unravel_index(np.argmax(d0), d0.shape))   # farthest from an arbitrary cell
        marks = [first]
        D = [self.dijkstra_from([first])[0]]
        while len(marks) < k:
            mind = np.min(np.stack(D), axis=0)
            mind[~np.isfinite(mind)] = -1
            nxt = tuple(np.unravel_index(np.argmax(mind), mind.shape))
            if mind[nxt] <= 0:
                break
            marks.append(nxt)
            D.append(self.dijkstra_from([nxt])[0])
        table = np.stack(D)
        ms = (time.perf_counter() - t0) * 1000
        self._landmarks[k] = (marks, table, ms)
        return self._landmarks[k]


# ---------------------------------------------------------------- heuristics
def heuristic_table(grid: Grid, name: str, goal: Cell) -> np.ndarray:
    """h(n) for every cell n (R, C). Built once per search, so lookups are O(1)."""
    R, C = grid.R, grid.C
    rr, cc = np.mgrid[0:R, 0:C]
    dr = np.abs(rr - goal[0]).astype(float)
    dc = np.abs(cc - goal[1]).astype(float)
    k = grid.cell
    if name == "h0":
        return np.zeros((R, C))
    if name == "h1":
        return k * np.sqrt(dr * dr + dc * dc)
    if name == "h2":
        return k * (dr + dc)
    if name == "h3":
        return k * (np.maximum(dr, dc) + (SQ2 - 1) * np.minimum(dr, dc))
    if name == "h4":
        _, table, _ = grid.landmarks()
        dg = table[:, goal[0], goal[1]][:, None, None]
        with np.errstate(invalid="ignore"):
            diff = np.abs(table - dg)
        diff[~np.isfinite(diff)] = 0
        return diff.max(axis=0)
    raise ValueError(f"unknown heuristic {name!r}; choose one of {list(HEURISTICS)}")


def astar(grid: Grid, start: Cell, goal: Cell, heuristic: str = "h4",
          keep_closed: bool = False) -> PlanResult:
    t0 = time.perf_counter()
    if not grid.passable(*start) or not grid.passable(*goal):
        return PlanResult(None, math.inf, 0, 0, 0, 0.0, heuristic)
    H = heuristic_table(grid, heuristic, goal)
    g = {start: 0.0}
    parent = {start: None}
    tie = 0
    open_ = [(H[start], 0.0, tie, start)]
    closed = set()
    expanded = generated = 0
    max_open = 1
    while open_:
        f, negg, _, n = heapq.heappop(open_)
        if n in closed:
            continue
        closed.add(n)
        expanded += 1
        if n == goal:
            path = [n]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.reverse()
            cm = None
            if keep_closed:
                cm = np.zeros((grid.R, grid.C), bool)
                for q in closed:
                    cm[q] = True
            return PlanResult(path, g[n], expanded, generated, max_open,
                              (time.perf_counter() - t0) * 1000, heuristic, cm)
        gn = g[n]
        for q, c in grid.neighbors(n):
            ng = gn + c
            if ng < g.get(q, math.inf):
                g[q] = ng
                parent[q] = n
                tie += 1
                heapq.heappush(open_, (ng + H[q], -ng, tie, q))
                generated += 1
                if q in closed:            # inadmissible h may need re-opening
                    closed.discard(q)
        if len(open_) > max_open:
            max_open = len(open_)
    return PlanResult(None, math.inf, expanded, generated, max_open,
                      (time.perf_counter() - t0) * 1000, heuristic)


# ---------------------------------------------------------------- path -> drivable line
@dataclass
class RacingLine:
    pts: np.ndarray            # (N, 2) px, ~SPACING apart
    s: np.ndarray              # cumulative length
    curvature: np.ndarray      # 1/px, signed
    length: float
    source: PlanResult

    def index_near(self, x, y, hint: int = 0, window: int = 60) -> int:
        n = len(self.pts)
        idx = (hint + np.arange(-5, window)) % n
        d = ((self.pts[idx] - (x, y)) ** 2).sum(1)
        return int(idx[np.argmin(d)])


SPACING = 6.0


def smooth_path(grid: Grid, res: PlanResult, closed_loop: bool = True) -> RacingLine:
    pts = np.array([grid.to_xy(c) for c in res.cells], float)
    # resample uniformly, then moving-average smoothing (cyclic for a lap)
    seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.arange(0, cum[-1], SPACING)
    pts = np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1)
    k = 3
    for _ in range(2):
        if closed_loop:
            pad = np.vstack([pts[-k:], pts, pts[:k]])
        else:
            pad = np.vstack([np.repeat(pts[:1], k, 0), pts, np.repeat(pts[-1:], k, 0)])
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        pts = np.stack([np.convolve(pad[:, 0], ker, "valid"), np.convolve(pad[:, 1], ker, "valid")], 1)
    seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    n = len(pts)
    m = 3
    ang = np.arctan2(np.roll(pts[:, 1], -m) - pts[:, 1], np.roll(pts[:, 0], -m) - pts[:, 0])
    ang_prev = np.arctan2(pts[:, 1] - np.roll(pts[:, 1], m), pts[:, 0] - np.roll(pts[:, 0], m))
    dth = (ang - ang_prev + np.pi) % (2 * np.pi) - np.pi
    curv = dth / (2 * m * SPACING)
    if not closed_loop:
        curv[:m] = curv[-m:] = 0
    return RacingLine(pts, s, curv, float(s[-1]), res)


def plan_lap(track: Track, heuristic: str = "h4", grid: Optional[Grid] = None):
    """Plan a full lap. On narrow painted tracks the safety radius is reduced step by step."""
    for r in ([CAR_RADIUS, 13.0, 10.0] if grid is None else [None]):
        g = grid or Grid(track, car_radius=r)
        start, goal = g.lap_endpoints()
        res = astar(g, start, goal, heuristic)
        if res.found:
            return g, res, smooth_path(g, res)
    raise RuntimeError(f"no lap path on {track.name}: the road is too narrow somewhere for a car")
