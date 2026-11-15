"""Heuristics for A* on the direction-dependent cost grid.

All heuristics are built by a factory ``make_heuristic(name, grid, goal)`` that
returns ``h(pos) -> float``. Anything that does not depend on the goal is
precomputed once per grid and cached on ``grid._cache``.

========  =====================================  ===========  ============
name      idea                                   admissible   consistent
========  =====================================  ===========  ============
h0        zero (Dijkstra baseline)               yes          yes
h1        Manhattan x c_min                      yes          yes
h2        Euclidean x c_min                      yes          yes
h3        Row/column corridor bound              yes          yes
h4        ALT landmarks (directed triangle ineq) yes          yes
h5        Weighted A*: 2 x h3 (bonus)            NO           no
========  =====================================  ===========  ============

h0 is the reference and h5 is an extra to show the speed / optimality
trade-off. The four heuristics we designed and compare are h1-h4.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List

from .world import INF, E, Grid, N, Pos, S, W

Heuristic = Callable[[Pos], float]

HEURISTIC_INFO: Dict[str, str] = {
    "h0": "Zero / Dijkstra (baseline)",
    "h1": "Manhattan x c_min",
    "h2": "Euclidean x c_min",
    "h3": "Row/Column corridor bound",
    "h4": "ALT landmarks",
    "h5": "Weighted A* (2 x h3, inadmissible)",
}
ADMISSIBLE = {"h0", "h1", "h2", "h3", "h4"}


# ----------------------------------------------------------------------------
# h0 / h1 / h2
# ----------------------------------------------------------------------------
def h_zero(grid: Grid, goal: Pos) -> Heuristic:
    def h0(p: Pos) -> float:
        return 0
    return h0


def h_manhattan(grid: Grid, goal: Pos) -> Heuristic:
    """Each of the |dr|+|dc| moves still needed costs at least c_min."""
    cmin = grid.min_step_cost
    gr, gc = goal

    def h1(p: Pos) -> float:
        return cmin * (abs(p[0] - gr) + abs(p[1] - gc))
    return h1


def h_euclidean(grid: Grid, goal: Pos) -> Heuristic:
    """Straight-line distance x c_min. Always <= Manhattan, so it is weaker."""
    cmin = grid.min_step_cost
    gr, gc = goal

    def h2(p: Pos) -> float:
        return cmin * math.hypot(p[0] - gr, p[1] - gc)
    return h2


# ----------------------------------------------------------------------------
# h3: row / column corridor bound (domain specific, uses terrain + currents)
# ----------------------------------------------------------------------------
def _corridor_tables(grid: Grid):
    """Cheapest way to enter each row going N/S and each column going E/W.

    enter_row[d][r] = min cost of any move in direction d (N or S) whose
    target tile is in row r. enter_col[d][c] is the same for E/W moves into
    column c. A path from row r1 to row r2 > r1 has to make at least one
    S-move into every row r1+1..r2, and those moves are all different.
    The same holds for columns, so summing the cheapest entries is a lower
    bound. Prefix sums make every query O(1).
    """
    key = "corridor"
    if key in grid._cache:
        return grid._cache[key]
    R, C = grid.rows, grid.cols
    enter_row = {N: [INF] * R, S: [INF] * R}
    enter_col = {E: [INF] * C, W: [INF] * C}
    for p in grid.passable_tiles():
        for q, d, cost in grid.neighbors(p):
            if d in (N, S):
                enter_row[d][q[0]] = min(enter_row[d][q[0]], cost)
            else:
                enter_col[d][q[1]] = min(enter_col[d][q[1]], cost)

    def prefix(vals: List[float]) -> List[float]:
        # A row/column that can never be entered in that direction contributes
        # 0. That is still a valid lower bound, and it avoids INF - INF = NaN
        # in the prefix differences.
        out = [0.0]
        for v in vals:
            out.append(out[-1] + (0 if v == INF else v))
        return out

    tables = {
        N: prefix(enter_row[N]), S: prefix(enter_row[S]),
        E: prefix(enter_col[E]), W: prefix(enter_col[W]),
    }
    grid._cache[key] = tables
    return tables


def h_corridor(grid: Grid, goal: Pos) -> Heuristic:
    t = _corridor_tables(grid)
    PS, PN, PE, PW = t[S], t[N], t[E], t[W]
    gr, gc = goal

    def h3(p: Pos) -> float:
        r, c = p
        if gr > r:      # need S-moves into rows r+1..gr
            v = PS[gr + 1] - PS[r + 1]
        elif gr < r:    # need N-moves into rows gr..r-1
            v = PN[r] - PN[gr]
        else:
            v = 0
        if gc > c:      # E-moves into cols c+1..gc
            v += PE[gc + 1] - PE[c + 1]
        elif gc < c:    # W-moves into cols gc..c-1
            v += PW[c] - PW[gc]
        return v
    return h3


# ----------------------------------------------------------------------------
# h4: ALT (A*, Landmarks, Triangle inequality) for a directed graph
# ----------------------------------------------------------------------------
def select_landmarks(grid: Grid, k: int = 6) -> List[Pos]:
    """Pits first (H + C), then farthest-point picks to cover the map."""
    from .search import dijkstra
    marks: List[Pos] = list(dict.fromkeys(grid.pits))[:k]
    tiles = list(grid.passable_tiles())
    if not tiles:
        return marks
    if not marks:
        marks.append(tiles[0])
    while len(marks) < k:
        d = dijkstra(grid, marks)
        # farthest reachable tile from the current landmarks
        far = max((p for p in tiles if p in d), key=lambda p: d[p], default=None)
        if far is None or far in marks or d.get(far, 0) == 0:
            break
        marks.append(far)
    return marks


def landmark_tables(grid: Grid, k: int = 6):
    key = ("alt", k)
    if key in grid._cache:
        return grid._cache[key]
    from .search import dijkstra
    marks = select_landmarks(grid, k)
    tables = []
    for L in marks:
        from_L = dijkstra(grid, [L])               # d(L -> p)
        to_L = dijkstra(grid, [L], reverse=True)   # d(p -> L)
        tables.append((from_L, to_L))
    grid._cache[key] = (marks, tables)
    return grid._cache[key]


def h_alt(grid: Grid, goal: Pos, k: int = 6) -> Heuristic:
    """For a directed graph the triangle inequality gives two bounds per landmark L:

        d(n, g) >= d(n, L) - d(g, L)
        d(n, g) >= d(L, g) - d(L, n)

    Both are admissible and consistent, and so is their maximum. Terms with an
    unknown (infinite) distance are skipped.
    """
    _, tables = landmark_tables(grid, k)
    per_L = []
    for from_L, to_L in tables:
        g_to = to_L.get(goal, INF)      # d(g -> L)
        g_from = from_L.get(goal, INF)  # d(L -> g)
        per_L.append((from_L, to_L, g_to, g_from))

    def h4(p: Pos) -> float:
        best = 0.0
        for from_L, to_L, g_to, g_from in per_L:
            nL = to_L.get(p, INF)
            if nL < INF and g_to < INF:
                v = nL - g_to
                if v > best:
                    best = v
            Ln = from_L.get(p, INF)
            if g_from < INF and Ln < INF:
                v = g_from - Ln
                if v > best:
                    best = v
        return best
    return h4


# ----------------------------------------------------------------------------
# h5: weighted (inadmissible), bonus
# ----------------------------------------------------------------------------
def h_weighted(grid: Grid, goal: Pos, w: float = 2.0) -> Heuristic:
    base = h_corridor(grid, goal)

    def h5(p: Pos) -> float:
        return w * base(p)
    return h5


_FACTORY = {
    "h0": h_zero, "h1": h_manhattan, "h2": h_euclidean,
    "h3": h_corridor, "h4": h_alt, "h5": h_weighted,
}


def make_heuristic(name: str, grid: Grid, goal: Pos) -> Heuristic:
    try:
        fn = _FACTORY[name](grid, goal)
    except KeyError:
        raise ValueError(f"Unknown heuristic {name!r}; choose from {list(_FACTORY)}") from None
    fn.__name__ = name
    return fn


def all_heuristics(include_bonus: bool = True) -> List[str]:
    return list(_FACTORY) if include_bonus else [k for k in _FACTORY if k != "h5"]
