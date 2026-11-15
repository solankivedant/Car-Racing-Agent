"""Tour planning: choose the order of checkpoints and build the full route.

* ``survey`` mode: visit every remaining checkpoint in the cheapest order
  (the PDF behaviour), then return to H. Because currents make costs
  asymmetric this is an *asymmetric* TSP. It is solved exactly with
  Held-Karp dynamic programming, O(2^n * n^2), which is instant for n <= 9.
* ``race`` mode: visit checkpoints in ascending order 1 -> 2 -> ... -> H, like
  a racing lap.

The tour ignores fuel. The reflex layer (``agent.py``) enforces the fuel rules
and asks for a re-plan after every pit stop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

from .search import SearchResult, astar
from .world import INF, Grid, Pos

SURVEY, RACE = "survey", "race"


@dataclass
class Tour:
    order: List[int]                       # checkpoint ids in visiting order
    legs: List[List[Pos]]                  # one path per leg (start->s1, ..., sn->end)
    cost: float                            # sum of leg costs (ignores refuelling)
    skipped: List[int] = field(default_factory=list)  # unreachable checkpoints
    nodes_expanded: int = 0
    runtime_ms: float = 0.0

    @property
    def route(self) -> List[Pos]:
        out: List[Pos] = []
        for leg in self.legs:
            out.extend(leg if not out else leg[1:])
        return out


class LegCache:
    """Remembers A* results so re-planning does not repeat searches."""

    def __init__(self, grid: Grid, heuristic: str = "h1"):
        self.grid = grid
        self.heuristic = heuristic
        self._cache: Dict[Tuple[Pos, Pos], SearchResult] = {}
        self.searches = 0
        self.nodes_expanded = 0
        self.runtime_ms = 0.0

    def get(self, a: Pos, b: Pos) -> SearchResult:
        key = (a, b)
        if key not in self._cache:
            res = astar(self.grid, a, b, self.heuristic)
            self._cache[key] = res
            self.searches += 1
            self.nodes_expanded += res.nodes_expanded
            self.runtime_ms += res.runtime_ms
        return self._cache[key]

    def cost(self, a: Pos, b: Pos) -> float:
        return self.get(a, b).cost

    def path(self, a: Pos, b: Pos) -> Optional[List[Pos]]:
        return self.get(a, b).path


def held_karp(start: int, targets: Sequence[int], end: int, cost) -> Tuple[float, List[int]]:
    """Exact asymmetric TSP path start -> (all targets) -> end.

    ``cost(i, j)`` gives the travel cost between node ids. Returns
    ``(total_cost, order_of_targets)``.
    """
    n = len(targets)
    if n == 0:
        return cost(start, end), []
    FULL = (1 << n) - 1
    # dp[mask][j] = cheapest cost from start visiting exactly `mask`, ending at targets[j]
    dp = [[INF] * n for _ in range(1 << n)]
    par = [[-1] * n for _ in range(1 << n)]
    for j in range(n):
        dp[1 << j][j] = cost(start, targets[j])
    for mask in range(1, FULL + 1):
        row = dp[mask]
        for j in range(n):
            cur = row[j]
            if cur == INF or not (mask >> j) & 1:
                continue
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                v = cur + cost(targets[j], targets[k])
                if v < dp[nm][k]:
                    dp[nm][k] = v
                    par[nm][k] = j
    best, last = INF, -1
    for j in range(n):
        v = dp[FULL][j] + cost(targets[j], end)
        if v < best:
            best, last = v, j
    if last < 0:
        return INF, []
    order, mask = [], FULL
    while last >= 0:
        order.append(targets[last])
        prev = par[mask][last]
        mask ^= 1 << last
        last = prev
    order.reverse()
    return best, order


def brute_force_order(start: int, targets: Sequence[int], end: int, cost) -> Tuple[float, List[int]]:
    """Reference solver used by the tests (only for small n)."""
    best, best_order = INF, []
    for perm in permutations(targets):
        seq = [start, *perm, end]
        v = sum(cost(a, b) for a, b in zip(seq, seq[1:]))
        if v < best:
            best, best_order = v, list(perm)
    return best, best_order


def plan_tour(grid: Grid, start: Pos, remaining: Sequence[int], mode: str = SURVEY,
              heuristic: str = "h1", legs: Optional[LegCache] = None,
              end: Optional[Pos] = None) -> Tour:
    """Plan start -> remaining checkpoints -> end (default H)."""
    legs = legs or LegCache(grid, heuristic)
    end = grid.home if end is None else end
    exp0, t0 = legs.nodes_expanded, legs.runtime_ms

    # Drop checkpoints that cannot be reached from start, or from which home
    # cannot be reached.
    reachable, skipped = [], []
    for k in remaining:
        p = grid.sites[k]
        if legs.cost(start, p) < INF and legs.cost(p, end) < INF:
            reachable.append(k)
        else:
            skipped.append(k)

    # node id 0 = start, -1 = end, k = checkpoint k
    def pos_of(i: int) -> Pos:
        return start if i == 0 else end if i == -1 else grid.sites[i]

    def cost(i: int, j: int) -> float:
        return legs.cost(pos_of(i), pos_of(j))

    if mode == RACE:
        order = sorted(reachable)
        seq = [0, *order, -1]
        total = sum(cost(a, b) for a, b in zip(seq, seq[1:]))
    elif mode == SURVEY:
        total, order = held_karp(0, reachable, -1, cost)
        seq = [0, *order, -1]
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    paths = [legs.path(pos_of(a), pos_of(b)) or [] for a, b in zip(seq, seq[1:])]
    return Tour(order, paths, total, skipped,
                legs.nodes_expanded - exp0, legs.runtime_ms - t0)
