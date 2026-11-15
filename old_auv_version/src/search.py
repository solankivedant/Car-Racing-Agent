"""Graph search on the grid: A* (with statistics) and Dijkstra helpers."""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Union

from .world import INF, Grid, Pos

Heuristic = Callable[[Pos], float]


@dataclass
class SearchResult:
    path: Optional[List[Pos]]          # None if goal unreachable
    cost: float                        # INF if unreachable
    nodes_expanded: int
    nodes_generated: int
    max_frontier: int
    runtime_ms: float
    heuristic: str = ""
    expanded: Set[Pos] = field(default_factory=set, repr=False)

    @property
    def found(self) -> bool:
        return self.path is not None


def astar(grid: Grid, start: Pos, goal: Pos,
          heuristic: Union[str, Heuristic, None] = "h1",
          keep_expanded: bool = False) -> SearchResult:
    """A* from ``start`` to ``goal``.

    ``heuristic`` is a heuristic name from :mod:`src.heuristics` (``"h0"`` ...
    ``"h5"``), a callable ``h(pos)``, or ``None`` (plain Dijkstra).

    Ties on f are broken in favour of the larger g (deeper node), which is the
    usual choice for grids. With an admissible heuristic the result is optimal.
    The closed set is used to skip stale queue entries, and a node is reopened
    if a cheaper g is found. That keeps the result optimal for admissible but
    inconsistent heuristics too.
    """
    from .heuristics import make_heuristic  # local import avoids a cycle

    if isinstance(heuristic, str) or heuristic is None:
        name = heuristic or "h0"
        h = make_heuristic(name, grid, goal)
    else:
        name = getattr(heuristic, "__name__", "custom")
        h = heuristic

    t0 = time.perf_counter()
    if not grid.passable(start) or not grid.passable(goal):
        return SearchResult(None, INF, 0, 0, 0, 0.0, name)

    counter = itertools.count()
    g: Dict[Pos, float] = {start: 0}
    parent: Dict[Pos, Optional[Pos]] = {start: None}
    frontier = [(h(start), 0, next(counter), start)]
    closed: Set[Pos] = set()
    expanded = 0
    generated = 1
    max_frontier = 1

    while frontier:
        f, neg_g, _, node = heapq.heappop(frontier)
        gn = -neg_g
        if node in closed and gn >= g.get(node, INF):
            continue
        if gn > g.get(node, INF):
            continue
        closed.add(node)
        expanded += 1
        if node == goal:
            path = _reconstruct(parent, goal)
            dt = (time.perf_counter() - t0) * 1000
            return SearchResult(path, gn, expanded, generated, max_frontier, dt, name,
                                closed if keep_expanded else set())
        for nxt, _, cost in grid.neighbors(node):
            ng = gn + cost
            if ng < g.get(nxt, INF):
                g[nxt] = ng
                parent[nxt] = node
                if nxt in closed:
                    closed.discard(nxt)          # reopen (only for inconsistent h)
                heapq.heappush(frontier, (ng + h(nxt), -ng, next(counter), nxt))
                generated += 1
        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

    dt = (time.perf_counter() - t0) * 1000
    return SearchResult(None, INF, expanded, generated, max_frontier, dt, name,
                        closed if keep_expanded else set())


def _reconstruct(parent: Dict[Pos, Optional[Pos]], goal: Pos) -> List[Pos]:
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def dijkstra(grid: Grid, sources: Iterable[Pos], reverse: bool = False) -> Dict[Pos, float]:
    """Multi-source Dijkstra.

    ``reverse=False``: returns d(source -> p) for every reachable p.
    ``reverse=True`` : returns d(p -> nearest source), computed by running on
    the reversed graph. This is how the safety map (cost to the nearest pit)
    is built in one pass.
    """
    dist: Dict[Pos, float] = {}
    pq = []
    for s in sources:
        if grid.passable(s):
            dist[s] = 0
            pq.append((0, s))
    heapq.heapify(pq)
    while pq:
        d, p = heapq.heappop(pq)
        if d > dist.get(p, INF):
            continue
        edges = grid.predecessors(p) if reverse else ((q, c) for q, _, c in grid.neighbors(p))
        for q, c in edges:
            nd = d + c
            if nd < dist.get(q, INF):
                dist[q] = nd
                heapq.heappush(pq, (nd, q))
    return dist
