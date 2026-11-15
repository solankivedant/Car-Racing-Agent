"""Safety map and pit-stop routing.

``cost_to_pit[p]`` is the cheapest fuel cost from ``p`` to *any* refuel point
(H or C). It is computed once with a multi-source Dijkstra on the reversed
graph, so the PDF rule "before every move, check that after the move the car
can still reach a charger" becomes an O(1) lookup::

    safe(cur -> nxt)  <=>  fuel - step_cost(cur, nxt) >= cost_to_pit[nxt]

The class also keeps the exact pit-to-pit distances. They are used to plan a
*relay* (pit-hopping) when a checkpoint is too far to reach in one go from the
pit the car is currently at.
"""
from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Optional

from .search import astar, dijkstra
from .world import INF, Grid, Pos


class SafetyMap:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.cost_to_pit: Dict[Pos, float] = dijkstra(grid, grid.pits, reverse=True)
        # exact d(P -> x) from every pit, used for pit-chains and staging
        self.from_pit: Dict[Pos, Dict[Pos, float]] = {P: dijkstra(grid, [P]) for P in grid.pits}

    # ------------------------------------------------------------------
    def ctp(self, p: Pos) -> float:
        return self.cost_to_pit.get(p, INF)

    def is_safe_move(self, fuel: float, cur: Pos, nxt: Pos) -> bool:
        c = self.grid.step_cost(cur, nxt)
        return fuel - c >= self.ctp(nxt)

    def cheapest_pit(self, pos: Pos, heuristic: str = "h1"):
        """Return (pit, path, cost) of the pit that is cheapest to reach."""
        best = (None, None, INF)
        for P in self.grid.pits:
            res = astar(self.grid, pos, P, heuristic)
            if res.found and res.cost < best[2]:
                best = (P, res.path, res.cost)
        return best

    # ------------------------------------------------------------------
    def pit_chain(self, start_pit: Pos, is_staging: Callable[[Pos], bool]) -> Optional[List[Pos]]:
        """Cheapest chain of pits ``[start_pit, P1, ..., P*]`` where each hop
        costs at most a full tank, so it is always safe, and ``P*`` satisfies
        ``is_staging``. Returns ``None`` if no such chain exists.
        """
        cap = self.grid.capacity
        pits = list(dict.fromkeys(self.grid.pits))
        dist = {start_pit: 0.0}
        parent: Dict[Pos, Optional[Pos]] = {start_pit: None}
        pq = [(0.0, start_pit)]
        done = set()
        while pq:
            d, P = heapq.heappop(pq)
            if P in done:
                continue
            done.add(P)
            if is_staging(P):
                chain = [P]
                while parent[chain[-1]] is not None:
                    chain.append(parent[chain[-1]])
                return chain[::-1]
            for Q in pits:
                if Q == P:
                    continue
                c = self.from_pit[P].get(Q, INF)
                if c <= cap and d + c < dist.get(Q, INF):
                    dist[Q] = d + c
                    parent[Q] = P
                    heapq.heappush(pq, (d + c, Q))
        return None
