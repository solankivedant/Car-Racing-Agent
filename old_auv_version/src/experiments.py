"""Experiment helpers shared by the comparison notebook and the report."""
from __future__ import annotations

import time
from typing import Dict, Iterable, List

import pandas as pd

from .agent import run_mission
from .heuristics import HEURISTIC_INFO, landmark_tables, make_heuristic
from .mapgen import random_map
from .search import astar, dijkstra
from .world import Grid, parse_map, grid_to_text

# categorical slots (fixed order: colour follows the heuristic, never its rank)
PALETTE = {
    "h0": "#2a78d6", "h1": "#eb6834", "h2": "#1baf7a",
    "h3": "#eda100", "h4": "#e87ba4", "h5": "#008300",
}
SEQ_BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
HEURS = ["h0", "h1", "h2", "h3", "h4", "h5"]


def fresh(g: Grid) -> Grid:
    """Copy of a grid with empty caches, so precompute time is measured fairly."""
    return parse_map(grid_to_text(g), name=g.name)


def key_points(g: Grid):
    return {"H": g.home, **{str(k): p for k, p in g.sites.items()},
            **{f"C{i}": p for i, p in enumerate(g.pits[1:], 1)}}


def effective_branching_factor(n_expanded: int, depth: int) -> float:
    """Solve N + 1 = 1 + b + b^2 + ... + b^d for b (bisection)."""
    if depth <= 0 or n_expanded <= 1:
        return 1.0
    target = n_expanded + 1
    lo, hi = 1.0, float(max(2, n_expanded))
    for _ in range(80):
        mid = (lo + hi) / 2
        s = (mid ** (depth + 1) - 1) / (mid - 1) if mid != 1 else depth + 1
        if s < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def pairwise_runs(g: Grid, heuristics: Iterable[str] = HEURS, map_label: str = None) -> pd.DataFrame:
    """A* for every ordered pair of key points (H, checkpoints, pits) x heuristic."""
    pts = key_points(g)
    rows = []
    for h in heuristics:
        for an, a in pts.items():
            for bn, b in pts.items():
                if a == b:
                    continue
                r = astar(g, a, b, h)
                if not r.found:
                    continue
                rows.append(dict(map=map_label or g.name, heuristic=h, start=an, goal=bn,
                                 cost=r.cost, expanded=r.nodes_expanded,
                                 generated=r.nodes_generated, frontier=r.max_frontier,
                                 ms=r.runtime_ms, depth=len(r.path) - 1))
    df = pd.DataFrame(rows)
    opt = df[df.heuristic == "h0"].set_index(["start", "goal"])["cost"]
    df["optimal"] = [opt[(s, t)] for s, t in zip(df.start, df.goal)]
    df["gap_pct"] = 100 * (df.cost - df.optimal) / df.optimal
    df["ebf"] = [effective_branching_factor(n, d) for n, d in zip(df.expanded, df.depth)]
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby("heuristic").agg(
        searches=("cost", "size"), total_expanded=("expanded", "sum"),
        mean_expanded=("expanded", "mean"), mean_ms=("ms", "mean"),
        mean_ebf=("ebf", "mean"), optimal_pct=("gap_pct", lambda x: 100 * (x.abs() < 1e-9).mean()),
        max_gap_pct=("gap_pct", "max"))
    base = s.loc["h0", "total_expanded"]
    s["expanded_vs_dijkstra"] = s.total_expanded / base
    s.insert(0, "description", [HEURISTIC_INFO[h] for h in s.index])
    return s.round(3)


def random_suite(n_maps: int = 60, sizes=((12, 16), (20, 30), (30, 40)), seed0: int = 1000,
                 n_sites: int = 5) -> List[Grid]:
    maps = []
    for i in range(n_maps):
        r, c = sizes[i % len(sizes)]
        g = random_map(rows=r, cols=c, n_sites=n_sites, n_pits=2 + (r * c) // 400,
                       seed=seed0 + i)
        g.name = f"rand{i:02d}_{r}x{c}"
        maps.append(g)
    return maps


def suite_runs(maps: List[Grid], heuristics=HEURS) -> pd.DataFrame:
    frames = []
    for g in maps:
        d = pairwise_runs(g, heuristics)
        d["size"] = f"{g.rows}x{g.cols}"
        d["cells"] = g.rows * g.cols
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def informedness(g: Grid, heuristics=("h1", "h2", "h3", "h4", "h5"), n_goals: int = 6) -> pd.DataFrame:
    """Mean h(n)/h*(n) over all tiles (1.0 = perfect, >1 means overestimates)."""
    goals = list(key_points(g).values())[:n_goals]
    rows = []
    for h in heuristics:
        ratios, over = [], 0
        for goal in goals:
            hf = make_heuristic(h, g, goal)
            true = dijkstra(g, [goal], reverse=True)
            for n, t in true.items():
                if t > 0:
                    v = hf(n)
                    ratios.append(v / t)
                    over += v > t + 1e-9
        rows.append(dict(heuristic=h, mean_h_over_hstar=sum(ratios) / len(ratios),
                         min_ratio=min(ratios), max_ratio=max(ratios),
                         overestimates=over, admissible=over == 0))
    return pd.DataFrame(rows).set_index("heuristic").round(3)


def consistency_violations(g: Grid, h: str, goals) -> int:
    bad = 0
    for goal in goals:
        hf = make_heuristic(h, g, goal)
        for n in g.passable_tiles():
            for m, _, c in g.neighbors(n):
                if hf(n) > c + hf(m) + 1e-9:
                    bad += 1
    return bad


def alt_precompute_ms(g: Grid, k: int = 6) -> float:
    g2 = fresh(g)
    t0 = time.perf_counter()
    landmark_tables(g2, k)
    return (time.perf_counter() - t0) * 1000


def mission_table(maps: Dict[str, Grid], modes=("survey", "race"), heuristics=HEURS) -> pd.DataFrame:
    rows = []
    for name, g in maps.items():
        for mode in modes:
            for h in heuristics:
                g2 = fresh(g)
                t0 = time.perf_counter()
                r = run_mission(g2, h, mode)
                wall = (time.perf_counter() - t0) * 1000
                rows.append(dict(map=name, mode=mode, heuristic=h, status=r.status,
                                 fuel_used=r.fuel_used, moves=r.moves, pit_stops=r.pit_stops,
                                 replans=r.replans, planned_tour_cost=r.planned_tour_cost,
                                 searches=r.searches, nodes_expanded=r.nodes_expanded,
                                 search_ms=round(r.search_ms, 2), total_ms=round(wall, 1),
                                 visit_order="-".join(map(str, r.visit_order)),
                                 violations=r.safety_violations))
    return pd.DataFrame(rows)
