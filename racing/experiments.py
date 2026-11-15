"""Experiment helpers shared by the comparison notebook and the report."""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd

from .agents import NeuroAgent, PathAgent, ReflexAgent, UtilityAgent
from .planner import HEURISTICS, Grid, astar, heuristic_table
from .race import FINISHED, run_race
from .track import Track

# fixed categorical order: colour follows the heuristic / agent, never its rank
H_COLORS = {"h0": "#2a78d6", "h1": "#eb6834", "h2": "#1baf7a", "h3": "#eda100", "h4": "#e87ba4"}
AGENT_COLORS = {"A* (h4)": "#e87ba4", "Utility": "#008300", "Reflex": "#6250d6", "Learned (GA)": "#e34948"}
HS = list(HEURISTICS)


def lap_search_table(tracks: Dict[str, Track]) -> pd.DataFrame:
    rows = []
    for name, t in tracks.items():
        g = Grid(t)
        s, goal = g.lap_endpoints()
        _, _, pre_ms = g.landmarks()
        ref = None
        for h in HS:
            r = astar(g, s, goal, h)
            if h == "h0":
                ref = r.cost
            rows.append(dict(track=name, heuristic=h, description=HEURISTICS[h], cost=round(r.cost, 1),
                             gap_pct=round(100 * (r.cost - ref) / ref, 2), expanded=r.expanded,
                             generated=r.generated, max_open=r.max_open, ms=round(r.ms, 2),
                             path_cells=len(r.cells), free_cells=int((g.free & ~g.blocked).sum()),
                             alt_precompute_ms=round(pre_ms, 1) if h == "h4" else 0.0))
    df = pd.DataFrame(rows)
    df["expanded_vs_dijkstra"] = df.groupby("track").expanded.transform(lambda x: x / x.iloc[0]).round(3)
    return df


def random_pairs_table(tracks: Dict[str, Track], n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, t in tracks.items():
        g = Grid(t)
        g.landmarks()
        free = np.argwhere(g.free)
        for q in range(n):
            i, j = rng.choice(len(free), 2, replace=False)
            a, b = tuple(free[i]), tuple(free[j])
            ref = None
            for h in HS:
                r = astar(g, a, b, h)
                if h == "h0":
                    ref = r.cost
                rows.append(dict(track=name, query=q, heuristic=h, cost=r.cost, expanded=r.expanded,
                                 ms=r.ms, gap_pct=100 * (r.cost - ref) / ref if ref else 0.0,
                                 straight_px=float(np.hypot(*(np.subtract(a, b))) * g.cell)))
    return pd.DataFrame(rows)


def informedness(tracks: Dict[str, Track]) -> pd.DataFrame:
    rows = []
    for name, t in tracks.items():
        g = Grid(t)
        s, goal = g.lap_endpoints()
        true = g.dijkstra_from([goal])[0]
        ok = np.isfinite(true) & (true > 0)
        for h in HS[1:]:
            H = heuristic_table(g, h, goal)
            ratio = H[ok] / true[ok]
            cells = np.argwhere(g.free & ~g.blocked)
            bad = 0
            for r, c in cells:
                for qq, cost in g.neighbors((r, c)):
                    if H[r, c] > cost + H[qq] + 1e-6:
                        bad += 1
            rows.append(dict(track=name, heuristic=h, mean_h_over_hstar=round(float(ratio.mean()), 3),
                             max_h_over_hstar=round(float(ratio.max()), 3),
                             overestimates=int((H[ok] > true[ok] + 1e-6).sum()),
                             admissible=bool((H[ok] <= true[ok] + 1e-6).all()),
                             consistency_violations=bad))
    return pd.DataFrame(rows)


def landmark_sweep(track: Track, ks=(1, 2, 4, 8, 12, 16)) -> pd.DataFrame:
    rows = []
    for k in ks:
        g = Grid(track)
        s, goal = g.lap_endpoints()
        t0 = time.perf_counter()
        marks, table, _ = g.landmarks(k)
        pre = (time.perf_counter() - t0) * 1000
        g._landmarks[8] = g._landmarks[k]            # heuristic_table uses the default key
        r = astar(g, s, goal, "h4")
        rows.append(dict(landmarks=len(marks), expanded=r.expanded, search_ms=round(r.ms, 2),
                         precompute_ms=round(pre, 1)))
    return pd.DataFrame(rows)


def time_trials(tracks: Dict[str, Track], neuro_w=None) -> pd.DataFrame:
    """Each agent alone on each track (fair: same start, no traffic)."""
    makers = {f"A* ({h})": (lambda h=h: PathAgent(h)) for h in ("h1", "h2", "h3", "h4")}
    makers.update({"Utility": UtilityAgent, "Reflex": ReflexAgent})
    if neuro_w is not None:
        makers["Learned (GA)"] = lambda: NeuroAgent(neuro_w)
    rows = []
    for name, t in tracks.items():
        for label, mk in makers.items():
            t0 = time.perf_counter()
            r = run_race(t, [mk()])
            c = r.cars[0]
            res = r.results()[0]
            rows.append(dict(track=name, agent=label, status=c.status,
                             lap_time=round(c.finish_time, 2) if c.status == FINISHED else np.nan,
                             avg_speed=res["avg_speed"], top_speed=res["top_speed"],
                             distance=res["distance"], cpu_ms_per_tick=round(1000 * (time.perf_counter() - t0) / max(1, r.ticks), 3)))
    return pd.DataFrame(rows)


def rule_usage(track: Track) -> pd.DataFrame:
    rows = []
    for ag in (ReflexAgent(), PathAgent("h4")):
        r = run_race(track, [ag])
        c = r.cars[0]
        tot = sum(c.rules.values())
        for rid, n in sorted(c.rules.items(), key=lambda kv: -kv[1]):
            rows.append(dict(agent=ag.name, rule=rid, ticks=n, share_pct=round(100 * n / tot, 1)))
    return pd.DataFrame(rows)


def race_table(track: Track, agents: List) -> pd.DataFrame:
    return pd.DataFrame(run_race(track, agents).results())
