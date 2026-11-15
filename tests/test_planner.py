import math

import numpy as np
import pytest

from racing.planner import ADMISSIBLE, HEURISTICS, Grid, astar, heuristic_table, plan_lap, smooth_path


@pytest.fixture(scope="module")
def lap_grids(tracks):
    out = {}
    for name, t in tracks.items():
        g = Grid(t)
        s, goal = g.lap_endpoints()
        out[name] = (g, s, goal)
    return out


def _valid(g, cells):
    for a, b in zip(cells, cells[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
        assert g.passable(*b)


@pytest.mark.parametrize("h", list(HEURISTICS))
def test_lap_path_valid_for_every_heuristic(lap_grids, h):
    for name, (g, s, goal) in lap_grids.items():
        r = astar(g, s, goal, h)
        assert r.found, (name, h)
        _valid(g, r.cells)
        assert abs(r.cost - g.path_cost(r.cells)) < 1e-6
        # it really drives a full lap, not back across the start line
        assert r.cost > 0.8 * g.track.length


def test_admissible_heuristics_are_optimal(lap_grids):
    for name, (g, s, goal) in lap_grids.items():
        ref = astar(g, s, goal, "h0").cost
        for h in ADMISSIBLE:
            assert abs(astar(g, s, goal, h).cost - ref) < 1e-6, (name, h)
        assert astar(g, s, goal, "h2").cost >= ref - 1e-6


def test_random_pairs_optimal(tracks):
    rng = np.random.default_rng(0)
    for t in tracks.values():
        g = Grid(t)
        free = np.argwhere(g.free)
        for _ in range(8):
            a, b = (tuple(free[i]) for i in rng.choice(len(free), 2, replace=False))
            ref = g.dijkstra_from([a])[0][b]
            for h in ADMISSIBLE:
                assert abs(astar(g, a, b, h).cost - ref) < 1e-6


@pytest.mark.parametrize("h", ["h1", "h3", "h4"])
def test_admissible_and_consistent(lap_grids, h):
    for name, (g, s, goal) in lap_grids.items():
        H = heuristic_table(g, h, goal)
        true = g.dijkstra_from([goal])[0]           # symmetric graph: d(goal, n) = d(n, goal)
        ok = np.isfinite(true)
        assert np.all(H[ok] <= true[ok] + 1e-6), (name, h)
        cells = np.argwhere(g.free & ~g.blocked)[::5]
        for r, c in cells:
            for q, cost in g.neighbors((r, c)):
                assert H[r, c] <= cost + H[q] + 1e-6


def test_manhattan_overestimates(lap_grids):
    g, s, goal = lap_grids["oval"]
    H = heuristic_table(g, "h2", goal)
    true = g.dijkstra_from([goal])[0]
    ok = np.isfinite(true)
    assert np.any(H[ok] > true[ok] + 1e-6)


def test_heuristic_ordering(lap_grids):
    g, s, goal = lap_grids["serpentine"]
    h1, h3, h4 = (heuristic_table(g, h, goal) for h in ("h1", "h3", "h4"))
    assert np.all(h1 <= h3 + 1e-9)                   # octile dominates euclidean
    exp = {h: astar(g, s, goal, h).expanded for h in ADMISSIBLE}
    assert exp["h4"] < exp["h3"] <= exp["h1"] <= exp["h0"]


def test_barrier_blocks_start_line(lap_grids):
    for g, s, goal in lap_grids.values():
        assert g.blocked.any()
        assert g.passable(*s) and g.passable(*goal)


def test_racing_line_is_on_track(tracks):
    for t in tracks.values():
        _, res, line = plan_lap(t, "h4")
        cl = t.clearance_at(line.pts[:, 0], line.pts[:, 1])
        assert cl.min() > 8, t.name                  # car body (radius ~11) mostly fits
        assert 0.8 * t.length < line.length < 1.1 * t.length
        assert np.all(np.isfinite(line.curvature))


def test_unreachable_goal(tracks):
    g = Grid(tracks["oval"])
    wall = tuple(np.argwhere(~g.free)[0])
    free = tuple(np.argwhere(g.free)[0])
    assert not astar(g, free, wall, "h1").found


def test_unknown_heuristic(tracks):
    g = Grid(tracks["oval"])
    with pytest.raises(ValueError):
        heuristic_table(g, "h9", (0, 0))
    _ = math, smooth_path
