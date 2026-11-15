import pytest

from conftest import all_map_paths
from conftest import load_test_map as tmap
from src.heuristics import ADMISSIBLE, all_heuristics, make_heuristic
from src.mapgen import random_map
from src.search import astar, dijkstra
from src.world import INF, direction_between, load_map

RANDOM_MAPS = [random_map(rows=12, cols=16, seed=s) for s in range(25)]


def _valid_path(g, path, start, goal):
    assert path[0] == start and path[-1] == goal
    for a, b in zip(path, path[1:]):
        assert direction_between(a, b) is not None
        assert g.passable(b)


@pytest.mark.parametrize("h", all_heuristics())
def test_astar_path_is_valid_and_cost_matches(main_map, h):
    g = main_map
    res = astar(g, g.home, g.sites[2], h)
    assert res.found
    _valid_path(g, res.path, g.home, g.sites[2])
    assert res.cost == g.path_cost(res.path)


@pytest.mark.parametrize("h", sorted(ADMISSIBLE))
def test_admissible_heuristics_are_optimal_on_random_maps(h):
    for g in RANDOM_MAPS:
        pts = [g.home, *g.sites.values(), *g.pits]
        for a in pts[:4]:
            ref = dijkstra(g, [a])
            for b in pts:
                res = astar(g, a, b, h)
                assert res.cost == ref.get(b, INF), (g.name, a, b, h)


def test_weighted_is_never_better_than_optimal(main_map):
    g = main_map
    for k, p in g.sites.items():
        opt = astar(g, g.home, p, "h0").cost
        w = astar(g, g.home, p, "h5").cost
        assert w >= opt


@pytest.mark.parametrize("h", sorted(ADMISSIBLE - {"h0"}))
def test_admissible_and_consistent_everywhere(h):
    maps = [load_map(p) for p in all_map_paths()] + RANDOM_MAPS[:8]
    for g in maps:
        goals = list(g.sites.values())[:3] + [g.home]
        for goal in goals:
            hf = make_heuristic(h, g, goal)
            true = dijkstra(g, [goal], reverse=True)      # exact cost-to-go
            for n in g.passable_tiles():
                if n in true:
                    assert hf(n) <= true[n] + 1e-9, (g.name, h, n, goal)
                for m, _, c in g.neighbors(n):
                    assert hf(n) <= c + hf(m) + 1e-9, ("inconsistent", g.name, h, n, m)
            assert hf(goal) == 0


def test_h5_is_inadmissible_somewhere(main_map):
    g = main_map
    hf = make_heuristic("h5", g, g.sites[2])
    true = dijkstra(g, [g.sites[2]], reverse=True)
    assert any(hf(n) > true[n] for n in true)


def test_informed_heuristics_expand_fewer_nodes(main_map):
    g = main_map
    tot = {h: 0 for h in ADMISSIBLE}
    for a in [g.home, *g.sites.values()]:
        for b in [g.home, *g.sites.values()]:
            for h in ADMISSIBLE:
                tot[h] += astar(g, a, b, h).nodes_expanded
    assert tot["h4"] < tot["h1"] < tot["h0"]
    assert tot["h3"] <= tot["h1"] <= tot["h2"] <= tot["h0"]


def test_unreachable_goal_returns_none():
    g = tmap("unreachable")
    res = astar(g, g.home, g.sites[1], "h1")
    assert not res.found and res.cost == INF and res.path is None


def test_start_equals_goal(main_map):
    res = astar(main_map, main_map.home, main_map.home, "h3")
    assert res.path == [main_map.home] and res.cost == 0


def test_reverse_dijkstra_matches_forward(main_map):
    g = main_map
    back = dijkstra(g, [g.home], reverse=True)
    for p in list(g.sites.values()):
        assert back[p] == astar(g, p, g.home, "h0").cost
