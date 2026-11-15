import pytest

from src.mapgen import random_map
from src.tour import RACE, SURVEY, LegCache, brute_force_order, held_karp, plan_tour


def test_held_karp_equals_brute_force_on_asymmetric_matrix():
    import random
    rng = random.Random(1)
    for _ in range(30):
        n = rng.randint(1, 6)
        ids = list(range(1, n + 1))
        M = {(a, b): rng.randint(1, 30) for a in [0, *ids, -1] for b in [0, *ids, -1]}

        def cost(a, b):
            return M[(a, b)]
        assert held_karp(0, ids, -1, cost)[0] == brute_force_order(0, ids, -1, cost)[0]


@pytest.mark.parametrize("seed", range(6))
def test_survey_tour_is_optimal_on_real_maps(seed):
    g = random_map(rows=10, cols=14, n_sites=5, seed=seed)
    legs = LegCache(g, "h3")
    tour = plan_tour(g, g.home, list(g.sites), SURVEY, legs=legs)
    pos = {0: g.home, -1: g.home, **g.sites}

    def cost(a, b):
        return legs.cost(pos[a], pos[b])
    bf, _ = brute_force_order(0, list(g.sites), -1, cost)
    assert tour.cost == bf


def test_tour_route_is_closed_and_visits_all(main_map):
    tour = plan_tour(main_map, main_map.home, list(main_map.sites), SURVEY)
    route = tour.route
    assert route[0] == main_map.home and route[-1] == main_map.home
    for p in main_map.sites.values():
        assert p in route
    assert main_map.path_cost(route) == tour.cost


def test_race_mode_keeps_order(main_map):
    tour = plan_tour(main_map, main_map.home, list(main_map.sites), RACE)
    assert tour.order == sorted(main_map.sites)


def test_survey_never_worse_than_race(main_map, track_map):
    for g in (main_map, track_map):
        s = plan_tour(g, g.home, list(g.sites), SURVEY).cost
        r = plan_tour(g, g.home, list(g.sites), RACE).cost
        assert s <= r


def test_unreachable_sites_are_skipped():
    from conftest import load_test_map as test_map
    g = test_map("unreachable")
    tour = plan_tour(g, g.home, list(g.sites), SURVEY)
    assert tour.skipped == [1] and tour.order == [2]
