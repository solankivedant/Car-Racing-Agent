"""Reflex rules (one scenario per rule) and full-mission integration tests."""
import pytest

from conftest import all_map_paths
from conftest import load_test_map as tmap
from src import reflex as rx
from src.agent import DONE, RacingAgent, run_mission
from src.heuristics import all_heuristics
from src.mapgen import random_map
from src.safety import SafetyMap
from src.search import astar
from src.sensors import read_rays
from src.world import load_map


def P(**kw):
    base = dict(pos=(0, 0), tile="asphalt", char=".", current=None, fuel=50, capacity=50,
                cost_to_pit=10, at_pit=False, checkpoint_here=None, next_tile=(0, 1))
    base.update(kw)
    return rx.Percept(**base)


# --------------------------------------------------------------- unit: rules
def test_R0_finish():
    assert rx.decide(P(all_done=True, at_home=True, at_pit=True, cost_to_pit=0)).rid == "R0"


def test_R6_mark_checkpoint():
    assert rx.decide(P(checkpoint_here=3)).rid == "R6"
    assert rx.decide(P(checkpoint_here=3, visited=frozenset({3}))).rid == "R7"


def test_R1_refuel_when_heading_to_this_pit():
    p = P(at_pit=True, cost_to_pit=0, fuel=30, heading_to_pit=True, pit_target=(0, 0))
    assert rx.decide(p).rid == "R1"


def test_R1_not_when_passing_pit_with_plenty_of_fuel():
    assert rx.decide(P(at_pit=True, cost_to_pit=0, fuel=45)).rid == "R7"


def test_R1_not_at_other_pit_while_relaying():
    p = P(at_pit=True, cost_to_pit=0, fuel=10, heading_to_pit=True, pit_target=(5, 5))
    assert rx.decide(p).rid == "R7"


def test_R2_low_fuel():
    assert rx.decide(P(fuel=12)).rid == "R2"            # 24% < 25%
    assert rx.decide(P(fuel=12, heading_to_pit=True)).rid == "R7"


def test_R4_opportunistic_pit():
    assert rx.decide(P(fuel=15, cost_to_pit=4)).rid == "R4"     # 30% and pit cost 4
    assert rx.decide(P(fuel=15, cost_to_pit=5)).rid == "R7"     # pit too far
    assert rx.decide(P(fuel=21, cost_to_pit=2)).rid == "R7"     # 42% > 40%
    assert rx.decide(P(fuel=20, cost_to_pit=2)).rid == "R4"     # exactly 40%
    assert rx.decide(P(fuel=12.5, cost_to_pit=2)).rid == "R4"   # exactly 25%


def test_R3_unsafe_move():
    assert rx.decide(P(fuel=40, safe_after_next=False)).rid == "R3"
    assert rx.decide(P(fuel=40, safe_after_next=False, heading_to_pit=True)).rid == "R7"


def test_R5_wall_guard_has_priority_over_move():
    assert rx.decide(P(next_is_wall=True)).rid == "R5"


def test_R7_default():
    assert rx.decide(P()).rid == "R7"


def test_rule_priority_order():
    ids = [r.rid for r in rx.RULES]
    assert ids == ["R0", "R6", "R1", "R2", "R4", "R5", "R3", "R7"]


# ------------------------------------------------------------- safety & sensors
def test_cost_to_pit_matches_astar(main_map):
    s = SafetyMap(main_map)
    for p in list(main_map.passable_tiles())[::7]:
        best = min(astar(main_map, p, c, "h0").cost for c in main_map.pits)
        assert s.ctp(p) == best


def test_rays(main_map):
    rays = read_rays(main_map, (13, 1), "N")            # H; wall directly east
    assert rays["w5_right"] == 0
    assert rays["w3_front"] == 13                       # rows 12..0 of column 1 are open
    assert rays["w1_left"] == 1


def test_wall_guard_replans(main_map):
    a = RacingAgent(main_map, "h3")
    a.plan.insert(0, (12, 2))                           # inject a wall tile (12,2) is '#'
    assert not main_map.passable((12, 2))
    a.step()
    assert a.log[-1].rule == "R5"
    res = a.run()
    assert res.status == DONE


# ---------------------------------------------------------------- integration
def _check_run(g, res):
    assert res.status == DONE, res.summary()
    assert res.path[0] == g.home and res.path[-1] == g.home
    assert res.safety_violations == 0
    assert res.min_fuel >= 0
    for k in g.sites:
        assert k in res.visit_order or k in res.skipped
    # replay: fuel never negative, refuels only at pits, every move adjacent & legal
    fuel = g.capacity
    pos = g.home
    for e in res.log:
        if e.action == "MOVE":
            assert g.step_cost(pos, e.pos) == e.cost
            fuel -= e.cost
            pos = e.pos
            assert fuel >= 0
            assert abs(fuel - e.fuel) < 1e-9
        elif e.action == "REFUEL":
            assert g.is_pit(pos)
            fuel = g.capacity


@pytest.mark.parametrize("path", all_map_paths(), ids=lambda p: p.stem)
@pytest.mark.parametrize("mode", ["survey", "race"])
def test_full_mission_on_every_map(path, mode):
    g = load_map(path)
    _check_run(g, run_mission(g, "h3", mode))


@pytest.mark.parametrize("h", all_heuristics())
def test_full_mission_every_heuristic(main_map, h):
    res = run_mission(main_map, h, "survey")
    _check_run(main_map, res)
    assert res.visit_order and not res.skipped


@pytest.mark.parametrize("seed", range(40))
def test_full_mission_random_maps(seed):
    g = random_map(rows=12, cols=16, n_sites=4, seed=100 + seed)
    _check_run(g, run_mission(g, "h4", "survey"))


def test_main_map_visits_all_sites(main_map):
    res = run_mission(main_map, "h4", "survey")
    assert sorted(res.visit_order) == [1, 2, 3, 4, 5]
    assert res.pit_stops >= 1


def test_race_mode_visits_in_order(track_map):
    res = run_mission(track_map, "h4", "race")
    assert res.visit_order == sorted(res.visit_order)


def test_relay_through_pits():
    g = tmap("relay")
    res = run_mission(g, "h3")
    assert res.visit_order == [1] and res.pit_stops >= 2


def test_infeasible_site_is_skipped_not_looped():
    g = tmap("infeasible")
    res = run_mission(g, "h3")
    assert res.status == DONE and res.skipped == [1]


def test_unreachable_site_skipped():
    res = run_mission(tmap("unreachable"), "h3")
    assert res.skipped == [1] and res.visit_order == [2]


def test_low_fuel_map_needs_pit_stop():
    res = run_mission(tmap("low_fuel"), "h3")
    assert res.visit_order == [1] and res.pit_stops >= 1
    assert "R1" in res.rule_counts


def test_safety_rule_fires_on_main_map(main_map):
    res = run_mission(main_map, "h3")
    assert res.rule_counts.get("R3", 0) >= 1
    assert res.rule_counts.get("R4", 0) >= 1


def test_no_sites_map_finishes_immediately():
    res = run_mission(tmap("no_sites"), "h3")
    assert res.status == DONE and res.moves == 0


def test_bad_mode_rejected(main_map):
    with pytest.raises(ValueError):
        RacingAgent(main_map, "h3", mode="drift")
