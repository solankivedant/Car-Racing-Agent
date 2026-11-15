import pytest

from conftest import load_test_map as tmap
from src.world import (ASPHALT, GRASS, INF, WALL, MapFormatError, grid_to_text, parse_map)


def test_main_map_header(main_map):
    g = main_map
    assert (g.rows, g.cols, g.capacity) == (15, 20, 50)
    assert g.home == (13, 1)
    assert set(g.pits) == {(13, 1), (1, 11), (11, 16)}
    assert g.sites == {1: (1, 1), 2: (1, 17), 3: (7, 18), 4: (13, 17), 5: (7, 4)}


def test_main_map_terrain_and_currents(main_map):
    g = main_map
    assert g.terrain[0][2] == GRASS
    assert g.terrain[0][11] == WALL
    assert g.terrain[0][0] == ASPHALT
    assert g.current[4][0] == "E" and g.current[1][0] == "N"
    assert g.current[6][19] == "S" and g.current[13][3] == "W"
    assert g.current[0][0] is None


def test_cost_model_with_against_across():
    g = tmap("currents")
    # row 1: H(1,0) .(1,1)> .(1,2)> .(1,3)< 1(1,4)
    assert g.step_cost((1, 0), (1, 1)) == 1        # E into an E current: half
    assert g.step_cost((1, 2), (1, 1)) == 4        # W into an E current: double
    assert g.step_cost((0, 1), (1, 1)) == 2        # S into an E current: across
    assert g.step_cost((1, 2), (1, 3)) == 4        # E into a W current: double
    assert g.step_cost((1, 2), (2, 2)) == 4        # S into an N current: double
    assert g.step_cost((1, 3), (2, 3)) == 1        # S into an S current: half


def test_kelp_and_wall_costs(main_map):
    g = main_map
    assert g.step_cost((1, 1), (1, 2)) == 6        # into grass/kelp, no current
    assert g.step_cost((0, 10), (0, 11)) == INF    # into a wall
    assert g.step_cost((0, 0), (5, 5)) == INF      # not adjacent


def test_costs_are_asymmetric(main_map):
    a, b = (4, 5), (4, 6)                          # E current row
    assert main_map.step_cost(a, b) == 1
    assert main_map.step_cost(b, a) == 4


def test_min_step_cost(main_map):
    assert main_map.min_step_cost == 1
    assert tmap("tiny_open").min_step_cost == 2


def test_roundtrip_serialisation(main_map):
    g2 = parse_map(grid_to_text(main_map))
    assert g2.raw == main_map.raw and g2.current == main_map.current


def test_aliases_S_P_FUEL():
    g = parse_map("ROWS 1\nCOLS 3\nFUEL 9\nTERRAIN\nSP1\nCURRENTS\n...\n")
    assert g.home == (0, 0) and g.pits == [(0, 0), (0, 1)] and g.capacity == 9


@pytest.mark.parametrize("text, msg", [
    ("ROWS 2\nCOLS 3\nBATTERY 5\nTERRAIN\nH..\n...\nCURRENTS\n...\n", "current rows"),
    ("ROWS 1\nCOLS 3\nBATTERY 5\nTERRAIN\nH.\nCURRENTS\n...\n", "columns"),
    ("ROWS 1\nCOLS 3\nBATTERY 5\nTERRAIN\nH.X\nCURRENTS\n...\n", "Unknown terrain"),
    ("ROWS 1\nCOLS 3\nBATTERY 5\nTERRAIN\nH..\nCURRENTS\n..x\n", "Unknown current"),
    ("ROWS 1\nCOLS 3\nBATTERY 5\nTERRAIN\n...\nCURRENTS\n...\n", "no home"),
    ("ROWS 1\nCOLS 3\nTERRAIN\nH..\nCURRENTS\n...\n", "BATTERY"),
    ("ROWS 1\nCOLS 3\nBATTERY 5\nTERRAIN\nH11\nCURRENTS\n...\n", "twice"),
])
def test_bad_maps_rejected(text, msg):
    with pytest.raises(MapFormatError, match=msg):
        parse_map(text)
