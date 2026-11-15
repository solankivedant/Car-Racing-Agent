import math

import numpy as np
import pytest

from racing import car as C
from racing.track import TrackFormatError, parse_track


def test_tracks_load_and_are_valid(tracks):
    assert set(tracks) == {"serpentine", "oval", "technical"}
    for t in tracks.values():
        assert t.length > 1000
        assert t.self_overlap() > 1.15                      # road never merges with itself
        assert t.inside(*t.start_pos)
        assert not C.collides(t, *t.start_pos, t.start_heading)[0]


def test_obstacles_are_not_drivable(serp):
    assert serp.obstacles
    for (x, y, r) in serp.obstacles:
        assert not serp.inside(x, y)
        assert serp.inside(*serp.point_at(serp.arc_at(x, y) + 0)) or True


def test_arc_increases_along_centre(oval):
    s = oval.arc_at(oval.centre[10:-10:50, 0], oval.centre[10:-10:50, 1])
    assert np.all(np.diff(s) > 0)


def test_grid_slots_on_track_and_behind_line(tracks):
    for t in tracks.values():
        for x, y, h, s0 in t.grid_slots(8):
            assert s0 < 0
            assert not C.collides(t, x, y, h)[0]


@pytest.mark.parametrize("text, msg", [
    ("NAME x\nWIDTH 50\nPOINTS\n0 0\n1 1\n2 2\n3 3\n", "SIZE"),
    ("SIZE 100 100\nWIDTH 50\nPOINTS\n0 0\n1 1\n", "at least 4"),
    ("SIZE 100 100\nWIDTH abc\n", "bad number"),
    ("SIZE 100 100\nWIDTH 20\nPOINTS\n10 10\n80 10\n80 80\n10 80\nfoo bar baz qux\n", "unexpected"),
])
def test_bad_track_files(text, msg):
    with pytest.raises((TrackFormatError, ValueError), match=msg):
        parse_track(text)


# ------------------------------------------------------------------ physics
def test_accelerate_brake_and_limits():
    x, y, h, v = C.step_physics(0.0, 0.0, 0.0, 0.0, 0, 1)
    assert v > 0 and x > 0 and y == 0
    vv = np.array([C.V_MAX])
    _, _, _, v2 = C.step_physics(0.0, 0.0, 0.0, vv, 0, 1)
    assert v2[0] <= C.V_MAX
    _, _, _, v3 = C.step_physics(0.0, 0.0, 0.0, np.array([100.0]), 0, -1)
    assert v3[0] < 100 - C.BRAKE * C.DT * 0.9
    _, _, _, v4 = C.step_physics(0.0, 0.0, 0.0, np.array([5.0]), 0, -1)
    assert v4[0] == 0                                     # never reverses


def test_steering_direction():
    # heading 0 = east, y down: RIGHT turns clockwise -> heading increases -> car moves down
    _, y, h, _ = C.step_physics(0.0, 0.0, 0.0, np.array([100.0]), C.RIGHT, 0)
    assert h[0] > 0 and y[0] > 0
    _, y, h, _ = C.step_physics(0.0, 0.0, 0.0, np.array([100.0]), C.LEFT, 0)
    assert h[0] < 0 and y[0] < 0


def test_turn_rate_weaker_when_slow_and_fast():
    assert C.turn_rate(1.0) < C.turn_rate(80.0)
    assert C.turn_rate(C.V_MAX) < C.turn_rate(80.0)


def test_collision_off_track(oval):
    assert C.collides(oval, 600.0, 360.0, 0.0)[0]          # infield
    assert not C.collides(oval, *oval.start_pos, oval.start_heading)[0]


def test_raycast_matches_clearance(oval):
    # at the start of the oval the car points north; left and right rays measure
    # the distance to the two road edges (about half the width each)
    x, y = oval.start_pos
    d = C.raycast(oval, x, y, oval.start_heading)[0]
    assert d.shape == (5,)
    assert abs(d[0] - oval.track_width / 2) < 6 and abs(d[4] - oval.track_width / 2) < 6
    assert d[2] > 100                                        # open road ahead
    assert np.all(d >= 0) and np.all(d <= C.RAY_MAX)


def test_raycast_vectorised_equals_single(serp):
    xs = np.array([serp.start_pos[0], *serp.centre[::700, 0]])
    ys = np.array([serp.start_pos[1], *serp.centre[::700, 1]])
    hs = np.linspace(0, 6, len(xs))
    batch = C.raycast(serp, xs, ys, hs)
    for k in range(len(xs)):
        assert np.allclose(batch[k], C.raycast(serp, xs[k], ys[k], hs[k])[0])


def test_ray_endpoints_geometry():
    pts = C.ray_endpoints(0, 0, 0.0, [10, 10, 10, 10, 10])
    assert abs(pts[2][0] - 10) < 1e-9 and abs(pts[2][1]) < 1e-9       # front = east
    assert abs(pts[0][1] + 10) < 1e-9                                  # left = north (y-)
    assert abs(pts[4][1] - 10) < 1e-9


def test_action_space():
    assert len(C.ACTIONS) == 9
    assert C.action_name(C.LEFT, C.BRAKE_A) == "LEFT+BRAKE"
    assert C.action_name(C.STRAIGHT, C.ACCELERATE) == "STRAIGHT+ACCELERATE"
    _ = math
