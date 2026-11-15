"""Painted tracks (track editor) and the sharp, resizable GUI layout."""
import math
from pathlib import Path

import numpy as np
import pytest

from racing.agents import PathAgent, ReflexAgent, UtilityAgent
from racing.race import FINISHED, run_race
from racing.track import Track, TrackFormatError, build_from_mask, load_track, save_painted_track
from racing.view import Layout, render_mask_rgb


def ring(W=1200, H=720, rx=480, ry=250, width=80):
    yy, xx = np.mgrid[0:H, 0:W]
    e_out = ((xx - 600) / rx) ** 2 + ((yy - 360) / ry) ** 2
    e_in = ((xx - 600) / (rx - width)) ** 2 + ((yy - 360) / (ry - width)) ** 2
    return (e_out <= 1) & (e_in >= 1)


def painted(road, start, rocks=()):
    t = Track("p", 0, 0, 0, 1, [], list(rocks), start=start, mask_file="x.png")
    return build_from_mask(t, road)


START = (160.0, 360.0, -math.pi / 2)


def test_painted_ring_lap_length_and_progress():
    t = painted(ring(), START)
    assert t.painted
    # geodesic lap is between the inner and outer ellipse perimeters
    def perim(a, b):
        return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    assert perim(400, 170) * 0.95 < t.length < perim(480, 250)
    assert abs(t.track_width - 80) < 6
    # progress increases in the driving direction (north from the left side, i.e. clockwise)
    a = [t.arc_at(*t.point_at(s)) for s in (50, 500, 1000, 1500)]
    assert all(np.diff(a) > 0)
    assert t.start_pos == (160.0, 360.0)


def test_painted_track_roundtrip_and_race(tmp_path):
    road = ring()
    path = save_painted_track(tmp_path / "ring.track", "Ring", road, START, [(1080, 360, 8)])
    assert path.exists() and path.with_suffix(".png").exists()
    t = load_track(path)
    assert t.name == "Ring" and len(t.obstacles) == 1
    assert not t.inside(1080, 360)                       # the rock is not drivable
    for ag in (ReflexAgent(), UtilityAgent(), PathAgent("h4")):
        r = run_race(t, [ag])
        assert r.cars[0].status == FINISHED, ag.name


def test_open_road_is_rejected():
    road = np.zeros((720, 1200), bool)
    road[300:380, 100:1100] = True                       # a straight, not a loop
    with pytest.raises(TrackFormatError, match="closed loop"):
        painted(road, (300, 340, 0.0))


def test_start_off_road_and_narrow_are_rejected():
    with pytest.raises(TrackFormatError, match="not on the road"):
        painted(ring(), (600, 360, 0.0))
    with pytest.raises(TrackFormatError, match="too narrow"):
        painted(ring(width=20), (122, 360, -math.pi / 2))


def test_mask_file_missing(tmp_path):
    (tmp_path / "x.track").write_text("NAME x\nMASK nope.png\nSTART 1 1 0\n")
    with pytest.raises(TrackFormatError, match="not found"):
        load_track(tmp_path / "x.track")


def test_editor_paint_validate_save(tmp_path):
    pg = pytest.importorskip("pygame")
    from racing.editor import ROCK_T, START_T, Editor
    ed = Editor(track_dir=tmp_path, headless=True)
    ed.template_oval()
    assert ed.validate() is not None, ed.msg
    # painting grass across the road breaks the loop -> validation fails with a message
    ed.push_undo()
    ed.brush = 60
    ed.stamp(1040, 360, False)                          # erase across the whole road width
    assert ed.validate() is None and "closed loop" in ed.msg
    ed.do_undo()
    assert ed.validate() is not None
    ed.name = "Unit Test Oval"
    p = ed.save()
    assert p == tmp_path / "unit_test_oval.track"
    t = load_track(p)
    assert t.length > 1500
    ed.draw()
    _ = (pg, ROCK_T, START_T)


def test_editor_refuses_to_overwrite_builtin(tmp_path):
    pytest.importorskip("pygame")
    from racing.editor import Editor
    (tmp_path / "oval.track").write_text("builtin")
    ed = Editor(track_dir=tmp_path, headless=True)
    ed.template_oval()
    ed.name = "oval"
    assert ed.save() is None and "built-in" in ed.msg


def test_layout_fits_and_maps_coordinates():
    L = Layout.fit(1920, 1040, 1200, 720)
    assert L.scale > 1.0                                  # big screens get a bigger, not stretched, view
    assert L.world_w * L.scale + L.panel_w <= 1920 + 1
    x, y = L.to_world(*L.to_screen(321.0, 123.0))
    assert abs(x - 321) < 1e-9 and abs(y - 123) < 1e-9


def test_background_rendered_at_screen_resolution():
    road = ring()
    rgb = render_mask_rgb(road, 1.5)
    assert rgb.shape[:2] == (1080, 1800)                  # rendered natively, not upscaled later


def test_gui_headless_draw(tracks):
    pytest.importorskip("pygame")
    from racing.gui import App
    files = [str(Path(__file__).resolve().parents[1] / "tracks" / f"{n}.track") for n in ("oval", "technical")]
    app = App(files, None, headless=True, scale=1.25)
    for _ in range(30):
        app.tick()
    app.draw()
    app.switch_track(1)
    app.mode = "train"
    app.start()
    app.tick()
    app.draw()
    assert app.lay.scale == 1.25
