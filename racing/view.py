"""Shared pygame rendering helpers (race GUI and track editor).

Sharp output on any screen:
* the process is made DPI-aware on Windows, so Windows does not blur the window
  by stretching it on high-DPI / 125-150 % scaled displays;
* nothing is drawn at 1200x720 and then stretched. The track background is
  rendered directly at the window's scale from a signed distance field, so road
  edges stay smooth at any size, and cars, rays and lines are drawn in screen
  coordinates.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, zoom

from .track import Track

GRASS = (34, 52, 74)
HATCH = (52, 84, 120)
ASPHALT = (70, 74, 82)
EDGE = (235, 235, 235)
DASH = (118, 122, 132)
ROCK = (120, 92, 60)
ROCK_EDGE = (60, 40, 25)


def make_dpi_aware():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)      # per-monitor aware
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def open_window(pg, title: str, size=None, headless=False):
    """Maximised, resizable window (or a fixed-size surface when headless)."""
    if headless:
        return pg.display.set_mode(size)
    info = pg.display.Info()
    w, h = size or (info.current_w, max(600, info.current_h - 80))
    screen = pg.display.set_mode((w, h), pg.RESIZABLE)
    pg.display.set_caption(title)
    if sys.platform.startswith("win") and size is None:
        try:
            import ctypes
            hwnd = pg.display.get_wm_info()["window"]
            ctypes.windll.user32.ShowWindow(hwnd, 3)                # SW_MAXIMIZE
        except Exception:
            pass
    return screen


@dataclass
class Layout:
    """Where the world (track) goes on screen, next to a side panel."""
    win_w: int
    win_h: int
    world_w: int
    world_h: int
    panel_w: int
    scale: float
    ox: int
    oy: int
    ui: float

    @staticmethod
    def fit(win_w, win_h, world_w, world_h, panel_base=360, scale=None):
        ui = max(1.0, min(2.0, win_h / 760))
        panel = int(panel_base * ui)
        if scale is None:
            scale = max(0.3, min((win_w - panel) / world_w, win_h / world_h))
        vw, vh = int(world_w * scale), int(world_h * scale)
        ox = max(0, (win_w - panel - vw) // 2)
        oy = max(0, (win_h - vh) // 2)
        return Layout(win_w, win_h, world_w, world_h, panel, scale, ox, oy, ui)

    def to_screen(self, x, y):
        return self.ox + x * self.scale, self.oy + y * self.scale

    def to_world(self, sx, sy):
        return (sx - self.ox) / self.scale, (sy - self.oy) / self.scale

    @property
    def panel_x(self):
        return self.win_w - self.panel_w

    @property
    def view_rect(self):
        return (self.ox, self.oy, int(self.world_w * self.scale), int(self.world_h * self.scale))


def render_mask_rgb(mask: np.ndarray, scale: float, obstacles=(), edge_px: float = 2.0) -> np.ndarray:
    """RGB image (h, w, 3) of a road mask at ``scale``, with smooth edges (via an SDF)."""
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    sdf = np.where(mask, inside, -outside + 1).astype(np.float32)
    z = zoom(sdf, scale, order=1) if abs(scale - 1) > 1e-6 else sdf
    h, w = z.shape
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[:] = GRASS
    yy, xx = np.mgrid[0:h, 0:w]
    per = max(8, int(14 * scale))
    rgb[((xx + yy) % per) < max(2, int(2 * scale))] = HATCH
    road = z > 0.5
    rgb[road] = ASPHALT
    rgb[road & (z < 0.5 + edge_px)] = EDGE
    return rgb


def render_track(pg, track: Track, layout: Layout):
    """Background surface for ``track`` at ``layout.scale`` (sharp, no stretching)."""
    s = layout.scale
    road = track.mask.copy()
    yy, xx = np.mgrid[0:track.height_px, 0:track.width_px]
    for (ox, oy, r) in track.obstacles:                    # rocks are drawn separately
        road |= ((xx - ox) ** 2 + (yy - oy) ** 2 <= r * r) & (track.clearance > -1)
    road[:3, :] = road[-3:, :] = False
    road[:, :3] = road[:, -3:] = False
    rgb = render_mask_rgb(road, s)
    surf = pg.surfarray.make_surface(rgb.swapaxes(0, 1))
    # dashed centre line
    c, sv = track.centre, track.s
    step = max(1, int(len(c) / max(1, track.length) * 3))
    on = (sv % 40) < 18
    seg = []
    for i in range(0, len(c), step):
        if on[i]:
            seg.append((c[i, 0] * s, c[i, 1] * s))
        elif len(seg) > 1:
            pg.draw.lines(surf, DASH, False, seg, max(1, int(s)))
            seg = []
        else:
            seg = []
    # rocks
    for (ox, oy, r) in track.obstacles:
        pg.draw.circle(surf, ROCK, (ox * s, oy * s), r * s)
        pg.draw.circle(surf, ROCK_EDGE, (ox * s, oy * s), r * s, max(1, int(2 * s)))
    draw_start_line(pg, surf, track.start_pos, track.start_heading, track.track_width, s)
    return surf


def draw_start_line(pg, surf, pos, heading, width, s, ox=0, oy=0):
    sx, sy = pos
    hx, hy = math.cos(heading), math.sin(heading)
    nx, ny = -hy, hx
    half = width / 2
    sq = 6
    for i in range(int(-half), int(half), sq):
        for j in range(2):
            col = (245, 245, 245) if ((i // sq) + j) % 2 == 0 else (20, 20, 20)
            cx = sx + nx * (i + sq / 2) + hx * (j - 0.5) * sq
            cy = sy + ny * (i + sq / 2) + hy * (j - 0.5) * sq
            pg.draw.rect(surf, col, (ox + (cx - sq / 2) * s, oy + (cy - sq / 2) * s,
                                     math.ceil(sq * s), math.ceil(sq * s)))


_SPRITES = {}


def car_sprite(pg, color, scale, alpha=255, selected=False, length=20.0, width=10.0):
    key = (tuple(color), round(scale, 3), alpha, selected)
    if key in _SPRITES:
        return _SPRITES[key]
    L, W = length * scale, width * scale
    pad = 2 * scale
    s = pg.Surface((L + 2 * pad, W + 2 * pad), pg.SRCALPHA)
    body = pg.Rect(pad, pad, L, W)
    rad = max(2, int(3 * scale))
    pg.draw.rect(s, (*color, alpha), body, border_radius=rad)
    pg.draw.rect(s, (20, 20, 20, alpha), (pad + L * 0.45, pad + W * 0.12, L * 0.25, W * 0.76), border_radius=rad)
    lw = max(2, int(3 * scale))
    pg.draw.rect(s, (255, 255, 210, alpha), (pad + L - lw, pad, lw, lw))
    pg.draw.rect(s, (255, 255, 210, alpha), (pad + L - lw, pad + W - lw, lw, lw))
    if selected:
        pg.draw.rect(s, (255, 255, 255, 255), body, max(1, int(scale)), border_radius=rad)
    _SPRITES[key] = s
    return s


def blit_car(pg, screen, sprite, sx, sy, heading):
    r = pg.transform.rotozoom(sprite, -math.degrees(heading), 1.0)
    screen.blit(r, r.get_rect(center=(sx, sy)))


def fonts(pg, ui: float):
    mono = "consolas,dejavusansmono,menlo,monospace"
    return dict(s=pg.font.SysFont(mono, int(12 * ui)), m=pg.font.SysFont(mono, int(14 * ui)),
                b=pg.font.SysFont(mono, int(16 * ui), bold=True), l=pg.font.SysFont(mono, int(20 * ui), bold=True))


def headless_env():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
