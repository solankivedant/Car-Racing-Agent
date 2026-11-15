"""Track editor: paint your own race track.

    python main.py editor                       # new track
    python main.py editor --track tracks/oval.track   # start from an existing track

Paint the road (asphalt) and the outside area (grass/wall) with a round brush,
drop rocks on the road, and set the start line and driving direction. The editor
checks that the road forms a closed loop, shows the driving direction as a colour
gradient, saves the track (PNG + .track) and can launch a race or a training
session on it straight away.

Tools / keys
    1  ROAD brush        2  GRASS brush (outside)     3  ROCK      4  START line
    left mouse = use tool     right mouse = grass brush / remove rock
    mouse wheel or [ ]  brush size      Ctrl+Z  undo      C  clear      G  oval template
    O  load the next existing track as a starting point
    V  validate (shows the lap direction)      S  save
    P  save + race on it      T  save + train on it      F11 fullscreen     ESC quit
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from . import view as V
from .track import Track, TrackFormatError, build_from_mask, load_track, save_painted_track

ROAD, GRASS_T, ROCK_T, START_T = "ROAD", "GRASS", "ROCK", "START"
BG = (16, 18, 24)
TEXT = (235, 236, 240)
DIM = (150, 156, 168)
GOLD = (255, 205, 80)
OK = (90, 220, 130)
BAD = (255, 110, 110)
WORLD = (1200, 720)


class Editor:
    def __init__(self, track_dir="tracks", start_from: Optional[str] = None, headless=False):
        if headless:
            V.headless_env()
        else:
            V.make_dpi_aware()
        import pygame
        self.pg = pygame
        pygame.init()
        self.headless = headless
        self.track_dir = Path(track_dir)
        self.screen = V.open_window(pygame, "Car Racing Agent  -  Track Editor",
                                    size=(1500, 760) if headless else None, headless=headless)
        self.W, self.H = WORLD
        self.road = np.zeros((self.H, self.W), bool)
        self.rocks: List[Tuple[float, float, float]] = []
        self.start: Optional[Tuple[float, float, float]] = None
        self.name = "my track"
        self.tool = ROAD
        self.brush = 42
        self.undo: List[tuple] = []
        self.msg, self.msg_col = "Paint a closed loop of road, then set the START (key 4).", DIM
        self.validated: Optional[Track] = None
        self.show_dir = False
        self.typing = None                       # text being typed for the save name
        self.after_save = None
        self.request = None
        self.drag = None
        self.last = None
        self.fullscreen = False
        self.load_index = -1
        if start_from:
            self.load_existing(start_from)
        self.relayout()

    # ------------------------------------------------------------ model
    def push_undo(self):
        self.undo.append((self.road.copy(), list(self.rocks), self.start))
        self.undo = self.undo[-30:]

    def do_undo(self):
        if self.undo:
            self.road, self.rocks, self.start = self.undo.pop()
            self.dirty()

    def dirty(self):
        self.validated = None
        self.show_dir = False
        self.render_canvas()

    def stamp(self, x, y, value: bool, r=None):
        r = r or self.brush
        x0, x1 = int(max(0, x - r)), int(min(self.W, x + r + 1))
        y0, y1 = int(max(0, y - r)), int(min(self.H, y + r + 1))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        self.road[y0:y1, x0:x1][disk] = value
        # fast preview on the canvas, the pretty version is re-rendered on mouse-up
        sx, sy = x * self.lay.scale, y * self.lay.scale
        self.pg.draw.circle(self.canvas, V.ASPHALT if value else V.GRASS, (sx, sy), r * self.lay.scale)

    def stroke(self, a, b, value):
        (x0, y0), (x1, y1) = a, b
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / max(2, self.brush / 3)))
        for i in range(n + 1):
            t = i / n
            self.stamp(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, value)

    def template_oval(self):
        self.push_undo()
        self.road[:] = False
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        e = ((xx - 600) / 480) ** 2 + ((yy - 360) / 250) ** 2
        e_in = ((xx - 600) / 380) ** 2 + ((yy - 360) / 150) ** 2
        self.road = (e <= 1) & (e_in >= 1)
        self.rocks = []
        self.start = (170.0, 360.0, -math.pi / 2)
        self.msg, self.msg_col = "Oval template loaded: reshape it with the brushes.", DIM
        self.dirty()

    def load_existing(self, path):
        t = load_track(path)
        if (t.width_px, t.height_px) != WORLD:
            self.W, self.H = t.width_px, t.height_px
        road = t.mask.copy()
        yy, xx = np.mgrid[0:t.height_px, 0:t.width_px]
        for (x, y, r) in t.obstacles:
            road |= (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        self.road = road
        self.rocks = list(t.obstacles)
        self.start = (*t.start_pos, t.start_heading)
        self.name = f"{t.name} (edited)" if not t.painted else t.name
        self.msg, self.msg_col = f"Loaded '{t.name}'. Paint to change it, S to save.", DIM

    def cleaned_road(self) -> np.ndarray:
        """Smooth brush jaggies and keep a wall border."""
        r = gaussian_filter(self.road.astype(np.float32), 1.5) > 0.5
        r[:3, :] = r[-3:, :] = False
        r[:, :3] = r[:, -3:] = False
        return r

    def validate(self) -> Optional[Track]:
        if self.start is None:
            self.msg, self.msg_col = "Set the START first: tool 4, click on the road and drag the direction.", BAD
            return None
        road = self.cleaned_road()
        t = Track(self.name, self.W, self.H, 0, 1, [], list(self.rocks), file=self.slug(),
                  start=self.start, mask_file=f"{self.slug()}.png")
        try:
            build_from_mask(t, road)
        except TrackFormatError as e:
            self.msg, self.msg_col = f"Not a valid track: {e}", BAD
            return None
        from .planner import plan_lap
        try:
            plan_lap(t, "h4")
        except RuntimeError:
            self.msg, self.msg_col = ("Not a valid track: the A* planner cannot fit the car through "
                                      "somewhere. Widen the road or move a rock."), BAD
            return None
        self.validated = t
        self.show_dir = True
        self.msg, self.msg_col = (f"Valid track. Lap {t.length:.0f} px, road width ~{t.track_width:.0f} px. "
                                  "Colours show the driving direction (light to dark)."), OK
        return t

    def slug(self) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")
        return s or "my_track"

    def save(self) -> Optional[Path]:
        t = self.validate()
        if t is None:
            return None
        self.track_dir.mkdir(exist_ok=True)
        path = self.track_dir / f"{self.slug()}.track"
        if path.exists() and not (path.with_suffix(".png")).exists():
            self.msg, self.msg_col = f"'{path.name}' is a built-in spline track. Choose another name.", BAD
            return None
        save_painted_track(path, self.name, self.cleaned_road(), self.start, self.rocks)
        self.msg, self.msg_col = f"Saved {path} (lap {t.length:.0f} px).", OK
        return path

    # ------------------------------------------------------------ view
    def relayout(self):
        w, h = self.screen.get_size()
        self.lay = V.Layout.fit(w, h, self.W, self.H, panel_base=380)
        self.F = V.fonts(self.pg, self.lay.ui)
        self.render_canvas()

    def render_canvas(self):
        rgb = V.render_mask_rgb(self.road, self.lay.scale)
        if self.show_dir and self.validated is not None:
            t = self.validated
            from scipy.ndimage import zoom
            arc = zoom(t.arc.astype(np.float32), self.lay.scale, order=0)[:rgb.shape[0], :rgb.shape[1]]
            m = zoom(t.mask.astype(np.uint8), self.lay.scale, order=0)[:rgb.shape[0], :rgb.shape[1]] > 0
            f = np.clip(arc / t.length, 0, 1)[..., None]
            light, dark = np.array([205, 226, 251]), np.array([13, 54, 107])
            col = (light * (1 - f) + dark * f).astype(np.uint8)
            rgb[:arc.shape[0], :arc.shape[1]][m] = col[m]
        self.canvas = self.pg.surfarray.make_surface(rgb.swapaxes(0, 1))

    def _t(self, s, pos, f="m", color=TEXT):
        self.screen.blit(self.F[f].render(str(s), True, color), pos)

    def draw(self):
        pg, L = self.pg, self.lay
        u = L.ui
        self.screen.fill(BG)
        self.screen.blit(self.canvas, (L.ox, L.oy))
        s = L.scale
        for (x, y, r) in self.rocks:
            pg.draw.circle(self.screen, V.ROCK, L.to_screen(x, y), r * s)
            pg.draw.circle(self.screen, V.ROCK_EDGE, L.to_screen(x, y), r * s, max(1, int(2 * s)))
        if self.start:
            x, y, h = self.start
            width = 2 * max(16.0, self._local_half(x, y))
            V.draw_start_line(pg, self.screen, (x, y), h, width, s, L.ox, L.oy)
            sx, sy = L.to_screen(x, y)
            ex, ey = sx + math.cos(h) * 60 * s, sy + math.sin(h) * 60 * s
            pg.draw.line(self.screen, GOLD, (sx, sy), (ex, ey), max(3, int(3 * s)))
            for da in (2.6, -2.6):
                pg.draw.line(self.screen, GOLD, (ex, ey), (ex + math.cos(h + da) * 16 * s,
                                                           ey + math.sin(h + da) * 16 * s), max(3, int(3 * s)))
        if self.drag and self.tool == START_T:
            pg.draw.line(self.screen, GOLD, L.to_screen(*self.drag), pg.mouse.get_pos(), 2)
        mx, my = pg.mouse.get_pos()
        if mx < L.panel_x and self.tool in (ROAD, GRASS_T, ROCK_T):
            r = self.brush if self.tool != ROCK_T else self._rock_r()
            pg.draw.circle(self.screen, (255, 255, 255), (mx, my), r * s, 1)
        self._panel()
        pg.display.flip()

    def _local_half(self, x, y):
        from scipy.ndimage import distance_transform_edt
        x0, x1 = int(max(0, x - 80)), int(min(self.W, x + 80))
        y0, y1 = int(max(0, y - 80)), int(min(self.H, y + 80))
        sub = self.road[y0:y1, x0:x1]
        if not sub.any():
            return 16.0
        d = distance_transform_edt(sub)
        yi, xi = int(min(y1 - y0 - 1, max(0, y - y0))), int(min(x1 - x0 - 1, max(0, x - x0)))
        return float(d[yi, xi])

    def _rock_r(self):
        return float(min(16, max(6, self.brush // 3)))

    def _panel(self):
        pg, L = self.pg, self.lay
        u = L.ui
        x0 = L.panel_x + int(14 * u)
        pg.draw.rect(self.screen, BG, (L.panel_x, 0, L.panel_w, L.win_h))
        y = int(10 * u)
        self._t("TRACK EDITOR", (x0, y), "l"); y += int(30 * u)
        self._t(f"name: {self.name}", (x0, y), "m", GOLD); y += int(26 * u)
        tools = [("1", ROAD, "road brush (asphalt)"), ("2", GRASS_T, "grass brush (outside)"),
                 ("3", ROCK_T, "rock on the road"), ("4", START_T, "start line + direction")]
        for key, tl, desc in tools:
            on = self.tool == tl
            col = GOLD if on else TEXT
            if on:
                pg.draw.rect(self.screen, (44, 48, 60), (x0 - 6 * u, y - 2 * u, L.panel_w - 16 * u, 20 * u),
                             border_radius=4)
            self._t(f"[{key}] {tl:<6} {desc}", (x0, y), "m", col); y += int(22 * u)
        y += int(6 * u)
        self._t(f"brush size {self.brush} px   (wheel / [ ])", (x0, y), "s", DIM); y += int(16 * u)
        self._t(f"rocks {len(self.rocks)}   start {'set' if self.start else 'NOT SET'}", (x0, y), "s",
                OK if self.start else BAD); y += int(26 * u)
        lines = ["left mouse   use tool", "right mouse  grass brush / remove rock",
                 "START: click on the road, drag the", "       direction of driving",
                 "", "V  validate (shows lap direction)", "S  save           Ctrl+Z  undo",
                 "P  save + race    T  save + train", "G  oval template  C  clear",
                 "O  load next existing track", "F11 fullscreen   ESC quit"]
        for ln in lines:
            self._t(ln, (x0, y), "s", DIM); y += int(15 * u)
        y += int(12 * u)
        for ln in _wrap(self.msg, 44):
            self._t(ln, (x0, y), "s", self.msg_col); y += int(15 * u)
        if self.typing is not None:
            y += int(10 * u)
            pg.draw.rect(self.screen, (44, 48, 60), (x0 - 6 * u, y - 4 * u, L.panel_w - 16 * u, 48 * u),
                         border_radius=4)
            self._t("track name (Enter = save, Esc = cancel):", (x0, y), "s", GOLD); y += int(18 * u)
            self._t(self.typing + "_", (x0, y), "m", TEXT)

    # ------------------------------------------------------------ input
    def handle(self, ev) -> bool:
        pg, L = self.pg, self.lay
        if ev.type == pg.QUIT:
            return False
        if ev.type == pg.VIDEORESIZE and not self.fullscreen:
            self.screen = pg.display.set_mode((max(900, ev.w), max(560, ev.h)), pg.RESIZABLE)
            self.relayout()
            return True
        if self.typing is not None and ev.type == pg.KEYDOWN:
            if ev.key == pg.K_RETURN:
                if self.typing.strip():
                    self.name = self.typing.strip()
                self.typing = None
                path = self.save()
                if path and self.after_save:
                    self.request = (self.after_save, str(path))
                    return False
            elif ev.key == pg.K_ESCAPE:
                self.typing, self.after_save = None, None
            elif ev.key == pg.K_BACKSPACE:
                self.typing = self.typing[:-1]
            elif ev.unicode and ev.unicode.isprintable() and len(self.typing) < 30:
                self.typing += ev.unicode
            return True
        if ev.type == pg.KEYDOWN:
            k, mods = ev.key, pg.key.get_mods()
            if k == pg.K_ESCAPE:
                if self.fullscreen:
                    self.toggle_fullscreen()
                    return True
                return False
            if k == pg.K_z and mods & pg.KMOD_CTRL:
                self.do_undo()
            elif k == pg.K_1:
                self.tool = ROAD
            elif k == pg.K_2:
                self.tool = GRASS_T
            elif k == pg.K_3:
                self.tool = ROCK_T
            elif k == pg.K_4:
                self.tool = START_T
            elif k == pg.K_RIGHTBRACKET:
                self.brush = min(90, self.brush + 4)
            elif k == pg.K_LEFTBRACKET:
                self.brush = max(8, self.brush - 4)
            elif k == pg.K_c:
                self.push_undo()
                self.road[:] = False
                self.rocks, self.start = [], None
                self.dirty()
            elif k == pg.K_g:
                self.template_oval()
            elif k == pg.K_o:
                files = sorted(self.track_dir.glob("*.track"))
                if files:
                    self.push_undo()
                    self.load_index = (self.load_index + 1) % len(files)
                    try:
                        self.load_existing(files[self.load_index])
                    except TrackFormatError as e:
                        self.msg, self.msg_col = f"Could not load {files[self.load_index].name}: {e}", BAD
                    self.dirty()
            elif k == pg.K_v:
                self.validate()
                self.render_canvas()
            elif k in (pg.K_s, pg.K_p, pg.K_t):
                self.after_save = {pg.K_s: None, pg.K_p: "race", pg.K_t: "train"}[k]
                self.typing = self.name
            elif k == pg.K_F11:
                self.toggle_fullscreen()
        if ev.type == pg.MOUSEWHEEL:
            self.brush = int(min(90, max(8, self.brush + 3 * ev.y)))
        if ev.type == pg.MOUSEBUTTONDOWN and ev.button in (1, 3) and ev.pos[0] < L.panel_x:
            wx, wy = L.to_world(*ev.pos)
            if self.tool == START_T and ev.button == 1:
                self.drag = (wx, wy)
            elif self.tool == ROCK_T:
                self.push_undo()
                if ev.button == 1:
                    self.rocks.append((round(wx), round(wy), self._rock_r()))
                elif self.rocks:
                    d = [(x - wx) ** 2 + (y - wy) ** 2 for x, y, _ in self.rocks]
                    i = int(np.argmin(d))
                    if d[i] < 40 ** 2:
                        self.rocks.pop(i)
                self.validated = None
            else:
                self.push_undo()
                self.paint_value = (ev.button == 1 and self.tool == ROAD)
                self.last = (wx, wy)
                self.stamp(wx, wy, self.paint_value)
        if ev.type == pg.MOUSEMOTION and self.last is not None:
            wx, wy = L.to_world(*ev.pos)
            self.stroke(self.last, (wx, wy), self.paint_value)
            self.last = (wx, wy)
        if ev.type == pg.MOUSEBUTTONUP:
            if self.last is not None:
                self.last = None
                self.dirty()
            if self.drag is not None and ev.button == 1:
                wx, wy = L.to_world(*ev.pos)
                x, y = self.drag
                h = math.atan2(wy - y, wx - x) if math.hypot(wx - x, wy - y) > 8 else (
                    self.start[2] if self.start else -math.pi / 2)
                self.push_undo()
                self.start = (x, y, h)
                self.drag = None
                self.validated = None
                self.msg, self.msg_col = "Start set. Press V to validate or S to save.", DIM
        return True

    def toggle_fullscreen(self):
        pg = self.pg
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            info = pg.display.Info()
            self.screen = pg.display.set_mode((info.current_w, info.current_h), pg.FULLSCREEN)
        else:
            self.screen = V.open_window(pg, "Car Racing Agent  -  Track Editor")
        self.relayout()

    def loop(self):
        """Returns a request ("race"|"train", track_file) or None."""
        clock = self.pg.time.Clock()
        running = True
        while running:
            for ev in self.pg.event.get():
                running = self.handle(ev) and running
            self.draw()
            clock.tick(60)
        return self.request


def _wrap(s: str, n: int):
    words, lines, cur = s.split(" "), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > n:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    if cur:
        lines.append(cur)
    return lines
