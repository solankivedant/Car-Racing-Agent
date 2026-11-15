"""pygame top-down viewer: replays a mission with a smoothly animated car.

Shows the track tiles, current/slope arrows, the planned A* route, the trail
driven so far, the car with its 5 ray sensors (w1..w5) and a HUD with the fuel
bar, the rule that fired, the car action (steer + throttle) and the mission
counters.

Keys
    SPACE  pause / resume          RIGHT  single step (while paused)
    + / -  faster / slower         R      restart
    0..5   re-run with heuristic h0..h5 (and 6 for the bonus weighted A*)
    M      toggle survey / race    T      toggle trail
    S      screenshot to demo/     ESC/Q  quit
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional

from .agent import RunResult, run_mission
from .reflex import RULE_BY_ID
from .sensors import ray_endpoints
from .world import GRASS, WALL, Grid

TILE = 36
HUD_W = 330
FPS = 60

# palette
C_BG = (18, 20, 26)
C_ASPHALT = (78, 82, 92)
C_ASPHALT_LINE = (96, 100, 110)
C_GRASS = (70, 140, 70)
C_GRASS_DARK = (58, 120, 58)
C_WALL = (30, 32, 38)
C_WALL_EDGE = (200, 60, 60)
C_HOME = (220, 70, 60)
C_PIT = (240, 160, 40)
C_SITE = (130, 80, 200)
C_SITE_DONE = (90, 200, 120)
C_PLAN = (90, 190, 255)
C_TRAIL = (255, 230, 120)
C_ARROW = (160, 200, 255)
C_TEXT = (235, 235, 240)
C_DIM = (150, 155, 165)
C_RAY = (255, 90, 90)
C_CAR = (230, 40, 50)
C_CAR_TOP = (250, 200, 60)

HEADING_ANGLE = {"N": 0, "E": 90, "S": 180, "W": 270}   # clockwise, 0 = up


def _ascii(s: str) -> str:
    """Logic symbols -> ASCII (pygame's default fonts often lack the glyphs)."""
    for a, b in (("∧", "&"), ("∨", "|"), ("¬", "~"), ("≤", "<="), ("⊤", "true"), ("–", "-")):
        s = s.replace(a, b)
    return s


def _center(p, ox=0, oy=0):
    return (ox + p[1] * TILE + TILE / 2, oy + p[0] * TILE + TILE / 2)


class Viewer:
    def __init__(self, grid: Grid, heuristic: str = "h3", mode: str = "survey",
                 speed: float = 6.0, headless: bool = False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame  # imported lazily so the rest of the project runs without it
        self.pg = pygame
        pygame.init()
        self.grid = grid
        self.heuristic, self.mode = heuristic, mode
        self.speed = speed                    # ticks per second
        self.W = grid.cols * TILE + HUD_W
        self.H = max(grid.rows * TILE, 560)
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Car Racing Agent – reflex + A*")
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 15)
        self.font_b = pygame.font.SysFont("consolas,dejavusansmono,monospace", 17, bold=True)
        self.font_s = pygame.font.SysFont("consolas,dejavusansmono,monospace", 12)
        self.show_trail = True
        self.background = self._draw_background()
        self.run_agent()

    # ----------------------------------------------------------------- data
    def run_agent(self):
        self.result: RunResult = run_mission(self.grid, self.heuristic, self.mode)
        self.frames = self._build_frames(self.result)
        self.i = 0
        self.phase = 0.0
        self.paused = False

    def _build_frames(self, res: RunResult):
        """One frame per log entry, with the position before and after."""
        frames = []
        pos, fuel, heading = self.grid.home, self.grid.capacity, "N"
        visited: List[int] = []
        trail = [pos]
        plan0 = res.log[0].plan if res.log else ()
        frames.append(dict(a=pos, b=pos, h0=heading, h1=heading, fuel=fuel, e=None,
                           visited=list(visited), trail=list(trail), plan=plan0))
        for e in res.log:
            a, h0 = pos, heading
            if e.action == "MOVE":
                pos, heading = e.pos, e.heading
                trail.append(pos)
            if e.action == "MARK_CHECKPOINT":
                visited.append(self.grid.site_at(e.pos))
            fuel = e.fuel
            frames.append(dict(a=a, b=pos, h0=h0, h1=heading, fuel=fuel, e=e,
                               visited=list(visited), trail=list(trail), plan=e.plan))
        return frames

    # ------------------------------------------------------------- drawing
    def _draw_background(self):
        pg = self.pg
        g = self.grid
        surf = pg.Surface((g.cols * TILE, g.rows * TILE))
        for r in range(g.rows):
            for c in range(g.cols):
                rect = pg.Rect(c * TILE, r * TILE, TILE, TILE)
                t = g.terrain[r][c]
                if t == WALL:
                    pg.draw.rect(surf, C_WALL, rect)
                    pg.draw.rect(surf, C_WALL_EDGE, rect.inflate(-6, -6), 2)
                elif t == GRASS:
                    pg.draw.rect(surf, C_GRASS, rect)
                    for k in range(4, TILE, 9):
                        pg.draw.line(surf, C_GRASS_DARK, (rect.x + k, rect.y + 4),
                                     (rect.x + k - 3, rect.y + TILE - 4), 1)
                else:
                    pg.draw.rect(surf, C_ASPHALT, rect)
                    pg.draw.rect(surf, C_ASPHALT_LINE, rect, 1)
                ch = g.raw[r][c]
                if ch == "H":
                    sq = TILE // 4
                    for i in range(4):
                        for j in range(4):
                            col = (240, 240, 240) if (i + j) % 2 == 0 else (20, 20, 20)
                            pg.draw.rect(surf, col, (rect.x + i * sq, rect.y + j * sq, sq, sq))
                    pg.draw.rect(surf, C_HOME, rect, 3)
                elif ch == "C":
                    pg.draw.rect(surf, C_PIT, rect.inflate(-4, -4), border_radius=5)
                    self._text(surf, "P", rect.center, self.font_b, (40, 30, 10), center=True)
                cur = g.current[r][c]
                if cur:
                    self._arrow(surf, rect.center, cur)
        return surf

    def _arrow(self, surf, center, d):
        pg = self.pg
        ang = math.radians(HEADING_ANGLE[d])
        dx, dy = math.sin(ang), -math.cos(ang)
        L = TILE * 0.28
        x, y = center
        tip = (x + dx * L, y + dy * L)
        tail = (x - dx * L, y - dy * L)
        pg.draw.line(surf, C_ARROW, tail, tip, 2)
        px, py = -dy, dx
        pg.draw.polygon(surf, C_ARROW, [tip, (tip[0] - dx * 7 + px * 5, tip[1] - dy * 7 + py * 5),
                                        (tip[0] - dx * 7 - px * 5, tip[1] - dy * 7 - py * 5)])

    def _text(self, surf, s, pos, font=None, color=C_TEXT, center=False):
        img = (font or self.font).render(str(s), True, color)
        r = img.get_rect()
        if center:
            r.center = pos
        else:
            r.topleft = pos
        surf.blit(img, r)

    def _draw_car(self, x, y, angle_deg):
        pg = self.pg
        car = pg.Surface((TILE * 0.55, TILE * 0.85), pg.SRCALPHA)
        w, h = car.get_size()
        pg.draw.rect(car, C_CAR, (0, 0, w, h), border_radius=7)
        pg.draw.rect(car, C_CAR_TOP, (w * 0.2, h * 0.28, w * 0.6, h * 0.35), border_radius=4)
        pg.draw.rect(car, (20, 20, 20), (0, h * 0.08, 3, h * 0.25))
        pg.draw.rect(car, (20, 20, 20), (w - 3, h * 0.08, 3, h * 0.25))
        pg.draw.rect(car, (20, 20, 20), (0, h * 0.67, 3, h * 0.25))
        pg.draw.rect(car, (20, 20, 20), (w - 3, h * 0.67, 3, h * 0.25))
        pg.draw.circle(car, (255, 255, 200), (w * 0.25, 3), 3)
        pg.draw.circle(car, (255, 255, 200), (w * 0.75, 3), 3)
        rot = pg.transform.rotate(car, -angle_deg)
        self.screen.blit(rot, rot.get_rect(center=(x, y)))

    def draw(self):
        pg = self.pg
        g = self.grid
        f = self.frames[self.i]
        nxt = self.frames[min(self.i + 1, len(self.frames) - 1)]
        s = self.screen
        s.fill(C_BG)
        s.blit(self.background, (0, 0))

        # checkpoints
        for k, p in g.sites.items():
            done = k in f["visited"]
            pg.draw.circle(s, C_SITE_DONE if done else C_SITE, _center(p), TILE * 0.38)
            self._text(s, k, _center(p), self.font_b, C_TEXT, center=True)

        # planned route
        plan = [f["b"], *f["plan"]]
        if len(plan) > 1:
            pts = [_center(p) for p in plan]
            pg.draw.lines(s, C_PLAN, False, pts, 3)
        # trail
        if self.show_trail and len(f["trail"]) > 1:
            pg.draw.lines(s, C_TRAIL, False, [_center(p) for p in f["trail"]], 2)

        # car (interpolate towards the next frame)
        t = self.phase
        a, b = nxt["a"], nxt["b"]
        ax, ay = _center(a)
        bx, by = _center(b)
        x, y = ax + (bx - ax) * t, ay + (by - ay) * t
        h0, h1 = HEADING_ANGLE[nxt["h0"]], HEADING_ANGLE[nxt["h1"]]
        dh = ((h1 - h0 + 180) % 360) - 180
        ang = h0 + dh * min(1.0, t * 2)
        # rays from the tile the car currently occupies
        cur_tile = a if t < 0.5 else b
        heading = nxt["h0"] if t < 0.5 else nxt["h1"]
        for name, end in ray_endpoints(g, cur_tile, heading):
            ex, ey = end[1] * TILE + TILE / 2, end[0] * TILE + TILE / 2
            pg.draw.line(s, C_RAY, (x, y), (ex, ey), 1)
            pg.draw.circle(s, C_RAY, (ex, ey), 3)
        self._draw_car(x, y, ang)
        self._draw_hud(nxt)
        pg.display.flip()

    def _draw_hud(self, f):
        pg = self.pg
        s = self.screen
        x0 = self.grid.cols * TILE + 14
        y = 12
        res = self.result
        self._text(s, "CAR RACING AGENT", (x0, y), self.font_b); y += 24
        self._text(s, f"map: {self.grid.name}", (x0, y), self.font, C_DIM); y += 18
        self._text(s, f"heuristic: {self.heuristic}   mode: {self.mode}", (x0, y), self.font, C_DIM)
        y += 26
        # fuel bar
        cap = self.grid.capacity
        pct = f["fuel"] / cap
        self._text(s, f"FUEL {f['fuel']:g}/{cap}  ({pct*100:.0f}%)", (x0, y)); y += 20
        bw = HUD_W - 30
        pg.draw.rect(s, (50, 50, 60), (x0, y, bw, 16), border_radius=4)
        col = (220, 60, 60) if pct < 0.25 else (240, 180, 50) if pct <= 0.40 else (80, 200, 110)
        pg.draw.rect(s, col, (x0, y, int(bw * pct), 16), border_radius=4)
        for th in (0.25, 0.40):
            pg.draw.line(s, C_TEXT, (x0 + bw * th, y - 2), (x0 + bw * th, y + 18), 1)
        y += 28
        e = f["e"]
        if e is not None:
            rule = RULE_BY_ID.get(e.rule)
            self._text(s, f"tick {e.t}", (x0, y), self.font, C_DIM); y += 20
            self._text(s, f"RULE {e.rule}: {rule.name if rule else ''}", (x0, y), self.font_b,
                       (255, 210, 90)); y += 20
            if rule:
                self._text(s, _ascii(rule.formula)[:44], (x0, y), self.font_s, C_DIM); y += 16
            self._text(s, f"action : {e.action}", (x0, y)); y += 18
            self._text(s, f"car    : {e.car_action or '-'}", (x0, y)); y += 18
            self._text(s, f"mode   : {e.mode}  target: {e.target}", (x0, y)); y += 18
            if e.note:
                self._text(s, e.note[:42], (x0, y), self.font_s, (140, 220, 255)); y += 16
            y += 6
            self._text(s, "ray sensors (tiles to wall)", (x0, y), self.font_s, C_DIM); y += 16
            for k, v in e.rays.items():
                self._text(s, f"  {k:<15}{v:>3}", (x0, y), self.font_s); y += 15
        y += 8
        done = len(f["visited"])
        self._text(s, f"checkpoints: {done}/{len(self.grid.sites)}  {f['visited']}", (x0, y)); y += 20
        self._text(s, f"moves {len(f['trail'])-1}   pit stops {res.pit_stops}", (x0, y)); y += 20
        self._text(s, f"final: {res.status}  fuel used {res.fuel_used:g}", (x0, y), self.font,
                   C_DIM); y += 20
        self._text(s, f"A*: {res.searches} searches, {res.nodes_expanded} expanded", (x0, y),
                   self.font_s, C_DIM)
        y = self.H - 64
        for line in ("SPACE pause  +/- speed  R restart", "0-5 heuristic  M mode  T trail",
                     "S screenshot  Q quit"):
            self._text(s, line, (x0, y), self.font_s, C_DIM); y += 15

    # ---------------------------------------------------------------- loop
    def advance(self, dt: float):
        if self.paused or self.i >= len(self.frames) - 1:
            return
        nxt = self.frames[self.i + 1]
        moving = nxt["a"] != nxt["b"]
        self.phase += dt * self.speed * (1 if moving else 2.5)
        if self.phase >= 1.0:
            self.phase = 0.0
            self.i += 1

    def loop(self):
        pg = self.pg
        clock = pg.time.Clock()
        running = True
        while running:
            dt = clock.tick(FPS) / 1000
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    k = ev.key
                    if k in (pg.K_ESCAPE, pg.K_q):
                        running = False
                    elif k == pg.K_SPACE:
                        self.paused = not self.paused
                    elif k == pg.K_RIGHT and self.paused:
                        self.i = min(self.i + 1, len(self.frames) - 1)
                        self.phase = 0
                    elif k in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                        self.speed = min(self.speed * 1.5, 80)
                    elif k in (pg.K_MINUS, pg.K_KP_MINUS):
                        self.speed = max(self.speed / 1.5, 0.5)
                    elif k == pg.K_r:
                        self.i, self.phase = 0, 0.0
                    elif k == pg.K_t:
                        self.show_trail = not self.show_trail
                    elif k == pg.K_m:
                        self.mode = "race" if self.mode == "survey" else "survey"
                        self.run_agent()
                    elif pg.K_0 <= k <= pg.K_6:
                        self.heuristic = f"h{k - pg.K_0}" if k != pg.K_6 else "h5"
                        self.run_agent()
                    elif k == pg.K_s:
                        Path("demo").mkdir(exist_ok=True)
                        out = f"demo/screenshot_{self.grid.name}_{self.heuristic}_{self.i}.png"
                        pg.image.save(self.screen, out)
                        print("saved", out)
            self.advance(dt)
            self.draw()
        pg.quit()

    def record(self, out_path: str, fps: int = 20, substeps: int = 3, hold_end: int = 30,
               max_frames: Optional[int] = None):
        """Render the whole mission to a GIF or MP4 (needs imageio)."""
        import imageio.v2 as imageio
        import numpy as np
        pg = self.pg
        images = []
        self.i, self.phase = 0, 0.0
        while self.i < len(self.frames) - 1:
            nxt = self.frames[self.i + 1]
            steps = substeps if nxt["a"] != nxt["b"] else 2
            for k in range(steps):
                self.phase = k / steps
                self.draw()
                arr = pg.surfarray.array3d(self.screen).swapaxes(0, 1)
                images.append(np.ascontiguousarray(arr))
            self.i += 1
            self.phase = 0.0
            if max_frames and len(images) >= max_frames:
                break
        self.draw()
        last = np.ascontiguousarray(pg.surfarray.array3d(self.screen).swapaxes(0, 1))
        images.extend([last] * hold_end)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_path.endswith(".gif"):
            imageio.mimsave(out_path, images, duration=1 / fps, loop=0)
        else:
            imageio.mimsave(out_path, images, fps=fps)
        pg.image.save(self.screen, str(Path(out_path).with_suffix(".png")))
        return out_path
