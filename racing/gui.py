"""pygame front-end: watch the race, or watch the learning agent evolve.

RACE mode  : the full lineup (A* h1-h4, Utility, Reflex, Learned GA) races on one track.
TRAIN mode : a whole population of neural-network cars learns to drive, generation
             by generation (genetic algorithm), live on screen.

The window opens maximised, can be resized, and F11 switches to fullscreen.
Everything is drawn directly at screen resolution (sharp, no stretching).

Keys
    SPACE  pause            TAB / click  select car        1-9 / [ ]  choose track
    T      race <-> train   V  rays on all cars            L  racing lines
    + / -  sim speed        R  restart                     S  screenshot
    E      open the track editor on this track             F11 fullscreen
    ESC / Q quit
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import car as C
from . import view as V
from .agents import Agent, PathAgent, UtilityAgent, default_lineup
from .learning import GAConfig, PopulationSim, next_generation, random_population
from .race import CRASHED, FINISHED, Race
from .track import Track, load_track

BG = (16, 18, 24)
TEXT = (235, 236, 240)
DIM = (150, 156, 168)
GOLD = (255, 205, 80)
INFO = (140, 210, 255)


def _ascii(s: str) -> str:
    for a, b in (("∧", "&"), ("¬", "~"), ("→", "->"), ("⊤", "true"), ("∨", "|")):
        s = s.replace(a, b)
    return s


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


class App:
    def __init__(self, track_files: List[str], neuro_weights: Optional[np.ndarray] = None,
                 headless: bool = False, scale: Optional[float] = None, mode: str = "race",
                 start_index: int = 0):
        if headless:
            V.headless_env()
        else:
            V.make_dpi_aware()
        import pygame
        self.pg = pygame
        pygame.init()
        self.headless = headless
        self.track_files = list(track_files)
        self.neuro = neuro_weights
        self.tracks = {}
        self.ti = start_index % len(self.track_files)
        self.track = self._track(self.ti)
        self.fixed_scale = scale
        if headless:
            s = scale or 1.0
            lay = V.Layout.fit(int(self.track.width_px * s) + 360, int(self.track.height_px * s),
                               self.track.width_px, self.track.height_px, scale=s)
            size = (int(self.track.width_px * s) + lay.panel_w, int(self.track.height_px * s))
            self.screen = V.open_window(pygame, "", size, headless=True)
        else:
            self.screen = V.open_window(pygame, "Car Racing Agent  -  A* vs Utility vs Reflex vs Learned")
        self.fullscreen = False
        self.mode = mode
        self.paused = False
        self.speed = 1
        self.show_all_rays = False
        self.show_lines = 1
        self.sel = 0
        self.request = None
        self.relayout()
        self.start()

    # ------------------------------------------------------------------ setup
    def _track(self, i) -> Track:
        f = self.track_files[i % len(self.track_files)]
        if f not in self.tracks:
            self.tracks[f] = load_track(f)
        return self.tracks[f]

    def relayout(self):
        w, h = self.screen.get_size()
        tr = self.track
        self.lay = V.Layout.fit(w, h, tr.width_px, tr.height_px, scale=self.fixed_scale if self.headless else None)
        self.F = V.fonts(self.pg, self.lay.ui)
        self.bg = V.render_track(self.pg, tr, self.lay)

    def start(self):
        if self.mode == "race":
            self.race = Race(self.track, default_lineup(self.neuro), max_time=120)
            self.sel = min(self.sel, len(self.race.cars) - 1)
        else:
            self.cfg = GAConfig(population=50)
            self.rng = np.random.default_rng(1)
            self.pop = random_population(self.cfg, self.rng)
            self.gen = 0
            self.best_hist: List[float] = []
            self.sim = PopulationSim(self.pop, self.track, max_time=40)

    def switch_track(self, i):
        self.ti = i % len(self.track_files)
        self.track = self._track(self.ti)
        self.relayout()
        self.start()

    # ---------------------------------------------------------------- drawing
    def _t(self, s, pos, f="m", color=TEXT):
        self.screen.blit(self.F[f].render(str(s), True, color), pos)

    def _rays(self, x, y, h, d, color=(255, 80, 80)):
        pg, L = self.pg, self.lay
        sx, sy = L.to_screen(x, y)
        w = max(1, int(L.scale))
        for (ex, ey) in C.ray_endpoints(x, y, h, d):
            px, py = L.to_screen(ex, ey)
            pg.draw.line(self.screen, color, (sx, sy), (px, py), w)
            pg.draw.circle(self.screen, color, (int(px), int(py)), max(3, int(3 * L.scale)))

    def _car(self, x, y, h, color, alpha=255, selected=False):
        sx, sy = self.lay.to_screen(x, y)
        spr = V.car_sprite(self.pg, color, self.lay.scale, alpha, selected, C.CAR_L, C.CAR_W)
        V.blit_car(self.pg, self.screen, spr, sx, sy, h)

    def _world(self):
        self.screen.fill(BG)
        self.screen.blit(self.bg, (self.lay.ox, self.lay.oy))

    def draw_race(self):
        pg, L = self.pg, self.lay
        self._world()
        cars = self.race.cars
        sel = cars[self.sel]
        lw = max(2, int(2 * L.scale))
        if self.show_lines:
            for c in cars:
                if isinstance(c.agent, PathAgent) and (c is sel or self.show_lines == 2):
                    pts = [L.to_screen(x, y) for x, y in c.agent.line.pts[::2]]
                    pg.draw.lines(self.screen, c.agent.color, True, pts, lw)
        if len(sel.trail) > 1:
            pg.draw.lines(self.screen, (255, 255, 255), False,
                          [L.to_screen(x, y) for x, y in sel.trail[-400:]], max(1, int(L.scale)))
        if self.show_all_rays:
            for c in cars:
                if c.alive:
                    self._rays(c.x, c.y, c.heading, c.rays, c.agent.color)
        if sel.status != CRASHED:
            self._rays(sel.x, sel.y, sel.heading, sel.rays)
        for i, c in enumerate(cars):
            self._car(c.x, c.y, c.heading, c.agent.color, 255 if c.status != CRASHED else 110, i == self.sel)
            if c.status == CRASHED:
                sx, sy = L.to_screen(c.x, c.y)
                k = 7 * L.scale
                pg.draw.line(self.screen, (255, 60, 60), (sx - k, sy - k), (sx + k, sy + k), lw + 1)
                pg.draw.line(self.screen, (255, 60, 60), (sx - k, sy + k), (sx + k, sy - k), lw + 1)
        for c in cars:
            sx, sy = L.to_screen(c.x, c.y)
            img = self.F["s"].render(c.name, True, c.agent.color)
            self.screen.blit(img, (sx - img.get_width() / 2, sy - 18 * L.scale - img.get_height()))
        self._panel_race()

    def _panel_race(self):
        pg, L = self.pg, self.lay
        u = L.ui
        x0 = L.panel_x + int(14 * u)
        pg.draw.rect(self.screen, BG, (L.panel_x, 0, L.panel_w, L.win_h))
        y = int(10 * u)
        lh, sh = int(18 * u), int(15 * u)
        self._t("CAR RACING AGENT", (x0, y), "l"); y += int(26 * u)
        self._t(f"[{self.ti + 1}/{len(self.track_files)}] {self.track.name}", (x0, y), "m", DIM); y += lh
        self._t(f"race time {self.race.t:6.2f}s   x{self.speed}", (x0, y), "m", DIM); y += int(24 * u)
        self._t("POS  CAR             STATUS        SPEED", (x0, y), "s", DIM); y += sh + 2
        for pos, c in enumerate(self.race.standings(), 1):
            i = self.race.cars.index(c)
            pg.draw.rect(self.screen, c.agent.color, (x0, y + 3 * u, 10 * u, 10 * u))
            if c.status == FINISHED:
                st = f"{c.finish_time:6.2f}s"
            elif c.status == "RACING":
                st = f"{100 * max(0, c.progress) / self.race.goal:5.1f}%"
            else:
                st = c.status
            self._t(f"{pos:>2}  {c.name:<15} {st:<12} {c.speed:5.0f}", (x0 + int(14 * u), y), "s",
                    GOLD if i == self.sel else TEXT)
            y += int(17 * u)
        y += int(10 * u)
        c = self.race.cars[self.sel]
        ag: Agent = c.agent
        pg.draw.line(self.screen, (60, 64, 74), (x0, y), (L.win_w - 14 * u, y)); y += int(8 * u)
        self._t(f"SELECTED: {c.name}  [{ag.kind}]", (x0, y), "b", ag.color); y += int(22 * u)
        self._t(f"speed {c.speed:5.1f} px/s   heading {math.degrees(c.heading) % 360:5.1f} deg", (x0, y), "s"); y += sh
        self._t(f"pos ({c.x:6.1f},{c.y:6.1f})  v=({c.speed * math.cos(c.heading):5.0f},"
                f"{c.speed * math.sin(c.heading):5.0f})", (x0, y), "s"); y += sh
        bw = L.panel_w - int(40 * u)
        pg.draw.rect(self.screen, (50, 52, 60), (x0, y + 2, bw, 8 * u), border_radius=3)
        pg.draw.rect(self.screen, ag.color, (x0, y + 2, int(bw * c.speed / C.V_MAX), 8 * u), border_radius=3)
        y += int(16 * u)
        info = ag.info
        self._t(f"action : {info.get('action', '-')}", (x0, y), "m", GOLD); y += lh
        self._t(f"rule   : {info.get('rule', '-')}", (x0, y), "m"); y += int(17 * u)
        if "formula" in info:
            for line in _wrap(_ascii(info["formula"]), 46):
                self._t(line, (x0, y), "s", DIM); y += int(14 * u)
        if info.get("facts"):
            for line in _wrap("facts: " + ", ".join(info["facts"]), 46)[:3]:
                self._t(line, (x0, y), "s", INFO); y += int(14 * u)
        if "utilities" in info:
            for k, v in sorted(info["utilities"].items(), key=lambda kv: -kv[1])[:4]:
                self._t(f"U({k}) = {v:7.1f}", (x0, y), "s", INFO); y += int(14 * u)
        if "net_out" in info:
            self._t(f"net out steer={info['net_out'][0]:+.2f} thr={info['net_out'][1]:+.2f}", (x0, y), "s", INFO)
            y += int(14 * u)
        if "v_limit" in info:
            self._t(f"planned speed limit {info['v_limit']} px/s", (x0, y), "s", INFO); y += int(14 * u)
        y += int(4 * u)
        wts = UtilityAgent.SENSOR_WEIGHTS
        for k, (nm, d) in enumerate(zip(C.RAY_NAMES, c.rays)):
            self._t(f"{nm:<15}{d:6.0f}px  w={wts[k]:.1f}", (x0, y), "s"); y += int(14 * u)
        y = L.win_h - int(64 * u)
        for ln in ("SPACE pause  TAB/click select  1-9 [ ] track", "T train  V rays  L lines  +/- speed  R restart",
                   "E track editor  S screenshot  F11 full  Q quit"):
            self._t(ln, (x0, y), "s", DIM); y += int(15 * u)

    def draw_train(self):
        pg, L = self.pg, self.lay
        u = L.ui
        self._world()
        sim = self.sim
        best = int(np.argmax(sim.prog))
        for i in range(sim.P):
            if not sim.alive[i]:
                self._car(sim.x[i], sim.y[i], sim.h[i], (120, 60, 60), 90)
        for i in range(sim.P):
            if sim.alive[i]:
                self._car(sim.x[i], sim.y[i], sim.h[i], (46, 204, 113), 235)
        if sim.alive[best]:
            d = C.raycast(self.track, sim.x[best], sim.y[best], sim.h[best])[0]
            self._rays(sim.x[best], sim.y[best], sim.h[best], d)
        x0 = L.panel_x + int(14 * u)
        pg.draw.rect(self.screen, BG, (L.panel_x, 0, L.panel_w, L.win_h))
        y = int(10 * u)
        lh = int(18 * u)
        self._t("LEARNING (genetic algorithm)", (x0, y), "b"); y += int(24 * u)
        self._t(f"[{self.ti + 1}/{len(self.track_files)}] {self.track.name}", (x0, y), "m", DIM); y += int(20 * u)
        self._t(f"generation  {self.gen}", (x0, y), "l", GOLD); y += int(28 * u)
        self._t(f"cars alive  {sim.alive.sum()}/{sim.P}", (x0, y)); y += lh
        self._t(f"time        {sim.t:5.1f}s   x{self.speed}", (x0, y)); y += lh
        self._t(f"best lap %  {100 * min(1.0, sim.prog.max() / self.track.length):5.1f}", (x0, y)); y += lh
        self._t(f"finished    {sim.finished.sum()}", (x0, y)); y += int(26 * u)
        self._t("best fitness per generation", (x0, y), "s", DIM); y += int(16 * u)
        hist = self.best_hist[-80:]
        gw, gh = L.panel_w - int(40 * u), int(100 * u)
        pg.draw.rect(self.screen, (40, 42, 50), (x0, y, gw, gh))
        if hist:
            hmax = max(1.0, max(hist)) * 1.1
            pts = [(x0 + k * gw / max(1, len(hist) - 1), y + gh - v / hmax * gh) for k, v in enumerate(hist)]
            if len(pts) > 1:
                pg.draw.lines(self.screen, (46, 204, 113), False, pts, max(2, int(2 * u)))
            self._t(f"{hmax:.1f}", (x0 + 3, y + 2), "s", DIM)
            self._t(f"last {hist[-1]:.2f}", (x0 + gw - int(84 * u), y + gh - int(16 * u)), "s", (46, 204, 113))
        self._t("0", (x0 + 3, y + gh - int(16 * u)), "s", DIM)
        y += gh + int(6 * u)
        self._t("fitness = lap % (+ bonus if finished fast)", (x0, y), "s", DIM); y += int(22 * u)
        for ln in ("Every car = one neural network", "(5 rays + speed -> steer, throttle).",
                   "Best cars survive, are crossed over", "and mutated: the next generation",
                   "drives a little better."):
            self._t(ln, (x0, y), "s", DIM); y += int(14 * u)
        y = L.win_h - int(36 * u)
        self._t("T race  1-9 [ ] track  +/- speed  E editor  Q quit", (x0, y), "s", DIM)

    # ---------------------------------------------------------------- update
    def tick(self):
        if self.paused:
            return
        for _ in range(self.speed):
            if self.mode == "race":
                self.race.step()
            else:
                self.sim.step()
                if self.sim.done:
                    res = self.sim.result()
                    self.best_hist.append(float(res.fitness.max()))
                    self.pop = next_generation(self.pop, res.fitness, self.rng, self.cfg)
                    self.gen += 1
                    self.sim = PopulationSim(self.pop, self.track, max_time=40)

    def draw(self):
        self.draw_race() if self.mode == "race" else self.draw_train()
        self.pg.display.flip()

    def toggle_fullscreen(self):
        pg = self.pg
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            info = pg.display.Info()
            self.screen = pg.display.set_mode((info.current_w, info.current_h), pg.FULLSCREEN)
        else:
            self.screen = V.open_window(pg, "Car Racing Agent  -  A* vs Utility vs Reflex vs Learned")
        self.relayout()

    def handle(self, ev) -> bool:
        pg = self.pg
        if ev.type == pg.QUIT:
            return False
        if ev.type == pg.VIDEORESIZE and not self.fullscreen:
            self.screen = pg.display.set_mode((max(800, ev.w), max(500, ev.h)), pg.RESIZABLE)
            self.relayout()
        if ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1 and self.mode == "race":
            mx, my = self.lay.to_world(*ev.pos)
            d = [(c.x - mx) ** 2 + (c.y - my) ** 2 for c in self.race.cars]
            if min(d) < 30 ** 2:
                self.sel = int(np.argmin(d))
        if ev.type == pg.KEYDOWN:
            k = ev.key
            if k in (pg.K_ESCAPE, pg.K_q):
                if self.fullscreen and k == pg.K_ESCAPE:
                    self.toggle_fullscreen()
                    return True
                return False
            if k == pg.K_F11:
                self.toggle_fullscreen()
            elif k == pg.K_SPACE:
                self.paused = not self.paused
            elif k == pg.K_TAB and self.mode == "race":
                self.sel = (self.sel + 1) % len(self.race.cars)
            elif pg.K_1 <= k <= pg.K_9 and k - pg.K_1 < len(self.track_files):
                self.switch_track(k - pg.K_1)
            elif k in (pg.K_RIGHTBRACKET, pg.K_PAGEDOWN):
                self.switch_track(self.ti + 1)
            elif k in (pg.K_LEFTBRACKET, pg.K_PAGEUP):
                self.switch_track(self.ti - 1)
            elif k == pg.K_t:
                self.mode = "train" if self.mode == "race" else "race"
                self.start()
            elif k == pg.K_v:
                self.show_all_rays = not self.show_all_rays
            elif k == pg.K_l:
                self.show_lines = (self.show_lines + 1) % 3
            elif k in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                self.speed = min(self.speed * 2, 32)
            elif k in (pg.K_MINUS, pg.K_KP_MINUS):
                self.speed = max(self.speed // 2, 1)
            elif k == pg.K_r:
                self.start()
            elif k == pg.K_e:
                self.request = ("editor", self.track_files[self.ti])
                return False
            elif k == pg.K_s:
                Path("demo").mkdir(exist_ok=True)
                out = f"demo/screenshot_{self.mode}_{self.track.file}.png"
                pg.image.save(self.screen, out)
                print("saved", out)
        return True

    def loop(self):
        """Run until the window closes. Returns a request such as ("editor", track_file) or None."""
        clock = self.pg.time.Clock()
        running = True
        while running:
            for ev in self.pg.event.get():
                running = self.handle(ev) and running
            self.tick()
            self.draw()
            clock.tick(30)
        return self.request

    def record(self, out: str, every: int = 2, max_seconds: float = 60, hold: int = 30, fps: int = 15):
        """Render a race (or some training generations) to a GIF/MP4."""
        import imageio.v2 as imageio
        frames = []
        ticks = 0
        limit = int(max_seconds / C.DT)
        while ticks < limit:
            if self.mode == "race" and self.race.done:
                break
            self.tick()
            ticks += self.speed
            if ticks % every == 0 or self.speed > 1:
                self.draw()
                frames.append(np.ascontiguousarray(self.pg.surfarray.array3d(self.screen).swapaxes(0, 1)))
        self.draw()
        last = np.ascontiguousarray(self.pg.surfarray.array3d(self.screen).swapaxes(0, 1))
        frames += [last] * hold
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if out.endswith(".gif"):
            imageio.mimsave(out, frames, duration=1 / fps, loop=0)
        else:
            imageio.mimsave(out, frames, fps=fps, macro_block_size=1)
        self.pg.image.save(self.screen, str(Path(out).with_suffix(".png")))
        return out
