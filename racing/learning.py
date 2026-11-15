"""Learning agent training: neuro-evolution with a genetic algorithm.

"Learning: repeatedly running and changing things" (proposal). A population
of cars, each driven by its own neural network, races on the training tracks.
The best drivers survive and are recombined and mutated, and this repeats for
many generations. No driving rules are coded: the network learns how to map the
5 ray sensors + speed to steering and throttle.

Fitness of a driver on one track::

    fitness = fraction_of_lap_driven  (+ 1 + 3 * time_bonus  if it finished)

with time_bonus = (max_time - lap_time) / max_time. Its total fitness is the mean
over the training tracks, each driven once from the start line and once from a
random starting-grid slot. The whole population
is simulated in parallel with numpy (one raycast / physics call per tick for all
cars), so a generation takes about a second.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from . import car as C
from .agents import n_params, nn_forward, nn_inputs, nn_to_actions
from .track import Track

STUCK_S = 3.0
TIME_BONUS = 3.0


@dataclass
class EpisodeResult:
    fitness: np.ndarray        # (P,)
    finished: np.ndarray       # (P,) bool
    lap_time: np.ndarray       # (P,) s (inf if not finished)
    frac: np.ndarray           # (P,) lap fraction driven
    trails: Optional[list] = None


class PopulationSim:
    """Step-by-step time trial of a whole population (used by evaluate() and the GUI)."""

    def __init__(self, W: np.ndarray, track: Track, max_time: float = 45.0, record: bool = False,
                 start=None):
        """``start``: None (centre of the start line) or a grid slot (x, y, heading, s0)."""
        self.W, self.track, self.max_time = W, track, max_time
        P = self.P = W.shape[0]
        if start is None:
            x0, y0 = track.start_pos
            h0, s0 = track.start_heading, 0.0
        else:
            x0, y0, h0, s0 = start
        self.s0 = s0
        self.x = np.full(P, x0); self.y = np.full(P, y0)
        self.h = np.full(P, h0); self.v = np.zeros(P)
        self.alive = np.ones(P, bool)
        self.finished = np.zeros(P, bool)
        self.lap_time = np.full(P, np.inf)
        self.prog = np.full(P, s0)
        self.last = track.arc_at(self.x, self.y).astype(float)
        self.best = np.full(P, s0); self.best_t = np.zeros(P)
        self.t = 0.0
        self.trails = [[(x0, y0)] for _ in range(P)] if record else None

    @property
    def done(self):
        return not self.alive.any() or self.t >= self.max_time

    def step(self):
        if self.done:
            return
        tr, L = self.track, self.track.length
        idx = np.nonzero(self.alive)[0]
        rays = C.raycast(tr, self.x[idx], self.y[idx], self.h[idx])
        out = nn_forward(self.W[idx], nn_inputs(rays, self.v[idx]))
        st, th = nn_to_actions(out)
        xn, yn, hn, vn = C.step_physics(self.x[idx], self.y[idx], self.h[idx], self.v[idx], st, th)
        crash = C.collides(tr, xn, yn, hn)
        self.x[idx], self.y[idx], self.h[idx], self.v[idx] = xn, yn, hn, vn
        a = tr.arc_at(xn, yn).astype(float)
        d = a - self.last[idx]
        d = np.where(d < -L / 2, d + L, np.where(d > L / 2, d - L, d))
        self.prog[idx] += d
        self.last[idx] = a
        self.t += C.DT
        better = self.prog[idx] > self.best[idx] + 1
        self.best[idx[better]] = self.prog[idx[better]]
        self.best_t[idx[better]] = self.t
        fin = self.prog[idx] >= L
        self.finished[idx[fin]] = True
        self.lap_time[idx[fin]] = self.t
        dead = crash | fin | (self.t - self.best_t[idx] > STUCK_S) | (self.prog[idx] < self.s0 - 30)
        self.alive[idx[dead]] = False
        if self.trails is not None:
            for k, i in enumerate(idx):
                self.trails[i].append((float(xn[k]), float(yn[k])))

    def result(self) -> "EpisodeResult":
        L = self.track.length
        frac = np.clip((self.prog - self.s0) / (L - self.s0), 0, 1)
        bonus = np.where(self.finished,
                         1 + TIME_BONUS * np.clip((self.max_time - self.lap_time) / self.max_time, 0, 1), 0)
        return EpisodeResult(frac + bonus, self.finished.copy(), self.lap_time.copy(), frac, self.trails)


def evaluate(W: np.ndarray, track: Track, max_time: float = 45.0, record: bool = False,
             start=None) -> EpisodeResult:
    """Run every network in ``W`` (P, n_params) for one lap on ``track`` (time trial)."""
    sim = PopulationSim(W, track, max_time, record, start)
    while not sim.done:
        sim.step()
    return sim.result()


@dataclass
class GAConfig:
    population: int = 60
    generations: int = 40
    elite: int = 6
    tournament: int = 3
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.35
    immigrants: int = 4
    init_sigma: float = 0.9
    max_time: float = 45.0
    seed: int = 7


@dataclass
class GAHistory:
    best: List[float] = field(default_factory=list)
    mean: List[float] = field(default_factory=list)
    finish_rate: List[float] = field(default_factory=list)
    best_lap: List[float] = field(default_factory=list)
    seconds: List[float] = field(default_factory=list)


def evolve(tracks: List[Track], cfg: GAConfig = GAConfig(), verbose: bool = True,
           on_generation: Optional[Callable] = None):
    """Returns (best_weights, history)."""
    rng = np.random.default_rng(cfg.seed)
    pop = random_population(cfg, rng)
    hist = GAHistory()
    best_w, best_f = None, -np.inf
    for gen in range(cfg.generations):
        t0 = time.perf_counter()
        # every driver races each training track twice: from the centre of the start line
        # and from a random starting-grid slot (robust starts, as in the real race)
        res = []
        for tr in tracks:
            res.append(evaluate(pop, tr, cfg.max_time))
            slot = tr.grid_slots(8)[int(rng.integers(0, 8))]
            res.append(evaluate(pop, tr, cfg.max_time, start=slot))
        fit = np.mean([r.fitness for r in res], axis=0)
        fin = np.mean([r.finished for r in res], axis=0)
        laps = np.mean([np.where(r.finished, r.lap_time, np.nan) for r in res], axis=0)
        order = np.argsort(-fit)
        if fit[order[0]] > best_f:
            best_f, best_w = float(fit[order[0]]), pop[order[0]].copy()
        hist.best.append(float(fit[order[0]]))
        hist.mean.append(float(fit.mean()))
        hist.finish_rate.append(float((fin == 1).mean()))
        fl = laps[np.isfinite(laps)]
        hist.best_lap.append(float(fl.min()) if len(fl) else float("nan"))
        hist.seconds.append(time.perf_counter() - t0)
        if verbose:
            print(f"gen {gen:3d}  best {fit[order[0]]:.3f}  mean {fit.mean():.3f}  "
                  f"finish-all {100 * hist.finish_rate[-1]:4.0f}%  best lap {hist.best_lap[-1]:.2f}s  "
                  f"({hist.seconds[-1]:.1f}s)")
        if on_generation:
            on_generation(gen, pop, fit)
        pop = next_generation(pop, fit, rng, cfg)
    # final pick: the all-time best and the final generation's elite, judged on every grid slot
    cands = np.vstack([best_w[None, :], pop[:cfg.elite]])
    score = np.zeros(len(cands))
    for tr in tracks:
        score += evaluate(cands, tr, cfg.max_time).fitness
        for slot in tr.grid_slots(8):
            score += evaluate(cands, tr, cfg.max_time, start=slot).fitness
    best_w = cands[int(np.argmax(score))].copy()
    if verbose:
        print(f"final selection score {score.max() / (9 * len(tracks)):.3f} (mean over all starts)")
    return best_w, hist


def random_population(cfg: GAConfig, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(cfg.seed)
    return rng.normal(0, cfg.init_sigma, (cfg.population, n_params()))


def next_generation(pop: np.ndarray, fit: np.ndarray, rng, cfg: GAConfig) -> np.ndarray:
    """Elitism + tournament selection + uniform crossover + Gaussian mutation + immigrants."""
    D = pop.shape[1]
    order = np.argsort(-fit)
    new = [pop[i].copy() for i in order[:cfg.elite]]
    while len(new) < cfg.population - cfg.immigrants:
        a = pop[_tournament(rng, fit, cfg.tournament)]
        b = pop[_tournament(rng, fit, cfg.tournament)]
        child = np.where(rng.random(D) < 0.5, a, b)
        m = rng.random(D) < cfg.mutation_rate
        new.append(child + m * rng.normal(0, cfg.mutation_sigma, D))
    for _ in range(cfg.immigrants):
        new.append(rng.normal(0, cfg.init_sigma, D))
    return np.array(new)


def _tournament(rng, fit, k):
    idx = rng.integers(0, len(fit), k)
    return idx[np.argmax(fit[idx])]


def save_model(path, weights, history: GAHistory, cfg: GAConfig, train_tracks: List[str]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "weights": [float(w) for w in weights], "config": cfg.__dict__,
        "train_tracks": train_tracks, "history": history.__dict__,
    }, indent=1))


def load_model(path):
    d = json.loads(Path(path).read_text())
    return np.array(d["weights"]), d
