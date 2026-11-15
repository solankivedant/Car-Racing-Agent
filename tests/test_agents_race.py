from pathlib import Path

import numpy as np
import pytest

from racing import car as C
from racing.agents import (REFLEX_RULES, Agent, NeuroAgent, PathAgent, ReflexAgent, UtilityAgent,
                           default_lineup, n_params, percept_atoms)
from racing.car import CarState
from racing.learning import GAConfig, evaluate, evolve, load_model, next_generation, random_population
from racing.logic import RuleBase, lit, rule
from racing.race import CRASHED, FINISHED, STUCK, Race, run_race

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "ga_best.json"


def S(rays, speed=100.0):
    return CarState(0, 0, 0, speed, np.array(rays, float), 0, 1000, 0)


# ------------------------------------------------------------- propositional logic
def test_literals_and_rules():
    assert lit("~A").positive is False and lit("¬A").atom == "A"
    r = rule("X", "go", "A & ~B", (0, 1))
    assert r.fires({"A"}) and not r.fires({"A", "B"}) and not r.fires(set())
    assert r.formula == "A ∧ ¬B → go"


def test_rulebase_priority_and_default():
    rb = RuleBase([rule("1", "a", "A", (1, 1)), rule("2", "b", "A & B", (2, 2)), rule("3", "c", "", (0, 0))])
    assert rb.decide({"A", "B"}, {})[1].rid == "1"          # first match wins
    assert rb.decide(set(), {})[1].rid == "3"
    with pytest.raises(RuntimeError):
        RuleBase([rule("1", "a", "A", (1, 1))]).decide(set(), {})


# ------------------------------------------------------------- reflex rules
def test_percepts():
    f = percept_atoms(S([100, 100, 20, 100, 100], speed=150))
    assert "FrontBlocked" in f and "TooFast" in f
    f = percept_atoms(S([10, 200, 220, 200, 200], speed=100))
    assert "LeftTight" in f and "FrontBlocked" not in f


@pytest.mark.parametrize("rays, speed, rid, steer, thr", [
    ([150, 120, 20, 40, 30], 150, "R1", C.LEFT, C.BRAKE_A),     # wall ahead, left open, fast
    ([30, 40, 20, 120, 150], 150, "R1", C.RIGHT, C.BRAKE_A),    # wall ahead, right open
    ([30, 40, 15, 120, 150], 10, "R2", C.RIGHT, C.ACCELERATE),  # blocked & slow: crawl out
    ([60, 20, 220, 90, 60], 80, "R3", C.RIGHT, C.COAST),        # front-left diagonal close
    ([60, 90, 220, 20, 60], 80, "R4", C.LEFT, C.COAST),
    ([10, 90, 220, 90, 60], 80, "R5", C.RIGHT, C.ACCELERATE),   # left side wall
    ([60, 90, 220, 90, 10], 80, "R6", C.LEFT, C.ACCELERATE),
    ([60, 150, 110, 60, 40], 240, "R7", C.LEFT, C.BRAKE_A),     # bend ahead, too fast
    ([60, 150, 90, 60, 40], 60, "R8", C.LEFT, C.COAST),
    ([60, 160, 220, 100, 60], 100, "R9", C.LEFT, C.ACCELERATE),
    ([60, 100, 220, 160, 60], 100, "R10", C.RIGHT, C.ACCELERATE),
    ([60, 120, 220, 120, 60], 100, "R11", C.STRAIGHT, C.ACCELERATE),
])
def test_each_reflex_rule(rays, speed, rid, steer, thr):
    ag = ReflexAgent()
    a = ag.act(S(rays, speed))
    assert ag.info["rule"] == rid, ag.info
    assert a == (steer, thr)


def test_rule_table_complete():
    assert [r for r, _ in REFLEX_RULES.table()][-1] == "R11"
    assert REFLEX_RULES.rules[-1].antecedent == ()


# ------------------------------------------------------------- agents finish laps
@pytest.mark.parametrize("make", [lambda: ReflexAgent(), lambda: UtilityAgent(),
                                  lambda: PathAgent("h1"), lambda: PathAgent("h2"),
                                  lambda: PathAgent("h3"), lambda: PathAgent("h4")],
                         ids=["reflex", "utility", "A*h1", "A*h2", "A*h3", "A*h4"])
def test_agent_finishes_every_track(tracks, make):
    for t in tracks.values():
        r = run_race(t, [make()])
        c = r.cars[0]
        assert c.status == FINISHED, (t.name, c.status, c.progress)
        assert c.progress >= t.length


def test_full_lineup_race(serp):
    r = run_race(serp, default_lineup())
    res = r.results()
    assert [x["pos"] for x in res] == list(range(1, len(res) + 1))
    finished = [x for x in res if x["status"] == FINISHED]
    assert len(finished) == 6
    assert finished == sorted(finished, key=lambda x: x["lap_time"])


def _drive(agent, track, x, y, h, v, seconds=3.0):
    agent.reset(track)
    for _ in range(int(seconds / C.DT)):
        s = CarState(x, y, h, v, C.raycast(track, x, y, h)[0], 0, track.length, 0)
        st, th = agent.act(s)
        x, y, h, v = (float(q[0]) for q in C.step_physics(x, y, h, np.array([v]), st, th))
        if C.collides(track, x, y, h)[0]:
            return False
    return True


def test_utility_recovers_from_bad_heading(oval):
    # car heading 35 degrees towards the inner wall at speed: straight on crashes,
    # the utility agent's look-ahead must steer/brake away from it
    x, y = oval.start_pos
    h = oval.start_heading + 0.6
    assert not _drive(Straight("s"), oval, x, y, h, 150.0)
    assert _drive(UtilityAgent(), oval, x, y, h, 150.0)
    U = UtilityAgent()
    U.reset(oval)
    assert U.utilities(CarState(x, y, h, 150.0, C.raycast(oval, x, y, h)[0], 0, oval.length, 0)).max() > -1e3


class Straight(Agent):
    def act(self, s):
        return (C.STRAIGHT, C.ACCELERATE)


class Parked(Agent):
    def act(self, s):
        return (C.STRAIGHT, C.BRAKE_A)


def test_crash_and_stuck_detection(serp):
    r = run_race(serp, [Straight("straight"), Parked("parked")], max_time=20)
    st = {c.name: c.status for c in r.cars}
    assert st["straight"] == CRASHED and st["parked"] == STUCK


def test_race_timeout(oval):
    r = Race(oval, [ReflexAgent()], max_time=2.0)
    r.run()
    assert r.cars[0].status == "TIMEOUT"


# ------------------------------------------------------------- learning
def test_network_shapes():
    W = np.zeros((3, n_params()))
    ag = NeuroAgent(W[0])
    a = ag.act(S([50, 50, 50, 50, 50]))
    assert a in C.ACTIONS


def test_ga_operators_keep_elite():
    cfg = GAConfig(population=20, elite=3)
    rng = np.random.default_rng(0)
    pop = random_population(cfg, rng)
    fit = np.arange(20.0)
    new = next_generation(pop, fit, rng, cfg)
    assert new.shape == pop.shape
    for i in (19, 18, 17):
        assert any(np.array_equal(pop[i], n) for n in new[:3])


def test_ga_learns_on_oval(oval):
    cfg = GAConfig(population=24, generations=6, seed=3, max_time=20)
    w, hist = evolve([oval], cfg, verbose=False)
    assert hist.best[-1] >= hist.best[0]
    assert max(hist.best) > 0.3                     # drives a decent part of the lap


@pytest.mark.skipif(not MODEL.exists(), reason="train first: python main.py train")
def test_trained_model_finishes_all_tracks(tracks):
    w, meta = load_model(MODEL)
    assert len(w) == n_params()
    for t in tracks.values():
        r = evaluate(w[None, :], t)
        assert r.finished[0], t.name
