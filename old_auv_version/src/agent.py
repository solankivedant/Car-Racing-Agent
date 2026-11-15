"""Car racing agent: reflex layer (rules) on top of a deliberative A* layer.

Control loop, one tick at a time::

    sense()  -> Percept  (tile, current, fuel, rays, cost-to-pit, plan facts)
    decide() -> Rule     (first matching rule from reflex.RULES)
    act()               (move / refuel / go to pit / mark / re-plan)

Deliberation (A* + Held-Karp) runs only when needed: at the start, after
every refuel, and when the wall guard fires.

Loop-freedom. A re-plan at a pit checks the first leg of the new tour by
simulating the reflex rules along it (``leg_ok``). If a rule would interrupt
the leg before it reaches its target, the car does not try that leg. It plans
a *relay* instead: a chain of pit hops, each at most one full tank, ending at
a *staging pit* from which the leg is known to succeed. So every pit-to-pit
cycle reaches a new checkpoint (or H), and a checkpoint that no staging pit
can serve is reported as infeasible and skipped, so the car never loops forever.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import reflex as rx
from .heuristics import ADMISSIBLE
from .safety import SafetyMap
from .search import astar
from .sensors import read_rays
from .tour import RACE, SURVEY, LegCache, plan_tour
from .world import (INF, LEFT_OF, OPPOSITE, RIGHT_OF, GRASS, Grid, Pos,
                    direction_between)

TOUR, TO_PIT = "TOUR", "TO_PIT"
DONE, FAILED, RUNNING, STEP_LIMIT = "DONE", "FAILED", "RUNNING", "STEP_LIMIT"


@dataclass
class LogEntry:
    t: int
    pos: Pos
    fuel: float
    rule: str
    action: str
    car_action: str = ""
    cost: float = 0
    heading: str = "N"
    mode: str = TOUR
    target: Optional[str] = None
    note: str = ""
    rays: Dict[str, int] = field(default_factory=dict)
    plan: tuple = field(default=(), repr=False)      # remaining planned tiles (for the GUI)


@dataclass
class RunResult:
    status: str
    map_name: str
    heuristic: str
    mode: str
    moves: int
    fuel_used: float
    pit_stops: int
    replans: int
    visit_order: List[int]
    skipped: List[int]
    planned_tour_cost: float
    searches: int
    nodes_expanded: int
    search_ms: float
    min_fuel: float
    safety_violations: int
    log: List[LogEntry] = field(repr=False, default_factory=list)
    path: List[Pos] = field(repr=False, default_factory=list)
    rule_counts: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        rc = ", ".join(f"{k}:{v}" for k, v in sorted(self.rule_counts.items()))
        return (f"[{self.map_name} | {self.heuristic} | {self.mode}] status={self.status} "
                f"moves={self.moves} fuel_used={self.fuel_used:g} pit_stops={self.pit_stops} "
                f"replans={self.replans} order={self.visit_order} skipped={self.skipped} "
                f"planned_tour_cost={self.planned_tour_cost:g} searches={self.searches} "
                f"nodes_expanded={self.nodes_expanded} min_fuel={self.min_fuel:g} "
                f"violations={self.safety_violations}\n  rules fired: {rc}")


class RacingAgent:
    def __init__(self, grid: Grid, heuristic: str = "h3", mode: str = SURVEY,
                 max_ticks: int = 20000, verbose: bool = False):
        if mode not in (SURVEY, RACE):
            raise ValueError(f"mode must be {SURVEY!r} or {RACE!r}")
        self.grid = grid
        self.heuristic = heuristic
        # exact searches (pit routing, relay legs) need an admissible heuristic
        self.exact_h = heuristic if heuristic in ADMISSIBLE else "h0"
        self.mode = mode
        self.max_ticks = max_ticks
        self.verbose = verbose

        self.safety = SafetyMap(grid)
        self.legs = LegCache(grid, heuristic)

        # state
        self.pos: Pos = grid.home
        self.fuel: float = grid.capacity
        self.heading: str = "N"
        self.visited: List[int] = []
        self.infeasible: List[int] = []
        self.plan: List[Pos] = []          # tiles still to drive (excluding pos)
        self.state = TOUR
        self.pit_target: Optional[Pos] = None
        self.relay: List[Pos] = []
        self.forced_target: Optional[Pos] = None
        self.target_label: Optional[str] = None

        # bookkeeping
        self.t = 0
        self.log: List[LogEntry] = []
        self.path: List[Pos] = [self.pos]
        self.fuel_used = 0.0
        self.pit_stops = 0
        self.replans = 0
        self.min_fuel = self.fuel
        self.violations = 0
        self.planned_tour_cost = INF
        self.rule_counts: Dict[str, int] = {}
        self.status = RUNNING
        self.extra_searches = 0
        self.extra_nodes = 0
        self.extra_ms = 0.0

        self._replan(initial=True)

    # ================================================================ helpers
    @property
    def remaining(self) -> List[int]:
        return [k for k in self.grid.sites if k not in self.visited and k not in self.infeasible]

    def _search(self, a: Pos, b: Pos, exact: bool = True):
        res = astar(self.grid, a, b, self.exact_h if exact else self.heuristic)
        self.extra_searches += 1
        self.extra_nodes += res.nodes_expanded
        self.extra_ms += res.runtime_ms
        return res

    def _note(self, msg: str):
        if self.verbose:
            print(f"t={self.t:4d} {msg}")

    def _percept(self, pos: Pos, fuel: float, nxt: Optional[Pos], heading_to_pit: bool,
                 pit_target: Optional[Pos], visited, with_rays: bool = True) -> rx.Percept:
        g = self.grid
        k = g.site_at(pos)
        if self.mode == RACE:
            # race rules: a checkpoint only counts when it is the next one in order
            pending = [s for s in g.sites if s not in visited and s not in self.infeasible]
            if k is not None and pending and k != pending[0]:
                k = None
        return rx.Percept(
            pos=pos, tile=g.terrain[pos[0]][pos[1]], char=g.char_at(pos),
            current=g.current[pos[0]][pos[1]], fuel=fuel, capacity=g.capacity,
            cost_to_pit=self.safety.ctp(pos), at_pit=g.is_pit(pos), checkpoint_here=k,
            rays=read_rays(g, pos, self.heading) if with_rays else {},
            heading=self.heading, heading_to_pit=heading_to_pit, pit_target=pit_target,
            visited=frozenset(visited),
            all_done=not [s for s in g.sites if s not in visited and s not in self.infeasible],
            at_home=pos == g.home, next_tile=nxt,
            next_is_wall=nxt is not None and not g.passable(nxt),
            safe_after_next=nxt is None or not g.passable(nxt)
            or self.safety.is_safe_move(fuel, pos, nxt),
        )

    def sense(self) -> rx.Percept:
        nxt = self.plan[0] if self.plan else None
        return self._percept(self.pos, self.fuel, nxt, self.state == TO_PIT,
                             self.pit_target, self.visited)

    # ------------------------------------------------------- leg simulation
    def leg_ok(self, path: List[Pos], fuel: float) -> bool:
        """Would the reflex rules let the car drive ``path`` uninterrupted?"""
        if not path:
            return False
        visited = set(self.visited)
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            p = self._percept(a, fuel, b, False, None, visited, with_rays=False)
            rule = rx.decide(p)
            while rule.action == rx.MARK:            # marking is free, decide again
                visited.add(p.checkpoint_here)
                p = self._percept(a, fuel, b, False, None, visited, with_rays=False)
                rule = rx.decide(p)
            if rule.action == rx.FINISH:
                return False
            if rule.action != rx.MOVE:
                return False
            fuel -= self.grid.step_cost(a, b)
            if fuel < 0:
                return False
        return True

    # --------------------------------------------------------------- planning
    def _replan(self, initial: bool = False):
        self.replans += 0 if initial else 1
        self.state, self.pit_target = TOUR, None
        while True:
            rem = self.remaining
            tour = plan_tour(self.grid, self.pos, rem, self.mode, self.heuristic, self.legs)
            if initial and self.planned_tour_cost == INF:
                self.planned_tour_cost = tour.cost
            for k in tour.skipped:
                self._skip(k, "unreachable")
            if not tour.legs or not tour.legs[0]:
                if self.pos == self.grid.home and not self.remaining:
                    self.plan = []
                    return
                self._fail("no route home")
                return
            target = self.grid.sites[tour.order[0]] if tour.order else self.grid.home
            self.target_label = str(tour.order[0]) if tour.order else "H"
            route = tour.route
            if not self.grid.is_pit(self.pos) or self.leg_ok(tour.legs[0], self.fuel):
                self.plan = route[1:]
                return
            # ---- the first leg would be interrupted: plan a relay through pits
            chain = self.safety.pit_chain(self.pos, lambda P: self._staging_ok(P, target))
            if chain is None:
                if tour.order:
                    self._skip(tour.order[0], "needs more fuel than any pit chain provides")
                    continue
                self._fail("home unreachable under the fuel rules")
                return
            self.forced_target = target
            self.relay = chain[1:]
            if self.relay:
                self._note(f"relay via pits {self.relay} to reach {self.target_label}")
                self._go_to_pit(self.relay.pop(0))
            else:
                self._drive_forced()
            return

    def _staging_ok(self, P: Pos, target: Pos) -> bool:
        res = self.legs.get(P, target) if self.heuristic in ADMISSIBLE else self._search(P, target)
        return res.found and self.leg_ok(res.path, self.grid.capacity)

    def _drive_forced(self):
        target, self.forced_target = self.forced_target, None
        res = self.legs.get(self.pos, target) if self.heuristic in ADMISSIBLE \
            else self._search(self.pos, target)
        self.state, self.pit_target = TOUR, None
        self.plan = res.path[1:] if res.found else []

    def _go_to_pit(self, pit: Optional[Pos] = None):
        if pit is None:
            pit, path, _ = self.safety.cheapest_pit(self.pos, self.exact_h)
            self.extra_searches += len(self.grid.pits)
        else:
            res = self._search(self.pos, pit)
            path = res.path
        if pit is None or path is None:
            self._fail("no pit reachable")
            return
        self.state, self.pit_target = TO_PIT, pit
        self.target_label = "H" if pit == self.grid.home else f"C{pit}"
        self.plan = path[1:]

    def _skip(self, k: int, why: str):
        if k not in self.infeasible:
            self.infeasible.append(k)
            self._note(f"checkpoint {k} skipped: {why}")
            self._log(rx.RULE_BY_ID["R7"].rid, "SKIP", note=f"checkpoint {k} skipped: {why}")

    def _fail(self, why: str):
        self.status = FAILED
        self.plan = []
        self._log("--", "FAIL", note=why)

    # ------------------------------------------------------------------ acting
    def _car_action(self, new_dir: str, nxt: Pos) -> str:
        if new_dir == self.heading:
            steer = "STRAIGHT"
        elif new_dir == LEFT_OF[self.heading]:
            steer = "LEFT"
        elif new_dir == RIGHT_OF[self.heading]:
            steer = "RIGHT"
        else:
            steer = "U_TURN"
        cur = self.grid.current[nxt[0]][nxt[1]]
        slow = self.grid.terrain[nxt[0]][nxt[1]] == GRASS or cur == OPPOSITE[new_dir]
        fast = cur == new_dir
        speed = "BRAKE" if slow else "ACCELERATE" if fast else "CRUISE"
        return f"{steer}+{speed}"

    def current_target(self) -> Optional[str]:
        """Label of what the car is driving towards right now (for logs / HUD)."""
        if self.state == TO_PIT and self.pit_target is not None:
            return "H" if self.pit_target == self.grid.home else f"C{self.pit_target}"
        rem = set(self.remaining)
        for q in self.plan:
            k = self.grid.site_at(q)
            if k in rem and (self.mode == SURVEY or k == min(rem)):
                return str(k)
        return "H"

    def _log(self, rid: str, action: str, car_action: str = "", cost: float = 0,
             note: str = "", rays=None):
        self.log.append(LogEntry(self.t, self.pos, self.fuel, rid, action, car_action, cost,
                                 self.heading, self.state, self.current_target(), note,
                                 rays or {}, tuple(self.plan)))

    def step(self) -> str:
        """Run one sense-decide-act tick. Returns the current status."""
        if self.status != RUNNING:
            return self.status
        if self.t >= self.max_ticks:
            self.status = STEP_LIMIT
            return self.status
        self.t += 1
        p = self.sense()
        rule = rx.decide(p)
        self.rule_counts[rule.rid] = self.rule_counts.get(rule.rid, 0) + 1
        a = rule.action

        if a == rx.FINISH:
            self.status = DONE
            self._log(rule.rid, a, note="finish line crossed", rays=p.rays)
        elif a == rx.MARK:
            self.visited.append(p.checkpoint_here)
            self._log(rule.rid, a, note=f"checkpoint {p.checkpoint_here} cleared", rays=p.rays)
        elif a == rx.REFUEL:
            before = self.fuel
            self.fuel = self.grid.capacity
            self.pit_stops += 1
            self._log(rule.rid, a, note=f"refuel {before:g} -> {self.fuel:g}", rays=p.rays)
            if self.relay:
                self._go_to_pit(self.relay.pop(0))
            elif self.forced_target is not None:
                self._drive_forced()
            else:
                self._replan()
        elif a == rx.GOTO_PIT:
            self._go_to_pit()
            self._log(rule.rid, a, note=f"heading to pit {self.pit_target}", rays=p.rays)
            self.relay, self.forced_target = [], None
        elif a == rx.REPLAN:
            self._log(rule.rid, a, note=f"wall at {p.next_tile}", rays=p.rays)
            self._replan()
        elif a == rx.MOVE:
            if not self.plan:
                # plan used up (e.g. after a forced relay leg): plan the rest of the tour
                self._log(rule.rid, "REPLAN", note="plan finished, re-planning remaining tour")
                self._replan()
                if not self.plan and self.status == RUNNING and \
                        not (self.pos == self.grid.home and not self.remaining):
                    self._fail("planner returned an empty route")
                return self.status
            nxt = self.plan.pop(0)
            cost = self.grid.step_cost(self.pos, nxt)
            if cost > self.fuel:
                self._fail(f"out of fuel moving to {nxt}")
                return self.status
            d = direction_between(self.pos, nxt)
            car = self._car_action(d, nxt)
            if not self.safety.is_safe_move(self.fuel, self.pos, nxt):
                self.violations += 1        # only possible while heading to a pit (never expected)
            self.fuel -= cost
            self.fuel_used += cost
            self.heading = d
            self.pos = nxt
            self.path.append(nxt)
            self.min_fuel = min(self.min_fuel, self.fuel)
            self._log(rule.rid, a, car, cost, rays=p.rays)
        return self.status

    def run(self) -> RunResult:
        while self.step() == RUNNING:
            pass
        return self.result()

    def result(self) -> RunResult:
        return RunResult(
            status=self.status, map_name=self.grid.name, heuristic=self.heuristic,
            mode=self.mode, moves=len(self.path) - 1, fuel_used=self.fuel_used,
            pit_stops=self.pit_stops, replans=self.replans, visit_order=list(self.visited),
            skipped=list(self.infeasible), planned_tour_cost=self.planned_tour_cost,
            searches=self.legs.searches + self.extra_searches,
            nodes_expanded=self.legs.nodes_expanded + self.extra_nodes,
            search_ms=self.legs.runtime_ms + self.extra_ms, min_fuel=self.min_fuel,
            safety_violations=self.violations, log=self.log, path=self.path,
            rule_counts=dict(self.rule_counts))


def run_mission(grid: Grid, heuristic: str = "h3", mode: str = SURVEY, **kw) -> RunResult:
    return RacingAgent(grid, heuristic, mode, **kw).run()
