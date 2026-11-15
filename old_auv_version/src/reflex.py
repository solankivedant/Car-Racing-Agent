"""Reflex module: percepts plus a prioritized table of condition -> action rules.

Each rule is a propositional formula over boolean percept/memory facts.
The rules are checked top to bottom and the first one that holds fires,
like a subsumption-style reflex agent::

    R0  AllDone & AtHome                         -> FINISH
    R6  AtCheckpoint(k) & ~Visited(k)            -> MARK_CHECKPOINT
    R1  AtPit & FuelNotFull & (HeadingToThisPit | (~HeadingToPit & Fuel<=40%))
                                                 -> REFUEL (+ re-plan)
    R2  Fuel<25% & ~HeadingToPit                 -> GOTO_CHEAPEST_PIT
    R4  25%<=Fuel<=40% & PitCost<=4 & ~HeadingToPit
                                                 -> GOTO_CHEAPEST_PIT (override goals)
    R5  NextTile=Wall                            -> REPLAN (never drive into a wall)
    R3  ~SafeAfter(next) & ~HeadingToPit         -> CANCEL move, GOTO_CHEAPEST_PIT
    R7  True                                     -> FOLLOW_PLAN (move one tile)

``R1``-``R4`` are the four rules from the problem statement. ``R5`` is the
reactive wall guard. A* never plans through walls, so this only fires if the
plan is corrupted. ``R6``/``R7``/``R0`` drive the mission.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .world import Pos

# Actions
FINISH = "FINISH"
MARK = "MARK_CHECKPOINT"
REFUEL = "REFUEL"
GOTO_PIT = "GOTO_PIT"
REPLAN = "REPLAN"
MOVE = "MOVE"

LOW_FUEL = 0.25
OPPORTUNISTIC_FUEL = 0.40
NEAR_PIT_COST = 4


@dataclass
class Percept:
    """What the car senses on its current tile, plus what it remembers."""
    pos: Pos
    tile: str                      # asphalt / grass
    char: str                      # raw map char (H, C, 1..9, ., ~)
    current: Optional[str]         # N/E/S/W/None
    fuel: float
    capacity: int
    cost_to_pit: float             # from the safety map
    at_pit: bool
    checkpoint_here: Optional[int]
    rays: Dict[str, int] = field(default_factory=dict)
    heading: str = "N"
    # memory / plan facts
    heading_to_pit: bool = False
    pit_target: Optional[Pos] = None
    visited: frozenset = frozenset()
    all_done: bool = False
    at_home: bool = False
    next_tile: Optional[Pos] = None
    next_is_wall: bool = False
    safe_after_next: bool = True

    @property
    def fuel_pct(self) -> float:
        return self.fuel / self.capacity


@dataclass(frozen=True)
class Rule:
    rid: str
    name: str
    formula: str
    condition: Callable[[Percept], bool]
    action: str


RULES: List[Rule] = [
    Rule("R0", "Mission complete", "AllDone ∧ AtHome",
         lambda p: p.all_done and p.at_home, FINISH),
    Rule("R6", "Checkpoint reached", "AtCheckpoint(k) ∧ ¬Visited(k)",
         lambda p: p.checkpoint_here is not None and p.checkpoint_here not in p.visited, MARK),
    Rule("R1", "Refuel at pit",
         "AtPit ∧ ¬FullTank ∧ (HeadingToThisPit ∨ (¬HeadingToPit ∧ Fuel≤40%))",
         lambda p: p.at_pit and p.fuel < p.capacity and (
             (p.heading_to_pit and p.pit_target == p.pos)
             or (not p.heading_to_pit and p.fuel_pct <= OPPORTUNISTIC_FUEL)),
         REFUEL),
    Rule("R2", "Low fuel (<25%)", "Fuel<25% ∧ ¬HeadingToPit",
         lambda p: p.fuel_pct < LOW_FUEL and not p.heading_to_pit, GOTO_PIT),
    Rule("R4", "Pit nearby, fuel 25–40%", "25%≤Fuel≤40% ∧ PitCost≤4 ∧ ¬HeadingToPit",
         lambda p: (LOW_FUEL <= p.fuel_pct <= OPPORTUNISTIC_FUEL
                    and p.cost_to_pit <= NEAR_PIT_COST and not p.heading_to_pit),
         GOTO_PIT),
    Rule("R5", "Wall ahead", "NextTile=Wall",
         lambda p: p.next_tile is not None and p.next_is_wall, REPLAN),
    Rule("R3", "Unsafe move", "¬SafeAfter(next) ∧ ¬HeadingToPit",
         lambda p: p.next_tile is not None and not p.safe_after_next and not p.heading_to_pit,
         GOTO_PIT),
    Rule("R7", "Follow plan", "⊤", lambda p: True, MOVE),
]

RULE_BY_ID = {r.rid: r for r in RULES}


def decide(percept: Percept, rules: List[Rule] = RULES) -> Rule:
    """Return the first (highest-priority) rule whose condition holds."""
    for rule in rules:
        if rule.condition(percept):
            return rule
    raise RuntimeError("No rule fired")  # unreachable: R7 always holds
