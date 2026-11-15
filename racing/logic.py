"""Tiny propositional-logic rule engine used by the reflex and safety layers.

Percepts are turned into a set of true atoms (propositions), e.g.
``{"FrontBlocked", "LeftMoreOpen"}``. A rule is a conjunction of literals
(atom or ¬atom) implying an action::

    FrontBlocked ∧ LeftMoreOpen ∧ ¬Slow  ->  LEFT + BRAKE

Rules are checked in priority order and the first one whose antecedent is
satisfied fires (a production system / subsumption style reflex agent).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

Action = Tuple[int, int]                      # (steer, throttle)
ActionFn = Union[Action, Callable[[dict], Action]]


@dataclass(frozen=True)
class Literal:
    atom: str
    positive: bool = True

    def holds(self, facts: set) -> bool:
        return (self.atom in facts) == self.positive

    def __str__(self):
        return self.atom if self.positive else f"¬{self.atom}"


def lit(s: str) -> Literal:
    s = s.strip()
    if s.startswith(("~", "¬", "!")):
        return Literal(s[1:].strip(), False)
    return Literal(s, True)


@dataclass(frozen=True)
class Rule:
    rid: str
    name: str
    antecedent: Tuple[Literal, ...]
    action: ActionFn

    def fires(self, facts: set) -> bool:
        return all(l.holds(facts) for l in self.antecedent)

    @property
    def formula(self) -> str:
        lhs = " ∧ ".join(str(l) for l in self.antecedent) or "⊤"
        return f"{lhs} → {self.name}"

    def resolve(self, ctx: dict) -> Action:
        return self.action(ctx) if callable(self.action) else self.action


def rule(rid: str, name: str, antecedent: str, action: ActionFn) -> Rule:
    """``rule("S1", "LEFT+BRAKE", "FrontBlocked & LeftMoreOpen & ~Slow", (-1, -1))``"""
    lits = tuple(lit(a) for a in antecedent.replace("∧", "&").split("&") if a.strip())
    return Rule(rid, name, lits, action)


class RuleBase:
    def __init__(self, rules: Sequence[Rule]):
        self.rules = list(rules)

    def first_match(self, facts: set) -> Optional[Rule]:
        for r in self.rules:
            if r.fires(facts):
                return r
        return None

    def decide(self, facts: set, ctx: dict) -> Tuple[Action, Rule]:
        r = self.first_match(facts)
        if r is None:
            raise RuntimeError("no rule fired; the rule base needs a default rule (⊤)")
        return r.resolve(ctx), r

    def table(self) -> List[Tuple[str, str]]:
        return [(r.rid, r.formula) for r in self.rules]


def atoms(pairs: Iterable[Tuple[str, bool]]) -> set:
    return {name for name, val in pairs if val}

