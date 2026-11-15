"""Car Racing Agent: command-line entry point.

Examples
--------
    python main.py run     --map maps/map_main.txt --heuristic h4 --mode survey --trace
    python main.py run     --map maps/track_main.txt --mode race --log demo/track_log.csv
    python main.py gui     --map maps/map_main.txt --heuristic h3
    python main.py record  --map maps/map_main.txt --out demo/map_main_h4.gif
    python main.py compare --map maps/map_main.txt
    python main.py plan    --map maps/map_main.txt --heuristic h1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from src.agent import run_mission
from src.heuristics import HEURISTIC_INFO, all_heuristics
from src.search import astar
from src.tour import plan_tour
from src.world import load_map


def cmd_run(a):
    g = load_map(a.map)
    res = run_mission(g, a.heuristic, a.mode, verbose=a.verbose)
    if a.trace:
        print(f"{'t':>4} {'pos':>9} {'fuel':>5} rule action           car                  target note")
        for e in res.log:
            print(f"{e.t:>4} {str(e.pos):>9} {e.fuel:>5g} {e.rule:<4} {e.action:<16} "
                  f"{e.car_action:<20} {str(e.target):<6} {e.note}")
        print()
    print(g.render(res.path))
    print()
    print(res.summary())
    if a.log:
        _write_log(res, a.log)
        print(f"log written to {a.log}")
    return 0 if res.status == "DONE" else 1


def _write_log(res, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in res.log:
        d = asdict(e)
        d.pop("plan", None)
        d.update({k: v for k, v in d.pop("rays").items()})
        d["pos"] = f"{e.pos[0]},{e.pos[1]}"
        rows.append(d)
    if path.endswith(".json"):
        Path(path).write_text(json.dumps(rows, indent=1))
    else:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                      if k in rows[0] else 99)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def cmd_compare(a):
    g = load_map(a.map)
    print(f"Map {g.name}: {g.rows}x{g.cols}, capacity {g.capacity}, "
          f"{len(g.sites)} checkpoints, {len(g.pits)} refuel points\n")
    pts = {"H": g.home, **{str(k): p for k, p in g.sites.items()}}
    print("Single-leg A* over all ordered pairs of {H, checkpoints}:")
    print(f"{'heur':<5}{'description':<36}{'total cost':>11}{'expanded':>10}{'ms':>9}")
    for h in all_heuristics():
        cost = exp = ms = 0
        for a_ in pts.values():
            for b_ in pts.values():
                if a_ == b_:
                    continue
                r = astar(g, a_, b_, h)
                cost += r.cost
                exp += r.nodes_expanded
                ms += r.runtime_ms
        print(f"{h:<5}{HEURISTIC_INFO[h]:<36}{cost:>11g}{exp:>10}{ms:>9.2f}")
    print("\nFull mission (reflex + A*):")
    for h in all_heuristics():
        r = run_mission(g, h, a.mode)
        print(f"  {h}: status={r.status} fuel_used={r.fuel_used:g} moves={r.moves} "
              f"pit_stops={r.pit_stops} expanded={r.nodes_expanded}")
    return 0


def cmd_plan(a):
    g = load_map(a.map)
    t = plan_tour(g, g.home, list(g.sites), a.mode, a.heuristic)
    print(g.render(t.route))
    print(f"\norder: H -> {' -> '.join(map(str, t.order))} -> H   cost (ignoring fuel) = {t.cost:g}")
    if t.skipped:
        print("skipped (unreachable):", t.skipped)
    return 0


def cmd_gui(a):
    from src.visualize import Viewer
    Viewer(load_map(a.map), a.heuristic, a.mode, speed=a.speed).loop()
    return 0


def cmd_record(a):
    from src.visualize import Viewer
    v = Viewer(load_map(a.map), a.heuristic, a.mode, headless=True)
    out = v.record(a.out, fps=a.fps)
    print(f"recorded {out}  ({v.result.summary().splitlines()[0]})")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Car Racing Agent (reflex + A*)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--map", default="maps/map_main.txt")
        sp.add_argument("--heuristic", "-H", default="h4", choices=all_heuristics())
        sp.add_argument("--mode", default="survey", choices=["survey", "race"])

    sp = sub.add_parser("run", help="headless mission + summary")
    common(sp)
    sp.add_argument("--trace", action="store_true", help="print every tick")
    sp.add_argument("--verbose", action="store_true")
    sp.add_argument("--log", help="write step log (.csv or .json)")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("gui", help="interactive pygame viewer")
    common(sp)
    sp.add_argument("--speed", type=float, default=6.0, help="ticks per second")
    sp.set_defaults(fn=cmd_gui)

    sp = sub.add_parser("record", help="render the mission to GIF/MP4")
    common(sp)
    sp.add_argument("--out", default="demo/mission.gif")
    sp.add_argument("--fps", type=int, default=20)
    sp.set_defaults(fn=cmd_record)

    sp = sub.add_parser("compare", help="quick heuristic comparison table")
    common(sp)
    sp.set_defaults(fn=cmd_compare)

    sp = sub.add_parser("plan", help="show the planned tour (no fuel rules)")
    common(sp)
    sp.set_defaults(fn=cmd_plan)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
