"""Car Racing Agent: command-line entry point.

    python main.py gui                          # watch the race (keys: see racing/gui.py)
    python main.py gui --mode train             # watch the cars learn (genetic algorithm)
    python main.py race  --track tracks/technical.track
    python main.py train --generations 80       # train the learning agent -> models/ga_best.json
    python main.py compare                      # A* heuristic comparison table (all tracks)
    python main.py editor                       # paint your own track (then race / train on it)
    python main.py record --out demo/race_serpentine.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TRACKS = ["tracks/serpentine.track", "tracks/technical.track", "tracks/oval.track"]


def all_tracks():
    """Built-in tracks first, then every track painted with the editor."""
    extra = sorted(str(p).replace("\\", "/") for p in Path("tracks").glob("*.track")
                   if str(p).replace("\\", "/") not in TRACKS)
    return TRACKS + extra
MODEL = "models/ga_best.json"


def _neuro():
    from racing.learning import load_model
    if Path(MODEL).exists():
        return load_model(MODEL)[0]
    print(f"(no {MODEL} yet: run `python main.py train` to add the learning agent)")
    return None


def cmd_race(a):
    from racing.agents import default_lineup
    from racing.race import run_race
    from racing.track import load_track
    for f in a.track or TRACKS:
        t = load_track(f)
        r = run_race(t, default_lineup(_neuro()))
        print(f"\n{t.name}  (lap {t.length:.0f}px)")
        print(f"{'pos':>3} {'car':<14}{'kind':<9}{'status':<10}{'lap time':>9}{'progress':>10}{'avg v':>8}{'top v':>8}")
        for x in r.results():
            lt = f"{x['lap_time']:.2f}s" if x["lap_time"] else "-"
            print(f"{x['pos']:>3} {x['car']:<14}{x['kind']:<9}{x['status']:<10}{lt:>9}"
                  f"{x['progress_pct']:>9.1f}%{x['avg_speed']:>8.0f}{x['top_speed']:>8.0f}")
    return 0


def _ui_session(request):
    """Switch between race/train view and the editor until the user closes a window."""
    from racing.editor import Editor
    from racing.gui import App
    neuro = _neuro()
    while request:
        kind, track = request
        if kind == "editor":
            request = Editor(start_from=track).loop()
        else:
            files = all_tracks()
            track = str(track).replace("\\", "/")
            idx = files.index(track) if track in files else 0
            request = App(files, neuro, mode=kind, start_index=idx).loop()
    import pygame
    pygame.quit()
    return 0


def cmd_gui(a):
    track = (a.track or [TRACKS[0]])[0]
    return _ui_session((a.mode, track))


def cmd_editor(a):
    return _ui_session(("editor", (a.track or [None])[0]))


def cmd_record(a):
    from racing.gui import App
    app = App(a.track or TRACKS[:1], _neuro(), headless=True, scale=a.scale or 0.8, mode=a.mode)
    app.speed = a.speed
    out = app.record(a.out, max_seconds=a.seconds)
    print("recorded", out)
    return 0


def cmd_train(a):
    from racing.learning import GAConfig, evaluate, evolve, save_model
    from racing.track import load_track
    cfg = GAConfig(population=a.population, generations=a.generations, seed=a.seed)
    train = a.track or TRACKS[:2]
    w, hist = evolve([load_track(f) for f in train], cfg)
    save_model(a.out, w, hist, cfg, [Path(f).stem for f in train])
    print("saved", a.out)
    for f in TRACKS:
        r = evaluate(w[None, :], load_track(f))
        tag = "train" if f in train else "UNSEEN"
        print(f"  {Path(f).stem:<11} ({tag}): finished={bool(r.finished[0])} lap={r.lap_time[0]:.2f}s")
    return 0


def cmd_compare(a):
    from racing.planner import HEURISTICS, Grid, astar
    from racing.track import load_track
    for f in a.track or TRACKS:
        t = load_track(f)
        g = Grid(t)
        s, goal = g.lap_endpoints()
        g.landmarks()
        print(f"\n{t.name}: full-lap A* on a {g.R}x{g.C} grid ({int(g.free.sum())} free cells)")
        print(f"{'h':<4}{'heuristic':<28}{'cost':>9}{'expanded':>10}{'ms':>9}")
        for h in HEURISTICS:
            r = astar(g, s, goal, h)
            print(f"{h:<4}{HEURISTICS[h]:<28}{r.cost:>9.1f}{r.expanded:>10}{r.ms:>9.1f}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Car Racing Agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("race"); sp.add_argument("--track", nargs="*"); sp.set_defaults(fn=cmd_race)
    sp = sub.add_parser("gui"); sp.add_argument("--track", nargs="*")
    sp.add_argument("--scale", type=float); sp.add_argument("--mode", default="race", choices=["race", "train"])
    sp.set_defaults(fn=cmd_gui)
    sp = sub.add_parser("record"); sp.add_argument("--track", nargs="*")
    sp.add_argument("--out", default="demo/race.gif"); sp.add_argument("--scale", type=float)
    sp.add_argument("--mode", default="race", choices=["race", "train"])
    sp.add_argument("--speed", type=int, default=1); sp.add_argument("--seconds", type=float, default=60)
    sp.set_defaults(fn=cmd_record)
    sp = sub.add_parser("train"); sp.add_argument("--track", nargs="*")
    sp.add_argument("--generations", type=int, default=80); sp.add_argument("--population", type=int, default=60)
    sp.add_argument("--seed", type=int, default=7); sp.add_argument("--out", default=MODEL)
    sp.set_defaults(fn=cmd_train)
    sp = sub.add_parser("compare"); sp.add_argument("--track", nargs="*"); sp.set_defaults(fn=cmd_compare)
    sp = sub.add_parser("editor", help="paint your own track"); sp.add_argument("--track", nargs="*")
    sp.set_defaults(fn=cmd_editor)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
