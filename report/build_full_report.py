"""Builds report/final_report.pdf: the complete report covering all deliverables.

    python notebooks/build_notebook.py      # produces report/figures/*
    python report/build_full_report.py
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

_FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for nm, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
              ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(nm, str(_FD / f)))
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B"); addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-BI")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"
DEMO = ROOT / "demo"
OUT = HERE / "final_report.pdf"
BLUE = colors.HexColor("#184f95")
GREY = colors.HexColor("#52514e")

BODY = ParagraphStyle("b", fontName="DV", fontSize=9.4, leading=12.4, alignment=TA_JUSTIFY, spaceAfter=5)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
H1 = ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=BLUE, spaceBefore=4, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="DV-B", fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=8, spaceAfter=4)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8, leading=10, textColor=GREY, alignment=TA_CENTER)
CELL = ParagraphStyle("c", fontName="DV", fontSize=8.2, leading=10)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="DV-B")
CODE = ParagraphStyle("code", fontName="DVM", fontSize=8.2, leading=10.5, backColor=colors.HexColor("#f2f4f7"),
                      borderPadding=5, leftIndent=4, spaceBefore=3, spaceAfter=7)
TITLE = ParagraphStyle("t", fontName="DV-B", fontSize=24, leading=30, alignment=TA_CENTER, textColor=BLUE)
SUBT = ParagraphStyle("st", fontName="DV", fontSize=12, leading=17, alignment=TA_CENTER, textColor=GREY)


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def code(text):
    return Paragraph(text.replace("\n", "<br/>").replace(" ", "&nbsp;"), CODE)


def table(rows, widths, header=True, zebra=True):
    data = [[(Paragraph(str(c), CELLB if (header and i == 0) else CELL) if isinstance(c, (str, int, float)) else c)
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c3c2b7")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0fb")))
    if zebra:
        for i in range(2 if header else 1, len(rows), 2):
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fafafa")))
    t.setStyle(TableStyle(st))
    return t


def img(path, width_cm, max_h_cm=None):
    w, h = PILImage.open(path).size
    width = width_cm * cm
    height = width * h / w
    if max_h_cm and height > max_h_cm * cm:
        height = max_h_cm * cm
        width = height * w / h
    return Image(str(path), width=width, height=height)


def figure(path, width_cm, caption, max_h_cm=None):
    return KeepTogether([img(path, width_cm, max_h_cm), Spacer(1, 2), P(caption, CAP), Spacer(1, 6)])


def count_tests():
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                             capture_output=True, text=True, timeout=900).stdout.strip().splitlines()
        return out[-1] if out else "not run"
    except Exception as e:                      # pragma: no cover
        return f"not run ({e})"


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(GREY)
    if doc.page > 1:
        canvas.drawString(1.8 * cm, 1.0 * cm, "CSE518 AI · Car Racing Agent · Final report on all deliverables")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"page {doc.page}")
    canvas.restoreState()


def main():
    lap = pd.read_csv(FIG / "lap_search.csv")
    rp = pd.read_csv(FIG / "random_pairs_summary.csv", index_col=0)
    inf = pd.read_csv(FIG / "informedness.csv")
    tt = pd.read_csv(FIG / "time_trials.csv")
    sw = pd.read_csv(FIG / "alt_sweep.csv")
    ru = pd.read_csv(FIG / "rule_usage.csv")
    model = json.loads((ROOT / "models" / "ga_best.json").read_text())
    tests = count_tests()
    TR = ["serpentine", "technical", "oval"]
    TN = {"serpentine": "Serpentine", "technical": "Technical", "oval": "Oval"}
    HN = {"h0": "Dijkstra (h = 0)", "h1": "Euclidean", "h2": "Manhattan", "h3": "Octile", "h4": "ALT landmarks"}

    def lp(t, h, col):
        return lap[(lap.track == t) & (lap.heuristic == h)][col].iloc[0]

    def mi(t, h, col="mean_h_over_hstar"):
        return inf[(inf.track == t) & (inf.heuristic == h)][col].iloc[0]

    piv = tt.pivot_table(index="agent", columns="track", values="lap_time")
    s = []

    # ------------------------------------------------------------------ title page
    s += [Spacer(1, 3.2 * cm), P("Car Racing Agent", TITLE), Spacer(1, 6),
          P("Reflex rules, A* path planning, utility and learning agents on a race track", SUBT),
          Spacer(1, 1.2 * cm),
          P("CSE518 Artificial Intelligence · Monsoon 2026 · Project Part 1", SUBT),
          P("Final report on all deliverables", SUBT), Spacer(1, 1.0 * cm),
          P("Vedant Solanki · Enrollment No. AU2340012", SUBT),
          P(date.today().strftime("%d %B %Y"), SUBT), Spacer(1, 1.2 * cm)]
    s += [img(DEMO / "race_serpentine.png", 16.5), Spacer(1, 10)]
    s += [table([["Deliverable", "What", "File(s)"],
                 ["1", "Code: reflex module, A* path planning, plus utility and learning agents, multi-car race, GUI, "
                       "track editor, tests", "racing/, main.py, tests/, tracks/, models/"],
                 ["2", "Comparison of the 4 A* heuristics (+ driving comparison)", "notebooks/heuristic_comparison.ipynb"],
                 ["3", "Short (~2 pages) progress report", "report/progress_report.pdf"],
                 ["+", "Demo videos and screenshots", "demo/*.mp4, demo/*.png"]],
                [2.2 * cm, 9.3 * cm, 6.3 * cm])]
    s += [PageBreak()]

    # ------------------------------------------------------------------ 1 overview
    s += [P("1. Project overview", H1), P(
        "The approved project is a <b>Car Racing Agent</b>. Several AI cars race one lap on a winding 2-D track, from "
        "the start line to the finish line, and every car must avoid the walls and rocks. The cars sense the track with "
        "five distance rays and choose a steering action and a throttle action 30 times per second. Different kinds of "
        "AI race against each other, which answers the question in the inspiration video: <i>\"Can I make a better AI "
        "than AI?\"</i> Hand-made agents (reflex rules, A* search, utility maximisation) race a learned agent "
        "(a neural network evolved by a genetic algorithm).")]
    s += [P("Mapping of the proposal to the implementation", H2)]
    s += [table([["Proposal item", "Implementation", "Where"],
                 ["1 Intelligent agent: car", "4 agent types (Reflex, A* h1–h4, Utility, Learned GA)", "racing/agents.py"],
                 ["2 Environment: race track", "top-down tracks from files; painted tracks from the editor; walls, rocks",
                  "racing/track.py, tracks/"],
                 ["3 State: position, direction, speed, sensor distances",
                  "x, y, heading, speed, velocity (vx, vy), five ray distances", "racing/car.py (CarState)"],
                 ["4 Actions: left, right, straight, accelerate, brake",
                  "steer LEFT/STRAIGHT/RIGHT × throttle ACCELERATE/COAST/BRAKE (9 combinations)", "racing/car.py"],
                 ["5 Goal: finish line / complete the track", "one lap; crash = out; ranking by lap time", "racing/race.py"],
                 ["6 Search: A*", "occupancy-grid A*, 4 heuristics + Dijkstra baseline", "racing/planner.py"],
                 ["7 Sensors with weights w1–w5", "rays at −90°, −45°, 0°, +45°, +90°; weights in the utility function",
                  "racing/car.py, agents.py"],
                 ["8 Utility: progress vs collision vs speed", "look-ahead utility agent", "racing/agents.py"],
                 ["9 Rules: safety / follow path", "reflex rule base; A* car = follow path + safety rules",
                  "racing/agents.py"],
                 ["10 Learning: repeatedly running", "genetic algorithm over a population of 60 cars", "racing/learning.py"],
                 ["11 Propositional logic for rules", "rule engine over boolean percept atoms", "racing/logic.py"]],
                [5.2 * cm, 8.0 * cm, 4.6 * cm])]
    s += [Spacer(1, 6), figure(FIG / "tracks.png", 17,
                               "Figure 1. The three built-in tracks with the A* (h4) racing line. The serpentine "
                               "track reproduces the hand-drawn track of the proposal. Brown dots are rocks.")]

    # ------------------------------------------------------------------ 2 deliverable 1
    s += [PageBreak(), P("2. Deliverable 1: the code", H1)]
    s += [P("2.1 Architecture", H2)]
    s += [table([["Module", "Responsibility"],
                 ["racing/track.py", "Loads tracks. A spline track (centre-line + width) or a painted track (PNG mask + "
                  "start pose). Pre-computes the drivable mask, the wall clearance and the lap position of every pixel."],
                 ["racing/car.py", "Kinematic physics (acceleration, braking, drag, speed-dependent grip), collision with "
                  "the 28×14 px car body, and five vectorised ray sensors."],
                 ["racing/logic.py", "Propositional rule engine: literals, conjunctive rules, priority order."],
                 ["racing/planner.py", "Occupancy grid, A* with statistics, heuristics h0–h4, ALT landmarks, racing line "
                  "and speed profile."],
                 ["racing/agents.py", "ReflexAgent, PathAgent (A* + safety rules), UtilityAgent, NeuroAgent."],
                 ["racing/race.py", "Multi-car race at 30 Hz; progress, finish, crash, stuck and timeout detection."],
                 ["racing/learning.py", "Genetic algorithm. The whole population is simulated in parallel with numpy."],
                 ["racing/gui.py, view.py", "pygame race view and live training view (sharp, DPI-aware, resizable)."],
                 ["racing/editor.py", "Track editor: paint road / grass, place rocks, set the start, validate, save."],
                 ["main.py", "Command line: gui · editor · race · train · compare · record."]],
                [3.8 * cm, 14.0 * cm])]

    s += [P("2.2 Environment, car and sensors", H2), P(
        "A track file gives the closed centre-line and the width (smoothed with a closed Catmull-Rom spline), or a painted "
        "road mask. From it, scipy distance transforms compute, for every pixel, whether it is drivable, the clearance to "
        "the nearest wall, and the <i>lap position</i>, used for race progress and for the finish line. Painted tracks have "
        "no centre-line, so their lap position is computed <b>geodesically</b>: a Dijkstra run starts from the start line, "
        "with a barrier just behind it. The car is a kinematic model: acceleration 170 px/s², braking 420 px/s², "
        "drag, top speed 260 px/s, and a yaw rate that is weak at standstill and reduced at high speed (less grip). "
        "The car touches a wall if any corner of its 28×14 px body leaves the road. Five rays w1–w5 measure the free "
        "distance (up to 220 px) to the left, front-left, front, front-right and right.")]

    s += [P("2.3 Reflex module", H2), P(
        "Raw readings become boolean propositions, for example <i>FrontBlocked</i> ⇔ front ray &lt; braking distance "
        "v²/(2·0.85·a<sub>brake</sub>) + margin, <i>LeftMoreOpen</i> ⇔ w1 + 1.4·w2 &gt; w5 + 1.4·w4, and <i>TooFast</i>, "
        "<i>Slow</i>, <i>LeftTight</i>, <i>DiagLeftTight</i>, <i>VeerLeft</i>, … The rule base is checked in priority "
        "order, and the first rule whose conjunction of literals holds fires:")]
    s += [table([["Rule", "Condition (propositional logic)", "Action"],
                 ["R1", "FrontBlocked ∧ ¬Slow", "turn to the open side + BRAKE"],
                 ["R2", "FrontBlocked ∧ Slow", "turn to the open side + crawl (ACCELERATE)"],
                 ["R3 / R4", "DiagLeftTight ∧ ¬DiagRightTight (and mirror)", "steer away (accelerate if slow)"],
                 ["R5 / R6", "LeftTight ∧ ¬RightTight (and mirror)", "steer away + ACCELERATE"],
                 ["R7", "FrontNear ∧ TooFast", "steer to the open side + BRAKE"],
                 ["R8", "FrontNear ∧ ¬TooFast", "steer to the open side + COAST"],
                 ["R9 / R10", "VeerLeft / VeerRight", "follow the bend + ACCELERATE"],
                 ["R11", "⊤ (default)", "STRAIGHT + ACCELERATE"]],
                [2.0 * cm, 8.6 * cm, 7.2 * cm])]
    s += [Spacer(1, 4), P(
        "The A* car uses the same propositions for its <b>safety layer</b>, matching the proposal's \"Rules: safety / follow "
        "path\": S1 FrontBlocked ∧ ¬Slow ∧ OffLine → emergency brake and evade; S2/S3 side wall → evade; P1 ⊤ → follow "
        "the planned path. On the serpentine track the A* car follows its path in "
        f"{ru[(ru.agent == 'A* h4') & (ru.rule == 'P1')].share_pct.iloc[0]:.0f}% of the ticks. The rest are safety "
        "interventions near walls and rocks.")]

    s += [P("2.4 Path-planning module (A*)", H2), P(
        "The track is cut into 8 px cells (90×150 grid, 8-connected, no corner cutting). A cell is free if the clearance "
        "at its centre is at least the car's safety radius (16 px), which makes the grid a configuration space. A move costs "
        "length × (p(a) + p(b))/2, with wall penalty p = 1 + 1.5·(1 − clearance / half-width) ≥ 1. The planned line therefore "
        "keeps a safety margin, and every move costs at least its straight-line length. Start and goal both lie on the "
        "start line, and a barrier of cells across the line forces A* to plan the <b>whole lap</b>. A* uses a binary heap, "
        "breaks ties towards larger g, and re-opens closed nodes (needed for inadmissible heuristics). The cell path is "
        "smoothed into a racing line. Pure-pursuit steering follows it, with a speed profile from the line's curvature "
        "and look-ahead braking: v<sub>lim</sub> = min<sub>j</sub> √(v<sub>corner,j</sub>² + 2·a·d<sub>j</sub>) over "
        "the next 320 px.")]

    s += [P("2.5 Utility agent and learning agent", H2), P(
        "<b>Utility agent.</b> Every tick it simulates each of the 9 actions 10 ticks ahead with the physics model, and "
        "picks the action with the highest utility U = progress + 0.15·speed − 6·Σ w<sub>i</sub>·e<super>−ray<sub>i</sub>/18"
        "</super> − 0.6·overspeed − 10<super>4</super>·crash. The sensor weights are w = (0.6, 1.2, 2.0, 1.2, 0.6). "
        "<b>Learning agent.</b> A 7-10-2 tanh network maps 5 rays, the speed and a bias to steering and throttle (102 "
        f"weights). A genetic algorithm (population {model['config']['population']}, {model['config']['generations']} "
        f"generations, elitism {model['config']['elite']}, tournament selection, uniform crossover, Gaussian mutation, "
        "random immigrants) evolves the weights. Each driver races every training track twice, from the start line and "
        "from a random grid slot. Fitness = lap fraction (+1 + 3·time bonus when finished). Training tracks: "
        f"{', '.join(model['train_tracks'])}. The oval is held out to test generalisation.")]
    s += [figure(FIG / "ga_learning.png", 16,
                 f"Figure 2. GA learning curve. The best fitness rises from {model['history']['best'][0]:.2f} "
                 f"to {model['history']['best'][-1]:.2f}; the right plot shows the share of the population that "
                 "finishes every training lap.")]

    s += [P("2.6 User interface and track editor", H2), P(
        "<font face='DVM'>python main.py gui</font> opens a maximised, resizable, DPI-aware window, drawn at the "
        "screen's own resolution. All cars race at once. A leaderboard shows position, lap time and speed. For the "
        "selected car the view shows its five rays, its planned A* line, the rule that fired with its logic formula, "
        "the true propositions, the utilities and the sensor weights. Key <font face='DVM'>T</font> switches to the live "
        "<b>training view</b>, where 50 neural-network cars learn the track generation by generation. "
        "<font face='DVM'>python main.py editor</font> opens the <b>track editor</b>. You paint the road and the outside "
        "area with a brush, drop rocks, and set the start line and direction. The editor checks that the road is a "
        "closed loop that a car can fit through and that A* can plan it, shows the driving direction, and saves the "
        "track. The new track can be raced or trained on straight away.")]
    s += [KeepTogether([table([[img(DEMO / "learning_serpentine.png", 8.6), img(DEMO / "editor.png", 8.6)]],
                              [8.9 * cm, 8.9 * cm], header=False, zebra=False),
                        P("Figure 3. Left: live training view (genetic algorithm). Right: track editor; the "
                          "colours show the computed driving direction.", CAP)])]

    s += [P("2.7 Testing", H2), P(f"Automated test suite (pytest, <font face='DVM'>tests/</font>): <b>{tests}</b>. "
                                  "The tests cover:")]
    s += bullets([
        "Tracks: parsing, malformed files, no self-overlap, rocks not drivable, grid slots on the road, painted tracks "
        "(lap length, progress direction, open roads and narrow roads rejected, save/load round trip).",
        "Physics and sensors: acceleration and braking limits, no reversing, steering direction, grip, wall collision, "
        "ray distances vs road geometry, vectorised vs single raycasts.",
        "Logic and rules: literal parsing, priority order, default rule, and <b>one test per reflex rule R1–R11</b>.",
        "A*: valid paths for every heuristic, full laps (no shortcut across the start line), optimality against Dijkstra "
        "on full laps and random queries, admissibility and consistency of h1/h3/h4 on every cell, Manhattan's "
        "over-estimation, Octile ≥ Euclidean, ALT expands the fewest nodes, racing-line clearance.",
        "Agents and race: every agent finishes every built-in track; a full race ranks by lap time; utility recovers from "
        "a bad heading; crash, stuck and timeout detection.",
        "Learning: network shapes, GA operators keep the elite, fitness improves, the trained model finishes all tracks.",
        "GUI and editor (headless): layout and coordinate mapping, rendering at screen resolution, painting, validation, "
        "undo, save, refusing to overwrite built-in tracks."])

    s += [P("2.8 How to run", H2),
          code("pip install -r requirements.txt\n"
               "python main.py gui                # watch the race (TAB car, 1-9 track, T train)\n"
               "python main.py gui --mode train   # watch the cars learn\n"
               "python main.py editor             # paint your own track\n"
               "python main.py race               # results table on all tracks\n"
               "python main.py compare            # A* heuristic table\n"
               "python main.py train              # re-train the learning agent\n"
               "python -m pytest -q               # run the tests")]

    # ------------------------------------------------------------------ 3 deliverable 2
    s += [Spacer(1, 10), P("3. Deliverable 2: comparison of the A* heuristics", H1), P(
        "The notebook <font face='DVM'>notebooks/heuristic_comparison.ipynb</font> runs every experiment and saves its "
        "tables and figures, which this report re-uses. Four heuristics were designed and compared, with Dijkstra as the "
        "baseline:")]
    s += [table([["id", "Heuristic", "Formula / idea", "Admissible?", "Consistent?"],
                 ["h0", "Dijkstra (baseline)", "h = 0", "yes", "yes"],
                 ["h1", "Euclidean", "√(dx² + dy²); every move costs ≥ its length", "yes", "yes"],
                 ["h2", "Manhattan", "|dx| + |dy|; a diagonal costs √2 &lt; 2, so it over-estimates", "<b>no</b>", "no"],
                 ["h3", "Octile", "max(dx,dy) + (√2−1)·min(dx,dy); exact on an empty 8-grid", "yes", "yes"],
                 ["h4", "ALT landmarks", "max<sub>L</sub> |d(L,n) − d(L,g)| from exact Dijkstra tables of 8 landmarks "
                  "(triangle inequality)", "yes", "yes"]],
                [1.0 * cm, 3.2 * cm, 8.6 * cm, 2.4 * cm, 2.6 * cm])]
    s += [Spacer(1, 6), P("3.1 Full-lap planning (the query the car actually makes)", H2), P(
        "Start and goal are both on the start line, so the goal is only a few cells away in a straight line, but the true "
        "cost is a whole lap. This is the hardest case for purely geometric heuristics.")]
    rows = [["h", "Heuristic"] + [f"{TN[t]}: expanded (% of h0)" for t in TR] + ["Serp. ms", "cost gap"]]
    for h in ["h0", "h1", "h2", "h3", "h4"]:
        rows.append([h, HN[h]] + [f"{int(lp(t, h, 'expanded'))} ({100 * lp(t, h, 'expanded_vs_dijkstra'):.0f}%)" for t in TR]
                    + [f"{lp('serpentine', h, 'ms'):.0f}", f"{lp('serpentine', h, 'gap_pct'):.2f}%"])
    s += [table(rows, [0.8 * cm, 3.0 * cm, 3.3 * cm, 3.3 * cm, 3.3 * cm, 1.7 * cm, 2.4 * cm]),
          P("Table 1. Full-lap A* on the three tracks. The cost gap is the extra path cost compared with the optimal lap.",
            CAP)]
    s += [figure(FIG / "lap_bars.png", 17, "Figure 4. Full-lap search: nodes expanded, time and cost gap per "
                                           "heuristic (darkest bar = serpentine, then technical, then oval).")]
    s += [figure(FIG / "closed_sets.png", 17, "Figure 5. Cells expanded (blue) for the full-lap query on the "
                                              "serpentine track. Red = path found. The last panel shows the ALT "
                                              "heuristic value per cell.", max_h_cm=10.5)]

    s += [P("3.2 Random start/goal queries", H2)]
    rrows = [["h", "Heuristic", "mean expanded", "% of h0", "mean ms", "optimal", "max cost gap"]]
    for h, r in rp.iterrows():
        rrows.append([h, HN[h], f"{r.mean_expanded:.0f}", f"{100 * r.vs_dijkstra:.0f}%", f"{r.mean_ms:.1f}",
                      f"{r.optimal_pct:.0f}%", f"{r.max_gap_pct:.1f}%"])
    s += [table(rrows, [0.8 * cm, 3.2 * cm, 2.7 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.6 * cm]),
          P("Table 2. 360 random pairs of free cells (120 per track).", CAP)]
    s += [figure(FIG / "random_pairs.png", 17, "Figure 6. Search effort relative to Dijkstra per query (left) and "
                                               "against query distance (right).")]

    s += [P("3.3 Admissibility, consistency and informedness", H2)]
    irows = [["h"] + [f"{TN[t]} mean h/h*" for t in TR] + ["over-estimates", "consistency violations"]]
    for h in ["h1", "h2", "h3", "h4"]:
        irows.append([f"{h} {HN[h]}"] + [f"{mi(t, h):.3f}" for t in TR]
                     + [str(int(sum(mi(t, h, 'overestimates') for t in TR))),
                        str(int(sum(mi(t, h, 'consistency_violations') for t in TR)))])
    s += [table(irows, [3.6 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.8 * cm, 3.6 * cm]),
          P("Table 3. Checked on every free cell (h* from Dijkstra) and every edge, summed over the three tracks.", CAP)]
    s += [Spacer(1, 4), P(
        f"<b>Landmark count.</b> On the serpentine lap ALT expands {int(sw[sw.landmarks == 1].expanded.iloc[0])} nodes "
        f"with a single farthest-point landmark, against {int(lp('serpentine', 'h0', 'expanded'))} for Dijkstra, and "
        f"{int(sw.expanded.min())} with {int(sw.loc[sw.expanded.idxmin(), 'landmarks'])} landmarks. Most of the gain comes "
        f"from the first landmark on a loop track. The pre-computation grows from {sw.precompute_ms.min():.0f} ms to "
        f"{sw.precompute_ms.max():.0f} ms.")]

    s += [P("3.4 Does the heuristic matter on the road? Who is the better AI?", H2)]
    order = [a for a in ["A* (h1)", "A* (h2)", "A* (h3)", "A* (h4)", "Utility", "Reflex", "Learned (GA)"] if a in piv.index]
    trows = [["Agent"] + [f"{TN[t]} lap (s)" for t in TR]]
    for a in order:
        trows.append([a] + [("DNF" if pd.isna(piv.loc[a, t]) else f"{piv.loc[a, t]:.2f}") for t in TR])
    s += [table(trows, [4.0 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm]),
          P("Table 4. Solo time trials. The oval is never seen by the learning agent in training.", CAP)]
    s += [figure(FIG / "time_trials.png", 17, "Figure 7. Lap times per agent and track.")]

    s += [P("3.5 Conclusions", H2)]
    s += bullets([
        "<b>Admissible heuristics (h1, h3, h4) always returned the optimal path.</b> Only the search effort differs. "
        f"<b>Manhattan (h2) is inadmissible</b> on an 8-connected grid. It was optimal in only {rp.loc['h2', 'optimal_pct']:.0f}% "
        f"of the random queries, and because it is inconsistent it re-opens nodes: up to "
        f"{100 * max(lp(t, 'h2', 'expanded_vs_dijkstra') for t in TR):.0f}% of Dijkstra's expansions on a lap.",
        f"<b>Geometry alone is a poor guide on a race track.</b> On a full lap Euclidean and Octile expand "
        f"{100 * lp('serpentine', 'h1', 'expanded_vs_dijkstra'):.0f}% and {100 * lp('serpentine', 'h3', 'expanded_vs_dijkstra'):.0f}% "
        f"of Dijkstra's nodes (mean h/h* ≈ {mi('serpentine', 'h1'):.2f}). On random queries they help (Octile "
        f"{100 * rp.loc['h3', 'vs_dijkstra']:.0f}%).",
        f"<b>ALT (h4) is the clear winner.</b> Mean h/h* ≈ {mi('serpentine', 'h4'):.2f}; it expands "
        f"{100 * min(lp(t, 'h4', 'expanded_vs_dijkstra') for t in TR):.0f}–{100 * max(lp(t, 'h4', 'expanded_vs_dijkstra') for t in TR):.0f}% "
        f"of Dijkstra's nodes on a lap and {100 * rp.loc['h4', 'vs_dijkstra']:.0f}% on random queries, because its "
        "landmark distances already know the walls and the shape of the loop.",
        "<b>On the road the heuristic hardly matters.</b> Admissible heuristics give the same optimal line, so the A* cars "
        "post near-identical lap times. The heuristic is a planning-time choice.",
        "<b>Better AI than AI?</b> The utility agent is the fastest hand-made AI. The learned GA driver, with no rules "
        "and no map, is second-fastest on every track and generalises to the unseen oval. The A* cars are the most "
        "predictable but conservative (centred safety line), and the reflex agent is the simplest and slowest."])

    # ------------------------------------------------------------------ 4 deliverable 3 + demos
    s += [PageBreak(), P("4. Deliverable 3: progress report", H1), P(
        "<font face='DVM'>report/progress_report.pdf</font> is the required short report: two A4 pages. It describes the "
        "implementation of the reflex module and the path-planning module, explains the four heuristics with their "
        "admissibility arguments, and gives the comparison (full-lap and random-query tables, the expansion figure, and "
        "the time-trial table) with a discussion. Its sections:")]
    s += bullets(["1 Project and agent design (PEAS, the four agent types)",
                  "2 Reflex module (propositional rules R1–R11 and safety rules S1–S3)",
                  "3 Path-planning module (grid, costs, full-lap barrier, racing line, speed profile)",
                  "4 The four heuristics (definitions, admissibility, consistency)",
                  "5 Heuristic comparison (Tables 1–2, Figure 1) and discussion",
                  "6 Driving results: who is the better AI? (Table 3)",
                  "7 Testing and deliverables"])
    s += [P("5. Demos", H1)]
    s += [table([["File", "Content"],
                 ["demo/race_serpentine.mp4", "Full race of all 7 cars on the proposal track"],
                 ["demo/race_technical.mp4", "Full race on the technical circuit"],
                 ["demo/learning_serpentine.mp4", "The genetic algorithm learning to drive, generation by generation"],
                 ["demo/*.png, demo/editor.png", "Screenshots of the race view, the training view and the editor"]],
                [6.0 * cm, 11.8 * cm])]
    s += [P("6. Submission contents", H1)]
    s += [code("main.py                  command-line entry point\n"
               "racing/                  all source code (track, car, logic, planner, agents, race,\n"
               "                         learning, gui, view, editor, experiments)\n"
               "tracks/                  serpentine, technical, oval (+ painted tracks)\n"
               "models/ga_best.json      trained learning agent\n"
               "tests/                   pytest suite\n"
               "notebooks/               Deliverable 2 (+ build_notebook.py)\n"
               "report/progress_report.pdf  Deliverable 3\n"
               "report/final_report.pdf  this report\n"
               "demo/                    videos and screenshots\n"
               "requirements.txt, README.md")]
    s += [PageBreak(), P("Appendix: the approved proposal", H1),
          figure(ROOT / "my-idea.jpeg", 9.5, "The hand-written project proposal that this implementation follows.",
                 max_h_cm=12.5)]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                            title="Car Racing Agent - Final report on all deliverables", author="Vedant Solanki")
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
