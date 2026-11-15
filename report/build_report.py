"""Builds report/progress_report.pdf (Deliverable 3) from the notebook outputs.

    python notebooks/build_notebook.py     # produces report/figures/*.csv|png
    python report/build_report.py
"""
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(_FD / f)))
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B"); addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-BI")

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "progress_report.pdf"
BODY = ParagraphStyle("b", fontName="DV", fontSize=8.5, leading=10.5, alignment=TA_JUSTIFY, spaceAfter=3)
H1 = ParagraphStyle("h1", fontName="DV-B", fontSize=14, leading=17, alignment=1, spaceAfter=1)
SUB = ParagraphStyle("s", parent=BODY, alignment=1, textColor=colors.HexColor("#52514e"))
H2 = ParagraphStyle("h2", fontName="DV-B", fontSize=10, leading=12, spaceBefore=4, spaceAfter=2,
                    textColor=colors.HexColor("#184f95"))
CELL = ParagraphStyle("c", fontName="DV", fontSize=7.4, leading=8.8)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.4, leading=9, textColor=colors.HexColor("#52514e"))


def P(t, st=BODY):
    return Paragraph(t, st)


def table(rows, widths):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c3c2b7")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0fb")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


def img(name, width_cm):
    w, h = PILImage.open(FIG / name).size
    return Image(str(FIG / name), width=width_cm * cm, height=width_cm * cm * h / w)


def main():
    lap = pd.read_csv(FIG / "lap_search.csv")
    rp = pd.read_csv(FIG / "random_pairs_summary.csv", index_col=0)
    inf = pd.read_csv(FIG / "informedness.csv")
    tt = pd.read_csv(FIG / "time_trials.csv")
    sw = pd.read_csv(FIG / "alt_sweep.csv")
    names = {"serpentine": "Serp.", "technical": "Tech.", "oval": "Oval"}
    HN = {"h0": "Dijkstra", "h1": "Euclidean", "h2": "Manhattan", "h3": "Octile", "h4": "ALT (8 landmarks)"}

    def lp(track, h, col):
        return lap[(lap.track == track) & (lap.heuristic == h)][col].iloc[0]

    def mi(track, h):
        r = inf[(inf.track == track) & (inf.heuristic == h)]
        return float(r.mean_h_over_hstar.iloc[0]) if len(r) else 0.0

    s = []
    s += [P("Car Racing Agent: Reflex, Planning, Utility and Learning (Part 1 Progress Report)", H1),
          P("CSE518 Artificial Intelligence · Monsoon 2026 · Vedant Solanki (AU2340012)", SUB), Spacer(1, 3)]

    s += [P("1. Project and agent design", H2), P(
        "The approved project is a <b>car racing agent</b>: several AI cars race one lap on a winding 2-D track "
        "(the track drawn in the proposal, plus an oval and a technical circuit loaded from track files), from the "
        "start line to the <b>finish line</b>. <b>Environment</b>: a top-down track with walls and rocks; continuous "
        "kinematic car physics at 30 Hz (acceleration, braking, drag, speed-dependent steering grip). <b>State</b>: "
        "position (x, y), direction (heading), speed / velocity (vx, vy) and five <b>sensor distances</b>. "
        "<b>Sensors</b>: rays w1–w5 at −90°, −45°, 0°, +45°, +90° relative to the heading, each measuring the free "
        "distance to the nearest wall or obstacle (as in the proposal sketch). <b>Actions</b>: every tick, steer "
        "LEFT / STRAIGHT / RIGHT and throttle ACCELERATE / COAST / BRAKE (9 combinations). <b>Goal</b>: finish the "
        "lap as fast as possible without touching a wall. Touching one means the car is out. Four agent types race each other:")]
    s += [table([["Agent", "Type", "How it decides"],
                 ["Reflex", "simple reflex", "percepts → propositions → prioritized rule base (propositional logic)"],
                 ["A* h1–h4", "deliberative + reflex safety", "A* plans the whole lap; pure-pursuit path following "
                  "with a curvature speed profile; safety rules override (\"Rules: safety / follow path\")"],
                 ["Utility", "model-based utility", "simulates all 9 actions 10 ticks ahead; maximises "
                  "U = progress + 0.15·speed − 6·Σ w<sub>i</sub>e<super>−ray<sub>i</sub>/18</super> − overspeed − crash"],
                 ["Learned (GA)", "learning agent", "7-10-2 neural net (rays, speed → steer, throttle) whose 102 weights "
                  "are evolved by a genetic algorithm over repeated runs"]],
                [2.2 * cm, 3.6 * cm, 12.1 * cm]), Spacer(1, 3)]

    s += [P("2. Reflex module (propositional rules)", H2), P(
        "Raw readings become boolean atoms, e.g. <i>FrontBlocked</i> ⇔ front ray &lt; braking distance "
        "v²/(2·0.85·a<sub>brake</sub>) + 22 px, <i>FrontNear</i>, <i>LeftMoreOpen</i> ⇔ w1 + 1.4·w2 &gt; w5 + 1.4·w4, "
        "<i>LeftTight</i>/<i>RightTight</i> (side ray &lt; 13 px), <i>DiagLeftTight</i>, <i>TooFast</i>, <i>Slow</i>, "
        "<i>VeerLeft</i>/<i>VeerRight</i>. A small rule engine (<font face='DVM'>racing/logic.py</font>) fires the first "
        "rule whose conjunction of literals holds: R1 FrontBlocked ∧ ¬Slow → turn to the open side + BRAKE; "
        "R2 FrontBlocked ∧ Slow → turn + crawl; R3/R4 DiagLeftTight ∧ ¬DiagRightTight → RIGHT (and mirror); "
        "R5/R6 side wall → steer away; R7 FrontNear ∧ TooFast → steer + BRAKE; R8 FrontNear ∧ ¬TooFast → steer + COAST; "
        "R9/R10 VeerLeft/VeerRight → follow the bend; R11 ⊤ → STRAIGHT + ACCELERATE. The A* car uses the same atoms "
        "for its safety layer: S1 FrontBlocked ∧ ¬Slow ∧ OffLine → emergency brake/evade, S2/S3 side wall → evade, "
        "P1 ⊤ → follow path.")]

    s += [P("3. Path-planning module (A*)", H2), P(
        "The track is rasterised into 8 px cells (90×150). A cell is free if the clearance to the wall is ≥ 13 px "
        "(the car fits: configuration space). The grid is 8-connected, with no corner cutting. A move costs length × "
        "(p(a)+p(b))/2 with wall penalty p = 1 + 1.5·(1 − clearance/half-width) ≥ 1, so the line keeps to the middle "
        "of the road and <b>every cost ≥ its Euclidean length</b>. Start and goal are on the start line, and a barrier "
        "of cells across the line forces A* to plan the <b>full lap</b> (no backward shortcut). A* uses a binary heap, "
        "ties broken towards larger g, and re-opening of closed nodes (needed for inadmissible h). The cell path is "
        "resampled at 6 px and smoothed into a racing line with curvature κ. Each point gets a corner speed "
        "v ≤ ω(v)/κ, and braking is planned ahead: v<sub>lim</sub> = min<sub>j</sub> √(v<sub>c,j</sub>² + "
        "2·a<sub>brake</sub>·d<sub>j</sub>) over the next 320 px.")]

    s += [P("4. The four heuristics", H2), P(
        "<b>h1 Euclidean</b> √(dx²+dy²): admissible and consistent, since costs ≥ length. "
        "<b>h2 Manhattan</b> |dx|+|dy|: <i>not admissible</i> on an 8-grid, because a diagonal costs √2 &lt; 2. "
        "<b>h3 Octile</b> max(dx,dy) + (√2−1)·min(dx,dy): the exact shortest distance on an obstacle-free 8-grid, so it is admissible, "
        "consistent, and dominates h1. <b>h4 ALT</b>: 8 landmarks chosen by farthest-point sampling, with exact distance "
        "tables from Dijkstra (scipy). h(n) = max<sub>L</sub>|d(L,n) − d(L,g)| ≤ d(n,g) by the triangle inequality "
        "(the graph is undirected), so it is admissible and consistent and it <i>knows walls and the shape of the track</i>. "
        "h0 = 0 (Dijkstra) is the baseline. Admissibility and consistency were checked on every cell and edge.")]

    rows = [["h", "heuristic"] + [f"{names[t]} expanded" for t in names] + ["Serp. ms", "Serp. cost gap", "h/h* (serp.)"]]
    for h in ["h0", "h1", "h2", "h3", "h4"]:
        rows.append([h, HN[h]] + [f"{int(lp(t, h, 'expanded'))} ({100 * lp(t, h, 'expanded_vs_dijkstra'):.0f}%)" for t in names]
                    + [f"{lp('serpentine', h, 'ms'):.1f}", f"{lp('serpentine', h, 'gap_pct'):.2f}%",
                       "-" if h == "h0" else f"{mi('serpentine', h):.2f}"])
    rrows = [["h", "mean expanded", "vs Dijkstra", "mean ms", "optimal", "max gap"]]
    for h, r in rp.iterrows():
        rrows.append([h, f"{r.mean_expanded:.0f}", f"{100 * r.vs_dijkstra:.0f}%", f"{r.mean_ms:.2f}",
                      f"{r.optimal_pct:.0f}%", f"{r.max_gap_pct:.1f}%"])
    s += [P("5. Heuristic comparison", H2),
          P("<b>Table 1.</b> Full-lap query on each track (expanded nodes, % of Dijkstra) and the serpentine details.", CAP),
          table(rows, [0.6 * cm, 2.8 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 1.6 * cm, 2.2 * cm, 1.9 * cm]), Spacer(1, 2),
          P("<b>Table 2.</b> 360 random start/goal queries (120 per track).", CAP),
          table(rrows, [0.8 * cm, 2.6 * cm, 2.2 * cm, 2 * cm, 1.8 * cm, 1.8 * cm]), Spacer(1, 3),
          img("closed_sets.png", 16.5),
          P("<b>Figure 1.</b> Cells expanded (blue) by each heuristic for the full-lap query on the proposal track. "
            "Red = path. The last panel shows the ALT value h4 per cell.", CAP)]

    best_alt = int(sw.expanded.min())
    s += [P(
        f"<b>Discussion.</b> All admissible heuristics returned the optimal cost in every query. On the <b>full lap</b> "
        f"the goal lies just behind the start, so Euclidean and Octile think the goal is close almost everywhere "
        f"(h/h* ≈ {mi('serpentine', 'h1'):.2f}). They expand nearly as many cells as Dijkstra. Manhattan finds a "
        f"{lp('serpentine', 'h2', 'gap_pct'):.1f}% costlier lap, and because it is inconsistent it <b>re-opens</b> nodes: "
        f"{int(lp('serpentine', 'h2', 'expanded'))} expansions on the serpentine, more than Dijkstra. <b>ALT is the clear winner</b> "
        f"(h/h* ≈ {mi('serpentine', 'h4'):.2f}): {100 * lp('serpentine', 'h4', 'expanded_vs_dijkstra'):.0f}% of Dijkstra's "
        f"expansions on the lap and {100 * rp.loc['h4', 'vs_dijkstra']:.0f}% on random queries, with a one-off "
        f"pre-computation of {lp('serpentine', 'h4', 'alt_precompute_ms'):.0f} ms. A landmark sweep shows that even one farthest-point landmark "
        f"captures most of the gain on a loop track ({int(sw[sw.landmarks == 1].expanded.iloc[0])} expansions vs "
        f"{int(lp('serpentine', 'h0', 'expanded'))} for Dijkstra), and 16 landmarks only reach {best_alt}. On random queries the geometric heuristics are useful "
        f"(Octile {100 * rp.loc['h3', 'vs_dijkstra']:.0f}% of Dijkstra), while inadmissible Manhattan is both slower "
        f"({100 * rp.loc['h2', 'vs_dijkstra']:.0f}% of Dijkstra, due to re-openings) and optimal in only "
        f"{rp.loc['h2', 'optimal_pct']:.0f}% of the queries.")]

    piv = tt.pivot_table(index="agent", columns="track", values="lap_time")
    order = [a for a in ["A* (h1)", "A* (h2)", "A* (h3)", "A* (h4)", "Utility", "Reflex", "Learned (GA)"] if a in piv.index]
    trows = [["agent"] + [names[t] for t in names]]
    for a in order:
        trows.append([a] + [("DNF" if pd.isna(piv.loc[a, t]) else f"{piv.loc[a, t]:.2f} s") for t in names])
    s += [P("6. Driving results: who is the better AI?", H2),
          P("<b>Table 3.</b> Solo time trials (lap time). The oval is never seen by the GA in training.", CAP),
          table(trows, [3 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm]), Spacer(1, 3), P(
        "The heuristic does not change how the car drives: admissible heuristics yield the same optimal lap, so the "
        "A* cars post near-identical times. It only changes planning effort. The <b>utility agent</b> is the fastest "
        "hand-made AI, because its look-ahead trades progress, speed and collision risk every tick. The <b>learned GA "
        "driver</b> has no rules and no map, yet it learns to finish every track, including the unseen oval, and is the "
        "<b>second-fastest</b> car on every track, beating the A* and reflex cars. The <b>reflex agent</b> is the simplest and slowest. The <b>A*</b> cars are the "
        "most predictable, but conservative, because the cost function keeps them centred.")]

    s += [P("7. Testing and deliverables", H2), P(
        "A pytest suite (<font face='DVM'>tests/</font>) covers: track parsing and validity (no self-overlap, rocks not drivable), "
        "physics (acceleration, braking limits, steering sign, grip), vectorised vs single raycasts, the logic "
        "engine, one test per reflex rule (R1–R11), A* path validity and optimality against Dijkstra (full laps and random "
        "pairs), admissibility and consistency of h1/h3/h4 on every cell, Manhattan's over-estimation, the start-line barrier, "
        "racing-line clearance, every agent finishing every track, crash, stuck and timeout detection, the GA operators, "
        "and learning progress. <b>Code</b>: <font face='DVM'>racing/</font> (track, car, logic, planner, agents, race, "
        "learning, gui, editor) and <font face='DVM'>main.py gui | editor | race | train | compare | record</font>. The pygame viewer "
        "shows all cars racing, the selected car's five rays, its planned A* line, the rule that fired and its utilities, "
        "and a live <i>training mode</i> where a population of cars learns to drive generation by generation. A "
        "<b>track editor</b> lets you paint new tracks (road, grass, rocks, start line). Lap progress on a painted track "
        "is computed geodesically (Dijkstra from the start line), so every agent can race on it immediately.")]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.1 * cm, bottomMargin=1.1 * cm,
                            title="Car Racing Agent - Part 1 Progress Report", author="Vedant Solanki")
    doc.build(s)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
