"""Builds report/progress_report.pdf (Deliverable 3) from the notebook outputs.

Run the notebook first (python notebooks/build_notebook.py), then:
    python report/build_report.py
"""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

import matplotlib
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

_FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(_FD / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_FD / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(_FD / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", str(_FD / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", str(_FD / "DejaVuSansMono.ttf")))
addMapping("DejaVu", 0, 0, "DejaVu")
addMapping("DejaVu", 1, 0, "DejaVu-Bold")
addMapping("DejaVu", 0, 1, "DejaVu-Oblique")
addMapping("DejaVu", 1, 1, "DejaVu-BoldOblique")

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "progress_report.pdf"

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="DejaVu", fontSize=8.6,
                      leading=10.6, alignment=TA_JUSTIFY, spaceAfter=3)
H1 = ParagraphStyle("h1", parent=ss["Title"], fontName="DejaVu-Bold", fontSize=14.5,
                    leading=17, spaceAfter=1)
SUB = ParagraphStyle("sub", parent=BODY, alignment=1, fontSize=8.6, textColor=colors.HexColor("#52514e"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="DejaVu-Bold", fontSize=10,
                    leading=12, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#184f95"))
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.6, leading=9, alignment=0, spaceAfter=0)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.6, leading=9, textColor=colors.HexColor("#52514e"))


def P(t, st=BODY):
    return Paragraph(t, st)


def table(rows, widths, header=True, font=7.6):
    data = [[c if not isinstance(c, str) else Paragraph(c, CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c3c2b7")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0fb"))]
    t.setStyle(TableStyle(st))
    return t


def main():
    A = pd.read_csv(FIG / "expA_main_summary.csv", index_col=0)
    T = pd.read_csv(FIG / "expA_track_summary.csv", index_col=0)
    B = pd.read_csv(FIG / "expB_summary.csv", index_col=0)
    M = pd.read_csv(FIG / "expE_missions.csv")
    I = pd.read_csv(FIG / "informedness.csv")
    inf = I[I["map"] == "map_main"].set_index("heuristic")["mean_h_over_hstar"]

    mm = M[(M["map"] == "map_main")].set_index(["mode", "heuristic"])
    s_h4 = mm.loc[("survey", "h4")]
    r_h4 = mm.loc[("race", "h4")]

    story = []
    story += [P("Car Racing Agent: Reactive–Deliberative Control (Part 1 Progress Report)", H1),
              P("CSE518 Artificial Intelligence · Monsoon 2026 · Vedant Solanki (AU2340012)", SUB),
              Spacer(1, 4)]

    story += [P("1. Problem and agent design", H2), P(
        "The project implements an autonomous <b>racing car</b> that must drive from the start/finish line "
        "<b>H</b> through all checkpoints <b>1–9</b> and back to H on a 4-connected grid track, while managing "
        "fuel. It is the AUV survey problem from the problem statement, re-themed: open water → asphalt (cost 2), "
        "kelp → grass/gravel (cost 6), reef → wall (impassable), charging dock → pit stop <b>C</b>, battery → "
        "fuel, ocean current → slope/tailwind lanes. Entering a tile <i>with</i> its current costs half, "
        "<i>against</i> it double, across it normal, so <b>costs are asymmetric</b> (c(a,b) ≠ c(b,a)). The parser "
        "reads the course format unchanged (<font face='DejaVuMono'>map_main.txt</font>, plus aliases S/P/FUEL), so "
        "the code runs on any map in that format. <b>PEAS</b>: Performance = all checkpoints cleared, back at H, "
        "minimum fuel, never stranded. Environment = track grid with terrain, currents, pits. Actuators = "
        "N/E/S/W moves (shown as LEFT/RIGHT/STRAIGHT/U-TURN + ACCELERATE/CRUISE/BRAKE relative to the heading), "
        "refuel. Sensors = tile terrain and current, fuel level, five distance rays (w1–w5 at −90°…+90°), "
        "cost-to-nearest-pit. Two tour modes exist: <i>survey</i> (any order, as in the problem statement) and <i>race</i> "
        "(checkpoints in order 1→n).")]

    story += [P("2. Reflex module", H2), P(
        "Every tick the agent <b>senses → decides → acts</b>. Decisions come from a prioritized table of "
        "propositional condition→action rules. The first rule whose formula holds fires. R1–R4 are the four "
        "rules of the problem statement. A <b>safety map</b> <i>ctp(p)</i> (cheapest cost from p to any pit) is "
        "pre-computed with one multi-source Dijkstra on the <i>reversed</i> graph, so the check \"after the move "
        "I can still reach a charger\" is an O(1) test: <i>fuel − c(cur,nxt) ≥ ctp(nxt)</i>.")]
    rules = [["#", "Condition (priority order)", "Action"],
             ["R0", "AllDone ∧ AtHome", "FINISH"],
             ["R6", "AtCheckpoint(k) ∧ ¬Visited(k) (race: k must be the next in order)", "mark k cleared"],
             ["R1", "AtPit ∧ ¬FullTank ∧ (HeadingToThisPit ∨ (¬HeadingToPit ∧ Fuel≤40%))",
              "refuel to full, re-plan remaining tour"],
             ["R2", "Fuel &lt; 25% ∧ ¬HeadingToPit", "A* to cheapest pit"],
             ["R4", "25% ≤ Fuel ≤ 40% ∧ ctp ≤ 4 ∧ ¬HeadingToPit", "override goals, A* to cheapest pit"],
             ["R5", "NextTile = Wall", "reactive guard: re-plan"],
             ["R3", "¬(fuel − c(next) ≥ ctp(next)) ∧ ¬HeadingToPit", "cancel move, A* to cheapest pit"],
             ["R7", "⊤", "follow plan (one tile)"]]
    story += [table(rules, [0.8 * cm, 11.2 * cm, 5.9 * cm]), Spacer(1, 3)]
    story += [P(
        "<b>Loop-freedom.</b> Taken literally, the rules can trap a car between a pit and a far checkpoint "
        "(refuel → drive → R2/R3 fires → return → refuel …). After each refuel the agent therefore <i>simulates "
        "the reflex rules</i> along the first leg of the new plan. If a rule would interrupt it, the agent plans "
        "a <b>relay</b>: the cheapest chain of pit hops (each ≤ one full tank, hence always safe) to a "
        "<i>staging pit</i> from which the leg provably completes. Checkpoints that no pit chain can serve are "
        "reported and skipped instead of looping. Since fuel ≥ ctp(pos) is an invariant, the car can never be "
        "stranded.")]

    story += [P("3. Path-planning module", H2), P(
        "<b>A*</b> (binary heap, closed set with re-opening, ties broken towards larger g) returns path, cost, "
        "nodes expanded/generated, frontier size and time. <b>Tour planning</b> computes A* legs between "
        "every ordered pair of {current position, remaining checkpoints, H} (cached), then solves the "
        "<b>asymmetric TSP exactly with Held-Karp DP</b>, O(2<super>n</super>n<super>2</super>), instant for n ≤ 9. "
        "This was verified against brute force in the tests. The tour ignores fuel. The reflex layer inserts pit stops and "
        f"triggers a re-plan from the pit. On <font face='DejaVuMono'>map_main.txt</font> (h4, survey) the car "
        f"clears all 5 checkpoints in {int(s_h4.moves)} moves with {int(s_h4.pit_stops)} pit stops, using "
        f"{s_h4.fuel_used:g} fuel. The minimum fuel ever is 0 at a pit, with 0 safety violations.")]

    story += [P("4. The four heuristics", H2), P(
        "Let c<sub>min</sub> be the cheapest single move on the map (1 with a tailwind, else 2). "
        "<b>h1 Manhattan × c<sub>min</sub></b>: at least |Δr|+|Δc| moves remain, each ≥ c<sub>min</sub>. "
        "<b>h2 Euclidean × c<sub>min</sub></b>: ≤ h1 on a 4-grid, so it is admissible but weaker. "
        "<b>h3 Row/column corridor bound</b> (domain-specific): for every row r we pre-compute the cheapest "
        "move that <i>enters</i> r going S (and going N), and similarly for columns going E/W, from terrain and "
        "currents. Any path from row r<sub>1</sub> to r<sub>2</sub> must enter each row in between with a "
        "distinct vertical move, and the same holds for columns, so the sum of the cheapest entries (prefix sums, O(1)) is a "
        "lower bound. It is ≥ h1 everywhere. "
        "<b>h4 ALT landmarks</b>: 6 landmarks (H, pits, then farthest-point picks), with exact d(L→·) and "
        "d(·→L) from 2 Dijkstra runs each. For a directed graph, h(n) = max<sub>L</sub> {d(n,L) − d(g,L), "
        "d(L,g) − d(L,n)} by the triangle inequality, so walls and asymmetric currents are captured. "
        "All four are <b>admissible and consistent</b>: each per-move change of h is at most the move's cost. "
        "This was also checked empirically on every tile of every map (0 violations). As a baseline we use h0 = 0 (Dijkstra). "
        "As a bonus we add h5 = 2·h3 (weighted A*, inadmissible).")]

    def row(df, h):
        r = df.loc[h]
        return [h, r["description"].replace(" (baseline)", "").replace(", inadmissible", ""),
                f"{int(r.total_expanded)}", f"{100 * r.expanded_vs_dijkstra:.0f}%", f"{r.mean_ms:.2f}",
                f"{r.mean_ebf:.3f}", f"{r.optimal_pct:.0f}%", f"{r.max_gap_pct:.0f}%"]
    hdr = ["h", "heuristic", "expanded", "% of h0", "ms", "b*", "optimal", "max gap"]
    rows = [hdr] + [row(A, h) for h in A.index]
    rowsB = [hdr] + [row(B, h) for h in B.index]
    w = [0.7 * cm, 4.4 * cm, 1.8 * cm, 1.5 * cm, 1.1 * cm, 1.2 * cm, 1.5 * cm, 1.6 * cm]

    story += [P("5. Comparison results", H2), P(
        f"Deliverable 2 (<font face='DejaVuMono'>notebooks/heuristic_comparison.ipynb</font>) runs A* for "
        f"<b>every ordered pair</b> of key points (H, checkpoints, pits) on the course map "
        f"({int(A.loc['h0', 'searches'])} searches), on the racing track ({int(T.loc['h0', 'searches'])}), and "
        f"on <b>60 random maps</b> of 12×16, 20×30 and 30×40 cells ({int(B.loc['h0', 'searches'])} searches). "
        "It also measures informedness h/h*, draws expansion heat-maps and runs full missions. "
        "b* = effective branching factor.")]
    story += [KeepTogether([P("<b>Table 1.</b> map_main.txt, all ordered pairs.", CAP),
                            table(rows, w), Spacer(1, 3),
                            P("<b>Table 2.</b> 60 random maps.", CAP), table(rowsB, w)])]
    story += [Spacer(1, 4)]
    from PIL import Image as PILImage
    iw, ih = PILImage.open(FIG / "expC_heatmaps.png").size
    img = Image(str(FIG / "expC_heatmaps.png"), width=15.2 * cm, height=15.2 * cm * ih / iw)
    story += [img, P("<b>Figure 1.</b> Cells expanded (blue) for H → checkpoint 2 on map_main. "
                     "Every admissible heuristic finds the same optimal path (red), but explores very different areas.", CAP)]

    story += [P("6. Discussion", H2), P(
        f"<b>All admissible heuristics return the optimal cost.</b> Only the effort differs. The ranking is "
        f"consistent on every map set: <b>h4 &lt; h3 &lt; h1 &lt; h2 &lt; h0</b>. Euclidean (h2) is the weakest "
        f"informed heuristic because on a 4-connected grid it never exceeds Manhattan. h3 improves on h1 a little "
        f"on the open course map ({100 * A.loc['h3', 'expanded_vs_dijkstra']:.0f}% vs "
        f"{100 * A.loc['h1', 'expanded_vs_dijkstra']:.0f}% of Dijkstra's expansions) and a lot on cluttered random maps "
        f"({100 * B.loc['h3', 'expanded_vs_dijkstra']:.0f}% vs {100 * B.loc['h1', 'expanded_vs_dijkstra']:.0f}%), "
        f"because whole rows/columns become expensive to enter. <b>h4 (ALT) is the clear winner</b>: mean h/h* = "
        f"{inf['h4']:.2f} (h1 {inf['h1']:.2f}, h3 {inf['h3']:.2f}). It expands only "
        f"{100 * A.loc['h4', 'expanded_vs_dijkstra']:.0f}% (course map) and {100 * B.loc['h4', 'expanded_vs_dijkstra']:.0f}% "
        f"(random maps) of Dijkstra's nodes, and the gap grows with map size. Its one-off pre-computation (a few ms) is "
        f"repaid because the tour planner runs dozens of searches per map. The weighted h5 is as fast as ALT but "
        f"is optimal in only {B.loc['h5', 'optimal_pct']:.0f}% of searches (worst gap {B.loc['h5', 'max_gap_pct']:.0f}%). "
        f"<b>Mission-level finding:</b> the fuel-agnostic optimal tour is not fuel-optimal. On map_main, survey "
        f"order (planned {s_h4.planned_tour_cost:g}) actually burns {s_h4.fuel_used:g} fuel, while race order 1→5 "
        f"(planned {r_h4.planned_tour_cost:g}) burns only {r_h4.fuel_used:g}, because pit-stop detours are added afterwards. "
        "A fuel-aware tour cost is the main improvement planned for Part 2.")]

    story += [P("7. Testing and deliverables", H2), P(
        "<b>136 pytest tests</b> (all passing) cover: the parser and malformed maps; with/against/across costs and "
        "asymmetry; A* path validity and optimality against Dijkstra on 25 random maps for every admissible "
        "heuristic; admissibility and consistency on every tile; Held-Karp = brute force; one scenario per "
        "reflex rule incl. threshold edges (exactly 25 %, 40 %); the safety map vs per-tile A*; wall-guard re-plan; "
        "relay pit-hopping; infeasible/unreachable checkpoints skipped (no infinite loop); and full missions on "
        "every map × mode × heuristic plus 40 random maps, replaying the log to check that fuel never goes below 0 and that refuels happen only at pits. "
        "<b>Code</b>: <font face='DejaVuMono'>src/</font> (world, search, heuristics, tour, safety, reflex, agent, "
        "sensors, visualize), with <font face='DejaVuMono'>main.py run|gui|record|compare|plan</font>. A pygame "
        "viewer animates the car with its five ray sensors, the planned A* route, the fuel bar and the active rule, and records "
        "GIF demos.")]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm,
                            title="Car Racing Agent – Part 1 Progress Report", author="Vedant Solanki")
    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
