"""Builds (and optionally executes) heuristic_comparison.ipynb.

    python notebooks/build_notebook.py            # build + execute in place
    python notebooks/build_notebook.py --no-exec  # only build
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "heuristic_comparison.ipynb"

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Car Racing Agent: A* heuristic comparison (Deliverable 2)

**CSE518 Artificial Intelligence, Project Part 1**

This notebook compares the heuristics used by the A* path planner of the car racing agent.
The car drives on a 4-connected grid. The cost of a move depends on the tile being entered
(asphalt 2, grass 6, wall impassable) **and on the direction of travel**. Moving with a
slope/current costs half, moving against it costs double. Costs are therefore **asymmetric**.

| id | heuristic | idea |
|----|-----------|------|
| h0 | Zero (Dijkstra) | baseline, reference optimal costs |
| **h1** | Manhattan × c_min | every remaining move costs at least the cheapest move on the map |
| **h2** | Euclidean × c_min | straight-line distance, weaker than Manhattan on a 4-grid |
| **h3** | Row/column corridor bound | cheapest way to *enter* every row/column between n and the goal, using terrain and currents |
| **h4** | ALT landmarks | triangle-inequality bounds from exact distances to/from 6 landmarks (directed version) |
| h5 | Weighted A* (2·h3) | bonus: inadmissible, shows the speed/optimality trade-off |

h1–h4 are the four heuristics we designed. h0 is the reference and h5 is an extra.

**Contents**
1. Setup and maps
2. Admissibility and consistency check (empirical)
3. Experiment A: all ordered pairs on the course map and the racing track
4. Experiment B: 60 random maps in three sizes
5. Experiment C: where does A* search? (expansion heat maps)
6. Experiment D: informedness h/h*
7. Experiment E: full missions (reflex + A*)
8. Conclusions
""")

code(r"""
import sys, time, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch

from src.world import load_map, WALL, GRASS
from src.search import astar
from src.heuristics import HEURISTIC_INFO
from src import experiments as X

FIG = ROOT / "report" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b5b3ad", "axes.grid": True, "grid.color": "#e6e4de",
    "grid.linewidth": 0.7, "axes.axisbelow": True, "axes.titleweight": "bold",
    "axes.titlesize": 11, "legend.frameon": False,
})
pd.set_option("display.width", 160, "display.max_columns", 30)
COL = X.PALETTE
main = load_map(ROOT / "maps/map_main.txt")
track = load_map(ROOT / "maps/track_main.txt")
for g in (main, track):
    print(f"{g.name:11s} {g.rows}x{g.cols}  capacity={g.capacity}  checkpoints={list(g.sites)}  "
          f"refuel points={g.pits}  c_min={g.min_step_cost}")
""")

md("## 1. Maps\nGray = asphalt (2), green = grass (6), black = wall, arrows = slope/current.")

code(r"""
def draw_map(g, ax, path=None, title=None, overlay=None, cmap=None):
    img = np.ones((g.rows, g.cols, 3))
    for r in range(g.rows):
        for c in range(g.cols):
            t = g.terrain[r][c]
            img[r, c] = (0.17,0.17,0.2) if t == WALL else (0.55,0.75,0.5) if t == GRASS else (0.86,0.86,0.86)
    ax.imshow(img, interpolation="nearest")
    if overlay is not None:
        ax.imshow(overlay, cmap=cmap, interpolation="nearest", alpha=0.85)
    arrow = {"N":(0,-0.32),"S":(0,0.32),"E":(0.32,0),"W":(-0.32,0)}
    for r in range(g.rows):
        for c in range(g.cols):
            d = g.current[r][c]
            if d: ax.arrow(c-arrow[d][0]/2, r-arrow[d][1]/2, arrow[d][0], arrow[d][1],
                           head_width=0.18, length_includes_head=True, color="#256abf", lw=0.6)
    if path:
        ax.plot([p[1] for p in path], [p[0] for p in path], color="#e34948", lw=2)
    ax.scatter(*g.home[::-1], marker="s", s=120, c="#e34948", edgecolors="k", zorder=5)
    for P in g.pits[1:]:
        ax.scatter(P[1], P[0], marker="s", s=120, c="#eda100", edgecolors="k", zorder=5)
    for k, p in g.sites.items():
        ax.scatter(p[1], p[0], s=170, c="#6250d6", zorder=5)
        ax.text(p[1], p[0], str(k), color="w", ha="center", va="center", fontsize=8, weight="bold", zorder=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if title: ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
draw_map(main, axes[0], title="map_main.txt (course map, 15x20, fuel 50)")
draw_map(track, axes[1], title="track_main.txt (racing circuit, 16x26, fuel 70)")
plt.tight_layout(); plt.savefig(FIG / "maps.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 2. Admissibility and consistency check

For every tile *n* and several goals we compute the exact cost-to-go *h\*(n)* with a reverse
Dijkstra. Then we count how often **h(n) > h\*(n)** (admissibility violated) and how often
**h(n) > c(n,n') + h(n')** (consistency violated).
""")

code(r"""
rows = []
for g in (main, track):
    goals = list(X.key_points(g).values())[:6]
    inf = X.informedness(g)
    for h in ["h1","h2","h3","h4","h5"]:
        rows.append(dict(map=g.name, heuristic=h, description=HEURISTIC_INFO[h],
                         overestimates=int(inf.loc[h,"overestimates"]),
                         admissible=bool(inf.loc[h,"admissible"]),
                         consistency_violations=X.consistency_violations(g, h, goals)))
adm = pd.DataFrame(rows)
adm["consistent"] = adm.consistency_violations == 0
adm
""")

md(r"""
h1–h4 have **zero** violations on both maps, so they are admissible and consistent, and A* with any of
them returns optimal paths. h5 (2·h3) over-estimates on many tiles, so it is **not** admissible.

## 3. Experiment A: every ordered pair of key points
Key points = H, all checkpoints and all pit stops. Each ordered pair (A→B and B→A are different
because of the currents) is solved with every heuristic.
""")

code(r"""
dfA = pd.concat([X.pairwise_runs(main), X.pairwise_runs(track)], ignore_index=True)
sumA = {m: X.summarise(dfA[dfA["map"] == m]) for m in ["map_main", "track_main"]}
print("map_main"); display(sumA["map_main"])
print("track_main"); display(sumA["track_main"])
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
labels = X.HEURS
for ax, (metric, title) in zip(axes, [("total_expanded", "Nodes expanded (sum over all pairs)"),
                                       ("mean_ms", "Mean runtime per search (ms)"),
                                       ("max_gap_pct", "Worst cost gap vs optimal (%)")]):
    w = 0.38
    x = np.arange(len(labels))
    for j, m in enumerate(["map_main", "track_main"]):
        vals = sumA[m].loc[labels, metric].values
        bars = ax.bar(x + (j - 0.5) * w, vals, w * 0.94, color=[COL[h] for h in labels],
                      alpha=1.0 if j == 0 else 0.55, edgecolor="white", linewidth=1)
    ax.set_xticks(x, labels); ax.set_title(title)
axes[0].legend(handles=[Patch(color="#777", alpha=1.0, label="map_main (solid)"),
                        Patch(color="#777", alpha=0.55, label="track_main (light)")], loc="upper right")
plt.tight_layout(); plt.savefig(FIG / "expA_bars.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 4. Experiment B: 60 random maps (12×16, 20×30, 30×40)
Random terrain (18 % walls, 15 % grass), straight current lanes, 5 checkpoints and several pits.
""")

code(r"""
t0 = time.perf_counter()
suite = X.random_suite(60)
dfB = X.suite_runs(suite)
print(f"{len(dfB)} searches on {len(suite)} maps in {time.perf_counter()-t0:.1f}s")
sumB = X.summarise(dfB); sumB
""")

code(r"""
by_size = dfB.groupby(["size", "heuristic"]).expanded.mean().unstack()
by_size = by_size.loc[sorted(by_size.index, key=lambda s: int(s.split('x')[0])*int(s.split('x')[1]))]
rel = dfB.pivot_table(index=["map","start","goal"], columns="heuristic", values="expanded")
rel = rel.div(rel["h0"], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
bp = ax.boxplot([rel[h].values for h in X.HEURS], tick_labels=X.HEURS, patch_artist=True,
                showfliers=False, widths=0.55, medianprops=dict(color="#0b0b0b"))
for patch, h in zip(bp["boxes"], X.HEURS):
    patch.set_facecolor(COL[h]); patch.set_edgecolor("white")
ax.set_ylabel("expanded / expanded by Dijkstra"); ax.set_title("Search effort relative to Dijkstra (per search)")
ax.axhline(1, color="#b5b3ad", lw=1)

ax = axes[1]
xs = [int(s.split('x')[0])*int(s.split('x')[1]) for s in by_size.index]
for h in X.HEURS:
    ax.plot(xs, by_size[h].values, marker="o", ms=6, lw=2, color=COL[h], label=h)
ax.set_xticks(xs, [f"{s}\n({x} cells)" for s, x in zip(by_size.index, xs)])
ax.set_ylabel("mean nodes expanded"); ax.set_title("Scaling with map size")
ax.legend(ncol=3, loc="upper left")
plt.tight_layout(); plt.savefig(FIG / "expB_box_scaling.png", bbox_inches="tight"); plt.show()
by_size.round(1)
""")

md(r"""
## 5. Experiment C: where does each heuristic search?
Cells expanded by A* for the leg **H → checkpoint 2** on the course map. Darker = expanded.
The red line is the optimal path.
""")

code(r"""
start, goal = main.home, main.sites[2]
fig, axes = plt.subplots(2, 3, figsize=(14, 8.6))
cmap = ListedColormap(["#00000000", "#3987e5"])
for ax, h in zip(axes.ravel(), X.HEURS):
    r = astar(main, start, goal, h, keep_expanded=True)
    ov = np.zeros((main.rows, main.cols))
    for p in r.expanded: ov[p] = 1
    ov = np.ma.masked_where(ov == 0, ov)
    draw_map(main, ax, path=r.path, overlay=ov, cmap=ListedColormap(["#3987e5"]),
             title=f"{h}: {HEURISTIC_INFO[h]}\nexpanded={r.nodes_expanded}  cost={r.cost:g}")
plt.tight_layout(); plt.savefig(FIG / "expC_heatmaps.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 6. Experiment D: informedness
Mean ratio h(n)/h\*(n) over all tiles and several goals. 1.0 is a perfect heuristic. A higher
ratio means fewer expansions, as long as it never goes above 1 (admissibility).
""")

code(r"""
inf = pd.concat({g.name: X.informedness(g) for g in (main, track)}, names=["map"]).reset_index()
display(inf)
fig, ax = plt.subplots(figsize=(7, 3.4))
hs = ["h1","h2","h3","h4","h5"]; x = np.arange(len(hs)); w = 0.38
for j, m in enumerate(["map_main", "track_main"]):
    v = inf[inf["map"] == m].set_index("heuristic").loc[hs, "mean_h_over_hstar"]
    ax.bar(x + (j-0.5)*w, v, w*0.94, color=[COL[h] for h in hs], alpha=1 if j == 0 else 0.55, edgecolor="white")
ax.axhline(1, color="#e34948", lw=1, ls="--"); ax.text(4.45, 1.02, "h = h*", color="#52514e", fontsize=8, ha="right")
ax.set_xticks(x, hs); ax.set_ylabel("mean h / h*"); ax.set_title("Informedness (solid map_main, light track_main)")
plt.tight_layout(); plt.savefig(FIG / "expD_informedness.png", bbox_inches="tight"); plt.show()
""")

md(r"""
**Pre-computation cost of h4 (ALT).** The landmark tables (2 Dijkstra runs per landmark) are built
once per map and reused by every later search:
""")

code(r"""
pre = pd.DataFrame([dict(map=g.name, cells=g.rows*g.cols, alt_precompute_ms=round(X.alt_precompute_ms(g), 2))
                    for g in [main, track] + suite[:3]])
pre
""")

md(r"""
## 7. Experiment E: full missions (reflex module + tour planner + A*)
Each run uses a fresh copy of the map, so the ALT pre-computation is included in the time.
The agent applies all four fuel rules (R1–R4), re-plans after every pit stop and
returns to H.
""")

code(r"""
mis = X.mission_table({"map_main": main, "track_main": track})
mis
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, metric, title in [(axes[0], "nodes_expanded", "Nodes expanded during the whole mission"),
                          (axes[1], "fuel_used", "Fuel used for the whole mission")]:
    sub = mis[mis["mode"] == "survey"]
    x = np.arange(len(X.HEURS)); w = 0.38
    for j, m in enumerate(["map_main", "track_main"]):
        v = sub[sub["map"] == m].set_index("heuristic").loc[X.HEURS, metric]
        ax.bar(x + (j-0.5)*w, v, w*0.94, color=[COL[h] for h in X.HEURS], alpha=1 if j == 0 else 0.55, edgecolor="white")
    ax.set_xticks(x, X.HEURS); ax.set_title(title + " (survey mode)")
axes[0].legend(handles=[Patch(color="#777", label="map_main (solid)"), Patch(color="#777", alpha=0.55, label="track_main (light)")])
plt.tight_layout(); plt.savefig(FIG / "expE_missions.png", bbox_inches="tight"); plt.show()
""")

code(r"""
from src.agent import run_mission
r = run_mission(main, "h4", "survey")
fig, ax = plt.subplots(figsize=(8, 6))
draw_map(main, ax, path=r.path, title=f"Driven route on map_main (h4, survey): fuel used {r.fuel_used:g}, {r.pit_stops} pit stops")
plt.tight_layout(); plt.savefig(FIG / "route_main.png", bbox_inches="tight"); plt.show()
print(r.summary())
events = pd.DataFrame([dict(t=e.t, pos=e.pos, fuel=e.fuel, rule=e.rule, action=e.action, note=e.note)
                       for e in r.log if e.action != "MOVE"])
events
""")

code(r"""
# Fuel over time with refuel events (shows the 25% / 40% thresholds)
fuel = [main.capacity] + [e.fuel for e in r.log if e.action in ("MOVE", "REFUEL")]
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(fuel, color=COL["h0"], lw=2)
for th, lab in [(0.25, "25% (R2)"), (0.40, "40% (R4)")]:
    ax.axhline(th*main.capacity, color="#b5b3ad", ls="--", lw=1)
    ax.text(len(fuel)-1, th*main.capacity+0.8, lab, ha="right", fontsize=8, color="#52514e")
ax.set_xlabel("tick (moves and refuels)"); ax.set_ylabel("fuel"); ax.set_title("Fuel level during the mission (map_main, h4)")
plt.tight_layout(); plt.savefig(FIG / "fuel_trace.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# numbers quoted in the report
summary = {
  "A_main": sumA["map_main"][["total_expanded","expanded_vs_dijkstra","mean_ms","mean_ebf","optimal_pct","max_gap_pct"]],
  "B_random": sumB[["total_expanded","expanded_vs_dijkstra","mean_ms","mean_ebf","optimal_pct","max_gap_pct"]],
}
for k, v in summary.items():
    print(k); print(v.to_string()); print()
sumA["map_main"].to_csv(FIG / "expA_main_summary.csv"); sumA["track_main"].to_csv(FIG / "expA_track_summary.csv")
sumB.to_csv(FIG / "expB_summary.csv"); mis.to_csv(FIG / "expE_missions.csv", index=False); adm.to_csv(FIG / "admissibility.csv", index=False)
inf.to_csv(FIG / "informedness.csv", index=False); by_size.to_csv(FIG / "expB_by_size.csv")
""")

md(r"""
## 8. Conclusions

* **All four designed heuristics (h1–h4) are admissible and consistent.** This is proven in the report
  and confirmed above with zero violations. They always return the same optimal cost as Dijkstra.
  Only the search effort changes.
* **Ranking by search effort: h4 (ALT) < h3 (corridor) < h1 (Manhattan) < h2 (Euclidean) < h0.**
  Euclidean is the weakest informed heuristic because on a 4-connected grid it is always ≤ Manhattan.
* **h3** reads terrain and currents. It is never weaker than h1 and is clearly better wherever whole rows
  or columns can only be entered expensively (grass bands, headwinds). It costs O(1) per call after an
  O(cells) pre-pass.
* **h4 (ALT)** is the most informed by far (mean h/h\* = 0.91 on map_main, 0.94 on the track). It expands
  **4–10× fewer nodes than Dijkstra** (17.6 % on map_main, 23 % on the track, 10 % on the random maps),
  because the landmark distances already contain the walls and the asymmetric currents. Its price is a one-off
  pre-computation (2 Dijkstra runs per landmark, a few ms). It pays off because the tour planner runs
  dozens of searches on the same map.
* **h3** is never weaker than h1 (each row/column entry costs ≥ c_min). It gains most on the random maps
  (28 % of Dijkstra's expansions vs 58 % for h1), where walls and grass make whole rows expensive to enter.
* **h5 (weighted)** expands about as few nodes as ALT on random maps, but is optimal in only ~50 % of
  searches (worst gap 67 %). It is useful only when planning time matters more than fuel.
* **Mission level:** every admissible heuristic gives the same tour cost. The small fuel differences
  (e.g. 283 vs 310 on the track in race mode) come only from tie-breaking between equal-cost paths,
  which changes where the fuel rules fire.
* **Interesting finding:** on map_main the fuel-agnostic optimal tour (survey, planned cost 126) uses
  **more** fuel in practice (232) than the fixed race order 1→5 (planned 138, actual 179), because it sends
  the car to checkpoints far from any pit, and then the reflex rules add long detours. A
  **fuel-aware tour planner** (tour cost evaluated with pit stops) is therefore the main improvement for Part 2.

**Recommendation:** use **h4 (ALT)** as the default planner heuristic, and **h3** when pre-computation is
not possible (e.g. maps that change online).
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, OUT)
print("wrote", OUT)

if "--no-exec" not in sys.argv:
    from nbconvert.preprocessors import ExecutePreprocessor
    nb = nbf.read(OUT, as_version=4)
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
    nbf.write(nb, OUT)
    print("executed", OUT)
