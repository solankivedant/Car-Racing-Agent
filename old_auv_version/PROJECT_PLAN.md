# Car Racing Agent (AI): Project Plan (Part 1)

**Course:** CSE518 Artificial Intelligence, Monsoon 2026
**Deadline:** 18 Nov 2026 (all three deliverables)
**Today:** 12 Nov 2026, so there are 6 working days plus 1 buffer day

---

## 1. Core idea: map the AUV problem onto a racing car

Your proposal is a **car racing agent** (car on a track, 5 ray sensors, A* search, rules, utility, learning).
The PDF uses an underwater AUV as its example. The grader still expects the same pieces: a reflex module,
an A* planner, 4 heuristics, a comparison notebook and a report. So we keep the PDF's mechanics and
present them as a racing game:

| PDF (AUV)                    | Your project (Car Racing)                            | Map char |
|------------------------------|------------------------------------------------------|----------|
| Open water (cost 2)          | Asphalt track (cost 2)                               | `.`      |
| Kelp (cost 6)                | Grass / gravel run-off (cost 6, car slows down)      | `~`      |
| Reef (impassable)            | Wall / barrier (crash = forbidden)                   | `#`      |
| Home base `H`                | Start / Finish line + garage                         | `H` (or `S`) |
| Charging dock `C`            | Pit stop (refuel to full)                            | `C` (or `P`) |
| Survey sites `1`–`9`         | Checkpoints `1`–`9`                                  | `1`–`9`  |
| Battery                      | Fuel                                                 | `BATTERY`/`FUEL` |
| Ocean current `^ > v <`      | Slope / tailwind lanes (downhill = half cost, uphill = double) | `^ > v <` |

**Important:** the parser must read the original `map_main.txt` exactly as given, because the PDF says
your code "will also be run on other maps". Accept both letter sets (`H/S`, `C/P`, `BATTERY/FUEL`).

**Two tour modes:**
- `survey` mode (PDF behaviour): visit all checkpoints **in any order** (optimal order), then return to `H`.
- `race` mode (your racing theme): visit checkpoints **in order 1→2→…→9**, then cross the finish `H`.

---

## 2. System architecture

```
car-racing-agent/
├── maps/
│   ├── map_main.txt          # given by the course (must work unchanged)
│   ├── track_main.txt        # your racing-themed track (same format)
│   └── tests/                # small hand-made maps for unit tests
│       ├── tiny_open.txt     # 5x5, no walls
│       ├── currents.txt      # checks with/against/across cost
│       ├── low_fuel.txt      # forces pit stops
│       ├── no_pit_needed.txt
│       └── unreachable.txt   # checkpoint boxed in by walls
├── src/
│   ├── world.py        # map parser, Grid, tile types, step_cost(from, to)
│   ├── heuristics.py   # h0 (baseline) + the 4 heuristics
│   ├── astar.py        # generic A* that returns path, cost, nodes expanded, time
│   ├── tour.py         # pairwise A* costs + Held-Karp DP (survey) / ordered (race)
│   ├── safety.py       # reverse multi-source Dijkstra: cost_to_nearest_pit[tile]
│   ├── reflex.py       # percepts + prioritized condition-action rules
│   ├── agent.py        # reflex + deliberative loop (sense → decide → act → replan)
│   ├── sensors.py      # 5 ray sensors (w1..w5 from your sketch) + tile percepts
│   ├── simulate.py     # headless run, writes a step-by-step log (CSV/JSON)
│   └── visualize.py    # pygame top-down view (car, rays, path, fuel bar, HUD)
├── tests/              # pytest
├── notebooks/
│   └── heuristic_comparison.ipynb   # Deliverable 2
├── report/
│   └── progress_report.pdf          # Deliverable 3 (~2 pages)
├── demo/                            # GIF/MP4 recordings like the YT video
├── main.py             # python main.py --map maps/map_main.txt --heuristic h3 --mode survey --gui
├── requirements.txt    # numpy, pygame, matplotlib, pandas, pytest, jupyter
└── README.md
```

---

## 3. Module design

### 3.1 World model (`world.py`)
- Parse `ROWS`, `COLS`, `BATTERY`, the `TERRAIN` block and the `CURRENTS` block. Check the dimensions and raise clear errors.
- `step_cost(a, b)`: base cost comes from the tile **being entered** (2 or 6, wall = ∞).
  - Move direction equals the current of tile `b`: cost × 0.5
  - Move direction is the opposite of the current: cost × 2
  - Move direction is perpendicular to the current, or there is no current: normal cost
- Because of the currents, costs are **asymmetric** (A→B ≠ B→A). Document this, because it affects heuristics and tour planning.
- Minimum possible step cost is `c_min = 1` (asphalt with a tailwind). Heuristics must use this to stay admissible.

### 3.2 A* planner (`astar.py`)
- State = `(row, col)`. Actions = N/E/S/W.
- Uses a priority queue (`heapq`), a closed set, tie-breaking on larger g, and path reconstruction.
- Returns `path, cost, nodes_expanded, nodes_generated, max_frontier, runtime_ms`. The notebook needs these numbers.

### 3.3 Four heuristics (`heuristics.py`), plus the Dijkstra baseline
| ID | Heuristic | Admissible? | Purpose in the comparison |
|----|-----------|-------------|---------------------------|
| h0 | Zero (Dijkstra), **baseline only** | yes | reference optimal cost and worst-case expansions |
| h1 | Manhattan × `c_min` | yes, consistent | standard grid heuristic |
| h2 | Euclidean × `c_min` | yes, weaker | shows that a less informed heuristic expands more nodes |
| h3 | **Current-aware Manhattan**: per axis, use the cheapest cost achievable in that direction on this map (1 if a helping current exists in that direction, else 2) | yes | domain-specific, tighter than h1 |
| h4 | **Landmark / ALT**: precompute Dijkstra to/from landmarks (H + pit stops + corners); h = max of triangle-inequality bounds (directed version) | yes | strongest informed heuristic, at a precompute cost |
| (bonus) h5 | Weighted A*: 2 × Manhattan × `c_min`. Inadmissible | no | fast but possibly suboptimal, to show the speed/optimality trade-off |

Prove or argue admissibility of each in the report. The notebook checks it empirically by testing h(n) ≤ true cost-to-go for all n.

### 3.4 Tour planner (`tour.py`)
- Compute A* cost and path for every ordered pair among {H, 1..9}. Costs are asymmetric, so A→B and B→A are computed separately.
- **Survey mode:** Held-Karp DP over subsets (asymmetric TSP). For ≤9 checkpoints this is 2⁹·9² ≈ 41k operations, which is instant and exact.
- **Race mode:** fixed order 1→…→n→H.
- The tour is planned **ignoring fuel**. The reflex module enforces the fuel rules and triggers a re-plan of the remaining tour.

### 3.5 Safety map (`safety.py`)
- Run multi-source Dijkstra **on the reversed graph** from all pits/H to get `cost_to_pit[tile]` for every tile in one pass.
- This makes the rule "before every move, check that I can still reach a charger after the move" an O(1) lookup:
  `fuel - step_cost(cur, nxt) >= cost_to_pit[nxt]`.

### 3.6 Reflex module (`reflex.py`): the rules as propositional logic
**Percepts:** current tile type, current of the tile, fuel %, 5 ray-sensor distances, `cost_to_pit`, at_pit, at_checkpoint.

Rules are checked **in priority order**, and the first one that fires wins:

| # | Condition (propositional form) | Action |
|---|--------------------------------|--------|
| R1 | `AtPit ∧ Fuel < 100%` (and the agent came here to refuel) | REFUEL, then RE-PLAN the remaining tour |
| R2 | `Fuel < 25%` | GOTO cheapest pit (A*), refuel, re-plan |
| R3 | `¬SafeAfter(nextMove)`, i.e. `fuel − cost(next) < cost_to_pit[next]` | CANCEL the move, GOTO cheapest pit, refuel, re-plan |
| R4 | `25% ≤ Fuel ≤ 40% ∧ cost_to_pit ≤ 4` | OVERRIDE goals, GOTO pit |
| R5 | `NextTile = Wall` | never move (A* already avoids walls; this is a reactive guard) |
| R6 | `AtCheckpoint(k) ∧ ¬Visited(k)` | MARK visited (log "checkpoint k cleared") |
| R7 | otherwise | FOLLOW the planned path (next waypoint) |

- **Car actions** (from your sketch: left / right / straight / accelerate / brake): every grid move is turned
  into car actions relative to the current heading (turn left, turn right, straight). Use *brake* when
  entering grass or an uphill tile and *accelerate* on asphalt or a tailwind. This is shown in the GUI
  and log; it does not change the PDF cost model.
- **Edge cases:** an unreachable checkpoint (skip with a warning, don't loop forever), a checkpoint that
  cannot be reached and returned from even on a full tank (detect it up front), and staying at the
  same pit with nothing to do (no infinite refuel loop).

### 3.7 Sensors and visualization (the part that looks like the YT video)
- 5 rays at −90°, −45°, 0°, +45°, +90° relative to the heading (your w1..w5), cast tile by tile until a wall or the edge.
- pygame window: coloured tiles, current arrows, rotating car sprite, rays drawn as lines, planned A*
  path, cells A* expanded (toggle with a key), fuel bar, HUD showing *active rule*, fuel, cost so far and checkpoints cleared.
- Keys: `SPACE` pause, `1–5` switch heuristic live, `E` show expansions, `R` restart, `S` screenshot.
- Record demo GIFs/MP4s for the report and presentation (`imageio`).

### 3.8 Hooks for Part 2 (utility and learning from your proposal; don't build them now)
- **Utility:** `U = w_p·progress − w_c·collision_risk + w_s·speed − w_f·fuel_used`
- **Learning:** tune the reflex thresholds or ray weights with a simple genetic algorithm over many simulated
  runs ("repeatedly running and changing things"). This is the same idea as the video's
  "can I beat an AI": compare a hand-written agent with a learned one.
- Keep these as interfaces in `agent.py` so Part 2 can plug in.

---

## 4. Testing plan (`tests/`, pytest)
| Area | Tests |
|------|-------|
| Parser | loads `map_main.txt` (15×20, battery 50, sites 1–5, 2 C, 1 H); rejects bad dimensions or unknown chars |
| Cost model | with current = half, against = double, across = normal; kelp vs water; wall = ∞; asymmetry A→B ≠ B→A |
| A* | path is valid (adjacent steps, no walls); cost equals the sum of step costs; **every admissible h gives the same cost as Dijkstra** on 100 random maps; no path → returns None |
| Heuristics | admissibility check h(n) ≤ h*(n) on every tile; consistency check h(n) ≤ c(n,n') + h(n') for h1–h4 |
| Tour | Held-Karp equals brute-force permutations on small maps; tour starts and ends at H |
| Safety map | `cost_to_pit` equals per-tile A* to the nearest pit |
| Reflex rules | one crafted scenario per rule R1–R7 confirms that the rule fires |
| Full mission (integration) | on all maps: every checkpoint visited, ends at H, **fuel never < 0**, the safety invariant holds at every step, terminates within N steps |
| Regression | save the step log for `map_main.txt` and compare future runs against it |

Run with `pytest -v --cov=src`. Put the coverage summary in the README.

---

## 5. Deliverables checklist
- **D1: Code.** `src/`, `main.py`, `maps/`, `tests/`, README with run commands, `requirements.txt`.
  Demo GIF of the car completing `map_main.txt` with pit stops.
- **D2: `notebooks/heuristic_comparison.ipynb`.**
  1. Load maps (main + racing track + ~50 random maps with a fixed seed).
  2. For each heuristic: nodes expanded, runtime, path cost, optimality gap vs h0, effective branching factor.
  3. Plots: bar charts (expansions and time per heuristic), boxplots over random maps, heatmaps of expanded
     cells on `map_main.txt` (one per heuristic), and full-tour totals (fuel used, pit stops, steps).
  4. Admissibility and consistency check table.
  5. Conclusion cell: which heuristic is best and why.
- **D3: ~2-page progress report** (PDF):
  1. Introduction: car racing agent, PEAS table (your sketch items 1–5, 7).
  2. Environment and cost model (asymmetric currents).
  3. Reflex module: rule table with priorities, safety map trick.
  4. Path planning: A*, tour planning with Held-Karp, re-planning after a pit stop.
  5. The 4 heuristics: definition and admissibility argument for each.
  6. Comparison results: one table plus one figure from the notebook.
  7. Testing summary and next steps (utility and learning for Part 2).

---

## 6. Day-by-day schedule (12 → 18 Nov)

| Day | Date | Work | Done when |
|-----|------|------|-----------|
| 1 | Thu 12 Nov | Project setup, `world.py` parser + cost model, test maps, parser and cost tests | `pytest` green; `map_main.txt` prints correctly |
| 2 | Fri 13 Nov | `astar.py`, the 4 heuristics + h0, `tour.py` (Held-Karp), A* and heuristic tests | Optimal tour cost for `map_main.txt` printed; all h give equal cost |
| 3 | Sat 14 Nov | `safety.py`, `reflex.py`, `agent.py`, headless `simulate.py` with step log | Full mission completes on all maps; safety invariant tests pass |
| 4 | Sun 15 Nov | `sensors.py` + pygame `visualize.py`, HUD, record demo GIF | Car visibly drives the tour, refuels, re-plans |
| 5 | Mon 16 Nov | Random map generator + comparison notebook (D2), plots | Notebook runs top-to-bottom with "Restart & Run All" |
| 6 | Tue 17 Nov | Report (D3), README, code cleanup, docstrings | 2-page PDF ready |
| 7 | Wed 18 Nov | **Buffer**: final full test run on a fresh venv, zip, submit | Submitted |

---

## 7. Risks and how to handle them
| Risk | Mitigation |
|------|------------|
| Infinite pit-stop loop (25–40% rule right after refuelling) | R4 only fires when not already at full fuel; log the rule history and add a test |
| Checkpoint needs more fuel than a full tank allows | Feasibility check at start; report it and skip instead of looping |
| Heuristic accidentally inadmissible because of half-cost currents | Always scale by `c_min = 1`; the automated admissibility test catches it |
| Grader's hidden maps have different sizes or more sites | No hard-coded sizes; tests on random maps of different sizes |
| Running out of time on the GUI | GUI is Day 4 only; the headless simulation plus logs already satisfies D1 |
