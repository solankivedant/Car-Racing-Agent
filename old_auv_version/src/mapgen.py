"""Random map generator (course file format), for experiments and tests."""
from __future__ import annotations

import random
from typing import Optional

from .search import dijkstra
from .world import Grid, grid_to_text, parse_map


def random_map(rows: int = 15, cols: int = 20, n_sites: int = 5, n_pits: int = 2,
               wall_p: float = 0.18, grass_p: float = 0.15, current_lanes: int = 4,
               capacity: Optional[int] = None, seed: Optional[int] = None,
               max_tries: int = 200) -> Grid:
    """Random map in which every special tile is reachable from H and can reach H back.

    Currents are placed as straight lanes (whole rows or columns), like the
    course map, so heuristics meet realistic structured currents and not noise.
    """
    rng = random.Random(seed)
    for _ in range(max_tries):
        terr = [["." for _ in range(cols)] for _ in range(rows)]
        cur = [["." for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                x = rng.random()
                if x < wall_p:
                    terr[r][c] = "#"
                elif x < wall_p + grass_p:
                    terr[r][c] = "~"
        for _ in range(current_lanes):
            if rng.random() < 0.5:
                r = rng.randrange(rows)
                sym = rng.choice("<>")
                lo, hi = sorted(rng.sample(range(cols), 2))
                for c in range(lo, hi + 1):
                    cur[r][c] = sym
            else:
                c = rng.randrange(cols)
                sym = rng.choice("^v")
                lo, hi = sorted(rng.sample(range(rows), 2))
                for r in range(lo, hi + 1):
                    cur[r][c] = sym
        open_cells = [(r, c) for r in range(rows) for c in range(cols) if terr[r][c] == "."]
        need = 1 + n_pits + n_sites
        if len(open_cells) < need:
            continue
        picks = rng.sample(open_cells, need)
        hr, hc = picks[0]
        terr[hr][hc] = "H"
        for (r, c) in picks[1:1 + n_pits]:
            terr[r][c] = "C"
        for i, (r, c) in enumerate(picks[1 + n_pits:], start=1):
            terr[r][c] = str(i)
        cap = capacity if capacity is not None else rng.choice([40, 50, 60, 80])
        text = (f"ROWS {rows}\nCOLS {cols}\nBATTERY {cap}\nTERRAIN\n"
                + "\n".join("".join(row) for row in terr) + "\nCURRENTS\n"
                + "\n".join("".join(row) for row in cur) + "\n")
        g = parse_map(text, name=f"random_{seed}")
        fwd = dijkstra(g, [g.home])
        back = dijkstra(g, [g.home], reverse=True)
        specials = list(g.sites.values()) + g.pits
        if all(p in fwd and p in back for p in specials):
            return g
    raise RuntimeError("Could not generate a connected map; lower wall_p")


def save_map(grid: Grid, path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(grid_to_text(grid))
