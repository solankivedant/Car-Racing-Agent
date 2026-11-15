"""World model: map parser, tile types and the direction-dependent cost model.

The map format is the one used by the course (``map_main.txt``)::

    ROWS 15
    COLS 20
    BATTERY 50          (FUEL is accepted as a synonym)
    TERRAIN
    <ROWS lines of COLS characters>
    CURRENTS
    <ROWS lines of COLS characters>

Terrain characters
    ``.``  asphalt / open water          (entering cost 2)
    ``~``  grass / gravel  (kelp)        (entering cost 6)
    ``#``  wall / barrier  (reef)        (impassable)
    ``H``  start-finish line + garage    (asphalt, refuel point)   alias ``S``
    ``C``  pit stop                      (asphalt, refuel point)   alias ``P``
    ``1``-``9`` checkpoints              (asphalt)

Current / slope characters: ``^ > v <`` for N, E, S, W and ``.`` for none.

Cost of moving from tile ``a`` into tile ``b`` in direction ``d``:
    base = cost of terrain(b)
    current(b) == d            -> base / 2   (downhill / tailwind)
    current(b) == opposite(d)  -> base * 2   (uphill / headwind)
    otherwise                  -> base
All resulting costs are integers (1, 2, 3, 4, 6, 12).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

Pos = Tuple[int, int]

# ----------------------------------------------------------------------------
# Directions
# ----------------------------------------------------------------------------
N, E, S, W = "N", "E", "S", "W"
DIRECTIONS: Dict[str, Pos] = {N: (-1, 0), E: (0, 1), S: (1, 0), W: (0, -1)}
OPPOSITE = {N: S, S: N, E: W, W: E}
LEFT_OF = {N: W, W: S, S: E, E: N}
RIGHT_OF = {N: E, E: S, S: W, W: N}
CURRENT_CHARS = {"^": N, ">": E, "v": S, "<": W, ".": None}
CURRENT_SYMBOL = {N: "^", E: ">", S: "v", W: "<", None: "."}

# ----------------------------------------------------------------------------
# Terrain
# ----------------------------------------------------------------------------
ASPHALT, GRASS, WALL = "asphalt", "grass", "wall"
TERRAIN_COST = {ASPHALT: 2, GRASS: 6, WALL: None}
INF = float("inf")


class MapFormatError(ValueError):
    """Raised when a map file does not follow the expected format."""


@dataclass
class Grid:
    rows: int
    cols: int
    capacity: int
    terrain: List[List[str]]                 # ASPHALT / GRASS / WALL
    current: List[List[Optional[str]]]       # N/E/S/W or None
    home: Pos
    pits: List[Pos]                          # every refuel point (H + all C)
    sites: Dict[int, Pos]                    # checkpoint id -> position
    raw: List[str] = field(default_factory=list, repr=False)
    name: str = "map"
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ basics
    def in_bounds(self, p: Pos) -> bool:
        return 0 <= p[0] < self.rows and 0 <= p[1] < self.cols

    def passable(self, p: Pos) -> bool:
        return self.in_bounds(p) and self.terrain[p[0]][p[1]] != WALL

    def is_pit(self, p: Pos) -> bool:
        return p in self._pit_set()

    def _pit_set(self):
        if "pitset" not in self._cache:
            self._cache["pitset"] = frozenset(self.pits)
        return self._cache["pitset"]

    def site_at(self, p: Pos) -> Optional[int]:
        for k, v in self.sites.items():
            if v == p:
                return k
        return None

    def char_at(self, p: Pos) -> str:
        return self.raw[p[0]][p[1]]

    # ------------------------------------------------------------ cost model
    def step_cost(self, a: Pos, b: Pos) -> float:
        """Fuel cost of moving from ``a`` into the adjacent tile ``b``."""
        d = direction_between(a, b)
        if d is None or not self.passable(b):
            return INF
        base = TERRAIN_COST[self.terrain[b[0]][b[1]]]
        cur = self.current[b[0]][b[1]]
        if cur is None:
            return base
        if cur == d:
            return base // 2
        if cur == OPPOSITE[d]:
            return base * 2
        return base

    def neighbors(self, p: Pos) -> Iterator[Tuple[Pos, str, float]]:
        """Successors of ``p``: (next_pos, direction, cost)."""
        r, c = p
        for d, (dr, dc) in DIRECTIONS.items():
            q = (r + dr, c + dc)
            if self.passable(q):
                yield q, d, self.step_cost(p, q)

    def predecessors(self, p: Pos) -> Iterator[Tuple[Pos, float]]:
        """Tiles ``q`` that can move into ``p`` and the cost of that move."""
        if not self.passable(p):
            return
        r, c = p
        for dr, dc in DIRECTIONS.values():
            q = (r + dr, c + dc)
            if self.passable(q):
                yield q, self.step_cost(q, p)

    def passable_tiles(self) -> Iterator[Pos]:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.terrain[r][c] != WALL:
                    yield (r, c)

    @property
    def min_step_cost(self) -> float:
        """Cheapest single move that exists on this map (``c_min``).

        Heuristics are multiplied by this so they never over-estimate. Without
        a tail-current the minimum is 2, with one it is 1.
        """
        if "cmin" not in self._cache:
            best = INF
            for p in self.passable_tiles():
                for _, _, cost in self.neighbors(p):
                    best = min(best, cost)
            self._cache["cmin"] = best if best < INF else 1
        return self._cache["cmin"]

    def path_cost(self, path: List[Pos]) -> float:
        return sum(self.step_cost(a, b) for a, b in zip(path, path[1:]))

    def render(self, path: Optional[List[Pos]] = None) -> str:
        """ASCII picture of the map, optionally with a path marked by ``*``."""
        on_path = set(path or [])
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                ch = self.raw[r][c]
                if (r, c) in on_path and ch in ".~":
                    ch = "*"
                row.append(ch)
            lines.append("".join(row))
        return "\n".join(lines)


def direction_between(a: Pos, b: Pos) -> Optional[str]:
    delta = (b[0] - a[0], b[1] - a[1])
    for d, v in DIRECTIONS.items():
        if v == delta:
            return d
    return None


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
_HOME_CHARS = {"H", "S"}
_PIT_CHARS = {"C", "P"}


def parse_map(text: str, name: str = "map") -> Grid:
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]
    header: Dict[str, int] = {}
    i = 0
    while i < len(lines) and lines[i].strip().upper() != "TERRAIN":
        parts = lines[i].split()
        if len(parts) != 2:
            raise MapFormatError(f"Bad header line: {lines[i]!r}")
        key, val = parts[0].upper(), parts[1]
        if key == "FUEL":
            key = "BATTERY"
        try:
            header[key] = int(val)
        except ValueError as exc:
            raise MapFormatError(f"Header value must be an integer: {lines[i]!r}") from exc
        i += 1
    for key in ("ROWS", "COLS", "BATTERY"):
        if key not in header:
            raise MapFormatError(f"Missing header field {key}")
    rows, cols, cap = header["ROWS"], header["COLS"], header["BATTERY"]
    if rows <= 0 or cols <= 0 or cap <= 0:
        raise MapFormatError("ROWS, COLS and BATTERY must be positive")
    if i >= len(lines):
        raise MapFormatError("Missing TERRAIN section")
    i += 1
    terrain_lines = lines[i:i + rows]
    i += rows
    if len(terrain_lines) != rows:
        raise MapFormatError(f"Expected {rows} terrain rows, got {len(terrain_lines)}")
    if i >= len(lines) or lines[i].strip().upper() != "CURRENTS":
        raise MapFormatError("Missing CURRENTS section after the terrain rows")
    i += 1
    current_lines = lines[i:i + rows]
    if len(current_lines) != rows:
        raise MapFormatError(f"Expected {rows} current rows, got {len(current_lines)}")

    terrain: List[List[str]] = []
    current: List[List[Optional[str]]] = []
    home: Optional[Pos] = None
    pits: List[Pos] = []
    sites: Dict[int, Pos] = {}
    raw: List[str] = []
    for r in range(rows):
        tl, cl = terrain_lines[r], current_lines[r]
        if len(tl) != cols:
            raise MapFormatError(f"Terrain row {r} has {len(tl)} columns, expected {cols}")
        if len(cl) != cols:
            raise MapFormatError(f"Current row {r} has {len(cl)} columns, expected {cols}")
        trow, crow, rawrow = [], [], []
        for c in range(cols):
            ch = tl[c]
            if ch == ".":
                trow.append(ASPHALT)
            elif ch == "~":
                trow.append(GRASS)
            elif ch == "#":
                trow.append(WALL)
            elif ch in _HOME_CHARS:
                if home is not None:
                    raise MapFormatError("More than one home base (H)")
                home = (r, c)
                trow.append(ASPHALT)
                ch = "H"
            elif ch in _PIT_CHARS:
                pits.append((r, c))
                trow.append(ASPHALT)
                ch = "C"
            elif ch.isdigit() and ch != "0":
                k = int(ch)
                if k in sites:
                    raise MapFormatError(f"Checkpoint {k} appears twice")
                sites[k] = (r, c)
                trow.append(ASPHALT)
            else:
                raise MapFormatError(f"Unknown terrain char {ch!r} at ({r},{c})")
            rawrow.append(ch)
            cc = cl[c]
            if cc not in CURRENT_CHARS:
                raise MapFormatError(f"Unknown current char {cc!r} at ({r},{c})")
            crow.append(CURRENT_CHARS[cc])
        terrain.append(trow)
        current.append(crow)
        raw.append("".join(rawrow))
    if home is None:
        raise MapFormatError("Map has no home base (H)")
    return Grid(rows, cols, cap, terrain, current, home, [home] + pits,
                dict(sorted(sites.items())), raw, name)


def load_map(path) -> Grid:
    path = Path(path)
    return parse_map(path.read_text(encoding="utf-8"), name=path.stem)


def grid_to_text(grid: Grid) -> str:
    """Serialise a grid back to the course file format."""
    out = [f"ROWS {grid.rows}", f"COLS {grid.cols}", f"BATTERY {grid.capacity}", "TERRAIN"]
    out += grid.raw
    out.append("CURRENTS")
    for r in range(grid.rows):
        out.append("".join(CURRENT_SYMBOL[grid.current[r][c]] for c in range(grid.cols)))
    return "\n".join(out) + "\n"
