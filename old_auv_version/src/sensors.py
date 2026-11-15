"""Car sensors: five distance rays (w1..w5 in the proposal sketch).

Rays are cast relative to the car heading at -90, -45, 0, +45, +90 degrees
(left, front-left, front, front-right, right). Each one steps tile by tile
until it hits a wall or the map edge and returns the number of free tiles.
Diagonal rays move one row and one column per step.
"""
from __future__ import annotations

from typing import Dict, Tuple

from .world import DIRECTIONS, LEFT_OF, RIGHT_OF, Grid, Pos

RAY_NAMES = ("w1_left", "w2_front_left", "w3_front", "w4_front_right", "w5_right")


def _ray_vectors(heading: str) -> Tuple[Tuple[int, int], ...]:
    f = DIRECTIONS[heading]
    l = DIRECTIONS[LEFT_OF[heading]]
    r = DIRECTIONS[RIGHT_OF[heading]]
    fl = (f[0] + l[0], f[1] + l[1])
    fr = (f[0] + r[0], f[1] + r[1])
    return (l, fl, f, fr, r)


def cast_ray(grid: Grid, pos: Pos, vec: Tuple[int, int], max_range: int = 50) -> int:
    r, c = pos
    dist = 0
    while dist < max_range:
        r, c = r + vec[0], c + vec[1]
        if not grid.passable((r, c)):
            break
        dist += 1
    return dist


def read_rays(grid: Grid, pos: Pos, heading: str) -> Dict[str, int]:
    return {name: cast_ray(grid, pos, v) for name, v in zip(RAY_NAMES, _ray_vectors(heading))}


def ray_endpoints(grid: Grid, pos: Pos, heading: str):
    """(name, end_pos) for drawing the rays in the GUI."""
    out = []
    for name, v in zip(RAY_NAMES, _ray_vectors(heading)):
        d = cast_ray(grid, pos, v)
        out.append((name, (pos[0] + v[0] * (d + 0.5), pos[1] + v[1] * (d + 0.5))))
    return out
