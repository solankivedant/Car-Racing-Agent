"""Race track: loading, geometry, drivable mask, progress along the lap.

A track file (``tracks/*.track``) describes a closed centre-line and its width::

    NAME Serpentine
    SIZE 1200 720        world size in pixels (width height)
    WIDTH 70             track width in pixels
    LAPS 1
    POINTS               control points of the closed centre-line (x y), in driving order
    90 420
    ...
    OBSTACLES            optional: rocks / oil drums on the track (x y radius)
    700 610 11

The control points are smoothed with a closed Catmull-Rom spline. Everything a car
needs is then pre-computed once with numpy/scipy:

* ``mask[y, x]``      True where the car may drive (asphalt)
* ``clearance[y, x]`` distance in px to the nearest wall/obstacle
* ``arc[y, x]``       arc-length position of the nearest centre-line point, used for
                      race progress and "distance to the finish line"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

Point = Tuple[float, float]


class TrackFormatError(ValueError):
    pass


@dataclass
class Track:
    name: str
    width_px: int
    height_px: int
    track_width: float
    laps: int
    control: List[Point]
    obstacles: List[Tuple[float, float, float]]
    centre: np.ndarray = field(repr=False, default=None)     # (M, 2) samples, ~1 px apart
    s: np.ndarray = field(repr=False, default=None)          # arc length of each sample
    length: float = 0.0
    mask: np.ndarray = field(repr=False, default=None)       # (H, W) bool
    clearance: np.ndarray = field(repr=False, default=None)  # (H, W) float px
    arc: np.ndarray = field(repr=False, default=None)        # (H, W) float px
    file: str = ""
    start: Optional[Tuple[float, float, float]] = None       # painted tracks: (x, y, heading rad)
    mask_file: str = ""                                      # painted tracks: PNG with the road

    # ------------------------------------------------------------ geometry
    @property
    def painted(self) -> bool:
        return not self.control

    @property
    def start_pos(self) -> Point:
        if self.start is not None:
            return float(self.start[0]), float(self.start[1])
        return float(self.centre[0, 0]), float(self.centre[0, 1])

    @property
    def start_heading(self) -> float:
        if self.start is not None:
            return float(self.start[2])
        return self.heading_at(0.0)

    def point_at(self, s: float) -> Point:
        i = int(np.searchsorted(self.s, s % self.length)) % len(self.s)
        return float(self.centre[i, 0]), float(self.centre[i, 1])

    def heading_at(self, s: float) -> float:
        d = 12 if self.painted else 4
        a = self.point_at(s - d)
        b = self.point_at(s + d)
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def normal_at(self, s: float) -> Point:
        h = self.heading_at(s)
        return -math.sin(h), math.cos(h)          # points to the driver's right (y is down)

    def grid_slots(self, n: int, spacing: float = 30.0, lateral: float = 14.0):
        """Staggered starting grid behind the start line: (x, y, heading, s0)."""
        slots = []
        for k in range(n):
            s0 = -(12 + spacing * (k // 2) + (spacing / 2) * (k % 2))
            side = -1 if k % 2 == 0 else 1
            x, y = self.point_at(s0)
            nx, ny = self.normal_at(s0)
            slots.append((x + nx * lateral * side, y + ny * lateral * side,
                          self.heading_at(s0), s0))
        return slots

    # ------------------------------------------------------------ queries
    def inside(self, x, y):
        """Vectorised: True where (x, y) is on the drivable asphalt."""
        xi = np.asarray(x).astype(int)
        yi = np.asarray(y).astype(int)
        ok = (xi >= 0) & (yi >= 0) & (xi < self.width_px) & (yi < self.height_px)
        out = np.zeros(np.shape(xi), dtype=bool)
        out[ok] = self.mask[yi[ok], xi[ok]]
        return out

    def arc_at(self, x, y):
        xi = np.clip(np.asarray(x).astype(int), 0, self.width_px - 1)
        yi = np.clip(np.asarray(y).astype(int), 0, self.height_px - 1)
        return self.arc[yi, xi]

    def clearance_at(self, x, y):
        xi = np.clip(np.asarray(x).astype(int), 0, self.width_px - 1)
        yi = np.clip(np.asarray(y).astype(int), 0, self.height_px - 1)
        return self.clearance[yi, xi]

    def self_overlap(self, margin: float = 1.15) -> float:
        """Smallest distance between centre-line points far apart along the lap,
        divided by the track width. A value < margin means the road would merge
        with itself, so the track file is invalid. Painted tracks are validated by
        their geodesic lap instead (see build_from_mask)."""
        if self.painted:
            return float("inf")
        c = self.centre[::6]
        s = self.s[::6]
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        ds = np.abs(s[:, None] - s[None, :])
        ds = np.minimum(ds, self.length - ds)
        far = ds > self.track_width * 2.0
        if not far.any():
            return float("inf")
        return float(d[far].min() / self.track_width)


# ---------------------------------------------------------------- building
def catmull_rom_closed(pts: np.ndarray, samples_per_seg: int = 60) -> np.ndarray:
    n = len(pts)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        t = np.linspace(0, 1, samples_per_seg, endpoint=False)[:, None]
        t2, t3 = t * t, t * t * t
        seg = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                     + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
        out.append(seg)
    return np.vstack(out)


def resample(poly: np.ndarray, step: float = 1.0):
    closed = np.vstack([poly, poly[:1]])
    seg = np.sqrt((np.diff(closed, axis=0) ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    s = np.arange(0, L, step)
    x = np.interp(s, cum, closed[:, 0])
    y = np.interp(s, cum, closed[:, 1])
    return np.stack([x, y], 1), s, float(L)


def build(track: Track) -> Track:
    pts = np.array(track.control, dtype=float)
    centre, s, L = resample(catmull_rom_closed(pts), 1.0)
    track.centre, track.s, track.length = centre, s, L
    W, H = track.width_px, track.height_px

    # distance from every pixel to the centre-line (+ nearest centre-line pixel)
    line = np.ones((H, W), dtype=bool)
    ci = np.clip(np.round(centre).astype(int), [0, 0], [W - 1, H - 1])
    line[ci[:, 1], ci[:, 0]] = False
    arc_of_pixel = np.zeros((H, W))
    arc_of_pixel[ci[:, 1], ci[:, 0]] = s
    dist, (iy, ix) = distance_transform_edt(line, return_indices=True)
    mask = dist <= track.track_width / 2
    # world border is always a wall
    mask[:3, :] = mask[-3:, :] = False
    mask[:, :3] = mask[:, -3:] = False
    yy, xx = np.mgrid[0:H, 0:W]
    for (ox, oy, r) in track.obstacles:
        mask &= (xx - ox) ** 2 + (yy - oy) ** 2 > r * r
    track.mask = mask
    track.arc = arc_of_pixel[iy, ix]
    track.clearance = distance_transform_edt(mask)
    return track


# ---------------------------------------------------------------- painted tracks
GEO_CELL = 4
_SQ2 = math.sqrt(2)


def build_from_mask(track: Track, road: np.ndarray) -> Track:
    """Build a track from a painted road mask and a start pose.

    A painted track has no centre-line, so lap progress is computed *geodesically*.
    A Dijkstra run (scipy, 4 px cells) starts just after the start line, with a
    barrier just behind it. It gives every road pixel its driving distance from the
    line. The lap length is the distance needed to come all the way round to the
    line again. A centre-line is then recovered as the road centroid of every
    distance slice.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    H, W = road.shape
    track.width_px, track.height_px = W, H
    mask = road.astype(bool).copy()
    mask[:3, :] = mask[-3:, :] = False
    mask[:, :3] = mask[:, -3:] = False
    yy, xx = np.mgrid[0:H, 0:W]
    for (ox, oy, r) in track.obstacles:
        mask &= (xx - ox) ** 2 + (yy - oy) ** 2 > r * r
    track.mask = mask
    track.clearance = distance_transform_edt(mask)
    if track.start is None:
        raise TrackFormatError("painted track needs a START x y heading")
    sx, sy, sh = track.start
    if not (0 <= int(sx) < W and 0 <= int(sy) < H) or not mask[int(sy), int(sx)]:
        raise TrackFormatError("the start position is not on the road")
    half = float(track.clearance[int(sy), int(sx)])
    if half < 18:
        raise TrackFormatError("the road at the start is too narrow (needs > 36 px)")
    if not track.track_width or track.track_width <= 0:
        track.track_width = 2 * half

    # ---- coarse geodesic grid
    c = GEO_CELL
    R, C = H // c, W // c
    cy = np.arange(R) * c + c // 2
    cx = np.arange(C) * c + c // 2
    free = track.clearance[np.ix_(cy, cx)] >= 9.0          # a car (14 px wide) must fit
    X, Y = np.meshgrid(cx, cy)
    dx, dy = math.cos(sh), math.sin(sh)
    along = (X - sx) * dx + (Y - sy) * dy
    across = -(X - sx) * dy + (Y - sy) * dx
    near = np.abs(across) <= half + 3 * c
    barrier = near & (along > -2.5 * c) & (along <= 0)
    ok = free & ~barrier
    idx = np.arange(R * C).reshape(R, C)
    rows, cols, wts = [], [], []
    for ddy, ddx, dl in [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                         (-1, -1, _SQ2), (-1, 1, _SQ2), (1, -1, _SQ2), (1, 1, _SQ2)]:
        r0, r1 = max(0, -ddy), R - max(0, ddy)
        c0, c1 = max(0, -ddx), C - max(0, ddx)
        a = ok[r0:r1, c0:c1] & ok[r0 + ddy:r1 + ddy, c0 + ddx:c1 + ddx]
        if ddy and ddx:
            a &= ok[r0 + ddy:r1 + ddy, c0:c1] & ok[r0:r1, c0 + ddx:c1 + ddx]
        rows.append(idx[r0:r1, c0:c1][a])
        cols.append(idx[r0 + ddy:r1 + ddy, c0 + ddx:c1 + ddx][a])
        wts.append(np.full(int(a.sum()), dl * c))
    G = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(R * C, R * C))
    src = np.nonzero((ok & near & (along > 0) & (along <= 1.5 * c)).ravel())[0]
    if len(src) == 0:
        raise TrackFormatError("no road just in front of the start line")
    dist = dijkstra(G, directed=False, indices=src, min_only=True).reshape(R, C)
    behind = ok & near & (along <= -2.5 * c) & (along > -6 * c)
    fin = behind & np.isfinite(dist)
    if not fin.any():
        raise TrackFormatError("the road is not a closed loop (or too narrow somewhere, a car needs "
                               "about 20 px): a car cannot drive round back to the start line")
    L = float(dist[fin].min()) + 3.0 * c
    if L < 400:
        raise TrackFormatError("the lap is too short (< 400 px)")
    reach = np.isfinite(dist)
    dist = np.where(reach, np.minimum(dist, L - 1e-3), 0.0)
    band = barrier & free
    dist[band] = L - 1.0                   # the barrier band belongs to the end of the lap
    have = reach | band

    # ---- per-pixel lap position (taken from the nearest coarse cell that has one)
    _, (iy, ix) = distance_transform_edt(~have, return_indices=True)
    coarse = dist[iy, ix]
    track.arc = coarse[np.clip(yy // c, 0, R - 1), np.clip(xx // c, 0, C - 1)]

    # ---- centre-line: road centroid of every 6 px slice of lap distance
    sel = have & free
    bins = (dist[sel] // 6).astype(int)
    nb = int(L // 6) + 1
    cnt = np.bincount(bins, minlength=nb).astype(float)
    mx = np.bincount(bins, weights=X[sel], minlength=nb)
    my = np.bincount(bins, weights=Y[sel], minlength=nb)
    good = cnt > 0
    pts = np.stack([mx[good] / cnt[good], my[good] / cnt[good]], 1)
    pts[0] = (sx, sy)
    k = 4                                  # cyclic smoothing (~50 px window) removes centroid noise
    for _ in range(2):
        pad = np.vstack([pts[-k:], pts, pts[:k]])
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        pts = np.stack([np.convolve(pad[:, 0], ker, "valid"), np.convolve(pad[:, 1], ker, "valid")], 1)
    pts[0] = (sx, sy)
    centre, sc, Lc = resample(pts, 1.0)
    track.centre = centre
    track.s = sc * (L / Lc)                # parametrise the centre-line by lap distance
    track.length = L
    track.control = []
    return track


def save_painted_track(track_path, name: str, road: np.ndarray, start, obstacles, laps: int = 1):
    """Write ``<name>.png`` (white = road) and ``<name>.track`` side by side."""
    from PIL import Image
    track_path = Path(track_path)
    png = track_path.with_suffix(".png")
    Image.fromarray(np.where(road, 255, 0).astype(np.uint8)).save(png)
    lines = ["# Car Racing Agent track, painted with the track editor (python main.py editor)",
             f"NAME {name}", f"MASK {png.name}",
             f"START {start[0]:.1f} {start[1]:.1f} {math.degrees(start[2]):.1f}", f"LAPS {laps}"]
    if obstacles:
        lines.append("OBSTACLES")
        lines += [f"{x:.0f} {y:.0f} {r:.0f}" for x, y, r in obstacles]
    track_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return track_path


def parse_track(text: str, file: str = "", base_dir=None) -> Track:
    name, size, width, laps = "track", None, None, 1
    points, obstacles = [], []
    mask_file, start = "", None
    section = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key = line.split()[0].upper()
        if key in ("POINTS", "OBSTACLES"):
            section = key
            continue
        if key == "NAME":
            name = line[4:].strip()
            continue
        if key == "MASK":
            mask_file = line[4:].strip()
            continue
        if key == "START":
            try:
                v = [float(q) for q in line.split()[1:]]
            except ValueError as e:
                raise TrackFormatError(f"bad number in {raw!r}") from e
            if len(v) != 3:
                raise TrackFormatError("START needs x y heading_degrees")
            start = (v[0], v[1], math.radians(v[2]))
            continue
        if key in ("SIZE", "WIDTH", "LAPS"):
            parts = line.split()[1:]
            try:
                vals = [float(v) for v in parts]
            except ValueError as e:
                raise TrackFormatError(f"bad number in {raw!r}") from e
            if key == "SIZE":
                if len(vals) != 2:
                    raise TrackFormatError("SIZE needs width and height")
                size = (int(vals[0]), int(vals[1]))
            elif key == "WIDTH":
                width = vals[0]
            else:
                laps = int(vals[0])
            continue
        try:
            vals = [float(v) for v in line.split()]
        except ValueError:
            raise TrackFormatError(f"unexpected line {raw!r}") from None
        if section == "POINTS" and len(vals) == 2:
            points.append((vals[0], vals[1]))
        elif section == "OBSTACLES" and len(vals) == 3:
            obstacles.append((vals[0], vals[1], vals[2]))
        else:
            raise TrackFormatError(f"unexpected line {raw!r}")
    if mask_file:
        from PIL import Image
        mp = Path(base_dir or ".") / mask_file
        if not mp.exists():
            raise TrackFormatError(f"mask image not found: {mp}")
        road = np.array(Image.open(mp).convert("L")) > 127
        t = Track(name, road.shape[1], road.shape[0], width or 0, max(1, laps), [], obstacles,
                  file=file, start=start, mask_file=mask_file)
        return build_from_mask(t, road)
    if size is None or width is None:
        raise TrackFormatError("SIZE and WIDTH are required")
    if len(points) < 4:
        raise TrackFormatError("need at least 4 centre-line points")
    if laps < 1 or width <= 0:
        raise TrackFormatError("LAPS >= 1 and WIDTH > 0 required")
    t = Track(name, size[0], size[1], width, laps, points, obstacles, file=file)
    return build(t)


def load_track(path) -> Track:
    path = Path(path)
    return parse_track(path.read_text(encoding="utf-8"), file=path.stem, base_dir=path.parent)
