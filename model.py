"""Agent-side forward model of the ball: walls, ceiling, brick damage and the paddle line (no server imports).

Written from the published rules; the lab's modelcheck measures its landing error against the real engine.
"""

import math
from typing import Any

import numpy as np

PADDLE_STEP = 28.0
INF_HEALTH = 1 << 20


class Field:
    """Static geometry plus a mutable brick table (expanded by the ball radius for point-vs-box tests)."""

    def __init__(self, st: dict[str, Any]) -> None:
        self.W = float(st["width"])
        self.r = float(st["ball_radius"])
        self.speed = float(st["ball_speed"])
        self.ceil = float(st["ceiling_y"]) + self.r
        self.y_pad = float(st["paddle_y"]) - self.r  # ball-centre height at paddle contact
        self.pw = float(st["paddle_width"])
        self.load_bricks(st["bricks"])

    def load_bricks(self, bricks: list[dict[str, Any]]) -> None:
        r = self.r
        self.x0 = np.array([b["x"] - r for b in bricks])
        self.x1 = np.array([b["x"] + b["width"] + r for b in bricks])
        self.y0 = np.array([b["y"] - r for b in bricks])
        self.y1 = np.array([b["y"] + b["height"] + r for b in bricks])
        hp = [INF_HEALTH if b["indestructible"] else b["hits_remaining"] for b in bricks]
        self.health = np.array(hp, dtype=np.int32)
        self.points = np.array([0 if b["indestructible"] else b["points"] for b in bricks], dtype=np.float64)
        self.alive = np.ones(len(bricks), dtype=bool)
        self.destructible = self.health < INF_HEALTH

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self.health.copy(), self.alive.copy()

    def restore(self, snap: tuple[np.ndarray, np.ndarray]) -> None:
        self.health[:], self.alive[:] = snap


def fly(f: Field, x: float, y: float, vx: float, vy: float, max_events: int = 40, to_paddle: bool = True):
    """Advance the ball until it reaches the paddle line while descending (mutates f.health / f.alive).

    Returns (x_land, t_land, hits, breaks, points, path) where hits counts damage on destructible bricks.
    x_land is None if the ball never comes down within max_events.
    """
    t = 0.0
    hits = breaks = 0
    pts = 0.0
    r, W = f.r, f.W
    path = [(x, y)]
    for _ in range(max_events):
        best, kind, bi, nx, ny = math.inf, 0, -1, 0.0, 0.0
        if vx < 0:
            tt = (r - x) / vx
            if tt < best:
                best, kind = tt, 1
        elif vx > 0:
            tt = (W - r - x) / vx
            if tt < best:
                best, kind = tt, 1
        if vy < 0:
            tt = (f.ceil - y) / vy
            if tt < best:
                best, kind = tt, 2
        elif vy > 0 and to_paddle:
            tt = (f.y_pad - y) / vy
            if tt < best:
                best, kind = tt, 3
        if best == math.inf:
            break
        best = max(best, 0.0)
        # bricks: slab test, vectorised over live bricks
        idx = np.flatnonzero(f.alive)
        if idx.size:
            if vx != 0.0:
                a, b = (f.x0[idx] - x) / vx, (f.x1[idx] - x) / vx
                t1x, t2x = np.minimum(a, b), np.maximum(a, b)
            else:
                inside = (x >= f.x0[idx]) & (x <= f.x1[idx])
                t1x = np.where(inside, -1e9, 1e9)
                t2x = np.where(inside, 1e9, -1e9)
            if vy != 0.0:
                a, b = (f.y0[idx] - y) / vy, (f.y1[idx] - y) / vy
                t1y, t2y = np.minimum(a, b), np.maximum(a, b)
            else:
                inside = (y >= f.y0[idx]) & (y <= f.y1[idx])
                t1y = np.where(inside, -1e9, 1e9)
                t2y = np.where(inside, 1e9, -1e9)
            t_in = np.maximum(t1x, t1y)
            ok = (t_in <= np.minimum(t2x, t2y)) & (np.minimum(t2x, t2y) >= 0.0) & (t_in < best)
            if ok.any():
                x_axis = t1x > t1y
                nxs = np.where(x_axis, np.where(vx > 0, -1.0, 1.0), 0.0)
                nys = np.where(x_axis, 0.0, np.where(vy > 0, -1.0, 1.0))
                ok &= (vx * nxs + vy * nys) < 0.0
                if ok.any():
                    cand = np.where(ok, np.maximum(t_in, 0.0), math.inf)
                    k = int(np.argmin(cand))
                    best, kind, bi, nx, ny = float(cand[k]), 4, int(idx[k]), float(nxs[k]), float(nys[k])
        x += vx * best
        y += vy * best
        t += best
        path.append((x, y))
        if kind == 3:
            return x, t, hits, breaks, pts, path
        if kind == 1:
            vx = -vx
            x = r if vx > 0 else W - r
        elif kind == 2:
            vy = -vy
            y = f.ceil
        elif kind == 4:
            if nx != 0.0:
                vx = -vx
                x = (f.x0[bi] - 1e-2) if nx < 0 else (f.x1[bi] + 1e-2)
            else:
                vy = -vy
                y = (f.y0[bi] - 1e-2) if ny < 0 else (f.y1[bi] + 1e-2)
            if f.destructible[bi]:
                hits += 1
                f.health[bi] -= 1
                if f.health[bi] <= 0:
                    f.alive[bi] = False
                    breaks += 1
                    pts += f.points[bi]
        else:
            break
    return None, t, hits, breaks, pts, path


def zone_velocity(f: Field, zone: int, angle_deg: float | None = None) -> tuple[float, float]:
    """Velocity after a paddle bounce. zone: -1 left, 0 centre, +1 right; angle_deg overrides the nominal angle."""
    a = math.radians(45.0 * zone if angle_deg is None else angle_deg)
    return f.speed * math.sin(a), -f.speed * math.cos(a)


def move_costs(px: float, pw: float, W: float) -> dict[float, int]:
    """Minimum number of moves to stand at every paddle x reachable from px (28 px steps, clamped at the walls)."""
    hi = W - pw
    dist = {px: 0}
    frontier = [px]
    while frontier:
        nxt = []
        for p in frontier:
            for q in (max(0.0, p - PADDLE_STEP), min(hi, p + PADDLE_STEP)):
                if q not in dist:
                    dist[q] = dist[p] + 1
                    nxt.append(q)
        frontier = nxt
    return dist
