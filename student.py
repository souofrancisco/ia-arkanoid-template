"""Track 4 - stochastic rollout planning (expected utility over the bounce-angle distribution).

Every time the ball's trajectory changes, the agent simulates (model.py) the flight to the paddle line, then for
each paddle contact zone that is reachable in time simulates the next flight with the nominal bounce angle, scores
the outcome (brick damage per second) and walks the paddle to the lattice position that produces the best zone.
"""

import argparse
import asyncio
import math
import random

import numpy as np
from typing import Any

from agents.base_agent import BaseAgent
from latency import PaddleTracker
from netloop import run_latest
from model import Field, fly, move_costs, zone_velocity

CFG = {"side": 45.0, "alpha": 8.0, "beta": 15.0}  # config.json paddle.angles (documented in the assignment)


def tri_ppf(u: float, c: float, h: float) -> float:
    """Inverse CDF of the symmetric triangular distribution on [c-h, c+h] with mode c."""
    return c - h + h * math.sqrt(2 * u) if u < 0.5 else c + h - h * math.sqrt(2 * (1 - u))


def angles(zone: int, k: int | None = None) -> list[float]:
    k = k or K_SAMPLES
    c, h = 45.0 * zone, (CFG["alpha"] if zone == 0 else CFG["beta"])
    return [tri_ppf((i + 0.5) / k, c, h) for i in range(k)]

LATEST_LOOP = True  # False = the template's BaseAgent.run (answers every queued state; collapses if a decision takes > ~33 ms)
AGENT_ID: str | None = None  # TODO(group): set to a group member's mechanographic number before submission
VALUE_TAIL = True  # when no zone is expected to hit, choose by a value function over landing x (value iteration, static bricks)
MC_DEPTH = 0  # >0: score each zone by damage/time over this many cycles of a greedy rollout policy (brick changes modelled)
MC_SAMPLES = 5
VALUE_BLEND = 0.0  # >0: add this weight times V(next landing x) to every sample's utility (all game phases)
VT_GRID = 21  # landing-x grid points
VT_K = 5  # angle samples per (x, zone) in the value table
VT_ITERS = 8
VT_GAMMA = 0.85
TAIL_DEPTH = 0  # >0: when no zone is expected to hit anything, search this many paddle cycles ahead (sequence of zones)
TAIL_THRESH = 1.0  # expected damage below which the tail search replaces the rate utility
TAIL_K = 3  # angle samples per zone at the first tail level (deeper levels use the nominal angle)
TAIL_GAMMA = 0.9
HORIZON = 1  # 2 adds a greedy second paddle cycle after each sampled first cycle
K2 = 3  # angle samples per zone in the second cycle
GAMMA = 0.9  # weight of the second cycle
K_SAMPLES = 9  # stratified quantiles of the triangular bounce-angle distribution per zone
W_BREAK = 0.5  # extra utility for finishing a brick (on top of 1 per hit)
W_NEAR = 0.002  # utility per px of closeness of the flight to the nearest live destructible brick
CYCLE_FLOOR = 1.0  # seconds; keeps the rate finite for very short flights


class StudentAgent(BaseAgent):
    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self._key: tuple | None = None
        self._target: float | None = None
        self._tracker: PaddleTracker | None = None
        self._vt_key: tuple | None = None
        self._rng = random.Random(12345)

    async def run(self) -> None:
        await (run_latest(self) if LATEST_LOOP else super().run())

    def plan(self, st: dict[str, Any], px: float, lat: int) -> float | None:
        f = Field(st)
        bx, by, vx, vy = st["ball_x"], st["ball_y"], st["ball_vx"], st["ball_vy"]
        xl, tl, *_ = fly(f, bx, by, vx, vy)
        if xl is None:
            return None
        after_base = f.snapshot()
        budget = max(1, math.ceil(tl * 30.0 - 1e-9) - lat)
        pw = st["paddle_width"]
        best_for_zone: dict[int, tuple[int, float, float]] = {}
        for p, moves in move_costs(px, pw, f.W).items():
            rel = xl - p
            if moves > budget or rel < -f.r or rel > pw + f.r:
                continue
            zone = -1 if rel < pw / 3 else (1 if rel > 2 * pw / 3 else 0)
            cand = (moves, abs(rel - pw * (zone + 1) / 3 - pw / 6), p)
            if zone not in best_for_zone or cand < best_for_zone[zone]:
                best_for_zone[zone] = cand
        if not best_for_zone:
            return None
        live = f.alive & f.destructible
        centres = (
            list(zip(((f.x0 + f.x1) / 2)[live], ((f.y0 + f.y1) / 2)[live], strict=True)) if live.any() else []
        )
        vtab = self.value_table(f, after_base) if VALUE_BLEND > 0 else None
        if MC_DEPTH:
            best_u, best_p = -math.inf, None
            for zone, (moves, _, p) in best_for_zone.items():
                tot_d = tot_t = 0.0
                for si, ang in enumerate(angles(zone, MC_SAMPLES)):
                    d, t = self.policy_rollout(f, after_base, xl, zone, ang, MC_DEPTH, random.Random(1000 + si))
                    tot_d += d
                    tot_t += t
                u = tot_d / max(tot_t, 1e-6) - 1e-4 * moves
                if u > best_u:
                    best_u, best_p = u, p
            return best_p
        best_u, best_p = -math.inf, None
        gains: dict[int, float] = {}
        trans_of: dict[int, list] = {}
        for zone, (moves, _, p) in best_for_zone.items():
            u = 0.0
            g = 0.0
            trans = []
            for ang in angles(zone):
                f.restore(after_base)
                zvx, zvy = zone_velocity(f, zone, ang)
                x2, t2, hits, breaks, _, path = fly(f, xl, f.y_pad, zvx, zvy)
                cycle = max(t2 if x2 is not None else 20.0, CYCLE_FLOOR)
                near = min((math.hypot(qx - cx, qy - cy) for qx, qy in path for cx, cy in centres), default=0.0)
                u += (hits + W_BREAK * breaks) / cycle - W_NEAR * near
                g += hits + W_BREAK * breaks
                if vtab is not None and x2 is not None:
                    u += VALUE_BLEND * float(np.interp(x2, self._vx, vtab))
                trans.append((hits + W_BREAK * breaks, x2))
                if HORIZON >= 2 and x2 is not None:
                    u += GAMMA * self.second_cycle(f, x2, centres)
            u = u / K_SAMPLES - 1e-4 * moves
            gains[zone] = g / K_SAMPLES
            trans_of[zone] = trans
            if u > best_u:
                best_u, best_p = u, p
        if VALUE_TAIL and max(gains.values()) < TAIL_THRESH:
            v = self.value_table(f, after_base)
            best_q, vt_p = -math.inf, best_p
            for zone, (moves, _, p) in best_for_zone.items():
                q = sum(g + VT_GAMMA * (float(np.interp(x2, self._vx, v)) if x2 is not None else 0.0)
                        for g, x2 in trans_of[zone]) / len(trans_of[zone]) - 1e-4 * moves
                if q > best_q:
                    best_q, vt_p = q, p
            return vt_p
        if TAIL_DEPTH and max(gains.values()) < TAIL_THRESH:
            best_v, tail_p = -math.inf, best_p
            for zone, (moves, _, p) in best_for_zone.items():
                v = self.deep(f, after_base, xl, zone, TAIL_DEPTH, TAIL_K) - 1e-4 * moves
                if v > best_v:
                    best_v, tail_p = v, p
            if best_v > 1e-9:  # only override when the search actually finds damage
                return tail_p
        return best_p

    def policy_rollout(self, f: Field, snap: Any, x: float, zone: int, ang: float, depth: int, rng: random.Random) -> tuple[float, float]:
        """(damage, time) of `depth` paddle cycles: first with (zone, ang), later ones greedy by nominal 1-step rate with a sampled angle."""
        f.restore(snap)
        dmg = tim = 0.0
        for i in range(depth):
            if i:
                snap = f.snapshot()
                best, zone = -math.inf, 0
                for z in (-1, 0, 1):
                    f.restore(snap)
                    x2, t2, h, b, _, _ = fly(f, x, f.y_pad, *zone_velocity(f, z))
                    rate = (h + W_BREAK * b) / max(t2 if x2 is not None else 20.0, CYCLE_FLOOR)
                    if rate > best:
                        best, zone = rate, z
                f.restore(snap)
                c, hh = 45.0 * zone, (CFG["alpha"] if zone == 0 else CFG["beta"])
                ang = tri_ppf(rng.random(), c, hh)  # common random numbers: same draws for every candidate zone
            x2, t2, hits, breaks, _, _ = fly(f, x, f.y_pad, *zone_velocity(f, zone, ang))
            dmg += hits + W_BREAK * breaks
            tim += max(t2 if x2 is not None else 20.0, CYCLE_FLOOR)
            if x2 is None:
                break
            x = x2
        return dmg, tim

    def value_table(self, f: Field, snap: Any) -> np.ndarray:
        """V(x): expected discounted damage over future cycles when the ball lands at x, bricks frozen at `snap`."""
        key = (tuple(snap[0].tolist()), tuple(snap[1].tolist()))
        if self._vt_key == key:
            return self._vt
        xs = np.linspace(f.r, f.W - f.r, VT_GRID)
        G = np.zeros((VT_GRID, 3, VT_K))
        X2 = np.zeros((VT_GRID, 3, VT_K))
        for i, x in enumerate(xs):
            for zi, zone in enumerate((-1, 0, 1)):
                for si, ang in enumerate(angles(zone, VT_K)):
                    f.restore(snap)
                    zvx, zvy = zone_velocity(f, zone, ang)
                    x2, _, hits, breaks, _, _ = fly(f, float(x), f.y_pad, zvx, zvy)
                    G[i, zi, si] = hits + W_BREAK * breaks
                    X2[i, zi, si] = x2 if x2 is not None else x
        f.restore(snap)
        V = np.zeros(VT_GRID)
        for _ in range(VT_ITERS):
            V = (G + VT_GAMMA * np.interp(X2, xs, V)).mean(axis=2).max(axis=1)
        self._vx, self._vt, self._vt_key = xs, V, key
        return V

    def deep(self, f: Field, snap: Any, x: float, zone: int, depth: int, k: int) -> float:
        """Mean damage (hits + W_BREAK*breaks, discounted) of a paddle contact in `zone` at x, then the best zone each later cycle."""
        tot = 0.0
        for ang in angles(zone, k):
            f.restore(snap)
            zvx, zvy = zone_velocity(f, zone, ang)
            x2, _, hits, breaks, _, _ = fly(f, x, f.y_pad, zvx, zvy)
            v = hits + W_BREAK * breaks
            if depth > 1 and x2 is not None:
                s2 = f.snapshot()
                v += TAIL_GAMMA * max(self.deep(f, s2, x2, z2, depth - 1, 1) for z2 in (-1, 0, 1))
            tot += v
        return tot / k

    def second_cycle(self, f: Field, x: float, centres: list) -> float:
        """Best zone's mean utility of the next paddle cycle from landing point x (state after the first cycle)."""
        s1 = f.snapshot()
        best = -math.inf
        for zone in (-1, 0, 1):
            tot = 0.0
            for ang in angles(zone, K2):
                f.restore(s1)
                zvx, zvy = zone_velocity(f, zone, ang)
                x2, t2, hits, breaks, _, path = fly(f, x, f.y_pad, zvx, zvy)
                cyc = max(t2 if x2 is not None else 20.0, CYCLE_FLOOR)
                near = min((math.hypot(qx - cx, qy - cy) for qx, qy in path for cx, cy in centres), default=0.0)
                tot += (hits + W_BREAK * breaks) / cyc - W_NEAR * near
            best = max(best, tot / K2)
        f.restore(s1)
        return best

    async def deliberate(self) -> dict[str, Any] | None:
        st = self.current_state
        if not st or st.get("game_over"):
            return None
        if self._tracker is None:
            self._tracker = PaddleTracker(st["paddle_width"], st["width"])
        tick = round(st["time_elapsed"] * 30)
        px, lat = self._tracker.observe(tick, st["paddle_x"])
        key = (lat, st["ball_vx"], st["ball_vy"], len(st["bricks"]), sum(b["hits_remaining"] for b in st["bricks"]),
               st["lives"], st["stage"])
        if key != self._key:
            self._key = key
            self._target = self.plan(st, px, lat)
        if self._target is None:  # no usable plan: just follow the ball
            diff = st["ball_x"] - st["paddle_width"] / 2.0 - px
        else:
            diff = self._target - px
        move = 0 if abs(diff) < 14.0 else (1 if diff > 0 else -1)
        self._tracker.record(tick, move)
        if not move:
            return None
        return {"action": "move", "direction": "EAST" if diff > 0 else "WEST"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Arkanoid student agent")
    parser.add_argument("--server", default="ws://localhost:8765/ws")
    parser.add_argument("--name", default="Group 30 agent")
    parser.add_argument("--id", default=None)
    args = parser.parse_args()
    asyncio.run(StudentAgent(args.server, args.name, args.id or AGENT_ID).run())


if __name__ == "__main__":
    main()
