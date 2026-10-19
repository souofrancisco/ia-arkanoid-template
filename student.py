"""Arkanoid agent: stochastic rollout planner with an end-game value function.

Every time the ball's trajectory changes, the agent simulates (model.py, an exact forward model of the published
physics) the flight to the paddle line. For each paddle contact zone (left / centre / right third) that is reachable in
time it simulates the next flight for K quantiles of the zone's bounce-angle distribution and scores the expected
brick damage per second. When no zone is expected to hit anything it picks the zone by a value function over the next
landing point (value iteration on a small grid with frozen bricks). The paddle then walks to the lattice position
(28 px steps) that makes the ball land in the chosen zone. latency.py infers any action delay from the agent's own
moves and netloop.py answers only the newest state, so a slow decision never builds a backlog.
"""

import argparse
import asyncio
import logging
import math
from typing import Any

import numpy as np

from agents.base_agent import BaseAgent
from latency import PaddleTracker
from model import Field, fly, move_costs, zone_velocity
from netloop import run_latest

# False = the template's BaseAgent.run (answers every queued state; collapses if a decision takes > ~33 ms)
LATEST_LOOP = True
AGENT_ID: str | None = None  # TODO(group): set to a group member's mechanographic number before submission

ANGLES = {"side": 45.0, "alpha": 8.0, "beta": 15.0}  # config.json paddle.angles (centre +-alpha, sides side +-beta)
K_SAMPLES = 9  # stratified quantiles of the triangular bounce-angle distribution per zone
W_BREAK = 0.5  # extra utility for finishing a brick (on top of 1 per hit)
CYCLE_FLOOR = 1.0  # seconds; keeps the rate finite for very short flights
TAIL_THRESH = 1.0  # expected damage below which the value function replaces the rate utility
VT_GRID = 13  # landing-x grid points of the value table
VT_K = 3  # angle samples per (x, zone) in the value table
VT_ITERS = 8
VT_GAMMA = 0.85


def tri_ppf(u: float, c: float, h: float) -> float:
    """Inverse CDF of the symmetric triangular distribution on [c-h, c+h] with mode c."""
    return c - h + h * math.sqrt(2 * u) if u < 0.5 else c + h - h * math.sqrt(2 * (1 - u))


def angles(zone: int, k: int = K_SAMPLES) -> list[float]:
    c, h = ANGLES["side"] * zone, (ANGLES["alpha"] if zone == 0 else ANGLES["beta"])
    return [tri_ppf((i + 0.5) / k, c, h) for i in range(k)]


class StudentAgent(BaseAgent):
    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self._key: tuple | None = None
        self._target: float | None = None
        self._tracker: PaddleTracker | None = None
        self._vt_key: tuple | None = None
        self._vt: np.ndarray | None = None
        self._vx: np.ndarray | None = None
        self._counts = np.zeros(VT_GRID, dtype=np.float64)
        self._last_hp = -1

    async def run(self) -> None:
        await (run_latest(self) if LATEST_LOOP else super().run())

    def zone_options(self, xl: float, tl: float, px: float, lat: int, pw: float, f: Field) -> dict[int, tuple]:
        """Per reachable zone: (moves, centring error, paddle x) of the cheapest paddle position that lands in it."""
        budget = max(1, math.ceil(tl * 30.0 - 1e-9) - lat)
        best: dict[int, tuple[int, float, float]] = {}
        for p, moves in move_costs(px, pw, f.W).items():
            rel = xl - p
            if moves > budget or rel < -f.r or rel > pw + f.r:
                continue
            zone = -1 if rel < pw / 3 else (1 if rel > 2 * pw / 3 else 0)
            cand = (moves, abs(rel - pw * (zone + 1) / 3 - pw / 6), p)
            if zone not in best or cand < best[zone]:
                best[zone] = cand
        return best

    def plan(self, st: dict[str, Any], px: float, lat: int) -> float | None:
        """Paddle x to walk to, or None when there is no usable plan (the caller then follows the ball)."""
        cur_hp = sum(b["hits_remaining"] for b in st["bricks"] if not b["indestructible"])
        if self._last_hp == -1 or cur_hp < self._last_hp:
            self._last_hp = cur_hp
            self._counts.fill(0.0)

        f = Field(st)
        xl, tl, *_ = fly(f, st["ball_x"], st["ball_y"], st["ball_vx"], st["ball_vy"])
        if xl is None:
            return None
        after_base = f.snapshot()
        options = self.zone_options(xl, tl, px, lat, st["paddle_width"], f)
        if not options:
            return None
        best_u, best_p = -math.inf, None
        gains: dict[int, float] = {}
        landings: dict[int, list[tuple[float, float | None]]] = {}
        for zone, (moves, _, p) in options.items():
            u = g = 0.0
            outcomes = []
            for ang in angles(zone):
                f.restore(after_base)
                x2, t2, hits, breaks, *_ = fly(f, xl, f.y_pad, *zone_velocity(f, zone, ang))
                cycle = max(t2 if x2 is not None else 20.0, CYCLE_FLOOR)
                u += (hits + W_BREAK * breaks) / cycle
                g += hits + W_BREAK * breaks
                outcomes.append((hits + W_BREAK * breaks, x2))
            u = u / K_SAMPLES - 1e-4 * moves
            gains[zone], landings[zone] = g / K_SAMPLES, outcomes
            if u > best_u:
                best_u, best_p = u, p
        if max(gains.values()) < TAIL_THRESH:
            v_raw = self.value_table(f, after_base)
            idx = int(np.clip(np.round((xl - f.r) / ((f.W - 2 * f.r) / (VT_GRID - 1))), 0, VT_GRID - 1))
            self._counts[idx] += 1.0
            v_eff = v_raw / (1.0 + self._counts)
            best_q, tail_p = -math.inf, best_p
            for zone, (moves, _, p) in options.items():
                q = sum(g + VT_GAMMA * (float(np.interp(x2, self._vx, v_eff)) if x2 is not None else 0.0)
                        for g, x2 in landings[zone]) / K_SAMPLES - 1e-4 * moves
                if q > best_q:
                    best_q, tail_p = q, p
            return tail_p
        self._counts.fill(0.0)
        return best_p

    def value_table(self, f: Field, snap: Any) -> np.ndarray:
        """V(x): expected discounted damage over future cycles when the ball lands at x, bricks frozen at `snap`."""
        key = (tuple(snap[0].tolist()), tuple(snap[1].tolist()))
        if self._vt_key == key and self._vt is not None:
            return self._vt
        xs = np.linspace(f.r, f.W - f.r, VT_GRID)
        gain = np.zeros((VT_GRID, 3, VT_K))
        nxt = np.zeros((VT_GRID, 3, VT_K))
        for i, x in enumerate(xs):
            for zi, zone in enumerate((-1, 0, 1)):
                for si, ang in enumerate(angles(zone, VT_K)):
                    f.restore(snap)
                    x2, _, hits, breaks, _, _ = fly(f, float(x), f.y_pad, *zone_velocity(f, zone, ang))
                    gain[i, zi, si] = hits + W_BREAK * breaks
                    nxt[i, zi, si] = x2 if x2 is not None else x
        f.restore(snap)
        v = np.zeros(VT_GRID)
        for _ in range(VT_ITERS):
            v = (gain + VT_GAMMA * np.interp(nxt, xs, v)).mean(axis=2).max(axis=1)
        self._vx, self._vt, self._vt_key = xs, v, key
        return v

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
            try:
                self._target = self.plan(st, px, lat)
            except Exception:  # a planner bug on an unseen map must not kill the run: follow the ball instead
                logging.exception("plan failed; following the ball")
                self._target = None
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
