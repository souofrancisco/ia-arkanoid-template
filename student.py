"""Track 4 - stochastic rollout planning (expected utility over the bounce-angle distribution).

Every time the ball's trajectory changes, the agent simulates (model.py) the flight to the paddle line, then for
each paddle contact zone that is reachable in time simulates the next flight with the nominal bounce angle, scores
the outcome (brick damage per second) and walks the paddle to the lattice position that produces the best zone.
"""

import argparse
import asyncio
import math
from typing import Any

from agents.base_agent import BaseAgent
from latency import PaddleTracker
from model import Field, fly, move_costs, zone_velocity

CFG = {"side": 45.0, "alpha": 8.0, "beta": 15.0}  # config.json paddle.angles (documented in the assignment)


def tri_ppf(u: float, c: float, h: float) -> float:
    """Inverse CDF of the symmetric triangular distribution on [c-h, c+h] with mode c."""
    return c - h + h * math.sqrt(2 * u) if u < 0.5 else c + h - h * math.sqrt(2 * (1 - u))


def angles(zone: int, k: int | None = None) -> list[float]:
    k = k or K_SAMPLES
    c, h = 45.0 * zone, (CFG["alpha"] if zone == 0 else CFG["beta"])
    return [tri_ppf((i + 0.5) / k, c, h) for i in range(k)]

AGENT_ID: str | None = None  # TODO(group): set to a group member's mechanographic number before submission
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
        self._last = 0

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
        best_u, best_p = -math.inf, None
        for zone, (moves, _, p) in best_for_zone.items():
            u = 0.0
            for ang in angles(zone):
                f.restore(after_base)
                zvx, zvy = zone_velocity(f, zone, ang)
                x2, t2, hits, breaks, _, path = fly(f, xl, f.y_pad, zvx, zvy)
                cycle = max(t2 if x2 is not None else 20.0, CYCLE_FLOOR)
                near = min((math.hypot(qx - cx, qy - cy) for qx, qy in path for cx, cy in centres), default=0.0)
                u += (hits + W_BREAK * breaks) / cycle - W_NEAR * near
                if HORIZON >= 2 and x2 is not None:
                    u += GAMMA * self.second_cycle(f, x2, centres)
            u = u / K_SAMPLES - 1e-4 * moves
            if u > best_u:
                best_u, best_p = u, p
        return best_p

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
        px, lat = self._tracker.update(st["paddle_x"], self._last)
        key = (lat, st["ball_vx"], st["ball_vy"], len(st["bricks"]), sum(b["hits_remaining"] for b in st["bricks"]),
               st["lives"], st["stage"])
        if key != self._key:
            self._key = key
            self._target = self.plan(st, px, lat)
        if self._target is None:  # no usable plan: just follow the ball
            diff = st["ball_x"] - st["paddle_width"] / 2.0 - px
        else:
            diff = self._target - px
        self._last = 0 if abs(diff) < 14.0 else (1 if diff > 0 else -1)
        if not self._last:
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
