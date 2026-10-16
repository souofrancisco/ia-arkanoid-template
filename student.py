"""Track 3 - deterministic one-cycle model-predictive control.

Every time the ball's trajectory changes, the agent simulates (model.py) the flight to the paddle line, then for
each paddle contact zone that is reachable in time simulates the next flight with the nominal bounce angle, scores
the outcome (brick damage per second) and walks the paddle to the lattice position that produces the best zone.
"""

import argparse
import asyncio
import math
from typing import Any

from agents.base_agent import BaseAgent
from model import Field, fly, move_costs, zone_velocity

AGENT_ID: str | None = None  # TODO(group): set to a group member's mechanographic number before submission
W_BREAK = 0.5  # extra utility for finishing a brick (on top of 1 per hit)
W_NEAR = 0.002  # utility per px of closeness of the flight to the nearest live destructible brick
CYCLE_FLOOR = 1.0  # seconds; keeps the rate finite for very short flights


class StudentAgent(BaseAgent):
    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self._key: tuple | None = None
        self._target: float | None = None

    def plan(self, st: dict[str, Any]) -> float | None:
        f = Field(st)
        bx, by, vx, vy = st["ball_x"], st["ball_y"], st["ball_vx"], st["ball_vy"]
        xl, tl, *_ = fly(f, bx, by, vx, vy)
        if xl is None:
            return None
        after_base = f.snapshot()
        budget = max(1, math.ceil(tl * 30.0 - 1e-9))
        px, pw = st["paddle_x"], st["paddle_width"]
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
            f.restore(after_base)
            zvx, zvy = zone_velocity(f, zone)
            x2, t2, hits, breaks, _, path = fly(f, xl, f.y_pad, zvx, zvy)
            cycle = max(t2 if x2 is not None else 20.0, CYCLE_FLOOR)
            near = min((math.hypot(qx - cx, qy - cy) for qx, qy in path for cx, cy in centres), default=0.0)
            u = (hits + W_BREAK * breaks) / cycle - W_NEAR * near - 1e-4 * moves
            if u > best_u:
                best_u, best_p = u, p
        return best_p

    async def deliberate(self) -> dict[str, Any] | None:
        st = self.current_state
        if not st or st.get("game_over"):
            return None
        key = (st["ball_vx"], st["ball_vy"], len(st["bricks"]), sum(b["hits_remaining"] for b in st["bricks"]),
               st["lives"], st["stage"])
        if key != self._key:
            self._key = key
            self._target = self.plan(st)
        if self._target is None:  # no usable plan: just follow the ball
            diff = st["ball_x"] - st["paddle_width"] / 2.0 - st["paddle_x"]
        else:
            diff = self._target - st["paddle_x"]
        if abs(diff) < 14.0:
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
