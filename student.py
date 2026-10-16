"""Track 2 - heuristic contact-zone steering, no simulation.

Lands the ball with the wall-reflection predictor, then picks the paddle third that points the ball toward the
weighted centre of the remaining destructible bricks (left / centre / right of the landing point).
"""

import argparse
import asyncio
from typing import Any

from agents.base_agent import BaseAgent

AGENT_ID: str | None = None  # TODO(group): set to a group member's mechanographic number before submission
STEP = 28.0
DEAD_BAND = 40.0  # px between landing point and target below which the ball is sent straight up


def landing_x(bx: float, by: float, vx: float, vy: float, st: dict[str, Any]) -> float:
    r = st["ball_radius"]
    left, right = r, st["width"] - r
    if vy < 0:
        ceil = st["ceiling_y"] + r
        bx += vx * (by - ceil) / -vy
        by, vy = ceil, -vy
    total = bx + vx * max(0.0, (st["paddle_y"] - r - by) / vy) if vy > 1e-9 else bx
    span = right - left
    u = (total - left) % (2 * span)
    return left + (u if u <= span else 2 * span - u)


def target_x(st: dict[str, Any]) -> float | None:
    """Weighted (by hits left) centre of the live destructible bricks."""
    live = [b for b in st["bricks"] if not b["indestructible"]]
    if not live:
        return None
    w = sum(b["hits_remaining"] for b in live)
    return sum((b["x"] + b["width"] / 2) * b["hits_remaining"] for b in live) / w


class StudentAgent(BaseAgent):
    async def deliberate(self) -> dict[str, Any] | None:
        st = self.current_state
        if not st or st.get("game_over"):
            return None
        pw = st["paddle_width"]
        land = landing_x(st["ball_x"], st["ball_y"], st["ball_vx"], st["ball_vy"], st) if st["ball_vy"] else st["ball_x"]
        tgt = target_x(st)
        zone_off = 0.0  # offset of the paddle centre from the landing point
        if tgt is not None and abs(tgt - land) > DEAD_BAND:
            # right third sends the ball right: ball must land on the right part of the paddle, so the paddle sits left
            zone_off = -pw / 3 if tgt > land else pw / 3
        diff = land + zone_off - pw / 2.0 - st["paddle_x"]
        if abs(diff) < STEP / 2:
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
