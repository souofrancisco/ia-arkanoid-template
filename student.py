"""Track 1 - predictive interception baseline.

Predicts where the ball crosses the paddle line (walls and ceiling only; bricks are ignored) and parks the paddle
centre there. MODE="track" degrades it to a pure ball follower, the floor of the comparison.
"""

import argparse
import asyncio
from typing import Any

from agents.base_agent import BaseAgent

AGENT_ID: str | None = None  # TODO(group): set to a group member's mechanographic number before submission
MODE = "intercept"  # "intercept" | "track"
STEP = 28.0


def landing_x(bx: float, by: float, vx: float, vy: float, st: dict[str, Any]) -> float:
    """x where the ball centre reaches the paddle contact height, unfolding wall and ceiling reflections."""
    r = st["ball_radius"]
    left, right = r, st["width"] - r
    y_hit = st["paddle_y"] - r
    if vy < 0:  # rises to the ceiling first, then comes down
        ceil = st["ceiling_y"] + r
        bx += vx * (by - ceil) / -vy
        by = ceil
        vy = -vy
    total_x = bx + vx * max(0.0, (y_hit - by) / vy) if vy > 1e-9 else bx
    span = right - left
    u = (total_x - left) % (2 * span)
    return left + (u if u <= span else 2 * span - u)


class StudentAgent(BaseAgent):
    """Moves one 28 px step per tick toward the position that centres the paddle on the landing point."""

    async def deliberate(self) -> dict[str, Any] | None:
        st = self.current_state
        if not st or st.get("game_over"):
            return None
        if MODE == "track" or st["ball_vy"] == 0:
            aim = st["ball_x"]
        else:
            aim = landing_x(st["ball_x"], st["ball_y"], st["ball_vx"], st["ball_vy"], st)
        diff = aim - st["paddle_width"] / 2.0 - st["paddle_x"]
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
