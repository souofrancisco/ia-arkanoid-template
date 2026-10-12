"""Dummy autonomous agent that randomly selects from valid actions."""

import argparse
import asyncio
import random
from typing import Any

from agents.base_agent import BaseAgent

AGENT_ID: str | None = None  # your id (e.g. "a12345"); --id overrides it; None = random


class DummyAgent(BaseAgent):
    """Agent that randomly chooses between valid actions or idling."""

    async def deliberate(self) -> dict[str, Any] | None:
        """Sample uniformly from valid moves provided in current state."""
        if not self.current_state or self.current_state.get("game_over"):
            return None

        valid_actions: list[dict[str, Any]] = (
            self.current_state.get("valid_actions")
            or self.current_state.get("actions")
            or []
        )
        if not valid_actions:
            return None

        # Randomly choose a valid move or do nothing (stay idle)
        choices: list[dict[str, Any] | None] = [*valid_actions, None]
        return random.choice(choices)


def main() -> None:
    """Run Dummy Agent against a game server."""
    parser = argparse.ArgumentParser(description="Arkanoid dummy agent")
    parser.add_argument("--server", default="ws://localhost:8765/ws", help="WebSocket URI")
    parser.add_argument("--name", default="Dummy Agent", help="Agent name shown in the high score table")
    parser.add_argument("--id", default=None, help="Agent id, used to rank it (default: AGENT_ID, else random)")
    args = parser.parse_args()
    asyncio.run(DummyAgent(args.server, args.name, args.id or AGENT_ID).run())


if __name__ == "__main__":
    main()
