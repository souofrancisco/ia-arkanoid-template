"""Base agent client for communicating with AI Game Framework servers."""

import json
import logging
import uuid
from typing import Any

import websockets

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - AGENT - %(levelname)s - %(message)s"
)


def resolve_agent_id(agent_id: str | None = None) -> str:
    """Return the trimmed agent id, or a random one (uuid4 hex) when it is None or empty."""
    agent_id = (agent_id or "").strip()
    return agent_id or uuid.uuid4().hex


class BaseAgent:
    """Abstract base agent handling WebSocket communication with the game server."""

    def __init__(
        self,
        server_uri: str = "ws://localhost:8765/ws",
        name: str = "Agent",
        agent_id: str | None = None,
    ) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Agent name must not be empty")
        self.server_uri = server_uri
        self.name = name
        self.agent_id = resolve_agent_id(agent_id)
        self.current_state: dict[str, Any] | None = None
        self.player_id: int | None = None
        self.setup_data: dict[str, Any] | None = None

    async def run(self) -> None:
        """Connect to game server and run the observation-action loop."""
        try:
            async with websockets.connect(self.server_uri) as websocket:
                await websocket.send(json.dumps({"client": "agent", "name": self.name, "id": self.agent_id}))
                logging.info("Connected to %s as %s (id %s)", self.server_uri, self.name, self.agent_id)

                async for message in websocket:
                    data = json.loads(message)

                    msg_type = data.get("type")
                    if msg_type == "setup":
                        self.player_id = data.get("player_id")
                        self.setup_data = data
                        logging.info("Assigned player ID: %s", self.player_id)
                        continue

                    if msg_type in ("state", "update"):
                        self.current_state = data
                        if data.get("game_won"):
                            logging.info("Game won! Victory achieved. Stopping agent.")
                            break
                        if data.get("game_over"):
                            logging.info("Game over received. Stopping agent.")
                            break

                        action = await self.deliberate()
                        if action is not None:
                            await websocket.send(json.dumps(action))

        except Exception as e:
            logging.error("Connection error: %s", e)

    async def deliberate(self) -> dict[str, Any] | None:
        """Analyze current state and decide on the next action."""
        raise NotImplementedError("Subclasses must implement deliberate()")
