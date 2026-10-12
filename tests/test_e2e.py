import asyncio
import contextlib
import unittest

from aigf.main import serve_game_instance

from agents.dummy_agent import DummyAgent
from server.config import load_config
from server.server import ArkanoidGameServer


class TestE2E(unittest.IsolatedAsyncioTestCase):
    async def test_server_agent_e2e_integration(self) -> None:
        port = 8799
        cfg = load_config("config.json")
        server = ArkanoidGameServer(config=cfg)

        # Launch server task in background
        server_task = asyncio.create_task(
            serve_game_instance(server, host="127.0.0.1", port=port)
        )
        # Give server time to bind and listen
        await asyncio.sleep(0.2)

        # Launch dummy agent
        agent = DummyAgent(server_uri=f"ws://127.0.0.1:{port}/ws")
        agent_task = asyncio.create_task(agent.run())

        # Wait for agent to receive setup and initial states
        for _ in range(20):
            await asyncio.sleep(0.05)
            if agent.player_id is not None and agent.current_state is not None:
                break

        self.assertIsNotNone(agent.player_id)
        self.assertIsNotNone(agent.current_state)
        assert agent.current_state is not None
        self.assertIn("ball_y", agent.current_state)
        self.assertIn("paddle_x", agent.current_state)

        # Cancel agent and server
        agent_task.cancel()
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await agent_task
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


if __name__ == "__main__":
    unittest.main()
