import asyncio
import unittest

from agents.base_agent import BaseAgent
from agents.dummy_agent import DummyAgent
from agents.manual_agent import receive_loop, send_loop


class ConcreteBaseAgent(BaseAgent):
    """Subclass implementing deliberate for testing BaseAgent."""

    async def deliberate(self) -> dict[str, str] | None:
        return {"action": "move", "direction": "WEST"}


class TestAgents(unittest.IsolatedAsyncioTestCase):
    async def test_base_agent_deliberate_not_implemented(self) -> None:
        agent = BaseAgent()
        self.assertEqual(agent.server_uri, "ws://localhost:8765/ws")
        self.assertIsNone(agent.player_id)
        self.assertIsNone(agent.setup_data)
        with self.assertRaises(NotImplementedError):
            await agent.deliberate()

        concrete = ConcreteBaseAgent()
        concrete_act = await concrete.deliberate()
        self.assertEqual(concrete_act, {"action": "move", "direction": "WEST"})

    async def test_dummy_agent_deliberation(self) -> None:
        agent = DummyAgent()

        # No state
        action = await agent.deliberate()
        self.assertIsNone(action)

        # Game over state
        agent.current_state = {"game_over": True, "valid_actions": [{"action": "move", "direction": "WEST"}]}
        action = await agent.deliberate()
        self.assertIsNone(action)

        # Empty valid actions
        agent.current_state = {"game_over": False, "valid_actions": []}
        action = await agent.deliberate()
        self.assertIsNone(action)

        # State with valid actions
        valid_moves = [
            {"action": "move", "direction": "WEST"},
            {"action": "move", "direction": "EAST"},
        ]
        agent.current_state = {"game_over": False, "valid_actions": valid_moves}

        # Run several times to test that choices come from valid_actions or None
        sampled_actions = set()
        for _ in range(50):
            chosen = await agent.deliberate()
            if chosen is not None:
                self.assertIn(chosen, valid_moves)
                sampled_actions.add(chosen["direction"])
            else:
                sampled_actions.add("NONE")

        # Expect to see valid choices sampled
        self.assertTrue(len(sampled_actions) > 0)

    async def test_base_agent_run_loop(self) -> None:
        agent = ConcreteBaseAgent()
        messages = [
            '{"type": "setup", "player_id": 42}',
            '{"type": "state", "game_over": false, "valid_actions": [{"action": "move", "direction": "WEST"}]}',
        ]
        sent_messages: list[str] = []

        class MockWebSocket:
            async def send(self, msg: str) -> None:
                sent_messages.append(msg)

            async def __aiter__(self):
                for m in messages:
                    yield m

        class MockConnectContext:
            async def __aenter__(self):
                return MockWebSocket()

            async def __aexit__(self, *args: object) -> None:
                return None

        import unittest.mock as mock

        with mock.patch("websockets.connect", return_value=MockConnectContext()):
            await agent.run()

        self.assertEqual(agent.player_id, 42)
        self.assertIsNotNone(agent.setup_data)
        assert agent.setup_data is not None
        self.assertEqual(agent.setup_data.get("player_id"), 42)
        # Verify action was sent
        self.assertEqual(len(sent_messages), 2)  # handshake + action

    async def test_base_agent_stops_on_game_over(self) -> None:
        agent = ConcreteBaseAgent()
        messages = [
            '{"type": "setup", "player_id": 1}',
            '{"type": "state", "game_over": true}',
            '{"type": "state", "game_over": false}',  # Should never be reached
        ]
        sent_messages: list[str] = []

        class MockWebSocket:
            async def send(self, msg: str) -> None:
                sent_messages.append(msg)

            async def __aiter__(self):
                for m in messages:
                    yield m

        class MockConnectContext:
            async def __aenter__(self):
                return MockWebSocket()

            async def __aexit__(self, *args: object) -> None:
                return None

        import unittest.mock as mock

        with mock.patch("websockets.connect", return_value=MockConnectContext()):
            await agent.run()

        # Only initial handshake sent; no move action after game over
        self.assertEqual(len(sent_messages), 1)
        self.assertIn("agent", sent_messages[0])

    async def test_base_agent_stops_on_game_won(self) -> None:
        agent = ConcreteBaseAgent()
        messages = [
            '{"type": "setup", "player_id": 1}',
            '{"type": "state", "game_won": true, "game_over": true}',
        ]
        sent_messages: list[str] = []

        class MockWebSocket:
            async def send(self, msg: str) -> None:
                sent_messages.append(msg)

            async def __aiter__(self):
                for m in messages:
                    yield m

        class MockConnectContext:
            async def __aenter__(self):
                return MockWebSocket()

            async def __aexit__(self, *args: object) -> None:
                return None

        import unittest.mock as mock

        with mock.patch("websockets.connect", return_value=MockConnectContext()):
            await agent.run()

        self.assertEqual(len(sent_messages), 1)

    async def test_manual_agent_receive_loop(self) -> None:
        state_json = (
            '{"type": "state", "score": 200, "lives": 3, "high_score": 500, '
            '"paddle_x": 100.0, "ball_x": 120.0, "ball_y": 300.0, '
            '"level": 1, "max_levels": 3, "bricks": []}'
        )

        class MockWebSocket:
            async def __aiter__(self):
                yield '{"type": "setup", "player_id": 10}'
                yield state_json
                yield '{"type": "state", "game_won": true, "game_over": true}'

            async def send(self, msg: str) -> None:
                pass

        stop_event = asyncio.Event()
        await receive_loop(MockWebSocket(), stop_event)
        self.assertTrue(stop_event.is_set())

        # Test send loop exits when stop_event is set
        await send_loop(MockWebSocket(), stop_event)


class TestAgentName(unittest.TestCase):
    def test_name_stored_and_validated(self) -> None:
        self.assertEqual(DummyAgent(name="  ann ").name, "ann")
        with self.assertRaises(ValueError):
            DummyAgent(name="  ")

    def test_id_given_or_generated(self) -> None:
        self.assertEqual(DummyAgent(agent_id=" a1 ").agent_id, "a1")
        auto = DummyAgent().agent_id
        self.assertEqual(len(auto), 32)
        self.assertNotEqual(auto, DummyAgent().agent_id)


if __name__ == "__main__":
    unittest.main()
