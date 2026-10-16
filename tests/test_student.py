"""Tests for the student agent: forward model vs engine, latency tracker, and a legal action on a fresh state."""

import asyncio
import random
import unittest

from latency import STEP, PaddleTracker
from model import Field, fly
from server.config import load_config
from server.highscores import HighScoreManager
from server.logic import Arkanoid
from student import StudentAgent


def new_game(seed: int = 1, level: str = "level1.json") -> Arkanoid:
    cfg = load_config("config.json")
    cfg.game.level_sequence = [level]
    random.seed(seed)
    return Arkanoid(cfg, high_score_manager=HighScoreManager(None))


class TestModel(unittest.TestCase):
    def test_landing_matches_engine(self) -> None:
        for seed, level in [(1, "level1.json"), (2, "level2.json"), (3, "level3.json")]:
            g = new_game(seed, level)
            hit: dict = {}
            orig = g._handle_paddle_bounce
            g._handle_paddle_bounce = lambda x, orig=orig, hit=hit: (hit.setdefault("x", x), orig(x))[1]
            st = g.get_state()
            x_pred, *_ = fly(Field(st), st["ball_x"], st["ball_y"], st["ball_vx"], st["ball_vy"])
            for _ in range(600):
                if hit:
                    break
                d = st["ball_x"] - g.paddle_width / 2 - g.paddle_x
                if abs(d) > 14:
                    g.move_paddle("EAST" if d > 0 else "WEST")
                g.update(1 / 30)
                st = g.get_state()
            self.assertTrue(hit, "ball never reached the paddle")
            self.assertAlmostEqual(x_pred, hit["x"], delta=1.0)


class TestTracker(unittest.TestCase):
    def test_infers_delay(self) -> None:
        for true_lat in (0, 1, 3, 6):
            tr = PaddleTracker(84.0, 600.0)
            px, moves = 258.0, {}
            inflight = {}
            for tick in range(40):
                px = min(516.0, max(0.0, px + inflight.get(tick, 0) * STEP))  # a move decided at t is visible in the observation of tick t+1+lat
                eff, lat = tr.observe(tick, px)
                move = 1 if (tick // 5) % 2 == 0 else -1
                tr.record(tick, move)
                inflight[tick + true_lat + 1] = move
            self.assertEqual(lat, true_lat)


class TestAgent(unittest.TestCase):
    def test_returns_legal_action(self) -> None:
        agent = StudentAgent("ws://none", "t", "t")
        g = new_game()
        for _ in range(50):
            agent.current_state = g.get_state()
            a = asyncio.run(agent.deliberate())
            self.assertTrue(a is None or (a["action"] == "move" and a["direction"] in ("WEST", "EAST")))
            if a:
                g.move_paddle(a["direction"])
            g.update(1 / 30)


if __name__ == "__main__":
    unittest.main()
