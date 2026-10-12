import os
import unittest

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame

from viewer.sprites import SpriteManager, generate_default_spritesheet
from viewer.viewer import ArkanoidViewer


def setUpModule() -> None:
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((600, 800))


def tearDownModule() -> None:
    pygame.quit()


class TestViewer(unittest.TestCase):

    def test_generate_spritesheet_and_manager(self) -> None:
        test_path = "assets/test_sprites.png"
        if os.path.exists(test_path):
            os.remove(test_path)

        generate_default_spritesheet(test_path)
        self.assertTrue(os.path.exists(test_path))

        manager = SpriteManager(sheet_path=test_path)
        self.assertIn("frost", manager.brick_sprites)
        self.assertIn("aurora_green", manager.brick_sprites)
        self.assertIn("metal", manager.brick_sprites)
        self.assertIsNotNone(manager.life_sprite)
        self.assertIsNotNone(manager.ball_sprite)

        # Scaling test
        brick_surf = manager.get_brick_sprite("frost", 60.0, 25.0)
        self.assertEqual(brick_surf.get_width(), 60)
        self.assertEqual(brick_surf.get_height(), 25)

        paddle_surf = manager.get_paddle_sprite(90.0, 15.0)
        self.assertEqual(paddle_surf.get_width(), 90)
        self.assertEqual(paddle_surf.get_height(), 15)

        ball_surf = manager.get_ball_sprite(7.0)
        self.assertEqual(ball_surf.get_width(), 14)
        self.assertEqual(ball_surf.get_height(), 14)

        if os.path.exists(test_path):
            os.remove(test_path)

    def test_viewer_rendering_headless(self) -> None:
        viewer = ArkanoidViewer(width=600, height=800)
        viewer.init_pygame()
        self.assertIsNotNone(viewer.screen)
        self.assertIsNotNone(viewer.font)

        # Render disconnected frame
        viewer.connected = False
        viewer.render()

        # Render active game frame with Nord brick states
        viewer.connected = True
        mock_state = {
            "score": 150,
            "high_score": 300,
            "stage": 1,
            "stage_name": "Stage 1 - Test Assault",
            "lives": 3,
            "paddle_x": 250.0,
            "paddle_y": 740.0,
            "paddle_width": 84.0,
            "paddle_height": 14.0,
            "ball_x": 292.0,
            "ball_y": 730.0,
            "ball_radius": 6.0,
            "ceiling_y": 82.0,
            "bricks": [
                {
                    "x": 100.0,
                    "y": 100.0,
                    "width": 50.0,
                    "height": 20.0,
                    "type": "frost",
                    "hits_remaining": 1,
                    "palette": "frost",
                    "color": "#88C0D0",
                },
                {
                    "x": 160.0,
                    "y": 100.0,
                    "width": 50.0,
                    "height": 20.0,
                    "type": "aurora_green",
                    "hits_remaining": 2,
                    "palette": "aurora",
                    "color": "#A3BE8C",
                },
                {
                    "x": 220.0,
                    "y": 100.0,
                    "width": 50.0,
                    "height": 20.0,
                    "type": "metal",
                    "hits_remaining": -1,
                    "palette": "snow_storm",
                    "color": "#D8DEE9",
                },
            ],
            "game_won": False,
            "game_over": False,
        }
        with viewer.state_lock:
            viewer.latest_state = mock_state

        viewer.render()

        # Render game won overlay
        mock_state["game_won"] = True
        viewer.render()

        # Render game over overlay
        mock_state["game_won"] = False
        mock_state["game_over"] = True
        viewer.render()


if __name__ == "__main__":
    unittest.main()
