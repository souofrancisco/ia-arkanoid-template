"""Every map must be winnable: each destructible brick has to be reachable by the ball."""

import glob
import json
import math
import os
import tempfile
import unittest
from collections import deque

from server.config import load_config

STEP = 4.0


def unreachable_bricks(map_path: str) -> list[tuple[float, float]]:
    """Iteratively 'destroy' every brick the ball can touch; return the destructible ones never reached."""
    cfg = load_config()
    with open(map_path, encoding="utf-8") as f:
        bricks = json.load(f)["bricks"]
    r = cfg.ball.radius
    w, h = cfg.bricks.width, cfg.bricks.height
    x0, x1 = r, cfg.screen.width - r
    y0, y1 = cfg.screen.ceiling_y + r, cfg.paddle.y - r
    cols, rows = int((x1 - x0) / STEP) + 1, int((y1 - y0) / STEP) + 1

    def dist(cx: float, cy: float, b: dict) -> float:
        """Distance from a ball center to the brick rectangle."""
        return math.hypot(max(b["x"] - cx, 0.0, cx - b["x"] - w), max(b["y"] - cy, 0.0, cy - b["y"] - h))

    def blocked(cx: float, cy: float, live: list[dict]) -> bool:
        return any(dist(cx, cy, b) < r for b in live)

    live = list(bricks)
    while any(not b["indestructible"] for b in live):
        # Flood fill from the paddle line (ball launch area) over free ball-center positions.
        seen = set()
        queue = deque((c, rows - 1) for c in range(cols))
        seen.update(queue)
        while queue:
            c, rw = queue.popleft()
            for nc, nr in ((c + 1, rw), (c - 1, rw), (c, rw + 1), (c, rw - 1)):
                if (
                    0 <= nc < cols
                    and 0 <= nr < rows
                    and (nc, nr) not in seen
                    and not blocked(x0 + nc * STEP, y0 + nr * STEP, live)
                ):
                    seen.add((nc, nr))
                    queue.append((nc, nr))
        hit = []
        for b in live:
            if b["indestructible"]:
                continue
            # Ball touches the brick if a reachable position lies within STEP of its expanded box.
            if any(dist(x0 + c * STEP, y0 + rw * STEP, b) <= r + STEP for c, rw in seen):
                hit.append(b)
        if not hit:
            break
        live = [b for b in live if b not in hit]
    return [(b["x"], b["y"]) for b in live if not b["indestructible"]]


class TestMaps(unittest.TestCase):
    def test_all_destructible_bricks_reachable(self) -> None:
        maps = sorted(glob.glob("maps/*.json"))
        self.assertTrue(maps)
        for path in maps:
            with self.subTest(map=path):
                self.assertEqual(unreachable_bricks(path), [], f"{path} has unreachable destructible bricks")


    def test_detects_enclosed_brick(self) -> None:
        metal = {"type": "metal", "health": -1, "indestructible": True}
        soft = {"type": "red", "health": 1, "indestructible": False}
        bricks = [
            {"x": 275.0, "y": 90.0, **metal},
            {"x": 215.0, "y": 120.0, **metal},
            {"x": 275.0, "y": 120.0, **soft},
            {"x": 335.0, "y": 120.0, **metal},
            {"x": 275.0, "y": 150.0, **metal},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"bricks": bricks}, f)
        try:
            self.assertEqual(unreachable_bricks(f.name), [(275.0, 120.0)])
        finally:
            os.remove(f.name)


if __name__ == "__main__":
    unittest.main()
