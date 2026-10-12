# Vulture whitelist for dataclasses, configuration fields, and framework hooks.
import server.config

_cfg = server.config.AppConfig()
_ = _cfg.screen.width
_ = _cfg.screen.height
_ = _cfg.paddle.width
_ = _cfg.paddle.height
_ = _cfg.paddle.y
_ = _cfg.paddle.speed
_ = _cfg.paddle.step_distance
_ = _cfg.paddle.left_ratio
_ = _cfg.paddle.center_ratio
_ = _cfg.paddle.right_ratio
_ = _cfg.paddle.angles.alpha_deg
_ = _cfg.paddle.angles.side_deg
_ = _cfg.paddle.angles.beta_deg
_ = _cfg.ball.radius
_ = _cfg.ball.speed
_ = _cfg.ball.initial_speed
_ = _cfg.bricks.width
_ = _cfg.bricks.height
_ = _cfg.bricks.types
_btype = server.config.BrickTypeConfig()
_ = _btype.health
_ = _btype.indestructible
_ = _btype.points
_ = _btype.color
_ = _cfg.game.lives
_ = _cfg.game.fps
_ = _cfg.game.maps_dir
_ = _cfg.game.level_sequence
import server.server

_ = server.server.main
import agents.dummy_agent

_ = agents.dummy_agent.main
import agents.manual_agent

_ = agents.manual_agent.main
import viewer.viewer

_ = viewer.viewer.main
