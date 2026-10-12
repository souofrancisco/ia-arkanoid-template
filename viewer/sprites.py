"""Sprite generator and manager for Arkanoid Pygame viewer using Nord palette."""

import os

import pygame

# --- Nord Color Scheme Constants ---
NORD_POLAR_DARK = (46, 52, 64)  # nord0 (Dark background)
NORD_POLAR_LINE = (76, 86, 106)  # nord3 (Dividers / ceiling)

NORD_SNOW_DARK = (216, 222, 233)  # nord4 (Metal shadow / highlights)
NORD_SNOW_BRIGHT = (236, 239, 244)  # nord6 (Paddle highlight / ball)

NORD_FROST_TEAL = (143, 188, 187)  # nord7 (Teal highlight)
NORD_FROST_ICE = (136, 192, 208)  # nord8 (1 hit left base)
NORD_FROST_DEEP = (94, 129, 172)  # nord10 (Shadow)

NORD_AURORA_RED = (191, 97, 106)  # nord11 (5+ hits)
NORD_AURORA_ORANGE = (208, 135, 112)  # nord12 (4 hits)
NORD_AURORA_YELLOW = (235, 203, 139)  # nord13 (3 hits)
NORD_AURORA_GREEN = (163, 190, 140)  # nord14 (2 hits)


def _mix(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Linear blend of two RGB colors (t=0 -> c1, t=1 -> c2)."""
    return (
        round(c1[0] + (c2[0] - c1[0]) * t),
        round(c1[1] + (c2[1] - c1[1]) * t),
        round(c1[2] + (c2[2] - c1[2]) * t),
    )


def _gradient_block(
    surf: pygame.Surface, rect: pygame.Rect, top: tuple[int, int, int], bottom: tuple[int, int, int]
) -> None:
    """Fill rect with a vertical gradient."""
    for i in range(rect.height):
        t = i / max(1, rect.height - 1)
        pygame.draw.line(surf, _mix(top, bottom, t), (rect.x, rect.y + i), (rect.right - 1, rect.y + i))


def _make_brick(base: tuple[int, int, int], metal: bool = False) -> pygame.Surface:
    """Draw a 50x20 brick: dark outline, gradient body, bevel, gloss strip, and tier-specific detail."""
    w, h = 50, 20
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    outline = _mix(base, NORD_POLAR_DARK, 0.65)
    light = _mix(base, NORD_SNOW_BRIGHT, 0.45)
    dark = _mix(base, NORD_POLAR_DARK, 0.35)

    pygame.draw.rect(surf, outline, (0, 0, w, h), border_radius=2)
    body = pygame.Rect(1, 1, w - 2, h - 2)
    _gradient_block(surf, body, _mix(base, NORD_SNOW_BRIGHT, 0.18), _mix(base, NORD_POLAR_DARK, 0.18))
    # Bevel: light top/left, dark bottom/right
    pygame.draw.line(surf, light, (2, 1), (w - 3, 1))
    pygame.draw.line(surf, light, (1, 2), (1, h - 3))
    pygame.draw.line(surf, dark, (2, h - 2), (w - 2, h - 2))
    pygame.draw.line(surf, dark, (w - 2, 2), (w - 2, h - 3))

    if metal:
        # Riveted steel plate with brushed hatch lines and diagonal sheen
        for x in range(6, w - 6, 4):
            pygame.draw.line(surf, _mix(base, NORD_POLAR_LINE, 0.18), (x, 5), (x, h - 6))
        for cx, cy in ((5, 4), (w - 6, 4), (5, h - 5), (w - 6, h - 5)):
            pygame.draw.rect(surf, NORD_POLAR_LINE, (cx - 1, cy - 1, 3, 3))
            pygame.draw.rect(surf, NORD_SNOW_BRIGHT, (cx - 1, cy - 1, 2, 2))
        pygame.draw.line(surf, NORD_SNOW_BRIGHT, (20, 3), (28, h - 4), 2)
    else:
        # Glossy highlight strip and inner shine dots
        pygame.draw.rect(surf, _mix(base, NORD_SNOW_BRIGHT, 0.5), (4, 3, w - 12, 2), border_radius=1)
        pygame.draw.rect(surf, NORD_SNOW_BRIGHT, (w - 7, 3, 2, 2))
    return surf


def _make_paddle() -> pygame.Surface:
    """Draw the 84x14 Vaus-style paddle: capped ends, steel body, frost core."""
    w, h = 84, 14
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(surf, NORD_POLAR_DARK, (0, 0, w, h), border_radius=6)
    steel = _mix(NORD_SNOW_DARK, NORD_POLAR_LINE, 0.5)
    _gradient_block(surf, pygame.Rect(3, 1, w - 6, h - 2), NORD_SNOW_BRIGHT, steel)
    # End caps in Aurora red with highlight
    for x0 in (0, w - 12):
        pygame.draw.rect(surf, NORD_AURORA_RED, (x0, 0, 12, h), border_radius=6)
        shade = _mix(NORD_AURORA_RED, NORD_POLAR_DARK, 0.4)
        pygame.draw.rect(surf, shade, (x0, h - 3, 12, 3), border_bottom_left_radius=6, border_bottom_right_radius=6)
        pygame.draw.rect(surf, (216, 139, 146), (x0 + 3, 2, 6, 2), border_radius=1)
    # Center frost core
    pygame.draw.rect(surf, NORD_POLAR_DARK, (28, 2, 28, 10), border_radius=3)
    _gradient_block(surf, pygame.Rect(29, 3, 26, 8), NORD_FROST_ICE, NORD_FROST_DEEP)
    pygame.draw.line(surf, NORD_FROST_TEAL, (31, 4), (52, 4))
    return surf


def _make_ball() -> pygame.Surface:
    """Draw the 14x14 pixel-art ball: chamfered block with radial-style shading."""
    surf = pygame.Surface((14, 14), pygame.SRCALPHA)
    rows = [(5, 9), (3, 11), (2, 12), (1, 13), (1, 13), (0, 14), (0, 14)]
    rows += rows[::-1]
    for y, (x0, x1) in enumerate(rows):
        pygame.draw.line(surf, NORD_POLAR_DARK, (x0, y), (x1 - 1, y))
    inner = [(5, 9), (3, 11), (2, 12), (2, 12), (1, 13), (1, 13), (1, 13), (1, 13), (2, 12), (2, 12), (3, 11), (5, 9)]
    for i, (x0, x1) in enumerate(inner):
        y = i + 1
        t = i / (len(inner) - 1)
        pygame.draw.line(surf, _mix(NORD_SNOW_BRIGHT, NORD_FROST_ICE, t), (x0, y), (x1 - 1, y))
    pygame.draw.rect(surf, (255, 255, 255), (4, 3, 4, 2))
    pygame.draw.rect(surf, (255, 255, 255), (3, 5, 2, 2))
    return surf


def _make_life() -> pygame.Surface:
    """Draw the 28x8 life icon: miniature paddle."""
    surf = pygame.Surface((28, 8), pygame.SRCALPHA)
    pygame.draw.rect(surf, NORD_POLAR_DARK, (0, 0, 28, 8), border_radius=3)
    _gradient_block(surf, pygame.Rect(2, 1, 24, 6), NORD_SNOW_BRIGHT, NORD_SNOW_DARK)
    for x0 in (0, 22):
        pygame.draw.rect(surf, NORD_AURORA_RED, (x0 + 1, 1, 5, 6), border_radius=2)
    pygame.draw.rect(surf, NORD_FROST_DEEP, (10, 2, 8, 4))
    return surf


def generate_default_spritesheet(path: str = "assets/sprites.png") -> None:
    """Generate the pixel-art sprite sheet using an adjusted Nord palette."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pygame.init()

    # Spritesheet canvas: 256x256 with alpha channel
    sheet = pygame.Surface((256, 256), pygame.SRCALPHA)

    # Bricks (50x20), by hits remaining: frost=1, green=2, yellow=3, orange=4, red=5+, metal=indestructible
    brick_definitions = [
        (0, NORD_FROST_ICE, False),
        (24, NORD_AURORA_GREEN, False),
        (48, NORD_AURORA_YELLOW, False),
        (72, NORD_AURORA_ORANGE, False),
        (96, NORD_AURORA_RED, False),
        (120, NORD_SNOW_DARK, True),
    ]
    for y_pos, base_col, is_metal in brick_definitions:
        sheet.blit(_make_brick(base_col, metal=is_metal), (0, y_pos))

    sheet.blit(_make_paddle(), (0, 145))
    sheet.blit(_make_ball(), (0, 165))
    sheet.blit(_make_life(), (24, 168))

    pygame.image.save(sheet, path)


class SpriteManager:
    """Loads and caches Arkanoid sprites from the generated spritesheet."""

    def __init__(self, sheet_path: str = "assets/sprites.png") -> None:
        self.sheet_path = sheet_path
        if not os.path.exists(sheet_path):
            generate_default_spritesheet(sheet_path)

        img = pygame.image.load(sheet_path)
        try:
            self.spritesheet = img.convert_alpha()
        except pygame.error:
            self.spritesheet = img

        self.brick_sprites: dict[str, pygame.Surface] = {
            "frost": self._subsurface(0, 0, 50, 20),
            "aurora_green": self._subsurface(0, 24, 50, 20),
            "aurora_yellow": self._subsurface(0, 48, 50, 20),
            "aurora_orange": self._subsurface(0, 72, 50, 20),
            "aurora_red": self._subsurface(0, 96, 50, 20),
            "metal": self._subsurface(0, 120, 50, 20),
            # Backwards compatibility and palette aliases
            "blue": self._subsurface(0, 0, 50, 20),  # frost
            "green": self._subsurface(0, 24, 50, 20),  # aurora_green
            "yellow": self._subsurface(0, 48, 50, 20),  # aurora_yellow
            "orange": self._subsurface(0, 72, 50, 20),  # aurora_orange
            "red": self._subsurface(0, 96, 50, 20),  # aurora_red
            "snow_storm": self._subsurface(0, 120, 50, 20),  # metal
        }
        self.paddle_sprite = self._subsurface(0, 145, 84, 14)
        self.ball_sprite = self._subsurface(0, 165, 14, 14)
        self.life_sprite = self._subsurface(24, 168, 28, 8)

    def _subsurface(self, x: int, y: int, width: int, height: int) -> pygame.Surface:
        rect = pygame.Rect(x, y, width, height)
        return self.spritesheet.subsurface(rect).copy()

    def get_brick_sprite(
        self, brick_type: str, width: float, height: float
    ) -> pygame.Surface:
        """Return appropriately scaled brick sprite."""
        surf = self.brick_sprites.get(brick_type, self.brick_sprites["frost"])
        w_int = int(round(width))
        h_int = int(round(height))
        if surf.get_width() != w_int or surf.get_height() != h_int:
            return pygame.transform.scale(surf, (w_int, h_int))
        return surf

    def get_paddle_sprite(self, width: float, height: float) -> pygame.Surface:
        """Return appropriately scaled paddle sprite."""
        w_int = int(round(width))
        h_int = int(round(height))
        if (
            self.paddle_sprite.get_width() != w_int
            or self.paddle_sprite.get_height() != h_int
        ):
            return pygame.transform.scale(self.paddle_sprite, (w_int, h_int))
        return self.paddle_sprite

    def get_ball_sprite(self, radius: float) -> pygame.Surface:
        """Return appropriately scaled non-round pixel block ball sprite."""
        diameter = int(round(radius * 2.0))
        if (
            self.ball_sprite.get_width() != diameter
            or self.ball_sprite.get_height() != diameter
        ):
            return pygame.transform.scale(self.ball_sprite, (diameter, diameter))
        return self.ball_sprite
