"""
14_hierarchical: Hierarchical graph — global layout + local letter positions.

Graph structure:
  z_layout  ~ Uniform(-1,1)^2   — global (dx, dy) offset applied to all letters
  z_hi_rel  ~ Uniform(-1,1)^4   — 'HI' positions RELATIVE to layout offset
  z_bye_rel ~ Uniform(-1,1)^6   — 'BYE' positions RELATIVE to layout offset
  img_hi    = render('HI',  z_layout + z_hi_rel)   [intermediate node]
  img_bye   = render('BYE', z_layout + z_bye_rel)   [intermediate node]
  x         = composite(img_hi, img_bye) + noise     [observed]

Inference:
  z_layout  <-- x   (global context CNN)
  z_hi_rel  <-- x   (local CNN for 'HI' region)
  z_bye_rel <-- x   (local CNN for 'BYE' region)

This tests that Falcon correctly handles intermediate deterministic nodes
in a composite hierarchy.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


def _try_font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except IOError:
        return ImageFont.load_default()


def _draw_word_abs(img, abs_positions, word, font, S):
    """Draw letters of `word` at absolute pixel positions."""
    draw = ImageDraw.Draw(img)
    to_px = lambda v: (v + 1) / 2 * S
    for i, letter in enumerate(word):
        x, y = to_px(abs_positions[2 * i]), to_px(abs_positions[2 * i + 1])
        try:
            bbox = draw.textbbox((0, 0), letter, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(letter, font=font)
        draw.text((x - w / 2, y - h / 2), letter, fill=200, font=font)


class HiRenderer:
    """Render 'HI' with absolute positions = z_layout + z_hi_rel (clipped to [-1,1])."""

    def __init__(self, image_size=64, font_size=13):
        self.S = image_size
        self._font = _try_font(font_size)

    def simulate(self, z_layout, z_hi_rel):
        # Broadcast layout offset across all letter positions
        layout_broadcast = np.tile(z_layout, 2)     # (4,): [dx,dy, dx,dy]
        abs_pos = np.clip(z_hi_rel + layout_broadcast, -1, 1)
        img = Image.new("L", (self.S, self.S), 0)
        _draw_word_abs(img, abs_pos, "HI", self._font, self.S)
        return np.array(img, dtype=np.float32)


class ByeRenderer:
    """Render 'BYE' with absolute positions = z_layout + z_bye_rel (clipped)."""

    def __init__(self, image_size=64, font_size=13):
        self.S = image_size
        self._font = _try_font(font_size)

    def simulate(self, z_layout, z_bye_rel):
        layout_broadcast = np.tile(z_layout, 3)     # (6,)
        abs_pos = np.clip(z_bye_rel + layout_broadcast, -1, 1)
        img = Image.new("L", (self.S, self.S), 0)
        _draw_word_abs(img, abs_pos, "BYE", self._font, self.S)
        return np.array(img, dtype=np.float32)


class CompositeRenderer:
    """Merge img_hi and img_bye into a noisy composite."""

    def __init__(self, noise_level=4.0):
        self.noise_level = noise_level

    def simulate(self, img_hi, img_bye):
        combined = img_hi + img_bye + self.noise_level
        return np.random.poisson(combined.clip(0)).astype(np.float32)


class SceneCNN(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.LazyLinear(128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x.float().unsqueeze(1) / 255.0)
