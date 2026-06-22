"""
12_four_components: Four-component scene inference in 64x64 image.

Components:
  z_hi   = [x_H, y_H, x_I, y_I]              (4) — Flow/NSF
  z_bye  = [x_B, y_B, x_Y, y_Y, x_E, y_E]   (6) — Flow/NSF
  z_dot  = [cx, cy, r]                        (3) — GaussianFullCov
  z_bg   = [log_bg]                           (1) — GaussianFullCov

x = 64x64 image with words 'HI' + 'BYE' + circle + Poisson noise at background level.
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


def _draw_word(img, theta, word, font, S):
    draw = ImageDraw.Draw(img)
    to_px = lambda v: (v + 1) / 2 * S
    for i, letter in enumerate(word):
        x, y = to_px(theta[2 * i]), to_px(theta[2 * i + 1])
        try:
            bbox = draw.textbbox((0, 0), letter, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(letter, font=font)
        draw.text((x - w / 2, y - h / 2), letter, fill=200, font=font)


class FourCompSimulator:
    """Render HI + BYE + circle into a 64x64 image."""

    def __init__(self, image_size=64, font_size=14, noise_level=3.0):
        self.S = image_size
        self.noise_level = noise_level
        self._font = _try_font(font_size)

    def simulate(self, z_hi, z_bye, z_dot, z_bg):
        S = self.S
        to_px = lambda v: (v + 1) / 2 * S
        img = Image.new("L", (S, S), 0)
        _draw_word(img, z_hi,  "HI",  self._font, S)
        _draw_word(img, z_bye, "BYE", self._font, S)
        draw = ImageDraw.Draw(img)
        cx, cy = to_px(z_dot[0]), to_px(z_dot[1])
        r = (z_dot[2] + 1) / 2 * (S / 5) + 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=220, width=2)
        arr = np.array(img, dtype=np.float32)
        bg = np.exp(z_bg[0] * 2) * 3
        rate = (arr + bg).clip(0)
        return np.random.poisson(rate).astype(np.float32)


class SceneCNN(nn.Module):
    """Shared CNN backbone for 64x64 → feature embedding."""

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


class GlobalStats(nn.Module):
    """Tiny embedding: global image statistics for background inference."""

    def __init__(self, out_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.LazyLinear(32), nn.ReLU(), nn.Linear(32, out_dim))

    def forward(self, x):
        xf = x.float().reshape(x.shape[0], -1)
        q = torch.stack([xf.mean(-1), xf.std(-1),
                         xf.quantile(0.05, dim=-1), xf.quantile(0.95, dim=-1)], dim=-1)
        return self.mlp(q)
