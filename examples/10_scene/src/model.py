"""
10_scene: Three-component scene inference.

A 64x64 image contains three independent components:
  z_word  = [x_H, y_H, x_I, y_I]   ~ Uniform(-1,1)^4  — positions of 'HI'
  z_dot   = [cx, cy, r]              ~ Uniform(-1,1)^3  — circle center + radius
  z_noise = [log_bg]                 ~ Uniform(-1,1)^1  — background level

Three parallel networks, each using a small CNN + GaussianFullCov.
The image is 64x64 to give more room for the three components to coexist.
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


class WordSimulator:
    """Prior + simulator for word positions."""

    def simulate(self, size=4):
        return np.random.uniform(-1, 1, size).astype(np.float32)


class DotSimulator:
    """Prior + simulator for circle: [cx, cy, r] in [-1,1]."""

    def simulate(self, size=3):
        return np.random.uniform(-1, 1, size).astype(np.float32)


class BgSimulator:
    """Prior for background log-level: scalar in [-1,1]."""

    def simulate(self, size=1):
        return np.random.uniform(-1, 1, size).astype(np.float32)


class SceneSimulator:
    """Render 'HI' + circle + background noise into a 64x64 image."""

    def __init__(self, word="HI", image_size=64, font_size=14, noise_level=4.0):
        self.word = word
        self.image_size = image_size
        self.font_size = font_size
        self.noise_level = noise_level
        self._font = _try_font(font_size)

    def simulate(self, z_word, z_dot, z_noise):
        S = self.image_size
        to_px = lambda v: (v + 1) / 2 * S

        img = Image.new("L", (S, S), 0)
        draw = ImageDraw.Draw(img)

        # Draw word letters
        for i, letter in enumerate(self.word):
            x = to_px(z_word[2 * i])
            y = to_px(z_word[2 * i + 1])
            try:
                bbox = draw.textbbox((0, 0), letter, font=self._font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                w, h = draw.textsize(letter, font=self._font)
            draw.text((x - w / 2, y - h / 2), letter, fill=200, font=self._font)

        # Draw circle
        cx, cy = to_px(z_dot[0]), to_px(z_dot[1])
        r = (z_dot[2] + 1) / 2 * (S / 6) + 2   # radius in [2, S/6+2] pixels
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=220, width=2)

        arr = np.array(img, dtype=np.float32)

        # Background level
        bg = np.exp(z_noise[0] * 2) * 3     # in [~0.4, ~22]
        rate = arr + bg
        noisy = np.random.poisson(rate.clip(0)).astype(np.float32)
        return noisy


class SceneCNN(nn.Module):
    """CNN embedding for 64x64 input."""

    def __init__(self, out_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),  # 4
        )
        self.mlp = nn.Sequential(nn.Flatten(), nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(self, x):
        return self.mlp(self.cnn(x.float().unsqueeze(1) / 255.0))


class BgCNN(nn.Module):
    """Lightweight embedding for background level (global image statistics)."""

    def __init__(self, out_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.LazyLinear(32), nn.ReLU(), nn.Linear(32, out_dim))

    def forward(self, x):
        xf = x.float().reshape(x.shape[0], -1)
        feats = torch.stack([xf.mean(-1), xf.std(-1),
                             xf.quantile(0.1, dim=-1), xf.quantile(0.9, dim=-1)], dim=-1)
        return self.mlp(feats)
