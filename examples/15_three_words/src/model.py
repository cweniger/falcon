"""
15_three_words: Three parallel NSF Flow networks in a shared 64x64 image.

Each of three words ('HI', 'BYE', 'SBI') has its own network inferring
letter positions from the composite image — pushing composite inference
to 4+6+6 = 16 total parameters.

z_hi   = [x_H, y_H, x_I, y_I]              (4) — NSF Flow
z_bye  = [x_B, y_B, x_Y, y_Y, x_E, y_E]   (6) — NSF Flow
z_sbi  = [x_S, y_S, x_B, y_B, x_I, y_I]   (6) — NSF Flow
x      = 64x64 composite of all three words + Poisson noise
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


def _draw_word(img, theta, word, font, S, fill=200):
    draw = ImageDraw.Draw(img)
    to_px = lambda v: (v + 1) / 2 * S
    for i, letter in enumerate(word):
        x, y = to_px(theta[2 * i]), to_px(theta[2 * i + 1])
        try:
            bbox = draw.textbbox((0, 0), letter, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(letter, font=font)
        draw.text((x - w / 2, y - h / 2), letter, fill=fill, font=font)


class ThreeWordSimulator:
    """Render 'HI', 'BYE', 'SBI' into a shared 64x64 image with Poisson noise."""

    def __init__(self, image_size=64, font_size=13, noise_level=3.5):
        self.S = image_size
        self.noise_level = noise_level
        self._font = _try_font(font_size)

    def simulate(self, z_hi, z_bye, z_sbi):
        img = Image.new("L", (self.S, self.S), 0)
        _draw_word(img, z_hi,  "HI",  self._font, self.S, fill=190)
        _draw_word(img, z_bye, "BYE", self._font, self.S, fill=210)
        _draw_word(img, z_sbi, "SBI", self._font, self.S, fill=230)
        arr = np.array(img, dtype=np.float32)
        return (arr + np.random.poisson(self.noise_level, arr.shape)).astype(np.float32)


class WordCNN(nn.Module):
    """CNN embedding: 64x64 → feature vector."""

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
