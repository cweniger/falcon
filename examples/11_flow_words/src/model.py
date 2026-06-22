"""
11_flow_words: Two-word inference with NSF Flow estimators on 64x64 images.

Revisits 09_two_words with:
  - 64x64 image (more room for letters)
  - NSF normalizing-flow estimators (handles non-Gaussian posteriors)
  - Hypercube prior (required by Flow)
  - Larger buffer (4096 samples)

z_hi  = [x_H, y_H, x_I, y_I]              ~ Uniform(-1,1)^4
z_bye = [x_B, y_B, x_Y, y_Y, x_E, y_E]   ~ Uniform(-1,1)^6
x     = 64x64 noisy composite image
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


def _draw_word(img, theta, word, font, image_size):
    draw = ImageDraw.Draw(img)
    to_px = lambda v: (v + 1) / 2 * image_size
    for i, letter in enumerate(word):
        x = to_px(theta[2 * i]);  y = to_px(theta[2 * i + 1])
        try:
            bbox = draw.textbbox((0, 0), letter, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(letter, font=font)
        draw.text((x - w / 2, y - h / 2), letter, fill=200, font=font)


class TwoWordSimulator:
    """Render 'HI' and 'BYE' into a shared 64x64 image with Poisson noise."""

    def __init__(self, word1="HI", word2="BYE", image_size=64, font_size=14, noise_level=4.0):
        self.word1 = word1
        self.word2 = word2
        self.image_size = image_size
        self.noise_level = noise_level
        self._font = _try_font(font_size)

    def simulate(self, z_hi, z_bye):
        img = Image.new("L", (self.image_size, self.image_size), 0)
        _draw_word(img, z_hi,  self.word1, self._font, self.image_size)
        _draw_word(img, z_bye, self.word2, self._font, self.image_size)
        arr = np.array(img, dtype=np.float32)
        return (arr + np.random.poisson(self.noise_level, arr.shape)).astype(np.float32)


class WordCNN(nn.Module):
    """CNN embedding for 64x64 image."""

    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8
            nn.AdaptiveAvgPool2d(4),                                        # 4x4
            nn.Flatten(),
            nn.LazyLinear(128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x.float().unsqueeze(1) / 255.0)
