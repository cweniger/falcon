"""
07_letters: Infer positions of two letters 'HI' in a 32x32 grayscale image.

z = [x1, y1, x2, y2]  in [-1, 1]^4  (normalized pixel positions)
x = noisy image of H and I placed at those positions
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


def _draw_letters(theta, text="HI", image_size=32, font_size=12):
    """Draw letters at positions given by theta in [-1,1]^(2*len(text))."""
    theta = np.asarray(theta, dtype=float)
    to_px = lambda v: (v + 1) / 2 * image_size

    img = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for i, letter in enumerate(text):
        x = to_px(theta[2 * i])
        y = to_px(theta[2 * i + 1])
        try:
            bbox = draw.textbbox((0, 0), letter, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(letter, font=font)
        draw.text((x - w / 2, y - h / 2), letter, fill=200, font=font)

    return np.array(img, dtype=np.float32)


class LetterSimulator:
    """Forward model: draw 'HI' at positions given by z, add Poisson noise."""

    def __init__(self, text="HI", image_size=32, font_size=12, noise_level=5.0):
        self.text = text
        self.image_size = image_size
        self.font_size = font_size
        self.noise_level = noise_level

    def simulate(self, z):
        img = _draw_letters(z, self.text, self.image_size, self.font_size)
        noisy = img + np.random.poisson(self.noise_level, img.shape).astype(np.float32)
        return noisy


class SimpleImageEmbedding(nn.Module):
    """Small CNN + MLP to embed 32x32 image into a feature vector."""

    def __init__(self, image_size=32, out_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                          # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                          # 8x8
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                  # 4x4
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        x = x.float().unsqueeze(1) / 255.0           # (B, 1, H, W)
        h = self.cnn(x)
        return self.mlp(h)
