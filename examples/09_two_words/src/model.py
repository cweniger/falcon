"""
09_two_words: Parallel inference of two words' positions in a shared image.

z_hi  = [x_H, y_H, x_I, y_I]  ~ Uniform(-1,1)^4   (positions of 'HI')
z_bye = [x_B, y_B, x_Y, y_Y, x_E, y_E]  ~ Uniform(-1,1)^6  (positions of 'BYE')
x = noisy composite 32x32 image containing both words

Each word has its own CNN-based GaussianFullCov network running in parallel.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


def _draw_word(img, theta, word, font, image_size=32, color=200):
    """Draw letters of `word` onto existing PIL image `img` in-place."""
    draw = ImageDraw.Draw(img)
    to_px = lambda v: (v + 1) / 2 * image_size
    for i, letter in enumerate(word):
        x = to_px(theta[2 * i])
        y = to_px(theta[2 * i + 1])
        try:
            bbox = draw.textbbox((0, 0), letter, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(letter, font=font)
        draw.text((x - w / 2, y - h / 2), letter, fill=color, font=font)


class WordPrior:
    """Uniform prior over letter positions for a given word."""

    def __init__(self, word="HI"):
        self.n_params = 2 * len(word)

    def simulate(self):
        return np.random.uniform(-1, 1, self.n_params).astype(np.float32)


class TwoWordSimulator:
    """Composite simulator: render 'HI' and 'BYE' into a shared 32x32 image."""

    def __init__(self, word1="HI", word2="BYE", image_size=32, font_size=11, noise_level=5.0):
        self.word1 = word1
        self.word2 = word2
        self.image_size = image_size
        self.font_size = font_size
        self.noise_level = noise_level
        try:
            self._font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except IOError:
            self._font = ImageFont.load_default()

    def simulate(self, z_hi, z_bye):
        img = Image.new("L", (self.image_size, self.image_size), 0)
        _draw_word(img, z_hi,  self.word1, self._font, self.image_size, color=180)
        _draw_word(img, z_bye, self.word2, self._font, self.image_size, color=220)
        arr = np.array(img, dtype=np.float32)
        noisy = arr + np.random.poisson(self.noise_level, arr.shape).astype(np.float32)
        return noisy


class WordEmbedding(nn.Module):
    """CNN embedding shared architecture for both word networks."""

    def __init__(self, out_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.mlp = nn.Sequential(nn.Flatten(), nn.LazyLinear(64), nn.ReLU(), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.mlp(self.cnn(x.float().unsqueeze(1) / 255.0))
