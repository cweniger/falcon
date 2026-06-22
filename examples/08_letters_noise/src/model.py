"""
08_letters_noise: Two-component inference from a noisy 'HI' image.

Node 1: z_pos = [x_H, y_H, x_I, y_I] ~ Uniform(-1,1)^4  (letter positions)
Node 2: z_noise = [log_brightness, log_noise] ~ N(0,1)^2  (rendering params)

x = Poisson(brightness * render(z_pos) + noise_floor)

Two separate GaussianFullCov networks observe the same image x.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


def _draw_letters(theta, text="HI", image_size=32, font_size=12):
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


class LetterPosSimulator:
    """Prior over letter positions: Uniform(-1,1)^4."""

    def simulate(self, size=4):
        return np.random.uniform(-1, 1, size).astype(np.float32)


class NoisePrior:
    """Prior over rendering params: N(0,1)^2.

    z_noise[0] = log_brightness  in [-1,1] -> brightness in exp([-1,1])
    z_noise[1] = log_noise       in [-1,1] -> noise floor in exp([-1,1])
    """

    def simulate(self, size=2):
        return np.random.uniform(-1, 1, size).astype(np.float32)


class ImageSimulator:
    """Combine positions and noise params to render an image."""

    def __init__(self, text="HI", image_size=32, font_size=12):
        self.text = text
        self.image_size = image_size
        self.font_size = font_size

    def simulate(self, z_pos, z_noise):
        brightness = np.exp(z_noise[0] * 2)      # in [e^-2, e^2] ~ [0.14, 7.4]
        noise_floor = np.exp(z_noise[1] * 2) * 3  # noise baseline
        img = _draw_letters(z_pos, self.text, self.image_size, self.font_size)
        rate = brightness * img + noise_floor
        noisy = np.random.poisson(rate.clip(0)).astype(np.float32)
        return noisy


class PosEmbedding(nn.Module):
    """CNN embedding for position network."""

    def __init__(self, out_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.mlp = nn.Sequential(nn.Flatten(), nn.LazyLinear(64), nn.ReLU(), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.mlp(self.cnn(x.float().unsqueeze(1) / 255.0))


class NoiseEmbedding(nn.Module):
    """Lightweight embedding for noise parameter network (global stats of image)."""

    def __init__(self, out_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LazyLinear(32), nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        x = x.float()
        # Summary statistics: mean, std, max, fraction above threshold
        mean = x.mean(dim=(-2, -1), keepdim=False) if x.dim() == 3 else x.mean(dim=-1)
        # flatten then compute stats
        xf = x.reshape(x.shape[0], -1)
        feats = torch.stack([xf.mean(-1), xf.std(-1), xf.max(-1).values,
                             (xf > xf.mean(-1, keepdim=True)).float().mean(-1)], dim=-1)
        return self.mlp(feats)
