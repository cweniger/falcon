import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch

def simulate_word(theta, word="FALCON", image_size=128, font_size=20):
    """
    Create a grayscale image with the given word's letters placed according to
    len(word) (x, y) coordinate pairs in theta ∈ [-1, 1]^(2 * len(word)).

    Parameters
    ----------
    theta : array-like, shape (2 * len(word),)
        Flat array of normalized coordinates in [-1, 1].
    word : str
        The word to draw (default: "FALCON").
    image_size : int
        Size (width = height) of the output image in pixels.
    font_size : int
        Font size for the letters.

    Returns
    -------
    np.ndarray
        Grayscale numpy array of shape (image_size, image_size).
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2 * len(word),):
        raise ValueError(f"theta must have shape ({2 * len(word)},), got {theta.shape}.")

    # Map [-1, 1] → [0, image_size)
    to_px = lambda v: (v + 1) / 2 * image_size

    # Create blank grayscale image
    img = Image.new("L", (image_size, image_size), color=0)
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Draw each character
    for i, letter in enumerate(word):
        x = to_px(theta[2 * i])
        y = to_px(theta[2 * i + 1])

        # Determine text size (handle Pillow versions)
        try:
            bbox = draw.textbbox((0, 0), letter, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(letter, font=font)

        draw.text((x - w / 2, y - h / 2), letter, fill=255, font=font)

    return np.array(img)



def simulate_snake(theta, image_size=128, step_size=6, radius=2, blur_sigma=0.0):
    """
    Draws a 32-circle 'snake' on a 128x128 grayscale image.

    Parameters
    ----------
    theta : array-like of shape (31,)
        Each entry in (-pi, pi) is the heading for the next step.
        theta[0] is the direction from circle 0 to circle 1, etc.
        Angle convention: 0 = +x (right), +pi/2 = up (CCW).
    image_size : int
        Image width/height in pixels.
    step_size : int
        Distance between consecutive circle centers (default 12 px).
    radius : int
        Circle radius in pixels (default 6 px, so step_size=2*radius).
    blur_sigma : float
        If > 0, applies Gaussian blur with this radius.

    Returns
    -------
    img_np : np.ndarray, shape (image_size, image_size), dtype=uint8
        Grayscale image with the snake drawn (0 background, 255 foreground).
    centers : np.ndarray, shape (32, 2)
        Float array of (x, y) centers of all 32 circles.
    """

    # Start in the image center
    cx = cy = image_size / 2.0
    centers = [(cx, cy)]

    # Generate all subsequent centers from headings
    ang = 0
    for delta_ang in theta:
        ang += delta_ang
        dx = step_size * np.sin(ang)
        dy = -step_size * np.cos(ang)  # minus: y grows downward in images
        cx += dx
        cy += dy
        centers.append((cx, cy))

    # Draw
    img = Image.new("L", (image_size, image_size), color=0)
    draw = ImageDraw.Draw(img)
    for (x, y) in centers:
        # Circle bounding box
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, fill=255)

    if blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    return np.array(img), np.array(centers, dtype=float)



class DrawFalconLetters:
    def __init__(self, text = "FALCON", font_size=20, image_size=128):
        self.font_size = font_size
        self.image_size = image_size
        self.text = text

    def simulate(self, positions):
        img = simulate_word(positions, self.text, image_size=self.image_size, font_size=self.font_size)
        return img.astype(np.float32)
    
class DrawSnake:
    def __init__(self, step_size=6, radius=2, image_size=128, blur_sigma=0.0):
        self.step_size = step_size
        self.radius = radius
        self.image_size = image_size
        self.blur_sigma = blur_sigma

    def simulate(self, headings):
        img, centers = simulate_snake(
            headings,
            image_size=self.image_size,
            step_size=self.step_size,
            radius=self.radius,
            blur_sigma=self.blur_sigma,
        )
        return img.astype(np.float32)
    
class SimulateData:
    def __init__(self, background_level=1000):
        self.background_level = background_level

    def simulate(self, *imgs):
        img = sum(imgs) + self.background_level
        x = np.random.poisson(img / 10)
        return x.astype(np.float32)
    

class SimulatePositions:
    def simulate(self, size=12):
        return np.random.uniform(-1, 1, size).astype(np.float32)
    
class SimulateAngles:
    def simulate(self, size=31):
        return np.random.uniform(-1.5, 1.5).astype(np.float32)



class E(torch.nn.Module):
    def __init__(self, log_prefix=None):
        super().__init__()
        from falcon.contrib.norms import LazyOnlineNorm

        #self.norm = LazyOnlineNorm(momentum=5e-3)
        self.linear = torch.nn.LazyLinear(100)
        self.log_prefix = (log_prefix + ":") if log_prefix else ""

    def forward(self, x, *args):
        #x = self.norm(x).float()
        x = x.float()/100
        x = self.linear(x.flatten(start_dim=1))
        return x