"""Generate mock observation: 'H' at (-0.3, -0.2), 'I' at (0.4, 0.3)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from model import LetterSimulator

rng = np.random.default_rng(42)
z_true = np.array([-0.3, -0.2, 0.4, 0.3], dtype=np.float32)
sim = LetterSimulator(text="HI", image_size=32, font_size=12, noise_level=5.0)
obs = sim.simulate(z_true)
np.savez(os.path.join(os.path.dirname(__file__), 'obs.npz'), x=obs, z_true=z_true)
print("Saved obs.npz — z_true:", z_true, "image shape:", obs.shape)
