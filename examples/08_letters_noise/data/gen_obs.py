"""Generate mock observation for 08_letters_noise."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from model import ImageSimulator

z_pos_true   = np.array([-0.3, -0.2,  0.4, 0.3], dtype=np.float32)
z_noise_true = np.array([0.3, -0.2], dtype=np.float32)   # moderate brightness, low noise

sim = ImageSimulator(text="HI", image_size=32, font_size=12)
obs = sim.simulate(z_pos_true, z_noise_true)
out = os.path.join(os.path.dirname(__file__), 'obs.npz')
np.savez(out, x=obs, z_pos_true=z_pos_true, z_noise_true=z_noise_true)
print(f"Saved {out} — image shape {obs.shape}")
