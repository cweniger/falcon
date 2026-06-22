"""Generate mock observation for 10_scene."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from model import SceneSimulator

z_word_true  = np.array([-0.4, -0.3, 0.3, -0.3], dtype=np.float32)
z_dot_true   = np.array([0.2,  0.5,  0.3], dtype=np.float32)
z_noise_true = np.array([0.0], dtype=np.float32)

sim = SceneSimulator(word="HI", image_size=64, font_size=14, noise_level=4.0)
obs = sim.simulate(z_word_true, z_dot_true, z_noise_true)
out = os.path.join(os.path.dirname(__file__), 'obs.npz')
np.savez(out, x=obs, z_word_true=z_word_true, z_dot_true=z_dot_true, z_noise_true=z_noise_true)
print(f"Saved {out} — shape {obs.shape}, max {obs.max():.1f}")
