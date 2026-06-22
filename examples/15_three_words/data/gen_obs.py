"""Generate observation for 15_three_words."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from model import ThreeWordSimulator

# Place words in three distinct vertical bands
z_hi_true  = np.array([-0.6, -0.55, -0.05, -0.55], dtype=np.float32)
z_bye_true = np.array([-0.65, 0.05,  0.0,   0.05, 0.65,  0.05], dtype=np.float32)
z_sbi_true = np.array([-0.65, 0.55,  0.0,   0.55, 0.65,  0.55], dtype=np.float32)

sim = ThreeWordSimulator(image_size=64, font_size=13, noise_level=3.5)
obs = sim.simulate(z_hi_true, z_bye_true, z_sbi_true)
out = os.path.join(os.path.dirname(__file__), 'obs.npz')
np.savez(out, x=obs, z_hi_true=z_hi_true, z_bye_true=z_bye_true, z_sbi_true=z_sbi_true)
print(f"Saved {out}  shape={obs.shape}  max={obs.max():.1f}")
