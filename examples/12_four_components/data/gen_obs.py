"""Generate observation for 12_four_components."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from model import FourCompSimulator

z_hi_true  = np.array([-0.55, -0.5,  0.05, -0.5],            dtype=np.float32)
z_bye_true = np.array([-0.6,   0.3,  0.0,   0.3, 0.6,  0.3], dtype=np.float32)
z_dot_true = np.array([0.5,  -0.2,   0.2],                    dtype=np.float32)
z_bg_true  = np.array([0.0],                                   dtype=np.float32)

sim = FourCompSimulator(image_size=64, font_size=14, noise_level=3.0)
obs = sim.simulate(z_hi_true, z_bye_true, z_dot_true, z_bg_true)
out = os.path.join(os.path.dirname(__file__), 'obs.npz')
np.savez(out, x=obs, z_hi_true=z_hi_true, z_bye_true=z_bye_true,
         z_dot_true=z_dot_true, z_bg_true=z_bg_true)
print(f"Saved {out}  shape={obs.shape}  max={obs.max():.1f}")
