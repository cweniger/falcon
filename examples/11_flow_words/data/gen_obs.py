"""Generate mock observation for 11_flow_words."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from model import TwoWordSimulator

z_hi_true  = np.array([-0.5, -0.45, 0.05, -0.45], dtype=np.float32)
z_bye_true = np.array([-0.55, 0.35, 0.0, 0.35, 0.55, 0.35], dtype=np.float32)

sim = TwoWordSimulator(word1="HI", word2="BYE", image_size=64, font_size=14, noise_level=4.0)
obs = sim.simulate(z_hi_true, z_bye_true)
out = os.path.join(os.path.dirname(__file__), 'obs.npz')
np.savez(out, x=obs, z_hi_true=z_hi_true, z_bye_true=z_bye_true)
print(f"Saved {out}  shape={obs.shape}  max={obs.max():.1f}")
