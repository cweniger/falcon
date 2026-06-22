"""Generate observation for 14_hierarchical."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from model import HiRenderer, ByeRenderer, CompositeRenderer

# Global layout offset shifts all letters
z_layout_true  = np.array([0.1, -0.05], dtype=np.float32)
# Relative positions (added to layout)
z_hi_rel_true  = np.array([-0.55, -0.45,  0.05, -0.45], dtype=np.float32)
z_bye_rel_true = np.array([-0.6,   0.35,  0.0,   0.35, 0.6,  0.35], dtype=np.float32)

hi_sim   = HiRenderer(image_size=64, font_size=13)
bye_sim  = ByeRenderer(image_size=64, font_size=13)
comp_sim = CompositeRenderer(noise_level=4.0)

img_hi  = hi_sim.simulate(z_layout_true, z_hi_rel_true)
img_bye = bye_sim.simulate(z_layout_true, z_bye_rel_true)
obs     = comp_sim.simulate(img_hi, img_bye)

out = os.path.join(os.path.dirname(__file__), 'obs.npz')
np.savez(out, x=obs, z_layout_true=z_layout_true,
         z_hi_rel_true=z_hi_rel_true, z_bye_rel_true=z_bye_rel_true)
print(f"Saved {out}  shape={obs.shape}  max={obs.max():.1f}")
