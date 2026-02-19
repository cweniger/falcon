"""Generate a mock EMRI observation at known ground-truth parameters.

Saves the noisy signal to obs.npz for use as the observed data in falcon.

Ground truth: f0=2.75e-3 Hz, chirp_mass=1.0, harmonic_decay=1.5
"""

import numpy as np
from fuge import emri_signal

TRUE_F0 = 2.75e-3
TRUE_CHIRP_MASS = 1.0
TRUE_HARMONIC_DECAY = 1.5

signal = emri_signal(
    f0=TRUE_F0,
    chirp_mass=TRUE_CHIRP_MASS,
    t_c=1e6,
    A0=5.0,
    harmonic_decay=TRUE_HARMONIC_DECAY,
    n_harmonics=4,
    N=100_000,
)

rng = np.random.default_rng(42)
noise = rng.standard_normal(len(signal))
observation = signal + noise

np.savez(
    "obs.npz",
    x=observation,
    true_theta=np.array([TRUE_F0, TRUE_CHIRP_MASS, TRUE_HARMONIC_DECAY]),
)
print(f"Saved obs.npz ({len(observation)} samples, SNR ~ {np.std(signal)/1.0:.1f})")
