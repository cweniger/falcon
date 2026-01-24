#!/usr/bin/env python3
"""Generate mock data for ExpPlusNoise model.

Forward model: x = exp(z)
Ground truth: z = [-5, 0, 5]
Asimov data (no noise).

Convention: Observations have no batch dimension - shape is [features].
"""

import numpy as np

# True parameters (no batch dimension)
z_true = np.array([-5.0, 0.0, 5.0])

# Observation: x = exp(z), no noise (Asimov)
x_obs = np.exp(z_true)

print(f"z_true: {z_true}")
print(f"x_obs:  {x_obs}")
print(f"        exp(-5)={np.exp(-5):.6f}, exp(0)={np.exp(0):.6f}, exp(5)={np.exp(5):.6f}")

# Save (shape: [3], not [1, 3])
np.savez("mock_data_exp.npz", x=x_obs)
print(f"\nSaved to mock_data_exp.npz (shape: {x_obs.shape})")
