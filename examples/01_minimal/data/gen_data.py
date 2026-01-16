"""Generate mock observation data with ground truth for testing inference."""
import numpy as np

SIGMA = 1e-1
NBINS = 3

# Generate ground truth parameters and noisy observation
z = np.random.randn(NBINS) * 10
x = z + np.random.randn(NBINS) * SIGMA

# Save as NPZ with multiple keys (demonstrates key extraction syntax)
np.savez('mock_data.npz', x=x, z=z)

print(f"Saved mock_data.npz with keys: x, z")
print(f"z = {z}")
print(f"x = {x}")
