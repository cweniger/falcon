"""Generate mock observation data with ground truth for testing inference."""
import numpy as np

SIGMA = 1e-1
NBINS = 3

# Generate ground truth parameters and noisy observation
z_truth = np.random.randn(NBINS) * 10
x = z_truth + np.random.randn(NBINS) * SIGMA

# Save as NPZ with multiple keys (demonstrates key extraction syntax)
np.savez('mock_data.npz', x=x, z_truth=z_truth)

print(f"Saved mock_data.npz with keys: x, z_truth")
print(f"z_truth = {z_truth}")
print(f"x = {x}")
