import numpy as np

SIGMA = 1e-1

for NBINS in [3,]:
    x = np.random.randn(NBINS) * SIGMA
    np.save(f'obs_{NBINS}.npy', x)
