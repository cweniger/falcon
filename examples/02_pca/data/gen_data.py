import numpy as np

SIGMA = 1e-3

for NBINS in [100, 1000, 10000, 100000, 1000000]:
    x = np.random.randn(NBINS) * SIGMA
    np.save(f'obs_{NBINS}.npy', x)
