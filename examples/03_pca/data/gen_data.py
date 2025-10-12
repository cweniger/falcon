import numpy as np

SIGMA = 1e-3

for NBINS in [100, 1000, 10000, 100000, 1000000]:
    z0 = 0.123456789
    x = np.random.randn(NBINS) * SIGMA + z0 
    np.save(f'obs_{NBINS}_offset.npy', x)

for NBINS in [100, 1000, 10000, 100000, 1000000]:
    z0 = 0
    x = np.random.randn(NBINS) * SIGMA + z0 
    np.save(f'obs_{NBINS}.npy', x)
