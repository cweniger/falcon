import numpy as np
SIGMA = 1e-3
x = np.random.randn(1, 10000) * SIGMA
np.savez('../data/obs_10000.npz', x=x)
