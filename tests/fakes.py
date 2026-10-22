"""Stand-ins for the Ray-backed buffer, used by the engine tests."""

from unittest.mock import MagicMock

import numpy as np

from falcon.core.raystore import Batch


class FakeLoader:
    def __init__(self, ids, dataset_manager):
        self.ids = np.asarray(ids)
        self.dataset_manager = dataset_manager
        self.refreshes = 0

    @property
    def count(self):
        return len(self.ids)

    def refresh(self):
        self.refreshes += 1

    def iter_batches(self, batch_size, shuffle=False, drop_last=False):
        for start in range(0, len(self.ids), batch_size):
            yield Batch(self.ids[start:start + batch_size], {}, self.dataset_manager)


class FakeBuffer:
    """10 training samples (ids 0-9) and 5 validation samples (ids 100-104)."""

    def __init__(self):
        self.dataset_manager = MagicMock()
        self.train = FakeLoader(range(10), self.dataset_manager)
        self.val = FakeLoader(range(100, 105), self.dataset_manager)

    def cached_loader(self, keys, max_cache_samples=0):
        return self.train

    def cached_val_loader(self, keys, max_cache_samples=0):
        return self.val

    def get_stats(self):
        return {"total_length": 15}
