"""
Samples reader for loading posterior/prior/proposal samples.

Provides lazy-loaded access to samples stored as individual NPZ files.

Usage:
    samples = read_samples('outputs/run_01/samples_dir')

    # Access by sample type
    samples.posterior              # SampleSetReader
    samples.types                  # ['posterior', 'prior', ...]

    # Index access (returns dict)
    samples.posterior[32]          # Dict for sample 32

    # Key access (returns list of arrays)
    samples.posterior['x']         # List of arrays, one per sample

    # Stacked access
    samples.posterior.stacked['x'] # np.stack() result

    # Filtering
    samples.posterior.where(batch='250113-1200')['x']
"""

from pathlib import Path
from typing import Optional, List, Set, Dict, Any, Union
import numpy as np


class StackedAccessor:
    """Helper class for .stacked['key'] access pattern."""

    def __init__(self, sample_set_reader: 'SampleSetReader'):
        self._reader = sample_set_reader

    def __getitem__(self, key: str) -> np.ndarray:
        """Return stacked array for key. Fails if shapes differ."""
        arrays = self._reader[key]
        if not arrays:
            raise KeyError(f"No samples contain key '{key}'")
        return np.stack(arrays)


class SampleSetReader:
    """Reader for one sample type (posterior/prior/proposal).

    Provides lazy-loaded access to samples via indexing.
    """

    def __init__(self, sample_type_dir: Path, indices: Optional[List[int]] = None):
        self.sample_type_dir = sample_type_dir
        self._sample_files: Optional[List[Path]] = None
        self._samples_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._keys_cache: Optional[Set[str]] = None
        self._batches_cache: Optional[List[str]] = None
        self._indices = indices  # For filtered views
        self._stacked = StackedAccessor(self)

    def _discover_files(self) -> List[Path]:
        """Find all sample NPZ files across all batches."""
        if self._sample_files is not None:
            return self._sample_files

        self._sample_files = []
        if not self.sample_type_dir.exists():
            return self._sample_files

        # Find all batch directories and their NPZ files
        for batch_dir in sorted(self.sample_type_dir.iterdir()):
            if batch_dir.is_dir():
                npz_files = sorted(batch_dir.glob("*.npz"))
                self._sample_files.extend(npz_files)

        return self._sample_files

    def _get_active_indices(self) -> List[int]:
        """Get list of valid indices (all or filtered subset)."""
        n_files = len(self._discover_files())
        if self._indices is not None:
            return self._indices
        return list(range(n_files))

    def _load_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """Load a single sample by index."""
        if idx in self._samples_cache:
            return self._samples_cache[idx]

        files = self._discover_files()
        if idx < 0 or idx >= len(files):
            raise IndexError(f"Sample index {idx} out of range (0-{len(files)-1})")

        data = dict(np.load(files[idx]))
        self._samples_cache[idx] = data
        return data

    def __getitem__(self, key: Union[int, str, slice]) -> Union[Dict, List, List[Dict]]:
        """Access samples by index (int), key (str), or slice."""
        active_indices = self._get_active_indices()

        if isinstance(key, int):
            # Handle negative indices
            if key < 0:
                key = len(active_indices) + key
            if key < 0 or key >= len(active_indices):
                raise IndexError(f"Index {key} out of range")
            real_idx = active_indices[key]
            return self._load_sample(real_idx)

        elif isinstance(key, slice):
            sliced_indices = active_indices[key]
            return [self._load_sample(i) for i in sliced_indices]

        elif isinstance(key, str):
            # Return list of arrays for this key across all samples
            result = []
            for idx in active_indices:
                sample = self._load_sample(idx)
                if key in sample:
                    result.append(sample[key])
            return result

        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __len__(self) -> int:
        """Number of samples (respects filtering)."""
        return len(self._get_active_indices())

    def __iter__(self):
        """Iterate over sample dicts."""
        for idx in self._get_active_indices():
            yield self._load_sample(idx)

    @property
    def stacked(self) -> StackedAccessor:
        """Access for stacked arrays: .stacked['x'] -> np.stack()"""
        return self._stacked

    @property
    def keys(self) -> Set[str]:
        """All keys across all samples (excludes metadata keys starting with _)."""
        if self._keys_cache is not None:
            return self._keys_cache

        all_keys = set()
        for idx in self._get_active_indices():
            sample = self._load_sample(idx)
            # Exclude metadata keys (starting with _)
            all_keys.update(k for k in sample.keys() if not k.startswith('_'))

        self._keys_cache = all_keys
        return self._keys_cache

    @property
    def batches(self) -> List[str]:
        """List of unique batch IDs."""
        if self._batches_cache is not None:
            return self._batches_cache

        batches = set()
        files = self._discover_files()
        for idx in self._get_active_indices():
            if idx < len(files):
                # Batch is the parent directory name
                batch_name = files[idx].parent.name
                batches.add(batch_name)

        self._batches_cache = sorted(batches)
        return self._batches_cache

    def where(self, batch: Optional[str] = None) -> 'SampleSetReader':
        """Filter samples by criteria. Returns a new reader with filtered view.

        Args:
            batch: Filter to samples from this batch only

        Returns:
            New SampleSetReader with filtered indices
        """
        files = self._discover_files()
        active_indices = self._get_active_indices()

        filtered_indices = []
        for idx in active_indices:
            if idx >= len(files):
                continue

            include = True

            # Filter by batch
            if batch is not None:
                batch_name = files[idx].parent.name
                if batch_name != batch:
                    include = False

            if include:
                filtered_indices.append(idx)

        # Return new reader with filtered indices
        filtered = SampleSetReader(self.sample_type_dir, indices=filtered_indices)
        filtered._sample_files = self._sample_files  # Share discovered files
        filtered._samples_cache = self._samples_cache  # Share cache
        return filtered

    def __repr__(self):
        n = len(self)
        batches = len(self.batches)
        return f"<SampleSetReader: {n} samples, {batches} batch(es)>"


class SamplesReader:
    """Top-level reader for all sample types in a samples directory.

    Usage:
        samples = read_samples('outputs/run_01/samples_dir')
        samples.posterior['x']  # Access posterior samples
        samples.types           # ['posterior', 'prior', ...]
    """

    def __init__(self, samples_dir: Path):
        self.samples_dir = Path(samples_dir)
        self._readers_cache: Dict[str, SampleSetReader] = {}

    @property
    def types(self) -> List[str]:
        """List available sample types (posterior, prior, proposal, etc.)."""
        if not self.samples_dir.exists():
            return []
        return [d.name for d in sorted(self.samples_dir.iterdir()) if d.is_dir()]

    def __getattr__(self, name: str) -> SampleSetReader:
        """Access sample type as attribute: samples.posterior"""
        if name.startswith('_'):
            raise AttributeError(name)

        if name not in self._readers_cache:
            type_dir = self.samples_dir / name
            self._readers_cache[name] = SampleSetReader(type_dir)

        return self._readers_cache[name]

    def __getitem__(self, name: str) -> SampleSetReader:
        """Access sample type by name: samples['posterior']"""
        return getattr(self, name)

    def __repr__(self):
        types = self.types
        return f"<SamplesReader({self.samples_dir}): {types}>"


def read_samples(path: str) -> SamplesReader:
    """Read samples from a samples directory.

    Args:
        path: Path to the samples directory (e.g., 'outputs/run_01/samples_dir')

    Returns:
        SamplesReader providing access to all sample types.

    Example:
        samples = read_samples('outputs/run_01/samples_dir')
        samples.posterior[32]           # Dict for sample 32
        samples.posterior['x']          # List of arrays
        samples.posterior.stacked['x']  # Stacked array
        samples.posterior.where(batch='250113-1200')['x']  # Filtered
    """
    return SamplesReader(Path(path))
