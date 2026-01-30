# Falcon Testing Strategy

## Philosophy

**Test what's stable, not what's experimental.**

- `falcon/core/` - Stable infrastructure shared by all estimators → **solid tests**
- `falcon/contrib/` - Experimental estimators → **protocol compliance only**
- `examples/` - Usage examples → **smoke tests**

Individual estimator correctness ("does SNPE_A produce good posteriors?") is a research question answered by benchmarks and papers, not unit tests.

---

## Test Structure

```
tests/
├── core/
│   ├── test_batch.py              # Batch class (after refactor)
│   ├── test_dataloader_factory.py # DataLoaderFactory (after refactor)
│   ├── test_graph.py              # Graph topology, node lookup
│   └── test_raystore.py           # DatasetManager, sample lifecycle
├── test_protocol_compliance.py    # Auto-discovers contrib/, checks protocol
└── test_examples_smoke.py         # Runs each example briefly
```

---

## 1. Core Infrastructure Tests

### 1.1 Batch (after refactor)

```python
# tests/core/test_batch.py
import numpy as np
from unittest.mock import MagicMock
from falcon.core.raystore import Batch

class TestBatch:
    def test_dictionary_access(self):
        """Batch supports dictionary-style access."""
        mock_manager = MagicMock()
        ids = np.array([0, 1, 2])
        data = {'theta': np.random.randn(3, 2), 'x': np.random.randn(3, 5)}

        batch = Batch(ids, data, mock_manager)

        assert 'theta' in batch
        assert 'y' not in batch
        assert batch['theta'].shape == (3, 2)
        assert batch['x'].shape == (3, 5)
        assert set(batch.keys()) == {'theta', 'x'}
        assert len(batch) == 3

    def test_discard_sends_correct_ids(self):
        """batch.discard(mask) sends correct sample IDs upstream."""
        mock_manager = MagicMock()
        ids = np.array([10, 20, 30, 40])
        data = {'theta': np.random.randn(4, 2)}

        batch = Batch(ids, data, mock_manager)
        mask = np.array([True, False, True, False])
        batch.discard(mask)

        mock_manager.deactivate.remote.assert_called_once()
        discarded_ids = mock_manager.deactivate.remote.call_args[0][0]
        assert discarded_ids == [10, 30]

    def test_discard_empty_mask(self):
        """Empty mask should not call deactivate."""
        mock_manager = MagicMock()
        ids = np.array([10, 20, 30])
        data = {'theta': np.random.randn(3, 2)}

        batch = Batch(ids, data, mock_manager)
        mask = np.array([False, False, False])
        batch.discard(mask)

        mock_manager.deactivate.remote.assert_not_called()

    def test_discard_with_torch_tensor(self):
        """Mask can be a torch tensor."""
        import torch
        mock_manager = MagicMock()
        ids = np.array([10, 20, 30])
        data = {'theta': np.random.randn(3, 2)}

        batch = Batch(ids, data, mock_manager)
        mask = torch.tensor([True, True, False])
        batch.discard(mask)

        discarded_ids = mock_manager.deactivate.remote.call_args[0][0]
        assert discarded_ids == [10, 20]
```

### 1.2 DataLoaderFactory (after refactor)

```python
# tests/core/test_dataloader_factory.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from falcon.core.raystore import DataLoaderFactory

class TestDataLoaderFactory:
    def test_creates_train_dataloader(self):
        """Factory creates dataloader with specified keys."""
        mock_manager = MagicMock()
        factory = DataLoaderFactory(mock_manager)

        loader = factory.get_train_dataloader(
            keys=['theta', 'theta.logprob', 'x'],
            batch_size=64
        )

        assert loader is not None
        assert loader.batch_size == 64

    def test_creates_val_dataloader(self):
        """Factory creates validation dataloader."""
        mock_manager = MagicMock()
        factory = DataLoaderFactory(mock_manager)

        loader = factory.get_val_dataloader(
            keys=['theta', 'x'],
            batch_size=128
        )

        assert loader is not None

    def test_pause_resume(self):
        """Flow control methods work."""
        mock_manager = MagicMock()
        factory = DataLoaderFactory(mock_manager)

        assert not factory._paused
        factory.pause()
        assert factory._paused
        factory.resume()
        assert not factory._paused

    def test_interrupt(self):
        """Interrupt sets flag."""
        mock_manager = MagicMock()
        factory = DataLoaderFactory(mock_manager)

        assert not factory._interrupted
        factory.interrupt()
        assert factory._interrupted
```

### 1.3 Graph

```python
# tests/core/test_graph.py
import pytest
from falcon.core.graph import Graph, Node

class TestGraph:
    def test_topological_sort_simple(self):
        """Linear chain: a -> b -> c"""
        nodes = [
            Node('a', simulator_cls='mock', parents=[]),
            Node('b', simulator_cls='mock', parents=['a']),
            Node('c', simulator_cls='mock', parents=['b']),
        ]
        graph = Graph(nodes)

        assert graph.sorted_node_names == ['a', 'b', 'c']

    def test_topological_sort_diamond(self):
        """Diamond: a -> b,c -> d"""
        nodes = [
            Node('a', simulator_cls='mock', parents=[]),
            Node('b', simulator_cls='mock', parents=['a']),
            Node('c', simulator_cls='mock', parents=['a']),
            Node('d', simulator_cls='mock', parents=['b', 'c']),
        ]
        graph = Graph(nodes)

        sorted_names = graph.sorted_node_names
        assert sorted_names.index('a') < sorted_names.index('b')
        assert sorted_names.index('a') < sorted_names.index('c')
        assert sorted_names.index('b') < sorted_names.index('d')
        assert sorted_names.index('c') < sorted_names.index('d')

    def test_topological_sort_cycle_raises(self):
        """Cycle should raise ValueError."""
        nodes = [
            Node('a', simulator_cls='mock', parents=['b']),
            Node('b', simulator_cls='mock', parents=['a']),
        ]
        with pytest.raises(ValueError, match="cycle"):
            Graph(nodes)

    def test_get_parents(self):
        """get_parents returns correct parents."""
        nodes = [
            Node('a', simulator_cls='mock', parents=[]),
            Node('b', simulator_cls='mock', parents=['a']),
        ]
        graph = Graph(nodes)

        assert graph.get_parents('a') == []
        assert graph.get_parents('b') == ['a']

    def test_get_evidence(self):
        """get_evidence returns correct evidence nodes."""
        nodes = [
            Node('theta', simulator_cls='mock', evidence=['x']),
            Node('x', simulator_cls='mock', parents=['theta']),
        ]
        graph = Graph(nodes)

        assert graph.get_evidence('theta') == ['x']
        assert graph.get_evidence('x') == []

    def test_graph_merge(self):
        """Two graphs can be merged with +."""
        nodes1 = [Node('a', simulator_cls='mock')]
        nodes2 = [Node('b', simulator_cls='mock', parents=['a'])]

        graph1 = Graph(nodes1)
        graph2 = Graph(nodes2)
        merged = graph1 + graph2

        assert 'a' in merged.node_dict
        assert 'b' in merged.node_dict
```

### 1.4 Raystore / DatasetManager

```python
# tests/core/test_raystore.py
import numpy as np
import pytest
from falcon.core.raystore import SampleStatus

class TestSampleStatus:
    def test_status_ordering(self):
        """Status values follow lifecycle order."""
        assert SampleStatus.VALIDATION < SampleStatus.TRAINING
        assert SampleStatus.TRAINING < SampleStatus.DISFAVOURED
        assert SampleStatus.DISFAVOURED < SampleStatus.TOMBSTONE
        assert SampleStatus.TOMBSTONE < SampleStatus.DELETED

# Note: Full DatasetManagerActor tests require Ray.
# These would be integration tests, not unit tests.
# Consider using pytest-ray or mocking Ray for unit tests.
```

---

## 2. Protocol Compliance Tests

Auto-discover all estimators in `contrib/` and verify they implement the required protocol.

```python
# tests/test_protocol_compliance.py
import inspect
import importlib
import pytest
from pathlib import Path

# Required methods and their minimum parameters
REQUIRED_METHODS = {
    'train': ['self', 'data'],
    'sample_prior': ['self', 'n'],
    'sample_posterior': ['self', 'n'],
}

# Optional methods (checked with hasattr in framework)
OPTIONAL_METHODS = {
    'sample_proposal': ['self', 'n'],
    'save': ['self', 'path'],
    'load': ['self', 'path'],
}

def discover_estimators():
    """
    Find all classes in falcon/contrib/ that appear to be estimators.
    An estimator is identified by having both 'train' and 'sample_prior' methods.
    """
    contrib_path = Path(__file__).parent.parent / 'falcon' / 'contrib'
    estimators = []

    for py_file in contrib_path.glob('*.py'):
        if py_file.name.startswith('_'):
            continue

        module_name = f'falcon.contrib.{py_file.stem}'
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip imported classes (only test classes defined in this module)
            if obj.__module__ != module_name:
                continue
            # Identify estimators by presence of key methods
            if hasattr(obj, 'train') and hasattr(obj, 'sample_prior'):
                estimators.append((name, obj))

    return estimators

class TestProtocolCompliance:
    @pytest.fixture
    def estimators(self):
        return discover_estimators()

    def test_estimators_discovered(self, estimators):
        """Sanity check: we should find at least SNPE_A."""
        names = [name for name, cls in estimators]
        assert len(estimators) > 0, "No estimators found in falcon/contrib/"
        assert 'SNPE_A' in names, "SNPE_A not found"

    @pytest.mark.parametrize('method,required_params', REQUIRED_METHODS.items())
    def test_required_methods_exist(self, estimators, method, required_params):
        """All estimators must implement required methods."""
        for name, cls in estimators:
            assert hasattr(cls, method), \
                f"{name} missing required method: {method}"

    @pytest.mark.parametrize('method,required_params', REQUIRED_METHODS.items())
    def test_required_method_signatures(self, estimators, method, required_params):
        """Required methods must accept required parameters."""
        for name, cls in estimators:
            if not hasattr(cls, method):
                continue  # Already caught by test_required_methods_exist

            sig = inspect.signature(getattr(cls, method))
            param_names = list(sig.parameters.keys())

            for required_param in required_params:
                assert required_param in param_names, \
                    f"{name}.{method}() missing parameter: {required_param}"

    def test_sample_methods_return_type_hints(self, estimators):
        """Sample methods should have return type hints (recommended)."""
        for name, cls in estimators:
            for method in ['sample_prior', 'sample_posterior']:
                if hasattr(cls, method):
                    hints = getattr(getattr(cls, method), '__annotations__', {})
                    if 'return' not in hints:
                        pytest.skip(f"{name}.{method}() has no return type hint (recommended)")
```

---

## 3. Example Smoke Tests

Run each example for a few epochs to verify the system works end-to-end.

```python
# tests/test_examples_smoke.py
import pytest
import subprocess
import os
from pathlib import Path

# Discover all example directories
EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'
EXAMPLE_DIRS = [
    d for d in EXAMPLES_DIR.iterdir()
    if d.is_dir() and (d / 'config.yaml').exists()
]

@pytest.mark.slow
@pytest.mark.parametrize('example_dir', EXAMPLE_DIRS, ids=lambda d: d.name)
def test_example_runs_without_error(example_dir, tmp_path):
    """
    Each example should run for a few epochs without crashing.
    Uses temporary directory for outputs to avoid polluting example dirs.
    """
    result = subprocess.run(
        [
            'falcon', 'launch',
            'buffer.num_epochs=5',           # Short training
            'buffer.min_training_samples=50', # Fewer samples
            'buffer.max_training_samples=100',
            'buffer.validation_window_size=20',
            f'hydra.run.dir={tmp_path}',     # Output to temp dir
        ],
        cwd=example_dir,
        capture_output=True,
        timeout=180,  # 3 minute timeout
        env={**os.environ, 'WANDB_MODE': 'disabled'},  # Disable wandb
    )

    assert result.returncode == 0, \
        f"Example {example_dir.name} failed:\n" \
        f"STDOUT:\n{result.stdout.decode()}\n" \
        f"STDERR:\n{result.stderr.decode()}"

@pytest.mark.slow
@pytest.mark.parametrize('example_dir', EXAMPLE_DIRS, ids=lambda d: d.name)
def test_example_sampling_works(example_dir, tmp_path):
    """
    After training, sampling should work.
    """
    # First train briefly
    train_result = subprocess.run(
        [
            'falcon', 'launch',
            'buffer.num_epochs=3',
            'buffer.min_training_samples=30',
            'buffer.max_training_samples=50',
            'buffer.validation_window_size=10',
            f'hydra.run.dir={tmp_path}',
        ],
        cwd=example_dir,
        capture_output=True,
        timeout=180,
        env={**os.environ, 'WANDB_MODE': 'disabled'},
    )

    if train_result.returncode != 0:
        pytest.skip("Training failed, skipping sampling test")

    # Then try sampling
    sample_result = subprocess.run(
        [
            'falcon', 'sample', 'posterior',
            f'hydra.run.dir={tmp_path}',
        ],
        cwd=example_dir,
        capture_output=True,
        timeout=60,
        env={**os.environ, 'WANDB_MODE': 'disabled'},
    )

    assert sample_result.returncode == 0, \
        f"Sampling failed for {example_dir.name}:\n" \
        f"STDERR:\n{sample_result.stderr.decode()}"
```

---

## 4. Running Tests

### pytest configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

### Commands

```bash
# Run fast tests only (core + protocol)
pytest tests/core tests/test_protocol_compliance.py

# Run all tests including slow smoke tests
pytest

# Run smoke tests only
pytest -m slow

# Run with coverage
pytest --cov=falcon --cov-report=html tests/core tests/test_protocol_compliance.py
```

---

## 5. CI Integration (optional)

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest tests/core tests/test_protocol_compliance.py

  smoke-tests:
    runs-on: ubuntu-latest
    needs: fast-tests  # Only run if fast tests pass
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest -m slow --timeout=300
```

---

## 6. Summary

| Test Type | Location | Runs | Catches |
|-----------|----------|------|---------|
| Core unit tests | `tests/core/` | Fast (<10s) | Infrastructure bugs |
| Protocol compliance | `tests/test_protocol_compliance.py` | Fast (<5s) | Interface drift |
| Smoke tests | `tests/test_examples_smoke.py` | Slow (~2min) | System regressions |

**Adding a new estimator to `contrib/`:**
- Protocol compliance tests run automatically (auto-discovery)
- No manual test maintenance required

**Adding a new example:**
- Smoke tests run automatically (auto-discovery)
- Just needs `config.yaml` in example directory
