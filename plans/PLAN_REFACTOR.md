# Falcon Estimator API Refactoring Plan

This document captures the planned refactoring of the estimator API to support O(100) future implementations (flows, diffusion models, flow matching, etc.) while simplifying the core protocol.

## Goals

1. **Minimal protocol** - Reduce required methods from 10 to 3
2. **Smart data handling** - Bidirectional batches, dictionary access
3. **Node-centric internals** - Nodes self-describing, graph emergent
4. **Extensibility** - Support diverse architectures without forcing patterns

---

## 1. Bidirectional Batch

### Current (tuple, callback-based)
```python
# Train signature
async def train(self, dataloader_train, dataloader_val, hook_fn=None, dataset_manager=None)

# Protocol requires
def get_discard_mask(self, theta, theta_logprob, conditions) -> mask

# Usage - position-dependent unpacking
for batch in dataloader_train:
    ids, theta, theta_logprob, *conditions = batch
    # Discarding via external hook
    mask = self.get_discard_mask(...)
    hook_fn(self, batch)  # hook calls dataset_manager.deactivate()
```

### Proposed (dictionary, bidirectional)
```python
# Train signature - simplified
async def train(self, data: DataLoaderFactory) -> None

# No get_discard_mask in protocol!

# Usage - self-documenting dictionary access
for batch in train_loader:
    theta = batch['theta']
    logprob = batch['theta.logprob']
    x = batch['x']

    # Discard directly via batch
    mask = self._compute_discard_mask(theta, logprob, x)
    batch.discard(mask)
```

### Implementation: `Batch` class

```python
# falcon/core/raystore.py

class Batch:
    """Bidirectional batch - provides data and accepts feedback."""

    def __init__(self, ids: np.ndarray, data: dict, dataset_manager):
        self._ids = ids
        self._data = data  # {key: tensor}
        self._dataset_manager = dataset_manager

    def __getitem__(self, key: str):
        """Dictionary-style access: batch['theta']"""
        return self._data[key]

    def __contains__(self, key: str):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def __len__(self):
        return len(self._ids)

    def discard(self, mask):
        """Mark samples as disfavoured. Mask is boolean array/tensor."""
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        if mask.any():
            ids_to_discard = self._ids[mask].tolist()
            self._dataset_manager.deactivate.remote(ids_to_discard)

    # Future extensions:
    # def reweight(self, weights): ...
    # def flag(self, mask, reason: str): ...
```

### Benefits
- Removes `get_discard_mask` from protocol
- Removes `hook_fn` and `dataset_manager` from train signature
- Self-documenting data access (no position-dependent unpacking)
- Extensible for future feedback mechanisms

---

## 2. DataLoaderFactory

### Current (two dataloaders, external key control)
```python
# NodeWrapper.train constructs dataloaders
keys_train = [self.name, self.name + ".logprob"] + self.evidence + self.scaffolds
dataset_train = dataset_manager.get_train_dataset_view(keys_train, ...)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size)

await self.estimator_instance.train(dataloader_train, dataloader_val, hook_fn, dataset_manager)
```

### Proposed (single factory, estimator controls)
```python
# NodeWrapper.train - simplified
data_factory = DataLoaderFactory(dataset_manager)
await self.estimator_instance.train(data_factory)

# Estimator.train - estimator controls everything
async def train(self, data):
    keys = [self.theta_key, f"{self.theta_key}.logprob"] + self.condition_keys
    train_loader = data.get_train_dataloader(keys=keys, batch_size=self.batch_size)
    val_loader = data.get_val_dataloader(keys=keys, batch_size=self.batch_size)

    for batch in train_loader:
        # ... training with smart batch ...
```

### Implementation: `DataLoaderFactory` class

```python
# falcon/core/raystore.py

class DataLoaderFactory:
    """Factory for creating dataloaders with bidirectional batches."""

    def __init__(self, dataset_manager_actor, observations: dict = None):
        self._dataset_manager = dataset_manager_actor
        self._observations = observations or {}
        self._paused = False
        self._interrupted = False

    def get_train_dataloader(self, keys: list[str], batch_size: int = 128,
                              **kwargs) -> DataLoader:
        """Returns dataloader for training data, yielding Batch objects."""
        dataset = BatchDatasetView(
            keys=keys,
            sample_status=[SampleStatus.TRAINING, SampleStatus.DISFAVOURED],
            dataset_manager=self._dataset_manager,
            observations=self._observations,
            flow_control=self,  # for pause/interrupt
        )
        return DataLoader(dataset, batch_size=batch_size, **kwargs)

    def get_val_dataloader(self, keys: list[str], batch_size: int = 128,
                            **kwargs) -> DataLoader:
        """Returns dataloader for validation data, yielding Batch objects."""
        dataset = BatchDatasetView(
            keys=keys,
            sample_status=SampleStatus.VALIDATION,
            dataset_manager=self._dataset_manager,
            observations=self._observations,
            flow_control=self,
        )
        return DataLoader(dataset, batch_size=batch_size, **kwargs)

    @property
    def available_keys(self) -> list[str]:
        """All keys available in the store."""
        return ray.get(self._dataset_manager.get_available_keys.remote())

    # Flow control (moved from estimator protocol)
    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def interrupt(self):
        self._interrupted = True

    async def check_flow_control(self):
        """Called by dataloader iteration. Blocks if paused, raises if interrupted."""
        while self._paused:
            await asyncio.sleep(0.1)
        if self._interrupted:
            raise TrainingInterrupted()
```

### Benefits
- Single argument to train() instead of 4
- Estimator controls batch_size, keys, timing
- Flow control (pause/resume/interrupt) moved out of estimator protocol
- Observations substitution handled transparently
- Easier testing (mock single factory)

---

## 3. Minimal Estimator Protocol

### Current (10 methods)
```python
class Estimator:
    async def train(self, dataloader_train, dataloader_val, hook_fn, dataset_manager)
    def sample_prior(self, n, parent_conditions)
    def sample_posterior(self, n, parent_conditions, evidence_conditions)
    def sample_proposal(self, n, parent_conditions, evidence_conditions)
    def get_discard_mask(self, theta, logprob, conditions)  # → removed via Batch
    def save(self, path)
    def load(self, path)
    def pause(self)   # → moved to DataLoaderFactory
    def resume(self)  # → moved to DataLoaderFactory
    def interrupt()   # → moved to DataLoaderFactory
```

### Proposed (3 required + optional)
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Estimator(Protocol):
    """Minimal interface required by falcon framework."""

    async def train(self, data: DataLoaderFactory) -> None:
        """Train the estimator using data from the factory."""
        ...

    def sample_prior(self, n: int, parent_conditions: list = []) -> np.ndarray:
        """Sample from prior/forward model."""
        ...

    def sample_posterior(self, n: int, parent_conditions: list = [],
                         evidence_conditions: list = []) -> np.ndarray:
        """Sample from posterior (with importance correction if applicable)."""
        ...


# Optional methods (checked with hasattr in framework)
class EstimatorWithProposal(Estimator, Protocol):
    def sample_proposal(self, n: int, parent_conditions: list = [],
                        evidence_conditions: list = []) -> np.ndarray:
        """Sample from proposal distribution (no importance correction).
        If absent, defaults to sample_posterior."""
        ...


class PersistableEstimator(Estimator, Protocol):
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

### Framework usage pattern
```python
# In DeployedGraph
def sample_proposal(self, n, ...):
    if hasattr(estimator, 'sample_proposal'):
        return estimator.sample_proposal(n, ...)
    return estimator.sample_posterior(n, ...)

def save_node(self, node, path):
    if hasattr(node.estimator, 'save'):
        node.estimator.save(path)
```

---

## 4. Node-Centric Internal Structure

### Current state
- `NodeWrapper` stores `self.graph` but never uses it
- Nodes already know own dependencies: `parents`, `evidence`, `scaffolds`
- Graph mainly used for topological ordering during simulation

### Changes needed

#### 4.1 Remove unused graph parameter from NodeWrapper
```python
# Before
class NodeWrapper:
    def __init__(self, node, graph, model_path=None):
        self.graph = graph  # never used!

# After
class NodeWrapper:
    def __init__(self, node, model_path=None):
        # No graph parameter
```

#### 4.2 Cleaner estimator configuration
```python
# Before (in NodeWrapper.__init__)
_embedding_keywords = self.node.evidence + self.node.scaffolds
self.estimator_instance = estimator_cls(
    self.simulator_instance,
    _embedding_keywords=_embedding_keywords,
    **node.estimator_config,
)

# After - clearer naming
self.estimator_instance = estimator_cls(
    self.simulator_instance,
    theta_key=self.name,
    condition_keys=self.evidence + self.scaffolds,
    **node.estimator_config,
)
```

#### 4.3 Observer nodes (future)
```python
# In Node class
class Node:
    def __init__(self, ..., participates_in_simulation=True):
        self.participates_in_simulation = participates_in_simulation

# In DeployedGraph.sample
for name in sorted_node_names:
    if not self.graph.node_dict[name].participates_in_simulation:
        continue
    # ... simulation logic
```

---

## 5. Estimator Class Hierarchy (for O(100) implementations)

### Architecture

```
Protocol (minimal interface - 3 methods)
    ↓
Optional Mixins (composable, opt-in)
    ↓
Family-Specific Bases (common patterns)
    ↓
Concrete Implementations
```

### 5.1 Optional Mixins

```python
# falcon/contrib/estimators/mixins.py

class CheckpointingMixin:
    """Best-weights tracking and persistence."""

    def _update_best_weights(self, network, loss, attr_name):
        if loss < getattr(self, f'best_{attr_name}_loss', float('inf')):
            setattr(self, f'best_{attr_name}_loss', loss)
            setattr(self, f'_best_{attr_name}', copy.deepcopy(network.state_dict()))

    def _restore_best_weights(self, network, attr_name):
        best = getattr(self, f'_best_{attr_name}', None)
        if best is not None:
            network.load_state_dict(best)


class EarlyStoppingMixin:
    """Patience-based early stopping."""

    def _init_early_stopping(self, patience: int, min_delta: float = 0.0):
        self._es_patience = patience
        self._es_min_delta = min_delta
        self._es_counter = 0
        self._es_best_loss = float('inf')

    def _check_early_stopping(self, loss: float) -> bool:
        if loss < self._es_best_loss - self._es_min_delta:
            self._es_best_loss = loss
            self._es_counter = 0
            return False
        self._es_counter += 1
        return self._es_counter >= self._es_patience


class EmbeddingMixin:
    """Summary/embedding network handling."""

    def _build_embedding(self, config, input_dim):
        # Build embedding network from config
        ...

    def _embed(self, conditions: list, train: bool = True):
        # Concatenate and embed conditions
        ...
```

### 5.2 Family-Specific Bases

```python
# falcon/contrib/estimators/bases/iterative.py

class BaseIterativeEstimator(CheckpointingMixin, EarlyStoppingMixin, ABC):
    """Base for epoch-based iterative training (flows, Gaussians, VAEs).

    Subclasses implement:
    - _create_networks(batch): Initialize networks from first batch
    - _training_step(batch) -> dict: Compute loss and metrics
    - _validation_step(batch) -> dict: Compute validation metrics
    - _configure_optimizers() -> tuple: Return (optimizer, scheduler)
    """

    @abstractmethod
    def _create_networks(self, batch: Batch) -> None: ...

    @abstractmethod
    def _training_step(self, batch: Batch) -> dict: ...

    @abstractmethod
    def _validation_step(self, batch: Batch) -> dict: ...

    @abstractmethod
    def _configure_optimizers(self) -> tuple: ...

    async def train(self, data: DataLoaderFactory) -> None:
        """Common epoch-based training loop."""
        keys = [self.theta_key, f"{self.theta_key}.logprob"] + self.condition_keys
        train_loader = data.get_train_dataloader(keys=keys, batch_size=self.batch_size)
        val_loader = data.get_val_dataloader(keys=keys, batch_size=self.batch_size)

        # Lazy network initialization
        first_batch = next(iter(train_loader))
        if not self._networks_initialized:
            self._create_networks(first_batch)
            self._optimizer, self._scheduler = self._configure_optimizers()

        for epoch in range(self.num_epochs):
            # Training
            for batch in train_loader:
                metrics = self._training_step(batch)
                self._optimizer.step()

            # Validation
            val_metrics = self._run_validation(val_loader)

            # Early stopping
            if self._check_early_stopping(val_metrics['loss']):
                break

            # LR scheduling
            if self._scheduler:
                self._scheduler.step()


# falcon/contrib/estimators/bases/diffusion.py

class BaseDiffusionEstimator(CheckpointingMixin, ABC):
    """Base for diffusion-based estimators.

    Different training loop structure - samples timesteps, adds noise.
    """

    @abstractmethod
    def _create_score_network(self, batch: Batch) -> None: ...

    @abstractmethod
    def _noise_schedule(self, t: torch.Tensor) -> tuple: ...

    @abstractmethod
    def _denoising_loss(self, batch: Batch, t: torch.Tensor) -> dict: ...

    async def train(self, data: DataLoaderFactory) -> None:
        """Diffusion-specific training loop."""
        # ... diffusion-specific logic with time sampling ...
```

### 5.3 Directory Structure

```
falcon/contrib/estimators/
├── __init__.py
├── protocol.py              # Estimator protocol definition
├── mixins.py                # CheckpointingMixin, EarlyStoppingMixin, etc.
├── bases/
│   ├── __init__.py
│   ├── iterative.py         # BaseIterativeEstimator
│   ├── diffusion.py         # BaseDiffusionEstimator
│   └── flow_matching.py     # BaseFlowMatchingEstimator
└── implementations/
    ├── __init__.py
    ├── snpe_a.py             # Current SNPE_A refactored
    ├── snpe_a_gaussian.py    # Current SNPE_A_gaussian refactored
    └── ...
```

---

## 6. Migration Path

### Phase 1: Core Infrastructure
1. Implement `Batch` class in `raystore.py`
2. Implement `DataLoaderFactory` in `raystore.py`
3. Update `DatasetView` to yield `Batch` objects

### Phase 2: Framework Updates
4. Remove `graph` parameter from `NodeWrapper`
5. Update `NodeWrapper.train()` to use `DataLoaderFactory`
6. Add flow control (pause/resume/interrupt) to `DataLoaderFactory`

### Phase 3: Estimator Updates
7. Update `SNPE_A.py` to use new interface:
   - Accept `DataLoaderFactory` instead of two dataloaders
   - Use `batch['key']` instead of positional unpacking
   - Use `batch.discard(mask)` instead of `get_discard_mask`
   - Remove `pause/resume/interrupt` methods
8. Update `SNPE_A_gaussian.py` similarly

### Phase 4: Protocol & Hierarchy (optional, for new estimators)
9. Create `falcon/contrib/estimators/protocol.py`
10. Extract mixins from SNPE_A into `mixins.py`
11. Create `BaseIterativeEstimator` in `bases/iterative.py`

### Backward Compatibility
- Keep `_embedding_keywords` as alias for `condition_keys` temporarily
- Framework checks `hasattr` for optional methods
- Existing estimators continue to work during transition

---

## 7. Summary of Changes

| Component | Before | After |
|-----------|--------|-------|
| Protocol methods | 10 | 3 required + optional |
| Train signature | 4 params | 1 param (DataLoaderFactory) |
| Batch access | Tuple, positional | Dictionary, by name |
| Discard mechanism | `get_discard_mask` + hook | `batch.discard(mask)` |
| Flow control | Estimator methods | DataLoaderFactory methods |
| NodeWrapper.graph | Stored, unused | Removed |
| Estimator knows keys | `_embedding_keywords` | `theta_key`, `condition_keys` |

---

## 8. Already Implemented

The following method renames have already been applied:

- `prior_sample` → `sample_prior`
- `conditioned_sample` → `sample_posterior`
- `proposal_sample` → `sample_proposal`
- `discardable` → `get_discard_mask`
- `_aux_sample` → `_importance_sample`
- `_summary` → `_embed`
- `_make_flow` → `_create_flow`
- `_save_checkpoint` → `_update_best_weights`
- `_traindist` → `_marginal_flow`
- `_posterior` → `_conditional_flow`

Files modified:
- `falcon/contrib/SNPE_A.py`
- `falcon/contrib/SNPE_A_gaussian.py`
- `falcon/core/deployed_graph.py`
- `falcon/cli.py`
