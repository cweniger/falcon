# Falcon Performance: Cached Dataloader

## Problem

With `replacement_fraction=0.01-0.1`, training reuses buffer samples 10-100x.
Ray handles simulation fine at this rate, but every training step re-fetches
data through Ray IPC (`ray.get()` per sample per key), and the dataloader is
recreated every epoch. This dominates runtime.

## Solution: Stateful cached dataloader

The dataloader is instantiated **once**, caches samples locally as CPU tensors,
and serves random batches via list indexing + stack. Periodic `sync()` pulls
only **new** samples from the DatasetManager. The epoch structure is preserved:
one "epoch" = `cache_size // batch_size` random mini-batches from the cache.

For buffers that exceed CPU memory (e.g. LISA global fit), only a random subset
is cached locally; the rest stays in the Ray object store.

### Data flow

```
Current (slow):
  Every epoch:  recreate dataloader
  Every step:   DatasetManager --ray.get() x 384--> numpy --stack--> torch --> GPU
  Every epoch:  full validation pass through all val samples

Proposed (fast):
  Once:         create dataloader, full sync (ray.get all samples to CPU cache)
  Every step:   dict index --> torch.stack --> .to(device) --> train
  Every N steps: incremental sync() -- ray.get ONLY new samples --> update cache
  Every M steps: validation on random subset
```

## Garbage collector interaction

The existing `DatasetManagerActor` uses reference counting to prevent garbage
collection of samples that are still in use:

- `get_samples_by_status()` increments `ref_counts` for returned sample IDs
- `release_samples()` decrements `ref_counts`
- `garbage_collect_tombstones()` (runs every 10s) only frees TOMBSTONE samples
  where `ref_counts <= 0`
- `ray.internal.free()` permanently frees the Ray object; any subsequent
  `ray.get()` on that ref would fail

**The cached dataloader must participate in this protocol correctly.**

Key insight: the cache holds **copies** of data (CPU tensors), not live Ray
refs. Once `sync()` resolves refs to numpy and converts to tensors, the Ray
refs are no longer needed. Therefore:

1. Refs should be held only during the `sync()` call, not between syncs
2. This minimizes the window where tombstoned samples are blocked from GC
3. With 20 actors, each holds refs for only the duration of `sync()` (~50-100ms),
   not for an entire epoch

### Design: decentralized resolve via `checkout_refs()` / `release_refs()`

The DatasetManager only provides **metadata** (sample IDs + Ray object refs).
Each training actor resolves refs **directly from the Ray object store**
(zero-copy shared memory on same node, network transfer across nodes). This
preserves Ray's decentralized data access — the DatasetManager is never a data
bottleneck, even with 20 actors syncing concurrently.

Two methods on DatasetManagerActor:

1. **`checkout_refs(status, keys, max_samples=0)`** — selects samples by status,
   increments ref_counts, returns `{key: [ObjectRef, ...], '_ids': np.array}`.
   No data movement — only metadata.

2. **`release_refs(ids)`** — decrements ref_counts. Called by the training actor
   after it has resolved all refs and converted to CPU tensors.

The training actor's `sync()` flow:
1. Release previous checkout: `release_refs(old_ids)`
2. Get new refs: `checkout_refs(status, keys, max_samples)` → refs + ids
3. Resolve directly: `ray.get(refs)` — reads from object store (decentralized)
4. Convert to CPU tensors, store in cache
5. Release: `release_refs(ids)`

Ref_counts are held for the duration of the resolve (~50-100ms). Between syncs,
no refs are held by the cached dataloader. This is strictly shorter than the
current DatasetView, which holds refs for an entire epoch.

Why decentralized resolve (not inside DatasetManager):
- With 20 actors, centralizing resolve through the DatasetManager means all
  training data flows through one actor — a serialization bottleneck
- Ray's object store is designed for direct access: shared memory on same node,
  plasma store across nodes
- Each training actor resolves in parallel with all others
- The DatasetManager stays lightweight: only metadata + ref_count bookkeeping

### Protocol for 20 actors

```
Actor A sync():
  1. release_refs(old_ids)           → DatasetManager: ref_counts[old_ids] -= 1
  2. checkout_refs(status, keys)     → DatasetManager: ref_counts[ids] += 1
                                       returns {key: [ObjectRef, ...], '_ids': ids}
  3. ray.get(refs) on Actor A        → reads from Ray object store (decentralized)
  4. convert to CPU tensors
  5. release_refs(ids)               → DatasetManager: ref_counts[ids] -= 1

Actor B sync(): (concurrent with A — no contention on data)
  Same protocol. Steps 1-2 and 5 are serialized by the DatasetManager actor,
  but step 3 (the expensive part) runs in parallel across all actors.
```

Ref_counts are held from step 2 to step 5 — the duration of the resolve
(~50-100ms). The GC cannot free a sample during this window because
ref_counts > 0. After step 5, ref_counts return to baseline and tombstoned
samples can be freed.

With 20 actors syncing at staggered times, at most a few actors hold refs
simultaneously. Each hold is brief. The GC runs every 10s and will find
windows where all ref_counts are back to zero.

## Implementation Plan

### Step 1: Add `checkout_refs()` and `release_refs()` to DatasetManagerActor

Two lightweight methods for ref_count management. No data resolution here —
that happens decentrally on each training actor.

`checkout_refs` supports **incremental sync**: the caller passes
`already_cached_ids` (IDs it already has in its local cache). The method
returns only refs for **new** IDs (samples not yet in the caller's cache),
plus the full set of **active** IDs so the caller can evict stale entries.

```python
def checkout_refs(self, status, keys, max_samples=0, already_cached_ids=None):
    """Select samples by status, return refs for uncached samples only.

    Increments ref_counts for new (uncached) sample IDs. The caller
    resolves these from the object store, then calls release_refs().

    Args:
        status: SampleStatus or list of SampleStatus
        keys: list of key names to retrieve
        max_samples: 0 = all samples, >0 = random subset
        already_cached_ids: np.array of IDs the caller already has cached.
            Only refs for IDs NOT in this set are returned.

    Returns:
        dict with:
            '_active_ids': np.array — all currently active sample IDs
            '_new_ids': np.array — IDs that need to be fetched (not in cache)
            key: [ObjectRef, ...] — refs for _new_ids only (one per key)
    """
    # Select samples by status
    if isinstance(status, list):
        ids = np.where(np.isin(self.status, status))[0]
    else:
        ids = np.where(self.status == status)[0]
    if max_samples > 0 and len(ids) > max_samples:
        ids = np.random.choice(ids, size=max_samples, replace=False)

    # Determine which IDs are new (not in caller's cache)
    if already_cached_ids is not None and len(already_cached_ids) > 0:
        new_mask = ~np.isin(ids, already_cached_ids)
        new_ids = ids[new_mask]
    else:
        new_ids = ids

    # Increment ref_counts only for new IDs (prevent GC while resolving)
    if len(new_ids) > 0:
        self.ref_counts[new_ids] += 1

    # Return refs for new IDs only (no data movement)
    result = {'_active_ids': ids, '_new_ids': new_ids}
    for key in keys:
        result[key] = [self.ray_store[i][key] for i in new_ids]
    return result

def release_refs(self, ids):
    """Decrement ref_counts after training actor has resolved data."""
    if ids is not None and len(ids) > 0:
        self.ref_counts[ids] -= 1
```

**File**: `falcon/core/raystore.py` (add methods to `DatasetManagerActor`)

### Step 2: Add `CachedDataLoader` to raystore.py

```python
class CachedDataLoader:
    """CPU-cached dataloader for fast training access.

    Created once, holds training samples as lists of CPU tensors.
    Supports heterogeneous shapes (variable-length data per key).
    sync() pulls current state from DatasetManager via decentralized
    object store access. sample() indexes the list, stacks a batch,
    and transfers to GPU.

    If max_cache_samples > 0, only a random subset of the buffer is
    cached. On each sync(), a new random subset is drawn, so all
    samples get coverage over multiple syncs.

    Garbage collector interaction:
    - sync() checks out refs (increments ref_counts), resolves them
      directly from the Ray object store, then releases (decrements)
    - Ref_counts held only during resolve (~50-100ms), not between syncs
    - Tombstoned samples can be freed as soon as no sync() is in flight
    """
    def __init__(self, dataset_manager, keys, sample_status, max_cache_samples=0):
        self.dataset_manager = dataset_manager
        self.keys = keys
        self.sample_status = sample_status
        self.max_cache_samples = max_cache_samples  # 0 = cache everything
        self.cache = {}   # sample_id -> {key: np.ndarray}
        self.active_ids = np.array([], dtype=int)  # current active sample IDs
        self.count = 0

    def sync(self):
        """Incremental sync: only fetch new samples, evict stale ones.

        Protocol:
        1. Checkout refs, passing already-cached IDs to skip
        2. Resolve ONLY new refs from object store (decentralized)
        3. Add new samples to cache, evict samples no longer active
        4. Release refs (decrement ref_counts)

        With replacement_fraction=0.01 and buffer_size=4096, a typical
        sync fetches ~41 new samples instead of 4096. This reduces
        ray.get calls by ~99%.
        """
        # 1. Checkout refs — only new IDs get refs + ref_count increment
        checkout = ray.get(
            self.dataset_manager.checkout_refs.remote(
                self.sample_status, self.keys, self.max_cache_samples,
                already_cached_ids=self.active_ids,
            )
        )
        active_ids = checkout['_active_ids']
        new_ids = checkout['_new_ids']

        # 2. Resolve only new refs from object store (decentralized)
        if len(new_ids) > 0:
            new_data = {}
            for key in self.keys:
                refs = checkout[key]  # ObjectRefs for new_ids only
                arrays = ray.get(refs)  # parallel resolve from object store
                new_data[key] = arrays

            # 3a. Add new samples to cache (store as numpy arrays)
            for i, sid in enumerate(new_ids):
                self.cache[sid] = {
                    key: new_data[key][i] for key in self.keys
                }

            # Release refs for new IDs (resolve done, data copied)
            ray.get(self.dataset_manager.release_refs.remote(new_ids))

        # 3b. Evict samples no longer active
        active_set = set(active_ids.tolist())
        stale = [sid for sid in self.cache if sid not in active_set]
        for sid in stale:
            del self.cache[sid]

        # Update index for fast sample()
        self.active_ids = active_ids
        self._id_list = list(active_set & set(self.cache.keys()))
        self.count = len(self._id_list)

    def sample_batch(self, batch_size):
        """Random mini-batch as a Batch object (compatible with train_step/val_step).

        Returns a Batch with numpy arrays (same format as the current DataLoader),
        so _compute_loss() works unchanged — it does torch.from_numpy() internally.
        """
        idx = np.random.randint(0, self.count, size=batch_size)
        selected = [self._id_list[i] for i in idx]

        ids = np.array(selected)
        data = {key: np.stack([self.cache[sid][key] for sid in selected])
                for key in self.keys}

        return Batch(ids, data, self.dataset_manager)
```

**File**: `falcon/core/raystore.py`

### Step 3: Add `cached_loader()` to BufferView

```python
class BufferView:
    def cached_loader(self, keys, max_cache_samples=0):
        """Create a CPU-cached training dataloader (created once, syncs periodically)."""
        return CachedDataLoader(
            self._dataset_manager, keys,
            sample_status=[SampleStatus.TRAINING, SampleStatus.DISFAVOURED],
            max_cache_samples=max_cache_samples,
        )

    def cached_val_loader(self, keys, max_cache_samples=0):
        """Create a CPU-cached validation dataloader."""
        return CachedDataLoader(
            self._dataset_manager, keys,
            sample_status=SampleStatus.VALIDATION,
            max_cache_samples=max_cache_samples,
        )
```

**File**: `falcon/core/raystore.py` (add methods to `BufferView`)

### Step 4: Integrate cached dataloader into StepwiseEstimator.train()

The existing `StepwiseEstimator.train()` has a clean epoch structure:

```python
# Current code (stepwise_estimator.py lines 164-264):
async def train(self, buffer):
    cfg = self.loop_config
    keys = [self.theta_key, f"{self.theta_key}.logprob", *self.condition_keys]
    dataloader_train = buffer.train_loader(keys, batch_size=cfg.batch_size)
    dataloader_val = buffer.val_loader(keys, batch_size=cfg.batch_size)
    ...
    for epoch in range(cfg.num_epochs):
        for batch in dataloader_train:     # <-- re-fetches every epoch
            metrics = self.train_step(batch)
            ...
        for batch in dataloader_val:       # <-- re-fetches every epoch
            metrics = self.val_step(batch)
            ...
        self.on_epoch_end(epoch, val_metrics_avg)
        # early stopping check
```

The cached dataloader preserves this epoch structure. An "epoch" remains
one pass through the training data. The only change is HOW batches are
produced: from a local CPU cache instead of per-sample `ray.get()`.

**Modified `train()` method:**

```python
async def train(self, buffer) -> None:
    cfg = self.loop_config
    keys = [self.theta_key, f"{self.theta_key}.logprob", *self.condition_keys]

    if getattr(cfg, 'cache_sync_every', 0) > 0:
        await self._train_cached(buffer, cfg, keys)
    else:
        await self._train_original(buffer, cfg, keys)

async def _train_original(self, buffer, cfg, keys):
    """Original epoch-based training (unchanged)."""
    # ... existing code moved here verbatim ...

async def _train_cached(self, buffer, cfg, keys):
    """Epoch-based training with GPU-cached dataloader."""
    max_cache = getattr(cfg, 'max_cache_samples', 0)
    train_cache = buffer.cached_loader(keys, max_cache_samples=max_cache)
    val_cache = buffer.cached_val_loader(keys, max_cache_samples=0)  # val is small

    # Initial sync (full fetch — cache is empty)
    train_cache.sync()
    val_cache.sync()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    t0 = time.perf_counter()

    for epoch in range(cfg.num_epochs):
        info(f"Epoch {epoch+1}/{cfg.num_epochs}")
        log({"epoch": epoch + 1})

        # Periodic incremental sync (only fetches NEW samples)
        if epoch > 0 and epoch % cfg.cache_sync_every == 0:
            train_cache.sync()
            val_cache.sync()

        # === Training phase ===
        # One "epoch" = one pass through cache in random mini-batches
        steps_per_epoch = max(1, train_cache.count // cfg.batch_size)
        train_metrics_sum = {}
        num_train_batches = 0

        for step in range(steps_per_epoch):
            batch = train_cache.sample_batch(cfg.batch_size)
            metrics = self.train_step(batch)

            for k, v in metrics.items():
                train_metrics_sum[k] = train_metrics_sum.get(k, 0) + v
            num_train_batches += 1

            for k, v in metrics.items():
                log({f"train:{k}": v})

            await asyncio.sleep(0)
            await self._pause_event.wait()
            if self._break_flag:
                self._break_flag = False
                break

        train_metrics_avg = {
            k: v / num_train_batches for k, v in train_metrics_sum.items()
        }

        # === Validation phase ===
        val_metrics_sum = {}
        num_val_samples = 0
        val_steps = max(1, val_cache.count // cfg.batch_size)

        for step in range(val_steps):
            batch = val_cache.sample_batch(cfg.batch_size)
            metrics = self.val_step(batch)
            bs = len(batch)

            for k, v in metrics.items():
                val_metrics_sum[k] = val_metrics_sum.get(k, 0) + v * bs
            num_val_samples += bs

            await asyncio.sleep(0)
            await self._pause_event.wait()
            if self._break_flag:
                self._break_flag = False
                break

        val_metrics_avg = {
            k: v / num_val_samples for k, v in val_metrics_sum.items()
        }

        for k, v in val_metrics_avg.items():
            log({f"val:{k}": v})

        # === End of epoch (unchanged) ===
        self.on_epoch_end(epoch, val_metrics_avg)

        val_loss = val_metrics_avg.get("loss", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        self._record_epoch_history(epoch, train_metrics_avg, val_metrics_avg, t0, buffer)

        if epochs_no_improve >= cfg.early_stop_patience:
            info("Early stopping triggered.")
            break

        await self._pause_event.wait()
        if self._terminated:
            break
```

**Key design decisions:**

1. **Epoch = one pass through cache.** `steps_per_epoch = cache_size // batch_size`.
   This means an "epoch" with caching is the same as without: every sample
   seen once (in expectation, since sampling is random with replacement).

2. **Sync at epoch boundaries.** `cache_sync_every=1` syncs every epoch,
   `cache_sync_every=5` syncs every 5 epochs. Syncs are incremental (only
   new samples fetched), so `cache_sync_every=1` is cheap.

3. **Same hooks.** `on_epoch_end()`, `train_step()`, `val_step()` are called
   identically. `LossBasedEstimator` and `GaussianPosterior` code is unchanged.

4. **Batch interface.** The `CachedDataLoader.sample_batch()` returns a `Batch`
   object (same as current DataLoader), so `batch[key]`, `batch.discard()`,
   `len(batch)` all work. The `Batch._ids` field contains the original sample
   IDs from the DatasetManager, enabling `discard()` feedback.

**File**: `falcon/contrib/stepwise_estimator.py`

### Step 5: Config update

```yaml
loop:
  num_epochs: 1000
  batch_size: 128
  early_stop_patience: 32
  cache_sync_every: 1       # sync CPU cache every N epochs (0 = disable, use old dataloader)
  max_cache_samples: 0      # 0 = cache all samples, >0 = cache random subset of this size
```

**File**: `examples/05_linear_regression/config.yaml` (and loop config schema)

## Compatibility

- `cache_sync_every: 0` → original epoch-based behavior, nothing changes
- `cache_sync_every: 1` → sync every epoch, step-based training
- `cache_sync_every: N` → sync every N epochs (for very slow replacement)
- `max_cache_samples: 0` → cache everything (default, fine for small problems)
- `max_cache_samples: 50000` → cache at most 50k samples (for large problems
  like LISA global fit where the full buffer may exceed CPU memory)
- All existing configs work unchanged (defaults: `cache_sync_every: 0`,
  `max_cache_samples: 0`)

## Memory management for large-scale problems

For problems like LISA global fit (millions of frequency bins, large buffers),
the full buffer may not fit in CPU memory on the training machine. The
`max_cache_samples` parameter caps the local cache size:

- On each `sync()`, a **random subset** of `max_cache_samples` samples is drawn
  from the DatasetManager. Over multiple syncs, all samples get coverage.
- Training draws batches from the cached subset. With heavy reuse
  (`replacement_fraction=0.01`), most of the buffer is stable between syncs,
  so the random subset provides good coverage.
- The cache rotates on each sync, so no sample is permanently excluded.

This gives a smooth tradeoff:
- Small problems (linear regression): `max_cache_samples: 0`, cache everything,
  maximum speed.
- Medium problems: `max_cache_samples: 100000`, fits in ~10 GB CPU memory for
  1000-dim observations.
- Large problems (LISA): `max_cache_samples: 10000`, keeps memory bounded,
  syncs more frequently to rotate coverage.

## Garbage collector safety analysis

### Ref count lifecycle with decentralized resolve

With 20 training actors, the ref_count lifecycle looks like:

```
Time →
Actor 1:  [checkout...resolve...release]                    [checkout...resolve...release]
Actor 2:       [checkout...resolve...release]                    [checkout...resolve...release]
...
Actor 20:                [checkout...resolve...release]
GC:       can run between any actor's checkout-release windows

During each sync():
  checkout_refs(): ref_counts += 1  (on DatasetManager, serialized)
  ray.get(refs): resolve on training actor (parallel across actors)
  release_refs(): ref_counts -= 1  (on DatasetManager, serialized)

Between syncs:
  ref_counts are unchanged — no holds from cached dataloaders
  GC can freely collect any tombstoned sample with ref_count=0
```

Compare with the current DatasetView:
```
Current: ref_counts held for entire epoch (~seconds to minutes)
Proposed: ref_counts held for ~50-100ms during sync() resolve
```

The cached dataloader is **strictly better** for GC responsiveness.

### Race condition: GC between checkout and resolve

1. Actor A calls `checkout_refs()` → ref_counts[X] += 1
2. GC runs (`garbage_collect_tombstones`) — sees ref_counts[X] > 0, skips X
3. Actor A calls `ray.get(ref_X)` — succeeds (object still in store)
4. Actor A calls `release_refs()` → ref_counts[X] -= 1
5. Next GC: ref_counts[X] == 0 → can free X

This is correct. The ref_count protects the object during resolve.

Note: `checkout_refs()` and `garbage_collect_tombstones()` are both methods
on the same Ray actor, so they are serialized. The GC **cannot** run during
`checkout_refs()`. The only window where GC could free an object is between
`checkout_refs()` returning and the training actor calling `release_refs()`.
But ref_counts > 0 during this window, so GC will skip those samples.

### Edge case: sample tombstoned during sync

1. Actor calls `checkout_refs()`, ref_counts[X] += 1
2. `rotate_sample_buffer()` marks X as TOMBSTONE (on a different call)
3. Actor resolves `ray.get(ref_X)` — succeeds (ref_count > 0 prevents GC)
4. Actor calls `release_refs()` — ref_counts[X] -= 1
5. Next GC cycle: X is TOMBSTONE with ref_count=0 → freed

Correct. The actor got its copy before GC freed the Ray object.

### Edge case: deleted sample in cache

After sync, the cache holds CPU tensors. The original Ray objects may be freed
by GC. This is fine — the cache has independent copies (numpy→torch conversion
copies the data). The tensors remain valid until the next sync replaces them.

### Concurrency: 20 actors resolving in parallel

The expensive step (ray.get on object refs) happens on each training actor,
not on the DatasetManager. The DatasetManager only handles lightweight
checkout/release calls (array indexing + ref_count arithmetic). With 20
actors, the DatasetManager processes ~40 lightweight calls per sync round
(20 checkouts + 20 releases). Each takes microseconds. No bottleneck.

## Files to modify

1. **`falcon/core/raystore.py`**
   - Add `CachedDataLoader` class
   - Add `DatasetManagerActor.checkout_refs()` and `release_refs()` methods
   - Add `BufferView.cached_loader()` and `cached_val_loader()` methods

2. **`falcon/contrib/stepwise_estimator.py`**
   - Add `_train_cached()` method alongside existing `train()`
   - Route based on `cache_sync_every` config
   - No changes to `train_step()`/`val_step()` — cached loader returns `Batch`

3. **`falcon/contrib/SNPE_gaussian.py`** (or its config schema)
   - Add `cache_sync_every` and `max_cache_samples` to loop config schema

4. **`examples/05_linear_regression/config.yaml`**
   - Add `cache_sync_every: 1`

## What stays the same

- Ray orchestration loop (`deployed_graph.py`)
- Sample lifecycle (VALIDATION → TRAINING → ...)
- DatasetManager append/rotate
- Simulation via Ray actors
- Logging, monitoring
- Original epoch-based training (when cache disabled)

## Expected performance

- Between syncs: ~standalone speed (~3.9ms/step)
- Sync cost: two lightweight remote calls (checkout_refs + release_refs) to
  DatasetManager, plus decentralized ray.get for NEW samples only.
  With replacement_fraction=0.01, buffer_size=4096: ~41 new samples per sync
  instead of 4096. First sync fetches all; subsequent syncs are incremental.
- Per-step batch transfer to GPU: ~0.1-2ms depending on observation dimension
- With `cache_sync_every=1` and 250 steps/epoch: sync overhead amortized
- **Expected speedup: 10-15x** vs current Falcon
- CPU memory: bounded by `max_cache_samples` × sample size
- GPU memory: only one batch at a time
- GC: strictly better than current (holds refs for ~50-100ms vs entire epoch)

## Verification

1. Run `examples/05_linear_regression` before/after, compare wall time
2. Compare posterior quality — should be identical
3. Verify cache syncs: log cache size at each sync
4. Test with `max_cache_samples` < buffer size to verify subset rotation works
5. Test GC: verify tombstoned samples are freed promptly with 2+ training actors
6. Stress test: 20 actors, small buffer, frequent resampling — verify no
   `ray.get()` failures on freed objects

## Motivation

Falcon should feel fast on simple problems. If a toy linear regression takes
minutes instead of seconds, users won't trust the framework for harder problems.
The cached dataloader makes the inner training loop as fast as a standalone
script while preserving all of Falcon's distributed simulation and orchestration
capabilities. Even with slow simulators, GPU utilization improves because the
GPU isn't waiting on Ray IPC during training.
