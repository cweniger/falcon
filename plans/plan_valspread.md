# plan_valspread — spread validation data across the buffer

## Context

Right now the validation set is always the **newest** `validation_samples` samples. As new
samples arrive, the oldest validation samples become training samples
(`rotate_sample_buffer`, `src/falcon/core/raystore.py:217`). So validation data always comes
from the latest proposal, while training data covers the whole buffer window. The two
distributions differ, which destabilises early stopping, the LR plateau scheduler and
best-checkpoint selection.

Goal: each sample gets a fixed purpose when it is added. A fraction `validation_fraction`
of samples, spread evenly by insertion id, is validation-only and the rest are
training-only. A sample's purpose never changes. Both sets age out through the same
lifecycle, so they always cover the same buffer window. `min_samples` and `max_samples` now
count **all live samples** (training plus validation).

## New config key: `buffer.validation_fraction: 0.15`

- **Name.** `validation_fraction` is clearer than an every-Nth stride. It can't be misread as
  "validate every N epochs". It keeps the `validation_` prefix of the old
  `validation_samples` key and of the `n_validation` metric.
- **Default 0.15.** Falcon keeps simulating fresh samples, so a held-out sample mostly costs
  compute, not information. Moving from 0.125 to 0.15 makes the validation loss about 9%
  less noisy and costs about 3% of the training pool. Validation adds about 9% to each
  epoch.
- **Exact, stateless id rule.** Convert once to an integer ratio:
  `p, q = Fraction(validation_fraction).limit_denominator(1000)` (0.15 → 3/20).
  Sample id `i` is validation-only if `(i * p) % q < p`.
  - This gives exactly `p` validation ids in every `q` consecutive ids, spread evenly
    (for 3/20: ids 0, 7 and 14).
  - For 1/8 it reduces to `i % 8 == 0`.
  - Id 0 is always a validation sample.
  - The rule avoids floating-point `floor(i·f)`, which can drift for large ids.

## Current fields are not enough, so decouple lifecycle from purpose

`SampleStatus` currently mixes two things: purpose (VALIDATION/TRAINING) and lifecycle
(DISFAVOURED/TOMBSTONE/DELETED). With the old tip scheme that worked, because validation
samples were never discarded. Here validation samples must follow the same discard rule as
training samples. A discarded validation sample would become DISFAVOURED, and the training
loader reads DISFAVOURED, so the two sets would mix. The fix is two independent per-sample
arrays:

```python
class SampleStatus(IntEnum):     # lifecycle only
    ACTIVE = 0        # merges old VALIDATION + TRAINING
    DISFAVOURED = 1   # still used by its loader until evicted
    TOMBSTONE = 2
    DELETED = 3

class SamplePurpose(IntEnum):    # fixed at insertion, never changes
    TRAINING = 0
    VALIDATION = 1
```

`self.status` holds the lifecycle and a new `self.purpose` (int8) holds the purpose. Each
loader filters on `status ∈ {ACTIVE, DISFAVOURED}` **and** its own purpose, so no sample is
ever read by both loaders. Renumbering the enums is safe: status is only compared by
equality or `isin`, and it is never persisted (snapshots store sample dicts only).

## Changes

### 1. `src/falcon/core/raystore.py` (the core of the change)

- **`BufferConfig`**: remove `validation_samples`. Add `validation_fraction: float = 0.15`,
  placed next to `min_samples`/`max_samples`.
- **`DatasetManagerActor.__init__`**:
  - Replace the `validation_samples` argument with `validation_fraction`.
  - Compute `self._val_p, self._val_q` with `Fraction(...).limit_denominator(1000)`.
  - Raise `ValueError` unless `1 <= p` and `p / q <= 0.5` (this also rejects fractions too
    small to represent), and `min_samples >= 2`. Id 0 is validation and id 1 is training
    for any fraction ≤ 0.5, so both pools are non-empty after the first append.
  - Add `self.purpose = np.zeros(0, dtype=np.int8)` and update the startup `info(...)` line.
- **New helper `_is_validation(ids)`**: returns `(ids * self._val_p) % self._val_q < self._val_p`.
  It is the single source of the purpose rule.
- **New helper `_register_new_samples(n)`**, used by both `append` and `append_refs`. It
  replaces their duplicated `status`/`ref_counts` appends:
  ```python
  ids = np.arange(len(self.status), len(self.status) + n)
  self.status = np.append(self.status, np.full(n, SampleStatus.ACTIVE))
  self.purpose = np.append(self.purpose, np.where(self._is_validation(ids),
                           SamplePurpose.VALIDATION, SamplePurpose.TRAINING).astype(np.int8))
  self.ref_counts = np.append(self.ref_counts, np.zeros(n))
  ```
  Purpose depends only on the global insertion id, so the split needs no state and runs
  across chunk boundaries and initial samples loaded from disk.
- **`rotate_sample_buffer`**: rewrite it as two steps (the old VAL→TRAIN step is removed).
  1. *Floor, per purpose*: tombstone old DISFAVOURED samples, but keep the newest ones as
     needed so each purpose's live pool stays at its floor.
     - The validation floor is `int(self._is_validation(np.arange(min_samples)).sum())`,
       computed once in `__init__`, and the training floor is the remainder.
     - A per-purpose floor means discards can never empty the validation set.
     - Use `ids_disf[:len(ids_disf) - keep]`; this also removes the `[:-0]` special case in
       the current code.
  2. *Cap, purpose-agnostic*: if `ACTIVE > max_samples`, tombstone the oldest ACTIVE ids.
     Purposes are interleaved by id, so removing the oldest samples keeps the ratio.
- **`deactivate`**: `ACTIVE → DISFAVOURED` for any purpose (currently `TRAINING` only).
- **`num_initial_samples`**: return `self.min_samples` (was `min + validation_samples`).
- **`num_resims`** (`simulate_when_full=False`): `max_samples - count(ACTIVE)`.
- **`checkout_refs(..., purpose=None)`**: select ids with
  `np.isin(status, statuses) & (self.purpose == purpose)` when a purpose is given.
- **`CachedDataLoader`**: new `sample_purpose=None` argument, passed to `checkout_refs` in
  `_checkout_and_fetch`.
- **`BufferView.cached_loader` / `cached_val_loader`**: both pass
  `sample_status=[ACTIVE, DISFAVOURED]`, with `sample_purpose=TRAINING` and `VALIDATION`
  respectively.
- **`get_store_stats`**: keep the same keys so `cli.py`, `interactive.py` and
  `deployed_graph._log_status` don't change. `training` and `validation` become the live
  (ACTIVE+DISFAVOURED) count for each purpose.
- **Buffer metric logging in `append`/`append_refs`**: move the duplicated block into one
  `_log_buffer_stats()` built on `get_store_stats`. Keep the metric names (`n_total`,
  `n_validation`, `n_training`, `n_disfavoured`, `n_tombstone`, `n_deleted`) so `read_run`
  and plots keep working.
- `garbage_collect_tombstones` and `load_initial_samples` are unchanged.

### 2. Estimators: apply the discard rule to validation samples

- **`src/falcon/estimators/flow.py` `val_step`**: add the same discard as in `train_step`,
  inside `torch.no_grad()`:
  `if self.discard_samples: batch.discard(self._compute_discard_mask(theta, theta_logprob, conditions_device))`.
- **`src/falcon/estimators/gaussian_fullcov.py`**: no change needed. `val_step` already
  calls `_compute_loss`, which already calls `batch.discard`. Those calls do nothing today
  because `deactivate` only accepts TRAINING samples, and they start working with the new
  `deactivate`.
- **`src/falcon/estimators/stepwise_base.py`**: update the `val_step` docstring ("NO
  batch.discard() calls"). Validation should now apply the same discard rule as training.
  The `_train` loop itself doesn't change.

### 3. Config plumbing and legacy key

- **`src/falcon/cli.py` (~line 611, before the structured `BufferConfig` merge)**: if
  `cfg.buffer` has `validation_samples`, log a `warning` that it is ignored (sizes now
  include validation; see `validation_fraction`) and drop it. Otherwise old configs and old
  run dirs crash on the struct merge.
- **`src/falcon/api.py` `_DEFAULT_GRAPH_CONFIG`**: replace `validation_samples: 256` with
  `validation_fraction: 0.15`.

### 4. Configs, docs and tests (same edit everywhere)

Remove `validation_samples` from every buffer block. In `examples/01_minimal/config.yml`,
put `validation_fraction: 0.15` in its place with a comment ("share of samples held out for
validation only, spread evenly over the buffer") and reword the min/max comments to say
"(training + validation)". In the other configs, just delete the line and rely on the
default. Representative files:
- `examples/0{1..5}_*/config*.yml` (8 files)
- source cells of `examples/01_minimal/notebook.ipynb` and `examples/04_gaussian/notebook.ipynb`
- `tests/test_examples_smoke.py` (drop the `buffer.validation_samples=16` override; min 64 → 10 validation samples)
- `docs/configuration.md` (yaml block and table: new `validation_fraction` row, min/max described as totals)
- `docs/getting-started.md`
- `CLAUDE.md` (the lifecycle line under `DatasetManagerActor` becomes lifecycle ACTIVE → DISFAVOURED → TOMBSTONE → DELETED, plus a fixed purpose TRAINING/VALIDATION)

## Behavioural consequences (expected, not bugs)

- Existing yaml numbers now mean totals. With the same values the training pool shrinks by
  15% (e.g. 01_minimal at `max_samples: 32768` → ≈27.9k train and ≈4.9k val, instead of
  32.8k train and 256 val). The example numbers are left as they are.
- Validation cost now grows with buffer size: `val_steps = n_val // batch_size`, plus one
  extra forward pass per validation batch for Flow's discard. Expect about 9% longer epochs.
  `02_bimodal/config_rounds_fill.yml` (`max_samples: 1e6`) can cache up to 150k validation
  samples. If that is too slow, a possible follow-up (not part of this change) is a
  validation cap through the `max_cache_samples` argument that `stepwise_base._train`
  currently hard-codes to 0 for the validation loader.
- Small buffers get fewer validation samples than today's 256 (04_gaussian, 1024 → ≈154).
  The fix there is a larger `min_samples`/`max_samples` or a larger `validation_fraction`.
- For mid-size and large buffers the validation loss averages over far more samples, so it
  should be much less noisy, and early stopping and the plateau scheduler should behave
  more stably.

## Verification

1. **New unit tests `tests/test_buffer_purpose.py`**. They follow the no-Ray pattern in
   `tests/test_snapshotting.py` (`DatasetManagerActor.__ray_actor_class__.__new__`, set
   attributes by hand, feed `append_refs` plain dicts with snapshotting disabled). Cases:
   - purpose = VALIDATION exactly when `(id * p) % q < p`: 3 of every 20 ids for 0.15, and `id % 8 == 0` for 0.125
   - purpose never changes through `deactivate` and `rotate_sample_buffer`
   - the cap tombstones the oldest ACTIVE samples and keeps the train/val ratio
   - under heavy discards each purpose's floor holds (the validation floor stays live)
   - `checkout_refs` with purpose TRAINING and VALIDATION returns disjoint id sets whose union is all live samples
   - `num_initial_samples() == min_samples`; `num_resims()` caps against total ACTIVE when `simulate_when_full=False`
   - the constructor rejects fractions ≤ 0 or > 0.5, fractions that round to 0/1, and `min_samples < 2`
2. `pytest tests/`, including the smoke test with the updated overrides.
3. End-to-end: `cd examples/01_minimal && falcon launch -o output/valspread`. Check that the
   `Buffer: X train, Y val` lines in `graph/output.log` show about 5.7:1, and that
   `read_run(...)` has `buffer:n_validation / (n_training + n_validation) ≈ 0.15` throughout.
4. Stability check: run `examples/02_bimodal/config_regular.yml` on this branch and on
   `main`, then compare the `val:loss` curves, LR decays and early-stop epochs.
