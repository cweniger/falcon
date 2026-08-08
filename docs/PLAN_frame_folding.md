# Continuous frame-folding for GaussianizedFlowMatching (v59)

## Context

The GFM estimator whitens theta_lat with a full-ZCA `_GlobalWhitener` whose
statistics EMA-update **every training step** (`gaussianized_flow_matching.py:84-96`,
called from `training_loss` at line 187). Every update moves the coordinate
frame the velocity field is trained in, so the network must chase the frame by
SGD — the measured "drifting target" pathology behind adaptation-debt val
spikes and slow re-learning after zoom steps (mackerel/muskrat analysis,
2026-07-14/15).

Fix (converged in design discussion): whenever the whitener frame changes,
**fold the affine frame delta analytically into the velocity network's input
and output layers** — a function-preserving warm start, applied continuously
at every update. The frame's effect on the model is then applied analytically,
never learned by SGD; gradients only ever learn shape residuals. Drop-in:
config-gated flag, no API or training-loop changes, old behavior when off.

## The math (fold recipe)

Old frame (μ₀, S₀=Σ₀^{1/2}, W₀=Σ₀^{-1/2}); new frame (μ₁, S₁, W₁), row-vector
convention w = (θ_lat − μ) @ W. The coordinate change old→new is affine:

    w' = w @ M + c,   M = S₀ @ W₁  (P×P),   c = (μ₀ − μ₁) @ W₁  (P,)

Warm-started field in the new frame (static conjugation; exact on the target
side, O(‖M−I‖) base-mismatch residual which is negligible for per-step deltas):

    v_new(w', t, s) = v_old((w' − c) @ M⁻¹, t, s) @ M

Folded into weights of `VelocityField` (`flow_matching.py`), which consumes
`cat([w, time_embed, cond])` in `net[0]` and outputs velocity from `net[-1]`
(PyTorch Linear: out = x @ weight.T + bias):

- **Input side** (w occupies columns `0:P` of `net[0].weight`; default config
  has `w_proj=None`):
  - `net[0].weight[:, :P] ← net[0].weight[:, :P] @ inverse(M).T`
  - `net[0].bias ← net[0].bias − (c @ inverse(M)) @ old_weight_block.T`
    (compute with the pre-update weight block)
- **Output side** (`net[-1]`: Linear(hidden, P)):
  - `net[-1].weight ← M.T @ net[-1].weight`
  - `net[-1].bias ← net[-1].bias @ M`

Apply identically to `self.velocity` **and** `self.velocity_ema` (sampling and
log_prob use the EMA net — forgetting it would desynchronize serving).

Notes:
- Between eigen refreshes only `_mean` moves → M = I, bias-only fold (cheap
  fast path). With `eig_update_freq: 1` (production configs) the full fold
  runs each step with ‖M−I‖ = O(momentum) ≈ 1e-2.
- Optimizer moments (single AdamW, `gaussianized_flow_matching.py:439-450`)
  are deliberately **not** transformed: per-step deltas are O(1e-2), moment
  mismatch is second order and re-adapts within ~1/(1−β) steps. Document this.
- Guard: only the default architecture is supported (shared trunk,
  `w_proj is None`, `per_param_nets=False`, `focus_mask is None`). If any of
  these deviate, raise at construction when the flag is on. LayerNorm variants
  are fine (fold touches only first/last Linear).
- `_best_flow` needs no change: it is a deepcopy whose whitener buffers and
  net weights are snapshotted together (`on_epoch_end:517-520`) — always a
  consistent pair.
- `logdet()` bookkeeping already live in `log_prob`/`val_nll`; unchanged.

## Changes (single file + config)

### 1. `falcon_main/src/falcon/estimators/gaussianized_flow_matching.py`

a. `_GlobalWhitener.update()` (lines 84-96): when the frame will change,
   stash `S₀ = self._sqrt.clone()`, `μ₀ = self._mean.clone()` before the EMA
   update; after the update (and `_refresh()` when it fires), compute
   `M = S₀ @ self._inv_sqrt`, `c = (μ₀ − self._mean) @ self._inv_sqrt`
   (float64), and invoke `self._on_frame_change(M, c)` if a callback is set.
   Add `self._on_frame_change = None` attribute + setter. Fast path: skip
   callback when `‖M−I‖_max` and `‖c‖_max` are below ~1e-12.

b. `_WhitenedFlow.__init__` (lines 148-154): accept `whiten_fold: bool = False`;
   when True, register the callback:
   `self.whitener._on_frame_change = self._fold_frame_delta` and validate the
   supported-architecture guard.

c. New method `_WhitenedFlow._fold_frame_delta(M, c)`: `@torch.no_grad()`;
   casts M, c to the nets' dtype; applies the four weight/bias updates above to
   `self.velocity.net` and `self.velocity_ema.net`.

d. `GaussianizedFlowMatching.__init__`: new kwarg `whiten_fold: bool = False`,
   stored and passed to `_WhitenedFlow` in `_build_module` (lines 413-424).

### 2. `dsbi config` — `falcon-LDC-MBHB/config_v59.yml`

Copy of `config_v56.yml` with one estimator line added: `whiten_fold: true`.
(Everything else untouched — this is the pure A/B of the fold.)

## Verification

1. **Algebra property test** (standalone script, scratchpad): build a small
   `VelocityField` (P=4, hidden=32) + random near-identity M, c; snapshot
   v_old; fold; assert
   `allclose(v_new(w@M+c, t, s), v_old(w, t, s) @ M, atol=1e-5)` on random
   batches. Also: fold with M=I, c=0 is a no-op (bitwise).
2. **Synthetic dynamics A/B** (script): tiny GFM on a toy stream whose target
   scale shrinks 10x over training (mimics zoom). Train twice
   (whiten_fold on/off, same seed): fold-on should show lower/smoother val NLL
   and no adaptation transients after eig refreshes. This validates "better
   dynamics" before any production run.
3. **Back-compat**: whiten_fold absent/false → behavioral no-op (callback
   never registered); run a few steps of an existing config to confirm
   identical losses for fixed seed.
4. **Production A/B**: launch `config_v59.yml`; compare against mackerel at
   matched sims — expect same-or-better val trajectory with fewer/shorter
   spike episodes (spikes from mode funerals will remain; adaptation-debt
   spikes should shrink). Monitor with the existing run_monitor.py loop.

## Out of scope (recorded, not done here)

Rung-quantized updates, boundary mode-audit, ping-pong twin heads, SDE-churn
sampling, symmetry quotient — the larger v59+ program. This change is the
smallest self-contained piece with independent value.
