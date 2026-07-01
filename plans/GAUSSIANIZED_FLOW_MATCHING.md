# Plan: Gaussianized Flow-Matching Estimator for falcon

## Context

Falcon currently has two posterior estimators that sit at opposite ends of a spectrum:
- `GaussianFullCov` (`src/falcon/estimators/gaussian_fullcov.py`) — a conditional Gaussian `N(μ(x), Σ)` in standard-normal latent space. Cheap, robust, but cannot represent multimodal / non-Gaussian posteriors.
- `Flow` (`src/falcon/estimators/flow.py`) — a conditional+marginal normalizing-flow pair (sbi-backed `FlowDensity`) with importance-sampling for posterior/proposal. Expressive, but training is heavier/fiddlier and it works in a hypercube latent space.

We want a new estimator that gets the best of both: use a learned Gaussian as a **whitening preconditioner** so a **flow-matching** model only has to learn the *residual* non-Gaussian structure in a well-conditioned (≈ N(0,I)) space. Prototyping in `diffusion_dev/` (flow matching N(0,I) → targets, CNF density, the affine-bridge whitening trick, Euler/Hutchinson cost benchmarks) showed: flow matching trains fast and robustly, sampling is ~free, density is cheap via Hutchinson, and **whitening the target to ≈N(0,I) is what makes the flow's job easy** — exactly the role the Gaussian preconditioner plays here, but now with *condition-dependent* moments.

Decisions already made with the user: (a) **flow-matching backend** (not sbi NFs); (b) implement as a **real wired falcon estimator** now. Target hardware: CUDA (A100 available).

## Design overview

Work in the **standard-normal latent space** of the `Product` prior (`simulator_instance.inverse/forward(θ, mode="standard_normal")`, free params only — same mechanism `GaussianFullCov` uses).

Two parallel "Gaussianized flow" densities over the latent `θ_lat`, mirroring `Flow`'s conditional+marginal pair:

1. **Conditional** `p_cond(θ_lat | x)`: Gaussian `μ(x)` (MLP, normalized inputs) + global residual covariance `Σ_cond` (EMA + eigendecomp), then a conditional flow-matching net on the whitened residual `w_cond = Σ_cond^{-1/2}(θ_lat − μ(x))`.
2. **Marginal** `p_marg(θ_lat)`: unconditional Gaussian `μ0` + global covariance `Σ_marg`, then a marginal flow-matching net on `w_marg = Σ_marg^{-1/2}(θ_lat − μ0)`.

Each whitened target is ≈ N(0,I), so each flow either learns ≈identity (Gaussian posterior) or just the residual multimodality/skew. Sampling and density then feed `flow.py`'s importance-sampling machinery (posterior / proposal modes) — **with one correction**: the latent prior is N(0,I) (not uniform), so it must be added to the weights.

## Files

Estimator name: **`GaussianizedFlowMatching`** (YAML `_target_: falcon.estimators.GaussianizedFlowMatching`). Training structure: **joint single loop** (recommended below; fits the StepwiseEstimator one-loop-per-node design).

**New:**
- `src/falcon/estimators/flow_matching.py` — backend, reusable, framework-agnostic:
  - `VelocityField(nn.Module)` — MLP + Gaussian-Fourier time embedding (port from `diffusion_dev/flow_matching_bimodal.py`).
  - `EMA` helper (weight averaging).
  - `fm_loss(net, w1, t_sampler)` — `‖v(w_t,t) − (w1−w0)‖²`, `w0~N(0,I)`, `t~U(0,1)`.
  - `euler_sample(net, n, dim, steps)` — `w0~N(0,I)` → ODE → `w1`.
  - `cnf_logprob(net, w, steps, divergence="exact"|"hutch", n_probe)` — backward Euler CNF: `logN(w0;0,I) + ∫ ∇·v dt`; `div_exact` (sum `∂v_i/∂w_i`, d VJPs) for low dim, `div_hutch` (Rademacher probes) for high dim.
- `src/falcon/estimators/gaussianized_flow_matching.py` — the estimator + preconditioner:
  - `_GaussianizedFlow(nn.Module)` — one preconditioner+flow pair; FlowDensity-compatible surface: `loss(θ_lat, s)`, `sample(n, s) -> θ_lat`, `log_prob(θ_lat, s)`. Holds the Gaussian (μ-MLP, EMA input/output norm buffers, EMA `Σ`, cached `V`,`λ`) — pattern lifted from `_GaussianPosterior` (`gaussian_fullcov.py:23–212`) — plus a `VelocityField`+EMA.
  - `GaussianizedFlowMatching(StepwiseEstimator)` — the estimator class.

**Modified:**
- `src/falcon/estimators/__init__.py` — add `"GaussianizedFlowMatching": "falcon.estimators.gaussianized_flow_matching"` to `_LAZY_IMPORTS`.

**Test asset:**
- `examples/06_gaussianized_flow_matching/` (adapt `examples/02_bimodal` or `04_gaussian`) — a bimodal posterior to exercise the non-Gaussian residual; `config.yml` with `_target_: falcon.estimators.GaussianizedFlowMatching`.

## Component detail

### `_GaussianizedFlow.loss(θ_lat, s)`  (s = `s*0` for marginal)
```
μ = mean_mlp(whiten_inputs(s))         # μ(x); for marginal s*0 -> constant μ0
update EMA input/output norm stats; update EMA residual cov Σ; refresh eig(V,λ) every k steps
nll = gaussian_nll(θ_lat, μ, Σ)        # trains μ-MLP + tracks Σ  (as in GaussianFullCov)
w   = (1/√λ) · Vᵀ (θ_lat − μ.detach()) # whiten with DETACHED μ,Σ (flow ⊥ Gaussian grads)
fm  = fm_loss(flow_net, w, s)          # flow-matching loss in whitened space
return nll + fm
```

### `_GaussianizedFlow.sample(n, s)`
```
w  = euler_sample(flow_net_ema, n, dim, sample_steps)   # N(0,I) -> w
return μ(s) + V·diag(√λ)·w                              # unwhiten -> θ_lat
```

### `_GaussianizedFlow.log_prob(θ_lat, s)`  (density over θ_lat)
```
w = (1/√λ)·Vᵀ(θ_lat − μ(s))
return cnf_logprob(flow_net_ema, w, density_steps, div) − 0.5·Σ log λ
#                                                      └ constant whitening log-det
```

### `GaussianizedFlowMatching.train_step(batch)`
```
θ      = to_tensor(batch[theta]);  θ_lat = simulator_instance.inverse(θ, "standard_normal")
s      = embed(conditions)                       # shared embedding for cond Gaussian + cond flow
loss   = cond.loss(θ_lat, s) + marg.loss(θ_lat, s*0)
opt.zero_grad(); loss.backward(); opt.step(); sched.step()
cond.flow_ema.update(); marg.flow_ema.update()
(optional) discard low-log-ratio samples  # mirror flow.py discard_samples
return {"loss": cond_fm, "loss_aux": marg_fm}
```
Single joint optimizer over μ-MLPs + flow nets + embedding (mirrors `GaussianFullCov`'s single-loop, on-the-fly-stats style). `val_step` runs the same forward without grads / EMA / discard. `on_epoch_end` checkpoints best EMA weights + steps the LR scheduler (copy `flow.py`'s structure).

### Importance sampling — adapt `flow.py:_importance_sample`
Reuse verbatim except the latent-prior term. In standard-normal latent space:
```
proposals θ_lat ~ mixture(cond.sample, marg.sample)         # balance-heuristic mixture
log p_cond = cond.log_prob(θ_lat, s);  log p_marg = marg.log_prob(θ_lat, s*0)
log_prior  = logN(θ_lat; 0, I)                              # NEW — replaces hypercube penalty
posterior:  log_target = log p_cond − log p_marg + log_prior
proposal :  log_target = (γ/(1+γ))·log p_cond  (+ prior term — see risk)
log_w = log_target − log_g_mix ; normalize ; multinomial resample
θ = simulator_instance.forward(θ_lat_selected, "standard_normal")   # back to physical at the very end
```
`sample_prior` = `simulate_batch` (as in flow.py). Keep `nan_replacement`, `num_proposals`, `gamma`, `proposal_mixture_beta`, `discard_samples`/`log_ratio_threshold`.

**Prior-width safeguard (like GaussianFullCov) — TWO layers:**
1. **Covariance-level clamp (primary).** Clamp the residual-covariance eigenvalues (equivalently the per-dim output std) at the **prior variance (= 1 in std-normal latent space)**, mirroring GaussianFullCov's `self._output_std.clamp(max=1.0)`. This is the real stabilizer for parameters that are hard to constrain in a unimodal way: their posterior can never become *wider* than the prior. Applied inside `_GaussianizedFlow`'s covariance EMA/eigendecomp update, for both conditional and marginal.
2. **Sample-level truncation (backstop).** After the full transform (flow + unwhiten), clip/penalize `θ_lat` beyond a configurable prior-width bound (`prior_sigma_bound`, std-normal units) on both proposal and posterior sampling, catching anything the flow re-widens.

Both clamp at the **prior width specifically** (posterior support ⊆ prior support), so they stay safe for genuinely multimodal/wide posteriors — they only suppress prior-exceeding runaway, never real modes.

### save / load  (mirror `flow.py:save/load`)
Persist, for cond & marg each: μ-MLP weights, input/output EMA norm buffers, `Σ`/`V`/`λ`, `μ0` (marg), flow-net EMA weights; plus embedding weights, `init_parameters`, `_total_epochs_trained`, and the `history` arrays.

### Registration & YAML
`__init__.py` lazy entry as above. Config mirrors `examples/04_gaussian/config.yml`, with estimator keys: `max_epochs, lr, gamma, batch_size, early_stop_patience, embedding`, Gaussian (`hidden_dim, num_layers, momentum, min_var, eig_update_freq`), flow (`flow_hidden, flow_layers, time_dim, ema_decay, sample_steps, density_steps, divergence: exact|hutch|auto, n_probe`), and IS (`num_proposals, proposal_mixture_beta, discard_samples, log_ratio_threshold`).

## Verification

1. **Unit / math checks** (standalone, `diffusion_dev/`-style, on CUDA):
   - CNF density normalizes (∫ p ≈ 1) in whitened space; matches `exact` vs `hutch` within the measured ~0.1-nat noise.
   - Whitening round-trips: `unwhiten(whiten(θ_lat)) == θ_lat`; log-det `= 0.5 Σ log λ`.
   - Gaussian-only sanity: if the target is Gaussian, the conditional flow learns ≈identity and `log_prob` matches the analytic Gaussian.
2. **End-to-end**: `falcon launch -c examples/06_gaussianized_flow_matching` on the bimodal example; then `falcon sample posterior -o <out>` and corner-plot the samples against the known bimodal posterior — modes recovered, no mode collapse. Compare against `Flow` and `GaussianFullCov` on the same example (quality + wall-time).
3. **Smoke test** in `tests/` mirroring `tests/test_examples_smoke.py` for the new example.

## Risks & open decisions

- **Co-evolving μ,Σ early in training** shift the whitened target under the flow. Mitigation: detach μ,Σ in whitening (planned); optionally a `prior_epochs`-style warmup that trains the Gaussian (NLL only) before enabling the flow loss — reuse `flow.py`'s `prior_epochs` pattern.
- **Proposal-mode prior term**: whether `(γ/(1+γ))·log p_cond` needs an added prior term in standard-normal space. Decide during impl (the proposal is heuristic; likely add a fractional prior term for support). Flag in code.
- **Exact vs Hutchinson** divergence threshold: gate on `param_dim` (e.g. exact ≤ 8–16, else Hutchinson). Expose as `divergence: auto`.
- **IS density cost**: weights need `num_proposals × density_steps` flow evals per query; with exact trace that is `×param_dim` more. Hutchinson probe count `n_probe` is a **tunable config knob** (not hardcoded) trading density noise vs cost; small values already suffice for the IS path (0.1-nat noise is immaterial to normalized weights — verified in `diffusion_dev`), raise it for cleaner reported densities.
- **Embedding sharing**: one embedding `s` feeds both the conditional Gaussian and the conditional flow (single network, jointly trained). Marginal side uses `s*0`.
- **Eigendecomp stability**: keep `min_var` Tikhonov reg + eigenvalue clamping (min `min_var`, **max prior variance = 1**) from `GaussianFullCov`; refresh eig every `eig_update_freq` steps.
- **Discard-logic space consistency**: `discard_samples` compares the conditional log-prob to the buffer's `theta_logprob` (physical space); our density is in latent space, so handle the prior-transform Jacobian consistently (same latent/physical subtlety `flow.py` already tolerates) or compare in latent space throughout.
- **Density self-consistency**: sampling and CNF density must use the SAME EMA flow weights. Hutchinson `n_probe` is tunable via config (not hardcoded); small values suffice on the IS path, larger for reported densities.
- **Lazy init + t-sampling**: discover `param_dim`/embedding dim on first batch (`_initialize_networks`, like flow.py); default FM training `t ~ U(0,1)` (uniform beat logit-normal for sharp features in the prototype).
- **Co-evolving μ,Σ warmup default**: whether the Gaussian-only warmup is on by default (recommended for stability) or opt-in via config.
