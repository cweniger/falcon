# FlowMatching Estimator

Flow-matching posterior estimation with a truncated-prior proposal.

## Overview

`FlowMatching` learns a conditional flow `q_c(u | x)` and a marginal flow
`q_m(u)` over the standard-normal latent space `u` of the prior. Each flow is a
velocity field `v(w, t, s)`, an MLP trained by flow matching: it regresses the
straight-line bridge from N(0, I) at `t = 0` to the data at `t = 1`. Draws
come from integrating the ODE forward with Euler steps. Densities come from
the continuous change of variables, with the ODE run backward.

Key features:

- **Exact per-round whitening.** Each flow works in the frame `w` of a ZCA
  whitener that is refit to the round's training data, so the velocity field
  always sees a unit-scale target, however far the data has contracted. The
  latent-frame NLL includes the whitener's log-determinant, so rounds with
  different whiteners are still compared correctly.
- **Moving averages.** The evaluated, published and saved networks are
  exponential moving averages of the trained ones.
- **Truncated-prior proposal.** Proposals are prior samples truncated to a
  region read off the conditional flow. A hysteretic two-region ladder moves
  that region once per round (below).
- **No importance weights for the posterior.** The training data is the
  truncated prior, so the conditional flow itself estimates the posterior.
  Posterior samples are conditional-flow draws cut at its `x_sigma` contour.

!!! note
    `FlowMatching` requires a `TransformedPrior` such as [`Product`](product.md);
    it works in the `"standard_normal"` mode.

## The proposal: a region ladder

A region is a level set `{ln q_c(u) > thr}` of a conditional flow at the
observation, intersected with its parent region. A region is frozen when it is
minted. It keeps copies of the flows it was read off, the embedded observation
and its threshold, so it keeps meaning the same thing as the networks change.

Two nested regions are live:

- the **inner region I** must contain the `x_sigma` mass of the current
  conditional flow;
- the **outer region O** is what proposals sample, so the training data
  reaches beyond I.

Wider regions wait on a stack. After every round, the train actor measures
`m_out`, the current conditional flow's mass outside I, and moves the ladder:

| Condition | Action |
|-----------|--------|
| `m_out > leak(x_sigma)` | **EXPAND**: I ← O, O ← pop the stack (or the prior) |
| otherwise | mint a candidate at `leak(x_sigma + delta_sigma)` inside I |
| candidate mass ≤ `vratio` × mass of I (with `v_min_ess`) | **CONTRACT**: push O, O ← I, I ← candidate |
| otherwise | hold |

`leak(x)` is the two-sided normal tail mass: `x = 3` means 2.7e-3 of the mass
may lie outside I. A region is sampled by importance sampling. The sampler
draws from the region's marginal flow, keeps the draws inside the region,
weights them by prior over marginal flow, and resamples without replacement,
so no simulation is spent twice. The pool of weighted draws is extended until
its effective sample size is `ess_factor` times the draws taken from it, so
proposals stay faithful samples of the truncated prior even where the
marginal flow has a hole.

The ladder needs the node's conditions at the observation. The train actor
gets them at launch; evidence derived from observed nodes is simulated once
from the observation. Without them, e.g. in amortized runs, the proposal stays
the prior.

The ladder logs `[mint]` and `[ladder]` lines to `graph/<node>/train/output.log`,
and `ladder:*` metrics. The sample actor logs `proposal:acceptance`,
`proposal:eps` (the ESS per accepted draw) and `proposal:pool_ess`.

## Reference settings

The defaults are the settings of the reference run that produced the O1b
region of the LDC MBHB study (`t7b_s4k_b8k_max64k_reject_essgate`: 9
parameters, 84 rounds, 3.3e5 simulations on one A100). Its buffer policy maps
onto falcon's buffer as follows:

| Reference (O1b) | falcon |
|-----------------|--------|
| drop samples outside the new outer region, oldest first, down to a floor of 8000 | `discard_samples: true` (default) with `buffer.min_samples: 8000` |
| stop generating while 64000 samples are inside the region | `buffer.max_samples: 64000`, `buffer.simulate_when_full: false` |
| 20% validation, handled like the training set | `buffer.validation_fraction: 0.2` |
| 4000 new simulations per round | no exact counterpart: falcon simulates `simulate_count` every `simulate_interval` seconds |

Keeping the buffer inside the outer region matters: the whitener can only
zoom in as far as the training data has contracted.

## Configuration

```yaml
estimator:
  _target_: falcon.estimators.FlowMatching
  max_epochs: 200
  val_every_epochs: 3        # validation solves the density ODE
  patience_epochs: 12
  lr: 3.0e-4
  embedding:
    _target_: model.E
    _input_: [x]
  hidden: 256
  layers: 4
```

See `examples/01_minimal/config_flow_matching.yml` for a complete example.

## Configuration Reference

### Networks and training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden` | int | 512 | Velocity-field MLP width |
| `layers` | int | 6 | Velocity-field MLP hidden layers |
| `time_dim` | int | 32 | Number of Fourier time features (even) |
| `layernorm` | bool | false | LayerNorm after every hidden layer |
| `lr` | float | 3e-4 | Learning rate; reset at the start of every round |
| `betas` | tuple | (0.9, 0.9) | AdamW betas |
| `grad_clip` | float | 1.0 | Gradient-norm clip, per flow (0 = off) |
| `ema_decay` | float | 0.995 | Decay of the moving averages, per step |
| `embedding_epochs` | int | 10 | Train the embedding only in the first N epochs of each round, then keep it fixed (null = throughout) |
| `time_late_k`, `time_late_mix` | float | 8.0, 0.5 | Training times: a `mix` share drawn from `k t^(k-1)`, which favours the sharp end of the path |
| `sample_steps` | int | 128 | Euler steps per draw |
| `density_steps` | int | 32 | ODE steps per density |
| `div_probes` | int | 4 | Hutchinson probes for proposal and posterior densities (0 = exact trace; validation is always exact) |
| `eval_chunk` | int | 16384 | Rows per forward pass when sampling or evaluating densities |

### Region ladder

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x_sigma` | float | 3.0 | Mass the inner region must hold, in sigmas; also the posterior truncation |
| `delta_sigma` | float | 1.0 | Dead band: new inner regions are minted at `x_sigma + delta_sigma` |
| `min_keep_frac` | float | 0.01 | At least this share of a region's own marginal-flow draws fall inside it |
| `vratio` | float | 0.8 | Contract only if the candidate has at most this share of the inner region's prior mass |
| `v_min_ess` | float | 1000 | Effective sample size that mass ratio must carry |
| `v_max_draws` | int | 1048576 | Maximum marginal-flow draws spent on it |
| `chain_depth` | int | 1 | Ancestors tested for region membership (-1 = all) |
| `n_region` | int | 65536 | Draws per pass when minting or sampling a region |
| `n_mout` | int | 65536 | Draws for `m_out` |
| `max_sample_passes` | int | 64 | Maximum passes per proposal request |
| `ess_factor` | float | 4.0 | Extend the pool of region draws until its ESS is this many times the draws taken from it |
| `readout_draws` | int | 65536 | Draws that set the posterior truncation |

The defaults are the O1b settings (above), sized for an A100; on a CPU,
reduce `hidden`, `layers` and the draw counts.

The training loop parameters (`max_epochs`, `patience_epochs`,
`val_every_epochs`, `max_rounds`, `patience_rounds`, `prior_rounds`,
`batch_size`, `lr_decay_factor`, `lr_patience_epochs`, `cache_on_device`,
`max_cache_samples`) are as in [Flow](flow.md#configuration-reference); see
[Training Loop](../training.md). `discard_samples` defaults to true here:
accepted rounds discard samples outside the outer region.

## Training details

- **Network groups.** The first group is the conditional flow together with
  the embedding, judged on `nll`. The second is the marginal flow, judged on
  `nll_aux`. Both are latent-frame NLLs with the exact trace. Early stopping
  watches `loss`, the conditional NLL without the rare rows whose ODE ran
  away. The training loss is the flow-matching loss, so the `train_loss` and
  `val_loss` in the log are on different scales; `val_fm` is the validation
  flow-matching loss.
- **Whitener frames.** Every round refits each flow's whitener to the round's
  training data before the first epoch. A flow that was not promoted in the
  previous round keeps the frame of the best network it resumes from. Once
  the proposal region has settled, rounds can alternate between a rejected
  round in a refit frame and an accepted round in the held one; the O1b
  reference run did this from round ~50 on.
- **Discretisation.** Densities and draws come from fixed-step Euler
  integration. When the conditional flow must contract its target strongly,
  the density at the default steps is biased high by a fraction of a nat. The
  whitener's zoom, as the proposal region contracts, keeps that contraction
  moderate.

## Class Reference

::: falcon.estimators.flow_matching.FlowMatching
    options:
      show_source: true
      members:
        - __init__
        - build
        - train_step
        - val_step
        - on_train_start
        - on_round_end

::: falcon.estimators.region_ladder.RegionLadder
    options:
      show_source: false
