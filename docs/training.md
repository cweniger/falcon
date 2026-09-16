# Training Loop

Falcon's estimators (`Flow`, `GaussianFullCov`) train in **rounds**. Each round
trains on a fixed snapshot of the simulation buffer until it converges, then
tests the result against the best network so far on the same validation data.
Only a network that wins replaces the best one, so the best network can only
get better. Simulation keeps running in the background, using the best network
as the proposal.

## Terminology

These words have exactly one meaning throughout the docs, the log output and the
metric names.

| Term | Meaning |
|------|---------|
| **step** | One optimizer update on one training batch |
| **epoch** | One pass over the round's training set |
| **validation** | One full pass over the round's validation set, producing the `val:*` metrics |
| **round** | Refresh data → train epochs → acceptance test → if accepted: promote and run the discard sweep |
| **current network** | The network being trained in this round |
| **best network** | The network used for proposals, posterior sampling and saving |
| **accepted / rejected round** | The current network beat / did not beat the best network on the round's validation set |

Two rules keep the parameters unambiguous:

1. Every count parameter ends in its unit: `_epochs` or `_rounds`. Validations are
   never used as a unit, so changing how often you validate changes the cost of
   training, not how long it waits for improvement.
2. All `*_epochs` counts are per round: they reset when a new round starts.

## One round

```text
refresh training and validation data from the buffer      (fixed for the round)
copy best network -> current network, reset learning rate

epoch 1, 2, ..., max_epochs:
    one shuffled pass over the training set (one step per batch)
    every val_every_epochs epochs, and at the last epoch:
        validate; remember the current network if it is the best epoch so far
        stop the round if the best epoch is patience_epochs epochs old

restore the current network to its best validated epoch
validate current and best network on the same validation set
promote every network group whose validation loss improved
if the primary group was promoted (round accepted):
    run the discard test once over all training and validation samples
```

Training ends after `patience_rounds` rejected rounds in a row, after
`max_rounds` rounds, or on a graceful stop.

### Network groups

Networks that belong together are compared and promoted together:

- **`Flow`**: the conditional flow together with its embedding (primary group,
  judged on `loss`), and the marginal flow (judged on `loss_aux`). Each is
  promoted independently; the round counts as accepted when the conditional
  flow improves.
- **`GaussianFullCov`**: the Gaussian posterior together with its embedding.

### Proposals during training

Simulation runs asynchronously while the estimator trains, and proposal sampling
is served between any two training steps. Proposals always come from the best
network, which only changes when a round is accepted. Until the first round has
been accepted, and for the first `prior_rounds` rounds, proposals come from the
prior.

### Discarding samples

With `discard_samples: true`, every accepted round ends with one discard sweep:
each training and validation sample is tested once with the new best network,
and samples that fail are marked for eviction from the buffer.

### Graceful stop

A graceful stop (Ctrl+C in the interactive display, or `--timeout`) takes effect
after the current training step. Progress since the last validation is dropped,
the round's acceptance test still runs, and the discard sweep is skipped. The
saved networks are the best networks.

## Parameters

All parameters are flat keyword arguments of the estimator (see
[Flow](api/flow.md) and [GaussianFullCov](api/gaussian.md)).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `max_rounds` | `null` | Maximum number of rounds (`null` = unlimited); a resumed run counts the rounds trained before |
| `patience_rounds` | 10 | Stop training after this many rejected rounds in a row |
| `max_epochs` | 100 | Maximum epochs per round |
| `patience_epochs` | 16 | End the round once the best validation loss is this many epochs old |
| `val_every_epochs` | 1 | Validate after every N-th epoch (and always at the last epoch of a round) |
| `lr_decay_factor` | 1.0 | Learning rate multiplier on a plateau (1.0 = constant learning rate) |
| `lr_patience_epochs` | 8 | Epochs without validation improvement before the learning rate decays |
| `prior_rounds` | 0 | Rounds that simulate from the prior before switching to the learned proposal |
| `discard_samples` | `true` (Flow) | Run the discard sweep after accepted rounds |
| `batch_size` | 128 | Samples per step |

Patience is checked only when validating, so with `val_every_epochs: 3` and
`patience_epochs: 12` a round ends at the fourth validation without improvement.
If `patience_epochs` is not a multiple of `val_every_epochs`, it is effectively
rounded up (a warning is logged).

### Expensive validation

When a validation pass is slow (for example densities that require solving an
ODE), validate less often and express the patience in the same epochs as before:

```yaml
estimator:
  max_epochs: 200
  val_every_epochs: 3
  patience_epochs: 12
  patience_rounds: 10
```

## Log output and metrics

Each epoch prints one line; validation fields appear only for validated epochs:

```text
Round 1 | epoch 9/300 | steps=243 | train_loss=-3.619e+00 | val_loss=-5.202e+00 | lr=1.000e-02
```

Each round prints its decision:

```text
Round 1 ACCEPTED | epochs=271 | n_train=3481 n_val=615 | n_sims=32128 | conditional: -1.405e+01 vs best none promoted | marginal: 4.209e+00 vs best none promoted
```

| Metric | Description |
|--------|-------------|
| `round`, `epoch` | Current round, and epoch within the round |
| `train:*` | Metrics of every training step |
| `val:*` | Metrics of each validation |
| `checkpoint:<group>` | Epoch at which a group reached its best validation loss in the round |
| `round:accepted` | 1 if the round was accepted |
| `round:<group>:candidate`, `round:<group>:best` | Validation losses compared in the acceptance test |
| `round:<group>:promoted` | 1 if the group replaced its best network |
| `round:epochs`, `round:n_train`, `round:n_val` | Size of the round |
| `n_samples` | Samples simulated in total, recorded once per round |
| `round:n_discarded_train`, `round:n_discarded_val` | Samples discarded by the sweep |

## Migrating from the epoch-based loop

| Old parameter | New parameter |
|---------------|---------------|
| `max_epochs` (total) | `max_epochs` (per round) |
| `early_stop_patience` | `patience_epochs` |
| `lr_patience` | `lr_patience_epochs` |
| `prior_epochs` | `prior_rounds` |
| `cache_sync_every` | removed: data is refreshed once per round |
