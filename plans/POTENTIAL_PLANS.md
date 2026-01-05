# Potential Future Improvements

## PRs Created (Pending Merge)

1. **PR #6** - BufferView/Batch refactoring (refactor/smart_batch)
   - Bidirectional Batch class with dict-based conditions
   - Framework-agnostic batch (numpy arrays instead of torch tensors)
   - BufferView for estimator-controlled data access

2. **PR #7** - Remove Hydra, add `--run-dir` with memorable names
   - Replace hydra-core with OmegaConf
   - Auto-generated run names: `adj-noun-YYMMDD-HHMM`
   - Config resume logic from `run_dir/config.yaml`

3. **PR #8** - Stable graph init (wait for actor initialization)
   - Wait for actors with `__ray_ready__`
   - Error handling for init failures
   - "Spinning up graph..." feedback

4. **PR #9** - `falcon graph` DAG visualization
   - Git-style ASCII graph
   - Shows forward model structure and inference direction

## Discussed But Deferred

1. **`get_status()` method for nodes**
   - Return node status during training (epoch, loss, state)
   - Useful for monitoring and periodic logging
   - Could log every N minutes

2. **Docstring cleanup**
   - Found 26 issues across codebase
   - 4 critical, 7 high priority, 15 medium/low
   - Recommended as separate PR

3. **Crossing lines in graph visualization**
   - Current impl doesn't handle complex DAGs with crossings
   - Would need graph layout algorithm
   - Sufficient for typical SBI graphs

## Potential New Features

1. **Config validation / resume safety**
   - Warn when `./config.yaml` differs from `run_dir/config.yaml`
   - Schema versioning for config compatibility
   - `--force-overrides` flag for resume with changes

2. **`falcon status` command**
   - Show training progress across all nodes
   - Current epoch, loss, learning rate, convergence state
   - Could poll actors periodically

3. **`falcon resume` shorthand**
   - Shorthand for `falcon launch --run-dir <existing>`
   - Tab-complete existing run directories
   - Show last run status before resuming

4. **Better error messages**
   - Config loading failures with suggestions
   - Missing module imports with helpful hints
   - Actor initialization failures with context

5. **Retry logic for actor initialization**
   - Configurable retry count for transient failures
   - Exponential backoff
   - Graceful degradation for non-critical nodes
