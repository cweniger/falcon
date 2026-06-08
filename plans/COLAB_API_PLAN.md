# Plan: Notebook / Colab API for Falcon

> **Status: living plan**, tracked by issue #58. The phasing and open questions
> here are roadmap and become obsolete as work lands (the checklist of record is
> the issue). The design rationale (the config-shape taxonomy, the flat surface,
> the `_target_` rule, the Ray lifecycle, the JAX notes) is durable and will
> graduate to a permanent design doc under `docs/` once implementation is well
> underway; this file is then pruned or removed.

## Motivation

Falcon is currently CLI-first: the only supported entry point is `falcon launch`,
and the runtime, the blessed TUI, and signal handling are fused inside
`launch_mode` in `falcon/cli.py`. This works for batch jobs on a workstation or
cluster, but it makes the framework hard to teach and hard to explore.

The intended end state is **pedagogical and expert-friendly at the same time**: a
set of notebooks that show off what Falcon does, where a learner can tweak a
config value or define a brand-new simulator in a cell and immediately re-run
inference, and where an ML expert finds an API consistent with the libraries
they already know (sklearn, Keras, sbi). For that, Python (not the shell) has to
be a first-class front door. The CLI should become one frontend over a clean
core, not the core itself.

This plan is the design for that API. It is independent of the end-of-run
summary work (PR #56).

## Status: decisions taken

The following were settled during design discussion and are treated as decided
in the rest of this document:

- **Flat config surface, committed.** Config is set through flat, prefixed,
  typed keyword arguments (`loop_num_epochs=600`), not nested dicts or nested
  config objects. The YAML file stays nested; a deterministic transform bridges
  the two.
- **`launch()` blocks by default.** A non-blocking `launch(wait=False)` mode is a
  supported, real escape hatch, justified by the live-monitoring use case.
- **Ray cluster setup is separate from running a graph.** `falcon.init()` sets up
  or connects to Ray; `launch()` reuses it.
- **The prior list-syntax is kept** as `Product`'s own field encoding. It is not
  desugared into `_target_` blocks, and the existing list-of-lists syntax also
  serves as the Python API — no separate typed-marginal objects needed for v1.
- **Worked notebooks are the onboarding vehicle.** The per-run auto-saved
  `config.yml` is the bridge to the CLI.
- **The v1 notebook display is one interleaved, color-tagged log stream** (driver
  plus all node logs). Structured ipywidgets displays are optional phase 2.
- **`falcon.Simulator` base class is not needed for v1.** Duck typing is
  sufficient; a base class adds value only when actor-environment hooks (JAX
  passthrough) are implemented.
- **`falcon.session()` context manager is deferred.** Not needed before the basic
  API works; add later for CI scoped lifetimes.

Still open (see Open Questions): the outcome of the cloudpickle spike (gates
everything notebook-class-related) and the exact flattened-signature parameter
counts.

## Design principles

1. **Outside-in.** The API is whatever makes the target notebook cells (below)
   read cleanly. We design the cells first and the surface second.
2. **The CLI and the API are siblings, not a hierarchy.** Both are thin wrappers
   over one pure pipeline function. Neither imports the other's concerns (no TUI
   in the API, no `Run`-returning in the CLI).
3. **The surface style of a config slot is derived from its type, not chosen by
   taste.** See the config-shape taxonomy below. This is what keeps a partly-flat,
   partly-object API from looking arbitrary.
4. **No `**kwargs` in any public constructor.** Jedi (Colab/Jupyter completion)
   cannot introspect `**kwargs`; a single `**kwargs` destroys autocomplete and
   the YAML-to-API mapping for that call. Every public signature is explicit.
5. **No hidden magic.** No environment auto-detection, no notebook-vs-terminal
   heuristics, no silently writing user cell code to temp modules.
6. **Borrow, do not invent.** This is a solved problem (Hydra, Pydantic, spaCy).
   See Standard Precedents. The only genuinely new piece is the notebook-class
   escape hatch.
7. **Everything is inspectable.** Configs, graphs, and runs all get rich notebook
   reprs.
8. **The notebooks are the spec and the test.** Example notebooks are executed in
   CI; if the API drifts, the notebooks break.

## Audience: experts and students, one design

The API serves ML experts and students with a single surface, not two.

- For **experts**, the flat typed kwargs, the sum-as-object pattern, the
  autocomplete-first design, and the YAML round-tripping are idiom-consistent
  with sklearn, Keras, and HuggingFace, and serve real needs (hyperparameter
  tuning, reproducible config, sweeps).
- For **students**, the same surface is reached through worked example
  notebooks. A beginner starts by mutating a working notebook, never by
  constructing config from a blank cell. The build-from-scratch surface is a
  later lesson.

There is no separate beginner API to maintain. The expert-optimal design is
gentle enough to be a teaching destination, provided the on-ramp is
example-driven. The example notebooks carry the beginners; the API carries the
experts.

## The two registers

| Register | Teaching mode | Entry | Backing object |
|----------|---------------|-------|----------------|
| **Config** | "Change a knob, re-run" | `falcon.config("config.yml")`, `.override(...)` | `Config` (wraps `DictConfig`) |
| **Programmatic** | "Build your own model" | `falcon.Graph()`, `.add_node(...)` | `Graph` |

Both lower to the same runtime `Graph`. YAML stays valid forever; the
programmatic path is additive. The two are mixable: a config can embed a live
Python class as a node's `simulator`, and a programmatically built `Graph` can be
launched with flat run-level overrides.

## The config-shape taxonomy

Every config slot has one of four shapes. The shape determines the API surface.
This is the single rule that makes the whole design coherent.

| Shape | Definition | Example | API surface | YAML form |
|-------|------------|---------|-------------|-----------|
| **Product** | Fixed set of fields, all always apply | `loop`, `optimizer`, `buffer` | Flat prefixed kwargs: `loop_num_epochs=600` | nested block / dotted keys |
| **Sum** | Pick one of N; valid fields depend on the choice | flow architecture, `estimator`, `simulator` | A typed object; the class is the choice: `network=Flow.MAF(...)` | tagged block: `{_target_: maf, ...}` |
| **Composite** | Recursive; structure is data, depth unbounded | embedding pipeline | A construction expression: `Sequential(...)` | recursive `_target_`/`_input_` |
| **Collection** | List of N homogeneous components | `priors` | A list of typed elements | a YAML list |

Why each is what it is:

- A **product** has a fixed, known field list, so it flattens into named kwargs.
  The names never change. Autocomplete shows one honest popup.
- A **sum** cannot be flattened: a flat `network_num_bins` is meaningless when the
  user picked MAF. So a sum stays an object, and the object's class is the choice.
  `Flow.MAF(...)` exposes exactly MAF's fields with autocomplete; `Flow.NSF(...)`
  exposes NSF's. Picking the constructor picks the valid field set.
- A **composite** cannot be flattened either, and is not even a fixed sum: an
  embedding `_input_` can itself be an embedding, or a list of them, to arbitrary
  depth. The surface is a construction expression, the same idiom as
  `torch.nn.Sequential`.
- A **collection** is a list, so its surface is a list. Each element is itself
  typed (and may itself be a product or sum).

**Flattening stops at sum, composite, and collection boundaries.** Products
flatten only within their owning scope. Crossing into a chosen-class object
restarts flattening inside that object. Consequence: `estimator_loop_num_epochs`
never exists. `estimator` is a sum slot (Gaussian, Flow, or none), so it is an
object, and the `loop_*` flattening happens inside the chosen estimator:
`estimator=Gaussian(loop_num_epochs=600)`.

The shape is **per-slot and per-estimator**, not global. Worked example: the
`network` slot is a product in `Gaussian` (a fixed MLP-shaped posterior network,
flat `network_hidden_dim=...`) but a sum in `Flow` (13 architectures, object
`network=Flow.MAF(...)`). Same field name, different shape, because the
underlying type differs. The surface follows the type.

## The flat config surface and the YAML bridge

### Flat Python, nested YAML

The Python API shape and the `config.yml` shape are connected by a mapping, not
by identity. The YAML stays nested (readable, editable, organizationally
grouped); the Python surface is flat (one autocomplete popup, all defaults
visible). They are bridged by a deterministic prefix transform:

```
Python kwarg            YAML key
loop_num_epochs    <->  loop.num_epochs
optimizer_lr       <->  optimizer.lr
network_hidden_dim <->  network.hidden_dim
```

The transform splits off only the first segment, and only when it is a known
section prefix. Field names may themselves contain underscores (`hidden_dim`)
with no ambiguity. The only constraint: section names contain no underscore.

### Implementation: synthesized signatures, no `**kwargs`

A flat `Gaussian(**kwargs)` autocompletes to nothing, so that is forbidden. But
the ~25 flat parameters must not be hand-maintained either. Instead:

- The nested `@dataclass` config classes (`GaussianConfig` with fields `loop`,
  `network`, `optimizer`, `inference`, etc.) remain the **single source of
  truth**: they are the YAML schema, the OmegaConf structured-config validation
  schema, and the basis for the flat signature.
- The flat `__init__` signature is **synthesized** from them: walk the nested
  fields, build an `inspect.Signature` of `loop_*` / `network_*` / etc.
  parameters with their annotations and defaults, and assign it to
  `Estimator.__init__.__signature__`. IPython and Jedi honor an explicit
  `__signature__`, so Colab's popup shows the full flat list with defaults even
  though the code does not literally spell them out.
- The constructor body expands `prefix_field` back into the nested `Config`.

One generator serves every estimator. Zero signature duplication.

### Defaults visibility

Scalar defaults show inline in the Colab signature popup. `field(default_factory=...)`
defaults show only as `<factory>`. Therefore:

- Prefer immutable scalar/tuple defaults over `default_factory`. For example
  `betas: tuple[float, float] = (0.9, 0.9)` shows in the popup; a
  `field(default_factory=lambda: [0.9, 0.9])` does not.
- `?` shows the docstring + signature; `??` shows the source (always the full
  truth); constructing the object with no args and reading its dataclass repr
  resolves every default including factory ones.
- Dataclass per-field docstrings do not surface in the popup, so each config
  class docstring enumerates its fields with units and semantics.

### Flat kwargs vs dotted overrides (do not conflate)

There are two distinct mechanisms:

- **Flat typed kwargs** are the *constructor surface* of a fixed-schema object
  (an estimator, the buffer). They autocomplete. Used in `Gaussian(loop_num_epochs=...)`.
- **Dotted-string overrides** are the *arbitrary-deep-path* escape hatch, used to
  override into a loaded `Config` whose paths include arbitrary user-chosen node
  names: `cfg.override("graph.theta.estimator.loop.num_epochs=150")`. These do
  not autocomplete; they cannot, because node names are data.

Flat kwargs are the discovery path; dotted overrides are the catch-all. They are
not interchangeable and the docs must keep them distinct.

## Step 0: unify `_target_` resolution — DEFERRED

**Decision (2026-06-08): Step 0 is deferred indefinitely.**

The original motivation was to replace `net_type` (a bare string discriminator)
with `Flow.MAF()` / `Flow.NSF()` variant classes so that per-variant
hyperparameters could be exposed with full autocomplete. That surface only becomes
load-bearing when those hyperparameters are actually exposed. Right now all 13
flow builders are called identically — `builder(theta, s, z_score_x=None,
z_score_y=None)` — so `net_type` is just a plain product field, and the
variant-class refactor is pure churn with no functional benefit.

If per-variant hyperparameters are ever needed before a full variant-class refactor
is worthwhile, the pragmatic escape hatch is `net_config: dict = {}` passed
through to the builder.

Similarly, `NetworkConfig` conflates architecture (`net_type`) and normalization
fields (`theta_norm`, `norm_momentum`, etc.), but untangling them has no practical
benefit until variant-specific params are added.

The prior list-syntax (`['uniform', -100, 100]`) is also left as-is: it is already
the Python API, the same list-of-lists form works in both YAML and notebook code,
and no typed-marginal object layer is needed.

**Consequence for sequencing**: implementation starts at Step 1.

## Target notebook UX (the spec)

### Cell story A: tweak a config (e.g. `examples/01_minimal` as a notebook)

```python
import falcon

cfg = falcon.config("config.yml")     # Config object, rich repr renders the YAML
cfg

cfg = cfg.override(                   # dotted strings for arbitrary deep paths
    "buffer.min_samples=2000",
    "graph.theta.estimator.loop.num_epochs=150",
)

run = falcon.launch(cfg)              # blocks; live progress in the cell
run                                   # rich repr: status, runtime, final losses, log paths

run.plot_metrics()
samples = run.sample_posterior(n=10_000)
falcon.corner(samples)
```

### Cell story B: define a new model (the "build your own" lesson)

```python
import falcon, torch

class MySimulator:                             # plain callable, duck typing
    def __call__(self, theta):
        return theta + 0.1 * torch.randn_like(theta)

graph = falcon.Graph()

graph.add_node(
    "theta",
    simulator=falcon.priors.Product([          # collection -> list of typed marginals
        ['uniform', -5, 5],
        ['uniform', -5, 5],
    ]),
    estimator=falcon.estimators.Gaussian(      # sum -> object; products inside flatten
        loop_num_epochs=300,
        optimizer_lr=1e-3,
        inference_gamma=0.5,
    ),
    evidence=["x"],
)

graph.add_node(
    "x",
    simulator=MySimulator(),                   # __main__ class, shipped via cloudpickle
    parents=["theta"],
    observed=obs_array,                        # ndarray accepted directly
    ray_num_gpus=0.5,                          # node-level product -> flat
)

graph                                          # rich repr: Mermaid DAG

run = falcon.launch(graph)
```

### A Flow estimator with a composite embedding

```python
estimator=falcon.estimators.Flow(
    loop_num_epochs=600,                       # product -> flat
    optimizer_lr=1e-3,                         # product -> flat
    network_net_type="maf",                    # product -> flat string field
    embedding={                                # composite -> nested _input_ config
        '_target_': 'MyCNN',
        'channels': 32,
        '_input_': {
            '_target_': 'falcon.embeddings.PCAProjector',
            'n_components': 64,
            '_input_': 'x',
        },
    },
)
```

These cells are the acceptance test: if a notebook needs an awkward cell, the API
is wrong.

## Architecture

### Step 1: extract the pure pipeline (no behavior change)

Split `launch_mode` in `falcon/cli.py` into three:

- `_run_pipeline(cfg, *, auto_sample, timeout, stop_check, log_sink) -> Path`:
  graph build, deploy, train, optional posterior sampling, end-of-run summary,
  teardown. No terminal control, no signal handlers, no Ray init/shutdown (see
  Ray lifecycle).
- `launch_mode(cfg, interactive, ...)`: CLI wrapper. Builds the TUI or the
  `_GracefulShutdown` handler, wires `stop_check`, owns Ray init/shutdown for the
  one-shot process, calls `_run_pipeline`.
- `falcon.launch(...)`: API wrapper (below).

`stop_check` and `log_sink` are injected, so the CLI passes TUI-aware versions
and the API passes notebook-aware ones. This refactor is worth doing on its own
merits (testability, separation of concerns) and ships with CLI behavior
byte-for-byte unchanged.

### The CLI conforms to the API

The test of the structure: `falcon launch` and `falcon.launch()` should read as
two thin adapters over one core, differing only in frontend (terminal TUI vs cell
output) and input format (argv vs Python objects). CLI flags map 1:1 onto API
parameters (`-o` to `output=`, `key=value` to `overrides=`,
`--no-auto-sample` to `auto_sample=`, `--timeout` to `timeout=`). If
the CLI ends up with pipeline logic the API path does not also exercise, the
split has leaked.

## The public API

```
falcon.init(**ray_init_kwargs)
falcon.config(source) -> Config                       # source: path | dict | DictConfig
falcon.launch(target, output=None, *, overrides=None,
              auto_sample=True, timeout=None, wait=True) -> Run | LaunchHandle
falcon.shutdown()
```

- **`Config`** wraps `DictConfig`: dict-like access, `.override(*dotted_strings)`,
  `.to_yaml()`, `_repr_markdown_`. OmegaConf does the real work.
- **`falcon.launch(target, ...)`** accepts a `Config` / dict / path **or** a
  `Graph`. Buffer, network, and other model config belongs in the config object
  (or via `overrides=`), not as kwargs on `launch()`. `output`, `timeout`,
  `auto_sample`, and `wait` are the only run-level options here. Cluster-level Ray
  config lives on `falcon.init()`. `launch()` calls `_run_pipeline` with no TUI
  and a notebook log sink.
- **`Run`** is returned (blocking mode). It gains methods, not top-level
  functions: `run.sample_posterior(n)`, `.sample_prior(n)`, `.sample_proposal(n)`
  (each writes NPZ for CLI parity and returns the samples), `run.plot_metrics()`,
  `run.status`, `run.runtime`, `run.config`. `load_run` is reused as-is. There is
  no `falcon.load` alias and no top-level `falcon.sample()`; sampling is a `Run`
  method because a `Run` owns the config and the trained graph.
- **`falcon.init(**ray_init_kwargs)`** is a thin wrapper around `ray.init()`.
  Named `num_cpus` / `num_gpus` parameters are omitted: when connecting to an
  existing cluster (`address=...`) they are meaningless; when starting a local
  cluster Ray detects resources automatically. Pass any `ray.init()` kwarg
  directly. Idempotent: a second call is a no-op.

### Blocking vs non-blocking

`launch()` **blocks by default** and returns a finished `Run`. This matches every
mainstream ML library (`model.fit()` in Keras, Lightning, `transformers.Trainer`,
sbi), gives the simplest mental model, and avoids a half-trained-`Run`
concurrency surface. The kernel-busy cost is mitigated by live in-cell progress
(see Live Monitoring) and a graceful interrupt: a notebook kernel-interrupt maps
to the existing graceful-stop machinery and returns a partial `Run`, not a
traceback.

`launch(wait=False)` is a **supported** non-blocking mode: training runs in a
background thread and `launch` returns a `LaunchHandle` with `.wait()`,
`.status`, `.stop()`, and the live-updating display. It exists because the
live-interactive-monitoring use case genuinely needs a free kernel. It is opt-in,
never the default, so the simple mental model stays intact for everyone who does
not need it.

### Programmatic graph builder

`Graph` and `Node` are already plain classes. Add `Graph.add_node`:

```
graph.add_node(name, simulator=..., estimator=None, parents=None,
               evidence=None, observed=None, ray_num_gpus=..., ray_num_cpus=..., ...)
```

- `simulator=` and `estimator=` are object slots (sums). `estimator` is optional;
  omitting it means the node is not inferred.
- `observed=` accepts an ndarray/tensor directly (the YAML path's
  `"file.npz['y']"` string is not forced on notebook users).
- `ray_num_gpus` etc. are node-level product config and flatten.
- `add_node` validates incrementally with notebook-friendly errors ("node 'x'
  lists parent 'theat', not defined; did you mean 'theta'?").

`falcon.Simulator` is a documented, optional base class so the "define your own"
lesson has an obvious starting point. Duck typing still works; the base class
anchors the docs and is where the actor-environment hooks live (see JAX).

## Ray lifecycle

Provisioning a real multi-node cluster is never `launch()`'s job; that happens
before any Falcon code runs, via Ray's own tooling, and Falcon only connects.
Starting a local Ray is cheap and a beginner should not have to think about it.
Either way, Ray is initialized **once per session** and reused.

```
falcon.init(**ray_init_kwargs)   # optional, once, idempotent; thin wrapper around ray.init()
falcon.launch(...)               # uses existing Ray; lazily calls init() if none; never shuts down
falcon.shutdown()                # explicit teardown
```

- `falcon.init()` connects to an existing cluster (`address=...` passed as a
  kwarg) or starts a local one. Idempotent: a second call is a no-op.
  **Cluster-level Ray resources live here**, not on `launch()`.
- `launch()` reuses an existing Ray, lazily calls `init()` with defaults if none
  exists (so a beginner does nothing), and **never shuts Ray down on return**, so
  state and actors survive across cells.
- The CLI keeps init-and-shutdown inside its one-shot process. This is a
  deliberate, documented divergence from the API path.
- `falcon.session()` context manager is deferred; use `falcon.init()` +
  `falcon.shutdown()` explicitly for now.

Beginner: do nothing, the first `launch()` brings up local Ray. Expert on a
cluster: `falcon.init(address=...)` once at the top, then many `launch()` calls
reuse it.

Note: per-node `ray_num_gpus` on `add_node` is node *placement*, a node property,
unrelated to cluster setup.

## Notebook-defined models: cloudpickle and the escape hatch

A class defined in a notebook lives in `__main__` and has no importable path, so
the `_target_` string mechanism cannot find it. The plan: notebook users pass the
class or instance object itself into `add_node`, and Ray ships it to actors via
cloudpickle.

### What cloudpickle does, and what it does not serialize

- A `__main__` class is serialized **by value** (its code and method bytecode).
- Modules that its methods *reference* are serialized **by reference**: if a
  method calls `torch.randn(...)` and `torch` was imported at the top of the
  notebook, cloudpickle records "the module named `torch`" and the Ray worker
  re-runs `import torch` on unpickle. The import statement does not travel; a
  re-import on the worker does.
- Therefore top-of-notebook imports work as-is. Moving `import torch` into
  `__init__` changes nothing for correctness.
- The real requirement is **environment parity**: the Ray worker must be able to
  import the referenced packages. For a local Ray started by the notebook the
  worker is the same environment, so this is automatic. For a remote cluster the
  packages must exist there (or be supplied via `runtime_env`).
- Cloudpickle captures, transitively, the global names the methods reference. A
  notebook class referencing another notebook-defined helper class drags the
  helper in by value too.

### Spike, gating the rest of the plan

This is the load-bearing assumption of the pedagogical story and **must be
validated in a spike before the API is committed**. Risks to test:

- Closures over large notebook globals bloating the pickle.
- Transitive capture of other notebook-defined classes.
- Torch modules / CUDA tensors as constructor args.
- Re-running the defining cell mid-session (class identity changes; a new
  `launch()` picks up the new class, which is the desired edit-rerun behavior, but
  an in-flight run keeps the old one).

If cloudpickle proves unreliable, the fallback is an explicit
`falcon.register(MyClass)` that snapshots source into a synthetic importable
module. We do not silently write temp modules.

### The escape hatch: serialization round-trip

Every standard config system (Hydra, spaCy, AllenNLP) assumes components are
importable. Notebook `__main__` classes break that: they have no import path, so
they cannot serialize back to a `_target_` string. This is handled with a
deliberate, narrow exception:

- When `launch()` saves the resolved `config.yml`, a live notebook-defined object
  is written as a placeholder, `"<live object: ClassName>"`, not a real `_target_`.
- The saved `config.yml` stays valid and readable but the run is flagged **not
  reproducible from YAML alone**.
- The object still ships to Ray fine, because that path is cloudpickle, not the
  import path.

So a run using only library components produces a fully runnable `config.yml`; a
run using notebook-defined classes produces a `config.yml` that is viewable and
instructive but not replayable without the notebook. The example notebooks must
state this honestly. Source extraction to a `_live_objects.py` artefact is not
implemented for v1 — the notebook itself is already the natural reproducibility
artefact.

## JAX and process-global state

JAX simulators and embeddings need extra care because JAX has process-global
state that does **not** serialize. Ray actors are separate processes, so:

- **`jax_enable_x64` does not travel.** Setting it in a notebook cell affects
  only the driver. Each actor starts in 32-bit and must re-establish x64 itself,
  before its first JAX array is created, either in the simulator's `__init__` or
  via a per-actor environment variable (`JAX_ENABLE_X64=1`). This is the one
  legitimate case where `__init__`-time setup is required (it is not about
  imports).
- **GPU memory preallocation.** JAX preallocates a large GPU fraction on first
  use, per process. Several actors on one GPU collide. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  (or a memory fraction) per actor.
- **Do not ship `jit`-compiled artifacts.** Store the plain function and `jit` it
  inside `__init__`; each actor compiles its own. Compilation is per-process
  anyway.
- **PRNG keys.** A key pickles fine, but a single key copied to every actor makes
  every actor produce identical "random" simulations. Each actor must fold in
  something unique (actor index, node name).
- **JAX arrays as constructor args** are device-committed and risky across the
  serialization boundary; pass plain numpy and convert inside `__init__`.

Design consequence: the node's Ray actor config should expose a first-class
`env` / `runtime_env` passthrough so users can set `JAX_ENABLE_X64`,
`XLA_PYTHON_CLIENT_PREALLOCATE`, etc. per actor without hand-rolling it. The
`add_node` docs should show the JAX pattern explicitly (x64 and `jit` in
`__init__`, per-actor key splitting).

## Live monitoring and dynamic output

### Do not port the blessed TUI

The CLI TUI is terminal mechanism: alternate-screen mode, escape codes, a fixed
footer carved from a scrolling region, `cbreak` keyboard capture. None of it
exists in a notebook cell. Mirror the *information*, not the mechanism.

### One data source, three frontends

The blessed TUI, `falcon monitor`, and the notebook display are all frontends
over one source, `MonitorBridge.get_status()` (per-node epoch/loss/sims, buffer
stats). The notebook display is a third renderer of the same status dict, not a
reimplementation.

```
MonitorBridge.get_status()
  +-- blessed TUI         (falcon launch, terminal)
  +-- falcon monitor      (separate TUI process)
  +-- notebook display    (new)
```

### v1: one interleaved, color-tagged log stream

The v1 notebook display is deliberately simple: dump **every** log source into
the single cell stream, interleaved. That is the driver's `output.log` plus every
node's `output.log`. No widgets, no status panel, no polling of `MonitorBridge`.

- The mechanism mostly exists: Ray already forwards actor stdout to the driver
  (`log_to_driver`, which the CLI sets from `console.level`), and the driver's own
  log is already on stdout. The notebook log sink just prints what arrives.
- Each line is tagged by its source for scanability: a stable per-node color (hash
  the node name into a small palette) and/or emoji, with the driver getting its
  own. ANSI color codes and emoji both render in Jupyter/Colab cell output. Line
  shape: `{tag} {node-name}  HH:MM:SS  message`.
- Interleaving across nodes is accepted, by design. The point is liveness: the
  cell visibly does something. A scannable color tag makes the mixed stream
  readable enough.

Honest caveat: Ray batches the forwarded actor output, so the interleaving is
approximately time-ordered, not exact. Fine for a liveness signal; the docs
should not promise precise ordering.

This is the v1 display. It always works, needs no extra dependency, and is enough
for the blocking `launch()` path.

### Phase 2: structured displays (optional, later)

Richer renderings are a later, optional add-on, not v1:

- **ipywidgets dashboard**: a fixed `VBox` of one compact status row per node
  (status, epoch progress bar, loss) plus an `Output` widget for the log stream.
  Polls `MonitorBridge.get_status()`. Reproduces the TUI's panel-plus-scroll
  split. Needs the `notebooks` extra. Per-node *status rows* shown all at once
  (scales to dozens of nodes); per-node *log tails* are not, since the v1 stream
  already carries them interleaved. Full per-node log inspection stays the job of
  `falcon monitor`.
- **`display(..., display_id=True)` + `.update()`**: a lighter middle option, one
  updating HTML status table, IPython-only.

These are sugar over the same information. v1 (the interleaved stream) ships
first; phase 2 is built only if the plain stream proves insufficient in practice.

### The blocking constraint, and the routes around it

While `launch()` blocks the kernel, the display is **push-only**: the launch loop
pushes updates out, but interactive widget callbacks (a dropdown to select
metrics or a node) cannot fire, because the kernel is busy. This is the default
outcome, not a hard law. Three routes give genuine live interactivity:

1. **Non-blocking mode (`launch(wait=False)`).** Training runs in a background
   thread, the kernel is free, ipywidgets callbacks fire normally; the user can
   select metrics and filter nodes live. The half-trained-`Run` concurrency worry
   does not apply, because monitoring is read-only.
2. **A separate monitor client (lowest risk).** Training runs on Ray actors,
   decoupled from the driver, and `MonitorBridge` persists in the cluster. A
   second kernel or notebook (or `falcon monitor`) connects to the same bridge
   and is fully interactive because it is a different, non-blocked process. The
   blocking `launch()` cell stays simple; interactivity lives in the companion.
3. **A cooperative-async launch loop.** If the launch loop is `async` and yields
   to the kernel's asyncio event loop periodically, widget callbacks can fire
   during the "blocking" cell. Legitimate but more implementation work; a
   fallback, not the primary path.

Interactive metric *exploration after* a run needs none of this: the run is done,
the kernel is free, so `run.plot_metrics(...)` with metric-picker widgets is
interactive out of the box. Only live-during-training selection needs routes 1-3.

Recommended: keep block-by-default with the push-only display for the simple
path; offer `wait=False` for users who want a live interactive dashboard; point
at the separate monitor client (route 2) for the richest experience at the
lowest risk.

## Rich display

- `Config._repr_markdown_`: pretty, foldable YAML.
- `Graph._repr_html_` / `_repr_mimebundle_`: render the DAG as **Mermaid**
  (Jupyter and GitHub render it natively); reuse the topology logic behind
  `render_git_graph_simple`. `falcon graph` keeps ASCII for the terminal.
- `Run._repr_html_`: status, runtime, Ray size, per-node final loss, log-file
  links; a compact sibling of the PR #56 end-of-run summary.
- `run.plot_metrics()`: matplotlib loss curves from `read_run`.
- `falcon.corner(samples)`: convenience posterior corner plot (optional
  dependency, graceful fallback like `wandb`).

## Standard precedents

This is a solved problem; borrow rather than invent.

- **Hydra / OmegaConf** (already a Falcon dependency): `instantiate()` with
  `_target_` plus kwargs is exactly the Step 0 convention. Target Hydra's
  `instantiate` semantics for the resolver.
- **Pydantic**: typed sectioned configs, discriminated unions, custom types that
  own their own encodings. The two-layer `_target_` rule is the Pydantic
  discriminated-union pattern.
- **spaCy / Thinc config + `catalogue`**: the most polished example of a
  registry-based, typed, nested config designed for hand-editing. Study its
  design; do not add it as a dependency (it would compete with OmegaConf/Hydra).
- **Keras** (`class_name` + `config`), **AllenNLP** (`Registrable` / `from_params`),
  **Kubernetes** (`kind:` discriminator): the same pattern in other domains.
- **PyTorch** deliberately does the opposite (architecture lives in code, only
  weights persist). That is the philosophy Falcon is choosing against; worth
  knowing as the alternative.

The only genuinely new piece is the notebook-class escape hatch; everything else
is on well-paved road.

## Example notebooks (deliverables)

Each `examples/0X_*` gets a companion `notebook.py` (jupytext percent format),
kept as source of truth alongside the existing CLI scripts. The `.ipynb` files
are generated build artefacts, not checked in. Existing `run.py` / `run_example.py`
files are left untouched; the notebook scripts are new, separate files.

- `01_minimal`: cell story A: load config, override, launch, inspect, sample.
- `02_bimodal`: config register: compare training strategies by editing config.
- `03_composite`: programmatic register: multi-node graph, composite embedding.
- `04_gaussian`: define-your-own: write a plain callable simulator in a cell.
- `05_linear_regression`: the full define-your-own story with an explicit FFT
  embedding network defined in a cell, the cloudpickle caveats surfaced, and a
  check against the analytic Gaussian posterior.
- A new `examples/00_tour.ipynb`: narrative tour of the whole framework.

These are the acceptance tests for the API. Onboarding flows through them: a
beginner starts by mutating a working notebook, and the per-run auto-saved
`config.yml` (the fully resolved config, which doubles as an all-defaults
overview) is how they later discover the file the CLI consumes. One explicit
lesson connects the saved file to `falcon launch`.

## CI

Execute every example notebook in CI with `pytest --nbmake` (or
`jupyter nbconvert --execute`) on a tiny config (few epochs, small buffer) so
notebook rot is caught. Add a `notebooks` extra to `pyproject.toml`
(`ipywidgets`, `matplotlib`, optionally `corner`).

## Phasing / sequencing

0. **Step 0: deferred.** See above.
1. **`_run_pipeline` extraction.** CLI behavior byte-for-byte unchanged; unit
   tests exercise `_run_pipeline` directly. No new public API.
2. **Cloudpickle spike.** Prove or disprove notebook-`__main__` classes surviving
   to Ray actors. Gate the rest of the plan on this.
3. **Flat config surface.** Synthesized signatures from the nested dataclasses,
   the prefix-transform bridge, the `Config` object. Typed configs, no `**kwargs`.
4. **`falcon.init` / `launch` / `shutdown` + the v1 interleaved color-tagged log
   stream.** `launch()` returns a `Run`, blocks. The CLI is refactored to conform
   to the API. The config register, end to end. `01_minimal` notebook.
5. **`Graph.add_node` builder + live-object support + the escape-hatch
   serialization + the JAX actor-env passthrough.** The programmatic register.
   `03` / `04` / `05` notebooks. (`falcon.Simulator` base class deferred.)
6. **Rich reprs + Mermaid graph + `plot_metrics` / `corner`.**
7. **Phase-2 monitoring (optional): the ipywidgets dashboard over `MonitorBridge`,
   `launch(wait=False)` + `LaunchHandle`, the separate monitor client.** Built
   only if the v1 stream proves insufficient.
8. **Notebook CI + `00_tour.ipynb`.**

## Explicitly out of scope (v1)

- Environment auto-detection (notebook vs terminal). The CLI is for terminals,
  the API for everything else; no `isatty` heuristics. This also dissolves the
  original "Colab garbage output" bug: notebook users call `falcon.launch()`,
  never `!falcon launch`.
- Silently writing user cell code to temp modules.
- A notebook-native config *editor* widget. Editing is done in code cells.
- A second, beginner-only API. There is one surface, reached by experts directly
  and by students through worked examples.

## Open questions

- **Cloudpickle spike outcome.** Gates everything notebook-class-related. If it
  fails, the `falcon.register` fallback applies.
- **Flattened-signature parameter counts.** The flat surface is comfortable at
  sklearn scale (~20-30 params per estimator) and degrades past that. Count the
  real totals during Step 3 (Flow's `InferenceConfig` alone is ~12 fields). If an
  estimator's flattened signature is much larger than ~30, revisit before
  implementing.
- **Where notebook runs write.** Default `output/<adj-noun-date>` as today; the
  rich `Run` repr surfaces the path so it is never lost.
