# Falcon TUI Monitoring Plan

## Overview

A terminal-based monitoring system for Falcon:

**`falcon monitor`** - Separate htop-style TUI that connects to Ray actors

## Design Principles

- **Separation of concerns**: Execution and monitoring are decoupled
- **Actor-based**: Monitor queries Ray actors directly for real-time data
- **Single discovery point**: One named coordinator actor provides access to everything
- **Robustness**: Training continues regardless of monitor state
- **HPC-friendly**: Works in SSH sessions, no port forwarding needed

---

## Architecture

### Actor Discovery Model

```
┌─────────────────────────────────────────────────────────────────┐
│                         Ray Cluster                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  FalconCoordinator (named: "falcon:coordinator")        │   │
│  │                                                         │   │
│  │  - Holds references to all NodeWrapper actors           │   │
│  │  - Holds reference to DatasetManagerActor               │   │
│  │  - Provides get_status() → aggregated status dict       │   │
│  │  - Provides get_node_log(name) → recent log lines       │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           │ holds references to                                 │
│           ▼                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ NodeWrapper │  │ NodeWrapper │  │ NodeWrapper │   ...       │
│  │   (theta)   │  │    (phi)    │  │  (x_embed)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  DatasetManagerActor (named: "DatasetManager")          │   │
│  │  - get_store_stats() already exists                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LoggerManager (named: "falcon:global_logger")          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
           │
           │ ray.get_actor("falcon:coordinator")
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      falcon monitor (TUI)                       │
│  - Connects to Ray cluster                                      │
│  - Discovers coordinator actor                                  │
│  - Polls get_status() every ~1 second                          │
│  - Displays node table, logs, progress                          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight

The monitor only needs to discover **one** named actor (`falcon:coordinator`). That actor holds references to all NodeWrapper actors and can query them on behalf of the monitor.

---

## Component 1: FalconCoordinator Actor

### Definition

```python
# falcon/core/coordinator.py

import ray
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NodeStatus:
    name: str
    status: str  # "idle", "simulating", "training", "paused", "done", "error"
    samples: int
    current_epoch: int
    total_epochs: int
    loss: Optional[float]
    loss_history: List[float]
    log_tail: List[str]  # Last N log lines

@dataclass
class ClusterStatus:
    run_dir: str
    started_at: datetime
    nodes: Dict[str, NodeStatus]
    buffer_stats: Dict[str, int]  # From DatasetManagerActor

@ray.remote(name="falcon:coordinator")
class FalconCoordinator:
    """
    Central coordination actor for monitoring.
    Holds references to all node actors and provides aggregated status.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.started_at = datetime.now()
        self.node_actors: Dict[str, ray.actor.ActorHandle] = {}
        self.dataset_manager = None

    def register_node(self, name: str, actor_handle: ray.actor.ActorHandle):
        """Called by DeployedGraph after creating each NodeWrapper."""
        self.node_actors[name] = actor_handle

    def register_dataset_manager(self, actor_handle: ray.actor.ActorHandle):
        """Called by DeployedGraph after creating DatasetManagerActor."""
        self.dataset_manager = actor_handle

    def get_status(self) -> dict:
        """
        Main method called by monitor.
        Queries all nodes and returns aggregated status.
        """
        # Query all nodes in parallel
        node_status_futures = {
            name: actor.get_status.remote()
            for name, actor in self.node_actors.items()
        }

        # Get buffer stats
        buffer_stats = {}
        if self.dataset_manager:
            buffer_stats = ray.get(self.dataset_manager.get_store_stats.remote())

        # Collect node statuses
        nodes = {}
        for name, future in node_status_futures.items():
            try:
                nodes[name] = ray.get(future, timeout=2.0)
            except Exception as e:
                nodes[name] = {"status": "error", "error": str(e)}

        return {
            "run_dir": self.run_dir,
            "started_at": self.started_at.isoformat(),
            "nodes": nodes,
            "buffer": buffer_stats,
        }

    def get_node_log(self, node_name: str, num_lines: int = 50) -> List[str]:
        """Get recent log lines for a specific node."""
        if node_name not in self.node_actors:
            return []
        return ray.get(self.node_actors[node_name].get_log_tail.remote(num_lines))

    def get_node_names(self) -> List[str]:
        """Return list of registered node names."""
        return list(self.node_actors.keys())
```

### NodeWrapper Status Method

Add to `NodeWrapper` in `deployed_graph.py`:

```python
class NodeWrapper:
    # ... existing code ...

    def __init__(self, node, graph, model_path):
        # ... existing code ...
        self._status = "idle"
        self._log_buffer = []  # Ring buffer for recent log lines
        self._max_log_lines = 100

    def get_status(self) -> dict:
        """Return current status for monitoring."""
        status = {
            "name": self.node.name,
            "status": self._status,
            "samples": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "loss": None,
            "loss_history": [],
        }

        # Get estimator state if available
        if hasattr(self, 'estimator_instance') and self.estimator_instance:
            est = self.estimator_instance
            if hasattr(est, 'history'):
                status["loss_history"] = est.history.get("val_loss", [])[-20:]
                if status["loss_history"]:
                    status["loss"] = status["loss_history"][-1]
            if hasattr(est, 'current_epoch'):
                status["current_epoch"] = est.current_epoch
            if hasattr(est, 'loop_config'):
                status["total_epochs"] = est.loop_config.num_epochs

        return status

    def get_log_tail(self, num_lines: int = 50) -> List[str]:
        """Return recent log lines."""
        return self._log_buffer[-num_lines:]

    def _log(self, message: str):
        """Internal logging that also stores to buffer."""
        self._log_buffer.append(f"[{datetime.now():%H:%M:%S}] {message}")
        if len(self._log_buffer) > self._max_log_lines:
            self._log_buffer.pop(0)
```

---

## Component 2: `falcon monitor` TUI

### Technology

- **Textual** library (`pip install textual`)
- Connects to Ray cluster, discovers coordinator actor
- Polls status every ~1 second

### Dashboard Layout

The node table is **scrollable** to handle many nodes:
- Shows ~5 visible rows by default with vertical scrollbar
- Supports sorting by name (alphabetical), status, or loss
- Press `S` to cycle through sort modes

```
┌─ Falcon Monitor ──────────────────────────────────────────────────┐
│ Run: outputs/exp01    Elapsed: 00:14:32                           │
│ Buffer: 1,204 training | 512 validation | 89 disfavoured          │
├───────────────────────────────────────────────────────────────────┤
│ Node        Status      Samples   Loss     Trend        Progress ▲│
│ ─────────────────────────────────────────────────────────────────│
│ ▶ theta     training      1,204   0.342    ▂▃▅▄▃▂▂▁     ████░░ 67%│
│   phi       training        512   1.203    ▇▆▅▅▄▄▃▃     ██░░░░ 34%│
│   x_embed   done            850   0.127    ▅▅▄▃▂▂▁▁     ██████ 100%│
│   y_embed   training        320   0.891    ▇▆▅▄▄▃▃▂     █░░░░░ 20%│
│   z_model   idle              0   -        -            ░░░░░░  0%░│
├───────────────────────────────────────────────────────────────────┤
│ Log (theta):                                                      │
│ [14:32:01] Epoch 145/200, loss=0.342, lr=0.0001                   │
│ [14:32:03] Epoch 146/200, loss=0.338, lr=0.0001                   │
├───────────────────────────────────────────────────────────────────┤
│ ^N Next  ^P Prev  S Sort  L Log  D Dashboard  Q Quit              │
└───────────────────────────────────────────────────────────────────┘
```

### Key Bindings

| Key | Action |
|-----|--------|
| `Ctrl+N` / `j` | Select next node |
| `Ctrl+P` / `k` | Select previous node |
| `S` | Cycle sort mode (name → status → loss) |
| `D` | Dashboard view (default) |
| `L` | Full log view for selected node |
| `R` | Force refresh |
| `Q` | Quit monitor |

### CLI Interface

```bash
# Auto-connect to local Ray cluster
falcon monitor

# Explicit run directory (for finding Ray address from config)
falcon monitor outputs/exp01/

# Explicit Ray address
falcon monitor --address ray://192.168.1.10:10001

# Refresh rate
falcon monitor --refresh 0.5  # seconds, default 1.0
```

### Implementation

```python
# falcon/monitor.py
#
# Standalone TUI monitor for Falcon training runs.
# Run with: python -m falcon.monitor
# Or via: falcon monitor

import click
import ray
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, RichLog, Static
from textual.binding import Binding
from datetime import datetime

class FalconMonitor(App):
    CSS = """
    #status_bar { height: 3; background: $surface; padding: 0 1; }
    #buffer_bar { height: 1; color: $text-muted; padding: 0 1; }
    #nodes { height: 40%; }
    #log { height: 45%; border: solid $primary; }
    """

    BINDINGS = [
        Binding("ctrl+n", "next_node", "Next"),
        Binding("ctrl+p", "prev_node", "Prev"),
        Binding("j", "next_node", "Next", show=False),
        Binding("k", "prev_node", "Prev", show=False),
        Binding("s", "cycle_sort", "Sort"),
        Binding("d", "dashboard", "Dashboard"),
        Binding("l", "log_view", "Log View"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
    ]

    SORT_MODES = ["name", "status", "loss"]

    def __init__(self, ray_address: str = None, refresh_interval: float = 1.0):
        super().__init__()
        self.ray_address = ray_address
        self.refresh_interval = refresh_interval
        self.coordinator = None
        self.nodes = []
        self.current_node_idx = 0
        self.started_at = None
        self.sort_mode = "name"  # Current sort mode

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="status_bar")
        yield Static(id="buffer_bar")
        yield DataTable(id="nodes")
        yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        # Connect to Ray
        if not ray.is_initialized():
            ray.init(address=self.ray_address or "auto")

        # Discover coordinator
        try:
            self.coordinator = ray.get_actor("falcon:coordinator")
        except ValueError:
            self.query_one("#status_bar", Static).update(
                "[red]No Falcon training running (coordinator not found)[/red]"
            )
            return

        # Setup table
        table = self.query_one("#nodes", DataTable)
        table.add_columns("Node", "Status", "Samples", "Loss", "Trend", "Progress")
        table.cursor_type = "row"

        # Start polling
        self.set_interval(1.0, self.refresh_data)
        self.refresh_data()

    def refresh_data(self) -> None:
        if not self.coordinator:
            return

        try:
            status = ray.get(self.coordinator.get_status.remote(), timeout=5.0)
        except Exception as e:
            self.query_one("#status_bar", Static).update(f"[red]Error: {e}[/red]")
            return

        self.update_status_bar(status)
        self.update_buffer_bar(status)
        self.update_node_table(status)
        self.update_log_view()

    def update_status_bar(self, status: dict) -> None:
        bar = self.query_one("#status_bar", Static)

        # Calculate elapsed time
        if self.started_at is None and status.get("started_at"):
            self.started_at = datetime.fromisoformat(status["started_at"])

        elapsed = "00:00:00"
        if self.started_at:
            delta = datetime.now() - self.started_at
            hours, rem = divmod(int(delta.total_seconds()), 3600)
            mins, secs = divmod(rem, 60)
            elapsed = f"{hours:02d}:{mins:02d}:{secs:02d}"

        bar.update(
            f"Run: [bold]{status.get('run_dir', 'unknown')}[/bold]    "
            f"Elapsed: {elapsed}    "
            f"Sort: {self.sort_mode}"
        )

    def update_buffer_bar(self, status: dict) -> None:
        bar = self.query_one("#buffer_bar", Static)
        buf = status.get("buffer", {})
        bar.update(
            f"Buffer: {buf.get('training', 0)} training | "
            f"{buf.get('validation', 0)} validation | "
            f"{buf.get('disfavoured', 0)} disfavoured"
        )

    def update_node_table(self, status: dict) -> None:
        table = self.query_one("#nodes", DataTable)
        table.clear()

        nodes_dict = status.get("nodes", {})

        # Sort nodes based on current sort mode
        if self.sort_mode == "name":
            sorted_items = sorted(nodes_dict.items(), key=lambda x: x[0])
        elif self.sort_mode == "status":
            status_order = {"training": 0, "simulating": 1, "idle": 2, "done": 3, "error": 4}
            sorted_items = sorted(nodes_dict.items(),
                                  key=lambda x: status_order.get(x[1].get("status", ""), 5))
        elif self.sort_mode == "loss":
            sorted_items = sorted(nodes_dict.items(),
                                  key=lambda x: x[1].get("loss") or float("inf"))
        else:
            sorted_items = list(nodes_dict.items())

        self.nodes = [name for name, _ in sorted_items]

        for i, (name, node) in enumerate(sorted_items):
            prefix = "▶ " if i == self.current_node_idx else "  "

            # Status with color
            node_status = node.get("status", "unknown")
            status_display = {
                "training": "[green]training[/green]",
                "simulating": "[blue]simulating[/blue]",
                "idle": "[dim]idle[/dim]",
                "done": "[green bold]done[/green bold]",
                "error": "[red]error[/red]",
                "paused": "[yellow]paused[/yellow]",
            }.get(node_status, node_status)

            # Loss and trend
            loss = node.get("loss")
            loss_str = f"{loss:.4f}" if loss is not None else "-"
            trend = self._make_sparkline(node.get("loss_history", []))

            # Progress
            progress = self._make_progress(node)

            table.add_row(
                f"{prefix}{name}",
                status_display,
                str(node.get("samples", 0)),
                loss_str,
                trend,
                progress,
            )

    def update_log_view(self) -> None:
        if not self.nodes or not self.coordinator:
            return

        node_name = self.nodes[self.current_node_idx]
        log_view = self.query_one("#log", RichLog)
        log_view.clear()

        try:
            lines = ray.get(
                self.coordinator.get_node_log.remote(node_name, 30),
                timeout=2.0
            )
            log_view.write(f"[bold]Log ({node_name}):[/bold]")
            for line in lines:
                log_view.write(line)
        except Exception as e:
            log_view.write(f"[red]Error fetching log: {e}[/red]")

    def _make_sparkline(self, values: list) -> str:
        if not values:
            return ""
        chars = "▁▂▃▄▅▆▇█"
        values = values[-8:]  # Last 8 values
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return chars[0] * len(values)
        return "".join(
            chars[min(7, int((v - min_v) / (max_v - min_v) * 7))]
            for v in values
        )

    def _make_progress(self, node: dict) -> str:
        if node.get("status") == "done":
            return "[green]██████[/green] done"
        current = node.get("current_epoch", 0)
        total = node.get("total_epochs", 1) or 1
        pct = min(1.0, current / total)
        filled = int(pct * 6)
        bar = "█" * filled + "░" * (6 - filled)
        return f"{bar} {int(pct*100)}%"

    def action_next_node(self) -> None:
        if self.nodes:
            self.current_node_idx = (self.current_node_idx + 1) % len(self.nodes)
            self.update_log_view()

    def action_prev_node(self) -> None:
        if self.nodes:
            self.current_node_idx = (self.current_node_idx - 1) % len(self.nodes)
            self.update_log_view()

    def action_refresh(self) -> None:
        self.refresh_data()

    def action_cycle_sort(self) -> None:
        """Cycle through sort modes: name → status → loss → name..."""
        idx = self.SORT_MODES.index(self.sort_mode)
        self.sort_mode = self.SORT_MODES[(idx + 1) % len(self.SORT_MODES)]
        self.refresh_data()


# --- CLI Entry Point ---

@click.command()
@click.option("--address", default="auto", help="Ray cluster address")
@click.option("--refresh", default=1.0, help="Refresh interval in seconds")
def main(address: str, refresh: float):
    """Falcon Monitor - TUI for monitoring training runs."""
    app = FalconMonitor(ray_address=address, refresh_interval=refresh)
    app.run()


if __name__ == "__main__":
    main()
```

---

## Integration with DeployedGraph

### Modifications to `deployed_graph.py`

```python
class DeployedGraph:
    def __init__(self, graph, model_path):
        # ... existing init ...

        # Create coordinator actor
        from falcon.core.coordinator import FalconCoordinator
        self.coordinator = FalconCoordinator.remote(
            run_dir=str(model_path)
        )

    def _create_node_actors(self):
        """Modified to register nodes with coordinator."""
        for node in self.graph.nodes:
            # ... existing actor creation ...
            actor = NodeWrapper.options(**node.actor_config).remote(
                node, self.graph, self.model_path
            )
            self.wrapped_nodes_dict[node.name] = actor

            # Register with coordinator
            ray.get(self.coordinator.register_node.remote(node.name, actor))

        # Register dataset manager
        dataset_manager = ray.get_actor("DatasetManager")
        ray.get(self.coordinator.register_dataset_manager.remote(dataset_manager))
```

---

## File Structure

```
falcon/
├── cli.py                 # Main CLI (falcon launch, falcon sample, etc.)
├── monitor.py             # Separate monitor module (falcon monitor)
└── core/
    ├── coordinator.py     # FalconCoordinator actor
    ├── deployed_graph.py  # Modified to create/register with coordinator
    └── ...
```

The monitor is a **completely separate module** from `cli.py`:
- `falcon launch` → `falcon/cli.py`
- `falcon monitor` → `falcon/monitor.py`

This keeps the TUI dependencies (Textual) isolated and allows the monitor to be developed/tested independently.

---

## Entry Points

In `pyproject.toml`:

```toml
[project.scripts]
falcon = "falcon.cli:main"

[project.gui-scripts]
falcon-monitor = "falcon.monitor:main"

# Or as a subcommand, add to cli.py:
# falcon monitor → subprocess call to falcon.monitor:main
```

**Option A: Separate binary**
```bash
falcon-monitor                    # Runs falcon/monitor.py:main()
falcon-monitor --address auto
```

**Option B: Subcommand that delegates** (preferred - single `falcon` command)
```bash
falcon monitor                    # cli.py spawns monitor.py as subprocess
falcon monitor --address auto
```

For Option B, add to `cli.py`:

```python
@cli.command()
@click.option("--address", default="auto", help="Ray cluster address")
@click.option("--refresh", default=1.0, help="Refresh interval in seconds")
def monitor(address: str, refresh: float):
    """Launch the Falcon monitor TUI."""
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "-m", "falcon.monitor",
        "--address", address,
        "--refresh", str(refresh),
    ])
```

---

## Implementation Steps

### Phase 1: Coordinator Infrastructure

1. **Create `falcon/core/coordinator.py`**
   - `FalconCoordinator` actor class (named `"falcon:coordinator"`)
   - `get_status()`, `get_node_log()`, `get_node_names()` methods

2. **Add status methods to NodeWrapper**
   - `get_status()` returning current state
   - `get_log_tail()` returning recent log lines
   - Log buffer for storing recent messages

3. **Integrate coordinator into DeployedGraph**
   - Create coordinator at init
   - Register each node after creation
   - Register dataset manager

### Phase 2: TUI Implementation

4. **Create `falcon/monitor.py`** (standalone module)
   - Textual app with scrollable node table
   - Sorting support (name, status, loss)
   - Connect to Ray, discover coordinator
   - Poll and display status
   - Has its own `main()` entry point
   - Can be run as `python -m falcon.monitor`

5. **Add `falcon monitor` CLI command**
   - Subcommand in `cli.py` that delegates to `falcon.monitor`
   - `--address` option for Ray cluster
   - `--refresh` option for poll interval

### Phase 3: Polish

6. **Enhanced features**
   - Full-screen log view (press `L`)
   - Color-coded status indicators
   - Error handling for disconnects

7. **Edge cases**
   - Monitor started before training
   - Training completed
   - Network interruptions
   - Graceful reconnection

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
monitor = ["textual>=0.40.0"]

# Or include in main dependencies if TUI is core
[project.dependencies]
textual = ">=0.40.0"
```

---

## Future Enhancements

- **Pause/resume from monitor**: Send commands to coordinator
- **Multiple runs**: List and switch between active training runs
- **Historical view**: Load completed run from saved state
- **Resource usage**: Show GPU/CPU utilization per node
- **Alerts**: Highlight nodes with increasing loss or errors
