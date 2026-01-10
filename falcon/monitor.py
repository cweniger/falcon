"""
Falcon Monitor - TUI for monitoring training runs.

Standalone module that connects to a running Ray cluster
and displays real-time status of Falcon training.

Run with:
    python -m falcon.monitor
    falcon monitor
"""

import click
import ray
import sys
from datetime import datetime
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, RichLog, Static


def init_ray_for_monitor(address: str = "auto") -> bool:
    """
    Initialize Ray connection before Textual takes over the terminal.

    Returns True if successful, False otherwise.
    """
    if ray.is_initialized():
        return True

    try:
        # Connect to existing cluster only - don't start a new one
        # Disable logging to avoid file descriptor issues with Textual
        # Use "falcon" namespace to find actors created by falcon launch
        ray.init(
            address=address,
            namespace="falcon",
            log_to_driver=False,
            configure_logging=False,
            logging_level="ERROR",
        )
        return True
    except Exception as e:
        print(f"Failed to connect to Ray cluster: {e}", file=sys.stderr)
        print("Make sure a Falcon training run is active.", file=sys.stderr)
        return False


class FalconMonitor(App):
    """Textual app for monitoring Falcon training runs."""

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

    def __init__(self, ray_address: Optional[str] = None, refresh_interval: float = 1.0):
        super().__init__()
        self.ray_address = ray_address
        self.refresh_interval = refresh_interval
        self.coordinator = None
        self.nodes = []
        self.current_node_idx = 0
        self.started_at = None
        self.sort_mode = "name"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="status_bar")
        yield Static(id="buffer_bar")
        yield DataTable(id="nodes")
        yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        # Ray should already be initialized before app starts
        if not ray.is_initialized():
            self.query_one("#status_bar", Static).update(
                "[red]Ray not initialized. Run 'falcon launch' first.[/red]"
            )
            return

        # Discover coordinator
        try:
            self.coordinator = ray.get_actor("falcon:coordinator")
        except ValueError:
            self.query_one("#status_bar", Static).update(
                "[yellow]No Falcon training running (coordinator not found)[/yellow]"
            )
            self.query_one("#buffer_bar", Static).update(
                "Start a training run with 'falcon launch' first"
            )
            return

        # Setup table
        table = self.query_one("#nodes", DataTable)
        table.add_columns("Node", "Status", "Samples", "Loss", "Trend", "Progress")
        table.cursor_type = "row"

        # Start polling
        self.set_interval(self.refresh_interval, self.refresh_data)
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
            try:
                self.started_at = datetime.fromisoformat(status["started_at"])
            except Exception:
                pass

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
        if "error" in buf:
            bar.update(f"Buffer: [red]{buf['error']}[/red]")
        else:
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
            sorted_items = sorted(
                nodes_dict.items(),
                key=lambda x: status_order.get(x[1].get("status", ""), 5)
            )
        elif self.sort_mode == "loss":
            sorted_items = sorted(
                nodes_dict.items(),
                key=lambda x: x[1].get("loss") or float("inf")
            )
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
                "initializing": "[cyan]init[/cyan]",
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
        return f"{bar} {int(pct * 100)}%"

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
    # Initialize Ray BEFORE Textual takes over the terminal
    # This avoids file descriptor conflicts
    if not init_ray_for_monitor(address):
        sys.exit(1)

    app = FalconMonitor(ray_address=address, refresh_interval=refresh)
    app.run()


if __name__ == "__main__":
    main()
