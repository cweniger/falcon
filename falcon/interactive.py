"""Interactive TUI for falcon launch.

Provides a scrolling log view with a fixed status footer showing node log tails.
Uses Blessed for terminal control and Rich for formatting.
"""

import atexit
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class NodeStatus:
    """Status information for a single node."""

    name: str
    status: str = "idle"
    current_epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    samples: int = 0


@dataclass
class InteractiveState:
    """Shared state for the interactive display."""

    nodes: dict[str, NodeStatus] = field(default_factory=dict)
    selected_node_idx: int = 0
    running: bool = True
    buffer_stats: dict = field(default_factory=lambda: {"training": 0, "validation": 0})
    log_dir: Optional[Path] = None


class InteractiveDisplay:
    """Interactive terminal display with scrolling driver logs and node log tail footer.

    Layout:
        ┌─────────────────────────────────────────────────────┐
        │ [Driver log - scrolling]                            │
        │ 08:01:46 [INFO] falcon v0.1.0                       │
        │ 08:01:46 [INFO] Output: outputs/test-run            │
        ├─────────────────────────────────────────────────────┤
        │ ▸ theta | training | 5/300 | loss: -2.91            │
        │ ─────────────────────────────────────────────────── │
        │ 08:02:15 [INFO] Epoch 1/300 | loss=2.048            │
        │ 08:02:18 [INFO] Epoch 2/300 | loss=-1.128           │
        │ 08:02:22 [INFO] Epoch 3/300 | loss=-2.911           │
        │ [j/k] node  [q] quit                                │
        └─────────────────────────────────────────────────────┘
    """

    def __init__(self, footer_height: int = 20):
        from blessed import Terminal

        self.term = Terminal()
        self.footer_height = footer_height
        self.state = InteractiveState()
        self._lock = threading.Lock()
        self._key_thread: Optional[threading.Thread] = None
        self._log_lines: list[str] = []  # Buffer for driver log lines
        self._original_sigint = None
        self._original_sigterm = None
        self._stopped = False
        self._interrupt_count = 0
        self._stopping = False  # Graceful stop requested

    def set_log_dir(self, log_dir: str) -> None:
        """Set the directory where node log files are stored."""
        self.state.log_dir = Path(log_dir)

    def start(self) -> None:
        """Start the interactive display."""
        # Install signal handlers for clean shutdown on Ctrl+C
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)

        # Register atexit handler as fallback
        atexit.register(self._atexit_handler)

        # Enter fullscreen-like mode with scroll region
        print(self.term.enter_fullscreen, end="", flush=True)
        print(self.term.hide_cursor, end="", flush=True)

        # Set scroll region to leave space for footer
        self._set_scroll_region()

        # Start keyboard listener thread
        self._key_thread = threading.Thread(target=self._key_listener, daemon=True)
        self._key_thread.start()

        # Draw initial footer
        self._draw_footer()

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM with double Ctrl+C pattern."""
        self._interrupt_count += 1

        if self._interrupt_count == 1:
            # First interrupt: request graceful stop
            self._stopping = True
            self.log("\x1b[33m⚠ Stopping gracefully... (Ctrl+C again to force quit)\x1b[0m")
            return  # Don't exit yet, let training finish

        # Second interrupt: force exit
        self.stop()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        else:
            sys.exit(128 + signum)

    def _atexit_handler(self):
        """Fallback cleanup on exit."""
        if not self._stopped:
            self.stop()

    def stop(self) -> None:
        """Stop the interactive display and restore terminal."""
        if self._stopped:
            return
        self._stopped = True
        self.state.running = False

        # Restore signal handlers
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

        # Unregister atexit handler
        try:
            atexit.unregister(self._atexit_handler)
        except Exception:
            pass

        # Reset scroll region to full terminal
        print(f"\x1b[1;{self.term.height}r", end="", flush=True)

        # Clear the footer area
        footer_top = self.term.height - self.footer_height
        for i in range(self.footer_height):
            row = footer_top + i
            print(f"\x1b[{row + 1};1H\x1b[K", end="", flush=True)

        # Move cursor to end of scroll region (where logs ended)
        print(f"\x1b[{footer_top};1H", end="", flush=True)

        # Restore terminal
        print(self.term.normal_cursor, end="", flush=True)
        print(self.term.exit_fullscreen, end="", flush=True)

    def _set_scroll_region(self) -> None:
        """Set terminal scroll region to exclude footer."""
        scroll_bottom = self.term.height - self.footer_height
        print(f"\x1b[1;{scroll_bottom}r", end="", flush=True)
        print(self.term.move(scroll_bottom - 1, 0), end="", flush=True)

    def log(self, message: str) -> None:
        """Print a log message in the scrolling region (driver logs)."""
        with self._lock:
            scroll_bottom = self.term.height - self.footer_height
            print(self.term.move(scroll_bottom - 1, 0), end="", flush=True)

            max_width = self.term.width - 1
            if len(message) > max_width:
                message = message[: max_width - 3] + "..."

            print(message, flush=True)

            self._log_lines.append(message)
            if len(self._log_lines) > 1000:
                self._log_lines = self._log_lines[-500:]

            self._draw_footer()

    def update_node(
        self,
        name: str,
        status: Optional[str] = None,
        current_epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        loss: Optional[float] = None,
        samples: Optional[int] = None,
    ) -> None:
        """Update status for a node."""
        with self._lock:
            if name not in self.state.nodes:
                self.state.nodes[name] = NodeStatus(name=name)

            node = self.state.nodes[name]
            if status is not None:
                node.status = status
            if current_epoch is not None:
                node.current_epoch = current_epoch
            if total_epochs is not None:
                node.total_epochs = total_epochs
            if loss is not None:
                node.loss = loss
            if samples is not None:
                node.samples = samples

            self._draw_footer()

    def update_buffer(self, training: int, validation: int) -> None:
        """Update buffer statistics."""
        with self._lock:
            self.state.buffer_stats["training"] = training
            self.state.buffer_stats["validation"] = validation
            self._draw_footer()

    def _get_node_log_tail(self, node_name: str, num_lines: int = None) -> list[str]:
        """Read the last N lines from a node's output.log file."""
        if num_lines is None:
            # Footer layout: separator(1) + status bar(1) + sub-separator(1) + log lines + help(1)
            num_lines = max(1, self.footer_height - 4)
        if self.state.log_dir is None:
            return ["[Log directory not set]"]

        log_file = self.state.log_dir / node_name / "output.log"
        if not log_file.exists():
            return [f"[Waiting for {node_name}/output.log...]"]

        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
            return [line.rstrip() for line in lines[-num_lines:]]
        except Exception as e:
            return [f"[Error reading log: {e}]"]

    def _draw_footer(self) -> None:
        """Draw the fixed status footer with node log tail."""
        save_pos = "\x1b[s"
        restore_pos = "\x1b[u"

        footer_top = self.term.height - self.footer_height
        lines = []

        # Separator
        lines.append("─" * self.term.width)

        # Node selector bar
        node_names = list(self.state.nodes.keys())
        if node_names:
            # Build node tabs
            tabs = []
            for i, name in enumerate(node_names):
                node = self.state.nodes[name]
                is_selected = i == self.state.selected_node_idx

                # Status indicator
                status_colors = {
                    "training": "\x1b[33m",  # Yellow
                    "idle": "\x1b[90m",      # Gray
                    "done": "\x1b[32m",      # Green
                    "error": "\x1b[31m",     # Red
                    "active": "\x1b[36m",    # Cyan
                }
                color = status_colors.get(node.status, "")
                reset = "\x1b[0m"

                if is_selected:
                    tab = f"\x1b[1m▸ {name}\x1b[0m"  # Bold + arrow for selected
                else:
                    tab = f"  {name}"

                tabs.append(f"{color}{tab}{reset}")

            # Current node details
            selected_name = node_names[self.state.selected_node_idx] if node_names else None
            if selected_name:
                node = self.state.nodes[selected_name]
                details = []
                details.append(f"\x1b[1m{selected_name}\x1b[0m")

                status_colors = {
                    "training": "\x1b[33m",
                    "idle": "\x1b[90m",
                    "done": "\x1b[32m",
                    "error": "\x1b[31m",
                }
                color = status_colors.get(node.status, "")
                details.append(f"{color}{node.status}\x1b[0m")

                if node.total_epochs > 0:
                    details.append(f"{node.current_epoch}/{node.total_epochs}")
                if node.loss is not None:
                    details.append(f"loss: {node.loss:.2e}")
                if node.samples > 0:
                    details.append(f"{node.samples} sims")

                # Buffer stats
                buf = self.state.buffer_stats
                if buf["training"] > 0:
                    details.append(f"buf: {buf['training']}")

                header_line = " | ".join(tabs) + "  │  " + " | ".join(details)
            else:
                header_line = " | ".join(tabs)

            lines.append(header_line[:self.term.width])
        else:
            lines.append("[No nodes registered yet]")

        # Sub-separator for log section
        lines.append("\x1b[90m" + "─" * self.term.width + "\x1b[0m")

        # Node log tail
        # Footer layout: separator(1) + status bar(1) + sub-separator(1) + log lines + help(1)
        num_log_lines = max(1, self.footer_height - 4)
        selected_name = self.get_selected_node()
        if selected_name:
            log_lines = self._get_node_log_tail(selected_name, num_log_lines)
            for log_line in log_lines:
                # Truncate long lines
                if len(log_line) > self.term.width - 1:
                    log_line = log_line[:self.term.width - 4] + "..."
                lines.append(log_line)

            # Pad with empty lines if needed
            while len(log_lines) < num_log_lines:
                lines.append("")
                log_lines.append("")  # For counting
        else:
            for _ in range(num_log_lines):
                lines.append("")

        # Help line
        help_line = "[j/k] switch node"
        lines.append(f"\x1b[90m{help_line}\x1b[0m")

        # Draw footer
        output = save_pos
        for i, line in enumerate(lines):
            row = footer_top + i
            if row < self.term.height:
                output += f"\x1b[{row + 1};1H\x1b[K{line}"
        output += restore_pos

        print(output, end="", flush=True)

    def _key_listener(self) -> None:
        """Listen for keyboard input in a separate thread."""
        with self.term.cbreak():
            while self.state.running:
                key = self.term.inkey(timeout=0.1)
                if not key:
                    continue

                with self._lock:
                    if key.lower() == "j" or key.name == "KEY_DOWN":
                        node_count = len(self.state.nodes)
                        if node_count > 0:
                            self.state.selected_node_idx = (
                                self.state.selected_node_idx + 1
                            ) % node_count
                            self._draw_footer()
                    elif key.lower() == "k" or key.name == "KEY_UP":
                        node_count = len(self.state.nodes)
                        if node_count > 0:
                            self.state.selected_node_idx = (
                                self.state.selected_node_idx - 1
                            ) % node_count
                            self._draw_footer()

    @property
    def is_running(self) -> bool:
        """Check if the display is still running."""
        return self.state.running

    @property
    def stop_requested(self) -> bool:
        """Check if graceful stop was requested (first Ctrl+C)."""
        return self._stopping

    def get_selected_node(self) -> Optional[str]:
        """Get the name of the currently selected node."""
        node_names = list(self.state.nodes.keys())
        if node_names and 0 <= self.state.selected_node_idx < len(node_names):
            return node_names[self.state.selected_node_idx]
        return None


class InteractiveLogHandler:
    """Adapter to route log messages to InteractiveDisplay."""

    def __init__(self, display: InteractiveDisplay):
        self.display = display

    def write(self, message: str) -> None:
        """Handle a log message."""
        message = message.rstrip()
        if message:
            self.display.log(message)

    def flush(self) -> None:
        """Flush output (no-op for interactive display)."""
        pass
