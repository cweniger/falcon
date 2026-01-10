"""Coordinator actor for falcon monitor."""

import ray
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@ray.remote
class FalconCoordinator:
    """
    Central coordination actor for monitoring.

    Holds references to all node actors and provides aggregated status.
    Discovered by falcon monitor via ray.get_actor("falcon:coordinator").
    """

    def __init__(self, run_dir: str):
        """
        Initialize the coordinator.

        Args:
            run_dir: Path to the run directory for display purposes
        """
        self.run_dir = run_dir
        self.log_dir = None  # Set via set_log_dir()
        self.started_at = datetime.now()
        self.node_actors: Dict[str, ray.actor.ActorHandle] = {}
        self.dataset_manager = None

    def set_log_dir(self, log_dir: str):
        """Set the directory where node log files are stored."""
        self.log_dir = Path(log_dir)

    def register_node(self, name: str, actor_handle: ray.actor.ActorHandle):
        """
        Register a node actor with the coordinator.

        Called by DeployedGraph after creating each NodeWrapper.

        Args:
            name: Node name
            actor_handle: Ray actor handle for the NodeWrapper
        """
        self.node_actors[name] = actor_handle

    def register_dataset_manager(self, actor_handle: ray.actor.ActorHandle):
        """
        Register the DatasetManagerActor with the coordinator.

        Args:
            actor_handle: Ray actor handle for DatasetManagerActor
        """
        self.dataset_manager = actor_handle

    def get_status(self) -> dict:
        """
        Get aggregated status from all nodes and buffer.

        Main method called by falcon monitor.
        Queries all nodes in parallel and returns aggregated status.

        Returns:
            dict with run_dir, started_at, nodes status, and buffer stats
        """
        # Query all nodes in parallel
        node_status_futures = {
            name: actor.get_status.remote()
            for name, actor in self.node_actors.items()
        }

        # Get buffer stats
        buffer_stats = {}
        if self.dataset_manager:
            try:
                buffer_stats = ray.get(
                    self.dataset_manager.get_store_stats.remote(),
                    timeout=2.0
                )
            except Exception:
                buffer_stats = {"error": "timeout"}

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
        """
        Get recent log lines for a specific node.

        Reads directly from the node's output.log file.

        Args:
            node_name: Name of the node
            num_lines: Number of recent log lines to return

        Returns:
            List of recent log lines
        """
        if self.log_dir is None:
            return ["[Log directory not set]"]

        log_file = self.log_dir / node_name / "output.log"
        if not log_file.exists():
            return [f"[Log file not found: {log_file}]"]

        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
            # Return last num_lines, stripped of trailing newlines
            return [line.rstrip() for line in lines[-num_lines:]]
        except Exception as e:
            return [f"[Error reading log: {e}]"]

    def get_node_names(self) -> List[str]:
        """
        Get list of registered node names.

        Returns:
            List of node names
        """
        return list(self.node_actors.keys())
