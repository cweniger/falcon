from typing import Optional, Dict, Any
import ray
import wandb


def start_wandb_logger(
    wandb_project: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_dir: Optional[str] = None,
):
    logger = WandBManager.options(
        name="falcon:global_logger", lifetime="detached"
    ).remote(
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_dir=wandb_dir,
    )


def finish_wandb_logger():
    """
    Stop the W&B logger actor.
    """
    logger = ray.get_actor(name="falcon:global_logger")
    ray.get(logger.shutdown.remote())
    ray.kill(logger)


@ray.remote
class WandBWrapper:
    def __init__(
        self,
        wandb_project: Optional[str] = None,
        wandb_group: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_dir: Optional[str] = None,
    ):
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        self.wandb_name = wandb_name
        self.wandb_config = wandb_config
        self.wandb_dir = wandb_dir

        wandb_init_kwargs = {
            "project": self.wandb_project,
            "group": self.wandb_group,
            "name": self.wandb_name,
            "config": self.wandb_config or {},
            "reinit": True,
        }
        if self.wandb_dir:
            wandb_init_kwargs["dir"] = self.wandb_dir

        self.wandb_run = wandb.init(**wandb_init_kwargs)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log scalar metrics, wandb style.
        """
        try:
            self.wandb_run.log(metrics, step=step)
        except:
            # Handle the case where the actor is not available
            print("Error logging metrics:", self.wandb_name, metrics)

    def shutdown(self):
        """
        Finish the W&B run.
        """
        self.wandb_run.finish()


@ray.remote
class WandBManager:
    def __init__(
        self,
        wandb_project: Optional[str] = None,
        wandb_group: Optional[str] = None,
        wandb_dir: Optional[str] = None,
    ):
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        self.wandb_dir = wandb_dir
        self.wandb_runs = {}

    def init(self, actor_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new W&B run.
        """
        self.wandb_runs[actor_id] = WandBWrapper.remote(
            wandb_project=self.wandb_project,
            wandb_group=self.wandb_group,
            wandb_name=actor_id,
            wandb_config=config,
            wandb_dir=self.wandb_dir,
        )

    def log(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        actor_id: str = None,
    ):
        """
        Log scalar metrics, wandb style.
        """
        self.wandb_runs[actor_id].log.remote(metrics, step=step)

    def shutdown(self):
        """
        Finish all W&B runs.
        """
        for _, wandb_run in self.wandb_runs.items():
            ray.get(wandb_run.shutdown.remote())
