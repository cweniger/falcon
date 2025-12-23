import asyncio
from pathlib import Path
import copy
import time

import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import ray
import sbi.utils  # Don't remove - needed for sbi.neural_nets.net_builders
from sbi.neural_nets import net_builders

from falcon.core.logging import log
from falcon.core.utils import LazyLoader, RVBatch
from falcon.contrib.torch_embedding import instantiate_embedding
from .hypercubemappingprior import HypercubeMappingPrior
from .norms import LazyOnlineNorm


# Network builder registry
NET_BUILDERS = {
    "nsf": net_builders.build_nsf,
    "made": net_builders.build_made,
    "maf": net_builders.build_maf,
    "maf_rqs": net_builders.build_maf_rqs,
    "zuko_nice": net_builders.build_zuko_nice,
    "zuko_maf": net_builders.build_zuko_maf,
    "zuko_nsf": net_builders.build_zuko_nsf,
    "zuko_ncsf": net_builders.build_zuko_ncsf,
    "zuko_sospf": net_builders.build_zuko_sospf,
    "zuko_naf": net_builders.build_zuko_naf,
    "zuko_unaf": net_builders.build_zuko_unaf,
    "zuko_gf": net_builders.build_zuko_gf,
    "zuko_bpf": net_builders.build_zuko_bpf,
}


class Flow(torch.nn.Module):
    def __init__(
        self,
        theta,
        s,
        theta_norm=False,
        norm_momentum=3e-3,
        net_type="nsf",
        use_log_update=False,
        adaptive_momentum=False,
    ):
        super().__init__()
        self.theta_norm = (
            LazyOnlineNorm(
                momentum=norm_momentum,
                use_log_update=use_log_update,
                adaptive_momentum=adaptive_momentum,
            )
            if theta_norm
            else None
        )

        builder = NET_BUILDERS.get(net_type)
        if builder is None:
            raise ValueError(f"Unknown net_type: {net_type}. Available: {list(NET_BUILDERS.keys())}")
        self.net = builder(theta.float(), s.float(), z_score_x=None, z_score_y=None)

        if self.theta_norm is not None:
            self.theta_norm(theta)  # Initialize normalization stats
        self.scale = 0.2

    def loss(self, theta, s):
        if self.theta_norm is not None:
            theta = self.theta_norm(theta)
        theta = theta.float() * self.scale
        loss = self.net.loss(theta, condition=s.float())
        loss = loss - np.log(self.scale) * theta.shape[-1]
        if self.theta_norm is not None:
            loss = loss + torch.log(self.theta_norm.volume())
        return loss

    def sample(self, num_samples, s):
        samples = self.net.sample((num_samples,), condition=s).detach()
        samples = samples / self.scale
        if self.theta_norm is not None:
            samples = self.theta_norm.inverse(samples).detach()
        return samples

    def log_prob(self, theta, s):
        if self.theta_norm is not None:
            theta = self.theta_norm(theta).detach()
        theta = theta * self.scale
        log_prob = self.net.log_prob(theta.float(), condition=s.float())
        log_prob = log_prob + np.log(self.scale) * theta.shape[-1]
        if self.theta_norm is not None:
            log_prob = log_prob - torch.log(self.theta_norm.volume().detach())
        return log_prob


class SNPE_A:
    def __init__(
        self,
        simulator_instance,
        device=None,
        num_epochs=100,
        lr_decay_factor=0.1,
        scheduler_patience=8,
        early_stop_patience=16,
        gamma=0.5,
        lr=1e-2,
        discard_samples=True,
        theta_norm=True,
        norm_momentum=1e-2,
        net_type="zuko_nice",
        sample_reference_posterior=False,
        batch_size=128,
        embedding=None,
        _embedding_keywords=[],
        use_best_models_during_inference=True,
        use_log_update=False,
        adaptive_momentum=False,
        log_ratio_threshold=-20.0,
        reset_network_after_pause=False,
    ):
        self.param_dim = simulator_instance.param_dim
        self.device = (
            torch.device(device) if device else
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device is None:
            print(f"Auto-detected device: {self.device}")

        # Store config
        self.num_epochs = num_epochs
        self.lr_decay_factor = lr_decay_factor
        self.scheduler_patience = scheduler_patience
        self.early_stop_patience = early_stop_patience
        self.discard_samples = discard_samples
        self.gamma = gamma
        self.lr = lr
        self.theta_norm = theta_norm
        self.norm_momentum = norm_momentum
        self.adaptive_momentum = adaptive_momentum
        self.net_type = net_type
        self.sample_reference_posterior = sample_reference_posterior
        self._use_best_models_during_inference = use_best_models_during_inference
        self._use_log_update = use_log_update
        self.log_ratio_threshold = log_ratio_threshold
        self.reset_network_after_pause = reset_network_after_pause

        # Embedding
        self._embedding = instantiate_embedding(embedding).to(self.device)
        self.embedding_keyword_order = _embedding_keywords

        # Prior/simulator
        self.simulator_instance = simulator_instance

        # Networks (initialized lazily)
        self.networks_initialized = False
        self._conditional_flow = None
        self._marginal_flow = None
        self._best_conditional_flow = None
        self._best_marginal_flow = None
        self._best_embedding = None
        self.best_conditional_flow_val_loss = float("inf")
        self.best_marginal_flow_val_loss = float("inf")

        # Consolidated history tracking
        self.history = {
            "train_ids": [],
            "val_ids": [],
            "theta_mins": [],
            "theta_maxs": [],
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "n_samples": [],
            "elapsed_min": [],
        }

        # Async control
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._terminated = False
        self._break_flag = False

    def _create_flow(self, theta, s, is_conditional=True):
        """Create a Flow network with standard settings."""
        return Flow(
            theta,
            s if is_conditional else s * 0,
            theta_norm=self.theta_norm,
            norm_momentum=self.norm_momentum,
            net_type=self.net_type,
            use_log_update=self._use_log_update,
            adaptive_momentum=self.adaptive_momentum,
        )

    def _initialize_networks(self, theta, conditions):
        self._init_parameters = [theta, conditions]
        print("Initializing networks...")
        print("GPU available:", torch.cuda.is_available())

        conditions = [c.to(self.device) for c in conditions]
        s = self._embed(conditions, train=False).detach()
        theta = theta.to(self.device)

        # Training networks
        self._conditional_flow = self._create_flow(theta, s, is_conditional=True)
        self._conditional_flow.to(self.device)

        self._marginal_flow = self._create_flow(theta, s, is_conditional=False)
        self._marginal_flow.to(self.device)

        # Best-fit networks (copies of training networks)
        self._best_conditional_flow = self._create_flow(theta, s, is_conditional=True)
        self._best_conditional_flow.to(self.device)
        self._best_conditional_flow.load_state_dict(self._conditional_flow.state_dict())

        self._best_marginal_flow = self._create_flow(theta, s, is_conditional=False)
        self._best_marginal_flow.to(self.device)
        self._best_marginal_flow.load_state_dict(self._marginal_flow.state_dict())

        self._best_embedding = copy.deepcopy(self._embedding)

        # Optimizer
        parameters = (
            list(self._conditional_flow.parameters()) +
            list(self._marginal_flow.parameters()) +
            list(self._embedding.parameters())
        )
        self._optimizer = AdamW(parameters, lr=self.lr)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=self.lr_decay_factor,
            patience=self.scheduler_patience,
        )

        self.networks_initialized = True
        print("...done initializing networks.")

    def _update_best_weights(self, network_type="conditional"):
        """Copy current network weights to best-fit checkpoint."""
        if network_type == "conditional":
            self._best_conditional_flow.load_state_dict(
                {k: v.clone() for k, v in self._conditional_flow.state_dict().items()}
            )
            self._best_embedding.load_state_dict(
                {k: v.clone() for k, v in self._embedding.state_dict().items()}
            )
        else:
            self._best_marginal_flow.load_state_dict(
                {k: v.clone() for k, v in self._marginal_flow.state_dict().items()}
            )

    def _embed(self, conditions, train=True, use_best_fit=False):
        """Run conditions through embedding network."""
        embedding = (
            self._best_embedding
            if use_best_fit and self._best_embedding is not None
            else self._embedding
        )
        embedding.train() if train else embedding.eval()

        # TODO: Remove hack once batches are dicts
        conditions = conditions + [None] * 10
        return embedding(
            {k: conditions[i] for i, k in enumerate(self.embedding_keyword_order)}
        )

    def sample_prior(self, num_samples, parent_conditions=[]):
        """Sample from the prior distribution."""
        assert parent_conditions == [], "Conditions are not supported."
        samples = self.simulator_instance.simulate_batch(num_samples)
        logprob = np.ones(num_samples) * (-np.log(4) ** self.param_dim)
        return RVBatch(samples, logprob=logprob)

    async def train(self, dataloader_train, dataloader_val, hook_fn=None, dataset_manager=None):
        """Train the neural spline flow."""
        best_val_loss = float("inf")
        n_train_batch = 0
        n_val_batch = 0
        counter_post = 0
        counter_aux = 0
        t0 = time.perf_counter()

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            log({"epoch": epoch + 1})

            # Training loop
            loss_train_avg = 0
            loss_aux_avg = 0
            num_batches = 0

            for batch in dataloader_train:
                log({"train:step": n_train_batch})
                n_train_batch += 1

                ids, theta, theta_logprob = batch[0], batch[1], batch[2]
                inf_conditions = batch[3:]

                ts = time.time()
                self.history["train_ids"].extend((ts, id) for id in ids.numpy().tolist())

                log({"train:theta_logprob_min": theta_logprob.min().item()})
                log({"train:theta_logprob_max": theta_logprob.max().item()})

                u = self.simulator_instance.inverse(theta)
                if not self.networks_initialized:
                    self._initialize_networks(u, inf_conditions)

                self._optimizer.zero_grad()
                inf_conditions = [c.to(self.device) for c in inf_conditions]
                s = self._embed(inf_conditions, train=True)
                uc = u.to(self.device)
                sc = s.to(self.device)

                # Track theta ranges
                with torch.no_grad():
                    self.history["theta_mins"].append(theta.min(dim=0).values.cpu().numpy())
                    self.history["theta_maxs"].append(theta.max(dim=0).values.cpu().numpy())

                # Compute losses
                self._conditional_flow.train()
                loss_train = self._conditional_flow.loss(uc, sc).mean()
                log({"train:loss": loss_train.item()})

                self._marginal_flow.train()
                loss_aux = self._marginal_flow.loss(uc, sc.detach() * 0).mean()
                log({"train:loss_aux": loss_aux.item()})

                (loss_train + loss_aux).backward()
                self._optimizer.step()

                num_batches += 1
                loss_train_avg += loss_train.item()
                loss_aux_avg += loss_aux.item()

                if hook_fn is not None:
                    hook_fn(self, batch)
                await asyncio.sleep(0)
                await self._pause_event.wait()
                if self._break_flag:
                    self._break_flag = False
                    break

            loss_train_avg /= num_batches
            loss_aux_avg /= num_batches

            # Validation loop
            val_post_loss = 0
            val_aux_loss = 0
            num_val = 0

            for batch in dataloader_val:
                log({"val:step": n_val_batch})
                n_val_batch += 1

                ids, theta, theta_logprob = batch[0], batch[1], batch[2]
                inf_conditions = batch[3:]

                ts = time.time()
                self.history["val_ids"].extend((ts, id) for id in ids.numpy().tolist())

                log({"val:theta_logprob_min": theta_logprob.min().item()})
                log({"val:theta_logprob_max": theta_logprob.max().item()})

                u = self.simulator_instance.inverse(theta)
                inf_conditions = [c.to(self.device) for c in inf_conditions]
                s = self._embed(inf_conditions, train=False)
                uc = u.to(self.device)
                sc = s.to(self.device)

                self._conditional_flow.eval()
                val_post_loss += self._conditional_flow.loss(uc, sc).sum().item()

                self._marginal_flow.eval()
                val_aux_loss += self._marginal_flow.loss(uc, sc * 0).sum().item()

                num_val += uc.shape[0]
                await asyncio.sleep(0)
                await self._pause_event.wait()
                if self._break_flag:
                    self._break_flag = False
                    break

            val_post_loss /= num_val
            val_aux_loss /= num_val

            log({"val:loss": val_post_loss})
            log({"val:loss_aux": val_aux_loss})

            # Checkpointing
            if val_post_loss < self.best_conditional_flow_val_loss:
                self.best_conditional_flow_val_loss = val_post_loss
                self._update_best_weights("conditional")
                log({"checkpoint:count": counter_post})
                counter_post += 1

            if val_aux_loss < self.best_marginal_flow_val_loss:
                self.best_marginal_flow_val_loss = val_aux_loss
                self._update_best_weights("marginal")
                log({"checkpoint:count_aux": counter_aux})
                counter_aux += 1

            self._scheduler.step(val_post_loss)
            log({"lr": self._optimizer.param_groups[0]["lr"]})

            # Early stopping
            if val_post_loss < best_val_loss:
                best_val_loss = val_post_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Record history
            self.history["epochs"].append(epoch + 1)
            self.history["train_loss"].append(loss_train_avg)
            self.history["val_loss"].append(val_post_loss)

            try:
                stats = ray.get(dataset_manager.get_store_stats.remote())
                self.history["n_samples"].append(stats["total_length"])
            except Exception:
                pass

            elapsed = (time.perf_counter() - t0) / 60.0
            self.history["elapsed_min"].append(elapsed)
            log({"elapsed_minutes": elapsed})

            if epochs_no_improve >= self.early_stop_patience:
                print("Early stopping triggered.")
                break

            await self._pause_event.wait()
            if self._terminated:
                break

    def _importance_sample(self, num_samples, mode="posterior", parent_conditions=[], evidence_conditions=[]):
        """Sample using importance sampling."""
        conditions = parent_conditions + evidence_conditions
        assert conditions, "Conditions must be provided."
        conditions = [c.to(self.device) for c in conditions]

        if self._use_best_models_during_inference:
            conditional_net = self._best_conditional_flow
            marginal_net = self._best_marginal_flow
            s = self._embed(conditions, train=False, use_best_fit=True)
        else:
            conditional_net = self._conditional_flow
            marginal_net = self._marginal_flow
            s = self._embed(conditions, train=False)

        s = s.expand(num_samples, *s.shape[1:])

        # Generate proposals from conditional flow
        num_proposals = 256
        conditional_net.eval()
        samples_proposals = conditional_net.sample(num_proposals, s).detach()

        log({
            "importance_sample:proposal_mean": samples_proposals.mean().item(),
            "importance_sample:proposal_std": samples_proposals.std().item(),
        })

        # Compute log probs
        log_prob_cond = conditional_net.log_prob(samples_proposals, s)
        marginal_net.eval()
        log_prob_marg = marginal_net.log_prob(samples_proposals, s * 0)

        # Mask samples outside [-2, 2] box
        mask = (samples_proposals < -2) | (samples_proposals > 2)
        mask = mask.any(dim=-1).float() * 100

        # Compute importance weights
        if mode == "proposal":
            log_weights = -1.0 / (1.0 + self.gamma) * log_prob_cond - mask
        else:  # "posterior" - reweight by marginal
            log_weights = -log_prob_marg - mask

        log_weights = torch.nan_to_num(log_weights, nan=-100.0, neginf=-100.0)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
        weights = torch.exp(log_weights)

        # Effective sample size
        n_eff = 1 / (weights**2).sum(dim=0).cpu().detach().numpy()
        log({"importance_sample:n_eff_min": n_eff.min()})
        log({"importance_sample:n_eff_max": n_eff.max()})

        # Resample
        idx = torch.multinomial(weights.T, 1, replacement=True).squeeze(-1)
        samples = samples_proposals[idx, torch.arange(num_samples), :]
        samples = self.simulator_instance.forward(samples).cpu()
        logprob = log_prob_cond[idx, torch.arange(num_samples)].cpu()

        return samples, logprob.detach()

    def sample_posterior(self, num_samples, parent_conditions=[], evidence_conditions=[]):
        """Sample from the posterior distribution q(θ|x)."""
        samples, logprob = self._importance_sample(
            num_samples, mode="posterior",
            parent_conditions=parent_conditions,
            evidence_conditions=evidence_conditions,
        )
        return RVBatch(samples.numpy(), logprob=logprob.numpy())

    def sample_proposal(self, num_samples, parent_conditions=[], evidence_conditions=[]):
        """Sample from the widened proposal distribution for adaptive resampling."""
        if self.sample_reference_posterior:
            post_samples, _ = self._importance_sample(
                128, mode="posterior",
                parent_conditions=parent_conditions,
                evidence_conditions=evidence_conditions,
            )
            mean, std = post_samples.mean(dim=0).cpu(), post_samples.std(dim=0).cpu()
            log({f"sample_proposal:posterior_mean_{i}": mean[i].item() for i in range(len(mean))})
            log({f"sample_proposal:posterior_std_{i}": std[i].item() for i in range(len(std))})

        samples, logprob = self._importance_sample(
            num_samples, mode="proposal",
            parent_conditions=parent_conditions,
            evidence_conditions=evidence_conditions,
        )
        log({
            "sample_proposal:mean": samples.mean().item(),
            "sample_proposal:std": samples.std().item(),
            "sample_proposal:logprob": logprob.mean().item(),
        })
        return RVBatch(samples.numpy(), logprob=logprob.numpy())

    def get_discard_mask(self, theta, theta_logprob, parent_conditions=[], evidence_conditions=[]):
        """Return boolean mask of low-likelihood samples to discard."""
        if not self.discard_samples:
            return torch.zeros(len(theta), dtype=torch.bool)

        conditions = parent_conditions + evidence_conditions
        u = self.simulator_instance.inverse(theta)
        conditions = [c.to(self.device) for c in conditions]
        s = self._embed(conditions, train=False, use_best_fit=True)

        u = u.expand(len(theta), *u.shape[1:]) if u.shape[0] == 1 else u
        s = s.expand(len(theta), *s.shape[1:]) if s.shape[0] == 1 else s

        u = u.to(self.device)
        self._conditional_flow.eval()
        log_prob = self._conditional_flow.log_prob(u.unsqueeze(0), s).squeeze(0).cpu()
        log_ratio = log_prob - theta_logprob
        return log_ratio < self.log_ratio_threshold

    def save(self, node_dir: Path):
        print("Saving:", str(node_dir))
        if not self.networks_initialized:
            raise RuntimeError("Networks not initialized.")

        torch.save(self._best_conditional_flow.state_dict(), node_dir / "conditional_flow.pth")
        torch.save(self._best_marginal_flow.state_dict(), node_dir / "marginal_flow.pth")
        torch.save(self._init_parameters, node_dir / "init_parameters.pth")

        # Save history
        torch.save(self.history["train_ids"], node_dir / "train_id_history.pth")
        torch.save(self.history["val_ids"], node_dir / "validation_id_history.pth")
        torch.save(self.history["theta_mins"], node_dir / "theta_mins_batches.pth")
        torch.save(self.history["theta_maxs"], node_dir / "theta_maxs_batches.pth")
        torch.save(self.history["epochs"], node_dir / "epochs.pth")
        torch.save(self.history["train_loss"], node_dir / "loss_train_posterior.pth")
        torch.save(self.history["val_loss"], node_dir / "loss_val_posterior.pth")
        torch.save(self.history["n_samples"], node_dir / "n_samples_total.pth")
        torch.save(self.history["elapsed_min"], node_dir / "elapsed_minutes.pth")

        if self._best_embedding is not None:
            torch.save(self._best_embedding.state_dict(), node_dir / "embedding.pth")

    def load(self, node_dir: Path):
        print("Loading:", str(node_dir))
        init_parameters = torch.load(node_dir / "init_parameters.pth")
        self._initialize_networks(init_parameters[0], init_parameters[1])

        self._best_conditional_flow.load_state_dict(torch.load(node_dir / "conditional_flow.pth"))
        self._best_marginal_flow.load_state_dict(torch.load(node_dir / "marginal_flow.pth"))

        if (node_dir / "embedding.pth").exists() and self._best_embedding is not None:
            self._best_embedding.load_state_dict(torch.load(node_dir / "embedding.pth"))

    def pause(self):
        self._pause_event.clear()

    def resume(self, reset_network=False):
        if self.reset_network_after_pause:
            self.networks_initialized = False
            self._break_flag = True
        self._pause_event.set()

    def interrupt(self):
        self._terminated = True
        self._pause_event.set()
