import asyncio
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sbi.utils  # Don't remove this import, it is needed for sbi.neural_nets.net_builders
from sbi.neural_nets import net_builders

from falcon.core.logging import log
from falcon.core.utils import LazyLoader
from falcon.contrib.torch_embedding import instantiate_embedding
from .hypercubemappingprior import HypercubeMappingPrior
from .norms import LazyOnlineNorm
import copy

class Flow(torch.nn.Module):
    def __init__(self, theta, s, theta_norm=False, log_prefix=None, norm_momentum = 3e-3, net_type = 'nsf'):
        super(Flow, self).__init__()
        self.log_prefix = log_prefix + ":" if log_prefix else ""
        self.theta_norm = LazyOnlineNorm(momentum=norm_momentum, log_prefix=self.log_prefix+"OnlineNorm") if theta_norm else None
        if net_type == 'nsf':
            self.net = net_builders.build_nsf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'made':
            self.net = net_builders.build_made(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'maf':
            self.net = net_builders.build_maf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'maf_rqs':
            self.net = net_builders.build_maf_rqs(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'zuko_nice':
            self.net = net_builders.build_zuko_nice(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'zuko_maf':
            self.net = net_builders.build_zuko_maf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'zuko_nsf':
            self.net = net_builders.build_zuko_nsf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'zuko_ncsf':
            self.net = net_builders.build_zuko_ncsf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'zuko_sospf':
            self.net = net_builders.build_zuko_sospf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'zuko_naf':
            self.net = net_builders.build_zuko_naf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'zuko_unaf':
            self.net = net_builders.build_zuko_unaf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'zuko_cnf':
            self.net = net_builders.build_zuko_cnf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'zuko_gf':
            self.net = net_builders.build_zuko_gf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        elif net_type == 'zuko_bpf':
            self.net = net_builders.build_zuko_bpf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        else:
            raise ValueError("Netowrk type not known", net_type)
        if self.theta_norm is not None:
            theta = self.theta_norm(theta)

    def loss(self, theta, s):
        log({f"{self.log_prefix}target_{i}_min": theta[:, i].min().item() for i in range(theta.shape[1])})
        log({f"{self.log_prefix}target_{i}_max": theta[:, i].max().item() for i in range(theta.shape[1])})
        log({f"{self.log_prefix}summary_{i}_min": s[:, i].min().item() for i in range(s.shape[1])})
        log({f"{self.log_prefix}summary_{i}_max": s[:, i].max().item() for i in range(s.shape[1])})

        if self.theta_norm is not None:
            theta = self.theta_norm(theta)
        loss = self.net.loss(theta.float(), condition=s.float())
        if self.theta_norm is not None:
            volume = self.theta_norm.volume()
            loss = loss + torch.log(volume)
        return loss

    def sample(self, num_samples, s):
        # Return (num_samples, num_conditions, theta_dim) - standard pyro 
        samples = self.net.sample((num_samples,), condition=s).detach()
        if self.theta_norm is not None:
            samples = self.theta_norm.inverse(samples).detach()
        return samples

    def log_prob(self, theta, s):
        # Pyro convention: sample_shape + batch_shape + event_shape
        # Has to work for general (*batch_shape, theta_dim)
        # (num_proposals, num_conditions, theta_dim)
        if self.theta_norm is not None:
            theta = self.theta_norm(theta).detach()
        log_prob = self.net.log_prob(theta.float(), condition=s.float())  # (num_proposals, num_samples)
        if self.theta_norm is not None:
            volume = self.theta_norm.volume().detach()
            log_prob = log_prob - torch.log(volume)
        return log_prob

class SNPE_A:
    def __init__(self, 
                 simulator_instance,
                 device=None,
                 num_epochs=100, 
                 lr_decay_factor=0.1,
                 scheduler_patience=8,
                 early_stop_patience=16,
                 gamma = 0.5, 
                 lr=1e-2,
                 discard_samples=True,
                 theta_norm=True,
                 norm_momentum=1e-2,
                 net_type='zuko_nice',
                 sample_reference_posterior=True,
                 batch_size=128,
                 embedding=None,
                 _embedding_keywords=[]
                 ):
        # Configuration
        self.param_dim = simulator_instance.param_dim

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Auto-detected device: {self.device}")
        else:
            self.device = torch.device(device)
        self.num_epochs = num_epochs
        self.lr_decay_factor = lr_decay_factor  # Factor to reduce LR
        self.scheduler_patience = scheduler_patience  # Patience for scheduler
        self.early_stop_patience = early_stop_patience  # Patience for early stopping
        self.discard_samples = discard_samples
        self.gamma = gamma
        self.lr = lr
        self.theta_norm = theta_norm
        self.norm_momentum = norm_momentum
        self.net_type = net_type
        self.sample_reference_posterior = sample_reference_posterior

        # New embedding instantiation
        self._embedding = instantiate_embedding(embedding).to(self.device)
        self.embedding_keyword_order = _embedding_keywords

        # Prior distribution
        self.simulator_instance = simulator_instance

        # Runtime variables
        self.log_ratio_threshold = -torch.inf  # Dynamic threshold for rejection sampling
        self.networks_initialized = False
        self.best_posterior_val_loss = float('inf')  # Best posterior validation loss
        self.best_traindist_val_loss = float('inf')  # Best traindist validation loss
        
        # Training networks (what optimizer tracks)
        self._posterior = None
        self._traindist = None
        
        # Best-fit networks (for inference only)
        self._best_posterior = None
        self._best_traindist = None
        self._best_embedding = None

    def _initialize_networks(self, theta, conditions):
        self._init_parameters = [theta, conditions]
        inf_conditions = conditions
        print("Initializing LearnableDistribution...")
        print("GPU available:", torch.cuda.is_available())

        # Initialize neural spline flow for posterior distribution
        inf_conditions = [c.to(self.device) for c in inf_conditions]
        s = self._summary(inf_conditions, train=False).detach()
        theta = theta.to(self.device)
        
        # Training networks
        self._posterior = Flow(theta, s, theta_norm = self.theta_norm, log_prefix = 'posterior', norm_momentum = self.norm_momentum, net_type=self.net_type)
        self._posterior.to(self.device)
        self._traindist = Flow(theta, s*0, theta_norm = self.theta_norm, log_prefix = 'traindist', norm_momentum = self.norm_momentum, net_type=self.net_type)
        self._traindist.to(self.device)
        
        # Best-fit networks (initialized as copies of training networks)
        self._best_posterior = Flow(theta, s, theta_norm = self.theta_norm, log_prefix = 'best_posterior', norm_momentum = self.norm_momentum, net_type=self.net_type)
        self._best_posterior.to(self.device)
        self._best_posterior.load_state_dict(self._posterior.state_dict())
        
        self._best_traindist = Flow(theta, s*0, theta_norm = self.theta_norm, log_prefix = 'best_traindist', norm_momentum = self.norm_momentum, net_type=self.net_type)
        self._best_traindist.to(self.device)
        self._best_traindist.load_state_dict(self._traindist.state_dict())
        
        # Best-fit embeddings (if applicable)
        if hasattr(self, '_embedding'):
            self._best_embedding = copy.deepcopy(self._embedding)

        # Initialize optimizer
        parameters = list(self._posterior.parameters())
        parameters += list(self._traindist.parameters())
        if hasattr(self._embedding, "parameters"):
            parameters += list(self._embedding.parameters())
        self._optimizer = AdamW(parameters, lr=self.lr)

        # Initialize Learning Rate Scheduler
        self._scheduler = ReduceLROnPlateau(self._optimizer, mode='min', 
                                           factor=self.lr_decay_factor, 
                                           patience=self.scheduler_patience, 
                                           verbose=True)

        # Set flag
        self.networks_initialized = True
        print("...done initializing LearnableDistribution.")
    
    def _save_posterior_checkpoint(self):
        """Copy current posterior + embeddings state to best-fit networks."""
        cloned_state = {k: v.clone() for k, v in self._posterior.state_dict().items()}
        self._best_posterior.load_state_dict(cloned_state)
        if self._best_embedding is None:
            self._best_embedding = copy.deepcopy(self._embedding)
        else:
            cloned_embedding_state = {k: v.clone() for k, v in self._embedding.state_dict().items()}
            self._best_embedding.load_state_dict(cloned_embedding_state)
    
    def _save_traindist_checkpoint(self):
        """Copy current traindist state to best-fit network."""
        cloned_state = {k: v.clone() for k, v in self._traindist.state_dict().items()}
        self._best_traindist.load_state_dict(cloned_state)

    def _align_singleton_batch_dims(self, tensors, length=None):
        """Broadcast singleton batch dimensions of tensors in a list to same length."""
        if length is None:
            length = max([len(t) for t in tensors])
        return [t.expand(length, *t.shape[1:]) for t in tensors]

    def _summary(self, inf_conditions, train = True, use_best_fit = False):
        """Run conditions through embedding networks and concatenate them."""
        embedding = self._best_embedding if use_best_fit and self._best_embedding is not None else self._embedding
        
        if train:
            embedding.train()
        else:
            embedding.eval()
        
        # TODO: Remove annoying hack once batches are dicts
        inf_conditions = inf_conditions + [None]*10
        s = self._embedding({k: inf_conditions[i] for i, k in enumerate(self.embedding_keyword_order)})
        return s

    def sample(self, num_samples, parent_conditions=[]):
        """Sample from the prior distribution."""
        assert parent_conditions == [], "Conditions are not supported."
        samples = self.simulator_instance.sample(num_samples)
        samples = samples.numpy()
        return samples

    async def train(self, dataloader_train, dataloader_val, hook_fn=None):
        """Train the neural spline flow on the given data."""
        best_val_loss = float('inf')  # Best validation loss
        n_train_batch = 0
        n_val_batch = 0
        for epoch in range(self.num_epochs):
            print(f"⌛ Epoch {epoch+1}/{self.num_epochs}")
            log({"epoch": epoch + 1})

            # Training loop
            #loss_aux_avg = 0
            #loss_train_avg = 0
            num_samples = 0
            for batch in dataloader_train:
                log({"n_train_batch": n_train_batch})
                n_train_batch += 1
                _, theta, inf_conditions = batch[0], batch[1], batch[2:]
                u = self.simulator_instance.inverse(theta)
                if not self.networks_initialized:
                    self._initialize_networks(u, inf_conditions)
                self._optimizer.zero_grad()
                inf_conditions = [c.to(self.device) for c in inf_conditions]
                s = self._summary(inf_conditions, train=True)
                uc = u.to(self.device)
                sc = s.to(self.device)

                self._posterior.train()
                losses_train = self._posterior.loss(uc, sc)
                loss_train = torch.mean(losses_train)

                log({"loss_train_posterior": loss_train.item()})
                #log({"loss_train_posterior_min": losses_train.min().item()})
                #log({"loss_train_posterior_max": losses_train.max().item()})

                self._traindist.train()
                losses_aux = self._traindist.loss(uc, sc.detach()*0)
                loss_aux = torch.mean(losses_aux)

                log({"loss_train_traindist": loss_aux.item()})

                loss_total = loss_train + loss_aux
                loss_total.backward()
                self._optimizer.step()

                #num_samples += len(batch)
                #loss_train_avg += loss_train.sum().item()
                #loss_aux_avg += loss_aux.sum().item()

                # Run hook and allow other tasks to run
                if hook_fn is not None:
                    hook_fn(self, batch)
                await asyncio.sleep(0)

            #loss_train_avg /= num_samples
            #loss_aux_avg /= num_samples

            # Validation loop
            val_posterior_loss = 0
            val_traindist_loss = 0
            num_val_samples = 0
            for batch in dataloader_val:
                log({"n_val_batch": n_val_batch})
                n_val_batch += 1
                _, theta, inf_conditions = batch[0], batch[1], batch[2:]
                u = self.simulator_instance.inverse(theta)
                inf_conditions = [c.to(self.device) for c in inf_conditions]
                s = self._summary(inf_conditions, train=False)
                uc = u.to(self.device)
                sc = s.to(self.device)

                self._posterior.eval()
                posterior_losses = self._posterior.loss(uc, sc)
                val_posterior_loss += torch.sum(posterior_losses).item()

                self._traindist.eval()
                traindist_losses = self._traindist.loss(uc, sc*0)
                val_traindist_loss += torch.sum(traindist_losses).item()

                num_val_samples += len(batch)
                #val_posterior_loss_avg += posterior_loss.sum().item()
                #val_traindist_loss_avg += traindist_loss.sum().item()
                await asyncio.sleep(0)

            val_posterior_loss /= num_val_samples
            val_traindist_loss /= num_val_samples

            log({"loss_val_posterior": val_posterior_loss})
            log({"loss_val_traindist": val_traindist_loss})

            # Check and save best checkpoints independently
            if val_posterior_loss < self.best_posterior_val_loss:
                self.best_posterior_val_loss = val_posterior_loss
                self._save_posterior_checkpoint()
                
            if val_traindist_loss < self.best_traindist_val_loss:
                self.best_traindist_val_loss = val_traindist_loss
                self._save_traindist_checkpoint()

            # Use posterior validation loss for scheduler and early stopping
            self._scheduler.step(val_posterior_loss)

            lr_current = self._optimizer.param_groups[0]['lr']
            log({"lr": lr_current})

            # Early Stopping based on posterior validation loss
            if val_posterior_loss < best_val_loss:
                best_val_loss = val_posterior_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.early_stop_patience:
                print("Early stopping triggered.")
                break
        
        # Training complete - best-fit networks are already updated via checkpoints

    def conditioned_sample(self, num_samples, parent_conditions=[], evidence_conditions=[]):
        samples = self._aux_sample(num_samples, mode = 'posterior', parent_conditions = parent_conditions, evidence_conditions = evidence_conditions)
        samples = samples.numpy()
        return samples

    def proposal_sample(self, num_samples, parent_conditions=[], evidence_conditions=[]):
        # Sample from posterior and log values for reference
        if self.sample_reference_posterior:
            posterior_samples = self._aux_sample(128, mode = 'posterior',
                parent_conditions = parent_conditions, evidence_conditions =
                evidence_conditions)
            vector_mean = posterior_samples.mean(axis=0).cpu()
            vector_std = posterior_samples.std(axis=0).cpu()
            log(
                {f"posterior_mean_{i}": vector_mean[i].item() for i in range(len(vector_mean))},
            )
            log(
                {f"posterior_std_{i}": vector_std[i].item() for i in range(len(vector_std))},
            )

        samples = self._aux_sample(num_samples, mode = 'proposal', parent_conditions = parent_conditions, evidence_conditions = evidence_conditions)
        log({
            "proposal_mean": samples.mean().item(),
            "proposal_std": samples.std().item(),
            })
        samples = samples.numpy()
        return samples

    def _aux_sample(self, num_samples, mode = None, parent_conditions=[], evidence_conditions=[]):
        """Sample from the proposal distribution given conditions."""
        inf_conditions = parent_conditions + evidence_conditions
        # Run conditions through summary network
        assert inf_conditions is not None, "Conditions must be provided."
        inf_conditions = [c.to(self.device) for c in inf_conditions]
        s = self._summary(inf_conditions, train=False, use_best_fit=True)
        s, = self._align_singleton_batch_dims([s], length=num_samples)

        num_proposals = 128

        self._best_traindist.eval()
        samples_proposals = self._best_traindist.sample(num_proposals, s*0).detach()
        # (num_proposals, num_samples, theta_dim)

        log({
            "traindist_mean": samples_proposals.mean().item(),
            "traindist_std": samples_proposals.std().item(),
            })

        self._best_posterior.eval()
        log_prob_post = self._best_posterior.log_prob(
            samples_proposals, s)  # (num_proposals, num_samples)

        self._best_traindist.eval()
        log_prob_dist = self._best_traindist.log_prob(
            samples_proposals, s*0)  # (num_proposals, num_samples)
        
        # Generate "mask" that equals one if samples are outside the [-1, 1] box
        mask = (samples_proposals < -2) | (samples_proposals > 2)
        mask = mask.any(dim=-1).float()*100     # (num_proposals, num_samples)

        gamma = self.gamma


        if mode == 'proposal':
            # Proposal samples, based on auxiliary distribution
	    
            # Option A
            log_weights = gamma/(1.+gamma)*log_prob_post - log_prob_dist - mask

            # Option B
            #log_weights = gamma*(log_prob_post-log_prob_dist) - log_prob_dist - mask

        elif mode == 'posterior':
            # General posterior samples, based on auxiliary distribution alone

            # Option A
            log_weights = log_prob_post - 2*log_prob_dist - mask

            # Option B1
            #log_weights = log_prob_post - gamma/(1.+gamma)*log_prob_post_x0 - log_prob_dist - mask

            # Option B2
            #log_weights = 1./(1.+gamma)*log_prob_post - log_prob_dist - mask

        elif mode == 'prior':
            # Prior samples, based on auxiliary distribution
            log_weights = -log_prob_dist - mask

        else:
            raise KeyError

        #  Use q(z) = q(z|x0)^gamma as proposal
        #  q(z|x) \propto p(x|z) q(z) is approximate posterior

        # Replace -inf and nan with -100
        log_weights = torch.nan_to_num(log_weights, nan=-100.0, neginf=-100.0)

        log_weights = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
        weights = torch.exp(log_weights)  # (num_proposals, num_samples) - sum up to one in first dimension

        weights *= num_proposals
        #print("Weights:", weights.sum(dim=0))
        n_eff = ((weights**2).sum(dim=0)**0.5).min().item()
        log({"n_eff": n_eff})

        idx = torch.multinomial(weights.T, 1, replacement=True).squeeze(-1)

        # samples_proposals have shape (num_proposals, num_samples, theta_dim)
        # samples will have shape (num_samples, theta_dim)
        # idx has shape (num_samples,) and ranges from 0 to num_proposals-1
        # samples by samples_proposals[idx[i], i, :] for i in range(num_samples)

        samples = samples_proposals[idx, torch.arange(num_samples), :]

        samples = self.simulator_instance.forward(samples)

        return samples.to('cpu')

    def discardable(self, theta, parent_conditions=[], evidence_conditions=[]):
        inf_conditions = parent_conditions + evidence_conditions
        u = self.simulator_instance.inverse(theta)
        inf_conditions = [c.to(self.device) for c in inf_conditions]
        s = self._summary(inf_conditions, train=False, use_best_fit=True)
        u, s = self._align_singleton_batch_dims([u, s])
        u = u.to(self.device)
        self._best_posterior.eval()
        self._best_traindist.eval()
        log_prob1 = self._best_posterior.log_prob(u.unsqueeze(0), s).squeeze(0).to('cpu')
        log_prob2 = self._best_traindist.log_prob(u.unsqueeze(0), s*0).squeeze(0).to('cpu')
        log_ratio = log_prob1 - 0*log_prob2  #  p(z|x)/p(z)

        alpha = 0.99
        eta = 1e-3
        t = self.log_ratio_threshold
        t += eta*(sum((log_ratio > t)*(log_ratio - t)*alpha) - 
                  sum((log_ratio < t)*(t - log_ratio)*(1-alpha))
                  )
        offset = 0.5*3**2*self.param_dim
        self.log_ratio_threshold = max(log_ratio.max().item()-offset, self.log_ratio_threshold)
        
        if self.discard_samples:
            #mask = log_ratio < self.log_ratio_threshold
            mask = log_ratio < torch.inf
        else:
            mask = torch.zeros_like(log_ratio).bool()
        #print("rejection fraction:", mask.float().mean().item())
        
        return mask

        # p(z|x)/p_tilde(z) = p(x|z)/p_tilde(x) > 1e-3**dim_params

    def save(self, node_dir: Path):
        # Save best-fit model states to files
        print("Saving:", str(node_dir))
        if not self.networks_initialized:
            raise RuntimeError("Networks not initialized. Call _initialize_networks() first.")
        
        # Save best-fit network states
        torch.save(self._best_posterior.state_dict(), node_dir / "posterior.pth")
        torch.save(self._best_traindist.state_dict(), node_dir / "traindist.pth")
        torch.save(self._init_parameters, node_dir / "init_parameters.pth")
        
        # Save best-fit embedding networks if they exist
        if self._best_embedding is not None:
            torch.save(self._best_embedding.state_dict(), node_dir / "embedding.pth")

    def load(self, node_dir: Path):
        # Load best-fit model states from files
        print("💾 Loading:", str(node_dir))
        init_parameters = torch.load(node_dir / "init_parameters.pth")
        self._initialize_networks(init_parameters[0], init_parameters[1])
        
        # Load saved states into best-fit networks
        posterior_state = torch.load(node_dir / "posterior.pth")
        traindist_state = torch.load(node_dir / "traindist.pth")

        self._best_posterior.load_state_dict(posterior_state)
        self._best_traindist.load_state_dict(traindist_state)

        # Load embedding networks if they exist
        if (node_dir / "embedding.pth").exists() and self._best_embedding is not None:
            embedding_state = torch.load(node_dir / "embedding.pth")
            self._best_embedding.load_state_dict(embedding_state)
