import torch
import matplotlib.pyplot as plt
from sbi.inference import SNPE, SNPE_C
from sbi.utils import BoxUniform

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Problem setup
dim = 10
max_epochs = 5
de = "maf"
prior = BoxUniform(low=-torch.ones(dim, device=device),
                   high=torch.ones(dim, device=device))

# Simulator: x = theta + eps
def simulator(theta):
    eps = 1e-2 * torch.randn_like(theta, device=device)
    return theta + eps

# Generate training data
num_simulations = 50_000
theta = prior.sample((num_simulations,))
x = simulator(theta)

def get_losses(summary_dict):
    """Find a plausible training loss array in summary dict."""
    for key in [
        "train_log_probs", "training_log_probs",
        "train_log_probs_per_epoch", "train_losses",
        "training_loss"
    ]:
        if key in summary_dict:
            return summary_dict[key]
    raise KeyError(f"No known loss key found. Available keys: {summary_dict.keys()}")

# --- Option 1: Regular NPE ---
inference = SNPE(prior=prior, density_estimator=de, device=device)
inference = inference.append_simulations(theta, x)
density_estimator = inference.train(max_num_epochs=max_epochs)
print("NPE summary keys:", inference.summary.keys())
losses_npe = get_losses(inference.summary)
posterior_npe = inference.build_posterior(density_estimator)

# --- Option 2: SNPE-C (atomic loss) ---
inference_atomic = SNPE_C(prior=prior, density_estimator=de, device=device)
inference_atomic = inference_atomic.append_simulations(theta, x)
density_estimator_atomic = inference_atomic.train(max_num_epochs=max_epochs, num_atoms = 0)
print("SNPE-C summary keys:", inference_atomic.summary.keys())
losses_atomic = get_losses(inference_atomic.summary)
posterior_atomic = inference_atomic.build_posterior(density_estimator_atomic)

# Observation to infer from
theta_true = torch.zeros(dim, device=device)
x_obs = simulator(theta_true.unsqueeze(0))[0]

# Sample posteriors
samples_npe = posterior_npe.sample((40_000,), x=x_obs).cpu()
samples_atomic = posterior_atomic.sample((40_000,), x=x_obs).cpu()

# --- Plot marginals for a few dims ---
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
dims_to_plot = [0, 1, 2, 3, 4, 5]
for ax, d in zip(axes.flat, dims_to_plot):
    ax.hist(samples_npe[:, d].numpy(), bins=100, density=True, alpha=0.6, label="NPE")
    ax.hist(samples_atomic[:, d].numpy(), bins=100, density=True, alpha=0.6, label="SNPE-C")
    ax.axvline(theta_true[d].item(), color="k", linestyle="--")
    ax.set_xlim(-1, 1)
    ax.set_title(f"θ[{d}] marginal")
ax.legend()
plt.tight_layout()
plt.savefig("test.png")
plt.close()

# --- Plot loss curves ---
plt.figure(figsize=(6, 4))
plt.plot(losses_npe, label="NPE loss")
plt.plot(losses_atomic, label="SNPE-C loss")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curves.png")
plt.close()