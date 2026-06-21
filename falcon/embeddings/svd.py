import torch
import numpy as np
from typing import Optional, List

from falcon.embeddings.builder import Apply


class _PCAProjector(torch.nn.Module):
    """
    A streaming, dual PCA projector with momentum-based updates.

    This class maintains a running mean and a set of principal components (eigenvectors)
    and eigenvalues (variances) of the input data, computed using a dual PCA approach.
    It stores incoming data in a buffer until the buffer is full, then performs an
    eigen-decomposition update. A momentum term blends the new PCA decomposition
    with the existing one to adapt over time.

    Optionally:
        - A "prior" can be applied, which acts like ridge (Tikhonov) regularization.
          It assumes white noise on the inputs and shrinks each principal component
          proportionally to 1 / (1 + 1 / eigenvalue).
        - The output can be normalized so that the expected variance (averaged over
          all features) is unity.
    """

    def __init__(
        self,
        n_components: int = 10,
        oversampling: int = 10,
        buffer_size: int = 256,
        momentum: float = 0.1,
        normalize_output: bool = True,
        use_prior: bool = True,
    ) -> None:
        super().__init__()
        self.n_components: int = n_components
        self.oversampling: int = oversampling
        self.buffer_size: int = buffer_size
        self.momentum: float = momentum
        self.normalize_output: bool = normalize_output
        self.use_prior: bool = use_prior

        self.buffer: List[torch.Tensor] = []
        self.buffer_counter: int = 0

        self.components: Optional[torch.Tensor] = None
        self.eigenvalues: Optional[torch.Tensor] = None

    def update(self, X: torch.Tensor) -> None:
        """Accumulate a batch of data. If the buffer is full, update the PCA decomposition."""
        batch_size = X.shape[0]

        self.buffer.append(X)
        self.buffer_counter += batch_size

        if self.buffer_counter >= self.buffer_size:
            self._compute_svd_update()
            self.buffer = []
            self.buffer_counter = 0

    def _compute_svd_update(self) -> None:
        """Perform a momentum-weighted dual PCA update of the top-k components and eigenvalues."""
        X_buffer = torch.cat(self.buffer, dim=0)
        N = X_buffer.shape[0]

        X_centered = X_buffer - X_buffer.mean(dim=0)

        K = X_centered @ X_centered.T / N

        eigvals, eigvecs = torch.linalg.eigh(K)

        top_indices = torch.argsort(eigvals, descending=True)[: self.n_components]

        Λ_new = eigvals[top_indices]
        Q = eigvecs[:, top_indices]

        U_new = (X_centered.T @ Q) / torch.sqrt(N * Λ_new + 1e-6)
        U_new = U_new.T

        if self.components is None:
            self.components = U_new
            self.eigenvalues = Λ_new
        else:
            alpha = self.momentum
            U_combined = torch.cat(
                [np.sqrt(1 - alpha) * self.components, np.sqrt(alpha) * U_new], dim=0
            )

            U, S, Vt = torch.linalg.svd(U_combined, full_matrices=False)

            self.components = Vt[: self.n_components]
            self.eigenvalues = (1 - alpha) * self.eigenvalues + alpha * Λ_new

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Project X onto the learned principal components."""
        if self.components is None:
            raise ValueError(
                "SVD components not computed yet. Call update() enough times first."
            )

        X_proj = X @ self.components.T

        if self.use_prior:
            if self.eigenvalues is None:
                raise ValueError(
                    "Eigenvalues not available. PCA must be computed first."
                )
            X_proj = X_proj / (1.0 + (1.0 / self.eigenvalues)).unsqueeze(0)

        X_reconstructed = X_proj @ self.components

        if self.normalize_output:
            if self.eigenvalues is None:
                raise ValueError(
                    "Eigenvalues not available. PCA must be computed first."
                )
            input_dim = X_reconstructed.shape[-1]
            scale_factor = (self.eigenvalues.sum() / input_dim) ** 0.5
            X_reconstructed /= scale_factor

        return X_reconstructed


def PCAProjector(inputs=None, *, n_components=10, oversampling=10, buffer_size=256,
                 momentum=0.1, normalize_output=True, use_prior=True):
    """Typed factory for PCAProjector embedding. Returns a pipeline config dict.

    In a notebook, use: PCAProjector(["x"], n_components=32)
    In YAML, use:  _target_: falcon.embeddings.PCAProjector  (inputs via _input_)
    """
    return Apply(_PCAProjector, inputs, n_components=n_components, oversampling=oversampling,
                 buffer_size=buffer_size, momentum=momentum, normalize_output=normalize_output,
                 use_prior=use_prior)
