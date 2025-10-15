import torch
import numpy as np
from typing import Optional, List


class PCAProjector(torch.nn.Module):
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
        # input_dim: int,
        oversampling: int = 10,
        buffer_size: int = 256,
        momentum: float = 0.1,
        normalize_output: bool = True,
        use_prior: bool = True,
        # add_mean: bool = False,
    ) -> None:
        """
        Args:
            n_components: Number of principal components to retain.
            oversampling: Extra dimensions to capture a slightly larger subspace (not currently used).
            buffer_size: Number of samples to accumulate before performing an SVD update.
            momentum: Blend factor for merging new PCA decomposition with the old one.
            normalize_output: Whether to normalize the reconstructed outputs
                              so that they have unit average variance.
            use_prior: Whether to apply a variance-based prior (ridge-like shrinkage).
        """
        super().__init__()
        self.n_components: int = n_components
        # self.input_dim: int = input_dim
        self.oversampling: int = oversampling
        self.buffer_size: int = buffer_size
        # self.device: str = device
        self.momentum: float = momentum
        self.normalize_output: bool = normalize_output
        self.use_prior: bool = use_prior
        # self.add_mean: bool = add_mean

        # Running mean of the input data, updated incrementally
        # self.mean: Optional[torch.Tensor] = None  # shape: (D,)
        # Number of samples accumulated so far (used for updating the mean)
        # self.n_samples: int = 0

        # Temporary buffer for incoming data points
        self.buffer: List[torch.Tensor] = []
        # Counts how many samples have been appended to the buffer
        self.buffer_counter: int = 0

        # Principal components (top-k right singular vectors) and eigenvalues
        self.components: Optional[torch.Tensor] = None  # shape: (k, D)
        self.eigenvalues: Optional[torch.Tensor] = None  # shape: (k,)

    def update(self, X: torch.Tensor) -> None:
        """
        Accumulate a batch of data in the buffer. If the buffer is full, update the PCA decomposition.

        Args:
            X: A batch of input data with shape (batch_size, D).
        """
        batch_size = X.shape[0]

        # Store in the buffer
        self.buffer.append(X)
        self.buffer_counter += batch_size

        # If we've accumulated enough samples in the buffer, update the PCA decomposition
        if self.buffer_counter >= self.buffer_size:
            self._compute_svd_update()
            # Clear the buffer and reset counter
            self.buffer = []
            self.buffer_counter = 0

    def _compute_svd_update(self) -> None:
        """
        Perform a momentum-weighted dual PCA update of the top-k components and eigenvalues.

        - Constructs a dual covariance matrix in the sample domain (N x N).
        - Computes its eigen-decomposition (eigh).
        - Projects eigenvectors back into feature space to get principal components.
        - Merges new components with old ones via momentum.
        """
        # Concatenate all samples in the buffer along the batch dimension
        X_buffer = torch.cat(self.buffer, dim=0)  # shape: (N, D)
        N = X_buffer.shape[0]

        # Center the buffer by subtracting the current global mean
        # X_centered = X_buffer - self.mean
        X_centered = X_buffer - X_buffer.mean(dim=0)

        # Dual covariance matrix K = (1/N) X_centered * X_centered^T
        # shape: (N, N)
        K = X_centered @ X_centered.T / N

        # Eigen-decomposition of the dual covariance matrix
        # eigvals: shape (N,), eigvecs: shape (N, N) (columns are eigenvectors)
        eigvals, eigvecs = torch.linalg.eigh(K)  # ascending order

        # Select indices of the top-k eigenvalues
        top_indices = torch.argsort(eigvals, descending=True)[: self.n_components]

        # Extract the top-k eigenvalues
        Λ_new = eigvals[top_indices]  # shape: (k,)
        # Extract the corresponding eigenvectors
        Q = eigvecs[:, top_indices]  # shape: (N, k)

        # Project back to feature space to get top-k eigenvectors.
        # We do U_new = (X_centered^T @ Q) / sqrt(N * Λ_new).
        # Note that each principal axis in feature space has length sqrt(Λ_new).
        U_new = (X_centered.T @ Q) / torch.sqrt(N * Λ_new + 1e-6)  # shape: (D, k)
        U_new = U_new.T  # shape: (k, D)

        # If this is the first update, just set the components
        if self.components is None:
            self.components = U_new
            self.eigenvalues = Λ_new
        else:
            # Merge the new components with the old ones using 'momentum'
            alpha = self.momentum
            # Weighted concatenation of the old and new components
            # We scale old components by sqrt(1 - alpha), new by sqrt(alpha) for energy balance
            U_combined = torch.cat(
                [np.sqrt(1 - alpha) * self.components, np.sqrt(alpha) * U_new], dim=0
            )  # shape: (2k, D)

            # SVD to re-orthonormalize the combined set of vectors
            U, S, Vt = torch.linalg.svd(U_combined, full_matrices=False)

            # Take the top-k vectors (right singular vectors)
            self.components = Vt[: self.n_components]
            # Blend the eigenvalues (variances) in a simpler linear way
            self.eigenvalues = (1 - alpha) * self.eigenvalues + alpha * Λ_new

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Filter the input data by projecting onto the learned principal components,
        optionally applying a variance-based prior, then reconstructing and possibly normalizing.

        Args:
            X: A batch of input data with shape (batch_size, D).

        Returns:
            torch.Tensor: A batch of data with shape (batch_size, D) after PCA
            transform (and optional prior & normalization).
        """
        # If no PCA has been computed yet, we can't project
        if self.components is None:
            raise ValueError(
                "SVD components not computed yet. Call update() enough times first."
            )

        # Shift input by the running mean
        # X_centered = X - self.mean
        # Project onto the principal components
        X_proj = X @ self.components.T  # shape: (batch_size, k)

        # Optionally apply the variance-based prior
        # This is like ridge regularization in a Bayesian sense,
        # assuming white noise on the inputs: X_proj / (1 + 1 / eigenvalues).
        if self.use_prior:
            if self.eigenvalues is None:
                raise ValueError(
                    "Eigenvalues not available. PCA must be computed first."
                )
            X_proj = X_proj / (1.0 + (1.0 / self.eigenvalues)).unsqueeze(0)

        # Reconstruct from the principal components
        X_reconstructed = X_proj @ self.components

        # Optionally normalize the output so that the average variance is ~1
        if self.normalize_output:
            if self.eigenvalues is None:
                raise ValueError(
                    "Eigenvalues not available. PCA must be computed first."
                )
            # The sum of eigenvalues is the total variance in the top-k subspace
            # We divide by sqrt( average variance per feature ) = sqrt( sum / D )
            input_dim = X_reconstructed.shape[-1]
            scale_factor = (self.eigenvalues.sum() / input_dim) ** 0.5
            X_reconstructed /= scale_factor

        return X_reconstructed
