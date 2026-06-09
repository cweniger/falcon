import torch
import numpy as np
from typing import Optional, List


class DynamicSVD(torch.nn.Module):
    """
    Streaming SVD with Procrustes-stabilized output and optional whitening.

    Maintains eigenbasis (V, Λ) via momentum-blended, eigenvalue-scaled SVD
    updates. Procrustes alignment ensures output coefficients have stable
    meaning across updates — critical when feeding into a neural network.

    Optionally wraps a whitener (e.g. DiagonalWhitener). When a whitener is
    provided, update(x, signal) computes noise = x - signal, updates the
    whitener from the noise, then whitens x before the SVD update. forward()
    applies whitening at inference without updating statistics.

    Update (when buffer full):
        U = [ √(1-α) · diag(√Λ_old) · V_old ;  √(α/M) · X_white ]
        SVD(U) → V_new, Λ_new = S²
        Procrustes(V_new, R_old @ V_old) → R_new

    forward(x) → stable k-dim coefficients:
        1. c = X_white @ V.T        (project onto eigenbasis)
        2. c *= λ/(λ+1)             (Wiener filter, diagonal)
        3. c /= √λ                  (normalize to ~unit variance)
        4. c_out = c @ R.T          (rotate to stable frame)
    """

    def __init__(
        self,
        n_components: int = 10,
        buffer_size: Optional[int] = None,
        momentum: float = 0.1,
        shrinkage: bool = True,
        whitener=None,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.buffer_size = buffer_size if buffer_size is not None else 4 * n_components
        self.momentum = momentum
        self.shrinkage = shrinkage
        self.whitener = whitener

        self.buffer: List[torch.Tensor] = []
        self.buffer_counter: int = 0

        self.components: Optional[torch.Tensor] = None   # (k, D)
        self.eigenvalues: Optional[torch.Tensor] = None  # (k,)
        self._R: Optional[torch.Tensor] = None           # (k, k)

    def update(self, x: torch.Tensor, signal: Optional[torch.Tensor] = None) -> None:
        """Accumulate a batch; trigger SVD update when buffer is full.

        Args:
            x: Input data, shape (batch_size, D).
            signal: True signal estimate, same shape as x. If provided and a
                    whitener is attached, noise = x - signal is used to update
                    the whitener before whitening x.
        """
        if self.whitener is not None and signal is not None:
            self.whitener.update((x - signal).detach())

        x_white = self.whitener(x) if self.whitener is not None else x

        self.buffer.append(x_white)
        self.buffer_counter += x_white.shape[0]

        if self.buffer_counter >= self.buffer_size:
            self._svd_update()
            self.buffer = []
            self.buffer_counter = 0

    def _svd_update(self) -> None:
        X = torch.cat(self.buffer, dim=0)
        M = X.shape[0]
        alpha = self.momentum

        if self.components is None:
            K = X @ X.T / M
            eigvals, eigvecs = torch.linalg.eigh(K)
            idx = torch.argsort(eigvals, descending=True)[: self.n_components]
            Q = eigvecs[:, idx]
            Λ = eigvals[idx]
            V = (X.T @ Q) / torch.sqrt(M * Λ.clamp(min=1e-12))
            self.components = V.T
            self.eigenvalues = Λ
            self._R = torch.eye(self.n_components, dtype=V.dtype, device=V.device)
        else:
            Λ_sqrt = torch.sqrt(self.eigenvalues.clamp(min=1e-12))
            scaled_old = np.sqrt(1 - alpha) * (Λ_sqrt.unsqueeze(1) * self.components)
            scaled_new = np.sqrt(alpha / M) * X
            U = torch.cat([scaled_old, scaled_new], dim=0)
            _, S, Vt = torch.linalg.svd(U, full_matrices=False)
            V_new = Vt[: self.n_components]
            Λ_new = S[: self.n_components] ** 2

            V_stable_old = self._R @ self.components
            C = V_stable_old @ V_new.T
            U2, _, Wt = torch.linalg.svd(C)
            R_new = U2 @ Wt

            self.components = V_new
            self.eigenvalues = Λ_new
            self._R = R_new

    def forward(self, x: torch.Tensor, signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project to stable k-dimensional coefficients.

        When in training mode, automatically accumulates x into the buffer and
        triggers an SVD update when the buffer is full. This makes DynamicSVD
        usable as a drop-in nn.Module embedding without a separate update() call.

        Args:
            x: Input data, shape (batch_size, D).
            signal: If provided and a whitener is attached, used to estimate
                    noise for whitener updates (passed through to update()).

        Returns:
            Coefficients of shape (batch_size, k), ~unit variance. Returns
            zeros before the first SVD update.
        """
        if self.training:
            with torch.no_grad():
                self.update(x.detach(), signal)

        if self.components is None:
            return torch.randn(x.shape[0], self.n_components, dtype=x.dtype, device=x.device)

        x_white = self.whitener(x) if self.whitener is not None else x
        c = x_white @ self.components.T

        if self.shrinkage and self.eigenvalues is not None:
            Λ = self.eigenvalues.clamp(min=1e-12)
            c = c * (Λ / (Λ + 1.0) / torch.sqrt(Λ)).unsqueeze(0)

        return c @ self._R.T

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Wiener-filter and reconstruct in whitened D-dimensional space."""
        if self.components is None:
            raise ValueError("Call update() enough times before reconstruct().")

        x_white = self.whitener(x) if self.whitener is not None else x
        x_proj = x_white @ self.components.T

        if self.shrinkage:
            shrink = self.eigenvalues / (self.eigenvalues + 1.0)
            x_proj = x_proj * shrink.unsqueeze(0)

        return x_proj @ self.components


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
