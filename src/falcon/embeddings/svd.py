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
            random noise before the first SVD update (avoids all-zero gradients).
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

    def get_extra_state(self):
        return {
            'components': self.components,
            'eigenvalues': self.eigenvalues,
            '_R': self._R,
        }

    def set_extra_state(self, state):
        self.components = state['components']
        self.eigenvalues = state['eigenvalues']
        self._R = state['_R']

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
