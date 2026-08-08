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
    provided, update(x, signal) computes noise = x - signal and updates the
    whitener from it, so the whitener normalizes the *noise* to unit variance.
    forward() applies whitening at inference without updating statistics.

    Inputs with more than two dimensions are flattened to (batch_size, D)
    internally, so image-shaped data can be fed in directly.

    The `signal` stream (a training-only scaffold; see falcon graph configs)
    has two independent uses:
      - it is always the noise estimate for the whitener, as above;
      - with fit_on_signal=True it is also the stream the eigenbasis is fitted
        from, while projections in forward() stay on x.

    Update (when buffer full):
        U = [ √(1-α) · diag(√Λ_old) · V_old ;  √(α/M) · X_white ]
        SVD(U) → V_new, Λ_new = S²
        Procrustes(V_new, R_old @ V_old) → R_new

    forward(x) → stable k-dim coefficients:
        1. c = X_white @ V.T        (project onto eigenbasis)
        2. c *= λ/(λ+1)             (Wiener filter, diagonal)
        3. c /= √λ                  (normalize to ~unit variance)
        4. c_out = c @ R.T          (rotate to stable frame)

    Steps 2-3 are applied jointly, and only when shrinkage=True; with
    shrinkage=False the raw projection is returned.

    Note that step 2 is the textbook Wiener gain only when λ is *signal* power
    measured in noise units, which requires both a whitener and
    fit_on_signal=True.  Fitting on x instead gives λ ≈ λ_signal + 1, so the
    gain becomes (λ_s+1)/(λ_s+2) >= 1/2 and a noise-only direction is passed
    through at half amplitude rather than suppressed.  Without a whitener at
    all, the `+ 1` is in raw data units and the threshold is arbitrary.
    """

    def __init__(
        self,
        n_components: int = 10,
        buffer_size: Optional[int] = None,
        momentum: float = 0.1,
        shrinkage: bool = True,
        whitener=None,
        fit_on_signal: bool = False,
    ) -> None:
        super().__init__()
        self.fit_on_signal = fit_on_signal
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
            x: Input data, shape (batch_size, D) or (batch_size, ...).
            signal: True signal estimate, same shape as x. If provided and a
                    whitener is attached, noise = x - signal is used to update
                    the whitener. With fit_on_signal=True it is also what the
                    eigenbasis is fitted from, in place of x.
        """
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        if signal is not None and signal.dim() > 2:
            signal = signal.flatten(start_dim=1)
        if self.whitener is not None and signal is not None:
            self.whitener.update((x - signal).detach())

        # fit_on_signal: the eigenbasis is fitted from the noise-free ``signal``
        # stream (a training-only scaffold); projections in forward() remain on
        # ``x``. Removes noise-contaminated components (empirical eigenvalues up
        # to the Marchenko-Pastur edge (1+sqrt(D/N))^2 sigma^2 masquerade as
        # structure when fitting on noisy x). With shrinkage=True the trailing
        # near-zero clean eigenvalues are damped, so the 1/sqrt(lambda)
        # normalization cannot amplify observation noise.
        fit_x = signal if (self.fit_on_signal and signal is not None) else x
        x_white = self.whitener(fit_x) if self.whitener is not None else fit_x

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
            signal: Passed through to update() during training, where it feeds
                    the whitener's noise estimate and, with fit_on_signal=True,
                    the eigenbasis fit.  Never used for the projection itself,
                    so it is not needed at inference.

        Returns:
            Coefficients of shape (batch_size, k), ~unit variance. Returns
            random noise before the first SVD update (avoids all-zero gradients).
        """
        if self.training:
            with torch.no_grad():
                self.update(x.detach(), signal)
        if x.dim() > 2:
            x = x.flatten(start_dim=1)

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

        if x.dim() > 2:
            x = x.flatten(start_dim=1)

        x_white = self.whitener(x) if self.whitener is not None else x
        x_proj = x_white @ self.components.T

        if self.shrinkage:
            shrink = self.eigenvalues / (self.eigenvalues + 1.0)
            x_proj = x_proj * shrink.unsqueeze(0)

        return x_proj @ self.components
