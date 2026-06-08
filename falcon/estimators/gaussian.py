"""Gaussian posterior estimation — deprecated factory wrapper.

Use ``falcon.estimators.GaussianFullCov`` directly instead.
"""

import warnings
from typing import List, Optional


def Gaussian(
    simulator_instance,
    theta_key: Optional[str] = None,
    condition_keys: Optional[List[str]] = None,
):
    """Create a GaussianFullCov estimator (deprecated factory).

    .. deprecated::
        Use :class:`falcon.estimators.GaussianFullCov` directly.
    """
    warnings.warn(
        "falcon.estimators.Gaussian is deprecated; use GaussianFullCov directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    from falcon.estimators.gaussian_fullcov import GaussianFullCov
    est = GaussianFullCov()
    est.setup(simulator_instance, theta_key, condition_keys)
    return est
