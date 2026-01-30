## `update_stats`

* Should be based on updating covariance matrix and mean with EMA
* Cholesky does not need to be tracked, since we need eigenvalue-based corrections anyway
* Eigenvalue decomposition should be done and stored, with `eig_freq = 1` by default, but adjustable to do only every `eig_freq` steps (but use better argument name)
* Drop support for diagonal statistics, assume that always `full_cov = True`

## `log_prob`

* `loss` and `log_prob` should be collapsed in single function, if possible, ideally just `log_prob`
* Focus on `full_cov = True` here and elsewhere
* Change to be based on eigenvalue decomposition as provided and stored by by `update_stats`

## `sample_proposal`

* Generalise this to work properly in a Bayesian setting for when the likelihood function is tempered, not the posterior
* Concretely, we want samples from `p_prop(z) \propto p(x_obs | z)^gamma p_prior(z)`
* We can estimate `p(x_obs | z) \propto q(z | x_obs) / p(z)`, where q is the learned posterior
* This implies that the precision matrix of `p(x_obs | z)` is simply the precision matrix of q minus the identity matrix. Calculate this based on the eigenvalue decomposition of the learned posterior (stored by `update_stats`), and use that the identity matrix is trivial in this contxt.  Truncate eigenvalues of the likelihood precision matrix at 0 from below. Essentially we should have `lambda_i^like = max(lambda_i^post - 1, 0)` or something like that.
* Tempering the likelihood means to rescale the eigenvalues of the precision matrix appropriately
* For the precision matrix of the proposal, we need to multiply with the prior again, which adds 1 to the precision matrix of the likelihood
* The proposal mean will be also affected by the tempering, and derived from the mean values of the posterior, after rotating them into eigenvalue space appropriately. Figure out the math. For directions with zero precision, the mean values should be zero. For other directions, the mean value should be appropriately rescaled, such that in the limit gamma -> 0 the mean goes back to zero.  Based on Bayes theorem.


