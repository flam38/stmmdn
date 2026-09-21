"""Multivariate Student's t log-density.

Extracted verbatim (modulo the notes below) from the reference implementation.
Extracted from the research notebooks that produced the paper's results.
"""

import torch

__all__ = ["log_t_dist"]


def log_t_dist(x, mu, inv_scale, nu):
    """Multivariate Student's t log-pdf, parameterised by the inverse scale.

    This evaluates the log-density of a ``d``-dimensional Student's t
    distribution with degrees of freedom ``nu``, location ``mu`` and scale
    matrix ``Sigma``, where the supplied ``inv_scale`` is ``Sigma^{-1}``.

    Note that ``inv_scale`` is *not* the inverse of the Student's t covariance.
    For ``nu > 2`` the covariance is ``Cov = (nu / (nu - 2)) * Sigma``, so
    ``Sigma^{-1} = ((nu - 2) / nu) * Cov^{-1}``. Downstream code that needs a
    covariance must apply the ``nu / (nu - 2)`` factor exactly once.

    Parameters
    ----------
    x : torch.Tensor, shape (B, d)
        Observations at which the log-density is evaluated.
    mu : torch.Tensor, shape (B, d)
        Location vector.
    inv_scale : torch.Tensor, shape (B, d, d)
        Precision of the scale matrix, ``Sigma^{-1}``. Must be symmetric
        positive definite.
    nu : float or torch.Tensor
        Degrees of freedom, ``nu > 0``.

    Returns
    -------
    torch.Tensor, shape (B, 1)
        Log probability density at each observation.

    Raises
    ------
    ValueError
        If ``inv_scale`` is not positive definite. Callers in the training loop
        treat this like a non-finite loss and skip the batch.
    """
    d = x.shape[1]

    # Centred observations (x - mu).
    x_mu = x - mu

    # Mahalanobis distance Q = (x - mu)^T Sigma^{-1} (x - mu), shape (B, 1).
    mahalanobis = torch.bmm(
        x_mu.unsqueeze(-1).transpose(1, 2),
        torch.bmm(inv_scale, x_mu.unsqueeze(-1)),
    ).squeeze(-1)

    # log|Sigma| = -log|Sigma^{-1}|; slogdet is used for numerical stability
    # and doubles as a positive-definiteness check.
    sign, logdet_prec = torch.slogdet(inv_scale)
    if not torch.all(sign > 0):
        raise ValueError("inv_scale must be positive definite.")

    log_det_scale = (-logdet_prec).unsqueeze(-1)

    # log Gamma((nu + d)/2) - log Gamma(nu/2)
    log_gamma_part = torch.lgamma((nu + d) / 2) - torch.lgamma(nu / 2)

    # log p(x) = log Gamma((nu + d)/2) - log Gamma(nu/2)
    #            - 0.5 * [ d log(nu pi) + log|Sigma| ]
    #            - 0.5 * (nu + d) log(1 + Q / nu)
    log_prob_density = (
        log_gamma_part
        - 0.5 * log_det_scale
        - 0.5 * d * torch.log(nu * torch.pi)
        - 0.5 * (nu + d) * torch.log1p(mahalanobis / nu)
    )

    if torch.any(torch.isnan(log_prob_density)):
        print("NaN detected in log_t_dist!")

    return log_prob_density
