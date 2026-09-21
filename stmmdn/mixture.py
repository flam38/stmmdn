"""Constrained two-component multivariate Student's t mixture output layer.

This is the distributional head of the model. It maps a flat network output to
the parameters of a two-component multivariate Student's t mixture, applying the
constraints that keep every scale matrix positive definite and, optionally,
order the two components' market-proxy variances.

Extracted from the research notebooks that produced the paper's results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import log_t_dist

__all__ = ["TTMOutputLayer", "output_extraction_TTM", "chunks_for"]


def chunks_for(n_assets):
    """Number of scalar outputs the network must emit for ``n_assets`` series.

    The layout is, in order:

    ``[mu1 (n), L100 (1), L1_diag (n-1), LT1 (n(n-1)/2),
       mu2 (n), L200 (1), L2_diag (n-1), LT2 (n(n-1)/2), w1 (1)]``

    which totals ``4n + n(n-1) + 1``.
    """
    low_triang = n_assets * (n_assets - 1) // 2
    return int(2 * (n_assets + n_assets + low_triang) + 1)


class TTMOutputLayer(nn.Module):
    """Constrained two-component multivariate Student's t mixture head.

    Parameters
    ----------
    n_assets : int
        Model dimension, written ``N`` in the paper. This is the market-proxy
        series plus the tradable assets, so the investable universe has
        ``n_assets - 1`` members.
    s_ref : float
        Reference scale for the market proxy, written ``s_ref`` in the paper:
        the sample standard deviation of the central 95 per cent of training
        market-proxy returns, computed with ``ddof=1``. It is estimated on the
        training split only and held fixed, so it carries no look-ahead.
    pi_1 : float, optional
        Fixed mixing probability of the first component. Default 0.95,
        the value used throughout the paper. The second component takes
        ``1 - pi_1``.
    market_band : bool, optional
        Apply the ordering band to the market-proxy entries. Default
        ``False``, the fleet configuration ruled on 15 September 2026. ``True``
        reproduces the head of the November 2025 fleet.
    market_scale, f_mkt, diag_scale, floors, mean_anchor, mu_bar, offdiag_scale, s_vec : optional
        Head variants tested in the September 2026 experiments and kept so
        those runs can be reproduced. All are off by default; see Notes.

    Notes
    -----
    **Why the factor is upper triangular.** Each inverse scale matrix is built
    as ``Sigma^{-1} = L L^T`` with ``L`` *upper* triangular. Under that
    orientation ``(Sigma)_11 = l_11^{-2}`` exactly, so constraining ``l_11``
    constrains the market proxy's marginal scale directly. With a
    lower-triangular factor the identity does not hold, because ``(Sigma)_11``
    would also depend on the off-diagonal entries.

    **The market entries.** With ``market_band=False`` (the default) both
    leading diagonal entries are ``softplus(raw)``, exactly like every other
    diagonal entry. The head then guarantees positive definiteness but not the
    ordering of the two components; which component carries the larger
    market-proxy variance is learned.

    With ``market_band=True`` the two leading diagonal entries are

    ``l_1,11 in [1/(1.5 s_ref), 1/(0.5 s_ref)]``  and
    ``l_2,11 = 1 / (s_ref (1.5 + softplus(raw)))``

    which gives ``(Sigma_1)_11 in [0.25 s_ref^2, 2.25 s_ref^2]`` and
    ``(Sigma_2)_11 > 2.25 s_ref^2``. The second component then always has the
    larger market-proxy variance, at every time step and for every parameter
    value. The band guarantees the ordering, not a minimum separation: as the
    softplus term approaches zero the two variances approach one another.

    **Experimental variants (arm record).** These reproduce the head switches
    of the September 2026 experiment notebooks and need ``s_vec``, the
    per-asset reference scales in training units:

    * ``market_scale``: ``l_11 = 1 / (s_1 (f_mkt + softplus(raw)))`` in both
      components; used only when ``market_band`` is off;
    * ``diag_scale``: ``l_jj = 1 / (s_j (f_j + softplus(raw)))`` for the
      non-market diagonals, with innovation floors ``floors[1:]``;
    * ``mean_anchor``: ``mu_j = mu_bar_j + s_j * raw``, so the means start at
      the training mean ``mu_bar``;
    * ``offdiag_scale``: each strict upper-triangular entry divided by
      ``s_row``.
    """

    def __init__(
        self,
        n_assets,
        s_ref,
        pi_1=0.95,
        market_band=False,
        market_scale=False,
        f_mkt=0.0,
        diag_scale=False,
        floors=None,
        mean_anchor=False,
        mu_bar=None,
        offdiag_scale=False,
        s_vec=None,
    ):
        super(TTMOutputLayer, self).__init__()
        self.p = int(n_assets)
        self.diag = self.p
        self.low_triang = self.p * (self.p - 1) // 2
        self.chunks = chunks_for(self.p)
        self.s_ref = float(s_ref)
        self.pi_1 = float(pi_1)

        self.market_band = bool(market_band)
        self.market_scale = bool(market_scale)
        self.f_mkt = float(f_mkt)
        self.diag_scale = bool(diag_scale)
        self.mean_anchor = bool(mean_anchor)
        self.offdiag_scale = bool(offdiag_scale)

        if (self.market_scale or self.diag_scale or self.mean_anchor or self.offdiag_scale) and s_vec is None:
            raise ValueError(
                "s_vec (per-asset reference scales) is required by market_scale, "
                "diag_scale, mean_anchor and offdiag_scale"
            )
        if self.mean_anchor and mu_bar is None:
            raise ValueError("mu_bar (training mean per asset) is required by mean_anchor")
        if self.diag_scale and floors is None:
            floors = [0.0] * self.p

        # Non-persistent buffers: they follow .to(device) but are not written to
        # the state dict, so checkpoints stay interchangeable across variants.
        self.register_buffer("s_vec", self._as_vector(s_vec, "s_vec"), persistent=False)
        self.register_buffer("floors", self._as_vector(floors, "floors"), persistent=False)
        self.register_buffer("mu_bar", self._as_vector(mu_bar, "mu_bar"), persistent=False)
        self.register_buffer(
            "lt_row_idx", torch.triu_indices(self.p, self.p, offset=1)[0], persistent=False
        )

    def _as_vector(self, values, name):
        if values is None:
            return None
        vector = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        if vector.numel() != self.p:
            raise ValueError(f"{name} must have {self.p} entries, got {vector.numel()}")
        return vector

    @property
    def guarantees_ordering(self):
        """Whether the head guarantees ``(Sigma_2)_11 > (Sigma_1)_11``."""
        return self.market_band

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def check_positive_diagonal(x):
        diagonal_matrix = torch.diag_embed(x)
        return torch.all(torch.diagonal(diagonal_matrix) > 0)

    @staticmethod
    def is_positive_definite(batch_of_matrices):
        if not torch.allclose(batch_of_matrices, batch_of_matrices.transpose(-1, -2)):
            print("Non symmetric Matrix.")
            return False
        eigenvalues = torch.linalg.eigvalsh(batch_of_matrices)
        positive_definite_mask = torch.all(eigenvalues > 0, dim=-1)
        if torch.all(positive_definite_mask):
            return True
        print("Not all matrices are positive definite.")
        return False

    # ------------------------------------------------------------------
    # constraint application
    # ------------------------------------------------------------------
    def _market_entry(self, raw, component, s_ref):
        """Leading diagonal entry ``l_g,11`` of the precision factor."""
        if self.market_band:
            if component == 1:
                # l_1,11 in [1/(1.5 s_ref), 1/(0.5 s_ref)]
                return 1 / (s_ref + 0.5 * (s_ref * torch.tanh(raw)))
            # l_2,11 = 1 / (s_ref (1.5 + softplus(raw)))
            return 1 / (s_ref * 1.5 + (s_ref * F.softplus(raw)))
        if self.market_scale:
            return 1.0 / (self.s_vec[0] * (self.f_mkt + F.softplus(raw)))
        return F.softplus(raw)

    def _diagonal(self, raw):
        """Non-market diagonal entries of the precision factor."""
        if self.diag_scale:
            return 1.0 / (self.s_vec[1:] * (self.floors[1:] + F.softplus(raw)))
        return F.softplus(raw)

    def _off_diagonal(self, raw):
        """Strict upper-triangular entries of the precision factor."""
        if self.offdiag_scale:
            return raw / self.s_vec[self.lt_row_idx]
        return raw

    def _means(self, raw):
        if self.mean_anchor:
            return self.mu_bar + self.s_vec * raw
        return raw

    def apply_constraints(self, x):
        """Apply the distributional constraints to a raw network output.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, chunks)
            Unconstrained network output.

        Returns
        -------
        torch.Tensor, shape (batch, chunks)
            Constrained parameter vector in the layout given by
            :func:`chunks_for`.

        Original name in the reference implementation:
        ``CustomLayer_TTM_Corr.TTM_dist_layer_corr2``.
        """
        p, low_triang, chunks = self.p, self.low_triang, self.chunks
        device = x.device
        num_dims = len(x.size())

        assert x.shape[-1] == chunks, f"Expected last dim {chunks}, got {x.shape[-1]}"
        s_ref = torch.tensor(self.s_ref, dtype=torch.float64, device=device)

        output = torch.chunk(x, chunks=chunks, dim=-1)

        # -------------------- component 1 ----------------------------------
        mu1 = torch.cat(output[0:p], dim=num_dims - 1)
        L100 = torch.cat(output[p:p + 1], dim=num_dims - 1)
        L1_diag = torch.cat(output[p + 1:2 * p], dim=num_dims - 1)
        LT1 = torch.cat(output[2 * p:int(2 * p + low_triang)], dim=num_dims - 1)
        # Strict UPPER triangle of the (upper-triangular) precision factor L; filled by triu_indices in mu_scale.

        # These diagonals live on the precision side, because the matrix formed
        # in mu_scale is Sigma^{-1} = L L^T.
        mu1 = self._means(mu1)
        L100 = self._market_entry(L100, component=1, s_ref=s_ref)
        L1_diag = self._diagonal(L1_diag)
        LT1 = self._off_diagonal(LT1)

        # -------------------- component 2 ------------------------------------
        mu2 = torch.cat(output[int(2 * p + low_triang):int(3 * p + low_triang)], dim=num_dims - 1)
        L200 = torch.cat(output[int(3 * p + low_triang):int(3 * p + low_triang + 1)], dim=num_dims - 1)
        L2_diag = torch.cat(
            output[int(3 * p + low_triang + 1):int(3 * p + low_triang + p)], dim=num_dims - 1
        )
        LT22 = torch.cat(output[int(4 * p + low_triang):int(4 * p + 2 * low_triang)], dim=num_dims - 1)

        mu2 = self._means(mu2)
        L200 = self._market_entry(L200, component=2, s_ref=s_ref)
        L2_diag = self._diagonal(L2_diag)
        LT22 = self._off_diagonal(LT22)

        # -------------------- mixing probability -----------------------------
        w1 = torch.cat(
            output[int(4 * p + 2 * low_triang):int(4 * p + 2 * low_triang + 1)],
            dim=num_dims - 1,
        )
        if w1.ndimension() == 0:
            w1 = w1.unsqueeze(0)
        # The mixing probability is fixed, not learned.
        w1 = torch.full_like(w1, self.pi_1, dtype=torch.float64).to(device)

        return torch.cat(
            [mu1, L100, L1_diag, LT1, mu2, L200, L2_diag, LT22, w1], dim=num_dims - 1
        )

    # ------------------------------------------------------------------
    # parsing
    # ------------------------------------------------------------------
    def mu_scale(self, output):
        """Parse a *constrained* parameter vector into distribution parameters.

        Returns ``(mu1, inv_scale1, mu2, inv_scale2, w1)`` where the scale
        matrices are **inverse** scale matrices, ``Sigma^{-1}``, ready to pass
        straight to :func:`~stmmdn.distributions.log_t_dist`.

        ``output`` must already have been through :meth:`apply_constraints`; do
        not apply Softplus or Tanh again here.

        To recover the Student's t scale, invert; to recover the covariance,
        multiply the scale by ``nu / (nu - 2)`` exactly once.
        """
        p, low_triang, chunks = self.p, self.low_triang, self.chunks
        device = output.device
        output = torch.chunk(output, chunks=chunks, dim=-1)

        # ----- component 1 -----
        mu1 = torch.cat(output[0:p], dim=-1).to(device)
        L100 = torch.cat(output[p:p + 1], dim=-1).to(device)
        D11 = torch.cat(output[p + 1:p + 1 + (p - 1)], dim=-1).to(device)

        lt1_start = p + 1 + (p - 1)
        LT11 = torch.cat(output[lt1_start:lt1_start + low_triang], dim=-1).to(device)

        batch_size = mu1.shape[0]
        L1 = torch.zeros(batch_size, p, p, dtype=torch.float64, device=device)

        # Fill strict upper-triangular entries (row<col); L is upper triangular so (Sigma)_11 = 1/l_11^2 exactly
        row_idx, col_idx = torch.triu_indices(p, p, offset=1, device=L1.device)
        L1[:, row_idx, col_idx] = LT11.to(torch.float64)

        diag_elements1 = torch.cat([L100, D11[:, 0:p - 1]], dim=1)
        L1 = L1 + torch.diag_embed(diag_elements1)

        # Sigma^{-1} = L L^T, symmetric positive definite by construction.
        inv_scale1 = torch.bmm(L1, L1.transpose(1, 2))

        # ----- component 2 -----
        base2 = lt1_start + low_triang

        mu2 = torch.cat(output[base2:base2 + p], dim=-1).to(device)
        L200 = torch.cat(output[base2 + p:base2 + p + 1], dim=-1).to(device)
        D22 = torch.cat(output[base2 + p + 1:base2 + p + 1 + (p - 1)], dim=-1).to(device)

        lt2_start = base2 + p + 1 + (p - 1)
        LT22 = torch.cat(output[lt2_start:lt2_start + low_triang], dim=-1).to(device)

        w1 = torch.cat(
            output[lt2_start + low_triang:lt2_start + low_triang + 1], dim=-1
        ).to(device)

        L2 = torch.zeros(batch_size, p, p, dtype=torch.float64, device=device)
        # Fill strict upper-triangular entries (row<col); L is upper triangular so (Sigma)_11 = 1/l_11^2 exactly
        row_idx2, col_idx2 = torch.triu_indices(p, p, offset=1, device=L2.device)
        L2[:, row_idx2, col_idx2] = LT22.to(torch.float64)

        diag_elements2 = torch.cat([L200, D22[:, 0:p - 1]], dim=1)
        L2 = L2 + torch.diag_embed(diag_elements2)

        inv_scale2 = torch.bmm(L2, L2.transpose(1, 2))

        return mu1, inv_scale1, mu2, inv_scale2, w1

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------
    def negative_log_likelihood(self, y_pred, y_true, nu_1, nu_2):
        """Mean negative log-likelihood of the two-component mixture.

        ``y_pred`` is the constrained parameter vector, ``y_true`` the observed
        return vectors.
        """
        mu1, inv_scale1, mu2, inv_scale2, w1 = self.mu_scale(y_pred)
        log_t_1 = log_t_dist(y_true, mu1, inv_scale1, nu_1)
        log_t_2 = log_t_dist(y_true, mu2, inv_scale2, nu_2)
        p1 = torch.exp(log_t_1)
        p2 = torch.exp(log_t_2)
        mixture_density = w1 * p1 + (1 - w1) * p2
        log_likelihood = torch.log(mixture_density)
        return -torch.mean(log_likelihood)


def output_extraction_TTM(layer, output):
    """Return means and Student's t **scale** matrices (not covariances).

    ``layer.mu_scale`` yields inverse scale matrices; this inverts them.
    The covariance is ``(nu / (nu - 2)) * scale``, which callers must apply
    exactly once.
    """
    mu1, inv_scale1, mu2, inv_scale2, w1 = layer.mu_scale(output)
    scale1 = torch.inverse(inv_scale1)
    scale2 = torch.inverse(inv_scale2)
    return mu1, scale1, mu2, scale2, w1
