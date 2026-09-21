"""Temporal attention and the dynamic covariance graph layer.

Extracted from the research notebooks that produced the paper's results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Attention", "DynamicCovarianceGNN", "GRAPH_NORMALISATIONS"]

GRAPH_NORMALISATIONS = ("inverse", "rsqrt")


class Attention(nn.Module):
    """Additive temporal attention over LSTM hidden states.

    Learns a single attention vector ``w_a`` and scores each hidden state by
    ``s_tau = h_tau^T w_a``, normalised across the sequence by a softmax. The
    context vector is the weighted sum of hidden states and has the same
    dimensionality as a hidden state.
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        nn.init.xavier_uniform_(self.att_weights)

    def forward(self, outputs):
        """
        Parameters
        ----------
        outputs : torch.Tensor, shape (batch, seq_len, hidden_size)

        Returns
        -------
        representations : torch.Tensor, shape (batch, hidden_size)
        att_weights : torch.Tensor, shape (batch, seq_len)
        """
        scores = torch.bmm(
            outputs,
            self.att_weights.permute(1, 0).unsqueeze(0).repeat(outputs.size(0), 1, 1),
        )
        # Softmax over the sequence so the weights sum to one per sample.
        att_weights = torch.softmax(scores.squeeze(2), dim=1)
        weighted = torch.mul(outputs, att_weights.unsqueeze(-1).expand_as(outputs))
        representations = weighted.sum(1)
        return representations, att_weights


class DynamicCovarianceGNN(nn.Module):
    """Graph layer whose adjacency matrix is a rolling return covariance.

    One propagation step followed by a shared linear map and a ReLU. The
    adjacency is the *signed* covariance matrix, symmetrically normalised by a
    degree matrix built from absolute row sums:

    ``S_tilde = D^{-1/2} S D^{-1/2}``,  ``D_ii = sum_j |S_ij|``.

    Two properties are deliberate and differ from a standard GCN:

    * edge weights keep their sign, so negative co-movement stays negative in
      message passing, while degrees use absolute values to stay positive;
    * no identity self-loops are added, because the covariance diagonal already
      carries each asset's own variance.

    The node set is fixed but the adjacency is supplied per observation, so the
    graph is time-varying.

    Parameters
    ----------
    input_dim, output_dim : int
    normalisation : {"inverse", "rsqrt"}, optional
        How ``D^{-1/2}`` is computed; the reference notebooks used two forms.
        ``"inverse"`` (default) is ``inv(sqrt(diag(max(d, 1e-9)) + 1e-9 I))``,
        as in the GNN-ALSTM notebooks. ``"rsqrt"`` is
        ``diag(rsqrt(max(d, 1e-8)))``, as in the November 2025 TPXC30 and DJIA
        GNN-LSTM Setup I notebooks and the DJIA 14Sep26 GNN-LSTM notebooks. The
        two agree to about ``1e-9 / d`` relative.

    Original name in the reference implementation: ``StaticGNN``. The name is a
    historical artefact; the adjacency varies by observation.
    """

    def __init__(self, input_dim, output_dim, normalisation="inverse"):
        super(DynamicCovarianceGNN, self).__init__()
        if normalisation not in GRAPH_NORMALISATIONS:
            raise ValueError(f"normalisation must be one of {GRAPH_NORMALISATIONS}, got {normalisation!r}")
        self.normalisation = normalisation
        self.linear = nn.Linear(input_dim, output_dim)

    def construct_degree_matrix(self, covariance_matrix):
        """Degree matrix from absolute covariance row sums.

        Parameters
        ----------
        covariance_matrix : torch.Tensor, shape (num_nodes, num_nodes)
        """
        if not isinstance(covariance_matrix, torch.Tensor):
            raise TypeError("covariance_matrix must be a PyTorch tensor")

        degree_values = torch.sum(torch.abs(covariance_matrix), dim=-1)
        # Guard against a vanishing degree.
        degree_values = torch.clamp(degree_values, min=1e-9)
        degree_matrix = torch.diag(degree_values)
        return degree_matrix

    def normalize_covariance(self, covariance_matrix):
        """Symmetric normalisation ``D^{-1/2} S D^{-1/2}`` with signed ``S``."""
        if self.normalisation == "rsqrt":
            deg = torch.sum(covariance_matrix.abs(), dim=-1)
            deg = torch.clamp(deg, min=1e-8)
            inv_sqrt_deg = torch.rsqrt(deg)
            D_inv_sqrt = torch.diag(inv_sqrt_deg)
            return D_inv_sqrt @ covariance_matrix @ D_inv_sqrt

        degree_matrix = self.construct_degree_matrix(covariance_matrix)
        inv_sqrt_degree = torch.linalg.inv(
            torch.sqrt(
                degree_matrix
                + torch.eye(degree_matrix.size(0), device=degree_matrix.device) * 1e-9
            )
        )
        normalized_covariance = inv_sqrt_degree @ covariance_matrix @ inv_sqrt_degree
        return normalized_covariance

    def forward(self, x, cov_matrix):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (num_nodes, input_dim)
            Node features, one row per asset.
        cov_matrix : torch.Tensor, shape (num_nodes, num_nodes)
            Rolling covariance for this observation.

        Returns
        -------
        torch.Tensor, shape (num_nodes, output_dim)
        """
        if not isinstance(cov_matrix, torch.Tensor):
            raise TypeError("cov_matrix must be a PyTorch tensor")

        normalized_cov_matrix = self.normalize_covariance(cov_matrix)
        x = torch.matmul(normalized_cov_matrix, x)  # propagate over the graph
        x = self.linear(x)
        x = F.relu(x)
        return x
