"""The STMMDN model: attention-LSTM and dynamic covariance GNN into a
constrained Student's t mixture head.

Extracted from the research notebooks that produced the paper's results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Attention, DynamicCovarianceGNN
from .mixture import TTMOutputLayer, chunks_for

__all__ = ["STMMDN", "init_weights", "initialise", "FLEET_CONFIG", "NOV25_CONFIG", "NU_2_MAPS"]

NU_2_MAPS = ("bounded", "softplus")

#: Configuration ruled for the September 2026 fleet (the defaults).
FLEET_CONFIG = {
    "market_band": False,
    "dropout": 0.2,
    "nu_2_map": "bounded",
    "nu_2_bounds": (4.0, 9.0),
}

#: Architecture and head of the November 2025 fleet, for reproducing those
#: models. Their training settings varied by notebook.
NOV25_CONFIG = {
    "market_band": True,
    "dropout": 0.0,
    "nu_2_map": "softplus",
    "nu_2_bounds": (4.0, 9.0),
}


class STMMDN(nn.Module):
    """Spatial-Temporal Multivariate Mixture Density Network.

    A temporal branch and, optionally, a spatial branch feed one distributional
    head:

    * the **temporal branch** is a stacked LSTM followed by a second LSTM and,
      optionally, additive attention over the hidden states;
    * the **spatial branch** is a graph layer whose adjacency is the rolling
      return covariance for that observation, mean-pooled over assets;
    * the **head** maps the encoding to the parameters of a two-component
      multivariate Student's t mixture.

    The four architectures compared in the paper are the four settings of the
    two flags, and each reproduces the layer layout of its reference notebook:

    =====================  =============  =================  ==========================================
    Paper name             ``use_graph``  ``use_attention``  Layers after the temporal encoder
    =====================  =============  =================  ==========================================
    ``GNN-ALSTM-TTM``      True           True               lstm_dense, gnn_dense, combined_dense1, combined_dense2
    ``GNN-LSTM-TTM``       True           False              lstm_dense, gnn_dense, combined_dense1, combined_dense2
    ``ALSTM-TTM``          False          True               dense1, dense2
    ``LSTM-TTM``           False          False              dense1, dense2
    =====================  =============  =================  ==========================================

    Parameters
    ----------
    input_size : int
        Number of temporal predictors per time step, after feature selection.
    n_assets : int
        Model dimension ``N``: the market proxy plus the tradable assets.
    s_ref : float
        Market-proxy reference scale; see :class:`~stmmdn.mixture.TTMOutputLayer`.
    nu_1_initial, nu_2_initial : float
        Initial values of the *unconstrained* degrees-of-freedom parameters.
        Obtain them by applying the inverse of the chosen maps (see ``nu_2_map``)
        to the target values of the degrees of freedom.
    hidden_size : int, optional
        Hidden width of the first LSTM. Default 48.
    num_layers : int, optional
        Layers in the first LSTM. Default 2, the value used for every published
        run.
    node_lookback : int, optional
        Length of the per-asset return history used as node features. Default
        21, matching the rolling covariance window.
    embed_dim : int, optional
        Width of the temporal and graph embeddings. Default 24.
    dense_dim : int, optional
        Width of the dense layers. Default 48.
    use_graph, use_attention : bool, optional
        Architecture switches; see the table above.
    pi_1 : float, optional
        Fixed mixing probability of the first component. Default 0.95.
    dropout : float, optional
        Dropout rate. Default 0.2, applied in three places: between the stacked
        layers of the first LSTM, on the temporal output (last hidden state or
        attention output) before the first dense layer, and after the ReLU of the
        first dense layer of the head path (``dense1`` or ``combined_dense1``).
        The graph branch and the mixture head are not dropped. ``0.0`` disables
        dropout and consumes no random numbers.
    market_band : bool, optional
        Ordering band on the market-proxy entries; see
        :class:`~stmmdn.mixture.TTMOutputLayer`. Default ``False``.
    nu_2_map : {"bounded", "softplus"}, optional
        Map from the raw parameter to ``nu_2``. ``"bounded"`` (default) is
        ``low + (high - low) * sigmoid(raw)`` with ``nu_2_bounds``;
        ``"softplus"`` is ``4 + softplus(raw)``, as for ``nu_1``.
    nu_2_bounds : tuple of float, optional
        ``(low, high)`` for the bounded map. Default ``(4.0, 9.0)``.
    graph_normalisation : {"inverse", "rsqrt"}, optional
        How the graph layer computes ``D^{-1/2}``; see
        :class:`~stmmdn.layers.DynamicCovarianceGNN`. Default ``"inverse"``.
    head_options : dict, optional
        Extra keyword arguments for :class:`~stmmdn.mixture.TTMOutputLayer`,
        used only to reproduce the experimental head variants.

    Notes
    -----
    The presets :data:`FLEET_CONFIG` (the defaults) and :data:`NOV25_CONFIG`
    can be unpacked into the constructor, e.g. ``STMMDN(..., **NOV25_CONFIG)``.

    Submodules are registered in the same order as in the reference notebooks.
    The same seed therefore gives the notebooks' initial weights when the model
    is built the same way: construct, ``.to(device)``, then
    :func:`initialise` with the notebook's number of passes. Most graph
    notebooks call ``apply(init_weights)`` and ``double()`` twice; the non-graph
    notebooks call them once. The count differs between notebooks.
    """

    def __init__(
        self,
        input_size,
        n_assets,
        s_ref,
        nu_1_initial,
        nu_2_initial,
        hidden_size=48,
        num_layers=2,
        node_lookback=21,
        embed_dim=24,
        dense_dim=48,
        use_graph=True,
        use_attention=True,
        pi_1=0.95,
        dropout=0.2,
        market_band=False,
        nu_2_map="bounded",
        nu_2_bounds=(4.0, 9.0),
        graph_normalisation="inverse",
        head_options=None,
    ):
        super(STMMDN, self).__init__()
        if nu_2_map not in NU_2_MAPS:
            raise ValueError(f"nu_2_map must be one of {NU_2_MAPS}, got {nu_2_map!r}")
        low, high = (float(b) for b in nu_2_bounds)
        if not low < high:
            raise ValueError("nu_2_bounds must satisfy low < high")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_graph = use_graph
        self.use_attention = use_attention
        self.n_assets = int(n_assets)
        self.chunks = chunks_for(self.n_assets)
        self.dropout_rate = float(dropout)
        self.nu_2_map = nu_2_map
        self.nu_2_bounds = (low, high)

        # Degrees of freedom are global learnable parameters, shared over t.
        self.raw_nu_1 = nn.Parameter(torch.tensor([nu_1_initial], dtype=torch.float64))
        self.raw_nu_2 = nn.Parameter(torch.tensor([nu_2_initial], dtype=torch.float64))

        # ---- temporal branch ----
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # nn.LSTM applies dropout between stacked layers only.
            dropout=self.dropout_rate if num_layers > 1 else 0.0,
        )
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=embed_dim, batch_first=True)
        if use_attention:
            self.attention = Attention(hidden_size=embed_dim)

        # ---- spatial branch and head path (registration order as in the notebooks) ----
        if use_graph:
            self.static_gnn = DynamicCovarianceGNN(
                input_dim=node_lookback, output_dim=embed_dim, normalisation=graph_normalisation
            )
            self.lstm_dense = nn.Linear(embed_dim, dense_dim)
            self.gnn_dense = nn.Linear(embed_dim, dense_dim)
            self.combined_dense1 = nn.Linear(2 * dense_dim, dense_dim)
            self.combined_dense2 = nn.Linear(dense_dim, self.chunks)
        else:
            self.dense1 = nn.Linear(embed_dim, dense_dim)
            self.dense2 = nn.Linear(dense_dim, self.chunks)
        self.dropout = nn.Dropout(self.dropout_rate)

        head_options = dict(head_options or {})
        self.custom_layer = TTMOutputLayer(
            n_assets=n_assets, s_ref=s_ref, pi_1=pi_1, market_band=market_band, **head_options
        )

    def forward(self, x, node_features=None, cov_matrix=None):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)
            Temporal predictors.
        node_features : torch.Tensor, shape (batch, n_assets, node_lookback)
            Per-asset return histories. Required when ``use_graph``.
        cov_matrix : torch.Tensor, shape (batch, n_assets, n_assets)
            Rolling covariance per observation. Required when ``use_graph``.

        Returns
        -------
        distribution_outputs : torch.Tensor, shape (batch, chunks)
            Constrained mixture parameters.
        attn_weights : torch.Tensor or None
            Attention weights, shape (batch, seq_len), or ``None`` when
            attention is disabled.
        """
        batch_size = x.size(0)
        device = x.device

        # ---- temporal branch ----
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=x.dtype)
        lstm_out, _ = self.lstm1(x, (h0, c0))
        lstm_out, _ = self.lstm2(lstm_out)

        if self.use_attention:
            temporal_out, attn_weights = self.attention(lstm_out)
        else:
            # Without attention the encoder uses the final hidden state.
            temporal_out, attn_weights = lstm_out[:, -1, :], None
        temporal_out = self.dropout(temporal_out)

        if self.use_graph:
            if node_features is None or cov_matrix is None:
                raise ValueError("node_features and cov_matrix are required when use_graph=True")
            lstm_dense_out = F.relu(self.lstm_dense(temporal_out))

            # ---- spatial branch ----
            gnn_out = []
            for i in range(batch_size):
                gnn_out_i = self.static_gnn(node_features[i], cov_matrix[i])
                gnn_out.append(gnn_out_i.mean(dim=0))  # mean-pool over assets
            gnn_out = torch.stack(gnn_out, dim=0)
            gnn_dense_out = F.relu(self.gnn_dense(gnn_out))

            combined_features = torch.cat((lstm_dense_out, gnn_dense_out), dim=-1)
            hidden = F.relu(self.combined_dense1(combined_features))
            hidden = self.dropout(hidden)
            raw_parameters = self.combined_dense2(hidden)
        else:
            hidden = F.relu(self.dense1(temporal_out))
            hidden = self.dropout(hidden)
            raw_parameters = self.dense2(hidden)

        distribution_outputs = self.custom_layer.apply_constraints(raw_parameters)
        return distribution_outputs, attn_weights

    @property
    def nu_1(self):
        """Degrees of freedom of the first component, ``> 4`` by construction."""
        return 4 + F.softplus(self.raw_nu_1)

    @property
    def nu_2(self):
        """Degrees of freedom of the second, heavier-tailed component.

        In ``nu_2_bounds`` under the bounded map, ``> 4`` under the softplus map.
        """
        if self.nu_2_map == "bounded":
            low, high = self.nu_2_bounds
            return low + (high - low) * torch.sigmoid(self.raw_nu_2)
        return 4 + F.softplus(self.raw_nu_2)

    def load_reference_state_dict(self, state_dict, strict=True):
        """Load a checkpoint saved by a reference notebook.

        Two differences are handled:

        * the LSTM and ALSTM notebooks name the degrees-of-freedom parameters
          ``raw_nu1`` and ``raw_nu2``; the package uses ``raw_nu_1`` and
          ``raw_nu_2``;
        * the ALSTM notebook class also registers two unused parameters,
          ``nu_1`` and ``nu_2`` (its forward pass computes the degrees of
          freedom from ``raw_nu1`` and ``raw_nu2``); they are dropped.

        Every other key matches. Build the model with the configuration the
        checkpoint was trained with.
        """
        renames = {"raw_nu1": "raw_nu_1", "raw_nu2": "raw_nu_2"}
        unused = {"nu_1", "nu_2"}
        state = {renames.get(k, k): v for k, v in state_dict.items() if k not in unused}
        return self.load_state_dict(state, strict=strict)


def initialise(model, passes=1):
    """Initialise weights as the reference drivers do and return the model.

    Each pass is ``model.apply(init_weights)`` followed by ``model.double()``.
    Every pass draws fresh random numbers, so the number of passes changes the
    initial weights. Most graph notebooks use two passes and the non-graph
    notebooks one. Call it
    after ``.to(device)``.
    """
    for _ in range(int(passes)):
        model.apply(init_weights)
        model.double()
    return model


def init_weights(m):
    """Xavier for linear and input-hidden weights, orthogonal for hidden-hidden.

    LSTM biases are set to 0.01 with the forget-gate block set to 1.0.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.01)
                n = param.size(0)
                param[n // 4:n // 2].data.fill_(1.0)  # forget-gate bias
