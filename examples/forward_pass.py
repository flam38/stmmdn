"""Forward pass of the four STMMDN architectures on synthetic data.

No market data is needed. The script builds each architecture, runs one batch
through it, and prints the shapes of the predicted mixture parameters.

Usage:  python examples/forward_pass.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stmmdn import FLEET_CONFIG, STMMDN, chunks_for, initialise, output_extraction_TTM  # noqa: E402

torch.manual_seed(0)
BATCH, SEQ_LEN, N_FEATURES, N_ASSETS, LOOKBACK = 8, 21, 12, 6, 21

x = torch.randn(BATCH, SEQ_LEN, N_FEATURES, dtype=torch.float64)            # temporal predictors
node_features = 0.01 * torch.randn(BATCH, N_ASSETS, LOOKBACK, dtype=torch.float64)  # last 21 daily returns per asset
centred = node_features - node_features.mean(dim=2, keepdim=True)
cov_matrix = centred @ centred.transpose(1, 2) / (LOOKBACK - 1)             # rolling covariance, the graph

ARCHITECTURES = {"GNN-ALSTM-TTM": (True, True), "GNN-LSTM-TTM": (True, False),
                 "ALSTM-TTM": (False, True), "LSTM-TTM": (False, False)}

for name, (use_graph, use_attention) in ARCHITECTURES.items():
    model = STMMDN(input_size=N_FEATURES, n_assets=N_ASSETS, s_ref=1.0, nu_1_initial=21.0, nu_2_initial=-1.1,  # nu_1 = 25, nu_2 = 5.25
                   use_graph=use_graph, use_attention=use_attention, **FLEET_CONFIG)
    model = initialise(model).eval()
    with torch.no_grad():
        out, attn = model(x, node_features, cov_matrix) if use_graph else model(x)
        mu1, scale1, mu2, scale2, w1 = output_extraction_TTM(model.custom_layer, out)
    assert out.shape == (BATCH, chunks_for(N_ASSETS))
    print("%-14s outputs %s | means %s | scale matrices %s | nu_1 %.2f, nu_2 %.2f"
          % (name, tuple(out.shape), tuple(mu1.shape), tuple(scale1.shape), float(model.nu_1), float(model.nu_2)))
