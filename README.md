# STMMDN: Spatial-Temporal Multivariate Mixture Density Networks

Network architecture of the **GNN-ALSTM-TTM** model and its three ablations, from

> Lam, F., Chan, J. S. K., Choy, S. T. B., and Gao, J.
> *Spatial-Temporal Multivariate Mixture Density Networks: From Distributional
> Forecasting to Portfolio Strategy Construction.*

This repository contains the **network architecture**: the temporal encoder, the
dynamic graph layer, and the mixture density output layer. It is published so
that the architecture can be read, cited and re-implemented. It does not contain
market data, trained models, the training pipeline, or the portfolio strategies
of the paper.

## The model

STMMDN forecasts the joint distribution of $k$-day forward asset returns. Its
components are:

- **Temporal branch.** A stacked LSTM, optionally with additive attention over
  the hidden states, encoding a window of market predictors.
- **Spatial branch (optional).** A graph layer whose adjacency matrix is the
  rolling return covariance for that observation, mean-pooled over assets:
  - edge weights keep their sign;
  - degrees are built from absolute row sums;
  - no identity self-loops are added, because the covariance diagonal already
    carries each asset's own variance.
- **Output layer.** A **two-component multivariate Student's $t$ mixture**, which
  emits time-varying component means and scale matrices, a fixed mixing weight
  (0.95 on the first component) and global learnable degrees of freedom.

The four architectures the paper compares are the four settings of two flags:

| Paper name | `use_graph` | `use_attention` |
|---|---|---|
| `GNN-ALSTM-TTM` | True | True |
| `GNN-LSTM-TTM` | True | False |
| `ALSTM-TTM` | False | True |
| `LSTM-TTM` | False | False |

**Degrees of freedom.** $\nu_1 = 4 + \mathrm{softplus}(\tilde\nu_1)$ and, in the
default configuration, $\nu_2 = 4 + 5\,\sigma(\tilde\nu_2) \in (4, 9)$, so both
exceed 4 by construction.

**Scale matrices, not covariances.** Each inverse scale matrix is built as
$\Sigma^{-1} = LL^{\top}$ with $L$ upper triangular. `output_extraction_TTM`
returns the Student's $t$ scale matrices; the covariance of component $g$ is
$\frac{\nu_g}{\nu_g - 2}\,\Sigma_g$.

**Two configurations.** `FLEET_CONFIG` holds the settings used for the results
of the paper (dropout 0.2, $\nu_2$ bounded in $(4, 9)$, no band on the market
proxy). `NOV25_CONFIG` holds the settings of an earlier version of the models.

## Files

| File | Content |
|---|---|
| `stmmdn/model.py` | `STMMDN`, the network; `initialise`, `init_weights` |
| `stmmdn/layers.py` | `Attention`, `DynamicCovarianceGNN` |
| `stmmdn/mixture.py` | `TTMOutputLayer`, the mixture density output layer; `output_extraction_TTM` |
| `stmmdn/distributions.py` | `log_t_dist`, the multivariate Student's $t$ log-density |
| `examples/forward_pass.py` | a forward pass of the four architectures on synthetic data |

## Quick start

```
pip install -r requirements.txt
python examples/forward_pass.py
```

```python
import torch
from stmmdn import STMMDN, FLEET_CONFIG, initialise, output_extraction_TTM

model = STMMDN(input_size=12, n_assets=6, s_ref=1.0,
               nu_1_initial=21.0, nu_2_initial=-1.1,
               use_graph=True, use_attention=True, **FLEET_CONFIG)
model = initialise(model)                        # double precision, as in the paper

out, attention = model(x, node_features, cov_matrix)
mu1, scale1, mu2, scale2, w1 = output_extraction_TTM(model.custom_layer, out)
```

Inputs: `x` has shape `(batch, seq_len, input_size)`, `node_features` has shape
`(batch, n_assets, 21)` and holds the last 21 daily returns of each asset, and
`cov_matrix` has shape `(batch, n_assets, n_assets)` and holds the rolling
covariance of those returns. The first series is the market proxy.

## Data

No market data is distributed here. The data of the study was retrieved from a
Bloomberg Terminal and cannot be redistributed.

## Citation

See `CITATION.cff`.

## Licence

See `LICENSE`.
