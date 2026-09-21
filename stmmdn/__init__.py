"""STMMDN: Spatial-Temporal Multivariate Mixture Density Networks.

Network architecture of the GNN-ALSTM-TTM model and its three ablations: an
(attention-)LSTM temporal encoder and, optionally, a rolling-covariance dynamic
graph neural network feeding a two-component multivariate Student's t mixture
density head.
"""

from ._version import __version__
from .distributions import log_t_dist
from .layers import Attention, DynamicCovarianceGNN
from .mixture import TTMOutputLayer, chunks_for, output_extraction_TTM
from .model import FLEET_CONFIG, NOV25_CONFIG, STMMDN, init_weights, initialise

__all__ = [
    "__version__",
    "log_t_dist",
    "Attention",
    "DynamicCovarianceGNN",
    "TTMOutputLayer",
    "chunks_for",
    "output_extraction_TTM",
    "STMMDN",
    "init_weights",
    "initialise",
    "FLEET_CONFIG",
    "NOV25_CONFIG",
]
