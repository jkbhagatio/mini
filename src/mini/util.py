"""Utility functions."""

from einops import reduce, repeat
from jaxtyping import Float
from torch import Tensor


def vec_r2(y_pred: Float[Tensor, "n inst feat"], y_true: Float[Tensor, "n feat"]) -> (
    Float[Tensor, "n"]
):
    """Calculates vectorized R² scores for each example in a batch."""
    # Calculate SST
    y_true_mean = reduce(y_true, "n feat -> n", "mean")
    ss_tot = reduce((y_true - y_true_mean.unsqueeze(-1)) ** 2, "n feat -> n", "sum")
    # Calculate SSR
    y_true = repeat(y_true, "n feat -> n inst feat", inst=y_pred.shape[1])  # match to broadcast
    ss_res = reduce((y_true - y_pred) ** 2, "n inst feat -> n inst", "sum")
    
    return 1 - (ss_res / ss_tot.unsqueeze(-1))
