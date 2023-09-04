import math

from torch import Tensor
import torch


def logspace_split(x: Tensor, dim: int = 0, *, min_size: int = 1) -> Tensor:
    """Split `x` into base-2 logarithmically spaced chunks along `dim`."""
    n = x.shape[dim]

    halvings = round(math.log2(n / min_size))
    shares = n / 2.0 ** torch.arange(halvings, 0, -1)

    shortfall = n - torch.sum(shares)
    shares += shortfall / len(shares)  # Distribute the shortfall evenly.
    sizes = oric(shares).tolist()      # Round to the nearest integer.

    return x.split(sizes, dim=dim)


def oric(x: Tensor) -> Tensor:
    """Optimal rounding under integer constraints.

    Given a vector of real numbers such that the sum is an integer, returns a vector
    of rounded integers that preserves the sum and which minimizes the Lp-norm of the
    difference between the rounded and original vectors for all p >= 1. Algorithm from
    https://arxiv.org/abs/1501.00014. Runs in O(n log n) time.

    Args:
        x: A 1D vector of real numbers that sum to an integer.

    Returns:
        A 1D vector of rounded integers, preserving the sum.
    """
    rounded = x.floor()
    shortfall = x - rounded

    # The total shortfall should be *exactly* an integer, but we
    # round to account for numerical error.
    total_shortfall = shortfall.sum().round().long()
    indices = shortfall.argsort(descending=True)

    # Apportion the total shortfall to the elements in order of
    # decreasing shortfall.
    rounded[indices[:total_shortfall]] += 1
    return rounded.long()
