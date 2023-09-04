from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import torch

from mdl.utils import oric


@given(
    arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=10),
        elements=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False),
    ).map(
        # Compute the fractional part of the sum of the elements, divide it by
        # the number of elements, and subtract this from every element.
        # This ensures that the sum of the elements is integral.
        lambda x: (x - (x.sum() - np.floor(x.sum())) / len(x)),
    ),
)
def test_integer_constrained_rounding(x_np: np.ndarray):
    x = torch.from_numpy(x_np)
    original_sum = x.sum().round().long()

    rounded = oric(x)
    torch.testing.assert_close(rounded.sum(), original_sum)
    assert torch.abs(x - rounded).max() <= 1.0
