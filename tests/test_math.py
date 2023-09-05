import math

from hypothesis import given
from hypothesis import strategies as st

from mdl.math import oric


@given(st.lists(st.floats(min_value=1e-3, max_value=1e6, allow_nan=False)))
def test_integer_constrained_rounding(x: list[float]):
    float_total = sum(x)
    int_total = int(float_total)

    # Compute the fractional part of the sum of the elements, divide it by
    # the number of elements, and subtract this from every element.
    # This ensures that the sum of the elements is integral.
    x = [v - (float_total - int_total) / len(x) for v in x]
    assert math.isclose(sum(x), int_total)

    # Now actually run ORIC
    rounded = oric(x)

    # Was the sum preserved?
    assert math.isclose(sum(rounded), int_total)

    # No element should be changed by more than 1
    assert all(abs(before - after) <= 1.0 for before, after in zip(x, rounded))
