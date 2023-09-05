from scipy.optimize import brentq


def oric(x: list[float]) -> list[int]:
    """Optimal rounding under integer constraints (ORIC).

    Given a list of real numbers such that the sum is an integer, returns a list
    of integers that preserves the sum and which minimizes the Lp-norm of the
    difference between the rounded and original lists for all p >= 1. Algorithm from
    https://arxiv.org/abs/1501.00014. Runs in O(n log n) time.

    Args:
        x: A list of floating point numbers that sum to an integer. To account for
        numerical error, we currently don't check that it actually sums to an integer.

    Returns:
        A list of rounded integers, preserving the sum.
    """
    # Round down each element and calculate the shortfall for each
    rounded = [int(v) for v in x]
    shortfall = [v - int(v) for v in x]

    # Calculate the total shortfall and round it to the nearest integer
    total_shortfall = round(sum(shortfall))

    # Sort the indices by their corresponding shortfall values, in descending order
    indices = sorted(range(len(shortfall)), key=lambda k: shortfall[k], reverse=True)

    # Apportion the total shortfall to the elements in order of decreasing shortfall
    for i in indices[:total_shortfall]:
        rounded[i] += 1

    return rounded


def partition_logspace(n: int, num_parts: int, start: int = 1) -> list[int]:
    """Partition a positive integer `n` into `num_parts` logarithmically spaced parts.

    In number theory, a partition of a natural number `n` is a set of naturals that sum
    to `n`. For example, the partitions of 3 are {1, 1, 1}, {1, 2}, and {3}.

    This function returns a partition of `n` into `num_parts` parts, where the sorted
    parts are as close as possible (in the Lp sense) to being logarithmically spaced.
    We first compute the exact solution to the corresponding problem over the reals-
    that is, finding the ratio `r` of the geometric series that sums to `n` and starts
    with `start`- then use ORIC (https://arxiv.org/abs/1501.00014) to optimally round
    the series while ensuring it still sums to `n`.

    Args:
        n: The integer to partition.
        num_parts: The number of parts in the partition.
        start: The size of the smallest part.
    """
    assert n > num_parts > 0, "num_parts must be positive and less than n"

    # Compute the base of the geometric series
    base = solve_geometric_series(n, num_parts, start)

    # Compute the floating point grid and round it
    return oric([start * base**i for i in range(num_parts)])


def solve_geometric_series(S: float, n: int, a: float = 1.0) -> float:
    """Solve the `n`-term geometric series starting at `a` summing to `S` for `r`."""
    result = brentq(
        # See https://en.wikipedia.org/wiki/Geometric_series#Finite_series
        lambda r: a * (1 - r**n) / (1 - r) - S,
        # We want to avoid r = 1, which is a singularity.
        1 + 2.220446049250313e-16,
        # This is a loose upper bound on the solution.
        n * S ** (1 / n),
    )
    assert isinstance(result, float)

    return result
