from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any, Callable, NamedTuple, Type

import torch
from scipy.optimize import curve_fit
from torch import Tensor
from tqdm.auto import tqdm

from .math import partition_logspace
from .mlp_probe import ResMlpProbe, Probe


class PowerLaw(NamedTuple):
    """Parameters of a power law."""

    a: float
    b: float


@dataclass
class MdlResult:
    """Result of a prequential minimum description length (MDL) estimation."""

    mdl: float
    """MDL in bits per sample."""

    ce_curve: list[float]
    """Next-step cross-entropy in bits per sample for each chunk."""

    sample_sizes: list[int]
    """Number of samples used for each chunk."""

    total_trials: int
    """(DEPRECATED) Total number of trials used for the estimation."""

    def scaling_law(self) -> PowerLaw:
        """Fits a power law to the cross-entropy curve."""

        (a_hat, b_hat), _ = curve_fit(
            lambda x, a, b: a / x**b, self.sample_sizes, self.ce_curve
        )
        return PowerLaw(a_hat, b_hat)


@dataclass
class Sweep:
    num_features: int
    num_classes: int = 2

    num_chunks: int = 10
    """Number of logarithmically-spaced chunks to split the data into."""

    batch_size: int = 32
    """Batch size to use for fitting the probes."""

    probe_cls: Type[Probe] = ResMlpProbe
    """Probe class to instantiate."""

    probe_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass to the probe constructor."""

    val_frac: float = 0.2
    """Fraction of each chunk to use for validation."""

    device: str | torch.device = "cpu"
    dtype: torch.dtype | None = None

    def __post_init__(self):
        assert self.num_features > 0
        assert self.num_classes > 1

    def run(
        self,
        x: Tensor,
        y: Tensor,
        seed: int = 0,
        transform: Callable[[Tensor, Tensor], Tensor] = lambda x, _: x,
        **fit_kwargs,
    ) -> MdlResult:
        N, d = len(x), self.num_features
        rng = torch.Generator(device=self.device).manual_seed(seed)

        val_size = min(2048, round(N * self.val_frac))
        nonval_size = N - val_size

        # Determining the appropriate size for the smallest chunk is a bit tricky. We
        # want to make sure that we have enough data for at least two minibatches
        # because it's too easy to overfit with full-batch training.
        min_size = min(1024, 2 * max(self.batch_size, self.num_classes, d))

        # Split data into num_chunks logarithmically spaced chunks
        parts = partition_logspace(nonval_size, self.num_chunks, min_size)

        cumsizes = list(accumulate(parts))
        pbar = tqdm(
            zip(cumsizes[:-1], cumsizes[1:]), total=len(cumsizes) - 1, unit="scales"
        )

        curve = []
        total_mdl = 0.0

        for n, next_n in pbar:
            # Shuffle data
            indices = torch.randperm(len(x), device=self.device, generator=rng)
            x, y = x[indices], y[indices]

            nonval_x, val_x = x.split([nonval_size, val_size])
            nonval_y, val_y = y.split([nonval_size, val_size])

            train_size, test_size = nonval_size - parts[-1], parts[-1]
            train_x, test_x = nonval_x.split([train_size, test_size])
            train_y, test_y = nonval_y.split([train_size, test_size])

            probe = self.probe_cls(
                num_features=self.num_features,
                num_classes=self.num_classes,
                device=self.device,
                dtype=self.dtype,
                **self.probe_kwargs,
            )
            probe.fit(
                train_x[:n],
                train_y[:n],
                x_val=val_x,
                y_val=val_y,
                verbose=False,
                transform=transform,
                **fit_kwargs,
            )

            # Compute test loss and add to scaling curve
            test_loss = probe.evaluate(
                transform(test_x, test_y), test_y, self.batch_size
            )
            curve.append(float(test_loss))
            pbar.set_postfix(loss=f"{test_loss:.4f}")

            # Update MDL estimate
            total_mdl += next_n * test_loss

        return MdlResult(total_mdl / nonval_size, curve, cumsizes, 0)
