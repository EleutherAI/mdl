import math
from dataclasses import dataclass, field
from functools import partial
from itertools import accumulate
from typing import Any, Callable, NamedTuple, Sequence, Type

import torch
from scipy.optimize import curve_fit
from torch import Tensor, nn, optim
from torch.func import functional_call, stack_module_state, vmap
from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_with_logits,
)
from torch.nn.functional import (
    cross_entropy,
)

from .math import partition_logspace
from .mlp_probe import MlpProbe


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
    """Total number of trials used for the estimation."""

    def scaling_law(self) -> PowerLaw:
        """Fits a power law to the cross-entropy curve."""

        (a_hat, b_hat), _ = curve_fit(
            lambda x, a, b: a / x**b, self.sample_sizes[1:], self.ce_curve[1:]
        )
        return PowerLaw(a_hat, b_hat)


@dataclass
class Sweep:
    num_features: int
    num_classes: int = 2

    num_trials: int = 5
    """Minimum number of trials to use for each chunk size."""

    num_chunks: int = 10
    """Number of logarithmically-spaced chunks to split the data into."""

    batch_size: int = 32
    """Batch size to use for fitting the probes."""

    optimizer_cls: Type[optim.Optimizer] = optim.AdamW
    """Optimizer class to use for fitting the probes."""

    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass to the optimizer constructor."""

    probe_cls: Type[nn.Module] = MlpProbe
    """Probe class to instantiate."""

    probe_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass to the probe constructor."""

    seed: int = 0
    """Seed for the random number generator."""

    val_frac: float = 0.2
    """Fraction of each chunk to use for validation."""

    device: str | torch.device = "cpu"
    dtype: torch.dtype | None = None

    def __post_init__(self):
        assert self.num_features > 0
        assert self.num_classes > 1

    def build_optimizer(
        self, n: int
    ) -> tuple[optim.Optimizer, Callable[[Tensor], Tensor]]:
        probes = [
            self.probe_cls(
                self.num_features,
                self.num_classes,
                self.device,
                self.dtype,
                **self.probe_kwargs,
            )
            for _ in range(n)
        ]
        params, buffers = stack_module_state(probes)  # type: ignore

        fwd = partial(functional_call, probes[0])
        fwd = partial(vmap(fwd), (params, buffers))
        opt = self.optimizer_cls(params.values(), **self.optimizer_kwargs)

        return opt, fwd

    def run(self, x: Sequence, y: Sequence) -> MdlResult:
        N, d = len(x), self.num_features
        rng = torch.Generator(device=self.device).manual_seed(self.seed)

        val_size = min(2048, round(N * self.val_frac))
        N = N - val_size

        # Determining the appropriate size for the smallest chunk is a bit tricky. We
        # want to make sure that we have enough data for at least two minibatches
        # because it's too easy to overfit with full-batch training.
        min_size = min(1024, 2 * max(self.batch_size, self.num_classes, d))

        # Split data into num_chunks logarithmically spaced chunks
        parts = partition_logspace(N, self.num_chunks, min_size)
        cumsizes = list(accumulate(parts))

        # Generate max_trials different permutations of the data
        master_indices = torch.stack(
            [
                torch.randperm(len(x), device=self.device, generator=rng)
                for _ in range(self.num_trials)
            ]
        )

        # Create vectorized loss function
        loss_fn = vmap(bce_with_logits if self.num_classes == 2 else cross_entropy)

        curve = []
        total_mdl = 0.0
        total_trials = 0

        for n, next_n in zip(cumsizes, cumsizes[1:]):
            num_trials = self.num_trials
            total_trials += num_trials

            print(f"Sample size: {n}")
            print(f"Number of trials: {num_trials}")

            indices = master_indices[:num_trials, :n]
            val_indices = master_indices[:num_trials, -val_size:]

            # Create new optimizer and forward function for this chunk size
            opt, fwd = self.build_optimizer(num_trials)
            schedule = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=0.5,
                patience=0,
                threshold=0.01,
            )

            # Train until we don't improve for four epochs
            # TODO: Perform early stopping and learning rate annealing separately
            # for each trial?
            while opt.param_groups[0]["lr"] > opt.defaults["lr"] * 0.5**4:
                # Single epoch on the training set
                for batch in indices.split(self.batch_size, dim=1):
                    opt.zero_grad()

                    # We just sum the loss across different trials since they don't
                    # affect one another
                    loss = loss_fn(fwd(x[batch]), y[batch]).sum()
                    loss.backward()
                    opt.step()

                # Evaluate on the validation set
                with torch.no_grad():
                    # Update learning rate schedule
                    val_loss = 0.0

                    for batch in val_indices.split(self.batch_size, dim=1):
                        losses = loss_fn(fwd(x[batch]), y[batch], reduction="sum")
                        val_loss += losses.mean() / math.log(2)  # Average over trials

                    val_loss /= val_size
                    schedule.step(val_loss)
                    print(f"Validation loss: {float(val_loss):.4f} n = {val_size}")

            # Evaluate on the next chunk
            with torch.no_grad():
                indices = master_indices[:num_trials, n:next_n]
                test_loss = 0.0

                for batch in indices.split(self.batch_size, dim=1):
                    losses = loss_fn(fwd(x[batch]), y[batch], reduction="sum")
                    test_loss += losses.mean() / math.log(2)  # Average over trials

                test_loss /= next_n - n

                curve.append(float(test_loss))
                print(f"Test loss: {test_loss:.4f} n = {next_n - n}")

                # Update MDL estimate
                total_mdl += n * test_loss

        return MdlResult(total_mdl / N, curve, cumsizes[1:], total_trials)
