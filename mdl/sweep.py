import math
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from itertools import accumulate
from typing import Any, Callable, NamedTuple, Type

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
from tqdm.auto import tqdm

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
            lambda x, a, b: a / x**b, self.sample_sizes, self.ce_curve
        )
        return PowerLaw(a_hat, b_hat)


@dataclass
class Sweep:
    num_features: int
    num_classes: int = 2

    num_trials: int = 1
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
                num_features=self.num_features,
                num_classes=self.num_classes,
                device=self.device,
                dtype=self.dtype,
                **self.probe_kwargs,
            )
            for _ in range(n)
        ]
        params, buffers = stack_module_state(probes)  # type: ignore

        fwd = partial(functional_call, probes[0])
        fwd = partial(vmap(fwd), (params, buffers))
        opt = self.optimizer_cls(params.values(), **self.optimizer_kwargs)

        return opt, fwd

    def run(self, x: Tensor, y: Tensor, seed: int = 0) -> MdlResult:
        N, d = len(x), self.num_features
        rng = torch.Generator(device=self.device).manual_seed(seed)

        val_size = min(2048, round(N * self.val_frac))
        test_size = min(2048, round(N * self.val_frac))
        train_size = N - val_size - test_size

        # Shuffle data
        indices = torch.randperm(len(x), device=self.device, generator=rng)
        x, y = x[indices], y[indices]

        train_x, val_x, test_x = x.split([train_size, val_size, test_size])
        train_y, val_y, test_y = y.split([train_size, val_size, test_size])

        # Determining the appropriate size for the smallest chunk is a bit tricky. We
        # want to make sure that we have enough data for at least two minibatches
        # because it's too easy to overfit with full-batch training.
        min_size = min(1024, 2 * max(self.batch_size, self.num_classes, d))

        # Split data into num_chunks logarithmically spaced chunks
        parts = partition_logspace(len(train_x), self.num_chunks, min_size)
        cumsizes = list(accumulate(parts))

        loss_fn = bce_with_logits if self.num_classes == 2 else cross_entropy

        curve = []
        pbar = tqdm(cumsizes, unit="scales")
        total_mdl = 0.0
        total_trials = 0

        for n in pbar:
            num_trials = self.num_trials
            total_trials += num_trials

            # Create new optimizer and forward function for this chunk size
            probe = self.probe_cls(
                num_features=self.num_features,
                num_classes=self.num_classes,
                device=self.device,
                dtype=self.dtype,
                **self.probe_kwargs,
            )
            opt = self.optimizer_cls(probe.parameters(), **self.optimizer_kwargs)

            best_loss = torch.inf
            best_state = probe.state_dict()
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
                for x_batch, y_batch in zip(
                    train_x[:n].split(self.batch_size),
                    train_y[:n].split(self.batch_size),
                ):
                    opt.zero_grad()

                    # We just sum the loss across different trials since they don't
                    # affect one another
                    loss = loss_fn(probe(x_batch), y_batch)
                    loss.backward()
                    opt.step()

                # Evaluate on the validation set
                with torch.no_grad():
                    # Update learning rate schedule
                    val_loss = 0.0

                    for x_batch, y_batch in zip(
                        val_x.split(self.batch_size), val_y.split(self.batch_size)
                    ):
                        loss = loss_fn(probe(x_batch), y_batch, reduction="sum")
                        val_loss += float(loss) / math.log(2)  # Average over trials

                    val_loss /= val_size
                    schedule.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(probe.state_dict())

            # Evaluate on the next chunk
            with torch.no_grad():
                probe.load_state_dict(best_state)
                test_loss = 0.0

                for x_batch, y_batch in zip(
                    test_x.split(self.batch_size), test_y.split(self.batch_size)
                ):
                    loss = loss_fn(probe(x_batch), y_batch, reduction="sum")
                    test_loss += float(loss) / math.log(2)  # Average over trials

                test_loss /= test_size

                curve.append(float(test_loss))
                pbar.set_postfix(loss=f"{test_loss:.4f}")

                # Update MDL estimate
                total_mdl += n * test_loss

        return MdlResult(total_mdl / len(test_x), curve, cumsizes, total_trials)
