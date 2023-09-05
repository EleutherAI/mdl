import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple

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

    def scaling_law(self) -> PowerLaw:
        """Fits a power law to the cross-entropy curve."""

        (a_hat, b_hat), _ = curve_fit(
            lambda x, a, b: a / x**b, self.sample_sizes[1:], self.ce_curve[1:]
        )
        return PowerLaw(a_hat, b_hat)


class Probe(nn.Module, ABC):
    """Base class for probes."""

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = num_features

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...

    def reset_parameters(self):
        """Reset parameters, if necessary, before fitting a new model."""

    @torch.enable_grad()
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        l2_penalty: float = 0.1,
        max_iter: int = 1000,
        num_chunks: int = 10,
        num_trials: int = 5,
        seed: int = 42,
        solver: Literal["adam", "lbfgs"] = "lbfgs",
        verbose: bool = False,
    ) -> MdlResult:
        """Fits the model to the input data using L-BFGS with L2 regularization.

        Args:
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            chunk_size: Number of samples to use for each parameter update during
                prequential minimum description length (MDL) estimation. If `None`,
                train on all samples at once.
            l2_penalty: Positive L2 regularization strength, or equivalently, the
                precision of the Gaussian prior over the parameters. If `None`, use
                the default value of `num_classes`.
            max_iter: Maximum number of iterations for the L-BFGS optimizer.
            num_chunks: Number of chunks to split the data into for prequential MDL.
            seed: Random seed for shuffling the data.
            tol: Tolerance for the L-BFGS optimizer.
            verbose: Whether to display a progress bar.

        Returns:
            The negative log-likelihood of each chunk of data.
        """
        assert len(x) == len(y), "Input and target must have the same number of samples"

        eps = torch.finfo(x.dtype).eps
        assert l2_penalty > eps, "Cannot have a uniform prior over the parameters"

        # Shuffle the data so we don't learn in a weirdly structured order
        rng = torch.Generator(device=x.device).manual_seed(seed)

        # Generate num_trials different permutations of the data
        indices = torch.stack(
            [
                torch.randperm(len(x), device=x.device, generator=rng)
                for _ in range(num_trials)
            ]
        )

        # Generate num_trials copies of the model
        models = [deepcopy(self) for _ in range(num_trials)]
        params, buffers = stack_module_state(models)  # type: ignore
        parallel_fwd = vmap(partial(functional_call, self))

        n, d = x.shape
        min_size = max(self.num_classes, d)
        parts = partition_logspace(n, num_chunks, min_size)

        # This adds an extra batch dimension
        x, y = x[indices], y[indices]

        loss_fn = bce_with_logits if self.num_classes == 2 else cross_entropy
        y = y.to(
            torch.get_default_dtype() if self.num_classes == 2 else torch.long,
        )

        def closure():
            nonlocal t
            optimizer.zero_grad()

            # We sum the loss over data points instead of averaging it, so that the
            # L2 penalty decreases in relative importance as the dataset grows. This
            # allows us to interpret the penalty as a prior over the parameters.
            logits = parallel_fwd((params, buffers), x[:, :t]).squeeze(-1)
            loss = loss_fn(
                logits.flatten(0, 1), y[:, :t].flatten(0, 1), reduction="sum"
            )

            norm_sq = sum(p.square().sum() for p in params.values())
            reg_loss = loss + l2_penalty * norm_sq
            reg_loss.backward()
            return float(reg_loss)

        # Split the data into chunks for prequential MDL estimation.
        x_chunks, y_chunks = x.split(parts, 1), y.split(parts, 1)

        # State for prequential MDL estimation
        mdl = 0.0
        losses = []
        sample_sizes = []
        t = 0

        for x_chunk, y_chunk in zip(tqdm(x_chunks, disable=not verbose), y_chunks):
            # First evaluate on this chunk
            with torch.no_grad():
                logits = parallel_fwd((params, buffers), x_chunk).flatten(0, 1)
                y_chunk = y_chunk.flatten(0, 1)

                per_sample_ce = loss_fn(logits, y_chunk).item() / math.log(2)
                losses.append(per_sample_ce)
                sample_sizes.append(t)

            # Weight the loss by the number of samples in the chunk
            mdl += per_sample_ce * x_chunk.shape[1]
            t += x_chunk.shape[1]

            models = [deepcopy(self) for _ in range(num_trials)]
            params, buffers = stack_module_state(models)  # type: ignore

            # Then train on it
            if solver == "adam":
                optimizer = optim.AdamW(params.values())
                schedule = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.5, patience=0, verbose=True
                )
                x_, y_ = x[:, :t], y[:, :t]

                # Split into train and val
                val_size = min(1000, x_.shape[1] // 10)
                assert val_size > 0, "Dataset is too small to split into train and val"

                x_train, y_train = x_[:, val_size:], y_[:, val_size:]
                x_val, y_val = x_[:, :val_size], y_[:, :val_size]

                n = x_train.shape[1]
                batch_size = min(32, n)

                for _ in range(100):
                    # Train loop
                    for i in range(0, n, batch_size):
                        optimizer.zero_grad()

                        x_batch = x_train[:, i : i + batch_size]
                        y_batch = y_train[:, i : i + batch_size]

                        logits = parallel_fwd((params, buffers), x_batch).squeeze(-1)
                        loss = loss_fn(
                            logits.flatten(0, 1),
                            y_batch.flatten(0, 1),
                        )
                        loss.backward()
                        optimizer.step()

                    # Val loop
                    with torch.no_grad():
                        logits = parallel_fwd((params, buffers), x_val).squeeze(-1)
                        loss = loss_fn(logits.flatten(0, 1), y_val.flatten(0, 1))

                    schedule.step(loss)
                    if optimizer.param_groups[0]["lr"] < 6e-5:
                        print(f"Early stopping with loss {loss.item()}")
                        break
            else:
                optimizer = optim.LBFGS(
                    params.values(),
                    line_search_fn="strong_wolfe",
                    max_iter=max_iter,
                )
                optimizer.step(closure)

        return MdlResult(mdl / t, losses, sample_sizes)
