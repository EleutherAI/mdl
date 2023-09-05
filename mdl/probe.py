import math
from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_with_logits,
)
from torch.nn.functional import (
    cross_entropy,
)
from tqdm.auto import tqdm

from .math import partition_logspace


class MdlResult(NamedTuple):
    """Result of a prequential minimum description length (MDL) estimation."""

    mdl: float
    """MDL in bits per sample."""

    ce_curve: list[float]
    """Next-step cross-entropy in bits per sample for each chunk."""

    sample_sizes: list[int]
    """Number of samples used for each chunk."""


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
        num_chunks: int = 5,
        seed: int = 42,
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
        indices = torch.randperm(len(x), device=x.device, generator=rng)
        x, y = x[indices], y[indices]

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
        )

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
            logits = self(x[:t]).squeeze(-1)
            loss = loss_fn(logits, y[:t], reduction="sum")

            norm_sq = sum(p.square().sum() for p in self.parameters())
            reg_loss = loss + l2_penalty * norm_sq
            reg_loss.backward()
            return float(reg_loss)

        # Split the data into chunks for prequential MDL estimation.
        min_size = max(self.num_classes, x.shape[-1])
        parts = partition_logspace(len(x), num_chunks, min_size)
        x_chunks, y_chunks = x.split(parts), y.split(parts)

        # State for prequential MDL estimation
        mdl = 0.0
        losses = []
        sample_sizes = []
        t = 0

        for x_chunk, y_chunk in zip(tqdm(x_chunks, disable=not verbose), y_chunks):
            # First evaluate on this chunk
            with torch.no_grad():
                per_sample_ce = loss_fn(self(x_chunk), y_chunk).item() / math.log(2)
                losses.append(per_sample_ce)
                sample_sizes.append(t)

            # Weight the loss by the number of samples in the chunk
            mdl += per_sample_ce * len(x_chunk)
            t += len(x_chunk)

            # Then train on it
            self.reset_parameters()
            optimizer.step(closure)

        return MdlResult(mdl / t, losses, sample_sizes)
