from typing import NamedTuple
import math

import torch
from torch import Tensor
from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_with_logits, cross_entropy,
)
from tqdm.auto import tqdm

from .utils import logspace_split


class MdlResult(NamedTuple):
    """Result of a prequential minimum description length (MDL) estimation."""

    mdl: float
    """MDL in bits per sample."""

    ce_curve: list[float]
    """Next-step cross-entropy in bits per sample for each chunk."""


class LinearClassifier(torch.nn.Module):
    """Linear classifier trained with supervised learning."""

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(
            num_features, num_classes if num_classes > 2 else 1, device=device, dtype=dtype
        )
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(-1)

    @torch.enable_grad()
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        l2_penalty: float | None = None,
        max_iter: int = 1000,
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
            seed: Random seed for shuffling the data.
            tol: Tolerance for the L-BFGS optimizer.
            verbose: Whether to display a progress bar.

        Returns:
            The negative log-likelihood of each chunk of data.
        """
        assert len(x) == len(y), "Input and target must have the same number of samples"

        # Number of classes
        k = max(self.linear.out_features, 2)

        # By default, we set the L2 penalty equal to log(num_classes) so that it's on
        # the same scale as the cross-entropy loss.
        if l2_penalty is None:
            l2_penalty = math.log(k)
        else:
            eps = torch.finfo(self.linear.weight.dtype).eps
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

        loss_fn = bce_with_logits if k == 2 else cross_entropy
        y = y.to(
            torch.get_default_dtype() if k == 2 else torch.long,
        )

        def closure():
            nonlocal t
            optimizer.zero_grad()

            # We sum the loss over data points instead of averaging it, so that the
            # L2 penalty decreases in relative importance as the dataset grows. This
            # allows us to interpret the penalty as a prior over the parameters.
            logits = self(x[:t]).squeeze(-1)
            loss = loss_fn(logits, y[:t], reduction="sum")

            reg_loss = loss + l2_penalty * self.linear.weight.square().sum()
            reg_loss.backward()
            return float(reg_loss)

        # Split the data into chunks for prequential MDL estimation. Each chunk is the
        # same size if `chunk_size` divides the number of samples evenly, otherwise
        # the last chunk is smaller.
        thresh = k ** 2
        x_chunks = logspace_split(x, min_size=thresh) #x.split(chunk_size) if chunk_size else [x]
        y_chunks = logspace_split(y, min_size=thresh) # y.split(chunk_size) if chunk_size else [y]

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
            optimizer.step(closure)

        return MdlResult(mdl / t, losses)
