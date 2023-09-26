import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable

import torch
from torch import Tensor, nn, optim
from torch.nn.functional import (
    cross_entropy,
)
from tqdm.auto import trange


class Probe(nn.Module, ABC):
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.dtype = dtype or torch.get_default_dtype()
        self.num_classes = num_classes
        self.num_features = num_features

    def augment_data(self, x: Tensor) -> Tensor:
        return x

    @abstractmethod
    def build_optimizer(self) -> optim.Optimizer:
        ...

    @torch.enable_grad()
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        batch_size: int = 128,
        early_stop_epochs: int = 4,
        max_epochs: int = 50,
        preprocessor: Callable[[Tensor, Tensor], Tensor] = lambda x, _: x,
        seed: int = 42,
        verbose: bool = False,
        return_validation_losses: bool = False,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
    ):
        """Fits the model to the input data using Adam with L2 regularization.

        Args:
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            batch_size: Batch size to use for fitting the probe.
            early_stop_epochs: Number of epochs to wait before stopping early if the
                validation loss does not improve.
            max_epochs: Maximum number of epochs to train for.
            seed: Random seed for shuffling the data.
            tol: Tolerance for the L-BFGS optimizer.
            verbose: Whether to display a progress bar.
            return_validation_losses: Whether to return val losses of each epoch.

        Returns:
            The negative log-likelihood of each chunk of data.
        """
        assert len(x) == len(y), "Input and target must have the same number of samples"
        x = x.to(self.dtype)

        # Shuffle the data so we don't learn in a weirdly structured order
        if x_val is None or y_val is None:
            rng = torch.Generator(device=x.device).manual_seed(seed)
            perm = torch.randperm(len(x), generator=rng, device=x.device)
            x, y = x[perm], y[perm]

            val_size = min(2048, len(x) // 5)
            assert val_size > 0, "Dataset is too small to split into train and val"

            x_train, y_train = x[val_size:], y[val_size:]
            x_val, y_val = x[:val_size], y[:val_size]
        else:
            x_train, y_train = x, y
            val_size = len(x_val)

        val_losses = []
        y = y.to(
            torch.get_default_dtype() if self.num_classes == 2 else torch.long,
        )

        opt = self.build_optimizer()
        pbar = trange(max_epochs, desc="Epoch", disable=not verbose)

        best_loss = torch.inf
        best_opt_state = opt.state_dict()
        best_state = self.state_dict()
        num_plateaus = 0

        self.eval()
        x_val = self.augment_data(x_val)
        x_val = preprocessor(x_val, y_val)

        for _ in pbar:
            val_loss = self.evaluate(x_val, y_val, batch_size)

            if val_loss < best_loss:
                best_loss = val_loss
                best_opt_state = deepcopy(opt.state_dict())
                best_state = deepcopy(self.state_dict())
                num_plateaus = 0
            else:
                num_plateaus += 1

                # Early stopping
                if num_plateaus >= early_stop_epochs:
                    break

                # Backtrack
                opt.load_state_dict(best_opt_state)
                self.load_state_dict(best_state)

                # Manual ReduceLROnPlateau
                opt.param_groups[0]["lr"] *= 0.5

            val_losses.append(best_loss)
            pbar.set_postfix(loss=best_loss)

            ### TRAIN LOOP ###
            self.train()
            for x_batch, y_batch in zip(
                x_train.split(batch_size), y_train.split(batch_size)
            ):
                opt.zero_grad()

                x_batch = self.augment_data(x_batch)
                x_batch = preprocessor(x_batch, y_batch)
                loss = self.loss(x_batch, y_batch)
                loss.backward()
                opt.step()

        # Load parameters with lowest validation loss
        self.load_state_dict(best_state)

        if return_validation_losses:
            return val_losses

    def loss_fn(self, logits: Tensor, target: Tensor) -> Tensor:
        """Computes the loss of the probe on the given data."""
        return cross_entropy(logits, target) / math.log(2)

    @torch.no_grad()
    def evaluate(self, x: Tensor, y: Tensor, batch_size: int) -> float:
        """Compute average loss on `(x, y)` in batches of size `batch_size`."""
        total_loss = sum(
            self.loss(x_batch, y_batch).item() * len(x_batch)
            for x_batch, y_batch in zip(x.split(batch_size), y.split(batch_size))
        )
        return total_loss / len(x)

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Computes the loss of the probe on the given data."""
        return self.loss_fn(self(x.to(self.dtype)).squeeze(-1), y)
