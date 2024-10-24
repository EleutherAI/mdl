import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable

import torch
from torch import Tensor, nn, optim
from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_loss,
)
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

    @abstractmethod
    def build_optimizer(self) -> optim.Optimizer:
        ...

    @torch.enable_grad()
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        augment: Callable[[Tensor], Tensor] = lambda x: x,
        batch_size: int = 128,
        early_stop_epochs: int = 4,
        max_epochs: int = 50,
        reduce_lr_on_plateau: bool = True,
        return_validation_losses: bool = False,
        seed: int = 42,
        transform: Callable[[Tensor, Tensor], Tensor] = lambda x, _: x,
        verbose: bool = False,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
        logger = None,
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
        schedule = (
            optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
            if reduce_lr_on_plateau
            else optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
        )
        schedule = (
            optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
            if reduce_lr_on_plateau
            else optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
        )
        pbar = trange(max_epochs, desc="Epoch", disable=not verbose)

        best_loss = torch.inf
        best_opt_state = opt.state_dict()
        best_state = self.state_dict()
        num_plateaus = 0

        self.eval()
        x_val = transform(x_val, y_val)

        # Record initial weights for weight change norm logging
        initial_weights = deepcopy(self.state_dict()) if logger is not None else None

        for ep in pbar:
            val_loss = self.evaluate(x_val, y_val, batch_size)
            val_acc = self.accuracy(x_val, y_val, batch_size)

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
                if reduce_lr_on_plateau:
                    opt.param_groups[0]["lr"] *= 0.5
                if reduce_lr_on_plateau:
                    opt.param_groups[0]["lr"] *= 0.5

            val_losses.append(best_loss)
            pbar.set_postfix(loss=best_loss)

            ### TRAIN LOOP ###
            self.train()
            train_losses = []

            for x_batch, y_batch in zip(
                x_train.split(batch_size), y_train.split(batch_size)
            ):
                opt.zero_grad()

                x_batch = transform(augment(x_batch), y_batch)
                loss = self.loss(x_batch, y_batch)
                train_losses.append(loss.item())
                loss.backward()
                opt.step()

            # Update learning rate
            schedule.step()

            # Calculate norm of parameters' mean differences from initialization
            w_frobenius_norm, w_spectral_norm, b_l1, b_frobenius = self.calculate_weight_change_norms(initial_weights)

            if logger is not None:
                logger.log({
                    "epoch": ep,
                    "train/loss": sum(train_losses) / len(train_losses),
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "learning_rate": opt.param_groups[0]["lr"],
                    "weight_frobenius_norm": w_frobenius_norm,
                    "weight_spectral_norm": w_spectral_norm,
                    "bias_l1_norm": b_l1,
                    "bias_frobenius_norm": b_frobenius
                })

        # Load parameters with lowest validation loss
        self.load_state_dict(best_state)

        if return_validation_losses:
            return val_losses

    @torch.no_grad()
    def accuracy(self, x: Tensor, y: Tensor, batch_size: int) -> float:
        """Compute average accuracy on `(x, y)` in batches of size `batch_size`."""
        total_correct = sum(
            self(x_batch).argmax(dim=-1).eq(y_batch).sum().item()
            for x_batch, y_batch in zip(x.split(batch_size), y.split(batch_size))
        )
        return total_correct / len(x)

    @torch.no_grad()
    def evaluate(self, x: Tensor, y: Tensor, batch_size: int) -> float:
        """Compute average loss on `(x, y)` in batches of size `batch_size`."""
        total_loss = sum(
            self.loss(x_batch, y_batch).item() * len(x_batch)
            for x_batch, y_batch in zip(x.split(batch_size), y.split(batch_size))
        )
        return total_loss / len(x)

    def loss_fn(self, logits: Tensor, target: Tensor, smoothing: float = 0) -> Tensor:
        """Computes the loss of the predictions on the given data."""
        return (
            cross_entropy(logits, target.long())
            if logits.ndim == 2
            else bce_loss(logits, target)
        ) / math.log(2)

    def loss(self, x: Tensor, y: Tensor, smoothing: float = 0.1) -> Tensor:
        """Computes the loss of the probe on the given data."""
        return self.loss_fn(self(x.to(self.dtype)).squeeze(-1), y, smoothing)

    def calculate_weight_change_norms(self, initial_weights):
        """Calculate Frobenius and spectral norms of weight changes for logging."""
        current_weights = self.state_dict()
        
        num_weights = len([weight for weight in current_weights if 'weight' in weight])
        num_biases = len([bias for bias in current_weights if 'bias' in bias])
        assert num_weights > 0, "No weights found in model"

        w_frobenius_norm = 0.0
        w_spectral_norm = 0.0
        b_l1 = 0.
        b_frobenius = 0.

        for name, current_param in current_weights.items():
            if 'weight' in name:
                weight_diff = current_param - initial_weights[name]
                if len(weight_diff.shape) > 2:
                    print("more than 2 dims")
                    weight_diff_2d = weight_diff.reshape(weight_diff.shape[0], -1)
                else:
                    weight_diff_2d = weight_diff
                    
                w_frobenius_norm += torch.norm(weight_diff_2d, p='fro') / num_weights

                # Calculate only the largest singular value
                U, S, Vh = torch.svd_lowrank(weight_diff_2d, q=1)
                w_spectral_norm += S[0].item() / num_weights
            
            if 'bias' in name:
                bias_diff = current_param - initial_weights[name]
                b_l1 += torch.norm(bias_diff, p=1) / num_biases
                b_frobenius += torch.norm(bias_diff, p=2) / num_biases

        return w_frobenius_norm, w_spectral_norm, b_l1, b_frobenius
