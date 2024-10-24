from functools import partial
from itertools import pairwise

import torch
from torch import Tensor, nn, optim

from .probe import Probe


class SeqMlpProbe(Probe):
    def __init__(
        self,
        # Unused
        num_features: int,
        num_classes: int = 2,
        hidden_size: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        num_layers: int = 2,
    ):
        super().__init__(num_features, num_classes, device, dtype)

        assert hidden_size is not None

        k, h = num_classes, hidden_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(32 * 32 * 3, h, device=device, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(h, h, device=device, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(h, k, device=device, dtype=dtype),
        )
        
        # TODO incorporate initial image size
        # self.net = torch.nn.Sequential(
        #     *[
        #         torch.nn.Sequential(
        #             torch.nn.Linear(h, h),
        #             torch.nn.ReLU(),
        #         )
        #         for _ in range(num_layers)
        #     ],
        #     torch.nn.Linear(h, k),
        # )
    def build_optimizer(self) -> optim.Optimizer:
        return optim.SGD(
            self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class ResMlpProbe(Probe):
    """Multi-layer perceptron with ResNet architecture."""

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden_size: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        num_layers: int = 2,
    ):
        super().__init__(num_features, num_classes, device, dtype)
        self.num_layers = num_layers

        if hidden_size is None:
            hidden_size = (
                4 * num_features if num_layers == 2 else round(num_features * 4 / 3)
            )

        output_dim = num_classes if num_classes > 2 else 1
        sizes = [num_features] + [hidden_size] * (num_layers - 1)

        self.trunk = nn.Sequential(
            *[
                MlpBlock(in_dim, out_dim, device=device, dtype=dtype)
                for in_dim, out_dim in pairwise(sizes)
            ]
        )

        self.fc = nn.Linear(hidden_size, output_dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        features = self.trunk(x)
        return self.fc(features).squeeze(-1)

    def build_optimizer(self) -> optim.Optimizer:
        if self.num_layers > 1:
            return torch.optim.SGD(
                self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
            )
            # return optim.AdamW(self.parameters())
        else:
            # Use Nesterov SGD for linear probes. The problem is convex and there's
            # really no need to use an adaptive learning rate. We can set the fixed
            # LR considerably higher and this seems to help with convergence.
            return optim.SGD(
                self.parameters(),
                # Learning rate of 0.1 with momentum 0.9 is "really" an LR of unity in
                # PyTorch's parametrization; see https://youtu.be/k8fTYJPd3_I
                lr=0.1,
                momentum=0.9,
                # Nesterov seems to be strictly better than regular momentum
                nesterov=True,
                # Use same weight decay as AdamW above
                weight_decay=0.01,
            )


class MlpBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        self.linear1 = nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(
            out_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.bn1 = nn.BatchNorm1d(out_features, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(out_features, device=device, dtype=dtype)
        self.downsample = (
            nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
            if in_features != out_features
            else None
        )

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.linear2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = nn.functional.relu(out)

        return out


# Convenience alias
LinearProbe = partial(ResMlpProbe, num_layers=1)
