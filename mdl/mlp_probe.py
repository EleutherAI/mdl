from itertools import pairwise
from functools import partial

import torch
from torch import Tensor, nn, optim
from schedulefree import AdamWScheduleFree, ScheduleFreeWrapper
from mup import MuReadout, MuAdam, MuSGD, load_base_shapes, set_base_shapes
from muon import Muon

from .probe import Probe


class SwiGLU(torch.nn.Module):
    r"""Applies the SwiGLU function element-wise.
    SwiGLU is defined as:
    .. math::
        \text{SwiGLU}(x, y) = x * \sigma(y)
    where :math:`\sigma` is the sigmoid function, and :math:`x` and :math:`y` are
    split from the input tensor along the given dimension.
    Args:
        dim (int): the dimension on which to split the input. Default: -1
    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`
    Examples::
        >>> m = nn.SwiGLU()
        >>> input = torch.randn(4, 2)
        >>> output = m(input)
    """

    __constants__ = ["dim"]
    dim: int

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        x, y = torch.chunk(input, 2, dim=self.dim)

        return x * torch.sigmoid(y)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class MlpProbe(Probe):
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden_size: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        activation: str = "relu",
        schedule_free: bool = False,
        base_shapes_path: str | None = None,
        muon=False,
        **kwargs
    ):
        super().__init__(num_features, num_classes, device, dtype)

        self.learning_rate = learning_rate
        self.schedule_free = schedule_free
        self.betas = betas
        self.mup = base_shapes_path is not None
        self.muon = muon

        act = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swiglu": SwiGLU(),
        }[activation]

        assert hidden_size is not None
        k, h = num_classes, hidden_size

        in_features, out_features = h, h

        # Reduce h by a factor of 2/3 to keep the number of parameters constant
        if activation == "swiglu":
            swiglu_h = h * 2 // 3
            in_features = swiglu_h  # Swiglu output is one vector of len (h * 2 // 3)
            out_features = (
                swiglu_h * 2
            )  # Swiglu input is equivalent to two concatenated vectors of len (h * 2 // 3)

        self.net = nn.Sequential(
            nn.Linear(num_features, out_features, device=device, dtype=dtype),
            act,
            *[
                nn.Sequential(
                    nn.Linear(in_features, out_features, device=device, dtype=dtype),
                    act,
                )
                for _ in range(num_layers - 1)
            ],
            MuReadout(
                in_features, k, device=device, dtype=dtype, readout_zero_init=True
            ),
        )

        # Configure MuP
        if base_shapes_path:
            base_shapes = load_base_shapes(base_shapes_path)
            set_base_shapes(self, base_shapes)

    def build_optimizer(self):
        if self.muon:
            print("Not using MuP - not implemented for muon")
            muon_params = [p for p in self.net.parameters() if p.ndim >= 2]
            adamw_params = [p for p in self.net.parameters() if p.ndim < 2]

            optimizer = Muon(
                muon_params, 
                lr=0.02, 
                momentum=0.95, 
                adamw_params=adamw_params, 
                adamw_lr=self.learning_rate, 
                adamw_betas=self.betas, 
                adamw_wd=0.01 # type: ignore
            )
            return ScheduleFreeWrapper(optimizer)
        # opt_cls = AdamWScheduleFree if self.schedule_free else optim.AdamW
        # opt_cls = AdamWScheduleFree if self.schedule_free else optim.AdamW
        if self.mup:
            return MuAdam(
                self.parameters(), AdamWScheduleFree, lr=self.learning_rate, betas=self.betas, warmup_steps=1000
            )
        return AdamWScheduleFree(self.parameters(), lr=self.learning_rate, betas=self.betas, warmup_steps=1000)

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
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        schedule_free: bool = False,
        base_shapes_path: str | None = None,
        **kwargs
    ):
        super().__init__(num_features, num_classes, device, dtype)
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.betas = betas
        self.schedule_free = schedule_free
        self.mup = base_shapes_path is not None

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

        self.fc = MuReadout(
            sizes[-1], output_dim, device=device, dtype=dtype, readout_zero_init=True
        )

        # Configure MuP
        if base_shapes_path:
            base_shapes = load_base_shapes(base_shapes_path)
            set_base_shapes(self, base_shapes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.trunk(x)

        return self.fc(features).squeeze(-1)

    def build_optimizer(self) -> optim.Optimizer:
        if self.num_layers > 1:
            opt_cls = AdamWScheduleFree if self.schedule_free else optim.AdamW
            if self.mup:
                return MuAdam(
                    self.parameters(), opt_cls, lr=self.learning_rate, betas=self.betas
                )
            return opt_cls(self.parameters(), lr=self.learning_rate, betas=self.betas)
        else:
            # Use Nesterov SGD for linear probes. The problem is convex and there's
            # really no need to use an adaptive learning rate. We can set the fixed
            # LR considerably higher and this seems to help with convergence.
            opt_cls = MuSGD if self.mup else optim.SGD
            return opt_cls(
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


LinearProbe = partial(MlpProbe, num_layers=1)


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
