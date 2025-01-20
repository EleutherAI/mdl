import math
import json
import torch
from dataclasses import dataclass
from torch import nn
from torch import optim
from mup import MuReadout, MuAdam, load_base_shapes, set_base_shapes
from schedulefree import AdamWScheduleFree

from .probe import Probe


@dataclass
class LeNetConfig:
    image_size: int
    num_channels: int
    conv_hidden_sizes: list[int]
    fc_hidden_sizes: list[int]
    kernel_sizes: list[int]
    num_labels: int
    kernel_size: int = 5

class LeNet5(nn.Module):
    def _conv_output_size(self, size, kernel_size):
        return (size - kernel_size) + 1
    
    def __init__(self, cfg: LeNetConfig):
        super(LeNet5, self).__init__()

        self.cfg = cfg
        fc_hidden_size_1, fc_hidden_size_2 = cfg.fc_hidden_sizes
        conv_hidden_size_1, conv_hidden_size_2 = cfg.conv_hidden_sizes
        kernel_size = cfg.kernel_size
        # Get feature map size after two convolutions and two max pools
        self.feature_map_size = self._conv_output_size(cfg.image_size, kernel_size) // 2
        self.feature_map_size = self._conv_output_size(self.feature_map_size, kernel_size) // 2
        
        # Define parameters
        self.conv1 = nn.Conv2d(cfg.num_channels, conv_hidden_size_1, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(conv_hidden_size_1, out_channels=conv_hidden_size_2, kernel_size=kernel_size)

        self.fc1 = nn.Linear(conv_hidden_size_2 * self.feature_map_size * self.feature_map_size, 
                             fc_hidden_size_1)
        self.fc2 = nn.Linear(fc_hidden_size_1, fc_hidden_size_2)
        self.fc3 = nn.Linear(fc_hidden_size_2, cfg.num_labels)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        
        x = x.view(-1, 
                   self.cfg.conv_hidden_sizes[1] * self.feature_map_size * self.feature_map_size)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNetProbe(Probe):
    """Only defines a single size of probe"""
    def __init__(
            self,
            num_features: int, 
            num_classes: int = 2, 
            num_layers: int = 2,
            hidden_size: int = 2,
            device: str | torch.device | None = None, 
            dtype: torch.dtype | None = None,
            *,
            learning_rate: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            schedule_free: bool = False,
            base_shapes_path: str | None = None,
            conv_hidden_sizes: list[int] | None = None,
            fc_hidden_sizes: list[int] | None = None,
            **kwargs
        ):
        if not conv_hidden_sizes and not fc_hidden_sizes:
            print("Single probe size being used, input size ignored. Provide conv_hidden_sizes and fc_hidden_sizes.")
            
        super().__init__(num_features, num_classes, device, dtype)

        self.learning_rate = learning_rate
        self.betas = betas
        self.schedule_free = schedule_free
        self.mup = base_shapes_path is not None

        image_size = int(math.sqrt(num_features // 3))

        conv_hidden_sizes = conv_hidden_sizes or [hidden_size] * num_layers
        fc_hidden_sizes = fc_hidden_sizes or [hidden_size] * num_layers
        
        cfg = LeNetConfig(
            image_size=image_size,
            num_channels=3,
            conv_hidden_sizes=conv_hidden_sizes,
            kernel_sizes=[5, 5],
            fc_hidden_sizes=fc_hidden_sizes,
            num_labels=10
        )
        self.net = LeNet5(cfg).to(device=device, dtype=dtype)

        # Configure MuP
        self.net.fc3 = MuReadout(
            self.net.fc3.in_features,
            self.net.fc3.out_features,
            device=device,
            dtype=dtype,
            readout_zero_init=True
        )

        if base_shapes_path:
            base_shapes = load_base_shapes(base_shapes_path)
            set_base_shapes(self, base_shapes)

    
    def build_optimizer(self):
        opt_cls = AdamWScheduleFree if self.schedule_free else optim.AdamW
        if self.mup:
            return MuAdam(self.parameters(), opt_cls, lr=self.learning_rate, betas=self.betas)
        return opt_cls(self.parameters(), lr=self.learning_rate, betas=self.betas)

    def forward(self, x):
        return self.net(x)

