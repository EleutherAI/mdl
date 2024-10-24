import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
from typing import List, Optional

from mdl.probe import Probe

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
            device=device, dtype=dtype
        )
        self.bn1 = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
            device=device, dtype=dtype
        )
        self.bn2 = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False,
                    device=device, dtype=dtype
                ),
                nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_classes: int = 2,
        num_blocks: int = 2,
        hidden_size: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            3, hidden_size, kernel_size=3, stride=1, padding='same', bias=False,
            device=device, dtype=dtype
        )
        self.bn1 = nn.BatchNorm2d(hidden_size, device=device, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        
        self.stages = nn.ModuleList()
        in_channels = hidden_size
        current_channels = hidden_size
        for i in range(num_layers):
            # Double channels and reduce spatial dimensions every stage after the first
            if i > 0:
                current_channels *= 2
                stride = 2
            else:
                stride = 1
                
            stage = self._make_stage(
                in_channels, current_channels, num_blocks, stride,
                device=device, dtype=dtype
            )
            self.stages.append(stage)
            in_channels = current_channels
            
        # Global average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            current_channels, num_classes,
            device=device, dtype=dtype
        )
        
        # Initialize weights
        self._initialize_weights()

    def _make_stage(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_blocks: int, 
        stride: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> nn.Sequential:
        layers = []
        # First block may have stride > 1 to reduce spatial dimensions
        layers.append(BasicBlock(in_channels, out_channels, stride, device=device, dtype=dtype))
        
        # Remaining blocks maintain spatial dimensions
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, device=device, dtype=dtype))
            
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        
        for stage in self.stages:
            x = stage(x)
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# TODO make this faster
class ResNetProbe(Probe):
    """Probe based on a custom ResNet implementation with configurable layers."""

    def __init__(
        self,
        num_classes: int = 2,
        num_layers: int = 4,  # ResNet-18 configuration
        hidden_size: int = 128,
        learning_rate: float = 0.005,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        num_features: int = 3,
    ):
        super().__init__(num_features, num_classes, device, dtype)

        self.net = ResNet(
            num_layers=num_layers,
            num_classes=num_classes,
            # TODO check hyperparameters
            num_blocks=2,   
            hidden_size=hidden_size,
            device=device,
            dtype=dtype
        )
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device

        """
        import torch
        import torchvision
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
        data = next(iter(trainloader))[0]
        mean = data.mean(dim=[0,2,3])  # mean for each channel
        std = data.std(dim=[0,2,3])    # std for each channel
        print(f'mean: {mean}, std: {std}')
        # Results:
        # mean: tensor([0.4914, 0.4822, 0.4465])
        # std:  tensor([0.2470, 0.2435, 0.2616])
        """
        self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))
        self.mean_device = self.mean.to(self.device)
        self.std_device = self.std.to(self.device)

    def build_optimizer(self) -> optim.Optimizer:
        return optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 3, 32, 32)
        x = (x - self.mean_device) / self.std_device
        return self.net(x)