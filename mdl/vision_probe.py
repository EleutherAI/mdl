import math
import torch
import torchvision as tv
from torch import Tensor, nn, optim
from transformers import (
    ConvNextV2Config, ConvNextV2ForImageClassification, 
    SwinForImageClassification, SwinConfig
)
from mup import MuReadout, MuAdam, MuSGD, load_base_shapes, set_base_shapes
from schedulefree import AdamWScheduleFree

from .probe import Probe

 
class VisionProbe(Probe):
    """Probe based on a TorchVision model. Defaults to ResNet-18."""

    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = 0.005,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        model: str = "resnet18",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        num_features: int = 3,  # Unused
        pretrained: bool = False,
        base_shapes_path: str | None = None,
        **kwargs
    ):
        super().__init__(num_features, num_classes, device, dtype)

        if not pretrained:
            net = tv.models.get_model(model, num_classes=num_classes)
        else:
            net = tv.models.resnet18(pretrained=pretrained)
            net.fc = nn.Linear(net.fc.in_features, num_classes)

        self.net = net.to(device=device, dtype=dtype)  # type: ignore
        if model == "resnet18":
            self.net.conv1 = torch.nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.net.maxpool = torch.nn.Identity(device=device, dtype=dtype)

        self.mup = base_shapes_path is not None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.norm = tv.transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )
        if model == "resnet18" and not pretrained:
            net.conv1 = nn.Conv2d(
                3,
                64,
                3,
                stride=1,
                padding="same",
                bias=False,
                device=device,
                dtype=dtype,
            )
            net.maxpool = nn.Identity()

    def build_optimizer(self) -> optim.Optimizer:
        opt_cls = MuSGD if self.mup else optim.SGD
        return opt_cls(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.norm(x))


class ConvNextProbe(Probe):
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
            arch: str | None = "atto",
            **kwargs
        ):
        from transformers import ConvNextV2Config, ConvNextV2ForImageClassification
        
        super().__init__(num_features, num_classes, device, dtype)

        self.learning_rate = learning_rate
        self.betas = betas
        self.schedule_free = schedule_free
        self.mup = base_shapes_path is not None

        match arch:
            case "atto" | "":  # default
                depths = [2, 2, 6, 2]
                hidden_sizes = [40, 80, 160, 320]
            case "femto":
                depths = [2, 2, 6, 2]
                hidden_sizes = [48, 96, 192, 384]
            case "pico":
                depths = [2, 2, 6, 2]
                hidden_sizes = [64, 128, 256, 512]
            case "nano":
                depths = [2, 2, 8, 2]
                hidden_sizes = [80, 160, 320, 640]
            case "tiny":
                depths = [3, 3, 9, 3]
                hidden_sizes = [96, 192, 384, 768]
            case other:
                raise ValueError(f"Unknown ConvNeXt architecture {other}")

        image_size = int(math.sqrt(num_features // 3))

        cfg = ConvNextV2Config(
            image_size=image_size,
            depths=depths,
            drop_path_rate=0.1,
            hidden_sizes=hidden_sizes,
            num_labels=num_classes,
            # The default of 4 x 4 patches shrinks the image too aggressively for
            # low-resolution images like CIFAR-10
            patch_size=1,
        )
        self.net = ConvNextV2ForImageClassification(cfg).to(device=device, dtype=dtype) # type: ignore

        # Configure MuP
        self.net.classifier = MuReadout(
            self.net.classifier.in_features,
            self.net.classifier.out_features,
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
        return self.net(x).logits


class SwinProbe(Probe):
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
            arch: str | None = "atto",
            **kwargs
        ) -> None:
        from torchvision.models.swin_transformer import (
            PatchMergingV2,
            SwinTransformer,
            SwinTransformerBlockV2,
        )

        super().__init__(num_features, num_classes, device, dtype)

        self.learning_rate = learning_rate
        self.betas = betas
        self.schedule_free = schedule_free
        self.mup = base_shapes_path is not None

        match arch:
            case "atto":
                num_heads = [2, 4, 8, 16]
                embed_dim = 40
            case "femto":
                num_heads = [2, 4, 8, 16]
                embed_dim = 48
            case "pico":
                num_heads = [2, 4, 8, 16]
                embed_dim = 64
            case "nano":
                num_heads = [2, 4, 8, 16]
                embed_dim = 80
            case "tiny" | "":  # default
                num_heads = [3, 6, 12, 24]
                embed_dim = 96
            case other:
                raise ValueError(f"Unknown Swin architecture {other}")

        # Tiny architecture with 2 x 2 patches
        self.net = SwinTransformer(
            patch_size=[2, 2],
            embed_dim=embed_dim,
            depths=[2, 2, 6, 2],
            num_heads=num_heads,
            window_size=[7, 7],
            num_classes=num_classes,
            stochastic_depth_prob=0.2,
            block=SwinTransformerBlockV2,
            downsample_layer=PatchMergingV2,
        )
        # Configure MuP        
        self.net.head = MuReadout(
            self.net.head.in_features,
            self.net.head.out_features,
            device=device,
            dtype=dtype,
            readout_zero_init=True
        )

        self.net = torch.compile(self.net).to(device=device, dtype=dtype)
        
        if base_shapes_path:
            base_shapes = load_base_shapes(base_shapes_path)
            set_base_shapes(self, base_shapes)

    def build_optimizer(self):
        opt_cls = AdamWScheduleFree if self.schedule_free else optim.AdamW
        if self.mup:
            return MuAdam(self.parameters(), opt_cls, lr=self.learning_rate, betas=self.betas)
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=self.betas)

    def forward(self, x):
        return self.net(x)