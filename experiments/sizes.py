from argparse import ArgumentParser

import torch
from torchvision.transforms.v2.functional import to_tensor
from torch import Tensor
from torchvision.datasets import CIFAR10
import lovely_tensors as lt
from mup import make_base_shapes
import random

from mdl.mlp_probe import ResMlpProbe, MlpProbe, LinearProbe
from mdl.vision_probe import ConvNextProbe, VisionProbe, SwinProbe
from mdl.resnet_probe import ResNetProbe
from experiments.sweep_eraser import sweep_params
lt.monkey_patch()
torch.set_default_tensor_type(torch.DoubleTensor)

import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np


def summary(model, input_size, dtype, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size, device=device).type(dtype) for in_size in input_size]

    summary = OrderedDict()

    hooks = []
    model.apply(register_hook)
    model(*x)
    for h in hooks:
        h.remove()

    total_params = 0
    for layer in summary:
        total_params += summary[layer]["nb_params"]

    # assume 4 bytes/number (float on cuda).
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))

    return total_params_size


def get_cifar10(device):
    nontest = CIFAR10("/home/lucia/cifar10", download=True)

    images, labels = zip(*nontest)
    X: Tensor = torch.stack(list(map(to_tensor, images))).to(device)
    Y = torch.tensor(labels).to(device)

    # Shuffle deterministically
    rng = torch.Generator(device=X.device).manual_seed(42)
    perm = torch.randperm(len(X), generator=rng, device=X.device)
    X, Y = X[perm], Y[perm]

    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024

    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    # Test set is entirely separate
    test = CIFAR10(root="/home/lucia/cifar10-test", train=False, download=True)
    test_images, test_labels = zip(*test)
    X_test: Tensor = torch.stack(list(map(to_tensor, test_images))).to(device)
    Y_test = torch.tensor(test_labels).to(device)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, k, X, Y


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (X_train, Y_train, X_val, Y_val, X_test, Y_test, k, X, Y) = get_cifar10(device)

    num_features = X.shape[1] * X.shape[2] * X.shape[3]

    model_cls = {
        "mlp": MlpProbe,
        "convnext": ConvNextProbe,
        "swin": SwinProbe,
    }
    flatten = {
        "mlp": True,
        "convnext": False,
        "swin": False,
    }

    sizes = []
    for key in model_cls.keys():
        class_ = model_cls[key]

        mup_width = sweep_params[key]['mup_width']
        mup_depth = sweep_params[key]['mup_depth']

        base_shapes_path = f"mup-{key}-{mup_width}-{mup_depth}.bsh"
        base_model = class_(
            num_classes=k,
            num_features=num_features,
            num_layers=mup_depth,
            hidden_size=mup_width,
        )
        delta_model = class_(
            num_classes=k,
            num_features=num_features,
            num_layers=mup_depth,
            hidden_size=mup_width + 1, # Any value other than mup_width
        )
        make_base_shapes(base_model, delta_model, savefile=base_shapes_path)

        widths = sweep_params[key]['widths']
        for width in widths:
            probe = class_(
                num_classes=k,
                num_features=num_features,
                num_layers=mup_depth,
                hidden_size=width,
                schedule_free=True,
                base_shapes_path=base_shapes_path,
                device=device,
                dtype=torch.float32,
            )

            print(class_.__name__ if hasattr(class_, "__name__") else '', width, torch.float32)
            sizes.append(
                summary(probe, (num_features,) if flatten[key] else (3, 32, 32), dtype=probe.dtype)
            )

        depths = sweep_params[key]['depths']
        for depth in depths:
            # Re-initialize at every depth to prevent muP lack of support for depth wise scaling 
            # https://github.com/microsoft/mup/issues/54
            base_shapes_path = f"mup-{key}-{mup_width}-{mup_depth}-{random.random()}.bsh"
            base_model = class_(
                num_classes=k,
                num_features=num_features,
                num_layers=depth,
                hidden_size=mup_width,
            )
            delta_model = class_(
                num_classes=k,
                num_features=num_features,
                num_layers=depth,
                hidden_size=mup_width, 
            )
            make_base_shapes(base_model, delta_model, savefile=base_shapes_path)
            
            probe = class_(
                num_classes=k,
                num_features=num_features,
                num_layers=depth,
                hidden_size=mup_width,
                schedule_free=True,
                base_shapes_path=base_shapes_path,
                device=device,
                dtype=torch.float32,
            )
            print(class_.__name__ if hasattr(class_, "__name__") else '', depth, torch.float32)

            sizes.append(
                summary(probe, (num_features,) if flatten[key] else (3, 32, 32), dtype=probe.dtype)
            )

    print("Mean model size (MB): %0.2f" % np.mean(sizes))
    print("Number of models: %d" % len(sizes))