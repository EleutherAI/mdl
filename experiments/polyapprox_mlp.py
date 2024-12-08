from pathlib import Path

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2.functional import to_tensor
from polyapprox.ols import ols
from mdl.mlp_probe import MlpProbe


def get_cifar10_mean():
    nontest = CIFAR10("/home/lucia/cifar10", download=True)

    images, labels = zip(*nontest)
    X = torch.stack(list(map(to_tensor, images)))
    X = X.view(X.shape[0], -1)

    return X.mean(dim=0)


# Load each MLP checkpoint ols 
ckpts = list(Path("probe-ckpts").glob("*.pth"))
ols_results = []
for ckpt in ckpts:
    if 'normalize' not in ckpt.name:
        continue

    probe = MlpProbe(num_features=32*32*3, num_classes=10, hidden_size=128, num_layers=1)
    probe.load_state_dict(torch.load(ckpt))
    probe.eval()

    ols_results.append(ols(
        probe.net[0].weight.data.double().numpy(), probe.net[0].bias.data.double().numpy(), 
        probe.net[2].weight.data.double().numpy(), probe.net[2].bias.data.double().numpy(), 
        act="relu", order="quadratic",
        return_fvu=True
    ))

torch.save(ols_results, "polyapprox_mlp.pth")
    

