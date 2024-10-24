from argparse import ArgumentParser
from pathlib import Path
from functools import partial

import wandb
import torch
import torch.nn.functional as F
import torchvision as tv
from concept_erasure import LeaceFitter, OracleEraser, OracleFitter, QuadraticFitter, LeaceEraser
from torch import Tensor
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm
import lovely_tensors as lt
from torchvision.transforms.functional import to_tensor

from mdl.mlp_probe import ResMlpProbe, SeqMlpProbe, LinearProbe
from mdl.sweep import MdlResult, Sweep
from mdl.vision_probe import ViTProbe, ConvNextProbe, ResNetProbe

lt.monkey_patch()

torch.set_default_tensor_type(torch.DoubleTensor)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--erasers", type=str, nargs="+", choices=["none", "leace", "oleace", "qleace"], default=["none"])
    parser.add_argument("--net", type=str, choices=("mlp", "resmlp", "resnet", "convnext", "vit", "linear"))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    nontest = CIFAR10(root="/mnt/ssd-1/alexm/cifar10/", download=True)
    images, labels = zip(*nontest)
    X: Tensor = torch.stack(list(map(to_tensor, images))).to(device)
    Y = torch.tensor(labels).to(device)

    # Shuffle deterministically
    rng = torch.Generator(device=X.device).manual_seed(42)
    perm = torch.randperm(len(X), generator=rng, device=X.device)
    X, Y = X[perm], Y[perm]

    X_vec = X.view(X.shape[0], -1)
    k = int(Y.max()) + 1

    # Split train and validation
    val_size = 1024

    X_vec_train = X_vec[:-val_size]
    X_vec_val = X_vec[-val_size:]

    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]

    # Test set is entirely separate
    test = CIFAR10(root="/home/lucia/cifar10-test", train=False, download=True)
    test_images, test_labels = zip(*test)
    X_test: Tensor = torch.stack(list(map(to_tensor, test_images))).to(device)
    Y_test = torch.tensor(test_labels).to(device)

    # Populate eraser cache for training set if necessary 
    state_path = Path("erasers_cache") / f"cifar10_state.pth"
    state_path.parent.mkdir(exist_ok=True)
    state = {} if not state_path.exists() else torch.load(state_path)
    for eraser_str in args.erasers:
        if eraser_str == "none" or eraser_str in state:
            continue

        cls = {
            "leace": LeaceFitter,
            "oleace": OracleFitter,
            "qleace": QuadraticFitter,
        }[eraser_str]

        fitter = cls(3 * 32 * 32, k, dtype=torch.float64, device=device, shrinkage=False)
        for x, y in tqdm(zip(X_train, Y_train)):
            y = torch.as_tensor(y).view(1)
            if eraser_str != "qleace":
                y = F.one_hot(y, k)

            fitter.update(x.view(1, -1).to(device), y.to(device))

        state[eraser_str] = fitter.eraser
        torch.save(state, state_path)
    
    # Reduce size after eraser computation - cache does not differentiate between train set sizes
    if args.debug:
        X_train = X_train[:10_000]
        Y_train = Y_train[:10_000]
    
    model_cls = {
        "mlp": SeqMlpProbe,
        "resmlp": ResMlpProbe,
        "resnet": ResNetProbe,
        "vit": ViTProbe,
        "convnext": ConvNextProbe,
        "linear": LinearProbe,
    }[args.net]

    num_epochs = 1
    num_seeds = 1

    def reshape(x):
        "reshape tensor to CxHxW"
        return x.view(-1, X.shape[1], X.shape[2], X.shape[3])

    image_size = X.shape[-1]
    padding = round(image_size * 0.125)
    flattened_image_augmentor = tv.transforms.Compose(
        transforms=[
            tv.transforms.Lambda(reshape),
            tv.transforms.RandomCrop(image_size, padding=padding),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.Lambda(lambda x: x.flatten(1)),
        ]
    )
    # TODO ensure class hyperparameter setting works
    # TODO pass in a transform that moves batch to device if we need more epochs 
    sweep = Sweep(
        X.shape[1], k, device=X.device, dtype=torch.float64, # from bfloat16
        num_chunks=10,
        probe_cls=model_cls,
        probe_kwargs=dict(num_layers=args.depth, hidden_size=args.width), # , num_classes=k
    )


    def erase(x: Tensor, y: Tensor, eraser):
        assert y.ndim == 1
        assert x.ndim > 1 # otherwise requires unsqueeze

        # TODO figure out why this works without .flatten(1) now?
        if isinstance(eraser, LeaceEraser):
            return eraser(x).reshape(x.shape)
        elif isinstance(eraser, OracleEraser):
            return eraser(x, y).reshape(x.shape)
        else:
            return eraser(x, y)

    def none_transform(x, y):
        if args.net == "resnet":
            return x, y
        return x.flatten(1)

    data = {}
    for eraser_str in args.erasers:
        transform = partial(erase, eraser=state[eraser_str].to(device)) if eraser_str != "none" else none_transform

        results = []
        for seed in range(num_seeds):
            if not 'test' in args.name:
                run = wandb.init(project="mdl", entity="eleutherai", name=f'{eraser_str} {args.name} seed={seed}', config={'eraser': eraser_str, **vars(args)})
            else:
                run = None
            results.append(sweep.run(
                # Val and train are split in the sweep
                # was bfloat16
                X.double().repeat(num_epochs, 1, 1, 1).flatten(1), Y.repeat(num_epochs), seed=seed, 
                transform=transform, augment=flattened_image_augmentor, reduce_lr_on_plateau=False, logger=run
            ))
            if not 'test' in args.name:
                wandb.finish()

        data[eraser_str] = results

    data_path = Path("/mnt/ssd-1/lucia/results")
    data_path.mkdir(exist_ok=True)
    
    torch.save(data, data_path / f"{args.net}_h={args.width}_d={args.depth}_{args.name}.pth")