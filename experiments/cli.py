from argparse import ArgumentParser
from pathlib import Path
from functools import partial

import wandb
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2.functional import to_tensor
# from concept_erasure import OracleFitter, QuadraticFitter
from concept_erasure.quadratic import QuadraticEraser, QuadraticFitter
from concept_erasure.leace import LeaceEraser, LeaceFitter
from concept_erasure.alf_qleace import AlfQLeaceFitter, AlfQLeaceEraser
from torch import Tensor
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm
import lovely_tensors as lt


from mdl.mlp_probe import ResMlpProbe, MlpProbe, LinearProbe
from mdl.sweep import Sweep
from mdl.vision_probe import ConvNextProbe, VisionProbe, SwinProbe, VisionProbeMain
from mdl.resnet_probe import ResNetProbe

lt.monkey_patch()
torch.set_default_tensor_type(torch.DoubleTensor)


if __name__ == "__main__":
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--out", type=str, default='results')
    parser.add_argument("--erasers", type=str, nargs="+", choices=["none", "leace", "oleace", "qleace", "qleace2"], default=["none"])
    parser.add_argument("--net", type=str, choices=("mlp", "resmlp", "resnet", "convnext", "linear", "vision", "swin", "vision_main"))
    parser.add_argument("--optimizer", type=str, choices=("adamw", "schedulefree"), default="adamw")
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_seeds", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nocache", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--all_data", action="store_true")
    args = parser.parse_args()

    # nontest = CIFAR10(root="/mnt/ssd-1/alexm/cifar10/", download=True)
    nontest = CIFAR10(
        "/home/lucia/cifar10", download=True
    ) # transform=tv.transforms.ToTensor()

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

    # Populate eraser cache for training set if necessary 
    state_path = Path("erasers_cache") / f"cifar10_state_2.pth"
    state_path.parent.mkdir(exist_ok=True)
    state = {} if not state_path.exists() else torch.load(state_path)
    for eraser_str in args.erasers:
        if (eraser_str == "none" or (eraser_str in state and not args.nocache)):
            continue

        cls = {
            "leace": LeaceFitter,
            # "oleace": OracleFitter,
            "qleace": QuadraticFitter,
            "qleace2": AlfQLeaceFitter,
        }[eraser_str]

        fitter = cls(3 * 32 * 32, k, dtype=torch.float64, device=device, shrinkage=True)
        for x, y in tqdm(zip(X_train, Y_train)):
            y = torch.as_tensor(y).view(1)
            if eraser_str != "qleace":
                y = F.one_hot(y, k)

            fitter.update(x.view(1, -1).to(device), y.to(device))

        state[eraser_str] = fitter.eraser
        torch.save(state, state_path)

    # Reduce size after eraser computation - cache does not differentiate between train set sizes
    # if args.debug:
    #     X_train = X_train[:10_000]
    #     Y_train = Y_train[:10_000]
    
    model_cls = {
        "mlp": MlpProbe,
        "resmlp": ResMlpProbe,
        "resnet": ResNetProbe,
        "convnext": ConvNextProbe,
        "linear": LinearProbe,
        "vision": VisionProbe,
        "swin": SwinProbe,
        "vision_main": VisionProbeMain
    }[args.net]

    flatten = {
        "mlp": True,
        "resmlp": True,
        "resnet": False,
        "convnext": False,
        "linear": True,
        "vision": False,
        "vision_main": False,
        "swin": False,
    }

    image_size = X.shape[-1]
    padding = round(image_size * 0.125)

    if flatten[args.net]:
        def reshape(x):
            "reshape tensor to CxHxW"
            return x.view(-1, X.shape[1], X.shape[2], X.shape[3])

        augment = transforms.Compose([
            transforms.Lambda(reshape),
            transforms.RandomCrop(image_size, padding=padding), 
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.flatten(1))
        ])
    else:
        augment = transforms.Compose([
            # TODO run with randaugment
            transforms.RandomCrop(image_size, padding=padding), 
            transforms.RandomHorizontalFlip(),
            # transforms.RandAugment()
        ])

    
        

    def none_transform(x, y):
        if not flatten[args.net]:
            return x
        return x.flatten(1)

    data = {}
    for eraser_str in args.erasers:
        # I don't trust python to cache the result of this
        if eraser_str == "leace" or eraser_str == "qleace2":
            def erase(x: Tensor, y: Tensor, eraser):
                # assert y.ndim == 1
                # assert x.ndim > 1 # otherwise requires unsqueeze
                x_erased = eraser(x.flatten(1))
                return x_erased if flatten[args.net] else x_erased.reshape_as(x)
        else:
            def erase(x: Tensor, y: Tensor, eraser):
                x_erased = eraser(x.flatten(1), y)
                return x_erased if flatten[args.net] else x_erased.reshape_as(x)

        transform = (
            partial(erase, eraser=state[eraser_str].to(device)) 
            if eraser_str != "none" 
            else none_transform
        )

        results = []
        for seed in range(args.num_seeds):
            if not 'test' in args.name and not args.debug:
                wandb_name = f'{eraser_str if eraser_str != "none" else "baseline"} {args.name} w={args.width} d={args.depth} s={seed} {args.net} lr={args.lr} b1={args.b1}'
                run = wandb.init(
                    project="mdl", entity="eleutherai", name=wandb_name, config={'eraser': eraser_str, **vars(args)}
                )
            else:
                run = None

            name = None if not args.save else f"{args.net}_h={args.width}_d={args.depth}_{'_'.join(args.erasers)}_{args.name}"                

            if not args.all_data:
                if model_cls == MlpProbe:
                    print("mup not enabled..."); exit(0)
                    sweep = Sweep(
                        X.shape[1] * X.shape[2] * X.shape[3], k, device=X.device, dtype=torch.float64,
                        num_chunks=10, logger=run, name=name, probe_cls=model_cls,
                        probe_kwargs=dict(num_layers=args.depth, hidden_size=args.width, lr=args.lr, optimizer=args.optimizer, betas=(args.b1, 0.999), mup=True),
                    )
                else:
                    sweep = Sweep(
                        X.shape[1] * X.shape[2] * X.shape[3], k, device=X.device, dtype=torch.float64,
                        num_chunks=10, logger=run, name=name,
                        probe_cls=model_cls,
                        probe_kwargs=dict(num_layers=args.depth, hidden_size=args.width),
                    )

                results.append(sweep.run(
                    X.double(), Y, seed=seed, transform=transform, 
                    augment=augment, reduce_lr_on_plateau=False, max_epochs=200, early_stop_epochs=30
                ))
            else:
                if model_cls == MlpProbe:
                    # Call mup base stuff
                    from mup import make_base_shapes, set_base_shapes
                    num_features = X.shape[1] * X.shape[2] * X.shape[3]

                    base_model = model_cls(num_classes=k, num_features=num_features, num_layers=2, hidden_size=128, mup=True)
                    delta_model = model_cls(num_classes=k, num_features=num_features, num_layers=2, hidden_size=2, mup=True)
                    probe = model_cls(
                        num_classes=k, num_features=num_features, num_layers=args.depth, hidden_size=args.width, 
                        device=device, lr=args.lr, optimizer=args.optimizer, betas=(args.b1, 0.999), mup=True
                    )
                    set_base_shapes(probe, base_model, delta=delta_model, savefile=f'mup-{args.net}.bsh')

                else:
                    probe = model_cls(
                        num_classes=k, num_features=X.shape[1] * X.shape[2] * X.shape[3], 
                        num_layers=args.depth, hidden_size=args.width, device=device
                    )

                probe.fit(X_train, Y_train, x_val=X_val, y_val=Y_val, 
                          seed=seed, transform=transform, augment=augment, 
                          max_epochs=20_000, logger=run, early_stop_epochs=1_000)
                wandb.finish()
            
            if not 'test' in args.name and not args.debug:
                wandb.finish()

        if args.all_data:
            exit(0)
        
        data[eraser_str] = results

    data_path = Path(f"/mnt/ssd-1/lucia/{args.out}" if not args.debug else f"/mnt/ssd-1/lucia/debug-{args.out}")
    data_path.mkdir(exist_ok=True)
    
    torch.save(data, data_path / f"{args.net}_h={args.width}_d={args.depth}_{'_'.join(args.erasers)}_{args.name}.pth")