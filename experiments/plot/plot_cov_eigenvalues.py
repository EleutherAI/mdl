from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from concept_erasure.quadratic import QuadraticFitter
from concept_erasure.leace import LeaceFitter
from concept_erasure.alf_qleace import AlfQLeaceFitter
from torch import Tensor
import lovely_tensors as lt
import plotly.express as px

from experiments.cli import get_cifar10, get_cifarnet, IdentityEraser


if __name__ == "__main__":
    lt.monkey_patch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    parser = ArgumentParser()
    parser.add_argument("--eraser", type=str, choices=("control", "leace", "oleace", "qleace", "alf_qleace"), default="control")
    parser.add_argument("--dataset", type=str, choices=("cifar10", "cifarnet"), default="cifar10")
    parser.add_argument("--nocache", action="store_true")
    args = parser.parse_args()

    (X_train, Y_train, X_val, Y_val, k, X, Y) = {
        "cifar10": get_cifar10(device="cuda"),
        "cifarnet": get_cifarnet(),
    }[args.dataset]

    num_features = X.shape[1] * X.shape[2] * X.shape[3]

    # Populate eraser cache using training data
    state_path = Path("data") / "erasers_cache" / f"{args.dataset}_{dtype}_state.pth"
    state_path.parent.mkdir(exist_ok=True)
    state = {} if not state_path.exists() else torch.load(state_path, weights_only=False)

    if args.eraser not in state or args.nocache:
        if args.eraser == "control":
            state[args.eraser] = IdentityEraser()
        else:
            cls = {
                "leace": LeaceFitter,
                "qleace": QuadraticFitter,
                "alf_qleace": AlfQLeaceFitter,
            }[args.eraser]

            dtype = torch.float32

            fitter = cls(
                num_features, k, dtype=dtype, device=device, shrinkage=True
            )

            Y_tensor = (
                F.one_hot(Y_train, k)
                if args.eraser != "qleace"
                else Y_train
            ).to(device)
            X_tensor = X_train.flatten(1).to(device).to(dtype)
            fitter.update(X_tensor, Y_tensor)

            if args.dataset == "cifarnet":
                fitter = fitter.to("cpu")
            eraser = fitter.eraser

            state[args.eraser] = fitter.eraser
            torch.save(state, state_path)

    eraser = state[args.eraser]


    # Unerased SVD
    def get_flipped_eigenvalues(data: Tensor, log=True):
        if not log:
            raise NotImplementedError("Only log scale is supported")
        
        cov = data.flatten(1).T.cov()
        eigenvals = torch.linalg.eigvalsh(cov)
        
        # Add 1 to allow log scale
        return torch.cat((torch.tensor([1], device=eigenvals.device), eigenvals.flip(dims=(0,))))

    # SVD of centered data, singular values = square roots of eigenvalues of covariance matrix
    # SVD on covariance matrix, identical to eigenvalues

    # Eigenvalues of data covariance
    flipped_eigenvalues = {
        'control': get_flipped_eigenvalues(X_train, log=True).cpu()
    }
    
    for eraser_str in ('leace', 'qleace', 'alf_qleace'):
        eraser = state[eraser_str].to('cuda')
        erased = (
            eraser(X_train.cuda().flatten(1), Y_train)
            if eraser_str == "qleace"
            else eraser(X_train.flatten(1)).reshape(X_train.shape)
        )

        flipped_eigenvalues[eraser_str] = get_flipped_eigenvalues(erased, log=True).cpu()

    all_flipped = torch.cat(list(flipped_eigenvalues.values()))
    global_min = torch.log(torch.min(all_flipped)).item()
    global_max = torch.log(torch.max(all_flipped)).item()

    for eraser_str, erased_flipped in flipped_eigenvalues.items():
        fig = px.line(x=range(len(erased_flipped)), y=erased_flipped, title=f"{eraser_str} data covariance eigenvalues spectrum", log_x=True, log_y=True)
        fig.update_layout(xaxis_title="Reversed eigenvalue index", yaxis_title="Eigenvalue", yaxis_range=[global_min, global_max])
        fig.write_image(f"svd_{eraser_str}.png")