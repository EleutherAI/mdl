from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from concept_erasure.quadratic import QuadraticFitter
from concept_erasure.leace import LeaceFitter
from concept_erasure.alf_qleace import AlfQLeaceFitter
from torch import Tensor
from tqdm.auto import tqdm
import lovely_tensors as lt

from experiments.cli import get_cifar10, get_cifarnet, IdentityEraser


def get_alf_qleace(target_erasure=0.999, shrinkage=True):
    state_path = Path("data") / "erasers_cache" / f"alf_qleace.pth"
    state_path.parent.mkdir(exist_ok=True)
    state = {} if not state_path.exists() else torch.load(state_path, weights_only=False)
    
    key = f'alf_qleace_{target_erasure}_s={shrinkage}'
    if key not in state or args.nocache:
        fitter = AlfQLeaceFitter(
            num_features, k, dtype=dtype, device=device, shrinkage=shrinkage, target_erasure=target_erasure
        )

        Y_tensor = (F.one_hot(Y_train, k)).to(device)
        X_tensor = X_train.flatten(1).to(device).to(dtype)
        fitter.update(X_tensor, Y_tensor)

        if args.dataset == "cifarnet":
            fitter = fitter.to("cpu")

        state[key] = fitter.eraser
        torch.save(state, state_path)
    
    return state[key]


def get_erasers():
    # Populate eraser cache using training data
    state_path = Path("data") / "erasers_cache" / f"{args.dataset}_{dtype}_state.pth"
    state_path.parent.mkdir(exist_ok=True)
    state = {} if not state_path.exists() else torch.load(state_path, weights_only=False)
    
    for eraser_str in ["leace", "alf_qleace"]:
        if eraser_str not in state or args.nocache:
            cls = {
                "leace": LeaceFitter,
                "qleace": QuadraticFitter,
                "alf_qleace": AlfQLeaceFitter,
            }[eraser_str]

            fitter = cls(
                num_features, k, dtype=dtype, device=device, shrinkage=True
            )

            Y_tensor = (
                F.one_hot(Y_train, k)
            ).to(device)
            X_tensor = X_train.flatten(1).to(device).to(dtype)
            fitter.update(X_tensor, Y_tensor)

            if args.dataset == "cifarnet":
                fitter = fitter.to("cpu")

            state[eraser_str] = fitter.eraser
    
    return state['leace'], state['alf_qleace']

if __name__ == "__main__":
    lt.monkey_patch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=("cifar10", "cifarnet"), default="cifar10")
    parser.add_argument("--nocache", action="store_true")
    args = parser.parse_args()

    (X_train, Y_train, _, _, k, X, Y) = {
        "cifar10": get_cifar10('cuda'),
        "cifarnet": get_cifarnet(),
    }[args.dataset]
    num_features = X.shape[1] * X.shape[2] * X.shape[3]

    leace_eraser, alf_qleace_eraser = get_erasers()
    alf_qleace_eraser = get_alf_qleace(target_erasure=0.999)

    leace_erased: Tensor = leace_eraser.to(X_train.device)(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
    alf_qleace_erased: Tensor = alf_qleace_eraser(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)

    for erased_data, eraser_str in zip([leace_erased, alf_qleace_erased], ["leace", "alf_qleace"]):
        mean_barycenter = erased_data.mean(0)
        covariance_barycenter = torch.cov(erased_data.flatten(1).T)
        class_means = [erased_data[Y_train == c].mean(0) for c in Y_train.unique()]
        class_covariances = [erased_data[Y_train == c].flatten(1).T.cov() for c in Y_train.unique()]

        class_covariance_diffs = [covariance_barycenter - class_cov for class_cov in class_covariances]
        class_mean_diffs = [mean_barycenter - class_mean for class_mean in class_means]

        print("Eraser: ", eraser_str)
        print("Class means distance from barycenter after erasure:", [class_mean_diff.norm() for class_mean_diff in class_mean_diffs])
        print("Class covs diffs from barycenter after erasure:", [class_cov_diff.norm() for class_cov_diff in class_covariance_diffs])

        max_mean_diff = torch.stack([
            (mean_barycenter - other_mean).flatten().norm()
            for other_mean in class_means
        ]).max()
        max_pixel_diff = torch.stack([
            (mean_barycenter - other_mean).flatten().abs().max()
            for other_mean in class_means
        ]).max()
        print("Max mean difference from barycenter norm", max_mean_diff.item())
        print("Max pixel difference from barycenter", max_pixel_diff.item())

        max_diff_from_cov_center = torch.stack([
            (class_covariances[i] - covariance_barycenter).flatten().norm()
            for i in range(len(class_covariances))
        ]).max()
        max_pixel_diff_from_cov_center = torch.stack([
            (class_covariances[i] - covariance_barycenter).flatten().abs().max()
            for i in range(len(class_covariances))
        ]).max()
        print("Max covariance difference from barycenter norm", max_diff_from_cov_center.item())
        print("Max covariance difference from barycenter pixel", max_pixel_diff_from_cov_center.item())