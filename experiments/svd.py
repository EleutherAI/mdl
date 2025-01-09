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

from experiments.cli import get_cifar10, get_cifarnet, IdentityEraser, load_eraser


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


    for eraser_str in ('leace', 'alf_qleace'):
        eraser = state[eraser_str]

        erased = eraser.to('cuda')(X_train.cuda().reshape(len(X_train), -1)).reshape(X_train.shape)
        class_means = [erased[Y_train == c].mean(0) for c in Y_train.unique()]
    
        universal_mean = torch.stack(class_means).mean(0)
        diffs_from_global_mean = [class_mean - torch.stack(class_means).mean(0) for class_mean in class_means]
        print(f"Class mean diffs from global mean after {eraser_str} erasure:", diffs_from_global_mean)

        max_mean_diff = torch.stack([
            (universal_mean - other_mean).flatten().norm()
            for other_mean in class_means
        ]).max()
        max_pixel_diff = torch.stack([
            (universal_mean - other_mean).flatten().abs().max()
            for other_mean in class_means
        ]).max()
        print("Max difference norm", max_mean_diff)
        print("Max pixel difference", max_pixel_diff)

        # universal_covariance = torch.cov(erased.flatten(1).T)
        # print("Universal covariance:", universal_covariance)

        class_covariances = [erased[Y_train == c].flatten(1).T.cov() for c in Y_train.unique()]
        max_diff_between_any_two_covariances = torch.stack([
            (class_covariances[i] - class_covariances[j]).flatten().norm()
            for i in range(len(class_covariances))
            for j in range(i + 1, len(class_covariances))
        ]).max()
        max_pixel_diff_between_any_two_covariances = torch.stack([
            (class_covariances[i] - class_covariances[j]).flatten().abs().max()
            for i in range(len(class_covariances))
            for j in range(i + 1, len(class_covariances))
        ]).max()
        # max_covariance_diff = torch.stack([
        #     (universal_covariance - other_covariance).flatten().norm()
        #     for other_covariance in class_covariances
        # ]).max()

        # Print covariance traces
        print(f"Max covariance difference norm for {eraser_str}", max_diff_between_any_two_covariances)
        print(f"Max covariance difference pixel for {eraser_str}", max_pixel_diff_between_any_two_covariances)

        # leace_eraser = state["leace"]
        # leace_erased: Tensor = leace_eraser.to(X_train.device)(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)

        # leace_class_means = [leace_erased[Y_train == c].mean(0) for c in Y_train.unique()]
        # leace_universal_mean = torch.stack(leace_class_means).mean(0)
        # leace_max_mean_diff = torch.stack([
        #     (leace_universal_mean - other_mean).flatten().norm()
        #     for other_mean in leace_class_means
        # ]).max()
        # leace_max_pixel_diff = torch.stack([
        #     (leace_universal_mean - other_mean).flatten().abs().max()
        #     for other_mean in leace_class_means
        # ]).max()
        # print("Max LEACE difference norm", leace_max_mean_diff)
        # print("Max LEACE pixel difference", leace_max_pixel_diff)

        # leace_universal_covariance = torch.cov(leace_erased.flatten(1).T)
        # print("LEACE universal covariance:", leace_universal_covariance)

        # leace_class_covariances = [leace_erased[Y_train == c].flatten(1).T.cov() for c in Y_train.unique()]
        # leace_max_covariance_diff = torch.stack([
        #     (leace_universal_covariance - other_covariance).flatten().norm()
        #     for other_covariance in leace_class_covariances
        # ]).max()
        # leace_max_diff_between_any_two_covariances = torch.stack([
        #     (leace_class_covariances[i] - leace_class_covariances[j]).flatten().norm()
        #     for i in range(len(leace_class_covariances))
        #     for j in range(i + 1, len(leace_class_covariances))
        # ]).max()
        # leace_max_pixel_diff_between_any_two_covariances = torch.stack([
        #     (leace_class_covariances[i] - leace_class_covariances[j]).flatten().abs().max()
        #     for i in range(len(leace_class_covariances))
        #     for j in range(i + 1, len(leace_class_covariances))
        # ]).max()

        # print("Max LEACE covariance difference norm", leace_max_diff_between_any_two_covariances)
        # print("Max LEACE covariance difference pixel", leace_max_pixel_diff_between_any_two_covariances)

        # Cov traces

        # 190.42 CIFAR-10
        unerased_cov = X_train.flatten(1).T.cov()
        print(f"Unerased Cov trace: {unerased_cov.trace().item():.2f}") 
        # 135.79 CIFAR-10
        # leace_cov = leace_erased.flatten(1).T.cov()
        # print(f"LEACE Cov trace: {leace_cov.trace().item():.2f}") 
        # 24.79 CIFAR-10
        cov = erased.flatten(1).T.cov()
        print(f"{eraser_str} Cov trace: {cov.trace().item():.2f}") 

        # Unerased eigenvalues spectrum of covariance
        # SVD of centered data, singular values = square roots of eigenvalues of covariance matrix
        # SVD on covariance matrix, identical to eigenvalues

        # unerased_eigenvals = torch.linalg.eigvalsh(unerased_cov)
        # unerased_flipped = torch.cat((torch.tensor([1], device=unerased_eigenvals.device), unerased_eigenvals.flip(dims=(0,))))
        # qleace_eigenvals = torch.linalg.eigvalsh(cov)
        # qleace_flipped = torch.cat((torch.tensor([1], device=qleace_eigenvals.device), qleace_eigenvals.flip(dims=(0,))))
        # # leace_eigenvals = torch.linalg.eigvalsh(leace_cov)
        # # leace_flipped = torch.cat((torch.tensor([1], device=leace_eigenvals.device), leace_eigenvals.flip(dims=(0,))))

        # all_flipped = torch.cat([unerased_flipped, qleace_flipped, leace_flipped])
        # global_min = torch.log(torch.min(all_flipped)).cpu()
        # global_max = torch.log(torch.max(all_flipped)).cpu()


        # fig = px.line(x=range(len(unerased_flipped)), y=unerased_flipped.cpu(), title="Unerased data covariance eigenvalues spectrum", log_x=True, log_y=True)
        # fig.update_layout(xaxis_title="Reversed eigenvalue index", yaxis_title="Eigenvalue", yaxis_range=[global_min, global_max])
        # fig.write_image("svd_unerased.png")

        # # QLEACE eigenvalues spectrum
        
        # fig = px.line(x=range(len(qleace_flipped)), y=qleace_flipped.cpu(), title="QLEACE data covariance eigenvalues spectrum", log_x=True, log_y=True)
        # fig.update_layout(xaxis_title="Reversed eigenvalue index", yaxis_title="Eigenvalue", yaxis_range=[global_min, global_max])
        # fig.write_image("svd_qleace.png")

        # # LEACE eigenvalues spectrum
        # leace_fig = px.line(x=range(len(leace_flipped)), y=leace_flipped.cpu(), title="LEACE data covariance eigenvalues spectrum", log_x=True, log_y=True)
        # leace_fig.update_layout(xaxis_title="Reversed eigenvalue index", yaxis_title="Eigenvalue", yaxis_range=[global_min, global_max])
        # leace_fig.write_image("svd_leace.png")


        # Average std of each pixel across the unerased data
        unerased_std = X_train.std(dim=0).mean()
        print(f"Unerased std: {unerased_std:.2f}")

        # Average std of each pixel across the QLEACE data
        std = erased.std(dim=0).mean()
        print(f"{eraser_str} std: {std:.2f}")

        # # Average std of each pixel across the LEACE data
        # leace_std = leace_erased.std(dim=0).mean()
        # print(f"LEACE std: {leace_std:.2f}")
