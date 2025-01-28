# Do reparametrization with Adam
# Ensure clipping is done
# Clip after LEACE which brings outside the hypercube


from pathlib import Path
from simple_parsing import ArgumentParser
from dataclasses import dataclass

import torch
from torch import nn, optim, Tensor
import torchvision.utils as vutils
from torchvision import transforms
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
from concept_erasure import assert_type, groupby, optimal_linear_shrinkage
from PIL import Image as PilImage
from huggingface_hub import HfApi
import lovely_tensors as lt

from experiments.cli import get_cifar10, load_eraser, get_cifarnet, get_fake_cifar10, get_fake_cifarnet, get_svhn, get_fake_svhn, IdentityEraser
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2.functional import to_dtype, to_image
import torch.nn.functional as F
from concept_erasure import assert_type, groupby, optimal_linear_shrinkage
from dataclasses import dataclass
from typing import Literal

lt.monkey_patch()


def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

hyperparameters = {
    "cifar10": {
        "mse_weight": 1.,
        "cov_weight": 0.01,
        "mean_weight": 0.01,
    },
    "cifarnet": {
        "mse_weight": 1e-9,
        "cov_weight": 0.5,
        "mean_weight": 1.,
    },
    "svhn": {
        "mse_weight": 1.,
        "cov_weight": 0.01,
        "mean_weight": 0.01,
    },
}

@dataclass
class CombinedErasureArgs:
    dataset: str = "cifar10"
    num_classes: int = 10
    max_iter: int = 100
    prefix: str = "erased"
    mse_weight: float | None = None
    cov_weight: float | None = None
    mean_weight: float | None = None
    method: Literal["leace", "orth", "none"] = "leace"
    shrinkage: bool = False
    linear_cache_key: str | None = None
    
    def __post_init__(self):
        # Set default weights based on dataset if not provided
        if self.dataset in hyperparameters:
            if self.mse_weight is None:
                self.mse_weight = hyperparameters[self.dataset]["mse_weight"]
            if self.cov_weight is None:
                self.cov_weight = hyperparameters[self.dataset]["cov_weight"]
            if self.mean_weight is None:
                self.mean_weight = hyperparameters[self.dataset]["mean_weight"]

def transform_with_combined_erasure(args: CombinedErasureArgs, X: Tensor, Y: Tensor, cached_linear_eraser):
    """
    Transform data by first applying cached linear erasure, then quadratic erasure.
    
    Args:
        args: Configuration parameters
        X: Input tensor of shape [N, C, H, W] 
        Y: Target tensor of shape [N]
        cached_linear_eraser: Pre-computed linear eraser from cache
    """
    device = X.device
    flattened_X = X.flatten(1)
    n, d = flattened_X.shape
    
    # Step 1: Apply cached linear eraser if provided
    flattened_X = cached_linear_eraser.to(flattened_X.device)(flattened_X)

    
    # Calculate global statistics for quadratic erasure
    global_mean = flattened_X.mean(0)
    global_cov = optimal_linear_shrinkage(flattened_X.mT.cov(), len(X))
    
    def transform_to_statistics(data: Tensor, target_mean: Tensor, target_cov: Tensor):
        """Transform data points to match target statistics while preserving structure."""
        eps = torch.finfo(data.dtype).eps
        x = torch.clamp(data, eps, 1 - eps)
        z = nn.Parameter(x.logit())

        print(z.device, 'z device')
        
        target_mean = torch.clamp(target_mean, eps, 1 - eps)
        target_mean = target_mean.logit().sigmoid()
        
        target_cov = torch.clamp(target_cov, eps, 1 - eps)
        target_cov = target_cov.logit().sigmoid()
        
        target_data = x.logit().sigmoid()
        
        opt = optim.LBFGS([z], line_search_fn="strong_wolfe", max_iter=args.max_iter)
        
        def closure():
            opt.zero_grad()
            x = z.sigmoid()
            mean_loss = torch.norm(x.mean(0) - target_mean)
            cov_loss = torch.norm(x.T.cov() - target_cov)
            mse_loss = ((x - target_data) ** 2).mean((0, 1))
            
            loss = (mean_loss * args.mean_weight) + \
                   (cov_loss * args.cov_weight) + \
                   (args.mse_weight * mse_loss)
            
            print(f"loss {loss}, mean loss {(mean_loss * args.mean_weight).item()}, cov loss {(cov_loss * args.cov_weight).item()}, weighted mse loss {(mse_loss * args.mse_weight).item()}")
            
            loss.backward()
            return float(loss)
        
        opt.step(closure)
        return z.sigmoid().detach()
    
    transformed_data = []
    transformed_labels = []
    
    # Transform each class to match global statistics
    for y, x in groupby(flattened_X, Y):
        print(f"Original mean norm for {y}", torch.norm(x.mean(0)).item())
        print(f"Original cov norm difference for {y}", torch.norm(x.T.cov()).item())
        
        transformed = transform_to_statistics(x, global_mean, global_cov)
        
        # Print statistics for verification
        mean_mse = nn.MSELoss()(transformed.mean(0), global_mean)
        cov_mse = nn.MSELoss()(transformed.T.cov(), global_cov)
        print(f"mean and cov mse for class {y}", mean_mse.item(), cov_mse.item())
        
        transformed_data.append(transformed)
        transformed_labels.extend([y] * len(x))
    
    # Combine all transformed data
    transformed_data = torch.cat(transformed_data, dim=0)
    transformed_labels = torch.tensor(transformed_labels, device=device)
    
    # Reshape back to original image dimensions
    transformed_data = transformed_data.reshape(X.shape)
    
    return transformed_data, transformed_labels


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--method", type=str, default="leace")
    parser.add_argument("--shrinkage", type=bool, default=False)
    args = parser.parse_args()

    args = CombinedErasureArgs(
        dataset=args.dataset,
    )

    # Get dataset
    device = "cuda"
    (X_train, Y_train, X_val, Y_val, k, X, Y) = {
        "cifar10": get_cifar10(device),
        "cifarnet": get_cifarnet(),
        "fake-cifar10": get_fake_cifar10(),
        "fake-cifarnet": get_fake_cifarnet(),
        "svhn": get_svhn(device),
        "fake-svhn": get_fake_svhn(),
    }[args.dataset]

    num_features = X.shape[1] * X.shape[2] * X.shape[3]

    # Load your cached linear eraser
    cached_eraser = load_eraser(
        eraser_str="leace",
        dataset_str=args.dataset,
        X_train=X_train,
        Y_train=Y_train,
        dtype=torch.float32,
        method=args.method,
        shrinkage=args.shrinkage,
        alf_qleace_target=-1.,
        num_features=num_features,
        k=k,
        nowritecache=True,
        device="cuda",
        fit_device="cuda",

    )
    print(f"Train data shape: {X_train.shape}, {Y_train.shape}")
    print(f"Val data shape: {X_val.shape}, {Y_val.shape}")

    # Apply combined erasure
    data_dict = {}
    for split, X_data, Y_data in [("train", X_train, Y_train), ("val", X_val, Y_val)]:
        transformed_X, transformed_Y = transform_with_combined_erasure(args, X_data, Y_data, cached_eraser)
        dataset = Dataset.from_dict({"image": transformed_X, "label": transformed_Y})
        data_dict[split] = dataset

    import os
    # # Save to disk as HF dataset, LEACE-and-quadratic-iterative-erasure
    data_dir = f"data/leace-and-quadratic-iterative-erasure-{args.dataset}"
    
    for split, ds in data_dict.items(): 
        ds.save_to_disk(os.path.join(data_dir, split))
    
    # split = "train"
    # from datasets import load_from_disk
    # ab = load_from_disk((os.path.join(data_dir, split)))
    # ab.set_format(type="torch", columns=["image", "label"])
    
    # # Write sample images to disk
    # Path('data/saved_images').mkdir(exist_ok=True)
    # Path('data/saved_images/leaced_iterative_erasure').mkdir(exist_ok=True)
    # for i, (tensor, label) in enumerate(zip(ab["image"], ab["label"])):
    #     vutils.save_image(tensor, f'data/saved_images/leaced_iterative_erasure/class_{label}_original_{args.dataset}.png', normalize=True)



    # breakpoint()
