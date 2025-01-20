from pathlib import Path
from typing import Literal
from dataclasses import dataclass
from functools import partial

import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser
import lovely_tensors as lt
import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
from torch import Tensor
from datasets import load_from_disk
from concept_erasure import groupby
from torchvision.transforms.v2.functional import to_dtype, to_image
from experiments.cli import (
    get_cifar10,
    get_cifarnet,
    get_fake_cifarnet,
    get_svhn,
    IdentityEraser,
    load_eraser,
    get_fake_svhn,
    get_fake_cifar10
)   

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.weight'] = 'bold'


@dataclass
class Args:
    # General settings
    out: str = "data/images"

    # Dataset options
    method: Literal["leace", "orth", "none"] = "leace"
    shrinkage: bool = False
    normalize: bool = False
    post_erase_normalize: bool = False
    alf_qleace_target: float = 0.9

    # Runtime flags
    debug: bool = False
    nocache: bool = False
    nowritecache: bool = False
    save: bool = False
    overwrite: bool = False
    trial: bool = False  # Run a single trial with all data
    wandb_run_id: str | None = None


# def fix_cache(args, device):
#     state_path = Path("data") / "erasers_cache" / "state.pth"
#     state_path.parent.mkdir(parents=True, exist_ok=True)
#     new_state = (
#         {} if not state_path.exists() else torch.load(state_path, weights_only=False)
#     )

#     # leace_cache_key = get_cache_key('cifarnet', 'leace', dtype, args.method, args.shrinkage, 0.9)
#     # new_state[leace_cache_key] = leace_eraser_cifarnet.to('cpu')
#     # qleace_cache_key = get_cache_key('cifarnet', 'qleace', dtype, args.method, args.shrinkage, 0.9)
#     # new_state[qleace_cache_key] = qleace_eraser_cifarnet.to('cpu')
#     # alf_qleace_99_cache_key = get_cache_key('cifarnet', 'alf_qleace', dtype, args.method, args.shrinkage, 0.99)
#     # new_state[alf_qleace_99_cache_key] = alf_qleace_99_cifarnet.to('cpu')

#     torch.save(new_state, state_path)


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    lt.monkey_patch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # plot_all_erasers_cifarnet(args, device)
    # print("Plotting erasers")
    plot_all_erasers_cifar10(args, device)
    # print("Plotting alf qleace")
    # plot_cifarnet_alf_qleace(args, device)
    # print("Plotting datasets w iterative erasure")
    # plot_all_datasets_iterative_erasure(args, device)


def plot_cifarnet_alf_qleace(args, device):
    # Plot CIFARNet
    (X_train, Y_train, X_val, Y_val, k, X, Y) = get_cifarnet(shuffle=False)
    num_features = X.shape[1] * X.shape[2] * X.shape[3]
    image_side = X.shape[2]
    dtype = torch.float32

    load_example_eraser = partial(
        load_eraser,
        dataset_str="cifarnet",
        dtype=dtype,
        shrinkage=args.shrinkage,
        X_train=X_train, 
        Y_train=Y_train,
        num_features=num_features,
        k=k,
        nowritecache=args.nowritecache,
        nocache=args.nocache,
    )
    alf_qleace_90_cifarnet = load_example_eraser(
        "alf_qleace", alf_qleace_target=0.9, method="leace"
    ).to(device)
    alf_qleace_99_cifarnet = load_example_eraser(
        "alf_qleace", alf_qleace_target=0.99, method=args.method
    ).to(device)

    # Get CIFARNet sample images and apply different erasers
    sample_image = X_train[1:2].to(device)
    sample_label = Y_train[1:2].to(device)
    sample_images = {
        "Original": sample_image.squeeze(),
        "ALF-QLEACE-90": alf_qleace_90_cifarnet(sample_image.flatten(1))
        .reshape_as(sample_image)
        .squeeze(),
        "ALF-QLEACE-99": alf_qleace_99_cifarnet(sample_image.flatten(1))
        .reshape_as(sample_image)
        .squeeze(),
    }

    grid = vutils.make_grid(
        [
            sample_images["Original"],
            sample_images["ALF-QLEACE-90"],
            sample_images["ALF-QLEACE-99"],
        ],
        nrow=3,  # 3 images per row
        padding=4,  # Increase padding between images
        normalize=True,
        pad_value=1,  # Use white padding (1 = white, 0 = black)
    )

    # Convert to numpy and transpose to correct format (H,W,C)
    grid_img = grid.cpu().permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(15, 6), facecolor="white")
    ax.imshow(grid_img)
    ax.axis("off")
    
    # Calculate positions for captions
    img_width = grid_img.shape[1] / 3  # 3 images in the grid
    captions = [
        "Original", 
        "ALF-QLEACE (90%)",
        "ALF-QLEACE (99%)",
    ]

    for i, caption in enumerate(captions):
        # Position text under each image
        x_pos = (i + 0.5) * img_width  # Center of each image
        y_pos = grid_img.shape[0]  # Below the image
        
        ax.text(x_pos, y_pos, caption, 
                fontsize=10, 
                family='DejaVu Serif', 
                ha='center', 
                va='top')

    plt.savefig(
        "data/images/sample_alf_qleace_spectrum.pdf",
        bbox_inches="tight",  # Remove excess white space
        facecolor="white",
        dpi=300,
        pad_inches=0.1
    )  # Ensure white background in saved file


# Get erased original CIFAR10 dataset (not HF version modified and uploaded to the HF hub)
def get_fake_cifar10_disk():
    train = load_from_disk("data/eraser-order-cifar10")["train"]
    X = torch.stack(
        [
            to_dtype(to_image(img), dtype=torch.float32, scale=True)
            for img in train["image"]
        ]
    )
    Y = torch.tensor(train["label"])

    # Split train and validation
    val_size = 1024
    X_train, X_val = X[:-val_size], X[-val_size:]
    Y_train, Y_val = Y[:-val_size], Y[-val_size:]
    return X_train, Y_train, None, None, 0, None, None
    
def plot_all_erasers_cifar10(args, device):
    (X_train, Y_train, X_val, Y_val, k, X, Y) = get_cifar10(device, shuffle=False)
    num_features = X.shape[1] * X.shape[2] * X.shape[3]
    image_side = X.shape[2]
    dtype = torch.float32

    # state_path = Path("data") / "erasers_cache" / "state.pth"
    # state_path.parent.mkdir(parents=True, exist_ok=True)
    # new_state = {} if not state_path.exists() else torch.load(state_path, weights_only=False)

    # old_state = torch.load(Path("data") / "erasers_cache" / f"cifar10_{dtype}_state.pth", weights_only=False)
    # for eraser in ['leace', 'qleace']:
    #     cache_key = get_cache_key('cifar10', eraser, dtype, args.method, args.shrinkage, 0.99)
    #     if cache_key not in new_state and eraser in old_state:
    #         new_state[cache_key] = old_state[eraser].to("cpu")

    # torch.save(new_state, state_path)

    # Load erasers
    load_example_eraser = partial(
        load_eraser,
        dataset_str="cifar10",
        dtype=dtype,
        shrinkage=args.shrinkage,
        X_train=X_train,
        Y_train=Y_train,
        num_features=num_features,
        k=k,
        nowritecache=args.nowritecache,
        random_erase_dims=25
    )

    leace_eraser_cifar10 = load_example_eraser(
        "leace", alf_qleace_target=-1, method=args.method, nocache=args.nocache
    ).to(device)
    qleace_eraser_cifar10 = load_example_eraser(
        "qleace", alf_qleace_target=-1, method=args.method, nocache=args.nocache
    ).to(device)
    alf_qleace_90_cifar10 = load_example_eraser(
        "alf_qleace", alf_qleace_target=0.90, method=args.method, nocache=args.nocache
    ).to(device)
    alf_qleace_99_cifar10 = load_example_eraser(
        "alf_qleace", alf_qleace_target=0.99, method=args.method, nocache=args.nocache
    ).to(device)
    random_eraser_cifar10 = load_example_eraser(
        "random", alf_qleace_target=-1, method=args.method, nocache=args.nocache
    ).to(device)

    sample_image = X_train[3:4].to(device)
    sample_label = Y_train[3:4].to(device)

    # Erased CIFAR-10 on HF hub originates in the HF dataset ordering. Using the original dataset to get a matched image.
    fake_cifar10_images, fake_cifar10_labels, _, _, _, _, _ = get_fake_cifar10_disk()

    # Get first item in fake_cifar10_images with a corresponding fake_cifar10_labels label
    fake_cifar10_image = fake_cifar10_images[
        fake_cifar10_labels == sample_label.item()
    ][0]

    sample_images = {
        "Original": sample_image.squeeze(),
        "Random": random_eraser_cifar10(sample_image.flatten(1))
        .reshape_as(sample_image)
        .squeeze(),
        "LEACE": leace_eraser_cifar10(sample_image.flatten(1))
        .reshape_as(sample_image)
        .squeeze(),
        "QLEACE": qleace_eraser_cifar10(sample_image.flatten(1), sample_label)
        .reshape_as(sample_image)
        .squeeze(),
        "ALF-QLEACE-90": alf_qleace_90_cifar10(sample_image.flatten(1))
        .reshape_as(sample_image)
        .squeeze(),
        "ALF-QLEACE-99": alf_qleace_99_cifar10(sample_image.flatten(1))
        .reshape_as(sample_image)
        .squeeze(),
        "Iterative-Erasure": fake_cifar10_image.to(device),
    }

    grid = vutils.make_grid(
        [
            sample_images["Original"],
            sample_images["LEACE"],
            sample_images["QLEACE"],
            sample_images["ALF-QLEACE-90"],
            sample_images["Iterative-Erasure"],
            sample_images["Random"],
        ],
        nrow=6,
        padding=4,
        normalize=True,
        value_range=(0, 1),
        pad_value=1,  # Use white padding (1 = white, 0 = black)
    )

    # Convert to numpy and transpose to correct format (H,W,C)
    grid_img = grid.cpu().permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(15, 6), facecolor="white")
    ax.imshow(grid_img)
    ax.axis("off")
    
    # Calculate positions for captions
    img_width = grid_img.shape[1] / 6  # 6 images in the grid
    captions = [
        "Original", 
        "LEACE", 
        "QLEACE",
        "ALF-QLEACE",
        "Iterative Erasure",
        "Random Erasure", 
    ]

    for i, caption in enumerate(captions):
        # Position text under each image
        x_pos = (i + 0.5) * img_width  # Center of each image
        y_pos = grid_img.shape[0]  # Below the image
        
        ax.text(x_pos, y_pos, caption, 
                fontsize=10, 
                family='DejaVu Serif', 
                ha='center', 
                va='top')
    
    # plt.subplots_adjust(bottom=0.02, top=)
    
    plt.savefig(
        "data/images/eraser_comparison_cifar10.pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=300,
        pad_inches=0.1 
    )
    plt.close()


def plot_all_erasers_cifarnet(args, device):
    (X_train, Y_train, X_val, Y_val, k, X, Y) = get_cifarnet(shuffle=False)
    num_features = X.shape[1] * X.shape[2] * X.shape[3]
    image_side = X.shape[2]
    dtype = torch.float32

    # Load erasers
    load_example_eraser = partial(
        load_eraser,
        dataset_str="cifarnet",
        dtype=dtype,
        X_train=X_train,
        Y_train=Y_train,
        num_features=num_features,
        k=k,
        nowritecache=args.nowritecache,
        # Rank 13 for 90%, rank 238 for 99%, rank 2000+ for 99.9% for method = leace
        # Rank 112 for 90% for method = orth 
        # Rank 13 for 90% LEACE with shrinkage
        random_erase_dims=15
    )

    leace_eraser_cifarnet = load_example_eraser(
        "leace", alf_qleace_target=-1, method=args.method, nocache=args.nocache, shrinkage=args.shrinkage
    ).to(device)
    qleace_eraser_cifarnet = load_example_eraser(
        "qleace", alf_qleace_target=-1, method=args.method, nocache=args.nocache, shrinkage=args.shrinkage
    ).to(device)
    alf_qleace_90_cifarnet = load_example_eraser(
        "alf_qleace", alf_qleace_target=0.90, method=args.method, nocache=args.nocache, shrinkage=args.shrinkage
    ).to(device)
    alf_qleace_99_cifarnet = load_example_eraser(
        "alf_qleace", alf_qleace_target=0.99, method=args.method, nocache=args.nocache, shrinkage=args.shrinkage
    ).to(device)
    random_eraser_cifarnet = load_example_eraser(
        "random", alf_qleace_target=-1, method=args.method, nocache=args.nocache, shrinkage=args.shrinkage
    ).to(device)
    
    sample_image = X_train[:1].to(device)
    sample_label = Y_train[:1].to(device)

    # Erased CIFAR-net on HF hub originates in the HF dataset ordering. Using the original dataset to get a matched image.
    fake_cifarnet_images, fake_cifarnet_labels, _, _, _, _, _ = get_fake_cifarnet(shuffle=False)

    # Get first item in fake_cifarnet_images with a corresponding fake_cifarnet_labels label of 6
    fake_cifarnet_image_6 = fake_cifarnet_images[
        fake_cifarnet_labels == sample_label.item()
    ][0]

    sample_images = {
        "Original": sample_image.squeeze(),
        "Random": random_eraser_cifarnet(sample_image.flatten(1))
        .reshape_as(sample_image)
        .squeeze(),
        "LEACE": leace_eraser_cifarnet(sample_image.flatten(1))
        .reshape_as(sample_image)
        .squeeze(),
        "QLEACE": qleace_eraser_cifarnet(sample_image.flatten(1), sample_label)
        .reshape_as(sample_image)
        .squeeze(),
        "ALF-QLEACE-90": alf_qleace_90_cifarnet(sample_image.flatten(1))
        .reshape_as(sample_image)
        .squeeze(),
        "ALF-QLEACE-99": alf_qleace_99_cifarnet(sample_image.flatten(1))
        .reshape_as(sample_image)
        .squeeze(),
        "Iterative-Erasure": fake_cifarnet_image_6.to(device),
    }
    breakpoint()

    grid = vutils.make_grid(
        [
            sample_images["Original"],
            sample_images["LEACE"],
            sample_images["QLEACE"],
            sample_images["ALF-QLEACE-90"], # Contains very negative numbers, maybe needs to be rescaled using a bias term?
            sample_images["Iterative-Erasure"],
            sample_images["Random"],
        ],
        nrow=6,
        padding=4,
        normalize=False,
        value_range=(0, 1),
        pad_value=1,  # Use white padding (1 = white, 0 = black)
    )

    # Convert to numpy and transpose to correct format (H,W,C)
    grid_img = grid.cpu().permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(15, 6), facecolor="white")
    ax.imshow(grid_img)
    ax.axis("off")
    
    # Calculate positions for captions
    img_width = grid_img.shape[1] / 6  # 6 images in the grid
    captions = [
        "Original", 
        "LEACE", 
        "QLEACE",
        "ALF-QLEACE",
        "Iterative Erasure",
        "Random Erasure", 
    ]

    for i, caption in enumerate(captions):
        # Position text under each image
        x_pos = (i + 0.5) * img_width  # Center of each image
        y_pos = grid_img.shape[0]  # Below the image
        
        ax.text(x_pos, y_pos, caption, 
                fontsize=10, 
                family='DejaVu Serif', 
                ha='center', 
                va='top')
    
    # plt.subplots_adjust(bottom=0.02, top=)
    
    plt.savefig(
        "data/images/eraser_comparison_cifarnet.pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=300,
        pad_inches=0.1 
    )
    plt.close()



def plot_all_datasets_iterative_erasure(args, device):
    target_size = (64, 64)
    datasets = ["cifar-10", "cifarnet", "svhn"]
    plot_images = []
    samples_per_dataset = 10  # Based on your nrow=10 setting

    for dataset in datasets:
        X, Y, _, _, _, _, _ = {
            "cifar-10": get_cifar10(device, shuffle=False),
            "cifarnet": get_cifarnet(shuffle=False),
            "svhn": get_svhn(device, shuffle=False),
        }[dataset]

        fake_X, fake_Y, _, _, _, _, _ = {
            "cifar-10": get_fake_cifar10_disk(),
            "cifarnet": get_fake_cifarnet(shuffle=False),
            "svhn": get_fake_svhn(shuffle=False),
        }[dataset]

        samples = []
        fake_samples = []
        for y, x in groupby(X, Y):
            if len(fake_samples) >= samples_per_dataset:    
                break
            sample = x[0]
            fake_sample = fake_X[fake_Y == y][0]
    
            if sample.shape[-2:] != target_size and (
                sample.shape[-2] < target_size[0] or sample.shape[-1] < target_size[1]
            ):
                sample = F.interpolate(sample.unsqueeze(0), size=target_size, mode="bicubic", align_corners=False).squeeze()
                fake_sample = F.interpolate(fake_sample.unsqueeze(0), size=target_size, mode="bicubic", align_corners=False).squeeze()

            samples.append(sample.cpu())
            fake_samples.append(fake_sample.cpu())
        
        plot_images.extend(fake_samples)

    grid = vutils.make_grid(plot_images, nrow=samples_per_dataset, padding=4, normalize=False, pad_value=1)
    grid_img = grid.cpu().permute(1, 2, 0).numpy()

    # Create figure with extra space on the left for labels
    fig, ax = plt.subplots(figsize=(14, 4), facecolor="white")
    
    # Display the grid
    ax.imshow(grid_img)
    
    # Add dataset labels
    cell_height = grid_img.shape[0] / len(datasets)
    for idx, dataset in enumerate(datasets):
        y_pos = (idx + 0.5) * cell_height
        ax.text(-20, y_pos, dataset.upper(), 
                horizontalalignment='right',
                verticalalignment='center',
                family='DejaVu Serif', 
                rotation=0,
                fontsize=10)
    
    ax.axis("off")
    
    # Adjust layout to prevent label cutoff
    plt.subplots_adjust(left=0.1)
    
    plt.savefig(
        "data/images/iterative_erasure_minimally_changes_dataset.pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=300,
    )
    plt.close()

if __name__ == "__main__":
    main()
