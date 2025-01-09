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
    }
}

@dataclass
class Args:
    dataset: str = "cifar10"
    num_classes: int = 10
    max_iter: int = 100
    # 1000 results in human-interpretable images, 100 is marginal
    prefix: str = "erased"
    mse_weight: float | None = None
    cov_weight: float | None = None
    mean_weight: float | None = None


def transform_dataset(args: Args):
    def transform_cifarnet_to_statistics(data: Tensor, target_mean: Tensor, target_cov: Tensor, *, max_iter: int, mean_weight: float, cov_weight: float, mse_weight: float):
        """Transform existing data points to match target statistics while preserving structure."""
        n, d = data.shape
        assert d == target_mean.shape[-1] == target_cov.shape[-1] == target_cov.shape[-2]
        assert n > 1, "Need at least two samples to compute covariance"

        z = nn.Parameter(data.clone())
        target_data = data
        opt = optim.Adam([z], lr=1e-2)

        def closure():
            opt.zero_grad()
            x = z
            mean_loss = torch.norm(x.mean(0) - target_mean)
            cov_loss = torch.norm(x.T.cov() - target_cov)
            mse_loss = ((x - target_data) ** 2).mean((0, 1))

            loss = (mean_loss * mean_weight) + (cov_loss * cov_weight) + (mse_weight * mse_loss)
            print(loss, "mean loss", (mean_loss * mean_weight).item(), "cov loss", (cov_loss * cov_weight).item(), "weighted mse loss", (mse_loss * mse_weight).item())            

            loss.backward()
            return float(loss)

        for _ in range(max_iter):
            loss = closure()
            opt.step()
        return z.detach()


    def transform_to_statistics(data: Tensor, target_mean: Tensor, target_cov: Tensor, *, max_iter: int, mean_weight: float, cov_weight: float, mse_weight: float):
        """Transform existing data points to match target statistics while preserving structure."""
        n, d = data.shape
        assert d == target_mean.shape[-1] == target_cov.shape[-1] == target_cov.shape[-2]
        assert n > 1, "Need at least two samples to compute covariance"

        eps = torch.finfo(data.dtype).eps
        x = torch.clamp(data, eps, 1 - eps)
        z = nn.Parameter(x.logit())

        target_mean = torch.clamp(target_mean, eps, 1 - eps)
        target_mean = target_mean.logit().sigmoid()

        target_cov = torch.clamp(target_cov, eps, 1 - eps)
        target_cov = target_cov.logit().sigmoid()

        target_data = x.logit().sigmoid()

        opt = optim.LBFGS([z], line_search_fn="strong_wolfe", max_iter=max_iter)

        def closure():
            opt.zero_grad()
            
            x = z.sigmoid()
            mean_loss = torch.norm(x.mean(0) - target_mean)
            cov_loss = torch.norm(x.T.cov() - target_cov)
            mse_loss = ((x - target_data) ** 2).mean((0, 1))

            loss = (mean_loss * mean_weight) + (cov_loss * cov_weight) + (mse_weight * mse_loss)
            print(loss, "mean loss", (mean_loss * mean_weight).item(), "cov loss", (cov_loss * cov_weight).item(), "weighted mse loss", (mse_loss * mse_weight).item())            

            loss.backward()
            return float(loss)

        opt.step(closure)
        return z.sigmoid().detach()
        

    def process_split(split: str):
        if args.dataset == "cifarnet":
            ds = assert_type(Dataset, load_dataset(f"EleutherAI/{args.dataset}", split=split))
        else:
            ds = assert_type(Dataset, load_dataset(args.dataset, split=split))

        if "img" in ds.column_names:
            ds = ds.rename_column("img", "image")

        with ds.formatted_as("torch"):
            X = assert_type(Tensor, ds["image"]).div(255)
            Y = assert_type(Tensor, ds["label"])

        # Calculate global statistics
        flattened_X = X.flatten(1)
        global_mean = flattened_X.mean(0).cpu()
        global_cov = optimal_linear_shrinkage(flattened_X.mT.cov(), len(ds)).cpu()
        del flattened_X
        
        transformed_images = []
        transformed_labels = []

        # Transform each class to match global statistics
        means = []
        covs = []
        for y, x in groupby(X, Y):
            flat_x = x.flatten(1)
            
            # TODO Print original cov and mean norm differences
            print(f"Original mean norm for {y}", torch.norm(x.flatten(1).mean(0)).item())
            print(f"Original cov norm difference for {y}", torch.norm(x.flatten(1).T.cov()).item())

            if args.dataset == "cifarnet":
                transformed = transform_cifarnet_to_statistics(
                    flat_x, 
                    global_mean,
                    global_cov,
                    max_iter=args.max_iter,
                    mean_weight=args.mean_weight,
                    cov_weight=args.cov_weight,
                    mse_weight=args.mse_weight
                )
            else:
                transformed = transform_to_statistics(
                    flat_x, 
                    global_mean,
                    global_cov,
                    max_iter=args.max_iter,
                    mean_weight=args.mean_weight,
                    cov_weight=args.cov_weight,
                    mse_weight=args.mse_weight
                )
            means.append(transformed.mean(0))
            covs.append(transformed.T.cov())
        
            # Fix: Reshape to [N, 3, 32, 32] then permute to [N, 32, 32, 3] for PIL
            reshaped = transformed.reshape_as(x).permute(0, 2, 3, 1).mul(255).clip(0, 255).byte()
            transformed_images.extend([
                PilImage.fromarray(img.cpu().numpy(), mode='RGB')
                for img in reshaped
            ])
            transformed_labels.extend([y] * len(x))

        # Print average cosine similarity between transformed classes
        for i in range(1, len(means)):
            torch.testing.assert_close(means[i], global_mean, rtol=0.5, atol=0.5)
            torch.testing.assert_close(covs[i], global_cov, rtol=0.5, atol=0.5)

            mean_mse = nn.MSELoss()(means[i], global_mean)
            cov_mse = nn.MSELoss()(covs[i], global_cov)
            print(f"mean and cov mse for class {i}", mean_mse.item(), cov_mse.item())

        # Get indices of first occurrence of each class
        print("Saving sample images")
        unique_labels = []
        unique_indices = []
        original_tensors = []
        for i, label in enumerate(transformed_labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_indices.append(i)

                # Get original image index
                original_tensors.append(X[Y == label][0])

                if len(unique_labels) == 10:  # Assuming 10 classes (e.g., CIFAR-10, CIFARNet)
                    break

        selected_tensors = torch.stack([transforms.ToTensor()(transformed_images[i]) for i in unique_indices])

        Path('data/saved_images').mkdir(exist_ok=True)
        for i, (tensor, orig_tensor, label) in enumerate(zip(selected_tensors, original_tensors, unique_labels)):
            vutils.save_image(tensor, f'data/saved_images/class_{label}_{args.dataset}_{args.mse_weight}_{args.cov_weight}_{args.mean_weight}.png', normalize=True)
            vutils.save_image(orig_tensor, f'data/saved_images/class_{label}_original_{args.dataset}.png', normalize=True)

        return Dataset.from_dict({
            "image": transformed_images,
            "label": transformed_labels
        })

    features = Features({
        "image": Image(),
        "label": ClassLabel(num_classes=args.num_classes),
    })

    transformed_train = process_split("train").cast(features)
    transformed_test = process_split("test").cast(features)

    return DatasetDict({
        "train": transformed_train,
        "test": transformed_test
    })


if __name__ == "__main__":
    set_seeds()
    
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args
    args.mse_weight = args.mse_weight or hyperparameters[args.dataset]["mse_weight"]
    args.cov_weight = args.cov_weight or hyperparameters[args.dataset]["cov_weight"]
    args.mean_weight = args.mean_weight or hyperparameters[args.dataset]["mean_weight"]

    transformed = transform_dataset(args)

    # Upload to hub
    api = HfApi()
    api.whoami()

    repo_id = f"EleutherAI/erased-{args.dataset}"

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    
    transformed.push_to_hub(
        repo_id,
        private=False,
        commit_message=f"Upload transformed {args.dataset} dataset"
    )
