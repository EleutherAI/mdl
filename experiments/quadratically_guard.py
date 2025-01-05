from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from concept_erasure import assert_type, groupby, optimal_linear_shrinkage
from PIL import Image as PilImage
from torchvision import transforms
from torch import nn, optim, Tensor
import torch
import torchvision.utils as vutils; 
from pathlib import Path; 

@dataclass
class Args:
    dataset: str = "cifar10"
    num_classes: int = 10
    max_iter: int = 100
    # 1000 results in human-interpretable images, 100 is marginal
    mse_weight: float = 1.
    cov_weight: float = 0.2
    mean_weight: float = 0.01
    prefix: str = "transformed"

def transform_dataset(args: Args):
    def transform_to_statistics(data: Tensor, target_mean: Tensor, target_cov: Tensor, *, max_iter: int, mean_weight: float, cov_weight: float, mse_weight: float):
        """Transform existing data points to match target statistics while preserving structure."""
        n, d = data.shape
        assert d == target_mean.shape[-1] == target_cov.shape[-1] == target_cov.shape[-2]
        assert n > 1, "Need at least two samples to compute covariance"

        eps = torch.finfo(data.dtype).eps
        x = torch.clamp(data, eps, 1 - eps)
        original = x.logit()
        z = nn.Parameter(x.logit())
        opt = optim.LBFGS([z], line_search_fn="strong_wolfe", max_iter=max_iter)

        def debug_loss(x):
            mean_loss = torch.norm(x.mean(0) - target_mean)
            cov_loss = torch.norm(x.T.cov() - target_cov)
            mse_loss = ((x - original) ** 2).mean((0, 1))
            print("mean loss", (mean_loss * mean_weight).item(), "cov loss", (cov_loss * cov_weight).item(), "weighted mse loss", (mse_loss * mse_weight).item())
        
        def closure():
            opt.zero_grad()
            x = z.sigmoid()
            mean_loss = torch.norm(x.mean(0) - target_mean)
            cov_loss = torch.norm(x.T.cov() - target_cov)
            mse_loss = ((x - original) ** 2).mean((0, 1))

            loss = (mean_loss * mean_weight) + (cov_loss * cov_weight) + (mse_weight * mse_loss)
            loss.backward()
            return float(loss)

        print("original loss")
        debug_loss(z.sigmoid())
        opt.step(closure)
        print("final loss")
        debug_loss(z.sigmoid())
        return z.sigmoid().detach()

    def process_split(split: str):
        ds = assert_type(Dataset, load_dataset(args.dataset, split=split))
        if "img" in ds.column_names:
            ds = ds.rename_column("img", "image")
        
        with ds.formatted_as("torch"):
            X = assert_type(Tensor, ds["image"]).div(255).cuda()
            Y = assert_type(Tensor, ds["label"]).cuda()

        # Calculate global statistics
        flattened_X = X.flatten(1)
        global_mean = flattened_X.mean(0)
        global_cov = optimal_linear_shrinkage(flattened_X.mT.cov(), len(ds))

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

            print("Final MSE:", nn.MSELoss()(transformed.flatten(1), flat_x).item())
        
            # Fix: Reshape to [N, 3, 32, 32] then permute to [N, 32, 32, 3] for PIL
            reshaped = transformed.reshape(-1, 3, 32, 32).permute(0, 2, 3, 1).mul(255).clip(0, 255).byte()
            transformed_images.extend([
                PilImage.fromarray(img.cpu().numpy(), mode='RGB')
                for img in reshaped
            ])
            transformed_labels.extend([y] * len(x))

        # Get average cosine similarity between transformed classes
        for i in range(1, len(means)):
            torch.testing.assert_close(means[i], global_mean, rtol=0.5, atol=0.5)
            torch.testing.assert_close(covs[i], global_cov, rtol=0.5, atol=0.5)

            mean_mse = nn.MSELoss()(means[i], global_mean)
            cov_mse = nn.MSELoss()(covs[i], global_cov)
            print(f"mean and cov mse for class {i}", mean_mse.item(), cov_mse.item())

        # TODO this should be out of loop (leaving for consistency)
        print("Saving sample image")

        # Get indices of first occurrence of each class
        unique_labels = []
        unique_indices = []
        original_tensors = []
        for i, label in enumerate(transformed_labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_indices.append(i)

                # Get original image index
                original_tensors.append(X[Y == label][0])

                if len(unique_labels) == 10:  # Assuming 10 classes (e.g., CIFAR-10)
                    break

        selected_tensors = torch.stack([transforms.ToTensor()(transformed_images[i]) for i in unique_indices])

        Path('saved_images').mkdir(exist_ok=True)
        for i, (tensor, orig_tensor, label) in enumerate(zip(selected_tensors, original_tensors, unique_labels)):
            vutils.save_image(tensor, f'saved_images/class_{label}_cifarnet_{args.mse_weight}.png', normalize=True)
            vutils.save_image(orig_tensor, f'saved_images/class_{label}_original_cifarnet.png', normalize=True)

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
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    transformed = transform_dataset(args)

    for split, dataset in transformed.items():
        dataset.save_to_disk(f"{args.prefix}-{args.dataset}/{split}")

        # TODO Upload to hub
