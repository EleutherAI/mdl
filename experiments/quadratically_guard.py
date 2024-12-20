from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset, load_from_disk
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from experiments.cli import get_cifar10, get_cifarnet

@dataclass
class Args:
    dataset: str = "cifar10"
    num_classes: int = 10
    koleo_weight: float = 1e-3
    max_iter: int = 100
    structure_weight: float = 0.1

def transform_dataset(args: Args):
    from concept_erasure import assert_type, groupby, optimal_linear_shrinkage
    from concept_erasure.optimal_transport import psd_sqrt
    from PIL import Image as PilImage
    from torch import nn, optim, Tensor
    import torch
    import numpy as np

    def koleo(x: Tensor) -> Tensor:
        """Kozachenko-Leonenko estimator of entropy."""
        return torch.cdist(x, x).kthvalue(2).values.log().mean()

    def transform_to_statistics(data: Tensor, target_mean: Tensor, target_cov: Tensor, *, koleo_weight: float, max_iter: int, structure_weight: float):
        """Transform existing data points to match target statistics while preserving structure."""
        n, d = data.shape
        assert d == target_mean.shape[-1] == target_cov.shape[-1] == target_cov.shape[-2]
        assert n > 1, "Need at least two samples to compute covariance"

        eps = torch.finfo(data.dtype).eps
        # x = torch.clamp(data, eps, 1 - eps) # Redundant since we're using logit
        z = nn.Parameter(data.logit())
        opt = optim.LBFGS([z], line_search_fn="strong_wolfe", max_iter=max_iter)

        def closure():
            opt.zero_grad()
            x = z.sigmoid()
            mean_loss = torch.norm(x.mean(0) - target_mean)
            cov_loss = torch.norm(x.T.cov() - target_cov)
            structure_loss = torch.norm(torch.cdist(x, x) - torch.cdist(data, data))
            loss = mean_loss + cov_loss + structure_weight * structure_loss
            loss -= koleo_weight * koleo(x)
            loss.backward()
            return float(loss)

        opt.step(closure)
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
        for y, x in groupby(X, Y):
            flat_x = x.flatten(1)
            transformed = transform_to_statistics(
                flat_x, 
                global_mean,
                global_cov,
                koleo_weight=args.koleo_weight,
                max_iter=args.max_iter,
                structure_weight=args.structure_weight
            )
        
            # Fix: Reshape to [N, 3, 32, 32] then permute to [N, 32, 32, 3] for PIL
            reshaped = transformed.reshape(-1, 3, 32, 32).permute(0, 2, 3, 1).mul(255).clip(0, 255).byte()
            transformed_images.extend([
                PilImage.fromarray(img.cpu().numpy(), mode='RGB')
                for img in reshaped
            ])
            transformed_labels.extend([y] * len(x))

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
        dataset.save_to_disk(f"transformed-{args.dataset}/{split}")
