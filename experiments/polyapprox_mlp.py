from pathlib import Path

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from polyapprox.ols import ols
from mdl.mlp_probe import MlpProbe
import lovely_tensors as lt
from mup import set_base_shapes
from experiments.cli import get_cifar10

lt.monkey_patch()

lt.monkey_patch()

class QuadraticModel:
    def __init__(self, alpha: Tensor, beta: Tensor, gamma: Tensor, d: int):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.quad_rows, self.quad_cols = torch.tril_indices(d, d)
    
    def __call__(self, X):
        return self.forward(X)
        
    def forward(self, X):
        linear = X @ self.beta
        pairs = X[:, self.quad_rows] * X[:, self.quad_cols]
        
        return self.alpha + linear + pairs @ self.gamma.T

        
class LinearModel:
    def __init__(self, alpha: Tensor, beta: Tensor):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        return X @ self.beta + self.alpha


def calculate_fvu(model: nn.Module, approx_model: QuadraticModel | LinearModel, 
                 data_loader: torch.utils.data.DataLoader, device="cuda") -> float:
    """Calculate Fraction of Variance Unexplained"""
    all_outputs = []
    all_approx = []
    
    for inputs in data_loader:
        inputs = inputs.to(device)
        model_output = model(inputs)
        quad_output = approx_model(inputs)
        
        all_outputs.extend(model_output)
        all_approx.extend(quad_output)
    
    all_outputs = torch.stack(all_outputs)
    all_approx = torch.stack(all_approx)
    
    # Calculate FVU = mean squared error / variance of true outputs
    mse = torch.mean((all_outputs - all_approx) ** 2).item()
    var = torch.var(all_outputs).item()
    
    return mse / var


def normalize_cifar10(X, X_train, X_val):
    X_flat = X.reshape(X.shape[0], -1)
    
    mean = X_flat.mean(dim=0, keepdim=True)
    X_centered = X_flat - mean
    
    cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
    
    scaling = torch.sqrt(torch.diagonal(cov))
    scaling = torch.where(scaling > 0, scaling, torch.ones_like(scaling))
    
    def normalize_data(data: Tensor) -> Tensor:
        data_flat = data.reshape(data.shape[0], -1)
        data_centered = data_flat - mean
        data_normalized = data_centered / scaling
        return data_normalized.reshape(data.shape)
    
    X = normalize_data(X)
    X_train = normalize_data(X_train)
    X_val = normalize_data(X_val)

    return X, X_train, X_val


def prepare_random_data(n_samples: int, input_dim: int) -> torch.utils.data.DataLoader:
    """Prepare normally distributed random data"""
    X = torch.randn(n_samples, input_dim)
    return torch.utils.data.DataLoader(X, batch_size=100, shuffle=True)


def plot(ols_results, filename='polyapprox_mlp_fvu'):
# Plot FVU over checkpoints - the final number in each name is the checkpoints
    fvu = []
    checkpoint = []
    eraser = []
    for key, value in ols_results.items():
        if value.fvu < -0.01:
            print(f"{key} has FVU {value.fvu}. Skipping.")
            continue

        fvu.append(value.fvu)
        chunks = key[:-4].split("-")
        checkpoint.append(int(chunks[-1]))
        eraser.append(chunks[0].split(" ")[0])

    df = pd.DataFrame({"fvu": fvu, "checkpoint": checkpoint, "eraser": eraser})
    df = df.sort_values(by="checkpoint")

    fig = make_subplots(rows=len(df.eraser.unique()), cols=1)

    for row, eraser in enumerate(df.eraser.unique(), start=1):
        df_eraser = df[df.eraser == eraser]
        fig.add_trace(go.Scatter(x=df_eraser.checkpoint, y=df_eraser.fvu, mode="lines", name=eraser), row=row, col=1)

    fig.update_layout(title="FVU over checkpoints")
    fig.write_image(f"data/{filename}.pdf", format="pdf")

@torch.no_grad()
def main():
    # Load each MLP checkpoint ols
    out_path = Path("data/polyapprox_mlp.pth")
    ckpts = list(Path("probe-ckpts").glob("*.pth"))
    ols_results = {} if not out_path.exists() else torch.load(out_path, weights_only=False)
    base_shapes_path = f"mup-mlp-128-1-128.bsh"
    probe = MlpProbe(
        num_features=32 * 32 * 3, num_classes=10, hidden_size=128, num_layers=1
    )

    n_samples, input_dim = 10_000, 32 * 32 * 3
    (X_train, Y_train, X_val, Y_val, k, X, Y) = get_cifar10(device="cpu")
    X, X_train, X_val = normalize_cifar10(X, X_train, X_val)

    X_d = X.shape[1] * X.shape[2] * X.shape[3]
    cifar10_dataloader = torch.utils.data.DataLoader(X_train[:n_samples].flatten(1), batch_size=100, shuffle=True)

    
    random_loader = prepare_random_data(n_samples, input_dim)
    device = "cuda"

    for ckpt in ckpts:
        if 'normalize' not in ckpt.name or 'control' not in ckpt.name or 'relu' not in ckpt.name:
            continue

        if ckpt.name in ols_results:
            print(f"Skipping {ckpt.name} because it already exists")
            continue

        print(f"Processing {ckpt.name}")

        probe.load_state_dict(torch.load(ckpt, weights_only=False))
        set_base_shapes(probe, base_shapes_path, rescale_params=False)
        probe.to(device)

        ols_results[ckpt.name] = {}

        ols_results[ckpt.name]['ols'] = ols(
            probe.net[0].weight.data.double().cpu().numpy(),
            probe.net[0].bias.data.double().cpu().numpy(),
            probe.net[2].weight.data.double().cpu().numpy(),
            probe.net[2].bias.data.double().cpu().numpy(),
            act="relu",
            order="quadratic",
        )

        quad_approx = QuadraticModel(
            torch.from_numpy(ols_results[ckpt.name]['ols'].alpha).float().to(device), 
            torch.from_numpy(ols_results[ckpt.name]['ols'].beta).float().to(device), 
            torch.from_numpy(ols_results[ckpt.name]['ols'].gamma).float().to(device),
            d=X_d
        )

        random_fvu = calculate_fvu(probe, quad_approx, random_loader, device)
        print(f"FVU for random data: {random_fvu:.4f}")
        ols_results[ckpt.name]['random_fvu'] = random_fvu
        
        cifar_fvu = calculate_fvu(probe, quad_approx, cifar10_dataloader, device)
        print(f"FVU for CIFAR-10: {cifar_fvu:.4f}")
        ols_results[ckpt.name]['cifar_fvu'] = cifar_fvu

        ols_linear = ols(
            probe.net[0].weight.data.double().cpu().numpy(),
            probe.net[0].bias.data.double().cpu().numpy(),
            probe.net[2].weight.data.double().cpu().numpy(),
            probe.net[2].bias.data.double().cpu().numpy(),
            act="relu",
            order="linear",
            # TODO use true data mean and covariance matrix on probes from non normalized 24-11-21
        )

        linear_approx = LinearModel(
            torch.from_numpy(ols_linear.alpha).float().to(device),
            torch.from_numpy(ols_linear.beta).float().to(device),
        )

        random_linear_fvu = calculate_fvu(probe, linear_approx, random_loader, device)
        print(f"Linear FVU for random data: {random_linear_fvu:.4f}")
        ols_results[ckpt.name]['random_linear_fvu'] = random_linear_fvu

        linear_fvu = calculate_fvu(probe, linear_approx, cifar10_dataloader, device)
        print(f"Linear FVU for CIFAR-10: {linear_fvu:.4f}")
        ols_results[ckpt.name]['linear_fvu'] = linear_fvu

    torch.save(ols_results, out_path)
    plot(ols_results)


if __name__ == "__main__":
    main()
