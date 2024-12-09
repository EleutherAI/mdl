from pathlib import Path

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import torch
from polyapprox.ols import ols
from mdl.mlp_probe import MlpProbe
import lovely_tensors as lt

lt.monkey_patch()

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
    fig.write_image(f"{filename}.pdf", format="pdf")

# Load each MLP checkpoint ols
out_path = Path("polyapprox_mlp.pth")
ckpts = list(Path("probe-ckpts").glob("*.pth"))
ols_results = {} if not out_path.exists() else torch.load(out_path)
plot(ols_results)

for ckpt in ckpts:
    if "normalize" not in ckpt.name:
        continue

    if ckpt.name in ols_results:
        print(f"Skipping {ckpt.name} because it already exists")
        continue

    print(f"Processing {ckpt.name}")

    probe = MlpProbe(
        num_features=32 * 32 * 3, num_classes=10, hidden_size=128, num_layers=1
    )
    probe.load_state_dict(torch.load(ckpt))
    probe.eval()

    ols_results[ckpt.name] = ols(
        probe.net[0].weight.data.double().numpy(),
        probe.net[0].bias.data.double().numpy(),
        probe.net[2].weight.data.double().numpy(),
        probe.net[2].bias.data.double().numpy(),
        act="relu",
        order="quadratic",
        return_fvu=True,
    )

torch.save(ols_results, out_path)
plot(ols_results)

# def polyapprox_linear(ckpts):
#     linear_results = {}
#     for ckpt in ckpts:
#         if "normalize" not in ckpt.name:
#             continue

#         if ckpt.name in ols_results:
#             print(f"Skipping {ckpt.name} because it already exists")
#             continue

#         print(f"Processing {ckpt.name}")

#         probe = MlpProbe(
#             num_features=32 * 32 * 3, num_classes=10, hidden_size=128, num_layers=1
#         )
#         probe.load_state_dict(torch.load(ckpt))
#         probe.eval()

#         ols_results[ckpt.name] = ols(
#             probe.net[0].weight.data.double().numpy(),
#             probe.net[0].bias.data.double().numpy(),
#             probe.net[2].weight.data.double().numpy(),
#             probe.net[2].bias.data.double().numpy(),
#             act="relu",
#             order="quadratic",
#             return_fvu=True,
#         )
#         print(f"FVU: {ols_results[ckpt.name].fvu}")
#     # exit()
#     torch.save(linear_results, out_path)
#     plot(linear_results, filename="polyapprox_mlp_linear")
