import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import torch
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go

from experiments.scrape_wandb import DISPLAY_NAMES
from experiments.sweep_eraser import sweep_params

pio.kaleido.scope.mathjax = None


def load_sweep_data(data_path: Path) -> pd.DataFrame:
    records = []
    for file in data_path.glob("*.pth"):
        stem = file.stem.replace("alf_qleace", "alf-qleace")

        try:
            net, act, width, depth, eraser, directory, dataset = stem.split("_")
        except ValueError:
            print(f"Skipping malformed filename: {stem}")
            continue

        width = int(width.split("=")[1])
        depth = int(depth.split("=")[1])

        data = torch.load(file, weights_only=False)
        for seed, result in enumerate(data):
            # Handle nested list structure
            while isinstance(result, list):
                result = result[0]

            base_dataset = dataset.replace('fake-leace-', '')
            base_dataset = base_dataset.replace('fake-', '')

            if 'fake-leace' in dataset:
                eraser_name = "LEACE and Iterative Erasure"
            elif 'fake' in dataset:
                eraser_name = "Iterative Erasure"
            else:
                eraser_name = DISPLAY_NAMES[eraser]
                
            records.append(
                {
                    "dataset": base_dataset,
                    "net_id": net,
                    "net": DISPLAY_NAMES[net],
                    "act": DISPLAY_NAMES[act],
                    "eraser": eraser_name,
                    "width": width,
                    "depth": depth,
                    "seed": seed,
                    "mdl": result.mdl,
                    "ce_curve": result.ce_curve,
                    "sample_sizes": result.sample_sizes,
                    "total_trials": result.total_trials,
                }
            )

    return pd.DataFrame(records)


def create_plots(df: pd.DataFrame, output_dir: Path, dataset: str):
    output_dir.mkdir(exist_ok=True, parents=True)
    colors = px.colors.qualitative.Plotly

    ordered_erasers = [
        e
        for e in ["Control", "LEACE", "QLEACE", "ALF-QLEACE", "Iterative Erasure"]
        if e in df["eraser"].unique()
    ]

    df = df[df["dataset"] == dataset].sort_values(["depth", "width"])

    for net_id in df["net_id"].unique():
        net = DISPLAY_NAMES[net_id]
        reference_width = sweep_params[net_id]["mup_width"]
        reference_depth = sweep_params[net_id]["mup_depth"]
        ordered_acts = ["ReLU", "GELU", "SwiGLU"] if net == "MLP" else ["ReLU"]

        fig = make_subplots(
            rows=len(ordered_erasers),
            cols=2,
            subplot_titles=sum(zip(ordered_erasers, ordered_erasers), ()),
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            row_heights=[400] * len(ordered_erasers),
            shared_yaxes="rows",
        )

        fig.update_layout(
            title=f"Minimum description length over 5 seeds ({net})",
            height=300 * len(ordered_erasers),
            width=1200,
            showlegend=len(ordered_acts) > 1,
            legend=(
                dict(
                    title="Activation function",
                    yanchor="top",
                    y=0.95,
                    xanchor="right",
                    x=0.95,
                )
                if len(ordered_acts) > 1
                else None
            ),
        )

        net_df = df[df["net"] == net]

        for row, eraser in enumerate(ordered_erasers, 1):
            # Set up axes
            fig.update_yaxes(
                title_text="MDL (bits per sample)",  # if col == 1 else "",
                showticklabels=True,  # (col == 1),
                matches="y1",
                row=row,
                col=1,
            )
            fig.update_yaxes(showticklabels=False, row=row, col=2)

            for col, param in enumerate(["depth", "width"], 1):
                fig.update_xaxes(
                    title_text=param.title() if row == len(ordered_erasers) else "",
                    showticklabels=(row == len(ordered_erasers)),
                    type="log",
                    tickvals=[
                        2**i
                        for i in range(
                            int(np.log2(min(net_df[param]))),
                            int(np.log2(max(net_df[param]))) + 1,
                        )
                    ],
                    ticktext=[
                        f"2<sup>{i}</sup>"
                        for i in range(
                            int(np.log2(min(net_df[param]))),
                            int(np.log2(max(net_df[param]))) + 1,
                        )
                    ],
                    row=row,
                    col=col,
                )

            # Plot data for each activation function
            for act_idx, act in enumerate(ordered_acts):
                data = df[
                    (df["eraser"] == eraser) & (df["act"] == act) & (df["net"] == net)
                ]
                if data.empty:
                    continue

                mean_data = (
                    data.groupby(["width", "depth"])["mdl"]
                    .agg(["mean", "std"])
                    .reset_index()
                )

                # Plot depth data
                depth_data = data[data["width"] == reference_width]
                mean_depth = mean_data[mean_data["width"] == reference_width]

                for col, (ref_val, param_data, mean_param) in enumerate(
                    [
                        (reference_width, depth_data, mean_depth),
                        (
                            reference_depth,
                            data[data["depth"] == reference_depth],
                            mean_data[mean_data["depth"] == reference_depth],
                        ),
                    ],
                    1,
                ):
                    param = "depth" if col == 1 else "width"

                    # Plot individual points
                    fig.add_trace(
                        go.Scatter(
                            x=param_data[param],
                            y=param_data["mdl"],
                            mode="markers",
                            marker=dict(color=colors[act_idx], size=5, opacity=0.3),
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

                    # Plot mean line
                    fig.add_trace(
                        go.Scatter(
                            x=mean_param[param],
                            y=mean_param["mean"],
                            mode="lines+markers",
                            line=dict(width=2),
                            marker=dict(color=colors[act_idx]),
                            name=act,
                            showlegend=(row == 1 and col == 1),
                        ),
                        row=row,
                        col=col,
                    )

        fig.write_image(output_dir / f"{net}_MDL_{dataset}.pdf", format="pdf")


def main():
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/mnt/ssd-1/lucia/24-11-21"))
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--out", type=Path, default=Path("data/images/sweep_plots"))
    args = parser.parse_args()

    if args.dataset == "cifar10":
        assert "cifarnet" not in args.data.name

    print("Loading disk data into dataframe...")
    df = load_sweep_data(args.data)

    print("Creating plots...")
    create_plots(df, args.out, args.dataset)


if __name__ == "__main__":
    main()
