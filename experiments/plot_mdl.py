import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

from experiments.sweep_eraser import sweep_params
from experiments.scrape_wandb import DISPLAY_NAMES

import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


def load_sweep_data(data_path: Path) -> pd.DataFrame:
    """Load and parse sweep data files into a DataFrame.
    
    Parameters:
    - data_path: Path to the directory containing .pth files.

    Returns:
    - DataFrame containing parsed data.
    """
    records = []

    for file in data_path.glob("*.pth"):
        # Parse filename
        try:
            net, act, width, depth, eraser, directory, dataset = file.stem.split("_")
        except ValueError:
            print(file.stem)
            continue

        width = int(width.split("=")[1])
        depth = int(depth.split("=")[1])

        # Load data and create records
        data = torch.load(file)
        for seed, mdl_result in enumerate(data):
            if type(mdl_result) == list:
                mdl_result = mdl_result[0]
            if type(mdl_result) == list:
                mdl_result = mdl_result[0]
            records.append(
                {
                    "dataset": dataset,
                    "net_id": net,  # to access the sweep_eraser.py hyperparameter dict
                    "net": DISPLAY_NAMES[net],
                    "act": DISPLAY_NAMES[act],
                    "eraser": DISPLAY_NAMES[eraser],
                    "width": width,
                    "depth": depth,
                    "seed": seed,
                    "mdl": mdl_result.mdl,
                    "ce_curve": mdl_result.ce_curve,
                    "sample_sizes": mdl_result.sample_sizes,
                    "total_trials": mdl_result.total_trials,
                }
            )

    return pd.DataFrame(records)


def create_plots(df: pd.DataFrame, output_dir: Path, dataset: str):
    """Create plots for each network and eraser type with a line for each activation function.
    Seed data is plotted as markers and mean data as lines."""

    output_dir.mkdir(exist_ok=True, parents=True)

    colors = px.colors.qualitative.Set1

    ordered_erasers = ["Control", "LEACE", "QLEACE", "ALF-QLEACE"]
    ordered_erasers = [eraser for eraser in ordered_erasers if eraser in df["eraser"].unique()]

    df = df.sort_values(["depth", "width"])

    for net_id in df["net_id"].unique():
        net = DISPLAY_NAMES[net_id]

        subplot_titles = [
            title for title in sum(zip(ordered_erasers, ordered_erasers), ())
        ]

        fig = make_subplots(
            rows=len(ordered_erasers),
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            row_heights=[400] * len(ordered_erasers),
        )
        fig.update_layout(
            title=f"Minimum description length over 5 seeds ({net})",
            height=300 * len(ordered_erasers),
            width=1200,
            showlegend=False,
        )

        fig.update_yaxes(matches="y1")

        reference_width = sweep_params[net_id]["mup_width"]
        reference_depth = sweep_params[net_id]["mup_depth"]
        ordered_acts = ["ReLU", "GELU", "SwiGLU"] if net == "MLP" else ["ReLU"]

        for row, eraser in enumerate(ordered_erasers, 1):
            # Update axis labels
            fig.update_yaxes(title_text="MDL (bits per sample)", row=row, col=1)
            fig.update_yaxes(showticklabels=False, row=row, col=2)

            for col, type in zip([1, 2], ["depth", "width"]):
                if row == len(ordered_erasers):
                    fig.update_xaxes(title_text=type.title(), row=row, col=col)
                else:
                    fig.update_xaxes(showticklabels=False, row=row, col=col)

                net_df = df[df["net"] == net]
                fig.update_xaxes(
                    type="log",
                    row=row,
                    col=col,
                    tickvals=[
                        2**i
                        for i in range(
                            int(np.log2(min(net_df[type]))),
                            int(np.log2(max(net_df[type]))) + 1,
                        )
                    ],
                    ticktext=[
                        f"2<sup>{i}</sup>"
                        for i in range(
                            int(np.log2(min(net_df[type]))),
                            int(np.log2(max(net_df[type]))) + 1,
                        )
                    ],
                )

            # Plot data
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

                seed_depth_data = data[data["width"] == reference_width]
                mean_depth_data = mean_data[mean_data["width"] == reference_width]

                fig.add_trace(
                    go.Scatter(
                        x=seed_depth_data["depth"],
                        y=seed_depth_data["mdl"],
                        mode="markers",
                        marker=dict(color=colors[act_idx], size=5, opacity=0.3),
                        name=act,
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=mean_depth_data["depth"],
                        y=mean_depth_data["mean"],
                        mode="lines+markers",
                        line=dict(width=2),
                        name=act,
                        showlegend=row == 1,
                        marker=dict(color=colors[act_idx]),
                    ),
                    row=row,
                    col=1,
                )

                seed_width_data = data[data["depth"] == reference_depth]
                mean_width_data = mean_data[mean_data["depth"] == reference_depth]

                fig.add_trace(
                    go.Scatter(
                        x=seed_width_data["width"],
                        y=seed_width_data["mdl"],
                        mode="markers",
                        marker=dict(size=5, opacity=0.3, color=colors[act_idx]),
                        name=act,
                        showlegend=False,
                    ),
                    row=row,
                    col=2,
                )
                fig.add_trace(
                    go.Scatter(
                        x=mean_width_data["width"],
                        y=mean_width_data["mean"],
                        mode="lines+markers",
                        line=dict(width=2),
                        name=act,
                        showlegend=False,
                        marker=dict(color=colors[act_idx]),
                    ),
                    row=row,
                    col=2,
                )

                # Add legend for multiple activation functions
                if len(ordered_acts) > 1:
                    fig.update_layout(
                        showlegend=True,
                        legend=dict(
                            title="Activation function",
                            yanchor="top",
                            y=0.95,
                            xanchor="right",
                            x=0.95,
                        ),
                    )

        fig.write_image(output_dir / f"{net}_MDL_{dataset}.pdf", format="pdf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/mnt/ssd-1/lucia/24-11-21"),
        help="Path to the directory containing .pth files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset to use for the plots.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/images/sweep_plots"),
        help="Path to the directory to save the output plots.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.dataset == "cifar10":
        print("Prevent user error; dataset awkwardly not in results metadata")
        assert "cifarnet" not in args.data.name

    print("Loading disk data into dataframe...")
    df = load_sweep_data(args.data)

    print("Creating plots...")
    create_plots(df, args.out, args.dataset)


if __name__ == "__main__":
    main()
