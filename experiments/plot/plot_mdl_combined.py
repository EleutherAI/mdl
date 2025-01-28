import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.io as pio
import plotly.express as px


from experiments.plot.plot_mdl import DISPLAY_NAMES, load_sweep_data
from experiments.sweep_eraser import sweep_params

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

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Depth sweep", "Width sweep"],
            horizontal_spacing=0.1,
        )

        fig.update_layout(
            title=f"Minimum description length over 5 seeds ({net})",
            height=400,
            width=1200,
            showlegend=True,
            legend=dict(
                title="Eraser type",
                yanchor="top",
                y=0.95,
                xanchor="right",
                x=0.95,
            ),
        )

        # Set up axes
        for col in [1, 2]:
            fig.update_yaxes(
                title_text="MDL (bits per sample)" if col == 1 else "",
                showticklabels=True,
                row=1,
                col=col,
            )

        # Set up x-axes
        for col, param in enumerate(["depth", "width"], 1):
            param_data = df[param].unique()
            fig.update_xaxes(
                title_text=param.title(),
                type="log",
                tickvals=[
                    2**i
                    for i in range(
                        int(np.log2(min(param_data))),
                        int(np.log2(max(param_data))) + 1,
                    )
                ],
                ticktext=[
                    f"2<sup>{i}</sup>"
                    for i in range(
                        int(np.log2(min(param_data))),
                        int(np.log2(max(param_data))) + 1,
                    )
                ],
                row=1,
                col=col,
            )

        net_df = df[(df["net"] == net) & (df["act"] == "ReLU")]

        # Plot data for each eraser type
        for eraser_idx, eraser in enumerate(ordered_erasers):
            data = net_df[net_df["eraser"] == eraser]
            if data.empty:
                continue

            mean_data = (
                data.groupby(["width", "depth"])["mdl"]
                .agg(["mean", "std"])
                .reset_index()
            )

            # Plot depth sweep (fixed width)
            depth_data = data[data["width"] == reference_width]
            mean_depth = mean_data[mean_data["width"] == reference_width]

            # Plot width sweep (fixed depth)
            width_data = data[data["depth"] == reference_depth]
            mean_width = mean_data[mean_data["depth"] == reference_depth]

            for col, (param_data, mean_param) in enumerate(
                [
                    (depth_data, mean_depth),
                    (width_data, mean_width),
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
                        marker=dict(color=colors[eraser_idx], size=5, opacity=0.3),
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )

                # Plot mean line
                fig.add_trace(
                    go.Scatter(
                        x=mean_param[param],
                        y=mean_param["mean"],
                        mode="lines+markers",
                        line=dict(width=2),
                        marker=dict(color=colors[eraser_idx]),
                        name=eraser,
                        showlegend=(col == 1),
                    ),
                    row=1,
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