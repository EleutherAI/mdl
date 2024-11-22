import pandas as pd
from pathlib import Path
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import numpy as np

from experiments.sweep_eraser import sweep_params

import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


def load_sweep_data(data_path: Path) -> pd.DataFrame:
    """Load and parse sweep data files into a DataFrame."""
    records = []

    for file in data_path.glob("*.pth"):
        # Parse filename
        net, act, width, depth, eraser, _ = file.stem.split("_")
        width = int(width.split("=")[1])
        depth = int(depth.split("=")[1])

        # Load data and create records
        data = torch.load(file)
        for seed, mdl_result in enumerate(data):
            records.append(
                {
                    "net": net,
                    "act": act,
                    "width": width,
                    "depth": depth,
                    "eraser": eraser,
                    "seed": seed,
                    "mdl": mdl_result.mdl,
                    "ce_curve": mdl_result.ce_curve,
                    "sample_sizes": mdl_result.sample_sizes,
                    "total_trials": mdl_result.total_trials,
                }
            )

    return pd.DataFrame(records)


def create_plots(df: pd.DataFrame, output_dir: Path):
    """Create plots for each network type. Include results over three activation functions for MLPs."""
    output_dir.mkdir(exist_ok=True)

    colors = px.colors.qualitative.Set1

    eraser_types = ["control", "leace", "qleace"]

    df = df.sort_values(["depth", "width"])

    for net in df["net"].unique():
        subplot_titles = [
            eraser.upper() if eraser != "control" else eraser.title()
            for eraser in eraser_types
        ]
        subplot_titles = [
            title for title in sum(zip(subplot_titles, subplot_titles), ())
        ]

        fig = make_subplots(
            rows=len(eraser_types),
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.2,
            row_heights=[400] * len(eraser_types),
        )
        fig.update_layout(
            title=f"{net.title()} Network Analysis",
            height=300 * len(eraser_types),
            width=1200,
            showlegend=False,
        )

        fig.update_yaxes(matches="y1")

        reference_width = sweep_params[net]["mup_width"]
        reference_depth = sweep_params[net]["mup_depth"]

        for row, eraser in enumerate(eraser_types, 1):
            if row == len(eraser_types):
                fig.update_xaxes(title_text="Depth", row=row, col=1)
                fig.update_xaxes(title_text="Width", row=row, col=2)

            for col, type in zip([1, 2], ["depth", "width"]):
                fig.update_xaxes(
                    type="log",
                    row=row,
                    col=col,
                    tickvals=[
                        2**i
                        for i in range(
                            int(np.log2(min(df[type]))),
                            int(np.log2(max(df[type]))) + 1,
                        )
                    ],
                    ticktext=[
                        f"2^{i}"
                        for i in range(
                            int(np.log2(min(df[type]))),
                            int(np.log2(max(df[type]))) + 1,
                        )
                    ],
                )
            fig.update_yaxes(title_text="MDL (bits per sample)", row=row, col=1)

            for act_idx, act in enumerate(df["act"].unique()):
                data = df[
                    (df["eraser"] == eraser)
                    & (df["act"] == act)
                    & (df["net"] == net)
                ]
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
                        name=f"{act}",
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
                        name=f"{act}",
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
                        name=f"{act}",
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
                        name=f"{act}",
                        showlegend=False,
                        marker=dict(color=colors[act_idx]),
                    ),
                    row=row,
                    col=2,
                )

                # If more than one act type
                if len(df["act"].unique()) > 1:
                    fig.update_layout(
                        showlegend=True,
                        legend=dict(
                            title="Activation Types",
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                    )

        fig.write_image(output_dir / f"{net}_mdl_analysis.pdf", format="pdf")


def main():
    data_path = Path("/mnt/ssd-1/lucia/24-11-21")
    output_dir = Path("data/images/sweep_plots")

    print("Loading disk data into dataframe...")
    df = load_sweep_data(data_path)

    print("Creating plots...")
    create_plots(df, output_dir)


if __name__ == "__main__":
    main()
