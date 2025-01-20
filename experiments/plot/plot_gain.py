from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

from experiments.sweep_eraser import sweep_params
from experiments.plot.plot_mdl import load_sweep_data


def analyze_conv_gain(df: pd.DataFrame, out: Path, tag: str):
    """Create plots showing how convolutional advantage varies with width/depth."""
    out.mkdir(exist_ok=True)

    reference_width = sweep_params["mlp"]["mup_width"]
    reference_depth = sweep_params["mlp"]["mup_depth"]
    width_depths = [(width, reference_depth) for width in sweep_params["mlp"]["widths"]]
    widths_depth = [(reference_width, depth) for depth in sweep_params["mlp"]["depths"]]

    def diff_of_diffs(
        lenet_unerased_loss, lenet_erased_loss, mlp_unerased_loss, mlp_erased_loss
    ):
        """The amount adding a convolution improves the loss on erased dataset - the same thing for vanilla dataset"""
        return (mlp_erased_loss - lenet_erased_loss) - (
            mlp_unerased_loss - lenet_unerased_loss
        )

    # Lists to store results
    width_results = []
    width_individual_results = []
    depth_results = []
    depth_individual_results = []

    for dataset in ["cifar10"]:  # cifarnet
        # Width sweep
        for width, depth in width_depths:
            unerased_mlp_df = df[
                (df["dataset"] == dataset)
                & (df["net_id"] == "mlp")
                & (df["eraser"] == "Control")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]
            leace_mlp_df = df[
                (df["dataset"] == dataset)
                & (df["net_id"] == "mlp")
                & (df["eraser"] == "LEACE")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]
            fake_data_mlp_df = df[
                (df["dataset"] == f"fake-{dataset}")
                & (df["net_id"] == "mlp")
                & (df["eraser"] == "Control")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]

            unerased_lenet_df = df[
                (df["dataset"] == dataset)
                & (df["net_id"] == "lenet")
                & (df["eraser"] == "Control")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]
            leace_lenet_df = df[
                (df["dataset"] == dataset)
                & (df["net_id"] == "lenet")
                & (df["eraser"] == "LEACE")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]
            fake_data_lenet_df = df[
                (df["dataset"] == f"fake-{dataset}")
                & (df["net_id"] == "lenet")
                & (df["eraser"] == "Control")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]

            if any(
                df.empty
                for df in [
                    unerased_lenet_df,
                    leace_lenet_df,
                    fake_data_lenet_df,
                    unerased_mlp_df,
                    leace_mlp_df,
                    fake_data_mlp_df,
                ]
            ):
                continue

            # Calculate means
            unerased_mean_mlp_mdl = unerased_mlp_df["mdl"].mean()
            leaced_mean_mlp_mdl = leace_mlp_df["mdl"].mean()
            erased_mean_mlp_mdl = fake_data_mlp_df["mdl"].mean()

            unerased_mean_lenet_mdl = unerased_lenet_df["mdl"].mean()
            leaced_mean_lenet_mdl = leace_lenet_df["mdl"].mean()
            erased_mean_lenet_mdl = fake_data_lenet_df["mdl"].mean()

            # Calculate differences
            leace_dod = diff_of_diffs(
                unerased_mean_lenet_mdl,
                leaced_mean_lenet_mdl,
                unerased_mean_mlp_mdl,
                leaced_mean_mlp_mdl,
            )
            qleace_dod = diff_of_diffs(
                unerased_mean_lenet_mdl,
                erased_mean_lenet_mdl,
                unerased_mean_mlp_mdl,
                erased_mean_mlp_mdl,
            )

            width_results.append(
                {"width": width, "leace_dod": leace_dod, "qleace_dod": qleace_dod}
            )

            # Calculate individual seed differences
            for seed in unerased_mlp_df["seed"].unique():
                seed_unerased_mlp = unerased_mlp_df[unerased_mlp_df["seed"] == seed][
                    "mdl"
                ].iloc[0]
                seed_leaced_mlp = leace_mlp_df[leace_mlp_df["seed"] == seed][
                    "mdl"
                ].iloc[0]
                seed_erased_mlp = fake_data_mlp_df[fake_data_mlp_df["seed"] == seed][
                    "mdl"
                ].iloc[0]

                seed_unerased_lenet = unerased_lenet_df[
                    unerased_lenet_df["seed"] == seed
                ]["mdl"].iloc[0]
                seed_leaced_lenet = leace_lenet_df[leace_lenet_df["seed"] == seed][
                    "mdl"
                ].iloc[0]
                seed_erased_lenet = fake_data_lenet_df[
                    fake_data_lenet_df["seed"] == seed
                ]["mdl"].iloc[0]

                seed_leace_dod = diff_of_diffs(
                    seed_unerased_lenet,
                    seed_leaced_lenet,
                    seed_unerased_mlp,
                    seed_leaced_mlp,
                )
                seed_qleace_dod = diff_of_diffs(
                    seed_unerased_lenet,
                    seed_erased_lenet,
                    seed_unerased_mlp,
                    seed_erased_mlp,
                )

                width_individual_results.append(
                    {
                        "width": width,
                        "leace_dod": seed_leace_dod,
                        "qleace_dod": seed_qleace_dod,
                        "type": "seed",
                    }
                )

        # Depth sweep
        for width, depth in widths_depth:
            unerased_mlp_df = df[
                (df["dataset"] == dataset)
                & (df["net_id"] == "mlp")
                & (df["eraser"] == "Control")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]
            leace_mlp_df = df[
                (df["dataset"] == dataset)
                & (df["net_id"] == "mlp")
                & (df["eraser"] == "LEACE")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]
            fake_data_mlp_df = df[
                (df["dataset"] == f"fake-{dataset}")
                & (df["net_id"] == "mlp")
                & (df["eraser"] == "Control")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]

            unerased_lenet_df = df[
                (df["dataset"] == dataset)
                & (df["net_id"] == "lenet")
                & (df["eraser"] == "Control")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]
            leace_lenet_df = df[
                (df["dataset"] == dataset)
                & (df["net_id"] == "lenet")
                & (df["eraser"] == "LEACE")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]
            fake_data_lenet_df = df[
                (df["dataset"] == f"fake-{dataset}")
                & (df["net_id"] == "lenet")
                & (df["eraser"] == "Control")
                & (df["act"] == "ReLU")
                & (df["width"] == width)
                & (df["depth"] == depth)
            ]

            if any(
                df.empty
                for df in [
                    unerased_lenet_df,
                    leace_lenet_df,
                    fake_data_lenet_df,
                    unerased_mlp_df,
                    leace_mlp_df,
                    fake_data_mlp_df,
                ]
            ):
                continue

            # Calculate means
            unerased_mean_mlp_mdl = unerased_mlp_df["mdl"].mean()
            leaced_mean_mlp_mdl = leace_mlp_df["mdl"].mean()
            erased_mean_mlp_mdl = fake_data_mlp_df["mdl"].mean()

            unerased_mean_lenet_mdl = unerased_lenet_df["mdl"].mean()
            leaced_mean_lenet_mdl = leace_lenet_df["mdl"].mean()
            erased_mean_lenet_mdl = fake_data_lenet_df["mdl"].mean()

            # Calculate differences
            leace_dod = diff_of_diffs(
                unerased_mean_lenet_mdl,
                leaced_mean_lenet_mdl,
                unerased_mean_mlp_mdl,
                leaced_mean_mlp_mdl,
            )
            qleace_dod = diff_of_diffs(
                unerased_mean_lenet_mdl,
                erased_mean_lenet_mdl,
                unerased_mean_mlp_mdl,
                erased_mean_mlp_mdl,
            )

            depth_results.append(
                {"depth": depth, "leace_dod": leace_dod, "qleace_dod": qleace_dod}
            )

            # Calculate individual seed differences
            for seed in unerased_mlp_df["seed"].unique():
                seed_unerased_mlp = unerased_mlp_df[unerased_mlp_df["seed"] == seed][
                    "mdl"
                ].iloc[0]
                seed_leaced_mlp = leace_mlp_df[leace_mlp_df["seed"] == seed][
                    "mdl"
                ].iloc[0]
                seed_erased_mlp = fake_data_mlp_df[fake_data_mlp_df["seed"] == seed][
                    "mdl"
                ].iloc[0]

                seed_unerased_lenet = unerased_lenet_df[
                    unerased_lenet_df["seed"] == seed
                ]["mdl"].iloc[0]
                seed_leaced_lenet = leace_lenet_df[leace_lenet_df["seed"] == seed][
                    "mdl"
                ].iloc[0]
                seed_erased_lenet = fake_data_lenet_df[
                    fake_data_lenet_df["seed"] == seed
                ]["mdl"].iloc[0]

                seed_leace_dod = diff_of_diffs(
                    seed_unerased_lenet,
                    seed_leaced_lenet,
                    seed_unerased_mlp,
                    seed_leaced_mlp,
                )
                seed_qleace_dod = diff_of_diffs(
                    seed_unerased_lenet,
                    seed_erased_lenet,
                    seed_unerased_mlp,
                    seed_erased_mlp,
                )

                depth_individual_results.append(
                    {
                        "depth": depth,
                        "leace_dod": seed_leace_dod,
                        "qleace_dod": seed_qleace_dod,
                        "type": "seed",
                    }
                )

        # Create plots
        width_df = pd.DataFrame(width_results)
        width_individual_df = pd.DataFrame(width_individual_results)
        depth_df = pd.DataFrame(depth_results)
        depth_individual_df = pd.DataFrame(depth_individual_results)

        fig = make_subplots(rows=1, cols=2)  # Removed subplot titles

        # Width subplot (left)
        fig.add_trace(
            px.line(width_df, x="width", y="leace_dod")
            .data[0]
            .update(line_color="red"),
            row=1,
            col=1,
        )
        fig.add_trace(
            px.line(width_df, x="width", y="qleace_dod")
            .data[0]
            .update(line_color="blue"),
            row=1,
            col=1,
        )
        fig.add_trace(
            px.scatter(width_individual_df, x="width", y="leace_dod")
            .data[0]
            .update(marker=dict(size=5, opacity=0.5, color="red"), showlegend=False),
            row=1,
            col=1,
        )
        fig.add_trace(
            px.scatter(width_individual_df, x="width", y="qleace_dod")
            .data[0]
            .update(marker=dict(size=5, opacity=0.5, color="blue"), showlegend=False),
            row=1,
            col=1,
        )

        # Depth subplot (right)
        fig.add_trace(
            px.line(depth_df, x="depth", y="leace_dod")
            .data[0]
            .update(line_color="red", name="1st order"),
            row=1,
            col=2,
        )
        fig.add_trace(
            px.line(depth_df, x="depth", y="qleace_dod")
            .data[0]
            .update(line_color="blue", name="2nd order"),
            row=1,
            col=2,
        )
        fig.add_trace(
            px.scatter(depth_individual_df, x="depth", y="leace_dod")
            .data[0]
            .update(marker=dict(size=5, opacity=0.5, color="red"), showlegend=False),
            row=1,
            col=2,
        )
        fig.add_trace(
            px.scatter(depth_individual_df, x="depth", y="qleace_dod")
            .data[0]
            .update(marker=dict(size=5, opacity=0.5, color="blue"), showlegend=False),
            row=1,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=350,
            width=1000,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=50, r=30, t=30, b=50),
        )

        fig.update_xaxes(
            title_text="Width",
            type="log",
            tickvals=[
                2**i
                for i in range(
                    int(np.log2(min(width_df["width"]))),
                    int(np.log2(max(width_df["width"]))) + 1,
                )
            ],
            ticktext=[
                f"2<sup>{i}</sup>"
                for i in range(
                    int(np.log2(min(width_df["width"]))),
                    int(np.log2(max(width_df["width"]))) + 1,
                )
            ],
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text="Depth",
            type="log",
            tickvals=[
                2**i
                for i in range(
                    int(np.log2(min(depth_df["depth"]))),
                    int(np.log2(max(depth_df["depth"]))) + 1,
                )
            ],
            ticktext=[
                f"2<sup>{i}</sup>"
                for i in range(
                    int(np.log2(min(depth_df["depth"]))),
                    int(np.log2(max(depth_df["depth"]))) + 1,
                )
            ],
            row=1,
            col=2,
        )

        # Update y-axes
        fig.update_yaxes(
            title_text="Difference of differences",
            range=[
                0,
                max(width_df["leace_dod"].max(), width_df["qleace_dod"].max()) * 1.1,
            ],
            row=1,
            col=1,
        )

        # Right plot lines
        for i in range(4, 6):
            fig.data[i].showlegend = True
 
        # Write the combined figure
        fig.write_image(out / f"combined_dod{'_' + tag if tag else ''}.pdf")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/mnt/ssd-1/lucia/24-11-21"),
        help="Path to the directory containing .pth files.",
    )
    parser.add_argument("--out", type=Path, default="data/images/sweep_plots")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    df = load_sweep_data(args.data)

    analyze_conv_gain(df, args.out, args.tag)
