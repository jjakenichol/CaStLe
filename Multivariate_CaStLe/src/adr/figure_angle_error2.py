from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def main():
    parser = argparse.ArgumentParser(
        description="Generate the angle_error2 figure from extracted CSV data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/angle_error2_exact_plot_data.csv",
        help="Path to extracted figure CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to write figure files",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=95,
        help="Confidence interval level for error bands (e.g. 95 for 95%% CI)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_data = pd.read_csv(input_path).copy()

    required_columns = [
        "diff_coefs_serial",
        "velocity_magnitude",
        "M-Stencil Angle Difference",
    ]
    missing = [c for c in required_columns if c not in plot_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    plot_data["angle_error_exceeds_45"] = (
        plot_data["M-Stencil Angle Difference"] > 45
    )

    diffs_to_filter = [0.005, 0.01, 0.05, 0.1, 0.2, 0.4]
    vels_to_filter = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    plot_data = plot_data[
        plot_data["diff_coefs_serial"].isin(diffs_to_filter)
    ].copy()
    plot_data = plot_data[
        plot_data["velocity_magnitude"].isin(vels_to_filter)
    ].copy()

    sns.set_theme(style="whitegrid", context="talk")

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: diffusion coefficient
    sns.lineplot(
        x="diff_coefs_serial",
        y="M-Stencil Angle Difference",
        data=plot_data,
        errorbar=("ci", args.ci_level),
        err_kws={"alpha": 0.4},
        ax=ax1,
        color="black",
        linewidth=1,
        marker="o",
    )

    sns.lineplot(
        x="diff_coefs_serial",
        y="M-Stencil Angle Difference",
        data=plot_data,
        estimator=np.median,
        errorbar=("ci", args.ci_level),
        err_kws={
            "facecolor": "none",
            "edgecolor": "black",
            "hatch": "////",
            "alpha": 0.4,
        },
        ax=ax1,
        color="black",
        linewidth=1,
        marker="^",
        linestyle="--",
    )

    ax1.set_xlabel("Diffusion Coefficient")
    ax1.set_ylabel("Angle Estimation Error", color="black")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}°"))
    ax1.set_ylim(0, 65)
    yticks = [0, 10, 20, 30, 40, 50, 60]
    ax1.set_yticks(yticks)
    ax1.grid(color="grey", linestyle="-", alpha=0.5)

    ax2 = ax1.twinx()
    sns.lineplot(
        x="diff_coefs_serial",
        y="angle_error_exceeds_45",
        data=plot_data,
        estimator="mean",
        errorbar=None,
        ax=ax2,
        color="green",
        linewidth=3,
        marker="s",
    )
    ax2.set_ylabel("")
    ax2.set_yticks([])
    ax2.grid(False)

    # Right panel: advection velocity
    right_panel_data = plot_data[plot_data["velocity_magnitude"] <= 3.0].copy()

    sns.lineplot(
        x="velocity_magnitude",
        y="M-Stencil Angle Difference",
        data=right_panel_data,
        errorbar=("ci", args.ci_level),
        err_kws={"alpha": 0.4},
        ax=ax3,
        color="black",
        linewidth=1,
        marker="o",
    )

    sns.lineplot(
        x="velocity_magnitude",
        y="M-Stencil Angle Difference",
        data=right_panel_data,
        estimator=np.median,
        errorbar=("ci", args.ci_level),
        err_kws={
            "facecolor": "none",
            "edgecolor": "black",
            "hatch": "////",
            "alpha": 0.4,
        },
        ax=ax3,
        color="black",
        linewidth=1,
        marker="^",
        linestyle="--",
    )

    ax3.set_xlabel("Advection Velocity")
    ax3.set_ylabel("")
    ax3.set_ylim(0, 65)
    ax3.set_yticks([])
    ax3.set_xlim(right=3.0)
    for y in yticks:
        ax3.axhline(y=y, color="grey", linestyle="-", alpha=0.5)

    ax4 = ax3.twinx()
    sns.lineplot(
        x="velocity_magnitude",
        y="angle_error_exceeds_45",
        data=right_panel_data,
        estimator="mean",
        errorbar=None,
        ax=ax4,
        color="green",
        linewidth=3,
        marker="s",
    )
    ax4.set_ylabel("Proportion with Angle Error > 45°", color="green")
    ax4.tick_params(axis="y", colors="green")
    ax4.yaxis.set_label_coords(1.115, 0.5)
    ax4.grid(False)
    ax4.set_xlim(ax3.get_xlim())

    custom_handles = [
        Line2D(
            [],
            [],
            color="black",
            lw=1,
            marker="o",
            linestyle="-",
            label="Mean",
        ),
        Line2D(
            [],
            [],
            color="black",
            lw=1,
            marker="^",
            linestyle="--",
            label="Median",
        ),
    ]
    ax3.legend(handles=custom_handles, loc="upper right", frameon=True)

    ax2.set_ylim(ax4.get_ylim())

    plt.tight_layout()
    plt.savefig(output_dir / "angle_error2.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "angle_error2.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Wrote: {output_dir / 'angle_error2.pdf'}")
    print(f"Wrote: {output_dir / 'angle_error2.png'}")


if __name__ == "__main__":
    main()
