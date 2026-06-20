from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from st_risk.paths import current_results_dir, ensure_results_layout, project_root, results_file, set_selected_run


DEFAULT_RUN_ID = "2026-06-20-revision-figures-v2-multimodel-donor"
PUBLICATION_FONT_FAMILY = "Liberation Sans"
PUBLICATION_FONT_FILES = [
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"),
]
OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
]
SCENARIO_COLORS = {
    "clean": "#0072B2",
    "low_depth": "#E69F00",
    "marker_dropout": "#D55E00",
    "diffuse_mixture": "#009E73",
}
SCENARIO_LABELS = {
    "clean": "clean",
    "low_depth": "low depth",
    "marker_dropout": "marker dropout",
    "diffuse_mixture": "diffuse mixture",
}
PERTURBATION_LABELS = {
    "baseline_full_reference": "full reference",
    "marker_gene_dropout": "marker gene dropout",
    "dominant_celltype_dropout": "dominant cell-type dropout",
    "excit_inhib_label_coarsening": "Excit/Inhib coarsening",
    "out_of_reference_contamination": "out-of-reference contamination",
}
SCORE_LABELS = {
    "risk_score": "ReliST risk",
    "local_uncertainty_risk_score": "local + ambiguity",
    "reference_risk_score": "reference risk",
    "abundance_entropy_risk": "entropy",
    "inverse_top1_margin": "inverse top-1 margin",
    "inverse_max_abundance": "inverse max abundance",
    "native_uncertainty_risk": "native uncertainty",
    "cross_model_disagreement": "cross-model disagreement",
    "phi_uncertainty": "ambiguity",
    "phi_local": "local",
    "phi_reference": "reference",
    "full_risk": "full risk",
    "local_uncertainty": "local + ambiguity",
    "uncertainty_only": "ambiguity only",
    "local_only": "local only",
    "reference_only": "reference only",
    "uncertainty_reference": "ambiguity + reference",
    "local_reference": "local + reference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build iScience revision single-panel figures.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root() / "results" / "revision_figures",
        help="Revision figure result root.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run id under results-root/runs/.")
    parser.add_argument(
        "--known-composition-root",
        type=Path,
        default=project_root() / "results" / "revision_known_composition_benchmark",
        help="Known-composition benchmark result root or run directory.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=project_root() / "results" / "revision_uncertainty_baseline_eval",
        help="Uncertainty baseline result root or run directory.",
    )
    parser.add_argument(
        "--multimodel-root",
        type=Path,
        default=project_root() / "results" / "revision_known_composition_multimodel_eval",
        help="Known-composition multi-model evaluation root or run directory.",
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=project_root() / "results" / "revision_component_ablation",
        help="Component ablation result root or run directory.",
    )
    parser.add_argument(
        "--threshold-root",
        type=Path,
        default=project_root() / "results" / "revision_threshold_sensitivity",
        help="Threshold sensitivity result root or run directory.",
    )
    parser.add_argument(
        "--reference-perturbation-root",
        type=Path,
        default=project_root() / "results" / "revision_reference_perturbation",
        help="Reference perturbation result root or run directory.",
    )
    parser.add_argument(
        "--reference-perturbation-component-root",
        type=Path,
        default=project_root() / "results" / "revision_reference_perturbation_component_ablation",
        help="Reference perturbation component-ablation result root or run directory.",
    )
    return parser.parse_args()


def register_publication_fonts() -> str:
    for font_path in PUBLICATION_FONT_FILES:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    return PUBLICATION_FONT_FAMILY


def publication_style() -> None:
    font_family = register_publication_fonts()
    sns.set_theme(style="white", context="paper", font=font_family)
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "legend.title_fontsize": 7.4,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.pad_inches": 0.03,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> tuple[Path, Path]:
    pdf_path = output_dir / "figures" / "pdf" / f"{name}.pdf"
    png_path = output_dir / "figures" / "png_preview" / f"{name}.png"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def require_csv(path: Path, *, index_col: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, index_col=index_col)


def selected_run(root_or_run: Path) -> Path:
    return current_results_dir(root_or_run)


def add_panel_label(ax: plt.Axes, label: str = "A") -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _schematic_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    title: str,
    body: str,
    facecolor: str,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=0.8,
        edgecolor="#4D4D4D",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height - 0.045, title, ha="center", va="top", fontsize=8.6, fontweight="bold")
    ax.text(x + width / 2, y + height / 2 - 0.02, body, ha="center", va="center", fontsize=7.0, linespacing=1.2)


def _schematic_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color="#4D4D4D",
            shrinkA=3,
            shrinkB=3,
        )
    )


def plot_known_composition_workflow(output_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.0, 2.55), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "A")

    boxes = [
        ((0.04, 0.34), "snRNA reference", "split cells\ntrain signatures\nhold out cells", "#E6F2F8"),
        ((0.29, 0.34), "Pseudo-spots", "sample 4-10 cells\nknown composition\nsmooth pseudo-space", "#E8F4EA"),
        ((0.54, 0.34), "Lightweight deconvolution", "NNLS signature fit\npredicted abundance\ntrue error", "#F0F0F0"),
        ((0.79, 0.34), "ReliST evaluation", "risk features\nrisk-error AUROC\nselective filtering", "#FCE8D5"),
    ]
    for xy, title, body, color in boxes:
        _schematic_box(ax, xy, 0.17, 0.34, title=title, body=body, facecolor=color)
    _schematic_arrow(ax, (0.21, 0.51), (0.29, 0.51))
    _schematic_arrow(ax, (0.46, 0.51), (0.54, 0.51))
    _schematic_arrow(ax, (0.71, 0.51), (0.79, 0.51))
    ax.text(
        0.5,
        0.12,
        "Known cell-type composition is used only for validation; the risk layer does not use true composition at runtime.",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#555555",
    )
    return save_figure(fig, output_dir, "fig_revision_1a_known_composition_workflow")


def plot_risk_error_scatter(known_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    table = require_csv(known_dir / "tables" / "known_composition_risk_error_table.csv", index_col=0)
    summary = require_csv(known_dir / "tables" / "known_composition_score_error_summary.csv")
    risk_summary = summary.loc[summary["score_name"] == "risk_score"].iloc[0]

    fig, ax = plt.subplots(figsize=(3.55, 3.15), constrained_layout=True)
    for scenario, group in table.groupby("scenario", sort=False):
        ax.scatter(
            group["risk_score"],
            group["total_variation_error"],
            s=9,
            alpha=0.42,
            linewidth=0,
            color=SCENARIO_COLORS.get(str(scenario), "#7F7F7F"),
            label=SCENARIO_LABELS.get(str(scenario), str(scenario)),
        )

    bins = pd.qcut(table["risk_score"], q=10, duplicates="drop")
    trend = table.groupby(bins, observed=True).agg(
        mean_risk=("risk_score", "mean"),
        mean_error=("total_variation_error", "mean"),
        q25_error=("total_variation_error", lambda x: float(np.quantile(x, 0.25))),
        q75_error=("total_variation_error", lambda x: float(np.quantile(x, 0.75))),
    )
    ax.plot(trend["mean_risk"], trend["mean_error"], color="#000000", marker="o", markersize=3.5, linewidth=1.2)
    ax.fill_between(
        trend["mean_risk"].to_numpy(dtype=float),
        trend["q25_error"].to_numpy(dtype=float),
        trend["q75_error"].to_numpy(dtype=float),
        color="#000000",
        alpha=0.12,
        linewidth=0,
    )
    ax.text(
        0.03,
        0.97,
        f"Spearman = {risk_summary['spearman_error']:.3f}\nAUROC(top 20%) = {risk_summary['auroc_top20_error']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "linewidth": 0.5},
    )
    ax.set_xlabel("ReliST risk score")
    ax.set_ylabel("True composition error")
    ax.set_title("Risk-error association in pseudo-spots")
    ax.legend(frameon=False, loc="lower right", title="Scenario", markerscale=1.5)
    despine(ax)
    return save_figure(fig, output_dir, "fig_revision_1b_risk_error_scatter")


def plot_selective_error_curve(known_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    curve = require_csv(known_dir / "tables" / "known_composition_selective_error_curve.csv")
    keep_scores = ["risk_score", "local_uncertainty_risk_score", "reference_risk_score", "abundance_entropy_risk"]
    curve = curve.loc[curve["score_name"].isin(keep_scores)].copy()

    fig, ax = plt.subplots(figsize=(3.55, 2.85), constrained_layout=True)
    styles = {
        "risk_score": ("#0072B2", "o", "-"),
        "local_uncertainty_risk_score": ("#D55E00", "s", "--"),
        "reference_risk_score": ("#009E73", "^", "-."),
        "abundance_entropy_risk": ("#7F7F7F", "D", ":"),
    }
    for score_name, group in curve.groupby("score_name", sort=False):
        color, marker, linestyle = styles.get(score_name, ("#333333", "o", "-"))
        ax.plot(
            group["keep_fraction"],
            group["mean_error"],
            marker=marker,
            linewidth=1.2,
            markersize=3.6,
            linestyle=linestyle,
            color=color,
            label=SCORE_LABELS.get(score_name, score_name),
        )
    full = curve["full_mean_error"].iloc[0]
    ax.axhline(full, color="#555555", linewidth=0.8, linestyle="--")
    ax.text(0.505, full + 0.004, "full set", fontsize=6.8, color="#555555", va="bottom")
    ax.set_xlabel("Low-risk fraction retained")
    ax.set_ylabel("Mean true error")
    ax.set_title("Selective filtering reduces true error")
    ax.set_xlim(0.48, 1.02)
    ax.legend(frameon=False, loc="upper left")
    despine(ax)
    return save_figure(fig, output_dir, "fig_revision_1c_selective_error_curve")


def plot_baseline_comparison(multimodel_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    summary = require_csv(multimodel_dir / "tables" / "known_composition_multimodel_score_summary.csv")
    selected_scores = {
        "risk_score",
        "local_uncertainty_risk_score",
        "abundance_entropy_risk",
        "native_uncertainty_risk",
        "cross_model_disagreement",
    }
    plot_df = summary.loc[summary["score_name"].isin(selected_scores)].copy()
    plot_df = plot_df.dropna(subset=["auroc_top20_error"])
    plot_df["label"] = plot_df["model_key"].astype(str) + " | " + plot_df["score_name"].map(SCORE_LABELS).fillna(plot_df["score_name"])
    plot_df = plot_df.sort_values(["model_key", "auroc_top20_error"], ascending=[True, True])
    colors = ["#0072B2" if name in {"risk_score", "local_uncertainty_risk_score"} else "#999999" for name in plot_df["score_name"]]

    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    ax.barh(plot_df["label"], plot_df["auroc_top20_error"], color=colors, edgecolor="#333333", linewidth=0.4)
    ax.axvline(0.5, color="#666666", linestyle="--", linewidth=0.8)
    for y, value in enumerate(plot_df["auroc_top20_error"]):
        ax.text(value + 0.008, y, f"{value:.3f}", va="center", fontsize=6.8)
    ax.set_xlabel("AUROC for top 20% true-error spots")
    ax.set_ylabel("")
    ax.set_xlim(0.48, max(0.90, plot_df["auroc_top20_error"].max() + 0.06))
    ax.set_title("Multi-model true-error benchmark")
    despine(ax)
    return save_figure(fig, output_dir, "fig_revision_1d_baseline_comparison")


def plot_dlpfc_contract_fairness(baseline_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    aggregate = require_csv(baseline_dir / "tables" / "dlpfc_contract_fairness_aggregate.csv")
    plot_df = aggregate.loc[
        aggregate["comparison_block"].isin(["all_dlpfc_anchored_proxies", "marker_discordance_only", "reference_residual_only", "structure_proxies"])
    ].copy()
    label_map = {
        "all_dlpfc_anchored_proxies": "All anchors",
        "marker_discordance_only": "Marker discordance",
        "reference_residual_only": "Signature residual",
        "structure_proxies": "Structure anchors",
    }
    plot_df["label"] = plot_df["comparison_block"].map(label_map)
    x = np.arange(plot_df.shape[0])
    width = 0.34

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    ax.bar(
        x - width / 2,
        plot_df["mean_risk_axis_auc"],
        width,
        color="#0072B2",
        edgecolor="#333333",
        linewidth=0.4,
        label="risk-axis score",
    )
    ax.bar(
        x + width / 2,
        plot_df["mean_best_common_auc"],
        width,
        color="#999999",
        edgecolor="#333333",
        linewidth=0.4,
        label="best common feature",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=25, ha="right")
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("Mean AUROC")
    ax.set_title("DLPFC anchored proxy control")
    ax.legend(frameon=False, loc="upper right")
    despine(ax)
    return save_figure(fig, output_dir, "fig_revision_s0_dlpfc_contract_fairness")


def plot_component_ablation(ablation_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    summary = require_csv(ablation_dir / "tables" / "component_ablation_summary.csv")
    selected = ["full_risk", "local_uncertainty", "uncertainty_only", "local_only", "reference_only", "uncertainty_reference", "local_reference"]
    plot_df = summary.loc[summary["score_name"].isin(selected)].copy()
    plot_df["label"] = plot_df["score_name"].map(SCORE_LABELS).fillna(plot_df["score_name"])
    plot_df = plot_df.sort_values("auroc_top20_error", ascending=True)
    colors = ["#0072B2" if name == "full_risk" else "#D55E00" if name == "local_uncertainty" else "#999999" for name in plot_df["score_name"]]

    fig, ax = plt.subplots(figsize=(4.1, 3.05), constrained_layout=True)
    ax.barh(plot_df["label"], plot_df["auroc_top20_error"], color=colors, edgecolor="#333333", linewidth=0.4)
    ax.axvline(0.5, color="#666666", linestyle="--", linewidth=0.8)
    for y, value in enumerate(plot_df["auroc_top20_error"]):
        ax.text(value + 0.008, y, f"{value:.3f}", va="center", fontsize=6.8)
    ax.set_xlabel("AUROC for top 20% true-error spots")
    ax.set_title("Component ablation")
    ax.set_xlim(0.48, max(0.85, plot_df["auroc_top20_error"].max() + 0.06))
    despine(ax)
    return save_figure(fig, output_dir, "fig_revision_s1_component_ablation")


def plot_threshold_sensitivity(threshold_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    keep = require_csv(threshold_dir / "tables" / "threshold_keep_fraction_summary.csv")
    keep = keep.loc[
        keep["score_name"].isin(["risk_score", "local_uncertainty_risk_score", "reference_risk_score", "random_reference", "oracle_error"])
    ].copy()
    styles = {
        "risk_score": ("#0072B2", "o", "-"),
        "local_uncertainty_risk_score": ("#D55E00", "s", "--"),
        "reference_risk_score": ("#009E73", "^", "-."),
        "random_reference": ("#777777", "x", ":"),
        "oracle_error": ("#000000", "D", "-"),
    }
    fig, ax = plt.subplots(figsize=(3.65, 2.9), constrained_layout=True)
    for score_name, group in keep.groupby("score_name", sort=False):
        color, marker, linestyle = styles.get(score_name, ("#333333", "o", "-"))
        ax.plot(
            group["abstain_fraction"],
            group["kept_error_reduction_vs_full"],
            color=color,
            marker=marker,
            markersize=3.8,
            linewidth=1.2,
            linestyle=linestyle,
            label=SCORE_LABELS.get(score_name, score_name),
        )
    ax.axvline(0.20, color="#666666", linestyle="--", linewidth=0.8)
    ax.text(0.205, 0.012, "example 20% review", rotation=90, fontsize=6.8, color="#555555", va="bottom")
    ax.set_xlabel("Fraction sent for review")
    ax.set_ylabel("Error reduction in retained spots")
    ax.set_title("Coverage-error review budgets")
    ax.set_xlim(0.03, 0.32)
    ax.legend(frameon=False, loc="upper left")
    despine(ax)
    return save_figure(fig, output_dir, "fig_revision_s2_threshold_sensitivity")


def plot_high_risk_capture(threshold_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    high = require_csv(threshold_dir / "tables" / "threshold_high_risk_summary.csv")
    high = high.loc[high["score_name"] == "risk_score"].copy()

    fig, ax = plt.subplots(figsize=(3.5, 2.75), constrained_layout=True)
    ax.plot(
        high["high_risk_fraction_target"],
        high["top20_true_error_capture_fraction"],
        color="#0072B2",
        marker="o",
        markersize=4,
        linewidth=1.2,
        label="capture",
    )
    ax.plot(
        high["high_risk_fraction_target"],
        high["top20_true_error_precision"],
        color="#D55E00",
        marker="s",
        markersize=4,
        linewidth=1.2,
        linestyle="--",
        label="precision",
    )
    ax.set_xlabel("High-risk fraction flagged")
    ax.set_ylabel("Fraction")
    ax.set_title("High-risk review captures true-error spots")
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=False, loc="upper left")
    despine(ax)
    return save_figure(fig, output_dir, "fig_revision_s3_high_risk_capture")


def plot_reference_perturbation(perturbation_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    summary = require_csv(perturbation_dir / "tables" / "reference_perturbation_summary.csv")
    plot_df = summary.copy()
    plot_df["label"] = plot_df["scenario"].map(PERTURBATION_LABELS).fillna(plot_df["scenario"])
    plot_df = plot_df.sort_values("mean_true_error", ascending=True)
    colors = [
        "#0072B2" if scenario == "baseline_full_reference" else "#D55E00" if scenario == "out_of_reference_contamination" else "#999999"
        for scenario in plot_df["scenario"]
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3.05), constrained_layout=True)
    y = np.arange(plot_df.shape[0])
    ax.barh(y, plot_df["mean_true_error"], color=colors, edgecolor="#333333", linewidth=0.4)
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            float(row.mean_true_error) + 0.012,
            idx,
            f"AUROC {row.auroc_top20_error:.3f}; rho {row.spearman_error:.3f}",
            va="center",
            fontsize=6.6,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Mean true composition error")
    ax.set_title("Reference perturbation stress test")
    ax.set_xlim(0, max(0.78, plot_df["mean_true_error"].max() + 0.18))
    despine(ax)
    return save_figure(fig, output_dir, "fig_revision_s4_reference_perturbation")


def plot_reference_component_delta(component_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    delta = require_csv(component_dir / "tables" / "reference_perturbation_full_vs_no_reference.csv")
    plot_df = delta.copy()
    plot_df["label"] = plot_df["scenario"].map(PERTURBATION_LABELS).fillna(plot_df["scenario"])
    plot_df = plot_df.sort_values("delta_full_minus_leave_out_reference_auroc", ascending=True)
    colors = [
        "#0072B2" if value >= 0 else "#D55E00"
        for value in plot_df["delta_full_minus_leave_out_reference_auroc"].to_numpy(dtype=float)
    ]

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    y = np.arange(plot_df.shape[0])
    ax.barh(y, plot_df["delta_full_minus_leave_out_reference_auroc"], color=colors, edgecolor="#333333", linewidth=0.4)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    for idx, value in enumerate(plot_df["delta_full_minus_leave_out_reference_auroc"]):
        ha = "left" if value >= 0 else "right"
        offset = 0.004 if value >= 0 else -0.004
        ax.text(value + offset, idx, f"{value:+.3f}", va="center", ha=ha, fontsize=6.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Delta AUROC: full risk - no-reference")
    ax.set_title("Reference component is scenario-dependent")
    x_abs = max(0.05, float(np.nanmax(np.abs(plot_df["delta_full_minus_leave_out_reference_auroc"]))) + 0.02)
    ax.set_xlim(-x_abs, x_abs)
    despine(ax)
    return save_figure(fig, output_dir, "fig_revision_s5_reference_component_delta")


def copy_source_tables(run_dir: Path, source_dirs: dict[str, Path]) -> None:
    table_rows = []
    for source_name, source_dir in source_dirs.items():
        for table_path in sorted((source_dir / "tables").glob("*.csv")):
            table_rows.append(
                {
                    "source_name": source_name,
                    "source_table": str(table_path),
                    "filename": table_path.name,
                }
            )
    pd.DataFrame(table_rows).to_csv(results_file(run_dir, "tables", "revision_figure_source_tables.csv"), index=False)


def write_figure_index(run_dir: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(results_file(run_dir, "tables", "revision_figure_index.csv"), index=False)


def build_contact_sheet(run_dir: Path, index_rows: list[dict[str, str]]) -> Path:
    png_paths = [Path(row["png_preview_path"]) for row in index_rows]
    thumb_w, thumb_h = 520, 360
    label_h = 42
    cols = 2
    rows = int(np.ceil(len(png_paths) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(str(PUBLICATION_FONT_FILES[0]), 16)
    except OSError:
        font = None
    for idx, path in enumerate(png_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w - 24, thumb_h - 20), Image.Resampling.LANCZOS)
        x0 = (idx % cols) * thumb_w + (thumb_w - image.width) // 2
        y0 = (idx // cols) * (thumb_h + label_h) + 8
        sheet.paste(image, (x0, y0))
        draw.text(
            ((idx % cols) * thumb_w + 12, (idx // cols) * (thumb_h + label_h) + thumb_h + 8),
            path.stem,
            fill=(30, 30, 30),
            font=font,
        )
    out = run_dir / "figures" / "revision_figures_contact_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main() -> int:
    args = parse_args()
    publication_style()
    run_dir = args.results_root / "runs" / args.run_id
    ensure_results_layout(run_dir)
    (run_dir / "figures" / "pdf").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures" / "png_preview").mkdir(parents=True, exist_ok=True)
    set_selected_run(args.results_root, args.run_id)

    known_dir = selected_run(args.known_composition_root)
    baseline_dir = selected_run(args.baseline_root)
    multimodel_dir = selected_run(args.multimodel_root)
    ablation_dir = selected_run(args.ablation_root)
    threshold_dir = selected_run(args.threshold_root)
    perturbation_dir = selected_run(args.reference_perturbation_root)
    perturbation_component_dir = selected_run(args.reference_perturbation_component_root)

    plot_specs = [
        (
            "fig_revision_1a_known_composition_workflow",
            "Figure X A",
            "known-composition workflow",
            plot_known_composition_workflow,
            (),
            str(known_dir / "metadata" / "known_composition_benchmark.json"),
        ),
        (
            "fig_revision_1b_risk_error_scatter",
            "Figure X B",
            "risk-error relationship",
            plot_risk_error_scatter,
            (known_dir,),
            str(known_dir / "tables" / "known_composition_risk_error_table.csv"),
        ),
        (
            "fig_revision_1c_selective_error_curve",
            "Figure X C",
            "selective filtering curve",
            plot_selective_error_curve,
            (known_dir,),
            str(known_dir / "tables" / "known_composition_selective_error_curve.csv"),
        ),
        (
            "fig_revision_1d_baseline_comparison",
            "Figure X D",
            "multi-model true-error baseline comparison",
            plot_baseline_comparison,
            (multimodel_dir,),
            str(multimodel_dir / "tables" / "known_composition_multimodel_score_summary.csv"),
        ),
        (
            "fig_revision_s0_dlpfc_contract_fairness",
            "Figure S0",
            "DLPFC anchored common-feature control",
            plot_dlpfc_contract_fairness,
            (baseline_dir,),
            str(baseline_dir / "tables" / "dlpfc_contract_fairness_aggregate.csv"),
        ),
        (
            "fig_revision_s1_component_ablation",
            "Figure SX",
            "component ablation",
            plot_component_ablation,
            (ablation_dir,),
            str(ablation_dir / "tables" / "component_ablation_summary.csv"),
        ),
        (
            "fig_revision_s2_threshold_sensitivity",
            "Figure SY",
            "threshold sensitivity",
            plot_threshold_sensitivity,
            (threshold_dir,),
            str(threshold_dir / "tables" / "threshold_keep_fraction_summary.csv"),
        ),
        (
            "fig_revision_s3_high_risk_capture",
            "Figure SY",
            "high-risk capture and precision",
            plot_high_risk_capture,
            (threshold_dir,),
            str(threshold_dir / "tables" / "threshold_high_risk_summary.csv"),
        ),
        (
            "fig_revision_s4_reference_perturbation",
            "Figure SZ",
            "reference perturbation stress test",
            plot_reference_perturbation,
            (perturbation_dir,),
            str(perturbation_dir / "tables" / "reference_perturbation_summary.csv"),
        ),
        (
            "fig_revision_s5_reference_component_delta",
            "Figure SZ",
            "reference component scenario dependence",
            plot_reference_component_delta,
            (perturbation_component_dir,),
            str(perturbation_component_dir / "tables" / "reference_perturbation_full_vs_no_reference.csv"),
        ),
    ]

    index_rows = []
    for stem, slot, description, func, extra_args, source_table in plot_specs:
        pdf_path, png_path = func(*extra_args, run_dir)
        index_rows.append(
            {
                "figure_id": stem,
                "slot": slot,
                "description": description,
                "pdf_path": str(pdf_path),
                "png_preview_path": str(png_path),
                "source_table": source_table,
                "status": "generated",
            }
        )

    source_dirs = {
        "known_composition": known_dir,
        "uncertainty_baseline": baseline_dir,
        "known_composition_multimodel": multimodel_dir,
        "component_ablation": ablation_dir,
        "threshold_sensitivity": threshold_dir,
        "reference_perturbation": perturbation_dir,
        "reference_perturbation_component": perturbation_component_dir,
    }
    copy_source_tables(run_dir, source_dirs)
    write_figure_index(run_dir, index_rows)
    contact_sheet = build_contact_sheet(run_dir, index_rows)

    metadata = {
        "run_id": args.run_id,
        "known_composition_run_dir": str(known_dir),
        "baseline_run_dir": str(baseline_dir),
        "multimodel_run_dir": str(multimodel_dir),
        "ablation_run_dir": str(ablation_dir),
        "threshold_run_dir": str(threshold_dir),
        "reference_perturbation_run_dir": str(perturbation_dir),
        "reference_perturbation_component_run_dir": str(perturbation_component_dir),
        "n_figures": len(index_rows),
        "contact_sheet": str(contact_sheet),
        "figure_format": "PDF vector with PNG previews",
        "font_family": PUBLICATION_FONT_FAMILY,
        "claim_boundary": (
            "These are single-panel revision figures. They are not automatically assembled into a composite figure; "
            "final manuscript placement should be decided during iScience revision writing."
        ),
    }
    results_file(run_dir, "metadata", "revision_figures.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme_lines = [
        "# Revision Figures",
        "",
        "This run contains single-panel PDF figures for the iScience revision analyses.",
        "",
        "## Figures",
        "",
    ]
    for row in index_rows:
        readme_lines.append(f"- `{Path(row['pdf_path']).name}`: {row['description']} ({row['slot']})")
    readme_lines.extend(
        [
            "",
            "## Preview",
            "",
            f"- `figures/{contact_sheet.name}`",
            "",
            "## Boundary",
            "",
            "These panels support the revision evidence package. They should be assembled into main or supplementary figures only after manuscript-level layout review.",
        ]
    )
    (run_dir / "revision_figure_readme.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"Wrote revision figures to {run_dir}")
    print(json.dumps({"run_id": args.run_id, "n_figures": len(index_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
