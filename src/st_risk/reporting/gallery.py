from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GalleryEntry:
    model_key: str
    display_name: str
    sample_id: str
    run_id: str
    figure_path: Path
    rationale: str
    figure_kind: str = "discordant_sample_spatial"
    source_figure_path: Path | None = None


def repo_relative_link(target: Path, *, start: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), start=Path(start).resolve())).as_posix()


def build_gallery_metadata_rows(entries: list[GalleryEntry], *, start: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in entries:
        row = asdict(entry)
        row["figure_path"] = repo_relative_link(entry.figure_path, start=start)
        if entry.source_figure_path is None:
            row.pop("source_figure_path", None)
        else:
            row["source_figure_path"] = repo_relative_link(entry.source_figure_path, start=start)
        rows.append(row)
    return rows


def render_gallery_markdown(entries: list[GalleryEntry], *, sample_id: str, start: Path) -> str:
    lines = [
        f"# Model Gallery {sample_id}",
        "",
        f"- Sample: `{sample_id}`",
        "- Figure type: `discordant_sample_<sample_id>_spatial.png`",
        "- Purpose: one directly comparable sample-level figure per model",
        "",
    ]

    for entry in entries:
        figure_link = repo_relative_link(entry.figure_path, start=start)
        lines.extend(
            [
                f"## {entry.display_name}",
                "",
                f"- Run: `{entry.run_id}`",
                f"- Why this figure: {entry.rationale}",
                f"- Figure: [{entry.figure_path.name}]({figure_link})",
                "",
                f"![{entry.display_name} sample {sample_id}]({figure_link})",
                "",
            ]
        )
    return "\n".join(lines)


def dominant_celltype_table(abundance: pd.DataFrame, risk_table: pd.DataFrame, *, sample_id: str) -> pd.DataFrame:
    aligned = abundance.loc[risk_table.index]
    sub = risk_table.loc[risk_table["sample_id"].astype(str) == str(sample_id), ["x_spatial", "y_spatial"]].copy()
    sub_abundance = aligned.loc[sub.index]
    dominant = sub_abundance.idxmax(axis=1).astype(str)
    dominant_value = sub_abundance.max(axis=1).astype(float)
    result = sub.copy()
    result["dominant_celltype"] = dominant.values
    result["dominant_value"] = dominant_value.values
    return result


def build_dominant_celltype_summary(
    table: pd.DataFrame,
    *,
    model_key: str,
    display_name: str,
    sample_id: str,
    top_n: int = 5,
) -> pd.DataFrame:
    counts = (
        table["dominant_celltype"]
        .value_counts(dropna=False)
        .rename_axis("celltype")
        .reset_index(name="n_spots")
    )
    total = max(int(counts["n_spots"].sum()), 1)
    counts["fraction_of_sample"] = counts["n_spots"] / total
    counts["model_key"] = model_key
    counts["display_name"] = display_name
    counts["sample_id"] = str(sample_id)
    return counts.loc[:, ["model_key", "display_name", "sample_id", "celltype", "n_spots", "fraction_of_sample"]].head(top_n)


def render_dominant_gallery_markdown(
    entries: list[GalleryEntry],
    summary_table: pd.DataFrame,
    *,
    sample_id: str,
    start: Path,
) -> str:
    lines = [
        f"# Dominant Celltype Gallery {sample_id}",
        "",
        f"- Sample: `{sample_id}`",
        "- Figure type: dominant cell type per spot",
        "- Purpose: compare each model's own abundance-derived spatial prediction at a glance",
        "",
        "## Top Counts",
        "",
        "| Model | Celltype | Spots | Fraction |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in summary_table.itertuples(index=False):
        lines.append(f"| {row.display_name} | {row.celltype} | {int(row.n_spots)} | {row.fraction_of_sample:.3f} |")
    lines.append("")

    for entry in entries:
        figure_link = repo_relative_link(entry.figure_path, start=start)
        model_rows = summary_table.loc[summary_table["model_key"] == entry.model_key].copy()
        top_text = ", ".join(
            f"{r['celltype']} ({r['fraction_of_sample']:.2%})"
            for _, r in model_rows.head(3).iterrows()
        )
        lines.extend(
            [
                f"## {entry.display_name}",
                "",
                f"- Run: `{entry.run_id}`",
                f"- Why this figure: {entry.rationale}",
                f"- Top dominant labels: {top_text}",
                f"- Figure: [{entry.figure_path.name}]({figure_link})",
                "",
                f"![{entry.display_name} dominant celltype {sample_id}]({figure_link})",
                "",
            ]
        )
    return "\n".join(lines)


def top1_margin_table(abundance: pd.DataFrame, risk_table: pd.DataFrame, *, sample_id: str) -> pd.DataFrame:
    aligned = abundance.loc[risk_table.index].astype(float)
    sub = risk_table.loc[risk_table["sample_id"].astype(str) == str(sample_id), ["x_spatial", "y_spatial"]].copy()
    sub_abundance = aligned.loc[sub.index]
    row_sums = sub_abundance.sum(axis=1).replace(0.0, 1.0)
    normalized = sub_abundance.div(row_sums, axis=0)
    values = normalized.to_numpy(dtype=float)
    top1_idx = values.argmax(axis=1)
    top1_val = values[np.arange(values.shape[0]), top1_idx]
    if values.shape[1] > 1:
        top2_val = np.partition(values, -2, axis=1)[:, -2]
    else:
        top2_val = np.zeros(values.shape[0], dtype=float)
    top1_label = normalized.columns.to_numpy()[top1_idx]
    result = sub.copy()
    result["top1_label"] = top1_label
    result["top1_prop"] = top1_val
    result["top2_prop"] = top2_val
    result["top1_margin"] = top1_val - top2_val
    return result


def build_top1_margin_summary(
    table: pd.DataFrame,
    *,
    model_key: str,
    display_name: str,
    sample_id: str,
    low_margin_threshold: float = 0.1,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_key": model_key,
                "display_name": display_name,
                "sample_id": str(sample_id),
                "n_spots": int(table.shape[0]),
                "mean_top1_margin": float(table["top1_margin"].mean()),
                "median_top1_margin": float(table["top1_margin"].median()),
                "p90_top1_margin": float(table["top1_margin"].quantile(0.9)),
                "low_margin_fraction": float((table["top1_margin"] < low_margin_threshold).mean()),
            }
        ]
    )


def render_margin_gallery_markdown(
    entries: list[GalleryEntry],
    summary_table: pd.DataFrame,
    *,
    sample_id: str,
    start: Path,
    title: str = "Top1 Margin Gallery",
    display_value_note: str = "row-normalized `top1 proportion - top2 proportion`",
    purpose: str = "show whether the dominant label is clearly ahead or only barely wins",
) -> str:
    lines = [
        f"# {title} {sample_id}",
        "",
        f"- Sample: `{sample_id}`",
        f"- Margin definition: {display_value_note}",
        f"- Purpose: {purpose}",
        "",
        "## Summary",
        "",
        "| Model | Mean | Median | P90 | Low-margin fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_table.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.mean_top1_margin:.3f} | {row.median_top1_margin:.3f} | {row.p90_top1_margin:.3f} | {row.low_margin_fraction:.3f} |"
        )
    lines.append("")

    for entry in entries:
        figure_link = repo_relative_link(entry.figure_path, start=start)
        row = summary_table.loc[summary_table["model_key"] == entry.model_key].iloc[0]
        lines.extend(
            [
                f"## {entry.display_name}",
                "",
                f"- Run: `{entry.run_id}`",
                f"- Why this figure: {entry.rationale}",
                f"- Mean / median margin: {row['mean_top1_margin']:.3f} / {row['median_top1_margin']:.3f}",
                f"- Low-margin fraction (`< 0.1`): {row['low_margin_fraction']:.3f}",
                f"- Figure: [{entry.figure_path.name}]({figure_link})",
                "",
                f"![{entry.display_name} top1 margin {sample_id}]({figure_link})",
                "",
            ]
        )
    return "\n".join(lines)


def abundance_heatmap_table(
    abundance: pd.DataFrame,
    risk_table: pd.DataFrame,
    *,
    sample_id: str,
    celltype: str,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    aligned = abundance.loc[risk_table.index].astype(float)
    if celltype not in aligned.columns:
        raise KeyError(f"Cell type {celltype} not found in abundance table")
    sub = risk_table.loc[risk_table["sample_id"].astype(str) == str(sample_id), ["x_spatial", "y_spatial"]].copy()
    raw = aligned.loc[sub.index, celltype].astype(float)
    lo = float(raw.quantile(lower_quantile))
    hi = float(raw.quantile(upper_quantile))
    if hi <= lo:
        scaled = pd.Series(0.0, index=raw.index)
    else:
        scaled = raw.clip(lower=lo, upper=hi).sub(lo).div(hi - lo)
    result = sub.copy()
    result["celltype"] = celltype
    result["abundance_raw"] = raw.values
    result["abundance_scaled"] = scaled.values
    result["clip_q01"] = lo
    result["clip_q99"] = hi
    return result


def build_abundance_heatmap_summary(
    table: pd.DataFrame,
    *,
    model_key: str,
    display_name: str,
    sample_id: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_key": model_key,
                "display_name": display_name,
                "sample_id": str(sample_id),
                "celltype": str(table["celltype"].iloc[0]),
                "n_spots": int(table.shape[0]),
                "raw_mean": float(table["abundance_raw"].mean()),
                "raw_median": float(table["abundance_raw"].median()),
                "raw_p90": float(table["abundance_raw"].quantile(0.9)),
                "clip_q01": float(table["clip_q01"].iloc[0]),
                "clip_q99": float(table["clip_q99"].iloc[0]),
            }
        ]
    )


def render_abundance_gallery_markdown(
    entries: list[GalleryEntry],
    summary_table: pd.DataFrame,
    *,
    sample_id: str,
    start: Path,
) -> str:
    lines = [
        f"# Abundance Heatmap Gallery {sample_id}",
        "",
        f"- Sample: `{sample_id}`",
        "- Display value: robust within-model scaling of abundance using `[q01, q99] -> [0, 1]`",
        "- Purpose: compare spatial patterns for representative cell types without winner-take-all collapse",
        "",
        "## Summary",
        "",
        "| Model | Celltype | Raw mean | Raw median | Raw p90 | q01 | q99 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_table.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.celltype} | {row.raw_mean:.4f} | {row.raw_median:.4f} | {row.raw_p90:.4f} | {row.clip_q01:.4f} | {row.clip_q99:.4f} |"
        )
    lines.append("")

    for entry in entries:
        figure_link = repo_relative_link(entry.figure_path, start=start)
        model_key, celltype = entry.model_key.split("::", 1)
        row = summary_table.loc[
            (summary_table["model_key"] == model_key) & (summary_table["celltype"] == celltype)
        ].iloc[0]
        lines.extend(
            [
                f"## {row['display_name']} / {celltype}",
                "",
                f"- Run: `{entry.run_id}`",
                f"- Why this figure: {entry.rationale}",
                f"- Raw mean / p90: {row['raw_mean']:.4f} / {row['raw_p90']:.4f}",
                f"- Display clipping: [{row['clip_q01']:.4f}, {row['clip_q99']:.4f}]",
                f"- Figure: [{entry.figure_path.name}]({figure_link})",
                "",
                f"![{row['display_name']} {celltype} {sample_id}]({figure_link})",
                "",
            ]
        )
    return "\n".join(lines)
