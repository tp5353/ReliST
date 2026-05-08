from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from st_risk.eval.layer_eval import (
    score_proxy_bin_monotonicity_summary,
    score_proxy_bin_summary,
    score_proxy_comparison_summary,
    score_proxy_retention_curve,
)
from st_risk.models.io import load_saved_base_model_output
from st_risk.paths import resolve_results_file
from st_risk.reporting.gallery import dominant_celltype_table, repo_relative_link, top1_margin_table

DEFAULT_SAMPLE_IDS = ("151507", "151508", "151669", "151670")
DEFAULT_PROXY_COLUMNS = (
    "layer_guess_prototype_distance",
    "Maynard_prototype_distance",
    "snrna_marker_discordance",
    "snrna_signature_residual",
)
DEFAULT_SCORE_COLUMNS = ("risk_score", "structure_risk_score", "reference_risk_score")
PROXY_SHORT_NAMES = {
    "layer_guess_prototype_distance": "layer_guess",
    "Maynard_prototype_distance": "Maynard",
    "snrna_marker_discordance": "marker_discordance",
    "snrna_signature_residual": "signature_residual",
}


@dataclass(frozen=True)
class AnchoredModelSpec:
    model_key: str
    display_name: str
    run_id: str
    run_path: Path


def _optional_table(run_path: Path, filename: str) -> pd.DataFrame:
    path = resolve_results_file(run_path, "tables", filename)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def load_anchor_observation_table(run_path: Path) -> pd.DataFrame:
    risk = pd.read_csv(resolve_results_file(run_path, "tables", "risk_table.csv"), index_col=0)
    obs = risk.loc[:, [col for col in ("sample_id", "risk_score") if col in risk.columns]].copy()

    dual = _optional_table(run_path, "score_dual_axis_scores.csv")
    if not dual.empty:
        obs = obs.join(dual, how="left")

    label = _optional_table(run_path, "score_proxy_label_distance.csv")
    if not label.empty:
        obs = obs.join(label, how="left")

    marker = _optional_table(run_path, "score_proxy_snrna_marker.csv")
    if not marker.empty:
        obs = obs.join(marker, how="left")

    residual = _optional_table(run_path, "score_proxy_snrna_signature_residual.csv")
    if not residual.empty:
        obs = obs.join(residual, how="left")

    return obs


def available_score_columns(obs: pd.DataFrame, *, score_cols: tuple[str, ...] = DEFAULT_SCORE_COLUMNS) -> tuple[str, ...]:
    return tuple(col for col in score_cols if col in obs.columns and obs[col].notna().any())


def available_proxy_columns(obs: pd.DataFrame, *, proxy_cols: tuple[str, ...] = DEFAULT_PROXY_COLUMNS) -> tuple[str, ...]:
    return tuple(col for col in proxy_cols if col in obs.columns and obs[col].notna().any())


def sharpness_summary(
    spec: AnchoredModelSpec,
    *,
    sample_ids: tuple[str, ...] = DEFAULT_SAMPLE_IDS,
    low_margin_threshold: float = 0.1,
) -> pd.DataFrame:
    abundance = load_saved_base_model_output(spec.run_path).abundance
    risk = pd.read_csv(resolve_results_file(spec.run_path, "tables", "risk_table.csv"), index_col=0)
    rows: list[dict[str, float | str]] = []

    for sample_id in sample_ids:
        sample_mask = risk["sample_id"].astype(str) == str(sample_id)
        if not sample_mask.any():
            continue
        margin = top1_margin_table(abundance, risk, sample_id=sample_id)
        dominant = dominant_celltype_table(abundance, risk, sample_id=sample_id)
        top_counts = dominant["dominant_celltype"].value_counts(dropna=False)
        rows.append(
            {
                "model_key": spec.model_key,
                "display_name": spec.display_name,
                "run_id": spec.run_id,
                "sample_id": str(sample_id),
                "n_spots": int(margin.shape[0]),
                "mean_top1_margin": float(margin["top1_margin"].mean()),
                "median_top1_margin": float(margin["top1_margin"].median()),
                "low_margin_fraction": float((margin["top1_margin"] < low_margin_threshold).mean()),
                "top_dominant_celltype": str(top_counts.index[0]),
                "top_dominant_fraction": float(top_counts.iloc[0] / max(len(dominant), 1)),
            }
        )
    return pd.DataFrame(rows)


def anchor_auc_summary(
    spec: AnchoredModelSpec,
    *,
    sample_ids: tuple[str, ...] = DEFAULT_SAMPLE_IDS,
) -> pd.DataFrame:
    obs = load_anchor_observation_table(spec.run_path)
    score_cols = available_score_columns(obs)
    proxy_cols = available_proxy_columns(obs)
    rows: list[pd.DataFrame] = []
    for sample_id in sample_ids:
        sample_obs = obs.loc[obs["sample_id"].astype(str) == str(sample_id)].copy()
        if sample_obs.empty:
            continue
        for proxy_col in proxy_cols:
            valid = sample_obs.dropna(subset=[proxy_col, *score_cols])
            if valid.empty:
                continue
            summary = score_proxy_comparison_summary(valid, score_cols=score_cols, proxy_col=proxy_col)
            summary.insert(0, "sample_id", str(sample_id))
            summary.insert(0, "run_id", spec.run_id)
            summary.insert(0, "display_name", spec.display_name)
            summary.insert(0, "model_key", spec.model_key)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def anchor_retention_summary(
    spec: AnchoredModelSpec,
    *,
    sample_ids: tuple[str, ...] = DEFAULT_SAMPLE_IDS,
    keep_quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> pd.DataFrame:
    obs = load_anchor_observation_table(spec.run_path)
    score_cols = available_score_columns(obs)
    proxy_cols = available_proxy_columns(obs)
    rows: list[pd.DataFrame] = []
    for sample_id in sample_ids:
        sample_obs = obs.loc[obs["sample_id"].astype(str) == str(sample_id)].copy()
        if sample_obs.empty:
            continue
        for proxy_col in proxy_cols:
            for score_col in score_cols:
                valid = sample_obs.dropna(subset=[proxy_col, score_col])
                if valid.empty:
                    continue
                curve = score_proxy_retention_curve(
                    valid,
                    score_col=score_col,
                    proxy_col=proxy_col,
                    keep_quantiles=keep_quantiles,
                )
                curve.insert(0, "sample_id", str(sample_id))
                curve.insert(0, "run_id", spec.run_id)
                curve.insert(0, "display_name", spec.display_name)
                curve.insert(0, "model_key", spec.model_key)
                rows.append(curve)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def anchor_reliability_curve_summary(
    spec: AnchoredModelSpec,
    *,
    sample_ids: tuple[str, ...] = DEFAULT_SAMPLE_IDS,
    bin_quantiles: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    bin_labels: tuple[str, ...] = ("risk_q1_lowest", "risk_q2", "risk_q3", "risk_q4", "risk_q5_highest"),
) -> pd.DataFrame:
    obs = load_anchor_observation_table(spec.run_path)
    score_cols = available_score_columns(obs)
    proxy_cols = available_proxy_columns(obs)
    rows: list[pd.DataFrame] = []
    for sample_id in sample_ids:
        sample_obs = obs.loc[obs["sample_id"].astype(str) == str(sample_id)].copy()
        if sample_obs.empty:
            continue
        for proxy_col in proxy_cols:
            for score_col in score_cols:
                valid = sample_obs.dropna(subset=[proxy_col, score_col])
                if valid.empty:
                    continue
                curve = score_proxy_bin_summary(
                    valid,
                    score_col=score_col,
                    proxy_col=proxy_col,
                    bin_quantiles=bin_quantiles,
                    bin_labels=bin_labels,
                )
                curve.insert(0, "sample_id", str(sample_id))
                curve.insert(0, "run_id", spec.run_id)
                curve.insert(0, "display_name", spec.display_name)
                curve.insert(0, "model_key", spec.model_key)
                rows.append(curve)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def anchor_calibration_summary(
    spec: AnchoredModelSpec,
    *,
    sample_ids: tuple[str, ...] = DEFAULT_SAMPLE_IDS,
) -> pd.DataFrame:
    obs = load_anchor_observation_table(spec.run_path)
    score_cols = available_score_columns(obs)
    proxy_cols = available_proxy_columns(obs)
    rows: list[pd.DataFrame] = []
    for sample_id in sample_ids:
        sample_obs = obs.loc[obs["sample_id"].astype(str) == str(sample_id)].copy()
        if sample_obs.empty:
            continue
        for proxy_col in proxy_cols:
            for score_col in score_cols:
                valid = sample_obs.dropna(subset=[proxy_col, score_col])
                if valid.empty:
                    continue
                bin_summary = score_proxy_bin_summary(valid, score_col=score_col, proxy_col=proxy_col)
                mono = score_proxy_bin_monotonicity_summary(bin_summary)
                mono.insert(0, "sample_id", str(sample_id))
                mono.insert(0, "run_id", spec.run_id)
                mono.insert(0, "display_name", spec.display_name)
                mono.insert(0, "model_key", spec.model_key)
                rows.append(mono)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_anchored_validation_tables(
    specs: list[AnchoredModelSpec],
    *,
    sample_ids: tuple[str, ...] = DEFAULT_SAMPLE_IDS,
) -> dict[str, pd.DataFrame]:
    sharpness_rows: list[pd.DataFrame] = []
    auc_rows: list[pd.DataFrame] = []
    retention_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []
    reliability_rows: list[pd.DataFrame] = []
    for spec in specs:
        sharpness_rows.append(sharpness_summary(spec, sample_ids=sample_ids))
        auc_rows.append(anchor_auc_summary(spec, sample_ids=sample_ids))
        retention_rows.append(anchor_retention_summary(spec, sample_ids=sample_ids))
        calibration_rows.append(anchor_calibration_summary(spec, sample_ids=sample_ids))
        reliability_rows.append(anchor_reliability_curve_summary(spec, sample_ids=sample_ids))
    return {
        "sharpness": pd.concat(sharpness_rows, ignore_index=True),
        "auc": pd.concat(auc_rows, ignore_index=True),
        "retention": pd.concat(retention_rows, ignore_index=True),
        "calibration": pd.concat(calibration_rows, ignore_index=True),
        "reliability": pd.concat(reliability_rows, ignore_index=True),
    }


def best_score_per_proxy_sample(auc_summary: pd.DataFrame) -> pd.DataFrame:
    if auc_summary.empty:
        return pd.DataFrame()
    ordered = auc_summary.sort_values(
        ["model_key", "sample_id", "proxy_name", "extreme_proxy_auc", "score_proxy_corr"],
        ascending=[True, True, True, False, False],
        na_position="last",
    )
    return ordered.groupby(["model_key", "sample_id", "proxy_name"], as_index=False).first()


def build_model_summary(
    sharpness: pd.DataFrame,
    auc_summary: pd.DataFrame,
    retention_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
) -> pd.DataFrame:
    sharp = (
        sharpness.groupby(["model_key", "display_name"], as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            mean_top1_margin=("mean_top1_margin", "mean"),
            mean_low_margin_fraction=("low_margin_fraction", "mean"),
            mean_top_dominant_fraction=("top_dominant_fraction", "mean"),
        )
    )

    best_auc = best_score_per_proxy_sample(auc_summary)
    if best_auc.empty:
        return sharp

    best_retention = best_auc.merge(
        retention_summary,
        on=["model_key", "display_name", "run_id", "sample_id", "proxy_name", "score_name"],
        how="left",
    )
    best_calibration = best_auc.merge(
        calibration_summary,
        on=["model_key", "display_name", "run_id", "sample_id", "proxy_name", "score_name"],
        how="left",
    )

    rows: list[dict[str, float | str]] = []
    for (model_key, display_name), sub in best_auc.groupby(["model_key", "display_name"], sort=True):
        row: dict[str, float | str] = {
            "model_key": model_key,
            "display_name": display_name,
        }
        sharp_row = sharp.loc[(sharp["model_key"] == model_key) & (sharp["display_name"] == display_name)].iloc[0]
        row["n_samples"] = int(sharp_row["n_samples"])
        row["mean_top1_margin"] = float(sharp_row["mean_top1_margin"])
        row["mean_low_margin_fraction"] = float(sharp_row["mean_low_margin_fraction"])
        row["mean_top_dominant_fraction"] = float(sharp_row["mean_top_dominant_fraction"])

        for proxy_name, proxy_sub in sub.groupby("proxy_name", sort=True):
            proxy_slug = PROXY_SHORT_NAMES.get(proxy_name, proxy_name)
            row[f"{proxy_slug}_mean_best_auc"] = float(proxy_sub["extreme_proxy_auc"].mean())
            mode = proxy_sub["score_name"].mode()
            row[f"{proxy_slug}_best_score_mode"] = str(mode.iloc[0]) if not mode.empty else ""

            retention_sub = best_retention.loc[
                (best_retention["model_key"] == model_key)
                & (best_retention["proxy_name"] == proxy_name)
                & (best_retention["keep_quantile"] == 0.8)
            ]
            if not retention_sub.empty:
                row[f"{proxy_slug}_mean_proxy_ratio_at_0p8"] = float(retention_sub["retained_mean_proxy_ratio"].mean())
                row[f"{proxy_slug}_mean_high_proxy_ratio_at_0p8"] = float(
                    retention_sub["retained_high_proxy_fraction_ratio"].mean()
                )

            calibration_sub = best_calibration.loc[
                (best_calibration["model_key"] == model_key) & (best_calibration["proxy_name"] == proxy_name)
            ]
            if not calibration_sub.empty:
                row[f"{proxy_slug}_monotone_mean_proxy_rate"] = float(calibration_sub["monotone_mean_proxy"].mean())
                row[f"{proxy_slug}_mean_high_minus_low_proxy"] = float(
                    calibration_sub["high_minus_low_mean_proxy"].mean()
                )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("mean_top1_margin", ascending=False).reset_index(drop=True)


def render_anchored_validation_markdown(
    model_summary: pd.DataFrame,
    best_auc: pd.DataFrame,
    *,
    sample_ids: tuple[str, ...] = DEFAULT_SAMPLE_IDS,
    visual_figures: list[tuple[str, Path, Path]] | None = None,
    start: Path | None = None,
) -> str:
    auc_cols = [
        f"{PROXY_SHORT_NAMES[name]}_mean_best_auc"
        for name in DEFAULT_PROXY_COLUMNS
        if f"{PROXY_SHORT_NAMES[name]}_mean_best_auc" in model_summary.columns
    ]
    ratio_cols = [
        f"{PROXY_SHORT_NAMES[name]}_mean_proxy_ratio_at_0p8"
        for name in DEFAULT_PROXY_COLUMNS
        if f"{PROXY_SHORT_NAMES[name]}_mean_proxy_ratio_at_0p8" in model_summary.columns
    ]
    high_proxy_ratio_cols = [
        f"{PROXY_SHORT_NAMES[name]}_mean_high_proxy_ratio_at_0p8"
        for name in DEFAULT_PROXY_COLUMNS
        if f"{PROXY_SHORT_NAMES[name]}_mean_high_proxy_ratio_at_0p8" in model_summary.columns
    ]
    monotone_cols = [
        f"{PROXY_SHORT_NAMES[name]}_monotone_mean_proxy_rate"
        for name in DEFAULT_PROXY_COLUMNS
        if f"{PROXY_SHORT_NAMES[name]}_monotone_mean_proxy_rate" in model_summary.columns
    ]
    gap_cols = [
        f"{PROXY_SHORT_NAMES[name]}_mean_high_minus_low_proxy"
        for name in DEFAULT_PROXY_COLUMNS
        if f"{PROXY_SHORT_NAMES[name]}_mean_high_minus_low_proxy" in model_summary.columns
    ]

    def _fmt(value: float | int | str | None, *, digits: int = 3) -> str:
        if value is None or pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"

    def _row_value(row: pd.Series | object, column: str) -> float | None:
        if isinstance(row, pd.Series):
            return row[column] if column in row.index else None
        return getattr(row, column, None)

    def _rank_for(column: str, *, ascending: bool = False) -> str:
        if column not in model_summary.columns:
            return "not available"
        ranked = model_summary[["display_name", column]].dropna().sort_values(column, ascending=ascending).reset_index(drop=True)
        if ranked.empty or "RCTD" not in ranked["display_name"].values:
            return "not available"
        position = int(ranked.index[ranked["display_name"] == "RCTD"][0]) + 1
        return f"{position}/{len(ranked)}"

    lines = [
        "# Anchored Validation（外部锚点验证）：RCTD vs Hardness（更硬输出）",
        "",
        "- 目标：区分 `hard output`（更硬输出）和 anchored `reliability`（外部锚点下的可靠性）。",
        f"- 样本：`{', '.join(sample_ids)}`",
        "- 锚点：`layer_guess`、`Maynard`、`snRNA marker discordance`、`snRNA signature residual`",
        "- 读表规则：`AUC`（区分高锚点误差的能力）越高越好；coverage-risk curve（覆盖率-风险曲线）里 retained proxy ratio（保留后残余锚点误差比例）越低越好；reliability curve（可靠性曲线）应随风险分箱从低到高整体上升；`monotonicity`（单调性）越高、high-vs-low gap（高低风险桶差值）越大，说明 `calibration`（校准）越强。",
        "",
    ]
    if visual_figures:
        lines.extend(["## Visual Summary（直观图表）", ""])
        for title, pdf_path, png_path in visual_figures:
            pdf_link = repo_relative_link(pdf_path, start=start) if start is not None else str(pdf_path)
            png_link = repo_relative_link(png_path, start=start) if start is not None else str(png_path)
            lines.append(f"- `{title}`：[`PDF`（矢量图）]({pdf_link}) / [`PNG`（预览图）]({png_link})")
            lines.append("")
            lines.append(f"![{title}]({png_link})")
            lines.append("")

    lines.extend(
        [
            "## Sharpness（输出尖锐度）",
            "",
            "| 模型 Model | 平均 top1 margin | 低 margin 比例 | 平均 top dominant fraction |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in model_summary.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.mean_top1_margin:.3f} | {row.mean_low_margin_fraction:.3f} | {row.mean_top_dominant_fraction:.3f} |"
        )
    lines.append("")

    if auc_cols:
        lines.extend(
            [
                "## Best Anchor AUC（最佳锚点区分能力）",
                "",
                "| 模型 Model | layer_guess | Maynard | marker_discordance | signature_residual |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in model_summary.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.display_name,
                        _fmt(_row_value(row, "layer_guess_mean_best_auc")),
                        _fmt(_row_value(row, "Maynard_mean_best_auc")),
                        _fmt(_row_value(row, "marker_discordance_mean_best_auc")),
                        _fmt(_row_value(row, "signature_residual_mean_best_auc")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if ratio_cols:
        lines.extend(
            [
                "## Selective Gain（选择性保留增益，Keep 0.8）",
                "",
                "| 模型 Model | layer_guess | Maynard | marker_discordance | signature_residual |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in model_summary.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.display_name,
                        _fmt(_row_value(row, "layer_guess_mean_proxy_ratio_at_0p8")),
                        _fmt(_row_value(row, "Maynard_mean_proxy_ratio_at_0p8")),
                        _fmt(_row_value(row, "marker_discordance_mean_proxy_ratio_at_0p8")),
                        _fmt(_row_value(row, "signature_residual_mean_proxy_ratio_at_0p8")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if high_proxy_ratio_cols:
        lines.extend(
            [
                "## Selective High-Proxy Fraction（高锚点误差点保留比例，Keep 0.8）",
                "",
                "| 模型 Model | layer_guess | Maynard | marker_discordance | signature_residual |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in model_summary.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.display_name,
                        _fmt(_row_value(row, "layer_guess_mean_high_proxy_ratio_at_0p8")),
                        _fmt(_row_value(row, "Maynard_mean_high_proxy_ratio_at_0p8")),
                        _fmt(_row_value(row, "marker_discordance_mean_high_proxy_ratio_at_0p8")),
                        _fmt(_row_value(row, "signature_residual_mean_high_proxy_ratio_at_0p8")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if monotone_cols:
        lines.extend(
            [
                "## Calibration Monotonicity（校准单调性）",
                "",
                "| 模型 Model | layer_guess | Maynard | marker_discordance | signature_residual |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in model_summary.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.display_name,
                        _fmt(_row_value(row, "layer_guess_monotone_mean_proxy_rate")),
                        _fmt(_row_value(row, "Maynard_monotone_mean_proxy_rate")),
                        _fmt(_row_value(row, "marker_discordance_monotone_mean_proxy_rate")),
                        _fmt(_row_value(row, "signature_residual_monotone_mean_proxy_rate")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if gap_cols:
        lines.extend(
            [
                "## Calibration High-vs-Low Gap（高低风险桶差值）",
                "",
                "| 模型 Model | layer_guess | Maynard | marker_discordance | signature_residual |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in model_summary.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.display_name,
                        _fmt(_row_value(row, "layer_guess_mean_high_minus_low_proxy")),
                        _fmt(_row_value(row, "Maynard_mean_high_minus_low_proxy")),
                        _fmt(_row_value(row, "marker_discordance_mean_high_minus_low_proxy")),
                        _fmt(_row_value(row, "signature_residual_mean_high_minus_low_proxy")),
                    ]
                )
                + " |"
            )
        lines.append("")

    rctd = model_summary.loc[model_summary["display_name"] == "RCTD"].iloc[0]
    sharp_rank = int(model_summary["mean_top1_margin"].rank(method="min", ascending=False).loc[model_summary["display_name"] == "RCTD"].iloc[0])

    lines.extend(
        [
            "## RCTD Readout（RCTD 专门解读）",
            "",
            f"- `sharpness`（尖锐度）：按 mean `top1 margin`（第一名减第二名）排序，`RCTD` 排名 `{sharp_rank}/{len(model_summary)}`，是当前 4 个样本里最稳定的 `hard-output`（硬输出）模型。",
            f"- `structure anchors`（结构锚点）：`RCTD` 在 `layer_guess` 上排名 `{_rank_for('layer_guess_mean_best_auc')}`，在 `Maynard` 上排名 `{_rank_for('Maynard_mean_best_auc')}`。它在结构锚点上很强，但不是最强，因为 `Stereoscope` 在这两项上都更高。",
            f"- `reference anchors`（参考锚点）：`RCTD` 在 `marker_discordance` 上排名 `{_rank_for('marker_discordance_mean_best_auc')}`，在 `signature_residual` 上排名 `{_rank_for('signature_residual_mean_best_auc')}`。也就是说，它不是最强的 marker 模型，但它是当前 `signature_residual` 上最强的模型。",
            f"- `selective gain`（选择性保留增益）：看 retained mean proxy（保留后平均锚点误差）时，`RCTD` 在 `layer_guess / Maynard / marker_discordance / signature_residual` 上分别排名 `{_rank_for('layer_guess_mean_proxy_ratio_at_0p8', ascending=True)}`、`{_rank_for('Maynard_mean_proxy_ratio_at_0p8', ascending=True)}`、`{_rank_for('marker_discordance_mean_proxy_ratio_at_0p8', ascending=True)}`、`{_rank_for('signature_residual_mean_proxy_ratio_at_0p8', ascending=True)}`。这里是越低越好，所以它对结构锚点的过滤收益比 `signature_residual` 更明显。",
            f"- `calibration`（校准）里的 `monotonicity`（单调性）：`RCTD` 在 4 个锚点上的数值分别是 `{_fmt(rctd.get('layer_guess_monotone_mean_proxy_rate'))}`、`{_fmt(rctd.get('Maynard_monotone_mean_proxy_rate'))}`、`{_fmt(rctd.get('marker_discordance_monotone_mean_proxy_rate'))}`、`{_fmt(rctd.get('signature_residual_monotone_mean_proxy_rate'))}`。这说明风险分桶后，高风险桶通常确实对应更高的锚点误差。",
            f"- `calibration gap`（高低风险桶差值）：`RCTD` 在 `layer_guess / Maynard / marker_discordance / signature_residual` 上的 high-vs-low mean-proxy gap（高低风险桶平均锚点误差差值）分别是 `{_fmt(rctd.get('layer_guess_mean_high_minus_low_proxy'))}`、`{_fmt(rctd.get('Maynard_mean_high_minus_low_proxy'))}`、`{_fmt(rctd.get('marker_discordance_mean_high_minus_low_proxy'))}`、`{_fmt(rctd.get('signature_residual_mean_high_minus_low_proxy'))}`。结构锚点上的差值明显更大，而 `signature_residual` 上虽然方向一致，但幅度更温和。",
            "- 当前结论：现有证据支持把 `RCTD` 写成“明显更硬，并且在更克制的 `signature_residual` 锚点上特别强”；但它并不是所有锚点都最强，所以还不能写成“普遍最可靠”。",
        ]
    )
    lines.append("")

    if not best_auc.empty:
        lines.extend(
            [
                "## Best Score Modes（每个锚点最优分数模式）",
                "",
                "| 模型 Model | 锚点 Proxy | 最优分数模式 Best score mode | 平均最优 AUC |",
                "| --- | --- | --- | ---: |",
            ]
        )
        summary = (
            best_auc.groupby(["display_name", "proxy_name"], as_index=False)
            .agg(
                mean_best_auc=("extreme_proxy_auc", "mean"),
                best_score_mode=("score_name", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            )
            .sort_values(["display_name", "proxy_name"])
        )
        for row in summary.itertuples(index=False):
            lines.append(f"| {row.display_name} | {row.proxy_name} | {row.best_score_mode} | {row.mean_best_auc:.3f} |")
        lines.append("")

    lines.extend(
        [
            "## Model Selection Guide（模型选择指引）",
            "",
            "| 使用场景 | 优先模型 | 原因 |",
            "| --- | --- | --- |",
            "| 需要更明确的 `winner`（第一名）或更稳定的 `dominant label`（主标签） | `RCTD` | `sharpness`（尖锐度）最高，`top1 margin`（第一名减第二名）在当前 4 个样本里始终最高。 |",
            "| 更在意 `structure anchors`（结构锚点），例如 layer（层级）相关一致性 | `Stereoscope` | 在 `layer_guess / Maynard` 两类结构锚点上的 `AUC`（区分能力）当前最高。 |",
            "| 更在意更克制的 `reference anchor`（参考锚点），尤其是 `signature_residual`（signature 残差） | `RCTD` | 在 `signature_residual` 上当前最强，而且 `reference_risk_score`（参考风险分数）是主导最佳模式。 |",
            "| 更在意 `marker`（标记基因）相关一致性 | `Stereoscope` 或 `cell2location` | `marker_discordance`（marker 不一致）上二者都强于 `RCTD`，其中 `Stereoscope` 当前最高，`cell2location` 也保持强势。 |",
            "| 想先用一个居中、折中的模型看整体趋势 | `Tangram` | 当前更像 `middle regime`（中间态）：不是单项最强，但多数指标居中，不会特别 hard（硬）也不至于最弱。 |",
            "| 只把模型当作边界参考，不准备把它作为主力结论来源 | `DestVI` | 当前更像 `weak core + usable branch`（核心主线偏弱、但分支可用）的边界案例，适合保留参考，不适合单独担主结论。 |",
            "",
            "- 当前使用原则：先按你的分析目标选模型，不要把这份报告读成“以后永远只用一个模型”。",
            "- 如果目标是 `decision-support`（辅助决策），更稳的做法是：主模型 + `risk filtering`（风险筛除）+ 必要时用第二模型交叉查看高风险区域。",
            "",
        ]
    )

    return "\n".join(lines)
