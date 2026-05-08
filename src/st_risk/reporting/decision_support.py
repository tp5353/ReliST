from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from st_risk.reporting.anchored_validation import PROXY_SHORT_NAMES
from st_risk.reporting.gallery import dominant_celltype_table, repo_relative_link


ABSTAIN_REVIEW = "review needed"
ABSTAIN_CAUTION = "use with caution"
ABSTAIN_TRUSTED = "trusted"
SELECTION_GOALS = ("dominant", "structure", "reference", "marker", "consensus")


@dataclass(frozen=True)
class DecisionSupportSummary:
    sample_id: str
    primary_model: str
    selection_goal: str
    high_risk_quantile: float
    risk_threshold: float
    disagreement_threshold: float
    n_spots: int
    trusted_fraction: float
    caution_fraction: float
    review_fraction: float
    consensus_support_fraction: float
    mean_consensus_fraction: float


def cross_model_dominant_disagreement(
    dominant_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for model_key, table in dominant_tables.items():
        rows.append(table["dominant_celltype"].rename(model_key))
    combined = pd.concat(rows, axis=1, join="inner")
    model_cols = list(combined.columns)
    consensus_label: list[str] = []
    consensus_fraction: list[float] = []
    n_unique: list[int] = []
    for row in combined.itertuples(index=False):
        values = pd.Series(list(row), dtype="object")
        counts = values.value_counts()
        top_count = int(counts.iloc[0])
        top_labels = sorted(counts.loc[counts == top_count].index.astype(str).tolist())
        consensus_label.append(top_labels[0])
        consensus_fraction.append(float(top_count / len(model_cols)))
        n_unique.append(int(counts.shape[0]))
    result = combined.copy()
    result["consensus_label"] = consensus_label
    result["consensus_fraction"] = consensus_fraction
    result["n_unique_labels"] = n_unique
    return result


def build_deployment_table(
    primary_table: pd.DataFrame,
    risk_table: pd.DataFrame,
    disagreement_table: pd.DataFrame,
    *,
    sample_id: str,
    high_risk_quantile: float = 0.8,
    high_disagreement_threshold: float = 0.6,
) -> pd.DataFrame:
    sample_risk = risk_table.loc[risk_table["sample_id"].astype(str) == str(sample_id), ["risk_score"]].copy()
    disagreement_cols = [
        col
        for col in ("consensus_label", "consensus_fraction", "n_unique_labels", "is_consensus_resolved")
        if col in disagreement_table.columns
    ]
    table = primary_table.join(sample_risk, how="left").join(
        disagreement_table.loc[:, disagreement_cols],
        how="left",
    )
    risk_threshold = float(table["risk_score"].quantile(high_risk_quantile))
    table["risk_rank_fraction"] = table["risk_score"].rank(method="average", pct=True)
    table["is_high_risk"] = table["risk_score"] >= risk_threshold
    table["is_high_disagreement"] = table["consensus_fraction"] < high_disagreement_threshold
    if "is_consensus_resolved" in table.columns:
        table["consensus_support"] = np.where(
            table["dominant_celltype"].astype(str) == "uncertain",
            table["is_consensus_resolved"].fillna(False),
            table["dominant_celltype"].astype(str) == table["consensus_label"].astype(str),
        )
    else:
        table["consensus_support"] = table["dominant_celltype"].astype(str) == table["consensus_label"].astype(str)
    table["abstention_status"] = np.where(
        table["is_high_risk"] & table["is_high_disagreement"],
        ABSTAIN_REVIEW,
        np.where(table["is_high_risk"], ABSTAIN_CAUTION, ABSTAIN_TRUSTED),
    )
    table["trusted_keep"] = table["abstention_status"] == ABSTAIN_TRUSTED
    table["sample_id"] = str(sample_id)
    table.attrs["high_risk_quantile"] = float(high_risk_quantile)
    table.attrs["risk_threshold"] = risk_threshold
    table.attrs["high_disagreement_threshold"] = float(high_disagreement_threshold)
    return table


def build_decision_support_summary(
    deployment_table: pd.DataFrame,
    *,
    primary_model: str,
    selection_goal: str,
) -> pd.DataFrame:
    n_spots = max(int(deployment_table.shape[0]), 1)
    counts = deployment_table["abstention_status"].value_counts(normalize=True)
    row = {
        "sample_id": str(deployment_table["sample_id"].iloc[0]),
        "primary_model": primary_model,
        "selection_goal": selection_goal,
        "high_risk_quantile": float(deployment_table.attrs["high_risk_quantile"]),
        "risk_threshold": float(deployment_table.attrs["risk_threshold"]),
        "disagreement_threshold": float(deployment_table.attrs["high_disagreement_threshold"]),
        "n_spots": n_spots,
        "trusted_fraction": float(counts.get(ABSTAIN_TRUSTED, 0.0)),
        "caution_fraction": float(counts.get(ABSTAIN_CAUTION, 0.0)),
        "review_fraction": float(counts.get(ABSTAIN_REVIEW, 0.0)),
        "consensus_support_fraction": float(deployment_table["consensus_support"].mean()),
        "mean_consensus_fraction": float(deployment_table["consensus_fraction"].mean()),
    }
    return pd.DataFrame([row])


def select_primary_model(model_summary: pd.DataFrame, *, goal: str) -> str:
    if goal not in SELECTION_GOALS:
        raise ValueError(f"Unsupported selection goal: {goal}")
    if goal == "consensus":
        return "consensus"

    ranked = model_summary.copy()
    if goal == "dominant":
        column = "mean_top1_margin"
        ranked = ranked.dropna(subset=[column]).sort_values(column, ascending=False)
    elif goal == "structure":
        cols = [col for col in ("layer_guess_mean_best_auc", "Maynard_mean_best_auc") if col in ranked.columns]
        ranked = ranked.dropna(subset=cols, how="all").copy()
        ranked["goal_score"] = ranked[cols].mean(axis=1)
        ranked = ranked.sort_values("goal_score", ascending=False)
    elif goal == "reference":
        if "signature_residual_mean_best_auc" in ranked.columns and ranked["signature_residual_mean_best_auc"].notna().any():
            ranked = ranked.dropna(subset=["signature_residual_mean_best_auc"]).sort_values(
                "signature_residual_mean_best_auc", ascending=False
            )
        else:
            cols = [col for col in ranked.columns if col.endswith("_mean_best_auc")]
            ranked["goal_score"] = ranked[cols].mean(axis=1)
            ranked = ranked.sort_values("goal_score", ascending=False)
    else:
        column = "marker_discordance_mean_best_auc"
        ranked = ranked.dropna(subset=[column]).sort_values(column, ascending=False)

    if ranked.empty:
        raise ValueError(f"Could not select a primary model for goal={goal}")
    return str(ranked.iloc[0]["model_key"])


def build_consensus_primary_inputs(
    dominant_tables: dict[str, pd.DataFrame],
    risk_tables: dict[str, pd.DataFrame],
    *,
    sample_id: str,
    min_consensus_fraction: float = 0.6,
    unresolved_label: str = "uncertain",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first_model_key, first_table = next(iter(dominant_tables.items()))
    merged = first_table.loc[:, ["x_spatial", "y_spatial", "dominant_celltype"]].rename(
        columns={"dominant_celltype": first_model_key}
    )
    index_lookup = (
        first_table.loc[:, ["x_spatial", "y_spatial"]]
        .reset_index()
        .rename(columns={"index": "spot_id"})
    )
    for model_key, table in list(dominant_tables.items())[1:]:
        coords = table.loc[:, ["x_spatial", "y_spatial", "dominant_celltype"]].rename(
            columns={"dominant_celltype": model_key}
        )
        merged = merged.merge(coords, on=["x_spatial", "y_spatial"], how="inner", validate="one_to_one")

    label_columns = [col for col in merged.columns if col not in ("x_spatial", "y_spatial")]
    consensus_label: list[str] = []
    consensus_fraction: list[float] = []
    n_unique: list[int] = []
    for row in merged[label_columns].itertuples(index=False):
        values = pd.Series(list(row), dtype="object")
        counts = values.value_counts()
        top_count = int(counts.iloc[0])
        top_labels = sorted(counts.loc[counts == top_count].index.astype(str).tolist())
        consensus_label.append(top_labels[0])
        consensus_fraction.append(float(top_count / len(label_columns)))
        n_unique.append(int(counts.shape[0]))

    disagreement = merged.copy()
    disagreement["consensus_label"] = consensus_label
    disagreement["consensus_fraction"] = consensus_fraction
    disagreement["n_unique_labels"] = n_unique
    disagreement["is_consensus_resolved"] = disagreement["consensus_fraction"] >= float(min_consensus_fraction)
    disagreement = disagreement.merge(index_lookup, on=["x_spatial", "y_spatial"], how="left", validate="one_to_one").set_index("spot_id")

    primary = disagreement.loc[:, ["x_spatial", "y_spatial"]].copy()
    primary["dominant_celltype"] = np.where(
        disagreement["is_consensus_resolved"],
        disagreement["consensus_label"].astype(str),
        unresolved_label,
    )

    def _risk_with_coords(model_key: str, risk_table: pd.DataFrame) -> pd.DataFrame:
        sample_risk = risk_table.loc[risk_table["sample_id"].astype(str) == str(sample_id)].copy()
        if {"x_spatial", "y_spatial"}.issubset(sample_risk.columns):
            return sample_risk
        coord_lookup = dominant_tables[model_key].loc[:, ["x_spatial", "y_spatial"]]
        return sample_risk.join(coord_lookup, how="left")

    first_risk_key, first_risk = next(iter(risk_tables.items()))
    first_risk = _risk_with_coords(first_risk_key, first_risk)
    risk_coords = first_risk.loc[:, ["x_spatial", "y_spatial"]].copy()
    risk_coords = risk_coords.reset_index().rename(columns={"index": "spot_id"})
    risk_merged = risk_coords.copy()
    for model_key, risk_table in risk_tables.items():
        sample_risk = _risk_with_coords(model_key, risk_table).loc[:, ["x_spatial", "y_spatial", "risk_score"]].copy()
        sample_risk[f"{model_key}_ranked_risk"] = sample_risk["risk_score"].rank(method="average", pct=True)
        sample_risk = sample_risk.loc[:, ["x_spatial", "y_spatial", f"{model_key}_ranked_risk"]]
        risk_merged = risk_merged.merge(sample_risk, on=["x_spatial", "y_spatial"], how="inner", validate="one_to_one")

    ranked_cols = [col for col in risk_merged.columns if col.endswith("_ranked_risk")]
    consensus_risk = risk_merged.loc[:, ["spot_id"]].copy().set_index("spot_id")
    consensus_risk["sample_id"] = str(sample_id)
    consensus_risk["risk_score"] = risk_merged[ranked_cols].mean(axis=1).to_numpy()
    return primary, consensus_risk, disagreement


def build_decision_validation_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    ratio_cols = [col for col in model_summary.columns if col.endswith("_mean_proxy_ratio_at_0p8")]
    high_ratio_cols = [col for col in model_summary.columns if col.endswith("_mean_high_proxy_ratio_at_0p8")]
    monotone_cols = [col for col in model_summary.columns if col.endswith("_monotone_mean_proxy_rate")]
    gap_cols = [col for col in model_summary.columns if col.endswith("_mean_high_minus_low_proxy")]

    rows: list[dict[str, float | str]] = []
    for row in model_summary.itertuples(index=False):
        ratio_values = pd.Series([getattr(row, col) for col in ratio_cols], dtype="float64").dropna()
        high_ratio_values = pd.Series([getattr(row, col) for col in high_ratio_cols], dtype="float64").dropna()
        monotone_values = pd.Series([getattr(row, col) for col in monotone_cols], dtype="float64").dropna()
        gap_values = pd.Series([getattr(row, col) for col in gap_cols], dtype="float64").dropna()
        rows.append(
            {
                "display_name": row.display_name,
                "mean_retained_proxy_ratio_at_0p8": float(ratio_values.mean()) if not ratio_values.empty else np.nan,
                "mean_removed_high_proxy_fraction_at_0p8": float((1.0 - high_ratio_values).mean()) if not high_ratio_values.empty else np.nan,
                "mean_monotonicity": float(monotone_values.mean()) if not monotone_values.empty else np.nan,
                "mean_high_vs_low_gap": float(gap_values.mean()) if not gap_values.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_retained_proxy_ratio_at_0p8", "mean_removed_high_proxy_fraction_at_0p8"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)


def render_decision_support_markdown(
    *,
    sample_id: str,
    primary_display_name: str,
    selection_goal: str,
    deconvolution_figure: Path,
    risk_figure: Path,
    filtered_figure: Path,
    status_figure: Path,
    disagreement_figure: Path,
    summary_table: pd.DataFrame,
    validation_summary: pd.DataFrame,
    abstention_csv: Path,
    start: Path,
) -> str:
    summary = summary_table.iloc[0]
    deconvolution_link = repo_relative_link(deconvolution_figure, start=start)
    risk_link = repo_relative_link(risk_figure, start=start)
    filtered_link = repo_relative_link(filtered_figure, start=start)
    status_link = repo_relative_link(status_figure, start=start)
    disagreement_link = repo_relative_link(disagreement_figure, start=start)
    abstention_link = repo_relative_link(abstention_csv, start=start)

    lines = [
        f"# Decision Support Bundle（决策支持输出）{sample_id}",
        "",
        f"- 主模型 primary model：`{primary_display_name}`",
        f"- 选择目标 selection goal：`{selection_goal}`",
        f"- 高风险阈值 high-risk quantile：`{float(summary['high_risk_quantile']):.3f}` 对应的 sample risk threshold（样本风险阈值）=`{float(summary['risk_threshold']):.4f}`",
        f"- 高分歧阈值 high-disagreement threshold：`consensus_fraction < {float(summary['disagreement_threshold']):.2f}`",
        "",
        "## Deployment Outputs（部署输出）",
        "",
        f"- `deconvolution map`（原始反卷积结果）：[{deconvolution_figure.name}]({deconvolution_link})",
        f"- `risk map`（风险图）：[{risk_figure.name}]({risk_link})",
        f"- `trusted-only filtered result`（低风险过滤结果）：[{filtered_figure.name}]({filtered_link})",
        f"- `abstention status map`（保留意见状态图）：[{status_figure.name}]({status_link})",
        f"- `disagreement map`（跨模型分歧图，辅助图层）：[{disagreement_figure.name}]({disagreement_link})",
        f"- `abstention table`（逐 spot 状态表）：[{abstention_csv.name}]({abstention_link})",
        "",
        f"![{primary_display_name} deconvolution {sample_id}]({deconvolution_link})",
        "",
        f"![{primary_display_name} risk {sample_id}]({risk_link})",
        "",
        f"![{primary_display_name} trusted-only {sample_id}]({filtered_link})",
        "",
        f"![{primary_display_name} abstention status {sample_id}]({status_link})",
        "",
        f"![{primary_display_name} disagreement {sample_id}]({disagreement_link})",
        "",
        "## Usage Note（使用说明）",
        "",
        f"- `trusted`（可优先使用）比例：`{float(summary['trusted_fraction']):.3f}`。这些 spot（点）默认进入 `trusted-only filtered result`（低风险过滤结果）。",
        f"- `use with caution`（谨慎使用）比例：`{float(summary['caution_fraction']):.3f}`。这些 spot 风险高，但跨模型主标签分歧不算高，更适合作为弱证据而不是主结论。",
        f"- `review needed`（需要复核）比例：`{float(summary['review_fraction']):.3f}`。这些 spot 同时满足高 `risk`（风险）和高 `disagreement`（分歧），最适合优先人工复核或追加验证。",
    ]
    if primary_display_name == "Consensus-first":
        lines.append(
            f"- 达到 `majority consensus`（多数共识，至少 3/5 模型同意）的比例：`{float(summary['consensus_support_fraction']):.3f}`。没有达到这个门槛的 spot 会直接标成 `uncertain`（不确定），而不是强行指定 cell type。"
        )
    else:
        lines.append(
            f"- 主模型与跨模型 `consensus label`（共识标签）一致的比例：`{float(summary['consensus_support_fraction']):.3f}`。这个值越低，越不适合把单模型结果直接当成稳定真相。"
        )
    lines.extend(
        [
            f"- 跨模型平均 `consensus fraction`（共识支持度）=`{float(summary['mean_consensus_fraction']):.3f}`。如果这个值整体偏低，更适合用 `consensus-first`（共识优先）而不是固定单模型。",
            f"- `disagreement map`（跨模型分歧图）只是辅助图层：它帮助识别模型间不一致的位置，但不替代 `risk map`（风险图），也不应被读成简单 `voting`（投票）结论。",
            "",
            "## Abstention Rule（保留意见规则）",
            "",
            "- `high risk + high disagreement => review needed`（高风险且高分歧 => 需要复核）",
            "- `high risk + low disagreement => use with caution`（高风险但低分歧 => 谨慎使用）",
            "- `not high risk => trusted`（非高风险 => 优先使用）",
            "",
            "## Decision-Oriented Validation（面向决策的验证）",
            "",
            "- 这里不再只看 `AUC`（区分能力），而是看：删掉高风险区域后结果是否更干净，以及高风险区域是否更值得优先复核。",
            "",
            "| 模型 Model | 保留后平均锚点误差比例 | 被送去复核的高锚点误差比例 | 平均单调性 | 平均高低风险差值 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in validation_summary.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.mean_retained_proxy_ratio_at_0p8:.3f} | {row.mean_removed_high_proxy_fraction_at_0p8:.3f} | {row.mean_monotonicity:.3f} | {row.mean_high_vs_low_gap:.3f} |"
        )

    lines.extend(
        [
            "",
            "- 读法：保留后平均锚点误差比例越低，说明 `risk filtering`（风险筛除）后剩下的结果越干净；被送去复核的高锚点误差比例越高，说明高风险区域越值得优先检查。",
            "",
        ]
    )
    return "\n".join(lines)


def render_sample_interpretation_markdown(
    *,
    sample_id: str,
    margin_summary: pd.DataFrame,
    dominant_summary: pd.DataFrame,
    abundance_summary: pd.DataFrame,
    start: Path,
) -> str:
    order = margin_summary.sort_values("mean_top1_margin", ascending=False)["display_name"].tolist()
    lines = [
        f"# Sample Interpretation（样本解读）{sample_id}",
        "",
        f"- 阅读顺序：先看 `top1 margin`（第一名减第二名），再看 `dominant celltype`（主标签图），最后看 `abundance heatmaps`（连续丰度热图）。",
        "",
        "## 结论",
        "",
        f"- `certainty shape`（确定性形状）排序仍然是：`{' > '.join(order)}`。",
    ]

    rctd_margin = margin_summary.loc[margin_summary["display_name"] == "RCTD"].iloc[0]
    stereo_margin = margin_summary.loc[margin_summary["display_name"] == "Stereoscope"].iloc[0]
    lines.append(
        f"- `RCTD` 的 mean `top1 margin`（平均第一名优势）=`{float(rctd_margin['mean_top1_margin']):.3f}`，而 `Stereoscope` 只有 `{float(stereo_margin['mean_top1_margin']):.3f}`；这继续支持“`RCTD` 更 hard（更硬），`Stereoscope` 更 soft（更软）”。"
    )

    top_counts = (
        dominant_summary.sort_values(["display_name", "fraction_of_sample"], ascending=[True, False])
        .groupby("display_name", as_index=False)
        .first()
    )
    for row in top_counts.itertuples(index=False):
        lines.append(
            f"- `{row.display_name}` 当前最常见的 `dominant celltype`（主标签）是 `{row.celltype}`，占比 `{float(row.fraction_of_sample):.3f}`。"
        )

    target = abundance_summary.loc[abundance_summary["display_name"].eq("RCTD") & abundance_summary["celltype"].eq("Excit_01")]
    if not target.empty:
        row = target.iloc[0]
        lines.append(
            f"- `RCTD` 在 `Excit_01` 上的 raw `p90`（90 分位原始丰度）=`{float(row['raw_p90']):.3f}`，说明它在这个样本上仍然保留较明显的连续主导区域。"
        )

    lines.extend(
        [
            "",
            "## 当前读法",
            "",
            "- 这个样本现在更适合作为“模型风格差异可复现”的第四个案例，而不是新的主叙事样本。",
            "- 更稳的说法仍然是：`output geometry`（输出形状）和 `certainty shape`（确定性形状）更像 model-specific（模型特异）风格；单个 `dominant winner`（主导赢家）的身份仍然可能受样本影响。",
            "",
        ]
    )
    return "\n".join(lines)
