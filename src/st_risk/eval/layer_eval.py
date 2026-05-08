from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _valid_layer_mask(values: pd.Series) -> pd.Series:
    return ~values.astype(str).str.strip().str.lower().isin({"", "nan", "none"})


def high_risk_fraction_by_group(obs: pd.DataFrame, group_col: str, *, quantile: float = 0.9) -> pd.Series:
    threshold = obs["risk_score"].quantile(quantile)
    high_risk = obs["risk_score"] >= threshold
    result = high_risk.groupby(obs[group_col]).mean().sort_index()
    result.name = "high_risk_fraction"
    return result


def boundary_spot_mask(labels: pd.Series, neighbors: np.ndarray) -> np.ndarray:
    labels = labels.reset_index(drop=True)
    neighbors = np.asarray(neighbors, dtype=int)
    mask = np.zeros(len(labels), dtype=bool)
    for i, row in enumerate(neighbors):
        valid = row[row >= 0]
        if len(valid) == 0:
            continue
        mask[i] = any(labels.iloc[j] != labels.iloc[i] for j in valid)
    return mask


def neighbor_label_agreement(labels: pd.Series, neighbors: np.ndarray) -> np.ndarray:
    labels = labels.reset_index(drop=True)
    neighbors = np.asarray(neighbors, dtype=int)
    agreement = np.zeros(len(labels), dtype=float)
    for i, row in enumerate(neighbors):
        valid = row[row >= 0]
        if len(valid) == 0:
            agreement[i] = 1.0
            continue
        agreement[i] = float(np.mean(labels.iloc[valid] == labels.iloc[i]))
    return agreement


def boundary_enrichment_summary(obs: pd.DataFrame, *, quantile: float = 0.9) -> pd.DataFrame:
    threshold = obs["risk_score"].quantile(quantile)
    high_risk = obs["risk_score"] >= threshold
    boundary = obs["boundary_mask"].astype(bool)
    groups = pd.Series(np.where(boundary, "boundary", "non_boundary"), index=obs.index, name="region")
    summary = pd.DataFrame(
        {
            "n_spots": groups.value_counts().sort_index(),
            "high_risk_fraction": high_risk.groupby(groups).mean().sort_index(),
            "mean_risk_score": obs["risk_score"].groupby(groups).mean().sort_index(),
        }
    ).reset_index(names="region")
    overall_high_risk = float(high_risk.mean())
    overall_mean_risk = float(obs["risk_score"].mean())
    summary["high_risk_enrichment"] = summary["high_risk_fraction"] / max(overall_high_risk, 1e-12)
    summary["mean_risk_ratio"] = summary["mean_risk_score"] / max(overall_mean_risk, 1e-12)
    return summary


def selective_retention_summary(obs: pd.DataFrame, *, keep_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    ordered = obs.sort_values("risk_score", ascending=True).reset_index(drop=True)
    total = len(ordered)
    for q in keep_quantiles:
        keep_n = max(1, int(np.ceil(total * q)))
        kept = ordered.iloc[:keep_n]
        rows.append(
            {
                "keep_quantile": q,
                "n_spots": float(len(kept)),
                "mean_risk_score": float(kept["risk_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def selective_layer_coherence_summary(
    obs: pd.DataFrame,
    *,
    keep_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    ordered = obs.sort_values("risk_score", ascending=True).reset_index(drop=True)
    total = len(ordered)
    for q in keep_quantiles:
        keep_n = max(1, int(np.ceil(total * q)))
        kept = ordered.iloc[:keep_n]
        rows.append(
            {
                "keep_quantile": q,
                "n_spots": float(len(kept)),
                "mean_risk_score": float(kept["risk_score"].mean()),
                "mean_neighbor_agreement": float(kept["neighbor_agreement"].mean()),
                "boundary_fraction": float(kept["boundary_mask"].mean()),
            }
        )
    return pd.DataFrame(rows)


def selective_score_coherence_summary(
    obs: pd.DataFrame,
    *,
    score_col: str,
    keep_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    higher_score_higher_risk: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    ordered = obs.sort_values(score_col, ascending=higher_score_higher_risk).reset_index(drop=True)
    total = len(ordered)
    for q in keep_quantiles:
        keep_n = max(1, int(np.ceil(total * q)))
        kept = ordered.iloc[:keep_n]
        row: dict[str, float | str] = {
            "score_name": score_col,
            "keep_quantile": q,
            "n_spots": float(len(kept)),
            "mean_score": float(kept[score_col].mean()),
            "mean_neighbor_agreement": float(kept["neighbor_agreement"].mean()),
        }
        if "boundary_mask" in kept.columns:
            row["boundary_fraction"] = float(kept["boundary_mask"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def risk_stratified_layer_coherence_summary(
    obs: pd.DataFrame,
    *,
    low_quantile: float = 0.1,
    high_quantile: float = 0.9,
) -> pd.DataFrame:
    low_threshold = obs["risk_score"].quantile(low_quantile)
    high_threshold = obs["risk_score"].quantile(high_quantile)
    groups = {
        "low_risk": obs["risk_score"] <= low_threshold,
        "high_risk": obs["risk_score"] >= high_threshold,
    }
    rows: list[dict[str, float | str]] = []
    for name, mask in groups.items():
        subset = obs.loc[mask]
        rows.append(
            {
                "risk_group": name,
                "n_spots": float(len(subset)),
                "mean_risk_score": float(subset["risk_score"].mean()),
                "mean_neighbor_agreement": float(subset["neighbor_agreement"].mean()),
                "boundary_fraction": float(subset["boundary_mask"].mean()),
            }
        )
    return pd.DataFrame(rows)


def score_stratified_coherence_summary(
    obs: pd.DataFrame,
    *,
    score_col: str,
    low_quantile: float = 0.1,
    high_quantile: float = 0.9,
    higher_score_higher_risk: bool = True,
) -> pd.DataFrame:
    low_threshold = obs[score_col].quantile(low_quantile)
    high_threshold = obs[score_col].quantile(high_quantile)
    if higher_score_higher_risk:
        groups = {
            "low_risk": obs[score_col] <= low_threshold,
            "high_risk": obs[score_col] >= high_threshold,
        }
    else:
        groups = {
            "low_risk": obs[score_col] >= high_threshold,
            "high_risk": obs[score_col] <= low_threshold,
        }
    rows: list[dict[str, float | str]] = []
    for name, mask in groups.items():
        subset = obs.loc[mask]
        row: dict[str, float | str] = {
            "score_name": score_col,
            "risk_group": name,
            "n_spots": float(len(subset)),
            "mean_score": float(subset[score_col].mean()),
            "mean_neighbor_agreement": float(subset["neighbor_agreement"].mean()),
        }
        if "boundary_mask" in subset.columns:
            row["boundary_fraction"] = float(subset["boundary_mask"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def score_baseline_comparison_summary(
    obs: pd.DataFrame,
    *,
    score_cols: tuple[str, ...],
    keep_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    low_quantile: float = 0.1,
    high_quantile: float = 0.9,
    higher_score_higher_risk: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    full_neighbor_agreement = float(obs["neighbor_agreement"].mean())
    for score_col in score_cols:
        selective = selective_score_coherence_summary(
            obs,
            score_col=score_col,
            keep_quantiles=keep_quantiles,
            higher_score_higher_risk=higher_score_higher_risk,
        )
        stratified = score_stratified_coherence_summary(
            obs,
            score_col=score_col,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            higher_score_higher_risk=higher_score_higher_risk,
        )
        best_keep = selective.sort_values(
            ["mean_neighbor_agreement", "keep_quantile"],
            ascending=[False, True],
        ).iloc[0]
        low_row = stratified.loc[stratified["risk_group"] == "low_risk"].iloc[0]
        high_row = stratified.loc[stratified["risk_group"] == "high_risk"].iloc[0]
        summary_row: dict[str, float | str] = {
            "score_name": score_col,
            "full_mean_neighbor_agreement": full_neighbor_agreement,
            "best_keep_quantile": float(best_keep["keep_quantile"]),
            "best_keep_mean_neighbor_agreement": float(best_keep["mean_neighbor_agreement"]),
            "best_keep_gain": float(best_keep["mean_neighbor_agreement"] - full_neighbor_agreement),
            "low_risk_mean_neighbor_agreement": float(low_row["mean_neighbor_agreement"]),
            "high_risk_mean_neighbor_agreement": float(high_row["mean_neighbor_agreement"]),
            "low_vs_high_neighbor_agreement_gap": float(
                low_row["mean_neighbor_agreement"] - high_row["mean_neighbor_agreement"]
            ),
        }
        first_keep = selective.sort_values("keep_quantile").iloc[0]
        summary_row["keep_first_quantile"] = float(first_keep["keep_quantile"])
        summary_row["keep_first_mean_neighbor_agreement"] = float(first_keep["mean_neighbor_agreement"])
        summary_row["keep_first_gain"] = float(first_keep["mean_neighbor_agreement"] - full_neighbor_agreement)
        rows.append(summary_row)
    return pd.DataFrame(rows).sort_values(
        ["low_vs_high_neighbor_agreement_gap", "best_keep_gain"],
        ascending=[False, False],
    ).reset_index(drop=True)


def score_proxy_comparison_summary(
    obs: pd.DataFrame,
    *,
    score_cols: tuple[str, ...],
    proxy_col: str,
    high_proxy_quantile: float = 0.9,
    low_proxy_quantile: float = 0.1,
    top_score_quantile: float = 0.9,
    higher_score_higher_risk: bool = True,
) -> pd.DataFrame:
    proxy = obs[proxy_col]
    high_proxy_threshold = proxy.quantile(high_proxy_quantile)
    low_proxy_threshold = proxy.quantile(low_proxy_quantile)
    high_proxy_mask = proxy >= high_proxy_threshold
    low_proxy_mask = proxy <= low_proxy_threshold
    extreme_mask = high_proxy_mask | low_proxy_mask
    rows: list[dict[str, float | str]] = []
    for score_col in score_cols:
        scores = obs[score_col]
        top_score_threshold = scores.quantile(top_score_quantile)
        bottom_score_threshold = scores.quantile(1.0 - top_score_quantile)
        top_score_mask = scores >= top_score_threshold if higher_score_higher_risk else scores <= bottom_score_threshold
        bottom_score_mask = scores <= bottom_score_threshold if higher_score_higher_risk else scores >= top_score_threshold

        auc = np.nan
        if extreme_mask.sum() > 0 and high_proxy_mask[extreme_mask].nunique() > 1:
            auc = float(roc_auc_score(high_proxy_mask[extreme_mask].astype(int), scores[extreme_mask]))

        rows.append(
            {
                "score_name": score_col,
                "proxy_name": proxy_col,
                "score_proxy_corr": float(scores.corr(proxy)),
                "high_proxy_mean_score": float(scores.loc[high_proxy_mask].mean()),
                "low_proxy_mean_score": float(scores.loc[low_proxy_mask].mean()),
                "high_vs_low_score_delta": float(scores.loc[high_proxy_mask].mean() - scores.loc[low_proxy_mask].mean()),
                "top_score_mean_proxy": float(proxy.loc[top_score_mask].mean()),
                "bottom_score_mean_proxy": float(proxy.loc[bottom_score_mask].mean()),
                "top_vs_bottom_proxy_gap": float(proxy.loc[top_score_mask].mean() - proxy.loc[bottom_score_mask].mean()),
                "top_score_high_proxy_fraction": float(high_proxy_mask.loc[top_score_mask].mean()),
                "extreme_proxy_auc": auc,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["extreme_proxy_auc", "top_vs_bottom_proxy_gap", "score_proxy_corr"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def score_proxy_retention_curve(
    obs: pd.DataFrame,
    *,
    score_col: str,
    proxy_col: str,
    keep_quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    high_proxy_quantile: float = 0.9,
    higher_score_higher_risk: bool = True,
) -> pd.DataFrame:
    ordered = obs.sort_values(score_col, ascending=higher_score_higher_risk).reset_index(drop=True)
    proxy = ordered[proxy_col]
    baseline_mean_proxy = float(proxy.mean())
    baseline_high_proxy_threshold = float(proxy.quantile(high_proxy_quantile))
    baseline_high_proxy_mask = proxy >= baseline_high_proxy_threshold
    baseline_high_proxy_fraction = float(baseline_high_proxy_mask.mean())
    total = len(ordered)

    rows: list[dict[str, float | str]] = []
    for q in keep_quantiles:
        keep_n = max(1, int(np.ceil(total * q)))
        kept = ordered.iloc[:keep_n]
        kept_proxy = kept[proxy_col]
        kept_high_proxy_fraction = float((kept_proxy >= baseline_high_proxy_threshold).mean())
        rows.append(
            {
                "score_name": score_col,
                "proxy_name": proxy_col,
                "keep_quantile": q,
                "n_spots": float(len(kept)),
                "retained_mean_proxy": float(kept_proxy.mean()),
                "retained_median_proxy": float(kept_proxy.median()),
                "retained_mean_proxy_ratio": float(kept_proxy.mean()) / max(baseline_mean_proxy, 1e-12),
                "retained_high_proxy_fraction": kept_high_proxy_fraction,
                "retained_high_proxy_fraction_ratio": kept_high_proxy_fraction / max(baseline_high_proxy_fraction, 1e-12),
            }
        )
    return pd.DataFrame(rows)


def score_proxy_retention_summary(
    curve: pd.DataFrame,
    *,
    focus_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (score_name, proxy_name), sub in curve.groupby(["score_name", "proxy_name"], sort=True):
        row: dict[str, float | str] = {
            "score_name": score_name,
            "proxy_name": proxy_name,
            "mean_retained_proxy_ratio": float(sub["retained_mean_proxy_ratio"].mean()),
            "mean_retained_high_proxy_fraction_ratio": float(sub["retained_high_proxy_fraction_ratio"].mean()),
        }
        for q in focus_quantiles:
            focus = sub.loc[np.isclose(sub["keep_quantile"], q)]
            if focus.empty:
                continue
            row[f"proxy_ratio_at_{str(q).replace('.', 'p')}"] = float(focus.iloc[0]["retained_mean_proxy_ratio"])
            row[f"high_proxy_ratio_at_{str(q).replace('.', 'p')}"] = float(
                focus.iloc[0]["retained_high_proxy_fraction_ratio"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["proxy_name", "mean_retained_proxy_ratio", "mean_retained_high_proxy_fraction_ratio"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def calibrate_proxy_threshold(
    obs: pd.DataFrame,
    *,
    score_col: str,
    proxy_col: str,
    target_high_proxy_fraction: float,
    high_proxy_quantile: float = 0.9,
    higher_score_higher_risk: bool = True,
) -> dict[str, float | bool]:
    ordered = obs.sort_values(score_col, ascending=higher_score_higher_risk).reset_index(drop=True)
    proxy = ordered[proxy_col].to_numpy(dtype=float)
    baseline_mean_proxy = float(np.mean(proxy))
    high_proxy_threshold = float(np.quantile(proxy, high_proxy_quantile))
    high_proxy_mask = proxy >= high_proxy_threshold
    baseline_high_proxy_fraction = float(np.mean(high_proxy_mask))
    keep_n = np.arange(1, len(ordered) + 1, dtype=int)
    cumulative_high_proxy_fraction = np.cumsum(high_proxy_mask.astype(float)) / keep_n
    feasible = cumulative_high_proxy_fraction <= (target_high_proxy_fraction + 1e-12)

    if feasible.any():
        selected_idx = int(np.flatnonzero(feasible)[-1])
        target_met = True
    else:
        selected_idx = int(np.argmin(cumulative_high_proxy_fraction))
        target_met = False

    threshold = float(ordered.iloc[selected_idx][score_col])
    kept = ordered.iloc[: selected_idx + 1]
    kept_proxy = kept[proxy_col].to_numpy(dtype=float)
    retained_high_proxy_fraction = float(np.mean(kept_proxy >= high_proxy_threshold))

    return {
        "score_threshold": threshold,
        "n_kept": float(len(kept)),
        "coverage": float(len(kept) / max(len(ordered), 1)),
        "retained_mean_proxy": float(np.mean(kept_proxy)),
        "retained_mean_proxy_ratio": float(np.mean(kept_proxy)) / max(baseline_mean_proxy, 1e-12),
        "retained_high_proxy_fraction": retained_high_proxy_fraction,
        "retained_high_proxy_fraction_ratio": retained_high_proxy_fraction / max(baseline_high_proxy_fraction, 1e-12),
        "high_proxy_threshold": high_proxy_threshold,
        "target_high_proxy_fraction": float(target_high_proxy_fraction),
        "target_met": target_met,
    }


def evaluate_proxy_threshold(
    obs: pd.DataFrame,
    *,
    score_col: str,
    proxy_col: str,
    score_threshold: float,
    target_high_proxy_fraction: float,
    high_proxy_quantile: float = 0.9,
    higher_score_higher_risk: bool = True,
) -> dict[str, float | bool]:
    if higher_score_higher_risk:
        kept = obs.loc[obs[score_col] <= score_threshold].copy()
    else:
        kept = obs.loc[obs[score_col] >= score_threshold].copy()

    baseline_mean_proxy = float(obs[proxy_col].mean())
    high_proxy_threshold = float(obs[proxy_col].quantile(high_proxy_quantile))
    baseline_high_proxy_mask = obs[proxy_col] >= high_proxy_threshold
    baseline_high_proxy_fraction = float(baseline_high_proxy_mask.mean())

    if kept.empty:
        retained_mean_proxy = np.nan
        retained_mean_proxy_ratio = np.nan
        retained_high_proxy_fraction = 0.0
        retained_high_proxy_fraction_ratio = 0.0
    else:
        retained_mean_proxy = float(kept[proxy_col].mean())
        retained_mean_proxy_ratio = retained_mean_proxy / max(baseline_mean_proxy, 1e-12)
        retained_high_proxy_fraction = float((kept[proxy_col] >= high_proxy_threshold).mean())
        retained_high_proxy_fraction_ratio = retained_high_proxy_fraction / max(baseline_high_proxy_fraction, 1e-12)

    return {
        "score_threshold": float(score_threshold),
        "n_kept": float(len(kept)),
        "coverage": float(len(kept) / max(len(obs), 1)),
        "retained_mean_proxy": retained_mean_proxy,
        "retained_mean_proxy_ratio": retained_mean_proxy_ratio,
        "retained_high_proxy_fraction": retained_high_proxy_fraction,
        "retained_high_proxy_fraction_ratio": retained_high_proxy_fraction_ratio,
        "high_proxy_threshold": high_proxy_threshold,
        "target_high_proxy_fraction": float(target_high_proxy_fraction),
        "target_met": bool(retained_high_proxy_fraction <= (target_high_proxy_fraction + 1e-12)),
    }


def score_proxy_bin_summary(
    obs: pd.DataFrame,
    *,
    score_col: str,
    proxy_col: str,
    bin_quantiles: tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
    bin_labels: tuple[str, ...] = ("low_risk", "mid_risk", "high_risk"),
    high_proxy_quantile: float = 0.9,
    higher_score_higher_risk: bool = True,
) -> pd.DataFrame:
    if len(bin_quantiles) != len(bin_labels) + 1:
        raise ValueError("bin_quantiles must be one longer than bin_labels")

    ordered = obs.sort_values(score_col, ascending=higher_score_higher_risk).reset_index(drop=True)
    total = len(ordered)
    if total == 0:
        return pd.DataFrame(columns=["score_name", "proxy_name", "risk_bin"])

    baseline_mean_proxy = float(ordered[proxy_col].mean())
    high_proxy_threshold = float(ordered[proxy_col].quantile(high_proxy_quantile))
    baseline_high_proxy_fraction = float((ordered[proxy_col] >= high_proxy_threshold).mean())
    rank_fraction = (np.arange(total, dtype=float) + 0.5) / total
    ordered["risk_bin"] = pd.cut(
        rank_fraction,
        bins=list(bin_quantiles),
        labels=list(bin_labels),
        include_lowest=True,
        right=True,
    )

    rows: list[dict[str, float | str]] = []
    for bin_label in bin_labels:
        subset = ordered.loc[ordered["risk_bin"] == bin_label]
        if subset.empty:
            rows.append(
                {
                    "score_name": score_col,
                    "proxy_name": proxy_col,
                    "risk_bin": bin_label,
                    "n_spots": 0.0,
                    "coverage": 0.0,
                    "mean_score": np.nan,
                    "mean_proxy": np.nan,
                    "mean_proxy_ratio": np.nan,
                    "high_proxy_fraction": np.nan,
                    "high_proxy_fraction_ratio": np.nan,
                }
            )
            continue

        high_proxy_fraction = float((subset[proxy_col] >= high_proxy_threshold).mean())
        rows.append(
            {
                "score_name": score_col,
                "proxy_name": proxy_col,
                "risk_bin": bin_label,
                "n_spots": float(len(subset)),
                "coverage": float(len(subset) / total),
                "mean_score": float(subset[score_col].mean()),
                "mean_proxy": float(subset[proxy_col].mean()),
                "mean_proxy_ratio": float(subset[proxy_col].mean()) / max(baseline_mean_proxy, 1e-12),
                "high_proxy_fraction": high_proxy_fraction,
                "high_proxy_fraction_ratio": high_proxy_fraction / max(baseline_high_proxy_fraction, 1e-12),
            }
        )
    return pd.DataFrame(rows)


def score_proxy_bin_monotonicity_summary(
    bin_summary: pd.DataFrame,
    *,
    bin_order: tuple[str, ...] = ("low_risk", "mid_risk", "high_risk"),
) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for (score_name, proxy_name), sub in bin_summary.groupby(["score_name", "proxy_name"], sort=True):
        by_bin = sub.set_index("risk_bin").reindex(bin_order)
        mean_proxy = by_bin["mean_proxy"].to_numpy(dtype=float)
        high_proxy_fraction = by_bin["high_proxy_fraction"].to_numpy(dtype=float)
        coverage = by_bin["coverage"].to_numpy(dtype=float)
        mean_proxy_ratio = by_bin["mean_proxy_ratio"].to_numpy(dtype=float)

        valid_mean = np.isfinite(mean_proxy)
        valid_high = np.isfinite(high_proxy_fraction)
        monotone_mean = bool(np.all(np.diff(mean_proxy[valid_mean]) >= -1e-12)) if valid_mean.sum() >= 2 else False
        monotone_high = (
            bool(np.all(np.diff(high_proxy_fraction[valid_high]) >= -1e-12)) if valid_high.sum() >= 2 else False
        )

        rows.append(
            {
                "score_name": score_name,
                "proxy_name": proxy_name,
                "low_risk_coverage": float(coverage[0]),
                "mid_risk_coverage": float(coverage[1]) if len(coverage) > 1 else np.nan,
                "high_risk_coverage": float(coverage[2]) if len(coverage) > 2 else np.nan,
                "low_risk_mean_proxy": float(mean_proxy[0]),
                "mid_risk_mean_proxy": float(mean_proxy[1]) if len(mean_proxy) > 1 else np.nan,
                "high_risk_mean_proxy": float(mean_proxy[2]) if len(mean_proxy) > 2 else np.nan,
                "low_risk_mean_proxy_ratio": float(mean_proxy_ratio[0]),
                "mid_risk_mean_proxy_ratio": float(mean_proxy_ratio[1]) if len(mean_proxy_ratio) > 1 else np.nan,
                "high_risk_mean_proxy_ratio": float(mean_proxy_ratio[2]) if len(mean_proxy_ratio) > 2 else np.nan,
                "low_risk_high_proxy_fraction": float(high_proxy_fraction[0]),
                "mid_risk_high_proxy_fraction": float(high_proxy_fraction[1]) if len(high_proxy_fraction) > 1 else np.nan,
                "high_risk_high_proxy_fraction": float(high_proxy_fraction[2]) if len(high_proxy_fraction) > 2 else np.nan,
                "high_minus_low_mean_proxy": float(mean_proxy[-1] - mean_proxy[0]),
                "high_minus_low_high_proxy_fraction": float(high_proxy_fraction[-1] - high_proxy_fraction[0]),
                "monotone_mean_proxy": monotone_mean,
                "monotone_high_proxy_fraction": monotone_high,
            }
        )
    return pd.DataFrame(rows)


def score_proxy_sample_retention_summary(
    obs: pd.DataFrame,
    *,
    score_col: str,
    proxy_col: str,
    sample_col: str = "sample_id",
    keep_quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    focus_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    high_proxy_quantile: float = 0.9,
    higher_score_higher_risk: bool = True,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sample_id, sample_obs in obs.groupby(sample_col, sort=True):
        curve = score_proxy_retention_curve(
            sample_obs,
            score_col=score_col,
            proxy_col=proxy_col,
            keep_quantiles=keep_quantiles,
            high_proxy_quantile=high_proxy_quantile,
            higher_score_higher_risk=higher_score_higher_risk,
        )
        summary = score_proxy_retention_summary(curve, focus_quantiles=focus_quantiles)
        summary.insert(0, sample_col, sample_id)
        rows.append(summary)
    if not rows:
        return pd.DataFrame(columns=[sample_col, "score_name", "proxy_name"])
    return pd.concat(rows, ignore_index=True)


def bootstrap_metric_summary(
    obs: pd.DataFrame,
    *,
    group_cols: tuple[str, ...],
    metric_cols: tuple[str, ...],
    sample_col: str = "sample_id",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 0,
) -> pd.DataFrame:
    if obs.empty:
        return pd.DataFrame()
    lower_q = (1.0 - ci) / 2.0
    upper_q = 1.0 - lower_q
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | str]] = []

    for group_key, group in obs.groupby(list(group_cols), sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group.loc[:, list(metric_cols)].to_numpy(dtype=float)
        sample_count = len(group)
        if sample_count == 0:
            continue
        bootstrap_means = np.empty((n_bootstrap, len(metric_cols)), dtype=float)
        for i in range(n_bootstrap):
            take = rng.integers(0, sample_count, size=sample_count)
            bootstrap_means[i] = values[take].mean(axis=0)

        row: dict[str, float | str] = dict(zip(group_cols, group_key, strict=True))
        row["sample_count"] = float(group[sample_col].nunique())
        for metric_idx, metric_col in enumerate(metric_cols):
            observed = float(values[:, metric_idx].mean())
            row[metric_col] = observed
            row[f"{metric_col}_ci_low"] = float(np.quantile(bootstrap_means[:, metric_idx], lower_q))
            row[f"{metric_col}_ci_high"] = float(np.quantile(bootstrap_means[:, metric_idx], upper_q))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


def paired_sample_metric_summary(
    obs: pd.DataFrame,
    *,
    baseline_run: str,
    challenger_run: str,
    group_cols: tuple[str, ...],
    metric_cols: tuple[str, ...],
    run_col: str = "run_label",
    sample_col: str = "sample_id",
    lower_is_better: bool = True,
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    if obs.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | str]] = []

    for group_key, group in obs.groupby(list(group_cols), sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        baseline = group.loc[group[run_col] == baseline_run].copy()
        challenger = group.loc[group[run_col] == challenger_run].copy()
        if baseline.empty or challenger.empty:
            continue

        merged = baseline[[sample_col, *metric_cols]].merge(
            challenger[[sample_col, *metric_cols]],
            on=sample_col,
            suffixes=("_baseline", "_challenger"),
            how="inner",
        )
        sample_count = len(merged)
        if sample_count == 0:
            continue

        row: dict[str, float | str] = dict(zip(group_cols, group_key, strict=True))
        row["baseline_run"] = baseline_run
        row["challenger_run"] = challenger_run
        row["sample_count"] = float(sample_count)

        for metric_col in metric_cols:
            baseline_values = merged[f"{metric_col}_baseline"].to_numpy(dtype=float)
            challenger_values = merged[f"{metric_col}_challenger"].to_numpy(dtype=float)
            diffs = challenger_values - baseline_values
            if lower_is_better:
                win_mask = diffs < 0
                loss_mask = diffs > 0
            else:
                win_mask = diffs > 0
                loss_mask = diffs < 0
            tie_mask = np.isclose(diffs, 0.0)

            bootstrap_means = np.empty(n_bootstrap, dtype=float)
            for i in range(n_bootstrap):
                take = rng.integers(0, sample_count, size=sample_count)
                bootstrap_means[i] = diffs[take].mean()

            sign_flips = np.array(np.meshgrid(*([[-1.0, 1.0]] * sample_count))).T.reshape(-1, sample_count)
            null_means = (sign_flips * np.abs(diffs)).mean(axis=1)
            observed_mean = float(diffs.mean())
            if lower_is_better:
                one_sided_p = float(np.mean(null_means <= observed_mean))
            else:
                one_sided_p = float(np.mean(null_means >= observed_mean))
            two_sided_p = float(np.mean(np.abs(null_means) >= abs(observed_mean)))

            row[f"{metric_col}_baseline_mean"] = float(baseline_values.mean())
            row[f"{metric_col}_challenger_mean"] = float(challenger_values.mean())
            row[f"{metric_col}_mean_diff"] = observed_mean
            row[f"{metric_col}_diff_ci_low"] = float(np.quantile(bootstrap_means, 0.025))
            row[f"{metric_col}_diff_ci_high"] = float(np.quantile(bootstrap_means, 0.975))
            row[f"{metric_col}_challenger_win_count"] = float(win_mask.sum())
            row[f"{metric_col}_challenger_loss_count"] = float(loss_mask.sum())
            row[f"{metric_col}_tie_count"] = float(tie_mask.sum())
            row[f"{metric_col}_challenger_win_rate"] = float(win_mask.mean())
            row[f"{metric_col}_one_sided_p"] = one_sided_p
            row[f"{metric_col}_two_sided_p"] = two_sided_p

        rows.append(row)

    sort_cols = list(group_cols)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def sample_risk_summary(
    obs: pd.DataFrame,
    *,
    sample_col: str = "sample_id",
    quantile: float = 0.9,
) -> pd.DataFrame:
    threshold = obs["risk_score"].quantile(quantile)
    high_risk = obs["risk_score"] >= threshold
    grouped = obs.groupby(sample_col, sort=True)
    columns = {
        "n_spots": grouped.size(),
        "mean_risk_score": grouped["risk_score"].mean(),
        "std_risk_score": grouped["risk_score"].std().fillna(0.0),
        "high_risk_fraction": high_risk.groupby(obs[sample_col]).mean(),
        "mean_phi_local": grouped["phi_local"].mean(),
        "mean_phi_uncertainty": grouped["phi_uncertainty"].mean(),
        "mean_phi_stability": grouped["phi_stability"].mean(),
    }
    if "phi_reference" in obs.columns:
        columns["mean_phi_reference"] = grouped["phi_reference"].mean()
    if "neighbor_agreement" in obs.columns:
        columns["mean_neighbor_agreement"] = grouped["neighbor_agreement"].mean()
    if "boundary_mask" in obs.columns:
        columns["boundary_fraction"] = grouped["boundary_mask"].mean()
    summary = pd.DataFrame(columns).reset_index()
    return summary.sort_values("mean_risk_score", ascending=False).reset_index(drop=True)


def sample_layer_risk_summary(
    obs: pd.DataFrame,
    *,
    sample_col: str = "sample_id",
    layer_col: str = "layer_guess",
    quantile: float = 0.9,
) -> pd.DataFrame:
    threshold = obs["risk_score"].quantile(quantile)
    data = obs.copy()
    data = data.loc[_valid_layer_mask(data[layer_col])].copy()
    data["is_high_risk"] = data["risk_score"] >= threshold
    grouped = data.groupby([sample_col, layer_col], sort=True)
    summary = pd.DataFrame(
        {
            "n_spots": grouped.size(),
            "mean_risk_score": grouped["risk_score"].mean(),
            "high_risk_fraction": grouped["is_high_risk"].mean(),
            "mean_neighbor_agreement": grouped["neighbor_agreement"].mean(),
        }
    ).reset_index()
    return summary.sort_values([sample_col, "mean_risk_score"], ascending=[True, False]).reset_index(drop=True)


def cross_sample_layer_consistency_summary(
    sample_layer_summary: pd.DataFrame,
    *,
    layer_col: str = "layer_guess",
) -> pd.DataFrame:
    grouped = sample_layer_summary.groupby(layer_col, sort=True)
    summary = pd.DataFrame(
        {
            "sample_count": grouped["sample_id"].nunique(),
            "mean_risk_score_mean": grouped["mean_risk_score"].mean(),
            "mean_risk_score_std": grouped["mean_risk_score"].std().fillna(0.0),
            "mean_high_risk_fraction": grouped["high_risk_fraction"].mean(),
            "mean_neighbor_agreement": grouped["mean_neighbor_agreement"].mean(),
        }
    ).reset_index()
    summary["risk_score_cv"] = summary["mean_risk_score_std"] / summary["mean_risk_score_mean"].clip(lower=1e-12)
    return summary.sort_values(["mean_risk_score_mean", "sample_count"], ascending=[False, False]).reset_index(drop=True)


def dominant_layer_frequency_summary(
    sample_layer_summary: pd.DataFrame,
    *,
    sample_col: str = "sample_id",
    layer_col: str = "layer_guess",
    top_k: int = 2,
) -> pd.DataFrame:
    ordered = sample_layer_summary.sort_values([sample_col, "mean_risk_score"], ascending=[True, False]).copy()
    ordered["rank_within_sample"] = ordered.groupby(sample_col).cumcount() + 1
    top1 = ordered.loc[ordered["rank_within_sample"] == 1]
    topk = ordered.loc[ordered["rank_within_sample"] <= top_k]
    summary = pd.DataFrame(
        {
            "top1_count": top1.groupby(layer_col).size(),
            f"top{top_k}_count": topk.groupby(layer_col).size(),
        }
    ).fillna(0.0)
    summary["top1_fraction"] = summary["top1_count"] / max(top1[sample_col].nunique(), 1)
    summary[f"top{top_k}_fraction"] = summary[f"top{top_k}_count"] / max(top1[sample_col].nunique(), 1)
    return summary.reset_index().sort_values(["top1_count", f"top{top_k}_count"], ascending=[False, False]).reset_index(drop=True)


def celltype_risk_association_summary(
    abundance: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    quantile: float = 0.9,
) -> pd.DataFrame:
    abundance = abundance.loc[obs.index]
    threshold = obs["risk_score"].quantile(quantile)
    high_mask = obs["risk_score"] >= threshold
    low_mask = ~high_mask
    rows: list[dict[str, float | str]] = []
    for celltype in abundance.columns:
        values = abundance[celltype]
        rows.append(
            {
                "celltype": celltype,
                "corr_with_risk": float(values.corr(obs["risk_score"])),
                "mean_abundance_high_risk": float(values.loc[high_mask].mean()),
                "mean_abundance_low_risk": float(values.loc[low_mask].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["abundance_delta"] = summary["mean_abundance_high_risk"] - summary["mean_abundance_low_risk"]
    summary["abundance_ratio"] = summary["mean_abundance_high_risk"] / summary["mean_abundance_low_risk"].clip(lower=1e-12)
    return summary.sort_values(["abundance_delta", "corr_with_risk"], ascending=[False, False]).reset_index(drop=True)


def layer_celltype_risk_summary(
    abundance: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    layer_col: str = "layer_guess",
    quantile: float = 0.9,
) -> pd.DataFrame:
    abundance = abundance.loc[obs.index]
    data_obs = obs.loc[_valid_layer_mask(obs[layer_col])].copy()
    abundance = abundance.loc[data_obs.index]
    threshold = obs["risk_score"].quantile(quantile)
    rows: list[dict[str, float | str]] = []
    for layer_name, layer_index in data_obs.groupby(layer_col).groups.items():
        layer_obs = data_obs.loc[layer_index]
        layer_abundance = abundance.loc[layer_index]
        high_mask = layer_obs["risk_score"] >= threshold
        low_mask = ~high_mask
        if high_mask.sum() == 0 or low_mask.sum() == 0:
            continue
        for celltype in layer_abundance.columns:
            values = layer_abundance[celltype]
            rows.append(
                {
                    "layer_guess": layer_name,
                    "celltype": celltype,
                    "mean_abundance_high_risk": float(values.loc[high_mask].mean()),
                    "mean_abundance_low_risk": float(values.loc[low_mask].mean()),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["abundance_delta"] = summary["mean_abundance_high_risk"] - summary["mean_abundance_low_risk"]
    summary["abundance_ratio"] = summary["mean_abundance_high_risk"] / summary["mean_abundance_low_risk"].clip(lower=1e-12)
    return summary.sort_values(["layer_guess", "abundance_delta"], ascending=[True, False]).reset_index(drop=True)


def selective_layer_retention_by_group(
    obs: pd.DataFrame,
    *,
    layer_col: str = "layer_guess",
    keep_quantiles: tuple[float, ...] = (0.8, 0.9, 1.0),
) -> pd.DataFrame:
    data = obs.loc[_valid_layer_mask(obs[layer_col])].copy()
    ordered = data.sort_values("risk_score", ascending=True).reset_index(drop=True)
    baseline = ordered[layer_col].value_counts(normalize=True).rename("baseline_fraction")
    total = len(ordered)
    rows: list[dict[str, float | str]] = []
    for q in keep_quantiles:
        keep_n = max(1, int(np.ceil(total * q)))
        kept = ordered.iloc[:keep_n]
        kept_fraction = kept[layer_col].value_counts(normalize=True)
        kept_count = kept[layer_col].value_counts()
        all_layers = sorted(set(baseline.index).union(set(kept_fraction.index)))
        for layer_name in all_layers:
            base_frac = float(baseline.get(layer_name, 0.0))
            keep_frac = float(kept_fraction.get(layer_name, 0.0))
            rows.append(
                {
                    "keep_quantile": q,
                    "layer_guess": layer_name,
                    "n_spots_retained": float(kept_count.get(layer_name, 0)),
                    "retained_fraction": keep_frac,
                    "baseline_fraction": base_frac,
                    "retention_ratio": keep_frac / max(base_frac, 1e-12),
                }
            )
    return pd.DataFrame(rows).sort_values(["keep_quantile", "retention_ratio"], ascending=[True, False]).reset_index(drop=True)


def selective_celltype_shift_summary(
    abundance: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    keep_quantiles: tuple[float, ...] = (0.8, 0.9, 1.0),
) -> pd.DataFrame:
    abundance = abundance.loc[obs.index]
    ordered_index = obs.sort_values("risk_score", ascending=True).index
    abundance = abundance.loc[ordered_index]
    baseline = abundance.mean(axis=0)
    total = len(abundance)
    rows: list[dict[str, float | str]] = []
    for q in keep_quantiles:
        keep_n = max(1, int(np.ceil(total * q)))
        kept = abundance.iloc[:keep_n]
        kept_mean = kept.mean(axis=0)
        for celltype in abundance.columns:
            base_mean = float(baseline[celltype])
            keep_mean_value = float(kept_mean[celltype])
            rows.append(
                {
                    "keep_quantile": q,
                    "celltype": celltype,
                    "mean_abundance_retained": keep_mean_value,
                    "baseline_mean_abundance": base_mean,
                    "abundance_delta": keep_mean_value - base_mean,
                    "abundance_ratio": keep_mean_value / max(base_mean, 1e-12),
                }
            )
    return pd.DataFrame(rows).sort_values(["keep_quantile", "abundance_delta"], ascending=[True, False]).reset_index(drop=True)


def layer_enrichment_by_strategy_summary(
    abundance: pd.DataFrame,
    obs: pd.DataFrame,
    strategy_masks: pd.DataFrame,
    *,
    targets: pd.DataFrame,
    sample_col: str = "sample_id",
    layer_col: str = "layer_guess",
) -> pd.DataFrame:
    abundance = abundance.loc[obs.index]
    strategy_masks = strategy_masks.loc[obs.index]
    data_obs = obs.loc[_valid_layer_mask(obs[layer_col])].copy()
    abundance = abundance.loc[data_obs.index]
    strategy_masks = strategy_masks.loc[data_obs.index]

    required_cols = {"celltype", "target_layer"}
    missing = required_cols.difference(targets.columns)
    if missing:
        raise ValueError(f"targets is missing required columns: {sorted(missing)}")

    rows: list[dict[str, float | str]] = []
    target_rows = targets.loc[targets["celltype"].isin(abundance.columns)].copy()
    for sample_id, sample_obs in data_obs.groupby(sample_col, sort=True):
        sample_abundance = abundance.loc[sample_obs.index]
        sample_masks = strategy_masks.loc[sample_obs.index]
        for strategy_name, keep_mask in sample_masks.items():
            kept_index = sample_obs.index[keep_mask.astype(bool)]
            if kept_index.empty:
                continue
            kept_obs = sample_obs.loc[kept_index]
            kept_abundance = sample_abundance.loc[kept_index]
            for _, target in target_rows.iterrows():
                celltype = str(target["celltype"])
                target_layer = str(target["target_layer"])
                in_layer_mask = kept_obs[layer_col].astype(str) == target_layer
                out_layer_mask = ~in_layer_mask
                if in_layer_mask.sum() == 0 or out_layer_mask.sum() == 0:
                    continue
                values = kept_abundance[celltype]
                target_mean = float(values.loc[in_layer_mask].mean())
                other_mean = float(values.loc[out_layer_mask].mean())
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "strategy_name": str(strategy_name),
                        "celltype": celltype,
                        "target_layer": target_layer,
                        "n_spots": float(len(kept_obs)),
                        "n_target_layer_spots": float(in_layer_mask.sum()),
                        "n_other_layer_spots": float(out_layer_mask.sum()),
                        "target_layer_mean_abundance": target_mean,
                        "other_layers_mean_abundance": other_mean,
                        "layer_enrichment_delta": target_mean - other_mean,
                        "layer_enrichment_ratio": target_mean / max(other_mean, 1e-12),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["celltype", "strategy_name", "sample_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def layer_profile_reproducibility_by_strategy_summary(
    abundance: pd.DataFrame,
    obs: pd.DataFrame,
    strategy_masks: pd.DataFrame,
    *,
    targets: pd.DataFrame,
    sample_col: str = "sample_id",
    layer_col: str = "layer_guess",
    layer_order: tuple[str, ...] = ("Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"),
    min_layers: int = 4,
) -> pd.DataFrame:
    abundance = abundance.loc[obs.index]
    strategy_masks = strategy_masks.loc[obs.index]
    data_obs = obs.loc[_valid_layer_mask(obs[layer_col])].copy()
    abundance = abundance.loc[data_obs.index]
    strategy_masks = strategy_masks.loc[data_obs.index]

    required_cols = {"celltype"}
    missing = required_cols.difference(targets.columns)
    if missing:
        raise ValueError(f"targets is missing required columns: {sorted(missing)}")
    target_layer_map = (
        targets.set_index("celltype")["target_layer"].astype(str).to_dict() if "target_layer" in targets.columns else {}
    )
    selected_celltypes = [str(celltype) for celltype in targets["celltype"] if celltype in abundance.columns]

    rows: list[dict[str, float | str]] = []
    for strategy_name in strategy_masks.columns:
        for celltype in selected_celltypes:
            profiles: dict[str, pd.Series] = {}
            for sample_id, sample_obs in data_obs.groupby(sample_col, sort=True):
                keep_index = sample_obs.index[strategy_masks.loc[sample_obs.index, strategy_name].astype(bool)]
                if len(keep_index) == 0:
                    continue
                kept_obs = sample_obs.loc[keep_index]
                kept_abundance = abundance.loc[keep_index]
                layer_profile = kept_abundance.groupby(kept_obs[layer_col].astype(str))[celltype].mean().reindex(layer_order)
                if layer_profile.notna().sum() < min_layers:
                    continue
                profiles[str(sample_id)] = layer_profile

            if len(profiles) < 2:
                continue

            for sample_id, profile in profiles.items():
                other_profiles = [other for other_id, other in profiles.items() if other_id != sample_id]
                if not other_profiles:
                    continue
                centroid = pd.concat(other_profiles, axis=1).mean(axis=1)
                valid = profile.notna() & centroid.notna()
                if valid.sum() < min_layers:
                    continue
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "strategy_name": str(strategy_name),
                        "celltype": celltype,
                        "target_layer": target_layer_map.get(celltype, ""),
                        "n_layers_used": float(valid.sum()),
                        "reproducibility_corr": float(profile.loc[valid].corr(centroid.loc[valid])),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["celltype", "strategy_name", "sample_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
