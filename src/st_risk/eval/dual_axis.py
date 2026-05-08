from __future__ import annotations

import numpy as np
import pandas as pd

STRUCTURE_AXIS_WEIGHTS = {
    "phi_local": 3.0,
    "phi_uncertainty": 1.0,
    "phi_stability": 0.0,
    "phi_reference": 0.0,
}

REFERENCE_AXIS_WEIGHTS = {
    "phi_local": 0.0,
    "phi_uncertainty": 2.0,
    "phi_stability": 1.0,
    "phi_reference": 0.0,
}


def normalized_axis_weights(weights: dict[str, float] | None, *, fallback: dict[str, float]) -> dict[str, float]:
    if not weights:
        return fallback.copy()
    result = {
        "phi_local": float(weights.get("phi_local", fallback.get("phi_local", 0.0))),
        "phi_uncertainty": float(weights.get("phi_uncertainty", fallback.get("phi_uncertainty", 0.0))),
        "phi_stability": float(weights.get("phi_stability", fallback.get("phi_stability", 0.0))),
        "phi_reference": float(weights.get("phi_reference", fallback.get("phi_reference", 0.0))),
    }
    if np.isclose(sum(abs(value) for value in result.values()), 0.0):
        raise ValueError("Dual-axis weights must include at least one non-zero entry.")
    return result


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = values.std()
    if np.isclose(std, 0.0):
        return np.zeros_like(values, dtype=float)
    return (values - values.mean()) / std


def _grouped_zscore(values: np.ndarray, groups: np.ndarray | pd.Series | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if groups is None:
        return _zscore(values)

    group_array = np.asarray(groups)
    if group_array.shape[0] != values.shape[0]:
        raise ValueError("groups must have the same length as the feature values")

    normalized = np.zeros_like(values, dtype=float)
    for group_name in pd.Index(group_array).unique():
        mask = group_array == group_name
        normalized[mask] = _zscore(values[mask])
    return normalized


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _combine_axis_features(
    features: pd.DataFrame,
    *,
    weights: dict[str, float],
    groups: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    linear = np.zeros(features.shape[0], dtype=float)
    total_weight = 0.0
    for name, weight in weights.items():
        weight = float(weight)
        if np.isclose(weight, 0.0):
            continue
        if name in features.columns:
            values = features[name].to_numpy(dtype=float)
        else:
            values = np.zeros(features.shape[0], dtype=float)
        linear = linear + weight * _grouped_zscore(values, groups=groups)
        total_weight += abs(weight)
    if np.isclose(total_weight, 0.0):
        raise ValueError("At least one non-zero dual-axis feature weight is required")
    return _sigmoid(linear / total_weight)


def build_dual_axis_scores(
    features: pd.DataFrame,
    *,
    groups: np.ndarray | pd.Series | None = None,
    structure_weights: dict[str, float] | None = None,
    reference_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    structure_weights = normalized_axis_weights(structure_weights, fallback=STRUCTURE_AXIS_WEIGHTS)
    reference_weights = normalized_axis_weights(reference_weights, fallback=REFERENCE_AXIS_WEIGHTS)
    result = pd.DataFrame(index=features.index)
    result["structure_risk_score"] = _combine_axis_features(features, weights=structure_weights, groups=groups)
    result["reference_risk_score"] = _combine_axis_features(features, weights=reference_weights, groups=groups)
    return result


def dual_axis_correlation_summary(
    obs: pd.DataFrame,
    *,
    sample_col: str = "sample_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_summary = pd.DataFrame(
        [
            {
                "scope": "global",
                "pearson_corr": float(obs["structure_risk_score"].corr(obs["reference_risk_score"])),
                "mean_structure_risk_score": float(obs["structure_risk_score"].mean()),
                "mean_reference_risk_score": float(obs["reference_risk_score"].mean()),
            }
        ]
    )

    rows: list[dict[str, float | str]] = []
    for sample_id, sub in obs.groupby(sample_col, sort=True):
        rows.append(
            {
                "sample_id": str(sample_id),
                "pearson_corr": float(sub["structure_risk_score"].corr(sub["reference_risk_score"])),
                "mean_structure_risk_score": float(sub["structure_risk_score"].mean()),
                "mean_reference_risk_score": float(sub["reference_risk_score"].mean()),
            }
        )
    sample_summary = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    return global_summary, sample_summary


def assign_dual_axis_quadrants(
    obs: pd.DataFrame,
    *,
    sample_col: str = "sample_id",
    threshold_quantile: float = 0.5,
) -> pd.DataFrame:
    result = obs.copy()
    structure_high = pd.Series(False, index=result.index)
    reference_high = pd.Series(False, index=result.index)

    for _, sub in result.groupby(sample_col, sort=True):
        structure_threshold = float(sub["structure_risk_score"].quantile(threshold_quantile))
        reference_threshold = float(sub["reference_risk_score"].quantile(threshold_quantile))
        structure_high.loc[sub.index] = sub["structure_risk_score"] >= structure_threshold
        reference_high.loc[sub.index] = sub["reference_risk_score"] >= reference_threshold

    result["structure_high"] = structure_high
    result["reference_high"] = reference_high
    result["dual_axis_quadrant"] = np.select(
        [
            (~structure_high) & (~reference_high),
            structure_high & (~reference_high),
            (~structure_high) & reference_high,
            structure_high & reference_high,
        ],
        [
            "low_both",
            "structure_only",
            "reference_only",
            "high_both",
        ],
        default="unassigned",
    )
    return result


def dual_axis_quadrant_proxy_summary(
    obs: pd.DataFrame,
    *,
    proxy_cols: tuple[str, ...],
    quadrant_col: str = "dual_axis_quadrant",
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    total = len(obs)
    quadrant_order = ["low_both", "structure_only", "reference_only", "high_both"]

    for proxy_col in proxy_cols:
        for quadrant in quadrant_order:
            subset = obs.loc[obs[quadrant_col] == quadrant]
            if subset.empty:
                continue
            rows.append(
                {
                    "proxy_name": proxy_col,
                    "quadrant": quadrant,
                    "n_spots": float(len(subset)),
                    "fraction": float(len(subset) / max(total, 1)),
                    "mean_structure_risk_score": float(subset["structure_risk_score"].mean()),
                    "mean_reference_risk_score": float(subset["reference_risk_score"].mean()),
                    "mean_proxy": float(subset[proxy_col].mean()),
                }
            )
    return pd.DataFrame(rows)


def dual_axis_filter_strategy_summary(
    obs: pd.DataFrame,
    *,
    proxy_cols: tuple[str, ...],
    sample_col: str = "sample_id",
    quadrant_col: str = "dual_axis_quadrant",
    high_proxy_quantile: float = 0.9,
) -> pd.DataFrame:
    strategy_masks = dual_axis_filter_strategy_masks(
        obs,
        sample_col=sample_col,
        quadrant_col=quadrant_col,
        include_unfiltered=False,
    )
    rows: list[dict[str, float | str]] = []

    for sample_id, sub in obs.groupby(sample_col, sort=True):
        sub_strategy_masks = strategy_masks.loc[sub.index]

        for proxy_col in proxy_cols:
            baseline_mean_proxy = float(sub[proxy_col].mean())
            high_proxy_threshold = float(sub[proxy_col].quantile(high_proxy_quantile))
            baseline_high_proxy_fraction = float((sub[proxy_col] >= high_proxy_threshold).mean())

            for strategy_name, mask in sub_strategy_masks.items():
                kept = sub.loc[mask.astype(bool)]
                if kept.empty:
                    continue
                retained_high_proxy_fraction = float((kept[proxy_col] >= high_proxy_threshold).mean())
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "proxy_name": proxy_col,
                        "strategy_name": strategy_name,
                        "n_spots": float(len(kept)),
                        "coverage": float(len(kept) / max(len(sub), 1)),
                        "retained_mean_proxy": float(kept[proxy_col].mean()),
                        "retained_mean_proxy_ratio": float(kept[proxy_col].mean()) / max(baseline_mean_proxy, 1e-12),
                        "retained_high_proxy_fraction": retained_high_proxy_fraction,
                        "retained_high_proxy_fraction_ratio": retained_high_proxy_fraction / max(
                            baseline_high_proxy_fraction, 1e-12
                        ),
                    }
                )

    return pd.DataFrame(rows)


def dual_axis_filter_strategy_masks(
    obs: pd.DataFrame,
    *,
    sample_col: str = "sample_id",
    quadrant_col: str = "dual_axis_quadrant",
    include_unfiltered: bool = True,
) -> pd.DataFrame:
    columns = ["low_both", "single_risk_matched", "structure_axis_matched", "reference_axis_matched"]
    if include_unfiltered:
        columns = ["unfiltered", *columns]
    result = pd.DataFrame(False, index=obs.index, columns=columns, dtype=bool)
    if include_unfiltered:
        result.loc[:, "unfiltered"] = True
    for sample_id, sub in obs.groupby(sample_col, sort=True):
        low_both_mask = sub[quadrant_col] == "low_both"
        keep_n = int(low_both_mask.sum())
        if keep_n <= 0:
            continue
        result.loc[sub.index, "low_both"] = low_both_mask.to_numpy(dtype=bool)
        result.loc[
            sub.sort_values("risk_score", ascending=True).index[:keep_n],
            "single_risk_matched",
        ] = True
        result.loc[
            sub.sort_values("structure_risk_score", ascending=True).index[:keep_n],
            "structure_axis_matched",
        ] = True
        result.loc[
            sub.sort_values("reference_risk_score", ascending=True).index[:keep_n],
            "reference_axis_matched",
        ] = True
    return result
