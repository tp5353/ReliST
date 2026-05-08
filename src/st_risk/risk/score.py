from __future__ import annotations

from itertools import product
from math import gcd

import numpy as np
import pandas as pd

REQUIRED_FEATURE_COLUMNS = ("phi_local", "phi_uncertainty", "phi_stability")


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = values.std()
    if np.isclose(std, 0.0):
        return np.zeros_like(values)
    return (values - values.mean()) / std


def grouped_zscore(values: np.ndarray, groups: np.ndarray | pd.Series | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if groups is None:
        return zscore(values)

    group_array = np.asarray(groups)
    if group_array.shape[0] != values.shape[0]:
        raise ValueError("groups must have the same length as the feature values")

    normalized = np.zeros_like(values, dtype=float)
    for group_name in pd.Index(group_array).unique():
        mask = group_array == group_name
        normalized[mask] = zscore(values[mask])
    return normalized


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def combine_risk_features(
    features: dict[str, np.ndarray] | pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
    groups: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    if isinstance(features, pd.DataFrame):
        feature_map = {col: features[col].to_numpy() for col in REQUIRED_FEATURE_COLUMNS}
    else:
        feature_map = {col: np.asarray(features[col], dtype=float) for col in REQUIRED_FEATURE_COLUMNS}

    if weights is None:
        weights = {col: 1.0 for col in REQUIRED_FEATURE_COLUMNS}

    linear = np.zeros_like(next(iter(feature_map.values())), dtype=float)
    total_weight = 0.0
    for name, values in feature_map.items():
        weight = float(weights.get(name, 0.0))
        if np.isclose(weight, 0.0):
            continue
        linear = linear + weight * grouped_zscore(values, groups=groups)
        total_weight += abs(weight)
    if np.isclose(total_weight, 0.0):
        raise ValueError("At least one non-zero risk-feature weight is required")
    return sigmoid(linear / total_weight)


def attach_risk_score(
    features: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
    groups: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    result = features.copy()
    result["risk_score"] = combine_risk_features(result, weights=weights, groups=groups)
    return result


def weight_scheme_name(weights: dict[str, float]) -> str:
    local = float(weights.get("phi_local", 0.0))
    uncertainty = float(weights.get("phi_uncertainty", 0.0))
    stability = float(weights.get("phi_stability", 0.0))
    return f"wL{local:g}_U{uncertainty:g}_S{stability:g}"


def enumerate_weight_schemes(levels: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)) -> dict[str, dict[str, float]]:
    schemes: dict[str, dict[str, float]] = {}
    seen: set[tuple[int, ...]] = set()
    for values in product(levels, repeat=len(REQUIRED_FEATURE_COLUMNS)):
        if np.isclose(sum(abs(v) for v in values), 0.0):
            continue
        integer_values = tuple(int(round(float(value) * 1000)) for value in values)
        nonzero = [abs(v) for v in integer_values if v != 0]
        divisor = nonzero[0]
        for value in nonzero[1:]:
            divisor = gcd(divisor, value)
        canonical = tuple(v // divisor for v in integer_values)
        if canonical in seen:
            continue
        seen.add(canonical)
        weights = {name: float(canonical[i]) for i, name in enumerate(REQUIRED_FEATURE_COLUMNS)}
        schemes[weight_scheme_name(weights)] = weights
    return schemes
