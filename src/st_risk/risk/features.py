from __future__ import annotations

import numpy as np
import pandas as pd

from st_risk.models.base import BaseSpatialModelOutput


def _as_2d_array(values: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    if isinstance(values, (pd.DataFrame, pd.Series)):
        array = values.to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    return array


def _unit_interval_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("_unit_interval_rank expects a 1D array")
    if values.size <= 1 or np.allclose(values, values[0]):
        return np.zeros_like(values, dtype=float)
    ranks = pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)
    return np.clip(ranks, 0.0, 1.0)


def local_residual_score(
    abundance: np.ndarray,
    neighbors: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    heterogeneity: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute a spot-level residual against the neighborhood average."""
    abundance = np.asarray(abundance, dtype=float)
    neighbors = np.asarray(neighbors, dtype=int)
    n_spots = abundance.shape[0]
    scores = np.zeros(n_spots, dtype=float)
    for i in range(n_spots):
        row = neighbors[i]
        valid_mask = row >= 0
        valid_neighbors = row[valid_mask]
        if len(valid_neighbors) == 0:
            continue
        neighbor_values = abundance[valid_neighbors]
        if weights is None:
            neighbor_mean = neighbor_values.mean(axis=0)
        else:
            local_weights = np.asarray(weights[i], dtype=float)[valid_mask]
            if np.allclose(local_weights.sum(), 0.0):
                neighbor_mean = neighbor_values.mean(axis=0)
            else:
                local_weights = local_weights / local_weights.sum()
                neighbor_mean = local_weights @ neighbor_values
        scores[i] = np.abs(abundance[i] - neighbor_mean).sum()
    if heterogeneity is not None:
        heterogeneity = np.asarray(heterogeneity, dtype=float)
        scores = scores / (heterogeneity + eps)
    return scores


def uncertainty_score(uncertainty: np.ndarray | pd.DataFrame | pd.Series | None) -> np.ndarray:
    if uncertainty is None:
        raise ValueError("uncertainty cannot be None when computing uncertainty_score")
    values = _as_2d_array(uncertainty)
    if values.ndim == 1:
        return values.astype(float)
    return values.mean(axis=1)


def ambiguity_score(abundance: np.ndarray | pd.DataFrame) -> np.ndarray:
    values = _as_2d_array(abundance)
    row_sums = values.sum(axis=1, keepdims=True)
    probs = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
    max_prob = probs.max(axis=1)
    if probs.shape[1] <= 1:
        normalized_entropy = np.zeros(probs.shape[0], dtype=float)
    else:
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        normalized_entropy = entropy / np.log(probs.shape[1])
    return 0.5 * normalized_entropy + 0.5 * (1.0 - max_prob)


def confidence_proxy_score(
    abundance: np.ndarray | pd.DataFrame,
    uncertainty: np.ndarray | pd.DataFrame | pd.Series | None = None,
    *,
    uncertainty_weight: float = 0.5,
) -> np.ndarray:
    ambiguity = ambiguity_score(abundance)
    if uncertainty is None:
        return ambiguity

    uncertainty_component = uncertainty_score(uncertainty)
    weight = float(np.clip(uncertainty_weight, 0.0, 1.0))
    ambiguity_component = _unit_interval_rank(ambiguity)
    uncertainty_component = _unit_interval_rank(uncertainty_component)
    return (weight * uncertainty_component) + ((1.0 - weight) * ambiguity_component)


def stability_score(predictions: np.ndarray) -> np.ndarray:
    """Compute spot-level prediction instability from repeated perturbation runs."""
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 3:
        raise ValueError("predictions must have shape (repeats, n_spots, n_features)")
    ddof = 1 if values.shape[0] > 1 else 0
    return values.var(axis=0, ddof=ddof).mean(axis=1)


def build_feature_table(
    model_output: BaseSpatialModelOutput,
    *,
    neighbors: np.ndarray,
    weights: np.ndarray | None = None,
    heterogeneity: np.ndarray | None = None,
    stability_predictions: np.ndarray | None = None,
    confidence_proxy_weight: float = 0.5,
    confidence_proxy_precomputed: bool = False,
) -> pd.DataFrame:
    feature_df = pd.DataFrame(index=model_output.abundance.index)
    feature_df["phi_local"] = local_residual_score(
        model_output.abundance.to_numpy(),
        neighbors,
        weights=weights,
        heterogeneity=heterogeneity,
    )
    if confidence_proxy_precomputed:
        feature_df["phi_uncertainty"] = uncertainty_score(model_output.uncertainty)
    else:
        feature_df["phi_uncertainty"] = confidence_proxy_score(
            model_output.abundance,
            model_output.uncertainty,
            uncertainty_weight=confidence_proxy_weight,
        )
    if stability_predictions is None:
        feature_df["phi_stability"] = 0.0
    else:
        feature_df["phi_stability"] = stability_score(stability_predictions)
    return feature_df
