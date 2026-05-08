from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def select_signature_markers(
    signatures: pd.DataFrame,
    *,
    top_k: int = 25,
    min_positive_markers: int = 10,
    exclude_prefixes: Iterable[str] = ("drop_",),
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    prefixes = tuple(exclude_prefixes)
    markers: dict[str, list[str]] = {}
    rows: list[dict[str, float | str]] = []

    for celltype in signatures.columns:
        if prefixes and str(celltype).startswith(prefixes):
            continue
        other_cols = [col for col in signatures.columns if col != celltype]
        if not other_cols:
            continue
        specificity = signatures[celltype] - signatures[other_cols].max(axis=1)
        positive = specificity.loc[specificity > 0].sort_values(ascending=False)
        if positive.shape[0] < min_positive_markers:
            continue
        selected = positive.head(top_k)
        markers[str(celltype)] = selected.index.astype(str).tolist()
        for rank, (gene, score) in enumerate(selected.items(), start=1):
            rows.append(
                {
                    "celltype": str(celltype),
                    "gene": str(gene),
                    "specificity": float(score),
                    "rank": float(rank),
                }
            )

    marker_table = pd.DataFrame(rows).sort_values(["celltype", "rank"]).reset_index(drop=True)
    return markers, marker_table


def subset_markers(
    markers: dict[str, list[str]],
    *,
    mode: str = "all",
) -> dict[str, list[str]]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "all":
        return {celltype: list(genes) for celltype, genes in markers.items()}

    subsets: dict[str, list[str]] = {}
    for celltype, genes in markers.items():
        if normalized_mode == "odd":
            selected = [gene for idx, gene in enumerate(genes) if idx % 2 == 0]
        elif normalized_mode == "even":
            selected = [gene for idx, gene in enumerate(genes) if idx % 2 == 1]
        elif normalized_mode == "top_half":
            selected = genes[: max(1, len(genes) // 2)]
        elif normalized_mode == "bottom_half":
            selected = genes[len(genes) // 2 :]
        else:
            raise ValueError(f"Unsupported marker subset mode: {mode}")
        if selected:
            subsets[celltype] = selected
    return subsets


def random_subset_markers(
    markers: dict[str, list[str]],
    *,
    fraction: float = 0.5,
    rng: np.random.Generator,
    min_genes: int = 2,
) -> dict[str, list[str]]:
    probability = float(np.clip(fraction, 0.0, 1.0))
    subsets: dict[str, list[str]] = {}
    for celltype, genes in markers.items():
        if not genes:
            continue
        n_select = max(min_genes, int(np.ceil(len(genes) * probability)))
        n_select = min(len(genes), n_select)
        if n_select <= 0:
            continue
        chosen = rng.choice(np.asarray(genes, dtype=object), size=n_select, replace=False)
        subsets[celltype] = [str(gene) for gene in chosen.tolist()]
    return subsets


def compute_reference_marker_scores(
    expression: pd.DataFrame,
    markers: dict[str, list[str]],
) -> pd.DataFrame:
    rows = {}
    for celltype, genes in markers.items():
        present = [gene for gene in genes if gene in expression.columns]
        if not present:
            continue
        rows[celltype] = expression[present].mean(axis=1)
    if not rows:
        return pd.DataFrame(index=expression.index)
    return pd.DataFrame(rows, index=expression.index)


def reference_marker_discordance_proxy(
    abundance: pd.DataFrame,
    marker_scores: pd.DataFrame,
) -> pd.Series:
    shared_celltypes = [celltype for celltype in abundance.columns if celltype in marker_scores.columns]
    if len(shared_celltypes) < 2:
        raise ValueError("At least two shared cell types are required to compute marker discordance.")

    abundance_values = abundance[shared_celltypes].to_numpy(dtype=float)
    row_sums = abundance_values.sum(axis=1, keepdims=True)
    safe_sums = np.where(np.isclose(row_sums, 0.0), 1.0, row_sums)
    abundance_values = abundance_values / safe_sums

    marker_values = marker_scores.loc[abundance.index, shared_celltypes].to_numpy(dtype=float)

    abundance_centered = abundance_values - abundance_values.mean(axis=1, keepdims=True)
    marker_centered = marker_values - marker_values.mean(axis=1, keepdims=True)
    numerator = (abundance_centered * marker_centered).sum(axis=1)
    denominator = np.sqrt((abundance_centered**2).sum(axis=1) * (marker_centered**2).sum(axis=1))
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    discordance = (1.0 - correlation) / 2.0
    return pd.Series(discordance, index=abundance.index, name="snrna_marker_discordance")


def reference_signature_residual_proxy(
    abundance: pd.DataFrame,
    expression: pd.DataFrame,
    signatures: pd.DataFrame,
    *,
    genes: Iterable[str] | None = None,
) -> pd.Series:
    shared_celltypes = [celltype for celltype in abundance.columns if celltype in signatures.columns]
    if len(shared_celltypes) < 2:
        raise ValueError("At least two shared cell types are required to compute signature residual.")

    if genes is None:
        shared_genes = [gene for gene in signatures.index.astype(str) if gene in expression.columns]
    else:
        shared_genes = [str(gene) for gene in genes if str(gene) in expression.columns and str(gene) in signatures.index]
    if len(shared_genes) < 2:
        raise ValueError("At least two shared genes are required to compute signature residual.")

    abundance_values = abundance[shared_celltypes].to_numpy(dtype=float)
    row_sums = abundance_values.sum(axis=1, keepdims=True)
    safe_sums = np.where(np.isclose(row_sums, 0.0), 1.0, row_sums)
    abundance_values = abundance_values / safe_sums

    signature_values = signatures.loc[shared_genes, shared_celltypes].to_numpy(dtype=float)
    predicted = abundance_values @ signature_values.T
    observed = expression.loc[abundance.index, shared_genes].to_numpy(dtype=float)

    predicted_centered = predicted - predicted.mean(axis=1, keepdims=True)
    observed_centered = observed - observed.mean(axis=1, keepdims=True)
    numerator = (predicted_centered * observed_centered).sum(axis=1)
    denominator = np.sqrt((predicted_centered**2).sum(axis=1) * (observed_centered**2).sum(axis=1))
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    residual = (1.0 - correlation) / 2.0
    return pd.Series(residual, index=abundance.index, name="snrna_signature_residual")


def reference_subsampling_instability(
    abundance: pd.DataFrame,
    expression: pd.DataFrame,
    markers: dict[str, list[str]],
    *,
    repeats: int = 8,
    subset_fraction: float = 0.5,
    random_state: int = 0,
) -> pd.Series:
    if repeats < 2:
        raise ValueError("reference_subsampling_instability requires repeats >= 2")

    rng = np.random.default_rng(random_state)
    draws: list[np.ndarray] = []
    for _ in range(repeats):
        subset = random_subset_markers(markers, fraction=subset_fraction, rng=rng)
        marker_scores = compute_reference_marker_scores(expression, subset)
        if marker_scores.shape[1] < 2:
            continue
        discordance = reference_marker_discordance_proxy(abundance, marker_scores)
        draws.append(discordance.to_numpy(dtype=float))

    if len(draws) < 2:
        raise ValueError("reference_subsampling_instability could not generate at least two valid marker subsets")

    stacked = np.stack(draws, axis=0)
    instability = stacked.std(axis=0, ddof=1 if stacked.shape[0] > 1 else 0)
    return pd.Series(instability, index=abundance.index, name="phi_reference")


def marker_subset_discordance_mean(
    abundance: pd.DataFrame,
    expression: pd.DataFrame,
    markers: dict[str, list[str]],
    *,
    repeats: int = 8,
    subset_fraction: float = 0.5,
    random_state: int = 0,
) -> pd.Series:
    if repeats < 1:
        raise ValueError("marker_subset_discordance_mean requires repeats >= 1")

    rng = np.random.default_rng(random_state)
    draws: list[np.ndarray] = []
    for _ in range(repeats):
        subset = random_subset_markers(markers, fraction=subset_fraction, rng=rng)
        marker_scores = compute_reference_marker_scores(expression, subset)
        if marker_scores.shape[1] < 2:
            continue
        discordance = reference_marker_discordance_proxy(abundance, marker_scores)
        draws.append(discordance.to_numpy(dtype=float))

    if not draws:
        raise ValueError("marker_subset_discordance_mean could not generate a valid marker subset")

    stacked = np.stack(draws, axis=0)
    mean_discordance = stacked.mean(axis=0)
    return pd.Series(mean_discordance, index=abundance.index, name="phi_reference")


def marker_consistency_by_strategy_summary(
    abundance: pd.DataFrame,
    marker_scores: pd.DataFrame,
    obs: pd.DataFrame,
    strategy_masks: pd.DataFrame,
    *,
    celltypes: Iterable[str],
    sample_col: str = "sample_id",
) -> pd.DataFrame:
    abundance = abundance.loc[obs.index]
    marker_scores = marker_scores.loc[obs.index]
    strategy_masks = strategy_masks.loc[obs.index]

    rows: list[dict[str, float | str]] = []
    selected_celltypes = [str(celltype) for celltype in celltypes if celltype in abundance.columns and celltype in marker_scores.columns]
    for sample_id, sample_obs in obs.groupby(sample_col, sort=True):
        sample_abundance = abundance.loc[sample_obs.index]
        sample_markers = marker_scores.loc[sample_obs.index]
        sample_masks = strategy_masks.loc[sample_obs.index]
        for strategy_name, keep_mask in sample_masks.items():
            kept_index = sample_obs.index[keep_mask.astype(bool)]
            if kept_index.empty:
                continue
            kept_abundance = sample_abundance.loc[kept_index]
            kept_markers = sample_markers.loc[kept_index]
            for celltype in selected_celltypes:
                abundance_values = kept_abundance[celltype]
                marker_values = kept_markers[celltype]
                if abundance_values.nunique(dropna=True) < 2 or marker_values.nunique(dropna=True) < 2:
                    continue
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "strategy_name": str(strategy_name),
                        "celltype": celltype,
                        "n_spots": float(len(kept_index)),
                        "mean_abundance": float(abundance_values.mean()),
                        "mean_marker_score": float(marker_values.mean()),
                        "abundance_marker_corr": float(abundance_values.corr(marker_values)),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["celltype", "strategy_name", "sample_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
