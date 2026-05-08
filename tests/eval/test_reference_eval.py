import numpy as np
import pandas as pd

from st_risk.eval.reference_eval import (
    marker_subset_discordance_mean,
    random_subset_markers,
    reference_signature_residual_proxy,
    reference_subsampling_instability,
)


def test_random_subset_markers_respects_fraction_and_minimum():
    markers = {
        "A": ["g1", "g2", "g3", "g4"],
        "B": ["g5", "g6", "g7", "g8"],
    }
    rng = np.random.default_rng(0)
    subset = random_subset_markers(markers, fraction=0.5, rng=rng, min_genes=2)
    assert set(subset) == {"A", "B"}
    assert len(subset["A"]) >= 2
    assert len(subset["B"]) >= 2


def test_reference_subsampling_instability_returns_nonnegative_series():
    abundance = pd.DataFrame(
        {
            "A": [0.9, 0.2, 0.7, 0.3],
            "B": [0.1, 0.8, 0.3, 0.7],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    expression = pd.DataFrame(
        {
            "g1": [10.0, 1.0, 8.0, 2.0],
            "g2": [9.0, 2.0, 7.0, 3.0],
            "g3": [8.0, 3.0, 6.0, 4.0],
            "g4": [7.0, 4.0, 5.0, 5.0],
            "g5": [1.0, 10.0, 2.0, 9.0],
            "g6": [2.0, 9.0, 3.0, 8.0],
            "g7": [3.0, 8.0, 4.0, 7.0],
            "g8": [4.0, 7.0, 5.0, 6.0],
        },
        index=abundance.index,
    )
    markers = {
        "A": ["g1", "g2", "g3", "g4"],
        "B": ["g5", "g6", "g7", "g8"],
    }
    result = reference_subsampling_instability(
        abundance,
        expression,
        markers,
        repeats=6,
        subset_fraction=0.5,
        random_state=0,
    )
    assert list(result.index) == list(abundance.index)
    assert (result >= 0.0).all()


def test_marker_subset_discordance_mean_returns_bounded_series():
    abundance = pd.DataFrame(
        {
            "A": [0.9, 0.2, 0.7, 0.3],
            "B": [0.1, 0.8, 0.3, 0.7],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    expression = pd.DataFrame(
        {
            "g1": [10.0, 1.0, 8.0, 2.0],
            "g2": [9.0, 2.0, 7.0, 3.0],
            "g3": [8.0, 3.0, 6.0, 4.0],
            "g4": [7.0, 4.0, 5.0, 5.0],
            "g5": [1.0, 10.0, 2.0, 9.0],
            "g6": [2.0, 9.0, 3.0, 8.0],
            "g7": [3.0, 8.0, 4.0, 7.0],
            "g8": [4.0, 7.0, 5.0, 6.0],
        },
        index=abundance.index,
    )
    markers = {
        "A": ["g1", "g2", "g3", "g4"],
        "B": ["g5", "g6", "g7", "g8"],
    }
    result = marker_subset_discordance_mean(
        abundance,
        expression,
        markers,
        repeats=6,
        subset_fraction=0.5,
        random_state=0,
    )
    assert list(result.index) == list(abundance.index)
    assert (result >= 0.0).all()
    assert (result <= 1.0).all()


def test_reference_signature_residual_proxy_prefers_matching_profiles():
    abundance = pd.DataFrame(
        {
            "A": [0.9, 0.1, 0.6, 0.4],
            "B": [0.1, 0.9, 0.4, 0.6],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    signatures = pd.DataFrame(
        {
            "A": [10.0, 8.0, 2.0, 1.0],
            "B": [1.0, 2.0, 8.0, 10.0],
        },
        index=["g1", "g2", "g3", "g4"],
    )
    expression = pd.DataFrame(
        {
            "g1": [9.0, 2.0, 7.0, 4.0],
            "g2": [8.0, 3.0, 6.0, 5.0],
            "g3": [2.0, 8.0, 4.0, 6.0],
            "g4": [1.0, 9.0, 5.0, 7.0],
        },
        index=abundance.index,
    )
    result = reference_signature_residual_proxy(
        abundance,
        expression,
        signatures,
        genes=["g1", "g2", "g3", "g4"],
    )
    assert list(result.index) == list(abundance.index)
    assert (result >= 0.0).all()
    assert (result <= 1.0).all()
    assert result.loc["s1"] < result.loc["s3"]
