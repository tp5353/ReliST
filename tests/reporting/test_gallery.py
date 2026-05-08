from __future__ import annotations

from pathlib import Path

import pandas as pd

from st_risk.reporting.gallery import (
    abundance_heatmap_table,
    GalleryEntry,
    build_abundance_heatmap_summary,
    build_dominant_celltype_summary,
    build_gallery_metadata_rows,
    build_top1_margin_summary,
    dominant_celltype_table,
    render_abundance_gallery_markdown,
    render_margin_gallery_markdown,
    render_dominant_gallery_markdown,
    render_gallery_markdown,
    repo_relative_link,
    top1_margin_table,
)


def test_repo_relative_link_uses_markdown_directory_relative_posix_path(tmp_path):
    start = tmp_path / "results" / "model_comparison" / "runs" / "gallery-run"
    target = tmp_path / "results" / "dlpfc_mvp" / "runs" / "v1" / "figures" / "discordant_sample_151508_spatial.png"

    assert repo_relative_link(target, start=start) == "../../../dlpfc_mvp/runs/v1/figures/discordant_sample_151508_spatial.png"


def test_render_gallery_markdown_contains_relative_figure_links(tmp_path):
    start = tmp_path / "results" / "model_comparison" / "runs" / "gallery-run"
    entry = GalleryEntry(
        model_key="cell2location",
        display_name="cell2location",
        sample_id="151508",
        run_id="run-a",
        figure_path=tmp_path / "results" / "dlpfc_mvp" / "runs" / "run-a" / "figures" / "discordant_sample_151508_spatial.png",
        rationale="core reference-axis example",
    )

    markdown = render_gallery_markdown([entry], sample_id="151508", start=start)

    assert "# Model Gallery 151508" in markdown
    assert "core reference-axis example" in markdown
    assert "(../../../dlpfc_mvp/runs/run-a/figures/discordant_sample_151508_spatial.png)" in markdown


def test_build_gallery_metadata_rows_are_serializable(tmp_path):
    start = tmp_path / "results" / "model_comparison" / "runs" / "gallery-run"
    entry = GalleryEntry(
        model_key="tangram",
        display_name="Tangram",
        sample_id="151508",
        run_id="run-b",
        figure_path=tmp_path / "results" / "dlpfc_tangram" / "runs" / "run-b" / "figures" / "discordant_sample_151508_spatial.png",
        rationale="marker auxiliary branch example",
    )

    rows = build_gallery_metadata_rows([entry], start=start)

    assert rows == [
        {
            "model_key": "tangram",
            "display_name": "Tangram",
            "sample_id": "151508",
            "run_id": "run-b",
            "figure_path": "../../../dlpfc_tangram/runs/run-b/figures/discordant_sample_151508_spatial.png",
            "rationale": "marker auxiliary branch example",
            "figure_kind": "discordant_sample_spatial",
        }
    ]


def test_build_dominant_celltype_summary_counts_fraction():
    table = pd.DataFrame(
        {
            "dominant_celltype": ["Astro", "Astro", "Excit_01", "OPC"],
            "x_spatial": [1, 2, 3, 4],
            "y_spatial": [5, 6, 7, 8],
            "dominant_value": [0.9, 0.8, 0.7, 0.6],
        }
    )

    summary = build_dominant_celltype_summary(
        table,
        model_key="cell2location",
        display_name="cell2location",
        sample_id="151508",
        top_n=2,
    )

    assert summary.to_dict("records") == [
        {
            "model_key": "cell2location",
            "display_name": "cell2location",
            "sample_id": "151508",
            "celltype": "Astro",
            "n_spots": 2,
            "fraction_of_sample": 0.5,
        },
        {
            "model_key": "cell2location",
            "display_name": "cell2location",
            "sample_id": "151508",
            "celltype": "Excit_01",
            "n_spots": 1,
            "fraction_of_sample": 0.25,
        },
    ]


def test_dominant_celltype_table_uses_sample_subset_and_argmax():
    abundance = pd.DataFrame(
        {
            "Astro": [0.8, 0.2, 0.4],
            "OPC": [0.1, 0.7, 0.3],
            "Excit_01": [0.1, 0.1, 0.9],
        },
        index=["spot1", "spot2", "spot3"],
    )
    risk_table = pd.DataFrame(
        {
            "sample_id": ["151508", "151507", "151508"],
            "x_spatial": [1.0, 2.0, 3.0],
            "y_spatial": [4.0, 5.0, 6.0],
        },
        index=["spot1", "spot2", "spot3"],
    )

    table = dominant_celltype_table(abundance, risk_table, sample_id="151508")

    assert table.index.tolist() == ["spot1", "spot3"]
    assert table["dominant_celltype"].tolist() == ["Astro", "Excit_01"]


def test_render_dominant_gallery_markdown_uses_local_figure_paths(tmp_path):
    start = tmp_path / "results" / "model_comparison" / "runs" / "gallery-run"
    entry = GalleryEntry(
        model_key="destvi",
        display_name="DestVI",
        sample_id="151508",
        run_id="run-c",
        figure_path=start / "figures" / "dominant_celltype" / "destvi.png",
        rationale="dominant-label boundary case view",
        figure_kind="dominant_celltype",
    )
    summary = pd.DataFrame(
        [
            {
                "model_key": "destvi",
                "display_name": "DestVI",
                "sample_id": "151508",
                "celltype": "Astro",
                "n_spots": 10,
                "fraction_of_sample": 0.5,
            }
        ]
    )

    markdown = render_dominant_gallery_markdown([entry], summary, sample_id="151508", start=start)

    assert "# Dominant Celltype Gallery 151508" in markdown
    assert "(figures/dominant_celltype/destvi.png)" in markdown
    assert "| DestVI | Astro | 10 | 0.500 |" in markdown


def test_top1_margin_table_uses_row_normalized_top1_minus_top2():
    abundance = pd.DataFrame(
        {
            "Astro": [8.0, 1.0],
            "OPC": [1.0, 4.0],
            "Excit_01": [1.0, 3.0],
        },
        index=["spot1", "spot2"],
    )
    risk_table = pd.DataFrame(
        {
            "sample_id": ["151508", "151508"],
            "x_spatial": [1.0, 2.0],
            "y_spatial": [3.0, 4.0],
        },
        index=["spot1", "spot2"],
    )

    table = top1_margin_table(abundance, risk_table, sample_id="151508")

    assert table["top1_label"].tolist() == ["Astro", "OPC"]
    assert table["top1_prop"].round(3).tolist() == [0.8, 0.5]
    assert table["top2_prop"].round(3).tolist() == [0.1, 0.375]
    assert table["top1_margin"].round(3).tolist() == [0.7, 0.125]


def test_build_top1_margin_summary_returns_single_row():
    table = pd.DataFrame({"top1_margin": [0.05, 0.2, 0.4]})

    summary = build_top1_margin_summary(
        table,
        model_key="rctd",
        display_name="RCTD",
        sample_id="151508",
        low_margin_threshold=0.1,
    )

    row = summary.iloc[0]
    assert row["model_key"] == "rctd"
    assert row["display_name"] == "RCTD"
    assert row["n_spots"] == 3
    assert round(float(row["mean_top1_margin"]), 3) == 0.217
    assert round(float(row["median_top1_margin"]), 3) == 0.2
    assert round(float(row["low_margin_fraction"]), 3) == 0.333


def test_render_top1_margin_gallery_markdown_uses_local_paths(tmp_path):
    start = tmp_path / "results" / "model_comparison" / "runs" / "gallery-run"
    entry = GalleryEntry(
        model_key="tangram",
        display_name="Tangram",
        sample_id="151508",
        run_id="run-d",
        figure_path=start / "figures" / "top1_margin" / "tangram.png",
        rationale="margin explains why dominant labels fragment",
        figure_kind="top1_margin",
    )
    summary = pd.DataFrame(
        [
            {
                "model_key": "tangram",
                "display_name": "Tangram",
                "sample_id": "151508",
                "n_spots": 10,
                "mean_top1_margin": 0.12,
                "median_top1_margin": 0.11,
                "p90_top1_margin": 0.3,
                "low_margin_fraction": 0.4,
            }
        ]
    )

    markdown = render_margin_gallery_markdown([entry], summary, sample_id="151508", start=start)

    assert "# Top1 Margin Gallery 151508" in markdown
    assert "(figures/top1_margin/tangram.png)" in markdown
    assert "| Tangram | 0.120 | 0.110 | 0.300 | 0.400 |" in markdown


def test_render_top1_margin_gallery_markdown_allows_custom_title_and_note(tmp_path):
    start = tmp_path / "results" / "model_comparison" / "runs" / "gallery-run"
    entry = GalleryEntry(
        model_key="rctd",
        display_name="RCTD",
        sample_id="151508",
        run_id="run-f",
        figure_path=start / "figures" / "top1_margin_clipped" / "rctd.png",
        rationale="shared clipped contrast view",
        figure_kind="top1_margin",
    )
    summary = pd.DataFrame(
        [
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "sample_id": "151508",
                "n_spots": 10,
                "mean_top1_margin": 0.25,
                "median_top1_margin": 0.21,
                "p90_top1_margin": 0.5,
                "low_margin_fraction": 0.2,
            }
        ]
    )

    markdown = render_margin_gallery_markdown(
        [entry],
        summary,
        sample_id="151508",
        start=start,
        title="Top1 Margin Gallery Clipped",
        display_value_note="row-normalized margin with display clipped to [0, 0.3]",
    )

    assert "# Top1 Margin Gallery Clipped 151508" in markdown
    assert "display clipped to [0, 0.3]" in markdown


def test_abundance_heatmap_table_adds_robust_scaled_values():
    abundance = pd.DataFrame(
        {"Astro": [0.0, 1.0, 2.0, 100.0]},
        index=["spot1", "spot2", "spot3", "spot4"],
    )
    risk_table = pd.DataFrame(
        {
            "sample_id": ["151508", "151508", "151508", "151508"],
            "x_spatial": [1.0, 2.0, 3.0, 4.0],
            "y_spatial": [5.0, 6.0, 7.0, 8.0],
        },
        index=["spot1", "spot2", "spot3", "spot4"],
    )

    table = abundance_heatmap_table(abundance, risk_table, sample_id="151508", celltype="Astro")

    assert table["abundance_raw"].tolist() == [0.0, 1.0, 2.0, 100.0]
    assert table["abundance_scaled"].between(0.0, 1.0).all()
    assert float(table["clip_q99"].iloc[0]) <= 100.0


def test_build_abundance_heatmap_summary_returns_single_row():
    table = pd.DataFrame(
        {
            "celltype": ["Astro", "Astro", "Astro"],
            "abundance_raw": [0.1, 0.2, 0.3],
            "clip_q01": [0.1, 0.1, 0.1],
            "clip_q99": [0.3, 0.3, 0.3],
        }
    )

    summary = build_abundance_heatmap_summary(
        table,
        model_key="cell2location",
        display_name="cell2location",
        sample_id="151508",
    )

    row = summary.iloc[0]
    assert row["celltype"] == "Astro"
    assert round(float(row["raw_mean"]), 3) == 0.2
    assert round(float(row["raw_p90"]), 3) == 0.28


def test_render_abundance_gallery_markdown_uses_local_paths(tmp_path):
    start = tmp_path / "results" / "model_comparison" / "runs" / "gallery-run"
    entry = GalleryEntry(
        model_key="cell2location::Astro",
        display_name="cell2location",
        sample_id="151508",
        run_id="run-e",
        figure_path=start / "figures" / "abundance_heatmaps" / "cell2location_Astro.png",
        rationale="Astro pattern contrast view",
        figure_kind="abundance_heatmap",
    )
    summary = pd.DataFrame(
        [
            {
                "model_key": "cell2location",
                "display_name": "cell2location",
                "sample_id": "151508",
                "celltype": "Astro",
                "n_spots": 10,
                "raw_mean": 0.12,
                "raw_median": 0.10,
                "raw_p90": 0.3,
                "clip_q01": 0.01,
                "clip_q99": 0.4,
            }
        ]
    )

    markdown = render_abundance_gallery_markdown([entry], summary, sample_id="151508", start=start)

    assert "# Abundance Heatmap Gallery 151508" in markdown
    assert "(figures/abundance_heatmaps/cell2location_Astro.png)" in markdown
    assert "| cell2location | Astro | 0.1200 | 0.1000 | 0.3000 | 0.0100 | 0.4000 |" in markdown
