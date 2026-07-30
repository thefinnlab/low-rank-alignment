import matplotlib.axes
import numpy as np
import pandas as pd

from src.utils.saturation_bar_summary import (
    build_auc_comparison_table,
    build_saturation_bar_table,
    plot_saturation_bars,
)


def test_auc_comparison_pairs_transfer_with_within_target(tmp_path):
    reference = tmp_path / "results" / "fmri" / "gpt2-xl_layer24" / "within_fmri_rrr"
    transfer = tmp_path / "results" / "fmri" / "gpt2-xl_layer24" / "cross_dataset_rrr"
    reference.mkdir(parents=True)
    transfer.mkdir(parents=True)
    (reference / "config.yml").write_text(
        "model_dir: gpt2-xl\nlayer: 24\nsource_mode: within_fmri\nfmri: {}\n"
    )
    (transfer / "config.yml").write_text(
        "model_dir: gpt2-xl\nlayer: 24\nsource_mode: cross_dataset\n"
        "source_rrr_unit_dir: results/gpt2-xl-batch_layer24_offset0/rrr\nfmri: {}\n"
    )
    summary = pd.DataFrame([
        {"run": str(reference), "method": "rrr", "layout": "fmri", "normalized_log_auc": 0.8},
        {"run": str(transfer), "method": "rrr", "layout": "fmri", "normalized_log_auc": 0.6},
    ])

    comparisons, skipped = build_auc_comparison_table(summary)

    assert skipped.empty
    assert len(comparisons) == 1
    assert np.isclose(comparisons.iloc[0]["cross_dataset_effectiveness"], 0.75)


def test_build_table_infers_path_metadata_and_percent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "results" / "gpt2-xl_layer24_offset0" / "rrr"
    run.mkdir(parents=True)
    summary = pd.DataFrame({
        "run": [str(run)],
        "method": ["rrr"],
        "layout": ["ieeg"],
        "saturation_rank": [16],
    })

    table, skipped = build_saturation_bar_table(summary)

    assert skipped.empty
    row = table.iloc[0]
    assert row["model"] == "gpt2-xl"
    assert row["layer"] == 24
    assert row["source_dataset"] == "podcast"
    assert row["target_dataset"] == "podcast"
    assert row["ambient_dimension"] == 1600
    assert row["saturation_percent"] == 1.0


def test_plot_draws_equal_width_hatched_pca_before_rrr(tmp_path, monkeypatch):
    calls = []
    original = matplotlib.axes.Axes.bar

    def recording_bar(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", recording_bar)
    frame = pd.DataFrame([
        {
            "method": "pca", "model": "gpt2-xl", "layer": 24,
            "saturation_percent": 4.0, "source_dataset": "fmri",
            "target_dataset": "fmri",
        },
        {
            "method": "rrr", "model": "gpt2-xl", "layer": 24,
            "saturation_percent": 1.0, "source_dataset": "fmri",
            "target_dataset": "fmri",
        },
    ])

    plot_saturation_bars(frame, tmp_path / "summary")

    assert calls[0]["hatch"] == "xxxx"
    assert calls[0]["width"] == calls[1]["width"]
    assert calls[0]["zorder"] < calls[1]["zorder"]
    assert calls[0].get("bottom", 0) == 0
    assert (tmp_path / "summary.png").exists()
    assert (tmp_path / "summary.svg").exists()
