import pandas as pd

from src.utils.language_decoding_saturation_summary import (
    build_decoding_saturation_table,
    plot_decoding_saturation_bars,
)


def test_table_uses_exact_rrr_saturation_and_excludes_glove(tmp_path):
    run = tmp_path / "gpt2_layer6" / "rrr"
    rows = []
    for task in ("surprisal", "glove"):
        for rank, gap in ((4, 0.1), (8, 0.3)):
            for fold in (0, 1):
                rows.append({
                    "source_id": "source", "source_dataset": "podcast",
                    "architecture": "gpt2", "model_dir": "gpt2", "layer": 6,
                    "control": False, "task": task,
                    "kind": "continuous" if task != "glove" else "glove",
                    "k": rank, "fold": fold, "rrr": 0.5 + gap,
                    "pca": 0.5, "rrr_minus_pca": gap,
                })
    table, skipped = build_decoding_saturation_table(
        pd.DataFrame(rows),
        pd.DataFrame([{"source_id": "source", "rrr_dir": str(run)}]),
        pd.DataFrame([{"run": str(run), "method": "rrr", "saturation_rank": 8}]),
    )
    assert skipped.empty
    assert table.task.tolist() == ["surprisal"]
    assert table.rrr_minus_pca.tolist() == [0.3]
    assert table.depth_percent.tolist() == [50.0]
    assert table.n_folds.tolist() == [2]


def test_missing_exact_rank_is_reported(tmp_path):
    paired = pd.DataFrame([{
        "source_id": "source", "source_dataset": "podcast", "architecture": "gpt2",
        "model_dir": "gpt2", "layer": 6, "control": False, "task": "surprisal",
        "kind": "continuous", "k": 4, "fold": 0, "rrr": .2, "pca": .1,
        "rrr_minus_pca": .1,
    }])
    run = tmp_path / "rrr"
    table, skipped = build_decoding_saturation_table(
        paired,
        pd.DataFrame([{"source_id": "source", "rrr_dir": str(run)}]),
        pd.DataFrame([{"run": str(run), "method": "rrr", "saturation_rank": 8}]),
    )
    assert table.empty
    assert skipped.reason.tolist() == ["RRR saturation rank was not decoded"]


def test_rank_specific_factor_child_matches_parent_saturation_run(tmp_path):
    paired = pd.DataFrame([{
        "source_id": "source", "source_dataset": "fmri", "architecture": "gpt2",
        "model_dir": "gpt2", "layer": 6, "control": False, "task": "surprisal",
        "kind": "continuous", "k": 8, "fold": 0, "rrr": .2, "pca": .1,
        "rrr_minus_pca": .1,
    }])
    run = tmp_path / "within_fmri_rrr"
    table, skipped = build_decoding_saturation_table(
        paired,
        pd.DataFrame([{"source_id": "source", "rrr_dir": str(run / "rrr")}]),
        pd.DataFrame([{"run": str(run), "method": "rrr", "saturation_rank": 8}]),
    )
    assert skipped.empty
    assert table.rrr_minus_pca.tolist() == [.1]


def test_plot_writes_png_and_svg(tmp_path):
    frame = pd.DataFrame([
        {"task": "surprisal", "model": "gpt2", "depth_percent": 0.0, "rrr_minus_pca": .1, "sem": .01},
        {"task": "surprisal", "model": "gpt2-xl", "depth_percent": 0.0, "rrr_minus_pca": .2, "sem": .02},
    ])
    plot_decoding_saturation_bars(frame, tmp_path / "plot")
    assert (tmp_path / "plot.png").exists()
    assert (tmp_path / "plot.svg").exists()
