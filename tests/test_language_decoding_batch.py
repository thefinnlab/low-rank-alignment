from pathlib import Path

from scripts.rrr.run_internal_language_decoding_batch import build_layer_configs
import numpy as np
import pandas as pd

from scripts.fmri.run_internal_language_decoding_batch import _layer_configs, _session_cv
from scripts.fmri.bootstrap_language_decoding import (
    _apply_rankwise_fdr,
    _load_fold_pair,
    hierarchical_block_bootstrap,
    paired_block_bootstrap,
)
from scripts.rrr.plot_rank_saturation import METHOD_COLORS
from src.utils.plot_helpers import RRR_PCA_COLORS


def test_internal_batch_expands_model_layers_and_result_templates():
    configs = build_layer_configs(
        "configs/language_decoding/internal/cars-2.yml", "gpt2"
    )
    assert [config.encoding.per_model_config["layer"] for config in configs] == [
        0,
        3,
        6,
        9,
        12,
    ]
    assert configs[1].rrr_run_path == (
        "results/gpt2_layer3_offset0/movie-cars-2/rrr"
    )
    assert configs[1].pca_run_path == (
        "results/gpt2_layer3_offset0/movie-cars-2/pca"
    )
    assert configs[1].encoding.dataset == "brain_treebank"
    assert "iu_position" not in configs[1].tasks
    assert "surprisal" in configs[1].tasks


def test_internal_batch_podcast_includes_requested_iu_and_norm_features():
    configs = build_layer_configs(
        "configs/language_decoding/internal/podcast.yml", "gpt2-xl"
    )
    assert [config.encoding.per_model_config["layer"] for config in configs] == [
        0,
        12,
        24,
        36,
        48,
    ]
    assert Path(configs[0].task_data["language_features"]).name == "tokens.tsv"
    assert {"iu_position", "iu_onset", "iu_end", "concreteness", "surprisal"} <= set(
        configs[0].tasks
    )


def test_fmri_batch_uses_only_heldout_story_available_features():
    manifest, configs = _layer_configs(
        "configs/language_decoding/internal/fmri.yml", "gpt2"
    )
    assert [config.layer for config in configs] == [0, 3, 6, 9, 12]
    assert configs[2].output_dir == "results/fmri/gpt2_layer6/within_fmri_rrr"
    assert {"surprisal", "concreteness", "part_of_speech", "dependency_depth"} <= set(
        manifest["tasks"]
    )
    assert {"iu_position", "iu_onset", "iu_end", "iu_length"} <= set(
        manifest["tasks"]
    )


def test_fmri_decoder_inner_cv_holds_out_complete_sessions_after_missing_mask():
    stories = ["a", "b", "c"]
    tables = {story: pd.DataFrame(index=range(2)) for story in stories}
    groups = {1: ["a", "b"], 2: ["c"]}
    valid = np.array([True, False, True, True, False, True])
    splits = _session_cv(stories, tables, groups, valid)
    assert [tuple(values.tolist()) for values in splits[0]] == [(3,), (0, 1, 2)]
    assert [tuple(values.tolist()) for values in splits[1]] == [(0, 1, 2), (3,)]


def test_paired_sentence_bootstrap_preserves_method_pairing():
    y_true = np.arange(12, dtype=float)
    pair = {
        "y_true": y_true,
        "rrr": y_true.copy(),
        "pca": y_true[::-1],
        "block_id": np.repeat(["s1", "s2", "s3"], 4),
    }
    pca, rrr, delta, boot = paired_block_bootstrap(
        pair, "continuous", 50, np.random.default_rng(2)
    )
    assert rrr == 1
    assert pca == -1
    assert delta == 2
    assert np.nanmin(boot) > 0


def test_hierarchical_bootstrap_gives_models_equal_observed_weight():
    blocks = np.repeat(["s1", "s2"], 3)

    def pair(delta):
        y_true = np.arange(6, dtype=float)
        return {
            "y_true": y_true,
            "rrr": y_true if delta > 0 else y_true[::-1],
            "pca": y_true[::-1] if delta > 0 else y_true,
            "block_id": blocks,
        }

    pca, rrr, observed, boot = hierarchical_block_bootstrap(
        {"many_layers": [pair(1), pair(1), pair(1)], "one_layer": [pair(-1)]},
        "continuous",
        20,
        np.random.default_rng(3),
    )
    assert pca == rrr
    assert observed == 0
    assert len(boot) == 20


def test_bootstrap_fdr_is_applied_separately_per_rank():
    frame = pd.DataFrame({
        "rank": [1, 1, 2, 2],
        "task": ["a", "b", "a", "b"],
        "delta": [1.0, 1.0, 1.0, -1.0],
        "p_value": [0.01, 0.5, 0.04, 0.01],
    })
    result = _apply_rankwise_fdr(frame, 0.05, ["rank"])
    np.testing.assert_allclose(result.q_value, [0.02, 0.5, 0.04, 0.02])
    assert result.significant_rrr_gt_pca.tolist() == [True, False, True, False]


def test_bootstrap_and_rank_saturation_share_rrr_pca_colors():
    assert METHOD_COLORS is RRR_PCA_COLORS
    assert RRR_PCA_COLORS == {"rrr": "#74cee5", "pca": "#56a89c"}


def test_generic_bootstrap_pools_and_orders_paired_outer_folds(tmp_path):
    for method, offset in (("rrr", 0.1), ("pca", -0.1)):
        for fold, rows in ((0, np.array([2, 3])), (1, np.array([0, 1]))):
            folder = tmp_path / "tasks" / "surprisal" / method / "k4" / f"fold_{fold}"
            folder.mkdir(parents=True)
            np.savez_compressed(
                folder / "heldout_predictions.npz",
                y_true=rows.astype(float),
                y_pred=rows.astype(float) + offset,
                row_index=rows,
                block_id=np.asarray([f"sentence-{row // 2}" for row in rows], dtype=str),
            )
    pair = _load_fold_pair(tmp_path, "surprisal", 4)
    np.testing.assert_array_equal(pair["row_index"], np.arange(4))
    np.testing.assert_allclose(pair["rrr"], np.arange(4) + 0.1)
    np.testing.assert_allclose(pair["pca"], np.arange(4) - 0.1)
    np.testing.assert_array_equal(
        pair["block_id"], ["sentence-0", "sentence-0", "sentence-1", "sentence-1"]
    )
