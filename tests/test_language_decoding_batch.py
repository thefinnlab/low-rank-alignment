from pathlib import Path

from scripts.rrr.run_internal_language_decoding_batch import build_layer_configs
import numpy as np
import pandas as pd

from scripts.fmri.run_internal_language_decoding_batch import _layer_configs, _session_cv
from scripts.fmri.bootstrap_language_decoding import (
    _apply_rankwise_fdr,
    _load_fold_pair,
    cross_dataset_mean_scores,
    hierarchical_block_bootstrap,
    paired_block_bootstrap,
    _relative_preference as _bootstrap_relative_preference,
)
from scripts.rrr.plot_rank_saturation import METHOD_COLORS
from src.utils.plot_helpers import RRR_PCA_COLORS
from scripts.plot_mean_language_decoding import (
    _relative_preference,
    _relative_preferences,
)
from scripts.plot_entropy_architecture_preference import (
    summarize_entropy_architectures,
)
from scripts.plot_surprisal_layer_group_preference import summarize_layer_groups


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
    assert "self_surprisal" in configs[1].tasks
    assert "self_entropy" in configs[1].tasks
    assert configs[1].task_data["self_surprisal"] == (
        "processed_data/language_features/self_surprisal/gpt2/"
        "brain_treebank/cars-2/tokens.tsv"
    )
    assert configs[1].task_data["self_entropy"] == configs[1].task_data["self_surprisal"]


def test_internal_batch_podcast_includes_requested_iu_and_norm_features():
    configs = build_layer_configs(
        "configs/language_decoding/internal/podcast.yml", "gpt2-xl-batch"
    )
    assert [config.encoding.per_model_config["layer"] for config in configs] == [
        0,
        12,
        24,
        36,
        48,
    ]
    assert Path(configs[0].task_data["language_features"]).name == "tokens.tsv"
    assert configs[0].task_data["self_surprisal"].endswith(
        "gpt2-xl-batch/podcast/podcast/tokens.tsv"
    )
    assert {"iu_position", "iu_onset", "iu_end", "concreteness", "surprisal"} <= set(
        configs[0].tasks
    )


def test_internal_batch_can_select_a_task_subset():
    configs = build_layer_configs(
        "configs/language_decoding/internal/podcast.yml",
        "gpt2",
        ["surprisal", "self_surprisal"],
    )
    assert configs[0].tasks == ["surprisal", "self_surprisal"]


def test_internal_batch_can_limit_maximum_rank():
    configs = build_layer_configs(
        "configs/language_decoding/internal/podcast.yml", "gpt2-xl-batch", max_rank=256
    )
    assert configs[0].rank_grid
    assert max(configs[0].rank_grid) == 256


def test_internal_batch_can_use_separate_run_identity():
    configs = build_layer_configs(
        "configs/language_decoding/internal/podcast.yml",
        "gpt2",
        run_name="predictability-context-quick",
    )
    assert configs[0].source_name.endswith("__predictability-context-quick")


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
    assert "self_surprisal" in manifest["tasks"]
    assert "self_entropy" in manifest["tasks"]


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

    pca, rrr, observed, boot, boot_pca, boot_rrr = hierarchical_block_bootstrap(
        {"many_layers": [pair(1), pair(1), pair(1)], "one_layer": [pair(-1)]},
        "continuous", 20, np.random.default_rng(3),
        return_method_bootstrap=True,
    )
    np.testing.assert_allclose(boot, boot_rrr - boot_pca)


def test_cross_dataset_mean_gives_datasets_equal_observed_weight():
    blocks = np.repeat(["s1", "s2"], 3)

    def record(rrr_wins):
        y_true = np.arange(6, dtype=float)
        return {"pair": {
            "y_true": y_true,
            "rrr": y_true if rrr_wins else y_true[::-1],
            "pca": y_true[::-1] if rrr_wins else y_true,
            "block_id": blocks,
        }}

    datasets = {
        "a": {"model": [record(True), record(True), record(True)]},
        "b": {"model": [record(False)]},
    }
    scores, pca, rrr, observed = cross_dataset_mean_scores(datasets, "continuous")
    assert scores["a"]["delta"] == 2
    assert scores["b"]["delta"] == -2
    assert pca == rrr
    assert observed == 0


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


def test_relative_preference_ratio_and_low_performance_threshold():
    assert _relative_preference(0.2, 0.1, 0.05) == 1
    assert _relative_preference(0.1, 0.2, 0.05) == -1
    assert _relative_preference(0.04, 0.03, 0.05) == 0
    np.testing.assert_allclose(
        _bootstrap_relative_preference([0.2, 0.1, 0.04], [0.1, 0.2, 0.03], 0.05),
        [1, -1, 0],
    )
    assert _bootstrap_relative_preference(0.2, 0.1, 0.05) == 1


def test_relative_preferences_transform_averaged_scores():
    scores = {
        "podcast": {"rrr": 0.2, "pca": 0.1},
        "fmri": {"rrr": 0.1, "pca": 0.2},
    }
    result = _relative_preferences(scores, mean_rrr=0.15, mean_pca=0.15, threshold=0.05)
    assert result == {"podcast": 1, "fmri": -1, "all": 0}


def test_entropy_architecture_summary_excludes_random_and_weights_models_equally():
    rows = []
    model_scores = {
        "gpt2": (0.1, 0.2),
        "gpt2-xl": (0.2, 0.4),
        "gpt-j-6b": (0.3, 0.6),
        "llama3-8b": (0.4, 0.8),
        "roberta-large": (0.4, 0.2),
        "random-gpt2-xl": (0.01, 1.0),
    }
    for model, (pca, rrr) in model_scores.items():
        for layer in (0, 1):
            rows.append({
                "model": model, "layer": layer, "rank": 4,
                "task": "self_entropy", "pca_score": pca, "rrr_score": rrr,
            })
    result = summarize_entropy_architectures(pd.DataFrame(rows), threshold=0.05)
    causal = result.set_index("architecture").loc["causal"]
    roberta = result.set_index("architecture").loc["roberta"]
    assert causal["n_models"] == 4
    assert causal["n_layers"] == 8
    assert causal["pca_score"] == 0.25
    assert causal["rrr_score"] == 0.5
    assert causal["relative_preference"] == 1
    assert roberta["relative_preference"] == -1


def test_smooth_layer_group_preference_preserves_delta_direction():
    rows = []
    for layer, pca, rrr in ((0, 0.1, 0.2), (3, 0.1, 0.2), (6, 0.1, 0.2),
                            (9, 0.076653, -0.020039), (12, 0.076653, -0.020039)):
        rows.append({
            "model": "gpt2", "layer": layer, "rank": 1,
            "task": "self_surprisal", "pca_score": pca, "rrr_score": rrr,
        })
    result = summarize_layer_groups(
        pd.DataFrame(rows), threshold=0.05,
        grouping="first-three-vs-final-two", normalization_tau=0.05,
    ).set_index("layer_group")
    assert result.loc["first three depths", "smooth_relative_preference"] > 0
    assert result.loc["final two depths", "smooth_relative_preference"] < 0
    assert np.isfinite(result["smooth_relative_preference"]).all()


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
