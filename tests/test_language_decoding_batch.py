from pathlib import Path

from scripts.rrr.run_internal_language_decoding_batch import build_layer_configs
import numpy as np
import pandas as pd

from scripts.fmri.run_internal_language_decoding_batch import _layer_configs, _session_cv


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
