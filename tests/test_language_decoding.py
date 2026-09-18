import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.core.config import LanguageDecodingConfig, LanguageDecodingMode
from scripts.run_language_decoding import Artifact as DecodingArtifact, run_config
from scripts.run_rrr import Artifact as RRRArtifact
from src.utils.shared.config import dict_to_dataclass
from src.utils.language_interpret.decoding import (
    _configure_decoder_backend,
    _decoder_metrics,
    _fit_decoder,
    _pool_speaker_labels,
    _matched_score_correlation,
    _run_subspace_decoding,
    _run_task_decoding,
    _prepare_multiview_subspaces,
    _prepare_single_view_subspaces,
    _weight_payload,
    compose_multiview_factor,
    plot_language_decoding,
    resolve_rrr_unit_dir,
    resolve_subspace_unit_dir,
    summarize_language_decoding,
    summarize_prediction_agreement,
)
from src.utils.language_interpret.tasks import TASK_REGISTRY, PodcastTaskContext, get_language_task
from src.utils.language_interpret.fmri_decoding import (
    _categorical_metrics,
    _prepare_rank_design,
    _stable_classifier_rows,
    _word_session_cv,
    build_word_targets,
    build_tr_targets,
)
from src.utils.shared.fmri import AlignedFMRIStimuli


def test_speaker_labels_pool_speakers_missing_from_outer_partitions():
    target = np.array([0, 1, 2, np.nan, 0, 1, 3, np.nan])
    folds = [(np.arange(4, 8), np.arange(4)),
             (np.arange(4), np.arange(4, 8))]
    mapped, metadata = _pool_speaker_labels(target, folds)
    np.testing.assert_equal(mapped, [0, 1, 4, np.nan, 0, 1, 4, np.nan])
    assert metadata["retained_class_ids"] == [0, 1]
    assert metadata["pooled_class_ids"] == [2, 3]
    assert metadata["other_class_id"] == 4
    np.testing.assert_equal(target, [0, 1, 2, np.nan, 0, 1, 3, np.nan])


def test_categorical_decoder_uses_all_available_cpus():
    config = LanguageDecodingConfig(rrr_run_path="unused", rank_grid=[2])
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.tile([0, 1], 6)

    pipeline = _fit_decoder(X, y, "categorical", config)

    assert pipeline[-1].n_jobs == -1


def test_speaker_labels_require_presence_in_training_as_well_as_test():
    target = np.array([0., 1., 0., 1.])
    mapped, metadata = _pool_speaker_labels(
        target, [(np.array([0, 2]), np.array([1, 3])),
                 (np.arange(4), np.arange(4))]
    )
    np.testing.assert_equal(mapped, [2, 2, 2, 2])
    assert metadata["retained_class_ids"] == []


def test_speaker_labels_do_not_add_other_when_all_speakers_are_shared():
    target = np.array([0., 1., 0., 1., np.nan])
    mapped, metadata = _pool_speaker_labels(
        target, [(np.array([0, 1]), np.array([2, 3, 4]))]
    )
    np.testing.assert_equal(mapped, target)
    assert metadata["other_class_id"] is None


def _task_context(tmp_path):
    word_df = pd.DataFrame(
        {
            "word_idx": np.arange(6),
            "word": ["Hello,", "small", "world!", "Run", "very", "fast."],
            "start": [0.0, 0.2, 0.4, 1.0, 1.2, 1.4],
            "end": [0.1, 0.45, 0.5, 1.1, 1.3, 1.5],
        }
    ).set_index("word_idx")
    annotations = pd.DataFrame(
        {
            "word_idx": np.arange(6),
            "word": word_df["word"].to_numpy(),
            "onset": word_df["start"].to_numpy(),
            "offset": word_df["end"].to_numpy(),
            "pos_class": [0, 2, 0, 1, 3, 3],
            "pos": ["UH", "JJ", "NN", "VB", "RB", "RB"],
            "is_content": [1, 1, 1, 1, 0, 1],
            "surprise": [1, 2, 3, 4, 5, 6],
            "surprise_class": [0, 0, 1, 1, 2, 2],
            "entropy": [6, 5, 4, 3, 2, 1],
            "sentence_idx": [-1, 0, 0, 1, 2, 2],
        }
    )
    features = pd.DataFrame(
        {
            "token_id": np.arange(6),
            "word": word_df["word"].to_numpy(),
            "start": word_df["start"].to_numpy(),
            "end": word_df["end"].to_numpy(),
        }
    )
    for name, task in TASK_REGISTRY.items():
        if name == "surprisal_class":
            continue
        features[name] = (
            np.arange(1, 7, dtype=float)
            if task.kind == "continuous"
            else ["z", "a", "z", "b", "a", "b"]
        )
    features["part_of_speech"] = ["INTJ", "ADJ", "NOUN", "VERB", "ADV", "ADV"]
    features["fine_part_of_speech"] = ["UH", "JJ", "NN", "VB", "RB", "RB"]
    features["native_speaker"] = ["A", "A", "B", "B", "A", "B"]
    features["content_word"] = [True, True, True, True, False, True]
    features["sentence_position"] = [np.nan, 0, 1, 0, 0, 1]
    features["sentence_onset"] = [np.nan, True, False, True, False, False]
    features["sentence_end"] = [np.nan, False, True, True, False, True]
    features["sentence_length"] = [np.nan, 2, 2, 1, 2, 2]
    features["iu_position"] = [0, 1, 0, 0.5, 1, 0]
    features["iu_onset"] = [True, False, True, False, False, True]
    features["iu_end"] = [False, True, False, False, True, True]
    features["iu_length"] = [2, 2, 3, 3, 3, 1]
    features["pre_word_pause"] = [np.nan, 0.1, 0, 0.5, 0.1, 0.1]
    features["story_position"] = np.linspace(0, 1, 6)
    features["story_time_position"] = [0, 1 / 7, 2 / 7, 5 / 7, 6 / 7, 1]
    features["word_length"] = [5, 5, 5, 3, 4, 4]
    features["sense_count"] = [5, np.nan, 2, np.nan, 1, 3]
    features["wordnet_lexname_mfs"] = [
        "noun.person",
        np.nan,
        "noun.object",
        np.nan,
        "noun.person",
        "verb.motion",
    ]
    word_path = tmp_path / "words.csv"
    feature_path = tmp_path / "tokens.tsv"
    annotations.to_csv(word_path)
    features.to_csv(feature_path, sep="\t", index=False)
    return PodcastTaskContext(
        word_df,
        paths={
            "word_annotations": str(word_path),
            "language_features": str(feature_path),
        },
    )


def test_configured_task_getters_are_registered_aligned_and_nonempty(tmp_path):
    context = _task_context(tmp_path)
    configured = yaml.safe_load(
        open("configs/rrr/language_decoding.yml", encoding="utf-8")
    )["tasks"]
    assert set(configured) <= set(TASK_REGISTRY)

    for name in configured:
        target = get_language_task(name).getter(context)
        assert target.shape == (6,)
        assert np.isfinite(target).any(), name

    np.testing.assert_allclose(
        get_language_task("sentence_position").getter(context),
        [np.nan, 0.0, 1.0, 0.0, 0.0, 1.0],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        get_language_task("native_speaker").getter(context),
        [0, 0, 1, 1, 0, 1],
    )
    np.testing.assert_allclose(
        get_language_task("story_position").getter(context),
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    np.testing.assert_allclose(
        get_language_task("story_time_position").getter(context),
        [0.0, 1 / 7, 2 / 7, 5 / 7, 6 / 7, 1.0],
    )
    np.testing.assert_allclose(
        get_language_task("sentence_onset").getter(context),
        [np.nan, 1, 0, 1, 0, 0],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        get_language_task("iu_onset").getter(context), [1, 0, 1, 0, 0, 1]
    )
    np.testing.assert_allclose(
        get_language_task("sentence_end").getter(context),
        [np.nan, 0, 1, 1, 0, 1],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        get_language_task("iu_end").getter(context), [0, 1, 0, 0, 1, 1]
    )
    np.testing.assert_allclose(
        get_language_task("sentence_length").getter(context),
        [np.nan, 2, 2, 1, 2, 2],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        get_language_task("iu_length").getter(context), [2, 2, 3, 3, 3, 1]
    )
    np.testing.assert_allclose(
        get_language_task("pre_word_pause").getter(context),
        [np.nan, 0.1, 0.0, 0.5, 0.1, 0.1],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        get_language_task("fine_part_of_speech").getter(context), [3, 0, 1, 4, 2, 2]
    )
    np.testing.assert_array_equal(
        get_language_task("surprisal_class").getter(context), [0, 0, 1, 1, 2, 2]
    )
    np.testing.assert_allclose(
        get_language_task("word_length").getter(context), [5, 5, 5, 3, 4, 4]
    )
    np.testing.assert_allclose(
        get_language_task("sense_count").getter(context),
        [5, np.nan, 2, np.nan, 1, 3],
        equal_nan=True,
    )
    assert np.isnan(get_language_task("wordnet_lexname_mfs").getter(context)[1])
    assert get_language_task("part_of_speech").chance is None


def test_categorical_targets_are_alphabetical_and_preserve_missingness(tmp_path):
    context = _task_context(tmp_path)
    features = context.language_features.copy()
    features["fine_part_of_speech"] = ["z", "a", np.nan, "z", "b", "a"]
    context._cache["language_features"] = features
    np.testing.assert_allclose(
        get_language_task("fine_part_of_speech").getter(context),
        [2, 0, np.nan, 2, 1, 0],
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda table: table.iloc[:-1], "lengths differ"),
        (lambda table: table.assign(token_id=[0, 1, 2, 4, 4, 5]), "contiguous"),
        (lambda table: table.assign(word=["wrong"] + table.word.tolist()[1:]), "word text"),
        (lambda table: table.assign(start=table.start + 0.01), "timing"),
        (lambda table: table.assign(end=table.end + 0.01), "timing"),
    ],
)
def test_artifact_alignment_failures(tmp_path, mutation, message):
    context = _task_context(tmp_path)
    table = pd.read_csv(context.paths["language_features"], sep="\t")
    mutation(table).to_csv(context.paths["language_features"], sep="\t", index=False)
    with pytest.raises(ValueError, match=message):
        _ = context.language_features


def test_artifact_preserves_literal_none_word_and_empty_missing_field(tmp_path):
    context = _task_context(tmp_path)
    table = pd.read_csv(context.paths["language_features"], sep="\t")
    table.loc[0, "word"] = "None"
    table.loc[0, "concreteness"] = np.nan
    table.to_csv(context.paths["language_features"], sep="\t", index=False)
    context.word_df.loc[0, "word"] = "None"

    assert context.language_features.loc[0, "word"] == "None"
    assert np.isnan(context.language_features.loc[0, "concreteness"])


def test_saved_raw_weights_reproduce_continuous_pipeline_decisions():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 1.5]) + 0.1 * rng.normal(size=80)
    config = LanguageDecodingConfig(rrr_run_path="unused", rank_grid=[2])
    pipeline = _fit_decoder(X, y, "continuous", config)
    payload = _weight_payload(pipeline, None)
    reconstructed = X @ payload["embedding_weights"] + payload["intercept"]
    np.testing.assert_allclose(reconstructed[:, 0], pipeline.predict(X), atol=1e-5)
    metrics = _decoder_metrics("continuous", y, pipeline.predict(X))
    assert metrics["correlation"] > 0.99


def test_saved_raw_weights_reproduce_binary_logistic_decisions():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(int)
    config = LanguageDecodingConfig(rrr_run_path="unused", rank_grid=[2])
    pipeline = _fit_decoder(X, y, "categorical", config)
    payload = _weight_payload(pipeline, None)
    reconstructed = X @ payload["embedding_weights"] + payload["intercept"]
    np.testing.assert_allclose(
        reconstructed[:, 0], pipeline.decision_function(X), atol=1e-5
    )
    metrics = _decoder_metrics("categorical", y, pipeline.predict(X))
    assert metrics["balanced_accuracy"] > 0.9


def test_resolve_rrr_unit_and_config_enum(tmp_path):
    unit = tmp_path / "run" / "rrr" / "fold_0"
    unit.mkdir(parents=True)
    np.save(unit / "A.npy", np.eye(3))
    assert resolve_rrr_unit_dir(str(tmp_path / "run")) == unit.parent.resolve()
    config = LanguageDecodingConfig(
        rrr_run_path=str(tmp_path / "run"), mode=LanguageDecodingMode.CROSS_DATASET
    )
    assert config.mode == LanguageDecodingMode.CROSS_DATASET


def test_resolve_pca_unit_from_run_root(tmp_path):
    unit = tmp_path / "run" / "pca" / "fold_0"
    unit.mkdir(parents=True)
    np.save(unit / "A.npy", np.eye(3))
    assert (
        resolve_subspace_unit_dir(str(tmp_path / "run"), "pca")
        == unit.parent.resolve()
    )


def test_nested_language_decoding_config_roundtrip():
    config = dict_to_dataclass(
        {
            "rrr_run_path": "results/example",
            "pca_run_path": "results/example/pca",
            "mode": "cross_dataset",
            "encoding": {
                "dataset": "podcast",
                "num_folds": 7,
                "per_model_config": {"model_dir": "gpt2-xl-batch", "layer": 24},
            },
            "tasks": ["surprisal"],
        },
        LanguageDecodingConfig,
    )
    assert config.mode == LanguageDecodingMode.CROSS_DATASET
    assert config.pca_run_path == "results/example/pca"
    assert config.encoding.num_folds == 7
    assert config.tasks == ["surprisal"]


@pytest.mark.parametrize("mode", ["fmri_word", "fmri_tr_pre_fir"])
def test_fmri_modes_require_fmri_rrr_source(mode):
    config = dict_to_dataclass(
        {
            "mode": mode,
            "rrr_source": {"dataset": "deep-fmri"},
            "rank_grid": [2],
            "tasks": ["concreteness"],
        },
        LanguageDecodingConfig,
    )
    assert config.mode.value == mode

    with pytest.raises(ValueError, match="requires a deep-fmri rrr_source"):
        dict_to_dataclass(
            {
                "mode": mode,
                "rrr_source": {"dataset": "podcast"},
                "rank_grid": [2],
                "tasks": ["concreteness"],
            },
            LanguageDecodingConfig,
        )

    cross = dict_to_dataclass(
        {
            "mode": mode,
            "rrr_source": {"dataset": "podcast", "mode": "full_fit"},
            "fmri_target_artifact": {"dataset": "deep-fmri", "mode": "internal"},
            "rank_grid": [2],
            "tasks": ["concreteness"],
        },
        LanguageDecodingConfig,
    )
    assert cross.fmri_target_artifact["dataset"] == "deep-fmri"


@pytest.mark.parametrize("mode", ["fmri_word", "fmri_tr_pre_fir"])
def test_fmri_modes_require_internal_rrr_artifacts(mode, tmp_path):
    config = LanguageDecodingConfig(
        mode=mode,
        rrr_source={"mode": "full_fit", "dataset": "deep-fmri"},
        results_root=str(tmp_path),
        rank_grid=[2],
        tasks=["concreteness"],
    )

    with pytest.raises(ValueError, match="normal internal RRR source Artifact"):
        run_config(config)


def test_fmri_word_targets_preserve_raw_values_and_missing_rows(tmp_path):
    frames = {}
    for story, words in {"one": ["a", "b", "c"], "two": ["d", "e"]}.items():
        frames[story] = pd.DataFrame({"word": words})
        root = tmp_path / "features" / story
        root.mkdir(parents=True)
        pd.DataFrame({
            "word": words,
            "concreteness": ([1.0, "", 3.0] if story == "one" else [8.0, 9.0]),
            "part_of_speech": (["N", "", "V"] if story == "one" else ["V", "N"]),
        }).to_csv(root / "tokens.tsv", sep="\t", index=False)

    targets = build_word_targets(
        stories=["one", "two"], frames=frames,
        tasks=["concreteness", "part_of_speech"],
        task_data={"language_features": str(tmp_path / "features")},
        model_dir="model",
    )

    np.testing.assert_allclose(
        targets["concreteness"].values["one"], [1.0, np.nan, 3.0],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        targets["concreteness"].availability["one"], [True, False, True]
    )
    np.testing.assert_array_equal(targets["part_of_speech"].classes, ["N", "V"])
    np.testing.assert_allclose(
        targets["part_of_speech"].values["one"], [0.0, np.nan, 1.0],
        equal_nan=True,
    )


def test_fmri_word_session_cv_remaps_after_task_missing_rows():
    frames = {story: pd.DataFrame(index=range(2)) for story in ["a", "b", "c"]}
    valid = np.array([True, False, True, True, False, True])
    cv = _word_session_cv(
        ["a", "b", "c"], frames, {1: ["a", "b"], 2: ["c"]}, valid
    )

    np.testing.assert_array_equal(cv[0][0], [3])
    np.testing.assert_array_equal(cv[0][1], [0, 1, 2])
    np.testing.assert_array_equal(cv[1][0], [0, 1, 2])
    np.testing.assert_array_equal(cv[1][1], [3])


def test_fmri_targets_lanczos_filter_mean_filled_and_one_hot_channels(
    tmp_path, monkeypatch
):
    stories = ["story"]
    frame = pd.DataFrame({
        "word_idx": np.arange(12),
        "word": [f"w{i}" for i in range(12)],
        "start": np.arange(12, dtype=float),
        "end": np.arange(12, dtype=float) + 0.5,
    })
    sequence = type("Sequence", (), {
        "data_times": np.arange(12, dtype=float),
        "tr_times": np.arange(12, dtype=float),
    })()
    story_root = tmp_path / "features" / "story"
    story_root.mkdir(parents=True)
    table = pd.DataFrame({
        "word": frame["word"],
        "concreteness": [1, 2, 3, 4, 5, "", 7, 8, 9, 10, 11, 12],
        "part_of_speech": ["N", "V", "N", "V", "N", "", "N", "V", "N", "V", "N", "V"],
    })
    table.to_csv(story_root / "tokens.tsv", sep="\t", index=False)
    monkeypatch.setattr(
        "src.utils.language_interpret.fmri_decoding._toolbox_modules",
        lambda _: (lambda values, old, new, window: np.asarray(values), None, None),
    )

    targets = build_tr_targets(
        stories=stories,
        frames={"story": frame},
        sequences={"story": sequence},
        tasks=["concreteness", "part_of_speech"],
        task_data={"language_features": str(tmp_path / "features")},
        model_dir="model",
        fmri_repo_root="unused",
        trim=0,
    )

    continuous = targets["concreteness"]
    assert continuous.filtered["story"].shape == (7, 1)
    assert continuous.filtered["story"][0, 0] == pytest.approx(72 / 11)
    assert continuous.availability["story"][0] == 0
    assert continuous.standardized["story"].mean() == pytest.approx(0, abs=1e-6)

    categorical = targets["part_of_speech"]
    assert categorical.classes.tolist() == ["N", "V"]
    np.testing.assert_allclose(categorical.filtered["story"][0], [6 / 11, 5 / 11])
    assert categorical.availability["story"][0] == 0
    np.testing.assert_allclose(
        categorical.standardized["story"].mean(axis=0), 0, atol=1e-6
    )


def test_speaking_rate_counts_onsets_including_silent_trs(monkeypatch):
    monkeypatch.setattr(
        "src.utils.language_interpret.fmri_decoding._toolbox_modules",
        lambda _: (None, None, None),
    )
    # Two-second TRs. Word midpoints deliberately disagree with starts.
    sequence = type("Sequence", (), {
        "tr_times": np.arange(1, 17, 2, dtype=float),
        "data_times": np.array([11.9, 12.5, 15.9, 17.0]),
    })()
    frame = pd.DataFrame({"start": [10.0, 10.5, 14.0, 16.0]})
    target = build_tr_targets(
        stories=["story"], frames={"story": frame},
        sequences={"story": sequence}, tasks=["speaking_rate"],
        task_data={}, model_dir="unused", fmri_repo_root="unused", trim=0,
    )["speaking_rate"]
    np.testing.assert_array_equal(target.filtered["story"][:, 0], [1.0, 0.0, 0.5])
    np.testing.assert_array_equal(target.availability["story"], [1, 1, 1])
    assert target.standardized["story"].mean() == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("offset,expected", [
    (-1, [np.nan, 10, 20]), (0, [10, 20, 30]),
    (1, [20, 30, np.nan]), (4, [np.nan, np.nan, np.nan]),
])
def test_word_label_offsets_direction_and_boundaries(offset, expected):
    from src.utils.language_interpret.tasks import offset_word_labels
    np.testing.assert_allclose(
        offset_word_labels(np.array([10, 20, 30]), offset), expected, equal_nan=True,
    )


def test_fmri_label_offsets_reset_per_story(monkeypatch):
    tables = {
        "a": pd.DataFrame({"word": ["a", "b"], "concreteness": [1, 2]}),
        "b": pd.DataFrame({"word": ["c", "d"], "concreteness": [3, 4]}),
    }
    monkeypatch.setattr(
        "src.utils.language_interpret.fmri_decoding._task_table",
        lambda story, *args: tables[story],
    )
    target = build_word_targets(
        stories=["a", "b"], frames=tables, tasks=["concreteness"],
        task_data={}, model_dir="unused", label_offset=1,
    )["concreteness"]
    np.testing.assert_allclose(target.values["a"], [2, np.nan], equal_nan=True)
    np.testing.assert_allclose(target.values["b"], [4, np.nan], equal_nan=True)
    np.testing.assert_array_equal(target.availability["a"], [True, False])


def test_speaking_rate_rejected_for_word_targets():
    with pytest.raises(ValueError, match="only available in fmri_tr_pre_fir"):
        build_word_targets(
            stories=[], frames={}, tasks=["speaking_rate"],
            task_data={}, model_dir="unused",
        )


def test_fmri_categorical_metrics_and_stable_classifier_classes():
    y_train = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    y_test = y_train.copy()
    summary, channels, eligible = _categorical_metrics(
        y_train, y_test, y_test.copy(), np.array(["a", "b", "heldout"])
    )
    assert summary["primary_score"] == pytest.approx(1.0)
    assert summary["multivariate_r2"] == pytest.approx(1.0)
    assert summary["mean_cosine_similarity"] == pytest.approx(1.0)
    np.testing.assert_array_equal(eligible, [True, True, False])
    assert channels["eligible"].tolist() == [True, True, False]

    labels = np.array([0, 1, 2, 0, 1, 0])
    cv = [
        (np.array([0, 1, 3, 4]), np.array([2, 5])),
        (np.array([0, 1, 2, 5]), np.array([3, 4])),
    ]
    rows, remapped, stable = _stable_classifier_rows(labels, cv)
    np.testing.assert_array_equal(stable, [0, 1])
    np.testing.assert_array_equal(rows, [0, 1, 3, 4, 5])
    assert all(train.max() < len(rows) for train, _ in remapped)


def test_fmri_pre_fir_rank_design_uses_saved_source_scaler():
    aligned = AlignedFMRIStimuli(stories={
        "train": np.arange(24, dtype=np.float32).reshape(8, 3),
        "test": np.arange(30, 54, dtype=np.float32).reshape(8, 3),
    })
    factor = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]], dtype=np.float32)
    source_scaler = {
        "mean": aligned.stories["train"].mean(axis=0),
        "scale": aligned.stories["train"].std(axis=0),
        "coordinates": "standardized-v1",
    }

    prepared = _prepare_rank_design(
        aligned=aligned, train_stories=["train"], test_stories=["test"],
        factor=factor, source_scaler=source_scaler, rank=2, ndelays=2,
    )

    np.testing.assert_allclose(prepared[2], prepared[0].pre_fir_stimulus)
    np.testing.assert_allclose(prepared[3], prepared[1].pre_fir_stimulus)
    assert prepared[2].shape == (8, 2)
    np.testing.assert_allclose(prepared[2].mean(axis=0), 0, atol=1e-6)
    assert not np.allclose(prepared[3].mean(axis=0), 0)
    np.testing.assert_array_equal(prepared[4], factor)


@pytest.mark.parametrize("mode", ["fmri_word", "fmri_tr_pre_fir"])
def test_primary_client_publishes_fmri_results(tmp_path, monkeypatch, mode):
    source = RRRArtifact(
        mode="internal", dataset="deep-fmri", offset=0,
        model="gpt2-xl-batch", layer=24, variant="source",
        results_root=tmp_path,
    ).begin(config={"rank_grid": [1]})
    source.save_field("effective_config", {
        "model_dir": "gpt2-xl-batch", "layer": 24, "pooling": "mean",
        "fmri": {
            "fmri_repo_root": "unused", "embedding_root": "unused",
            "sessions": [1, 2], "trim": 5, "ndelays": 2,
        },
    })
    source.save_field("input_factors", np.eye(2, 1),
                      selector={"method": "rrr", "rank": 1})
    source.save_field(
        "input_scalers",
        {"mean": np.zeros(2), "scale": np.ones(2),
         "coordinates": "standardized-v1"},
    )

    def fake_run(
        config, source_config, factors, source_scaler, output, artifact=None
    ):
        from src.utils.language_interpret.output import DecodingOutput
        assert config.mode.value == mode
        assert source_config["fmri"]["ndelays"] == 2
        np.testing.assert_array_equal(factors[("rrr", 1)], np.eye(2, 1))
        np.testing.assert_array_equal(source_scaler["scale"], np.ones(2))
        assert artifact is not None
        writer = DecodingOutput(output, artifact)
        writer.save("sample_selection", pd.DataFrame({
            "analysis_row": [0], "split": ["test"], "story": ["heldout"],
            "story_tr_index": [0], "tr_time": [1.0],
        }))
        writer.save("run_metadata", {"mode": mode})
        metrics = {
            "task": "concreteness", "kind": "continuous", "fold": 0,
            "space": "rank", "subspace": "rrr", "k": 1,
            "primary_score": 0.5, "correlation": 0.5, "r2": 0.1,
        }
        writer.save("grid_results", pd.DataFrame([metrics]))
        writer.save("summary", pd.DataFrame([metrics]))
        writer.save_fit(metrics, {"reduced_weights": np.ones((1, 1))},
                        {"y_true": np.array([0.0]), "y_pred": np.array([0.0])})
        writer.save("target_diagnostics", {
            "task": "concreteness", "normalization": "per_story_zscore",
        }, {"task": "concreteness", "name": "metadata"})
        writer.save("target_diagnostics", {"heldout": np.ones(1)},
                    {"task": "concreteness", "name": "availability"})
        writer.save("target_diagnostics", {
            "filtered__heldout": np.ones((1, 1)),
            "standardized__heldout": np.zeros((1, 1)),
        }, {"task": "concreteness", "name": "targets"})
        assert not output.exists()  # No intermediate result files.
        return output

    monkeypatch.setattr(
        "src.utils.language_interpret.fmri_decoding.run_fmri_decoding", fake_run
    )
    config = LanguageDecodingConfig(
        mode=LanguageDecodingMode(mode),
        rrr_source={
            "mode": "internal", "dataset": "deep-fmri", "offset": 0,
            "model": "gpt2-xl-batch", "layer": 24, "variant": "source",
        },
        results_root=str(tmp_path), rank_grid=[1], tasks=["concreteness"],
    )

    artifact = run_config(config)

    assert isinstance(artifact, DecodingArtifact)
    assert artifact.identity["mode"] == mode
    assert artifact.complete
    assert (artifact._variant_dir() / "artifact.h5").is_file()
    assert not list(artifact._variant_dir().rglob("*.npz"))
    assert not list(artifact._variant_dir().rglob("*.csv"))
    loaded = artifact.load_field("sample_selection")
    assert loaded.loc[0, "story"] == "heldout"
    diagnostics = artifact.load_collection(
        "target_diagnostics", {"task": "concreteness"}
    )
    assert len(diagnostics) == 3


def test_language_decoding_device_validation_and_cpu_backend(monkeypatch):
    with pytest.raises(ValueError, match="device must be one of"):
        LanguageDecodingConfig(rrr_run_path="unused", device="tpu")

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert _configure_decoder_backend("auto") == "cpu"
    assert _configure_decoder_backend("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _configure_decoder_backend("cuda")


def test_batched_continuous_fits_preserve_per_task_predictions(tmp_path):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(90, 6))
    targets = {
        "surprisal": X @ np.array([1.0, 0.0, -0.5, 0.2, 0.0, 0.1]),
        "entropy": X @ np.array([0.0, -1.0, 0.3, 0.0, 0.5, 0.0]),
    }
    train_idx = np.arange(70)
    test_idx = np.arange(70, 90)
    config = LanguageDecodingConfig(
        rrr_run_path="unused",
        rank_grid=[2],
        tasks=list(targets),
        device="cpu",
    )
    rows = _run_task_decoding(
        output_dir=tmp_path,
        fold=0,
        rank=None,
        X=X,
        task_targets=targets,
        train_idx=train_idx,
        test_idx=test_idx,
        config=config,
        A=None,
        subspace="full",
    )
    assert [row["task"] for row in rows] == list(targets)

    for task_name, y in targets.items():
        weights = np.load(
            tmp_path / "tasks" / task_name / "full" / "fold_0" / "weights.npz"
        )
        reconstructed = (
            X[test_idx] @ weights["embedding_weights"] + weights["intercept"]
        )[:, 0]
        independent = _fit_decoder(
            X[train_idx], y[train_idx], "continuous", config
        ).predict(X[test_idx])
        np.testing.assert_allclose(reconstructed, independent, rtol=1e-5, atol=1e-5)


def test_summary_uses_task_specific_primary_metrics():
    results = pd.DataFrame(
        [
            {
                "task": "surprisal",
                "kind": "continuous",
                "space": "rank",
                "subspace": "rrr",
                "k": 1,
                "primary_score": 0.2,
                "correlation": 0.2,
                "r2": 0.0,
            },
            {
                "task": "surprisal",
                "kind": "continuous",
                "space": "rank",
                "subspace": "rrr",
                "k": 1,
                "primary_score": 0.4,
                "correlation": 0.4,
                "r2": 0.1,
            },
            {
                "task": "content_word",
                "kind": "categorical",
                "space": "full",
                "subspace": "full",
                "k": None,
                "primary_score": 0.6,
                "balanced_accuracy": 0.6,
                "macro_f1": 0.5,
                "accuracy": 0.7,
            },
            {
                "task": "content_word",
                "kind": "categorical",
                "space": "full",
                "subspace": "full",
                "k": None,
                "primary_score": 0.8,
                "balanced_accuracy": 0.8,
                "macro_f1": 0.7,
                "accuracy": 0.8,
            },
        ]
    )
    summary = summarize_language_decoding(results)
    assert len(summary) == 2
    categorical = summary[summary.task == "content_word"].iloc[0]
    assert categorical.primary_score == pytest.approx(0.7)
    assert categorical.n_folds == 2


def test_summary_keeps_rrr_and_pca_rank_rows_separate():
    rows = []
    for subspace, score in (("rrr", 0.4), ("pca", 0.2)):
        for fold in range(2):
            rows.append(
                {
                    "task": "surprisal",
                    "kind": "continuous",
                    "space": "rank",
                    "subspace": subspace,
                    "k": 2,
                    "primary_score": score,
                    "correlation": score,
                    "r2": 0.0,
                    "fold": fold,
                }
            )
    summary = summarize_language_decoding(pd.DataFrame(rows))
    assert set(summary["subspace"]) == {"rrr", "pca"}
    assert len(summary) == 2


def test_plot_overlays_rrr_and_pca(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "task": "surprisal",
                "kind": "continuous",
                "space": "full",
                "subspace": "full",
                "k": np.nan,
                "primary_score": 0.5,
                "sem_lower": 0.01,
                "sem_upper": 0.01,
            },
            *[
                {
                    "task": "surprisal",
                    "kind": "continuous",
                    "space": "rank",
                    "subspace": subspace,
                    "k": rank,
                    "primary_score": score,
                    "sem_lower": 0.02,
                    "sem_upper": 0.02,
                }
                for subspace, rank, score in (
                    ("rrr", 1, 0.2),
                    ("rrr", 2, 0.3),
                    ("pca", 1, 0.1),
                    ("pca", 2, 0.15),
                )
            ],
        ]
    )
    output = tmp_path / "rank_analysis.png"
    agreement = pd.DataFrame(
        [
            {
                "task": "surprisal",
                "kind": "continuous",
                "k": rank,
                "prediction_correlation": prediction,
                "prediction_correlation_sem_lower": 0.02,
                "prediction_correlation_sem_upper": 0.02,
                "error_correlation": error,
                "error_correlation_sem_lower": 0.03,
                "error_correlation_sem_upper": 0.03,
            }
            for rank, prediction, error in ((1, 0.4, 0.2), (2, 0.7, 0.5))
        ]
    )
    plot_language_decoding(summary, output, agreement=agreement)
    assert output.exists()


def test_matched_score_correlation_fisher_averages_multiclass_columns():
    rng = np.random.default_rng(18)
    left = rng.normal(size=(200, 3))
    right = np.column_stack(
        [
            left[:, 0] + 0.1 * rng.normal(size=200),
            left[:, 1] + 0.5 * rng.normal(size=200),
            left[:, 2] + rng.normal(size=200),
        ]
    )
    correlation, n_columns = _matched_score_correlation(left, right)
    individual = [np.corrcoef(left[:, index], right[:, index])[0, 1] for index in range(3)]
    expected = np.tanh(np.mean(np.arctanh(individual)))
    assert correlation == pytest.approx(expected)
    assert n_columns == 3


def test_prediction_agreement_summary_fisher_averages_folds_and_keeps_errors():
    results = pd.DataFrame(
        [
            {
                "task": "surprisal",
                "kind": "continuous",
                "k": 2,
                "fold": fold,
                "prediction_correlation": prediction,
                "error_correlation": error,
            }
            for fold, prediction, error in ((0, 0.2, -0.1), (1, 0.6, 0.3))
        ]
    )
    summary = summarize_prediction_agreement(results).iloc[0]
    assert summary.prediction_correlation == pytest.approx(
        np.tanh(np.mean(np.arctanh([0.2, 0.6])))
    )
    assert summary.error_correlation == pytest.approx(
        np.tanh(np.mean(np.arctanh([-0.1, 0.3])))
    )


def test_run_subspace_decoding_loads_paths_and_validates_fold_set(
    tmp_path, monkeypatch
):
    unit = tmp_path / "pca"
    for fold in (0, 1):
        fold_dir = unit / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        np.save(fold_dir / "A.npy", np.eye(4))
    config = LanguageDecodingConfig(
        rrr_run_path="unused", rank_grid=[1, 3], tasks=["surprisal"]
    )
    seen_shapes = []

    def capture_fit(*args):
        seen_shapes.append((args[4], args[5].shape, args[-1]))
        return {"fold": args[3], "k": args[4], "subspace": args[-1]}

    monkeypatch.setattr("src.utils.language_interpret.decoding._run_one_fit", capture_fit)
    folds = [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ]
    prepared = _prepare_single_view_subspaces(
        subspace="pca",
        factor_paths=sorted(unit.glob("fold_*/A.npy")),
        folds=folds,
        X=np.eye(4),
        config=config,
    )
    rows = _run_subspace_decoding(
        output_dir=tmp_path / "output",
        subspace="pca",
        prepared=prepared,
        folds=folds,
        task_targets={"surprisal": np.arange(4, dtype=float)},
        config=config,
    )
    assert len(rows) == 4
    assert seen_shapes == [
        (1, (4, 1), "pca"),
        (3, (4, 3), "pca"),
        (1, (4, 1), "pca"),
        (3, (4, 3), "pca"),
    ]

    with pytest.raises(ValueError, match="do not match decoding folds"):
        _prepare_single_view_subspaces(
            subspace="pca",
            factor_paths=sorted(unit.glob("fold_*/A.npy")),
            folds=[*folds, (np.array([0, 1]), np.array([2, 3]))],
            X=np.eye(4),
            config=config,
        )


def test_cross_dataset_single_full_fit_factor_is_applied_without_aggregation(
    tmp_path, monkeypatch
):
    unit = tmp_path / "rrr"
    fold = unit / "fold_0"
    fold.mkdir(parents=True)
    factor = np.array([[2.0], [0.5]], dtype=np.float32)
    np.save(fold / "A.npy", factor)
    source_mean = np.array([10.0, -2.0], dtype=np.float32)
    source_scale = np.array([4.0, 2.0], dtype=np.float32)
    np.savez(fold / "input_scaler.npz", mean=source_mean, scale=source_scale)
    X = np.array([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]], dtype=np.float32)
    folds = [(np.array([0, 1]), np.array([2]))]
    config = LanguageDecodingConfig(
        rrr_run_path="unused",
        mode="cross_dataset",
        rank_grid=[1],
        tasks=["surprisal"],
    )
    monkeypatch.setattr(
        "src.utils.language_interpret.decoding.aggregate_subspace_basis",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a single full-fit factor must not be aggregated")
        ),
    )

    prepared = _prepare_single_view_subspaces(
        subspace="rrr", factor_paths=[fold / "A.npy"], folds=folds, X=X,
        config=config,
    )

    expected = ((X - source_mean) / source_scale) @ factor
    np.testing.assert_allclose(prepared[(0, 1)][0], expected)
    np.testing.assert_array_equal(prepared[(0, 1)][1], factor)


def test_compose_multiview_factor_preserves_separate_raw_blocks():
    view_a = np.array([[1.0, 0.0], [0.0, 1.0]])
    view_b = np.array([[2.0], [3.0], [4.0]])
    final = np.array([[1.0], [2.0], [5.0]])
    composed = compose_multiview_factor([view_a, view_b], final)
    np.testing.assert_allclose(composed[:, 0], [1.0, 2.0, 10.0, 15.0, 20.0])


def test_compose_before_fold_average_retains_canceling_signs():
    component_folds = [np.array([[1.0], [0.0]]), np.array([[-1.0], [0.0]])]
    final_folds = [np.array([[1.0]]), np.array([[-1.0]])]
    composed = [
        compose_multiview_factor([component], final)
        for component, final in zip(component_folds, final_folds)
    ]
    np.testing.assert_allclose(composed[0], composed[1])
    np.testing.assert_allclose(np.mean(composed, axis=0), [[1.0], [0.0]])
    np.testing.assert_allclose(np.mean(final_folds, axis=0), [[0.0]])


def test_cross_multiview_decoding_composes_before_aggregation(tmp_path, monkeypatch):
    paths = []
    for fold, value in enumerate((1.0, -1.0)):
        folder = tmp_path / f"fold_{fold}"
        folder.mkdir()
        path = folder / "A.npy"
        np.save(path, np.array([[value]]))
        paths.append(path)
    component_bases = {
        "view": {
            0: np.array([[1.0], [0.0]]),
            1: np.array([[-1.0], [0.0]]),
        }
    }
    captured = []

    def capture_fit(*args):
        captured.append((args[5].copy(), args[-2].copy()))
        return {"fold": args[3], "k": args[4], "subspace": args[-1]}

    monkeypatch.setattr("src.utils.language_interpret.decoding._run_one_fit", capture_fit)
    config = LanguageDecodingConfig(
        rrr_run_path="unused",
        mode=LanguageDecodingMode.CROSS_DATASET,
        rank_grid=[1],
        tasks=["surprisal"],
    )
    embeddings = {"view": np.array([[1.0, 2.0], [3.0, 4.0]])}
    folds = [(np.array([1]), np.array([0])), (np.array([0]), np.array([1]))]
    prepared = _prepare_multiview_subspaces(
        subspace="rrr",
        final_factor_paths=paths,
        component_bases=component_bases,
        embeddings=embeddings,
        view_names=["view"],
        selection=np.arange(2),
        folds=folds,
        config=config,
    )
    rows = _run_subspace_decoding(
        output_dir=tmp_path / "output",
        subspace="rrr",
        prepared=prepared,
        folds=folds,
        task_targets={"surprisal": np.array([0.0, 1.0])},
        config=config,
    )
    assert len(rows) == 2
    for design, factor in captured:
        np.testing.assert_allclose(np.abs(factor), [[1.0], [0.0]], atol=1e-6)
        np.testing.assert_allclose(np.abs(design[:, 0]), [1.0, 3.0], atol=1e-6)
