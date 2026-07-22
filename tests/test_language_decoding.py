import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.core.config import LanguageDecodingConfig, LanguageDecodingMode
from src.utils.config_utils import dict_to_dataclass
from src.utils.language_decoding import (
    _decoder_metrics,
    _fit_decoder,
    _run_subspace_decoding,
    _weight_payload,
    plot_language_decoding,
    resolve_rrr_unit_dir,
    resolve_subspace_unit_dir,
    summarize_language_decoding,
)
from src.utils.language_tasks import PodcastTaskContext, get_language_task


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
    sentences = pd.DataFrame({"sentence_onset": [0.2, 1.0]})
    iu = pd.DataFrame(
        {
            "time": [0.35, 1.35],
            "prev_word": ["small", "missing"],
            "next_word": ["world", "also-missing"],
        }
    )
    word_path = tmp_path / "words.csv"
    sentence_path = tmp_path / "sentences.csv"
    iu_path = tmp_path / "iu.csv"
    annotations.to_csv(word_path)
    sentences.to_csv(sentence_path, index=False)
    iu.to_csv(iu_path, index=False)
    return PodcastTaskContext(
        word_df,
        paths={
            "word_annotations": str(word_path),
            "sentence_annotations": str(sentence_path),
            "iu_annotations": str(iu_path),
        },
    )


def test_task_getters_are_word_aligned_and_positions_are_normalized(tmp_path, monkeypatch):
    fake_wordfreq = types.SimpleNamespace(
        zipf_frequency=lambda word, language, minimum: float(len(word))
    )
    monkeypatch.setitem(sys.modules, "wordfreq", fake_wordfreq)
    context = _task_context(tmp_path)

    for name in (
        "part_of_speech",
        "content_word",
        "surprisal",
        "entropy",
        "story_position",
        "story_time_position",
        "fine_part_of_speech",
        "surprisal_class",
        "sentence_onset",
        "iu_onset",
        "sentence_position",
        "iu_position",
        "sentence_end",
        "iu_end",
        "sentence_length",
        "iu_length",
        "pre_word_pause",
        "word_duration",
        "word_length",
        "word_zipf_frequency",
    ):
        target = get_language_task(name).getter(context)
        assert target.shape == (6,)

    np.testing.assert_allclose(
        get_language_task("sentence_position").getter(context),
        [np.nan, 0.0, 1.0, 0.0, 0.0, 1.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        get_language_task("story_position").getter(context),
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    np.testing.assert_allclose(
        get_language_task("story_time_position").getter(context),
        [0.0, 1 / 7, 2 / 7, 5 / 7, 6 / 7, 1.0],
    )
    np.testing.assert_array_equal(
        get_language_task("sentence_onset").getter(context), [0, 1, 0, 1, 0, 0]
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
    provenance = context.iu_alignment()[2]
    assert provenance["method"].tolist() == ["exact_pair", "time_fallback"]
    np.testing.assert_allclose(
        get_language_task("word_length").getter(context), [5, 5, 5, 3, 4, 4]
    )


def test_fine_part_of_speech_rejects_missing_tags(tmp_path):
    context = _task_context(tmp_path)
    words = context.words.copy()
    words.loc[0, "pos"] = np.nan
    context._cache["words"] = words

    with pytest.raises(ValueError, match="missing tags"):
        get_language_task("fine_part_of_speech").getter(context)


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
    plot_language_decoding(summary, output)
    assert output.exists()


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

    monkeypatch.setattr("src.utils.language_decoding._run_one_fit", capture_fit)
    folds = [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ]
    rows = _run_subspace_decoding(
        output_dir=tmp_path / "output",
        subspace="pca",
        factor_paths=sorted(unit.glob("fold_*/A.npy")),
        folds=folds,
        X=np.eye(4),
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
        _run_subspace_decoding(
            output_dir=tmp_path / "output",
            subspace="pca",
            factor_paths=sorted(unit.glob("fold_*/A.npy")),
            folds=[*folds, (np.array([0, 1]), np.array([2, 3]))],
            X=np.eye(4),
            task_targets={"surprisal": np.arange(4, dtype=float)},
            config=config,
        )
