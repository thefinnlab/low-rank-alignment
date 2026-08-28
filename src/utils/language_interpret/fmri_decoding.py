"""TR-aligned language decoding for saved deep-fMRI RRR/PCA spaces."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV as SklearnRidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.core.config import LanguageDecodingConfig, LanguageDecodingMode
from src.utils.language_interpret.decoding import (
    _configure_decoder_backend,
    _decoder_metrics,
    _fit_decoder,
    _safe_pearson,
    _weight_payload,
    plot_language_decoding,
    summarize_language_decoding,
)
from src.utils.language_interpret.tasks import get_language_task
from src.utils.shared.correlation import fisher_z_mean_sem
from src.utils.shared.constants import RIDGE_ALPHAS
from src.utils.shared.fmri import (
    _toolbox_modules,
    align_fmri_stimuli,
    build_fmri_stimulus,
    load_story_embeddings,
    load_story_word_data,
    resolve_fmri_stories,
    session_cv_splits,
)
from src.utils.shared.tensors import to_numpy


@dataclass
class TRTarget:
    """One task represented on the shared fMRI TR grid."""

    standardized: dict[str, np.ndarray]
    filtered: dict[str, np.ndarray]
    availability: dict[str, np.ndarray]
    classes: np.ndarray


@dataclass
class WordTarget:
    """One task represented on the original fMRI transcript words."""

    values: dict[str, np.ndarray]
    availability: dict[str, np.ndarray]
    classes: np.ndarray


def _story_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    return ((values - mean) / scale).astype(np.float32)


def _task_table(
    story: str,
    task: str,
    task_data: dict,
    model_dir: str,
) -> pd.DataFrame:
    if task in {"self_surprisal", "self_entropy"}:
        root = task_data.get("self_surprisal_root")
        if not root:
            raise ValueError(f"Task {task} requires task_data.self_surprisal_root")
        path = Path(str(root).format(model=model_dir))
        candidates = [
            path / story / "tokens.tsv",
            path / model_dir / "fmri" / story / "tokens.tsv",
        ]
    else:
        root = task_data.get("language_features")
        if not root:
            raise ValueError("The fMRI modes require task_data.language_features")
        candidates = [Path(root) / story / "tokens.tsv"]
    paths = [path for path in candidates if path.is_file()]
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one feature artifact for {task}/{story}; checked {candidates}"
        )
    return pd.read_csv(
        paths[0], sep="\t", keep_default_na=False, low_memory=False
    )


def _validate_table_alignment(
    table: pd.DataFrame, frame: pd.DataFrame, story: str, task: str
) -> None:
    if "word" not in table:
        raise ValueError(f"Story {story} feature artifact is missing column 'word'")
    if task not in table:
        raise ValueError(f"Story {story} is missing requested feature column {task!r}")
    if len(table) != len(frame):
        raise ValueError(
            f"Feature/embedding transcript lengths differ for {story}: "
            f"{len(table)} vs {len(frame)}"
        )
    if not np.array_equal(
        table["word"].astype(str).str.lower().to_numpy(),
        frame["word"].astype(str).str.lower().to_numpy(),
    ):
        raise ValueError(f"Feature/embedding word alignment failed for {story}/{task}")


def build_tr_targets(
    *,
    stories: list[str],
    frames: dict[str, pd.DataFrame],
    sequences: dict[str, object],
    tasks: list[str],
    task_data: dict,
    model_dir: str,
    fmri_repo_root: str,
    trim: int,
) -> dict[str, TRTarget]:
    """Create story-standardized Lanczos targets and audit traces."""
    lanczosinterp2D, _, _ = _toolbox_modules(fmri_repo_root)
    result = {}
    for task in tasks:
        spec = get_language_task(task)
        tables = {}
        for story in stories:
            table = _task_table(story, task, task_data, model_dir)
            _validate_table_alignment(table, frames[story], story, task)
            tables[story] = table

        classes = np.array([], dtype=str)
        if spec.kind == "categorical":
            labels = []
            for table in tables.values():
                values = table[task].astype(str).str.strip()
                labels.extend(values[values != ""].tolist())
            classes = np.asarray(sorted(set(labels)), dtype=str)
            if len(classes) < 2:
                raise ValueError(f"Categorical task {task} has fewer than two labels")
            class_index = {label: index for index, label in enumerate(classes)}

        filtered = {}
        standardized = {}
        availability_by_story = {}
        for story in stories:
            raw = tables[story][task]
            if spec.kind == "continuous":
                numeric = pd.to_numeric(raw, errors="coerce").to_numpy(float)
                available = np.isfinite(numeric)
                if not available.any():
                    raise ValueError(f"Task {task} has no finite values in {story}")
                filled = numeric.copy()
                filled[~available] = float(np.mean(numeric[available]))
                word_values = filled[:, None]
            else:
                labels = raw.astype(str).str.strip().to_numpy()
                available = labels != ""
                word_values = np.zeros((len(labels), len(classes)), dtype=float)
                for row, label in enumerate(labels):
                    if available[row]:
                        word_values[row, class_index[label]] = 1.0
                if available.any():
                    prevalence = word_values[available].mean(axis=0)
                else:
                    raise ValueError(f"Task {task} has no labels in {story}")
                word_values[~available] = prevalence

            downsampled = lanczosinterp2D(
                word_values,
                sequences[story].data_times,
                sequences[story].tr_times,
                window=3,
            )
            filtered_availability = lanczosinterp2D(
                available.astype(float)[:, None],
                sequences[story].data_times,
                sequences[story].tr_times,
                window=3,
            )[:, 0]
            stop = -int(trim) if trim else None
            keep = slice(5 + int(trim), stop)
            filtered[story] = np.asarray(downsampled[keep], dtype=np.float32)
            availability_by_story[story] = np.asarray(
                filtered_availability[keep], dtype=np.float32
            )
            standardized[story] = _story_zscore(filtered[story])
        result[task] = TRTarget(
            standardized=standardized,
            filtered=filtered,
            availability=availability_by_story,
            classes=classes,
        )
    return result


def build_word_targets(
    *,
    stories: list[str],
    frames: dict[str, pd.DataFrame],
    tasks: list[str],
    task_data: dict,
    model_dir: str,
) -> dict[str, WordTarget]:
    """Load word-aligned targets while preserving task-specific missing rows."""
    result = {}
    for task in tasks:
        spec = get_language_task(task)
        tables = {}
        for story in stories:
            table = _task_table(story, task, task_data, model_dir)
            _validate_table_alignment(table, frames[story], story, task)
            tables[story] = table

        classes = np.array([], dtype=str)
        if spec.kind == "categorical":
            labels = []
            for table in tables.values():
                values = table[task].astype(str).str.strip()
                labels.extend(values[values != ""].tolist())
            classes = np.asarray(sorted(set(labels)), dtype=str)
            if len(classes) < 2:
                raise ValueError(f"Categorical task {task} has fewer than two labels")
            class_index = {label: index for index, label in enumerate(classes)}

        values_by_story = {}
        availability_by_story = {}
        for story, table in tables.items():
            raw = table[task]
            if spec.kind == "continuous":
                values = pd.to_numeric(raw, errors="coerce").to_numpy(float)
                available = np.isfinite(values)
            else:
                labels = raw.astype(str).str.strip().to_numpy()
                available = labels != ""
                values = np.full(len(labels), np.nan, dtype=float)
                values[available] = [class_index[label] for label in labels[available]]
            values_by_story[story] = values
            availability_by_story[story] = available
        result[task] = WordTarget(
            values=values_by_story,
            availability=availability_by_story,
            classes=classes,
        )
    return result


def _stack(stories: list[str], values: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([np.asarray(values[story]) for story in stories])


def _cosine_mean(true: np.ndarray, predicted: np.ndarray) -> float:
    numerator = np.sum(true * predicted, axis=1)
    denominator = np.linalg.norm(true, axis=1) * np.linalg.norm(predicted, axis=1)
    valid = denominator > 0
    return (
        float(np.mean(numerator[valid] / denominator[valid]))
        if valid.any()
        else float("nan")
    )


def _categorical_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    prediction: np.ndarray,
    classes: np.ndarray,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    eligible = (y_train.std(axis=0) > 0) & (y_test.std(axis=0) > 0)
    rows = []
    for index, label in enumerate(classes):
        correlation = _safe_pearson(y_test[:, index], prediction[:, index])
        channel_r2 = (
            float(r2_score(y_test[:, index], prediction[:, index]))
            if eligible[index]
            else float("nan")
        )
        rows.append({
            "class": str(label),
            "channel": int(index),
            "eligible": bool(eligible[index]),
            "correlation": correlation if eligible[index] else float("nan"),
            "r2": channel_r2,
        })
    channel_metrics = pd.DataFrame(rows)
    correlations = channel_metrics.loc[
        channel_metrics["eligible"], "correlation"
    ].to_numpy(float)
    correlations = correlations[np.isfinite(correlations)]
    primary = (
        float(fisher_z_mean_sem(correlations)[0])
        if len(correlations)
        else float("nan")
    )
    if eligible.any():
        vector_r2 = float(
            r2_score(
                y_test[:, eligible],
                prediction[:, eligible],
                multioutput="uniform_average",
            )
        )
        cosine = _cosine_mean(y_test[:, eligible], prediction[:, eligible])
    else:
        vector_r2 = cosine = float("nan")
    return {
        "primary_score": primary,
        "correlation": primary,
        "multivariate_r2": vector_r2,
        "mean_cosine_similarity": cosine,
        "n_eligible_channels": int(eligible.sum()),
        "n_excluded_channels": int((~eligible).sum()),
    }, channel_metrics, eligible


def _stable_classifier_rows(
    labels: np.ndarray,
    cv: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Keep labels learnable in every session-held-out training partition."""
    labels = np.asarray(labels, dtype=int)
    stable = set(np.unique(labels).tolist())
    for train, _ in cv:
        stable &= set(np.unique(labels[np.asarray(train, dtype=int)]).tolist())
    stable_classes = np.asarray(sorted(stable), dtype=int)
    retained = np.flatnonzero(np.isin(labels, stable_classes))
    position = np.full(len(labels), -1, dtype=int)
    position[retained] = np.arange(len(retained))
    remapped = []
    for train, validation in cv:
        train = np.asarray(train, dtype=int)
        validation = np.asarray(validation, dtype=int)
        train = train[np.isin(labels[train], stable_classes)]
        validation = validation[np.isin(labels[validation], stable_classes)]
        if len(validation):
            remapped.append((position[train], position[validation]))
    return retained, remapped, stable_classes


def _sample_selection(
    stories: list[str], train_stories: list[str], sequences: dict, trim: int
) -> pd.DataFrame:
    rows = []
    for story in stories:
        stop = -int(trim) if trim else None
        times = np.asarray(sequences[story].tr_times)[5 + int(trim) : stop]
        split = "train" if story in train_stories else "test"
        for index, time in enumerate(times):
            rows.append({
                "analysis_row": len(rows),
                "split": split,
                "story": story,
                "story_tr_index": int(index),
                "tr_time": float(time),
            })
    return pd.DataFrame(rows)


def _word_sample_selection(
    stories: list[str], train_stories: list[str], frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    parts = []
    cursor = 0
    for story in stories:
        frame = frames[story].reset_index(drop=True).copy()
        frame.insert(0, "analysis_row", np.arange(cursor, cursor + len(frame)))
        frame.insert(1, "split", "train" if story in train_stories else "test")
        frame.insert(2, "story", story)
        parts.append(frame)
        cursor += len(frame)
    return pd.concat(parts, ignore_index=True)


def _word_session_cv(
    train_stories: list[str],
    frames: dict[str, pd.DataFrame],
    groups: dict[int, list[str]],
    valid: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    slices = {}
    cursor = 0
    for story in train_stories:
        slices[story] = np.arange(cursor, cursor + len(frames[story]))
        cursor += len(frames[story])
    retained = np.flatnonzero(valid)
    position = np.full(len(valid), -1, dtype=int)
    position[retained] = np.arange(len(retained))
    splits = []
    for session_stories in groups.values():
        validation = np.concatenate([slices[story] for story in session_stories])
        validation = validation[valid[validation]]
        validation_positions = position[validation]
        train_positions = np.setdiff1d(
            np.arange(len(retained)), validation_positions, assume_unique=True
        )
        splits.append((train_positions, validation_positions))
    return splits


def _fit_fmri_continuous_decoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: LanguageDecodingConfig,
    cv: list[tuple[np.ndarray, np.ndarray]],
):
    """Fit primal ridge CV without materializing sample-by-sample kernels."""
    model = SklearnRidgeCV(
        alphas=np.asarray(config.encoding.ridge_alphas or RIDGE_ALPHAS),
        cv=cv,
        fit_intercept=True,
        scoring="neg_mean_squared_error",
    )
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, np.asarray(y_train).reshape(-1))
    # Normalize sklearn's scalar selection to the shared decoder payload contract.
    model.best_alphas_ = np.atleast_1d(model.alpha_)
    return pipeline


def _prepare_rank_design(
    *,
    aligned,
    train_stories: list[str],
    test_stories: list[str],
    factor: np.ndarray,
    rank: int,
    ndelays: int,
):
    """Return one rank's pre-FIR train/test input and ambient projection."""
    factor = np.asarray(factor, dtype=np.float32)
    ambient_dim = next(iter(aligned.stories.values())).shape[1]
    if factor.shape != (ambient_dim, int(rank)):
        raise ValueError(
            f"Rank {rank} factor has shape {factor.shape}; "
            f"expected {(ambient_dim, int(rank))}"
        )
    train = build_fmri_stimulus(
        aligned, train_stories, ndelays=int(ndelays), basis=factor
    )
    test = build_fmri_stimulus(
        aligned, test_stories, ndelays=int(ndelays), basis=factor
    )
    return train, test, train.pre_fir_stimulus, test.pre_fir_stimulus, factor


def run_fmri_word_decoding(
    config: LanguageDecodingConfig,
    source_config: dict,
    factors: dict[tuple[str, int], np.ndarray],
    output_dir: Path,
) -> Path:
    """Decode held-out fMRI transcript words from projected word embeddings."""
    if config.mode != LanguageDecodingMode.FMRI_WORD:
        raise ValueError(f"Unsupported fMRI word decoding mode: {config.mode}")
    _configure_decoder_backend(config.device)
    fmri = source_config["fmri"]
    train_stories, test_stories, groups = resolve_fmri_stories(
        fmri["fmri_repo_root"], fmri["sessions"]
    )
    if len(test_stories) != 1:
        raise ValueError(f"Expected one held-out fMRI story, got {test_stories}")
    stories = train_stories + test_stories
    frames, _ = load_story_word_data(fmri["fmri_repo_root"], stories)
    embedding_cache = tempfile.TemporaryDirectory(prefix="fmri-word-embeddings-")
    embedding_paths = {}
    ambient_dim = None
    for story in stories:
        embedding = load_story_embeddings(
            fmri["embedding_root"],
            source_config["model_dir"],
            [story],
            source_config["layer"],
            source_config.get("pooling", "mean"),
        )[story]
        if len(embedding) != len(frames[story]):
            raise ValueError(f"Embedding/transcript alignment failed for {story}")
        if ambient_dim is None:
            ambient_dim = embedding.shape[1]
        elif embedding.shape[1] != ambient_dim:
            raise ValueError(
                f"Embedding width differs for {story}: "
                f"{embedding.shape[1]} vs {ambient_dim}"
            )
        path = Path(embedding_cache.name) / f"story-{len(embedding_paths)}.npy"
        np.save(path, embedding)
        embedding_paths[story] = path
    targets = build_word_targets(
        stories=stories,
        frames=frames,
        tasks=config.tasks,
        task_data=config.task_data,
        model_dir=source_config["model_dir"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _word_sample_selection(stories, train_stories, frames).to_csv(
        output_dir / "sample_selection.csv", index=False
    )
    for task, target in targets.items():
        target_dir = output_dir / "target_diagnostics" / task
        target_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target_dir / "availability.npz",
            **{
                story: values.astype(np.uint8)
                for story, values in target.availability.items()
            },
        )
        np.savez_compressed(
            target_dir / "targets.npz",
            **{story: values for story, values in target.values.items()},
        )
        (target_dir / "metadata.json").write_text(json.dumps({
            "task": task,
            "kind": get_language_task(task).kind,
            "classes": target.classes.tolist(),
            "normalization": "none",
            "missing_values": "drop_per_task",
        }, indent=2))

    rows = []
    for (method, rank), factor in sorted(factors.items()):
        factor = np.asarray(factor, dtype=np.float32)
        if factor.shape != (ambient_dim, int(rank)):
            raise ValueError(
                f"{method.upper()} rank {rank} factor has shape {factor.shape}; "
                f"expected {(ambient_dim, int(rank))}"
            )
        # Project before stacking to avoid a second full ambient-dimensional copy
        # of every training story.
        X_train = np.vstack([
            np.asarray(
                np.load(embedding_paths[story], mmap_mode="r") @ factor,
                dtype=np.float32,
            )
            for story in train_stories
        ])
        X_test = np.asarray(
            np.load(embedding_paths[test_stories[0]], mmap_mode="r") @ factor,
            dtype=np.float32,
        )
        for task in config.tasks:
            spec = get_language_task(task)
            target = targets[task]
            y_train = np.concatenate([target.values[story] for story in train_stories])
            y_test = np.asarray(target.values[test_stories[0]])
            train_valid = np.isfinite(y_train)
            test_valid = np.isfinite(y_test)
            if not train_valid.any() or not test_valid.any():
                raise ValueError(f"Task {task} has no valid train or test words")
            cv = _word_session_cv(train_stories, frames, groups, train_valid)
            if spec.kind == "continuous":
                pipeline = _fit_fmri_continuous_decoder(
                    X_train[train_valid], y_train[train_valid], config, cv
                )
            else:
                pipeline = _fit_decoder(
                    X_train[train_valid], y_train[train_valid], spec.kind,
                    config, cv=cv,
                )
            prediction = np.asarray(to_numpy(pipeline.predict(X_test[test_valid])))
            metrics = {
                "task": task,
                "kind": spec.kind,
                "fold": 0,
                "space": "rank",
                "subspace": method,
                "k": int(rank),
                "n_train": int(train_valid.sum()),
                "n_test": int(test_valid.sum()),
                "primary_metric": (
                    "correlation" if spec.kind == "continuous"
                    else "balanced_accuracy"
                ),
                **_decoder_metrics(spec.kind, y_test[test_valid], prediction),
            }
            artifact_dir = (
                output_dir / "tasks" / task / method / f"k{rank}" / "fold_0"
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                artifact_dir / "weights.npz", **_weight_payload(pipeline, factor)
            )
            np.savez_compressed(
                artifact_dir / "heldout_predictions.npz",
                y_true=np.asarray(y_test[test_valid]),
                y_pred=prediction,
                row_index=np.flatnonzero(test_valid),
                block_id=np.repeat(test_stories[0], int(test_valid.sum())),
                classes=target.classes,
            )
            (artifact_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=True)
            )
            rows.append(metrics)
    embedding_cache.cleanup()

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "grid_results.csv", index=False)
    summary = summarize_language_decoding(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    plot_language_decoding(summary, output_dir / "rank_analysis.png")
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "mode": config.mode.value,
        "train_stories": train_stories,
        "test_story": test_stories[0],
        "analysis_unit": "word",
        "input_representation": "pooled_word_embeddings_projected_to_saved_basis",
        "target_representation": "raw_word_aligned_feature",
        "continuous_decoder": "primal_ridge_cv_neg_mean_squared_error",
    }, indent=2))
    return output_dir


def run_fmri_decoding(
    config: LanguageDecodingConfig,
    source_config: dict,
    factors: dict[tuple[str, int], np.ndarray],
    output_dir: Path,
) -> Path:
    """Dispatch one primary fMRI language-decoding mode."""
    if config.mode == LanguageDecodingMode.FMRI_WORD:
        return run_fmri_word_decoding(config, source_config, factors, output_dir)
    if config.mode == LanguageDecodingMode.FMRI_TR_PRE_FIR:
        return run_fmri_tr_decoding(config, source_config, factors, output_dir)
    raise ValueError(f"Unsupported fMRI language-decoding mode: {config.mode}")


def run_fmri_tr_decoding(
    config: LanguageDecodingConfig,
    source_config: dict,
    factors: dict[tuple[str, int], np.ndarray],
    output_dir: Path,
) -> Path:
    """Run one held-out-story fMRI TR decoding sweep into a staging directory."""
    if config.mode != LanguageDecodingMode.FMRI_TR_PRE_FIR:
        raise ValueError(f"Unsupported fMRI TR decoding mode: {config.mode}")
    _configure_decoder_backend(config.device)
    fmri = source_config["fmri"]
    train_stories, test_stories, groups = resolve_fmri_stories(
        fmri["fmri_repo_root"], fmri["sessions"]
    )
    if len(test_stories) != 1:
        raise ValueError(f"Expected one held-out fMRI story, got {test_stories}")
    stories = train_stories + test_stories
    embeddings = load_story_embeddings(
        fmri["embedding_root"],
        source_config["model_dir"],
        stories,
        source_config["layer"],
        source_config.get("pooling", "mean"),
    )
    frames, sequences = load_story_word_data(fmri["fmri_repo_root"], stories)
    for story in stories:
        if len(embeddings[story]) != len(frames[story]):
            raise ValueError(f"Embedding/transcript alignment failed for {story}")
    aligned = align_fmri_stimuli(
        fmri["fmri_repo_root"], stories, embeddings, trim=int(fmri["trim"])
    )
    targets = build_tr_targets(
        stories=stories,
        frames=frames,
        sequences=sequences,
        tasks=config.tasks,
        task_data=config.task_data,
        model_dir=source_config["model_dir"],
        fmri_repo_root=fmri["fmri_repo_root"],
        trim=int(fmri["trim"]),
    )
    for task, target in targets.items():
        for story in stories:
            if len(target.standardized[story]) != len(aligned.stories[story]):
                raise ValueError(
                    f"TR target/input alignment failed for {story}/{task}: "
                    f"{len(target.standardized[story])} vs {len(aligned.stories[story])}"
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    selection = _sample_selection(stories, train_stories, sequences, int(fmri["trim"]))
    selection.to_csv(output_dir / "sample_selection.csv", index=False)
    rows = []
    for task, target in targets.items():
        target_dir = output_dir / "target_diagnostics" / task
        target_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target_dir / "availability.npz",
            **{story: values for story, values in target.availability.items()},
        )
        np.savez_compressed(
            target_dir / "targets.npz",
            **{
                **{
                    f"filtered__{story}": values
                    for story, values in target.filtered.items()
                },
                **{
                    f"standardized__{story}": values
                    for story, values in target.standardized.items()
                },
            },
        )
        (target_dir / "metadata.json").write_text(json.dumps({
            "task": task,
            "kind": get_language_task(task).kind,
            "classes": target.classes.tolist(),
            "normalization": "per_story_zscore",
            "missing_fill": "within_story_finite_mean_or_class_prevalence",
        }, indent=2))

    for (method, rank), factor in sorted(factors.items()):
        try:
            (
                train_prepared,
                test_prepared,
                X_train,
                X_test,
                ambient_factor,
            ) = _prepare_rank_design(
                aligned=aligned,
                train_stories=train_stories,
                test_stories=test_stories,
                factor=factor,
                rank=rank,
                ndelays=int(fmri["ndelays"]),
            )
        except ValueError as error:
            raise ValueError(f"{method.upper()} {error}") from error
        cv = session_cv_splits(train_prepared.story_slices, groups)

        for task in config.tasks:
            spec = get_language_task(task)
            target = targets[task]
            y_train = _stack(train_stories, target.standardized)
            y_test = _stack(test_stories, target.standardized)
            artifact_dir = output_dir / "tasks" / task / method / f"k{rank}" / "fold_0"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            if spec.kind == "continuous":
                pipeline = _fit_fmri_continuous_decoder(
                    X_train, y_train[:, 0], config, cv
                )
                prediction = np.asarray(to_numpy(pipeline.predict(X_test)))
                metrics = _decoder_metrics("continuous", y_test[:, 0], prediction)
                np.savez(artifact_dir / "weights.npz", **_weight_payload(
                    pipeline, ambient_factor
                ))
                prediction_payload = {
                    "y_true": y_test[:, 0],
                    "y_pred": prediction,
                    "row_index": np.arange(len(y_test)),
                    "block_id": np.repeat(test_stories[0], len(y_test)),
                }
            else:
                train_variable = y_train.std(axis=0) > 0
                if not train_variable.any():
                    raise ValueError(
                        f"Categorical task {task} has no variable training channels"
                    )
                ridge = _fit_decoder(
                    X_train,
                    y_train[:, train_variable],
                    "continuous",
                    config,
                    cv=cv,
                )
                variable_prediction = np.asarray(to_numpy(ridge.predict(X_test)))
                if variable_prediction.ndim == 1:
                    variable_prediction = variable_prediction[:, None]
                prediction = np.zeros_like(y_test)
                prediction[:, train_variable] = variable_prediction
                channel_summary, channel_metrics, eligible = _categorical_metrics(
                    y_train, y_test, prediction, target.classes
                )
                raw_train = _stack(train_stories, target.filtered)
                raw_test = _stack(test_stories, target.filtered)
                train_labels = np.argmax(raw_train, axis=1)
                test_labels = np.argmax(raw_test, axis=1)
                classifier_rows, classifier_cv, stable_classes = _stable_classifier_rows(
                    train_labels, cv
                )
                classifier_available = (
                    len(stable_classes) >= 2 and len(classifier_cv) >= 2
                )
                label_valid = (
                    np.isin(test_labels, stable_classes)
                    if classifier_available
                    else np.zeros(len(test_labels), dtype=bool)
                )
                label_prediction = np.array([], dtype=int)
                label_metrics = {
                    "balanced_accuracy": float("nan"),
                    "macro_f1": float("nan"),
                    "accuracy": float("nan"),
                }
                classifier = None
                if classifier_available:
                    classifier = _fit_decoder(
                        X_train[classifier_rows],
                        train_labels[classifier_rows],
                        "categorical",
                        config,
                        cv=classifier_cv,
                    )
                    if label_valid.any():
                        label_prediction = np.asarray(
                            classifier.predict(X_test[label_valid]), dtype=int
                        )
                        label_metrics = _decoder_metrics(
                            "categorical",
                            test_labels[label_valid],
                            label_prediction,
                        )
                metrics = {
                    **channel_summary,
                    "balanced_accuracy": label_metrics["balanced_accuracy"],
                    "macro_f1": label_metrics["macro_f1"],
                    "accuracy": label_metrics["accuracy"],
                    "n_label_test": int(label_valid.sum()),
                    "n_excluded_label_rows": int((~label_valid).sum()),
                    "n_classifier_train": int(len(classifier_rows)),
                    "n_excluded_classifier_train": int(
                        len(train_labels) - len(classifier_rows)
                    ),
                    "classifier_classes": [
                        str(target.classes[index]) for index in stable_classes
                    ],
                    "classifier_available": classifier_available,
                }
                channel_metrics.to_csv(artifact_dir / "channel_metrics.csv", index=False)
                ridge_weights = _weight_payload(ridge, ambient_factor)
                ridge_weights["target_channels"] = np.flatnonzero(train_variable)
                np.savez(artifact_dir / "weights.npz", **ridge_weights)
                if classifier is not None:
                    np.savez(
                        artifact_dir / "classifier_weights.npz",
                        **_weight_payload(classifier, ambient_factor),
                    )
                prediction_payload = {
                    "y_true": y_test,
                    "y_pred": prediction,
                    "filtered_y_true": raw_test,
                    "dominant_y_true": test_labels,
                    "dominant_y_pred": label_prediction,
                    "dominant_row_index": np.flatnonzero(label_valid),
                    "classes": target.classes,
                    "eligible_channels": eligible,
                    "row_index": np.arange(len(y_test)),
                    "block_id": np.repeat(test_stories[0], len(y_test)),
                }
            full_metrics = {
                "task": task,
                "kind": spec.kind,
                "fold": 0,
                "space": "rank",
                "subspace": method,
                "k": int(rank),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "primary_metric": "correlation",
                **metrics,
            }
            np.savez_compressed(
                artifact_dir / "heldout_predictions.npz", **prediction_payload
            )
            (artifact_dir / "metrics.json").write_text(
                json.dumps(full_metrics, indent=2, allow_nan=True)
            )
            rows.append(full_metrics)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "grid_results.csv", index=False)
    summary = summarize_language_decoding(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    plot_language_decoding(summary, output_dir / "rank_analysis.png")
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "mode": config.mode.value,
        "train_stories": train_stories,
        "test_story": test_stories[0],
        "trim": int(fmri["trim"]),
        "fixed_leading_trim": 5,
        "ndelays": int(fmri["ndelays"]),
        "target_alignment": "lanczos_window_3_current_tr",
        "input_representation": (
            "tr_aligned_trimmed_projected_story_zscored_pre_fir"
        ),
        "continuous_decoder": "primal_ridge_cv_neg_mean_squared_error",
    }, indent=2))
    return output_dir
