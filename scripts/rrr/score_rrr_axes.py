"""Score individual internal-RRR axes against interpretable word features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.embeddings import load_embeddings
from src.utils.encoding import load_encoding_results
from src.utils.language_tasks import TASK_REGISTRY


DATASETS = {
    "podcast": {
        "manifest": Path("configs/language_decoding/internal/podcast.yml"),
        "result_glob": "results/*_layer*_offset0/rrr",
    },
    "cars-2": {
        "manifest": Path("configs/language_decoding/internal/cars-2.yml"),
        "result_glob": "results/*_layer*_offset0/movie-cars-2/rrr",
    },
}


def _saturation_rank(dataset: str, model: str, layer: int) -> int:
    paths = list(
        Path("results/rank_saturation", dataset, "internal_rrr", model, f"layer{layer}")
        .glob("*__rrr_curve.csv")
    )
    if len(paths) != 1:
        raise ValueError(
            f"Expected one internal RRR saturation curve for {dataset}/{model}/layer{layer}, "
            f"found {paths}"
        )
    values = pd.read_csv(paths[0])["saturation_rank"].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"Ambiguous saturation rank in {paths[0]}: {values}")
    return int(values[0])


def _load_run(unit_dir: Path, dataset_name: str):
    config = yaml.safe_load((unit_dir / "config.yml").read_text())
    enc = config["encoding"]
    per_model = dict(enc["per_model_config"])
    layer = int(per_model["layer"])
    model = str(per_model["model_dir"])
    embeddings, words = load_embeddings(
        enc["data_root"],
        enc["model"],
        dataset=enc["dataset"],
        per_dataset_config=enc.get("per_dataset_config"),
        **per_model,
    )
    manifest = yaml.safe_load(DATASETS[dataset_name]["manifest"].read_text())
    feature_path = Path(manifest["task_data"]["language_features"])
    features = pd.read_csv(
        feature_path, sep="\t", keep_default_na=False, na_values=[""]
    )
    if len(features) != len(embeddings) or not np.array_equal(
        pd.to_numeric(features["token_id"], errors="coerce").to_numpy(),
        np.arange(len(features)),
    ):
        raise ValueError(
            f"Feature rows in {feature_path} do not align with {len(embeddings)} embeddings"
        )
    targets = {}
    for name, spec in TASK_REGISTRY.items():
        if name not in features:
            continue
        if spec.kind == "continuous":
            target = pd.to_numeric(features[name], errors="coerce").to_numpy(float)
        else:
            present = features[name].notna()
            target = np.full(len(features), np.nan)
            labels = features.loc[present, name].astype(str)
            categories = {value: i for i, value in enumerate(sorted(labels.unique()))}
            target[present.to_numpy()] = labels.map(categories).to_numpy(float)
        targets[name] = (spec, target)
    result_root = unit_dir.parent
    result_path = sorted(
        (result_root / "subjects").glob("sub_*/encoding_results.joblib")
    )[0]
    encoding = load_encoding_results(str(result_path))
    selection = np.asarray(encoding["word_selection"]["analysis_selection"], dtype=int)
    return model, layer, embeddings[selection], selection, encoding["test_idxs"], targets


def _continuous_score(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _categorical_score(x_train, y_train, x_test, y_test) -> float:
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or len(np.unique(y_test)) < 2:
        return float("nan")
    cv = min(5, int(counts.min()))
    if cv < 2:
        return float("nan")
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 9),
            cv=cv,
            scoring="balanced_accuracy",
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            n_jobs=1,
            l1_ratios=(0.0,),
            use_legacy_attributes=False,
        ),
    )
    pipeline.fit(x_train[:, None], y_train)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true"
        )
        return float(
            balanced_accuracy_score(y_test, pipeline.predict(x_test[:, None]))
        )


def score_unit(unit_dir: Path, dataset: str) -> list[dict]:
    model, layer, X, selection, test_idxs, targets = _load_run(unit_dir, dataset)
    saturation = _saturation_rank(dataset, model, layer)
    all_rows = np.arange(len(X), dtype=int)
    rows = []
    for fold, test_idx in enumerate(test_idxs):
        test_idx = np.asarray(test_idx, dtype=int)
        train_idx = np.setdiff1d(all_rows, test_idx, assume_unique=False)
        A = np.load(unit_dir / f"fold_{fold}" / "A.npy", mmap_mode="r")
        width = min(saturation, A.shape[1])
        scores = np.asarray(X @ A[:, :width], dtype=np.float32)
        for task_name, (spec, target_full) in targets.items():
            target = np.asarray(target_full)[selection]
            train_valid = train_idx[np.isfinite(target[train_idx])]
            test_valid = test_idx[np.isfinite(target[test_idx])]
            if len(train_valid) < 2 or len(test_valid) < 2:
                continue
            for axis in range(width):
                if spec.kind == "continuous":
                    signed = _continuous_score(
                        scores[test_valid, axis], target[test_valid]
                    )
                    score = abs(signed)
                    metric = "absolute_pearson_r"
                else:
                    signed = float("nan")
                    score = _categorical_score(
                        scores[train_valid, axis],
                        target[train_valid],
                        scores[test_valid, axis],
                        target[test_valid],
                    )
                    metric = "balanced_accuracy"
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "layer": layer,
                        "unit_dir": str(unit_dir),
                        "fold": fold,
                        "saturation_rank": saturation,
                        "axis": axis + 1,
                        "feature": task_name,
                        "feature_label": spec.display_name,
                        "kind": spec.kind,
                        "metric": metric,
                        "score": score,
                        "signed_correlation": signed,
                        "n_train": len(train_valid),
                        "n_test": len(test_valid),
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/axis_readout/scores.csv"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    units = [
        (dataset, path)
        for dataset in args.datasets
        for path in sorted(Path().glob(DATASETS[dataset]["result_glob"]))
        if list(path.glob("fold_*/A.npy"))
    ]
    if args.limit is not None:
        units = units[: args.limit]
    output_groups = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(score_unit)(unit, dataset) for dataset, unit in units
    )
    output = [row for group in output_groups for row in group]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(args.output, index=False)
    print(f"Saved {len(output)} scores to {args.output}")


if __name__ == "__main__":
    main()
