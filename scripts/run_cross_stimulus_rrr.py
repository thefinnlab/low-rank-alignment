"""Standalone within-subject cross-stimulus orthogonal RRR analysis."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import ClassVar

import numpy as np
import pandas as pd

from src.core.artifacts import ArtifactStore, normalize_model, normalized_config
from src.utils.rrr.cross_stimulus import (
    cache_header,
    discover_subject_movies,
    evaluate_projected_ridge,
    load_stimulus_data,
    ordered_electrode_intersection,
)
from src.utils.rrr.core import fit_full_rrr, orthogonal_rank_factors
from src.utils.rrr.evaluation import fit_pca_input_basis, score_predictions
from src.utils.rrr.inputs import build_target_manifest
from src.utils.shared.encoding import contiguous_folds
from src.utils.shared.config import load_config, parse_args


@dataclass
class Config:
    """Configuration for one model/layer cross-stimulus study."""

    STRICT_CONFIG: ClassVar[bool] = True

    data_root: str = "data/brain-treebank"
    results_root: str = "results"
    model: str = "transformer"
    per_model_config: dict = field(
        default_factory=lambda: {"model_dir": "gpt2", "layer": 12}
    )
    subjects: list[int] | None = None
    movies: dict = field(default_factory=dict)
    offset: int = 0
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    num_folds: int = 5
    rereference: str = "laplacian"
    ridge_alphas: list[float] | None = None
    ridge_cv_folds: int = 5
    ridge_solver_params: dict | None = None
    ridge_y_in_cpu: bool | None = None
    ridge_force_cpu: bool | None = None
    alpha_grid: dict = field(default_factory=dict)
    log_alphas: bool = False
    use_low_rank_svd: bool = True
    low_rank_svd_seed: int = 0
    variant_name: str | None = None
    refresh: bool = False

    def validate(self):
        if self.rereference != "laplacian":
            raise ValueError("Cross-stimulus RRR currently requires rereference='laplacian'")
        if self.num_folds != 5:
            raise ValueError("Cross-stimulus RRR currently requires five target folds")
        if not self.rank_grid or any(int(rank) < 1 for rank in self.rank_grid):
            raise ValueError("rank_grid must contain positive integers")
        if len(set(map(int, self.rank_grid))) != len(self.rank_grid):
            raise ValueError("rank_grid cannot contain duplicate ranks")
        if self.ridge_cv_folds < 2:
            raise ValueError("ridge_cv_folds must be at least two")
        layer = self.per_model_config.get("layer")
        if layer is None and self.model != "glove":
            raise ValueError("per_model_config.layer is required")
        if not self.per_model_config.get("model_dir") and self.model != "glove":
            raise ValueError("per_model_config.model_dir is required")


class Artifact(ArtifactStore):
    """Storage contract owned only by the cross-stimulus producer."""

    SCRIPT = "run_cross_stimulus_rrr"
    SELECTOR_ORDER = (
        "subject", "source_stimulus", "target_stimulus", "method", "rank", "fold",
    )
    FIELD_CONTRACT = {
        "effective_config": True,
        "eligibility": True,
        "exclusions": True,
        "target_manifest": False,
        "source_factors": False,
        "rank_metrics": False,
        "fold_results": True,
        "summary": True,
        "diagnostics": False,
    }

    def __init__(self, *, model="gpt2", layer=12, offset=0, variant=None,
                 results_root="results"):
        self._init_store(
            mode="cross_stimulus", dataset="brain-treebank", offset=offset,
            model=normalize_model(model), layer=layer, variant=variant,
            results_root=results_root,
        )

    @classmethod
    def from_config(cls, config: Config):
        model = config.per_model_config.get("model_dir", config.model)
        model = normalize_model(model, glove_dim=config.per_model_config.get("dim"))
        layer = None if config.model == "glove" else config.per_model_config.get("layer")
        digest = hashlib.sha256(
            json.dumps(normalized_config(config), sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        variant = config.variant_name or f"rereference-{config.rereference}-{digest}"
        return cls(model=model, layer=layer, offset=config.offset, variant=variant,
                   results_root=config.results_root)


def _ridge_kwargs(config: Config) -> dict:
    return {
        "alphas": config.ridge_alphas,
        "alpha_policy": config.alpha_grid,
        "ridge_cv_folds": config.ridge_cv_folds,
        "ridge_solver_params": config.ridge_solver_params,
        "ridge_y_in_cpu": config.ridge_y_in_cpu,
        "ridge_force_cpu": config.ridge_force_cpu,
        "use_low_rank_svd": config.use_low_rank_svd,
        "low_rank_svd_seed": config.low_rank_svd_seed,
    }


def _metric_payload(result: dict, *, log_alphas: bool) -> dict:
    """Drop sample predictions while retaining primitive per-target metrics."""
    payload = {}
    for key, value in result.items():
        if key.endswith("Y_hat") or key == "Y_hat":
            continue
        if not log_alphas and key.endswith("best_alphas"):
            continue
        payload[key] = value
    return payload


def _metric_row(selector: dict, payload: dict, source_words: int, target_words: int) -> dict:
    return {
        **selector,
        "source_words": int(source_words), "target_words": int(target_words),
        "corr": float(np.asarray(payload["corr"])),
        "r2": float(np.asarray(payload["r2"])),
    }


def _selected_movies(config: Config) -> dict[int, list[str]]:
    discovered = discover_subject_movies(config.data_root, config.subjects)
    if not config.movies:
        return discovered
    selected = {}
    for subject, available in discovered.items():
        requested = config.movies.get(str(subject), config.movies.get(subject, available))
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"Subject {subject} does not have requested movies {unknown}")
        if len(requested) >= 2:
            selected[subject] = list(requested)
    return selected


def _load_subject(config: Config, subject: int, movies: list[str]):
    """Load every viable movie, recomputing the contact intersection after failures."""
    exclusions = []
    headers = {}
    for movie in movies:
        try:
            headers[movie] = cache_header(
                config.data_root, subject, movie, config.rereference
            )
        except Exception as error:
            exclusions.append({
                "subject": subject, "stimulus": movie, "stage": "cache_header",
                "reason": f"{type(error).__name__}: {error}",
            })
    while len(headers) >= 2:
        electrodes = ordered_electrode_intersection(headers)
        if not electrodes:
            exclusions.append({
                "subject": subject, "stimulus": "*", "stage": "electrodes",
                "reason": "No usable Laplacian contact is common to all movies",
            })
            return {}, [], exclusions
        loaded, failures = {}, []
        for movie in headers:
            try:
                loaded[movie] = load_stimulus_data(
                    data_root=config.data_root, subject=subject, stimulus=movie,
                    rereference=config.rereference, model=config.model,
                    per_model_config=deepcopy(config.per_model_config), offset=config.offset,
                    electrodes=electrodes,
                )
            except Exception as error:
                failures.append(movie)
                exclusions.append({
                    "subject": subject, "stimulus": movie, "stage": "data_loading",
                    "reason": f"{type(error).__name__}: {error}",
                })
        if not failures:
            return loaded, electrodes, exclusions
        for movie in failures:
            headers.pop(movie, None)
    return {}, [], exclusions


def _save_metric(artifact: Artifact, selector: dict, result: dict, refresh: bool,
                 log_alphas: bool):
    payload = _metric_payload(result, log_alphas=log_alphas)
    artifact.save_field(
        "rank_metrics", payload, selector=selector, overwrite=refresh
    )
    return payload


def _run_subject(config: Config, artifact: Artifact, subject: int, movies: list[str]):
    data, electrodes, exclusions = _load_subject(config, subject, movies)
    if len(data) < 2:
        exclusions.append({
            "subject": subject, "stimulus": "*", "stage": "eligibility",
            "reason": "Fewer than two fully loadable movies remain",
        })
        return [], [], exclusions, []

    ordered_movies = list(data)
    reference = data[ordered_movies[0]]
    artifact.save_field(
        "target_manifest", build_target_manifest(
            [subject], {subject: electrodes}, [electrodes], reference.lags
        ), selector={"subject": subject},
        overwrite=config.refresh,
    )
    max_rank = max(map(int, config.rank_grid))
    for movie, item in data.items():
        feasible = min(item.X.shape[0], item.X.shape[1], item.Y.shape[1])
        if max_rank > feasible:
            raise ValueError(
                f"Subject {subject} movie {movie!r} cannot estimate rank {max_rank}; "
                f"maximum is {feasible} from X={item.X.shape}, Y={item.Y.shape}"
            )

    ridge_kwargs = _ridge_kwargs(config)
    source_models = {}
    for movie, item in data.items():
        model = fit_full_rrr(
            item.X, item.Y, max_rank,
            alpha_context={"script": "run_cross_stimulus_rrr", "role": "source_full"},
            **ridge_kwargs,
        )
        model["pca_Q"] = fit_pca_input_basis(item.X, max_rank)
        model["rank_factors"] = {}
        source_models[movie] = model
        base_selector = {
            "subject": subject, "source_stimulus": movie,
        }
        artifact.save_field(
            "source_factors", model["A_raw"],
            selector={**base_selector, "method": "rrr"},
            overwrite=config.refresh,
        )
        for rank in map(int, config.rank_grid):
            factors = orthogonal_rank_factors(
                model["A_raw"], model["B_raw"], rank
            )
            model["rank_factors"][rank] = factors

    rows = []
    for target_movie, target in data.items():
        folds = contiguous_folds(len(target.X), config.num_folds)
        for fold, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = target.X[train_idx], target.X[test_idx]
            raw_train, raw_test = target.Y[train_idx], target.Y[test_idx]
            oracle = fit_full_rrr(
                X_train, raw_train, max_rank, X_test=X_test, Y_test=raw_test,
                alpha_context={
                    "script": "run_cross_stimulus_rrr", "role": "source_full"
                },
                **ridge_kwargs,
            )
            Y_train, Y_test = oracle["Y_train"], oracle["Y_test"]
            oracle_pca = fit_pca_input_basis(X_train, max_rank)
            full_prediction = oracle["full_prediction_test"]
            full_score = score_predictions(Y_test, full_prediction)
            full_selector = {
                "subject": subject, "source_stimulus": target_movie,
                "target_stimulus": target_movie, "method": "target_full_ridge",
                "fold": fold,
            }
            payload = _save_metric(
                artifact, full_selector, full_score, config.refresh, config.log_alphas
            )
            rows.append(_metric_row(full_selector, payload, len(target.X), len(target.X)))

            for rank in map(int, config.rank_grid):
                oracle_factors = orthogonal_rank_factors(
                    oracle["A_raw"], oracle["B_raw"], rank
                )
                for method, Q in (
                    ("within_rrr", oracle_factors["Q"]),
                    ("within_pca", oracle_pca),
                ):
                    result = evaluate_projected_ridge(
                        Q, rank, X_train, X_test, Y_train, Y_test, **ridge_kwargs
                    )
                    selector = {
                        "subject": subject, "source_stimulus": target_movie,
                        "target_stimulus": target_movie, "method": method,
                        "rank": rank, "fold": fold,
                    }
                    payload = _save_metric(
                        artifact, selector, result, config.refresh, config.log_alphas
                    )
                    rows.append(_metric_row(
                        selector, payload, len(target.X), len(target.X)
                    ))

                for source_movie, source in data.items():
                    if source_movie == target_movie:
                        continue
                    source_model = source_models[source_movie]
                    source_factors = source_model["rank_factors"][rank]
                    for method, Q in (
                        ("transfer_rrr_basis", source_factors["Q"]),
                        ("transfer_pca_basis", source_model["pca_Q"]),
                    ):
                        result = evaluate_projected_ridge(
                            Q, rank, X_train, X_test, Y_train, Y_test, **ridge_kwargs
                        )
                        selector = {
                            "subject": subject, "source_stimulus": source_movie,
                            "target_stimulus": target_movie, "method": method,
                            "rank": rank, "fold": fold,
                        }
                        payload = _save_metric(
                            artifact, selector, result, config.refresh, config.log_alphas
                        )
                        rows.append(_metric_row(
                            selector, payload, len(source.X), len(target.X)
                        ))

                    B_raw = source_factors["B_raw"]
                    B_orth = source_factors["B_orth"]
                    intercept = source_model["ridge_intercept"] @ B_raw @ B_raw.T
                    prediction = (
                        X_test @ source_factors["Q"] @ B_orth.T + intercept
                    )
                    result = score_predictions(Y_test, prediction)
                    selector = {
                        "subject": subject, "source_stimulus": source_movie,
                        "target_stimulus": target_movie, "method": "transfer_rrr_full",
                        "rank": rank, "fold": fold,
                    }
                    payload = _save_metric(
                        artifact, selector, result, config.refresh, config.log_alphas
                    )
                    rows.append(_metric_row(
                        selector, payload, len(source.X), len(target.X)
                    ))

    eligibility = [{
        "subject": subject, "stimulus": movie, "n_words": len(item.X),
        "n_common_electrodes": len(electrodes), "n_lags": len(item.lags),
        "eligible": True,
    } for movie, item in data.items()]
    return rows, eligibility, exclusions, ordered_movies


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = [
        "subject", "source_stimulus", "target_stimulus", "method", "rank",
        "source_words", "target_words",
    ]
    summary = frame.groupby(keys, as_index=False, dropna=False)[["corr", "r2"]].mean()
    oracle = summary[summary["method"] == "within_rrr"][
        ["subject", "target_stimulus", "rank", "corr", "r2"]
    ].rename(columns={"corr": "oracle_corr", "r2": "oracle_r2"})
    summary = summary.merge(
        oracle, on=["subject", "target_stimulus", "rank"], how="left"
    )
    summary["target_oracle_corr_gap"] = summary["corr"] - summary["oracle_corr"]
    summary["target_oracle_r2_gap"] = summary["r2"] - summary["oracle_r2"]
    return summary


def run_config(config: Config):
    config.validate()
    layers = config.per_model_config.get("layer")
    if isinstance(layers, list):
        outputs = []
        for layer in layers:
            current = replace(config, per_model_config={**config.per_model_config, "layer": layer})
            outputs.extend(run_config(current))
        return outputs

    artifact = Artifact.from_config(config)
    artifact.begin(config=config, refresh=config.refresh)
    if artifact.complete and not config.refresh:
        print(f"Artifact complete; reusing {artifact.identity_record()}")
        return [artifact]
    artifact.save_field("effective_config", asdict(config), overwrite=config.refresh)

    all_rows, eligibility, exclusions = [], [], []
    all_discovered = discover_subject_movies(
        config.data_root, config.subjects, min_movies=1
    )
    selected = _selected_movies(config)
    for subject, movies in all_discovered.items():
        if subject not in selected:
            exclusions.append({
                "subject": subject, "stimulus": movies[0] if movies else "*",
                "stage": "eligibility",
                "reason": "Subject has fewer than two selected movies",
            })
    for subject, movies in selected.items():
        rows, eligible, excluded, _ = _run_subject(
            config, artifact, subject, movies
        )
        all_rows.extend(rows)
        eligibility.extend(eligible)
        exclusions.extend(excluded)

    eligibility_frame = pd.DataFrame(eligibility, columns=[
        "subject", "stimulus", "n_words", "n_common_electrodes", "n_lags", "eligible",
    ])
    exclusion_frame = pd.DataFrame(
        exclusions, columns=["subject", "stimulus", "stage", "reason"]
    )
    fold_frame = pd.DataFrame(all_rows)
    summary = _summary(fold_frame)
    artifact.save_field("eligibility", eligibility_frame, overwrite=config.refresh)
    artifact.save_field("exclusions", exclusion_frame, overwrite=config.refresh)
    artifact.save_field("fold_results", fold_frame, overwrite=config.refresh)
    artifact.save_field("summary", summary, overwrite=config.refresh)
    if fold_frame.empty:
        raise ValueError(
            "Cross-stimulus preflight found no subject with at least two fully "
            "loadable movies; inspect the saved exclusions table"
        )
    artifact.mark_complete()
    print(f"Artifact: {artifact.identity_record()}")
    return [artifact]


def main():
    args, overrides = parse_args(__doc__)
    config = load_config(args.config, overrides, Config)
    run_config(config)


if __name__ == "__main__":
    main()
