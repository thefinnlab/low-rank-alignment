"""Multiview iEEG RRR mode."""

import numpy as np

from scripts.run_rrr import Artifact, UnitSelector
from src.core.config import RRRConfig
from src.core.registry import resolve_dataset_context
from src.utils.rrr.multiview import fit_banded_ridge_baseline, project_view_bands
from src.utils.rrr.significance import PreparedSignificance
from src.utils.shared.alpha_grids import resolve_ridge_alphas
from src.utils.shared.embeddings import get_glove_embeddings, load_embeddings
from src.utils.shared.tensors import to_numpy
from src.utils.subspace_comparison.metrics import aggregate_subspace_basis

from .common import (_build_rrr_decomposition_inputs,
                     _load_shared_word_selection_metadata,
                     _reference_encoding_results, _rrr_output_resume_plan,
                     _run_rrr_for_unit)


def run_multi_view(
    config: RRRConfig,
    artifact: Artifact,
    significance: PreparedSignificance,
    subject_ids: list[int],
) -> list[tuple[UnitSelector, int]]:
    """Fit banded ridge over saved RRR views, then run the standard RRR sweep."""
    from sklearn.preprocessing import StandardScaler

    sources = {}
    source_configs = {}
    for view in config.views:
        kwargs = dict(view.ieeg_rrr_source or view.fmri_rrr_source or {})
        kwargs.setdefault("results_root", config.results_root)
        source = Artifact(**kwargs)
        loaded = source.load_artifacts(
            fields=["effective_config", "upstream_identities", "target_manifest"]
        )
        sources[view.name] = loaded
        source_configs[view.name] = loaded.effective_config

    sig_frame = significance.significant_electrodes
    union = {
        int(subject): group["elec"].astype(str).tolist()
        for subject, group in sig_frame.groupby("subject")
    }
    significance_sources = [significance.reference.identity_record()]
    for identity in significance.source_identities:
        if identity not in significance_sources:
            significance_sources.append(identity)
    target_context = resolve_dataset_context(
        dataset=config.encoding.dataset, subject_ids=subject_ids,
        per_dataset_config=config.encoding.per_dataset_config,
        data_root=config.encoding.data_root,
    )
    embeddings = {}
    word_df = None
    for view in config.views:
        view_enc = source_configs[view.name]["encoding"]
        if int(view_enc.get("offset", 0)) != int(config.encoding.offset):
            raise ValueError(
                f"View {view.name} uses offset {view_enc.get('offset')}; "
                f"target uses {config.encoding.offset}"
            )
        values, words = load_embeddings(
            config.encoding.data_root, view_enc["model"],
            dataset=config.encoding.dataset,
            per_dataset_config=config.encoding.per_dataset_config,
            dataset_context=target_context, **view_enc.get("per_model_config", {}),
        )
        if word_df is not None and not np.array_equal(
            word_df["word"].astype(str).to_numpy(), words["word"].astype(str).to_numpy()
        ):
            raise ValueError(f"Embedding transcript differs for view {view.name}")
        word_df = words
        embeddings[view.name] = np.asarray(values)

    enc = config.encoding
    word_in_glove = None
    if enc.limit_to_glove or enc.model == "glove":
        word_in_glove = get_glove_embeddings(
            enc.data_root, word_df, **enc.per_model_config
        )[1]["in_glove"]
    shared_selection_metadata = None
    if enc.shared_word_selection:
        shared_selection_metadata = _load_shared_word_selection_metadata(
            significance, subject_ids
        )

    reference = _reference_encoding_results(significance, subject_ids)
    prepared_values = _build_rrr_decomposition_inputs(
        subject_ids,
        union,
        reference,
        significance.detailed_significance,
        enc,
        embeddings[config.views[0].name],
        word_df,
        compute_sig_nonsig=config.compute_sig_nonsig,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
    )
    if prepared_values is None:
        return []
    _, prepared_baselines, test_idxs, sig_mask_flat, target_manifest = prepared_values
    selection = np.asarray(reference["selection"])
    bases, basis_modes = {}, {}
    for view in config.views:
        source = sources[view.name]
        rank = int(view.rank)
        factor_items = source.load_collection("input_factors", {"method": "rrr"})
        fold_factors = {int(dict(key)["fold"]): value for key, value in factor_items.items()
                        if "fold" in dict(key)}
        if source.dataset == artifact.dataset:
            if len(fold_factors) != len(test_idxs):
                raise ValueError(f"View {view.name} does not provide all target folds")
            bases[view.name] = {fold: np.asarray(value)[:, :rank]
                                for fold, value in fold_factors.items()}
            basis_modes[view.name] = "matching_target_fold_raw"
        else:
            mean_basis = aggregate_subspace_basis(
                list(fold_factors.values()), rank=rank,
                method=config.subspace_aggregation,
            )
            bases[view.name] = {fold: mean_basis for fold in range(len(test_idxs))}
            basis_modes[view.name] = f"cross_dataset_fold_{config.subspace_aggregation}"
    view_names = [view.name for view in config.views]
    n_words = len(selection)
    all_rows = np.arange(n_words, dtype=int)
    widths = [int(view.rank) for view in config.views]
    stops = np.cumsum(widths)
    band_boundaries = list(zip([0, *stops[:-1].tolist()], stops.tolist()))
    empty_design = np.empty((n_words, int(stops[-1])), dtype=np.float32)
    fold_designs: list[np.ndarray] = [empty_design] * len(test_idxs)
    baselines: list[dict | None] = [None] * len(test_idxs)
    unit = UnitSelector(method="rrr")
    artifact.save_target_manifest(unit=unit, target_manifest=target_manifest,
                                  overwrite=config.refresh)
    resume_plan = _rrr_output_resume_plan(
        artifact, unit, config, len(test_idxs)
    )
    folds_to_build = resume_plan["folds_to_build"]
    explicit_banded_alphas = (
        np.asarray(config.banded_ridge_alphas, dtype=float)
        if config.banded_ridge_alphas is not None
        else None
    )

    for fold_f, test_idx in enumerate(test_idxs):
        if fold_f not in folds_to_build:
            continue
        test_idx = np.asarray(test_idx, dtype=int)
        train_idx = np.setdiff1d(all_rows, test_idx, assume_unique=False)
        bands, boundaries = project_view_bands(
            embeddings, bases, view_names, fold_f
        )
        if band_boundaries != boundaries:
            raise ValueError("Multiview band boundaries differ across folds")
        X_full = np.concatenate(bands, axis=1).astype(np.float32, copy=False)
        fold_designs[fold_f] = X_full

        raw_targets = prepared_baselines[fold_f]["Y_train_raw"] if "Y_train_raw" in prepared_baselines[fold_f] else None
        baseline = prepared_baselines[fold_f]
        Y_train = baseline["Y_train"]
        Y_test = baseline["Y_test"]
        if raw_targets is not None:
            target_scaler = StandardScaler().fit(raw_targets)
            Y_train = target_scaler.transform(raw_targets).astype(
            np.float32
            )
        alpha_context = {
            "script": "run_rrr",
            "role": "multi_view_banded_full",
            "fold": int(fold_f),
            "n_features": int(X_full.shape[1]),
            "bands": view_names,
            "standardize": False,
        }
        banded_alphas = resolve_ridge_alphas(
            explicit_banded_alphas,
            X_full[train_idx],
            standardize=False,
            context=alpha_context,
            cache_dir=None,
            policy=enc.alpha_grid,
        )
        fitted = fit_banded_ridge_baseline(
            [band[train_idx] for band in bands],
            [band[test_idx] for band in bands],
            Y_train,
            alphas=banded_alphas,
            ridge_cv_folds=enc.ridge_cv_folds,
            solver_params=config.banded_ridge_solver_params,
            y_in_cpu=enc.ridge_y_in_cpu,
            force_cpu=enc.ridge_force_cpu,
        )
        artifact.save_diagnostics(unit=unit, fold=fold_f, diagnostics={
            "view_names": view_names, "band_boundaries": boundaries,
            "alphas": banded_alphas, "deltas": fitted.deltas,
            "kernel_weights": np.exp(fitted.deltas),
            "best_alphas": fitted.best_alphas, "intercept": fitted.intercept,
        }, overwrite=config.refresh)
        baselines[fold_f] = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "Y_train": Y_train,
            "Y_test": Y_test,
            "Y_hat_full_train": fitted.Y_hat_train,
            "Y_hat_full_test": fitted.Y_hat_test,
            "beta_full_raw": fitted.beta_full_raw,
        }

    metadata = {
        "mode": "multi_view",
        "target_dataset": artifact.dataset,
        "views": view_names,
        "view_ranks": {view.name: int(view.rank) for view in config.views},
        "basis_modes": basis_modes,
        "subspace_aggregation": config.subspace_aggregation,
        "band_boundaries": band_boundaries,
        "significance_sources": significance_sources,
    }
    artifact.save_field("mode_metadata", metadata, selector={"kind": "multi_view"},
                        overwrite=config.refresh)

    print(
        f"  Multiview X: {(n_words, int(stops[-1]))}, "
        f"bands: {dict(zip(view_names, [b - a for a, b in band_boundaries]))}"
    )
    _run_rrr_for_unit(
        artifact=artifact,
        unit=unit,
        X_raw=fold_designs,
        test_idxs=prepared.test_idxs,
        per_fold_baselines=baselines,
        config=config,
        sig_mask_flat=sig_mask_flat,
        ranks_by_fold=resume_plan["rrr_ranks_by_fold"],
        completed_pairs=resume_plan["rrr_completed_pairs"],
        pca_ranks_by_fold=resume_plan["pca_ranks_by_fold"],
        pca_completed_pairs=resume_plan["pca_completed_pairs"],
    )
    return [(unit, len(test_idxs))]


__all__ = ["run_multi_view"]
