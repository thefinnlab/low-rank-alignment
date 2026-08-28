"""Find fMRI voxels predicted significantly in session-held-out encoding models."""

import json
import io
from dataclasses import asdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from himalaya.backend import set_backend

from src.core.config import FMRISignificanceConfig
from src.utils.shared.alpha_grids import resolve_ridge_alphas
from src.utils.shared.encoding import extract_best_alphas, get_ridge_model_constructor
from src.utils.shared.fmri import (
    finite_response_voxel_masks,
    load_story_embeddings,
    prepare_fmri_data,
    remap_inner_splits,
    resolve_fmri_stories,
    session_cv_splits,
    TEMPORARILY_HELD_OUT_FMRI_STORIES,
)
from src.utils.significance.fmri import bh_fdr, circular_shift_pvalues
from src.utils.shared.tensors import to_numpy


def _fit_oof(X, Y, story_slices, groups, config):
    predictions = np.zeros_like(Y, dtype=np.float32)
    best_alphas = []
    outer_folds = session_cv_splits(story_slices, groups)
    explicit = np.asarray(config.ridge.alphas) if config.ridge.alphas else None
    sessions = list(groups)
    for fold, ((train_idx, test_idx), held_out_session) in tqdm(enumerate(
        zip(outer_folds, sessions)
    ), desc="folds"):
        context = {
            "script": "fmri_significance",
            "fold": fold,
            "held_out_session": held_out_session,
            "n_features": int(X.shape[1]),
            "standardize": False,
        }
        alphas = resolve_ridge_alphas(
            explicit,
            X[train_idx],
            standardize=False,
            context=context,
            cache_dir=None,
            policy=config.ridge.alpha_grid,
        )
        inner = remap_inner_splits(
            train_idx, story_slices, groups, held_out_session
        )
        model = get_ridge_model_constructor(
            alphas=alphas,
            cv=inner,
            standardize=False,
            solver_params=config.ridge.solver_params,
            y_in_cpu=config.ridge.y_in_cpu,
            force_cpu=config.ridge.force_cpu,
        )(fold)
        model.fit(X[train_idx], Y[train_idx])
        predictions[test_idx] = to_numpy(model.predict(X[test_idx])).astype(np.float32)
        selected = extract_best_alphas(model)
        best_alphas.append(selected)
    return predictions, best_alphas


def _acf_diagnostic_figure(true, predicted, story_slices, observed):
    voxel = int(np.nanargmax(observed))
    max_lag = 30

    def acf(values):
        values = values - values.mean()
        denom = np.dot(values, values)
        return np.asarray(
            [1.0 if lag == 0 else np.dot(values[:-lag], values[lag:]) / denom
             for lag in range(max_lag + 1)]
        )

    true_acfs, pred_acfs, null_acfs = [], [], []
    for part in story_slices.values():
        y = true[part, voxel]
        p = predicted[part, voxel]
        true_acfs.append(acf(y))
        pred_acfs.append(acf(p))
        null_acfs.append(acf(np.roll(p, min(10, max(1, len(p) // 3)))))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.mean(true_acfs, axis=0), label="response")
    ax.plot(np.mean(pred_acfs, axis=0), label="prediction")
    ax.plot(np.mean(null_acfs, axis=0), "--", label="shifted prediction")
    ax.set(xlabel="Lag (TR)", ylabel="Autocorrelation", title=f"Voxel {voxel} ACF")
    ax.legend()
    fig.tight_layout()
    return fig


def _analyze_subject_block(subjects, config, embeddings, train_stories, groups, voxel_masks):
    prepared = prepare_fmri_data(
        config.fmri.fmri_repo_root,
        train_stories,
        embeddings,
        trim=config.fmri.trim,
        ndelays=config.fmri.ndelays,
        subjects=subjects,
        voxel_masks=voxel_masks,
    )
    predictions, best_alphas = _fit_oof(
        prepared.stimulus,
        prepared.responses,
        prepared.story_slices,
        groups,
        config,
    )
    return prepared, predictions, best_alphas


def run_fmri(config: FMRISignificanceConfig):
    from scripts.run_significance import Artifact
    config.validate()
    if torch.cuda.is_available():
        set_backend("torch_cuda")
    artifact = Artifact.from_fmri_config(config).begin(config=config, refresh=config.refresh)
    train_stories, test_stories, groups = resolve_fmri_stories(
        config.fmri.fmri_repo_root, config.fmri.sessions
    )
    embeddings = load_story_embeddings(
        config.fmri.embedding_root,
        config.model_dir,
        train_stories,
        config.layer,
        config.pooling,
    )
    voxel_masks, excluded_by_story = finite_response_voxel_masks(
        config.fmri.fmri_repo_root, config.fmri.subjects, train_stories
    )
    retained_counts = {
        subject: int(mask.sum()) for subject, mask in voxel_masks.items()
    }
    subject_fields = ("target_significance", "significance_details")
    completed_subjects = {
        subject for subject in config.fmri.subjects
        if not config.refresh and all(
            artifact.has_field(field, selector={"subject": str(subject)})
            for field in subject_fields
        )
    }
    pending_subjects = [s for s in config.fmri.subjects if s not in completed_subjects]
    for subject, mask in voxel_masks.items():
        print(
            f"{subject}: excluding {int((~mask).sum()):,}/{len(mask):,} "
            "voxels with non-finite responses"
        )
    subject_payloads = {}
    if config.batch_subjects:
        for subject in tqdm(pending_subjects, desc="Running subjects"):
            prepared, predictions, best_alphas = _analyze_subject_block(
                [subject], config, embeddings, train_stories, groups, voxel_masks
            )
            subject_payloads[subject] = (
                prepared.responses,
                predictions,
                prepared.story_slices,
                best_alphas,
            )
    elif pending_subjects:
        prepared, predictions, best_alphas = _analyze_subject_block(
            config.fmri.subjects, config, embeddings, train_stories, groups,
            voxel_masks,
        )
        cursor = 0
        for subject in config.fmri.subjects:
            stop = cursor + retained_counts[subject]
            subject_payloads[subject] = (
                prepared.responses[:, cursor:stop],
                predictions[:, cursor:stop],
                prepared.story_slices,
                [values[cursor:stop] for values in best_alphas],
            )
            cursor = stop

    for subject_index, subject in enumerate(config.fmri.subjects):
        if subject in completed_subjects:
            print(f"{subject}: reusing complete subject checkpoint")
            continue
        true, predicted, story_slices, best_alphas = subject_payloads[subject]
        pvalues, null_info = circular_shift_pvalues(
            true,
            predicted,
            story_slices,
            num_shifts=config.num_null_shifts,
            minimal_shift=config.minimal_shift,
            seed=config.null_seed + subject_index,
            batch_size=config.null_batch_size,
        )
        significant, qvalues = bh_fdr(pvalues, config.fdr_q)
        observed = null_info.pop("observed")
        voxel_mask = voxel_masks[subject]
        retained_indices = np.flatnonzero(voxel_mask)
        full_observed = np.full(len(voxel_mask), np.nan)
        full_pvalues = np.ones(len(voxel_mask))
        full_qvalues = np.ones(len(voxel_mask))
        full_significant = np.zeros(len(voxel_mask), dtype=bool)
        full_observed[voxel_mask] = observed
        full_pvalues[voxel_mask] = pvalues
        full_qvalues[voxel_mask] = qvalues
        full_significant[voxel_mask] = significant
        significance = pd.DataFrame(
            {
                "subject": subject,
                "voxel_index": np.arange(len(voxel_mask)),
                "finite_response": voxel_mask,
                "corr": full_observed,
                "p_value": full_pvalues,
                "q_value": full_qvalues,
                "sig": full_significant,
            }
        )
        voxel_filter = {
            "total_voxels": len(voxel_mask),
            "retained_voxels": int(voxel_mask.sum()),
            "excluded_voxels": int((~voxel_mask).sum()),
            "excluded_by_story": excluded_by_story[subject],
        }
        plot_payload = None
        if getattr(config, "plot", True):
            figure = _acf_diagnostic_figure(
                true, predicted, story_slices, observed
            )
            plot_payload = figure
        targets = significance.rename(columns={
            "voxel_index": "target_id", "sig": "significant"
        }).copy()
        targets["target_kind"] = "voxel"
        artifact.save_subject_field("target_significance", subject, targets,
                                    overwrite=config.refresh)
        artifact.save_subject_field("significance_details", subject, {
            "ridge_alphas": {"voxel_indices": retained_indices,
                             "folds": list(best_alphas)},
            "null_metadata": null_info,
            "voxel_filter": voxel_filter,
        }, overwrite=config.refresh)
        if config.ridge.log_alphas:
            artifact.save_subject_field("diagnostics", subject,
                                        {"best_alphas": list(best_alphas)},
                                        overwrite=config.refresh)
        if plot_payload is not None:
            buffer = io.BytesIO()
            plot_payload.savefig(buffer, format="png", dpi=160)
            plt.close(plot_payload)
            artifact.save_subject_field("plots", subject, buffer.getvalue(),
                                        overwrite=config.refresh)
        print(
            f"{subject}: {int(significant.sum()):,}/{int(voxel_mask.sum()):,} "
            "retained voxels significant"
        )

    metadata = {
        "subjects": config.fmri.subjects,
        "sessions": config.fmri.sessions,
        "train_stories": train_stories,
        "test_stories": test_stories,
        "temporarily_held_out_stories": sorted(TEMPORARILY_HELD_OUT_FMRI_STORIES),
        "trim": config.fmri.trim,
        "fixed_leading_trim": 5,
        "effective_start_trim": 5 + config.fmri.trim,
        "effective_end_trim": config.fmri.trim,
        "ndelays": config.fmri.ndelays,
        "null_method": "independent_within_story_circular_shift",
        "num_null_shifts": config.num_null_shifts,
        "minimal_shift": config.minimal_shift,
        "fdr_q": config.fdr_q,
        "fdr_scope": "subject",
        "voxel_filter": "finite_in_every_tr_of_every_selected_training_story",
        "excluded_voxel_counts": {
            subject: int((~mask).sum()) for subject, mask in voxel_masks.items()
        },
    }
    artifact.save_field("effective_config", asdict(config), overwrite=True)
    artifact.save_field("run_metadata", {**metadata, "method": "held_out_session"},
                        overwrite=True)
    subject_selectors = [{"subject": str(subject)} for subject in config.fmri.subjects]
    artifact.mark_complete({
        field: subject_selectors
        for field in ("target_significance", "significance_details")
    })
    return artifact
