"""Find fMRI voxels predicted significantly in session-held-out encoding models."""

import json
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from himalaya.backend import set_backend

from src.core.config import FMRISignificanceConfig
from src.utils.alpha_grids import resolve_ridge_alphas, save_alpha_diagnostics
from src.utils.config_utils import load_config, parse_args
from src.utils.encoding import extract_best_alphas, get_ridge_model_constructor
from src.utils.fmri import (
    bh_fdr,
    circular_shift_pvalues,
    finite_response_voxel_masks,
    load_story_embeddings,
    prepare_fmri_data,
    remap_inner_splits,
    resolve_fmri_stories,
    session_cv_splits,
    TEMPORARILY_HELD_OUT_FMRI_STORIES,
)
from src.utils.tensor_utils import to_numpy


def _fit_oof(X, Y, story_slices, groups, config, output_dir):
    predictions = np.zeros_like(Y, dtype=np.float32)
    best_alphas = []
    outer_folds = session_cv_splits(story_slices, groups)
    explicit = np.asarray(config.ridge.alphas) if config.ridge.alphas else None
    sessions = list(groups)
    for fold, ((train_idx, test_idx), held_out_session) in enumerate(
        zip(outer_folds, sessions)
    ):
        fold_dir = Path(output_dir) / f"fold_{fold}"
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
            cache_dir=str(fold_dir),
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
        if config.ridge.log_alphas:
            save_alpha_diagnostics(
                str(fold_dir), alphas, best_alphas=[selected], context=context,
                policy=config.ridge.alpha_grid,
            )
    return predictions, best_alphas


def _save_acf_diagnostic(path: Path, true, predicted, story_slices, observed):
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
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _analyze_subject_block(
    subjects, config, embeddings, train_stories, groups, voxel_masks
):
    prepared = prepare_fmri_data(
        config.fmri.fmri_repo_root,
        train_stories,
        embeddings,
        trim=config.fmri.trim,
        ndelays=config.fmri.ndelays,
        subjects=subjects,
        voxel_masks=voxel_masks,
    )
    block_name = subjects[0] if len(subjects) == 1 else "combined"
    block_dir = Path(config.output_dir) / "fits" / block_name
    block_dir.mkdir(parents=True, exist_ok=True)
    predictions, best_alphas = _fit_oof(
        prepared.stimulus,
        prepared.responses,
        prepared.story_slices,
        groups,
        config,
        block_dir,
    )
    return prepared, predictions, best_alphas


def run_config(config: FMRISignificanceConfig) -> None:
    config.validate()
    if torch.cuda.is_available():
        set_backend("torch_cuda")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    for subject, mask in voxel_masks.items():
        print(
            f"{subject}: excluding {int((~mask).sum()):,}/{len(mask):,} "
            "voxels with non-finite responses"
        )
    subject_payloads = {}
    if config.batch_subjects:
        for subject in tqdm(config.fmri.subjects, desc="Running subjects"):
            prepared, predictions, best_alphas = _analyze_subject_block(
                [subject], config, embeddings, train_stories, groups, voxel_masks
            )
            subject_payloads[subject] = (
                prepared.responses,
                predictions,
                prepared.story_slices,
                best_alphas,
            )
    else:
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
        subject_dir = output_dir / "subjects" / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "subject": subject,
                "voxel_index": np.arange(len(voxel_mask)),
                "finite_response": voxel_mask,
                "corr": full_observed,
                "p_value": full_pvalues,
                "q_value": full_qvalues,
                "sig": full_significant,
            }
        ).to_csv(subject_dir / "significance.csv", index=False)
        np.savez_compressed(
            subject_dir / "ridge_alphas.npz",
            voxel_indices=retained_indices,
            **{f"fold_{fold}": values for fold, values in enumerate(best_alphas)},
        )
        (subject_dir / "null_metadata.json").write_text(
            json.dumps(null_info, indent=2)
        )
        _save_acf_diagnostic(
            subject_dir / "acf_diagnostic.png",
            true,
            predicted,
            story_slices,
            observed,
        )
        (subject_dir / "voxel_filter.json").write_text(json.dumps({
            "total_voxels": len(voxel_mask),
            "retained_voxels": int(voxel_mask.sum()),
            "excluded_voxels": int((~voxel_mask).sum()),
            "excluded_by_story": excluded_by_story[subject],
        }, indent=2))
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
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args, overrides = parse_args(__doc__)
    run_config(load_config(args.config, overrides, FMRISignificanceConfig))


if __name__ == "__main__":
    main()
