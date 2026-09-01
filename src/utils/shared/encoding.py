"""Ridge encoding helpers shared by retained workflows."""

import glob
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.utils.shared.ieeg import get_arbitrary_embeddings
from src.utils.shared.tensors import to_numpy


def contiguous_folds(n_rows: int, n_splits: int):
    """Return unshuffled K-fold indices in sample order."""
    if n_rows < n_splits:
        raise ValueError(f"Cannot make {n_splits} folds from {n_rows} rows")
    return list(KFold(n_splits=n_splits, shuffle=False).split(np.arange(n_rows)))


def train_encoding(
    X,
    Y,
    model_constructor_fn,
    modify_inputs_fn=None,
    modify_targets_fn=None,
    epochs_shape=None,
    fix_model=False,
    provided_idxs=None,
    provided_scalers=None,
    num_folds=2,
    return_preds=False,
):
    """
    Train an encoding model using k-fold cross-validation.

    Args:
        X: Feature array (n_words, n_features) or list of per-fold arrays.
        Y: Target array (n_words, n_targets).
        model_constructor_fn: Callable(fold_index) -> sklearn-compatible model.
        modify_inputs_fn: Optional callable to preprocess X per fold.
        modify_targets_fn: Optional callable to preprocess Y per fold.
        epochs_shape: Optional shape to reshape correlation scores.
        fix_model: If True, skip fitting (use pre-trained model).
        provided_idxs: Optional list of (train_idx, test_idx) tuples.
        provided_scalers: Optional list of pre-fitted StandardScalers.
        num_folds: Number of cross-validation folds.

    Returns:
        Tuple of (corrs, models, tr_idxs, test_idxs, scalers).
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from himalaya.scoring import correlation_score
    from himalaya.backend import get_backend

    if provided_scalers is None:
        provided_scalers = []

    num_folds = len(provided_idxs) if provided_idxs is not None else num_folds
    if not isinstance(X, list):
        X = [X for _ in range(num_folds)]

    models = []
    corrs = []
    all_Y_preds = []
    scalers = []
    tr_idxs, test_idxs = [], []
    if provided_idxs is None:
        folds = contiguous_folds(len(X[0]), num_folds)
    else:
        folds = provided_idxs

    for i, (train_index, test_index) in enumerate(folds):
        X_curr = X[i]
        X1_train, X1_test = X_curr[train_index], X_curr[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if modify_inputs_fn:
            X1_train = modify_inputs_fn(X1_train, i, idx=train_index)
            X1_test = modify_inputs_fn(X1_test, i, idx=test_index)

        if modify_targets_fn:
            Y_train = modify_targets_fn(Y_train, i, idx=train_index)
            Y_test = modify_targets_fn(Y_test, i, idx=test_index)

        if not provided_scalers:
            scaler = StandardScaler().fit(Y_train)
        else:
            scaler = provided_scalers[i]

        Y_train = scaler.transform(Y_train)
        Y_test = scaler.transform(Y_test)
        scalers.append(scaler)

        model = model_constructor_fn(i)

        if not fix_model:
            model.fit(X1_train, Y_train)
        Y_preds = model.predict(X1_test)
        all_Y_preds.append(Y_preds)
        corr = to_numpy(correlation_score(Y_test, Y_preds))
        if epochs_shape:
            corr = corr.reshape(epochs_shape)

        corrs.append(corr)
        models.append(model)
        tr_idxs.append(train_index)
        test_idxs.append(test_index)
    if return_preds:
        return np.stack(corrs), models, tr_idxs, test_idxs, scalers, all_Y_preds
    return np.stack(corrs), models, tr_idxs, test_idxs, scalers


def get_ridge_model_constructor(
    alphas=None,
    ridge_cv_folds: int = 5,
    solver_params: dict | None = None,
    y_in_cpu: bool | None = None,
    force_cpu: bool | None = None,
    cv=None,
    standardize: bool = True,
):
    """
    Return a factory function for StandardScaler + RidgeCV pipelines.
    Compatible with the model_constructor_fn arg of train_encoding().
    Sets the himalaya torch_cuda backend if CUDA is available.

    Args:
        alphas: Ridge alpha grid for RidgeCV. When None, uses RIDGE_ALPHAS
            from utils.constants.
        ridge_cv_folds: Number of inner CV folds used by RidgeCV alpha
            selection.
        solver_params: Optional Himalaya RidgeCV solver_params. Useful memory
            controls include n_alphas_batch, n_targets_batch, and
            n_targets_batch_refit. Ignored by sklearn fallback.
        y_in_cpu: Optional Himalaya RidgeCV Y_in_cpu override. Ignored by
            sklearn fallback.
        force_cpu: Optional Himalaya RidgeCV force_cpu override. Ignored by
            sklearn fallback.
        cv: Optional explicit CV splitter or split iterable. When omitted,
            contiguous K-fold CV is used as before.
        standardize: Whether to prepend a StandardScaler to the ridge model.
    """
    from src.utils.shared.constants import RIDGE_ALPHAS

    is_himalaya = False
    try:
        from himalaya.ridge import RidgeCV
        from himalaya.backend import set_backend
        import torch

        is_himalaya = True
        if torch.cuda.is_available():
            set_backend("torch_cuda")
    except ImportError:
        from sklearn.linear_model import RidgeCV

    if alphas is None:
        alphas = RIDGE_ALPHAS
    inner_cv = cv if cv is not None else KFold(n_splits=ridge_cv_folds, shuffle=False)

    def create_model(_):
        ridge_kwargs = {"fit_intercept": True, "cv": inner_cv}
        if is_himalaya:
            if solver_params is not None:
                ridge_kwargs["solver_params"] = solver_params
            if y_in_cpu is not None:
                ridge_kwargs["Y_in_cpu"] = y_in_cpu
            if force_cpu is not None:
                ridge_kwargs["force_cpu"] = force_cpu
        ridge = RidgeCV(alphas, **ridge_kwargs)
        return make_pipeline(StandardScaler(), ridge) if standardize else make_pipeline(ridge)

    return create_model


def extract_best_alphas(model) -> np.ndarray:
    """Extract per-target best alphas from a fitted Pipeline containing RidgeCV.

    Args:
        model: Fitted sklearn Pipeline with a 'ridgecv' step (as created by
            get_ridge_model_constructor).

    Returns:
        1-D numpy array of shape (n_targets,) with the selected alpha per target.
    """
    ridge = model.named_steps["ridgecv"]
    return to_numpy(ridge.best_alphas_)


def save_encoding_results(results: dict, filepath: str) -> None:
    """Persist results dict to disk via joblib.dump."""
    import joblib

    joblib.dump(results, filepath)


def load_encoding_results(filepath: str, location=None) -> dict:
    """Load results dict previously saved by save_encoding_results."""
    import joblib
    from unittest.mock import patch

    # TODO: This is super hacky and I should just change where these are written to make sure they're always on CPU. But for now, this allows loading GPU-saved results on CPU-only machines without error.
    if location == "cpu":
        # We need to return a function that accepts (storage, location)
        # and just returns the storage (which defaults to CPU).
        def force_cpu(storage, loc):
            return storage

        with patch("torch.serialization._get_restore_location", return_value=force_cpu):
            return joblib.load(filepath)
    else:
        return joblib.load(filepath)


def fit_transform_target_pca(
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    target_pca_dim: int,
    *,
    context: str = "target PCA",
) -> tuple[np.ndarray, np.ndarray]:
    """Fit exact PCA on training targets and transform train/test targets."""
    from sklearn.decomposition import PCA

    available = min(train_targets.shape)
    if int(target_pca_dim) > available:
        raise ValueError(
            f"target_pca_dim={target_pca_dim} exceeds {context} estimable rank "
            f"{available}"
        )
    target_pca = PCA(n_components=int(target_pca_dim), svd_solver="full")
    return (
        target_pca.fit_transform(train_targets).astype(np.float32),
        target_pca.transform(test_targets).astype(np.float32),
    )


def fit_encoding_baseline(
    X_train: np.ndarray,
    raw_Y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    raw_Y_test: np.ndarray | None = None,
    *,
    alphas: np.ndarray | None = None,
    ridge_cv_folds: int = 5,
    ridge_solver_params: dict | None = None,
    ridge_y_in_cpu: bool | None = None,
    ridge_force_cpu: bool | None = None,
    alpha_policy: dict | None = None,
    alpha_cache_dir: str | None = None,
    alpha_context: dict | None = None,
) -> dict:
    """Fit the standardized-target full ridge baseline used by RRR workflows."""
    from sklearn.preprocessing import StandardScaler
    from src.utils.shared.alpha_grids import resolve_ridge_alphas, save_alpha_diagnostics
    from src.utils.shared.regression import solve_regression

    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_scaler = StandardScaler().fit(raw_Y_train)
    Y_train = y_scaler.transform(raw_Y_train).astype(np.float32)
    Y_test = (
        None if raw_Y_test is None
        else y_scaler.transform(raw_Y_test).astype(np.float32)
    )
    context = {
        "role": "full_baseline", "n_features": int(X_train.shape[1]),
        "standardize": True, **(alpha_context or {}),
    }
    resolved_alphas = resolve_ridge_alphas(
        alphas, X_train, standardize=True, context=context,
        cache_dir=alpha_cache_dir, policy=alpha_policy,
    )
    W, b, best_alphas = solve_regression(
        X_train, Y_train, "ridge", device, alphas=resolved_alphas,
        ridge_cv_folds=ridge_cv_folds, solver_params=ridge_solver_params,
        y_in_cpu=ridge_y_in_cpu, force_cpu=ridge_force_cpu,
        return_alphas=True,
    )
    if alpha_cache_dir is not None:
        save_alpha_diagnostics(
            alpha_cache_dir, resolved_alphas, best_alphas=best_alphas,
            context=context, policy=alpha_policy,
        )
    W, b = to_numpy(W), to_numpy(b)
    return {
        "Y_train": Y_train,
        "Y_test": Y_test,
        "Y_hat_full_train": (X_train @ W + b).astype(np.float32),
        "Y_hat_full_test": (
            None if X_test is None else (X_test @ W + b).astype(np.float32)
        ),
        "beta_full_raw": W.astype(np.float32),
        "ridge_intercept": b.astype(np.float32),
        "target_mean": y_scaler.mean_.astype(np.float32),
        "target_scale": y_scaler.scale_.astype(np.float32),
        "alpha_grid": np.asarray(resolved_alphas),
        "best_alphas": np.asarray(best_alphas),
    }


def build_per_fold_baselines(
    X_raw: np.ndarray,
    Y_full: np.ndarray,
    test_idxs: list,
    alphas: np.ndarray | None = None,
    ridge_cv_folds: int = 5,
    ridge_solver_params: dict | None = None,
    ridge_y_in_cpu: bool | None = None,
    ridge_force_cpu: bool | None = None,
    alpha_policy: dict | None = None,
    alpha_cache_dir: str | None = None,
    alpha_context: dict | None = None,
    folds_to_build: set[int] | None = None,
    target_pca_dim: int | None = None,
) -> list:
    """Train a fresh ridge encoding model per fold and pre-compute baselines.

    For each fold, fits a StandardScaler on Y_train, then trains a ridge model
    via solve_regression (which absorbs the X-scaler into W, b). Returns the
    dict structure consumed by RRR evaluation.

    Args:
        X_raw: (n_words, n_embed) raw embedding matrix.
        Y_full: (n_words, n_targets) assembled neural response matrix.
        test_idxs: List of test index arrays, one per fold.
        alphas: Optional explicit ridge alpha grid for RidgeCV.
        ridge_cv_folds: Number of inner CV folds used by RidgeCV alpha
            selection.
        ridge_solver_params: Optional Himalaya RidgeCV solver_params.
        ridge_y_in_cpu: Optional Himalaya RidgeCV Y_in_cpu override.
        ridge_force_cpu: Optional Himalaya RidgeCV force_cpu override.
        alpha_policy: Optional dimension-aware alpha grid policy.
        alpha_cache_dir: Optional root for fold-level alpha caches.
        alpha_context: Optional metadata included in alpha cache keys.
        folds_to_build: Optional set of fold indices to train. Other entries
            are returned as None so callers can skip already-complete folds.

    Returns:
        List of dicts, one per fold, each containing:
            train_idx:       training index array
            test_idx:        test index array
            Y_train:         (n_train, n_targets) scaled neural responses
            Y_test:          (n_test, n_targets) scaled neural responses
            Y_hat_full_train:(n_train, n_targets) baseline model predictions on train
            Y_hat_full_test: (n_test, n_targets) baseline model predictions on test
            beta_full_raw:   (n_embed, n_targets) coefficients mapping raw X → scaled Y
    """
    n_words = X_raw.shape[0]
    baselines = []

    for fold_f, test_idx in enumerate(test_idxs):
        if folds_to_build is not None and fold_f not in folds_to_build:
            baselines.append(None)
            continue

        train_mask = np.ones(n_words, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        raw_train = Y_full[train_idx]
        raw_test = Y_full[test_idx]
        if target_pca_dim is not None:
            raw_train, raw_test = fit_transform_target_pca(
                raw_train, raw_test, int(target_pca_dim), context=f"fold {fold_f}"
            )

        fold_context = {
            "fold": int(fold_f),
            **(alpha_context or {}),
        }
        fold_cache_dir = (
            os.path.join(alpha_cache_dir, f"fold_{fold_f}", "full_baseline_alpha")
            if alpha_cache_dir is not None
            else None
        )
        fitted = fit_encoding_baseline(
            X_raw[train_idx], raw_train, X_raw[test_idx], raw_test,
            alphas=alphas,
            ridge_cv_folds=ridge_cv_folds,
            ridge_solver_params=ridge_solver_params,
            ridge_y_in_cpu=ridge_y_in_cpu,
            ridge_force_cpu=ridge_force_cpu,
            alpha_policy=alpha_policy,
            alpha_cache_dir=fold_cache_dir,
            alpha_context=fold_context,
        )
        baselines.append(
            {
                "train_idx": train_idx,
                "test_idx": test_idx,
                **fitted,
            }
        )

    return baselines


def apply_offset(averaged_embeddings, data, offset, valid_rows=None):
    """Apply word offset: trim embeddings and neural data to align them.

    Args:
        averaged_embeddings: Array of shape (n_words, n_features).
        data: Array of shape (n_words, n_elecs, n_lags) or (n_words, n_elecs*n_lags).
        offset: Integer word offset. Positive shifts embeddings forward in time;
                negative shifts them backward.
        valid_rows: An optional array of shape n_words. If set then will only keep those examples
            which have valid_rows[i] = True

    Returns:
        Tuple of (X_arr, Y) both float32 with matching first dimension.
    """
    n_words = data.shape[0]
    if data.ndim == 3:
        Y = data.reshape(n_words, -1).astype(np.float32, copy=False)
    else:
        Y = data.astype(np.float32, copy=False)

    if offset < 0:
        averaged_embeddings, Y = averaged_embeddings[:offset], Y[-offset:]
        if valid_rows is not None:
            valid_rows = valid_rows[:offset]
    elif offset > 0:
        averaged_embeddings, Y = averaged_embeddings[offset:], Y[:-offset]
        if valid_rows is not None:
            valid_rows = valid_rows[offset:]

    if valid_rows is not None:
        averaged_embeddings, Y = averaged_embeddings[valid_rows], Y[valid_rows]

    return averaged_embeddings.astype(np.float32, copy=False), Y


def save_subject_results(
    output_dir,
    sub_id,
    corrs,
    tr_idxs,
    test_idxs,
    scalers,
    elec_names,
    offset,
    lags,
    per_model_config=None,
    corrs_arb=None,
    word_selection_metadata=None,
):
    """Build results dict and write outputs to {output_dir}/subjects/sub_{sub_id}/.

    Always writes encoding_results.joblib and corrs.npy.
    If corrs_arb is not None, also writes arbitrary_corrs.npy and corrs_plot.png.

    Args:
        output_dir: Root output directory.
        sub_id: Subject BIDS ID.
        corrs: Correlation array (n_folds, n_elecs, n_lags).
        tr_idxs: List of train index arrays.
        test_idxs: List of test index arrays.
        scalers: List of fitted StandardScalers.
        elec_names: Electrode name list.
        offset: Word offset used during alignment.
        lags: Lag axis array.
        per_model_config: Model metadata to store under the per_model_config key.
        corrs_arb: Optional null-model correlation array (same shape as corrs).
        word_selection_metadata: Optional row-provenance metadata to persist.
    """
    from src.utils.plotting.encoding import plot_corrs_vs_arbitrary

    results = {
        "corrs": corrs,
        "train_idxs": tr_idxs,
        "test_idxs": test_idxs,
        "scalers": scalers,
        "elec_names": list(elec_names),
        "subject_id": sub_id,
        "offset": offset,
        "lags": lags,
        "per_model_config": dict(per_model_config or {}),
    }
    if corrs_arb is not None:
        results["arb_corrs"] = corrs_arb
    if word_selection_metadata is not None:
        results["word_selection"] = word_selection_metadata

    sub_dir = os.path.join(output_dir, "subjects", f"sub_{sub_id}")
    os.makedirs(sub_dir, exist_ok=True)
    save_encoding_results(results, os.path.join(sub_dir, "encoding_results.joblib"))
    np.save(os.path.join(sub_dir, "corrs.npy"), corrs)
    if corrs_arb is not None:
        np.save(os.path.join(sub_dir, "arbitrary_corrs.npy"), corrs_arb)
        plot_corrs_vs_arbitrary(
            corrs,
            corrs_arb,
            lags,
            save_path=os.path.join(sub_dir, "corrs_plot.png"),
        )
    print(f"  Subject {sub_id}: {corrs.shape[1]} electrodes → {sub_dir}")


def _iter_subject_results(output_dir, subjects=None):
    """Yield (subject_id, results_dict) for each subject directory, sorted by subject ID.

    Args:
        output_dir: Root output directory containing subjects/sub_*/ subdirs.
        subjects: Optional list of subject IDs to include; if None, include all found.

    Yields:
        (subject_id: int, results: dict) tuples in ascending subject_id order.
    """
    pattern = os.path.join(output_dir, "subjects", "sub_*", "encoding_results.joblib")
    paths = sorted(glob.glob(pattern))
    if subjects is not None:
        subjects_set = set(subjects)
        paths = [
            p
            for p in paths
            if int(os.path.basename(os.path.dirname(p)).split("_")[1]) in subjects_set
        ]
    if not paths:
        raise FileNotFoundError(
            f"No subject encoding results found under {output_dir}/subjects/ "
            f"(subjects filter: {subjects})"
        )
    for path in paths:
        sub_id = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        yield sub_id, load_encoding_results(path)


def load_all_subject_correlations(output_dir: str, subjects=None):
    """Load and concatenate correlations across subjects.

    Args:
        output_dir: Root output directory containing subjects/sub_*/ subdirs.
        subjects: Optional list of subject IDs to load; loads all found if None.

    Returns:
        corrs_all: np.ndarray of shape (n_folds, n_total_elecs, n_lags).
        metadata_df: pd.DataFrame with columns ['subject', 'electrode'], one row per electrode.

    Raises:
        FileNotFoundError: If no subject directories are found (after filtering).
    """
    corrs_list = []
    rows = []
    for sub_id, results in _iter_subject_results(output_dir, subjects):
        corrs = results["corrs"]  # (n_folds, n_elecs, n_lags)
        elec_names = results["elec_names"]
        corrs_list.append(corrs)
        for name in elec_names:
            rows.append({"subject": sub_id, "electrode": name})
    corrs_all = np.concatenate(corrs_list, axis=1)  # (n_folds, n_total_elecs, n_lags)
    metadata_df = pd.DataFrame(rows, columns=["subject", "electrode"])
    return corrs_all, metadata_df
