import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.data_loaders import get_arbitrary_embeddings
from src.utils.tensor_utils import to_numpy


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
        kfold = KFold(num_folds, shuffle=False)
        folds = list(kfold.split(X[0]))
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


def get_all_targets_test_set(test_idxs, Y, num_electrodes, num_lags):
    """
    Reassemble held-out targets from all folds into a single array.

    For k-fold cross-validation, each fold holds out a disjoint subset.
    This function stacks those subsets back in order, then reshapes to
    (n_words, num_electrodes, num_lags).

    Args:
        test_idxs: List of index arrays, one per fold (fold test indices into Y).
        Y: Target array of shape (n_words, num_electrodes * num_lags).
        num_electrodes: Number of electrodes.
        num_lags: Number of time lags.

    Returns:
        np.ndarray of shape (n_words, num_electrodes, num_lags).
    """
    all_targets = []
    for test_idx in test_idxs:
        all_targets.append(Y[test_idx])
    all_targets = np.vstack(all_targets).reshape((Y.shape[0], num_electrodes, num_lags))
    return all_targets


def get_ridge_model_constructor(
    alphas=None,
    ridge_cv_folds: int = 5,
    solver_params: dict | None = None,
    y_in_cpu: bool | None = None,
    force_cpu: bool | None = None,
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
    """
    from src.utils.constants import RIDGE_ALPHAS

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
    inner_cv = KFold(n_splits=ridge_cv_folds, shuffle=False)

    def create_model(_):
        ridge_kwargs = {"fit_intercept": True, "cv": inner_cv}
        if is_himalaya:
            if solver_params is not None:
                ridge_kwargs["solver_params"] = solver_params
            if y_in_cpu is not None:
                ridge_kwargs["Y_in_cpu"] = y_in_cpu
            if force_cpu is not None:
                ridge_kwargs["force_cpu"] = force_cpu
        return make_pipeline(
            StandardScaler(),
            RidgeCV(alphas, **ridge_kwargs),
        )

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


def train_encoding_two_pass(
    X,
    Y,
    alphas=None,
    ridge_cv_folds: int = 5,
    n_refined=20,
    margin_decades=1.0,
    solver_params: dict | None = None,
    y_in_cpu: bool | None = None,
    force_cpu: bool | None = None,
    **train_kwargs,
):
    """Train encoding with coarse alphas, then re-train with a refined grid.

    Pass 1: Train with the coarse alpha grid (default RIDGE_ALPHAS).
    Extracts best_alphas_ from all folds, calls refine_alpha_grid to build
    a tighter grid, then re-trains with the refined grid on the same folds.

    Args:
        X: Feature array or list of per-fold arrays (passed to train_encoding).
        Y: Target array (passed to train_encoding).
        alphas: Initial coarse alpha grid. None → RIDGE_ALPHAS.
        ridge_cv_folds: Number of inner CV folds used by RidgeCV alpha
            selection.
        n_refined: Number of points in the refined grid.
        margin_decades: Extra decades beyond observed range.
        solver_params: Optional Himalaya RidgeCV solver_params.
        y_in_cpu: Optional Himalaya RidgeCV Y_in_cpu override.
        force_cpu: Optional Himalaya RidgeCV force_cpu override.
        **train_kwargs: Forwarded to train_encoding (epochs_shape, etc.).

    Returns:
        Tuple of (corrs, models, tr_idxs, test_idxs, scalers, coarse_best_alphas,
        refined_grid) where the first five match train_encoding's return value.
    """
    from src.utils.constants import refine_alpha_grid

    # Pass 1: coarse sweep
    model_constructor = get_ridge_model_constructor(
        alphas=alphas,
        ridge_cv_folds=ridge_cv_folds,
        solver_params=solver_params,
        y_in_cpu=y_in_cpu,
        force_cpu=force_cpu,
    )
    corrs, models, tr_idxs, test_idxs, scalers = train_encoding(
        X, Y, model_constructor, **train_kwargs
    )
    coarse_best = np.concatenate([extract_best_alphas(m) for m in models])

    # Build refined grid
    refined_grid = refine_alpha_grid(
        coarse_best, n_points=n_refined, margin_decades=margin_decades
    )

    # Pass 2: refined sweep on same folds
    model_constructor_2 = get_ridge_model_constructor(
        alphas=refined_grid,
        ridge_cv_folds=ridge_cv_folds,
        solver_params=solver_params,
        y_in_cpu=y_in_cpu,
        force_cpu=force_cpu,
    )
    corrs, models, _, _, scalers = train_encoding(
        X,
        Y,
        model_constructor_2,
        provided_idxs=list(zip(tr_idxs, test_idxs)),
        provided_scalers=scalers,
        **train_kwargs,
    )
    return corrs, models, tr_idxs, test_idxs, scalers, coarse_best, refined_grid


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


def _make_col_idxs(enc_results: dict, target_elec_names: list) -> np.ndarray:
    """Map target electrode names to column indices in the saved encoding result.

    For an encoding result with shape (n_words, n_elec * n_lags), electrode e at
    index i in enc_results["elec_names"] occupies columns i*n_lags : (i+1)*n_lags.

    Args:
        enc_results: Dict loaded from encoding_results.joblib.
        target_elec_names: Ordered list of electrode names to include.

    Returns:
        Integer array of column indices into the saved model's output.

    Raises:
        ValueError: If any electrode in target_elec_names is not found.
    """
    elec_names = enc_results["elec_names"]
    n_lags = len(enc_results["lags"])
    col_idxs = []
    for e in target_elec_names:
        try:
            i = elec_names.index(e)
        except ValueError:
            raise ValueError(
                f"Electrode '{e}' not found in enc_results['elec_names']. "
                "Ensure encoding results match the current run's electrode list."
            )
        col_idxs.extend(range(i * n_lags, (i + 1) * n_lags))
    return np.array(col_idxs, dtype=int)


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
    from sklearn.preprocessing import StandardScaler
    from src.utils.alpha_grids import resolve_ridge_alphas, save_alpha_diagnostics
    from src.utils.sae import solve_regression
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_words, n_embed = X_raw.shape
    baselines = []

    for fold_f, test_idx in enumerate(test_idxs):
        if folds_to_build is not None and fold_f not in folds_to_build:
            baselines.append(None)
            continue

        train_mask = np.ones(n_words, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        y_scaler = StandardScaler()
        Y_train = y_scaler.fit_transform(Y_full[train_idx]).astype(np.float32)
        Y_test = y_scaler.transform(Y_full[test_idx]).astype(np.float32)

        fold_context = {
            "role": "full_baseline",
            "fold": int(fold_f),
            "n_features": int(n_embed),
            "standardize": True,
            **(alpha_context or {}),
        }
        fold_cache_dir = (
            os.path.join(alpha_cache_dir, f"fold_{fold_f}", "full_baseline_alpha")
            if alpha_cache_dir is not None
            else None
        )
        fold_alphas = resolve_ridge_alphas(
            alphas,
            X_raw[train_idx],
            standardize=True,
            context=fold_context,
            cache_dir=fold_cache_dir,
            policy=alpha_policy,
        )
        W, b, best_alphas = solve_regression(
            X_raw[train_idx],
            Y_train,
            "ridge",
            device,
            alphas=fold_alphas,
            ridge_cv_folds=ridge_cv_folds,
            solver_params=ridge_solver_params,
            y_in_cpu=ridge_y_in_cpu,
            force_cpu=ridge_force_cpu,
            return_alphas=True,
        )
        if fold_cache_dir is not None:
            save_alpha_diagnostics(
                fold_cache_dir,
                fold_alphas,
                best_alphas=best_alphas,
                context=fold_context,
                policy=alpha_policy,
            )
        W = to_numpy(W)
        b = to_numpy(b)

        Y_hat_train = (X_raw[train_idx] @ W + b).astype(np.float32)
        Y_hat_test = (X_raw[test_idx] @ W + b).astype(np.float32)

        baselines.append(
            {
                "train_idx": train_idx,
                "test_idx": test_idx,
                "Y_train": Y_train,
                "Y_test": Y_test,
                "Y_hat_full_train": Y_hat_train,
                "Y_hat_full_test": Y_hat_test,
                "beta_full_raw": W.astype(np.float32),
                "best_alphas": to_numpy(best_alphas),
                "target_mean": y_scaler.mean_.astype(np.float64),
                "target_scale": y_scaler.scale_.astype(np.float64),
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


def get_and_offset_arbitrary_embeddings(
    aligned_embeddings, selection, df_word, offset, seed, valid_rows=None
):
    X_arbitrary = get_arbitrary_embeddings(
        aligned_embeddings,
        selection,
        df_word,
        seed=seed,
    ).astype(np.float32)
    X_arbitrary, _ = apply_offset(
        X_arbitrary, X_arbitrary, offset, valid_rows=valid_rows
    )
    return X_arbitrary


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
    from src.utils.plotting import plot_corrs_vs_arbitrary

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


def get_all_predictions_and_targets_test_set(
    encoding_models, test_idxs, X, Y, num_electrodes, num_timepoints
):
    """
    Generate predictions and assemble targets for all test folds.

    Args:
        encoding_models: List of fitted models (one per fold).
        test_idxs: List of test index arrays.
        X: List of feature arrays (one per fold) or single array.
        Y: Target array of shape (n_words, num_electrodes * num_timepoints).
        num_electrodes: Number of electrodes.
        num_timepoints: Number of time lags/timepoints.

    Returns:
        Tuple of (all_predictions, all_targets), each of shape
        (n_words, num_electrodes, num_timepoints).
    """
    all_predictions = []

    for i, trained_model in enumerate(encoding_models):
        all_predictions.append(trained_model.predict(X[i][test_idxs[i]]))

    all_predictions = (
        torch.vstack(all_predictions)
        .numpy()
        .reshape((X[0].shape[0], num_electrodes, num_timepoints))
    )

    return all_predictions, get_all_targets_test_set(
        test_idxs, Y, num_electrodes, num_timepoints
    )
