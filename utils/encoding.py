import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.tensor_utils import to_numpy


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
        corr = correlation_score(Y_test, Y_preds)
        if epochs_shape:
            corr = corr.reshape(epochs_shape)

        corr = to_numpy(corr)

        corrs.append(corr)
        models.append(model)
        tr_idxs.append(train_index)
        test_idxs.append(test_index)
    return np.stack(corrs), models, tr_idxs, test_idxs, scalers


def get_scaled_model_coefficients(encoding_model, num_electrodes, num_lags):
    """
    Extract ridge coefficients reshaped to (num_electrodes, num_lags, n_features).

    Args:
        encoding_model: Fitted sklearn Pipeline with 'ridgecv' step.
        num_electrodes: Number of electrodes.
        num_lags: Number of time lags.

    Returns:
        Tuple of (ridge_coefficients, ridge_intercept).
    """
    ridge_coefficients = encoding_model.named_steps['ridgecv'].coef_
    ridge_coefficients = ridge_coefficients.view(
        ridge_coefficients.shape[0], num_electrodes, num_lags
    ).permute(1, 2, 0)
    ridge_intercept = encoding_model.named_steps['ridgecv'].intercept_
    return ridge_coefficients, ridge_intercept


def absorb_scaler(encoding_model, num_electrodes, num_lags):
    """
    Convert Ridge coefficients from standardized space back to raw input space.

    In standardized space: y = coef @ x_scaled + intercept
    where x_scaled = (x - mean) / scale.
    So in raw space: y = (coef / scale) @ x + (intercept - coef @ mean / scale).

    Works with both the himalaya numpy backend and torch backend.
    Note: himalaya RidgeCV does not fit an intercept (fit_intercept=False), so
    raw_intercept captures only the de-standardization offset: -(mean / scale) @ coef.

    Args:
        encoding_model: Fitted sklearn Pipeline with 'standardscaler' and 'ridgecv' steps.
        num_electrodes: Number of electrodes.
        num_lags: Number of time lags.

    Returns:
        Tuple of (raw_coefs, raw_intercept) as numpy arrays.
        raw_coefs shape: (num_electrodes, num_lags, n_features).
        raw_intercept shape: (num_electrodes * num_lags,).
    """
    scaler = encoding_model.named_steps["standardscaler"]
    ridge = encoding_model.named_steps["ridgecv"]

    # coef_ may be a torch tensor (torch backend) or numpy array (numpy backend).
    coef_np = to_numpy(ridge.coef_)

    # coef_ shape from himalaya: (n_features, n_targets)
    raw_coefs_np = coef_np / scaler.scale_[:, np.newaxis]  # (n_features, n_targets)

    # himalaya Ridge does not fit an intercept; the de-standardization offset is:
    # intercept_raw = -(mean / scale) @ coef = -mean @ raw_coefs
    if hasattr(ridge, "intercept_"):
        intercept_np = to_numpy(ridge.intercept_)
        raw_intercept = intercept_np - (scaler.mean_ @ raw_coefs_np)
    else:
        raw_intercept = -(scaler.mean_ @ raw_coefs_np)  # (n_targets,)

    # Reshape: (n_features, n_targets) -> (n_electrodes, n_lags, n_features)
    raw_coefs_np = np.transpose(
        raw_coefs_np.reshape(raw_coefs_np.shape[0], num_electrodes, num_lags),
        axes=(1, 2, 0),
    )

    return raw_coefs_np, np.asarray(raw_intercept)


def get_models_coefficients(encoding_models, num_electrodes, num_lags, scaled_model=False):
    """
    Extract coefficients from a list of encoding models.

    Args:
        encoding_models: List of fitted sklearn Pipelines.
        num_electrodes: Number of electrodes.
        num_lags: Number of time lags.
        scaled_model: If True, use get_scaled_model_coefficients; else use absorb_scaler.

    Returns:
        Tuple of (coefficients list, intercepts list).
    """
    coefficients, intercepts = [], []
    for encoding_model in encoding_models:
        if scaled_model:
            coefs = get_scaled_model_coefficients(encoding_model, num_electrodes, num_lags)
        else:
            coefs = absorb_scaler(encoding_model, num_electrodes, num_lags)

        coefficients.append(coefs[0])
        intercepts.append(coefs[1])

    return coefficients, intercepts


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


def get_ridge_model_constructor():
    """
    Return a factory function for StandardScaler + RidgeCV pipelines.
    Uses alphas = np.logspace(1, 10, 10) and inner 5-fold CV (no shuffle).
    Compatible with the model_constructor_fn arg of train_encoding().
    Sets the himalaya torch_cuda backend if CUDA is available.
    """
    try:
        from himalaya.ridge import RidgeCV
        from himalaya.backend import set_backend
        import torch
        if torch.cuda.is_available():
            set_backend("torch_cuda")
    except ImportError:
        from sklearn.linear_model import RidgeCV

    alphas = np.logspace(1, 10, 10)
    inner_cv = KFold(n_splits=5, shuffle=False)

    def create_model(_):
        return make_pipeline(
            StandardScaler(),
            RidgeCV(alphas, fit_intercept=True, cv=inner_cv),
        )

    return create_model


def save_encoding_results(results: dict, filepath: str) -> None:
    """Persist results dict to disk via joblib.dump."""
    import joblib
    joblib.dump(results, filepath)


def load_encoding_results(filepath: str) -> dict:
    """Load results dict previously saved by save_encoding_results."""
    import joblib
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


def build_enc_subjects_info(
    output_dir: str,
    subs_in_unit: list,
    target_elecs_per_sub: dict,
) -> list:
    """Load per-subject encoding results and build info dicts for baseline computation.

    Args:
        output_dir: Root results directory containing subjects/sub_{id}/.
        subs_in_unit: Ordered list of subject IDs contributing to the analysis unit.
        target_elecs_per_sub: Dict mapping sub_id → list of target electrode names.

    Returns:
        List of dicts, one per subject, each containing:
            models:      list of fitted Pipeline per fold
            scalers:     list of StandardScaler per fold (fitted on all-electrode Y)
            col_idxs:    int array of column indices into the saved model's output
            y_col_start: start column in the assembled Y_full
            y_col_end:   end column (exclusive) in the assembled Y_full
    """
    import os
    info_list = []
    y_cursor = 0
    for sub_id in subs_in_unit:
        enc_results = load_encoding_results(
            os.path.join(output_dir, "subjects", f"sub_{sub_id}", "encoding_results.joblib")
        )
        target_elecs = target_elecs_per_sub[sub_id]
        col_idxs = _make_col_idxs(enc_results, target_elecs)
        n_target_cols = len(target_elecs) * len(enc_results["lags"])
        info_list.append({
            "models": enc_results["models"],
            "scalers": enc_results["scalers"],
            "col_idxs": col_idxs,
            "y_col_start": y_cursor,
            "y_col_end": y_cursor + n_target_cols,
        })
        y_cursor += n_target_cols
    return info_list


def build_per_fold_baselines(
    X_raw: np.ndarray,
    Y_full: np.ndarray,
    test_idxs: list,
    enc_subjects_info: list,
) -> list:
    """Pre-compute per-fold scaled Y arrays, baseline predictions, and raw coefficients.

    Replaces the fresh StandardScaler + ridge model training inside the per-fold loops
    of _run_rrr_for_unit and _run_srrr_for_unit by reusing the saved encoding models.

    Args:
        X_raw: (n_words, n_embed) raw embedding matrix.
        Y_full: (n_words, n_targets) assembled neural response matrix.
        test_idxs: List of test index arrays, one per fold.
        enc_subjects_info: Output of build_enc_subjects_info.

    Returns:
        List of dicts, one per fold, each containing:
            train_idx:       training index array
            test_idx:        test index array
            Y_train:         (n_train, n_targets) scaled neural responses
            Y_test:          (n_test, n_targets) scaled neural responses
            Y_hat_full_train:(n_train, n_targets) baseline model predictions on train
            Y_hat_full_test: (n_test, n_targets) baseline model predictions on test
            beta_full_raw:   (n_embed, n_targets) unscaled coefficients in raw embed space
    """
    n_words, n_embed = X_raw.shape
    n_targets = Y_full.shape[1]
    baselines = []

    for fold_f, test_idx in enumerate(test_idxs):
        train_mask = np.ones(n_words, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        Y_train_parts, Y_test_parts = [], []
        Y_hat_train_parts, Y_hat_test_parts = [], []
        beta_full_raw = np.zeros((n_embed, n_targets), dtype=np.float32)

        for info in enc_subjects_info:
            model = info["models"][fold_f]
            scaler = info["scalers"][fold_f]
            col_idxs = info["col_idxs"]
            y_start = info["y_col_start"]
            y_end = info["y_col_end"]

            # Scale Y using saved scaler parameters (no refit)
            scaler_mean = scaler.mean_[col_idxs]
            scaler_scale = scaler.scale_[col_idxs]
            Y_sub_train = Y_full[train_idx, y_start:y_end]
            Y_sub_test = Y_full[test_idx, y_start:y_end]
            Y_train_parts.append((Y_sub_train - scaler_mean) / scaler_scale)
            Y_test_parts.append((Y_sub_test - scaler_mean) / scaler_scale)

            # Predictions from saved model (already in scaled Y space).
            # himalaya's predict does backend.to_cpu(X) @ backend.to_cpu(coef_).
            # coef_ is a torch tensor (saved with torch backend), so X must also
            # be a torch tensor on the same device for the matmul to type-match.
            ridge_coef = model[-1].coef_
            if isinstance(ridge_coef, torch.Tensor):
                X_train = to_torch(X_raw[train_idx], device=str(ridge_coef.device))
                X_test = to_torch(X_raw[test_idx], device=str(ridge_coef.device))
            else:
                X_train = X_raw[train_idx]
                X_test = X_raw[test_idx]
            Y_hat_all_train = to_numpy(model.predict(X_train))
            Y_hat_all_test = to_numpy(model.predict(X_test))
            Y_hat_train_parts.append(Y_hat_all_train[:, col_idxs])
            Y_hat_test_parts.append(Y_hat_all_test[:, col_idxs])

            # Unscaled coefficients in raw embedding space
            coef = to_numpy(model[-1].coef_)
            if coef.shape[0] != n_embed:
                coef = coef.T  # → (n_embed, n_all_cols)
            beta_sub_all = coef / model[0].scale_[:, np.newaxis]
            beta_full_raw[:, y_start:y_end] = beta_sub_all[:, col_idxs]

        baselines.append({
            "train_idx": train_idx,
            "test_idx": test_idx,
            "Y_train": np.concatenate(Y_train_parts, axis=1).astype(np.float32),
            "Y_test": np.concatenate(Y_test_parts, axis=1).astype(np.float32),
            "Y_hat_full_train": np.concatenate(Y_hat_train_parts, axis=1).astype(np.float32),
            "Y_hat_full_test": np.concatenate(Y_hat_test_parts, axis=1).astype(np.float32),
            "beta_full_raw": beta_full_raw,
        })

    return baselines


def apply_offset(averaged_embeddings, data, offset):
    """Apply word offset: trim embeddings and neural data to align them.

    Args:
        averaged_embeddings: Array of shape (n_words, n_features).
        data: Array of shape (n_words, n_elecs, n_lags) or (n_words, n_elecs*n_lags).
        offset: Integer word offset. Positive shifts embeddings forward in time;
                negative shifts them backward.

    Returns:
        Tuple of (X_arr, Y) both float32 with matching first dimension.
    """
    n_words = data.shape[0]
    if data.ndim == 3:
        Y = data.reshape(n_words, -1).astype(np.float32)
    else:
        Y = data.astype(np.float32)
    if offset < 0:
        return averaged_embeddings[:offset].astype(np.float32), Y[-offset:]
    elif offset > 0:
        return averaged_embeddings[offset:].astype(np.float32), Y[:-offset]
    return averaged_embeddings.astype(np.float32), Y


def save_subject_results(
    output_dir, sub_id, corrs, coefficients, intercepts, models,
    tr_idxs, test_idxs, scalers, elec_names, offset, layer_int, lags,
    corrs_arb=None,
):
    """Build results dict and write outputs to {output_dir}/subjects/sub_{sub_id}/.

    Always writes encoding_results.joblib and corrs.npy.
    If corrs_arb is not None, also writes arbitrary_corrs.npy and corrs_plot.png.

    Args:
        output_dir: Root output directory.
        sub_id: Subject BIDS ID.
        corrs: Correlation array (n_folds, n_elecs, n_lags).
        coefficients: List of coefficient arrays, one per fold.
        intercepts: List of intercept arrays, one per fold.
        models: List of fitted models.
        tr_idxs: List of train index arrays.
        test_idxs: List of test index arrays.
        scalers: List of fitted StandardScalers.
        elec_names: Electrode name list.
        offset: Word offset used during alignment.
        layer_int: Integer layer index.
        lags: Lag axis array.
        corrs_arb: Optional null-model correlation array (same shape as corrs).
    """
    from utils.plotting import plot_corrs_vs_arbitrary

    results = {
        "corrs": corrs,
        "coefficients": coefficients,
        "intercepts": intercepts,
        "models": models,
        "train_idxs": tr_idxs,
        "test_idxs": test_idxs,
        "scalers": scalers,
        "elec_names": list(elec_names),
        "subject_id": sub_id,
        "offset": offset,
        "layer": layer_int,
        "lags": lags,
    }
    if corrs_arb is not None:
        results["arb_corrs"] = corrs_arb

    sub_dir = os.path.join(output_dir, "subjects", f"sub_{sub_id}")
    os.makedirs(sub_dir, exist_ok=True)
    save_encoding_results(results, os.path.join(sub_dir, "encoding_results.joblib"))
    np.save(os.path.join(sub_dir, "corrs.npy"), corrs)
    if corrs_arb is not None:
        np.save(os.path.join(sub_dir, "arbitrary_corrs.npy"), corrs_arb)
        plot_corrs_vs_arbitrary(
            corrs, corrs_arb, lags,
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
        paths = [p for p in paths if int(os.path.basename(os.path.dirname(p)).split("_")[1]) in subjects_set]
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


def load_all_subject_coefficients(output_dir: str, subjects=None):
    """Load and concatenate model coefficients and intercepts across subjects.

    Args:
        output_dir: Root output directory containing subjects/sub_*/ subdirs.
        subjects: Optional list of subject IDs to load; loads all found if None.

    Returns:
        coefficients_all: np.ndarray of shape (n_folds, n_total_elecs, n_lags, n_features).
        intercepts_all: list of np.ndarray, one per fold, each shape (n_total_elecs * n_lags,).
        metadata_df: pd.DataFrame with columns ['subject', 'electrode'], one row per electrode.

    Raises:
        FileNotFoundError: If no subject directories are found (after filtering).
    """
    # Collect per-subject coefficient/intercept lists and metadata
    all_subject_coefs = []  # list of (list-of-folds per subject)
    all_subject_intercepts = []  # list of (list-of-folds per subject)
    rows = []
    for sub_id, results in _iter_subject_results(output_dir, subjects):
        coefficients = results["coefficients"]  # list of (n_elecs, n_lags, n_features) per fold
        intercepts = results["intercepts"]  # list of (n_elecs * n_lags,) per fold
        elec_names = results["elec_names"]
        all_subject_coefs.append(coefficients)
        all_subject_intercepts.append(intercepts)
        for name in elec_names:
            rows.append({"subject": sub_id, "electrode": name})

    n_folds = len(all_subject_coefs[0])
    # Stack per-fold concatenations into a single (n_folds, n_total_elecs, n_lags, n_features) array
    coefficients_all = np.stack([
        np.concatenate([subj_coefs[fold] for subj_coefs in all_subject_coefs], axis=0)
        for fold in range(n_folds)
    ])
    intercepts_all = [
        np.concatenate([subj_intercepts[fold] for subj_intercepts in all_subject_intercepts], axis=0)
        for fold in range(n_folds)
    ]
    metadata_df = pd.DataFrame(rows, columns=["subject", "electrode"])
    return coefficients_all, intercepts_all, metadata_df


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
