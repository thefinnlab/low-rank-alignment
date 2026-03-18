import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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

        if "torch" in get_backend().__name__:
            corr = corr.numpy(force=True)

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
    coef = ridge.coef_
    coef_np = coef.numpy(force=True) if hasattr(coef, "numpy") else np.asarray(coef)

    # coef_ shape from himalaya: (n_features, n_targets)
    raw_coefs_np = coef_np / scaler.scale_[:, np.newaxis]  # (n_features, n_targets)

    # himalaya Ridge does not fit an intercept; the de-standardization offset is:
    # intercept_raw = -(mean / scale) @ coef = -mean @ raw_coefs
    if hasattr(ridge, "intercept_"):
        intercept = ridge.intercept_
        intercept_np = intercept.numpy(force=True) if hasattr(intercept, "numpy") else np.asarray(intercept)
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
    """
    try:
        from himalaya.ridge import RidgeCV
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
