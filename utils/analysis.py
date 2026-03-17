import os

import joblib
import numpy as np
import torch


def get_correlation_gpu(array_a, array_b, device=None):
    """
    Compute the full cross-correlation matrix between two feature sets.

    Each column of array_a and array_b is z-scored before computing the
    pairwise dot product, yielding Pearson correlations between all column
    pairs.

    Args:
        array_a (np.ndarray): Shape (n_samples, n_features_a).
        array_b (np.ndarray): Shape (n_samples, n_features_b).
        device (str or torch.device, optional): Device to run on. Defaults to
            CUDA if available, else CPU.

    Returns:
        torch.Tensor: Shape (n_features_a, n_features_b) on CPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a = torch.tensor(array_a, device=device, dtype=torch.float32)
    b = torch.tensor(array_b, device=device, dtype=torch.float32)

    # Use population std (correction=0 / ddof=0) so the result equals
    # the Pearson correlation matrix: (z_a.T @ z_b) / n.
    a = (a - a.mean(dim=0)) / a.std(dim=0, correction=0)
    b = (b - b.mean(dim=0)) / b.std(dim=0, correction=0)

    corr_matrix = (a.T @ b) / array_a.shape[0]
    return corr_matrix.cpu()


def partial_out_direction(X, Y, directions, predictions_list, fold_indices, scalers):
    """
    Remove variance explained by a direction in embedding space from X and Y.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray, shape [n_words, n_dim]
        Word embeddings (one array per fold, or a single shared array).
    Y : np.ndarray, shape [n_words, n_elecs * n_timepoints]
        Neural data (raw, not z-scored).
    directions : list of list of np.ndarray, each of shape [n_dim]
        Direction to partial out, one per partial direction, one per fold.
    predictions_list : list of np.ndarray, each [n_words]
        One prediction vector per direction to partial out.
    fold_indices : list of np.ndarray
        Test indices for each fold.
    scalers : list of StandardScaler
        The fitted scaler from each fold's training set.

    Returns
    -------
    X_residuals : list of np.ndarray, each of shape [n_words, n_dim]
        X with the component along the direction removed, one per fold.
    Y_residual : np.ndarray, shape [n_words, n_elecs * n_timepoints]
        Y with the out-of-sample predicted variance subtracted in raw space.
    """
    predictions_stack = np.column_stack(predictions_list)

    Y_pred_full = np.zeros_like(Y)

    for idx, scaler in zip(fold_indices, scalers):
        fold_preds_raw = []
        for d_i in range(predictions_stack.shape[1]):
            pred_tiled = np.tile(
                predictions_stack[idx, d_i:d_i+1], (1, Y.shape[1])
            )
            fold_preds_raw.append(scaler.inverse_transform(pred_tiled))

        preds_matrix = np.stack(fold_preds_raw, axis=-1)  # [n_samples, n_cols, n_dirs]

        A = preds_matrix[:, 0, :]  # [n_samples, n_dirs]
        betas, _, _, _ = np.linalg.lstsq(A, Y[idx], rcond=None)
        Y_pred_full[idx] = A @ betas

    Y_residual = Y - Y_pred_full

    X_residuals = [X[i].copy() for i in range(len(directions[0]))]
    for fold_dir in directions:
        for i, direction in enumerate(fold_dir):
            d = direction / np.linalg.norm(direction)
            projections = X_residuals[i] @ d
            X_residuals[i] -= np.outer(projections, d)

    return X_residuals, Y_residual


def save_partial_results(
    partial_clusters,
    X,
    Y,
    all_mean_cluster_dirs,
    partialed_dirs,
    scalers=None,
    filename="cluster_results.npz",
):
    """Save partial-out results to a .npz file."""
    from utils.analysis import _get_partial_cluster_filepath
    partial_cluster_dir = _get_partial_cluster_filepath(partial_clusters)
    os.makedirs(partial_cluster_dir, exist_ok=True)
    print(f"Saving results to {partial_cluster_dir}")
    np.savez(
        os.path.join(partial_cluster_dir, filename),
        *X,
        Y=Y,
        all_mean_cluster_dirs=all_mean_cluster_dirs,
        partialed_dirs=np.array(partialed_dirs),
        num_X=np.array(len(X)),
    )

    if scalers is not None:
        joblib.dump(scalers, os.path.join(partial_cluster_dir, "scalers.joblib"))


def load_partial_results(partial_clusters, filename="cluster_results.npz"):
    """Load partial-out results saved by save_partial_results."""
    partial_cluster_dir = _get_partial_cluster_filepath(partial_clusters)
    data = np.load(os.path.join(partial_cluster_dir, filename), allow_pickle=False)
    num_X = int(data["num_X"])
    X = [data[f"arr_{i}"] for i in range(num_X)]
    Y = data["Y"]
    all_mean_cluster_dirs = data["all_mean_cluster_dirs"]
    partialed_dirs = data["partialed_dirs"].tolist()

    scalers_path = os.path.join(partial_cluster_dir, "scalers.joblib")
    scalers = joblib.load(scalers_path) if os.path.exists(scalers_path) else None

    return X, Y, all_mean_cluster_dirs, partialed_dirs, scalers


def _get_partial_cluster_filepath(partial_clusters):
    """
    Build a filesystem path fragment from a list of (num_clusters, cluster_ids) tuples.

    Args:
        partial_clusters: List of (num_clusters, [partialed_cluster_ids]) tuples.

    Returns:
        str: Relative directory path string.
    """
    end_path_directory = []
    for num_clusters, curr_cluster in partial_clusters:
        end_path_directory.append(
            f"{num_clusters}-" + "_".join([str(x) for x in sorted(curr_cluster)])
        )

    target_dir = ""
    if end_path_directory:
        target_dir = os.path.join(*end_path_directory)
    return target_dir


def run_comparative_analysis_on_folds(
    encoding_models,
    sig,
    X,
    Y,
    train_folds,
    num_electrodes,
    num_lags,
    n_components=50,
):
    """
    Run Dictionary Learning, ICA, and PLS on significant encoding model coefficients.

    Args:
        encoding_models: List of trained sklearn Pipelines (one per fold).
        sig: Boolean array of shape (num_electrodes, num_lags).
        X: List of feature arrays (one per fold).
        Y: Target array of shape (n_words, num_electrodes, num_lags).
        train_folds: List of training index arrays.
        num_electrodes: Number of electrodes.
        num_lags: Number of time lags.
        n_components: Number of components for decomposition methods.

    Returns:
        dict with keys 'dict_learning', 'icas', 'plss'.
    """
    from sklearn.decomposition import DictionaryLearning, FastICA
    from sklearn.cross_decomposition import PLSRegression
    from utils.encoding import absorb_scaler

    all_coefs = []
    for model in encoding_models:
        coef, _ = absorb_scaler(model, num_electrodes, num_lags)
        all_coefs.append(coef)

    dls = []
    icas = []
    plss = []

    for i, train_idx in enumerate(train_folds):
        W_sig = all_coefs[i][sig]

        print(f"Number of significant electrode-lags found: {W_sig.shape[0]}")

        if W_sig.shape[0] < n_components:
            print("Warning: More components requested than significant signals available.")
            n_components = W_sig.shape[0]

        dl = DictionaryLearning(n_components=n_components, transform_alpha=1.0, random_state=42)
        dls.append(dl)

        ica = FastICA(n_components=n_components, random_state=42)
        icas.append(ica)

        Y_reshaped = Y.reshape(X[0].shape[0], -1)
        sig_flat = sig.flatten()
        Y_sig_only = Y_reshaped[:, sig_flat]

        pls = PLSRegression(n_components=min(n_components, Y_sig_only.shape[1]))
        pls.fit(X[train_idx], Y_sig_only[train_idx])
        plss.append(pls)

    return {
        "dict_learning": dls,
        "icas": icas,
        "plss": plss,
    }


def get_method_mapping(fitted_model, W_sig, sig_mask, method='dl'):
    """
    Extract spatial-temporal maps for different dimensionality reduction methods.

    Args:
        fitted_model: Fitted DictionaryLearning, FastICA, or PLSRegression model.
        W_sig: Significant weight matrix of shape (n_sig, n_features).
        sig_mask: Boolean mask of shape (n_elecs, n_lags).
        method: One of 'dl', 'ica', 'pls'.

    Returns:
        np.ndarray of shape (n_components, n_elecs, n_lags).
    """
    n_elecs, n_lags = sig_mask.shape

    if method == 'dl':
        loadings = fitted_model.transform(W_sig)
    elif method == 'ica':
        loadings = fitted_model.transform(W_sig)
    elif method == 'pls':
        loadings = fitted_model.y_weights_
    else:
        raise ValueError("Method must be 'dl', 'ica', or 'pls'")

    n_components = loadings.shape[1]
    maps = np.zeros((n_components, n_elecs, n_lags))

    for i in range(n_components):
        maps[i][sig_mask] = np.abs(loadings[:, i])

    return maps
