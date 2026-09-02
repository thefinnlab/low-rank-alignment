"""Loading and aggregation of saved RRR/PCA subspace factors."""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.utils.extmath import randomized_svd


def _parse_fold(path: Path) -> int | None:
    name = path.parent.name
    if not name.startswith("fold_"):
        return None
    suffix = name.split("_", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def orthonormal_basis(matrix: np.ndarray, rank: int | None = None) -> np.ndarray:
    """Return an orthonormal basis for the leading columns of ``matrix``."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}")
    if rank is None:
        rank = arr.shape[1]
    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if rank > arr.shape[1]:
        raise ValueError(f"rank {rank} exceeds matrix width {arr.shape[1]}")
    q, r = np.linalg.qr(arr[:, :rank], mode="reduced")
    keep = np.abs(np.diag(r)) > np.finfo(np.float64).eps * max(arr.shape)
    if not np.any(keep):
        raise ValueError("selected columns are numerically rank deficient")
    return q[:, keep]


def load_unit_subspace_matrices(unit_dir: str | Path) -> list[np.ndarray]:
    """Load fold-level ``A.npy`` matrices directly under a unit directory."""
    root = Path(unit_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Subspace unit directory does not exist: {root}")

    paths = sorted(
        path
        for path in root.glob("fold_*/A.npy")
        if _parse_fold(path) is not None
    )
    if not paths:
        raise FileNotFoundError(f"No fold_*/A.npy factors found under {root}")
    return [np.load(path) for path in paths]


def extrinsic_mean_basis(
    matrices: list[np.ndarray],
    rank: int,
) -> np.ndarray:
    """Compute the extrinsic Grassmann mean of leading matrix subspaces.

    Each input matrix contributes the subspace spanned by its first ``rank``
    columns. The returned columns are the top eigenvectors of the average
    projection matrix.
    """
    if not matrices:
        raise ValueError("At least one subspace matrix is required")
    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be at least 1")

    first = np.asarray(matrices[0])
    if first.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got shape {first.shape}")
    ambient_dim = first.shape[0]
    if rank > first.shape[1]:
        raise ValueError(
            f"rank {rank} exceeds source matrix width {first.shape[1]}"
        )

    bases = []
    for i, matrix in enumerate(matrices):
        arr = np.asarray(matrix)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D matrix at index {i}, got shape {arr.shape}")
        if arr.shape[0] != ambient_dim:
            raise ValueError(
                "All subspace matrices must share ambient dimension, got "
                f"{ambient_dim} and {arr.shape[0]} at index {i}"
            )
        if rank > arr.shape[1]:
            raise ValueError(
                f"rank {rank} exceeds source matrix width {arr.shape[1]} at index {i}"
            )
        basis = orthonormal_basis(arr, rank)
        if basis.shape[1] != rank:
            raise ValueError(
                f"Source matrix at index {i} is rank deficient for requested rank {rank}"
            )
        bases.append(basis)

    concatenated = np.concatenate(bases, axis=1)
    left_vectors, _, _ = np.linalg.svd(concatenated, full_matrices=False)
    return left_vectors[:, :rank].astype(np.float32, copy=False)


def factor_weighted_mean_basis(
    matrices: list[np.ndarray],
    rank: int,
) -> np.ndarray:
    """Aggregate leading factors with their raw magnitudes as weights.

    The returned basis contains the leading left singular vectors of the raw
    ``matrix[:, :rank]`` blocks concatenated across inputs.  Consequently both
    between-matrix scale and within-matrix singular-value structure contribute
    to the selected subspace, while the returned basis itself is orthonormal.
    """
    if not matrices:
        raise ValueError("At least one subspace matrix is required")
    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be at least 1")

    first = np.asarray(matrices[0])
    if first.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got shape {first.shape}")
    ambient_dim = first.shape[0]
    blocks = []
    for i, matrix in enumerate(matrices):
        arr = np.asarray(matrix)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D matrix at index {i}, got shape {arr.shape}")
        if arr.shape[0] != ambient_dim:
            raise ValueError(
                "All subspace matrices must share ambient dimension, got "
                f"{ambient_dim} and {arr.shape[0]} at index {i}"
            )
        if rank > arr.shape[1]:
            raise ValueError(
                f"rank {rank} exceeds source matrix width {arr.shape[1]} at index {i}"
            )
        blocks.append(arr[:, :rank])

    concatenated = np.concatenate(blocks, axis=1)
    left_vectors, singular_values, _ = np.linalg.svd(
        concatenated, full_matrices=False
    )
    if (
        np.count_nonzero(singular_values > np.finfo(singular_values.dtype).eps)
        < rank
    ):
        raise ValueError(
            f"Aggregated source matrices are rank deficient for rank {rank}"
        )
    return left_vectors[:, :rank].astype(np.float32, copy=False)


def prediction_weighted_mean_basis(
    matrices: list[np.ndarray],
    rank: int,
    prediction_singular_values: np.ndarray,
) -> np.ndarray:
    """Aggregate FIR factors using fitted-prediction energy.

    For prediction component ``j``, the complete factor across all FIR delays is
    normalized before its delay blocks are scaled by the corresponding prediction
    singular value. Thus the component contributes total squared energy ``S_j^2``
    without retaining arbitrary coefficient scale, while relative strength across
    its delay blocks is preserved.
    """
    if not matrices:
        raise ValueError("At least one subspace matrix is required")
    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be at least 1")

    singular_values = np.asarray(prediction_singular_values, dtype=np.float64)
    if singular_values.ndim != 1:
        raise ValueError(
            "prediction_singular_values must be one-dimensional, got shape "
            f"{singular_values.shape}"
        )
    if len(singular_values) < rank:
        raise ValueError(
            f"rank {rank} exceeds prediction singular-value count "
            f"{len(singular_values)}"
        )
    selected_singular_values = singular_values[:rank]
    if not np.isfinite(selected_singular_values).all():
        raise ValueError("prediction singular values must be finite")
    if np.any(selected_singular_values < 0):
        raise ValueError("prediction singular values must be non-negative")

    first = np.asarray(matrices[0], dtype=np.float64)
    if first.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got shape {first.shape}")
    ambient_dim = first.shape[0]
    blocks = []
    for i, matrix in enumerate(matrices):
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D matrix at index {i}, got shape {arr.shape}")
        if arr.shape[0] != ambient_dim:
            raise ValueError(
                "All subspace matrices must share ambient dimension, got "
                f"{ambient_dim} and {arr.shape[0]} at index {i}"
            )
        if rank > arr.shape[1]:
            raise ValueError(
                f"rank {rank} exceeds source matrix width {arr.shape[1]} at index {i}"
            )
        blocks.append(arr[:, :rank])

    full_components = np.concatenate(blocks, axis=0)
    component_norms = np.linalg.norm(full_components, axis=0)
    scale = np.zeros(rank, dtype=np.float64)
    nonzero = component_norms > np.finfo(np.float64).eps * max(full_components.shape)
    inconsistent = (~nonzero) & (selected_singular_values > 0)
    if np.any(inconsistent):
        indices = np.flatnonzero(inconsistent).tolist()
        raise ValueError(
            "Prediction components have positive singular values but zero FIR "
            f"factor norm at indices {indices}"
        )
    scale[nonzero] = selected_singular_values[nonzero] / component_norms[nonzero]
    weighted = np.concatenate(
        [block * scale[np.newaxis, :] for block in blocks], axis=1
    )
    left_vectors, aggregate_singular_values, _ = np.linalg.svd(
        weighted, full_matrices=False
    )
    tolerance = (
        np.finfo(aggregate_singular_values.dtype).eps
        * max(weighted.shape)
        * aggregate_singular_values[0]
        if aggregate_singular_values.size
        else 0.0
    )
    if np.count_nonzero(aggregate_singular_values > tolerance) < rank:
        raise ValueError(
            f"Prediction-weighted source matrices are rank deficient for rank {rank}"
        )
    return left_vectors[:, :rank].astype(np.float32, copy=False)


def aggregate_subspace_basis(
    matrices: list[np.ndarray],
    rank: int,
    method: str = "extrinsic",
    *,
    prediction_singular_values: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate matrices into a rank-k basis using the requested method."""
    if method == "extrinsic":
        return extrinsic_mean_basis(matrices, rank=rank)
    if method == "factor_weighted":
        return factor_weighted_mean_basis(matrices, rank=rank)
    if method == "prediction_weighted":
        if prediction_singular_values is None:
            raise ValueError(
                "prediction_weighted aggregation requires "
                "prediction_singular_values"
            )
        return prediction_weighted_mean_basis(
            matrices,
            rank=rank,
            prediction_singular_values=prediction_singular_values,
        )
    raise ValueError(
        "method must be 'extrinsic', 'factor_weighted', or "
        f"'prediction_weighted', got {method!r}"
    )


def aggregate_subspace_from_unit_dir(
    unit_dir: str | Path,
    rank: int,
    method: str = "extrinsic",
) -> np.ndarray:
    """Load fold factors from a unit directory and aggregate them."""
    return aggregate_subspace_basis(
        load_unit_subspace_matrices(unit_dir), rank=rank, method=method
    )


def validate_aggregation_cache(
    output_dir: str | Path,
    method: str,
    *,
    refresh: bool = False,
) -> None:
    """Record aggregation provenance and reject incompatible cached outputs."""
    if method not in {"extrinsic", "factor_weighted"}:
        raise ValueError(f"Unsupported subspace aggregation method: {method!r}")
    root = Path(output_dir)
    marker = root / "subspace_aggregation.json"
    if marker.exists():
        existing = json.loads(marker.read_text()).get("method", "extrinsic")
        if existing != method and not refresh:
            raise ValueError(
                f"Existing results in {root} use subspace aggregation {existing!r}, "
                f"but {method!r} was requested. Set refresh=true or use another "
                "output directory."
            )
    elif (
        method != "extrinsic"
        and root.exists()
        and any(path.name != "config.yml" for path in root.iterdir())
        and not refresh
    ):
        raise ValueError(
            f"Existing legacy results in {root} are assumed to use 'extrinsic'; "
            f"cannot reuse them with {method!r}. Set refresh=true or use another "
            "output directory."
        )
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"method": method}, indent=2))
