"""Utilities for discovering and comparing saved subspace factors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.utils.extmath import randomized_svd


@dataclass(frozen=True)
class SubspaceSource:
    """One fold-level subspace matrix saved as ``A.npy``."""

    label: str
    path: Path
    unit_dir: Path
    fold: int
    kind: str
    max_rank: int
    ambient_dim: int


@dataclass(frozen=True)
class MaxvarGCCAResult:
    """A fitted ridge-regularized MAXVAR GCCA model."""

    view_names: tuple[str, ...]
    means: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    rotations: dict[str, np.ndarray]
    shared_scores: np.ndarray
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    view_support: dict[str, np.ndarray]
    objective_contributions: dict[str, np.ndarray]
    contribution_shares: dict[str, np.ndarray]
    weights: dict[str, float]
    alpha: float
    ridge_values: dict[str, float]
    effective_ranks: dict[str, int]


def _parse_fold(path: Path) -> int | None:
    name = path.parent.name
    if not name.startswith("fold_"):
        return None
    suffix = name.split("_", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def _infer_kind(unit_dir: Path) -> str:
    parts = unit_dir.parts
    for kind in (
        "per_subject_rrr",
        "per_subject_pca",
        "per_region_rrr",
        "per_region_pca",
        "loso_pca",
        "loso_rrr",
        "rrr",
        "pca",
    ):
        if kind in parts:
            return kind
    return "unknown"


def is_pca_source_kind(kind: str) -> bool:
    """Return whether a discovered subspace kind is a PCA baseline."""
    return kind.endswith("pca")


def default_cross_rank_text(max_rank: int = 1600) -> str:
    """Return the default cross-rank sweep text."""
    ranks = []
    rank = 1
    while rank <= max_rank:
        ranks.append(rank)
        rank *= 2
    if max_rank not in ranks:
        ranks.append(max_rank)
    return ",".join(str(rank) for rank in ranks)


def parse_rank_list(text: str) -> list[int]:
    """Parse comma-separated ranks and inclusive ranges into sorted ranks."""
    ranks = []
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = [bound.strip() for bound in part.split("-", 1)]
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(f"Invalid rank range: {part!r}")
            start, stop = (int(bounds[0]), int(bounds[1]))
            if start < 1 or stop < 1:
                raise ValueError("Ranks must be positive integers")
            if stop < start:
                raise ValueError(f"Rank range must be increasing: {part!r}")
            ranks.extend(range(start, stop + 1))
        else:
            rank = int(part)
            if rank < 1:
                raise ValueError("Ranks must be positive integers")
            ranks.append(rank)
    if not ranks:
        raise ValueError("Enter at least one rank")
    return sorted(set(ranks))


def filter_available_ranks(
    ranks: list[int],
    max_rank: int,
) -> tuple[list[int], list[int]]:
    """Split requested ranks into ranks available at or below ``max_rank`` and skipped."""
    max_rank = int(max_rank)
    available = [int(rank) for rank in ranks if int(rank) <= max_rank]
    skipped = [int(rank) for rank in ranks if int(rank) > max_rank]
    return available, skipped


def _short_label(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return str(rel.parent)


def discover_subspace_sources(roots: list[str | Path]) -> list[SubspaceSource]:
    """Find fold-level ``A.npy`` factors under the provided roots.

    Args:
        roots: Directories to recursively scan.

    Returns:
        Sorted, de-duplicated list of discovered subspace sources.
    """
    sources: list[SubspaceSource] = []
    seen: set[Path] = set()
    for root_like in roots:
        root = Path(root_like).expanduser()
        if not root.exists():
            continue
        root = root.resolve()
        for path in sorted(root.rglob("A.npy")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            fold = _parse_fold(path)
            if fold is None:
                continue
            try:
                arr = np.load(path, mmap_mode="r")
                ambient_dim, max_rank = int(arr.shape[0]), int(arr.shape[1])
            except Exception:
                continue
            unit_dir = path.parent.parent
            kind = _infer_kind(unit_dir)
            label = _short_label(path, root)
            sources.append(
                SubspaceSource(
                    label=label,
                    path=path,
                    unit_dir=unit_dir,
                    fold=fold,
                    kind=kind,
                    max_rank=max_rank,
                    ambient_dim=ambient_dim,
                )
            )
    return sorted(sources, key=lambda s: (s.label, s.fold, str(s.path)))


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


def principal_angles_from_bases(
    basis_a: np.ndarray, basis_b: np.ndarray
) -> np.ndarray:
    """Compute principal angles in radians from two orthonormal bases."""
    if basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError(
            "Subspaces must have the same ambient dimension, got "
            f"{basis_a.shape[0]} and {basis_b.shape[0]}"
        )
    singular_values = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return np.arccos(singular_values)


def principal_angles(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    rank_a: int,
    rank_b: int,
) -> np.ndarray:
    """Compute principal angles in radians for leading-ranked subspaces."""
    basis_a = orthonormal_basis(matrix_a, rank_a)
    basis_b = orthonormal_basis(matrix_b, rank_b)
    return principal_angles_from_bases(basis_a, basis_b)


def subspace_distance_metrics(angles: np.ndarray) -> dict[str, float]:
    """Compute standard distances from principal angles in radians."""
    angles = np.asarray(angles, dtype=np.float64)
    cos_product = float(np.prod(np.cos(angles)))
    cos_product = float(np.clip(cos_product, -1.0, 1.0))
    return {
        "grassmann": float(np.sqrt(np.sum(angles**2))),
        "spectral": float(np.sin(np.max(angles))) if angles.size else float("nan"),
        "fubini_study": float(np.arccos(cos_product)),
    }


def sample_random_principal_angles(
    ambient_dim: int,
    rank_a: int,
    rank_b: int,
    n_draws: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample principal angles between random subspaces."""
    if ambient_dim < max(rank_a, rank_b):
        raise ValueError("ambient_dim must be at least both ranks")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_draws)):
        basis_a = orthonormal_basis(rng.standard_normal((ambient_dim, rank_a)))
        basis_b = orthonormal_basis(rng.standard_normal((ambient_dim, rank_b)))
        draws.append(principal_angles_from_bases(basis_a, basis_b))
    return np.stack(draws, axis=0)


def projected_rsm(vectors: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Return a representational dissimilarity matrix for row vectors."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}")
    if metric == "cosine":
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe = np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 0)
        similarities = safe @ safe.T
        rsm = 1.0 - np.clip(similarities, -1.0, 1.0)
    elif metric == "euclidean":
        sq_norms = np.sum(arr**2, axis=1, keepdims=True)
        sq_distances = np.maximum(sq_norms + sq_norms.T - 2.0 * (arr @ arr.T), 0.0)
        rsm = np.sqrt(sq_distances)
    else:
        raise ValueError(f"Unknown RSM metric: {metric!r}")
    np.fill_diagonal(rsm, 0.0)
    return rsm


def lower_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Return lower-triangular entries below the diagonal."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {arr.shape}")
    rows, cols = np.tril_indices(arr.shape[0], k=-1)
    return arr[rows, cols]


def rsa_lower_triangle_correlations(
    rsm_a: np.ndarray,
    rsm_b: np.ndarray,
) -> dict[str, float]:
    """Correlate lower-triangle RSM entries with Pearson and Spearman."""
    vec_a = lower_triangle_values(rsm_a)
    vec_b = lower_triangle_values(rsm_b)
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            "RSM lower triangles must have the same length, got "
            f"{vec_a.shape[0]} and {vec_b.shape[0]}"
        )
    if vec_a.size < 2:
        return {"pearson": float("nan"), "spearman": float("nan")}
    if np.std(vec_a) == 0 or np.std(vec_b) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(vec_a, vec_b)[0, 1])
    spearman = float(spearmanr(vec_a, vec_b).statistic)
    return {"pearson": pearson, "spearman": spearman}


def validate_matching_fold_indices(test_idxs_a: list, test_idxs_b: list) -> None:
    """Validate that two within-dataset comparisons use identical fold test rows."""
    if len(test_idxs_a) != len(test_idxs_b):
        raise ValueError(
            f"Within-dataset RSA requires the same number of folds, got "
            f"{len(test_idxs_a)} and {len(test_idxs_b)}"
        )
    for fold, (idx_a, idx_b) in enumerate(zip(test_idxs_a, test_idxs_b)):
        if not np.array_equal(np.asarray(idx_a), np.asarray(idx_b)):
            raise ValueError(
                "Within-dataset RSA requires matching test indices for each fold; "
                f"fold {fold} differs."
            )


def rsa_for_projected_pair(
    embeddings: np.ndarray,
    test_idx: np.ndarray,
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    metric: str = "cosine",
) -> dict:
    """Compute one fold's projected RSMs and RSA correlations."""
    test_embeddings = np.asarray(embeddings)[np.asarray(test_idx, dtype=int)]
    rsm_a = projected_rsm(test_embeddings @ basis_a, metric=metric)
    rsm_b = projected_rsm(test_embeddings @ basis_b, metric=metric)
    return {
        "rsm_a": rsm_a,
        "rsm_b": rsm_b,
        **rsa_lower_triangle_correlations(rsm_a, rsm_b),
    }


def score_projected_rsa_alignment(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    test_idx: np.ndarray,
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    metric: str = "cosine",
    *,
    return_vectors: bool = False,
) -> dict:
    """Score RSA between two projected embedding matrices on one held-out fold."""
    idx = np.asarray(test_idx, dtype=int)
    rsm_a = projected_rsm(np.asarray(embeddings_a)[idx] @ basis_a, metric=metric)
    rsm_b = projected_rsm(np.asarray(embeddings_b)[idx] @ basis_b, metric=metric)
    result = {
        "n_test": int(idx.size),
        **rsa_lower_triangle_correlations(rsm_a, rsm_b),
    }
    if return_vectors:
        result["rsm_a_vector"] = lower_triangle_values(rsm_a)
        result["rsm_b_vector"] = lower_triangle_values(rsm_b)
    return result


def summarize_rsa_fold_results(results: list[dict]) -> dict[str, float]:
    """Return mean RSA metrics across fold result dictionaries."""
    return {
        "mean_pearson": float(np.nanmean([row["pearson"] for row in results])),
        "mean_spearman": float(np.nanmean([row["spearman"] for row in results])),
    }


def fit_scaled_orthogonal_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    *,
    allow_scale: bool = True,
) -> dict[str, np.ndarray | float]:
    """Fit a centered orthogonal Procrustes map from ``source`` to ``target``.

    The returned map applies as ``scale * ((X - source_mean) @ rotation)`` and
    predicts target coordinates centered by ``target_mean``.
    """
    source_arr = np.asarray(source, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    if source_arr.ndim != 2 or target_arr.ndim != 2:
        raise ValueError("source and target must both be 2D arrays")
    if source_arr.shape != target_arr.shape:
        raise ValueError(
            "Procrustes source and target must have matching shapes, got "
            f"{source_arr.shape} and {target_arr.shape}"
        )
    if source_arr.shape[0] < 2:
        raise ValueError("Procrustes alignment needs at least two training rows")

    source_mean = source_arr.mean(axis=0)
    target_mean = target_arr.mean(axis=0)
    source_centered = source_arr - source_mean
    target_centered = target_arr - target_mean
    source_norm_sq = float(np.sum(source_centered**2))
    if source_norm_sq <= np.finfo(np.float64).eps:
        raise ValueError("Procrustes source training rows have zero variance")

    u, singular_values, vt = np.linalg.svd(
        source_centered.T @ target_centered,
        full_matrices=False,
    )
    rotation = u @ vt
    scale = float(np.sum(singular_values) / source_norm_sq) if allow_scale else 1.0
    return {
        "rotation": rotation,
        "scale": scale,
        "source_mean": source_mean,
        "target_mean": target_mean,
        "singular_values": singular_values,
    }


def apply_scaled_orthogonal_procrustes(
    source: np.ndarray,
    fit: dict[str, np.ndarray | float],
) -> np.ndarray:
    """Apply a fitted scaled orthogonal Procrustes map to source rows."""
    source_arr = np.asarray(source, dtype=np.float64)
    rotation = np.asarray(fit["rotation"], dtype=np.float64)
    source_mean = np.asarray(fit["source_mean"], dtype=np.float64)
    scale = float(fit["scale"])
    if source_arr.ndim != 2:
        raise ValueError("source must be a 2D array")
    if source_arr.shape[1] != rotation.shape[0]:
        raise ValueError(
            "Procrustes source dimension does not match fitted rotation, got "
            f"{source_arr.shape[1]} and {rotation.shape[0]}"
        )
    return scale * ((source_arr - source_mean) @ rotation)


def score_procrustes_alignment(
    source_train: np.ndarray,
    target_train: np.ndarray,
    source_test: np.ndarray,
    target_test: np.ndarray,
    *,
    allow_scale: bool = True,
) -> dict:
    """Fit Procrustes on train rows and score aligned source rows on test rows."""
    fit = fit_scaled_orthogonal_procrustes(
        source_train,
        target_train,
        allow_scale=allow_scale,
    )
    target_test_arr = np.asarray(target_test, dtype=np.float64)
    aligned_test = apply_scaled_orthogonal_procrustes(source_test, fit)
    target_centered_test = target_test_arr - np.asarray(fit["target_mean"])
    residual = aligned_test - target_centered_test

    sse = float(np.sum(residual**2))
    target_sse = float(np.sum(target_centered_test**2))
    normalized_mse = sse / target_sse if target_sse > 0 else float("nan")
    row_norms = np.linalg.norm(aligned_test, axis=1) * np.linalg.norm(
        target_centered_test,
        axis=1,
    )
    row_dots = np.sum(aligned_test * target_centered_test, axis=1)
    row_cosines = np.divide(
        row_dots,
        row_norms,
        out=np.full_like(row_dots, np.nan, dtype=np.float64),
        where=row_norms > 0,
    )
    return {
        "fit": fit,
        "aligned_test": aligned_test,
        "target_centered_test": target_centered_test,
        "scale": float(fit["scale"]),
        "test_rmse": float(np.sqrt(np.mean(residual**2))),
        "test_normalized_mse": float(normalized_mse),
        "test_r2": float(1.0 - normalized_mse) if np.isfinite(normalized_mse) else float("nan"),
        "test_mean_row_cosine": float(np.nanmean(row_cosines)),
    }


def summarize_procrustes_fold_results(results: list[dict]) -> dict[str, float]:
    """Return mean Procrustes metrics across fold result dictionaries."""
    return {
        "mean_test_rmse": float(np.nanmean([row["test_rmse"] for row in results])),
        "mean_test_normalized_mse": float(
            np.nanmean([row["test_normalized_mse"] for row in results])
        ),
        "mean_test_r2": float(np.nanmean([row["test_r2"] for row in results])),
        "mean_test_mean_row_cosine": float(
            np.nanmean([row["test_mean_row_cosine"] for row in results])
        ),
    }


def _validate_maxvar_views(views: dict[str, np.ndarray]) -> tuple[tuple[str, ...], int]:
    if len(views) < 2:
        raise ValueError("MAXVAR GCCA requires at least two views")
    names = tuple(views)
    row_counts = set()
    for name, values in views.items():
        arr = np.asarray(values)
        if arr.ndim != 2 or arr.shape[1] < 1:
            raise ValueError(f"MAXVAR view {name!r} must be a non-empty 2D array")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"MAXVAR view {name!r} contains non-finite values")
        row_counts.add(int(arr.shape[0]))
    if len(row_counts) != 1:
        raise ValueError(f"MAXVAR views must share rows, got counts {sorted(row_counts)}")
    n_rows = row_counts.pop()
    if n_rows < 2:
        raise ValueError("MAXVAR GCCA requires at least two rows")
    return names, n_rows


def _symmetric_ridge_inverse(gram, ridge, *, rcond):
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    threshold = float(rcond) * max(float(eigenvalues[-1]), 1.0)
    keep = eigenvalues > threshold
    if ridge == 0.0:
        inverse_values = np.zeros_like(eigenvalues)
        inverse_values[keep] = 1.0 / eigenvalues[keep]
    else:
        inverse_values = 1.0 / (eigenvalues + float(ridge))
    inverse = (eigenvectors * inverse_values) @ eigenvectors.T
    inverse_sqrt = (eigenvectors * np.sqrt(inverse_values)) @ eigenvectors.T
    return inverse, inverse_sqrt, int(np.sum(keep))


def fit_regularized_maxvar_gcca(
    views: dict[str, np.ndarray],
    n_components: int,
    *,
    alpha: float = 0.0,
    weights: dict[str, float] | None = None,
    scale: bool = False,
    rcond: float = 1e-10,
) -> MaxvarGCCAResult:
    """Fit MAXVAR GCCA using dimensionless per-view ridge shrinkage."""
    names, n_rows = _validate_maxvar_views(views)
    n_components, alpha = int(n_components), float(alpha)
    if n_components < 1 or alpha < 0 or rcond <= 0:
        raise ValueError("n_components and rcond must be positive; alpha must be non-negative")
    if weights is None:
        weights = {name: 1.0 for name in names}
    if set(weights) != set(names) or any(float(weights[name]) <= 0 for name in names):
        raise ValueError("MAXVAR weights must contain positive values for every view")

    means, scales, standardized = {}, {}, {}
    inverses, ridge_values, effective_ranks = {}, {}, {}
    blocks = []
    for name in names:
        arr = np.asarray(views[name], dtype=np.float64)
        mean = np.mean(arr, axis=0)
        centered = arr - mean
        std = np.std(centered, axis=0)
        safe_scale = (
            np.where(std > np.finfo(np.float64).eps, std, 1.0)
            if scale
            else np.ones(arr.shape[1])
        )
        z = centered / safe_scale
        gram = z.T @ z
        ridge = alpha * float(np.trace(gram)) / float(z.shape[1])
        inverse, inverse_sqrt, effective_rank = _symmetric_ridge_inverse(
            gram, ridge, rcond=float(rcond)
        )
        means[name], scales[name], standardized[name] = mean, safe_scale, z
        inverses[name], ridge_values[name] = inverse, ridge
        effective_ranks[name] = effective_rank
        blocks.append(np.sqrt(float(weights[name])) * (z @ inverse_sqrt))

    compound = np.concatenate(blocks, axis=1)
    compound_rank = int(np.linalg.matrix_rank(compound, tol=float(rcond) * np.linalg.norm(compound, 2)))
    capacity = min(n_rows - 1, sum(effective_ranks.values()), compound_rank)
    if n_components > capacity:
        raise ValueError(f"n_components={n_components} exceeds MAXVAR capacity {capacity}")
    if compound.size <= 2_000_000 or n_components >= min(compound.shape):
        left, singular_values, _ = np.linalg.svd(compound, full_matrices=False)
    else:
        left, singular_values, _ = randomized_svd(
            compound,
            n_components=n_components,
            n_oversamples=max(10, n_components),
            n_iter=5,
            random_state=0,
        )
    singular_values = singular_values[:n_components]
    zero_threshold = float(rcond) * max(float(singular_values[0]), 1.0)
    if singular_values[-1] <= zero_threshold:
        raise ValueError(
            f"Requested component {n_components} has numerically zero compound singular value"
        )
    shared_scores = left[:, :n_components]
    rotations = {
        name: inverses[name] @ standardized[name].T @ shared_scores for name in names
    }
    view_support = {
        name: np.einsum(
            "ij,ij->j", shared_scores, standardized[name] @ rotations[name]
        )
        for name in names
    }
    objective_contributions = {
        name: float(weights[name]) * view_support[name] for name in names
    }
    contribution_total = np.sum(np.stack(list(objective_contributions.values())), axis=0)
    contribution_shares = {
        name: np.divide(
            objective_contributions[name],
            contribution_total,
            out=np.zeros_like(contribution_total),
            where=contribution_total > 0,
        )
        for name in names
    }
    return MaxvarGCCAResult(
        view_names=names,
        means=means,
        scales=scales,
        rotations=rotations,
        shared_scores=shared_scores,
        eigenvalues=singular_values**2,
        singular_values=singular_values,
        view_support=view_support,
        objective_contributions=objective_contributions,
        contribution_shares=contribution_shares,
        weights={name: float(weights[name]) for name in names},
        alpha=alpha,
        ridge_values=ridge_values,
        effective_ranks=effective_ranks,
    )


def select_maxvar_support_views(result, *, cumulative_mass=0.8, min_views=2):
    """Select the smallest training-support set reaching cumulative contribution mass."""
    threshold = float(cumulative_mass)
    minimum = int(min_views)
    if not 0 < threshold <= 1 or minimum < 1 or minimum > len(result.view_names):
        raise ValueError("cumulative_mass must be in (0, 1] and min_views must be valid")
    selected = []
    for component in range(result.shared_scores.shape[1]):
        ordered = sorted(
            result.view_names,
            key=lambda name: (-result.contribution_shares[name][component], name),
        )
        chosen, mass = [], 0.0
        for name in ordered:
            chosen.append(name)
            mass += float(result.contribution_shares[name][component])
            if len(chosen) >= minimum and mass >= threshold:
                break
        selected.append(tuple(chosen))
    return tuple(selected)


def score_maxvar_partial_support(
    result, train_views, test_views, *, support_views=None, cumulative_mass=0.8,
    n_components=None,
):
    """Score each axis using its training-selected views and retain all-view diagnostics."""
    all_view = score_maxvar_leave_one_view_out(
        result, train_views, test_views, n_components=n_components
    )
    keep = all_view["test_scores"][result.view_names[0]].shape[1]
    supports = (
        select_maxvar_support_views(result, cumulative_mass=cumulative_mass)[:keep]
        if support_views is None else tuple(tuple(group) for group in support_views[:keep])
    )
    selected_corrs = {name: np.full(keep, np.nan) for name in result.view_names}
    consensus_corrs = {name: np.full(keep, np.nan) for name in result.view_names}
    component_scores = np.full(keep, np.nan)
    normalized = all_view["normalized_test_scores"]
    for component, group in enumerate(supports):
        if len(group) < 2:
            continue
        per_selected = []
        for name in group:
            peers = [normalized[peer][:, [component]] for peer in group if peer != name]
            corr = _componentwise_pearson(
                normalized[name][:, [component]], np.mean(np.stack(peers), axis=0)
            )[0]
            selected_corrs[name][component] = corr
            per_selected.append(corr)
        component_scores[component] = float(np.nanmean(per_selected))
        consensus = np.mean(np.stack([normalized[name][:, component] for name in group]), axis=0)
        for name in result.view_names:
            consensus_corrs[name][component] = _componentwise_pearson(
                normalized[name][:, [component]], consensus[:, None]
            )[0]
    return {
        **all_view,
        "support_views": supports,
        "support_counts": np.asarray([len(group) for group in supports], dtype=int),
        "selected_view_corrs": selected_corrs,
        "selected_consensus_corrs": consensus_corrs,
        "component_support_scores": component_scores,
        "mean_partial_support_corr": float(np.nanmean(component_scores)),
    }


def transform_maxvar_gcca_view(result, view_name, values, *, n_components=None):
    """Map rows from one fitted view into shared MAXVAR coordinates."""
    if view_name not in result.rotations:
        raise KeyError(f"Unknown MAXVAR view {view_name!r}")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != result.means[view_name].size:
        raise ValueError(f"View {view_name!r} has incompatible shape {arr.shape}")
    available = result.rotations[view_name].shape[1]
    keep = available if n_components is None else int(n_components)
    if keep < 1 or keep > available:
        raise ValueError(f"n_components must be in [1, {available}]")
    z = (arr - result.means[view_name]) / result.scales[view_name]
    return z @ result.rotations[view_name][:, :keep]


def maxvar_ambient_map(result, view_name, basis, *, n_components=None):
    """Return ``(C, b)`` so raw embeddings map to shared scores as ``X @ C + b``."""
    q = np.asarray(basis, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != result.means[view_name].size:
        raise ValueError("Basis width must match the fitted projected-view width")
    available = result.rotations[view_name].shape[1]
    keep = available if n_components is None else int(n_components)
    rotation = result.rotations[view_name][:, :keep]
    scaled_rotation = rotation / result.scales[view_name][:, None]
    return q @ scaled_rotation, -result.means[view_name] @ scaled_rotation


def score_maxvar_leave_one_view_out(
    result, train_views, test_views, *, n_components=None
):
    """Score held-out view estimates against the consensus of all other views."""
    if tuple(train_views) != result.view_names or tuple(test_views) != result.view_names:
        raise ValueError("Train/test view order must match the fitted MAXVAR model")
    available = result.shared_scores.shape[1]
    keep = available if n_components is None else int(n_components)
    train_scores = {
        name: transform_maxvar_gcca_view(result, name, train_views[name], n_components=keep)
        for name in result.view_names
    }
    test_scores = {
        name: transform_maxvar_gcca_view(result, name, test_views[name], n_components=keep)
        for name in result.view_names
    }
    normalized = {}
    for name in result.view_names:
        mean, std = np.mean(train_scores[name], axis=0), np.std(train_scores[name], axis=0)
        std = np.where(std > np.finfo(np.float64).eps, std, 1.0)
        normalized[name] = (test_scores[name] - mean) / std
    per_view = {}
    for name in result.view_names:
        consensus = np.mean(
            np.stack([normalized[peer] for peer in result.view_names if peer != name]),
            axis=0,
        )
        per_view[name] = _componentwise_pearson(normalized[name], consensus)
    pairwise = {
        (left, right): _componentwise_pearson(normalized[left], normalized[right])
        for i, left in enumerate(result.view_names)
        for right in result.view_names[i + 1 :]
    }
    return {
        "mean_loo_corr": float(np.nanmean(np.concatenate(list(per_view.values())))),
        "mean_pairwise_corr": float(np.nanmean(np.concatenate(list(pairwise.values())))),
        "per_view_corrs": per_view,
        "pairwise_corrs": pairwise,
        "test_scores": test_scores,
        "normalized_test_scores": normalized,
    }


def select_maxvar_alpha(
    views, inner_splits, shared_dimensions, alphas, *, rcond=1e-10,
    support_cumulative_mass=0.8,
):
    """Select ridge per dimension using only its training-selected support views."""
    names, _ = _validate_maxvar_views(views)
    dimensions = sorted(set(int(value) for value in shared_dimensions))
    candidates = sorted(set(float(value) for value in alphas))
    if not dimensions or dimensions[0] < 1 or not candidates or candidates[0] < 0:
        raise ValueError("Dimensions must be positive and alphas non-negative")
    if not inner_splits:
        raise ValueError("At least one inner split is required")
    rows, max_dimension = [], max(dimensions)
    for alpha in candidates:
        collected = {dimension: [] for dimension in dimensions}
        for inner_train, inner_validation in inner_splits:
            train = {name: np.asarray(views[name])[inner_train] for name in names}
            validation = {name: np.asarray(views[name])[inner_validation] for name in names}
            try:
                fitted = fit_regularized_maxvar_gcca(
                    train, max_dimension, alpha=alpha, rcond=rcond
                )
            except ValueError:
                continue
            for dimension in dimensions:
                score = score_maxvar_partial_support(
                    fitted, train, validation, n_components=dimension,
                    cumulative_mass=support_cumulative_mass,
                )
                collected[dimension].append(score["component_support_scores"][dimension - 1])
        for dimension in dimensions:
            values = collected[dimension]
            rows.append(
                {
                    "shared_dimension": dimension,
                    "alpha": alpha,
                    "mean_inner_loo_corr": float(np.nanmean(values)) if values else float("nan"),
                    "mean_inner_partial_support_corr": (
                        float(np.nanmean(values)) if values else float("nan")
                    ),
                    "n_inner_splits": len(values),
                }
            )
    selected = {}
    for dimension in dimensions:
        valid = [
            row for row in rows
            if row["shared_dimension"] == dimension
            and np.isfinite(row["mean_inner_loo_corr"])
        ]
        if not valid:
            raise ValueError(f"No valid ridge candidate for dimension {dimension}")
        best = max(valid, key=lambda row: (row["mean_inner_loo_corr"], -row["alpha"]))
        selected[dimension] = float(best["alpha"])
    return selected, rows


def _componentwise_pearson(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> np.ndarray:
    """Return signed Pearson correlations for matching coordinate columns."""
    arr_a = np.asarray(coords_a, dtype=np.float64)
    arr_b = np.asarray(coords_b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            "CCA transformed coordinates must have matching shapes, got "
            f"{arr_a.shape} and {arr_b.shape}"
        )
    if arr_a.ndim != 2:
        raise ValueError(f"Expected 2D CCA coordinates, got shape {arr_a.shape}")

    correlations = np.empty(arr_a.shape[1], dtype=np.float64)
    for col in range(arr_a.shape[1]):
        x = arr_a[:, col]
        y = arr_b[:, col]
        finite = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(finite) < 2:
            correlations[col] = np.nan
            continue
        x = x[finite]
        y = y[finite]
        if np.std(x) == 0 or np.std(y) == 0:
            correlations[col] = np.nan
        else:
            correlations[col] = float(np.corrcoef(x, y)[0, 1])
    return correlations


def align_maxvar_folds(anchor_scores, eigenvalues_by_fold, *, match_threshold=0.8,
                       eigengap_threshold=0.05):
    """Align fold axes on common anchor rows and classify identifiable axes."""
    folds = tuple(sorted(anchor_scores))
    if len(folds) < 2:
        raise ValueError("At least two folds are required for alignment")
    arrays = {fold: np.asarray(anchor_scores[fold], dtype=np.float64) for fold in folds}
    shape = arrays[folds[0]].shape
    if any(values.shape != shape or values.ndim != 2 for values in arrays.values()):
        raise ValueError("All fold anchor-score matrices must have the same 2D shape")
    normalized = {}
    for fold, values in arrays.items():
        centered = values - np.mean(values, axis=0)
        norms = np.linalg.norm(centered, axis=0)
        normalized[fold] = centered / np.where(norms > 0, norms, 1.0)
    similarities = np.eye(len(folds))
    for left_index, left in enumerate(folds):
        for right_index in range(left_index + 1, len(folds)):
            right = folds[right_index]
            singular = np.linalg.svd(normalized[left].T @ normalized[right], compute_uv=False)
            value = float(np.mean(np.clip(singular, 0.0, 1.0)))
            similarities[left_index, right_index] = similarities[right_index, left_index] = value
    medoid = folds[int(np.argmax(np.mean(similarities, axis=1)))]
    assignments, correlations, signs = {}, {}, {}
    for fold in folds:
        corr = normalized[medoid].T @ normalized[fold]
        rows, cols = linear_sum_assignment(-np.abs(corr))
        order = np.empty(shape[1], dtype=int)
        matches = np.empty(shape[1], dtype=float)
        direction = np.empty(shape[1], dtype=float)
        order[rows] = cols
        matches[rows] = np.abs(corr[rows, cols])
        direction[rows] = np.where(corr[rows, cols] < 0, -1.0, 1.0)
        assignments[fold], correlations[fold], signs[fold] = order, matches, direction
    match_matrix = np.stack([correlations[fold] for fold in folds if fold != medoid])
    median_matches = np.median(match_matrix, axis=0)
    eigenvalues = np.asarray(eigenvalues_by_fold[medoid], dtype=np.float64)
    eigengaps = np.empty(shape[1], dtype=float)
    for component in range(shape[1]):
        neighboring = []
        if component > 0:
            neighboring.append(abs(eigenvalues[component - 1] - eigenvalues[component]))
        if component + 1 < shape[1]:
            neighboring.append(abs(eigenvalues[component] - eigenvalues[component + 1]))
        eigengaps[component] = min(neighboring, default=np.inf) / max(
            abs(eigenvalues[component]), np.finfo(float).eps
        )
    stable = (median_matches >= float(match_threshold)) & (
        eigengaps >= float(eigengap_threshold)
    )
    return {
        "medoid_fold": medoid,
        "folds": folds,
        "subspace_similarity": similarities,
        "assignments": assignments,
        "match_correlations": correlations,
        "signs": signs,
        "median_match_correlation": median_matches,
        "relative_neighboring_eigengap": eigengaps,
        "stable": stable,
    }


def maxvar_component_blocks(stable, eigengaps, *, eigengap_threshold=0.05):
    """Group adjacent non-identifiable or near-degenerate axes into blocks."""
    stable = np.asarray(stable, dtype=bool)
    gaps = np.asarray(eigengaps, dtype=float)
    blocks, start = [], None
    for index in range(stable.size):
        ambiguous = not stable[index]
        if index and min(gaps[index - 1], gaps[index]) < eigengap_threshold:
            ambiguous = True
        if ambiguous and start is None:
            start = index
        if start is not None and (not ambiguous or index == stable.size - 1):
            stop = index + 1 if ambiguous and index == stable.size - 1 else index
            blocks.append(tuple(range(start, stop)))
            start = None
    return tuple(blocks)


def maxvar_block_overlap(reference, candidate):
    """Return canonical correlations (cosines of principal angles) for a score block."""
    left, _ = np.linalg.qr(np.asarray(reference, dtype=np.float64))
    right, _ = np.linalg.qr(np.asarray(candidate, dtype=np.float64))
    return np.clip(np.linalg.svd(left.T @ right, compute_uv=False), 0.0, 1.0)


def cca_shared_dimensionality(
    correlations: np.ndarray,
    *,
    threshold: float = 0.2,
) -> dict[str, float | int]:
    """Compute effective shared dimensionality from canonical correlations."""
    corrs = np.asarray(correlations, dtype=np.float64)
    finite_corrs = corrs[np.isfinite(corrs)]
    weights = np.maximum(finite_corrs, 0.0) ** 2
    weight_sum = float(np.sum(weights))
    weight_sq_sum = float(np.sum(weights**2))
    if weight_sum <= 0.0 or weight_sq_sum <= 0.0:
        shared_dim_pr = 0.0
    else:
        shared_dim_pr = float((weight_sum**2) / weight_sq_sum)
    return {
        "shared_dim_pr": shared_dim_pr,
        "n_corr_ge_threshold": int(np.sum(finite_corrs >= float(threshold))),
    }


def score_cca_alignment(
    source_train: np.ndarray,
    target_train: np.ndarray,
    source_test: np.ndarray,
    target_test: np.ndarray,
    *,
    scale: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-6,
    shared_corr_threshold: float = 0.2,
) -> dict:
    """Fit CCA on train rows and score canonical correlations on held-out rows."""
    source_train_arr = np.asarray(source_train, dtype=np.float64)
    target_train_arr = np.asarray(target_train, dtype=np.float64)
    source_test_arr = np.asarray(source_test, dtype=np.float64)
    target_test_arr = np.asarray(target_test, dtype=np.float64)
    if source_train_arr.ndim != 2 or target_train_arr.ndim != 2:
        raise ValueError("CCA source and target train arrays must both be 2D")
    if source_test_arr.ndim != 2 or target_test_arr.ndim != 2:
        raise ValueError("CCA source and target test arrays must both be 2D")
    if source_train_arr.shape[0] != target_train_arr.shape[0]:
        raise ValueError(
            "CCA source and target train rows must match, got "
            f"{source_train_arr.shape[0]} and {target_train_arr.shape[0]}"
        )
    if source_test_arr.shape[0] != target_test_arr.shape[0]:
        raise ValueError(
            "CCA source and target test rows must match, got "
            f"{source_test_arr.shape[0]} and {target_test_arr.shape[0]}"
        )
    if source_train_arr.shape[1] != source_test_arr.shape[1]:
        raise ValueError("CCA source train/test coordinate dimensions must match")
    if target_train_arr.shape[1] != target_test_arr.shape[1]:
        raise ValueError("CCA target train/test coordinate dimensions must match")

    n_train = int(source_train_arr.shape[0])
    n_components = min(
        int(source_train_arr.shape[1]),
        int(target_train_arr.shape[1]),
        n_train - 1,
    )
    if n_components < 1:
        raise ValueError(
            "CCA alignment needs at least two training rows and one coordinate "
            "on each side"
        )

    cca = CCA(
        n_components=n_components,
        scale=scale,
        max_iter=max_iter,
        tol=tol,
    )
    cca.fit(source_train_arr, target_train_arr)
    source_train_cca, target_train_cca = cca.transform(
        source_train_arr,
        target_train_arr,
    )
    source_test_cca, target_test_cca = cca.transform(
        source_test_arr,
        target_test_arr,
    )
    train_corrs = _componentwise_pearson(source_train_cca, target_train_cca)
    test_corrs = _componentwise_pearson(source_test_cca, target_test_cca)
    train_shared = cca_shared_dimensionality(
        train_corrs,
        threshold=shared_corr_threshold,
    )
    test_shared = cca_shared_dimensionality(
        test_corrs,
        threshold=shared_corr_threshold,
    )
    top_n = min(10, n_components)
    return {
        "model": cca,
        "source_train_cca": source_train_cca,
        "target_train_cca": target_train_cca,
        "source_test_cca": source_test_cca,
        "target_test_cca": target_test_cca,
        "train_corrs": train_corrs,
        "test_corrs": test_corrs,
        "n_components": int(n_components),
        "mean_top10_test_corr": float(np.nanmean(test_corrs[:top_n])),
        "mean_all_test_corr": float(np.nanmean(test_corrs)),
        "first_test_corr": float(test_corrs[0]),
        "median_test_corr": float(np.nanmedian(test_corrs)),
        "mean_top10_train_corr": float(np.nanmean(train_corrs[:top_n])),
        "test_shared_dim_pr": test_shared["shared_dim_pr"],
        "train_shared_dim_pr": train_shared["shared_dim_pr"],
        "n_test_corr_ge_threshold": test_shared["n_corr_ge_threshold"],
        "n_train_corr_ge_threshold": train_shared["n_corr_ge_threshold"],
        "shared_corr_threshold": float(shared_corr_threshold),
    }


def cca_component_artifact(
    result: dict,
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    *,
    save_ambient_axes: bool = False,
) -> dict[str, np.ndarray]:
    """Return arrays worth saving for one fitted CCA fold/rank comparison."""
    model = result["model"]
    artifact = {
        "train_corrs": np.asarray(result["train_corrs"], dtype=np.float32),
        "test_corrs": np.asarray(result["test_corrs"], dtype=np.float32),
        "test_weight": (np.maximum(result["test_corrs"], 0.0) ** 2).astype(np.float32),
        "base_x_rotations": np.asarray(model.x_rotations_, dtype=np.float32),
        "peer_y_rotations": np.asarray(model.y_rotations_, dtype=np.float32),
    }
    if save_ambient_axes:
        artifact["base_ambient_axes"] = (
            np.asarray(basis_a, dtype=np.float64) @ model.x_rotations_
        ).astype(np.float32)
        artifact["peer_ambient_axes"] = (
            np.asarray(basis_b, dtype=np.float64) @ model.y_rotations_
        ).astype(np.float32)
    return artifact


def score_projected_cca_alignment(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    *,
    shared_corr_threshold: float = 0.2,
    save_ambient_axes: bool = False,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Fit and score CCA after projecting two embedding matrices through bases."""
    train = np.asarray(train_idx, dtype=int)
    test = np.asarray(test_idx, dtype=int)
    result = score_cca_alignment(
        np.asarray(embeddings_a)[train] @ basis_a,
        np.asarray(embeddings_b)[train] @ basis_b,
        np.asarray(embeddings_a)[test] @ basis_a,
        np.asarray(embeddings_b)[test] @ basis_b,
        shared_corr_threshold=shared_corr_threshold,
    )
    metric_keys = [
        "n_components",
        "mean_top10_test_corr",
        "mean_all_test_corr",
        "first_test_corr",
        "median_test_corr",
        "mean_top10_train_corr",
        "test_shared_dim_pr",
        "train_shared_dim_pr",
        "n_test_corr_ge_threshold",
        "n_train_corr_ge_threshold",
        "shared_corr_threshold",
    ]
    metrics = {
        "n_train": int(train.size),
        "n_test": int(test.size),
        **{key: result[key] for key in metric_keys},
    }
    return metrics, cca_component_artifact(
        result,
        basis_a,
        basis_b,
        save_ambient_axes=save_ambient_axes,
    )


def summarize_cca_fold_results(results: list[dict]) -> dict[str, float]:
    """Return mean CCA metrics across fold result dictionaries."""
    return {
        "mean_top10_test_corr": float(
            np.nanmean([row["mean_top10_test_corr"] for row in results])
        ),
        "mean_all_test_corr": float(
            np.nanmean([row["mean_all_test_corr"] for row in results])
        ),
        "mean_first_test_corr": float(
            np.nanmean([row["first_test_corr"] for row in results])
        ),
        "mean_median_test_corr": float(
            np.nanmean([row["median_test_corr"] for row in results])
        ),
        "mean_top10_train_corr": float(
            np.nanmean([row["mean_top10_train_corr"] for row in results])
        ),
        "mean_n_components": float(
            np.nanmean([row["n_components"] for row in results])
        ),
        "mean_test_shared_dim_pr": float(
            np.nanmean([row["test_shared_dim_pr"] for row in results])
        ),
        "mean_train_shared_dim_pr": float(
            np.nanmean([row["train_shared_dim_pr"] for row in results])
        ),
        "mean_n_test_corr_ge_threshold": float(
            np.nanmean([row["n_test_corr_ge_threshold"] for row in results])
        ),
        "mean_n_train_corr_ge_threshold": float(
            np.nanmean([row["n_train_corr_ge_threshold"] for row in results])
        ),
    }
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


def extrinsic_mean_from_unit_dir(unit_dir: str | Path, rank: int) -> np.ndarray:
    """Load fold factors from a unit directory and return their mean basis."""
    return extrinsic_mean_basis(load_unit_subspace_matrices(unit_dir), rank=rank)
