"""Utilities for discovering and comparing saved subspace factors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
        "loso_rrr",
        "rrr",
        "pca",
    ):
        if kind in parts:
            return kind
    return "unknown"


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
