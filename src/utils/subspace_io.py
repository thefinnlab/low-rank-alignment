"""I/O helpers for saved subspace comparison analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from src.core.config import (
    RRRConfig,
    ScriptMode,
    SubspaceComparisonRun,
    get_results_dir,
)
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import expand_per_model_layer_configs, load_config
from src.utils.data_loaders import build_XY_from_subjects, load_sig_electrodes
from src.utils.embeddings import get_glove_embeddings, load_embeddings
from src.utils.encoding import apply_offset, load_encoding_results
from src.utils.subspace_compare import orthonormal_basis


@dataclass(frozen=True)
class ResolvedSubspaceRun:
    """A configured comparison run with resolved RRR config and artifact paths."""

    name: str
    kind: str
    config_path: Path
    config: RRRConfig
    unit_dir: Path
    output_dir: Path
    sig_dir: Path
    subject_ids: list[int]
    dataset_key: tuple[str, str]


@dataclass
class BaseComparisonData:
    """Rows and folds used by one ordered base comparison."""

    X_base: np.ndarray
    train_idxs: list[np.ndarray]
    test_idxs: list[np.ndarray]
    selection: np.ndarray
    valid_rows_on_selection: np.ndarray | None
    dataset_key: tuple[str, str]
    word_count: int


def sanitize_name(name: str) -> str:
    """Return a filesystem-safe short name for comparison output paths."""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in name)
    safe = safe.strip("-._")
    if not safe:
        raise ValueError(f"Cannot sanitize empty run name from {name!r}")
    return safe


def resolve_rrr_output_dir(config: RRRConfig, output_dir: str | Path) -> Path:
    """Resolve the primary RRR unit root for a config and output directory."""
    if config.rrr_output_dir:
        return Path(config.rrr_output_dir)
    if config.mode == ScriptMode.PER_REGION:
        return Path(output_dir) / "per_region_rrr"
    if config.mode == ScriptMode.PER_SUBJECT:
        return Path(output_dir) / "per_subject_rrr"
    if config.mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT:
        return Path(output_dir) / "loso_rrr"
    return Path(output_dir) / "rrr"


def resolve_output_dir(config: RRRConfig, dataset_context) -> Path:
    """Resolve the canonical result root for an RRR config."""
    enc = config.encoding
    return Path(
        config.output_dir
        or get_results_dir(
            enc.model,
            enc.offset,
            enc.per_model_config,
            dataset_context=dataset_context,
        )
    )


def _path_matches(a: Path, b: Path) -> bool:
    return a.expanduser().resolve() == b.expanduser().resolve()


def _resolve_config_for_unit(config_path: Path, unit_dir: Path) -> RRRConfig:
    """Load an RRR config and select the layer-expanded config matching unit_dir."""
    loaded = load_config(str(config_path), {}, RRRConfig)
    configs = expand_per_model_layer_configs(loaded)
    unit_dir = unit_dir.expanduser()
    if len(configs) == 1:
        return configs[0]

    matches = []
    for config in configs:
        enc = config.encoding
        subject_ids = enc.subjects or list(range(1, 10))
        context = resolve_dataset_context(
            dataset=enc.dataset,
            subject_ids=subject_ids,
            per_dataset_config=enc.per_dataset_config,
            data_root=enc.data_root,
        )
        output_dir = resolve_output_dir(config, context)
        rrr_dir = resolve_rrr_output_dir(config, output_dir)
        pca_dir = Path(str(rrr_dir).replace("rrr", "pca", 1))
        if _path_matches(unit_dir, rrr_dir) or _path_matches(unit_dir, pca_dir):
            matches.append(config)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"No layer-expanded config from {config_path} matches unit_dir {unit_dir}."
        )
    raise ValueError(f"Multiple configs from {config_path} match unit_dir {unit_dir}.")


def resolve_subspace_run(run: SubspaceComparisonRun) -> ResolvedSubspaceRun:
    """Resolve one configured comparison run."""
    config_path = Path(run.rrr_config)
    unit_dir = Path(run.unit_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"RRR config not found for run {run.name}: {config_path}")
    if not unit_dir.exists():
        raise FileNotFoundError(f"unit_dir not found for run {run.name}: {unit_dir}")

    config = _resolve_config_for_unit(config_path, unit_dir)
    enc = config.encoding
    subject_ids = enc.subjects or list(range(1, 10))
    context = resolve_dataset_context(
        dataset=enc.dataset,
        subject_ids=subject_ids,
        per_dataset_config=enc.per_dataset_config,
        data_root=enc.data_root,
    )
    get_dataset_adapter(enc.dataset).validate_subjects_share_stimulus(
        subject_ids,
        enc.per_dataset_config,
        enc.data_root,
    )
    output_dir = resolve_output_dir(config, context)
    sig_dir = Path(config.sig_dir or output_dir)
    stimulus_id = getattr(context, "stimulus_id", enc.dataset)
    return ResolvedSubspaceRun(
        name=run.name,
        kind=run.kind or _infer_unit_kind(unit_dir),
        config_path=config_path,
        config=config,
        unit_dir=unit_dir,
        output_dir=output_dir,
        sig_dir=sig_dir,
        subject_ids=[int(s) for s in subject_ids],
        dataset_key=(getattr(context, "dataset", enc.dataset), str(stimulus_id)),
    )


def _infer_unit_kind(unit_dir: Path) -> str:
    parts = unit_dir.parts
    for kind in ("rrr", "pca", "per_subject_rrr", "per_subject_pca", "loso_rrr", "loso_pca"):
        if kind in parts:
            return kind
    return "unknown"


def load_shared_selection_metadata(sig_dir: Path, subject_ids: list[int]) -> dict:
    """Load shared word-selection metadata from subject encoding outputs."""
    reference = None
    for sub_id in subject_ids:
        path = sig_dir / "subjects" / f"sub_{sub_id}" / "encoding_results.joblib"
        enc_results = load_encoding_results(str(path))
        metadata = enc_results.get("word_selection")
        if metadata is None:
            raise ValueError(f"Missing word_selection metadata in {path}")
        if reference is None:
            reference = metadata
            continue
        for key in ("epoch_selection", "analysis_selection"):
            if not np.array_equal(reference[key], metadata[key]):
                raise ValueError(f"Shared word-selection metadata differs for key {key!r}")
    if reference is None:
        raise ValueError("Cannot load shared selection metadata for no subjects")
    return reference


def load_base_comparison_data(base: ResolvedSubspaceRun) -> BaseComparisonData:
    """Load base-run selected embeddings and fold indices."""
    enc = base.config.encoding
    context = resolve_dataset_context(
        dataset=enc.dataset,
        subject_ids=base.subject_ids,
        per_dataset_config=enc.per_dataset_config,
        data_root=enc.data_root,
    )
    sig_electrodes, _ = load_sig_electrodes(
        str(base.output_dir),
        base.subject_ids,
        sig_dir=str(base.sig_dir),
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=context,
        data_root=enc.data_root,
    )
    embeddings, word_df = load_embeddings(
        enc.data_root,
        enc.model,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=context,
        **enc.per_model_config,
    )
    valid_rows = None
    if enc.limit_to_glove or enc.model == "glove":
        valid_rows = get_glove_embeddings(enc.data_root, word_df, **enc.per_model_config)[1][
            "in_glove"
        ]
    shared_metadata = (
        load_shared_selection_metadata(base.sig_dir, base.subject_ids)
        if enc.shared_word_selection
        else None
    )
    X_base, _, selection, _ = build_XY_from_subjects(
        base.subject_ids,
        enc,
        sig_electrodes,
        embeddings,
        valid_rows=valid_rows,
        shared_selection=(
            shared_metadata["epoch_selection"] if shared_metadata is not None else None
        ),
    )
    enc_results_path = (
        base.output_dir
        / "subjects"
        / f"sub_{base.subject_ids[0]}"
        / "encoding_results.joblib"
    )
    enc_results = load_encoding_results(str(enc_results_path))
    test_idxs = [np.asarray(idx, dtype=int) for idx in enc_results["test_idxs"]]
    train_idxs = enc_results.get("train_idxs")
    if train_idxs is None:
        all_idxs = np.arange(X_base.shape[0], dtype=int)
        train_idxs = [
            np.setdiff1d(all_idxs, test_idx, assume_unique=False) for test_idx in test_idxs
        ]
    else:
        train_idxs = [np.asarray(idx, dtype=int) for idx in train_idxs]
    return BaseComparisonData(
        X_base=np.asarray(X_base),
        train_idxs=train_idxs,
        test_idxs=test_idxs,
        selection=np.asarray(selection, dtype=int),
        valid_rows_on_selection=(
            np.asarray(valid_rows)[np.asarray(selection, dtype=int)]
            if valid_rows is not None
            else None
        ),
        dataset_key=base.dataset_key,
        word_count=int(X_base.shape[0]),
    )


def load_peer_embeddings_on_base_rows(
    base: ResolvedSubspaceRun,
    peer: ResolvedSubspaceRun,
    base_data: BaseComparisonData,
) -> np.ndarray:
    """Load peer model embeddings over the base run's selected word rows."""
    base_enc = base.config.encoding
    peer_enc = peer.config.encoding
    if int(base_enc.offset) != int(peer_enc.offset):
        raise ValueError(
            f"Cannot compare runs with different offsets in v1: "
            f"{base.name} offset={base_enc.offset}, {peer.name} offset={peer_enc.offset}"
        )
    context = resolve_dataset_context(
        dataset=base_enc.dataset,
        subject_ids=base.subject_ids,
        per_dataset_config=base_enc.per_dataset_config,
        data_root=base_enc.data_root,
    )
    embeddings, _ = load_embeddings(
        base_enc.data_root,
        peer_enc.model,
        dataset=base_enc.dataset,
        per_dataset_config=base_enc.per_dataset_config,
        dataset_context=context,
        **peer_enc.per_model_config,
    )
    dummy_y = np.zeros((len(base_data.selection), 1), dtype=np.float32)
    X_peer, _ = apply_offset(
        embeddings[base_data.selection],
        dummy_y,
        int(base_enc.offset),
        valid_rows=base_data.valid_rows_on_selection,
    )
    if X_peer.shape[0] != base_data.X_base.shape[0]:
        raise ValueError(
            "Peer embeddings on base rows produced a different row count: "
            f"{X_peer.shape[0]} vs {base_data.X_base.shape[0]}"
        )
    return np.asarray(X_peer)


def load_fold_bases(unit_dir: Path, rank: int) -> dict[int, np.ndarray]:
    """Load orthonormalized fold bases from fold_*/A.npy for one rank."""
    bases = {}
    for path in sorted(unit_dir.glob("fold_*/A.npy")):
        name = path.parent.name
        try:
            fold = int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        arr = np.load(path, mmap_mode="r")
        if int(rank) > arr.shape[1]:
            continue
        bases[fold] = orthonormal_basis(arr, int(rank))
    return bases


def max_available_rank(unit_dir: Path) -> int:
    """Return the minimum available A.npy width across folds in a unit directory."""
    widths = []
    for path in sorted(unit_dir.glob("fold_*/A.npy")):
        widths.append(int(np.load(path, mmap_mode="r").shape[1]))
    if not widths:
        raise FileNotFoundError(f"No fold_*/A.npy files found under {unit_dir}")
    return min(widths)


def ambient_dim(unit_dir: Path) -> int:
    """Return the common A.npy row count across folds in a unit directory."""
    dims = []
    for path in sorted(unit_dir.glob("fold_*/A.npy")):
        dims.append(int(np.load(path, mmap_mode="r").shape[0]))
    if not dims:
        raise FileNotFoundError(f"No fold_*/A.npy files found under {unit_dir}")
    unique = sorted(set(dims))
    if len(unique) != 1:
        raise ValueError(f"Inconsistent A.npy ambient dimensions under {unit_dir}: {unique}")
    return unique[0]


def run_metadata(run: ResolvedSubspaceRun) -> dict:
    """Return JSON-serializable metadata for a resolved run."""
    enc = run.config.encoding
    return {
        "name": run.name,
        "kind": run.kind,
        "rrr_config": str(run.config_path),
        "unit_dir": str(run.unit_dir),
        "output_dir": str(run.output_dir),
        "sig_dir": str(run.sig_dir),
        "dataset": enc.dataset,
        "dataset_key": list(run.dataset_key),
        "subjects": run.subject_ids,
        "model": enc.model,
        "offset": int(enc.offset),
        "per_model_config": enc.per_model_config,
        "per_dataset_config": enc.per_dataset_config,
    }
