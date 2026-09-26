"""Read artifacts once for the curated results notebook.

The returned values are ordinary tables, arrays, and dictionaries. Plot functions
never receive an Artifact or open a result file.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pandas as pd

from scripts.run_rrr import Artifact as RRRArtifact
from scripts.run_significance import Artifact as SignificanceArtifact
from scripts.run_language_decoding import Artifact as DecodingArtifact
from scripts.utilities.check_final_significance import MODEL_LAYERS
from src.utils.rrr.plot_metrics import fisher_mean, rank_quantiles


DATASETS = {"podcast": "podcast", "cars-2": "brain-treebank-cars-2", "fmri": "deep-fmri"}
QUANTILES = (.05, .10, .25, .50, .75, .90)
LABELS = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}
PAIRS = tuple((model, layer) for model, layers in MODEL_LAYERS.values() for layer in layers)
FMRI_VARIANT = "input_scaling-pooled-training-stories-v1"
CACHE_VERSION = 1

CONTROL_MODELS = frozenset({"random-gpt2", "glove", "glove-300"})


def primary_analysis_data(internal, loso, cross, quantiles, loso_quantiles, decoding):
    """Exclude appendix controls from primary summaries without changing loaded data."""
    def primary_frame(frame):
        if frame.empty:
            return frame.copy()
        return frame.loc[~frame["model"].isin(CONTROL_MODELS)].copy()

    return (
        primary_frame(internal), primary_frame(loso), primary_frame(cross),
        {key: value for key, value in quantiles.items() if key[1] not in CONTROL_MODELS},
        {key: value for key, value in loso_quantiles.items() if key[1] not in CONTROL_MODELS},
        primary_frame(decoding),
    )


def _cache_fingerprint(results_root):
    """Track artifact manifests and summary code without opening large payloads."""
    digest = hashlib.blake2b(digest_size=20)
    sources = (Path(__file__), Path(__file__).parents[1] / "rrr" / "plot_metrics.py")
    manifests = (
        path for category in ("run_rrr", "run_significance", "run_language_decoding")
        for path in (results_root / category).rglob("manifest.json")
    )
    for path in sorted((*sources, *manifests)):
        stat = path.stat()
        digest.update(str(path.relative_to(results_root) if path.is_relative_to(results_root)
                          else path).encode())
        digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
    digest.update(str(CACHE_VERSION).encode())
    return digest.hexdigest()


def _read_cache(path, fingerprint):
    try:
        with path.open("rb") as handle:
            if handle.read(len(fingerprint)) != fingerprint.encode():
                return None
            return pickle.load(handle)
    except (FileNotFoundError, EOFError, OSError, pickle.UnpicklingError):
        return None


def _write_cache(path, fingerprint, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".final_plots_", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(fingerprint.encode())
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _variant(artifact, dataset, mode, source=None):
    index = artifact._read_json(artifact._leaf() / "manifest.json", default={})
    candidates = list(index.get("variants", {}))
    if dataset == "fmri" and mode in {"internal", "cross_dataset"}:
        candidates = [v for v in candidates if FMRI_VARIANT in v.split("/")]
    if dataset == "cars-2":
        candidates = [v for v in candidates if "rereference-laplacian" in v.split("/")]
    if source is not None:
        candidates = [v for v in candidates if DATASETS[source] in v.split("/")]
    candidates = [v for v in candidates if artifact._read_json(
        artifact._leaf() / v / "manifest.json", default={}
    ).get("complete")]
    if len(candidates) != 1:
        raise ValueError(f"expected one completed {mode} variant, found {candidates}")
    artifact.variant = candidates[0]
    return artifact


def _rrr(results_root, mode, dataset, model, layer, source=None):
    artifact = RRRArtifact(
        mode=mode, dataset=DATASETS[dataset], model=model, layer=layer,
        results_root=results_root,
    )
    return _variant(artifact, dataset, mode, source)


def _groups(manifest, size):
    if manifest is None:
        return np.arange(size), size
    if len(manifest) != size:
        raise ValueError(f"target manifest has {len(manifest)} rows for {size} metrics")
    if "electrode" in manifest:
        columns = ["subject", "electrode"]
    elif "voxel_index" in manifest:
        columns = ["subject", "voxel_index"]
    else:
        return np.arange(size), size
    groups = manifest.groupby(columns, sort=False).ngroup().to_numpy()
    return groups, int(groups.max()) + 1


def _collapse(values, manifest):
    values = np.asarray(values, dtype=float).reshape(-1)
    groups, count = _groups(manifest, len(values))
    z = np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7))
    finite = np.isfinite(z)
    totals = np.bincount(groups[finite], weights=z[finite], minlength=count)
    sizes = np.bincount(groups[finite], minlength=count)
    return np.tanh(np.divide(totals, sizes, out=np.full(count, np.nan), where=sizes > 0))


def _manifest_for(artifact, selector):
    choices = artifact.target_manifest
    keys = {k: str(v) for k, v in selector.items() if k in {"method", "loso_phase", "held_out_subject"}}
    for encoded, frame in choices.items():
        found = {k: str(v) for k, v in dict(encoded).items()}
        if all(found.get(k) == value for k, value in keys.items()):
            return frame
    return choices.get(())


def _metric_name(dataset, method, *, phase=None):
    if dataset == "fmri":
        return "corr_per_target"
    if phase == "transfer":
        return "transfer_orth_corr_per_target"
    return "nr_orth_corr_per_target" if method == "rrr" else "pca_corr_per_target"


def _quantiles_from_values(values_by_rank):
    ranks, target_rows = [], []
    for rank, arrays in sorted(values_by_rank.items()):
        lengths = {len(a) for a in arrays}
        if len(lengths) != 1:
            raise ValueError(f"target count changes at rank {rank}")
        ranks.append(rank)
        target_rows.append(fisher_mean(np.stack(arrays), axis=0))
    return np.asarray(ranks, dtype=float), rank_quantiles(ranks, np.stack(target_rows), QUANTILES)


def _internal_subject_rrr(rank_metrics, manifest, metric_name, collapsed_by_rank):
    """Get full-training RRR curves and target quantiles for each subject."""
    if manifest is None or "subject" not in manifest:
        return [], {}
    raw_subjects = manifest["subject"].astype(str).to_numpy()
    groups, count = _groups(manifest, len(raw_subjects))
    target_subjects = np.empty(count, dtype=object)
    target_subjects[groups] = raw_subjects
    subjects = np.unique(raw_subjects)
    raw_masks = {subject: raw_subjects == subject for subject in subjects}
    target_masks = {subject: target_subjects == subject for subject in subjects}
    curve_folds = defaultdict(list)
    for encoded, metrics in rank_metrics.items():
        selector = dict(encoded)
        if selector.get("method") != "rrr" or "rank" not in selector or metric_name not in metrics:
            continue
        rank = int(selector["rank"])
        raw = np.asarray(metrics[metric_name], dtype=float).reshape(-1)
        if len(raw) != len(raw_subjects):
            raise ValueError(f"internal RRR rank {rank} has {len(raw)} targets; expected {len(raw_subjects)}")
        for subject in subjects:
            curve_folds[(subject, rank)].append(float(fisher_mean(raw[raw_masks[subject]])))
    curves = [(subject, "RRR baseline", rank, float(fisher_mean(values)))
              for (subject, rank), values in curve_folds.items()]
    quantiles = {}
    for subject in subjects:
        by_rank = {rank: [values[target_masks[subject]] for values in folds]
                   for rank, folds in collapsed_by_rank.items()}
        if by_rank:
            quantiles[(subject, "RRR baseline")] = _quantiles_from_values(by_rank)
    return curves, quantiles


def _internal_one(results_root, dataset, model, layer):
    artifact = _rrr(results_root, "internal", dataset, model, layer).load_artifacts(
        ["rank_metrics", "target_manifest", "full_metrics"]
    )
    curves, target_values = [], defaultdict(list)
    for encoded, metrics in artifact.rank_metrics.items():
        selector = dict(encoded)
        method, rank = selector.get("method"), selector.get("rank")
        if method not in {"rrr", "pca"} or rank is None:
            continue
        name = _metric_name(dataset, method)
        if name not in metrics:
            continue
        raw = np.asarray(metrics[name], dtype=float)
        curves.append((method, int(rank), float(fisher_mean(raw))))
        target_values[(method, int(rank))].append(_collapse(raw, _manifest_for(artifact, {"method": method})))
    if not curves:
        raise ValueError("no internal rank correlations")
    quantiles = {}
    for method in ("rrr", "pca"):
        by_rank = {rank: values for (item_method, rank), values in target_values.items() if item_method == method}
        if by_rank:
            quantiles[method] = _quantiles_from_values(by_rank)
    full = []
    for encoded, metrics in artifact.full_metrics.items():
        selector = dict(encoded)
        if dataset != "fmri" and selector.get("method") != "rrr":
            continue
        name = "corr_per_target" if dataset == "fmri" else "full_corr_per_target"
        if name in metrics:
            full.append(_collapse(metrics[name], _manifest_for(artifact, {"method": "rrr"})))
    full_values = fisher_mean(np.stack(full), axis=0) if full else np.array([])
    subject_curves, subject_quantiles = _internal_subject_rrr(
        artifact.rank_metrics, _manifest_for(artifact, {"method": "rrr"}),
        _metric_name(dataset, "rrr"),
        {rank: values for (method, rank), values in target_values.items()
         if method == "rrr"},
    )
    return curves, quantiles, full_values, subject_curves, subject_quantiles


def _loso_one(results_root, dataset, model, layer):
    artifact = _rrr(results_root, "leave_one_subject_out", dataset, model, layer).load_artifacts(
        ["target_manifest"]
    )
    names = {("rrr", "transfer"): "RRR transfer", ("pca", "baseline"): "PCA baseline"}
    curves, target_values = [], defaultdict(list)
    rank_metrics = {}
    for method, phase in names:
        rank_metrics.update(artifact.load_collection(
            "rank_metrics", {"method": method, "loso_phase": phase}
        ))
    for encoded, metrics in rank_metrics.items():
        selector = dict(encoded)
        key = (selector.get("method"), selector.get("loso_phase"))
        if key not in names or "rank" not in selector:
            continue
        name = _metric_name(dataset, key[0], phase=key[1])
        if name not in metrics:
            continue
        rank, subject = int(selector["rank"]), str(selector["held_out_subject"])
        raw = np.asarray(metrics[name], dtype=float)
        curves.append((subject, names[key], rank, float(fisher_mean(raw))))
        manifest = _manifest_for(artifact, selector)
        target_values[(subject, names[key], rank)].append(_collapse(raw, manifest))
    quantiles = {}
    for subject, method, _ in target_values:
        by_rank = {rank: values for (s, m, rank), values in target_values.items()
                   if s == subject and m == method}
        quantiles[(subject, method)] = _quantiles_from_values(by_rank)
    return curves, quantiles


def _cross_one(results_root, target, source, model, layer, internal_curves):
    artifact = _rrr(results_root, "cross_dataset", target, model, layer, source).load_artifacts(["rank_metrics"])
    records = []
    for encoded, metrics in artifact.rank_metrics.items():
        selector = dict(encoded)
        method, rank = selector.get("method"), selector.get("rank")
        if method not in {"rrr", "pca"} or rank is None:
            continue
        if target == "fmri":
            name = "corr_per_target"
        elif method == "rrr":
            name = "nr_orth_corr" if "nr_orth_corr" in metrics else "nr_corr"
        else:
            name = "pca_corr"
        if name in metrics:
            records.append((f"Transfer {method.upper()}", int(rank), float(fisher_mean(metrics[name]))))
    for method, rank, value in internal_curves:
        records.append((f"Within {method.upper()}", rank, value))
    frame = pd.DataFrame(records, columns=["method", "rank", "correlation"])
    if not {"Within RRR", "Within PCA", "Transfer RRR", "Transfer PCA"}.issubset(frame["method"]):
        raise ValueError("incomplete four-method directed curves")
    frame["fisher_z"] = np.arctanh(np.clip(frame.pop("correlation"), -1 + 1e-7, 1 - 1e-7))
    frame = frame.groupby(["method", "rank"], as_index=False)["fisher_z"].mean()
    frame["correlation"] = np.tanh(frame.pop("fisher_z"))
    return frame


def _significance_one(results_root, dataset, model, layer):
    mode = "held_out_session" if dataset == "fmri" else "sig_lags"
    variant = "rereference-laplacian" if dataset == "cars-2" else "default"
    artifact = SignificanceArtifact(
        mode=mode, dataset=DATASETS[dataset], model=model, layer=layer,
        variant=variant, results_root=results_root,
    )
    if not artifact.manifest().get("complete"):
        raise ValueError("incomplete significance artifact")
    targets = artifact.load_targets()
    return {(str(subject), str(row.target_id)) for subject, frame in targets.items()
            for row in frame.itertuples() if bool(row.significant)}


def _decoding_summaries(results_root):
    root = Path(results_root) / "run_language_decoding"
    rows = []
    for mode in ("within_dataset", "cross_dataset"):
        for path in root.glob(f"{mode}/*/offset-*/*/layer-*/manifest.json"):
            try:
                index = json.loads(path.read_text(encoding="utf-8"))
                identity = index["identity"]
                variants = index["variants"]
            except (ValueError, KeyError, OSError):
                continue
            for variant in variants:
                try:
                    artifact = DecodingArtifact(
                        mode=identity["mode"], dataset=identity["dataset"],
                        offset=identity["offset"], model=identity["model"],
                        layer=identity["layer"], variant=variant, results_root=results_root,
                    )
                    summary = artifact.load_field("summary").copy()
                    config = artifact.manifest().get("configuration", {})
                except (FileNotFoundError, KeyError, ValueError, OSError, RuntimeError):
                    continue
                summary["dataset"] = next((d for d, root_name in DATASETS.items()
                                           if root_name == identity["dataset"]), identity["dataset"])
                summary["model"] = identity["model"]
                summary["layer"] = identity["layer"]
                summary["label_offset"] = int(config.get("label_offset", 0))
                summary["variant"] = variant
                summary["generation"] = (
                    "cross_dataset" if mode == "cross_dataset" or variant.startswith("source-")
                    else "within_dataset"
                )
                rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_final_data(results_root):
    """Eagerly load all requested figures' inputs and return separate plain values."""
    results_root = Path(results_root)
    cache_path = results_root / ".final_plots_cache.pkl"
    fingerprint = _cache_fingerprint(results_root)
    cached = _read_cache(cache_path, fingerprint)
    if cached is not None:
        return cached
    internal_rows, loso_rows, cross_rows, full_rows = [], [], [], []
    quantiles, loso_quantiles, significant, gaps = {}, {}, {}, []
    for dataset in DATASETS:
        for model, layer in PAIRS:
            identity = (dataset, model, layer)
            subject_curves, subject_quantiles = [], {}
            try:
                curves, matrices, full, subject_curves, subject_quantiles = _internal_one(
                    results_root, *identity
                )
                quantiles.update({(*identity, method): value for method, value in matrices.items()})
                internal_rows.extend((dataset, model, layer, *row) for row in curves)
                if full.size:
                    full_rows.append((dataset, model, layer, full))
            except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as error:
                gaps.append(f"internal {identity}: {error}")
                curves = []
            try:
                values, matrices = _loso_one(results_root, *identity)
                transfer_subjects = {subject for subject, method, _, _ in values
                                     if method == "RRR transfer"}
                pca_subjects = {subject for subject, method, _, _ in values
                                if method == "PCA baseline"}
                control_subjects = {subject for subject, _, _, _ in subject_curves}
                matched_subjects = transfer_subjects & pca_subjects & control_subjects
                values = [row for row in values if row[0] in matched_subjects]
                values.extend(row for row in subject_curves if row[0] in matched_subjects)
                matrices = {key: value for key, value in matrices.items()
                            if key[0] in matched_subjects}
                matrices.update({key: value for key, value in subject_quantiles.items()
                                 if key[0] in matched_subjects})
                loso_quantiles.update({(*identity, *key): value for key, value in matrices.items()})
                loso_rows.extend((dataset, model, layer, *row) for row in values)
            except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as error:
                gaps.append(f"LOSO {identity}: {error}")
            try:
                significant[identity] = _significance_one(results_root, *identity)
            except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as error:
                gaps.append(f"significance {identity}: {error}")
            for source in DATASETS:
                if source == dataset:
                    continue
                try:
                    frame = _cross_one(results_root, dataset, source, model, layer, curves)
                    frame["target_dataset"], frame["source_dataset"] = dataset, source
                    frame["model"], frame["layer"] = model, layer
                    cross_rows.append(frame)
                except (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError) as error:
                    gaps.append(f"directed {source} → {dataset} {model}/layer-{layer}: {error}")
    internal = pd.DataFrame(internal_rows, columns=["dataset", "model", "layer", "method", "rank", "correlation"])
    loso = pd.DataFrame(loso_rows, columns=["dataset", "model", "layer", "subject", "method", "rank", "correlation"])
    cross = pd.concat(cross_rows, ignore_index=True) if cross_rows else pd.DataFrame(
        columns=["method", "rank", "correlation", "target_dataset", "source_dataset", "model", "layer"]
    )
    decoding = _decoding_summaries(results_root)
    result = (internal, loso, cross, quantiles, loso_quantiles, full_rows,
              significant, decoding, gaps)
    _write_cache(cache_path, fingerprint, result)
    return result
