"""Shared significance selection and union handling for RRR runners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.config import SignificanceSelectionConfig
from scripts.run_significance import Artifact as SignificanceArtifact


@dataclass
class PreparedSignificance:
    """Artifact-like significance values consumed by downstream RRR code."""

    target_significance: dict[str, pd.DataFrame]
    significance_details: dict[str, pd.DataFrame]
    encoding_results: dict
    reference: SignificanceArtifact
    sources: list[SignificanceArtifact]

    @property
    def source_identities(self) -> list[dict]:
        return [source.identity_record() for source in self.sources]

    def identity_record(self) -> dict:
        return self.reference.identity_record()


def expand_significance_artifact_specs(specs: list[dict]) -> list[dict]:
    """Expand list-valued ``layer`` fields into concrete artifact specifications."""
    expanded = []
    seen = set()
    for spec in specs:
        item = dict(spec)
        layers = item.get("layer")
        if isinstance(layers, list):
            if not layers:
                raise ValueError("Significance artifact layer lists cannot be empty")
            concrete_layers = layers
        else:
            concrete_layers = [layers]
        for layer in concrete_layers:
            concrete = dict(item)
            concrete["layer"] = layer
            key = tuple(sorted((name, repr(value)) for name, value in concrete.items()))
            if key in seen:
                raise ValueError(f"Duplicate significance artifact identity: {concrete}")
            seen.add(key)
            expanded.append(concrete)
    return expanded


def _canonical_targets(artifact: SignificanceArtifact) -> pd.DataFrame:
    frames = []
    for subject, frame in artifact.target_significance.items():
        required = {"target_id", "target_kind", "significant"}
        if not required.issubset(frame.columns):
            raise ValueError(
                "Significance target tables must contain target_id, target_kind, "
                "and significant"
            )
        current = frame.copy()
        if "subject" not in current:
            current["subject"] = str(subject)
        current["subject"] = current["subject"].astype(str)
        current["target_kind"] = current["target_kind"].astype(str)
        current["significant"] = current["significant"].astype(bool)
        frames.append(current[["subject", "target_id", "target_kind", "significant"]])
    if not frames:
        return pd.DataFrame(
            columns=["subject", "target_id", "target_kind", "significant"]
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(
        ["subject", "target_id", "target_kind"], as_index=False, sort=False
    )["significant"].any()


def _canonical_ieeg_details(artifact: SignificanceArtifact) -> pd.DataFrame:
    frames = []
    for frame in artifact.significance_details.values():
        required = {"subject", "electrode", "lag_idx", "sig"}
        if not required.issubset(frame.columns):
            raise ValueError(
                "iEEG significance details must contain subject, electrode, lag_idx, sig"
            )
        current = frame[["subject", "electrode", "lag_idx", "sig"]].copy()
        current["subject"] = current["subject"].astype(str)
        current["electrode"] = current["electrode"].astype(str)
        current["lag_idx"] = current["lag_idx"].astype(int)
        current["sig"] = current["sig"].astype(bool)
        frames.append(current)
    if not frames:
        return pd.DataFrame(columns=["subject", "electrode", "lag_idx", "sig"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(
        ["subject", "electrode", "lag_idx"], as_index=False, sort=False
    )["sig"].any()


def union_significance_artifacts(
    artifacts: list[SignificanceArtifact], *, modality: str
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """OR target and iEEG lag significance across loaded artifacts."""
    if not artifacts:
        raise ValueError("At least one SignificanceArtifact is required")
    targets = pd.concat(
        [_canonical_targets(artifact) for artifact in artifacts], ignore_index=True
    )
    targets = targets.groupby(
        ["subject", "target_id", "target_kind"], as_index=False, sort=False
    )["significant"].any()
    target_by_subject = {
        str(subject): frame.reset_index(drop=True)
        for subject, frame in targets.groupby("subject", sort=False)
    }
    details_by_subject = {}
    if modality == "ieeg":
        details = pd.concat(
            [_canonical_ieeg_details(artifact) for artifact in artifacts],
            ignore_index=True,
        )
        details = details.groupby(
            ["subject", "electrode", "lag_idx"], as_index=False, sort=False
        )["sig"].any()
        details_by_subject = {
            str(subject): frame.reset_index(drop=True)
            for subject, frame in details.groupby("subject", sort=False)
        }
    return target_by_subject, details_by_subject


def _load_artifact(spec: dict, results_root: str, fields: list[str], subjects) -> SignificanceArtifact:
    kwargs = dict(spec)
    kwargs.setdefault("results_root", results_root)
    return SignificanceArtifact(**kwargs).load_artifacts(
        fields=fields, selectors={"subjects": subjects}
    )


def _values_equal(left, right) -> bool:
    """Compare nested encoding metadata containing NumPy arrays."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right))
        except (TypeError, ValueError):
            return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def validate_ieeg_encoding_alignment(
    reference: SignificanceArtifact,
    sources: list[SignificanceArtifact],
    subjects,
) -> None:
    """Require union sources to use the reference stimulus rows and CV folds."""
    for subject in subjects:
        subject_key = str(subject)
        try:
            reference_result = reference.encoding_results[subject_key]
        except KeyError as exc:
            raise ValueError(
                f"Reference significance artifact lacks encoding results for subject {subject}"
            ) from exc
        for source in sources:
            try:
                source_result = source.encoding_results[subject_key]
            except KeyError as exc:
                raise ValueError(
                    "Significance union source "
                    f"{source.identity_record()} lacks encoding results for subject {subject}"
                ) from exc
            for field in ("selection", "test_idxs"):
                if field not in reference_result or field not in source_result:
                    raise ValueError(
                        f"Cannot validate iEEG folds: encoding field {field!r} is "
                        f"missing for subject {subject}"
                    )
                if not _values_equal(reference_result[field], source_result[field]):
                    raise ValueError(
                        "Significance union source "
                        f"{source.identity_record()} has different {field} for "
                        f"subject {subject}"
                    )
            reference_words = reference_result.get("word_selection")
            source_words = source_result.get("word_selection")
            if not _values_equal(reference_words, source_words):
                raise ValueError(
                    "Significance union source "
                    f"{source.identity_record()} has different word_selection "
                    f"metadata for subject {subject}"
                )


def load_significance_selection(
    selection: SignificanceSelectionConfig,
    *,
    results_root: str,
    subjects,
    modality: str,
) -> PreparedSignificance:
    """Load, validate, and prepare one configured RRR significance selection."""
    selection.validate()
    if not selection.reference_artifact:
        raise ValueError("significance_selection.reference_artifact is required")
    fields = ["target_significance"]
    if modality == "ieeg":
        fields.extend(["significance_details", "encoding_results"])
    reference = _load_artifact(
        selection.reference_artifact, results_root, fields, subjects
    )
    if selection.mode == "within-layer":
        sources = [reference]
    else:
        specs = expand_significance_artifact_specs(selection.artifacts)
        source_fields = fields if modality == "ieeg" else fields[:1]
        sources = [
            _load_artifact(spec, results_root, source_fields, subjects)
            for spec in specs
        ]
        identities = [source.identity_record() for source in sources]
        identity_keys = [
            tuple(sorted((name, repr(value)) for name, value in identity.items()))
            for identity in identities
        ]
        if len(identity_keys) != len(set(identity_keys)):
            raise ValueError("Duplicate expanded significance artifact identities")
        if modality == "ieeg":
            validate_ieeg_encoding_alignment(reference, sources, subjects)

    expected_subjects = {str(subject) for subject in subjects}
    reference_subjects = set(reference.target_significance)
    if reference_subjects != expected_subjects:
        raise ValueError(
            "Reference significance subjects differ from requested subjects: "
            f"{sorted(reference_subjects)} != {sorted(expected_subjects)}"
        )
    reference_kinds = set(_canonical_targets(reference)["target_kind"])
    for source in sources:
        if (
            source.mode != reference.mode
            or source.dataset != reference.dataset
            or int(source.offset) != int(reference.offset)
        ):
            raise ValueError(
                "Significance union sources must match reference modality, dataset, "
                "and offset"
            )
        if set(source.target_significance) != expected_subjects:
            raise ValueError("Every significance union source must contain requested subjects")
        if set(_canonical_targets(source)["target_kind"]) != reference_kinds:
            raise ValueError("Significance union sources must match reference target kind")

    targets, details = union_significance_artifacts(sources, modality=modality)
    encoding_results = getattr(reference, "encoding_results", {})
    return PreparedSignificance(targets, details, encoding_results, reference, sources)
