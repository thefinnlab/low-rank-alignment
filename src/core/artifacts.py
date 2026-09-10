"""Private mechanics for script-owned result Artifacts.

This module deliberately does not expose a user-facing Artifact or reference type.
Producer scripts subclass :class:`ArtifactStore` and declare their own contracts.
"""

from __future__ import annotations

import io
import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, ClassVar


SCHEMA_VERSION = 2
COMPACT_SCHEMA_VERSION = 3
_OPERATIONAL_KEYS = {"refresh", "plot", "results_root"}


class ArtifactError(RuntimeError):
    """Base error for managed result artifacts."""


class ArtifactAmbiguityError(ArtifactError):
    """Raised when a canonical identity contains more than one variant."""


class ArtifactIncompleteError(ArtifactError):
    """Raised when a run cannot be marked complete."""


def _primitive(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        value = value.value
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def normalized_config(value: Any, *, drop_paths: bool = True) -> dict:
    """Return deterministic configuration data suitable for artifact identity."""
    raw = _primitive(value)
    if not isinstance(raw, dict):
        raise TypeError("Artifact configuration must be a dataclass or mapping")

    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                if key in _OPERATIONAL_KEYS:
                    continue
                if drop_paths and (key.endswith("_root") or key.endswith("_dir") or key.endswith("_path")):
                    continue
                result[key] = clean(child)
            return result
        if isinstance(item, list):
            return [clean(child) for child in item]
        return item

    return clean(raw)


def readable_variant(config: Any, upstreams: list[dict] | None = None,
                     extras: dict | None = None) -> str:
    """Return a stable variant path with readable roles and compact selection IDs."""
    normalized = normalized_config(config)
    explicit = normalized.get("variant_name")
    if explicit:
        return _relative_components(str(explicit), "variant name")
    rrr_sources = [item for item in (upstreams or [])
                   if item.get("script") == "run_rrr"]
    parts = []
    if rrr_sources:
        for index, source in enumerate(rrr_sources):
            if len(rrr_sources) > 1:
                parts.extend((f"source-{index + 1}",))
            else:
                parts.extend(("source",))
            parts.extend((
                str(source["mode"]), str(source["dataset"]),
                f"offset-{int(source.get('offset', 0))}", str(source["model"]),
                f"layer-{'none' if source.get('layer') is None else int(source['layer'])}",
            ))
            source_variant = str(source.get("variant") or "")
            for prefix in ("rereference-", "aggregation-"):
                source_component = next(
                    (
                        component
                        for component in source_variant.split("/")
                        if component.startswith(prefix)
                    ),
                    None,
                )
                if source_component is not None:
                    parts.append(f"source-{source_component}")
    if extras:
        parts.extend(
            f"{key}-{value}" for key, value in sorted(_primitive(extras).items())
        )

    aggregation = normalized.get("subspace_aggregation")
    if aggregation is not None:
        parts.append(f"aggregation-{aggregation}")

    selection = normalized.get("significance_selection")
    if isinstance(selection, dict) and selection.get("reference_artifact"):
        canonical = canonical_significance_selection(selection)
        encoded = json.dumps(_primitive(canonical), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:10]
        parts.append(f"significance-{canonical['mode']}-{digest}")

    return "/".join(
        _component(part, "variant component") for part in (parts or ["default"])
    )


def canonical_significance_selection(selection: Any) -> dict:
    """Canonicalize a significance selection for RRR identity and validation."""
    normalized = normalized_config({"significance_selection": selection})[
        "significance_selection"
    ]
    if not isinstance(normalized, dict) or not normalized.get("reference_artifact"):
        raise ValueError(
            "significance_selection must define a reference_artifact"
        )
    canonical = {
        "mode": normalized.get("mode", "within-layer"),
        "reference_artifact": normalized["reference_artifact"],
        "artifacts": [],
    }
    expanded = []
    for artifact in normalized.get("artifacts", []):
        layers = artifact.get("layer")
        for layer in layers if isinstance(layers, list) else [layers]:
            concrete = dict(artifact)
            concrete["layer"] = layer
            expanded.append(concrete)
    canonical["artifacts"] = sorted(
        expanded,
        key=lambda item: json.dumps(_primitive(item), sort_keys=True),
    )
    return canonical


def normalize_dataset(dataset: str, stimulus: str | None = None) -> str:
    if dataset == "podcast":
        return "podcast"
    if dataset in {"brain_treebank", "brain-treebank"}:
        if not stimulus:
            raise ValueError("Brain Treebank Artifact identity requires a stimulus")
        return f"brain-treebank-{stimulus}"
    if dataset in {"fmri", "deep_fmri", "deep-fmri"}:
        return "deep-fmri"
    return _component(dataset, "dataset")


def normalize_model(model: str, *, glove_dim: int | None = None) -> str:
    if model == "glove" or model.startswith("glove-"):
        dim = glove_dim
        if dim is None and model.startswith("glove-"):
            dim = int(model.split("-", 1)[1])
        if dim is None:
            raise ValueError("GloVe Artifact identity requires its dimension")
        return f"glove-{int(dim)}"
    return _component(model, "model")


def _component(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"Invalid {label} path component: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", text):
        raise ValueError(f"Invalid {label} path component: {value!r}")
    return text


def _relative_components(value: str, label: str) -> str:
    parts = Path(value).parts
    if not parts or Path(value).is_absolute():
        raise ValueError(f"Invalid {label}: {value!r}")
    return "/".join(_component(part, label) for part in parts)


class ArtifactStore:
    """Private base used only by concrete producer-script Artifact classes."""

    SCRIPT: ClassVar[str]
    FIELD_CONTRACT: ClassVar[dict[str, bool]]
    SELECTOR_ORDER: ClassVar[tuple[str, ...]] = ()
    COMPACT_STORAGE: ClassVar[bool] = False
    CONTAINER_NAME: ClassVar[str] = "artifact.h5"

    def _init_store(
        self,
        *,
        mode: str,
        dataset: str,
        offset: int,
        model: str,
        layer: int | None,
        variant: str | None,
        results_root: str | os.PathLike[str],
    ) -> None:
        self.mode = _component(mode, "mode")
        self.dataset = _component(dataset, "dataset")
        self.offset = int(offset)
        self.model = _component(model, "model")
        self.layer = None if layer is None else int(layer)
        self.variant = variant
        self.results_root = Path(results_root)
        self.complete = False
        for field in self.FIELD_CONTRACT:
            setattr(self, field, None)

    @property
    def identity(self) -> dict:
        return {
            "script": self.SCRIPT,
            "mode": self.mode,
            "dataset": self.dataset,
            "offset": self.offset,
            "model": self.model,
            "layer": self.layer,
        }

    def identity_record(self) -> dict:
        return {**self.identity, "variant": self.variant}

    def _leaf(self) -> Path:
        return (
            self.results_root
            / self.SCRIPT
            / self.mode
            / self.dataset
            / f"offset-{self.offset}"
            / self.model
            / f"layer-{'none' if self.layer is None else self.layer}"
        )

    def _variant_dir(self) -> Path:
        if self.variant is None:
            self._resolve_variant()
        return self._leaf() / _relative_components(str(self.variant), "variant")

    def _manifest_path(self) -> Path:
        return self._variant_dir() / "manifest.json"

    def _resolve_variant(self) -> str:
        index = self._read_json(self._leaf() / "manifest.json", default={})
        variants = sorted(index.get("variants", {}))
        if not variants:
            raise FileNotFoundError(f"No Artifact variants exist for {self.identity}")
        if len(variants) != 1:
            raise ArtifactAmbiguityError(
                f"Artifact identity {self.identity} has {len(variants)} variants; "
                "specify variant explicitly: " + ", ".join(variants)
            )
        self.variant = variants[0]
        return self.variant

    def begin(self, *, config: Any, upstreams: list[dict] | None = None,
              refresh: bool = False, variant_extras: dict | None = None) -> "ArtifactStore":
        computed = readable_variant(config, upstreams, variant_extras)
        selected = _relative_components(str(self.variant), "variant") if self.variant else computed
        self.variant = selected
        path = self._manifest_path()
        if path.exists() and not refresh:
            manifest = self._read_json(path)
            expected_config = normalized_config(config)
            expected_upstreams = _primitive(upstreams or [])
            stored_upstreams = manifest.get("upstreams")
            if stored_upstreams is None:
                stored_upstreams = []
            if (manifest.get("configuration") != expected_config
                    or stored_upstreams != expected_upstreams):
                raise ArtifactAmbiguityError(
                    f"Readable variant {selected!r} already represents a different "
                    "configuration; set an explicit variant_name"
                )
            self.complete = bool(manifest.get("complete"))
            return self
        manifest = {
            "schema_version": (
                COMPACT_SCHEMA_VERSION if self.COMPACT_STORAGE else SCHEMA_VERSION
            ),
            "identity": self.identity,
            "variant": selected,
            "configuration": normalized_config(config),
            "upstreams": _primitive(upstreams or []),
            "declared_fields": {
                name: {"required": required} for name, required in self.FIELD_CONTRACT.items()
            },
            "entries": {},
            "complete": False,
        }
        if self.COMPACT_STORAGE:
            manifest["storage"] = "hdf5"
        self._write_json(path, manifest)
        self._update_index(selected, complete=False)
        return self

    def save_field(self, field: str, value: Any, *, selector: dict | None = None,
                   overwrite: bool = False) -> None:
        self._validate_field(field)
        manifest = self._manifest()
        key = self._selector_key(selector)
        existing = manifest["entries"].get(field, {}).get(key)
        if existing and not overwrite:
            return
        codec, suffix, payload, codec_metadata = self._encode(field, value)
        relative = self._field_path(field, selector or {}, suffix)
        if manifest.get("storage") == "hdf5" and field != "plots":
            object_path = self._write_container_payload(relative, payload)
            entry = {
                "selector": _primitive(selector or {}),
                "codec": codec,
                "container": self.CONTAINER_NAME,
                "object": object_path,
                "size": len(payload),
                **codec_metadata,
            }
        else:
            destination = self._variant_dir() / relative
            self._atomic_bytes(destination, payload)
            entry = {
                "selector": _primitive(selector or {}),
                "codec": codec,
                "file": relative.as_posix(),
                "size": len(payload),
                **codec_metadata,
            }
        manifest["entries"].setdefault(field, {})[key] = entry
        manifest["complete"] = False
        self._write_json(self._manifest_path(), manifest)
        self._update_index(str(self.variant), complete=False)
        if existing and existing.get("container") and existing.get("object"):
            self._delete_container_object(existing["object"])

    def has_field(self, field: str, *, selector: dict | None = None) -> bool:
        self._validate_field(field)
        return self._selector_key(selector) in self._manifest().get("entries", {}).get(field, {})

    def load_field(self, field: str, *, selector: dict | None = None) -> Any:
        self._validate_field(field)
        entry = self._manifest().get("entries", {}).get(field, {}).get(
            self._selector_key(selector)
        )
        if entry is None:
            raise KeyError(f"Artifact field {field!r} selector {selector or {}} is not saved")
        if "container" in entry:
            payload = self._read_container_payload(entry)
        else:
            payload = (self._variant_dir() / entry["file"]).read_bytes()
        try:
            return self._decode(entry, payload)
        except Exception as error:
            raise ArtifactError(
                f"Could not decode field {field!r} selector {selector or {}}"
            ) from error

    def load_collection(self, field: str, filters: dict | None = None) -> dict:
        self._validate_field(field)
        filters = filters or {}
        result = {}
        for entry in self._manifest().get("entries", {}).get(field, {}).values():
            selector = entry["selector"]
            if all(selector.get(key) in (value if isinstance(value, (list, tuple, set)) else [value])
                   for key, value in filters.items()):
                result[tuple(sorted(selector.items()))] = self.load_field(field, selector=selector)
        return result

    def mark_complete(self, required_selectors: dict[str, list[dict]] | None = None) -> None:
        manifest = self._manifest()
        missing = []
        required_selectors = required_selectors or {}
        fields_to_check = {
            field for field, required in self.FIELD_CONTRACT.items() if required
        } | set(required_selectors)
        for field in fields_to_check:
            required = self.FIELD_CONTRACT[field]
            selectors = required_selectors.get(field, [{}] if required else [])
            for selector in selectors:
                if self._selector_key(selector) not in manifest["entries"].get(field, {}):
                    missing.append((field, selector))
        if missing:
            raise ArtifactIncompleteError(f"Cannot complete Artifact; missing fields/selectors: {missing}")
        manifest["complete"] = True
        self._write_json(self._manifest_path(), manifest)
        self._update_index(str(self.variant), complete=True)
        self.complete = True

    def manifest(self) -> dict:
        """Return a defensive copy of logical manifest metadata, never storage paths."""
        return json.loads(json.dumps(self._manifest()))

    def _manifest(self) -> dict:
        return self._read_json(self._manifest_path())

    def _validate_field(self, field: str) -> None:
        if field not in self.FIELD_CONTRACT:
            raise ValueError(f"Unsupported {self.SCRIPT} Artifact field {field!r}; expected {sorted(self.FIELD_CONTRACT)}")

    @classmethod
    def _selector_key(cls, selector: dict | None) -> str:
        selector = _primitive(selector or {})
        positions = {key: index for index, key in enumerate(cls.SELECTOR_ORDER)}
        keys = sorted(selector, key=lambda key: (
            positions.get(key, len(positions)), key
        ))
        if not keys:
            return "default"
        return "/".join(
            f"{_component(key, 'selector key')}-"
            f"{_component(selector[key], 'selector value')}"
            for key in keys
        )

    @classmethod
    def _field_path(cls, field: str, selector: dict, suffix: str) -> Path:
        selector = _primitive(selector)
        name = selector.pop("name", None)
        directory = Path(field)
        key = cls._selector_key(selector)
        if key != "default":
            directory /= key
        filename = _component(name, "field name") if name is not None else field
        aliases = {
            "input_factors": "factors", "target_factors": "target_factors",
            "rank_metrics": "metrics", "full_metrics": "metrics",
            "fold_results": "fold_results", "mean_results": "mean_results",
        }
        filename = aliases.get(field, filename)
        return directory / (filename if filename.endswith(suffix) else f"{filename}{suffix}")

    def _update_index(self, variant: str, *, complete: bool) -> None:
        path = self._leaf() / "manifest.json"
        index = self._read_json(path, default={
            "schema_version": SCHEMA_VERSION,
            "identity": self.identity,
            "variants": {},
        })
        index["variants"][variant] = {"complete": bool(complete)}
        self._write_json(path, index)

    def _write_container_payload(self, relative: Path, payload: bytes) -> str:
        """Write one versioned payload into the variant's HDF5 container."""
        import h5py
        import numpy as np

        logical = relative.with_suffix("").as_posix()
        object_path = f"/entries/{logical}/versions/{uuid.uuid4().hex}"
        container = self._variant_dir() / self.CONTAINER_NAME
        container.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(container, "a") as handle:
            handle.create_dataset(
                object_path,
                data=np.frombuffer(payload, dtype=np.uint8),
                compression="gzip",
                shuffle=True,
            )
            handle.flush()
        return object_path

    def _read_container_payload(self, entry: dict) -> bytes:
        import h5py

        container = self._variant_dir() / entry["container"]
        with h5py.File(container, "r") as handle:
            return handle[entry["object"]][...].tobytes()

    def _delete_container_object(self, object_path: str) -> None:
        """Remove a superseded payload after its replacement is manifested."""
        import h5py

        container = self._variant_dir() / self.CONTAINER_NAME
        if not container.exists():
            return
        with h5py.File(container, "a") as handle:
            if object_path in handle:
                del handle[object_path]
                handle.flush()

    @staticmethod
    def _read_json(path: Path, default: Any = None) -> Any:
        if not path.exists():
            if default is not None:
                return default
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def _write_json(cls, path: Path, value: Any) -> None:
        cls._atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"))

    @staticmethod
    def _encode(field: str, value: Any) -> tuple[str, str, bytes, dict]:
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                buffer = io.BytesIO()
                np.save(buffer, value, allow_pickle=value.dtype == object)
                return "npy", ".npy", buffer.getvalue(), {}
        except ImportError:
            pass
        try:
            import pandas as pd
            if isinstance(value, pd.DataFrame):
                payload = value.to_csv(index=False).encode("utf-8")
                return "csv", ".csv", payload, {
                    "columns": [str(column) for column in value.columns],
                    "dtypes": {str(column): str(dtype)
                               for column, dtype in value.dtypes.items()},
                    "string_columns": [
                        str(column) for column in value.columns
                        if str(value[column].dtype) == "object"
                        and all(
                            isinstance(item, str)
                            or (
                                pd.api.types.is_scalar(item)
                                and bool(pd.isna(item))
                            )
                            for item in value[column]
                        )
                    ],
                }
        except ImportError:
            pass
        if isinstance(value, bytes):
            if field != "plots":
                raise TypeError(f"Raw bytes are only supported for plot fields, not {field}")
            return "png", ".png", value, {}
        if value is None or isinstance(value, (str, int, float, bool, list, dict)):
            try:
                payload = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
                return "json", ".json", payload, {}
            except (TypeError, ValueError):
                pass
        if field in {"encoding_results", "fitted_models"}:
            import joblib
            buffer = io.BytesIO()
            joblib.dump(value, buffer)
            return "joblib", ".joblib", buffer.getvalue(), {}
        if isinstance(value, dict):
            try:
                import numpy as np
                buffer = io.BytesIO()
                packed = {}
                npz_types = {}
                for key, item in value.items():
                    key = str(key)
                    if isinstance(item, np.ndarray):
                        packed[key] = item
                        npz_types[key] = "ndarray"
                    else:
                        wrapped = np.empty((), dtype=object)
                        wrapped[()] = item
                        packed[key] = wrapped
                        npz_types[key] = "python"
                np.savez(buffer, **packed)
                return "npz", ".npz", buffer.getvalue(), {
                    "npz_types": npz_types
                }
            except (TypeError, ValueError):
                pass
        raise TypeError(
            f"Artifact field {field!r} has no portable codec for {type(value).__name__}"
        )

    @staticmethod
    def _decode(entry: dict, payload: bytes) -> Any:
        codec = entry["codec"]
        if codec == "json":
            return json.loads(payload)
        if codec == "npy":
            import numpy as np
            return np.load(io.BytesIO(payload), allow_pickle=True)
        if codec == "npz":
            import numpy as np
            with np.load(io.BytesIO(payload), allow_pickle=True) as loaded:
                result = {}
                for key in loaded.files:
                    value = loaded[key]
                    if (entry.get("npz_types", {}).get(key) == "python"
                            and value.shape == ()):
                        value = value.item()
                    result[key] = value
                return result
        if codec == "csv":
            import pandas as pd
            frame = pd.read_csv(
                io.BytesIO(payload),
                dtype={column: str for column in entry.get("string_columns", [])},
            )
            for column, dtype in entry.get("dtypes", {}).items():
                if column not in frame:
                    continue
                if dtype == "object":
                    frame[column] = frame[column].astype(object)
                else:
                    try:
                        frame[column] = frame[column].astype(dtype)
                    except (TypeError, ValueError):
                        pass
            return frame
        if codec == "png":
            return payload
        if codec == "joblib":
            import joblib
            return joblib.load(io.BytesIO(payload))
        raise ArtifactError(f"Unsupported Artifact codec {codec!r}")
