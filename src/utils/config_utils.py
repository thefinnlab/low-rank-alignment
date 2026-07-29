"""Configuration utilities for argument parsing and config overrides."""

import argparse
import enum
from dataclasses import asdict
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import get_type_hints, get_origin, get_args, Union

import yaml

from src.core.config import EncodingConfig, SignificanceConfig


_REDACTED_CONFIG_KEYS = {"hf_token", "token", "api_key", "access_token"}
_OPERATIONAL_CONFIG_KEYS = {
    "refresh",
    "prefetch_rank_designs",
    "pca_svd_device",
}


def _config_value(value, *, redact: bool = True):
    """Convert a config value into stable, safe YAML primitives."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if redact and str(key).lower() in _REDACTED_CONFIG_KEYS and item
                else _config_value(item, redact=redact)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_config_value(item, redact=redact) for item in value]
    return value


def effective_config_dict(config, *, redact: bool = True) -> dict:
    """Return the effective dataclass configuration as YAML-safe primitives."""
    value = _config_value(config, redact=redact)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dataclass or mapping config, got {type(config)}")
    return value


def _artifact_config(value):
    """Remove operational refresh switches before provenance comparison."""
    if isinstance(value, dict):
        return {
            key: _artifact_config(item)
            for key, item in value.items()
            if key not in _OPERATIONAL_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [_artifact_config(item) for item in value]
    return value


def save_effective_config(config, output_dir, filename: str = "config.yml") -> Path:
    """Save effective config, rejecting incompatible reuse unless refresh is true.

    ``refresh`` is deliberately ignored for equality because it controls execution,
    not the artifact contract. Credentials are redacted both on disk and during the
    comparison.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    current = effective_config_dict(config)
    if path.exists():
        saved = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if _artifact_config(saved) != _artifact_config(current) and not bool(
            getattr(config, "refresh", False)
        ):
            raise ValueError(
                f"Existing artifact config differs at {path}; set refresh=true "
                "to replace artifacts or use a different output directory."
            )
    path.write_text(
        yaml.safe_dump(current, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def parse_override_args(unknown_args: list[str]) -> dict:
    """Convert ['--encoding.layer=48', '--fdr_q=0.05'] to {'encoding.layer': 48, 'fdr_q': 0.05}.

    Uses yaml.safe_load to preserve types (int, float, bool, list).
    """
    overrides = {}
    for arg in unknown_args:
        if arg.startswith("--") and "=" in arg:
            key, val = arg[2:].split("=", 1)
            if key and val is not None:
                overrides[key] = yaml.safe_load(val)
    return overrides


def get_nested_value(obj, path: str):
    """Get nested value from dict or dataclass using dot-notation path."""
    current = obj
    for field in path.split("."):
        if isinstance(current, dict):
            current = current[field]
        elif is_dataclass(current):
            current = getattr(current, field)
        else:
            raise TypeError(f"Cannot access '{field}' on {type(current)}")
    return current


def set_nested_attr(obj, key_path: str, value) -> None:
    """Set nested attribute on dict or dataclass using dot-notation path."""
    keys = key_path.split(".")
    target = obj
    for key in keys[:-1]:
        if is_dataclass(target):
            target = getattr(target, key)
        elif isinstance(target, dict):
            target = target.setdefault(key, {})
        else:
            raise TypeError(f"Unsupported type {type(target)} for key '{key}'")

    final_key = keys[-1]
    if is_dataclass(target):
        setattr(target, final_key, value)
    elif isinstance(target, dict):
        target[final_key] = value
    else:
        raise TypeError(f"Unsupported type {type(target)} for final key '{final_key}'")


def apply_overrides(config, overrides: dict):
    """Deep-copy config and apply override dict using dot-notation keys."""
    config = deepcopy(config)
    for key_path, value in overrides.items():
        set_nested_attr(config, key_path, value)
    validate = getattr(config, "validate", None)
    if callable(validate):
        validate()
    return config


def _resolve_type(hint):
    """Unwrap Optional[X] → X; return hint unchanged otherwise."""
    origin = get_origin(hint)
    if origin is Union:
        args = [a for a in get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def dict_to_dataclass(d: dict, cls):
    """Recursively convert dict d to an instance of cls (a dataclass).

    - Nested dicts whose corresponding field is itself a dataclass are converted recursively.
    - Unknown keys in d are silently ignored.
    - Missing keys use the dataclass field defaults.
    """
    hints = get_type_hints(cls)
    valid_fields = {f.name for f in fields(cls)}
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        value = d[f.name]
        resolved = _resolve_type(hints[f.name])
        if is_dataclass(resolved) and isinstance(value, dict):
            kwargs[f.name] = dict_to_dataclass(value, resolved)
        elif get_origin(resolved) is list and isinstance(value, list):
            args = get_args(resolved)
            if args and is_dataclass(args[0]):
                kwargs[f.name] = [
                    dict_to_dataclass(item, args[0]) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                kwargs[f.name] = value
        elif (
            isinstance(resolved, type)
            and issubclass(resolved, enum.Enum)
            and isinstance(value, str)
        ):
            kwargs[f.name] = resolved(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(config_path: str | None, overrides: dict, config_cls):
    """Load YAML config, convert to config_cls dataclass, then apply overrides.

    If config_path is None or empty, starts from dataclass defaults.
    """
    if config_path:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        config = dict_to_dataclass(raw, config_cls)
    else:
        config = config_cls()
    return apply_overrides(config, overrides)


def expand_per_model_layer_configs(config):
    """Return one config per ``encoding.per_model_config.layer`` value.

    Existing scalar-layer configs are returned unchanged. List-valued layer configs
    are deep-copied into scalar-layer configs so downstream code derives output
    directories and loads embeddings one layer at a time.
    """
    encoding = getattr(config, "encoding", config)
    layer = getattr(encoding, "per_model_config", {}).get("layer")
    if not isinstance(layer, list):
        return [config]

    configs = []
    for scalar_layer in layer:
        expanded = deepcopy(config)
        expanded_encoding = getattr(expanded, "encoding", expanded)
        expanded_encoding.per_model_config["layer"] = scalar_layer
        for field_name in ("cross_result_dir", "output_dir", "sig_dir", "rrr_output_dir"):
            value = getattr(expanded, field_name, None)
            if isinstance(value, str) and "{layer}" in value:
                setattr(expanded, field_name, value.format(layer=scalar_layer))
        configs.append(expanded)
    return configs


def expand_fmri_layer_configs(config):
    """Fan out list-valued fMRI layers and resolve ``{layer}`` path templates."""
    layer = getattr(config, "layer", None)
    if not isinstance(layer, list):
        return [config]

    configs = []
    for scalar_layer in layer:
        expanded = deepcopy(config)
        expanded.layer = scalar_layer
        for field_name in ("output_dir", "significance_dir", "source_rrr_unit_dir"):
            value = getattr(expanded, field_name, None)
            if isinstance(value, str) and "{layer}" in value:
                setattr(expanded, field_name, value.format(layer=scalar_layer))
        configs.append(expanded)
    return configs


def parse_args(description: str = ""):
    """Return (args, overrides_dict) where args.config is the YAML path."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    args, unknown = parser.parse_known_args()
    overrides = parse_override_args(unknown)
    return args, overrides
