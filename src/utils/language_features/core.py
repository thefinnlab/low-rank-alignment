"""Generation of reusable stimulus-aligned language features."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from src.core.config import LanguageFeatureGenerationConfig
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.shared.config import save_effective_config


SCHEMA_VERSION = 2
EMPTY_SIDECAR_COLUMNS = {
    "spans.tsv": [
        "provider",
        "span_id",
        "start_token_id",
        "end_token_id",
        "label",
        "span_type",
    ],
    "relations.tsv": [
        "provider",
        "relation_id",
        "source_span_id",
        "target_span_id",
        "label",
    ],
    "provider_tokens.tsv": [
        "provider",
        "provider_token_id",
        "sentence_id",
        "token_id",
        "text",
    ],
    "segments.tsv": [
        "provider",
        "segment_id",
        "start",
        "end",
        "label",
        "segment_type",
    ],
}


@dataclass
class StimulusDocument:
    """Canonical word-level input shared by all annotation providers."""

    dataset: str
    stimulus_id: str
    tokens: pd.DataFrame
    native_tokens: pd.DataFrame
    media: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        required = {"token_id", "word", "start", "end", "sentence_id"}
        missing = required - set(self.tokens)
        if missing:
            raise ValueError(f"Stimulus tokens are missing columns: {sorted(missing)}")
        ids = self.tokens["token_id"].to_numpy(dtype=int)
        if not np.array_equal(ids, np.arange(len(self.tokens))):
            raise ValueError("token_id must be contiguous and zero-based")
        starts = pd.to_numeric(self.tokens["start"], errors="coerce").to_numpy()
        ends = pd.to_numeric(self.tokens["end"], errors="coerce").to_numpy()
        finite = np.isfinite(starts) & np.isfinite(ends)
        if np.any(ends[finite] < starts[finite]):
            raise ValueError("Token end times must not precede their start times")
        if self.tokens["word"].isna().any():
            raise ValueError("Canonical words must not be missing")


@dataclass
class AnnotationResult:
    """One provider's token features and optional structured sidecars."""

    token_features: pd.DataFrame
    spans: pd.DataFrame = field(default_factory=pd.DataFrame)
    relations: pd.DataFrame = field(default_factory=pd.DataFrame)
    provider_tokens: pd.DataFrame = field(default_factory=pd.DataFrame)
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    version: str
    runner: Callable
    requires: tuple[str, ...] = ()


DOCUMENT_LOADERS: dict[str, Callable] = {}
PROVIDERS: dict[str, ProviderSpec] = {}
PROVIDER_OVERRIDES: dict[tuple[str, str], Callable] = {}


def register_document_loader(dataset: str):
    def decorator(loader):
        if dataset in DOCUMENT_LOADERS:
            raise ValueError(f"Duplicate language-feature document loader: {dataset}")
        DOCUMENT_LOADERS[dataset] = loader
        return loader

    return decorator


def register_annotation_provider(
    name: str, *, version: str = "1", requires: tuple[str, ...] = ()
):
    def decorator(runner):
        if name in PROVIDERS:
            raise ValueError(f"Duplicate annotation provider: {name}")
        PROVIDERS[name] = ProviderSpec(name, version, runner, requires)
        return runner

    return decorator


def register_provider_override(provider: str, dataset: str):
    def decorator(runner):
        key = (provider, dataset)
        if key in PROVIDER_OVERRIDES:
            raise ValueError(f"Duplicate provider override: {key}")
        PROVIDER_OVERRIDES[key] = runner
        return runner

    return decorator


def _collapse_word_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse subtoken transcript rows without discarding source columns."""
    if "word_idx" not in df:
        result = df.copy().reset_index(drop=True)
        result["word_idx"] = np.arange(len(result))
        return result
    rows = []
    for word_idx, group in df.groupby("word_idx", sort=True):
        row = group.iloc[0].copy()
        row["word_idx"] = int(word_idx)
        if "start" in group:
            row["start"] = pd.to_numeric(group["start"], errors="coerce").min()
        if "end" in group:
            row["end"] = pd.to_numeric(group["end"], errors="coerce").max()
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _canonical_tokens(native: pd.DataFrame) -> pd.DataFrame:
    word_col = "word" if "word" in native else "text"
    sentence = (
        pd.to_numeric(native["sentence_idx"], errors="coerce")
        if "sentence_idx" in native
        else pd.Series(np.zeros(len(native), dtype=int))
    )
    sentence = sentence.fillna(-1).astype(int)
    negative = sentence.lt(0)
    if negative.any():
        next_sentence_id = int(sentence.loc[~negative].max()) + 1 if (~negative).any() else 0
        run_starts = negative & ~negative.shift(fill_value=False)
        run_ids = run_starts.cumsum() - 1
        sentence.loc[negative] = next_sentence_id + run_ids.loc[negative]
    tokens = pd.DataFrame(
        {
            "token_id": np.arange(len(native), dtype=int),
            "word": native[word_col].astype(str).to_numpy(),
            "start": pd.to_numeric(native.get("start"), errors="coerce"),
            "end": pd.to_numeric(native.get("end"), errors="coerce"),
            "sentence_id": sentence.to_numpy(),
        }
    )
    if "idx_in_sentence" in native:
        tokens["token_in_sentence"] = pd.to_numeric(
            native["idx_in_sentence"], errors="coerce"
        ).astype("Int64")
    else:
        tokens["token_in_sentence"] = tokens.groupby("sentence_id").cumcount()
    if "speaker" in native:
        tokens["speaker"] = native["speaker"].replace("", pd.NA)
    source_mapping = {
        "word_idx": "source_word_idx",
        "source_word_idx": "native_source_word_idx",
        "source_word_idxs": "native_source_word_idxs",
    }
    for column, output_column in source_mapping.items():
        if column in native:
            tokens[output_column] = native[column].to_numpy()
    return tokens


def _attach_iu_ids(native: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Align PSST IU boundaries while preserving canonical transcript order."""
    if not path.exists():
        return native
    boundaries = pd.read_csv(path).sort_values("time").reset_index(drop=True)
    required = {"time", "prev_word", "next_word"}
    if not required <= set(boundaries):
        raise ValueError(f"IU annotations are missing columns {sorted(required - set(boundaries))}")
    normalized = [re.sub(r"^[^\w]+|[^\w]+$", "", str(word).lower()) for word in native["word"]]
    onsets = pd.to_numeric(native["start"], errors="coerce").to_numpy(float)
    starts = {0}
    previous_start = 0
    fallback_count = 0
    for _, boundary in boundaries.iterrows():
        previous = re.sub(r"^[^\w]+|[^\w]+$", "", str(boundary["prev_word"]).lower())
        following = re.sub(r"^[^\w]+|[^\w]+$", "", str(boundary["next_word"]).lower())
        candidates = [
            index + 1
            for index in range(previous_start, len(normalized) - 1)
            if normalized[index] == previous and normalized[index + 1] == following
        ]
        if candidates:
            word_index = min(
                candidates,
                key=lambda index: abs(onsets[index] - float(boundary["time"])),
            )
        else:
            finite_onsets = np.where(np.isfinite(onsets), onsets, np.inf)
            word_index = int(np.argmin(np.abs(finite_onsets - float(boundary["time"]))))
            fallback_count += 1
        starts.add(word_index)
        previous_start = max(previous_start, word_index)
    ordered = np.array(sorted(starts), dtype=int)
    stops = np.r_[ordered[1:], len(native)]
    iu_ids = np.full(len(native), -1, dtype=int)
    for iu_id, (start, stop) in enumerate(zip(ordered, stops)):
        iu_ids[start:stop] = iu_id
    result = native.copy()
    result["iu_id"] = iu_ids
    result.attrs["iu_alignment"] = {
        "boundaries": int(len(boundaries)),
        "time_fallbacks": int(fallback_count),
    }
    return result


def _annotation_path(template: str | None, stimulus_id: str) -> Path | None:
    """Resolve an optional stimulus-scoped annotation path template."""
    if not template:
        return None
    return Path(str(template).format(story=stimulus_id, stimulus_id=stimulus_id))


@register_document_loader("podcast")
def load_podcast_document(config: LanguageFeatureGenerationConfig) -> StimulusDocument:
    context = resolve_dataset_context(
        "podcast", [], config.per_dataset_config, config.data_root
    )
    adapter = get_dataset_adapter("podcast")
    native = _collapse_word_rows(
        adapter.load_word_table(
            config.data_root,
            context,
            model_dir=config.model_dir,
            drop_untimed_words=True,
        )
    )
    annotation_path = config.dataset_options.get(
        "word_annotations", "processed_data/df_word_onset_with_pos_class.csv"
    )
    annotation_path = Path(annotation_path)
    if annotation_path.exists():
        annotations = pd.read_csv(annotation_path)
        annotations = annotations.loc[
            :, ~annotations.columns.astype(str).str.startswith("Unnamed:")
        ]
        if len(annotations) != len(native):
            raise ValueError(
                "Podcast word annotations do not match the canonical transcript: "
                f"{len(annotations)} vs {len(native)}"
            )
        if not np.array_equal(
            annotations["word"].astype(str).to_numpy(),
            native["word"].astype(str).to_numpy(),
        ):
            raise ValueError("Podcast annotation words do not match the transcript")
        annotations = annotations.rename(columns={"onset": "start", "offset": "end"})
        for column in annotations:
            if column not in {"word", "word_idx", "start", "end"}:
                native[column] = annotations[column].to_numpy()
        native["start"] = annotations["start"].to_numpy()
        native["end"] = annotations["end"].to_numpy()
    iu_path = Path(
        config.dataset_options.get("iu_annotations", "processed_data/iu_boundaries.csv")
    )
    native = _attach_iu_ids(native, iu_path)
    audio = config.dataset_options.get(
        "audio", os.path.join(config.data_root, "stimuli", "podcast.wav")
    )
    document = StimulusDocument(
        "podcast",
        context.stimulus_id,
        _canonical_tokens(native),
        native.reset_index(drop=True),
        media={"audio": str(audio)} if Path(audio).exists() else {},
        metadata={"iu_alignment": native.attrs.get("iu_alignment", {})},
    )
    document.validate()
    return document


@register_document_loader("brain_treebank")
def load_brain_treebank_document(
    config: LanguageFeatureGenerationConfig,
) -> StimulusDocument:
    context = resolve_dataset_context(
        "brain_treebank", [], config.per_dataset_config, config.data_root
    )
    native = get_dataset_adapter("brain_treebank").load_word_table(
        config.data_root, context, drop_untimed_words=True
    )
    iu_path = _annotation_path(
        config.dataset_options.get("iu_annotations"), context.stimulus_id
    )
    if iu_path is not None:
        native = _attach_iu_ids(native, iu_path)
    document = StimulusDocument(
        "brain_treebank",
        context.stimulus_id,
        _canonical_tokens(native),
        native.reset_index(drop=True),
        metadata={"stimulus_label": context.stimulus_label},
    )
    document.validate()
    return document


@register_document_loader("fmri")
def load_fmri_document(config: LanguageFeatureGenerationConfig) -> StimulusDocument:
    """Load one deep-fMRI story from its exact word-tier TextGrid."""
    from src.utils.shared.fmri import load_story_word_data

    story = config.per_dataset_config.get("story")
    if not story or not isinstance(story, str):
        raise ValueError("fMRI language features require per_dataset_config.story")
    frames, _ = load_story_word_data(config.data_root, [story])
    native = frames[story].copy().reset_index(drop=True)

    sentence_path = _annotation_path(
        config.dataset_options.get("sentence_annotations"), story
    )
    if sentence_path is None or not sentence_path.exists():
        raise FileNotFoundError(
            "fMRI language features require SaT sentence annotations at "
            f"{sentence_path}"
        )
    sentences = pd.read_csv(sentence_path, sep="\t", keep_default_na=False)
    required = {"token_id", "word", "sentence_id"}
    if not required <= set(sentences):
        raise ValueError(
            f"Sentence annotations are missing {sorted(required - set(sentences))}"
        )
    if len(sentences) != len(native) or not np.array_equal(
        sentences["word"].astype(str).str.lower().to_numpy(),
        native["word"].astype(str).str.lower().to_numpy(),
    ):
        raise ValueError(f"SaT sentence annotations do not match fMRI story {story}")
    native["sentence_idx"] = pd.to_numeric(
        sentences["sentence_id"], errors="raise"
    ).astype(int)
    native["idx_in_sentence"] = native.groupby("sentence_idx").cumcount()

    iu_path = _annotation_path(config.dataset_options.get("iu_annotations"), story)
    if iu_path is not None:
        native = _attach_iu_ids(native, iu_path)
    audio = Path(config.data_root) / "data" / "ds003020" / "stimuli" / f"{story}.wav"
    document = StimulusDocument(
        "fmri",
        story,
        _canonical_tokens(native),
        native,
        media={"audio": str(audio)} if audio.exists() else {},
        metadata={
            "sentence_annotations": str(sentence_path),
            "iu_annotations": str(iu_path) if iu_path is not None else None,
            "iu_alignment": native.attrs.get("iu_alignment", {}),
        },
    )
    document.validate()
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    for item in paths:
        if path.is_dir():
            digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def stage_language_resources(
    config: LanguageFeatureGenerationConfig, *, download: bool = True
) -> dict[str, dict]:
    """Download permitted resources and validate every configured checksum."""
    root = Path(config.resource_dir)
    root.mkdir(parents=True, exist_ok=True)
    staged = {}
    for name, raw_spec in config.resources.items():
        spec = dict(raw_spec)
        path = Path(spec.get("path", spec.get("filename", name)))
        if not path.is_absolute():
            path = root / path
        resource_type = spec.get("type")
        needs_staging = not path.exists()
        if resource_type == "stanza_model" and path.is_dir():
            required_models = spec.get("required_models", [])
            needs_staging = (
                any(not (path / model).exists() for model in required_models)
                if required_models
                else not any(path.rglob("*.pt"))
            )
        elif resource_type == "huggingface_model" and path.is_dir():
            required_files = spec.get("required_files", ["config.json"])
            needs_staging = any(not (path / filename).exists() for filename in required_files)
        elif resource_type == "nltk_corpus" and path.is_dir():
            corpus = spec.get("corpus", name)
            needs_staging = not (
                (path / "corpora" / corpus).exists()
                or (path / "corpora" / f"{corpus}.zip").exists()
            )
        if needs_staging:
            if spec.get("manual", False):
                raise FileNotFoundError(
                    f"Resource {name!r} requires manual staging at {path}"
                )
            if download and resource_type == "stanza_model":
                try:
                    import stanza
                except ImportError as exc:
                    raise ImportError("Staging Stanza models requires 'stanza'") from exc
                stanza.download(
                    spec.get("language", "en"),
                    package=spec.get("package", "ewt"),
                    processors=spec.get("processors", {}),
                    model_dir=str(path),
                )
            elif download and resource_type == "huggingface_model":
                try:
                    from huggingface_hub import snapshot_download
                except ImportError as exc:
                    raise ImportError(
                        "Staging Hugging Face models requires 'huggingface_hub'"
                    ) from exc
                path.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=spec["repo_id"],
                    revision=spec.get("revision"),
                    local_dir=str(path),
                    allow_patterns=spec.get("allow_patterns"),
                )
            elif download and resource_type == "nltk_corpus":
                try:
                    import nltk
                except ImportError as exc:
                    raise ImportError("Staging NLTK corpora requires 'nltk'") from exc
                path.mkdir(parents=True, exist_ok=True)
                nltk_data_path = str(path.resolve())
                if nltk_data_path not in nltk.data.path:
                    nltk.data.path.append(nltk_data_path)
                if not nltk.download(spec.get("corpus", name), download_dir=str(path)):
                    raise RuntimeError(f"NLTK failed to stage corpus {name!r}")
            elif not download or not spec.get("url"):
                raise FileNotFoundError(f"Resource {name!r} not found at {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".part")
                urllib.request.urlretrieve(spec["url"], temporary)
                temporary.replace(path)
        actual = _sha256(path)
        expected = spec.get("sha256")
        if expected and actual.lower() != str(expected).lower():
            raise ValueError(
                f"Checksum mismatch for {name}: expected {expected}, got {actual}"
            )
        staged[name] = {**spec, "path": str(path.resolve()), "sha256": actual}
    return staged


def hydrate_provider_resources(config: LanguageFeatureGenerationConfig) -> None:
    """Reuse provider/resource declarations from another explicit YAML config."""
    source = config.provider_resource_config
    if not source:
        return
    payload = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    inherited_options = payload.get("provider_options", {})
    inherited_resources = payload.get("resources", {})
    config.provider_options = {
        name: {**inherited_options.get(name, {}), **config.provider_options.get(name, {})}
        for name in set(inherited_options) | set(config.provider_options)
    }
    config.resources = {**inherited_resources, **config.resources}


def _provider_order(requested: list[str]) -> list[str]:
    missing = [name for name in requested if name not in PROVIDERS]
    if missing:
        raise ValueError(
            f"Unknown annotation provider(s) {missing}; available={sorted(PROVIDERS)}"
        )
    requested_set = set(requested)
    order: list[str] = []
    visiting: set[str] = set()

    def visit(name: str):
        if name in order:
            return
        if name in visiting:
            raise ValueError(f"Annotation provider dependency cycle at {name}")
        visiting.add(name)
        for dependency in PROVIDERS[name].requires:
            if dependency not in requested_set:
                raise ValueError(f"Provider {name!r} requires provider {dependency!r}")
            visit(dependency)
        visiting.remove(name)
        order.append(name)

    for provider in requested:
        visit(provider)
    return order


def _input_fingerprint(document: StimulusDocument, config, resources) -> str:
    token_payload = document.tokens.to_csv(index=False, lineterminator="\n")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tokens": token_payload,
        "config": asdict(config),
        "resources": resources,
        "provider_versions": {
            name: PROVIDERS[name].version for name in config.providers
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def generate_language_features(config: LanguageFeatureGenerationConfig) -> Path:
    """Run configured providers and atomically write one stimulus artifact."""
    hydrate_provider_resources(config)
    # Import registers built-in providers without coupling this core to their code.
    from src.utils.language_features import providers as _builtins  # noqa: F401

    if config.dataset not in DOCUMENT_LOADERS:
        raise ValueError(
            f"No language-feature loader for {config.dataset!r}; "
            f"available={sorted(DOCUMENT_LOADERS)}"
        )
    document = DOCUMENT_LOADERS[config.dataset](config)
    resources = stage_language_resources(config, download=False)
    output = Path(config.output_dir) / document.dataset / document.stimulus_id
    fingerprint = _input_fingerprint(document, config, resources)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not config.refresh:
        previous = json.loads(manifest_path.read_text())
        if previous.get("input_fingerprint") == fingerprint:
            save_effective_config(config, output)
            return output
        raise ValueError(
            f"Existing feature artifact at {output} has different inputs; set refresh=true"
        )

    token_table = document.tokens.copy()
    span_tables = []
    relation_tables = []
    provider_token_tables = []
    segment_tables = []
    provider_metadata = {}
    feature_providers: dict[str, str] = {}
    for name in _provider_order(config.providers):
        spec = PROVIDERS[name]
        runner = PROVIDER_OVERRIDES.get((name, document.dataset), spec.runner)
        result = runner(
            document,
            token_table.copy(),
            config.provider_options.get(name, {}),
            resources,
        )
        if not isinstance(result, AnnotationResult):
            raise TypeError(f"Provider {name} did not return AnnotationResult")
        features = result.token_features.reset_index(drop=True)
        if len(features) != len(token_table):
            raise ValueError(
                f"Provider {name} returned {len(features)} rows for {len(token_table)} tokens"
            )
        collisions = (set(features) - {"token_id"}) & set(token_table)
        if collisions:
            raise ValueError(f"Provider {name} overwrote columns {sorted(collisions)}")
        if "token_id" in features and not np.array_equal(
            features["token_id"].to_numpy(), token_table["token_id"].to_numpy()
        ):
            raise ValueError(f"Provider {name} returned misaligned token IDs")
        additions = features.drop(columns=["token_id"], errors="ignore")
        for column in additions:
            feature_providers[column] = name
        token_table = pd.concat([token_table, additions], axis=1)
        for table, collection in (
            (result.spans, span_tables),
            (result.relations, relation_tables),
            (result.provider_tokens, provider_token_tables),
            (result.segments, segment_tables),
        ):
            if not table.empty:
                tagged = table.copy()
                tagged.insert(0, "provider", name)
                if collection is span_tables:
                    required = {"span_id", "start_token_id", "end_token_id"}
                    missing = required - set(tagged)
                    if missing:
                        raise ValueError(
                            f"Provider {name} span table is missing {sorted(missing)}"
                        )
                    starts = tagged["start_token_id"].to_numpy(dtype=int)
                    ends = tagged["end_token_id"].to_numpy(dtype=int)
                    if np.any(starts < 0) or np.any(ends < starts) or np.any(ends >= len(token_table)):
                        raise ValueError(f"Provider {name} returned invalid span bounds")
                if collection is provider_token_tables:
                    if "token_id" not in tagged:
                        raise ValueError(f"Provider {name} token crosswalk lacks token_id")
                    ids = tagged["token_id"].to_numpy(dtype=int)
                    if np.any(ids < 0) or np.any(ids >= len(token_table)):
                        raise ValueError(f"Provider {name} returned invalid crosswalk token IDs")
                if collection is segment_tables:
                    required = {"segment_id", "start", "end", "label", "segment_type"}
                    missing = required - set(tagged)
                    if missing:
                        raise ValueError(
                            f"Provider {name} segment table is missing {sorted(missing)}"
                        )
                    starts = pd.to_numeric(tagged["start"], errors="coerce").to_numpy()
                    ends = pd.to_numeric(tagged["end"], errors="coerce").to_numpy()
                    if (
                        not np.isfinite(starts).all()
                        or not np.isfinite(ends).all()
                        or np.any(starts < 0)
                        or np.any(ends <= starts)
                    ):
                        raise ValueError(f"Provider {name} returned invalid timed segments")
                collection.append(tagged)
        provider_metadata[name] = {
            "version": spec.version,
            **result.metadata,
        }

    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    save_effective_config(config, temporary)
    token_table.to_csv(temporary / "tokens.tsv", sep="\t", index=False)
    for filename, tables in (
        ("spans.tsv", span_tables),
        ("relations.tsv", relation_tables),
        ("provider_tokens.tsv", provider_token_tables),
        ("segments.tsv", segment_tables),
    ):
        table = (
            pd.concat(tables, ignore_index=True)
            if tables
            else pd.DataFrame(columns=EMPTY_SIDECAR_COLUMNS[filename])
        )
        table.to_csv(temporary / filename, sep="\t", index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": document.dataset,
        "stimulus_id": document.stimulus_id,
        "n_tokens": len(token_table),
        "input_fingerprint": fingerprint,
        "resources": resources,
        "providers": provider_metadata,
        "feature_providers": feature_providers,
        "feature_coverage": {
            column: {
                "available": int(token_table[column].notna().sum()),
                "total": int(len(token_table)),
            }
            for column in feature_providers
        },
        "config": asdict(config),
        "document_metadata": document.metadata,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(output)
    return output
