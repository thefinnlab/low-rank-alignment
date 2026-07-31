"""Extensible word-aligned linguistic targets for RRR subspace decoding."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd


TaskKind = Literal["continuous", "categorical"]


@dataclass(frozen=True)
class TaskSpec:
    """Metadata and target getter for one decoding task."""

    name: str
    display_name: str
    kind: TaskKind
    getter: Callable[["PodcastTaskContext"], np.ndarray]
    chance: float | None = None


@dataclass
class PodcastTaskContext:
    """Lazy word annotations aligned to a dataset's embedding transcript."""

    word_df: pd.DataFrame
    data_root: str = ""
    paths: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def _resolve_path(self, key: str, default: str) -> str:
        configured = str(self.paths.get(key, default))
        if os.path.isabs(configured) or os.path.exists(configured):
            return configured
        return os.path.join(self.data_root, configured)

    def _embedding_words(self) -> pd.DataFrame:
        words = self.word_df.reset_index()
        if "word_idx" not in words:
            words["word_idx"] = np.arange(len(words))
        return words

    def _validate_alignment(
        self,
        table: pd.DataFrame,
        *,
        id_column: str,
        start_column: str,
        end_column: str,
        source_name: str,
    ) -> None:
        model_words = self._embedding_words()
        required = {id_column, "word", start_column, end_column}
        missing = required - set(table)
        if missing:
            raise ValueError(f"{source_name} is missing columns: {sorted(missing)}")
        if len(table) != len(model_words):
            raise ValueError(
                f"{source_name} and embedding transcript lengths differ: "
                f"{len(table)} vs {len(model_words)}"
            )
        token_ids = pd.to_numeric(table[id_column], errors="coerce").to_numpy()
        expected_ids = np.arange(len(table))
        if not np.array_equal(token_ids, expected_ids):
            raise ValueError(f"{source_name} token IDs must be contiguous from zero")
        model_ids = pd.to_numeric(model_words["word_idx"], errors="coerce").to_numpy()
        if not np.array_equal(token_ids, model_ids):
            raise ValueError(f"{source_name} token IDs do not match embedding word_idx")
        if not np.array_equal(
            table["word"].astype(str).to_numpy(),
            model_words["word"].astype(str).to_numpy(),
        ):
            raise ValueError(f"{source_name} word text does not match embedding transcript")
        for artifact_column, transcript_column in (
            (start_column, "start"),
            (end_column, "end"),
        ):
            artifact_times = pd.to_numeric(
                table[artifact_column], errors="coerce"
            ).to_numpy(dtype=float)
            transcript_times = pd.to_numeric(
                model_words[transcript_column], errors="coerce"
            ).to_numpy(dtype=float)
            errors = np.abs(artifact_times - transcript_times)
            mismatched = np.isfinite(errors) & (errors > 1e-6)
            if mismatched.mean() > 0.01 or np.nanmax(errors) > 0.25:
                raise ValueError(
                    f"{source_name} {artifact_column} timing does not match "
                    f"embedding transcript {transcript_column}"
                )
            if mismatched.any():
                warnings.warn(
                    f"Accepted {int(mismatched.sum())} minor {artifact_column} "
                    f"timing differences after exact token ID/text alignment",
                    RuntimeWarning,
                )

    @property
    def language_features(self) -> pd.DataFrame:
        """Return a generated token artifact after strict alignment checks."""
        if "language_features" not in self._cache:
            path = self._resolve_path(
                "language_features",
                "processed_data/language_features/podcast/podcast/tokens.tsv",
            )
            # Only empty TSV fields are missing; values such as the literal token
            # ``None`` are transcript text and must remain strings.
            features = pd.read_csv(
                path, sep="\t", keep_default_na=False, na_values=[""]
            )
            self._validate_alignment(
                features,
                id_column="token_id",
                start_column="start",
                end_column="end",
                source_name="Language-feature artifact",
            )
            self._cache["language_features"] = features
        return self._cache["language_features"]

    @property
    def self_surprisal(self) -> pd.DataFrame:
        """Return model-specific prediction metrics after strict alignment."""
        if "self_surprisal" not in self._cache:
            default = str(self.paths.get(
                "language_features",
                "processed_data/language_features/podcast/podcast/tokens.tsv",
            ))
            path = self._resolve_path("self_surprisal", default)
            features = pd.read_csv(
                path, sep="\t", keep_default_na=False, na_values=[""]
            )
            self._validate_alignment(
                features,
                id_column="token_id",
                start_column="start",
                end_column="end",
                source_name="Self-surprisal artifact",
            )
            required = {"self_surprisal", "self_entropy"}
            missing = required - set(features)
            if missing:
                raise ValueError(
                    f"Self-surprisal artifact {path} is missing {sorted(missing)}"
                )
            self._cache["self_surprisal"] = features
        return self._cache["self_surprisal"]

    @property
    def words(self) -> pd.DataFrame:
        """Return legacy Podcast annotations used only by ``surprisal_class``."""
        if "words" not in self._cache:
            path = self._resolve_path(
                "word_annotations", "processed_data/df_word_onset_with_pos_class.csv"
            )
            annotations = pd.read_csv(path, index_col=0).reset_index(drop=True)
            self._validate_alignment(
                annotations,
                id_column="word_idx",
                start_column="onset",
                end_column="offset",
                source_name="Legacy word annotations",
            )
            self._cache["words"] = annotations
        return self._cache["words"]


TASK_REGISTRY: dict[str, TaskSpec] = {}


def register_language_task(
    name: str,
    display_name: str,
    kind: TaskKind,
    chance: float | None = None,
):
    """Register a target getter; adding a task requires only a function and metadata."""

    def decorator(getter):
        if name in TASK_REGISTRY:
            raise ValueError(f"Duplicate language task: {name}")
        TASK_REGISTRY[name] = TaskSpec(name, display_name, kind, getter, chance)
        return getter

    return decorator


def get_language_task(name: str) -> TaskSpec:
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown language task {name!r}; available={sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[name]


def _continuous_artifact_target(
    context: PodcastTaskContext, column: str
) -> np.ndarray:
    """Read one generated numeric feature, retaining missing values as NaN."""
    if column not in context.language_features:
        raise ValueError(f"Language-feature artifact is missing column {column!r}")
    return pd.to_numeric(
        context.language_features[column], errors="coerce"
    ).to_numpy(dtype=np.float64)


def _categorical_artifact_target(
    context: PodcastTaskContext, column: str
) -> np.ndarray:
    """Encode present labels alphabetically and retain missing labels as NaN."""
    if column not in context.language_features:
        raise ValueError(f"Language-feature artifact is missing column {column!r}")
    values = context.language_features[column]
    present = values.notna()
    labels = values.loc[present].astype(str)
    label_ids = {label: index for index, label in enumerate(sorted(labels.unique()))}
    target = np.full(len(values), np.nan, dtype=np.float64)
    target[present.to_numpy()] = labels.map(label_ids).to_numpy(dtype=np.float64)
    return target


def _register_artifact_task(
    name: str,
    kind: TaskKind,
    display_name: str,
    chance: float | None = None,
) -> None:
    getter_helper = (
        _continuous_artifact_target if kind == "continuous" else _categorical_artifact_target
    )

    def getter(context: PodcastTaskContext, column: str = name) -> np.ndarray:
        return getter_helper(context, column)

    getter.__name__ = f"{name}_target"
    register_language_task(name, display_name, kind, chance)(getter)


# Generated continuous targets. These include native controls, standardized timing and
# structure, lexical norms, sensorimotor dimensions/composites, associations, WordNet
# counts, and deterministic dependency-tree measures.
_CONTINUOUS_TASKS = {
    "native_surprisal": "Native surprisal",
    "native_entropy": "Native entropy",
    "surprisal": "Standardized surprisal",
    "entropy": "Standardized entropy",
    "word_length": "Word length",
    "word_zipf_frequency": "Word Zipf frequency",
    "word_duration": "Word duration",
    "pre_word_pause": "Pause before word",
    "post_word_pause": "Pause after word",
    "sentence_position": "Position in sentence",
    "sentence_length": "Sentence length",
    "iu_position": "Position in IU",
    "iu_length": "IU length",
    "story_position": "Position in story (words)",
    "story_time_position": "Position in story (time)",
    "concreteness": "Concreteness",
    "age_of_acquisition": "Age of acquisition",
    "word_prevalence": "Word prevalence",
    "semantic_diversity": "Semantic diversity",
    "socialness": "Socialness",
    "valence": "Valence",
    "arousal": "Arousal",
    "dominance": "Dominance",
    "emotional_intensity": "Emotional intensity",
    "visual_strength": "Visual strength",
    "auditory_strength": "Auditory strength",
    "haptic_strength": "Haptic strength",
    "olfactory_strength": "Olfactory strength",
    "gustatory_strength": "Gustatory strength",
    "interoceptive_strength": "Interoceptive strength",
    "hand_arm_action_strength": "Hand/arm action strength",
    "mouth_throat_action_strength": "Mouth/throat action strength",
    "foot_leg_action_strength": "Foot/leg action strength",
    "head_action_strength": "Head action strength",
    "torso_action_strength": "Torso action strength",
    "sensorimotor_strength": "Aggregate sensorimotor strength",
    "sensorimotor_exclusivity": "Sensorimotor exclusivity",
    "association_degree": "Association degree",
    "association_entropy": "Association entropy",
    "sense_count": "WordNet sense count",
    "number_of_hypernyms_mfs": "WordNet MFS hypernym count",
    "ontology_depth_mfs": "WordNet MFS ontology depth",
    "dependency_depth": "Dependency depth",
    "dependency_length": "Dependency length",
    "number_of_dependents": "Number of dependents",
    "subtree_size": "Dependency subtree size",
    "left_dependents": "Left dependents",
    "right_dependents": "Right dependents",
    "open_dependencies": "Open dependencies",
    "dependency_locality_cost": "Dependency locality cost",
    "prior_word_count": "Prior occurrence count",
    "distance_from_previous_occurrence": "Distance from previous occurrence",
    "coreference_mention_order": "Coreference mention order",
    "coreference_chain_size": "Coreference chain size",
    "coreference_antecedent_distance_words": "Coreference antecedent distance",
}


@register_language_task("self_surprisal", "Self surprisal", "continuous")
def self_surprisal_target(context: PodcastTaskContext) -> np.ndarray:
    """Read surprisal produced by the embedding model's own LM head."""
    return pd.to_numeric(
        context.self_surprisal["self_surprisal"], errors="coerce"
    ).to_numpy(dtype=np.float64)


@register_language_task("self_entropy", "Self entropy", "continuous")
def self_entropy_target(context: PodcastTaskContext) -> np.ndarray:
    """Read first-subtoken entropy produced by the embedding model's LM head."""
    return pd.to_numeric(
        context.self_surprisal["self_entropy"], errors="coerce"
    ).to_numpy(dtype=np.float64)

_CATEGORICAL_TASKS = {
    "speaker_identity": ("Speaker identity", None),
    "native_part_of_speech": ("Native part of speech", None),
    "part_of_speech": ("Standardized UD part of speech", None),
    "fine_part_of_speech": ("Fine-grained part of speech", None),
    "content_word": ("Content vs function", 0.5),
    "sentence_onset": ("Sentence onset", 0.5),
    "sentence_end": ("Sentence end", 0.5),
    "iu_onset": ("IU onset", 0.5),
    "iu_end": ("IU end", 0.5),
    "wordnet_lexname_mfs": ("WordNet MFS lexname", None),
    "person_relatedness_mfs": ("WordNet MFS person relatedness", 0.5),
    "animacy_mfs": ("WordNet MFS animacy", None),
    "motion_concept_mfs": ("WordNet MFS motion concept", 0.5),
    "cognition_concept_mfs": ("WordNet MFS cognition concept", 0.5),
    "communication_concept_mfs": ("WordNet MFS communication concept", 0.5),
    "perception_concept_mfs": ("WordNet MFS perception concept", 0.5),
    "dependency_relation": ("UD dependency relation", None),
    "tense": ("UD tense", None),
    "mood": ("UD mood", None),
    "voice": ("UD voice", None),
    "grammatical_person": ("UD grammatical person", None),
    "number": ("UD number", None),
    "definiteness": ("UD definiteness", None),
    "head_direction": ("Dependency head direction", None),
    "filled_pause": ("Filled pause", 0.5),
    "immediate_word_repetition": ("Immediate word repetition", 0.5),
    "repeated_phrase": ("Repeated phrase", 0.5),
    "discourse_marker": ("Discourse marker", 0.5),
    "discourse_marker_type": ("Discourse marker type", None),
    "word_seen_before": ("Word seen previously", 0.5),
    "word_in_previous_5": ("Word in previous five tokens", 0.5),
    "is_named_entity": ("Named-entity membership", 0.5),
    "named_entity_type": ("Named-entity type", None),
    "is_coreference_mention": ("Coreference-mention membership", 0.5),
    "coreference_newness": ("Coreference mention newness", 0.5),
    "coreference_mention_form": ("Coreference mention form", None),
}

for _name, _display_name in _CONTINUOUS_TASKS.items():
    _register_artifact_task(_name, "continuous", _display_name, 0.0)
for _name, (_display_name, _chance) in _CATEGORICAL_TASKS.items():
    _register_artifact_task(_name, "categorical", _display_name, _chance)


@register_language_task(
    "surprisal_class", "Surprisal class", "categorical", 1.0 / 3.0
)
def surprisal_class_target(context: PodcastTaskContext) -> np.ndarray:
    """Read the one retained target that has no generated-artifact equivalent."""
    return context.words["surprise_class"].to_numpy(dtype=np.int64)
