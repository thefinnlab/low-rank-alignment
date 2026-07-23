import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.core.config import LanguageFeatureGenerationConfig
from src.utils.language_feature_providers import (
    LANCASTER_COLUMNS,
    _choose_parser_component,
    _dependency_derivatives,
    lexical_norms_provider,
    native_provider,
    swow_provider,
)
from src.utils.language_features import (
    AnnotationResult,
    StimulusDocument,
    _canonical_tokens,
    generate_language_features,
    register_annotation_provider,
    register_document_loader,
    stage_language_resources,
)


def _document() -> StimulusDocument:
    native = pd.DataFrame(
        {
            "word_idx": range(4),
            "word": ["The", "ice", "cream", "melts."],
            "start": [0.0, 0.2, 0.4, 0.8],
            "end": [0.1, 0.3, 0.7, 1.0],
            "sentence_idx": [0, 0, 0, 0],
            "lemma": ["the", "ice", "cream", "melt"],
            "pos": ["DET", "NOUN", "NOUN", "VERB"],
            "gpt2_surprisal": [1.0, 2.0, 3.0, 4.0],
        }
    )
    tokens = pd.DataFrame(
        {
            "token_id": range(4),
            "word": native["word"],
            "start": native["start"],
            "end": native["end"],
            "sentence_id": native["sentence_idx"],
            "token_in_sentence": range(4),
        }
    )
    document = StimulusDocument("fixture", "story", tokens, native)
    document.validate()
    return document


def test_canonical_tokens_assign_contiguous_negative_sentence_runs_separately():
    native = pd.DataFrame(
        {
            "word": list("abcdefg"),
            "start": np.arange(7, dtype=float),
            "end": np.arange(7, dtype=float) + 0.5,
            "sentence_idx": [-1, -1, 2, -1, 3, -1, -1],
        }
    )

    tokens = _canonical_tokens(native)

    assert tokens["sentence_id"].tolist() == [4, 4, 2, 5, 3, 6, 6]


def test_native_controls_are_word_aligned_and_non_destructive():
    document = _document()
    result = native_provider(document, document.tokens, {}, {})
    assert result.token_features["native_lemma"].tolist() == [
        "the",
        "ice",
        "cream",
        "melt",
    ]
    np.testing.assert_allclose(result.token_features["word_duration"], [0.1, 0.1, 0.3, 0.2])
    np.testing.assert_allclose(
        result.token_features["pre_word_pause"], [np.nan, 0.1, 0.1, 0.1], equal_nan=True
    )
    assert "lemma" not in result.token_features


def test_lexical_lookup_uses_longest_phrase_and_emits_missingness(tmp_path):
    document = _document()
    token_table = document.tokens.copy()
    token_table["lemma"] = document.native_tokens["lemma"]
    token_table["part_of_speech"] = document.native_tokens["pos"]
    resource = tmp_path / "norms.csv"
    pd.DataFrame(
        {
            "term": ["ice", "cream", "ice cream", "melt"],
            "score": [1.0, 2.0, 5.0, 3.0],
        }
    ).to_csv(resource, index=False)
    resources = {
        "example": {
            "path": str(resource),
            "type": "lexical_norm",
            "key_column": "term",
            "columns": {"example_score": "score"},
        }
    }
    result = lexical_norms_provider(
        document, token_table, {"resources": ["example"]}, resources
    )
    np.testing.assert_allclose(
        result.token_features["example_score"], [np.nan, 5.0, 5.0, 3.0], equal_nan=True
    )
    assert result.token_features["example_score_available"].tolist() == [False, True, True, True]
    assert result.spans[["start_token_id", "end_token_id"]].values.tolist() == [[1, 2]]


def test_lexical_lookup_ignores_missing_resource_keys(tmp_path):
    document = _document()
    resource = tmp_path / "norms.csv"
    pd.DataFrame(
        {"term": ["ice", None, ""], "score": [1.0, 2.0, 3.0]}
    ).to_csv(resource, index=False)
    result = lexical_norms_provider(
        document,
        document.tokens,
        {"resources": ["example"]},
        {
            "example": {
                "path": str(resource),
                "key_column": "term",
                "columns": {"score": "score"},
            }
        },
    ).token_features
    assert result.loc[1, "score"] == 1.0


def test_emotional_intensity_is_absolute_valence(tmp_path):
    document = _document()
    resource = tmp_path / "vad.csv"
    pd.DataFrame(
        {"term": ["the", "ice", "cream", "melt"], "valence": [0.0, -0.8, 0.4, 1.0]}
    ).to_csv(resource, index=False)
    result = lexical_norms_provider(
        document,
        document.tokens.assign(
            lemma=document.native_tokens["lemma"],
            part_of_speech=document.native_tokens["pos"],
        ),
        {"resources": ["vad"]},
        {
            "vad": {
                "path": str(resource),
                "type": "lexical_norm",
                "key_column": "term",
                "columns": {"valence": "valence"},
            }
        },
    ).token_features

    np.testing.assert_allclose(result["emotional_intensity"], [0.0, 0.8, 0.4, 1.0])


def test_swow_provider_normalizes_weights_on_pandas_three(tmp_path):
    document = _document()
    resource = tmp_path / "swow.tsv"
    pd.DataFrame(
        {
            "cue": ["ice", "ice"],
            "response": ["cold", "cream"],
            "strength": [3.0, 1.0],
        }
    ).to_csv(resource, sep="\t", index=False)
    result = swow_provider(
        document,
        document.tokens.assign(lemma=["the", "ice", "cream", "melt"]),
        {"resource": "swow"},
        {
            "swow": {
                "path": str(resource),
                "separator": "\t",
                "cue_column": "cue",
                "response_column": "response",
                "weight_column": "strength",
            }
        },
    ).token_features
    assert result.loc[1, "association_degree"] == 2.0
    assert result.loc[1, "association_entropy"] == pytest.approx(
        -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
    )


def test_lancaster_derivatives_use_l3_and_max_over_sum(tmp_path):
    document = _document()
    token_table = document.tokens.copy()
    token_table["lemma"] = document.native_tokens["lemma"]
    token_table["part_of_speech"] = document.native_tokens["pos"]
    resource = tmp_path / "lancaster.csv"
    row = {"word": "melt", **{column: 1.0 for column in LANCASTER_COLUMNS}}
    pd.DataFrame([row]).to_csv(resource, index=False)
    resources = {
        "lancaster": {
            "path": str(resource),
            "type": "lexical_norm",
            "key_column": "word",
            "columns": {column: column for column in LANCASTER_COLUMNS},
        }
    }
    result = lexical_norms_provider(
        document, token_table, {"resources": ["lancaster"]}, resources
    ).token_features
    assert result.loc[3, "sensorimotor_strength"] == pytest.approx(11 ** (1 / 3))
    assert result.loc[3, "sensorimotor_exclusivity"] == pytest.approx(1 / 11)
    assert np.isnan(result.loc[0, "sensorimotor_strength"])


def test_pos_distinguished_resource_does_not_mix_lemma_classes(tmp_path):
    document = _document()
    token_table = document.tokens.iloc[:1].copy()
    token_table["word"] = ["recorded"]
    token_table["lemma"] = ["record"]
    token_table["part_of_speech"] = ["verb"]
    resource = tmp_path / "pos_norm.csv"
    pd.DataFrame(
        {
            "term": ["record", "record"],
            "pos": ["noun", "verb"],
            "score": [1.0, 9.0],
        }
    ).to_csv(resource, index=False)
    result = lexical_norms_provider(
        document,
        token_table,
        {"resources": ["pos_norm"]},
        {
            "pos_norm": {
                "path": str(resource),
                "key_column": "term",
                "pos_column": "pos",
                "columns": {"score": "score"},
            }
        },
    ).token_features
    assert result.loc[0, "score"] == 9.0
    assert result.loc[0, "pos_norm_match_level"] == "lemma_pos"


def test_dependency_derivatives_match_hand_computed_tree():
    # 0 <- 1 <- {2, 3}; token 0 is the root.
    result = _dependency_derivatives(np.array([-1, 0, 1, 1]))
    np.testing.assert_array_equal(result["dependency_depth"], [0, 1, 2, 2])
    np.testing.assert_array_equal(result["subtree_size"], [4, 3, 1, 1])
    np.testing.assert_array_equal(result["number_of_dependents"], [1, 2, 0, 0])
    np.testing.assert_array_equal(result["head_direction"], ["root", "left", "left", "left"])
    np.testing.assert_array_equal(result["open_dependencies"], [1, 2, 1, 0])
    np.testing.assert_array_equal(result["dependency_locality_cost"], [0, 1, 1, 2])


def test_parser_component_selection_prefers_lexical_word_over_punctuation():
    words = [
        SimpleNamespace(upos="NOUN"),
        SimpleNamespace(upos="PUNCT"),
    ]

    chosen = _choose_parser_component(words, [0, 1], np.array([2.0, 1.0]))

    assert chosen == 0


def test_parser_component_selection_keeps_syntactic_head_for_contraction():
    words = [
        SimpleNamespace(upos="PRON"),
        SimpleNamespace(upos="AUX"),
        SimpleNamespace(upos="PUNCT"),
    ]

    chosen = _choose_parser_component(words, [0, 1, 2], np.array([2.0, 1.0, 0.0]))

    assert chosen == 1


def test_parser_component_selection_allows_punctuation_only_token():
    words = [
        SimpleNamespace(upos="PUNCT"),
        SimpleNamespace(upos="PUNCT"),
    ]

    chosen = _choose_parser_component(words, [0, 1], np.array([2.0, 1.0]))

    assert chosen == 1


def test_resource_staging_validates_checksum(tmp_path):
    resource = tmp_path / "resource.csv"
    resource.write_text("word,value\ntest,1\n")
    config = LanguageFeatureGenerationConfig(
        providers=["native"],
        resource_dir=str(tmp_path),
        resources={"test": {"path": str(resource), "sha256": "wrong"}},
    )
    with pytest.raises(ValueError, match="Checksum mismatch"):
        stage_language_resources(config, download=False)


def test_generator_writes_manifest_and_reuses_identical_artifact(tmp_path):
    dataset_name = "language_feature_fixture"
    provider_name = "language_feature_fixture_provider"

    if dataset_name not in __import__(
        "src.utils.language_features", fromlist=["DOCUMENT_LOADERS"]
    ).DOCUMENT_LOADERS:

        @register_document_loader(dataset_name)
        def fixture_loader(config):
            del config
            return _document()

    if provider_name not in __import__(
        "src.utils.language_features", fromlist=["PROVIDERS"]
    ).PROVIDERS:

        @register_annotation_provider(provider_name)
        def fixture_provider(document, token_table, options, resources):
            del document, options, resources
            return AnnotationResult(
                pd.DataFrame(
                    {
                        "token_id": token_table["token_id"],
                        "fixture_value": np.arange(len(token_table)),
                    }
                )
            )

    config = LanguageFeatureGenerationConfig(
        dataset=dataset_name,
        output_dir=str(tmp_path / "output"),
        resource_dir=str(tmp_path / "resources"),
        providers=[provider_name],
    )
    first = generate_language_features(config)
    second = generate_language_features(config)
    assert first == second
    tokens = pd.read_csv(first / "tokens.tsv", sep="\t")
    assert tokens["fixture_value"].tolist() == [0, 1, 2, 3]
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["feature_providers"]["fixture_value"] == provider_name
