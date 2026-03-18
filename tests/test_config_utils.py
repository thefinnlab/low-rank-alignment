"""Tests for utils/config_utils.py and utils/config.py."""

import tempfile
import os

import pytest
import yaml

from utils.config import EncodingConfig, SignificanceConfig, get_results_dir
from utils.config_utils import (
    parse_override_args,
    dict_to_dataclass,
    apply_overrides,
    load_config,
)


# ---------------------------------------------------------------------------
# parse_override_args
# ---------------------------------------------------------------------------

class TestParseOverrideArgs:
    def test_int(self):
        result = parse_override_args(["--layer=48"])
        assert result == {"layer": 48}
        assert isinstance(result["layer"], int)

    def test_float(self):
        result = parse_override_args(["--fdr_q=0.05"])
        assert result == {"fdr_q": 0.05}
        assert isinstance(result["fdr_q"], float)

    def test_bool_true(self):
        result = parse_override_args(["--batch=true"])
        assert result == {"batch": True}
        assert isinstance(result["batch"], bool)

    def test_bool_false(self):
        result = parse_override_args(["--batch=false"])
        assert result == {"batch": False}
        assert isinstance(result["batch"], bool)

    def test_list(self):
        result = parse_override_args(["--subjects=[1,2,3]"])
        assert result == {"subjects": [1, 2, 3]}
        assert isinstance(result["subjects"], list)

    def test_dot_notation_key(self):
        result = parse_override_args(["--encoding.layer=48"])
        assert result == {"encoding.layer": 48}

    def test_multiple_args(self):
        result = parse_override_args(["--encoding.layer=48", "--n_permutations=100"])
        assert result == {"encoding.layer": 48, "n_permutations": 100}

    def test_ignores_non_flag_args(self):
        result = parse_override_args(["not_a_flag", "--layer=24"])
        assert result == {"layer": 24}

    def test_ignores_flag_without_equals(self):
        result = parse_override_args(["--batch"])
        assert result == {}


# ---------------------------------------------------------------------------
# dict_to_dataclass
# ---------------------------------------------------------------------------

class TestDictToDataclass:
    def test_flat_encoding_config(self):
        d = {"layer": 48, "offset": 1, "data_root": "other/"}
        config = dict_to_dataclass(d, EncodingConfig)
        assert config.layer == 48
        assert config.offset == 1
        assert config.data_root == "other/"
        assert config.batch is False  # default

    def test_nested_significance_config(self):
        d = {
            "encoding": {"layer": 48, "data_root": "my_data/"},
            "n_permutations": 200,
        }
        config = dict_to_dataclass(d, SignificanceConfig)
        assert isinstance(config.encoding, EncodingConfig)
        assert config.encoding.layer == 48
        assert config.encoding.data_root == "my_data/"
        assert config.n_permutations == 200
        assert config.fdr_q == 0.01  # default

    def test_partial_dict_uses_defaults(self):
        config = dict_to_dataclass({}, EncodingConfig)
        assert config.layer == 24
        assert config.offset == 0
        assert config.subjects is None

    def test_unknown_keys_silently_ignored(self):
        d = {"layer": 48, "unknown_field": "oops"}
        config = dict_to_dataclass(d, EncodingConfig)
        assert config.layer == 48
        assert not hasattr(config, "unknown_field")


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------

class TestApplyOverrides:
    def test_flat_override_encoding(self):
        config = EncodingConfig()
        result = apply_overrides(config, {"layer": 48})
        assert result.layer == 48

    def test_nested_override_significance(self):
        config = SignificanceConfig()
        result = apply_overrides(config, {"encoding.layer": 48})
        assert result.encoding.layer == 48

    def test_original_unchanged(self):
        config = SignificanceConfig()
        apply_overrides(config, {"encoding.layer": 48, "n_permutations": 100})
        assert config.encoding.layer == 24
        assert config.n_permutations == 5000

    def test_multiple_overrides(self):
        config = SignificanceConfig()
        result = apply_overrides(config, {"n_permutations": 100, "fdr_q": 0.05})
        assert result.n_permutations == 100
        assert result.fdr_q == 0.05


# ---------------------------------------------------------------------------
# get_results_dir
# ---------------------------------------------------------------------------

class TestGetResultsDir:
    def test_default_base_dir(self):
        result = get_results_dir("gpt2xl", 24, 0)
        assert result == os.path.join("results/", "gpt2xl_layer24_offset0")

    def test_custom_base_dir(self):
        result = get_results_dir("gpt2xl", 48, 0, base_dir="my_results/")
        assert result == os.path.join("my_results/", "gpt2xl_layer48_offset0")

    def test_nonzero_offset(self):
        result = get_results_dir("gpt2xl", 24, 2)
        assert "offset2" in result

    def test_negative_offset(self):
        result = get_results_dir("gpt2xl", 24, -1)
        assert "offset-1" in result

    def test_different_models_different_dirs(self):
        a = get_results_dir("gpt2xl", 24, 0)
        b = get_results_dir("llama", 24, 0)
        assert a != b

    def test_different_layers_different_dirs(self):
        a = get_results_dir("gpt2xl", 24, 0)
        b = get_results_dir("gpt2xl", 48, 0)
        assert a != b

    def test_consistent_with_encoding_config_defaults(self):
        cfg = EncodingConfig()
        expected = get_results_dir(cfg.model, cfg.layer, cfg.offset)
        assert expected == os.path.join("results/", "gpt2xl_layer24_offset0")


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def _write_yaml(self, d, path):
        with open(path, "w") as f:
            yaml.dump(d, f)

    def test_round_trip_significance(self):
        data = {
            "encoding": {"layer": 48, "data_root": "data/", "offset": 0,
                         "subjects": None, "run_name": None, "batch": False},
            "output_dir": None,
            "n_permutations": 5000,
            "fdr_q": 0.01,
            "save_weights": False,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sig.yml")
            self._write_yaml(data, path)
            config = load_config(path, {}, SignificanceConfig)

        assert isinstance(config, SignificanceConfig)
        assert config.encoding.layer == 48
        assert config.n_permutations == 5000
        assert config.fdr_q == 0.01

    def test_cli_overrides_take_precedence(self):
        data = {
            "encoding": {"layer": 24},
            "n_permutations": 5000,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sig.yml")
            self._write_yaml(data, path)
            config = load_config(path, {"encoding.layer": 48, "n_permutations": 100}, SignificanceConfig)

        assert config.encoding.layer == 48
        assert config.n_permutations == 100

    def test_none_config_path_uses_defaults(self):
        config = load_config(None, {}, EncodingConfig)
        assert config.layer == 24
        assert config.offset == 0
