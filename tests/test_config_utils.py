"""Tests for utils/config_utils.py and utils/config.py."""

import tempfile
import os

import pytest
import yaml

from src.core.config import (
    EncodingConfig,
    RRRConfig,
    SignificanceConfig,
    get_results_dir,
)
from src.core.registry import DatasetContext
from src.utils.config_utils import (
    parse_override_args,
    dict_to_dataclass,
    apply_overrides,
    load_config,
    expand_per_model_layer_configs,
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

    def test_significance_sampling_and_fdr_args(self):
        result = parse_override_args(
            [
                "--num_null_shifts=250",
                "--null_seed=7",
                "--fdr_mode=cluster",
            ]
        )
        assert result == {
            "num_null_shifts": 250,
            "null_seed": 7,
            "fdr_mode": "cluster",
        }

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
        d = {"per_model_config": {"layer": 48}, "offset": 1, "data_root": "other/"}
        config = dict_to_dataclass(d, EncodingConfig)
        assert config.per_model_config == {"layer": 48}
        assert config.offset == 1
        assert config.data_root == "other/"
        assert config.batch is False  # default

    def test_nested_significance_config(self):
        d = {
            "encoding": {"per_model_config": {"layer": 48}, "data_root": "my_data/"},
            "minimal_shift": 250,
        }
        config = dict_to_dataclass(d, SignificanceConfig)
        assert isinstance(config.encoding, EncodingConfig)
        assert config.encoding.per_model_config == {"layer": 48}
        assert config.encoding.data_root == "my_data/"
        assert config.minimal_shift == 250
        assert config.fdr_q == 0.01  # default
        assert config.num_null_shifts == 10000
        assert config.null_seed == 42
        assert config.fdr_mode == "electrode"

    def test_partial_dict_uses_defaults(self):
        config = dict_to_dataclass({}, EncodingConfig)
        assert config.per_model_config == {"layer": 24}
        assert config.dataset == "podcast"
        assert config.per_dataset_config == {}
        assert config.offset == 0
        assert config.subjects is None

    def test_unknown_keys_silently_ignored(self):
        d = {"offset": 2, "unknown_field": "oops"}
        config = dict_to_dataclass(d, EncodingConfig)
        assert config.offset == 2
        assert not hasattr(config, "unknown_field")

    def test_rrr_refresh_roundtrip(self):
        config = dict_to_dataclass({"refresh": True, "rank_grid": [1, 2]}, RRRConfig)
        assert config.refresh is True
        assert config.rank_grid == [1, 2]

    def test_alpha_grid_policy_roundtrip(self):
        config = dict_to_dataclass(
            {"alpha_grid": {"points_per_decade": 5}},
            EncodingConfig,
        )
        assert config.alpha_grid == {"points_per_decade": 5}

    def test_ridge_memory_options_roundtrip(self):
        config = dict_to_dataclass(
            {
                "ridge_cv_folds": 3,
                "ridge_solver_params": {
                    "n_alphas_batch": 8,
                    "n_targets_batch": 256,
                },
                "ridge_y_in_cpu": True,
                "ridge_force_cpu": False,
            },
            EncodingConfig,
        )
        assert config.ridge_cv_folds == 3
        assert config.ridge_solver_params == {
            "n_alphas_batch": 8,
            "n_targets_batch": 256,
        }
        assert config.ridge_y_in_cpu is True
        assert config.ridge_force_cpu is False

    def test_rrr_subspace_standardization_roundtrip(self):
        config = dict_to_dataclass(
            {"standardize_subspace_inputs": False},
            RRRConfig,
        )
        assert config.standardize_subspace_inputs is False

    def test_rrr_subspace_standardization_defaults_false(self):
        assert RRRConfig().standardize_subspace_inputs is False


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def test_flat_override_encoding(self):
        config = EncodingConfig()
        result = apply_overrides(config, {"offset": 2})
        assert result.offset == 2

    def test_nested_override_significance(self):
        config = SignificanceConfig()
        result = apply_overrides(config, {"encoding.offset": 2})
        assert result.encoding.offset == 2

    def test_original_unchanged(self):
        config = SignificanceConfig()
        apply_overrides(config, {"encoding.offset": 2, "fdr_q": 0.05})
        assert config.encoding.offset == 0
        assert config.fdr_q == 0.01

    def test_multiple_overrides(self):
        config = SignificanceConfig()
        result = apply_overrides(config, {"minimal_shift": 250, "fdr_q": 0.05})
        assert result.minimal_shift == 250
        assert result.fdr_q == 0.05

    def test_significance_sampling_and_fdr_overrides(self):
        config = SignificanceConfig()
        result = apply_overrides(
            config,
            {
                "num_null_shifts": 250,
                "null_seed": 7,
                "fdr_mode": "cluster",
            },
        )
        assert result.num_null_shifts == 250
        assert result.null_seed == 7
        assert result.fdr_mode == "cluster"

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"num_null_shifts": 0}, "positive integer"),
            ({"null_seed": 1.5}, "must be an integer"),
            ({"fdr_mode": "lag"}, "electrode.*cluster"),
        ],
    )
    def test_invalid_significance_overrides(self, override, message):
        with pytest.raises(ValueError, match=message):
            apply_overrides(SignificanceConfig(), override)

    def test_rrr_refresh_override(self):
        config = RRRConfig()
        result = apply_overrides(config, {"refresh": True})
        assert result.refresh is True
        assert config.refresh is False


# ---------------------------------------------------------------------------
# get_results_dir
# ---------------------------------------------------------------------------


class TestGetResultsDir:
    def test_default_base_dir(self):
        result = get_results_dir("gpt2xl", 0, {"layer": 24})
        assert result == os.path.join("results/", "gpt2xl_layer24_offset0")

    def test_custom_base_dir(self):
        result = get_results_dir("gpt2xl", 0, {"layer": 48}, base_dir="my_results/")
        assert result == os.path.join("my_results/", "gpt2xl_layer48_offset0")

    def test_nonzero_offset(self):
        result = get_results_dir("gpt2xl", 2, {"layer": 24})
        assert "offset2" in result

    def test_negative_offset(self):
        result = get_results_dir("gpt2xl", -1, {"layer": 24})
        assert "offset-1" in result

    def test_different_models_different_dirs(self):
        a = get_results_dir("gpt2xl", 0, {"layer": 24})
        b = get_results_dir("glove", 0, {"dim": 50})
        assert a != b

    def test_different_layers_different_dirs(self):
        a = get_results_dir("gpt2xl", 0, {"layer": 24})
        b = get_results_dir("gpt2xl", 0, {"layer": 48})
        assert a != b

    def test_consistent_with_encoding_config_defaults(self):
        cfg = EncodingConfig()
        expected = get_results_dir(cfg.model, cfg.offset, cfg.per_model_config)
        assert expected == os.path.join("results/", "gpt2xl_layer24_offset0")

    def test_glove_default_dir(self):
        result = get_results_dir("glove", 0, {"dim": 50})
        assert "glove_dim50_offset0" in result

    def test_transformer_model_dir_layer_dir(self):
        result = get_results_dir(
            "transformer", 0, {"model_dir": "gpt2-xl", "layer": 24}
        )
        assert result == os.path.join("results/", "gpt2-xl_layer24_offset0")

    def test_brain_treebank_results_dir_includes_movie(self):
        context = DatasetContext(
            dataset="brain_treebank",
            stimulus_id="coraline",
            stimulus_label="coraline",
        )
        result = get_results_dir("gpt2xl", 0, {"layer": 24}, dataset_context=context)
        assert result.endswith(os.path.join("gpt2xl_layer24_offset0", "movie-coraline"))

    def test_different_brain_treebank_movies_get_different_dirs(self):
        coraline = DatasetContext(
            dataset="brain_treebank",
            stimulus_id="coraline",
            stimulus_label="coraline",
        )
        venom = DatasetContext(
            dataset="brain_treebank",
            stimulus_id="venom",
            stimulus_label="venom",
        )
        a = get_results_dir("gpt2xl", 0, {"layer": 24}, dataset_context=coraline)
        b = get_results_dir("gpt2xl", 0, {"layer": 24}, dataset_context=venom)
        assert a != b

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            get_results_dir("llama", 0, {})


# ---------------------------------------------------------------------------
# expand_per_model_layer_configs
# ---------------------------------------------------------------------------


class TestExpandPerModelLayerConfigs:
    def test_scalar_layer_returns_original_config(self):
        config = SignificanceConfig(
            encoding=EncodingConfig(
                model="transformer",
                per_model_config={"model_dir": "gpt2-xl", "layer": 24},
            )
        )

        expanded = expand_per_model_layer_configs(config)

        assert expanded == [config]
        assert expanded[0].encoding.per_model_config["layer"] == 24

    def test_list_layer_expands_to_scalar_layer_configs(self):
        config = RRRConfig(
            encoding=EncodingConfig(
                model="transformer",
                per_model_config={"model_dir": "gpt2-xl", "layer": [0, 24, 48]},
            )
        )

        expanded = expand_per_model_layer_configs(config)

        assert [c.encoding.per_model_config["layer"] for c in expanded] == [0, 24, 48]
        assert all(
            not isinstance(c.encoding.per_model_config["layer"], list) for c in expanded
        )
        assert config.encoding.per_model_config["layer"] == [0, 24, 48]

    def test_encoding_config_list_layer_expands_to_scalar_layer_configs(self):
        config = EncodingConfig(
            model="transformer",
            per_model_config={"model_dir": "gpt2-xl", "layer": [0, 24, 48]},
        )

        expanded = expand_per_model_layer_configs(config)

        assert [c.per_model_config["layer"] for c in expanded] == [0, 24, 48]
        assert config.per_model_config["layer"] == [0, 24, 48]


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def _write_yaml(self, d, path):
        with open(path, "w") as f:
            yaml.dump(d, f)

    def test_round_trip_significance(self):
        data = {
            "encoding": {
                "per_model_config": {"layer": 48},
                "data_root": "data/",
                "offset": 0,
                "subjects": None,
                "run_name": None,
                "batch": False,
            },
            "output_dir": None,
            "minimal_shift": 500,
            "num_null_shifts": 123,
            "null_seed": 9,
            "fdr_q": 0.01,
            "fdr_mode": "cluster",
            "save_weights": False,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sig.yml")
            self._write_yaml(data, path)
            config = load_config(path, {}, SignificanceConfig)

        assert isinstance(config, SignificanceConfig)
        assert config.encoding.per_model_config == {"layer": 48}
        assert config.minimal_shift == 500
        assert config.num_null_shifts == 123
        assert config.null_seed == 9
        assert config.fdr_q == 0.01
        assert config.fdr_mode == "cluster"

    def test_cli_overrides_take_precedence(self):
        data = {
            "encoding": {"per_model_config": {"layer": 24}},
            "minimal_shift": 500,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sig.yml")
            self._write_yaml(data, path)
            config = load_config(
                path,
                {
                    "encoding.offset": 2,
                    "minimal_shift": 250,
                    "num_null_shifts": 50,
                    "null_seed": 11,
                    "fdr_mode": "cluster",
                },
                SignificanceConfig,
            )

        assert config.encoding.offset == 2
        assert config.minimal_shift == 250
        assert config.num_null_shifts == 50
        assert config.null_seed == 11
        assert config.fdr_mode == "cluster"

    def test_none_config_path_uses_defaults(self):
        config = load_config(None, {}, EncodingConfig)
        assert config.per_model_config == {"layer": 24}
        assert config.offset == 0
