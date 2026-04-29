"""Tests for list-valued downstream layer fanout."""

from argparse import Namespace

from src.core.config import EncodingConfig, RRRConfig, SignificanceConfig


def test_significance_main_expands_layer_list_before_running(monkeypatch):
    import scripts.sig_lags.run_significance as run_significance

    config = SignificanceConfig(
        encoding=EncodingConfig(
            model="transformer",
            per_model_config={"model_dir": "gpt2-xl", "layer": [0, 24, 48]},
        )
    )
    seen_layers = []

    monkeypatch.setattr(
        run_significance, "parse_args", lambda _description: (Namespace(config=None), {})
    )
    monkeypatch.setattr(
        run_significance,
        "load_config",
        lambda _config_path, _overrides, _config_cls: config,
    )
    monkeypatch.setattr(
        run_significance,
        "run_config",
        lambda cfg: seen_layers.append(cfg.encoding.per_model_config["layer"]),
    )

    run_significance.main()

    assert seen_layers == [0, 24, 48]
    assert all(not isinstance(layer, list) for layer in seen_layers)


def test_rrr_main_expands_layer_list_before_running(monkeypatch):
    import scripts.rrr.run_rrr as run_rrr

    config = RRRConfig(
        encoding=EncodingConfig(
            model="transformer",
            per_model_config={"model_dir": "gpt2-xl", "layer": [0, 24, 48]},
        )
    )
    seen_layers = []

    monkeypatch.setattr(
        run_rrr, "parse_args", lambda _description: (Namespace(config=None), {})
    )
    monkeypatch.setattr(
        run_rrr,
        "load_config",
        lambda _config_path, _overrides, _config_cls: config,
    )
    monkeypatch.setattr(
        run_rrr,
        "run_config",
        lambda cfg: seen_layers.append(cfg.encoding.per_model_config["layer"]),
    )

    run_rrr.main()

    assert seen_layers == [0, 24, 48]
    assert all(not isinstance(layer, list) for layer in seen_layers)
