"""Tests for list-valued downstream layer fanout."""


def test_significance_client_expands_layer_list_before_running(monkeypatch):
    import scripts.run_significance as client
    import src.utils.significance.significance_ieeg as implementation

    config = client.Config(
        ieeg={"encoding": {"model": "transformer",
                           "per_model_config": {"model_dir": "gpt2-xl",
                                                "layer": [0, 24, 48]}}}
    )
    seen_layers = []

    monkeypatch.setattr(
        implementation,
        "run_ieeg",
        lambda cfg: seen_layers.append(cfg.encoding.per_model_config["layer"]),
    )
    client.run_config(config)

    assert seen_layers == [0, 24, 48]
    assert all(not isinstance(layer, list) for layer in seen_layers)


def test_rrr_client_expands_layer_list_before_running(monkeypatch):
    import scripts.run_rrr as client
    import src.utils.rrr.rrr_ieeg as implementation

    config = client.Config(
        ieeg={"encoding": {"model": "transformer",
                           "per_model_config": {"model_dir": "gpt2-xl",
                                                "layer": [0, 24, 48]}}}
    )
    seen_layers = []

    monkeypatch.setattr(
        implementation,
        "run_ieeg",
        lambda cfg: seen_layers.append(cfg.encoding.per_model_config["layer"]),
    )
    client.run_config(config)

    assert seen_layers == [0, 24, 48]
    assert all(not isinstance(layer, list) for layer in seen_layers)
