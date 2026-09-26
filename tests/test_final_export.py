"""Validate batch example selection before any figures are written."""

import pytest

from src.utils.plotting import final_data, final_export


def _data():
    quantiles = {(dataset, model, layer, method): None
                 for dataset in final_data.DATASETS
                 for model, layer in (("gpt2", 0), ("glove", None))
                 for method in ("rrr", "pca")}
    loso_quantiles = {(dataset, model, layer, subject, method): None
                      for dataset in final_data.DATASETS
                      for model, layer in (("gpt2", 0), ("glove", None))
                      for subject in (1, 2)
                      for method in ("RRR baseline", "RRR transfer", "PCA baseline")}
    return (None, None, None, quantiles, loso_quantiles, [], {}, None, [])


@pytest.mark.parametrize("arguments,internal,loso", [
    ([], ("gpt2", 0), ("gpt2", 0)),
    (["--example-model", "glove", "--example-layer", "none"],
     ("glove", None), ("glove", None)),
    (["--loso-example-model", "glove", "--loso-example-layer", "none"],
     ("gpt2", 0), ("glove", None)),
])
def test_example_selection(monkeypatch, tmp_path, arguments, internal, loso):
    data = _data()
    monkeypatch.setattr(final_data, "load_final_data", lambda root: data)
    calls = []
    monkeypatch.setattr(final_export, "export_figures", lambda *args: calls.append(args))
    final_export.main(["--output-dir", str(tmp_path), *arguments])
    assert calls == [(data, tmp_path, ["pdf", "png"], internal, loso)]


@pytest.mark.parametrize("index,key", [
    (3, ("fmri", "gpt2", 0, "pca")),
    (4, ("fmri", "gpt2", 0, 1, "RRR transfer")),
])
def test_incomplete_example_rejected_before_export(monkeypatch, tmp_path, capsys, index, key):
    data = _data()
    del data[index][key]
    if index == 4:
        # LOSO eligibility requires at least one subject for each method.
        del data[index][("fmri", "gpt2", 0, 2, "RRR transfer")]
    monkeypatch.setattr(final_data, "load_final_data", lambda root: data)
    output = tmp_path / "figures"
    with pytest.raises(SystemExit) as error:
        final_export.main(["--output-dir", str(output)])
    assert error.value.code == 2
    assert "Available model/layer pairs: glove/none" in capsys.readouterr().err
    assert not output.exists()
