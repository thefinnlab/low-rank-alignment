from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.core.config import MaxvarGCCAConfig, MaxvarGCCAViewConfig
from src.utils.gcca import _calibration, resolve_gcca_output_dir, run_gcca
from src.utils.subspace_io import BaseComparisonData


def _config(tmp_path, *, refresh=False):
    return MaxvarGCCAConfig(
        anchor="anchor",
        views=[
            MaxvarGCCAViewConfig("anchor", "a.yml", "a", 2),
            MaxvarGCCAViewConfig("peer", "b.yml", "b", 4),
        ],
        shared_dimensions=[1, 3], ridge_alphas=[0.0, 0.01],
        output_dir=str(tmp_path / "gcca"), refresh=refresh,
    )


def test_gcca_config_requires_anchor_membership_and_union_capacity():
    views = [
        MaxvarGCCAViewConfig("a", "a.yml", "a", 2),
        MaxvarGCCAViewConfig("b", "b.yml", "b", 3),
    ]
    with pytest.raises(ValueError, match="anchor"):
        MaxvarGCCAConfig(anchor="missing", views=views)
    with pytest.raises(ValueError, match="union-rank"):
        MaxvarGCCAConfig(anchor="a", views=views, shared_dimensions=[6])


def test_calibration_reproduces_affine_target():
    rng = np.random.default_rng(2)
    estimate = rng.standard_normal((50, 3))
    target = estimate * np.array([2.0, -1.0, 0.5]) + np.array([1.0, 3.0, -2.0])
    result = _calibration(estimate, target)
    np.testing.assert_allclose(estimate * result["gain"] + result["intercept"], target)
    np.testing.assert_allclose(result["residual_variance"], 0.0, atol=1e-28)


def test_run_gcca_round_trips_downstream_artifacts(tmp_path, monkeypatch):
    import src.utils.gcca as module

    rng = np.random.default_rng(3)
    latent = rng.standard_normal((60, 4))
    x_anchor = latent @ rng.standard_normal((4, 6))
    x_peer = latent @ rng.standard_normal((4, 6))
    tests = [np.arange(0, 20), np.arange(20, 40), np.arange(40, 60)]
    trains = [np.setdiff1d(np.arange(60), test) for test in tests]
    base = BaseComparisonData(
        X_base=x_anchor, train_idxs=trains, test_idxs=tests,
        selection=np.arange(60), valid_rows_on_selection=None,
        dataset_key=("podcast", "podcast"), word_count=60,
    )
    encoding = SimpleNamespace(
        dataset="podcast", subjects=[1], per_dataset_config={}, data_root="data",
        model="transformer", per_model_config={"model_dir": "fake", "layer": 0}, offset=0,
    )
    resolved = {
        name: SimpleNamespace(
            name=name, kind="rrr", config_path=tmp_path / f"{name}.yml",
            config=SimpleNamespace(encoding=encoding), unit_dir=tmp_path / name,
            output_dir=tmp_path, sig_dir=tmp_path, subject_ids=[1],
            dataset_key=("podcast", "podcast"),
        ) for name in ("anchor", "peer")
    }
    monkeypatch.setattr(module, "resolve_subspace_run", lambda run: resolved[run.name])
    monkeypatch.setattr(module, "load_base_comparison_data", lambda run: base)
    monkeypatch.setattr(module, "load_peer_embeddings_on_base_rows", lambda *args: x_peer)
    monkeypatch.setattr(
        module, "load_fold_bases",
        lambda unit, rank: {fold: np.eye(6, rank) for fold in range(3)},
    )
    words = pd.DataFrame({
        "word_idx": np.arange(60), "word": [f"w{i}" for i in range(60)],
        "start": np.arange(60, dtype=float), "end": np.arange(60, dtype=float) + 0.5,
    })
    monkeypatch.setattr(module, "load_embeddings", lambda *args, **kwargs: (x_anchor, words))

    output = run_gcca(_config(tmp_path))
    expected = {
        "metadata.json", "summary.csv", "view_component_support.csv",
        "pairwise_component_correlations.csv", "inner_reliability.csv",
        "inner_component_alignment.csv", "component_stability.csv",
        "component_blocks.csv", "anchor_rows.csv", "folds.npz", "quicklook.png",
    }
    assert expected <= {path.name for path in output.iterdir()}
    rows = pd.read_csv(output / "anchor_rows.csv")
    assert rows.word.tolist()[:2] == ["w0", "w1"]
    reliability = pd.read_csv(output / "inner_reliability.csv")
    assert {"assignment_correlation", "validation_calibration_mse", "contribution_share"} <= set(reliability)
    with np.load(output / "fold_0/shared_dim_3/view_anchor.npz") as artifact:
        np.testing.assert_allclose(
            x_anchor @ artifact["ambient_map"] + artifact["intercept"],
            artifact["all_anchor_scores"], atol=1e-10,
        )
        np.testing.assert_allclose(
            artifact["all_anchor_scores"] * artifact["calibration_gain"]
            + artifact["calibration_intercept"],
            artifact["calibrated_all_anchor_scores"], atol=1e-10,
        )
    with pytest.raises(FileExistsError):
        run_gcca(_config(tmp_path))
    assert run_gcca(_config(tmp_path, refresh=True)) == output


def test_explicit_output_directory_wins(tmp_path):
    config = _config(tmp_path)
    assert resolve_gcca_output_dir(config, object()) == tmp_path / "gcca"
