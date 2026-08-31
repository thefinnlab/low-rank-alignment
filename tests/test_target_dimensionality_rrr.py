from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.run_target_dimensionality_rrr as client
from src.utils.rrr.significance import (
    PreparedSignificance,
    ieeg_target_dimension_summary,
)
from src.utils.shared.encoding import fit_transform_target_pca


def test_power_grid_includes_exact_endpoint():
    assert client._power_grid(1) == [1]
    assert client._power_grid(8) == [1, 2, 4, 8]
    assert client._power_grid(10) == [1, 2, 4, 8, 10]


def test_target_dim_override_validates_effective_cap():
    config = client.Config(target_pca_dims=[1, 3])
    assert client._target_dims(config, 3) == [1, 3]
    with pytest.raises(ValueError, match="full endpoint 4"):
        client._target_dims(config, 4)
    with pytest.raises(ValueError, match="exceed effective PCA cap 2"):
        client._target_dims(config, 2)


def test_client_rejects_non_internal_modes():
    config = client.Config(
        mode="cross_dataset", rrr_source={"dataset": "podcast"}
    )
    with pytest.raises(ValueError, match="only mode=internal"):
        config.validate()


def test_target_pca_is_fit_on_training_rows_only():
    train = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [2.0, 0.2, 0.0],
        [3.0, 0.3, 0.0],
    ])
    test_a = np.array([[4.0, 0.4, 0.0]])
    test_b = np.array([[4.0, 1000.0, -500.0]])

    train_a, projected_a = fit_transform_target_pca(train, test_a, 2)
    train_b, projected_b = fit_transform_target_pca(train, test_b, 2)

    np.testing.assert_allclose(train_a, train_b)
    assert not np.allclose(projected_a, projected_b)
    assert train_a.shape == (4, 2)


def test_target_pca_rejects_unestimable_dimension():
    with pytest.raises(ValueError, match="estimable rank 2"):
        fit_transform_target_pca(np.ones((2, 5)), np.ones((1, 5)), 3)


def test_ieeg_dimension_summary_uses_saved_significance_metadata():
    targets = {
        "1": pd.DataFrame({"significant": [True, False, True]}),
        "2": pd.DataFrame({"significant": [True, True]}),
    }
    encoding = {
        subject: {
            "lags": np.array([-1.0, 0.0, 1.0]),
            "selection": np.arange(10),
            "train_idxs": [np.arange(7), np.arange(8)],
        }
        for subject in ("1", "2")
    }
    significance = PreparedSignificance(
        targets, {}, encoding, reference=None, sources=[]
    )

    original, cap = ieeg_target_dimension_summary(significance, [1, 2])

    assert original == 12
    assert cap == 7


def test_artifact_target_dimension_selects_distinct_variants(tmp_path):
    first = client.Artifact(
        target_pca_dim=2, results_root=tmp_path
    ).begin(config={"target_pca_dim": 2})
    second = client.Artifact(
        target_pca_dim=4, results_root=tmp_path
    ).begin(config={"target_pca_dim": 4})

    assert first.identity["script"] == "run_target_dimensionality_rrr"
    assert first.variant == "target_pca_dim-2"
    assert second.variant == "target_pca_dim-4"


def test_run_config_derives_inner_ranks_and_compiles_sweep(tmp_path, monkeypatch):
    seen = []

    monkeypatch.setattr(client, "_inspect_ieeg_dimensions", lambda config: (5, 3, 2))

    def fake_run(config, target_pca_dim=None):
        seen.append((target_pca_dim, list(config.rank_grid), config.compute_sig_nonsig))
        artifact = client.Artifact(
            target_pca_dim=target_pca_dim,
            results_root=tmp_path,
        ).begin(config={"target_pca_dim": target_pca_dim})
        artifact.target_pca_dim = target_pca_dim
        artifact.save_mean_results(results=pd.DataFrame({
            "method": ["rrr"] * len(config.rank_grid),
            "rank": config.rank_grid,
            "corr": np.linspace(0.1, 0.2, len(config.rank_grid)),
            "r2": np.linspace(0.0, 0.1, len(config.rank_grid)),
        }))
        return artifact

    import src.utils.rrr.rrr_ieeg as implementation
    monkeypatch.setattr(implementation, "run_ieeg", fake_run)

    artifacts = client.run_config(client.Config(
        results_root=str(tmp_path), plot=False, target_pca_dims=[1, 3]
    ))

    assert seen == [(1, [1], False), (3, [1, 2], False)]
    assert len(artifacts) == 2
    owner = artifacts[-1]
    combined = owner.load_sweep_results()
    assert set(combined["target_pca_dim"]) == {1, 3}
    assert set(combined["original_target_count"]) == {5}
    assert set(combined["effective_target_pca_cap"]) == {3}
    plots = owner.load_collection("plots")
    assert len(plots) == 2
    assert owner.complete
