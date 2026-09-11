import numpy as np
import pandas as pd

from scripts.run_rrr import Artifact, UnitSelector
from src.utils.rrr.rrr_ieeg import _run_rrr_for_unit
from src.utils.rrr.ieeg_rrr_mode.common import _run_common_subspace_fold_k
from src.core.config import RRRConfig
from src.utils.shared.encoding import build_per_fold_baselines
from src.utils.plotting.encoding import plot_rrr_rank_results


def test_common_rank_evaluation_uses_raw_and_reduced_qr_designs(
    tmp_path, monkeypatch
):
    import src.utils.rrr.ieeg_rrr_mode.common as runner

    rng = np.random.default_rng(3)
    x = rng.standard_normal((20, 4)).astype(np.float32)
    y = rng.standard_normal((20, 3)).astype(np.float32)
    a = np.array(
        [[2.0, 1.0], [0.0, 3.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    captured = {}

    def evaluate(factor, x_train, x_test, y_train, y_test, *, subspace_name, **kwargs):
        del x_test, y_train, kwargs
        captured[subspace_name] = x_train @ factor
        n_targets = y_test.shape[1]
        return {
            f"{subspace_name}_corr": 0.0,
            f"{subspace_name}_r2": 0.0,
            f"{subspace_name}_corr_per_target": np.zeros(n_targets),
            f"{subspace_name}_r2_per_target": np.zeros(n_targets),
            f"{subspace_name}_best_alphas": np.ones(n_targets),
        }, np.array([1.0])

    monkeypatch.setattr(runner, "evaluate_projected_input_space", evaluate)
    artifact = Artifact(results_root=tmp_path).begin(config={"test": True})
    unit = UnitSelector(method="rrr")
    train, test = np.arange(10), np.arange(10, 20)

    _run_common_subspace_fold_k(
        a, 2, x[train], x[test], y[train], y[test], None, 0,
        np.array([1.0]), None, False, 2, None, None, None, "cpu",
        artifact, unit, "nr", evaluate_orthonormalized=True,
    )

    from src.utils.rrr.core import canonical_qr
    q = canonical_qr(a[:, :2])[0]
    np.testing.assert_allclose(captured["nr"], x[train] @ a[:, :2])
    np.testing.assert_allclose(captured["nr_orth"], x[train] @ q)
    metrics = artifact.load_rank_metrics(unit=unit, rank=2, fold=0)
    assert "nr_best_alphas" not in metrics
    assert "nr_orth_best_alphas" not in metrics
    assert not artifact.has_field(
        "diagnostics",
        selector=artifact._selector(unit, rank=2, fold=0, name="nr_orth"),
    )
    np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-6)


def test_common_rank_evaluation_can_run_only_reduced_qr_design(tmp_path, monkeypatch):
    import src.utils.rrr.ieeg_rrr_mode.common as runner

    rng = np.random.default_rng(13)
    x = rng.standard_normal((20, 4)).astype(np.float32)
    y = rng.standard_normal((20, 3)).astype(np.float32)
    a = rng.standard_normal((4, 2)).astype(np.float32)
    captured = {}

    def evaluate(factor, x_train, x_test, y_train, y_test, *, subspace_name, **kwargs):
        del x_test, y_train, kwargs
        captured[subspace_name] = x_train @ factor
        n_targets = y_test.shape[1]
        return {
            f"{subspace_name}_corr": 0.0,
            f"{subspace_name}_r2": 0.0,
            f"{subspace_name}_corr_per_target": np.zeros(n_targets),
            f"{subspace_name}_r2_per_target": np.zeros(n_targets),
            f"{subspace_name}_best_alphas": np.ones(n_targets),
        }, np.array([1.0])

    monkeypatch.setattr(runner, "evaluate_projected_input_space", evaluate)
    artifact = Artifact(results_root=tmp_path).begin(config={"test": True})
    unit = UnitSelector(method="rrr")

    _run_common_subspace_fold_k(
        a, 2, x[:10], x[10:], y[:10], y[10:], None, 0,
        np.array([1.0]), None, False, 2, None, None, None, "cpu",
        artifact, unit, "nr", orthogonal_only=True,
    )

    from src.utils.rrr.core import canonical_qr
    q = canonical_qr(a)[0]
    assert set(captured) == {"nr_orth"}
    np.testing.assert_allclose(captured["nr_orth"], x[:10] @ q)
    metrics = artifact.load_rank_metrics(unit=unit, rank=2, fold=0)
    assert "nr_corr" not in metrics
    assert "nr_orth_corr" in metrics


def test_numerical_rrr_writes_and_resumes_only_through_artifact(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((48, 8)).astype(np.float32)
    a = rng.standard_normal((8, 2)).astype(np.float32)
    b = rng.standard_normal((4, 2)).astype(np.float32)
    y = x @ a @ b.T + .01 * rng.standard_normal((48, 4)).astype(np.float32)
    test_idxs = [np.arange(24, 48)]
    baselines = build_per_fold_baselines(x, y, test_idxs)
    config = RRRConfig(
        rank_grid=[1, 2], plot=False, save_b_matrices=True, log_alphas=True
    )
    artifact = Artifact(results_root=tmp_path).begin(config=config)
    unit = UnitSelector(method="rrr")

    _run_rrr_for_unit(artifact, unit, x, test_idxs, baselines, config)

    assert artifact.has_field("input_factors", selector=artifact._selector(unit, fold=0))
    assert artifact.has_field("input_scalers", selector=artifact._selector(unit, fold=0))
    assert artifact.has_field("target_factors", selector=artifact._selector(unit, fold=0))
    assert set(artifact.load_summary(unit=unit)["rank"]) == {1, 2}
    rank_metrics = artifact.load_rank_metrics(unit=unit, rank=2, fold=0)
    assert {
        "nr_corr",
        "nr_r2_per_target",
        "nr_orth_corr",
        "nr_orth_r2_per_target",
        "nr_orth_best_alphas",
    }.issubset(rank_metrics)
    assert artifact.has_field(
        "diagnostics",
        selector=artifact._selector(unit, rank=2, fold=0, name="nr_orth"),
    )
    assert artifact.has_field(
        "diagnostics",
        selector=artifact._selector(unit, rank=2, fold=0, name="nr_orth"),
    )
    plan = artifact.resume_plan(unit=unit, ranks=[1, 2], folds=[0],
                                require_target_factors=True)
    assert plan["ranks_by_fold"] == {0: []}

    import src.utils.rrr.rrr_ieeg as runner
    monkeypatch.setattr(
        runner, "fit_dense_rrr",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected refit")),
    )
    _run_rrr_for_unit(artifact, unit, x, test_idxs, baselines, config)


def test_full_rank_qr_matches_standardized_full_ridge(tmp_path):
    rng = np.random.default_rng(23)
    n_features = 6
    x = rng.standard_normal((90, n_features)).astype(np.float32)
    x *= np.geomspace(0.05, 20.0, n_features).astype(np.float32)
    weights = rng.standard_normal((n_features, 8)).astype(np.float32)
    y = x @ weights + 0.2 * rng.standard_normal((90, 8)).astype(np.float32)
    test_idxs = [np.arange(60, 90)]
    alphas = np.array([0.01, 0.1, 1.0, 10.0])
    baselines = build_per_fold_baselines(x, y, test_idxs, alphas=alphas)
    config = RRRConfig(
        rank_grid=[n_features], encoding=RRRConfig().encoding,
        plot=False,
    )
    config.encoding.ridge_alphas = alphas.tolist()
    artifact = Artifact(results_root=tmp_path).begin(config=config)
    unit = UnitSelector(method="rrr")

    _run_rrr_for_unit(artifact, unit, x, test_idxs, baselines, config)

    metrics = artifact.load_rank_metrics(
        unit=unit, rank=n_features, fold=0
    )
    np.testing.assert_allclose(metrics["nr_orth_corr"], metrics["full_corr"],
                               rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(metrics["nr_orth_r2"], metrics["full_r2"],
                               rtol=2e-4, atol=2e-4)
    scaler = artifact.load_factors(unit=unit, fold=0)["input_scaler"]
    assert scaler["coordinates"] == "standardized-v1"
    np.testing.assert_allclose(
        scaler["mean"], x[:60].mean(axis=0, dtype=np.float64),
        rtol=1e-6, atol=1e-6,
    )


def test_raw_only_rank_metrics_resume_from_saved_factors_without_rrr_refit(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((36, 6)).astype(np.float32)
    y = rng.standard_normal((36, 3)).astype(np.float32)
    test_idxs = [np.arange(18, 36)]
    baselines = build_per_fold_baselines(x, y, test_idxs)
    config = RRRConfig(rank_grid=[1], plot=False)
    artifact = Artifact(results_root=tmp_path).begin(config=config)
    unit = UnitSelector(method="rrr")

    _run_rrr_for_unit(artifact, unit, x, test_idxs, baselines, config)
    saved = artifact.load_rank_metrics(unit=unit, rank=1, fold=0)
    raw_only = {key: value for key, value in saved.items() if "_orth" not in key}
    artifact.save_rank_metrics(
        unit=unit, rank=1, fold=0, metrics=raw_only, overwrite=True
    )

    import src.utils.rrr.ieeg_rrr_mode.common as runner
    monkeypatch.setattr(
        runner,
        "fit_dense_rrr",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected refit")),
    )
    _run_rrr_for_unit(artifact, unit, x, test_idxs, baselines, config)

    rebuilt = artifact.load_rank_metrics(unit=unit, rank=1, fold=0)
    assert "nr_orth_corr" in rebuilt
    assert "nr_orth_r2_per_target" in rebuilt


def test_rank_plot_overlays_raw_and_qr_mean_and_fold_lines():
    summary = pd.DataFrame({
        "k": [1, 2],
        "nr_corr": [0.1, 0.2],
        "nr_orth_corr": [0.15, 0.25],
        "nr_r2": [0.01, 0.04],
        "nr_orth_r2": [0.02, 0.06],
    })
    grid = pd.DataFrame({
        "fold": [0, 0, 1, 1],
        "k": [1, 2, 1, 2],
        "nr_corr": [0.1, 0.2, 0.11, 0.21],
        "nr_orth_corr": [0.15, 0.25, 0.16, 0.26],
        "nr_r2": [0.01, 0.04, 0.02, 0.05],
        "nr_orth_r2": [0.02, 0.06, 0.03, 0.07],
    })

    figure = plot_rrr_rank_results(summary, grid_df=grid, primary_label="nr")
    for axis_index in (0, 1, 3, 4):
        labels = {line.get_label() for line in figure.axes[axis_index].lines}
        assert "nr (raw A)" in labels
        assert "nr (QR A)" in labels
