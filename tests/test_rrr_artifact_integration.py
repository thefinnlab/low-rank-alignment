import numpy as np

from scripts.rrr.run_rrr import Artifact, UnitSelector, _run_rrr_for_unit
from src.core.config import RRRConfig
from src.utils.shared.encoding import build_per_fold_baselines


def test_numerical_rrr_writes_and_resumes_only_through_artifact(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((48, 8)).astype(np.float32)
    a = rng.standard_normal((8, 2)).astype(np.float32)
    b = rng.standard_normal((4, 2)).astype(np.float32)
    y = x @ a @ b.T + .01 * rng.standard_normal((48, 4)).astype(np.float32)
    test_idxs = [np.arange(24, 48)]
    baselines = build_per_fold_baselines(x, y, test_idxs)
    config = RRRConfig(rank_grid=[1, 2], plot=False, save_b_matrices=True)
    artifact = Artifact(results_root=tmp_path).begin(config=config)
    unit = UnitSelector(method="rrr")

    _run_rrr_for_unit(artifact, unit, x, test_idxs, baselines, config)

    assert artifact.has_field("input_factors", selector=artifact._selector(unit, fold=0))
    assert artifact.has_field("target_factors", selector=artifact._selector(unit, fold=0))
    assert set(artifact.load_summary(unit=unit)["k"]) == {1, 2}
    plan = artifact.resume_plan(unit=unit, ranks=[1, 2], folds=[0],
                                require_target_factors=True)
    assert plan["ranks_by_fold"] == {0: []}

    import scripts.rrr.run_rrr as runner
    monkeypatch.setattr(
        runner, "fit_dense_rrr",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected refit")),
    )
    _run_rrr_for_unit(artifact, unit, x, test_idxs, baselines, config)
