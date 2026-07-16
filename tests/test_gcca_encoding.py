import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.core.config import GCCAEncodingConfig, RRRConfig
from src.utils.gcca_encoding import (
    component_weights,
    get_gcca_embeddings,
    get_rrr_embeds,
    transform_rrr_to_gcca,
)
from src.utils.decomposition import DecompositionInputs
from src.utils.gcca_encoding_runner import _comparison_for_run


def _artifact(x, basis, rotation, mean, scale, gain, intercept, contribution):
    rrr = x @ basis
    raw = ((rrr - mean) / scale) @ rotation
    calibrated = raw * gain + intercept
    return {
        "basis": basis,
        "projected_mean": mean,
        "projected_scale": scale,
        "rotation": rotation,
        "all_anchor_scores": raw,
        "calibration_gain": gain,
        "calibration_intercept": intercept,
        "calibrated_all_anchor_scores": calibrated,
        "contribution_share": contribution,
    }


def test_default_config_uses_contribution_weighting():
    config = GCCAEncodingConfig(gcca_dir="gcca", run_name="weighted")
    assert config.averaging_method == "contribution"
    assert config.standardize_inputs == RRRConfig().standardize_subspace_inputs


def test_rrr_and_gcca_transform_reproduce_saved_scores():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 5))
    basis = np.linalg.qr(rng.normal(size=(5, 3)))[0]
    artifact = _artifact(
        x,
        basis,
        rng.normal(size=(3, 2)),
        rng.normal(size=3),
        np.exp(rng.normal(size=3)),
        np.array([2.0, -1.0]),
        np.array([1.0, 3.0]),
        np.array([0.7, 0.4]),
    )
    rrr = get_rrr_embeds(x, artifact)
    np.testing.assert_allclose(rrr, x @ basis)
    np.testing.assert_allclose(
        transform_rrr_to_gcca(rrr, artifact), artifact["calibrated_all_anchor_scores"]
    )


def test_contribution_weights_fall_back_to_equal_for_invalid_component():
    shares = np.array([[0.7, np.nan], [0.2, np.nan], [0.1, np.nan]])
    weights = component_weights("contribution", shares)
    np.testing.assert_allclose(weights[:, 0], shares[:, 0])
    np.testing.assert_allclose(weights[:, 1], 1 / 3)
    np.testing.assert_allclose(weights.sum(0), 1)


def test_get_gcca_embeddings_calibrates_then_weights_views():
    rng = np.random.default_rng(9)
    x1 = rng.normal(size=(12, 4))
    x2 = rng.normal(size=(12, 4))
    basis = np.eye(4, 2)
    rotation = np.eye(2)
    first = _artifact(
        x1, basis, rotation, np.zeros(2), np.ones(2), np.ones(2),
        np.zeros(2), np.array([0.75, 0.25]),
    )
    second = _artifact(
        x2, basis, rotation, np.zeros(2), np.ones(2), np.ones(2),
        np.zeros(2), np.array([0.25, 0.75]),
    )
    combined, weights = get_gcca_embeddings(
        {"first": first, "second": second},
        {"first": x1, "second": x2},
    )
    expected = np.stack(
        [first["calibrated_all_anchor_scores"], second["calibrated_all_anchor_scores"]]
    )
    np.testing.assert_allclose(combined, np.sum(expected * weights[:, None, :], axis=0))


def test_rrr_comparison_subsamples_gcca_targets_by_identity(tmp_path, monkeypatch):
    manifest = pd.DataFrame(
        {
            "subject": [1, 1],
            "electrode": ["B", "A"],
            "lag_index": [0, 0],
            "lag": [0.0, 0.0],
        }
    )
    prepared = DecompositionInputs(
        embeddings={},
        targets=np.zeros((4, 2)),
        test_idxs=[np.array([2, 3])],
        selection=np.arange(4),
        target_manifest=manifest,
    )
    gcca_fold = tmp_path / "k1" / "fold_0"
    gcca_fold.mkdir(parents=True)
    np.savez(gcca_fold / "metrics.npz", gcca_corr_per_target=np.array([0.1, 0.8]))
    rrr_dir = tmp_path / "rrr"
    for rank, corr in [(1, 0.2), (2, 0.3)]:
        fold = rrr_dir / f"k{rank}" / "fold_0"
        fold.mkdir(parents=True)
        np.savez(fold / "metrics.npz", nr_corr_per_target=np.array([corr]))
    run_manifest = manifest.iloc[[1]].reset_index(drop=True)
    monkeypatch.setattr(
        "src.utils.gcca_encoding_runner._run_target_manifest",
        lambda run: run_manifest,
    )
    curves, _, deltas = _comparison_for_run(
        tmp_path,
        prepared,
        "model_a",
        SimpleNamespace(unit_dir=rrr_dir),
        [1],
    )
    gcca_corr = curves[(curves.source == "GCCA") & (curves.k == 1)]["corr"].item()
    assert np.isclose(gcca_corr, 0.8)
    assert set(curves[curves.source == "RRR NR"].k) == {1, 2}
    assert deltas.k.tolist() == [1]
