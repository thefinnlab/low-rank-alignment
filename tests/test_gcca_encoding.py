import numpy as np
import pandas as pd

from src.core.config import GCCAEncodingConfig
from src.utils.gcca_encoding import (
    baseline_fingerprint, component_weights, deduplicate_embedding_runs,
    reconstruct_view_scores, union_significant_targets,
)


def test_default_config_is_calibrated_contribution():
    config = GCCAEncodingConfig(gcca_dir="gcca", run_name="weighted")
    assert config.averaging_method == "contribution"
    assert config.calibrated_scores is True


def test_contribution_weights_include_every_view_and_fallback_equally():
    shares = np.array([[0.7, np.nan], [0.2, np.nan], [0.1, np.nan]])
    weights = component_weights("contribution", shares)
    np.testing.assert_allclose(weights[:, 0], shares[:, 0])
    np.testing.assert_allclose(weights[:, 1], 1 / 3)
    np.testing.assert_allclose(weights.sum(0), 1)
    assert np.all(weights > 0)


def test_reconstruction_round_trip_with_calibration():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 5)); basis = np.linalg.qr(rng.normal(size=(5, 3)))[0]
    mean = rng.normal(size=3); scale = np.exp(rng.normal(size=3)); rotation = rng.normal(size=(3, 2))
    raw = ((x @ basis - mean) / scale) @ rotation
    ambient = basis @ (rotation / scale[:, None]); intercept = -mean @ (rotation / scale[:, None])
    artifact = {"basis": basis, "projected_mean": mean, "projected_scale": scale,
                "rotation": rotation, "ambient_map": ambient, "intercept": intercept,
                "all_anchor_scores": raw, "calibration_gain": np.array([2., -1.]),
                "calibration_intercept": np.array([1., 3.]),
                "calibrated_all_anchor_scores": raw * [2., -1.] + [1., 3.]}
    np.testing.assert_allclose(reconstruct_view_scores(artifact, x), artifact["calibrated_all_anchor_scores"])


def test_fingerprint_ignores_averaging_but_tracks_evaluation_inputs():
    kwargs = dict(target_manifest=[{"subject": 1, "electrode": "A"}], selection=[0, 1],
                  folds=[[0], [1]], embeddings={"id": {"model": "m"}},
                  evaluation_policy={"alphas": [1.]}, source_paths=["."])
    first = baseline_fingerprint(**kwargs)
    assert first == baseline_fingerprint(**kwargs)
    assert first != baseline_fingerprint(**{**kwargs, "folds": [[1], [0]]})
    # There is intentionally no averaging-method parameter in this API.


def test_union_excludes_cross_dataset_and_preserves_numeric_lag_order():
    frames = [pd.DataFrame({"dataset": ["podcast", "movie", "podcast"],
                            "subject": [1, 1, 1], "electrode": ["A", "B", "A"],
                            "lag": [2, 0, 1]})]
    result = union_significant_targets(frames, anchor_dataset="podcast")
    assert result[["electrode", "lag"]].values.tolist() == [["A", 1], ["A", 2]]


def test_embedding_identity_deduplicates_view_aliases():
    run = {"model": "transformer", "offset": 0,
           "per_model_config": {"model_dir": "m", "layer": 4, "pooling": "last"}}
    unique, mapping = deduplicate_embedding_runs({"a": run, "alias": dict(run)})
    assert len(unique) == 1
    assert mapping["a"] == mapping["alias"]
