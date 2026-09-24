"""Focused behavior tests for multiview RRR helpers."""

import numpy as np

from src.utils.rrr.multiview import (
    fit_banded_ridge_baseline,
    load_view_bases,
    project_view_bands,
)


def test_project_view_bands_preserves_raw_A_and_config_order():
    embeddings = {
        "second": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "first": np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32),
    }
    raw_a_first = np.array([[2.0], [0.0]], dtype=np.float32)
    raw_a_second = np.array([[0.0], [3.0]], dtype=np.float32)
    bases = {"first": {0: raw_a_first}, "second": {0: raw_a_second}}

    bands, boundaries = project_view_bands(
        embeddings, bases, ["first", "second"], fold=0
    )

    np.testing.assert_array_equal(bands[0], embeddings["first"] @ raw_a_first)
    np.testing.assert_array_equal(bands[1], embeddings["second"] @ raw_a_second)
    assert boundaries == [(0, 1), (1, 2)]


def test_matching_view_uses_fold_scaler_and_rank_specific_qr():
    from types import SimpleNamespace

    factor = np.array([[3.0, 1.0, 8.0], [0.0, 2.0, 9.0], [0.0, 0.0, 4.0]])
    scaler = {
        "coordinates": "standardized-v1",
        "mean": np.array([1.0, 2.0, 3.0]),
        "scale": np.array([2.0, 3.0, 4.0]),
    }

    class Source:
        mode = "internal"

        def load_collection(self, field, selector):
            return {(('fold', 0),): factor}

        def load_input_scaler(self, selector):
            return scaler

    view = SimpleNamespace(name="view", rank=2)
    config = SimpleNamespace(views=[view])
    resolved = {"view": SimpleNamespace(
        artifact=Source(), dataset_key=("podcast", "podcast")
    )}
    bases, modes, scalers = load_view_bases(
        config, resolved, ("podcast", "podcast"), np.arange(3),
        [np.array([0])], return_scalers=True,
    )
    embeddings = {"view": np.array([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])}
    bands, _ = project_view_bands(embeddings, bases, ["view"], 0, scalers)
    from src.utils.rrr.core import canonical_qr
    q = canonical_qr(factor[:, :2])[0]
    np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(
        bands[0], ((embeddings["view"] - scaler["mean"]) / scaler["scale"]) @ q
    )
    assert modes == {"view": "matching_target_fold_standardized_qr"}


def test_banded_ridge_recovers_concatenated_primal_weights():
    rng = np.random.default_rng(3)
    bands = [
        rng.normal(size=(48, 3)).astype(np.float32),
        rng.normal(size=(48, 2)).astype(np.float32),
    ]
    true_weights = [rng.normal(size=(3, 4)), rng.normal(size=(2, 4))]
    targets = (bands[0] @ true_weights[0] + bands[1] @ true_weights[1]).astype(
        np.float32
    )

    fitted = fit_banded_ridge_baseline(
        [band[:36] for band in bands],
        [band[36:] for band in bands],
        targets[:36],
        alphas=np.logspace(-2, 2, 5),
        cv=[
            (np.r_[0:12, 24:36], np.r_[12:24]),
            (np.r_[0:24], np.r_[24:36]),
        ],
        solver_params={"n_iter": 5, "random_state": 7, "progress_bar": False},
        force_cpu=True,
    )

    assert fitted.beta_full_raw.shape == (5, 4)
    assert fitted.deltas.shape == (2, 4)
    np.testing.assert_allclose(
        np.concatenate([band[:36] for band in bands], axis=1)
        @ fitted.beta_full_raw
        + fitted.intercept,
        fitted.Y_hat_train,
        rtol=2e-4,
        atol=2e-4,
    )


def test_load_view_bases_uses_fmri_cross_dataset_helpers(monkeypatch):
    class FakeArtifact:
        dataset = "deep-fmri"
        mode = "internal"

    source = FakeArtifact()
    calls = []

    def load_factors(artifact, **kwargs):
        calls.append(("factor", artifact, kwargs))
        return {2: np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)}

    def load_scaler(artifact, kind, rank):
        calls.append(("scaler", artifact, kind, rank))
        return {
            "coordinates": "standardized-v1",
            "mean": np.zeros(2),
            "scale": np.array([2.0, 4.0]),
        }

    monkeypatch.setattr("src.utils.rrr.multiview.load_cross_dataset_A_by_rank", load_factors)
    monkeypatch.setattr("src.utils.rrr.multiview.load_cross_dataset_scaler", load_scaler)

    from types import SimpleNamespace

    view = SimpleNamespace(name="fmri", rank=2)
    config = SimpleNamespace(views=[view], subspace_aggregation="extrinsic")
    resolved = {
        "fmri": SimpleNamespace(
            artifact=source, dataset_key=("deep-fmri", "deep-fmri")
        )
    }

    bases, modes = load_view_bases(
        config, resolved, ("podcast", "podcast"), np.arange(3), [np.array([0])]
    )

    np.testing.assert_allclose(bases["fmri"][0], [[1.0, 2.0], [1.5, 2.0]])
    assert modes == {"fmri": "cross_dataset_fmri_standardized_qr"}
    assert calls == [
        ("factor", source, {
            "kind": "rrr", "ranks": [2], "source_format": "rank_bases",
            "orthonormalize": True,
        }),
        ("scaler", source, "rrr", 2),
    ]


def test_load_view_bases_uses_full_fit_for_cross_dataset_ieeg(monkeypatch):
    from types import SimpleNamespace

    source = SimpleNamespace(dataset="brain-treebank-cars-2", mode="full_fit")
    view = SimpleNamespace(name="cars", rank=2)
    config = SimpleNamespace(views=[view])
    calls = []

    def load_factors(artifact, **kwargs):
        calls.append(("factor", artifact, kwargs))
        return {2: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}

    def load_scaler(artifact, kind, rank):
        calls.append(("scaler", artifact, kind, rank))
        return {
            "coordinates": "standardized-v1",
            "mean": np.zeros(2),
            "scale": np.array([2.0, 4.0]),
        }

    monkeypatch.setattr("src.utils.rrr.multiview.load_cross_dataset_A_by_rank", load_factors)
    monkeypatch.setattr("src.utils.rrr.multiview.load_cross_dataset_scaler", load_scaler)

    bases, modes = load_view_bases(
        config,
        {"cars": SimpleNamespace(
            artifact=source,
            dataset_key=("brain-treebank-cars-2", "cars-2"),
        )},
        ("podcast", "podcast"),
        np.arange(3),
        [np.array([0]), np.array([1])],
    )

    expected = np.array([[0.5, 0.0], [0.0, 0.25]])
    np.testing.assert_allclose(bases["cars"][0], expected)
    np.testing.assert_allclose(bases["cars"][1], expected)
    assert modes == {"cars": "cross_dataset_full_fit_standardized_qr"}
    assert calls[0][2]["source_format"] == "rank_bases"
    assert calls[0][2]["orthonormalize"] is True


def test_load_view_bases_rejects_cross_dataset_ieeg_fold_artifact():
    from types import SimpleNamespace

    view = SimpleNamespace(name="cars", rank=2)
    config = SimpleNamespace(views=[view])
    source = SimpleNamespace(dataset="brain-treebank-cars-2", mode="internal")

    with np.testing.assert_raises_regex(ValueError, "must use a full_fit"):
        load_view_bases(
            config,
            {"cars": SimpleNamespace(
                artifact=source,
                dataset_key=("brain-treebank-cars-2", "cars-2"),
            )},
            ("podcast", "podcast"),
            np.arange(3),
            [np.array([0])],
        )
