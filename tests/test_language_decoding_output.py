import numpy as np
import pandas as pd
import pytest

from scripts.run_language_decoding import Artifact
from src.core.config import LanguageDecodingConfig
from src.utils.language_interpret.decoding import (
    _run_task_decoding, compute_prediction_agreement,
)
from src.utils.language_interpret.output import DecodingOutput, fit_selector


def test_direct_h5_fits_match_file_sink_and_resume(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3))
    targets = {
        "surprisal": X[:, 0] + X[:, 1],
        "entropy": X[:, 2],
        "content_word": (X[:, 0] > 0).astype(float),
    }
    config = LanguageDecodingConfig(rrr_run_path="unused", tasks=list(targets),
                                    rank_grid=[3], device="cpu", logistic_cs=[1.0])
    artifact = Artifact(results_root=tmp_path / "results", variant="test").begin(config=config)
    writer = DecodingOutput(tmp_path / "direct", artifact)
    args = dict(fold=0, rank=3, X=X, task_targets=targets,
                train_idx=np.arange(30), test_idx=np.arange(30, 40),
                config=config, A=np.eye(3), subspace="rrr")
    legacy = _run_task_decoding(output_dir=tmp_path / "files", **args)
    direct = _run_task_decoding(output_dir=writer.directory, writer=writer, **args)
    pd.testing.assert_frame_equal(pd.DataFrame(direct), pd.DataFrame(legacy))
    for task in targets:
        selector = fit_selector(task, "rrr", 3, 0)
        for field, filename in (("decoder_weights", "weights.npz"),
                                ("heldout_predictions", "heldout_predictions.npz")):
            loaded = artifact.load_field(field, selector=selector)
            with np.load(tmp_path / f"files/tasks/{task}/rrr/k3/fold_0/{filename}") as saved:
                assert set(loaded) == set(saved)
                for name in saved:
                    if saved[name].dtype.kind in "OUS":
                        np.testing.assert_array_equal(loaded[name], saved[name])
                    else:
                        np.testing.assert_allclose(loaded[name], saved[name], equal_nan=True)
    assert not writer.directory.exists()
    assert (artifact._variant_dir() / "artifact.h5").is_file()

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Completed HDF5 fits should be reused")
    monkeypatch.setattr("src.utils.language_interpret.decoding._fit_decoder", unexpected_fit)
    cached = _run_task_decoding(output_dir=writer.directory, writer=writer, **args)
    pd.testing.assert_frame_equal(pd.DataFrame(cached), pd.DataFrame(direct))

    # Identical saved PCA weights should produce perfectly matched decision scores.
    for task in targets:
        artifact.save_field("decoder_weights", writer.weights(task, "rrr", 3, 0),
                            selector=fit_selector(task, "pca", 3, 0))
    agreement = compute_prediction_agreement(
        output_dir=writer.directory, writer=writer,
        rrr_prepared={(0, 3): (X, np.eye(3))}, pca_prepared={(0, 3): (X, np.eye(3))},
        folds=[(args["train_idx"], args["test_idx"])], task_targets=targets, config=config,
    )
    np.testing.assert_allclose(agreement.prediction_correlation, 1)


def test_incomplete_h5_fit_is_not_cached(tmp_path):
    artifact = Artifact(results_root=tmp_path, variant="test").begin(config={})
    writer = DecodingOutput(tmp_path / "unused", artifact)
    selector = fit_selector("surprisal", "rrr", 1, 0)
    writer.save("fit_metrics", pd.DataFrame([{"task": "surprisal"}]), selector)
    writer.save("decoder_weights", {"weights": np.ones(1)}, selector)
    assert writer.cached_fit("surprisal", "rrr", 1, 0) is None


def test_compact_decoder_reads_existing_file_backed_artifacts(tmp_path):
    class FileArtifact(Artifact):
        COMPACT_STORAGE = False

    legacy = FileArtifact(results_root=tmp_path, variant="legacy").begin(config={})
    selector = fit_selector("surprisal", "rrr", 1, 0)
    legacy.save_field("decoder_weights", {"weights": np.ones((2, 1))}, selector=selector)
    compact = Artifact(results_root=tmp_path, variant="legacy")
    np.testing.assert_array_equal(compact.load_field("decoder_weights", selector=selector)["weights"], np.ones((2, 1)))
