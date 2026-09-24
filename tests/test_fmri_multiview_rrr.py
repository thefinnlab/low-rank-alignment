from types import SimpleNamespace

import numpy as np

from src.utils.rrr.fmri_rrr_mode.multi_view import (
    _view_delay_bands,
    load_projected_fmri_views,
    load_projected_fmri_word_views,
)
from src.utils.shared.fmri import AlignedFMRIStimuli, make_delayed


def test_view_delay_bands_reconstruct_delay_major_fir_design():
    pre_fir = np.arange(30, dtype=np.float32).reshape(6, 5)
    design = make_delayed(pre_fir, 3)

    bands, boundaries, labels = _view_delay_bands(design, [2, 3], 3)

    np.testing.assert_array_equal(np.concatenate(bands, axis=1), design)
    assert labels == [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    assert boundaries == [(0, 2), (2, 5), (5, 7), (7, 10), (10, 12), (12, 15)]


def test_projected_fmri_views_apply_each_saved_scaler_before_projection(monkeypatch):
    config = SimpleNamespace(
        views=[SimpleNamespace(name="a"), SimpleNamespace(name="b")],
        fmri=SimpleNamespace(
            embedding_root="embeddings", fmri_repo_root="fmri", trim=0,
            glove_data_root="data", limit_to_glove=False, glove_dim=300,
        ),
    )
    resolved = {
        "a": SimpleNamespace(config=SimpleNamespace(embedding_spec=SimpleNamespace(
            model_dir="a", layer=1, pooling="mean"
        ))),
        "b": SimpleNamespace(config=SimpleNamespace(embedding_spec=SimpleNamespace(
            model_dir="b", layer=2, pooling="mean"
        ))),
    }
    raw = {
        "a": {"story": np.array([[3.0, 6.0], [5.0, 10.0]], dtype=np.float32)},
        "b": {"story": np.array([[11.0], [13.0]], dtype=np.float32)},
    }
    monkeypatch.setattr(
        "src.utils.rrr.fmri_rrr_mode.multi_view.load_story_embeddings",
        lambda root, model, stories, *args, **kwargs: raw[model],
    )
    monkeypatch.setattr(
        "src.utils.rrr.fmri_rrr_mode.multi_view.align_fmri_stimuli",
        lambda root, stories, values, trim: AlignedFMRIStimuli(values),
    )
    factors = {
        "a": np.array([[1.0], [2.0]], dtype=np.float32),
        "b": np.array([[3.0]], dtype=np.float32),
    }
    scalers = {
        "a": {"mean": np.array([1.0, 2.0]), "scale": np.array([2.0, 4.0]),
              "coordinates": "standardized-v1"},
        "b": {"mean": np.array([7.0]), "scale": np.array([2.0]),
              "coordinates": "standardized-v1"},
    }

    words = load_projected_fmri_word_views(
        config, resolved, factors, scalers, ["story"]
    )
    aligned = load_projected_fmri_views(
        config, resolved, factors, scalers, ["story"]
    )

    expected = np.array([[3.0, 6.0], [6.0, 9.0]], dtype=np.float32)
    np.testing.assert_allclose(words["story"], expected)
    np.testing.assert_allclose(aligned.stories["story"], expected)
