import numpy as np
import pandas as pd
import h5py
from types import SimpleNamespace

from src.utils import fmri


def test_resolve_fmri_stories_uses_sessions_and_shared_test(tmp_path):
    em_data = tmp_path / "em_data"
    em_data.mkdir()
    (em_data / "sess_to_story.json").write_text(
        '{"1": [["a", "b"], "test"], "2": [["c"], "test"]}'
    )
    train, test, groups = fmri.resolve_fmri_stories(
        str(tmp_path), [1, 2]
    )
    assert train == ["a", "b", "c"]
    assert len(train) == len(set(train))
    assert test == ["test"]
    assert set(train).isdisjoint(test)
    assert [len(groups[index]) for index in range(1, 3)] == [2, 1]


def test_resolve_fmri_stories_applies_explicit_temporary_holdouts(tmp_path):
    em_data = tmp_path / "em_data"
    em_data.mkdir()
    (em_data / "sess_to_story.json").write_text(
        '{"1": [["kept", "life"], "test"], '
        '"2": [["also_kept", "fromboyhoodtofatherhood"], "test"]}'
    )

    train, _, groups = fmri.resolve_fmri_stories(str(tmp_path), [1, 2])

    assert train == ["kept", "also_kept"]
    assert groups == {1: ["kept"], 2: ["also_kept"]}


def test_make_delayed_matches_positive_zero_padded_delays():
    values = np.arange(1, 5, dtype=float)[:, None]
    result = fmri.make_delayed(values, 2)
    np.testing.assert_array_equal(
        result,
        [[0, 0], [1, 0], [2, 1], [3, 2]],
    )


def test_finite_response_voxel_masks_require_complete_stories(tmp_path):
    base = (
        tmp_path
        / "data/ds003020/derivatives/preprocessed_data/UTS01"
    )
    base.mkdir(parents=True)
    first = np.arange(12, dtype=float).reshape(3, 4)
    second = first.copy()
    first[:, 1] = np.nan
    second[1, 3] = np.inf
    for story, values in (("first", first), ("second", second)):
        with h5py.File(base / f"{story}.hf5", "w") as handle:
            handle.create_dataset("data", data=values)

    masks, diagnostics = fmri.finite_response_voxel_masks(
        str(tmp_path), ["UTS01"], ["first", "second"]
    )

    np.testing.assert_array_equal(masks["UTS01"], [True, False, True, False])
    assert diagnostics == {"UTS01": {"first": 1, "second": 1}}


def test_prepare_uses_fixed_leading_five_plus_symmetric_trim(monkeypatch):
    frame = pd.DataFrame(
        {"word_idx": np.arange(20), "word": ["x"] * 20, "start": np.arange(20), "end": np.arange(20) + 0.5}
    )
    sequence = type(
        "Sequence", (), {"data_times": np.arange(20), "tr_times": np.arange(20)}
    )()

    def fake_modules(_):
        return (lambda values, old, new, window: values), None, None

    monkeypatch.setattr(fmri, "_toolbox_modules", fake_modules)
    monkeypatch.setattr(
        fmri, "load_story_word_data", lambda root, stories: ({"story": frame}, {"story": sequence})
    )
    embeddings = {"story": np.arange(40, dtype=float).reshape(20, 2)}
    prepared = fmri.prepare_fmri_data(
        "unused", ["story"], embeddings, trim=5, ndelays=1
    )
    assert prepared.stimulus.shape == (5, 2)
    assert prepared.pre_fir_stimulus.shape == (5, 2)
    assert prepared.story_slices == {"story": slice(0, 5)}


def test_prepare_zero_pads_delays_at_each_story_boundary(monkeypatch):
    frames = {
        story: pd.DataFrame(
            {
                "word_idx": np.arange(8),
                "word": ["x"] * 8,
                "start": np.arange(8),
                "end": np.arange(8) + 0.5,
            }
        )
        for story in ("first", "second")
    }
    sequences = {
        story: type(
            "Sequence", (), {"data_times": np.arange(8), "tr_times": np.arange(8)}
        )()
        for story in frames
    }

    monkeypatch.setattr(
        fmri,
        "_toolbox_modules",
        lambda _: (lambda values, old, new, window: values, None, None),
    )
    monkeypatch.setattr(
        fmri, "load_story_word_data", lambda root, stories: (frames, sequences)
    )
    prepared = fmri.prepare_fmri_data(
        "unused",
        ["first", "second"],
        {
            "first": np.arange(8, dtype=float)[:, None],
            "second": np.arange(10, 18, dtype=float)[:, None],
        },
        trim=0,
        ndelays=1,
    )

    assert prepared.story_slices == {"first": slice(0, 3), "second": slice(3, 6)}
    np.testing.assert_array_equal(prepared.stimulus[[0, 3]], 0)


def test_session_cv_splits_hold_out_whole_story_groups():
    slices = {"a": slice(0, 3), "b": slice(3, 5), "c": slice(5, 9)}
    folds = fmri.session_cv_splits(slices, {1: ["a", "b"], 2: ["c"]})
    np.testing.assert_array_equal(folds[0][1], np.arange(5))
    np.testing.assert_array_equal(folds[1][1], np.arange(5, 9))
    assert not np.intersect1d(*folds[0]).size


def test_circular_shift_null_is_deterministic_and_respects_story_bounds():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(60, 3))
    predicted = true + rng.normal(scale=0.1, size=true.shape)
    slices = {"a": slice(0, 30), "b": slice(30, 60)}
    first, first_meta = fmri.circular_shift_pvalues(
        true, predicted, slices, num_shifts=20, minimal_shift=5, seed=7, batch_size=4
    )
    second, second_meta = fmri.circular_shift_pvalues(
        true, predicted, slices, num_shifts=20, minimal_shift=5, seed=7, batch_size=6
    )
    np.testing.assert_array_equal(first, second)
    assert first_meta["representative_shifts"] == second_meta["representative_shifts"]
    for draw in first_meta["representative_shifts"]:
        assert all(5 <= shift <= 25 for shift in draw.values())
    assert np.all(first >= 1 / 21)


def test_bh_fdr_and_target_manifest_are_subject_scoped():
    rejected, adjusted = fmri.bh_fdr(np.array([0.001, 0.2, 0.01]), 0.05)
    assert rejected.tolist() == [True, False, True]
    assert np.all((0 <= adjusted) & (adjusted <= 1))
    manifest = fmri.target_manifest(
        ["UTS01", "UTS02"],
        {"UTS01": np.array([True, False]), "UTS02": np.array([False, True, True])},
    )
    assert manifest.to_dict("records") == [
        {"subject": "UTS01", "voxel_index": 0},
        {"subject": "UTS02", "voxel_index": 1},
        {"subject": "UTS02", "voxel_index": 2},
    ]


def test_pca_sibling_resolution():
    assert fmri.resolve_pca_sibling("results/model/rrr") == "results/model/pca"
    assert (
        fmri.resolve_pca_sibling("results/model/cross_dataset_rrr/source")
        == "results/model/cross_dataset_pca/source"
    )


def test_within_fmri_bases_save_fir_blocks_rank_bases_and_pre_fir_pca(
    tmp_path, monkeypatch
):
    from scripts.fmri import run_cross_dataset_rrr as runner

    pre_fir = np.column_stack(
        [np.arange(8, dtype=float), np.zeros(8), np.zeros(8)]
    ).astype(np.float32)
    prepared = fmri.PreparedFMRI(
        stimulus=np.zeros((8, 6), dtype=np.float32),
        pre_fir_stimulus=pre_fir,
        responses=np.zeros((8, 3), dtype=np.float32),
        story_slices={"story": slice(0, 8)},
    )
    fir_a = np.array([[10.0], [0.0], [0.0], [0.0], [1.0], [0.0]], dtype=np.float32)
    fit_kwargs = {}

    def fake_fit_dense_rrr(*args, **kwargs):
        fit_kwargs.update(kwargs)
        return {"A_dense": fir_a}

    monkeypatch.setattr(
        runner,
        "fit_dense_rrr",
        fake_fit_dense_rrr,
    )
    config = SimpleNamespace(
        rank_grid=[1],
        subspace_aggregation="unnormalized",
        rrr_svd_on_cpu=True,
        rrr_use_low_rank_svd=True,
        rrr_low_rank_svd_seed=17,
        fmri=SimpleNamespace(ndelays=2),
    )
    ridge_fit = runner.FinalRidgeFit(
        test_prediction=np.zeros((2, 3), dtype=np.float32),
        train_prediction=np.zeros((8, 3), dtype=np.float32),
        coefficients=np.zeros((6, 3), dtype=np.float32),
        best_alphas=np.ones(3),
    )

    bases, metadata = runner._fit_within_fmri_bases(
        prepared, ridge_fit, ambient_dim=3, config=config, output_dir=tmp_path
    )

    np.testing.assert_array_equal(np.load(metadata["rrr"]), fir_a)
    assert len(metadata["rrr_delay_blocks"]) == 2
    assert abs(float(bases["rrr"][1][:, 0] @ np.array([1.0, 0.0, 0.0]))) > 0.999
    assert abs(float(bases["pca"][1][:, 0] @ np.array([1.0, 0.0, 0.0]))) > 0.999
    assert (tmp_path / "rrr/k1/A.npy").exists()
    assert (tmp_path / "pca/k1/A.npy").exists()
    assert fit_kwargs["svd_on_cpu"] is True
    assert fit_kwargs["use_low_rank_svd"] is True
    assert fit_kwargs["low_rank_svd_seed"] == 17
