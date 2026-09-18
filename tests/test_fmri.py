import numpy as np
import pandas as pd
import h5py
from types import SimpleNamespace

from src.utils.shared import fmri
from src.utils.significance import fmri as fmri_significance


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


def test_load_story_embeddings_pools_words_from_hdf5(tmp_path):
    story_dir = tmp_path / "model" / "story"
    story_dir.mkdir(parents=True)
    pd.DataFrame({"word_idx": [0, 0, 1]}).to_csv(
        story_dir / "transcript.tsv", sep="\t", index=False
    )
    token_values = np.array([[1.0, 2.0], [3.0, 6.0], [8.0, 10.0]])
    with h5py.File(story_dir / "features.hdf5", "w") as handle:
        handle.create_dataset("layer-2", data=token_values)

    result = fmri.load_story_embeddings(
        str(tmp_path), "model", ["story"], layer=2, pooling="mean"
    )

    np.testing.assert_allclose(result["story"], [[2.0, 4.0], [8.0, 10.0]])


def test_load_story_embeddings_supports_glove(monkeypatch):
    frame = pd.DataFrame({
        "word_idx": [0, 1], "word": ["one", "two"],
        "start": [0.0, 1.0], "end": [0.5, 1.5],
    })
    monkeypatch.setattr(
        fmri, "load_story_word_data",
        lambda root, stories: ({"story": frame}, {}),
    )
    monkeypatch.setattr(
        "src.utils.shared.embeddings.get_glove_embeddings",
        lambda root, words, dim: (np.ones((2, dim), dtype=np.float32), words),
    )

    result = fmri.load_story_embeddings(
        "unused", "glove-300", ["story"], None,
        fmri_repo_root="fmri", glove_data_root="data",
    )

    assert result["story"].shape == (2, 300)


def test_load_story_embeddings_zero_masks_glove_oov_words(tmp_path, monkeypatch):
    story_dir = tmp_path / "model" / "story"
    story_dir.mkdir(parents=True)
    pd.DataFrame({"word_idx": [0, 1, 2]}).to_csv(
        story_dir / "transcript.tsv", sep="\t", index=False
    )
    token_values = np.arange(6, dtype=np.float32).reshape(3, 2) + 1
    with h5py.File(story_dir / "features.hdf5", "w") as handle:
        handle.create_dataset("layer-2", data=token_values)
    frame = pd.DataFrame({
        "word_idx": [0, 1, 2], "word": ["one", "oov", "three"],
        "start": [0.0, 1.0, 2.0], "end": [0.5, 1.5, 2.5],
    })
    monkeypatch.setattr(
        fmri, "load_story_word_data",
        lambda root, stories: ({"story": frame}, {}),
    )

    def fake_glove(root, words, dim):
        covered = words.copy()
        covered["in_glove"] = [True, False, True]
        return np.ones((3, dim), dtype=np.float32), covered

    monkeypatch.setattr(
        "src.utils.shared.embeddings.get_glove_embeddings", fake_glove
    )

    result = fmri.load_story_embeddings(
        str(tmp_path), "model", ["story"], layer=2,
        fmri_repo_root="fmri", glove_data_root="data",
        limit_to_glove=True, glove_dim=300,
    )

    np.testing.assert_array_equal(
        result["story"], [[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]
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


def test_cached_tr_alignment_standardizes_before_projection(monkeypatch):
    frame = pd.DataFrame(
        {
            "word_idx": np.arange(12),
            "word": ["x"] * 12,
            "start": np.arange(12),
            "end": np.arange(12) + 0.5,
        }
    )
    sequence = type(
        "Sequence", (), {"data_times": np.arange(12), "tr_times": np.arange(12)}
    )()
    monkeypatch.setattr(
        fmri,
        "_toolbox_modules",
        lambda _: (lambda values, old, new, window: values, None, None),
    )
    monkeypatch.setattr(
        fmri,
        "load_story_word_data",
        lambda root, stories: ({"story": frame}, {"story": sequence}),
    )
    embeddings = {"story": np.arange(36, dtype=np.float32).reshape(12, 3)}
    basis = np.array([[1.0, 0.5], [0.25, -1.0], [0.75, 0.25]], dtype=np.float32)

    aligned = fmri.align_fmri_stimuli("unused", ["story"], embeddings, trim=0)
    cached = fmri.build_fmri_stimulus(aligned, ["story"], ndelays=2, basis=basis)
    standardized = fmri._zscore_columns(aligned.stories["story"]).astype(np.float32)
    expected_pre_fir = np.asarray(standardized @ basis, dtype=np.float32)

    np.testing.assert_allclose(cached.pre_fir_stimulus, expected_pre_fir)
    np.testing.assert_allclose(
        cached.stimulus,
        fmri.make_delayed(expected_pre_fir, ndelays=2),
    )


def test_full_rank_orthonormal_projection_preserves_fmri_design():
    rng = np.random.default_rng(4)
    first = rng.normal(size=(19, 5)) * np.array([1.0, 3.0, 0.2, 7.0, 2.0])
    second = rng.normal(size=(23, 5)) * np.array([4.0, 0.5, 8.0, 1.5, 0.1])
    basis, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    aligned = fmri.AlignedFMRIStimuli(stories={"first": first, "second": second})

    full = fmri.build_fmri_stimulus(
        aligned, ["first", "second"], ndelays=3
    )
    projected = fmri.build_fmri_stimulus(
        aligned, ["first", "second"], ndelays=3, basis=basis
    )

    np.testing.assert_allclose(
        projected.pre_fir_stimulus,
        full.pre_fir_stimulus @ basis,
        rtol=1e-5,
        atol=1e-6,
    )
    expected_fir = np.hstack(
        [full.stimulus[:, start : start + 5] @ basis for start in range(0, 15, 5)]
    )
    np.testing.assert_allclose(projected.stimulus, expected_fir, rtol=1e-5, atol=1e-6)


def test_pooled_embedding_scaler_weights_training_rows_and_excludes_test_story():
    aligned = fmri.AlignedFMRIStimuli(stories={
        "short": np.array([[0.0, 2.0], [2.0, 2.0]], dtype=np.float32),
        "long": np.array(
            [[4.0, 2.0], [6.0, 2.0], [8.0, 2.0], [10.0, 2.0]],
            dtype=np.float32,
        ),
        "test": np.full((20, 2), 1000.0, dtype=np.float32),
    })

    scaler = fmri.fit_pooled_embedding_scaler(aligned, ["short", "long"])
    pooled = np.vstack([aligned.stories["short"], aligned.stories["long"]])

    np.testing.assert_allclose(scaler["mean"], pooled.mean(axis=0))
    np.testing.assert_allclose(scaler["scale"][0], pooled[:, 0].std())
    assert scaler["scale"][1] == 1.0
    transformed = np.vstack([
        fmri.apply_embedding_scaler(aligned.stories[name], scaler)
        for name in ("short", "long")
    ])
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(transformed[:, 0].std(), 1.0, atol=1e-6)


def test_pooled_embedding_scaler_preserves_small_variance_around_large_mean():
    values = 1e10 + np.array(
        [[-3.0, 2.0], [-1.0, -2.0], [1.0, 2.0], [3.0, -2.0]],
        dtype=np.float64,
    )
    aligned = fmri.AlignedFMRIStimuli(stories={"train": values})

    scaler = fmri.fit_pooled_embedding_scaler(aligned, ["train"])

    np.testing.assert_allclose(scaler["scale"], values.std(axis=0), rtol=1e-6)


def test_load_responses_reads_selected_voxels_in_subject_order(tmp_path):
    root = tmp_path
    stories = ["first", "second"]
    slices = {"first": slice(0, 2), "second": slice(2, 5)}
    masks = {
        "UTS01": np.array([False, True, False, True]),
        "UTS02": np.array([True, False, True, False]),
    }
    expected_parts = []
    for story_index, story in enumerate(stories):
        story_parts = []
        rows = slices[story].stop - slices[story].start
        for subject_index, subject in enumerate(masks):
            path = fmri.response_path(str(root), subject, story)
            path.parent.mkdir(parents=True, exist_ok=True)
            values = (
                np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
                + 100 * story_index
                + 10 * subject_index
            )
            with h5py.File(path, "w") as handle:
                handle.create_dataset("data", data=values)
            story_parts.append(values[:, masks[subject]])
        expected_parts.append(np.hstack(story_parts))

    loaded = fmri.load_fmri_responses(
        str(root), stories, slices, list(masks), masks, column_batch_size=1
    )

    np.testing.assert_array_equal(loaded, np.vstack(expected_parts))
    assert loaded.dtype == np.float32


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
    first, first_meta = fmri_significance.circular_shift_pvalues(
        true, predicted, slices, num_shifts=20, minimal_shift=5, seed=7, batch_size=4
    )
    second, second_meta = fmri_significance.circular_shift_pvalues(
        true, predicted, slices, num_shifts=20, minimal_shift=5, seed=7, batch_size=6
    )
    np.testing.assert_array_equal(first, second)
    assert first_meta["representative_shifts"] == second_meta["representative_shifts"]
    for draw in first_meta["representative_shifts"]:
        assert all(5 <= shift <= 25 for shift in draw.values())
    assert np.all(first >= 1 / 21)


def test_bh_fdr_and_target_manifest_are_subject_scoped():
    rejected, adjusted = fmri_significance.bh_fdr(
        np.array([0.001, 0.2, 0.01]), 0.05
    )
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
    from src.utils.rrr import rrr_fmri as runner

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
        return {
            "A_dense": fir_a,
            "singular_values": np.array([7.0], dtype=np.float32),
        }

    monkeypatch.setattr(
        runner,
        "fit_dense_rrr",
        fake_fit_dense_rrr,
    )
    config = SimpleNamespace(
        rank_grid=[1],
        subspace_aggregation="factor_weighted",
        rrr_svd_on_cpu=True,
        rrr_use_low_rank_svd=True,
        rrr_low_rank_svd_seed=17,
        rrr_save_b_matrix=False,
        fmri=SimpleNamespace(ndelays=2),
    )
    ridge_fit = runner.FinalRidgeFit(
        test_prediction=np.zeros((2, 3), dtype=np.float32),
        train_prediction=np.zeros((8, 3), dtype=np.float32),
        coefficients=np.zeros((6, 3), dtype=np.float32),
        best_alphas=np.ones(3),
        alpha_grid=np.ones(1),
    )

    bases, factors = runner._fit_within_fmri_bases(
        prepared, ridge_fit, ambient_dim=3, config=config
    )

    np.testing.assert_array_equal(factors["full_fir"], fir_a)
    np.testing.assert_array_equal(factors["prediction_singular_values"], [7.0])
    assert len(factors["delay_blocks"]) == 2
    assert abs(float(bases["rrr"][1][:, 0] @ np.array([1.0, 0.0, 0.0]))) > 0.999
    assert abs(float(bases["pca"][1][:, 0] @ np.array([1.0, 0.0, 0.0]))) > 0.999
    assert fit_kwargs["svd_on_cpu"] is True
    assert fit_kwargs["use_low_rank_svd"] is True
    assert fit_kwargs["low_rank_svd_seed"] == 17
    assert fit_kwargs["return_b"] is False


def test_within_fmri_prediction_weighted_basis_uses_prediction_singular_values(
    monkeypatch,
):
    from src.utils.rrr import rrr_fmri as runner

    prepared = fmri.PreparedFMRI(
        stimulus=np.zeros((8, 6), dtype=np.float32),
        pre_fir_stimulus=np.eye(8, 3, dtype=np.float32),
        responses=np.zeros((8, 3), dtype=np.float32),
        story_slices={"story": slice(0, 8)},
    )
    fir_a = np.array(
        [
            [100.0, 0.0],
            [0.0, 0.0],
            [0.0, 1000.0],
            [0.0, 0.0],
            [100.0, 0.0],
            [0.0, 1000.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        runner,
        "fit_dense_rrr",
        lambda *args, **kwargs: {
            "A_dense": fir_a,
            "singular_values": np.array([10.0, 1.0], dtype=np.float32),
        },
    )
    config = SimpleNamespace(
        rank_grid=[2],
        subspace_aggregation="prediction_weighted",
        rrr_svd_on_cpu=False,
        rrr_use_low_rank_svd=False,
        rrr_low_rank_svd_seed=0,
        rrr_save_b_matrix=False,
        pca_svd_device="cpu",
        fmri=SimpleNamespace(ndelays=2),
    )
    ridge_fit = runner.FinalRidgeFit(
        test_prediction=np.zeros((2, 3), dtype=np.float32),
        train_prediction=np.zeros((8, 3), dtype=np.float32),
        coefficients=np.zeros((6, 3), dtype=np.float32),
        best_alphas=np.ones(3),
        alpha_grid=np.ones(1),
    )

    bases, _ = runner._fit_within_fmri_bases(
        prepared, ridge_fit, ambient_dim=3, config=config
    )

    basis = bases["rrr"][2]
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(
        basis @ basis.T,
        np.diag([1.0, 1.0, 0.0]),
        atol=1e-6,
    )


def test_final_ridge_skips_unused_train_prediction_and_coefficients(
    tmp_path, monkeypatch
):
    from src.utils.rrr import rrr_fmri as runner

    predictions = []

    class FakeModel:
        named_steps = {"ridgecv": SimpleNamespace(coef_=None)}

        def fit(self, X, Y):
            return self

        def predict(self, X):
            predictions.append(X)
            return np.zeros((len(X), 2), dtype=np.float32)

    monkeypatch.setattr(runner, "resolve_ridge_alphas", lambda *args, **kwargs: [1.0])
    monkeypatch.setattr(
        runner, "get_ridge_model_constructor", lambda **kwargs: lambda _: FakeModel()
    )
    monkeypatch.setattr(runner, "extract_best_alphas", lambda model: np.ones(2))
    config = SimpleNamespace(
        ridge=SimpleNamespace(
            alphas=[1.0],
            alpha_grid={},
            solver_params=None,
            y_in_cpu=True,
            force_cpu=False,
            log_alphas=False,
        )
    )
    X_train = np.ones((5, 3), dtype=np.float32)
    X_test = np.ones((2, 3), dtype=np.float32)
    result = runner._fit_final_ridge(
        X_train,
        np.ones((5, 2), dtype=np.float32),
        X_test,
        [],
        config,
        {},
    )

    assert len(predictions) == 1
    assert predictions[0] is X_test
    assert result.train_prediction is None
    assert result.coefficients is None


def test_cross_dataset_completed_resume_skips_embeddings_and_responses(
    tmp_path, monkeypatch
):
    from src.utils.rrr import rrr_fmri as runner
    from src.core.config import (
        FMRIDataConfig, FMRICrossDatasetRRRConfig, SignificanceSelectionConfig,
    )

    fmri_root = tmp_path / "fmri"
    (fmri_root / "em_data").mkdir(parents=True)
    (fmri_root / "em_data" / "sess_to_story.json").write_text(
        '{"1": [["train"], "test"]}'
    )
    from scripts.run_significance import Artifact as SignificanceArtifact
    from scripts.run_rrr import Artifact as IEEGArtifact, UnitSelector
    results_root = tmp_path / "results"
    sig = SignificanceArtifact(mode="held_out_session", dataset="deep-fmri",
                               model="gpt2-xl", layer=24,
                               results_root=results_root).begin(config={"sig": 1})
    sig.save_subject_field(
        "target_significance", "UTS01",
        pd.DataFrame({"subject": ["UTS01"], "target_id": [0],
                      "target_kind": ["voxel"], "significant": [True]}),
    )
    source_selection = {
        "mode": "within-layer",
        "reference_artifact": {
            "mode": "sig_lags", "dataset": "podcast", "offset": 0,
            "model": "gpt2-xl", "layer": 24,
        },
    }
    source = IEEGArtifact(
        model="gpt2-xl", layer=24, results_root=results_root,
        significance_selection=source_selection,
    ).begin(config={"rrr": 1, "significance_selection": source_selection})
    source.save_factors(unit=UnitSelector(method="rrr"), fold=0,
                        input_factors=np.ones((2, 1), dtype=np.float32))
    config = FMRICrossDatasetRRRConfig(
        fmri=FMRIDataConfig(
            fmri_repo_root=str(fmri_root), sessions=[1], subjects=["UTS01"]
        ),
        results_root=str(results_root),
        plot=False,
        significance_selection=SignificanceSelectionConfig(
            reference_artifact={
                "mode": "held_out_session", "dataset": "deep-fmri",
                "offset": 0, "model": "gpt2-xl", "layer": 24,
                "variant": sig.variant,
            }
        ),
        ieeg_rrr_source={
            "mode": "internal", "dataset": "podcast", "offset": 0,
            "model": "gpt2-xl", "layer": 24,
            "significance_selection": source_selection,
        },
        source_mode="cross_dataset",
        rank_grid=[1],
    )
    monkeypatch.setattr(
        runner,
        "load_story_embeddings",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed resume should not load embeddings")
        ),
    )
    target = IEEGArtifact.from_fmri_config(config)
    target.begin(
        config=config,
        upstreams=[sig.identity_record(), source.identity_record()],
        variant_extras=target.variant_extras,
    )
    target.save_field("full_metrics", {
        "corr_per_target": np.array([0.25]), "r2_per_target": np.array([0.1]),
        "best_alphas": np.array([1.0]),
    })
    target.save_rank_metrics(method="rrr", rank=1, metrics={
        "corr_per_target": np.array([0.2]), "r2_per_target": np.array([0.05]),
        "best_alphas": np.array([1.0]),
    })
    result = runner.run_fmri(config)

    results = result.load_field("mean_results")
    assert results[["method", "rank"]].to_dict("records") == [
        {"method": "rrr", "rank": 1}
    ]
