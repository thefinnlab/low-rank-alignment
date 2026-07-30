import numpy as np
import pandas as pd
import yaml

from src.core.config import LanguageDecodingConfig, LanguageDecodingMode
from src.utils.language_decoding import (
    _prepare_single_view_subspaces,
    _shift_vector_target,
    resolve_subspace_unit_dir,
)
from src.utils.language_decoding_batch import (
    discover_sources,
    hierarchical_bootstrap,
    run_language_decoding_batch,
)


def _saved_config(path, *, dataset="podcast", model="gpt2", layer=6):
    path.write_text(
        yaml.safe_dump(
            {
                "encoding": {
                    "dataset": dataset,
                    "per_dataset_config": {"movie": "cars-2"}
                    if dataset == "brain_treebank"
                    else {},
                    "per_model_config": {"model_dir": model, "layer": layer},
                }
            }
        )
    )


def test_batch_discovers_fold_source_from_path_only(tmp_path, monkeypatch):
    root = tmp_path / "run"
    for space in ("rrr", "pca"):
        for fold in range(2):
            directory = root / space / f"fold_{fold}"
            directory.mkdir(parents=True)
            np.save(directory / "A.npy", np.eye(4))
    _saved_config(root / "rrr" / "config.yml", dataset="brain_treebank")
    config = LanguageDecodingConfig(
        result_paths=[str(root)], batch_output_dir=str(tmp_path / "out"), rank_grid=[2]
    )
    monkeypatch.setattr(
        "src.utils.language_decoding_batch.load_embeddings",
        lambda *args, **kwargs: (np.zeros((5, 4)), pd.DataFrame()),
    )
    source = discover_sources(config)[0]
    assert source.source_dataset == "cars-2"
    assert source.factor_format == "folds"
    assert source.model_dir == "gpt2"


def test_batch_skips_missing_result_path_and_keeps_valid_source(tmp_path, monkeypatch, capsys):
    root = tmp_path / "run"
    for space in ("rrr", "pca"):
        directory = root / space / "fold_0"
        directory.mkdir(parents=True)
        np.save(directory / "A.npy", np.eye(4))
    _saved_config(root / "rrr" / "config.yml")
    monkeypatch.setattr(
        "src.utils.language_decoding_batch.load_embeddings",
        lambda *args, **kwargs: (np.zeros((5, 4)), pd.DataFrame()),
    )
    config = LanguageDecodingConfig(
        result_paths=[str(tmp_path / "absent"), str(root)],
        batch_output_dir=str(tmp_path / "out"),
        rank_grid=[2],
    )
    sources = discover_sources(config)
    assert [source.root for source in sources] == [root]
    assert "[missing source]" in capsys.readouterr().out


def test_all_missing_batch_writes_manifest_and_finishes(tmp_path):
    output = tmp_path / "output"
    config = LanguageDecodingConfig(
        result_paths=[str(tmp_path / "one"), str(tmp_path / "two")],
        batch_output_dir=str(output),
        rank_grid=[1],
    )
    assert run_language_decoding_batch(config) == output.resolve()
    missing = pd.read_csv(output / "missing_sources.csv")
    assert len(missing) == 2
    assert "No valid sources" in (output / "REPORT.md").read_text()


def test_rank_specific_fmri_factors_are_repeated_across_probe_folds(tmp_path, monkeypatch):
    root = tmp_path / "fmri"
    for space in ("rrr", "pca"):
        for rank in (1, 2):
            directory = root / space / f"k{rank}"
            directory.mkdir(parents=True)
            np.save(directory / "A.npy", np.eye(3, rank))
    (root / "config.yml").write_text(
        yaml.safe_dump({"fmri": {}, "model_dir": "gpt2", "layer": 6})
    )
    config = LanguageDecodingConfig(
        result_paths=[str(root)], batch_output_dir=str(tmp_path / "out"), rank_grid=[1, 2]
    )
    monkeypatch.setattr(
        "src.utils.language_decoding_batch.load_embeddings",
        lambda *args, **kwargs: (np.zeros((5, 3)), pd.DataFrame()),
    )
    source = discover_sources(config)[0]
    assert source.source_dataset == "fmri"
    assert resolve_subspace_unit_dir(str(root), "rrr") == (root / "rrr").resolve()
    child = LanguageDecodingConfig(
        rrr_run_path=str(root / "rrr"),
        mode=LanguageDecodingMode.CROSS_DATASET,
        rank_grid=[1, 2],
    )
    prepared = _prepare_single_view_subspaces(
        subspace="rrr",
        factor_paths=sorted((root / "rrr").glob("k*/A.npy")),
        folds=[(np.array([0, 1]), np.array([2])), (np.array([1, 2]), np.array([0]))],
        X=np.arange(12, dtype=float).reshape(4, 3),
        config=child,
    )
    np.testing.assert_array_equal(prepared[(0, 2)][0], prepared[(1, 2)][0])


def test_story_safe_vector_shift_masks_boundaries():
    target = np.arange(10, dtype=float).reshape(5, 2)
    story = np.array([0, 0, 0, 1, 1])
    shifted, valid = _shift_vector_target(target, story, -1)
    np.testing.assert_array_equal(valid, [False, True, True, False, True])
    np.testing.assert_array_equal(shifted[4], target[3])


def test_hierarchical_bootstrap_is_seeded_and_dataset_separated():
    rows = []
    for dataset, base in (("podcast", 0.1), ("fmri", -0.1)):
        for architecture in ("a", "b"):
            for fold in range(3):
                rows.append(
                    {
                        "source_dataset": dataset,
                        "control": False,
                        "task": "surprisal",
                        "kind": "continuous",
                        "offset": 0,
                        "k": 8,
                        "architecture": architecture,
                        "source_id": f"{dataset}-{architecture}",
                        "fold": fold,
                        "rrr_minus_pca": base + fold * 0.01,
                    }
                )
    frame = pd.DataFrame(rows)
    left = hierarchical_bootstrap(frame, draws=50, seed=7, confidence_level=0.95)
    right = hierarchical_bootstrap(frame, draws=50, seed=7, confidence_level=0.95)
    pd.testing.assert_frame_equal(left, right)
    assert set(left.source_dataset) == {"podcast", "fmri"}
    assert left.set_index("source_dataset").loc["podcast", "ci_lower"] > 0
    assert left.set_index("source_dataset").loc["fmri", "ci_upper"] < 0
