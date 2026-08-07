from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml

from scripts.utilities import backfill_podcast_word_selection as backfill


def _write_run(tmp_path: Path, model: str = "transformer") -> Path:
    root = tmp_path / "results" / "gpt2_layer0_offset0"
    config_path = root / "significance" / "config.yml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "encoding": {
                    "dataset": "podcast",
                    "model": model,
                    "offset": 0,
                    "data_root": str(tmp_path / "data"),
                    "limit_to_glove": False,
                }
            }
        )
    )
    result_path = root / "subjects" / "sub_1" / "encoding_results.joblib"
    result_path.parent.mkdir(parents=True)
    joblib.dump(
        {
            "subject_id": 1,
            "offset": 0,
            "train_idxs": [np.array([2, 3]), np.array([0, 1])],
            "test_idxs": [np.array([0, 1]), np.array([2, 3])],
        },
        result_path,
    )
    return root


def test_preflight_and_apply_backfill(monkeypatch, tmp_path):
    root = _write_run(tmp_path)
    monkeypatch.setattr(
        backfill,
        "load_precomputed_epochs",
        lambda *args, **kwargs: {"selection": np.array([10, 11, 12, 13])},
    )

    plans = backfill.build_backfill_plans(tmp_path / "results")
    assert len(plans) == 1
    assert not plans[0].already_complete
    assert backfill.apply_backfill(plans) == 1

    saved = joblib.load(root / "subjects" / "sub_1" / "encoding_results.joblib")
    assert saved["word_selection"]["mode"] == "per_subject"
    np.testing.assert_array_equal(
        saved["word_selection"]["analysis_selection"], [10, 11, 12, 13]
    )


def test_preflight_rejects_incomplete_fold_partition(monkeypatch, tmp_path):
    root = _write_run(tmp_path)
    path = root / "subjects" / "sub_1" / "encoding_results.joblib"
    result = joblib.load(path)
    result["test_idxs"] = [np.array([0]), np.array([2, 3])]
    joblib.dump(result, path)
    monkeypatch.setattr(
        backfill,
        "load_precomputed_epochs",
        lambda *args, **kwargs: {"selection": np.array([10, 11, 12, 13])},
    )

    with pytest.raises(ValueError, match="partition all 4 analysis rows"):
        backfill.build_backfill_plans(tmp_path / "results")


def test_discovery_excludes_glove(tmp_path):
    _write_run(tmp_path, model="glove")
    with pytest.raises(ValueError, match="No non-GloVe Podcast"):
        backfill.discover_podcast_result_roots(tmp_path / "results")
