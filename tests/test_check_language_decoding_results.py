import json
from pathlib import Path

from scripts.utilities.check_language_decoding_results import (
    expected_artifacts,
    find_failures,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_expected_matrix_covers_every_direction_layer_and_offset(tmp_path):
    expected = expected_artifacts(
        REPO_ROOT, tmp_path, models=["gpt2"], datasets=["podcast", "cars-2", "fmri"]
    )

    # Four word-aligned directions have 5 layers * 3 offsets, while the two
    # directions targeting fMRI have 5 layers * 1 offset.
    assert len(expected) == 70
    assert {(item.source_dataset, item.target_dataset) for item in expected} == {
        ("podcast", "cars-2"),
        ("podcast", "fmri"),
        ("cars-2", "podcast"),
        ("cars-2", "fmri"),
        ("fmri", "podcast"),
        ("fmri", "cars-2"),
    }
    assert {item.layer for item in expected} == {0, 3, 6, 9, 12}
    assert {item.label_offset for item in expected} == {-1, 0, 1}


def test_find_failures_distinguishes_missing_and_incomplete(tmp_path):
    expected = expected_artifacts(
        REPO_ROOT, tmp_path, models=["glove"], datasets=["podcast", "fmri"]
    )
    complete, missing, incomplete = expected[0], expected[2], expected[3]
    for item, is_complete in ((complete, True), (incomplete, False)):
        variant = "test-variant"
        item.leaf.mkdir(parents=True, exist_ok=True)
        (item.leaf / "manifest.json").write_text(
            json.dumps({"variants": {variant: {"complete": is_complete}}}),
            encoding="utf-8",
        )
        variant_dir = item.leaf / variant
        variant_dir.mkdir()
        (variant_dir / "manifest.json").write_text(
            json.dumps(
                {"configuration": item.configuration, "complete": is_complete}
            ),
            encoding="utf-8",
        )

    failures = find_failures([complete, incomplete, missing])

    assert [failure.status for failure in failures] == ["INCOMPLETE", "MISSING"]
