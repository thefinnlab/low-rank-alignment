import json
from pathlib import Path

from scripts.utilities.check_rrr_results import expected_artifacts, find_failures


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_expected_rrr_matrix_covers_all_modes_datasets_directions_and_layers(
    tmp_path,
):
    expected = expected_artifacts(
        REPO_ROOT,
        tmp_path,
        models=["gpt2"],
        modes=["internal", "leave_one_subject_out", "cross_dataset"],
        datasets=["podcast", "cars-2", "fmri"],
    )

    assert len(expected) == 60
    assert {item.mode for item in expected} == {
        "internal",
        "leave_one_subject_out",
        "cross_dataset",
    }
    cross_directions = {
        (item.dataset, item.source_dataset)
        for item in expected
        if item.mode == "cross_dataset"
    }
    assert cross_directions == {
        ("podcast", "brain-treebank-cars-2"),
        ("podcast", "deep-fmri"),
        ("brain-treebank-cars-2", "podcast"),
        ("brain-treebank-cars-2", "deep-fmri"),
        ("deep-fmri", "podcast"),
        ("deep-fmri", "brain-treebank-cars-2"),
    }


def test_find_failures_distinguishes_missing_and_incomplete(tmp_path):
    expected = expected_artifacts(
        REPO_ROOT,
        tmp_path,
        models=["gpt2"],
        modes=["internal"],
        datasets=["podcast"],
    )[:3]

    complete, incomplete, missing = expected
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
            json.dumps({"configuration": item.configuration, "complete": is_complete}),
            encoding="utf-8",
        )

    failures = find_failures([complete, incomplete, missing])

    assert [(failure.status, failure.expected.layer) for failure in failures] == [
        ("INCOMPLETE", incomplete.layer),
        ("MISSING", missing.layer),
    ]
