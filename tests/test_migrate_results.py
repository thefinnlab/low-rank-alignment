from pathlib import Path

import pytest

from scripts.utilities.migrate_results import (
    Mapping, copy_mapping, discover_ambiguous, discover_mappings,
)


def mapping(source):
    return Mapping(
        source=str(source), producer="run_significance", mode="sig_lags",
        dataset="podcast", offset=0, model="gpt2-xl-batch", layer=24,
        variant="a" * 64,
    )


def test_dry_run_apply_and_idempotence(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "all_subject_sig.csv").write_text("subject,elec\n1,A1\n")
    results = tmp_path / "canonical"
    dry = copy_mapping(mapping(legacy), results, apply=False)
    assert dry["operations"][0]["action"] == "copy"
    assert not results.exists()
    copy_mapping(mapping(legacy), results, apply=True)
    second = copy_mapping(mapping(legacy), results, apply=True)
    assert second["operations"][0]["action"] == "skip-identical"


def test_conflict_refusal_preserves_destination(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    source = legacy / "all_subject_sig.csv"
    source.write_text("new")
    results = tmp_path / "canonical"
    item = mapping(legacy)
    copy_mapping(item, results, apply=True)
    source.write_text("different")
    with pytest.raises(FileExistsError, match="Differing-content"):
        copy_mapping(item, results, apply=True)


def test_discovery_excludes_new_canonical_tree(tmp_path):
    legacy = tmp_path / "results" / "old"
    legacy.mkdir(parents=True)
    (legacy / "grid_results.csv").write_text("k\n1\n")
    canonical = tmp_path / "results" / "run_rrr" / "cross_subject"
    canonical.mkdir(parents=True)
    (canonical / "grid_results.csv").write_text("k\n1\n")
    found = discover_ambiguous(tmp_path / "results", tmp_path / "results")
    assert [entry["source"] for entry in found] == [str(legacy)]


def test_discovery_infers_configured_identity_and_reports_shared_root(tmp_path):
    legacy = tmp_path / "results" / "gpt2-xl_layer24_offset0"
    legacy.mkdir(parents=True)
    (legacy / "config.yml").write_text(
        "encoding:\n  dataset: podcast\n  model: transformer\n"
        "  offset: 0\n  per_model_config:\n    model_dir: gpt2-xl\n    layer: 24\n"
    )
    (legacy / "all_subject_sig.csv").write_text("subject,elec\n1,A1\n")
    (legacy / "summary.csv").write_text("k\n1\n")
    inferred, ambiguous = discover_mappings(tmp_path / "results", tmp_path / "results")
    assert inferred == []
    assert any(item["reason"] == "shared or conflicting producer markers"
               for item in ambiguous)


def test_shared_mapping_copies_only_the_selected_producer(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "all_subject_sig.csv").write_text("subject,elec\n1,A1\n")
    (legacy / "summary.csv").write_text("k\n1\n")
    results = tmp_path / "canonical"
    report = copy_mapping(mapping(legacy), results, apply=True)
    destinations = [item["destination"] for item in report["operations"]]
    assert any(name.endswith("all_subject_sig.csv") for name in destinations)
    assert not any(name.endswith("summary.csv") for name in destinations)
