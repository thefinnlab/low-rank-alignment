from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.utilities.migrate_brain_treebank_epoch_provenance import migrate
from scripts.utilities.precompute_epochs import provenance_config_filename


def test_brain_treebank_provenance_filename_is_stimulus_scoped():
    context = SimpleNamespace(
        dataset="brain_treebank", stimulus_id="megamind", rereference="laplacian"
    )
    assert (
        provenance_config_filename(context)
        == "config_movie-megamind_ref-laplacian.yml"
    )


def test_migrate_validates_and_moves_legacy_provenance(tmp_path: Path):
    directory = tmp_path / "precomputed_epochs" / "sub_7"
    directory.mkdir(parents=True)
    source = directory / "config_ref-laplacian.yml"
    source.write_text(
        yaml.safe_dump(
            {
                "dataset": "brain_treebank",
                "per_dataset_config": {
                    "movie": "cars-2",
                    "rereference": "laplacian",
                },
            }
        ),
        encoding="utf-8",
    )

    moves = migrate(tmp_path, "cars-2", "laplacian", [7], dry_run=False)

    destination = directory / "config_movie-cars-2_ref-laplacian.yml"
    assert moves == [(source, destination)]
    assert destination.exists()
    assert not source.exists()


def test_migrate_refuses_mismatched_legacy_provenance(tmp_path: Path):
    directory = tmp_path / "precomputed_epochs" / "sub_7"
    directory.mkdir(parents=True)
    source = directory / "config_ref-laplacian.yml"
    source.write_text(
        yaml.safe_dump(
            {
                "dataset": "brain_treebank",
                "per_dataset_config": {
                    "movie": "megamind",
                    "rereference": "laplacian",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match requested"):
        migrate(tmp_path, "cars-2", "laplacian", [7], dry_run=False)
    assert source.exists()
