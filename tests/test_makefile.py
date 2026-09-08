"""Tests for the compact Makefile workflow interface."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_sbatch(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "sbatch"
    log_path = tmp_path / "sbatch.log"
    executable.write_text(
        """#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$SBATCH_LOG"
counter_path="$SBATCH_LOG.counter"
counter=0
[ ! -f "$counter_path" ] || counter=$(cat "$counter_path")
counter=$((counter + 1))
printf '%s' "$counter" > "$counter_path"
printf '%s\n' "$counter"
"""
    )
    executable.chmod(0o755)
    return executable, log_path


def _run_make(
    tmp_path: Path, target: str, *args: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    sbatch, log_path = _fake_sbatch(tmp_path)
    env = os.environ.copy()
    env["SBATCH_LOG"] = str(log_path)
    result = subprocess.run(
        ["make", target, f"SBATCH={sbatch}", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    submissions = log_path.read_text().splitlines() if log_path.exists() else []
    return result, submissions


def _fake_significance_check(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    executable = tmp_path / "check-significance"
    output = "".join("\t".join(row) + "\n" for row in rows)
    executable.write_text(f"#!/bin/sh\nprintf %s '{output}'\n")
    executable.chmod(0o755)
    return executable


def test_significance_uses_shared_dataset_and_models(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path, "significance", "DATASET=podcast", "MODELS=gpt2 glove"
    )

    assert result.returncode == 0, result.stderr
    assert len(submissions) == 2
    assert "configs/significance/gpt2/significance.yml" in submissions[0]
    assert "configs/significance/glove/significance.yml" in submissions[1]
    assert all("scripts/run_significance.py" in item for item in submissions)


def test_fmri_significance_supports_glove(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path, "significance", "DATASET=fmri", "MODELS=gpt2 glove"
    )

    assert result.returncode == 0, result.stderr
    assert len(submissions) == 2
    assert "configs/cross_dataset/fmri/significance/gpt2.yml" in submissions[0]
    assert "configs/cross_dataset/fmri/significance/glove.yml" in submissions[1]


def test_rrr_resolves_dataset_and_mode(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path,
        "rrr",
        "DATASET=cars-2",
        "RRR_MODE=temporal_windows",
        "MODELS=gpt2 gpt2-xl",
    )

    assert result.returncode == 0, result.stderr
    assert len(submissions) == 2
    assert "configs/brain_treebank/cars_2/rrr/gpt2/temporal_windows.yml" in submissions[0]
    assert "configs/brain_treebank/cars_2/rrr/gpt2-xl/temporal_windows.yml" in submissions[1]


def test_fmri_rrr_rejects_an_ieeg_only_mode(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path,
        "rrr",
        "DATASET=fmri",
        "RRR_MODE=leave_one_subject_out",
        "MODELS=gpt2",
    )

    assert result.returncode != 0
    assert "unsupported for fmri" in result.stderr
    assert submissions == []


def test_leave_one_subject_out_submits_one_job_per_model_layer(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path,
        "rrr",
        "DATASET=podcast",
        "RRR_MODE=leave_one_subject_out",
        "MODELS=gpt2 glove",
    )

    assert result.returncode == 0, result.stderr
    assert len(submissions) == 6
    assert all(
        "configs/rrr/gpt2/leave_one_subject_out.yml" in submission
        for submission in submissions[:5]
    )
    assert [
        submission.rsplit("--ieeg.encoding.per_model_config.layer=", 1)[-1]
        for submission in submissions[:5]
    ] == ["0", "3", "6", "9", "12"]
    assert "configs/rrr/glove/leave_one_subject_out.yml" in submissions[-1]
    assert "--ieeg.encoding.per_model_config.layer=" not in submissions[-1]


def test_cross_dataset_rrr_requires_and_uses_source_dataset(tmp_path: Path):
    missing, submissions = _run_make(
        tmp_path, "rrr", "DATASET=fmri", "RRR_MODE=cross_dataset", "MODELS=gpt2"
    )
    assert missing.returncode != 0
    assert "SOURCE_DATASET is required" in missing.stderr
    assert submissions == []

    result, submissions = _run_make(
        tmp_path,
        "rrr",
        "DATASET=fmri",
        "RRR_MODE=cross_dataset",
        "SOURCE_DATASET=podcast",
        "MODELS=gpt2",
    )
    assert result.returncode == 0, result.stderr
    assert "configs/cross_dataset/fmri/from_podcast/gpt2.yml" in submissions[0]


def test_cross_condition_resolves_region_and_stimulus_configs(tmp_path: Path):
    region, region_submissions = _run_make(
        tmp_path,
        "cross-condition",
        "DATASET=podcast",
        "CROSS_MODE=region",
        "MODELS=gpt2 glove",
    )
    assert region.returncode == 0, region.stderr
    assert len(region_submissions) == 6
    assert all(
        "configs/cross_condition/region/gpt2.yml" in submission
        for submission in region_submissions[:5]
    )
    assert [
        submission.rsplit("--per_model_config.layer=", 1)[-1]
        for submission in region_submissions[:5]
    ] == ["0", "3", "6", "9", "12"]
    assert "configs/cross_condition/region/glove.yml" in region_submissions[-1]
    assert "--per_model_config.layer=" not in region_submissions[-1]

    stimulus, stimulus_submissions = _run_make(
        tmp_path,
        "cross-condition",
        "DATASET=brain-treebank",
        "CROSS_MODE=stimulus",
        "MODELS=roberta-large",
    )
    assert stimulus.returncode == 0, stimulus.stderr
    assert len(stimulus_submissions) == 11
    stimulus_submissions = stimulus_submissions[-5:]
    assert all(
        "configs/cross_condition/stimulus/roberta-large.yml" in submission
        for submission in stimulus_submissions
    )
    assert [
        submission.rsplit("--per_model_config.layer=", 1)[-1]
        for submission in stimulus_submissions
    ] == ["0", "6", "12", "18", "24"]


def test_makefile_validates_every_config_before_submission(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path, "significance", "MODELS=gpt2 missing-model"
    )

    assert result.returncode != 0
    assert "Missing significance config" in result.stderr
    assert submissions == []


def test_significance_missing_groups_missing_layers_by_dataset_and_model(tmp_path: Path):
    check = _fake_significance_check(tmp_path, [
        ("podcast", "gpt2", "0"),
        ("podcast", "gpt2", "6"),
        ("deep-fmri", "gpt2", "3"),
        ("podcast", "glove", "none"),
    ])
    result, submissions = _run_make(
        tmp_path,
        "significance-missing",
        f"SIGNIFICANCE_CHECK={check}",
        "SIGNIFICANCE_MODELS=gpt2 glove",
    )

    assert result.returncode == 0, result.stderr
    assert len(submissions) == 3
    assert "configs/significance/gpt2/significance.yml" in submissions[0]
    assert "--ieeg.encoding.per_model_config.layer=[0,6]" in submissions[0]
    assert "configs/cross_dataset/fmri/significance/gpt2.yml" in submissions[1]
    assert "--fmri.layer=[3]" in submissions[1]
    assert "configs/significance/glove/significance.yml" in submissions[2]
    assert "per_model_config.layer" not in submissions[2]


def test_cross_condition_configs_cover_final_models_and_depths():
    expected_layers = {
        "gpt2": [0, 3, 6, 9, 12],
        "gpt2-xl": [0, 12, 24, 36, 48],
        "llama3-8b": [0, 8, 16, 24, 32],
        "gpt-j-6b": [0, 7, 14, 21, 28],
        "roberta-large": [0, 6, 12, 18, 24],
        "random-gpt2": [0, 3, 6, 9, 12],
    }
    for mode in ("region", "stimulus"):
        root = REPO_ROOT / "configs" / "cross_condition" / mode
        assert {path.stem for path in root.glob("*.yml")} == {
            *expected_layers, "glove"
        }
        for model, layers in expected_layers.items():
            payload = yaml.safe_load((root / f"{model}.yml").read_text())
            assert payload["per_model_config"]["layer"] == layers
            assert payload["limit_to_glove"] is True
            assert payload["glove_dim"] == 300

        glove = yaml.safe_load((root / "glove.yml").read_text())
        assert glove["model"] == "glove"
        assert glove["per_model_config"] == {"dim": 300}


def test_final_results_submits_cross_dataset_jobs_after_all_fmri_jobs(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path,
        "final-results",
        "SIGNIFICANCE_CHECK=true",
    )

    assert result.returncode == 0, result.stderr
    assert len(submissions) == 63
    assert any("random-gpt2/cross_model_layer.yml" in item for item in submissions)
    assert not any("random-gpt2-xl" in item for item in submissions)
    fmri_ids = [3, 10, 17, 24, 31, 38, 45]
    dependency = "--dependency=afterok:" + ":".join(map(str, fmri_ids))
    assert all(dependency in item for item in submissions[-14:])


def test_final_rrr_configs_use_one_equivalent_significance_union():
    expected = {
        ("gpt2", (0, 3, 6, 9, 12)),
        ("gpt2-xl-batch", (0, 12, 24, 36, 48)),
        ("gpt-j-6b", (0, 7, 14, 21, 28)),
        ("roberta-large", (0, 6, 12, 18, 24)),
        ("llama3-8b", (0, 8, 16, 24, 32)),
        ("random-gpt2", (0, 3, 6, 9, 12)),
        ("glove-300", (None,)),
    }

    def signature(selection):
        assert selection["mode"] == "cross-model-layer"
        return {
            (
                artifact["model"],
                tuple(
                    artifact["layer"]
                    if isinstance(artifact["layer"], list)
                    else [artifact["layer"]]
                ),
            )
            for artifact in selection["artifacts"]
        }

    models = [
        "gpt2", "gpt2-xl", "gpt-j-6b", "roberta-large", "llama3-8b",
        "random-gpt2", "glove",
    ]
    configs = []
    for model in models:
        configs.extend([
            REPO_ROOT / "configs/rrr" / model / "cross_model_layer.yml",
            REPO_ROOT / "configs/rrr" / model / "leave_one_subject_out.yml",
            REPO_ROOT / "configs/brain_treebank/cars_2/rrr" / model
            / "cross_model_layer.yml",
            REPO_ROOT / "configs/brain_treebank/cars_2/rrr" / model
            / "leave_one_subject_out.yml",
            REPO_ROOT / "configs/fmri/within_fmri_cross_model_layer"
            / f"{model}.yml",
        ])
        for target in ("podcast", "cars-2"):
            path = (
                REPO_ROOT / "configs/cross_dataset" / target / "from_fmri"
                / f"{model}.yml"
            )
            payload = yaml.safe_load(path.read_text())
            assert signature(payload["significance_selection"]) == expected
            assert signature(
                payload["rrr_source"]["significance_selection"]
            ) == expected

    for model in [*models, "random-gpt2-xl"]:
        for target, source in (
            ("podcast", "cars-2"),
            ("cars-2", "podcast"),
            ("fmri", "podcast"),
            ("fmri", "cars-2"),
        ):
            path = (
                REPO_ROOT / "configs/cross_dataset" / target / f"from_{source}"
                / f"{model}.yml"
            )
            payload = yaml.safe_load(path.read_text())
            assert signature(payload["significance_selection"]) == expected
            assert signature(
                payload["rrr_source"]["significance_selection"]
            ) == expected

    for path in configs:
        text = path.read_text()
        assert "random-gpt2-xl" not in text
        assert signature(yaml.safe_load(text)["significance_selection"]) == expected
