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


def test_significance_uses_shared_dataset_and_models(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path, "significance", "DATASET=podcast", "MODELS=gpt2 glove"
    )

    assert result.returncode == 0, result.stderr
    assert len(submissions) == 2
    assert "configs/significance/gpt2/significance.yml" in submissions[0]
    assert "configs/significance/glove/significance.yml" in submissions[1]
    assert all("scripts/run_significance.py" in item for item in submissions)


def test_fmri_significance_uses_supported_models_and_skips_glove(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path, "significance", "DATASET=fmri", "MODELS=gpt2 glove"
    )

    assert result.returncode == 0, result.stderr
    assert len(submissions) == 1
    assert "configs/cross_dataset/fmri/significance/gpt2.yml" in submissions[0]
    assert "Skipping glove" in result.stderr


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
    assert "configs/cross_condition/region/gpt2.yml" in region_submissions[0]
    assert "configs/cross_condition/region/glove.yml" in region_submissions[1]

    stimulus, stimulus_submissions = _run_make(
        tmp_path,
        "cross-condition",
        "DATASET=brain-treebank",
        "CROSS_MODE=stimulus",
        "MODELS=roberta-large",
    )
    assert stimulus.returncode == 0, stimulus.stderr
    assert "configs/cross_condition/stimulus/roberta-large.yml" in stimulus_submissions[-1]


def test_makefile_validates_every_config_before_submission(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path, "significance", "MODELS=gpt2 missing-model"
    )

    assert result.returncode != 0
    assert "Missing significance config" in result.stderr
    assert submissions == []


def test_cross_condition_configs_cover_final_models_and_depths():
    expected_layers = {
        "gpt2": [0, 3, 6, 9, 12],
        "gpt2-xl": [0, 12, 24, 36, 48],
        "llama3-8b": [0, 8, 16, 24, 32],
        "gpt-j-6b": [0, 7, 14, 21, 28],
        "roberta-large": [0, 6, 12, 18, 24],
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
