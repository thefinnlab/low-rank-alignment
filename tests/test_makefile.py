"""Tests for Makefile job-submission workflows."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_sbatch(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "sbatch"
    log_path = tmp_path / "sbatch.log"
    executable.write_text("""#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$SBATCH_LOG"
count_file="${SBATCH_LOG}.count"
if [ -f "$count_file" ]; then
    count=$(sed -n '1p' "$count_file")
else
    count=0
fi
count=$((count + 1))
printf '%s\n' "$count" > "$count_file"
if [ "${SBATCH_FAIL_ON_CALL:-}" = "$count" ]; then
    exit 1
fi
printf '%s\n' "$((1000 + count))"
""")
    executable.chmod(0o755)
    return executable, log_path


def _run_make(
    tmp_path: Path, *args: str, target: str = "sig-rrr-batch"
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
    lines = log_path.read_text().splitlines() if log_path.exists() else []
    return result, lines


def test_sig_rrr_batch_submits_independent_model_dependencies(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path,
        "PIPELINE_MODELS=gpt2 gpt2-xl",
        "DATASET=podcast",
        "SIG_OVERRIDES=--num_null_shifts=100",
        "RRR_OVERRIDES=--refresh=false",
    )

    assert result.returncode == 0, result.stderr
    assert len(submissions) == 4
    assert "scripts/run_significance.py" in submissions[0]
    assert "configs/significance/gpt2/significance.yml" in submissions[0]
    assert "--num_null_shifts=100" in submissions[0]
    assert "--refresh=false" not in submissions[0]
    assert "--dependency=afterok:1001" in submissions[1]
    assert "configs/rrr/gpt2/cross_subject.yml" in submissions[1]
    assert "--refresh=false" in submissions[1]
    assert "--num_null_shifts=100" not in submissions[1]
    assert "configs/significance/gpt2-xl/significance.yml" in submissions[2]
    assert "--dependency=afterok:1003" in submissions[3]
    assert "configs/rrr/gpt2-xl/cross_subject.yml" in submissions[3]


def test_sig_rrr_batch_resolves_cars_2_configs(tmp_path: Path):
    result, submissions = _run_make(tmp_path, "PIPELINE_MODELS=glove", "DATASET=cars-2")

    assert result.returncode == 0, result.stderr
    assert len(submissions) == 2
    assert (
        "configs/brain_treebank/cars_2/significance/glove/significance.yml"
        in submissions[0]
    )
    assert "configs/brain_treebank/cars_2/rrr/glove/cross_subject.yml" in submissions[1]
    assert "--dependency=afterok:1001" in submissions[1]


def test_sig_rrr_batch_rejects_unsupported_dataset_before_submission(tmp_path: Path):
    result, submissions = _run_make(tmp_path, "PIPELINE_MODELS=gpt2", "DATASET=unknown")

    assert result.returncode != 0
    assert "Unsupported DATASET 'unknown'" in result.stderr
    assert submissions == []


def test_sig_rrr_batch_validates_all_configs_before_submission(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path, "PIPELINE_MODELS=gpt2 missing-model", "DATASET=podcast"
    )

    assert result.returncode != 0
    assert "Missing significance config" in result.stderr
    assert submissions == []


def test_sig_rrr_batch_stops_when_significance_submission_fails(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("SBATCH_FAIL_ON_CALL", "1")

    result, submissions = _run_make(
        tmp_path, "PIPELINE_MODELS=gpt2 gpt2-xl", "DATASET=podcast"
    )

    assert result.returncode != 0
    assert len(submissions) == 1
    assert "scripts/run_significance.py" in submissions[0]


def test_cross_temporal_pipeline_submits_complete_dependency_chain(tmp_path: Path):
    result, submissions = _run_make(
        tmp_path,
        "CROSS_TEMPORAL_MODELS=gpt2 gpt2-xl",
        "RRR_OVERRIDES=--refresh=false",
        target="cross-temporal-pipeline",
    )

    assert result.returncode == 0, result.stderr
    assert len(submissions) == 8

    assert "configs/rrr/gpt2/temporal_windows.yml" in submissions[0]
    assert "--dependency" not in submissions[0]
    assert "--dependency=afterok:1001" in submissions[1]
    assert "configs/rrr/gpt2/cross_temporal_windows.yml" in submissions[1]
    assert "--refresh=false" in submissions[1]
    assert "configs/rrr/gpt2-xl/temporal_windows.yml" in submissions[2]
    assert "--dependency" not in submissions[2]
    assert "configs/rrr/gpt2-xl/cross_temporal_windows.yml" in submissions[3]
    assert "--dependency=afterok:1003" in submissions[3]

    cars_root = "configs/brain_treebank/cars_2/rrr"
    assert f"{cars_root}/gpt2/temporal_windows.yml" in submissions[4]
    assert "--dependency" not in submissions[4]
    assert f"{cars_root}/gpt2/cross_temporal_windows.yml" in submissions[5]
    assert "--dependency=afterok:1005" in submissions[5]
    assert f"{cars_root}/gpt2-xl/temporal_windows.yml" in submissions[6]
    assert f"{cars_root}/gpt2-xl/cross_temporal_windows.yml" in submissions[7]
    assert "--dependency=afterok:1007" in submissions[7]


def test_cross_temporal_pipeline_validates_every_config_before_submission(
    tmp_path: Path,
):
    result, submissions = _run_make(
        tmp_path,
        "CROSS_TEMPORAL_MODELS=gpt2 missing-model",
        target="cross-temporal-pipeline",
    )

    assert result.returncode != 0
    assert "Missing cross-temporal pipeline config" in result.stderr
    assert submissions == []
