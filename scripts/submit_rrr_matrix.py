#!/usr/bin/env python
"""Submit dependency-aware within- and cross-dataset RRR jobs to SLURM.

Each job handles every configured layer for one model. By default the launcher
submits significance, within-dataset RRR, and every directed transfer among the
selected datasets. Restrict models, datasets, or stages to release a subset.
"""

from __future__ import annotations

import argparse
import getpass
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DATASETS = ("podcast", "cars-2", "fmri")
MODELS = (
    "gpt-j-6b",
    "gpt2-xl",
    "llama3-8b",
    "roberta-large",
    "random-gpt2-xl",
    "gpt2",
)
STAGES = ("significance", "within", "cross")


@dataclass(frozen=True)
class Job:
    key: tuple[str, ...]
    name: str
    script: str
    config: str
    dependencies: tuple[tuple[str, ...], ...] = ()


def _significance_config(dataset: str, model: str) -> tuple[str, str]:
    if dataset == "podcast":
        return (
            "scripts/sig_lags/run_significance.py",
            f"configs/significance/{model}/significance.yml",
        )
    if dataset == "cars-2":
        return (
            "scripts/sig_lags/run_significance.py",
            f"configs/brain_treebank/cars_2/significance/{model}/significance.yml",
        )
    return (
        "scripts/fmri/run_significance.py",
        f"configs/cross_dataset/fmri/significance/{model}.yml",
    )


def _within_config(dataset: str, model: str) -> tuple[str, str]:
    if dataset == "podcast":
        return (
            "scripts/rrr/run_rrr.py",
            f"configs/rrr/{model}/cross_subject.yml",
        )
    if dataset == "cars-2":
        return (
            "scripts/rrr/run_rrr.py",
            f"configs/brain_treebank/cars_2/rrr/{model}/cross_subject.yml",
        )
    return (
        "scripts/fmri/run_cross_dataset_rrr.py",
        f"configs/cross_dataset/fmri/within/{model}.yml",
    )


def _cross_config(target: str, source: str, model: str) -> tuple[str, str]:
    if target == "fmri":
        script = "scripts/fmri/run_cross_dataset_rrr.py"
    else:
        script = "scripts/rrr/run_rrr.py"
    return (
        script,
        f"configs/cross_dataset/{target}/from_{source}/{model}.yml",
    )


def build_jobs(
    datasets: list[str], models: list[str], stages: list[str]
) -> list[Job]:
    """Build jobs in submission order with dependencies on selected stages."""
    selected_stages = set(stages)
    jobs: list[Job] = []
    for model in models:
        for dataset in datasets:
            sig_key = ("significance", dataset, model)
            within_key = ("within", dataset, model)
            if "significance" in selected_stages:
                script, config = _significance_config(dataset, model)
                jobs.append(
                    Job(sig_key, f"sig-{dataset}-{model}", script, config)
                )
            if "within" in selected_stages:
                script, config = _within_config(dataset, model)
                dependencies = (sig_key,) if "significance" in selected_stages else ()
                jobs.append(
                    Job(
                        within_key,
                        f"rrr-{dataset}-{model}",
                        script,
                        config,
                        dependencies,
                    )
                )

        if "cross" in selected_stages:
            for source in datasets:
                for target in datasets:
                    if source == target:
                        continue
                    dependencies = []
                    if "within" in selected_stages:
                        dependencies.append(("within", source, model))
                    if "significance" in selected_stages:
                        dependencies.append(("significance", target, model))
                    script, config = _cross_config(target, source, model)
                    jobs.append(
                        Job(
                            ("cross", source, target, model),
                            f"x-{source}-to-{target}-{model}",
                            script,
                            config,
                            tuple(dependencies),
                        )
                    )
    return jobs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sbatch", default="sbatch", help="SLURM submission command")
    parser.add_argument("--submit-script", default="submit.sh")
    parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Additional sbatch argument; repeat as needed (for example --sbatch-arg=--time=24:00:00).",
    )
    parser.add_argument(
        "--significance-override",
        action="append",
        default=[],
        help="Config override passed to every significance job; repeat as needed.",
    )
    parser.add_argument(
        "--rrr-override",
        action="append",
        default=[],
        help="Config override passed to every within/cross RRR job; repeat as needed.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Job-name tag; defaults to user and current timestamp.",
    )
    return parser.parse_args()


def _validate_jobs(jobs: list[Job], submit_script: str) -> None:
    missing = sorted(
        {path for job in jobs for path in (job.script, job.config) if not Path(path).is_file()}
    )
    if not Path(submit_script).is_file():
        missing.append(submit_script)
    if missing:
        raise FileNotFoundError("Missing submission input(s): " + ", ".join(missing))


def _submit(
    job: Job,
    job_ids: dict[tuple[str, ...], str],
    args: argparse.Namespace,
    tag: str,
) -> str:
    command = [args.sbatch, "--parsable", *args.sbatch_arg]
    dependency_ids = [job_ids[key] for key in job.dependencies]
    if dependency_ids:
        command.append("--dependency=afterok:" + ":".join(dependency_ids))
    command.extend(
        [
            f"--job-name={job.name}-{tag}",
            args.submit_script,
            job.script,
            "--config",
            job.config,
        ]
    )
    overrides = (
        args.significance_override
        if job.key[0] == "significance"
        else args.rrr_override
    )
    command.extend(overrides)

    print(shlex.join(command), flush=True)
    if args.dry_run:
        return f"dry{len(job_ids) + 1}"
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    submission = completed.stdout.strip()
    job_id = submission.split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse SLURM job ID from {submission!r}")
    print(f"  submitted {job_id}", flush=True)
    return job_id


def main() -> None:
    args = _parse_args()
    datasets = list(dict.fromkeys(args.datasets))
    models = list(dict.fromkeys(args.models))
    stages = [stage for stage in STAGES if stage in set(args.stages)]
    jobs = build_jobs(datasets, models, stages)
    _validate_jobs(jobs, args.submit_script)
    Path("logs").mkdir(exist_ok=True)
    tag = args.tag or f"{getpass.getuser()[:2]}-{datetime.now():%Y%m%d-%H%M%S}"
    print(
        f"Submitting {len(jobs)} jobs for datasets={datasets}, models={models}, "
        f"stages={stages}",
        flush=True,
    )
    job_ids: dict[tuple[str, ...], str] = {}
    for job in jobs:
        job_ids[job.key] = _submit(job, job_ids, args, tag)


if __name__ == "__main__":
    main()
