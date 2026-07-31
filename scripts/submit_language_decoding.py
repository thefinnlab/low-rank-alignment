#!/usr/bin/env python
"""Submit one internal RRR/PCA language-decoding job per model class."""

from __future__ import annotations

import argparse
import getpass
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["all", "podcast", "cars-2", "fmri"], nargs="+", required=True
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sbatch", default="sbatch")
    parser.add_argument("--submit-script", default="submit.sh")
    parser.add_argument("--sbatch-arg", action="append", default=[])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--generate-predictability",
        action="store_true",
        help=(
            "Submit one model-specific surprisal/entropy prerequisite per model and "
            "hold its decoding jobs until generation succeeds"
        ),
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Submit predictability generation jobs without decoding jobs",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    if args.generation_only and not args.generate_predictability:
        parser.error("--generation-only requires --generate-predictability")
    if args.max_rank is not None and args.max_rank < 1:
        parser.error("--max-rank must be positive")
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        parser.error("--run-name may contain only letters, numbers, '.', '_', and '-'")

    datasets = ["podcast", "cars-2", "fmri"] if "all" in args.dataset else args.dataset
    if "all" in args.dataset and len(args.dataset) != 1:
        raise ValueError("--dataset all cannot be combined with named datasets")
    if len(set(datasets)) != len(datasets):
        raise ValueError("--dataset must not contain duplicates")
    if not Path(args.submit_script).is_file():
        raise FileNotFoundError(f"Missing submission script {args.submit_script}")
    tag = args.tag or f"{getpass.getuser()[:2]}-{datetime.now():%Y%m%d-%H%M%S}"
    Path("logs").mkdir(exist_ok=True)
    generation_jobs: dict[str, str] = {}
    if args.generate_predictability:
        predictability_payload = yaml.safe_load(
            Path("configs/language_features/self_surprisal.yml").read_text(
                encoding="utf-8"
            )
        ) or {}
        generation_models = args.models or list(predictability_payload.get("models", {}))
        unknown_generation = sorted(
            set(generation_models) - set(predictability_payload.get("models", {}))
        )
        if unknown_generation:
            raise ValueError(
                f"Unknown predictability model(s) {unknown_generation}"
            )
        for model in generation_models:
            command = [
                args.sbatch,
                "--parsable",
                *args.sbatch_arg,
                f"--job-name=lang-labels-{model}-{tag}",
                args.submit_script,
                "scripts/utilities/generate_self_surprisal.py",
                "--models",
                model,
                "--datasets",
                *datasets,
            ]
            if args.refresh:
                command.append("--refresh")
            print(shlex.join(command), flush=True)
            if args.dry_run:
                generation_jobs[model] = f"GENERATION_JOB_ID_{model}"
            else:
                completed = subprocess.run(
                    command, check=True, text=True, capture_output=True
                )
                job_id = completed.stdout.strip()
                generation_jobs[model] = job_id
                print(f"  submitted {job_id}", flush=True)
    if args.generation_only:
        return
    for dataset in datasets:
        manifest = Path(f"configs/language_decoding/internal/{dataset}.yml")
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing batch manifest {manifest}")
        payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        available = list(payload.get("models", {}))
        models = args.models or available
        unknown = sorted(set(models) - set(available))
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown} for {dataset}; available={available}")
        runner = (
            "scripts/fmri/run_internal_language_decoding_batch.py"
            if dataset == "fmri"
            else "scripts/rrr/run_internal_language_decoding_batch.py"
        )
        for model in models:
            command = [
                args.sbatch,
                "--parsable",
                *args.sbatch_arg,
                *(
                    [f"--dependency=afterok:{generation_jobs[model]}"]
                    if model in generation_jobs
                    else []
                ),
                f"--job-name=lang-{dataset}-{model}-{tag}",
                args.submit_script,
                runner,
                "--config",
                str(manifest),
                "--model",
                model,
                "--device",
                args.device,
            ]
            if args.tasks:
                command.extend(["--tasks", *args.tasks])
            if args.max_rank is not None:
                command.extend(["--max-rank", str(args.max_rank)])
            if args.run_name:
                command.extend(["--run-name", args.run_name])
            if args.refresh:
                command.append("--refresh")
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                completed = subprocess.run(
                    command, check=True, text=True, capture_output=True
                )
                print(f"  submitted {completed.stdout.strip()}", flush=True)


if __name__ == "__main__":
    main()
