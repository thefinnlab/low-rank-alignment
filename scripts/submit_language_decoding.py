#!/usr/bin/env python
"""Submit one internal RRR/PCA language-decoding job per model class."""

from __future__ import annotations

import argparse
import getpass
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["podcast", "cars-2", "fmri"], required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sbatch", default="sbatch")
    parser.add_argument("--submit-script", default="submit.sh")
    parser.add_argument("--sbatch-arg", action="append", default=[])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    manifest = Path(f"configs/language_decoding/internal/{args.dataset}.yml")
    if not manifest.is_file() or not Path(args.submit_script).is_file():
        raise FileNotFoundError("Missing batch manifest or submission script")
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    available = list(payload.get("models", {}))
    models = args.models or available
    unknown = sorted(set(models) - set(available))
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; available={available}")
    tag = args.tag or f"{getpass.getuser()[:2]}-{datetime.now():%Y%m%d-%H%M%S}"
    Path("logs").mkdir(exist_ok=True)
    runner = (
        "scripts/fmri/run_internal_language_decoding_batch.py"
        if args.dataset == "fmri"
        else "scripts/rrr/run_internal_language_decoding_batch.py"
    )
    for model in models:
        command = [
            args.sbatch,
            "--parsable",
            *args.sbatch_arg,
            f"--job-name=lang-{args.dataset}-{model}-{tag}",
            args.submit_script,
            runner,
            "--config",
            str(manifest),
            "--model",
            model,
            "--device",
            args.device,
        ]
        if args.refresh:
            command.append("--refresh")
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            completed = subprocess.run(command, check=True, text=True, capture_output=True)
            print(f"  submitted {completed.stdout.strip()}", flush=True)


if __name__ == "__main__":
    main()
