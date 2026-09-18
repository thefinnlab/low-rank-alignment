"""Submit the complete scaler-sensitive transfer pipeline to Slurm and wait.

The source jobs are submitted first.  Every consumer receives an ``afterok``
dependency on the source artifact it reads, so a consumer can never race a
partially written source artifact.
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml


DATASETS = ("podcast", "cars-2", "fmri")
TERMINAL_STATES = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
}


def submit(
    label: str,
    command: list[str],
    dependencies: list[str] | None = None,
    *,
    memory: str,
    gpu: bool,
    gpu80: bool,
) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    args = [
        "sbatch", "--parsable", f"--job-name={label}-{getpass.getuser()[:2]}-{stamp}",
        f"--mem={memory}",
    ]
    if gpu:
        args.append("--gres=gpu:1")
        args.append("--constraint=gpu80" if gpu80 else "--constraint=")
    else:
        # submit.sh has GPU defaults for RRR. Empty command-line values clear
        # those directives for jobs that do not request a GPU.
        args.extend(["--gres=", "--constraint="])
    if dependencies:
        args.append(f"--dependency=afterok:{':'.join(dependencies)}")
    result = subprocess.run(
        [*args, "submit.sh", *command], check=True, text=True, capture_output=True
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id:
        raise RuntimeError(f"sbatch returned no job ID for {label}")
    print(f"Submitted {label}: {job_id}", flush=True)
    return job_id


def fmri_layers(config: Path) -> list[str]:
    payload = yaml.safe_load(config.read_text()) or {}
    value = payload.get("fmri", {}).get("layer")
    return [str(layer) for layer in (value if isinstance(value, list) else [value])]


def job_states(job_ids: list[str]) -> dict[str, str]:
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", ",".join(job_ids), "--format=JobIDRaw,State"],
        check=True, text=True, capture_output=True,
    )
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in job_ids:
            states[fields[0]] = fields[1].split("+", 1)[0]
    return states


def wait_for_jobs(job_ids: list[str], interval: int) -> None:
    print(f"Waiting for {len(job_ids)} jobs...", flush=True)
    while True:
        states = job_states(job_ids)
        # Cancel jobs whose successful dependency can no longer be satisfied;
        # otherwise they can remain pending indefinitely on some clusters.
        pending = [job for job, state in states.items() if state == "PENDING"]
        if pending:
            details = subprocess.run(
                ["squeue", "-h", "-j", ",".join(pending), "-o", "%i %r"],
                check=True, text=True, capture_output=True,
            ).stdout
            blocked = [line.split()[0] for line in details.splitlines()
                       if "DependencyNeverSatisfied" in line]
            if blocked:
                subprocess.run(["scancel", *blocked], check=True)
        if len(states) == len(job_ids) and all(
            state in TERMINAL_STATES for state in states.values()
        ):
            break
        time.sleep(interval)

    failures = {job: state for job, state in states.items() if state != "COMPLETED"}
    if failures:
        raise SystemExit("Pipeline jobs failed: " + ", ".join(
            f"{job}={state}" for job, state in failures.items()
        ))
    print("All transfer-regeneration jobs completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    all_jobs: list[str] = []
    source_jobs: dict[tuple[str, str], list[str]] = {}

    for model in args.models:
        for dataset, config in (
            ("podcast", Path(f"configs/rrr/{model}/full_fit.yml")),
            ("cars-2", Path(f"configs/brain_treebank/cars_2/rrr/{model}/full_fit.yml")),
        ):
            job = submit(
                f"full-fit-{dataset}-{model}",
                ["scripts/run_rrr.py", "--config", str(config), "--refresh=true"],
                memory="128G",
                gpu=True,
                gpu80=True,
            )
            source_jobs[dataset, model] = [job]
            all_jobs.append(job)

        config = Path(f"configs/fmri/within_fmri_cross_model_layer/{model}.yml")
        jobs: list[str] = []
        for layer in fmri_layers(config):
            # A narrowed layer is its own source-artifact identity. Keeping these
            # as separate jobs avoids holding every layer's embeddings in memory.
            jobs.append(submit(
                f"internal-fmri-{model}-layer-{layer}",
                ["scripts/run_rrr.py", "--config", str(config),
                 f"--fmri.layer=[{layer}]", "--refresh=true"],
                memory="200G",
                gpu=True,
                gpu80=True,
            ))
        source_jobs["fmri", model] = jobs
        all_jobs.extend(jobs)

    # Cross-RRR and decoding are siblings: both consume source factors directly.
    for model in args.models:
        for source in DATASETS:
            dependencies = source_jobs[source, model]
            for target in DATASETS:
                if source == target:
                    continue
                config = Path(f"configs/cross_dataset/{target}/from_{source}/{model}.yml")
                all_jobs.append(submit(
                    f"rrr-{source}-to-{target}-{model}",
                    ["scripts/run_rrr.py", "--config", str(config), "--refresh=true"],
                    dependencies,
                    memory="200G" if target == "fmri" else "128G",
                    gpu=True,
                    gpu80=True,
                ))

                decode_config = Path(
                    f"configs/language_decoding/cross_dataset/{target}/from_{source}.yml"
                )
                all_jobs.append(submit(
                    f"decode-{source}-to-{target}-{model}",
                    ["scripts/run_language_decoding_batch.py", "--config",
                     str(decode_config), "--model", model, "--refresh=true"],
                    dependencies,
                    memory="32G",
                    gpu=True,
                    gpu80=False,
                ))

        # Regenerate all internal decoding variants. Only fMRI reads a source
        # artifact rebuilt by this pipeline, hence only it needs a dependency.
        for target in DATASETS:
            dependencies = source_jobs[target, model] if target == "fmri" else []
            config = Path(f"configs/language_decoding/internal/{target}.yml")
            all_jobs.append(submit(
                f"decode-{target}-{model}",
                ["scripts/run_language_decoding_batch.py", "--config", str(config),
                 "--model", model, "--refresh=true"],
                dependencies,
                memory="32G",
                gpu=True,
                gpu80=False,
            ))

    wait_for_jobs(all_jobs, args.poll_seconds)


if __name__ == "__main__":
    main()
