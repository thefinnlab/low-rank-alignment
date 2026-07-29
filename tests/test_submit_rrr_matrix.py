"""Tests for the dependency-aware SLURM RRR matrix launcher."""

from scripts.submit_rrr_matrix import build_jobs


def _by_key(jobs):
    return {job.key: job for job in jobs}


def test_full_single_model_graph_has_expected_jobs_and_dependencies():
    jobs = build_jobs(
        ["podcast", "cars-2", "fmri"],
        ["gpt2"],
        ["significance", "within", "cross"],
    )
    indexed = _by_key(jobs)

    assert len(jobs) == 12
    assert indexed[("within", "podcast", "gpt2")].dependencies == (
        ("significance", "podcast", "gpt2"),
    )
    assert indexed[("cross", "podcast", "fmri", "gpt2")].dependencies == (
        ("within", "podcast", "gpt2"),
        ("significance", "fmri", "gpt2"),
    )
    assert indexed[("cross", "fmri", "cars-2", "gpt2")].dependencies == (
        ("within", "fmri", "gpt2"),
        ("significance", "cars-2", "gpt2"),
    )


def test_dataset_subset_only_builds_connections_within_subset():
    jobs = build_jobs(
        ["podcast", "fmri"],
        ["gpt2"],
        ["significance", "within", "cross"],
    )

    assert len(jobs) == 6
    assert {job.key for job in jobs if job.key[0] == "cross"} == {
        ("cross", "podcast", "fmri", "gpt2"),
        ("cross", "fmri", "podcast", "gpt2"),
    }


def test_cross_only_assumes_prerequisite_artifacts_already_exist():
    jobs = build_jobs(["podcast", "cars-2"], ["gpt2"], ["cross"])

    assert len(jobs) == 2
    assert all(job.dependencies == () for job in jobs)


def test_each_job_uses_one_model_config_with_all_layers():
    jobs = build_jobs(["podcast"], ["gpt2", "gpt2-xl"], ["significance", "within"])

    assert len(jobs) == 4
    assert {job.config for job in jobs} == {
        "configs/significance/gpt2/significance.yml",
        "configs/significance/gpt2-xl/significance.yml",
        "configs/rrr/gpt2/cross_subject.yml",
        "configs/rrr/gpt2-xl/cross_subject.yml",
    }
