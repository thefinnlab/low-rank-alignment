from pathlib import Path
import importlib.util

import pytest


_PATH = Path("scripts/run_language_decoding_batch.py")
_SPEC = importlib.util.spec_from_file_location("run_language_decoding_batch", _PATH)
batch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(batch)


@pytest.mark.parametrize(
    "manifest",
    sorted(str(path) for path in Path("configs/language_decoding").glob("**/*.yml")),
)
@pytest.mark.parametrize(
    "model,layers",
    [
        ("gpt2", [0, 3, 6, 9, 12]),
        ("gpt2-xl", [0, 12, 24, 36, 48]),
        ("glove", [None]),
    ],
)
def test_all_language_decoding_manifests_expand_per_model(manifest, model, layers):
    configs = batch.build_configs(manifest, model)
    assert [item.encoding.per_model_config.get("layer") for item in configs] == layers
    assert all("self_surprisal" not in item.tasks for item in configs)
    assert all("self_entropy" not in item.tasks for item in configs)
    assert all("surprisal" in item.tasks for item in configs)
    assert all("standardized_entropy" in item.tasks for item in configs)
    assert all(("speaking_rate" in item.tasks) == (item.mode.value == "fmri_tr_pre_fir") for item in configs)


def test_directed_fmri_uses_ieeg_basis_and_internal_fmri_data_source():
    [config, *_] = batch.build_configs(
        "configs/language_decoding/cross_dataset/fmri/from_podcast.yml", "gpt2"
    )
    assert config.rrr_source["dataset"] == "podcast"
    assert config.rrr_source["mode"] == "full_fit"
    assert config.fmri_target_artifact["dataset"] == "deep-fmri"
    assert config.mode.value == "fmri_tr_pre_fir"
    assert config.rrr_source["results_root"] == "results"
    assert config.fmri_target_artifact["results_root"] == "results"


def test_cars_uses_native_speaker_and_omits_unavailable_iu_targets():
    [config, *_] = batch.build_configs(
        "configs/language_decoding/internal/cars-2.yml", "gpt2"
    )
    assert "native_speaker" in config.tasks
    assert not {"iu_onset", "iu_position"} & set(config.tasks)


def test_label_offsets_have_distinct_artifact_variants():
    from src.core.artifacts import readable_variant
    [config, *_] = batch.build_configs("configs/language_decoding/internal/podcast.yml", "gpt2")
    variants = []
    for offset in [-1, 0, 1]:
        config.label_offset = offset
        variants.append(readable_variant(config))
    assert len(set(variants)) == 3
    assert all(f"label-offset-{offset}" in variant for offset, variant in zip([-1, 0, 1], variants))


@pytest.mark.parametrize("mode,offsets", [
    ("fmri_word", [0]), ("fmri_tr_pre_fir", [0]),
])
def test_word_mode_override_excludes_tr_only_tasks(monkeypatch, mode, offsets):
    import sys

    seen = []
    class Artifact:
        def _variant_dir(self):
            return "unused"

    def run(config):
        seen.append(config)
        return Artifact()

    monkeypatch.setattr(batch, "run_config", run)
    monkeypatch.setattr(sys, "argv", [
        "batch", "--config", "configs/language_decoding/internal/fmri.yml",
        "--model", "gpt2", "--layer", "12", f"--mode={mode}",
    ])
    batch.main()
    assert len(seen) == len(offsets)
    assert [config.label_offset for config in seen] == offsets
    assert all(config.mode.value == mode for config in seen)
    assert all(("speaking_rate" in config.tasks) == (mode == "fmri_tr_pre_fir") for config in seen)
    assert all(config.encoding.offset == 0 for config in seen)
    assert all(config.rrr_source["offset"] == 0 for config in seen)
