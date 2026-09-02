#!/usr/bin/env python
"""Generate the directed cross-dataset RRR config matrix for PIPELINE_MODELS.

GloVe is intentionally excluded from fMRI-related runs until fMRI GloVe story
embeddings are defined. The generated YAML files use list-valued layers, which
the iEEG and fMRI entrypoints fan out into independent runs.
"""

from __future__ import annotations

from pathlib import Path

import yaml


MODEL_LAYERS = {
    "gpt-j-6b": ("gpt-j-6b", [0, 7, 14, 21, 28]),
    "gpt2-xl": ("gpt2-xl-batch", [0, 12, 24, 36, 48]),
    "llama3-8b": ("llama3-8b", [0, 8, 16, 24, 32]),
    "roberta-large": ("roberta-large", [0, 6, 12, 18, 24]),
    "random-gpt2-xl": ("random-gpt2-xl", [0, 12, 24, 36, 48]),
    "gpt2": ("gpt2", [0, 3, 6, 9, 12]),
}

FMRI_SUBJECTS = [f"UTS{number:02d}" for number in range(1, 10)]
FMRI_DATA = {
    "fmri_repo_root": "data/deep-fMRI-dataset",
    "embedding_root": "data/fmri/stimuli",
    "sessions": [1, 2, 3, 4, 5],
    "subjects": FMRI_SUBJECTS,
    "trim": 5,
    "ndelays": 4,
}
FMRI_RIDGE = {
    "alphas": None,
    "alpha_grid": {},
    "solver_params": {"n_targets_batch": 8192, "n_targets_batch_refit": 8192},
    "y_in_cpu": True,
    "force_cpu": False,
    "log_alphas": True,
}


def _write(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _rank_grid(model: str) -> list[int]:
    source = Path("configs/rrr") / model / "cross_subject.yml"
    return list(yaml.safe_load(source.read_text(encoding="utf-8"))["rank_grid"])


def _ieeg_encoding(target: str, model_dir: str, layers: list[int]) -> dict:
    config = {
        "data_root": "data/",
        "model": "transformer",
        "offset": 0,
        "subjects": None,
        "run_name": None,
        "batch": False,
        "per_model_config": {"model_dir": model_dir, "layer": layers},
    }
    if target == "cars-2":
        config.update(
            {
                "data_root": "data/brain-treebank",
                "dataset": "brain_treebank",
                "per_dataset_config": {"movie": "cars-2"},
                "subjects": [3, 7, 10],
                "shared_word_selection": True,
                "num_folds": 5,
            }
        )
    return config


def _ieeg_cross_config(
    target: str,
    source: str,
    model_dir: str,
    layers: list[int],
    ranks: list[int],
) -> dict:
    if source == "podcast":
        source_root = f"results/{model_dir}_layer{{layer}}_offset0"
        source_format = "fold_factors"
    elif source == "cars-2":
        source_root = f"results/{model_dir}_layer{{layer}}_offset0/movie-cars-2"
        source_format = "fold_factors"
    else:
        source_root = f"results/fmri/{model_dir}_layer{{layer}}/within_fmri_rrr"
        source_format = "rank_bases"
    return {
        "encoding": _ieeg_encoding(target, model_dir, layers),
        "output_dir": None,
        "sig_dir": None,
        "rrr_output_dir": None,
        "mode": "cross_dataset",
        "cross_result_dir": source_root,
        "cross_source_format": source_format,
        "subspace_aggregation": None,
        "rank_grid": ranks,
        "refresh": True,
        "run_pca_baseline": True,
        "run_null_augmented": False,
    }


def _fmri_significance(model_dir: str, layers: list[int]) -> dict:
    return {
        "fmri": FMRI_DATA,
        "ridge": FMRI_RIDGE,
        "model_dir": model_dir,
        "layer": layers,
        "pooling": "mean",
        "output_dir": f"results/fmri/{model_dir}_layer{{layer}}/significance",
        "batch_subjects": False,
        "num_null_shifts": 10000,
        "null_seed": 42,
        "null_batch_size": 1024,
        "minimal_shift": 10,
        "fdr_q": 0.05,
        "refresh": False,
    }


def _fmri_rrr(
    model_dir: str,
    layers: list[int],
    ranks: list[int],
    source: str,
) -> dict:
    within = source == "fmri"
    source_dir = None
    if source == "podcast":
        source_dir = f"results/{model_dir}_layer{{layer}}_offset0/rrr"
    elif source == "cars-2":
        source_dir = (
            f"results/{model_dir}_layer{{layer}}_offset0/movie-cars-2/rrr"
        )
    run_name = "within_fmri_rrr" if within else f"from_{source}_rrr"
    return {
        "fmri": FMRI_DATA,
        "ridge": FMRI_RIDGE,
        "model_dir": model_dir,
        "layer": layers,
        "pooling": "mean",
        "significance_dir": f"results/fmri/{model_dir}_layer{{layer}}/significance",
        "source_rrr_unit_dir": source_dir,
        "output_dir": f"results/fmri/{model_dir}_layer{{layer}}/{run_name}",
        "source_mode": "within_fmri" if within else "cross_dataset",
        "subspace_aggregation": "prediction_weighted" if within else None,
        "rrr_svd_on_cpu": False,
        "rrr_use_low_rank_svd": within,
        "rrr_low_rank_svd_seed": 0,
        "rrr_save_b_matrix": False,
        "rank_grid": ranks,
        "refresh": True,
    }


def main() -> None:
    root = Path("configs/cross_dataset")
    for model, (model_dir, layers) in MODEL_LAYERS.items():
        ranks = _rank_grid(model)
        _write(root / "fmri" / "significance" / f"{model}.yml", _fmri_significance(model_dir, layers))
        _write(root / "fmri" / "within" / f"{model}.yml", _fmri_rrr(model_dir, layers, ranks, "fmri"))
        for source in ("podcast", "cars-2"):
            _write(
                root / "fmri" / f"from_{source}" / f"{model}.yml",
                _fmri_rrr(model_dir, layers, ranks, source),
            )
        for target, sources in {
            "podcast": ("cars-2", "fmri"),
            "cars-2": ("podcast", "fmri"),
        }.items():
            for source in sources:
                _write(
                    root / target / f"from_{source}" / f"{model}.yml",
                    _ieeg_cross_config(target, source, model_dir, layers, ranks),
                )

    print(f"Wrote directed cross-dataset configs under {root}")


if __name__ == "__main__":
    main()
