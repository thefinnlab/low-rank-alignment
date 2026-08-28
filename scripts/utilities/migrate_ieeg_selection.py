"""Add missing iEEG stimulus selections to existing significance artifacts."""

from __future__ import annotations

import numpy as np

from scripts.run_significance import Artifact
from scripts.run_significance import Config as SignificanceClientConfig
from src.core.registry import resolve_dataset_context
from src.utils.shared.config import (
    expand_per_model_layer_configs,
    load_config,
    parse_args,
)
from src.utils.shared.ieeg import compute_shared_word_selection, load_precomputed_epochs


def migrate_config(config: SignificanceClientConfig) -> int:
    """Migrate every iEEG significance artifact described by ``config``."""
    config.validate()
    if config.modality != "ieeg":
        raise ValueError("This migration only supports iEEG significance configs")

    migrated = 0
    for selected in expand_per_model_layer_configs(config.selected_config()):
        enc = selected.encoding
        subjects = enc.subjects or list(range(1, 10))
        context = resolve_dataset_context(
            dataset=enc.dataset,
            subject_ids=subjects,
            per_dataset_config=enc.per_dataset_config,
            data_root=enc.data_root,
        )
        shared = compute_shared_word_selection(
            subjects,
            enc.data_root,
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            dataset_context=context,
        )

        artifact = Artifact.from_ieeg_config(selected, context)
        artifact.load_artifacts(
            fields=["encoding_results"], selectors={"subjects": subjects}
        )
        for subject in subjects:
            result = dict(artifact.encoding_results[str(subject)])
            changed = False
            word_selection = result.get("word_selection")
            if word_selection is not None:
                stored_epoch_selection = np.asarray(
                    word_selection["epoch_selection"], dtype=int
                )
                if not np.array_equal(stored_epoch_selection, shared):
                    raise ValueError(
                        f"Cannot repair subject {subject}: stored word selection "
                        "does not match the current all-subject cache intersection"
                    )
                updated_words = dict(word_selection)
                if [int(s) for s in updated_words.get("subjects", [])] != subjects:
                    updated_words["subjects"] = [int(s) for s in subjects]
                    result["word_selection"] = updated_words
                    changed = True
            cache = load_precomputed_epochs(
                subject,
                enc.data_root,
                dataset=enc.dataset,
                per_dataset_config=enc.per_dataset_config,
                dataset_context=context,
            )
            if "selection" not in result:
                result["selection"] = np.asarray(
                    shared if word_selection is not None else cache["selection"],
                    dtype=int,
                )
                changed = True
            if changed:
                artifact.save_subject_field(
                    "encoding_results", subject, result, overwrite=True
                )
                migrated += 1
        print(f"Updated {artifact.identity_record()} ({len(subjects)} subjects)")
    return migrated


def main() -> None:
    args, overrides = parse_args(__doc__)
    config = load_config(args.config, overrides, SignificanceClientConfig)
    print(f"Updated {migrate_config(config)} subject metadata entries.")


if __name__ == "__main__":
    main()
