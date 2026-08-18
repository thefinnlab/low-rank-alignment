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
        shared = None
        if enc.shared_word_selection:
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
            if "selection" in result:
                continue
            cache = load_precomputed_epochs(
                subject,
                enc.data_root,
                dataset=enc.dataset,
                per_dataset_config=enc.per_dataset_config,
                dataset_context=context,
            )
            selection = (
                np.asarray(shared, dtype=int)
                if shared is not None
                else np.asarray(cache["selection"], dtype=int)
            )
            result["selection"] = selection
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
