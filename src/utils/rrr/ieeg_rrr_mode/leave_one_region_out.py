"""Leave-one-region-out iEEG RRR mode."""

from scripts.run_rrr import Artifact, UnitSelector
from src.core.config import RRRConfig
from src.utils.atlas.core import build_electrode_region_map

from .leave_one_out import LeaveOneOutPartition, run_leave_one_out


def build_region_partitions(electrode_region_map, sig_electrodes):
    """Build disjoint significant-electrode partitions for configured regions."""
    significant_by_region = {}
    for region, per_subject_electrodes in electrode_region_map.items():
        selected = {}
        for subject, electrodes in per_subject_electrodes.items():
            significant = set(sig_electrodes.get(subject, []))
            matches = [electrode for electrode in electrodes
                       if electrode in significant]
            if matches:
                selected[int(subject)] = matches
        significant_by_region[region] = selected

    partitions = []
    for held_out_region, held_electrodes in significant_by_region.items():
        if not held_electrodes:
            print(
                f"Region {held_out_region!r} has no significant electrodes; "
                "skipping leave-one-region-out analysis."
            )
            continue
        train_electrodes = {}
        for region, per_subject_electrodes in significant_by_region.items():
            if region == held_out_region:
                continue
            for subject, electrodes in per_subject_electrodes.items():
                train_electrodes.setdefault(subject, []).extend(electrodes)
        train_electrodes = {
            subject: list(dict.fromkeys(electrodes))
            for subject, electrodes in train_electrodes.items()
            if electrodes
        }
        common = {"held_out_region": str(held_out_region)}
        partitions.append(
            LeaveOneOutPartition(
                label=f"region {held_out_region!r}",
                train_subjects=sorted(train_electrodes),
                train_electrodes=train_electrodes,
                held_subjects=sorted(held_electrodes),
                held_electrodes=held_electrodes,
                train_unit=UnitSelector(
                    method="rrr", **common, loro_phase="train_regions"
                ),
                transfer_unit=UnitSelector(
                    method="rrr", **common, loro_phase="transfer"
                ),
                baseline_unit=UnitSelector(
                    method="rrr", **common, loro_phase="baseline"
                ),
                allow_baseline_without_training=True,
            )
        )
    return partitions


def run_leave_one_region_out(
    config: RRRConfig,
    enc,
    artifact: Artifact,
    significance,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
    shared_selection_metadata=None,
):
    """Train on mapped regions other than each held-out region and transfer."""
    print("Building electrode region map...")
    electrode_region_map = build_electrode_region_map(
        subject_ids,
        enc.data_root,
        config.atlas_region_groups,
        config.atlas_path,
    )
    print(f"  Regions found: {list(electrode_region_map)}")
    return run_leave_one_out(
        config=config,
        enc=enc,
        artifact=artifact,
        significance=significance,
        subject_ids=subject_ids,
        sig_electrodes=sig_electrodes,
        aligned_embeddings=aligned_embeddings,
        word_df=word_df,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
        partitions=build_region_partitions(electrode_region_map, sig_electrodes),
    )


__all__ = ["build_region_partitions", "run_leave_one_region_out"]
