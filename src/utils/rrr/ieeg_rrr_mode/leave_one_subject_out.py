"""Leave-one-subject-out iEEG RRR mode."""

from scripts.run_rrr import Artifact, UnitSelector
from src.core.config import RRRConfig

from .leave_one_out import (
    LeaveOneOutPartition,
    _run_leave_one_out_transfer_for_unit,
    run_leave_one_out,
)


def _subject_partitions(subject_ids, sig_electrodes):
    partitions = []
    for held_out_subject in subject_ids:
        held_electrodes = sig_electrodes.get(held_out_subject, [])
        train_subjects = [subject for subject in subject_ids
                          if subject != held_out_subject]
        common = {
            "held_out_subject": int(held_out_subject),
        }
        partitions.append(
            LeaveOneOutPartition(
                label=f"subject {held_out_subject}",
                train_subjects=train_subjects,
                train_electrodes={
                    subject: list(sig_electrodes.get(subject, []))
                    for subject in train_subjects
                },
                held_subjects=[held_out_subject],
                held_electrodes={held_out_subject: list(held_electrodes)},
                train_unit=UnitSelector(
                    method="rrr", **common, loso_phase="train_subjects"
                ),
                transfer_unit=UnitSelector(
                    method="rrr", **common, loso_phase="transfer"
                ),
                baseline_unit=UnitSelector(
                    method="rrr", **common, loso_phase="baseline"
                ),
            )
        )
    return partitions


def run_leave_one_subject_out(
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
    """Train on all other subjects and evaluate each held-out subject."""
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
        partitions=_subject_partitions(subject_ids, sig_electrodes),
    )


# Compatibility alias for downstream imports of the former private helper.
_run_loso_transfer_for_unit = _run_leave_one_out_transfer_for_unit


__all__ = ["run_leave_one_subject_out"]
