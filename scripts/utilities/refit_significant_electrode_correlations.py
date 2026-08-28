#!/usr/bin/env python
"""Refit selected electrodes from a significance-comparison console report."""

from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from himalaya.backend import set_backend


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utilities.compare_significance_methods import (
    METHODS,
    METHOD_LABELS,
    THREE_WAY_METHODS,
    _fit_subject,
    load_comparison_config,
)
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.shared.correlation import fisher_z_mean
from src.utils.shared.embeddings import get_glove_embeddings, load_embeddings
from src.utils.shared.ieeg import compute_shared_word_selection


MEMBERSHIP_METHODS = {
    "all three": THREE_WAY_METHODS,
    "lag correlation + lag SEM only": ("lag_corr", "lag_sem"),
    "lag correlation + cluster only": ("lag_corr", "cluster"),
    "lag SEM + cluster only": ("lag_sem", "cluster"),
    "lag correlation only": ("lag_corr",),
    "lag SEM only": ("lag_sem",),
    "cluster only": ("cluster",),
}
_MEMBERSHIP_HEADER = re.compile(
    r"^(?P<label>" + "|".join(map(re.escape, MEMBERSHIP_METHODS)) + r") "
    r"\((?P<count>\d+)\):\s*$"
)
_ELECTRODE_ID = re.compile(r"(?<!\S)(\d+):([^,\s]+)")
_ADDITIONAL_METHOD_HEADER = re.compile(
    rf"^{re.escape(METHOD_LABELS['lag_corr_shared_max'])} \((?P<count>\d+)\):\s*$"
)


def parse_comparison_report(text: str) -> dict[str, set[tuple[int, str]]]:
    """Reconstruct method electrode sets from the report's membership section."""
    marker = "Three-way membership"
    if marker not in text:
        raise ValueError("Comparison report has no 'Three-way membership' section")
    membership_text = text.split(marker, 1)[1]
    additional_marker = "Additional method selections"
    if additional_marker in membership_text:
        membership_text, additional_text = membership_text.split(
            additional_marker, 1
        )
    else:
        additional_text = ""
    lines = membership_text.splitlines()
    groups: dict[str, set[tuple[int, str]]] = {}
    current_label = None
    declared_count = None

    def finish_group():
        if current_label is None:
            return
        actual = len(groups[current_label])
        if actual != declared_count:
            raise ValueError(
                f"Report group {current_label!r} declares {declared_count} electrodes "
                f"but {actual} identifiers were parsed"
            )

    for raw_line in lines:
        line = raw_line.strip()
        header = _MEMBERSHIP_HEADER.match(line)
        if header:
            finish_group()
            current_label = header.group("label")
            declared_count = int(header.group("count"))
            groups[current_label] = set()
            continue
        if current_label is None or not line or line == "—":
            continue
        for subject, electrode in _ELECTRODE_ID.findall(line):
            groups[current_label].add((int(subject), electrode))
    finish_group()

    missing = sorted(set(MEMBERSHIP_METHODS) - set(groups))
    if missing:
        raise ValueError(f"Comparison report is missing membership groups: {missing}")

    selections = {method: set() for method in METHODS}
    for label, members in groups.items():
        for method in MEMBERSHIP_METHODS[label]:
            selections[method].update(members)

    additional_lines = additional_text.splitlines()
    for index, raw_line in enumerate(additional_lines):
        header = _ADDITIONAL_METHOD_HEADER.match(raw_line.strip())
        if not header:
            continue
        declared = int(header.group("count"))
        members = set()
        for member_line in additional_lines[index + 1 :]:
            members.update(
                (int(subject), electrode)
                for subject, electrode in _ELECTRODE_ID.findall(member_line.strip())
            )
        if len(members) != declared:
            raise ValueError(
                "Report shared-max method declares "
                f"{declared} electrodes but {len(members)} identifiers were parsed"
            )
        selections["lag_corr_shared_max"] = members
        break
    return selections


def _selected_by_subject(selections) -> dict[int, list[str]]:
    union = set().union(*selections.values())
    subjects = sorted({subject for subject, _ in union})
    return {
        subject: sorted(electrode for item_subject, electrode in union if item_subject == subject)
        for subject in subjects
    }


def _load_analysis_inputs(config, subjects):
    enc = config.encoding
    configured_subjects = enc.subjects or list(range(1, 10))
    unknown = sorted(set(subjects) - set(configured_subjects))
    if unknown:
        raise ValueError(
            f"Report contains subjects absent from the significance config: {unknown}"
        )
    context = resolve_dataset_context(
        dataset=enc.dataset,
        subject_ids=configured_subjects,
        per_dataset_config=enc.per_dataset_config,
        data_root=enc.data_root,
    )
    get_dataset_adapter(enc.dataset).validate_subjects_share_stimulus(
        configured_subjects, enc.per_dataset_config, enc.data_root
    )
    aligned_embeddings, transcript = load_embeddings(
        enc.data_root,
        enc.model,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=context,
        **enc.per_model_config,
    )
    word_in_glove = None
    if enc.limit_to_glove or enc.model == "glove":
        word_in_glove = get_glove_embeddings(
            enc.data_root, transcript, **enc.per_model_config
        )[1]["in_glove"]
    shared = None
    if enc.shared_word_selection:
        shared = compute_shared_word_selection(
            configured_subjects,
            enc.data_root,
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            dataset_context=context,
        )
    return context, aligned_embeddings, word_in_glove, shared


def refit_selected_correlations(config, selections) -> pd.DataFrame:
    """Fit the union once per subject and return one all-lag r per electrode."""
    by_subject = _selected_by_subject(selections)
    if not by_subject:
        raise ValueError("Comparison report contains no significant electrodes")
    context, embeddings, word_in_glove, shared = _load_analysis_inputs(
        config, list(by_subject)
    )

    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    rows = []
    for subject, electrodes in by_subject.items():
        print(
            f"Processing subject {subject}: fitting {len(electrodes)} selected "
            "electrodes once across the configured folds..."
        )
        fitted = _fit_subject(
            config,
            subject,
            context,
            embeddings,
            word_in_glove,
            shared,
            selected_electrodes=electrodes,
            return_predictions=False,
        )
        correlations = np.asarray(fitted["correlations"], dtype=float)
        electrode_correlations = fisher_z_mean(correlations, axis=(0, 2))
        for electrode, correlation in zip(
            fitted["electrodes"], electrode_correlations
        ):
            rows.append(
                {
                    "subject": int(subject),
                    "electrode": str(electrode),
                    "mean_all_lags_corr": float(correlation),
                }
            )
        del fitted, correlations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize_correlations(frame, selections) -> pd.DataFrame:
    """Summarize common fitted correlations for each method's electrode set."""
    lookup = {
        (int(row.subject), str(row.electrode)): float(row.mean_all_lags_corr)
        for row in frame.itertuples()
    }
    rows = []
    for method in METHODS:
        missing = sorted(selections[method] - set(lookup))
        if missing:
            preview = ", ".join(f"{s}:{e}" for s, e in missing[:10])
            raise ValueError(
                f"Fitted results are missing {len(missing)} {METHOD_LABELS[method]} "
                f"electrodes: {preview}"
            )
        values = np.asarray([lookup[item] for item in selections[method]], dtype=float)
        rows.append(
            {
                "method": METHOD_LABELS[method],
                "electrodes": len(values),
                "mean all-lag r (Fisher z→r)": (
                    float(fisher_z_mean(values)) if len(values) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["podcast", "cars-2"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument(
        "--report",
        required=True,
        help="Text file containing the prior comparison script's complete output",
    )
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report_path = Path(args.report)
    selections = parse_comparison_report(report_path.read_text(encoding="utf-8"))
    config, config_path = load_comparison_config(
        args.dataset, args.model, args.layer, args.config
    )
    print(f"Config: {config_path}")
    print(f"Parsed comparison report: {report_path.resolve()}")
    frame = refit_selected_correlations(config, selections)
    summary = summarize_correlations(frame, selections)
    print("\nSelected-electrode all-lag correlations")
    print(
        summary.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
