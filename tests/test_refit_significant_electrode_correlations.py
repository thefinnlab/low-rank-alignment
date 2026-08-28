import numpy as np
import pandas as pd
import pytest

from scripts.utilities.refit_significant_electrode_correlations import (
    parse_comparison_report,
    summarize_correlations,
)


REPORT = """
irrelevant header
Three-way membership

all three (2):
  1:A, 2:B

lag correlation + lag SEM only (1):
  1:C

lag correlation + cluster only (1):
  1:D

lag SEM + cluster only (1):
  2:E

lag correlation only (1):
  2:F

lag SEM only (1):
  3:G

cluster only (0):
  —
"""


def test_parses_report_memberships_into_each_method_set():
    selections = parse_comparison_report(REPORT)

    assert selections["lag_corr"] == {
        (1, "A"), (2, "B"), (1, "C"), (1, "D"), (2, "F")
    }
    assert selections["lag_sem"] == {
        (1, "A"), (2, "B"), (1, "C"), (2, "E"), (3, "G")
    }
    assert selections["cluster"] == {
        (1, "A"), (2, "B"), (1, "D"), (2, "E")
    }


def test_report_parser_rejects_truncated_membership_lists():
    with pytest.raises(ValueError, match="declares 2.*but 1"):
        parse_comparison_report(REPORT.replace("1:A, 2:B", "1:A"))


def test_report_parser_requires_every_membership_group():
    truncated = REPORT.split("cluster only (0):", 1)[0]
    with pytest.raises(ValueError, match="missing membership groups"):
        parse_comparison_report(truncated)


def test_summarizes_same_fitted_correlations_for_each_selected_set():
    selections = parse_comparison_report(REPORT)
    union = set().union(*selections.values())
    values = {
        item: value
        for item, value in zip(sorted(union), np.linspace(0.1, 0.7, len(union)))
    }
    frame = pd.DataFrame(
        [
            {"subject": subject, "electrode": electrode, "mean_all_lags_corr": value}
            for (subject, electrode), value in values.items()
        ]
    )

    summary = summarize_correlations(frame, selections).set_index("method")

    lag_corr_values = np.array([values[item] for item in selections["lag_corr"]])
    expected = np.tanh(np.mean(np.arctanh(lag_corr_values)))
    assert summary.loc["Lag correlation", "electrodes"] == 5
    assert summary.loc[
        "Lag correlation", "mean all-lag r (Fisher z→r)"
    ] == pytest.approx(expected)


def test_summary_rejects_missing_fitted_electrodes():
    selections = parse_comparison_report(REPORT)
    frame = pd.DataFrame(
        [{"subject": 1, "electrode": "A", "mean_all_lags_corr": 0.1}]
    )

    with pytest.raises(ValueError, match="Fitted results are missing"):
        summarize_correlations(frame, selections)
