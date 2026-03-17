"""Shared fixtures for the test suite."""
import io

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# CSV / TSV helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def subject_mapping_tsv():
    """Tab-delimited participant mapping with two subjects."""
    return io.StringIO("nyu_id\tparticipant_id\n661\tsub-05\n717\tsub-12\n")


@pytest.fixture
def electrode_csv_single():
    """CSV with one subject and two electrodes."""
    return io.StringIO("subject,elec\n1,A1\n1,A2\n")


@pytest.fixture
def electrode_csv_multi():
    """CSV with two subjects."""
    return io.StringIO("subject,elec\n1,A1\n1,A2\n2,C1\n")


# ---------------------------------------------------------------------------
# Synthetic numpy arrays
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def small_X_Y(rng):
    """Small feature / target pair: 20 words, 4 features, 3 elecs x 2 lags."""
    n_words, n_features = 20, 4
    n_electrodes, n_lags = 3, 2
    X = rng.standard_normal((n_words, n_features)).astype(np.float32)
    Y = rng.standard_normal((n_words, n_electrodes * n_lags)).astype(np.float32)
    return X, Y, n_electrodes, n_lags
