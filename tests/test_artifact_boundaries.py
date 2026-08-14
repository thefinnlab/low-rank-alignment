from pathlib import Path


COVERED = [
    Path("scripts/sig_lags/run_significance.py"),
    Path("scripts/rrr/run_rrr.py"),
    Path("scripts/fmri/run_significance.py"),
    Path("scripts/fmri/run_cross_dataset_rrr.py"),
    Path("src/utils/rrr/inputs.py"),
    Path("src/utils/rrr/analysis.py"),
]


def test_covered_analysis_has_no_legacy_result_storage_seams():
    forbidden = [
        "ArtifactReference", "get_results_dir", "load_sig_electrodes",
        "output_dir", "sig_dir", "rrr_output_dir", "cross_result_dir",
        "significance_dir", "source_rrr_unit_dir", "unit_dir",
        "os.path", "Path(", "np.save", "np.load", ".to_csv(",
        "pd.read_csv(", "joblib.dump", "joblib.load",
    ]
    failures = []
    for path in COVERED:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                failures.append(f"{path}: {token}")
    assert not failures, "legacy result storage escaped an Artifact:\n" + "\n".join(failures)


def test_downstream_scripts_import_the_producing_artifact_directly():
    ieeg = Path("scripts/rrr/run_rrr.py").read_text(encoding="utf-8")
    fmri = Path("scripts/fmri/run_cross_dataset_rrr.py").read_text(encoding="utf-8")
    assert "from scripts.sig_lags.run_significance import Artifact as SignificanceArtifact" in ieeg
    assert "from scripts.fmri.run_significance import Artifact as SignificanceArtifact" in fmri
    assert "from scripts.rrr.run_rrr import Artifact as IEEGRRRArtifact" in fmri
