from pathlib import Path


COVERED = [
    Path("scripts/run_significance.py"),
    Path("scripts/run_rrr.py"),
    Path("src/utils/significance/significance_ieeg.py"),
    Path("src/utils/rrr/rrr_ieeg.py"),
    Path("src/utils/significance/significance_fmri.py"),
    Path("src/utils/rrr/rrr_fmri.py"),
    Path("src/utils/rrr/significance.py"),
    Path("src/utils/rrr/inputs.py"),
    Path("src/utils/rrr/analysis.py"),
]
COVERED.extend(sorted(Path("src/utils/rrr/fmri_rrr_mode").glob("*.py")))


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
    ieeg = Path("src/utils/rrr/rrr_ieeg.py").read_text(encoding="utf-8")
    fmri = Path("src/utils/rrr/rrr_fmri.py").read_text(encoding="utf-8")
    shared = Path("src/utils/rrr/significance.py").read_text(encoding="utf-8")
    assert "load_significance_selection" in ieeg
    assert "load_significance_selection" in fmri
    assert "from scripts.run_significance import Artifact as SignificanceArtifact" in shared
    assert "from scripts.run_rrr import Artifact as IEEGRRRArtifact" in fmri
