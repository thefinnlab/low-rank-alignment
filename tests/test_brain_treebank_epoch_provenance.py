from types import SimpleNamespace

from scripts.utilities.precompute_epochs import provenance_config_filename


def test_brain_treebank_provenance_filename_is_stimulus_scoped():
    context = SimpleNamespace(
        dataset="brain_treebank", stimulus_id="megamind", rereference="laplacian"
    )
    assert (
        provenance_config_filename(context)
        == "config_movie-megamind_ref-laplacian.yml"
    )
