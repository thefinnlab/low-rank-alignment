import numpy as np
import pandas as pd
import pytest

from src.utils.rrr.plot_metrics import mean_saturation_rank


def test_mean_saturation_interpolates_first_peak_fraction_crossing():
    assert mean_saturation_rank([1, 4, 8, 16], [.1, .4, .3, .5]) == pytest.approx(15.8)
    assert mean_saturation_rank([16, 1, 4, 8], [.5, .1, .4, .3]) == pytest.approx(15.8)
    assert mean_saturation_rank([1, 4, 8], [.1, .4, .3], .5) == pytest.approx(2)
    assert np.isnan(mean_saturation_rank([1], [np.nan]))
    with pytest.raises(ValueError):
        mean_saturation_rank([1], [.1], 0)


def test_final_saturation_uses_mean_curves_and_averages_loso_subject_ranks():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import _loso_saturation_values, _saturation_values
    from src.utils.plotting.final_figures import rrr_aggregate

    frame = pd.DataFrame([
        ("podcast", "glove", None, subject, "rrr", rank, value)
        for subject, values in (("a", [.1, .5]), ("b", [.4, .5]))
        for rank, value in zip([1, 9], values)
    ], columns=["dataset", "model", "layer", "subject", "method", "rank", "correlation"])
    expected = np.mean([8.9, 8.6])
    assert _loso_saturation_values(frame, "podcast", .99)["rrr"] == [(('glove', None), expected)]
    assert _saturation_values(frame, "podcast", ("rrr",), .99)["rrr"][0][0] == ('glove', None)
    for mode in ("internal", "loso"):
        figure = rrr_aggregate(frame, frame, {}, {}, [], ('glove', None), mode=mode)
        plt.close(figure)


def test_notebook_and_export_parse():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in ('notebooks/final_plots.py', 'src/utils/plotting/final_export.py'):
        ast.parse((root / path).read_text())


def test_significance_table_groups_models_by_relative_depth():
    from src.utils.plotting.final_figures import significance_latex

    significant = {
        ("podcast", "gpt2", 0): {"a"},
        ("podcast", "gpt2", 12): {"a", "b"},
        ("podcast", "gpt2-xl", 12): {"c"},
        ("podcast", "gpt2-xl-batch", 24): {"c"},
        ("podcast", "glove", None): {"a", "c"},
        ("cars-2", "gpt2", 3): {"d"},
    }
    table = significance_latex(significant, ["gpt2", "gpt2-xl", "glove"])
    assert r"0\% & 25\% & 50\% & 75\% & 100\%" in table
    assert "Podcast & gpt2 & 1 & -- & -- & -- & 2" in table
    assert " & gpt2-xl & -- & 1 & -- & -- & --" in table
    assert " & gpt2-xl-batch & -- & -- & 1 & -- & --" in table
    assert r" & glove & \multicolumn{5}{c}{2 (layerless)}" in table
    assert r" & Union & \multicolumn{5}{c}{3}" in table
    assert "Cars 2 & gpt2 & -- & 1 & -- & -- & --" in table
    assert table.count("Podcast") == table.count("Cars 2") == table.count("fMRI") == 1


def test_quantile_figures_keep_six_levels_and_mean_saturation_marker():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_data import QUANTILES
    from src.utils.plotting.final_figures import average_quantile_grid, quantile_grid, sensitivity_grid

    assert QUANTILES == (.05, .10, .25, .50, .75, .90)
    quantiles = {("podcast", "gpt2", 0, "rrr"): (np.array([1, 9]), np.tile([.1, .5], (6, 1)).T)}
    internal = pd.DataFrame([
        ("podcast", "gpt2", 0, "rrr", 1, .4),
        ("podcast", "gpt2", 0, "rrr", 9, .5),
    ], columns=["dataset", "model", "layer", "method", "rank", "correlation"])
    sample = quantile_grid(quantiles, ("gpt2", 0), internal)
    assert len(sample.axes[0].lines) == 7
    assert sample.axes[0].lines[-1].get_xdata()[0] == pytest.approx(8.6)
    normalized = average_quantile_grid(quantiles)
    assert len(normalized.axes[0].lines) == 6
    sensitivity = sensitivity_grid(internal, [.5, .99])
    assert len(sensitivity.axes) == 6
    for figure in (sample, normalized, sensitivity):
        plt.close(figure)


def test_primary_model_filter_keeps_controls_available_without_mutation():
    from src.utils.plotting.final_data import primary_analysis_data
    from src.utils.plotting.final_export import available_pairs

    frame = pd.DataFrame({"model": ["gpt2", "random-gpt2", "glove-300", "glove"]})
    quantiles = {(dataset, model, 0, method): ([1, 2], [[.1], [.2]])
                 for dataset in ("podcast", "cars-2", "fmri")
                 for model in frame["model"] for method in ("rrr", "pca")}
    filtered = primary_analysis_data(frame, frame, frame, quantiles, quantiles, frame)
    for index in (0, 1, 2, 5):
        assert filtered[index]["model"].tolist() == ["gpt2"]
    for index in (3, 4):
        assert {key[1] for key in filtered[index]} == {"gpt2"}
    assert len(frame) == 4
    assert len(quantiles) == 24
    assert available_pairs(quantiles) == [("gpt2", 0)]


def test_appendix_control_figures_pair_layer_dots_without_violins():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import control_saturation_grid

    frame = pd.DataFrame([
        (dataset, model, layer, method, rank, correlation)
        for dataset in ("podcast", "cars-2", "fmri")
        for model, layers in (("random-gpt2", [0, 3]), ("glove-300", [None]), ("gpt2", [0]))
        for layer in layers for method in ("rrr", "pca")
        for rank, correlation in ((1, .1), (9, .5))
    ], columns=["dataset", "model", "layer", "method", "rank", "correlation"])
    for model, count in (("random-gpt2", 2), ("glove-300", 1)):
        figure = control_saturation_grid(frame, model)
        for ax in figure.axes:
            assert len(ax.lines) == count
            assert not ax.collections
            assert all(len(line.get_ydata()) == 2 for line in ax.lines)
            assert all(line.get_marker() == 'o' for line in ax.lines)
            assert ax.yaxis.get_major_formatter()(32) == '32'
            if model == 'glove-300':
                assert ax.get_legend() is None
        plt.close(figure)


def test_loso_sensitivity_compares_three_methods_at_each_fraction():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import sensitivity_grid

    frame = pd.DataFrame([
        ("podcast", "gpt2", 0, subject, method, rank, correlation)
        for subject in ('a', 'b')
        for method in ('RRR baseline', 'RRR transfer', 'PCA baseline')
        for rank, correlation in ((1, .1), (9, .5))
    ], columns=['dataset', 'model', 'layer', 'subject', 'method', 'rank', 'correlation'])
    figure = sensitivity_grid(frame, [.5, .99], mode='loso')
    assert len(figure.axes) == 6
    for ax, expected_mean in ((figure.axes[0], 4), (figure.axes[3], 8.9)):
        assert [label.get_text() for label in ax.get_xticklabels()] == [
            'RRR\nbaseline', 'RRR\ntransfer', 'PCA\nbaseline'
        ]
        assert any(f'mean = {expected_mean:.1f}' in text.get_text() for text in ax.texts)
    plt.close(figure)
