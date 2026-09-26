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


def test_final_saturation_uses_quantiles_and_averages_loso_subject_ranks():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import _loso_saturation_values, _saturation_values
    from src.utils.plotting.final_figures import rrr_aggregate

    frame = pd.DataFrame([
        ("podcast", "glove", None, subject, "rrr", rank, value)
        for subject, values in (("a", [.1, .5]), ("b", [.4, .5]))
        for rank, value in zip([1, 9], values)
    ], columns=["dataset", "model", "layer", "subject", "method", "rank", "correlation"])
    expected = np.mean([8.9, 8.6])
    quantiles = {('podcast', 'glove', None, 'rrr'): ([1, 9], [[.1, .004], [.5, .005]])}
    loso_quantiles = {
        ('podcast', 'glove', None, subject, 'rrr'): ([1, 9], np.tile(values, (6, 1)).T)
        for subject, values in (("a", [.1, .5]), ("b", [.4, .5]))
    }
    assert _loso_saturation_values(loso_quantiles, "podcast", .99)["rrr"][0][1] == pytest.approx(expected)
    assert _saturation_values(quantiles, "podcast", ("rrr",), .99)["rrr"] == [(('glove', None), 8.9)]
    for mode in ("internal", "loso"):
        figure = rrr_aggregate(frame, frame, quantiles, loso_quantiles, [], ('glove', None), mode=mode)
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


def test_quantile_figures_keep_six_levels_and_quantile_saturation_marker():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_data import QUANTILES
    from src.utils.plotting.final_figures import average_quantile_grid, quantile_grid, sensitivity_grid

    assert QUANTILES == (.05, .10, .25, .50, .75, .90)
    quantiles = {("podcast", "gpt2", 0, "rrr"): (np.array([1, 9]), np.tile([.1, .5], (6, 1)).T)}
    internal = pd.DataFrame([
        ("podcast", "gpt2", 0, "rrr", 1, .4),
        ("podcast", "gpt2", 0, "rrr", 9, .5),
    ], columns=["dataset", "model", "layer", "method", "rank", "correlation"])
    sample = quantile_grid(quantiles, ("gpt2", 0))
    assert len(sample.axes[0].lines) == 7
    assert sample.axes[0].lines[-1].get_xdata()[0] == pytest.approx(8.9)
    normalized = average_quantile_grid(quantiles)
    assert len(normalized.axes[0].lines) == 6
    sensitivity = sensitivity_grid(quantiles, [.5, .99])
    assert len(sensitivity.axes) == 6
    for figure in (sample, normalized, sensitivity):
        plt.close(figure)


def test_main_analysis_selectors_include_control_models():
    from src.utils.plotting.final_export import available_pairs

    frame = pd.DataFrame({"model": ["gpt2", "random-gpt2", "glove-300", "glove"]})
    quantiles = {(dataset, model, 0, method): ([1, 2], [[.1], [.2]])
                 for dataset in ("podcast", "cars-2", "fmri")
                 for model in frame["model"] for method in ("rrr", "pca")}
    assert set(available_pairs(quantiles)) == {(model, 0) for model in frame["model"]}


def test_main_saturation_includes_glove_and_random_gpt2():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import _saturation_values, rrr_aggregate

    frame = pd.DataFrame([
        (dataset, model, layer, method, rank, correlation)
        for dataset in ("podcast", "cars-2", "fmri")
        for model, layers in (("random-gpt2", [0, 3]), ("glove-300", [None]), ("gpt2", [0]))
        for layer in layers for method in ("rrr", "pca")
        for rank, correlation in ((1, .1), (9, .5))
    ], columns=["dataset", "model", "layer", "method", "rank", "correlation"])
    quantiles = {
        (dataset, model, layer, method): ([1, 9], np.tile([.1, .5], (6, 1)).T)
        for dataset in ('podcast', 'cars-2', 'fmri')
        for model, layers in (("random-gpt2", [0, 3]), ("glove-300", [None]), ("gpt2", [0]))
        for layer in layers for method in ('rrr', 'pca')
    }
    values = _saturation_values(quantiles, 'podcast', ('rrr', 'pca'), .99)
    assert {pair for pair, rank in values['rrr']} == {
        ('random-gpt2', 0), ('random-gpt2', 3), ('glove-300', None), ('gpt2', 0)
    }
    figure = rrr_aggregate(frame, frame, quantiles, {}, [], ('glove-300', None))
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
    quantiles = {
        ('podcast', 'gpt2', 0, subject, method): ([1, 9], np.tile([.1, .5], (6, 1)).T)
        for subject in ('a', 'b')
        for method in ('RRR baseline', 'RRR transfer', 'PCA baseline')
    }
    figure = sensitivity_grid(quantiles, [.5, .99], mode='loso')
    assert len(figure.axes) == 6
    for ax, expected_mean in ((figure.axes[0], 4), (figure.axes[3], 8.9)):
        assert [label.get_text() for label in ax.get_xticklabels()] == [
            'RRR\nbaseline', 'RRR\ntransfer', 'PCA\nbaseline'
        ]
        assert any(f'mean = {expected_mean:.1f}' in text.get_text() for text in ax.texts)
    plt.close(figure)


def test_peak_cutoff_sensitivity_excludes_low_peaks_at_fixed_99_percent():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import sensitivity_grid

    # The weak quantile saturates later, until its peak is excluded by the cutoff.
    matrix = np.array([[.01, 0], [.1, .003], [.1, .006]])
    quantiles = {('podcast', 'gpt2', 0, method): ([1, 4, 9], matrix)
                for method in ('rrr', 'pca')}
    figure = sensitivity_grid(quantiles, (0, .005, .01), variable='floor')
    assert len(figure.axes) == 9
    for index, cutoff, expected in ((0, '0', 8.9), (3, '0.005', 8.9), (6, '0.01', 4.0)):
        ax = figure.axes[index]
        assert ax.get_ylabel() == cutoff
        assert any(f'mean = {expected:.1f}' in text.get_text() for text in ax.texts)
    plt.close(figure)


def test_quantile_robustness_ignores_negative_peaks_and_outer_quantiles():
    import matplotlib.pyplot as plt
    from src.utils.plotting.final_figures import _quantile_robustness_values, quantile_saturation_robustness

    # Outer quantiles would cross much later; the median dips after its first crossing.
    matrix = np.array([
        [.01, .01, -.3, .1, .1, .01],
        [.01, .01, -.2, .5, .2, .01],
        [.5, .5, -.1, .3, .4, .5],
    ])
    quantiles = {('podcast', 'gpt2', 0, method): (np.array([1, 4, 10]), matrix)
                 for method in ('rrr', 'pca')}
    # 75th percentile determines the final crossing, at 9.88.
    ranks = _quantile_robustness_values(quantiles, 'podcast')
    assert ranks['rrr'][0][1] == pytest.approx(9.88)
    figure = quantile_saturation_robustness(quantiles)
    assert len(figure.axes) == 3
    plt.close(figure)
    negative = {('podcast', 'gpt2', 0, 'rrr'): ([1, 4, 10], -np.ones((3, 6)))}
    assert not _quantile_robustness_values(negative, 'podcast')
