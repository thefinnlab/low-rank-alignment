# Public API for the utils package.
# Import the most commonly used functions for convenience.

from utils.config import EncodingConfig, SignificanceConfig, SVDConfig, get_results_dir
from utils.config_utils import (
    parse_override_args,
    get_nested_value,
    set_nested_attr,
    apply_overrides,
    dict_to_dataclass,
    load_config,
    parse_args,
)

from utils.data_loaders import (
    load_raws,
    read_subject_mapping,
    read_electrode_file,
    get_arbitrary_embeddings,
    build_word_epochs,
)
from utils.embeddings import load_embeddings
from utils.encoding import (
    train_encoding,
    get_scaled_model_coefficients,
    absorb_scaler,
    get_models_coefficients,
    get_all_targets_test_set,
    get_all_predictions_and_targets_test_set,
    get_ridge_model_constructor,
    save_encoding_results,
    load_encoding_results,
    apply_offset,
    save_subject_results,
    load_all_subject_correlations,
    load_all_subject_coefficients,
)
from utils.plotting import (
    add_corrs,
    plot_corrs_vs_arbitrary,
    plot_svd_truncation_results,
    plot_corrs_by_significance,
)
from utils.significance import (
    batch_corrcoef,
    significant_lags_all_electrodes_gpu,
    save_significance_results,
    load_significance_results,
    get_sig_results,
)
from utils.analysis import (
    get_correlation_gpu,
    partial_out_direction,
    save_partial_results,
    load_partial_results,
    run_comparative_analysis_on_folds,
    get_method_mapping,
    find_dominant_directions,
    compute_truncated_correlations,
    find_best_k,
)
from utils.examples import (
    _make_context,
    _sample,
    get_example_display_controls,
    generate_example_table,
    get_mo_display_activations,
)

__all__ = [
    # config
    "EncodingConfig",
    "SignificanceConfig",
    "SVDConfig",
    "get_results_dir",
    "parse_override_args",
    "get_nested_value",
    "set_nested_attr",
    "apply_overrides",
    "dict_to_dataclass",
    "load_config",
    "parse_args",
    # data_loaders
    "load_raws",
    "read_subject_mapping",
    "read_electrode_file",
    "get_arbitrary_embeddings",
    "build_word_epochs",
    # embeddings
    "load_embeddings",
    # encoding
    "train_encoding",
    "get_scaled_model_coefficients",
    "absorb_scaler",
    "get_models_coefficients",
    "get_all_targets_test_set",
    "get_all_predictions_and_targets_test_set",
    "get_ridge_model_constructor",
    "save_encoding_results",
    "load_encoding_results",
    "apply_offset",
    "save_subject_results",
    "load_all_subject_correlations",
    "load_all_subject_coefficients",
    # plotting
    "add_corrs",
    "plot_corrs_vs_arbitrary",
    "plot_svd_truncation_results",
    "plot_corrs_by_significance",
    # significance
    "batch_corrcoef",
    "significant_lags_all_electrodes_gpu",
    "save_significance_results",
    "load_significance_results",
    "get_sig_results",
    # analysis
    "get_correlation_gpu",
    "partial_out_direction",
    "save_partial_results",
    "load_partial_results",
    "run_comparative_analysis_on_folds",
    "get_method_mapping",
    "find_dominant_directions",
    "compute_truncated_correlations",
    "find_best_k",
    # examples
    "_make_context",
    "_sample",
    "get_example_display_controls",
    "generate_example_table",
    "get_mo_display_activations",
]
