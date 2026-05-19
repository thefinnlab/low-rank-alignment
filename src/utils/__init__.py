# Public API for the utils package.
# Import the most commonly used functions for convenience.

from src.utils.constants import (
    RIDGE_ALPHAS,
    refine_alpha_grid,
    check_alpha_grid_boundaries,
)
from src.utils.alpha_grids import (
    build_dimension_aware_alpha_grid,
    resolve_ridge_alphas,
    save_alpha_diagnostics,
    maybe_refine_alpha_grid,
)
from src.utils.tensor_utils import to_numpy, to_torch
from src.core.config import (
    EncodingConfig,
    SignificanceConfig,
    NullDistConfig,
    NewSignificanceConfig,
    RRRConfig,
    RRRSufficiencyConfig,
    PCAConfig,
    PartialRRRConfig,
    SAEConfig,
    ScriptMode,
    get_results_dir,
)
from src.utils.config_utils import (
    parse_override_args,
    get_nested_value,
    set_nested_attr,
    apply_overrides,
    dict_to_dataclass,
    load_config,
    parse_args,
)

from src.utils.data_loaders import (
    load_raws,
    read_subject_mapping,
    read_electrode_file,
    get_arbitrary_embeddings,
    build_word_epochs,
    compute_epoch_cache_indices,
    build_XY_from_subjects,
    load_sig_electrodes,
    load_fold_indices,
    get_precomputed_epochs_path,
    save_precomputed_epochs,
    load_precomputed_epochs,
    epochs_from_cache,
)
from src.utils.embeddings import load_embeddings, get_glove_embeddings
from src.utils.encoding import (
    train_encoding,
    get_scaled_model_coefficients,
    absorb_scaler,
    get_models_coefficients,
    get_all_targets_test_set,
    get_all_predictions_and_targets_test_set,
    save_encoding_results,
    load_encoding_results,
    apply_offset,
    save_subject_results,
    load_all_subject_correlations,
    load_all_subject_coefficients,
    extract_best_alphas,
    train_encoding_two_pass,
    get_and_offset_arbitrary_embeddings,
)
from src.utils.plotting import (
    add_corrs,
    plot_corrs_vs_arbitrary,
    plot_svd_truncation_results,
    plot_corrs_by_significance,
    plot_svd_corrs,
    plot_partial_svd_sweep,
    plot_partial_svd_lag_heatmap,
    plot_partial_rrr_sweep,
    plot_sae_training_curves,
    plot_sae_recovery_summary,
    plot_alpha_diagnostics,
    plot_alpha_vs_rank,
)


def _missing_optional_dependency(dependency_name, utility_name):
    def _missing(*args, **kwargs):
        raise ModuleNotFoundError(
            f"{dependency_name} is required for {utility_name} utilities."
        )

    return _missing


try:
    from src.utils.significance import (
        batch_corrcoef,
        significant_lags_all_electrodes_gpu,
        save_significance_results,
        load_significance_results,
        get_sig_results,
        phase_randomize_raw,
        phase_randomize_data,
        compute_observed_max_per_electrode,
        split_fold_into_blocks,
        reshape_fold_predictions_and_targets,
        compute_fold_block_correlations,
        compute_fold_correlations,
        build_cluster_filter,
        label_lag_clusters,
        compute_sem_normalized_target_scores,
        score_lag_clusters,
        score_fold_lag_clusters,
        valid_fold_shifts,
        fold_local_circular_shift_null_scores,
        aggregate_global_null,
    )
except ModuleNotFoundError as exc:
    if exc.name != "statsmodels":
        raise

    batch_corrcoef = _missing_optional_dependency("statsmodels", "significance")
    significant_lags_all_electrodes_gpu = _missing_optional_dependency(
        "statsmodels", "significance"
    )
    save_significance_results = _missing_optional_dependency(
        "statsmodels", "significance"
    )
    load_significance_results = _missing_optional_dependency(
        "statsmodels", "significance"
    )
    get_sig_results = _missing_optional_dependency("statsmodels", "significance")
    phase_randomize_raw = _missing_optional_dependency("statsmodels", "significance")
    phase_randomize_data = _missing_optional_dependency("statsmodels", "significance")
    compute_observed_max_per_electrode = _missing_optional_dependency(
        "statsmodels", "significance"
    )
    split_fold_into_blocks = _missing_optional_dependency("statsmodels", "significance")
    reshape_fold_predictions_and_targets = _missing_optional_dependency(
        "statsmodels", "significance"
    )
    compute_fold_block_correlations = _missing_optional_dependency(
        "statsmodels", "significance"
    )
    compute_fold_correlations = _missing_optional_dependency(
        "statsmodels", "significance"
    )
    build_cluster_filter = _missing_optional_dependency("statsmodels", "significance")
    label_lag_clusters = _missing_optional_dependency("statsmodels", "significance")
    score_lag_clusters = _missing_optional_dependency("statsmodels", "significance")
    score_fold_lag_clusters = _missing_optional_dependency(
        "statsmodels", "significance"
    )
    valid_fold_shifts = _missing_optional_dependency("statsmodels", "significance")
    fold_local_circular_shift_null_scores = _missing_optional_dependency(
        "statsmodels", "significance"
    )
    aggregate_global_null = _missing_optional_dependency("statsmodels", "significance")
from src.utils.analysis import (
    build_global_sig_mask,
    get_correlation_gpu,
    partial_out_direction,
    partial_out_subspace,
    save_partial_results,
    load_partial_results,
    partial_out_svd_directions,
    run_comparative_analysis_on_folds,
    get_method_mapping,
    find_dominant_directions,
    compute_truncated_correlations,
    find_best_k,
)
from src.utils.atlas import (
    get_coord_atlas_labels,
    group_electrodes_by_region,
    build_electrode_region_map,
)

try:
    from src.utils.srrr import (
        load_svd_decomposition,
        prox_l21,
        solve_B_closed_form,
        fit_srrr_proxgd,
        run_stability_selection,
        select_directions,
        fit_dense_rrr,
        predict_from_factors,
        run_sufficiency_test,
        compute_cca_similarity,
        compute_procrustes_similarity,
        save_srrr_result,
        load_srrr_result,
        save_rrr_factors,
        save_rrr_result,
        load_rrr_result,
        find_knee_lambda,
    )
except ModuleNotFoundError as exc:
    if exc.name != "himalaya":
        raise
    load_svd_decomposition = _missing_optional_dependency("himalaya", "SRRR")
    prox_l21 = _missing_optional_dependency("himalaya", "SRRR")
    solve_B_closed_form = _missing_optional_dependency("himalaya", "SRRR")
    fit_srrr_proxgd = _missing_optional_dependency("himalaya", "SRRR")
    run_stability_selection = _missing_optional_dependency("himalaya", "SRRR")
    select_directions = _missing_optional_dependency("himalaya", "SRRR")
    fit_dense_rrr = _missing_optional_dependency("himalaya", "SRRR")
    predict_from_factors = _missing_optional_dependency("himalaya", "SRRR")
    run_sufficiency_test = _missing_optional_dependency("himalaya", "SRRR")
    compute_cca_similarity = _missing_optional_dependency("himalaya", "SRRR")
    compute_procrustes_similarity = _missing_optional_dependency("himalaya", "SRRR")
    save_srrr_result = _missing_optional_dependency("himalaya", "SRRR")
    load_srrr_result = _missing_optional_dependency("himalaya", "SRRR")
    save_rrr_factors = _missing_optional_dependency("himalaya", "SRRR")
    save_rrr_result = _missing_optional_dependency("himalaya", "SRRR")
    load_rrr_result = _missing_optional_dependency("himalaya", "SRRR")
    find_knee_lambda = _missing_optional_dependency("himalaya", "SRRR")

try:
    from src.utils.sae import (
        SAEEncoder,
        SAERegressorEndToEnd,
        compute_sae_metrics,
        train_sae_end_to_end,
        train_sae_alternating,
    )
except ModuleNotFoundError as exc:
    if exc.name != "himalaya":
        raise
    SAEEncoder = _missing_optional_dependency("himalaya", "SAE")
    SAERegressorEndToEnd = _missing_optional_dependency("himalaya", "SAE")
    compute_sae_metrics = _missing_optional_dependency("himalaya", "SAE")
    train_sae_end_to_end = _missing_optional_dependency("himalaya", "SAE")
    train_sae_alternating = _missing_optional_dependency("himalaya", "SAE")

try:
    from src.utils.subspace_utils import (
        evaluate_complement,
        evaluate_arbitrary,
    )
except ModuleNotFoundError as exc:
    if exc.name != "himalaya":
        raise
    evaluate_complement = _missing_optional_dependency("himalaya", "subspace")
    evaluate_arbitrary = _missing_optional_dependency("himalaya", "subspace")
from src.utils.examples import (
    _make_context,
    _sample,
    get_example_display_controls,
    generate_example_table,
    get_mo_display_activations,
)

__all__ = [
    # constants
    "RIDGE_ALPHAS",
    "refine_alpha_grid",
    "check_alpha_grid_boundaries",
    "build_dimension_aware_alpha_grid",
    "resolve_ridge_alphas",
    "save_alpha_diagnostics",
    "maybe_refine_alpha_grid",
    # tensor utils
    "to_numpy",
    "to_torch",
    # config
    "EncodingConfig",
    "SignificanceConfig",
    "NullDistConfig",
    "NewSignificanceConfig",
    "RRRConfig",
    "RRRSufficiencyConfig",
    "PartialRRRConfig",
    "SAEConfig",
    "ScriptMode",
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
    "compute_epoch_cache_indices",
    "build_XY_from_subjects",
    "load_sig_electrodes",
    "load_fold_indices",
    "get_precomputed_epochs_path",
    "save_precomputed_epochs",
    "load_precomputed_epochs",
    "epochs_from_cache",
    # embeddings
    "load_embeddings",
    "get_glove_embeddings",
    # encoding
    "train_encoding",
    "get_scaled_model_coefficients",
    "absorb_scaler",
    "get_models_coefficients",
    "get_all_targets_test_set",
    "get_all_predictions_and_targets_test_set",
    "save_encoding_results",
    "load_encoding_results",
    "apply_offset",
    "save_subject_results",
    "load_all_subject_correlations",
    "load_all_subject_coefficients",
    "extract_best_alphas",
    "train_encoding_two_pass",
    "get_and_offset_arbitrary_embeddings",
    # plotting
    "add_corrs",
    "plot_corrs_vs_arbitrary",
    "plot_svd_truncation_results",
    "plot_corrs_by_significance",
    "plot_svd_corrs",
    "plot_partial_svd_sweep",
    "plot_partial_svd_lag_heatmap",
    "plot_partial_rrr_sweep",
    "plot_sae_training_curves",
    "plot_sae_recovery_summary",
    "plot_alpha_diagnostics",
    "plot_alpha_vs_rank",
    # significance
    "batch_corrcoef",
    "significant_lags_all_electrodes_gpu",
    "save_significance_results",
    "load_significance_results",
    "get_sig_results",
    "phase_randomize_raw",
    "phase_randomize_data",
    "compute_observed_max_per_electrode",
    "split_fold_into_blocks",
    "reshape_fold_predictions_and_targets",
    "compute_fold_block_correlations",
    "compute_fold_correlations",
    "build_cluster_filter",
    "label_lag_clusters",
    "compute_sem_normalized_target_scores",
    "score_lag_clusters",
    "score_fold_lag_clusters",
    "valid_fold_shifts",
    "fold_local_circular_shift_null_scores",
    "aggregate_global_null",
    # analysis
    "build_global_sig_mask",
    "get_correlation_gpu",
    "partial_out_subspace",
    "partial_out_svd_directions",
    "partial_out_direction",
    "save_partial_results",
    "load_partial_results",
    "run_comparative_analysis_on_folds",
    "get_method_mapping",
    "find_dominant_directions",
    "compute_truncated_correlations",
    "find_best_k",
    # atlas
    "get_coord_atlas_labels",
    "group_electrodes_by_region",
    "build_electrode_region_map",
    # srrr
    "load_svd_decomposition",
    "prox_l21",
    "solve_B_closed_form",
    "fit_srrr_proxgd",
    "run_stability_selection",
    "select_directions",
    "fit_dense_rrr",
    "predict_from_factors",
    "run_sufficiency_test",
    "compute_cca_similarity",
    "compute_procrustes_similarity",
    "save_srrr_result",
    "load_srrr_result",
    "save_rrr_factors",
    "save_rrr_result",
    "load_rrr_result",
    "find_knee_lambda",
    # sae
    "SAEEncoder",
    "SAERegressorEndToEnd",
    "compute_sae_metrics",
    "train_sae_end_to_end",
    "train_sae_alternating",
    # subspace_utils
    "evaluate_complement",
    "evaluate_arbitrary",
    # examples
    "_make_context",
    "_sample",
    "get_example_display_controls",
    "generate_example_table",
    "get_mo_display_activations",
]
