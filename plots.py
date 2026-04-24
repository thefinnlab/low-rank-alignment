import marimo

__generated_with = "0.20.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import os
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    import pandas as pd
    from statsmodels.stats.multitest import multipletests

    from src.utils import load_sig_electrodes, build_electrode_region_map, load_raws,load_precomputed_epochs, epochs_from_cache
    from src.utils.plot_helpers import (
        apply_publication_style,
        get_star_label,
        make_viridis_cmap,
        plot_cross_k_panel,
        plot_lag_panel,
    )

    return (
        apply_publication_style,
        build_electrode_region_map,
        cm,
        colors,
        get_star_label,
        load_precomputed_epochs,
        load_sig_electrodes,
        make_viridis_cmap,
        multipletests,
        np,
        os,
        pd,
        plot_cross_k_panel,
        plot_lag_panel,
        plt,
        re,
    )


@app.cell
def _(multipletests, np):
    def permutation_test_fdr(baseline, comparison_models, n_perms=10000, q_level=0.01):
        """
        Performs a vectorized permutation test with FDR correction.

        Parameters:
        - baseline: Array of shape (n_lags,) or (n_electrodes, n_lags)
        - comparison_models: Array of shape (n_models, n_electrodes, n_lags)
        - n_perms: Number of permutations for the null distribution
        - q_level: FDR threshold (default 0.01)

        Returns:
        - significant_mask: Boolean array (n_models, n_lags)
        - adj_p_values: FDR corrected p-values (n_models, n_lags)
        """
        # 1. Ensure baseline matches the comparison dimensions for broadcasting
        # If baseline is (n_lags,), we broadcast it to (n_electrodes, n_lags)
        if baseline.ndim == 1:
            baseline = np.broadcast_to(baseline, comparison_models.shape[1:])

        n_models, n_electrodes, n_lags = comparison_models.shape

        # 2. Compute observed mean difference across electrodes for each model and lag
        # Shape: (n_models, n_lags)
        obs_diff = np.mean(comparison_models - baseline, axis=1)

        # 3. Compute pairwise differences for permutation
        # Shape: (n_models, n_electrodes, n_lags)
        pairwise_diffs = comparison_models - baseline

        # 4. Generate Random Signs for Permutation (-1 or 1)
        # Shape: (n_perms, n_electrodes)
        rng = np.random.default_rng()
        signs = rng.choice([-1, 1], size=(n_perms, n_electrodes))

        # 5. Generate Null Distribution
        # We use np.dot to sum across electrodes efficiently.
        # (n_perms, n_electrodes) @ (n_models, n_electrodes, n_lags) -> (n_models, n_perms, n_lags)
        # Note: We need to swap axes for dot product compatibility
        null_dist = np.einsum('pe,mel->mpl', signs, pairwise_diffs) / n_electrodes

        # 4. ONE-TAILED P-VALUE CALCULATION
        # We count how often the null difference is GREATER than the observed difference
        # We add 1 to numerator and denominator to avoid p-values of 0
        p_values = (np.sum(null_dist >= obs_diff[:, np.newaxis, :], axis=1) + 1) / (n_perms + 1)

        # 7. FDR Correction (across lags for each model)
        adj_p_values = np.zeros_like(p_values)
        sig_mask = np.zeros_like(p_values, dtype=bool)

        for m in range(n_models):
            rejected, corrected, _, _ = multipletests(p_values[m], alpha=q_level, method='fdr_bh')
            adj_p_values[m] = corrected
            sig_mask[m] = rejected

        return sig_mask, adj_p_values


    def global_permutation_test(baseline, comparison_models, n_perms=10000):
        """
        Performs a one-tailed global permutation test (Model > Baseline) 
        averaging across both electrodes and lags.

        Returns:
        - p_values: Array of shape (n_models,)
        - obs_diffs: Array of shape (n_models,) observed global mean differences
        """
        if baseline.ndim == 1:
            baseline = np.broadcast_to(baseline, comparison_models.shape[1:])

        n_models, n_electrodes, n_lags = comparison_models.shape

        # 1. Compute observed global mean difference
        # Average over both Axis 1 (electrodes) and Axis 2 (lags)
        obs_diffs = np.mean(comparison_models - baseline, axis=(1, 2))

        # 2. Pairwise differences
        pairwise_diffs = comparison_models - baseline

        # 3. Generate Null Distribution via Electrode-Level Sign-Flipping
        rng = np.random.default_rng()
        # Shape: (n_perms, n_electrodes)
        # We apply the same sign to all lags within an electrode to preserve temporal structure
        signs = rng.choice([-1, 1], size=(n_perms, n_electrodes))

        # 4. Vectorized Null Generation
        # (n_perms, n_electrodes) @ (n_models, n_electrodes, n_lags) -> (n_models, n_perms, n_lags)
        # Then average over the resulting lags
        null_dist_lags = np.einsum('pe,mel->mpl', signs, pairwise_diffs) / n_electrodes
        null_dist_global = np.mean(null_dist_lags, axis=2) # Shape: (n_models, n_perms)

        # 5. One-tailed P-value (Model > Baseline)
        # Compare each model's observed global diff to its specific null distribution
        p_values = (np.sum(null_dist_global >= obs_diffs[:, np.newaxis], axis=1) + 1) / (n_perms + 1)

        return p_values, obs_diffs


    def compare_multi_models_to_baseline(baseline, comparison_models, n_perms=10000, block_size=5, q_level=0.01):
        """
        Compares multiple models against a single baseline across all electrodes.

        Parameters:
        - baseline: (n_electrodes, n_lags)
        - comparison_models: (n_models, n_electrodes, n_lags)
        - n_perms: Number of permutations
        - block_size: Lags to flip together for temporal autocorrelation
        - q_level: FDR threshold

        Returns:
        - sig_mask: (n_models, n_electrodes) boolean mask
        - adj_p_values: (n_models, n_electrodes) corrected p-values
        """
        n_mods, n_elec, n_lags = comparison_models.shape

        # 1. Compute observed differences (n_models, n_electrodes)
        # Broadcasting baseline (e, l) to match (m, e, l)
        diffs = comparison_models - baseline[None, :, :]
        obs_diffs = np.mean(diffs, axis=2)

        # 2. Blocked Signs (n_perms, n_lags)
        n_blocks = int(np.ceil(n_lags / block_size))
        rng = np.random.default_rng()
        block_signs = rng.choice([-1, 1], size=(n_perms, n_blocks))
        perm_signs = np.repeat(block_signs, block_size, axis=1)[:, :n_lags]

        # 3. Vectorized Null Generation
        # 'pl' is (perms, lags)
        # 'mel' is (models, electrodes, lags)
        # Result 'mpe' is (models, perms, electrodes)
        null_dist = np.einsum('pl,mel->mpe', perm_signs, diffs) / n_lags

        # 4. One-Tailed P-values
        # Compare (m, p, e) null to (m, e) observed
        # We expand obs_diffs to (m, 1, e) for broadcasting
        p_values = (np.sum(null_dist >= obs_diffs[:, None, :], axis=1) + 1) / (n_perms + 1)

        # 5. FDR Correction
        # We flatten to correct across ALL models and ALL electrodes simultaneously
        p_flat = p_values.flatten()
        sig_flat, adj_p_flat, _, _ = multipletests(p_flat, alpha=q_level, method='fdr_bh')

        # Reshape back to (n_models, n_electrodes)
        return sig_flat.reshape(n_mods, n_elec), adj_p_flat.reshape(n_mods, n_elec)

    return (
        compare_multi_models_to_baseline,
        global_permutation_test,
        permutation_test_fdr,
    )


@app.cell
def _():
    # base_result_dir = "results/gpt2xl_layer24_offset0/per_subject_rrr/subjects/sub_1"
    # base_result_dir = "results/gpt2xl_layer24_offset0/rrr/"
    base_result_dir = "results/gpt2xl_layer24_offset0/rrr/"
    # base_corrs_path = "results/gpt2xl_layer24_offset0/subjects/sub_1/corrs.npy"
    # base_corrs_path = "results/gpt2xl_layer24_offset0/subjects/sub_{sub}/corrs.npy"
    # subs = [1]
    subs = list(range(1, 10))
    return base_result_dir, subs


@app.cell
def _(base_result_dir, load_sig_electrodes, np, os, re, subs):
    ks = []
    results = []
    joint_results = []

    sub_elec_dict, _ = load_sig_electrodes("results/gpt2xl_layer24_offset0/", subs)
    for k_dir in os.listdir(base_result_dir):
        match = re.search(r'k(-?\d+)', k_dir)
        if match:
            ks.append(int(match.group(1)))
            curr_k_path = os.path.join(base_result_dir, k_dir)
            fold_results = []
            fold_joint_results = []
            for fold_dir in os.listdir(curr_k_path):
                metrics_path = os.path.join(curr_k_path, fold_dir, "metrics.npz")
                curr_data = np.load(metrics_path)
                fold_results.append(curr_data["nr_corr_per_target"].reshape((sum([len(sub_elec_dict[_sub]) for _sub in subs]), -1)))
                print(curr_data["nr_complement_joint_corr_per_target"])
                fold_joint_results.append(curr_data["nr_complement_joint_corr_per_target"].reshape((sum([len(sub_elec_dict[_sub]) for _sub in subs]), -1)))
            results.append(np.stack(fold_results))
            joint_results.append(np.stack(fold_joint_results))

    valid_ks = [1, 2, 5, 10, 20, 40, 100, 200, 400, 800, 1600]
    final_ks = []
    final_results = []
    final_joint_results = []
    for valid_k in valid_ks:
        k_idx = ks.index(valid_k)
        final_ks.append(valid_k)
        final_results.append(results[k_idx])
        final_joint_results.append(joint_results[k_idx])

    k_idxs = np.argsort(final_ks)
    ks = np.array(final_ks)[k_idxs]
    vmin, vmax = min(ks), max(ks)
    data = np.array(final_results)[k_idxs]
    joint_corr_data = np.array(final_joint_results)[k_idxs]
    return data, joint_corr_data, ks, sub_elec_dict, valid_ks, vmax, vmin


@app.cell
def _(base_corrs_path, data, np, permutation_test_fdr, subs):
    base_corrs = np.concatenate([np.load(base_corrs_path.format(sub=sub)) for sub in subs], axis=1)
    sig_mask, adj_p_values = permutation_test_fdr(base_corrs.mean(0), data.mean(1))
    return base_corrs, sig_mask


@app.cell
def _(base_corrs, compare_multi_models_to_baseline, data):
    elec_sig_mask, elec_ps = compare_multi_models_to_baseline(base_corrs.mean(0), data.mean(1))
    elec_sig_mask
    return (elec_sig_mask,)


@app.cell
def _(base_corrs, cm, colors, data, ks, np, plt, sig_mask, vmax, vmin):
    n_models, n_folds, n_elecs, l_size = data.shape
    lags = np.linspace(-2.0, 2.0, l_size, endpoint=False)

    # 3. Create the normalization and colormap
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    colormap = cm.viridis  # You can use 'plasma', 'inferno', etc.

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Plot the baseline and models
    for i in range(n_models):
        ax.plot(lags, data[i].mean((0, 1)), color=colormap(norm(ks[i])), alpha=0.7)
    ax.plot(lags, base_corrs.mean((0, 1)), color="red", label="Baseline")

    # 2. Add Significance Markers
    # significant_lags shape: [n_models, n_lags]
    # We need to determine the vertical spacing
    y_min, y_max = ax.get_ylim()
    marker_base = y_max + (y_max - y_min) * 0.05  # Start 5% above the max data point
    spacing = (y_max - y_min) * 0.03              # Space between rows of stars

    # Get indices that would sort ks so we stack them correctly
    sorted_indices = np.argsort(ks)

    for rank, i in enumerate(sorted_indices):
        # Find the x-coordinates (lags) where this model is significant
        curr_sig_mask = sig_mask[i]

        if np.any(sig_mask):
            # Calculate the y-position for this model's row
            y_pos = marker_base + (rank * spacing)

            # Plot '*' at each significant lag
            ax.scatter(lags[curr_sig_mask], [y_pos] * np.sum(curr_sig_mask), 
                       marker='*', 
                       color=colormap(norm(ks[i])), 
                       s=20) # 's' controls star size

    # 3. Adjust limits to make room for the stars at the top
    ax.set_ylim(y_min, marker_base + (n_models * spacing))

    # 4. Colorbar and Labels
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, extend='both')
    cbar.set_label('Rank r of subspace')
    plt.legend()

    plt.show()
    return


@app.cell
def _(
    build_electrode_region_map,
    load_precomputed_epochs,
    sub_elec_dict,
    subs,
):
    brain_regions = {
        "EAC": [
            "L G_temp_sup-G_T_transv",
            "R G_temp_sup-G_T_transv",
            "L S_temporal_transverse",
            "R S_temporal_transverse",
            "L G_temp_sup-Plan_tempo",
            "R G_temp_sup-Plan_tempo",
        ],
        "STG": [
            "L G_temp_sup-Lateral",
            "R G_temp_sup-Lateral",
            "L S_temporal_sup",
            "R S_temporal_sup",
        ],
        "IFG": [
            "L G_front_inf-Opercular",
            "R G_front_inf-Opercular",
            "L G_front_inf-Triangul",
            "R G_front_inf-Triangul",
            "L G_front_inf-Orbital",
            "R G_front_inf-Orbital",
        ],
        "PRC": [
            "L G_precentral",
            "R G_precentral",
            "L S_precentral-inf-part",
            "R S_precentral-inf-part",
            "L S_precentral-sup-part",
            "R S_precentral-sup-part",
            "L G_and_S_subcentral",
            "R G_and_S_subcentral",
        ],
        "TMP": [
            "L Pole_temporal",
            "R Pole_temporal",
            "L G_temp_sup-Plan_polar",
            "R G_temp_sup-Plan_polar",
        ],
        "MTG": [
            "L G_temporal_middle",
            "R G_temporal_middle",
        ],
        # "SMG": [
        #     "L G_pariet_inf-Supramar",
        #     "R G_pariet_inf-Supramar",
        # ],
        # "ANG": [
        #     "L G_pariet_inf-Angular",
        #     "R G_pariet_inf-Angular",
        # ],
        # "INS": [
        #     "L G_insular_short",
        #     "R G_insular_short",
        #     "L G_Ins_lg_and_S_cent_ins",
        #     "R G_Ins_lg_and_S_cent_ins",
        #     "L S_circular_insula_ant",
        #     "L S_circular_insula_inf",
        #     "L S_circular_insula_sup",
        #     "R S_circular_insula_inf",
        #     "R S_circular_insula_sup",
        # ],
        # "SFG": [
        #     "L G_front_sup",
        #     "R G_front_sup",
        #     "L S_front_sup",
        # ],
        # "OTHER": [
        #     "L G_postcentral",
        #     "R G_postcentral",
        #     "L S_postcentral",
        #     "L S_central",
        #     "L G_parietal_sup",
        #     "R G_parietal_sup",
        #     "L S_intrapariet_and_P_trans",
        #     "L G_and_S_paracentral",
        #     "L G_and_S_transv_frontopol",
        #     "L G_orbital",
        #     "R G_orbital",
        #     "L S_orbital-H_Shaped",
        #     "L G_temporal_inf",
        #     "R G_temporal_inf",
        #     "L G_oc-temp_lat-fusifor",
        #     "R G_oc-temp_lat-fusifor",
        #     "L S_oc-temp_lat",
        #     "L S_collat_transv_ant",
        #     "R S_collat_transv_ant",
        #     "L G_oc-temp_med-Parahip",
        #     "R G_oc-temp_med-Parahip",
        #     "L S_oc-temp_med_and_Lingual",
        #     "R S_oc-temp_med_and_Lingual",
        #     "L G_occipital_middle",
        #     "R G_occipital_middle",
        #     "L G_occipital_sup",
        #     "R G_occipital_sup",
        #     "L Pole_occipital",
        #     "R Pole_occipital",
        #     "L G_and_S_occipital_inf",
        #     "R G_and_S_occipital_inf",
        #     "L S_calcarine",
        #     "R S_calcarine",
        #     "R S_parieto_occipital",
        #     "L G_and_S_cingul-Ant",
        #     "L G_and_S_cingul-Mid-Ant",
        #     "R G_and_S_cingul-Mid-Ant",
        #     "L G_cingul-Post-ventral",
        #     "R G_cingul-Post-ventral",
        #     "L S_cingul-Marginalis",
        #     "L G_precuneus",
        #     "L S_pericallosal",
        #     "L Lat_Fis-post",
        #     "R Lat_Fis-post",
        #     "L Lat_Fis-ant-Vertical",
        # ],
    }

    electrode_groups = build_electrode_region_map(subs, "data/", brain_regions, "atlases/destrieux_2009/destrieux2009_rois_lateralized.nii.gz")
    # Map electrodes to idxs in data
    _curr_elec_idx = 0
    _electrode_idx_mapping = {}
    for sub in sorted(subs):
        for elec_name in sub_elec_dict[sub]:
            if sub not in _electrode_idx_mapping:
                _electrode_idx_mapping[sub] = {}
            _electrode_idx_mapping[sub][elec_name] = _curr_elec_idx
            _curr_elec_idx += 1

    groups = {}
    for _region, _region_dict in electrode_groups.items():
        for _sub, _elec_names in _region_dict.items():
            _cache = load_precomputed_epochs(_sub, "data/")
            for _elec_name in _elec_names:
                if _elec_name not in sub_elec_dict[_sub]:
                    continue

                if _region not in groups:
                    groups[_region] = []

                groups[_region].append(_electrode_idx_mapping[_sub][_elec_name])
    return brain_regions, groups


@app.cell
def _(groups):
    {t: len(g) for t, g in groups.items()}
    return


@app.cell
def _(base_corrs, data, global_permutation_test, groups):
    region_p_vals_dict = {}
    for _region, _elecs in groups.items():
        _region_base_corrs = base_corrs[:, _elecs, :].mean(0)
        _region_compare_corrs = data[:, :, _elecs, :].mean(1)
        _region_p_vals, _ = global_permutation_test(_region_base_corrs, _region_compare_corrs)
        region_p_vals_dict[_region] = _region_p_vals
    return (region_p_vals_dict,)


@app.cell
def _(base_corrs, data, get_star_label, groups, ks, plt, region_p_vals_dict):
    def plot_all_regions_grid(groups, base_corrs, data, ks, global_p_values_dict):
        """
        Generates a grid of semilogx plots, one for each brain region.

        Args:
            groups: Dict of {region_name: electrode_indices}
            base_corrs: Baseline correlation array
            data: Comparator data array [models, folds, electrodes, targets]
            ks: List/array of rank values
            global_p_values_dict: Dict of {region_name: p_values_array}
        """
        # Define colors to match your style
        RED_BASELINE = "#e41a1c"
        VIRIDIS_MID = "#21918c"
        VIRIDIS_LIGHT = "#5ec962"

        num_regions = len(groups)
        cols = 3  # Adjust based on preference
        rows = (num_regions + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
        axes = axes.flatten()

        for i, (region, elecs) in enumerate(groups.items()):
            ax = axes[i]

            # 1. Extract data for this specific region
            # data shape assumed: (models, folds, electrodes, time/targets)
            region_data = data[:, :, elecs, :]
            region_base = base_corrs[:, elecs, :]

            mean_corrs = region_data.mean((1, 2, 3))
            fold_corrs = region_data.mean((2, 3)) # (models, folds)
            base_mean = region_base.mean()

            # 2. Plot individual folds (Background)
            for fold in range(fold_corrs.shape[1]):
                ax.semilogx(ks, fold_corrs[:, fold],
                            color=VIRIDIS_LIGHT, linewidth=0.5, alpha=0.4)

            # 3. Plot Mean line (Foreground)
            ax.semilogx(ks, mean_corrs, color=VIRIDIS_MID, marker="o", 
                        markersize=3, linewidth=1.2, label="RRR Space", zorder=5)

            # 4. Plot Baseline
            ax.axhline(base_mean, color=RED_BASELINE, ls="--", 
                       linewidth=1.0, label="Baseline")

            # 5. Significance stars
            y_lo, y_hi = ax.get_ylim()
            offset = (y_hi - y_lo) * 0.05
            p_vals = global_p_values_dict[region]

            for idx, (k, val) in enumerate(zip(ks, mean_corrs)):
                stars = get_star_label(p_vals[idx])
                if stars:
                    ax.text(k, val + offset, stars, ha="center", 
                            va="bottom", fontsize=7, color="k")

            # Formatting subplots
            ax.set_title(region, fontsize=10, fontweight='bold')
            if i % cols == 0:
                ax.set_ylabel("Correlation (r)")
            if i >= (num_regions - cols):
                ax.set_xlabel("Rank $k$")

        # Remove empty subplots if num_regions is not a multiple of cols
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        return fig

    plot_all_regions_grid(groups, base_corrs, data, ks, region_p_vals_dict)
    return


@app.cell(hide_code=True)
def _():
    DESTRIEUX_2009_LABELS: list[str] = [
        "Background",                       # 0
        "L G_and_S_frontomargin",          # 1
        "L G_and_S_occipital_inf",         # 2
        "L G_and_S_paracentral",           # 3
        "L G_and_S_subcentral",            # 4
        "L G_and_S_transv_frontopol",      # 5
        "L G_and_S_cingul-Ant",            # 6
        "L G_and_S_cingul-Mid-Ant",        # 7
        "L G_and_S_cingul-Mid-Post",       # 8
        "L G_cingul-Post-dorsal",          # 9
        "L G_cingul-Post-ventral",         # 10
        "L G_cuneus",                       # 11
        "L G_front_inf-Opercular",         # 12
        "L G_front_inf-Orbital",           # 13
        "L G_front_inf-Triangul",          # 14
        "L G_front_middle",                # 15
        "L G_front_sup",                   # 16
        "L G_Ins_lg_and_S_cent_ins",       # 17
        "L G_insular_short",               # 18
        "L G_occipital_middle",            # 19
        "L G_occipital_sup",               # 20
        "L G_oc-temp_lat-fusifor",         # 21
        "L G_oc-temp_med-Lingual",         # 22
        "L G_oc-temp_med-Parahip",         # 23
        "L G_orbital",                     # 24
        "L G_pariet_inf-Angular",          # 25
        "L G_pariet_inf-Supramar",         # 26
        "L G_parietal_sup",                # 27
        "L G_postcentral",                 # 28
        "L G_precentral",                  # 29
        "L G_precuneus",                   # 30
        "L G_rectus",                      # 31
        "L G_subcallosal",                 # 32
        "L G_temp_sup-G_T_transv",         # 33
        "L G_temp_sup-Lateral",            # 34
        "L G_temp_sup-Plan_polar",         # 35
        "L G_temp_sup-Plan_tempo",         # 36
        "L G_temporal_inf",                # 37
        "L G_temporal_middle",             # 38
        "L Lat_Fis-ant-Horizont",          # 39
        "L Lat_Fis-ant-Vertical",          # 40
        "L Lat_Fis-post",                  # 41
        "L Medial_wall",                   # 42
        "L Pole_occipital",                # 43
        "L Pole_temporal",                 # 44
        "L S_calcarine",                   # 45
        "L S_central",                     # 46
        "L S_cingul-Marginalis",           # 47
        "L S_circular_insula_ant",         # 48
        "L S_circular_insula_inf",         # 49
        "L S_circular_insula_sup",         # 50
        "L S_collat_transv_ant",           # 51
        "L S_collat_transv_post",          # 52
        "L S_front_inf",                   # 53
        "L S_front_middle",                # 54
        "L S_front_sup",                   # 55
        "L S_interm_prim-Jensen",          # 56
        "L S_intrapariet_and_P_trans",     # 57
        "L S_oc_middle_and_Lunatus",       # 58
        "L S_oc_sup_and_transversal",      # 59
        "L S_occipital_ant",               # 60
        "L S_oc-temp_lat",                 # 61
        "L S_oc-temp_med_and_Lingual",     # 62
        "L S_orbital_lateral",             # 63
        "L S_orbital_med-olfact",          # 64
        "L S_orbital-H_Shaped",            # 65
        "L S_parieto_occipital",           # 66
        "L S_pericallosal",                # 67
        "L S_postcentral",                 # 68
        "L S_precentral-inf-part",         # 69
        "L S_precentral-sup-part",         # 70
        "L S_suborbital",                  # 71
        "L S_subparietal",                 # 72
        "L S_temporal_inf",                # 73
        "L S_temporal_sup",                # 74
        "L S_temporal_transverse",         # 75
        "R G_and_S_frontomargin",          # 76
        "R G_and_S_occipital_inf",         # 77
        "R G_and_S_paracentral",           # 78
        "R G_and_S_subcentral",            # 79
        "R G_and_S_transv_frontopol",      # 80
        "R G_and_S_cingul-Ant",            # 81
        "R G_and_S_cingul-Mid-Ant",        # 82
        "R G_and_S_cingul-Mid-Post",       # 83
        "R G_cingul-Post-dorsal",          # 84
        "R G_cingul-Post-ventral",         # 85
        "R G_cuneus",                       # 86
        "R G_front_inf-Opercular",         # 87
        "R G_front_inf-Orbital",           # 88
        "R G_front_inf-Triangul",          # 89
        "R G_front_middle",                # 90
        "R G_front_sup",                   # 91
        "R G_Ins_lg_and_S_cent_ins",       # 92
        "R G_insular_short",               # 93
        "R G_occipital_middle",            # 94
        "R G_occipital_sup",               # 95
        "R G_oc-temp_lat-fusifor",         # 96
        "R G_oc-temp_med-Lingual",         # 97
        "R G_oc-temp_med-Parahip",         # 98
        "R G_orbital",                     # 99
        "R G_pariet_inf-Angular",          # 100
        "R G_pariet_inf-Supramar",         # 101
        "R G_parietal_sup",                # 102
        "R G_postcentral",                 # 103
        "R G_precentral",                  # 104
        "R G_precuneus",                   # 105
        "R G_rectus",                      # 106
        "R G_subcallosal",                 # 107
        "R G_temp_sup-G_T_transv",         # 108
        "R G_temp_sup-Lateral",            # 109
        "R G_temp_sup-Plan_polar",         # 110
        "R G_temp_sup-Plan_tempo",         # 111
        "R G_temporal_inf",                # 112
        "R G_temporal_middle",             # 113
        "R Lat_Fis-ant-Horizont",          # 114
        "R Lat_Fis-ant-Vertical",          # 115
        "R Lat_Fis-post",                  # 116
        "R Medial_wall",                   # 117
        "R Pole_occipital",                # 118
        "R Pole_temporal",                 # 119
        "R S_calcarine",                   # 120
        "R S_central",                     # 121
        "R S_cingul-Marginalis",           # 122
        "R S_circular_insula_ant",         # 123
        "R S_circular_insula_inf",         # 124
        "R S_circular_insula_sup",         # 125
        "R S_collat_transv_ant",           # 126
        "R S_collat_transv_post",          # 127
        "R S_front_inf",                   # 128
        "R S_front_middle",                # 129
        "R S_front_sup",                   # 130
        "R S_interm_prim-Jensen",          # 131
        "R S_intrapariet_and_P_trans",     # 132
        "R S_oc_middle_and_Lunatus",       # 133
        "R S_oc_sup_and_transversal",      # 134
        "R S_occipital_ant",               # 135
        "R S_oc-temp_lat",                 # 136
        "R S_oc-temp_med_and_Lingual",     # 137
        "R S_orbital_lateral",             # 138
        "R S_orbital_med-olfact",          # 139
        "R S_orbital-H_Shaped",            # 140
        "R S_parieto_occipital",           # 141
        "R S_pericallosal",                # 142
        "R S_postcentral",                 # 143
        "R S_precentral-inf-part",         # 144
        "R S_precentral-sup-part",         # 145
        "R S_suborbital",                  # 146
        "R S_subparietal",                 # 147
        "R S_temporal_inf",                # 148
        "R S_temporal_sup",                # 149
        "R S_temporal_transverse",         # 150
    ]
    return (DESTRIEUX_2009_LABELS,)


@app.cell(hide_code=True)
def _(DESTRIEUX_2009_LABELS: list[str], brain_regions, np, os, subs):
    from scipy.spatial import KDTree
    import mne
    from mne_bids import BIDSPath
    from nilearn import datasets, image as nli_image

    def _lookup_atlas_labels(
        coords: np.ndarray,
        atlas_image: np.ndarray,
        affine: np.ndarray,
        atlas_labels: list[str],
    ) -> list[str]:
        """Map MNI coordinates to atlas labels using nearest-neighbor lookup.

        Pure numpy/scipy implementation — no nilearn required. Suitable for unit tests.

        Args:
            coords: (n_elecs, 3) array of MNI coordinates in mm.
            atlas_image: 3-D integer array (voxel label indices).
            affine: (4, 4) affine matrix mapping voxel indices to MNI space.
            atlas_labels: List of label strings indexed by atlas integer value (0 = background).

        Returns:
            List of atlas label strings, one per electrode.
        """
        image_label_coords = np.nonzero(atlas_image)  # (i_arr, j_arr, k_arr) for non-zero voxels
        i_arr, j_arr, k_arr = image_label_coords
        n = len(i_arr)

        # Transform voxel indices to MNI space using affine (no nilearn needed)
        vox_homo = np.vstack([
            i_arr.astype(float),
            j_arr.astype(float),
            k_arr.astype(float),
            np.ones(n),
        ])  # (4, n)
        mni = (affine @ vox_homo)[:3].T  # (n, 3)

        tree = KDTree(mni)
        _, nearest_neighbor = tree.query(coords, k=1)

        x = image_label_coords[0][nearest_neighbor]
        y = image_label_coords[1][nearest_neighbor]
        z = image_label_coords[2][nearest_neighbor]

        elec_label_ids = atlas_image[x, y, z]
        return [atlas_labels[i] for i in elec_label_ids]

    img = nli_image.load_img("atlases/destrieux_2009/destrieux2009_rois_lateralized.nii.gz")
    atlas_image = img.get_fdata().astype(int)
    affine = img.affine

    per_subject_data = {}
    for sub_id in subs:
        file_path = BIDSPath(
            root=os.path.join("data/", "derivatives/ecogprep"),
            subject=f"{sub_id:02}",
            task="podcast",
            datatype="ieeg",
            description="highgamma",
            suffix="ieeg",
            extension=".fif",
        )
        raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)
        coords = np.array([ch["loc"][:3] * 1000 for ch in raw.info["chs"]])
        per_subject_data[sub_id] = (raw.ch_names, coords)

    all_labels = []
    for sub_id, (elec_names, coords) in per_subject_data.items():
        all_labels += _lookup_atlas_labels(coords, atlas_image, affine, DESTRIEUX_2009_LABELS)

    not_covered = []
    for label in all_labels:
        in_any = False
        for _group, _regions in brain_regions.items():
            if label in _regions:
                in_any = True

        if not in_any and label not in not_covered:
            not_covered.append(label)
    print(not_covered)
    return


@app.cell(hide_code=True)
def _(base_corrs, cm, colors, data, elec_sig_mask, groups, ks, np, plt):
    def _():
        from matplotlib.gridspec import GridSpec

        n_models, n_folds, n_elecs, l_size = data.shape

        # ── Style (match panel b) ────────────────────────────────────────────
        _raw = cm.get_cmap("viridis", 256)
        CMAP = colors.LinearSegmentedColormap.from_list(
            "viridis_trunc", _raw(np.linspace(0.08, 0.92, 256))
        )
        RED_BASELINE = "#d62728"

        norm = colors.LogNorm(vmin=min(ks), vmax=max(ks))

        # ── Aggregate across folds and lags → [n_models, n_elecs] ────────────
        model_means = data.mean((1, 3))        # shape: [n_models, n_elecs]
        base_means  = base_corrs.mean((0, 2))  # shape: [n_elecs]

        # ── Build electrode ordering from groups dict ────────────────────────
        # groups: dict[str, list[int]]  e.g. {"Group A": [0, 3, 5], "Group B": [1, 2]}
        # If groups is not defined, fall back to a single group with all electrodes
        if "groups" not in dir() and "groups" not in globals():
            _groups = {"All Electrodes": list(range(n_elecs))}
        else:
            _groups = groups

        ordered_indices = []       # final electrode ordering
        group_boundaries = []      # (start_x, end_x, label) for each group
        gap = 3                    # x-gap between groups

        pos = 0
        for group_name, elec_ids in _groups.items():
            # Sort electrodes within group by baseline performance (descending)
            elec_ids_sorted = sorted(elec_ids, key=lambda e: base_means[e], reverse=True)
            start = pos
            ordered_indices.extend(elec_ids_sorted)
            pos += len(elec_ids_sorted)
            group_boundaries.append((start, pos - 1, group_name))
            pos += gap  # leave a gap before next group

        n_bars = pos - gap  # remove trailing gap
        ordered_indices = np.array(ordered_indices)

        # Build x-positions with gaps between groups
        x = np.empty(len(ordered_indices), dtype=int)
        offset = 0
        elec_cursor = 0
        for start, end, _ in group_boundaries:
            n_in_group = end - start + 1
            x[elec_cursor:elec_cursor + n_in_group] = np.arange(n_in_group) + offset
            offset += n_in_group + gap
            elec_cursor += n_in_group

        # Reorder data arrays
        model_means_sorted = model_means[:, ordered_indices]
        base_means_sorted  = base_means[ordered_indices]
        elec_sig_sorted    = elec_sig_mask[:, ordered_indices]

        # ── Figure ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))

        # Draw models back-to-front: tallest bars first
        model_order = np.argsort([-model_means_sorted[i].mean() for i in range(n_models)])
        bar_width = 1.0

        for i in model_order:
            ax.bar(x, model_means_sorted[i], width=bar_width,
                   color=CMAP(norm(ks[i])), alpha=0.45,
                   linewidth=0, label=f"k = {ks[i]}")

        # Baseline on top
        ax.bar(x, base_means_sorted, width=bar_width,
               color=RED_BASELINE, alpha=0.55,
               linewidth=0, label="Full Model (Baseline)")

        # ── Significance markers ─────────────────────────────────────────────
        y_min, y_max = ax.get_ylim()
        marker_base = y_max + (y_max - y_min) * 0.01
        spacing = (y_max - y_min) * 0.025

        sorted_model_indices = np.argsort(ks)

        for rank, mi in enumerate(sorted_model_indices):
            mask = elec_sig_sorted[mi]
            if np.any(mask):
                y_pos = marker_base + rank * spacing
                ax.scatter(x[mask], np.full(mask.sum(), y_pos),
                           marker="*", s=10,
                           color=CMAP(norm(ks[mi])),
                           linewidths=0, zorder=10)

        ax.set_ylim(y_min, marker_base + n_models * spacing + spacing * 0.5)

        # ── Group labels & separators ────────────────────────────────────────
        x_cursor = 0
        for start, end, label in group_boundaries:
            n_in_group = end - start + 1
            center = x_cursor + (n_in_group - 1) / 2
            ax.text(center, y_min - (y_max - y_min) * 0.06, label,
                    ha="center", va="top", fontsize=8, fontweight="bold")

            # Vertical separator between groups (skip before first)
            if x_cursor > 0:
                sep_x = x_cursor - gap / 2
                ax.axvline(sep_x, color="#d1d5db", linewidth=0.6, ls=":",
                           ymin=0, ymax=0.95, zorder=0)

            x_cursor += n_in_group + gap

        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # ── Labels & cosmetics ───────────────────────────────────────────────
        ax.set_xlabel("Electrode (grouped, sorted by baseline within group)")
        ax.set_ylabel("Correlation (r)")
        ax.set_title("Per-Electrode Correlation by Subspace Rank", fontsize=10, pad=8)

        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles, labels_, loc="upper right", fontsize=7, ncol=2)

        # Colorbar
        sm = cm.ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, extend="both", pad=0.02, aspect=30)
        cbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
        cbar.set_label("Rank $k$", fontsize=8, labelpad=1)
        cbar.outline.set_linewidth(0.6)

        plt.savefig("figure.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.savefig("figure.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
        plt.show()

    _()
    return


@app.cell
def _(base_corrs, data, global_permutation_test):
    global_p_values, global_obs_diff = global_permutation_test(base_corrs.mean(0), data.mean(1))
    return (global_p_values,)


@app.cell
def _(global_p_values):
    global_p_values
    return


@app.cell
def _(base_corrs, data, get_star_label, global_p_values, ks, plt):
    mean_corrs = data.mean((1, 2, 3))
    fold_corrs = data.mean((2, 3))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(ks, mean_corrs, color="blue", linestyle="-",
                 marker="o", markersize=4, label="RRR Space", linewidth=1.5)

    for fold in range(fold_corrs.shape[1]):
        plt.semilogx(ks, fold_corrs[:, fold], color="blue", alpha=0.2, linewidth=0.8)

    plt.axhline(base_corrs.mean((0, 1, 2)), color="#9ca3af", ls="--", label="Full Model Correlation (Baseline)")

    # --- Significance Annotation Logic ---
    # Adjust 'offset' based on the scale of your y-axis (e.g., 0.02 for correlations)
    offset = 0.001

    for _i, (k, val) in enumerate(zip(ks, mean_corrs)):
        stars = get_star_label(global_p_values[_i])
        if stars:
            plt.text(k, val + offset, stars,
                     ha='center', va='bottom', color='black', fontsize=12)

    plt.xlabel("k (Log Scale)")
    plt.ylabel("Correlation")
    plt.legend()
    plt.show()
    return (offset,)


@app.cell
def _(
    apply_publication_style,
    base_corrs,
    data,
    global_p_values,
    ks,
    make_viridis_cmap,
    np,
    plot_cross_k_panel,
    plot_lag_panel,
    plt,
    sig_mask,
):
    def _():
        from matplotlib.gridspec import GridSpec

        apply_publication_style()
        CMAP = make_viridis_cmap()

        fig = plt.figure(figsize=(7.2, 3.4))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.25],
                      wspace=0.35, left=0.08, right=0.97, bottom=0.15, top=0.92)

        # =====================================================================
        # Panel (a) — Mean Correlation vs. Subspace Rank
        # =====================================================================
        ax_a = fig.add_subplot(gs[0])
        plot_cross_k_panel(
            ax_a, ks,
            mean_corrs=data.mean((1, 2, 3)),
            fold_corrs=data.mean((2, 3)),
            p_values=global_p_values,
            base_mean=base_corrs.mean((0, 1, 2)),
        )
        ax_a.set_title("Mean Correlation vs. Subspace Rank", fontsize=9, pad=8)
        ax_a.text(-0.12, 1.05, "a", transform=ax_a.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")

        # =====================================================================
        # Panel (b) — Correlation by Lag and Subspace Rank
        # =====================================================================
        ax_b = fig.add_subplot(gs[1])
        lags = np.linspace(-2.0, 2.0, data.shape[-1], endpoint=False)
        plot_lag_panel(
            ax_b, lags, data, ks,
            base_corrs_mean=base_corrs.mean((0, 1)),
            sig_mask=sig_mask,
            saturation_k=20,
            cmap=CMAP,
            fig=fig,
        )
        ax_b.text(-0.12, 1.05, "b", transform=ax_b.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")

        plt.savefig("figure.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.savefig("figure.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
        plt.show()

    _()
    return


@app.cell
def _(
    apply_publication_style,
    base_corrs,
    data,
    get_star_label,
    global_p_values,
    groups,
    ks,
    make_viridis_cmap,
    np,
    plot_cross_k_panel,
    plot_lag_panel,
    plt,
    region_p_vals_dict,
    sig_mask,
):
    def _():
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        apply_publication_style()
        CMAP = make_viridis_cmap()

        # ── Shared palette ───────────────────────────────────────────────────────
        RED_BASELINE = "#d62728"
        VIRIDIS_MID  = CMAP(0.45)
        VIRIDIS_LIGHT = (*CMAP(0.45)[:3], 0.3)
        GRAY = "#6b7280"

        # ── Figure layout ────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(9.5, 3.5))
        gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.1],
                        wspace=0.25, left=0.07, right=0.96, bottom=0.10, top=0.95)

        gs_left = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[0],
                                      height_ratios=[1, 1.2], hspace=0.48)

        # =====================================================================
        # Panel (a) — Mean Correlation vs. Subspace Rank (top-left)
        # =====================================================================
        ax_a = fig.add_subplot(gs_left[0])
        mean_corrs = data.mean((1, 2, 3))
        fold_corrs = data.mean((2, 3))
        base_mean = base_corrs.mean((0, 1, 2))
        plot_cross_k_panel(
            ax_a, ks,
            mean_corrs=mean_corrs,
            fold_corrs=fold_corrs,
            p_values=global_p_values,
            base_mean=base_mean,
        )
        ax_a.set_title("Mean Correlation vs. Subspace Rank", fontsize=9, pad=8)
        ax_a.set_ylim(bottom=0.0)
        ax_a.text(-0.12, 1.05, "a", transform=ax_a.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")

        # =====================================================================
        # Panel (b) — Region Grid (bottom-left)
        # =====================================================================
        num_regions = len(groups)
        cols_b = 3
        rows_b = (num_regions + cols_b - 1) // cols_b

        gs_b = GridSpecFromSubplotSpec(rows_b, cols_b, subplot_spec=gs_left[1],
                                   hspace=0.29, wspace=0.1)

        region_items = list(groups.items())
        first_region_ax = None
        for idx in range(rows_b * cols_b):
            ax = fig.add_subplot(gs_b[idx])
            if idx == 0:
                first_region_ax = ax
            if idx >= num_regions:
                ax.set_visible(False)
                continue

            region, elecs = region_items[idx]

            region_data = data[:, :, elecs, :]
            region_base = base_corrs[:, elecs, :]

            r_mean_corrs = region_data.mean((1, 2, 3))
            r_fold_corrs = region_data.mean((2, 3))
            r_base_mean = region_base.mean()

            for fold in range(r_fold_corrs.shape[1]):
                ax.semilogx(ks, r_fold_corrs[:, fold],
                            color=VIRIDIS_LIGHT, linewidth=0.5)

            ax.semilogx(ks, r_mean_corrs, color=VIRIDIS_MID, marker="o",
                        markersize=2, linewidth=1.0, zorder=5)

            ax.axhline(r_base_mean, color=RED_BASELINE, ls="--", linewidth=0.8)
            ax.text(0.95, 0.2, f"n={len(elecs)}", transform=ax.transAxes,
                fontsize=5, ha="right", va="top", color="gray")

            y_lo, y_hi = ax.get_ylim()
            off = (y_hi - y_lo) * 0.05
            p_vals = region_p_vals_dict[region]
            for pidx, (k, val) in enumerate(zip(ks, r_mean_corrs)):
                stars = get_star_label(p_vals[pidx])
                if stars:
                    ax.text(k, val + off, stars, ha="center",
                            va="bottom", fontsize=5, color="k")

            ax.set_title(region, fontsize=7, fontweight="bold", pad=3)
            ax.tick_params(labelsize=6)

            if idx % cols_b == 0:
                ax.set_ylabel("r", fontsize=7)
            else:
                ax.set_yticklabels([])
            if idx >= num_regions - cols_b:
                ax.set_xlabel("Rank $k$", fontsize=7)
            else:
                ax.set_xticklabels([])

        first_region_ax.text(-0.30, 1.15, "b", transform=first_region_ax.transAxes,
                             fontsize=12, fontweight="bold", va="bottom")

        # =====================================================================
        # Panel (c) — Correlation by Lag and Subspace Rank (right, full height)
        # =====================================================================
        ax_c = fig.add_subplot(gs_outer[1])
        lags = np.linspace(-2.0, 2.0, data.shape[-1], endpoint=False)
        plot_lag_panel(
            ax_c, lags, data, ks,
            base_corrs_mean=base_corrs.mean((0, 1)),
            sig_mask=sig_mask,
            saturation_k=20,
            cmap=CMAP,
            fig=fig,
        )
        ax_c.text(-0.10, 1.02, "c", transform=ax_c.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")

        plt.savefig("figure.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.savefig("figure.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
        plt.show()

    _()
    return


@app.cell
def _(base_result_dir, os, pd, valid_ks):
    joint_data = pd.read_csv(os.path.join(base_result_dir, "grid_results.csv"))
    joint_data = joint_data[joint_data["k"].isin(valid_ks)]
    # TODO: support permutation testing here by writing per-target correlations for our OLS models
    # global_permutation_test()
    return (joint_data,)


@app.cell
def _(joint_corr_data, ks, np):
    joint_corr_data[np.where(ks == 20)]
    return


@app.cell
def _(data, global_permutation_test, joint_corr_data, ks, np):
    all_p_joint = []
    for _k in ks:
        p_values_joint, obs_diffs_joint = global_permutation_test(data[np.where(ks == _k)].mean((0, 1)), joint_corr_data[np.where(ks == _k)].mean((1)))
        all_p_joint.append(p_values_joint)
    all_p_joint = np.stack(all_p_joint)
    all_p_joint
    return (all_p_joint,)


@app.cell
def _(all_p_joint, cm, colors, get_star_label, joint_data, np, offset, plt):
    def _():
        # ── Style setup (same as your existing figure) ──────────────────────────
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 300,
        })

        # ── Colormap (same truncated viridis) ────────────────────────────────────
        _raw = cm.get_cmap("viridis", 256)
        CMAP = colors.LinearSegmentedColormap.from_list(
            "viridis_trunc", _raw(np.linspace(0.08, 0.92, 256))
        )

        # ── Colors: Subspace + Complement visually "mix" into Joint ─────────────
        CLR_SUBSPACE   = CMAP(0.45)                    # Teal-green (same as panel a)
        CLR_COMPLEMENT = CMAP(0.82)                    # Light yellow-green (lightest)
        CLR_JOINT      = CMAP(0.63)                    # Mid viridis — between the two

        # ── Data ─────────────────────────────────────────────────────────────────
        _data_subset = joint_data[["k", "fold", "nr_corr", "nr_complement_corr", "nr_complement_joint_corr"]]
        _data_subset = _data_subset.sort_values("k")

        # ── Figure ───────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        fig.subplots_adjust(left=0.16, right=0.95, bottom=0.15, top=0.92)

        ks_sorted = sorted(_data_subset["k"].unique())
        mean_results = _data_subset.groupby("k").agg({
            "nr_corr": "mean", 
            "nr_complement_corr": "mean", 
            "nr_complement_joint_corr": "mean"
        }).reindex(ks_sorted)
        ax.semilogx(mean_results.index, mean_results["nr_corr"],
                    color=CLR_SUBSPACE, marker="o", markersize=3.5,
                    linewidth=1.4, label="RRR Subspace", zorder=4)

        ax.semilogx(mean_results.index, mean_results["nr_complement_corr"],
                    color=CLR_COMPLEMENT, marker="s", markersize=3.5,
                    linewidth=1.4, label="Complement", zorder=4)

        ax.semilogx(mean_results.index, mean_results["nr_complement_joint_corr"],
                    color=CLR_JOINT, marker="D", markersize=3.5,
                    linewidth=1.4, label="RRR Subspace + Complement",
                    ls="--", zorder=5)

        # Individual folds — light version of the viridis midpoint
        for fold in _data_subset["fold"].unique():
            fold_subset = _data_subset[_data_subset["fold"] == fold]
            ax.semilogx(fold_subset["k"], fold_subset["nr_corr"],
                          color=CLR_SUBSPACE, linewidth=0.7, alpha=0.2)
            ax.semilogx(fold_subset["k"], fold_subset["nr_complement_corr"],
                          color=CLR_COMPLEMENT, linewidth=0.7, alpha=0.2)
            ax.semilogx(fold_subset["k"], fold_subset["nr_complement_joint_corr"],
                          color=CLR_JOINT, linewidth=0.7, alpha=0.2)

        # Assuming 'joint_p_values' is your array of shape [len(ks)]
        for i, k in enumerate(ks_sorted):
            stars = get_star_label(all_p_joint[i])
            if stars:
                # Position above the highest point at this K (usually the joint model)
                y_val = mean_results.loc[k, "nr_complement_joint_corr"]
                ax.text(k, y_val + offset, stars,
                        ha="center", va="bottom", fontsize=8, color="k")

        ax.set_xlabel("Rank $k$ (log scale)")
        ax.set_ylabel("Correlation (r)")
        ax.set_title("Joint Model Correlation vs. Subspace Rank", fontsize=9, pad=8)
        ax.legend(loc="center right", bbox_to_anchor=(1.05, 0.65), markerfirst=False, fontsize=7)

        plt.savefig("figure_joint.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.savefig("figure_joint.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
        return plt.show()


    _()
    return


@app.cell
def _(
    all_p_joint,
    apply_publication_style,
    base_corrs,
    data,
    get_star_label,
    global_p_values,
    groups,
    joint_data,
    ks,
    make_viridis_cmap,
    np,
    plot_cross_k_panel,
    plot_lag_panel,
    plt,
    region_p_vals_dict,
    sig_mask,
):
    def _():
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        apply_publication_style()
        CMAP = make_viridis_cmap()

        # ── Shared palette ───────────────────────────────────────────────────────
        RED_BASELINE = "#d62728"
        VIRIDIS_MID  = CMAP(0.45)
        VIRIDIS_LIGHT = (*CMAP(0.45)[:3], 0.3)
        GRAY = "#6b7280"

        # Panel (d) colors
        CLR_SUBSPACE   = CMAP(0.45)
        CLR_COMPLEMENT = CMAP(0.82)
        CLR_JOINT      = CMAP(0.63)

        # ── Figure layout ────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 3.5))
        gs_outer = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.1, 0.85],
                            wspace=0.32, left=0.05, right=0.97, bottom=0.10, top=0.95)

        gs_left = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[0],
                                          height_ratios=[1, 1.2], hspace=0.48)

        # =====================================================================
        # Panel (a) — Mean Correlation vs. Subspace Rank (top-left)
        # =====================================================================
        ax_a = fig.add_subplot(gs_left[0])
        mean_corrs = data.mean((1, 2, 3))
        fold_corrs = data.mean((2, 3))
        base_mean = base_corrs.mean((0, 1, 2))

        # Draw vertical line at the first k where RRR beats the baseline.
        for i, gt_check in enumerate(mean_corrs > base_mean):
            if gt_check:
                ax_a.axvline(x=ks[i], ls="--", color=GRAY, alpha=0.5, linewidth=1.0)
                break

        plot_cross_k_panel(
            ax_a, ks,
            mean_corrs=mean_corrs,
            fold_corrs=fold_corrs,
            p_values=global_p_values,
            base_mean=base_mean,
        )
        ax_a.set_title("Mean Correlation vs. Subspace Rank", fontsize=9, pad=8)
        ax_a.set_ylim(bottom=0.0)
        ax_a.text(-0.12, 1.05, "a", transform=ax_a.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")

        # =====================================================================
        # Panel (b) — Region Grid (bottom-left)
        # =====================================================================
        num_regions = len(groups)
        cols_b = 3
        rows_b = (num_regions + cols_b - 1) // cols_b

        gs_b = GridSpecFromSubplotSpec(rows_b, cols_b, subplot_spec=gs_left[1],
                                       hspace=0.29, wspace=0.1)

        region_items = list(groups.items())
        first_region_ax = None
        for idx in range(rows_b * cols_b):
            ax = fig.add_subplot(gs_b[idx])
            if idx == 0:
                first_region_ax = ax
            if idx >= num_regions:
                ax.set_visible(False)
                continue

            region, elecs = region_items[idx]

            region_data = data[:, :, elecs, :]
            region_base = base_corrs[:, elecs, :]

            r_mean_corrs = region_data.mean((1, 2, 3))
            r_fold_corrs = region_data.mean((2, 3))
            r_base_mean = region_base.mean()

            # Draw vertical line where we first pass the base mean.
            min_better = -1
            for i, gt_check in enumerate(r_mean_corrs > r_base_mean):
                if gt_check:
                    min_better = i
                    break

            if min_better >= 0:
                ax.axvline(x=ks[min_better], ls="--", color=GRAY, alpha=0.5, linewidth=0.8)

            for fold in range(r_fold_corrs.shape[1]):
                ax.semilogx(ks, r_fold_corrs[:, fold],
                            color=VIRIDIS_LIGHT, linewidth=0.5)

            ax.semilogx(ks, r_mean_corrs, color=VIRIDIS_MID, marker="o",
                        markersize=2, linewidth=1.0, zorder=5)

            ax.axhline(r_base_mean, color=RED_BASELINE, ls="--", linewidth=0.8)
            ax.text(0.95, 0.2, f"n={len(elecs)}", transform=ax.transAxes,
                    fontsize=5, ha="right", va="top", color="gray")

            y_lo, y_hi = ax.get_ylim()
            off = (y_hi - y_lo) * 0.05
            p_vals = region_p_vals_dict[region]
            for pidx, (k, val) in enumerate(zip(ks, r_mean_corrs)):
                stars = get_star_label(p_vals[pidx])
                if stars:
                    ax.text(k, val + off, stars, ha="center",
                            va="bottom", fontsize=5, color="k")

            ax.set_title(region, fontsize=7, fontweight="bold", pad=3)
            ax.tick_params(labelsize=6)

            if idx % cols_b == 0:
                ax.set_ylabel("r", fontsize=7)
            else:
                ax.set_yticklabels([])
            if idx >= num_regions - cols_b:
                ax.set_xlabel("Rank $k$", fontsize=7)
            else:
                ax.set_xticklabels([])

        first_region_ax.text(-0.30, 1.15, "b", transform=first_region_ax.transAxes,
                             fontsize=12, fontweight="bold", va="bottom")

        # =====================================================================
        # Panel (c) — Correlation by Lag and Subspace Rank (middle)
        # =====================================================================
        ax_c = fig.add_subplot(gs_outer[1])
        lags = np.linspace(-2.0, 2.0, data.shape[-1], endpoint=False)
        plot_lag_panel(
            ax_c, lags, data, ks,
            base_corrs_mean=base_corrs.mean((0, 1)),
            sig_mask=sig_mask,
            saturation_k=20,
            cmap=CMAP,
            fig=fig,
        )
        ax_c.text(-0.10, 1.02, "c", transform=ax_c.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")

        # =====================================================================
        # Panel (d) — Joint Model Correlation vs. Subspace Rank (right)
        # =====================================================================
        ax_d = fig.add_subplot(gs_outer[2])

        _data_subset = joint_data[["k", "fold", "nr_corr", "nr_complement_corr", "nr_complement_joint_corr"]]
        _data_subset = _data_subset.sort_values("k")

        ks_sorted = sorted(_data_subset["k"].unique())
        mean_results = _data_subset.groupby("k").agg({
            "nr_corr": "mean",
            "nr_complement_corr": "mean",
            "nr_complement_joint_corr": "mean"
        }).reindex(ks_sorted)

        ax_d.semilogx(mean_results.index, mean_results["nr_corr"],
                      color=CLR_SUBSPACE, marker="o", markersize=3.5,
                      linewidth=1.4, label="RRR Subspace", zorder=4)

        ax_d.semilogx(mean_results.index, mean_results["nr_complement_corr"],
                      color=CLR_COMPLEMENT, marker="s", markersize=3.5,
                      linewidth=1.4, label="Complement", zorder=4)

        ax_d.semilogx(mean_results.index, mean_results["nr_complement_joint_corr"],
                      color=CLR_JOINT, marker="D", markersize=3.5,
                      linewidth=1.4, label="RRR Subspace + Complement",
                      ls="--", zorder=5)

        ax_d.axhline(base_mean, color=RED_BASELINE, ls="--", linewidth=1.0,
                     label="Full Model (Baseline)")

        # Individual folds
        for fold in _data_subset["fold"].unique():
            fold_subset = _data_subset[_data_subset["fold"] == fold]
            ax_d.semilogx(fold_subset["k"], fold_subset["nr_corr"],
                          color=CLR_SUBSPACE, linewidth=0.7, alpha=0.2)
            ax_d.semilogx(fold_subset["k"], fold_subset["nr_complement_corr"],
                          color=CLR_COMPLEMENT, linewidth=0.7, alpha=0.2)
            ax_d.semilogx(fold_subset["k"], fold_subset["nr_complement_joint_corr"],
                          color=CLR_JOINT, linewidth=0.7, alpha=0.2)

        # Significance stars
        y_lo_d, y_hi_d = ax_d.get_ylim()
        offset_d = (y_hi_d - y_lo_d) * 0.03
        for i, k in enumerate(ks_sorted):
            stars = get_star_label(all_p_joint[i])
            if stars:
                y_val = mean_results.loc[k, "nr_complement_joint_corr"]
                ax_d.text(k, y_val + offset_d, stars,
                          ha="center", va="bottom", fontsize=8, color="k")

        ax_d.set_xlabel("Rank $k$ (log scale)")
        ax_d.set_ylabel("Correlation (r)")
        ax_d.set_title("Joint Model Correlation vs. Subspace Rank", fontsize=9, pad=8)
        ax_d.legend(loc="center right", bbox_to_anchor=(1.05, 0.45),
                    markerfirst=False, fontsize=7)

        ax_d.text(-0.10, 1.02, "d", transform=ax_d.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")

        # ── Save ─────────────────────────────────────────────────────────────────
        plt.savefig("figure.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.savefig("figure.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
        plt.show()

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
