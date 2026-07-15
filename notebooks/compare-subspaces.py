import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    _cwd = Path.cwd().resolve()
    _root = _cwd
    while _root != _root.parent:
        if (_root / "src").is_dir():
            break
        _root = _root.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from src.core.config import (
        RRRConfig,
        ScriptMode,
        SubspaceComparisonConfig,
        get_results_dir,
    )
    from src.core.registry import get_dataset_adapter, resolve_dataset_context
    from src.utils.config_utils import expand_per_model_layer_configs, load_config
    from src.utils.data_loaders import build_XY_from_subjects, load_sig_electrodes
    from src.utils.embeddings import get_glove_embeddings, load_embeddings
    from src.utils.encoding import load_encoding_results
    from src.utils.subspace_compare import (
        default_cross_rank_text,
        discover_subspace_sources,
        filter_available_ranks,
        fit_regularized_maxvar_gcca,
        is_pca_source_kind,
        align_maxvar_folds,
        maxvar_block_overlap,
        maxvar_component_blocks,
        maxvar_ambient_map,
        orthonormal_basis,
        parse_rank_list,
        principal_angles_from_bases,
        projected_rsm,
        rsa_lower_triangle_correlations,
        sample_random_principal_angles,
        score_cca_alignment,
        score_maxvar_leave_one_view_out,
        score_maxvar_partial_support,
        score_procrustes_alignment,
        summarize_cca_fold_results,
        select_maxvar_alpha,
        subspace_distance_metrics,
        summarize_procrustes_fold_results,
        summarize_rsa_fold_results,
        validate_matching_fold_indices,
    )
    from src.utils.subspace_io import (
        load_base_comparison_data,
        load_fold_bases,
        load_peer_embeddings_on_base_rows,
        max_available_rank,
        resolve_subspace_run,
        run_metadata,
        sanitize_name,
    )

    return (
        Path,
        RRRConfig,
        ScriptMode,
        SubspaceComparisonConfig,
        align_maxvar_folds,
        build_XY_from_subjects,
        default_cross_rank_text,
        discover_subspace_sources,
        expand_per_model_layer_configs,
        filter_available_ranks,
        fit_regularized_maxvar_gcca,
        get_dataset_adapter,
        get_glove_embeddings,
        get_results_dir,
        is_pca_source_kind,
        load_base_comparison_data,
        load_config,
        load_embeddings,
        load_encoding_results,
        load_fold_bases,
        load_peer_embeddings_on_base_rows,
        load_sig_electrodes,
        max_available_rank,
        maxvar_ambient_map,
        maxvar_block_overlap,
        maxvar_component_blocks,
        mo,
        np,
        orthonormal_basis,
        parse_rank_list,
        pd,
        plt,
        principal_angles_from_bases,
        projected_rsm,
        resolve_dataset_context,
        resolve_subspace_run,
        rsa_lower_triangle_correlations,
        run_metadata,
        sample_random_principal_angles,
        sanitize_name,
        score_cca_alignment,
        score_maxvar_partial_support,
        score_procrustes_alignment,
        select_maxvar_alpha,
        subspace_distance_metrics,
        summarize_cca_fold_results,
        summarize_procrustes_fold_results,
        summarize_rsa_fold_results,
        validate_matching_fold_indices,
    )


@app.cell
def _(mo):
    mo.md("""
    # Compare RRR/PCA Subspaces
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    roots_input = mo.ui.text(
        value="results,.",
        label="Scan roots",
        full_width=True,
    )
    include_null_input = mo.ui.checkbox(value=False, label="Random null")
    null_draws_input = mo.ui.number(
        value=250, start=1, stop=10000, step=1, label="Null draws"
    )
    null_seed_input = mo.ui.number(value=0, start=0, step=1, label="Null seed")
    null_band_input = mo.ui.slider(
        start=50,
        stop=99,
        step=1,
        value=95,
        label="Null band %",
    )
    mo.vstack(
        [
            roots_input,
            mo.hstack(
                [
                    include_null_input,
                    null_draws_input,
                    null_seed_input,
                    null_band_input,
                ]
            ),
        ]
    )
    return (
        include_null_input,
        null_band_input,
        null_draws_input,
        null_seed_input,
        roots_input,
    )


@app.cell(hide_code=True)
def _(discover_subspace_sources, is_pca_source_kind, mo, roots_input):
    _roots = [r.strip() for r in roots_input.value.split(",") if r.strip()]
    sources = discover_subspace_sources(_roots)
    mo.stop(
        not sources,
        mo.callout(
            mo.md("No `fold_*/A.npy` subspace factors found under the scan roots."),
            kind="warn",
        ),
    )
    source_options = {
        f"{s.label}  |  kind={s.kind}, fold={s.fold}, shape=({s.ambient_dim}, {s.max_rank})": i
        for i, s in enumerate(sources)
    }
    primary_source_options = {
        label: idx
        for label, idx in source_options.items()
        if not is_pca_source_kind(sources[idx].kind)
    }
    mo.stop(
        not primary_source_options,
        mo.callout(
            mo.md(
                "Only PCA subspace factors were found. Select scan roots containing "
                "RRR outputs to compare primary subspaces."
            ),
            kind="warn",
        ),
    )
    return primary_source_options, sources


@app.cell(hide_code=True)
def _(mo, primary_source_options):
    _source_labels = list(primary_source_options.keys())
    _default_b = 1 if len(_source_labels) > 1 else 0
    source_a_input = mo.ui.dropdown(
        options=primary_source_options,
        value=_source_labels[0],
        label="Subspace A",
        full_width=True,
    )
    source_b_input = mo.ui.dropdown(
        options=primary_source_options,
        value=_source_labels[_default_b],
        label="Subspace B",
        full_width=True,
    )
    pca_overlay_input = mo.ui.dropdown(
        options={
            "Off": "off",
            "A RRR vs B PCA": "b_pca",
            "A PCA vs B RRR": "a_pca",
            "All PCA baselines": "all",
        },
        value="Off",
        label="PCA overlay",
    )
    mo.vstack([source_a_input, source_b_input, pca_overlay_input])
    return pca_overlay_input, source_a_input, source_b_input


@app.cell(hide_code=True)
def _(default_cross_rank_text, mo, source_a_input, source_b_input, sources):
    source_a = sources[source_a_input.value]
    source_b = sources[source_b_input.value]
    rank_a_input = mo.ui.number(
        start=1,
        stop=source_a.max_rank,
        step=1,
        value=min(10, source_a.max_rank),
        label="Rank A",
    )
    rank_b_input = mo.ui.number(
        start=1,
        stop=source_b.max_rank,
        step=1,
        value=min(10, source_b.max_rank),
        label="Rank B",
    )
    cross_rank_input = mo.ui.text(
        value=default_cross_rank_text(),
        label="Cross-rank ranks",
        full_width=True,
    )
    mo.vstack([mo.hstack([rank_a_input, rank_b_input]), cross_rank_input])
    return cross_rank_input, rank_a_input, rank_b_input, source_a, source_b


@app.cell(hide_code=True)
def _(is_pca_source_kind, mo, pd, sources):
    _rows = [
        {
            "label": s.label,
            "kind": s.kind,
            "role": "PCA baseline candidate" if is_pca_source_kind(s.kind) else "primary selectable",
            "fold": s.fold,
            "ambient_dim": s.ambient_dim,
            "max_rank": s.max_rank,
            "path": str(s.path),
        }
        for s in sources
    ]
    mo.ui.table(pd.DataFrame(_rows), selection=None)
    return


@app.cell(hide_code=True)
def _(np, orthonormal_basis, principal_angles_from_bases):
    def group_subspace_sources_by_unit(sources):
        """Group fold-level sources by the directory containing fold_* children."""
        groups = {}
        for source in sources:
            groups.setdefault(source.unit_dir, []).append(source)
        return {
            unit_dir: sorted(group, key=lambda source: source.fold)
            for unit_dir, group in sorted(groups.items(), key=lambda item: str(item[0]))
        }

    # def load_fold_bases(group, rank):
    #     """Load and orthonormalize the leading rank columns for each fold."""
    #     rank = int(rank)
    #     bases = []
    #     for source in group:
    #         matrix = np.load(source.path, mmap_mode="r")
    #         bases.append(orthonormal_basis(matrix, rank))
    #     return bases

    def _orthonormalize_basis(matrix):
        _q, _r = np.linalg.qr(matrix, mode="reduced")
        _signs = np.sign(np.diag(_r))
        _signs[_signs == 0] = 1
        return _q * _signs

    def _grassmann_tangent_norm(tangent):
        return float(np.linalg.norm(tangent, ord="fro"))

    def grassmann_objective(basis, fold_bases):
        """Return 0.5 * mean squared Grassmann distance to fold bases."""
        _sq_distances = [
            float(np.sum(principal_angles_from_bases(basis, _target) ** 2))
            for _target in fold_bases
        ]
        return 0.5 * float(np.mean(_sq_distances))

    def grassmann_log_map(base, target):
        """Log map on the Grassmann manifold from base to target."""
        _base_to_target = base.T @ target
        _normal_component = target - base @ _base_to_target
        _tangent_factor = _normal_component @ np.linalg.pinv(_base_to_target)
        _u, _s, _vt = np.linalg.svd(_tangent_factor, full_matrices=False)
        return (_u * np.arctan(_s)) @ _vt

    def grassmann_exp_map(base, tangent):
        """Exp map on the Grassmann manifold from base along tangent."""
        _u, _s, _vt = np.linalg.svd(tangent, full_matrices=False)
        _v = _vt.T
        _candidate = base @ ((_v * np.cos(_s)) @ _vt) + (_u * np.sin(_s)) @ _vt
        return _orthonormalize_basis(_candidate)

    def grassmann_intrinsic_mean(fold_bases, initial_basis, options):
        """Refine an extrinsic initial basis into an intrinsic Grassmann mean."""
        _opts = dict(options or {})
        _max_iter = int(_opts.get("max_iter", 50))
        _tol = float(_opts.get("tol", 1e-6))
        _initial_step = float(_opts.get("initial_step", 1.0))
        _min_step = float(_opts.get("min_step", 1e-6))
        _shrink = float(_opts.get("backtracking_shrink", 0.5))
        _max_backtracking = int(_opts.get("max_backtracking", 12))
        _log_frequency = max(1, int(_opts.get("log_frequency", 1)))
        if _max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if _tol <= 0:
            raise ValueError("tol must be positive")
        if _initial_step <= 0:
            raise ValueError("initial_step must be positive")
        if _min_step <= 0:
            raise ValueError("min_step must be positive")
        if not 0 < _shrink < 1:
            raise ValueError("backtracking_shrink must be between 0 and 1")
        if _max_backtracking < 0:
            raise ValueError("max_backtracking must be non-negative")

        _basis = _orthonormalize_basis(initial_basis)
        _objective = grassmann_objective(_basis, fold_bases)
        _trace = []
        _accepted_steps = 0
        _rejected_steps = 0
        _converged = False
        _stop_reason = "max_iter"
        _step_size = _initial_step
        _last_iteration = 0

        for _iteration in range(_max_iter + 1):
            _last_iteration = _iteration
            _logs = [grassmann_log_map(_basis, _target) for _target in fold_bases]
            _mean_tangent = np.mean(np.stack(_logs, axis=0), axis=0)
            _grad_norm = _grassmann_tangent_norm(_mean_tangent)
            if _iteration % _log_frequency == 0 or _iteration == _max_iter:
                _trace.append(
                    {
                        "iteration": _iteration,
                        "objective": _objective,
                        "gradient_norm": _grad_norm,
                        "step_size": _step_size,
                        "accepted": True,
                    }
                )
            if _grad_norm <= _tol:
                _converged = True
                _stop_reason = "gradient_norm"
                break
            if _iteration == _max_iter:
                break

            _candidate_step = _step_size
            _accepted = False
            _candidate_basis = None
            _candidate_objective = None
            for _ in range(_max_backtracking + 1):
                if _candidate_step < _min_step:
                    break
                _candidate_basis = grassmann_exp_map(
                    _basis,
                    _candidate_step * _mean_tangent,
                )
                _candidate_objective = grassmann_objective(
                    _candidate_basis,
                    fold_bases,
                )
                if _candidate_objective <= _objective:
                    _accepted = True
                    break
                _candidate_step *= _shrink
                _rejected_steps += 1

            if not _accepted:
                _stop_reason = "step_size"
                break

            _accepted_steps += 1
            _improvement = _objective - _candidate_objective
            _basis = _candidate_basis
            _objective = _candidate_objective
            _step_size = min(_initial_step, _candidate_step / _shrink)
            if _improvement <= _tol * max(1.0, abs(_objective)):
                _converged = True
                _stop_reason = "objective_improvement"
                break

        _final_logs = [grassmann_log_map(_basis, _target) for _target in fold_bases]
        _final_mean_tangent = np.mean(np.stack(_final_logs, axis=0), axis=0)
        _final_grad_norm = _grassmann_tangent_norm(_final_mean_tangent)
        if not _trace or _trace[-1]["iteration"] != _last_iteration:
            _trace.append(
                {
                    "iteration": _last_iteration,
                    "objective": _objective,
                    "gradient_norm": _final_grad_norm,
                    "step_size": _step_size,
                    "accepted": True,
                }
            )

        _diagnostics = {
            "method": "intrinsic",
            "converged": _converged,
            "stop_reason": _stop_reason,
            "iterations": _last_iteration,
            "final_objective": _objective,
            "final_gradient_norm": _final_grad_norm,
            "final_step_size": _step_size,
            "accepted_steps": _accepted_steps,
            "rejected_backtracks": _rejected_steps,
            "trace": _trace,
        }
        return _basis, _diagnostics

    def stacked_svd_subspace_mean(fold_bases, rank):
        """Average fold subspaces via SVD of stacked orthonormal bases.

        If Q_i are fold bases, the left singular vectors of [Q_1 ... Q_n]
        are the eigenvectors of sum_i Q_i Q_i.T, so this returns the same
        rank-k subspace as eigendecomposing the mean projection matrix.
        """
        rank = int(rank)
        if not fold_bases:
            raise ValueError("At least one fold basis is required")
        ambient_dims = {basis.shape[0] for basis in fold_bases}
        if len(ambient_dims) != 1:
            raise ValueError(f"Fold bases have mixed ambient dimensions: {ambient_dims}")
        stacked = np.concatenate(fold_bases, axis=1)
        u, singular_values, _ = np.linalg.svd(stacked, full_matrices=False)
        if rank > u.shape[1]:
            raise ValueError(
                f"rank {rank} exceeds stacked basis rank capacity {u.shape[1]}"
            )
        projector_eigenvalues = singular_values[:rank] ** 2 / len(fold_bases)
        diagnostics = {
            "method": "stacked_svd",
            "singular_values": singular_values[:rank],
            "projector_eigenvalues": projector_eigenvalues,
            "projector_energy": float(np.sum(projector_eigenvalues)),
            "converged": True,
            "stop_reason": "extrinsic",
            "iterations": 0,
            "final_objective": grassmann_objective(u[:, :rank], fold_bases),
            "final_gradient_norm": float("nan"),
            "final_step_size": float("nan"),
            "accepted_steps": 0,
            "rejected_backtracks": 0,
            "trace": [],
        }
        return u[:, :rank], diagnostics

    def average_subspace_bases(
        fold_bases,
        rank,
        method="stacked_svd",
        intrinsic_options=None,
    ):
        """Return a representative basis for a list of fold subspaces."""
        _extrinsic_basis, _extrinsic_diag = stacked_svd_subspace_mean(fold_bases, rank)
        if method == "stacked_svd":
            return _extrinsic_basis, _extrinsic_diag
        if method == "intrinsic":
            _intrinsic_basis, _intrinsic_diag = grassmann_intrinsic_mean(
                fold_bases,
                _extrinsic_basis,
                intrinsic_options,
            )
            _intrinsic_diag["extrinsic"] = _extrinsic_diag
            _intrinsic_diag["projector_eigenvalues"] = _extrinsic_diag[
                "projector_eigenvalues"
            ]
            _intrinsic_diag["projector_energy"] = _extrinsic_diag["projector_energy"]
            _intrinsic_diag["singular_values"] = _extrinsic_diag["singular_values"]
            return _intrinsic_basis, _intrinsic_diag
        raise ValueError(f"Unknown subspace averaging method: {method!r}")

    def sample_fold_averaged_random_principal_angles(
        ambient_dim,
        rank,
        n_folds_a,
        n_folds_b,
        n_draws,
        seed=0,
        method="stacked_svd",
        intrinsic_options=None,
    ):
        """Sample null angles after averaging random fold subspaces per side."""
        _rank = int(rank)
        _n_folds_a = int(n_folds_a)
        _n_folds_b = int(n_folds_b)
        if ambient_dim < _rank:
            raise ValueError("ambient_dim must be at least rank")
        if _n_folds_a < 1 or _n_folds_b < 1:
            raise ValueError("n_folds_a and n_folds_b must both be at least 1")
        _rng = np.random.default_rng(seed)
        _draws = []
        for _ in range(int(n_draws)):
            _fold_bases_a = [
                orthonormal_basis(_rng.standard_normal((ambient_dim, _rank)), _rank)
                for _ in range(_n_folds_a)
            ]
            _fold_bases_b = [
                orthonormal_basis(_rng.standard_normal((ambient_dim, _rank)), _rank)
                for _ in range(_n_folds_b)
            ]
            _mean_a, _ = average_subspace_bases(
                _fold_bases_a,
                _rank,
                method=method,
                intrinsic_options=intrinsic_options,
            )
            _mean_b, _ = average_subspace_bases(
                _fold_bases_b,
                _rank,
                method=method,
                intrinsic_options=intrinsic_options,
            )
            _draws.append(principal_angles_from_bases(_mean_a, _mean_b))
        return np.stack(_draws, axis=0)

    return (
        average_subspace_bases,
        group_subspace_sources_by_unit,
        sample_fold_averaged_random_principal_angles,
        stacked_svd_subspace_mean,
    )


@app.cell(hide_code=True)
def _(np, plt):
    def principal_angle_null_band(null_angles, band_percent):
        alpha = (100 - int(band_percent)) / 2
        return {
            "low": np.percentile(null_angles, alpha, axis=0),
            "median": np.percentile(null_angles, 50, axis=0),
            "high": np.percentile(null_angles, 100 - alpha, axis=0),
        }

    def plot_principal_angles(
        angles,
        *,
        title,
        primary_label,
        null_band=None,
        overlay_angles=None,
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(1, len(angles) + 1)
        ax.plot(x, np.degrees(angles), marker="o", linewidth=2, label=primary_label)
        if null_band is not None:
            ax.fill_between(
                x,
                np.degrees(null_band["low"]),
                np.degrees(null_band["high"]),
                alpha=0.18,
                color="0.4",
                label="Random null band",
            )
            ax.plot(
                x,
                np.degrees(null_band["median"]),
                color="0.35",
                linestyle="--",
                linewidth=1.5,
                label="Random null median",
            )
        for label, values in (overlay_angles or {}).items():
            overlay_x = np.arange(1, len(values) + 1)
            ax.plot(
                overlay_x,
                np.degrees(values),
                marker=".",
                linewidth=1.6,
                label=label,
            )
        ax.set_xlabel("Principal angle index")
        ax.set_ylabel("Angle (degrees)")
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    return plot_principal_angles, principal_angle_null_band


@app.cell(hide_code=True)
def _(plt):
    def plot_cross_rank_metric(df, metric, *, title, ylabel):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for comparison, group in df.groupby("comparison", sort=False):
            ordered = group.sort_values("rank")
            ax.plot(
                ordered["rank"],
                ordered[metric],
                marker="o",
                linewidth=2,
                label=comparison,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Rank")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    return (plot_cross_rank_metric,)


@app.cell(hide_code=True)
def _(np, orthonormal_basis, rank_a_input, rank_b_input, source_a, source_b):
    matrix_a = np.load(source_a.path, mmap_mode="r")
    matrix_b = np.load(source_b.path, mmap_mode="r")
    basis_a = orthonormal_basis(matrix_a, int(rank_a_input.value))
    basis_b = orthonormal_basis(matrix_b, int(rank_b_input.value))
    return basis_a, basis_b


@app.cell(hide_code=True)
def _(basis_a, basis_b, mo, principal_angles_from_bases):
    mo.stop(
        basis_a.shape[0] != basis_b.shape[0],
        mo.callout(
            mo.md(
                "Selected subspaces have different ambient dimensions: "
                f"{basis_a.shape[0]} vs {basis_b.shape[0]}."
            ),
            kind="warn",
        ),
    )
    angles = principal_angles_from_bases(basis_a, basis_b)
    return (angles,)


@app.cell(hide_code=True)
def _(np, source_a, source_b, sources):
    def _pca_sibling(source):
        parts = list(source.unit_dir.parts)
        replacements = {
            "rrr": "pca",
            "per_subject_rrr": "per_subject_pca",
            "per_region_rrr": "per_region_pca",
            "loso_rrr": "loso_pca",
        }
        candidates = []
        for idx, part in enumerate(parts):
            if part in replacements:
                new_parts = parts.copy()
                new_parts[idx] = replacements[part]
                candidates.append(source.unit_dir.__class__(*new_parts))
        for candidate in candidates:
            for other in sources:
                if (
                    other.unit_dir == candidate
                    and other.fold == source.fold
                    and other.ambient_dim == source.ambient_dim
                ):
                    return other
        return None

    pca_a = _pca_sibling(source_a) if not source_a.kind.endswith("pca") else source_a
    pca_b = _pca_sibling(source_b) if not source_b.kind.endswith("pca") else source_b
    _ = np
    return pca_a, pca_b


@app.cell(hide_code=True)
def _(
    basis_a,
    basis_b,
    np,
    orthonormal_basis,
    pca_a,
    pca_b,
    pca_overlay_input,
    principal_angles_from_bases,
    rank_a_input,
    rank_b_input,
    source_a,
    source_b,
):
    overlay_angles = {}
    _mode = pca_overlay_input.value
    if (
        _mode in {"b_pca", "all"}
        and pca_b is not None
        and pca_b.path != source_b.path
        and int(rank_b_input.value) <= pca_b.max_rank
    ):
        _pca_basis_b = orthonormal_basis(
            np.load(pca_b.path, mmap_mode="r"),
            rank_b_input.value,
        )
        if basis_a.shape[0] == _pca_basis_b.shape[0]:
            overlay_angles["A RRR vs B PCA"] = principal_angles_from_bases(
                basis_a,
                _pca_basis_b,
            )
    if (
        _mode in {"a_pca", "all"}
        and pca_a is not None
        and pca_a.path != source_a.path
        and int(rank_a_input.value) <= pca_a.max_rank
    ):
        _pca_basis_a = orthonormal_basis(
            np.load(pca_a.path, mmap_mode="r"),
            rank_a_input.value,
        )
        if _pca_basis_a.shape[0] == basis_b.shape[0]:
            overlay_angles["A PCA vs B RRR"] = principal_angles_from_bases(
                _pca_basis_a,
                basis_b,
            )
    if (
        _mode == "all"
        and pca_a is not None
        and pca_b is not None
        and int(rank_a_input.value) <= pca_a.max_rank
        and int(rank_b_input.value) <= pca_b.max_rank
    ):
        _pca_basis_a = orthonormal_basis(
            np.load(pca_a.path, mmap_mode="r"),
            rank_a_input.value,
        )
        _pca_basis_b = orthonormal_basis(
            np.load(pca_b.path, mmap_mode="r"),
            rank_b_input.value,
        )
        if _pca_basis_a.shape[0] == _pca_basis_b.shape[0]:
            overlay_angles["A PCA vs B PCA"] = principal_angles_from_bases(
                _pca_basis_a,
                _pca_basis_b,
            )
    return (overlay_angles,)


@app.cell(hide_code=True)
def _(
    angles,
    include_null_input,
    null_band_input,
    null_draws_input,
    null_seed_input,
    principal_angle_null_band,
    rank_a_input,
    rank_b_input,
    sample_random_principal_angles,
    source_a,
    source_b,
):
    null_angles = None
    null_band = None
    if include_null_input.value:
        null_angles = sample_random_principal_angles(
            source_a.ambient_dim,
            int(rank_a_input.value),
            int(rank_b_input.value),
            int(null_draws_input.value),
            int(null_seed_input.value),
        )
        null_band = principal_angle_null_band(null_angles, null_band_input.value)
    _ = angles, source_b
    return null_angles, null_band


@app.cell(hide_code=True)
def _(
    angles,
    mo,
    null_angles,
    null_band,
    overlay_angles,
    plot_principal_angles,
    source_a,
    source_b,
):
    _fig = plot_principal_angles(
        angles,
        title="Principal angles between selected subspaces",
        primary_label="A RRR vs B RRR",
        null_band=null_band,
        overlay_angles=overlay_angles,
    )
    _notes = [
        f"A: `{source_a.label}`",
        f"B: `{source_b.label}`",
    ]
    if null_angles is not None:
        _notes.append(f"Null draws: {null_angles.shape[0]}")
    mo.vstack([mo.md("\n\n".join(_notes)), _fig])
    return


@app.cell(hide_code=True)
def _(angles, mo, np, overlay_angles, pd, subspace_distance_metrics):
    _rows = [{"comparison": "A RRR vs B RRR", **subspace_distance_metrics(angles)}]
    for _label, _angles in overlay_angles.items():
        _rows.append({"comparison": _label, **subspace_distance_metrics(_angles)})
    _df = pd.DataFrame(_rows)
    for _col in ["grassmann", "fubini_study"]:
        _df[f"{_col}_degrees"] = np.degrees(_df[_col])
    mo.ui.table(_df, selection=None)
    return


@app.cell(hide_code=True)
def _(
    cross_rank_input,
    filter_available_ranks,
    mo,
    np,
    orthonormal_basis,
    parse_rank_list,
    pca_a,
    pca_b,
    pd,
    plot_cross_rank_metric,
    principal_angles_from_bases,
    source_a,
    source_b,
    subspace_distance_metrics,
):
    try:
        _requested_ranks = parse_rank_list(cross_rank_input.value)
    except ValueError as exc:
        mo.stop(True, mo.callout(mo.md(str(exc)), kind="warn"))

    _rows = []
    _notes = []

    def _append_grassmann_rows(label, left, right):
        if left is None or right is None:
            _notes.append(f"`{label}` omitted because a PCA sibling was not found.")
            return
        if left.ambient_dim != right.ambient_dim:
            _notes.append(
                f"`{label}` omitted because ambient dimensions differ: "
                f"{left.ambient_dim} vs {right.ambient_dim}."
            )
            return
        _valid_ranks, _skipped_ranks = filter_available_ranks(
            _requested_ranks,
            min(left.max_rank, right.max_rank),
        )
        if _skipped_ranks:
            _notes.append(
                f"`{label}` skipped unavailable ranks: "
                + ", ".join(str(rank) for rank in _skipped_ranks)
            )
        if not _valid_ranks:
            return
        _left_matrix = np.load(left.path, mmap_mode="r")
        _right_matrix = np.load(right.path, mmap_mode="r")
        for _rank in _valid_ranks:
            _left_basis = orthonormal_basis(_left_matrix, _rank)
            _right_basis = orthonormal_basis(_right_matrix, _rank)
            _metrics = subspace_distance_metrics(
                principal_angles_from_bases(_left_basis, _right_basis)
            )
            _rows.append(
                {
                    "rank": _rank,
                    "comparison": label,
                    "grassmann": _metrics["grassmann"],
                    "grassmann_degrees": np.degrees(_metrics["grassmann"]),
                }
            )

    _append_grassmann_rows("A RRR vs B RRR", source_a, source_b)
    _append_grassmann_rows("A RRR vs B PCA", source_a, pca_b)
    _append_grassmann_rows("A PCA vs B RRR", pca_a, source_b)
    _append_grassmann_rows("A PCA vs B PCA", pca_a, pca_b)

    _df = pd.DataFrame(_rows)
    mo.stop(
        _df.empty,
        mo.callout(
            mo.md("No cross-rank subspace comparisons could be computed."),
            kind="warn",
        ),
    )
    _items = [
        mo.md("### Cross-rank Grassmann summary"),
        plot_cross_rank_metric(
            _df,
            "grassmann",
            title="Cross-rank selected subspace comparison",
            ylabel="Grassmann distance (radians)",
        ),
        mo.ui.table(_df, selection=None),
    ]
    if _notes:
        _items.insert(1, mo.callout(mo.md("\n\n".join(_notes)), kind="warn"))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Fold-Averaged Subspace Comparison
    """)
    return


@app.cell(hide_code=True)
def _(group_subspace_sources_by_unit, is_pca_source_kind, mo, sources):
    aggregate_groups_by_unit = group_subspace_sources_by_unit(sources)
    aggregate_group_list = list(aggregate_groups_by_unit.values())
    mo.stop(
        not aggregate_group_list,
        mo.callout(
            mo.md("No aggregate subspace groups found under the scan roots."),
            kind="warn",
        ),
    )

    def _aggregate_label(group):
        first = group[0]
        label = str(first.unit_dir)
        folds = ",".join(str(source.fold) for source in group)
        min_rank = min(source.max_rank for source in group)
        max_rank = max(source.max_rank for source in group)
        rank_label = str(min_rank) if min_rank == max_rank else f"{min_rank}-{max_rank}"
        return (
            f"{label}  |  kind={first.kind}, folds=[{folds}], "
            f"shape=({first.ambient_dim}, {rank_label})"
        )

    aggregate_group_options = {
        _aggregate_label(group): idx for idx, group in enumerate(aggregate_group_list)
    }
    aggregate_primary_group_options = {
        label: idx
        for label, idx in aggregate_group_options.items()
        if not is_pca_source_kind(aggregate_group_list[idx][0].kind)
    }
    mo.stop(
        not aggregate_primary_group_options,
        mo.callout(
            mo.md(
                "No RRR aggregate groups found. PCA groups are kept as baseline "
                "candidates but cannot be selected as primary averages."
            ),
            kind="warn",
        ),
    )
    return (
        aggregate_group_list,
        aggregate_groups_by_unit,
        aggregate_primary_group_options,
    )


@app.cell(hide_code=True)
def _(aggregate_primary_group_options, mo):
    _aggregate_labels = list(aggregate_primary_group_options.keys())
    _default_b = 1 if len(_aggregate_labels) > 1 else 0
    aggregate_a_input = mo.ui.dropdown(
        options=aggregate_primary_group_options,
        value=_aggregate_labels[0],
        label="Average subspace A",
        full_width=True,
    )
    aggregate_b_input = mo.ui.dropdown(
        options=aggregate_primary_group_options,
        value=_aggregate_labels[_default_b],
        label="Average subspace B",
        full_width=True,
    )
    mo.vstack([aggregate_a_input, aggregate_b_input])
    return aggregate_a_input, aggregate_b_input


@app.cell(hide_code=True)
def _(aggregate_groups_by_unit):
    def pca_sibling_group(group):
        """Return the fold group in the mirrored PCA output directory, if present."""
        first = group[0]
        if first.kind.endswith("pca"):
            return group
        replacements = {
            "rrr": "pca",
            "per_subject_rrr": "per_subject_pca",
            "per_region_rrr": "per_region_pca",
            "loso_rrr": "loso_pca",
        }
        candidates = []
        parts = list(first.unit_dir.parts)
        for idx, part in enumerate(parts):
            if part in replacements:
                new_parts = parts.copy()
                new_parts[idx] = replacements[part]
                candidates.append(first.unit_dir.__class__(*new_parts))
        wanted_folds = [source.fold for source in group]
        wanted_ambient = {source.ambient_dim for source in group}
        for candidate in candidates:
            sibling = aggregate_groups_by_unit.get(candidate)
            if sibling is None:
                continue
            if [source.fold for source in sibling] != wanted_folds:
                continue
            if {source.ambient_dim for source in sibling} != wanted_ambient:
                continue
            return sibling
        return None

    return (pca_sibling_group,)


@app.cell(hide_code=True)
def _(
    aggregate_a_input,
    aggregate_b_input,
    aggregate_group_list,
    default_cross_rank_text,
    mo,
):
    aggregate_group_a = aggregate_group_list[aggregate_a_input.value]
    aggregate_group_b = aggregate_group_list[aggregate_b_input.value]
    _max_rank = min(
        min(source.max_rank for source in aggregate_group_a),
        min(source.max_rank for source in aggregate_group_b),
    )
    aggregate_rank_input = mo.ui.text(
        value=str(min(10, _max_rank)),
        label="Averaging rank",
        full_width=False,
    )
    aggregate_cross_rank_input = mo.ui.text(
        value=default_cross_rank_text(),
        label="Cross-rank averaging ranks",
        full_width=True,
    )
    mo.vstack([mo.hstack([aggregate_rank_input]), aggregate_cross_rank_input])
    return (
        aggregate_cross_rank_input,
        aggregate_group_a,
        aggregate_group_b,
        aggregate_rank_input,
    )


@app.cell(hide_code=True)
def _(mo):
    use_intrinsic_mean_input = mo.ui.checkbox(
        value=False,
        label="Use intrinsic Grassmann mean",
    )
    intrinsic_max_iter_input = mo.ui.number(
        value=50,
        start=1,
        stop=1000,
        step=1,
        label="Max iterations",
    )
    intrinsic_tol_input = mo.ui.text(
        value="1e-6",
        label="Tolerance",
        full_width=False,
    )
    intrinsic_initial_step_input = mo.ui.text(
        value="1.0",
        label="Initial step",
        full_width=False,
    )
    intrinsic_min_step_input = mo.ui.text(
        value="1e-6",
        label="Min step",
        full_width=False,
    )
    intrinsic_shrink_input = mo.ui.text(
        value="0.5",
        label="Backtracking shrink",
        full_width=False,
    )
    intrinsic_max_backtracking_input = mo.ui.number(
        value=12,
        start=0,
        stop=100,
        step=1,
        label="Max backtracks",
    )
    intrinsic_log_frequency_input = mo.ui.number(
        value=1,
        start=1,
        stop=1000,
        step=1,
        label="Log frequency",
    )
    mo.vstack(
        [
            use_intrinsic_mean_input,
            mo.hstack(
                [
                    intrinsic_max_iter_input,
                    intrinsic_tol_input,
                    intrinsic_initial_step_input,
                    intrinsic_min_step_input,
                ]
            ),
            mo.hstack(
                [
                    intrinsic_shrink_input,
                    intrinsic_max_backtracking_input,
                    intrinsic_log_frequency_input,
                ]
            ),
        ]
    )
    return (
        intrinsic_initial_step_input,
        intrinsic_log_frequency_input,
        intrinsic_max_backtracking_input,
        intrinsic_max_iter_input,
        intrinsic_min_step_input,
        intrinsic_shrink_input,
        intrinsic_tol_input,
        use_intrinsic_mean_input,
    )


@app.cell(hide_code=True)
def _(
    intrinsic_initial_step_input,
    intrinsic_log_frequency_input,
    intrinsic_max_backtracking_input,
    intrinsic_max_iter_input,
    intrinsic_min_step_input,
    intrinsic_shrink_input,
    intrinsic_tol_input,
    mo,
    use_intrinsic_mean_input,
):
    def _parse_float_input(ui_element, label):
        try:
            return float(str(ui_element.value).strip())
        except ValueError:
            mo.stop(True, mo.callout(mo.md(f"{label} must be numeric."), kind="warn"))

    intrinsic_options = {
        "max_iter": int(intrinsic_max_iter_input.value),
        "tol": _parse_float_input(intrinsic_tol_input, "Tolerance"),
        "initial_step": _parse_float_input(
            intrinsic_initial_step_input,
            "Initial step",
        ),
        "min_step": _parse_float_input(intrinsic_min_step_input, "Min step"),
        "backtracking_shrink": _parse_float_input(
            intrinsic_shrink_input,
            "Backtracking shrink",
        ),
        "max_backtracking": int(intrinsic_max_backtracking_input.value),
        "log_frequency": int(intrinsic_log_frequency_input.value),
    }
    aggregate_average_method = (
        "intrinsic" if use_intrinsic_mean_input.value else "stacked_svd"
    )
    mo.stop(
        intrinsic_options["tol"] <= 0,
        mo.callout(mo.md("Tolerance must be positive."), kind="warn"),
    )
    mo.stop(
        intrinsic_options["initial_step"] <= 0,
        mo.callout(mo.md("Initial step must be positive."), kind="warn"),
    )
    mo.stop(
        intrinsic_options["min_step"] <= 0,
        mo.callout(mo.md("Min step must be positive."), kind="warn"),
    )
    mo.stop(
        not 0 < intrinsic_options["backtracking_shrink"] < 1,
        mo.callout(
            mo.md("Backtracking shrink must be between 0 and 1."),
            kind="warn",
        ),
    )
    return aggregate_average_method, intrinsic_options


@app.cell(hide_code=True)
def _(
    aggregate_average_method,
    aggregate_group_a,
    aggregate_group_b,
    aggregate_rank_input,
    average_subspace_bases,
    intrinsic_options,
    load_fold_bases,
    mo,
    pca_sibling_group,
):
    try:
        aggregate_rank = int(str(aggregate_rank_input.value).strip())
    except ValueError:
        aggregate_rank = None
    mo.stop(
        aggregate_rank is None or aggregate_rank < 1,
        mo.callout(mo.md("Averaging rank must be a positive integer."), kind="warn"),
    )
    _max_rank_a = min(source.max_rank for source in aggregate_group_a)
    _max_rank_b = min(source.max_rank for source in aggregate_group_b)
    mo.stop(
        aggregate_rank > min(_max_rank_a, _max_rank_b),
        mo.callout(
            mo.md(
                "Averaging rank exceeds the available columns for at least one "
                f"selected group. Max shared rank is {min(_max_rank_a, _max_rank_b)}."
            ),
            kind="warn",
        ),
    )
    _ambient_a = {source.ambient_dim for source in aggregate_group_a}
    _ambient_b = {source.ambient_dim for source in aggregate_group_b}
    mo.stop(
        len(_ambient_a) != 1 or len(_ambient_b) != 1 or _ambient_a != _ambient_b,
        mo.callout(
            mo.md(
                "Selected aggregate groups must have one matching ambient dimension. "
                f"Got A={sorted(_ambient_a)} and B={sorted(_ambient_b)}."
            ),
            kind="warn",
        ),
    )
    aggregate_fold_bases_a = load_fold_bases(aggregate_group_a, aggregate_rank)
    aggregate_fold_bases_b = load_fold_bases(aggregate_group_b, aggregate_rank)
    aggregate_basis_a, aggregate_diag_a = average_subspace_bases(
        aggregate_fold_bases_a,
        aggregate_rank,
        method=aggregate_average_method,
        intrinsic_options=intrinsic_options,
    )
    aggregate_basis_b, aggregate_diag_b = average_subspace_bases(
        aggregate_fold_bases_b,
        aggregate_rank,
        method=aggregate_average_method,
        intrinsic_options=intrinsic_options,
    )
    aggregate_pca_group_a = pca_sibling_group(aggregate_group_a)
    aggregate_pca_group_b = pca_sibling_group(aggregate_group_b)
    aggregate_pca_basis_a = None
    aggregate_pca_basis_b = None
    aggregate_pca_diag_a = None
    aggregate_pca_diag_b = None
    if aggregate_pca_group_a is not None and aggregate_pca_group_b is not None:
        _pca_max_rank_a = min(source.max_rank for source in aggregate_pca_group_a)
        _pca_max_rank_b = min(source.max_rank for source in aggregate_pca_group_b)
        _pca_ambient_a = {source.ambient_dim for source in aggregate_pca_group_a}
        _pca_ambient_b = {source.ambient_dim for source in aggregate_pca_group_b}
        if (
            aggregate_rank <= min(_pca_max_rank_a, _pca_max_rank_b)
            and len(_pca_ambient_a) == 1
            and len(_pca_ambient_b) == 1
            and _pca_ambient_a == _pca_ambient_b
        ):
            _pca_fold_bases_a = load_fold_bases(aggregate_pca_group_a, aggregate_rank)
            _pca_fold_bases_b = load_fold_bases(aggregate_pca_group_b, aggregate_rank)
            aggregate_pca_basis_a, aggregate_pca_diag_a = average_subspace_bases(
                _pca_fold_bases_a,
                aggregate_rank,
                method=aggregate_average_method,
                intrinsic_options=intrinsic_options,
            )
            aggregate_pca_basis_b, aggregate_pca_diag_b = average_subspace_bases(
                _pca_fold_bases_b,
                aggregate_rank,
                method=aggregate_average_method,
                intrinsic_options=intrinsic_options,
            )
    return (
        aggregate_basis_a,
        aggregate_basis_b,
        aggregate_diag_a,
        aggregate_diag_b,
        aggregate_pca_basis_a,
        aggregate_pca_basis_b,
        aggregate_pca_diag_a,
        aggregate_pca_diag_b,
        aggregate_pca_group_a,
        aggregate_pca_group_b,
        aggregate_rank,
    )


@app.cell(hide_code=True)
def _(
    aggregate_basis_a,
    aggregate_basis_b,
    aggregate_pca_basis_a,
    aggregate_pca_basis_b,
    principal_angles_from_bases,
):
    aggregate_angles = principal_angles_from_bases(aggregate_basis_a, aggregate_basis_b)
    aggregate_overlay_angles = {}
    if aggregate_pca_basis_a is not None and aggregate_pca_basis_b is not None:
        aggregate_overlay_angles["A RRR mean vs B PCA mean"] = (
            principal_angles_from_bases(aggregate_basis_a, aggregate_pca_basis_b)
        )
        aggregate_overlay_angles["A PCA mean vs B RRR mean"] = (
            principal_angles_from_bases(aggregate_pca_basis_a, aggregate_basis_b)
        )
        aggregate_overlay_angles["A PCA mean vs B PCA mean"] = (
            principal_angles_from_bases(aggregate_pca_basis_a, aggregate_pca_basis_b)
        )
    return aggregate_angles, aggregate_overlay_angles


@app.cell(hide_code=True)
def _(
    aggregate_angles,
    aggregate_average_method,
    aggregate_basis_a,
    aggregate_group_a,
    aggregate_group_b,
    aggregate_rank,
    include_null_input,
    intrinsic_options,
    null_band_input,
    null_draws_input,
    null_seed_input,
    principal_angle_null_band,
    sample_fold_averaged_random_principal_angles,
):
    aggregate_null_angles = None
    aggregate_null_band = None
    if include_null_input.value:
        aggregate_null_angles = sample_fold_averaged_random_principal_angles(
            aggregate_basis_a.shape[0],
            aggregate_rank,
            len(aggregate_group_a),
            len(aggregate_group_b),
            int(null_draws_input.value),
            int(null_seed_input.value),
            method=aggregate_average_method,
            intrinsic_options=intrinsic_options,
        )
        aggregate_null_band = principal_angle_null_band(
            aggregate_null_angles,
            null_band_input.value,
        )
    _ = aggregate_angles, aggregate_group_a, aggregate_group_b
    return aggregate_null_angles, aggregate_null_band


@app.cell(hide_code=True)
def _(
    aggregate_angles,
    aggregate_average_method,
    aggregate_group_a,
    aggregate_group_b,
    aggregate_null_angles,
    aggregate_null_band,
    aggregate_overlay_angles,
    aggregate_pca_group_a,
    aggregate_pca_group_b,
    mo,
    plot_principal_angles,
):
    _fig = plot_principal_angles(
        aggregate_angles,
        title="Principal angles between fold-averaged subspaces",
        primary_label="A RRR mean vs B RRR mean",
        null_band=aggregate_null_band,
        overlay_angles=aggregate_overlay_angles,
    )
    _notes = [
        f"A: `{aggregate_group_a[0].unit_dir}`",
        f"B: `{aggregate_group_b[0].unit_dir}`",
        f"Averaging method: `{aggregate_average_method}`",
    ]
    if aggregate_overlay_angles:
        _notes.extend(
            [
                f"PCA A: `{aggregate_pca_group_a[0].unit_dir}`",
                f"PCA B: `{aggregate_pca_group_b[0].unit_dir}`",
            ]
        )
    if aggregate_null_angles is not None:
        _notes.append(
            "Fold-averaged null draws: "
            f"{aggregate_null_angles.shape[0]} "
            f"(A folds={len(aggregate_group_a)}, B folds={len(aggregate_group_b)})"
        )
    mo.vstack([mo.md("\n\n".join(_notes)), _fig])
    return


@app.cell(hide_code=True)
def _(
    aggregate_angles,
    aggregate_diag_a,
    aggregate_diag_b,
    aggregate_group_a,
    aggregate_group_b,
    aggregate_overlay_angles,
    aggregate_pca_diag_a,
    aggregate_pca_diag_b,
    aggregate_pca_group_a,
    aggregate_pca_group_b,
    aggregate_rank,
    mo,
    np,
    pd,
    subspace_distance_metrics,
):
    _metrics = pd.DataFrame(
        [
            {
                "comparison": "A RRR mean vs B RRR mean",
                **subspace_distance_metrics(aggregate_angles),
            }
        ]
    )
    if aggregate_overlay_angles:
        _metrics = pd.concat(
            [
                _metrics,
                pd.DataFrame(
                    [
                        {"comparison": _label, **subspace_distance_metrics(_angles)}
                        for _label, _angles in aggregate_overlay_angles.items()
                    ]
                ),
            ],
            ignore_index=True,
        )
    for _col in ["grassmann", "fubini_study"]:
        _metrics[f"{_col}_degrees"] = np.degrees(_metrics[_col])

    def _summary_row(label, group, diagnostics):
        return {
            "group": label,
            "path": str(group[0].unit_dir),
            "kind": group[0].kind,
            "folds": ",".join(str(source.fold) for source in group),
            "n_folds": len(group),
            "ambient_dim": group[0].ambient_dim,
            "rank": aggregate_rank,
            "method": diagnostics["method"],
            "converged": diagnostics["converged"],
            "stop_reason": diagnostics["stop_reason"],
            "iterations": diagnostics["iterations"],
            "final_objective": diagnostics["final_objective"],
            "final_gradient_norm": diagnostics["final_gradient_norm"],
            "final_step_size": diagnostics["final_step_size"],
            "accepted_steps": diagnostics["accepted_steps"],
            "rejected_backtracks": diagnostics["rejected_backtracks"],
            "projector_energy": diagnostics["projector_energy"],
            "retained_projector_eigenvalues": ", ".join(
                f"{value:.4g}" for value in diagnostics["projector_eigenvalues"]
            ),
        }

    _summary = pd.DataFrame(
        [
            _summary_row("A", aggregate_group_a, aggregate_diag_a),
            _summary_row("B", aggregate_group_b, aggregate_diag_b),
        ]
    )
    if aggregate_overlay_angles:
        _summary = pd.concat(
            [
                _summary,
                pd.DataFrame(
                    [
                        _summary_row("PCA A", aggregate_pca_group_a, aggregate_pca_diag_a),
                        _summary_row("PCA B", aggregate_pca_group_b, aggregate_pca_diag_b),
                    ]
                ),
            ],
            ignore_index=True,
        )
    _trace_rows = []
    for _label, _diagnostics in [
        ("A", aggregate_diag_a),
        ("B", aggregate_diag_b),
        ("PCA A", aggregate_pca_diag_a),
        ("PCA B", aggregate_pca_diag_b),
    ]:
        if _diagnostics is None:
            continue
        for _entry in _diagnostics.get("trace", []):
            _trace_rows.append({"group": _label, **_entry})
    _trace = pd.DataFrame(_trace_rows)
    mo.vstack(
        [
            mo.md("### Fold-averaged distance metrics"),
            mo.ui.table(_metrics, selection=None),
            mo.md("### Averaging and optimization diagnostics"),
            mo.ui.table(_summary, selection=None),
            mo.md("### Intrinsic mean optimization trace"),
            mo.ui.table(_trace, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    aggregate_average_method,
    aggregate_cross_rank_input,
    aggregate_group_a,
    aggregate_group_b,
    average_subspace_bases,
    filter_available_ranks,
    intrinsic_options,
    load_fold_bases,
    mo,
    np,
    parse_rank_list,
    pca_sibling_group,
    pd,
    plot_cross_rank_metric,
    principal_angles_from_bases,
    subspace_distance_metrics,
):
    try:
        _requested_ranks = parse_rank_list(aggregate_cross_rank_input.value)
    except ValueError as exc:
        mo.stop(True, mo.callout(mo.md(str(exc)), kind="warn"))

    _rows = []
    _notes = []

    def _group_max_rank(group):
        return min(source.max_rank for source in group)

    def _group_ambient(group):
        return {source.ambient_dim for source in group}

    def _mean_basis(group, rank):
        _fold_bases = load_fold_bases(group, rank)
        return average_subspace_bases(
            _fold_bases,
            rank,
            method=aggregate_average_method,
            intrinsic_options=intrinsic_options,
        )[0]

    def _append_grassmann_rows(label, left_group, right_group):
        if left_group is None or right_group is None:
            _notes.append(f"`{label}` omitted because a PCA sibling group was not found.")
            return
        _left_ambient = _group_ambient(left_group)
        _right_ambient = _group_ambient(right_group)
        if len(_left_ambient) != 1 or len(_right_ambient) != 1 or _left_ambient != _right_ambient:
            _notes.append(
                f"`{label}` omitted because ambient dimensions differ: "
                f"A={sorted(_left_ambient)}, B={sorted(_right_ambient)}."
            )
            return
        _valid_ranks, _skipped_ranks = filter_available_ranks(
            _requested_ranks,
            min(_group_max_rank(left_group), _group_max_rank(right_group)),
        )
        if _skipped_ranks:
            _notes.append(
                f"`{label}` skipped unavailable ranks: "
                + ", ".join(str(rank) for rank in _skipped_ranks)
            )
        for _rank in _valid_ranks:
            _left_basis = _mean_basis(left_group, _rank)
            _right_basis = _mean_basis(right_group, _rank)
            _metrics = subspace_distance_metrics(
                principal_angles_from_bases(_left_basis, _right_basis)
            )
            _rows.append(
                {
                    "rank": _rank,
                    "comparison": label,
                    "grassmann": _metrics["grassmann"],
                    "grassmann_degrees": np.degrees(_metrics["grassmann"]),
                }
            )

    _pca_group_a = pca_sibling_group(aggregate_group_a)
    _pca_group_b = pca_sibling_group(aggregate_group_b)
    _append_grassmann_rows(
        "A RRR mean vs B RRR mean",
        aggregate_group_a,
        aggregate_group_b,
    )
    _append_grassmann_rows(
        "A RRR mean vs B PCA mean",
        aggregate_group_a,
        _pca_group_b,
    )
    _append_grassmann_rows(
        "A PCA mean vs B RRR mean",
        _pca_group_a,
        aggregate_group_b,
    )
    _append_grassmann_rows(
        "A PCA mean vs B PCA mean",
        _pca_group_a,
        _pca_group_b,
    )

    _df = pd.DataFrame(_rows)
    mo.stop(
        _df.empty,
        mo.callout(
            mo.md("No cross-rank fold-averaged comparisons could be computed."),
            kind="warn",
        ),
    )
    _items = [
        mo.md("### Cross-rank fold-averaged Grassmann summary"),
        mo.md(f"Averaging method: `{aggregate_average_method}`"),
        plot_cross_rank_metric(
            _df,
            "grassmann",
            title="Cross-rank fold-averaged subspace comparison",
            ylabel="Grassmann distance (radians)",
        ),
        mo.ui.table(_df, selection=None),
    ]
    if _notes:
        _items.insert(2, mo.callout(mo.md("\n\n".join(_notes)), kind="warn"))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md("""
    ## RSA over Projected Word Embeddings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rsa_config_a_input = mo.ui.text(
        value="configs/rrr/gpt2-xl/cross_subject.yml",
        label="RRR config 1",
        full_width=True,
    )
    rsa_config_b_input = mo.ui.text(
        value="configs/brain_treebank/cars_2/rrr_gpt2xl_layer24.yml",
        label="RRR config 2",
        full_width=True,
    )
    rsa_metric_input = mo.ui.dropdown(
        options={"Cosine distance": "cosine", "Euclidean distance": "euclidean"},
        value="Cosine distance",
        label="RSM distance",
    )

    mo.vstack([rsa_config_a_input, rsa_config_b_input, rsa_metric_input])
    return rsa_config_a_input, rsa_config_b_input, rsa_metric_input


@app.cell(hide_code=True)
def _(
    RRRConfig,
    expand_per_model_layer_configs,
    load_config,
    mo,
    rsa_config_a_input,
    rsa_config_b_input,
):
    def _load_expanded_rrr_configs(path_value, label):
        from pathlib import Path

        path_text = str(path_value).strip()
        mo.stop(
            not path_text,
            mo.callout(mo.md(f"Enter `{label}` to enable the RSA section."), kind="warn"),
        )
        path = Path(path_text).expanduser()
        mo.stop(
            not path.exists(),
            mo.callout(mo.md(f"`{label}` does not exist: `{path}`"), kind="warn"),
        )
        return expand_per_model_layer_configs(load_config(str(path), {}, RRRConfig))

    rsa_configs_a = _load_expanded_rrr_configs(rsa_config_a_input.value, "RRR config 1")
    rsa_configs_b = _load_expanded_rrr_configs(rsa_config_b_input.value, "RRR config 2")
    return rsa_configs_a, rsa_configs_b


@app.cell(hide_code=True)
def _(mo, rsa_configs_a, rsa_configs_b):
    def _rsa_config_options(configs):
        options = {}
        for idx, config in enumerate(configs):
            enc = config.encoding
            model_cfg = ", ".join(
                f"{key}={value}" for key, value in sorted(enc.per_model_config.items())
            )
            stimulus = enc.per_dataset_config.get("movie", enc.dataset)
            options[
                (
                    f"{idx}: dataset={enc.dataset}, stimulus={stimulus}, "
                    f"model={enc.model}, {model_cfg}"
                )
            ] = idx
        return options

    rsa_config_a_select = mo.ui.dropdown(
        options=_rsa_config_options(rsa_configs_a),
        value=list(_rsa_config_options(rsa_configs_a).keys())[0],
        label="Expanded config 1",
        full_width=True,
    )
    rsa_config_b_select = mo.ui.dropdown(
        options=_rsa_config_options(rsa_configs_b),
        value=list(_rsa_config_options(rsa_configs_b).keys())[0],
        label="Expanded config 2",
        full_width=True,
    )
    mo.vstack([rsa_config_a_select, rsa_config_b_select])
    return rsa_config_a_select, rsa_config_b_select


@app.cell(hide_code=True)
def _(
    aggregate_primary_group_options,
    mo,
    rsa_config_a_select,
    rsa_config_b_select,
    rsa_configs_a,
    rsa_configs_b,
):
    rsa_config_a = rsa_configs_a[rsa_config_a_select.value]
    rsa_config_b = rsa_configs_b[rsa_config_b_select.value]
    _group_labels = list(aggregate_primary_group_options.keys())
    _default_b = 1 if len(_group_labels) > 1 else 0
    rsa_group_a_input = mo.ui.dropdown(
        options=aggregate_primary_group_options,
        value=_group_labels[0],
        label="A-matrix group 1",
        full_width=True,
    )
    rsa_group_b_input = mo.ui.dropdown(
        options=aggregate_primary_group_options,
        value=_group_labels[_default_b],
        label="A-matrix group 2",
        full_width=True,
    )
    mo.vstack([rsa_group_a_input, rsa_group_b_input])
    return rsa_config_a, rsa_config_b, rsa_group_a_input, rsa_group_b_input


@app.cell(hide_code=True)
def _(
    aggregate_group_list,
    default_cross_rank_text,
    mo,
    rsa_group_a_input,
    rsa_group_b_input,
):
    rsa_group_a = aggregate_group_list[rsa_group_a_input.value]
    rsa_group_b = aggregate_group_list[rsa_group_b_input.value]
    _max_rank_a = min(source.max_rank for source in rsa_group_a)
    _max_rank_b = min(source.max_rank for source in rsa_group_b)
    rsa_rank_a_input = mo.ui.number(
        start=1,
        stop=_max_rank_a,
        step=1,
        value=min(10, _max_rank_a),
        label="RSA rank 1",
    )
    rsa_rank_b_input = mo.ui.number(
        start=1,
        stop=_max_rank_b,
        step=1,
        value=min(10, _max_rank_b),
        label="RSA rank 2",
    )
    rsa_cross_rank_input = mo.ui.text(
        value=default_cross_rank_text(),
        label="Cross-rank RSA ranks",
        full_width=True,
    )
    cca_cross_rank_input = mo.ui.text(
        value="1,2,4,8,16,32",
        label="Cross-rank CCA ranks",
        full_width=True,
    )
    cca_shared_threshold_input = mo.ui.number(
        start=-1.0,
        stop=1.0,
        step=0.01,
        value=0.2,
        label="CCA shared corr threshold",
    )
    mo.vstack(
        [
            mo.hstack([rsa_rank_a_input, rsa_rank_b_input]),
            rsa_cross_rank_input,
            cca_cross_rank_input,
            cca_shared_threshold_input,
        ]
    )
    return (
        cca_cross_rank_input,
        cca_shared_threshold_input,
        rsa_cross_rank_input,
        rsa_group_a,
        rsa_group_b,
        rsa_rank_a_input,
        rsa_rank_b_input,
    )


@app.cell(hide_code=True)
def _(
    ScriptMode,
    build_XY_from_subjects,
    get_dataset_adapter,
    get_glove_embeddings,
    get_results_dir,
    load_embeddings,
    load_encoding_results,
    load_sig_electrodes,
    np,
    orthonormal_basis,
    resolve_dataset_context,
):
    def _mode_value(mode):
        return mode.value if hasattr(mode, "value") else str(mode)

    def _resolve_rrr_paths(config):
        enc = config.encoding
        subject_ids = enc.subjects or list(range(1, 10))
        context = resolve_dataset_context(
            dataset=enc.dataset,
            subject_ids=subject_ids,
            per_dataset_config=enc.per_dataset_config,
            data_root=enc.data_root,
        )
        get_dataset_adapter(enc.dataset).validate_subjects_share_stimulus(
            subject_ids,
            enc.per_dataset_config,
            enc.data_root,
        )
        output_dir = config.output_dir or get_results_dir(
            enc.model,
            enc.offset,
            enc.per_model_config,
            dataset_context=context,
        )
        mode = _mode_value(config.mode)
        if config.rrr_output_dir:
            rrr_dir = config.rrr_output_dir
        elif mode == ScriptMode.PER_REGION.value:
            import os

            rrr_dir = os.path.join(output_dir, "per_region_rrr")
        elif mode == ScriptMode.PER_SUBJECT.value:
            import os

            rrr_dir = os.path.join(output_dir, "per_subject_rrr")
        elif mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT.value:
            import os

            rrr_dir = os.path.join(output_dir, "loso_rrr")
        else:
            import os

            rrr_dir = os.path.join(output_dir, "rrr")
        return {
            "context": context,
            "subject_ids": subject_ids,
            "output_dir": output_dir,
            "sig_dir": config.sig_dir or output_dir,
            "rrr_dir": rrr_dir,
        }

    def _load_shared_selection_metadata(sig_dir, subject_ids):
        import os

        reference = None
        for sub_id in subject_ids:
            path = os.path.join(
                sig_dir,
                "subjects",
                f"sub_{sub_id}",
                "encoding_results.joblib",
            )
            enc_results = load_encoding_results(path)
            metadata = enc_results.get("word_selection")
            if metadata is None:
                raise ValueError(
                    "shared_word_selection=True requires word_selection metadata in "
                    f"{path}"
                )
            if reference is None:
                reference = metadata
                continue
            for key in ("epoch_selection", "analysis_selection"):
                if not np.array_equal(reference[key], metadata[key]):
                    raise ValueError(
                        "Shared word-selection metadata differs across subjects "
                        f"for key {key!r}."
                    )
        return reference

    def _load_reference_fold_idxs(output_dir, subject_ids, n_rows):
        import os

        path = os.path.join(
            output_dir,
            "subjects",
            f"sub_{subject_ids[0]}",
            "encoding_results.joblib",
        )
        enc_results = load_encoding_results(path)
        test_idxs = enc_results["test_idxs"]
        train_idxs = enc_results.get("train_idxs")
        if train_idxs is None:
            all_idxs = np.arange(int(n_rows))
            train_idxs = [
                np.setdiff1d(all_idxs, np.asarray(test_idx, dtype=int), assume_unique=False)
                for test_idx in test_idxs
            ]
        return train_idxs, test_idxs

    def load_rsa_example(config, group, rank):
        enc = config.encoding
        paths = _resolve_rrr_paths(config)
        sig_electrodes, _ = load_sig_electrodes(
            paths["output_dir"],
            paths["subject_ids"],
            sig_dir=paths["sig_dir"],
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            dataset_context=paths["context"],
            data_root=enc.data_root,
        )
        aligned_embeddings, word_df = load_embeddings(
            enc.data_root,
            enc.model,
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            dataset_context=paths["context"],
            **enc.per_model_config,
        )
        word_in_glove = None
        if enc.limit_to_glove or enc.model == "glove":
            word_in_glove = get_glove_embeddings(
                enc.data_root,
                word_df,
                **enc.per_model_config,
            )[1]["in_glove"]
        shared_metadata = (
            _load_shared_selection_metadata(paths["sig_dir"], paths["subject_ids"])
            if enc.shared_word_selection
            else None
        )
        X_raw, _, _, _ = build_XY_from_subjects(
            paths["subject_ids"],
            enc,
            sig_electrodes,
            aligned_embeddings,
            valid_rows=word_in_glove,
            shared_selection=(
                shared_metadata["epoch_selection"]
                if shared_metadata is not None
                else None
            ),
        )
        train_idxs, test_idxs = _load_reference_fold_idxs(
            paths["output_dir"],
            paths["subject_ids"],
            X_raw.shape[0],
        )
        max_test_idx = max(int(np.max(idx)) for idx in test_idxs if len(idx))
        if max_test_idx >= X_raw.shape[0]:
            raise ValueError(
                "Saved fold indices are incompatible with loaded RSA rows: "
                f"max test index {max_test_idx} for {X_raw.shape[0]} rows."
            )
        if X_raw.shape[1] != group[0].ambient_dim:
            raise ValueError(
                "Embedding dimension does not match selected A matrix ambient dimension: "
                f"{X_raw.shape[1]} vs {group[0].ambient_dim} for {group[0].unit_dir}."
            )
        rank = int(rank)
        bases_by_fold = {}
        for source in group:
            matrix = np.load(source.path, mmap_mode="r")
            bases_by_fold[source.fold] = orthonormal_basis(matrix, rank)
        missing = sorted(set(range(len(test_idxs))) - set(bases_by_fold))
        if missing:
            raise ValueError(
                f"Selected group {group[0].unit_dir} is missing A.npy for folds {missing}."
            )
        return {
            "config": config,
            "paths": paths,
            "dataset_key": (
                paths["context"].dataset,
                getattr(paths["context"], "stimulus_id", enc.dataset),
            ),
            "X": X_raw,
            "train_idxs": train_idxs,
            "test_idxs": test_idxs,
            "bases_by_fold": bases_by_fold,
            "group": group,
            "rank": rank,
        }

    return (load_rsa_example,)


@app.cell(hide_code=True)
def _(
    load_rsa_example,
    pca_sibling_group,
    rsa_config_a,
    rsa_config_b,
    rsa_group_a,
    rsa_group_b,
    rsa_rank_a_input,
    rsa_rank_b_input,
):
    rsa_example_a = load_rsa_example(
        rsa_config_a,
        rsa_group_a,
        int(rsa_rank_a_input.value),
    )
    rsa_example_b = load_rsa_example(
        rsa_config_b,
        rsa_group_b,
        int(rsa_rank_b_input.value),
    )
    rsa_pca_group_a = pca_sibling_group(rsa_group_a)
    rsa_pca_group_b = pca_sibling_group(rsa_group_b)
    rsa_pca_example_a = None
    rsa_pca_example_b = None
    if rsa_pca_group_a is not None and rsa_pca_group_b is not None:
        try:
            rsa_pca_example_a = load_rsa_example(
                rsa_config_a,
                rsa_pca_group_a,
                int(rsa_rank_a_input.value),
            )
            rsa_pca_example_b = load_rsa_example(
                rsa_config_b,
                rsa_pca_group_b,
                int(rsa_rank_b_input.value),
            )
        except Exception:
            rsa_pca_example_a = None
            rsa_pca_example_b = None
    return rsa_example_a, rsa_example_b, rsa_pca_example_a, rsa_pca_example_b


@app.cell(hide_code=True)
def _(mo, np, plt, projected_rsm):
    def rsa_rsms_for_pair(X_a, X_b, test_idx, basis_a, basis_b, metric):
        _idx = np.asarray(test_idx, dtype=int)
        rsm_a = projected_rsm(np.asarray(X_a)[_idx] @ basis_a, metric=metric)
        rsm_b = projected_rsm(np.asarray(X_b)[_idx] @ basis_b, metric=metric)
        return rsm_a, rsm_b

    def plot_rsm_pair(rsm_a, rsm_b, title_a, title_b):
        vmax = float(np.nanmax([np.nanmax(rsm_a), np.nanmax(rsm_b)]))
        vmin = float(np.nanmin([np.nanmin(rsm_a), np.nanmin(rsm_b)]))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
        im0 = axes[0].imshow(rsm_a, vmin=vmin, vmax=vmax, aspect="auto")
        axes[0].set_title(title_a)
        axes[0].set_xlabel("Test word")
        axes[0].set_ylabel("Test word")
        axes[1].imshow(rsm_b, vmin=vmin, vmax=vmax, aspect="auto")
        axes[1].set_title(title_b)
        axes[1].set_xlabel("Test word")
        axes[1].set_ylabel("Test word")
        fig.colorbar(im0, ax=axes, shrink=0.85, label="Distance")
        return fig

    _ = mo
    return plot_rsm_pair, rsa_rsms_for_pair


@app.cell(hide_code=True)
def _(
    mo,
    np,
    pd,
    projected_rsm,
    rsa_lower_triangle_correlations,
    rsa_metric_input,
    stacked_svd_subspace_mean,
    summarize_rsa_fold_results,
    validate_matching_fold_indices,
):
    def compute_rsa_outputs(
        example_a,
        example_b,
        pca_example_a=None,
        pca_example_b=None,
    ):
        def _correlate_pair(X_a, X_b, test_idx, basis_a, basis_b, metric):
            _idx = np.asarray(test_idx, dtype=int)
            _rsm_a = projected_rsm(np.asarray(X_a)[_idx] @ basis_a, metric=metric)
            _rsm_b = projected_rsm(np.asarray(X_b)[_idx] @ basis_b, metric=metric)
            return rsa_lower_triangle_correlations(_rsm_a, _rsm_b)

        _distance_metric = rsa_metric_input.value
        _same_dataset = example_a["dataset_key"] == example_b["dataset_key"]
        _tables = {}
        _cross_mean_bases = {}
        _plot_specs = {}
        _has_pca_baseline = pca_example_a is not None and pca_example_b is not None

        if _same_dataset:
            try:
                validate_matching_fold_indices(
                    example_a["test_idxs"],
                    example_b["test_idxs"],
                )
            except ValueError as exc:
                mo.stop(True, mo.callout(mo.md(str(exc)), kind="warn"))
            if _has_pca_baseline:
                try:
                    validate_matching_fold_indices(
                        pca_example_a["test_idxs"],
                        pca_example_b["test_idxs"],
                    )
                except ValueError as exc:
                    mo.stop(True, mo.callout(mo.md(f"PCA baseline: {exc}"), kind="warn"))
                for _label, _left, _right in [
                    ("1 RRR vs 1 PCA", example_a, pca_example_a),
                    ("2 RRR vs 2 PCA", example_b, pca_example_b),
                    ("1 RRR vs 2 PCA", example_a, pca_example_b),
                    ("1 PCA vs 2 RRR", pca_example_a, example_b),
                ]:
                    try:
                        validate_matching_fold_indices(
                            _left["test_idxs"],
                            _right["test_idxs"],
                        )
                    except ValueError as exc:
                        mo.stop(True, mo.callout(mo.md(f"{_label}: {exc}"), kind="warn"))
            _rows = []
            _within_specs = [
                ("RRR-vs-RRR", "1 RRR vs 2 RRR", example_a, example_b)
            ]
            if _has_pca_baseline:
                _within_specs.extend(
                    [
                        ("RRR-vs-PCA", "1 RRR vs 2 PCA", example_a, pca_example_b),
                        ("PCA-vs-RRR", "1 PCA vs 2 RRR", pca_example_a, example_b),
                        ("PCA-vs-PCA", "1 PCA vs 2 PCA", pca_example_a, pca_example_b),
                    ]
                )
            for _basis_type, _comparison, _example_a, _example_b in _within_specs:
                for _fold, _test_idx in enumerate(_example_a["test_idxs"]):
                    _corrs = _correlate_pair(
                        _example_a["X"],
                        _example_b["X"],
                        _test_idx,
                        _example_a["bases_by_fold"][_fold],
                        _example_b["bases_by_fold"][_fold],
                        _distance_metric,
                    )
                    _rows.append(
                        {
                            "basis": _basis_type,
                            "comparison": _comparison,
                            "fold": _fold,
                            "n_items": int(len(_test_idx)),
                            **_corrs,
                        }
                    )
            _tables["within"] = pd.DataFrame(_rows)
            _plot_specs = {
                _comparison: {
                    "left": _example_a,
                    "right": _example_b,
                    "left_title": f"{_comparison.split(' vs ')[0]} fold",
                    "right_title": f"{_comparison.split(' vs ')[1]} fold",
                }
                for _, _comparison, _example_a, _example_b in _within_specs
            }
            _summary = pd.DataFrame(
                [
                    {
                        "basis": _basis_type,
                        "comparison": _comparison,
                        **summarize_rsa_fold_results(
                            [
                                row
                                for row in _rows
                                if row["basis"] == _basis_type
                                and row["comparison"] == _comparison
                            ]
                        ),
                    }
                    for _basis_type, _comparison in sorted(
                        {(row["basis"], row["comparison"]) for row in _rows}
                    )
                ]
            )
        else:
            _basis_a_mean, _ = stacked_svd_subspace_mean(
                [
                    example_a["bases_by_fold"][fold]
                    for fold in sorted(example_a["bases_by_fold"])
                ],
                example_a["rank"],
            )
            _basis_b_mean, _ = stacked_svd_subspace_mean(
                [
                    example_b["bases_by_fold"][fold]
                    for fold in sorted(example_b["bases_by_fold"])
                ],
                example_b["rank"],
            )
            _cross_mean_bases = {"1": _basis_a_mean, "2": _basis_b_mean}
            _direction_specs = [
                (
                    "RRR-vs-RRR",
                    "1 RRR fold vs 2 RRR mean",
                    example_a,
                    _basis_b_mean,
                    "1 RRR fold",
                    "2 RRR mean",
                ),
                (
                    "RRR-vs-RRR",
                    "2 RRR fold vs 1 RRR mean",
                    example_b,
                    _basis_a_mean,
                    "2 RRR fold",
                    "1 RRR mean",
                ),
            ]
            if _has_pca_baseline:
                _pca_basis_a_mean, _ = stacked_svd_subspace_mean(
                    [
                        pca_example_a["bases_by_fold"][fold]
                        for fold in sorted(pca_example_a["bases_by_fold"])
                    ],
                    pca_example_a["rank"],
                )
                _pca_basis_b_mean, _ = stacked_svd_subspace_mean(
                    [
                        pca_example_b["bases_by_fold"][fold]
                        for fold in sorted(pca_example_b["bases_by_fold"])
                    ],
                    pca_example_b["rank"],
                )
                _cross_mean_bases.update(
                    {"pca_1": _pca_basis_a_mean, "pca_2": _pca_basis_b_mean}
                )
                _direction_specs.extend(
                    [
                        (
                            "RRR-vs-PCA",
                            "1 RRR fold vs 2 PCA mean",
                            example_a,
                            _pca_basis_b_mean,
                            "1 RRR fold",
                            "2 PCA mean",
                        ),
                        (
                            "PCA-vs-RRR",
                            "1 PCA fold vs 2 RRR mean",
                            pca_example_a,
                            _basis_b_mean,
                            "1 PCA fold",
                            "2 RRR mean",
                        ),
                        (
                            "PCA-vs-PCA",
                            "1 PCA fold vs 2 PCA mean",
                            pca_example_a,
                            _pca_basis_b_mean,
                            "1 PCA fold",
                            "2 PCA mean",
                        ),
                        (
                            "RRR-vs-PCA",
                            "2 RRR fold vs 1 PCA mean",
                            example_b,
                            _pca_basis_a_mean,
                            "2 RRR fold",
                            "1 PCA mean",
                        ),
                        (
                            "PCA-vs-RRR",
                            "2 PCA fold vs 1 RRR mean",
                            pca_example_b,
                            _basis_a_mean,
                            "2 PCA fold",
                            "1 RRR mean",
                        ),
                        (
                            "PCA-vs-PCA",
                            "2 PCA fold vs 1 PCA mean",
                            pca_example_b,
                            _pca_basis_a_mean,
                            "2 PCA fold",
                            "1 PCA mean",
                        ),
                    ]
                )
            _summary_rows = []
            for (
                _basis_type,
                _label,
                _target,
                _source_mean,
                _left_title,
                _right_title,
            ) in _direction_specs:
                mo.stop(
                    _target["X"].shape[1] != _source_mean.shape[0],
                    mo.callout(
                        mo.md(
                            f"Cross-dataset `{_label}` needs the target embeddings to "
                            "share the source subspace ambient dimension. Got "
                            f"{_target['X'].shape[1]} vs {_source_mean.shape[0]}."
                        ),
                        kind="warn",
                    ),
                )
                _rows = []
                for _fold, _test_idx in enumerate(_target["test_idxs"]):
                    _corrs = _correlate_pair(
                        _target["X"],
                        _target["X"],
                        _test_idx,
                        _target["bases_by_fold"][_fold],
                        _source_mean,
                        _distance_metric,
                    )
                    _rows.append(
                        {
                            "basis": _basis_type,
                            "comparison": _label,
                            "fold": _fold,
                            "n_items": int(len(_test_idx)),
                            **_corrs,
                        }
                    )
                if _label in _tables:
                    _tables[_label] = pd.concat(
                        [_tables[_label], pd.DataFrame(_rows)],
                        ignore_index=True,
                    )
                else:
                    _tables[_label] = pd.DataFrame(_rows)
                _summary_rows.append(
                    {
                        "basis": _basis_type,
                        "comparison": _label,
                        **summarize_rsa_fold_results(_rows),
                    }
                )
                _plot_specs[_label] = {
                    "target": _target,
                    "source_mean": _source_mean,
                    "left_title": _left_title,
                    "right_title": _right_title,
                }
            _summary = pd.DataFrame(_summary_rows)
        return {
            "cross_mean_bases": _cross_mean_bases,
            "distance_metric": _distance_metric,
            "plot_specs": _plot_specs,
            "same_dataset": _same_dataset,
            "summary": _summary,
            "tables": _tables,
        }

    return (compute_rsa_outputs,)


@app.cell(hide_code=True)
def _(
    mo,
    np,
    pd,
    projected_rsm,
    rsa_example_a,
    rsa_example_b,
    rsa_lower_triangle_correlations,
    rsa_metric_input,
    rsa_pca_example_a,
    rsa_pca_example_b,
    stacked_svd_subspace_mean,
    summarize_rsa_fold_results,
    validate_matching_fold_indices,
):
    def _correlate_pair(X_a, X_b, test_idx, basis_a, basis_b, metric):
        _idx = np.asarray(test_idx, dtype=int)
        _rsm_a = projected_rsm(np.asarray(X_a)[_idx] @ basis_a, metric=metric)
        _rsm_b = projected_rsm(np.asarray(X_b)[_idx] @ basis_b, metric=metric)
        return rsa_lower_triangle_correlations(_rsm_a, _rsm_b)

    rsa_distance_metric = rsa_metric_input.value
    rsa_same_dataset = rsa_example_a["dataset_key"] == rsa_example_b["dataset_key"]
    rsa_tables = {}
    rsa_cross_mean_bases = {}
    rsa_plot_specs = {}
    rsa_has_pca_baseline = (
        rsa_pca_example_a is not None and rsa_pca_example_b is not None
    )

    if rsa_same_dataset:
        try:
            validate_matching_fold_indices(
                rsa_example_a["test_idxs"],
                rsa_example_b["test_idxs"],
            )
        except ValueError as exc:
            mo.stop(True, mo.callout(mo.md(str(exc)), kind="warn"))
        if rsa_has_pca_baseline:
            try:
                validate_matching_fold_indices(
                    rsa_pca_example_a["test_idxs"],
                    rsa_pca_example_b["test_idxs"],
                )
            except ValueError as exc:
                mo.stop(True, mo.callout(mo.md(f"PCA baseline: {exc}"), kind="warn"))
            for _label, _left, _right in [
                ("1 RRR vs 1 PCA", rsa_example_a, rsa_pca_example_a),
                ("2 RRR vs 2 PCA", rsa_example_b, rsa_pca_example_b),
                ("1 RRR vs 2 PCA", rsa_example_a, rsa_pca_example_b),
                ("1 PCA vs 2 RRR", rsa_pca_example_a, rsa_example_b),
            ]:
                try:
                    validate_matching_fold_indices(
                        _left["test_idxs"],
                        _right["test_idxs"],
                    )
                except ValueError as exc:
                    mo.stop(True, mo.callout(mo.md(f"{_label}: {exc}"), kind="warn"))
        _rows = []
        _within_specs = [
            ("RRR-vs-RRR", "1 RRR vs 2 RRR", rsa_example_a, rsa_example_b)
        ]
        if rsa_has_pca_baseline:
            _within_specs.extend(
                [
                    ("RRR-vs-PCA", "1 RRR vs 2 PCA", rsa_example_a, rsa_pca_example_b),
                    ("PCA-vs-RRR", "1 PCA vs 2 RRR", rsa_pca_example_a, rsa_example_b),
                    ("PCA-vs-PCA", "1 PCA vs 2 PCA", rsa_pca_example_a, rsa_pca_example_b),
                ]
            )
        for _basis_type, _comparison, _example_a, _example_b in _within_specs:
            for _fold, _test_idx in enumerate(_example_a["test_idxs"]):
                _corrs = _correlate_pair(
                    _example_a["X"],
                    _example_b["X"],
                    _test_idx,
                    _example_a["bases_by_fold"][_fold],
                    _example_b["bases_by_fold"][_fold],
                    rsa_distance_metric,
                )
                _rows.append(
                    {
                        "basis": _basis_type,
                        "comparison": _comparison,
                        "fold": _fold,
                        "n_items": int(len(_test_idx)),
                        **_corrs,
                    }
                )
        rsa_tables["within"] = pd.DataFrame(_rows)
        rsa_plot_specs = {
            _comparison: {
                "left": _example_a,
                "right": _example_b,
                "left_title": f"{_comparison.split(' vs ')[0]} fold",
                "right_title": f"{_comparison.split(' vs ')[1]} fold",
            }
            for _, _comparison, _example_a, _example_b in _within_specs
        }
        rsa_summary = pd.DataFrame(
            [
                {
                    "basis": _basis_type,
                    "comparison": _comparison,
                    **summarize_rsa_fold_results(
                        [
                            row
                            for row in _rows
                            if row["basis"] == _basis_type
                            and row["comparison"] == _comparison
                        ]
                    ),
                }
                for _basis_type, _comparison in sorted(
                    {(row["basis"], row["comparison"]) for row in _rows}
                )
            ]
        )
    else:
        _basis_a_mean, _ = stacked_svd_subspace_mean(
            [rsa_example_a["bases_by_fold"][fold] for fold in sorted(rsa_example_a["bases_by_fold"])],
            rsa_example_a["rank"],
        )
        _basis_b_mean, _ = stacked_svd_subspace_mean(
            [rsa_example_b["bases_by_fold"][fold] for fold in sorted(rsa_example_b["bases_by_fold"])],
            rsa_example_b["rank"],
        )
        rsa_cross_mean_bases = {"1": _basis_a_mean, "2": _basis_b_mean}
        _direction_specs = [
            (
                "RRR-vs-RRR",
                "1 RRR fold vs 2 RRR mean",
                rsa_example_a,
                _basis_b_mean,
                "1 RRR fold",
                "2 RRR mean",
            ),
            (
                "RRR-vs-RRR",
                "2 RRR fold vs 1 RRR mean",
                rsa_example_b,
                _basis_a_mean,
                "2 RRR fold",
                "1 RRR mean",
            ),
        ]
        if rsa_has_pca_baseline:
            _pca_basis_a_mean, _ = stacked_svd_subspace_mean(
                [
                    rsa_pca_example_a["bases_by_fold"][fold]
                    for fold in sorted(rsa_pca_example_a["bases_by_fold"])
                ],
                rsa_pca_example_a["rank"],
            )
            _pca_basis_b_mean, _ = stacked_svd_subspace_mean(
                [
                    rsa_pca_example_b["bases_by_fold"][fold]
                    for fold in sorted(rsa_pca_example_b["bases_by_fold"])
                ],
                rsa_pca_example_b["rank"],
            )
            rsa_cross_mean_bases.update(
                {"pca_1": _pca_basis_a_mean, "pca_2": _pca_basis_b_mean}
            )
            _direction_specs.extend(
                [
                    (
                        "RRR-vs-PCA",
                        "1 RRR fold vs 2 PCA mean",
                        rsa_example_a,
                        _pca_basis_b_mean,
                        "1 RRR fold",
                        "2 PCA mean",
                    ),
                    (
                        "PCA-vs-RRR",
                        "1 PCA fold vs 2 RRR mean",
                        rsa_pca_example_a,
                        _basis_b_mean,
                        "1 PCA fold",
                        "2 RRR mean",
                    ),
                    (
                        "PCA-vs-PCA",
                        "1 PCA fold vs 2 PCA mean",
                        rsa_pca_example_a,
                        _pca_basis_b_mean,
                        "1 PCA fold",
                        "2 PCA mean",
                    ),
                    (
                        "RRR-vs-PCA",
                        "2 RRR fold vs 1 PCA mean",
                        rsa_example_b,
                        _pca_basis_a_mean,
                        "2 RRR fold",
                        "1 PCA mean",
                    ),
                    (
                        "PCA-vs-RRR",
                        "2 PCA fold vs 1 RRR mean",
                        rsa_pca_example_b,
                        _basis_a_mean,
                        "2 PCA fold",
                        "1 RRR mean",
                    ),
                    (
                        "PCA-vs-PCA",
                        "2 PCA fold vs 1 PCA mean",
                        rsa_pca_example_b,
                        _pca_basis_a_mean,
                        "2 PCA fold",
                        "1 PCA mean",
                    ),
                ]
            )
        _summary_rows = []
        for (
            _basis_type,
            _label,
            _target,
            _source_mean,
            _left_title,
            _right_title,
        ) in _direction_specs:
            mo.stop(
                _target["X"].shape[1] != _source_mean.shape[0],
                mo.callout(
                    mo.md(
                        f"Cross-dataset `{_label}` needs the target embeddings to "
                        "share the source subspace ambient dimension. Got "
                        f"{_target['X'].shape[1]} vs {_source_mean.shape[0]}."
                    ),
                    kind="warn",
                ),
            )
            _rows = []
            for _fold, _test_idx in enumerate(_target["test_idxs"]):
                _corrs = _correlate_pair(
                    _target["X"],
                    _target["X"],
                    _test_idx,
                    _target["bases_by_fold"][_fold],
                    _source_mean,
                    rsa_distance_metric,
                )
                _rows.append(
                    {
                        "basis": _basis_type,
                        "comparison": _label,
                        "fold": _fold,
                        "n_items": int(len(_test_idx)),
                        **_corrs,
                    }
                )
            if _label in rsa_tables:
                rsa_tables[_label] = pd.concat(
                    [rsa_tables[_label], pd.DataFrame(_rows)],
                    ignore_index=True,
                )
            else:
                rsa_tables[_label] = pd.DataFrame(_rows)
            _summary_rows.append(
                {
                    "basis": _basis_type,
                    "comparison": _label,
                    **summarize_rsa_fold_results(_rows),
                }
            )
            rsa_plot_specs[_label] = {
                "target": _target,
                "source_mean": _source_mean,
                "left_title": _left_title,
                "right_title": _right_title,
            }
        rsa_summary = pd.DataFrame(_summary_rows)
    return (
        rsa_distance_metric,
        rsa_plot_specs,
        rsa_same_dataset,
        rsa_summary,
        rsa_tables,
    )


@app.cell(hide_code=True)
def _(
    compute_rsa_outputs,
    filter_available_ranks,
    load_rsa_example,
    mo,
    parse_rank_list,
    pca_sibling_group,
    pd,
    plot_cross_rank_metric,
    rsa_config_a,
    rsa_config_b,
    rsa_cross_rank_input,
    rsa_group_a,
    rsa_group_b,
):
    try:
        _requested_ranks = parse_rank_list(rsa_cross_rank_input.value)
    except ValueError as exc:
        mo.stop(True, mo.callout(mo.md(str(exc)), kind="warn"))

    _primary_max_rank = min(
        min(source.max_rank for source in rsa_group_a),
        min(source.max_rank for source in rsa_group_b),
    )
    _valid_ranks, _skipped_ranks = filter_available_ranks(
        _requested_ranks,
        _primary_max_rank,
    )
    mo.stop(
        not _valid_ranks,
        mo.callout(mo.md("No requested RSA ranks are available."), kind="warn"),
    )

    _pca_group_a = pca_sibling_group(rsa_group_a)
    _pca_group_b = pca_sibling_group(rsa_group_b)
    _notes = []
    if _skipped_ranks:
        _notes.append(
            "Skipped unavailable primary RSA ranks: "
            + ", ".join(str(rank) for rank in _skipped_ranks)
        )
    if _pca_group_a is None or _pca_group_b is None:
        _notes.append("PCA RSA lines omitted because a PCA sibling group was not found.")

    _summary_rows = []
    for _rank in _valid_ranks:
        _example_a = load_rsa_example(rsa_config_a, rsa_group_a, _rank)
        _example_b = load_rsa_example(rsa_config_b, rsa_group_b, _rank)
        _pca_example_a = None
        _pca_example_b = None
        if _pca_group_a is not None and _pca_group_b is not None:
            _pca_max_rank = min(
                min(source.max_rank for source in _pca_group_a),
                min(source.max_rank for source in _pca_group_b),
            )
            if _rank <= _pca_max_rank:
                try:
                    _pca_example_a = load_rsa_example(rsa_config_a, _pca_group_a, _rank)
                    _pca_example_b = load_rsa_example(rsa_config_b, _pca_group_b, _rank)
                except Exception as exc:
                    _notes.append(f"PCA RSA rank {_rank} omitted: {exc}")
            else:
                _notes.append(f"PCA RSA rank {_rank} omitted; max PCA rank is {_pca_max_rank}.")

        _outputs = compute_rsa_outputs(
            _example_a,
            _example_b,
            _pca_example_a,
            _pca_example_b,
        )
        _summary = _outputs["summary"].copy()
        if _outputs["same_dataset"]:
            for _row in _summary.to_dict("records"):
                _summary_rows.append({"rank": _rank, **_row})
        else:
            _basis_summary = (
                _summary.groupby("basis", as_index=False)[
                    ["mean_pearson", "mean_spearman"]
                ]
                .mean()
                .sort_values("basis")
            )
            for _row in _basis_summary.to_dict("records"):
                _summary_rows.append(
                    {
                        "rank": _rank,
                        "basis": _row["basis"],
                        "comparison": "cross-dataset directional mean",
                        "mean_pearson": _row["mean_pearson"],
                        "mean_spearman": _row["mean_spearman"],
                    }
                )

    _summary_df = pd.DataFrame(_summary_rows)
    mo.stop(
        _summary_df.empty,
        mo.callout(mo.md("No cross-rank RSA summaries could be computed."), kind="warn"),
    )
    _plot_df = _summary_df.copy()
    _plot_df["comparison"] = _plot_df["basis"]
    _items = [
        mo.md("### Cross-rank RSA summary"),
        mo.ui.table(_summary_df, selection=None),
        plot_cross_rank_metric(
            _plot_df,
            "mean_spearman",
            title="Cross-rank RSA Spearman summary",
            ylabel="Mean Spearman correlation",
        ),
        plot_cross_rank_metric(
            _plot_df,
            "mean_pearson",
            title="Cross-rank RSA Pearson summary",
            ylabel="Mean Pearson correlation",
        ),
    ]
    if _notes:
        _items.insert(1, mo.callout(mo.md("\n\n".join(dict.fromkeys(_notes))), kind="warn"))
    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(mo, rsa_same_dataset, rsa_tables):
    if rsa_same_dataset:
        rsa_direction_input = None
        _display = mo.md("")
    else:
        _direction_options = {label: label for label in rsa_tables}
        rsa_direction_input = mo.ui.dropdown(
            options=_direction_options,
            value=list(_direction_options.keys())[0],
            label="RSA direction",
        )
        _display = rsa_direction_input
    _display
    return (rsa_direction_input,)


@app.cell(hide_code=True)
def _(mo, rsa_direction_input, rsa_same_dataset, rsa_tables):
    if rsa_same_dataset:
        _table_key = "within"
    else:
        _table_key = rsa_direction_input.value
    _fold_options = {
        f"fold {int(row.fold)}": int(row.fold)
        for row in rsa_tables[_table_key].itertuples()
    }
    rsa_fold_input = mo.ui.dropdown(
        options=_fold_options,
        value=list(_fold_options.keys())[0],
        label="Displayed fold",
    )
    rsa_fold_input
    return (rsa_fold_input,)


@app.cell(hide_code=True)
def _(
    mo,
    plot_rsm_pair,
    rsa_direction_input,
    rsa_distance_metric,
    rsa_fold_input,
    rsa_plot_specs,
    rsa_rsms_for_pair,
    rsa_same_dataset,
    rsa_summary,
    rsa_tables,
):
    _fold = int(rsa_fold_input.value)
    _fig_items = []
    if rsa_same_dataset:
        _table = rsa_tables["within"]
        for _comparison, _spec in rsa_plot_specs.items():
            _left = _spec["left"]
            _right = _spec["right"]
            _rsm_a, _rsm_b = rsa_rsms_for_pair(
                _left["X"],
                _right["X"],
                _left["test_idxs"][_fold],
                _left["bases_by_fold"][_fold],
                _right["bases_by_fold"][_fold],
                rsa_distance_metric,
            )
            _fig_items.extend(
                [
                    mo.md(f"### {_comparison} RSMs"),
                    plot_rsm_pair(
                        _rsm_a,
                        _rsm_b,
                        f"{_spec['left_title']} {_fold}",
                        f"{_spec['right_title']} {_fold}",
                    ),
                ]
            )
    else:
        _direction = rsa_direction_input.value
        _table = rsa_tables[_direction]
        _spec = rsa_plot_specs[_direction]
        _target = _spec["target"]
        _source_mean = _spec["source_mean"]
        _rsm_a, _rsm_b = rsa_rsms_for_pair(
            _target["X"],
            _target["X"],
            _target["test_idxs"][_fold],
            _target["bases_by_fold"][_fold],
            _source_mean,
            rsa_distance_metric,
        )
        _fig_items.extend(
            [
                mo.md(f"### {_direction} RSMs"),
                plot_rsm_pair(
                    _rsm_a,
                    _rsm_b,
                    f"{_spec['left_title']} {_fold}",
                    _spec["right_title"],
                ),
            ]
        )

    _items = [
        mo.md(f"Distance metric: `{rsa_distance_metric}`"),
    ]
    _items.extend(_fig_items)
    _items.extend(
        [
            mo.md("### Per-fold RSA correlations"),
            mo.ui.table(_table, selection=None),
            mo.md("### Mean RSA correlations"),
            mo.ui.table(rsa_summary, selection=None),
        ]
    )
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Procrustes Alignment
    """)
    return


@app.cell(hide_code=True)
def _(
    mo,
    np,
    pd,
    score_procrustes_alignment,
    stacked_svd_subspace_mean,
    summarize_procrustes_fold_results,
    validate_matching_fold_indices,
):
    def compute_procrustes_outputs(
        example_a,
        example_b,
        pca_example_a=None,
        pca_example_b=None,
    ):
        def _score_pair(X_a, X_b, train_idx, test_idx, basis_a, basis_b):
            _train_idx = np.asarray(train_idx, dtype=int)
            _test_idx = np.asarray(test_idx, dtype=int)
            _result = score_procrustes_alignment(
                np.asarray(X_a)[_train_idx] @ basis_a,
                np.asarray(X_b)[_train_idx] @ basis_b,
                np.asarray(X_a)[_test_idx] @ basis_a,
                np.asarray(X_b)[_test_idx] @ basis_b,
            )
            _row = {
                key: _result[key]
                for key in [
                    "scale",
                    "test_rmse",
                    "test_normalized_mse",
                    "test_r2",
                    "test_mean_row_cosine",
                ]
            }
            return _row, _result

        _same_dataset = example_a["dataset_key"] == example_b["dataset_key"]
        _tables = {}
        _plot_specs = {}
        _has_pca_baseline = pca_example_a is not None and pca_example_b is not None

        if example_a["rank"] != example_b["rank"]:
            mo.stop(
                True,
                mo.callout(
                    mo.md(
                        "Procrustes alignment requires equal projected ranks. "
                        f"Got rank {example_a['rank']} and {example_b['rank']}."
                    ),
                    kind="warn",
                ),
            )

        if _same_dataset:
            try:
                validate_matching_fold_indices(
                    example_a["test_idxs"],
                    example_b["test_idxs"],
                )
            except ValueError as exc:
                mo.stop(True, mo.callout(mo.md(str(exc)), kind="warn"))
            if _has_pca_baseline:
                for _label, _left, _right in [
                    ("1 RRR vs 1 PCA", example_a, pca_example_a),
                    ("2 RRR vs 2 PCA", example_b, pca_example_b),
                    ("1 RRR vs 2 PCA", example_a, pca_example_b),
                    ("1 PCA vs 2 RRR", pca_example_a, example_b),
                ]:
                    try:
                        validate_matching_fold_indices(
                            _left["test_idxs"],
                            _right["test_idxs"],
                        )
                    except ValueError as exc:
                        mo.stop(True, mo.callout(mo.md(f"{_label}: {exc}"), kind="warn"))

            _rows = []
            _within_specs = [
                ("RRR-vs-RRR", "1 RRR vs 2 RRR", example_a, example_b)
            ]
            if _has_pca_baseline:
                _within_specs.extend(
                    [
                        ("RRR-vs-PCA", "1 RRR vs 2 PCA", example_a, pca_example_b),
                        ("PCA-vs-RRR", "1 PCA vs 2 RRR", pca_example_a, example_b),
                        ("PCA-vs-PCA", "1 PCA vs 2 PCA", pca_example_a, pca_example_b),
                    ]
                )
            for _basis_type, _comparison, _example_a, _example_b in _within_specs:
                for _fold, _test_idx in enumerate(_example_a["test_idxs"]):
                    _metrics, _result = _score_pair(
                        _example_a["X"],
                        _example_b["X"],
                        _example_a["train_idxs"][_fold],
                        _test_idx,
                        _example_a["bases_by_fold"][_fold],
                        _example_b["bases_by_fold"][_fold],
                    )
                    _rows.append(
                        {
                            "basis": _basis_type,
                            "comparison": _comparison,
                            "fold": _fold,
                            "n_train": int(len(_example_a["train_idxs"][_fold])),
                            "n_test": int(len(_test_idx)),
                            **_metrics,
                        }
                    )
                    _plot_specs[(_comparison, _fold)] = {
                        "aligned_test": _result["aligned_test"],
                        "target_centered_test": _result["target_centered_test"],
                        "title": f"{_comparison} fold {_fold}",
                    }
            _tables["within"] = pd.DataFrame(_rows)
            _summary = pd.DataFrame(
                [
                    {
                        "basis": _basis_type,
                        "comparison": _comparison,
                        **summarize_procrustes_fold_results(
                            [
                                row
                                for row in _rows
                                if row["basis"] == _basis_type
                                and row["comparison"] == _comparison
                            ]
                        ),
                    }
                    for _basis_type, _comparison in sorted(
                        {(row["basis"], row["comparison"]) for row in _rows}
                    )
                ]
            )
        else:
            _basis_a_mean, _ = stacked_svd_subspace_mean(
                [
                    example_a["bases_by_fold"][fold]
                    for fold in sorted(example_a["bases_by_fold"])
                ],
                example_a["rank"],
            )
            _basis_b_mean, _ = stacked_svd_subspace_mean(
                [
                    example_b["bases_by_fold"][fold]
                    for fold in sorted(example_b["bases_by_fold"])
                ],
                example_b["rank"],
            )
            _direction_specs = [
                (
                    "RRR-vs-RRR",
                    "1 RRR fold vs 2 RRR mean",
                    example_a,
                    _basis_b_mean,
                ),
                (
                    "RRR-vs-RRR",
                    "2 RRR fold vs 1 RRR mean",
                    example_b,
                    _basis_a_mean,
                ),
            ]
            if _has_pca_baseline:
                _pca_basis_a_mean, _ = stacked_svd_subspace_mean(
                    [
                        pca_example_a["bases_by_fold"][fold]
                        for fold in sorted(pca_example_a["bases_by_fold"])
                    ],
                    pca_example_a["rank"],
                )
                _pca_basis_b_mean, _ = stacked_svd_subspace_mean(
                    [
                        pca_example_b["bases_by_fold"][fold]
                        for fold in sorted(pca_example_b["bases_by_fold"])
                    ],
                    pca_example_b["rank"],
                )
                _direction_specs.extend(
                    [
                        (
                            "RRR-vs-PCA",
                            "1 RRR fold vs 2 PCA mean",
                            example_a,
                            _pca_basis_b_mean,
                        ),
                        (
                            "PCA-vs-RRR",
                            "1 PCA fold vs 2 RRR mean",
                            pca_example_a,
                            _basis_b_mean,
                        ),
                        (
                            "PCA-vs-PCA",
                            "1 PCA fold vs 2 PCA mean",
                            pca_example_a,
                            _pca_basis_b_mean,
                        ),
                        (
                            "RRR-vs-PCA",
                            "2 RRR fold vs 1 PCA mean",
                            example_b,
                            _pca_basis_a_mean,
                        ),
                        (
                            "PCA-vs-RRR",
                            "2 PCA fold vs 1 RRR mean",
                            pca_example_b,
                            _basis_a_mean,
                        ),
                        (
                            "PCA-vs-PCA",
                            "2 PCA fold vs 1 PCA mean",
                            pca_example_b,
                            _pca_basis_a_mean,
                        ),
                    ]
                )

            _summary_rows = []
            for _basis_type, _label, _target, _source_mean in _direction_specs:
                mo.stop(
                    _target["X"].shape[1] != _source_mean.shape[0],
                    mo.callout(
                        mo.md(
                            f"Cross-dataset `{_label}` needs the target embeddings to "
                            "share the source subspace ambient dimension. Got "
                            f"{_target['X'].shape[1]} vs {_source_mean.shape[0]}."
                        ),
                        kind="warn",
                    ),
                )
                _rows = []
                for _fold, _test_idx in enumerate(_target["test_idxs"]):
                    _metrics, _result = _score_pair(
                        _target["X"],
                        _target["X"],
                        _target["train_idxs"][_fold],
                        _test_idx,
                        _target["bases_by_fold"][_fold],
                        _source_mean,
                    )
                    _rows.append(
                        {
                            "basis": _basis_type,
                            "comparison": _label,
                            "fold": _fold,
                            "n_train": int(len(_target["train_idxs"][_fold])),
                            "n_test": int(len(_test_idx)),
                            **_metrics,
                        }
                    )
                    _plot_specs[(_label, _fold)] = {
                        "aligned_test": _result["aligned_test"],
                        "target_centered_test": _result["target_centered_test"],
                        "title": f"{_label} fold {_fold}",
                    }
                _tables[_label] = pd.DataFrame(_rows)
                _summary_rows.append(
                    {
                        "basis": _basis_type,
                        "comparison": _label,
                        **summarize_procrustes_fold_results(_rows),
                    }
                )
            _summary = pd.DataFrame(_summary_rows)

        return {
            "plot_specs": _plot_specs,
            "same_dataset": _same_dataset,
            "summary": _summary,
            "tables": _tables,
        }

    return (compute_procrustes_outputs,)


@app.cell(hide_code=True)
def _(
    compute_procrustes_outputs,
    rsa_example_a,
    rsa_example_b,
    rsa_pca_example_a,
    rsa_pca_example_b,
):
    _outputs = compute_procrustes_outputs(
        rsa_example_a,
        rsa_example_b,
        rsa_pca_example_a,
        rsa_pca_example_b,
    )
    procrustes_plot_specs = _outputs["plot_specs"]
    procrustes_same_dataset = _outputs["same_dataset"]
    procrustes_summary = _outputs["summary"]
    procrustes_tables = _outputs["tables"]
    return (
        procrustes_plot_specs,
        procrustes_same_dataset,
        procrustes_summary,
        procrustes_tables,
    )


@app.cell(hide_code=True)
def _(
    compute_procrustes_outputs,
    filter_available_ranks,
    load_rsa_example,
    mo,
    parse_rank_list,
    pca_sibling_group,
    pd,
    plot_cross_rank_metric,
    rsa_config_a,
    rsa_config_b,
    rsa_cross_rank_input,
    rsa_group_a,
    rsa_group_b,
):
    try:
        _requested_ranks = parse_rank_list(rsa_cross_rank_input.value)
    except ValueError as exc:
        mo.stop(True, mo.callout(mo.md(str(exc)), kind="warn"))

    _primary_max_rank = min(
        min(source.max_rank for source in rsa_group_a),
        min(source.max_rank for source in rsa_group_b),
    )
    _valid_ranks, _skipped_ranks = filter_available_ranks(
        _requested_ranks,
        _primary_max_rank,
    )
    mo.stop(
        not _valid_ranks,
        mo.callout(mo.md("No requested Procrustes ranks are available."), kind="warn"),
    )

    _pca_group_a = pca_sibling_group(rsa_group_a)
    _pca_group_b = pca_sibling_group(rsa_group_b)
    _notes = []
    if _skipped_ranks:
        _notes.append(
            "Skipped unavailable primary Procrustes ranks: "
            + ", ".join(str(rank) for rank in _skipped_ranks)
        )
    if _pca_group_a is None or _pca_group_b is None:
        _notes.append("PCA Procrustes lines omitted because a PCA sibling group was not found.")

    _summary_rows = []
    for _rank in _valid_ranks:
        _example_a = load_rsa_example(rsa_config_a, rsa_group_a, _rank)
        _example_b = load_rsa_example(rsa_config_b, rsa_group_b, _rank)
        _pca_example_a = None
        _pca_example_b = None
        if _pca_group_a is not None and _pca_group_b is not None:
            _pca_max_rank = min(
                min(source.max_rank for source in _pca_group_a),
                min(source.max_rank for source in _pca_group_b),
            )
            if _rank <= _pca_max_rank:
                try:
                    _pca_example_a = load_rsa_example(rsa_config_a, _pca_group_a, _rank)
                    _pca_example_b = load_rsa_example(rsa_config_b, _pca_group_b, _rank)
                except Exception as exc:
                    _notes.append(f"PCA Procrustes rank {_rank} omitted: {exc}")
            else:
                _notes.append(
                    f"PCA Procrustes rank {_rank} omitted; max PCA rank is {_pca_max_rank}."
                )

        _outputs = compute_procrustes_outputs(
            _example_a,
            _example_b,
            _pca_example_a,
            _pca_example_b,
        )
        _summary = _outputs["summary"].copy()
        if _outputs["same_dataset"]:
            for _row in _summary.to_dict("records"):
                _summary_rows.append({"rank": _rank, **_row})
        else:
            _basis_summary = (
                _summary.groupby("basis", as_index=False)[
                    ["mean_test_r2", "mean_test_mean_row_cosine"]
                ]
                .mean()
                .sort_values("basis")
            )
            for _row in _basis_summary.to_dict("records"):
                _summary_rows.append(
                    {
                        "rank": _rank,
                        "basis": _row["basis"],
                        "comparison": "cross-dataset directional mean",
                        "mean_test_r2": _row["mean_test_r2"],
                        "mean_test_mean_row_cosine": _row[
                            "mean_test_mean_row_cosine"
                        ],
                    }
                )

    _summary_df = pd.DataFrame(_summary_rows)
    mo.stop(
        _summary_df.empty,
        mo.callout(
            mo.md("No cross-rank Procrustes summaries could be computed."),
            kind="warn",
        ),
    )
    _plot_df = _summary_df.copy()
    _plot_df["comparison"] = _plot_df["basis"]
    _items = [
        mo.md("### Cross-rank Procrustes summary"),
        mo.ui.table(_summary_df, selection=None),
        plot_cross_rank_metric(
            _plot_df,
            "mean_test_r2",
            title="Cross-rank held-out Procrustes R2",
            ylabel="Mean held-out R2",
        ),
        plot_cross_rank_metric(
            _plot_df,
            "mean_test_mean_row_cosine",
            title="Cross-rank held-out Procrustes row cosine",
            ylabel="Mean held-out row cosine",
        ),
    ]
    if _notes:
        _items.insert(1, mo.callout(mo.md("\n\n".join(dict.fromkeys(_notes))), kind="warn"))
    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(mo, procrustes_same_dataset, procrustes_tables):
    if procrustes_same_dataset:
        procrustes_direction_input = None
        _display = mo.md("")
    else:
        _direction_options = {label: label for label in procrustes_tables}
        procrustes_direction_input = mo.ui.dropdown(
            options=_direction_options,
            value=list(_direction_options.keys())[0],
            label="Procrustes direction",
        )
        _display = procrustes_direction_input
    _display
    return (procrustes_direction_input,)


@app.cell(hide_code=True)
def _(
    mo,
    procrustes_direction_input,
    procrustes_same_dataset,
    procrustes_tables,
):
    if procrustes_same_dataset:
        _table_key = "within"
    else:
        _table_key = procrustes_direction_input.value
    _fold_options = {
        f"fold {int(row.fold)}": int(row.fold)
        for row in procrustes_tables[_table_key].itertuples()
    }
    procrustes_fold_input = mo.ui.dropdown(
        options=_fold_options,
        value=list(_fold_options.keys())[0],
        label="Displayed Procrustes fold",
    )
    procrustes_fold_input
    return (procrustes_fold_input,)


@app.cell(hide_code=True)
def _(np, plt):
    def plot_procrustes_scatter(aligned_test, target_centered_test, title):
        _aligned = np.asarray(aligned_test, dtype=float).ravel()
        _target = np.asarray(target_centered_test, dtype=float).ravel()
        _finite = np.isfinite(_aligned) & np.isfinite(_target)
        _aligned = _aligned[_finite]
        _target = _target[_finite]
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.scatter(_target, _aligned, s=12, alpha=0.35)
        if _target.size:
            _low = float(np.nanmin([np.min(_target), np.min(_aligned)]))
            _high = float(np.nanmax([np.max(_target), np.max(_aligned)]))
            ax.plot([_low, _high], [_low, _high], color="0.2", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Target centered coordinate")
        ax.set_ylabel("Aligned source coordinate")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig

    return (plot_procrustes_scatter,)


@app.cell(hide_code=True)
def _(
    mo,
    plot_procrustes_scatter,
    procrustes_direction_input,
    procrustes_fold_input,
    procrustes_plot_specs,
    procrustes_same_dataset,
    procrustes_summary,
    procrustes_tables,
):
    _fold = int(procrustes_fold_input.value)
    _fig_items = []
    if procrustes_same_dataset:
        _table = procrustes_tables["within"]
        _comparisons = list(dict.fromkeys(_table["comparison"].tolist()))
        for _comparison in _comparisons:
            _spec = procrustes_plot_specs.get((_comparison, _fold))
            if _spec is None:
                continue
            _fig_items.extend(
                [
                    mo.md(f"### {_comparison} aligned coordinates"),
                    plot_procrustes_scatter(
                        _spec["aligned_test"],
                        _spec["target_centered_test"],
                        _spec["title"],
                    ),
                ]
            )
    else:
        _direction = procrustes_direction_input.value
        _table = procrustes_tables[_direction]
        _spec = procrustes_plot_specs[(_direction, _fold)]
        _fig_items.extend(
            [
                mo.md(f"### {_direction} aligned coordinates"),
                plot_procrustes_scatter(
                    _spec["aligned_test"],
                    _spec["target_centered_test"],
                    _spec["title"],
                ),
            ]
        )

    _items = [
        mo.md("Fit: train rows from the original CV fold. Evaluation: held-out fold rows."),
    ]
    _items.extend(_fig_items)
    _items.extend(
        [
            mo.md("### Per-fold Procrustes scores"),
            mo.ui.table(_table, selection=None),
            mo.md("### Mean Procrustes scores"),
            mo.ui.table(procrustes_summary, selection=None),
        ]
    )
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md("""
    ## CCA Alignment
    """)
    return


@app.cell(hide_code=True)
def _(
    mo,
    np,
    pd,
    score_cca_alignment,
    stacked_svd_subspace_mean,
    summarize_cca_fold_results,
    validate_matching_fold_indices,
):
    def compute_cca_outputs(
        example_a,
        example_b,
        pca_example_a=None,
        pca_example_b=None,
        shared_corr_threshold=0.2,
    ):
        def _score_pair(X_a, X_b, train_idx, test_idx, basis_a, basis_b):
            _train_idx = np.asarray(train_idx, dtype=int)
            _test_idx = np.asarray(test_idx, dtype=int)
            _result = score_cca_alignment(
                np.asarray(X_a)[_train_idx] @ basis_a,
                np.asarray(X_b)[_train_idx] @ basis_b,
                np.asarray(X_a)[_test_idx] @ basis_a,
                np.asarray(X_b)[_test_idx] @ basis_b,
                shared_corr_threshold=float(shared_corr_threshold),
            )
            _metric_keys = [
                "n_components",
                "mean_top10_test_corr",
                "mean_all_test_corr",
                "first_test_corr",
                "median_test_corr",
                "mean_top10_train_corr",
                "test_shared_dim_pr",
                "train_shared_dim_pr",
                "n_test_corr_ge_threshold",
                "n_train_corr_ge_threshold",
                "shared_corr_threshold",
            ]
            return {key: _result[key] for key in _metric_keys}, _result

        _same_dataset = example_a["dataset_key"] == example_b["dataset_key"]
        _tables = {}
        _component_specs = {}
        _has_pca_baseline = pca_example_a is not None and pca_example_b is not None

        if _same_dataset:
            try:
                validate_matching_fold_indices(
                    example_a["test_idxs"],
                    example_b["test_idxs"],
                )
            except ValueError as exc:
                mo.stop(True, mo.callout(mo.md(str(exc)), kind="warn"))
            if _has_pca_baseline:
                for _label, _left, _right in [
                    ("1 RRR vs 1 PCA", example_a, pca_example_a),
                    ("2 RRR vs 2 PCA", example_b, pca_example_b),
                    ("1 RRR vs 2 PCA", example_a, pca_example_b),
                    ("1 PCA vs 2 RRR", pca_example_a, example_b),
                ]:
                    try:
                        validate_matching_fold_indices(
                            _left["test_idxs"],
                            _right["test_idxs"],
                        )
                    except ValueError as exc:
                        mo.stop(True, mo.callout(mo.md(f"{_label}: {exc}"), kind="warn"))

            _rows = []
            _within_specs = [
                ("RRR-vs-RRR", "1 RRR vs 2 RRR", example_a, example_b)
            ]
            if _has_pca_baseline:
                _within_specs.extend(
                    [
                        ("RRR-vs-PCA", "1 RRR vs 2 PCA", example_a, pca_example_b),
                        ("PCA-vs-RRR", "1 PCA vs 2 RRR", pca_example_a, example_b),
                        ("PCA-vs-PCA", "1 PCA vs 2 PCA", pca_example_a, pca_example_b),
                    ]
                )
            for _basis_type, _comparison, _example_a, _example_b in _within_specs:
                for _fold, _test_idx in enumerate(_example_a["test_idxs"]):
                    _metrics, _result = _score_pair(
                        _example_a["X"],
                        _example_b["X"],
                        _example_a["train_idxs"][_fold],
                        _test_idx,
                        _example_a["bases_by_fold"][_fold],
                        _example_b["bases_by_fold"][_fold],
                    )
                    _rows.append(
                        {
                            "basis": _basis_type,
                            "comparison": _comparison,
                            "fold": _fold,
                            "n_train": int(len(_example_a["train_idxs"][_fold])),
                            "n_test": int(len(_test_idx)),
                            **_metrics,
                        }
                    )
                    _component_specs[(_comparison, _fold)] = pd.DataFrame(
                        {
                            "component": np.arange(1, _result["n_components"] + 1),
                            "train_corr": _result["train_corrs"],
                            "test_corr": _result["test_corrs"],
                            "test_weight": np.maximum(_result["test_corrs"], 0.0) ** 2,
                        }
                    )
            _tables["within"] = pd.DataFrame(_rows)
            _summary = pd.DataFrame(
                [
                    {
                        "basis": _basis_type,
                        "comparison": _comparison,
                        **summarize_cca_fold_results(
                            [
                                row
                                for row in _rows
                                if row["basis"] == _basis_type
                                and row["comparison"] == _comparison
                            ]
                        ),
                    }
                    for _basis_type, _comparison in sorted(
                        {(row["basis"], row["comparison"]) for row in _rows}
                    )
                ]
            )
        else:
            _basis_a_mean, _ = stacked_svd_subspace_mean(
                [
                    example_a["bases_by_fold"][fold]
                    for fold in sorted(example_a["bases_by_fold"])
                ],
                example_a["rank"],
            )
            _basis_b_mean, _ = stacked_svd_subspace_mean(
                [
                    example_b["bases_by_fold"][fold]
                    for fold in sorted(example_b["bases_by_fold"])
                ],
                example_b["rank"],
            )
            _direction_specs = [
                (
                    "RRR-vs-RRR",
                    "1 RRR fold vs 2 RRR mean",
                    example_a,
                    _basis_b_mean,
                ),
                (
                    "RRR-vs-RRR",
                    "2 RRR fold vs 1 RRR mean",
                    example_b,
                    _basis_a_mean,
                ),
            ]
            if _has_pca_baseline:
                _pca_basis_a_mean, _ = stacked_svd_subspace_mean(
                    [
                        pca_example_a["bases_by_fold"][fold]
                        for fold in sorted(pca_example_a["bases_by_fold"])
                    ],
                    pca_example_a["rank"],
                )
                _pca_basis_b_mean, _ = stacked_svd_subspace_mean(
                    [
                        pca_example_b["bases_by_fold"][fold]
                        for fold in sorted(pca_example_b["bases_by_fold"])
                    ],
                    pca_example_b["rank"],
                )
                _direction_specs.extend(
                    [
                        (
                            "RRR-vs-PCA",
                            "1 RRR fold vs 2 PCA mean",
                            example_a,
                            _pca_basis_b_mean,
                        ),
                        (
                            "PCA-vs-RRR",
                            "1 PCA fold vs 2 RRR mean",
                            pca_example_a,
                            _basis_b_mean,
                        ),
                        (
                            "PCA-vs-PCA",
                            "1 PCA fold vs 2 PCA mean",
                            pca_example_a,
                            _pca_basis_b_mean,
                        ),
                        (
                            "RRR-vs-PCA",
                            "2 RRR fold vs 1 PCA mean",
                            example_b,
                            _pca_basis_a_mean,
                        ),
                        (
                            "PCA-vs-RRR",
                            "2 PCA fold vs 1 RRR mean",
                            pca_example_b,
                            _basis_a_mean,
                        ),
                        (
                            "PCA-vs-PCA",
                            "2 PCA fold vs 1 PCA mean",
                            pca_example_b,
                            _pca_basis_a_mean,
                        ),
                    ]
                )

            _summary_rows = []
            for _basis_type, _label, _target, _source_mean in _direction_specs:
                mo.stop(
                    _target["X"].shape[1] != _source_mean.shape[0],
                    mo.callout(
                        mo.md(
                            f"Cross-dataset `{_label}` needs the target embeddings to "
                            "share the source subspace ambient dimension. Got "
                            f"{_target['X'].shape[1]} vs {_source_mean.shape[0]}."
                        ),
                        kind="warn",
                    ),
                )
                _rows = []
                for _fold, _test_idx in enumerate(_target["test_idxs"]):
                    _metrics, _result = _score_pair(
                        _target["X"],
                        _target["X"],
                        _target["train_idxs"][_fold],
                        _test_idx,
                        _target["bases_by_fold"][_fold],
                        _source_mean,
                    )
                    _rows.append(
                        {
                            "basis": _basis_type,
                            "comparison": _label,
                            "fold": _fold,
                            "n_train": int(len(_target["train_idxs"][_fold])),
                            "n_test": int(len(_test_idx)),
                            **_metrics,
                        }
                    )
                    _component_specs[(_label, _fold)] = pd.DataFrame(
                        {
                            "component": np.arange(1, _result["n_components"] + 1),
                            "train_corr": _result["train_corrs"],
                            "test_corr": _result["test_corrs"],
                            "test_weight": np.maximum(_result["test_corrs"], 0.0) ** 2,
                        }
                    )
                _tables[_label] = pd.DataFrame(_rows)
                _summary_rows.append(
                    {
                        "basis": _basis_type,
                        "comparison": _label,
                        **summarize_cca_fold_results(_rows),
                    }
                )
            _summary = pd.DataFrame(_summary_rows)

        return {
            "component_specs": _component_specs,
            "same_dataset": _same_dataset,
            "summary": _summary,
            "tables": _tables,
        }

    return (compute_cca_outputs,)


@app.cell(hide_code=True)
def _(
    cca_shared_threshold_input,
    compute_cca_outputs,
    rsa_example_a,
    rsa_example_b,
    rsa_pca_example_a,
    rsa_pca_example_b,
):
    _outputs = compute_cca_outputs(
        rsa_example_a,
        rsa_example_b,
        rsa_pca_example_a,
        rsa_pca_example_b,
        float(cca_shared_threshold_input.value),
    )
    cca_component_specs = _outputs["component_specs"]
    cca_same_dataset = _outputs["same_dataset"]
    cca_summary = _outputs["summary"]
    cca_tables = _outputs["tables"]
    return cca_component_specs, cca_same_dataset, cca_summary, cca_tables


@app.cell(hide_code=True)
def _(
    cca_cross_rank_input,
    cca_shared_threshold_input,
    compute_cca_outputs,
    filter_available_ranks,
    load_rsa_example,
    mo,
    parse_rank_list,
    pca_sibling_group,
    pd,
    plot_cross_rank_metric,
    rsa_config_a,
    rsa_config_b,
    rsa_group_a,
    rsa_group_b,
):
    try:
        _requested_ranks = parse_rank_list(cca_cross_rank_input.value)
    except ValueError as exc:
        mo.stop(True, mo.callout(mo.md(str(exc)), kind="warn"))

    _primary_max_rank = min(
        min(source.max_rank for source in rsa_group_a),
        min(source.max_rank for source in rsa_group_b),
    )
    _valid_ranks, _skipped_ranks = filter_available_ranks(
        _requested_ranks,
        _primary_max_rank,
    )
    mo.stop(
        not _valid_ranks,
        mo.callout(mo.md("No requested CCA ranks are available."), kind="warn"),
    )

    _pca_group_a = pca_sibling_group(rsa_group_a)
    _pca_group_b = pca_sibling_group(rsa_group_b)
    _notes = []
    if _skipped_ranks:
        _notes.append(
            "Skipped unavailable primary CCA ranks: "
            + ", ".join(str(rank) for rank in _skipped_ranks)
        )
    if _pca_group_a is None or _pca_group_b is None:
        _notes.append("PCA CCA lines omitted because a PCA sibling group was not found.")

    _summary_rows = []
    for _rank in _valid_ranks:
        _example_a = load_rsa_example(rsa_config_a, rsa_group_a, _rank)
        _example_b = load_rsa_example(rsa_config_b, rsa_group_b, _rank)
        _pca_example_a = None
        _pca_example_b = None
        if _pca_group_a is not None and _pca_group_b is not None:
            _pca_max_rank = min(
                min(source.max_rank for source in _pca_group_a),
                min(source.max_rank for source in _pca_group_b),
            )
            if _rank <= _pca_max_rank:
                try:
                    _pca_example_a = load_rsa_example(rsa_config_a, _pca_group_a, _rank)
                    _pca_example_b = load_rsa_example(rsa_config_b, _pca_group_b, _rank)
                except Exception as exc:
                    _notes.append(f"PCA CCA rank {_rank} omitted: {exc}")
            else:
                _notes.append(
                    f"PCA CCA rank {_rank} omitted; max PCA rank is {_pca_max_rank}."
                )

        _outputs = compute_cca_outputs(
            _example_a,
            _example_b,
            _pca_example_a,
            _pca_example_b,
            float(cca_shared_threshold_input.value),
        )
        _summary = _outputs["summary"].copy()
        if _outputs["same_dataset"]:
            for _row in _summary.to_dict("records"):
                _summary_rows.append({"rank": _rank, **_row})
        else:
            _basis_summary = (
                _summary.groupby("basis", as_index=False)[
                    [
                        "mean_top10_test_corr",
                        "mean_all_test_corr",
                        "mean_test_shared_dim_pr",
                        "mean_n_test_corr_ge_threshold",
                    ]
                ]
                .mean()
                .sort_values("basis")
            )
            for _row in _basis_summary.to_dict("records"):
                _summary_rows.append(
                    {
                        "rank": _rank,
                        "basis": _row["basis"],
                        "comparison": "cross-dataset directional mean",
                        "mean_top10_test_corr": _row["mean_top10_test_corr"],
                        "mean_all_test_corr": _row["mean_all_test_corr"],
                        "mean_test_shared_dim_pr": _row["mean_test_shared_dim_pr"],
                        "mean_n_test_corr_ge_threshold": _row[
                            "mean_n_test_corr_ge_threshold"
                        ],
                    }
                )

    _summary_df = pd.DataFrame(_summary_rows)
    mo.stop(
        _summary_df.empty,
        mo.callout(
            mo.md("No cross-rank CCA summaries could be computed."),
            kind="warn",
        ),
    )
    _plot_df = _summary_df.copy()
    _plot_df["comparison"] = _plot_df["basis"]
    _items = [
        mo.md("### Cross-rank CCA summary"),
        mo.ui.table(_summary_df, selection=None),
        plot_cross_rank_metric(
            _plot_df,
            "mean_top10_test_corr",
            title="Cross-rank held-out CCA top-10 correlation",
            ylabel="Mean held-out canonical correlation",
        ),
        plot_cross_rank_metric(
            _plot_df,
            "mean_all_test_corr",
            title="Cross-rank held-out CCA all-component correlation",
            ylabel="Mean held-out canonical correlation",
        ),
        plot_cross_rank_metric(
            _plot_df,
            "mean_test_shared_dim_pr",
            title="Cross-rank held-out CCA effective shared dimensionality",
            ylabel="Mean held-out PR dimension",
        ),
        plot_cross_rank_metric(
            _plot_df,
            "mean_n_test_corr_ge_threshold",
            title="Cross-rank held-out CCA shared component count",
            ylabel="Mean held-out component count",
        ),
    ]
    if _notes:
        _items.insert(1, mo.callout(mo.md("\n\n".join(dict.fromkeys(_notes))), kind="warn"))
    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(cca_same_dataset, cca_tables, mo):
    if cca_same_dataset:
        cca_direction_input = None
        _display = mo.md("")
    else:
        _direction_options = {label: label for label in cca_tables}
        cca_direction_input = mo.ui.dropdown(
            options=_direction_options,
            value=list(_direction_options.keys())[0],
            label="CCA direction",
        )
        _display = cca_direction_input
    _display
    return (cca_direction_input,)


@app.cell(hide_code=True)
def _(cca_direction_input, cca_same_dataset, cca_tables, mo):
    if cca_same_dataset:
        _table_key = "within"
    else:
        _table_key = cca_direction_input.value
    _fold_options = {
        f"fold {int(row.fold)}": int(row.fold)
        for row in cca_tables[_table_key].itertuples()
    }
    cca_fold_input = mo.ui.dropdown(
        options=_fold_options,
        value=list(_fold_options.keys())[0],
        label="Displayed CCA fold",
    )
    cca_fold_input
    return (cca_fold_input,)


@app.cell(hide_code=True)
def _(np, plt):
    def plot_cca_spectrum(component_tables, title):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for _label, _table in component_tables.items():
            ax.plot(
                np.asarray(_table["component"]),
                np.asarray(_table["test_corr"]),
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                label=_label,
            )
        ax.axhline(0.0, color="0.2", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Canonical component")
        ax.set_ylabel("Held-out canonical correlation")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if len(component_tables) > 1:
            ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    return (plot_cca_spectrum,)


@app.cell(hide_code=True)
def _(
    cca_component_specs,
    cca_direction_input,
    cca_fold_input,
    cca_same_dataset,
    cca_summary,
    cca_tables,
    mo,
    pd,
    plot_cca_spectrum,
):
    _fold = int(cca_fold_input.value)
    if cca_same_dataset:
        _table = cca_tables["within"]
        _comparisons = list(dict.fromkeys(_table["comparison"].tolist()))
        _component_tables = {
            _comparison: cca_component_specs[(_comparison, _fold)]
            for _comparison in _comparisons
            if (_comparison, _fold) in cca_component_specs
        }
        _component_table = pd.concat(
            [
                _component_table.assign(comparison=_comparison)
                for _comparison, _component_table in _component_tables.items()
            ],
            ignore_index=True,
        )
        _title = f"Held-out CCA spectrum, fold {_fold}"
    else:
        _direction = cca_direction_input.value
        _table = cca_tables[_direction]
        _component_tables = {
            _direction: cca_component_specs[(_direction, _fold)],
        }
        _component_table = _component_tables[_direction]
        _title = f"{_direction} held-out CCA spectrum, fold {_fold}"

    mo.vstack(
        [
            mo.md("Fit: train rows from the original CV fold. Evaluation: held-out fold rows."),
            mo.md("### Displayed-fold CCA spectrum"),
            plot_cca_spectrum(_component_tables, _title),
            mo.md("### Displayed-fold CCA components"),
            mo.ui.table(_component_table, selection=None),
            mo.md("### Per-fold CCA scores"),
            mo.ui.table(_table, selection=None),
            mo.md("### Mean CCA scores"),
            mo.ui.table(cca_summary, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(gcca_component_table, mo):
    _dimensions = sorted(gcca_component_table["shared_dimension"].unique())
    _folds = sorted(gcca_component_table["outer_fold"].unique())
    gcca_explorer_dimension = mo.ui.dropdown(
        options={str(value): int(value) for value in _dimensions},
        value=int(_dimensions[-1]), label="Explorer shared dimension",
    )
    gcca_explorer_fold = mo.ui.dropdown(
        options={str(value): int(value) for value in _folds},
        value=int(_folds[0]), label="Explorer outer fold",
    )
    mo.hstack([gcca_explorer_dimension, gcca_explorer_fold])
    return gcca_explorer_dimension, gcca_explorer_fold


@app.cell(hide_code=True)
def _(
    gcca_component_table,
    gcca_explorer_dimension,
    gcca_explorer_fold,
    mo,
    np,
    plt,
):
    _frame = gcca_component_table[
        (gcca_component_table["shared_dimension"] == gcca_explorer_dimension.value)
        & (gcca_component_table["outer_fold"] == gcca_explorer_fold.value)
    ]
    _metrics = [
        ("view_support", "Absolute support"),
        ("contribution_share", "Contribution share"),
        ("selected_consensus_corr", "Held-out consensus correlation"),
    ]
    _fig, _axes = plt.subplots(1, 4, figsize=(19, 4.5))
    for _ax, (_metric, _title) in zip(_axes[:3], _metrics):
        _pivot = _frame.pivot(index="component", columns="view", values=_metric)
        _image = _ax.imshow(_pivot.to_numpy(), aspect="auto", interpolation="nearest")
        _ax.set_xticks(np.arange(len(_pivot.columns)), _pivot.columns, rotation=45, ha="right")
        _ax.set_yticks(np.arange(len(_pivot.index)), _pivot.index)
        _ax.set_title(_title)
        _fig.colorbar(_image, ax=_ax, fraction=0.046)
    _counts = _frame.groupby("component")["support_view_count"].first()
    _axes[3].bar(_counts.index, _counts.values)
    _axes[3].set_title("Support-view count")
    _axes[3].set_xlabel("Component")
    _fig.tight_layout()
    mo.vstack([mo.md("### Fold-stable component explorer"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Regularized MAXVAR GCCA

    Learn one shared response from multiple model/layer/dataset subspaces. The
    selected anchor run supplies transcript rows and outer folds. Views from the
    anchor dataset use matching fold-specific `A.npy` bases; views from other
    datasets use their stacked-SVD fold mean.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    gcca_config_input = mo.ui.text(
        value="configs/rrr/compare_subspaces.yml",
        label="Comparison config",
        full_width=True,
    )
    gcca_config_input
    return (gcca_config_input,)


@app.cell(hide_code=True)
def _(
    SubspaceComparisonConfig,
    gcca_config_input,
    load_config,
    mo,
    resolve_subspace_run,
):
    try:
        _gcca_config = load_config(
            gcca_config_input.value,
            {},
            SubspaceComparisonConfig,
        )
        gcca_run_map = {
            _run.name: resolve_subspace_run(_run) for _run in _gcca_config.runs
        }
    except Exception as _exc:
        mo.stop(True, mo.callout(mo.md(f"GCCA config: {_exc}"), kind="warn"))
    mo.stop(
        len(gcca_run_map) < 2,
        mo.callout(mo.md("GCCA needs at least two configured runs."), kind="warn"),
    )
    return (gcca_run_map,)


@app.cell(hide_code=True)
def _(gcca_run_map, is_pca_source_kind, mo):
    _names = list(gcca_run_map)
    _primary = [
        _name
        for _name, _run in gcca_run_map.items()
        if not is_pca_source_kind(_run.kind)
    ]
    _defaults = (_primary or _names)[: min(6, len(_primary or _names))]
    gcca_anchor_input = mo.ui.dropdown(
        options={_name: _name for _name in _names},
        value=_names[0],
        label="Anchor run",
        full_width=True,
    )
    gcca_views_input = mo.ui.multiselect(
        options={_name: _name for _name in _names},
        value=_defaults,
        label="GCCA views",
        full_width=True,
    )
    mo.vstack([gcca_anchor_input, gcca_views_input])
    return gcca_anchor_input, gcca_views_input


@app.cell(hide_code=True)
def _(gcca_run_map, gcca_views_input, max_available_rank, mo):
    gcca_rank_inputs = mo.ui.dictionary(
        {
            _name: mo.ui.number(
                value=min(32, max_available_rank(gcca_run_map[_name].unit_dir)),
                start=1,
                stop=max_available_rank(gcca_run_map[_name].unit_dir),
                step=1,
                label=_name,
            )
            for _name in gcca_views_input.value
        },
        label="Per-view A ranks",
    )
    gcca_rank_inputs
    return (gcca_rank_inputs,)


@app.cell(hide_code=True)
def _(mo):
    gcca_dimensions_input = mo.ui.text(
        value="1,2,4,8,16,32",
        label="Shared dimensions",
        full_width=True,
    )
    gcca_alphas_input = mo.ui.text(
        value="0,1e-6,1e-4,1e-2,1e-1,1",
        label="Dimensionless ridge alphas",
        full_width=True,
    )
    gcca_support_mass_input = mo.ui.number(
        value=0.80, start=0.5, stop=1.0, step=0.05,
        label="Support cumulative mass",
    )
    gcca_match_threshold_input = mo.ui.number(
        value=0.80, start=0.0, stop=1.0, step=0.05,
        label="Axis match-correlation threshold",
    )
    gcca_eigengap_threshold_input = mo.ui.number(
        value=0.05, start=0.0, stop=1.0, step=0.01,
        label="Relative neighboring eigengap threshold",
    )
    gcca_run_button = mo.ui.run_button(label="Run MAXVAR GCCA")
    mo.vstack(
        [
            gcca_dimensions_input,
            gcca_alphas_input,
            gcca_support_mass_input,
            gcca_match_threshold_input,
            gcca_eigengap_threshold_input,
            gcca_run_button,
        ]
    )
    return (
        gcca_alphas_input,
        gcca_dimensions_input,
        gcca_eigengap_threshold_input,
        gcca_match_threshold_input,
        gcca_run_button,
        gcca_support_mass_input,
    )


@app.function
def parse_gcca_alpha_grid(text):
    _values = sorted(
        set(float(_part.strip()) for _part in str(text).split(",") if _part.strip())
    )
    if not _values or _values[0] < 0:
        raise ValueError("Enter one or more non-negative ridge alphas")
    return _values


@app.cell(hide_code=True)
def _(
    align_maxvar_folds,
    fit_regularized_maxvar_gcca,
    gcca_alphas_input,
    gcca_anchor_input,
    gcca_dimensions_input,
    gcca_eigengap_threshold_input,
    gcca_match_threshold_input,
    gcca_rank_inputs,
    gcca_run_button,
    gcca_run_map,
    gcca_support_mass_input,
    gcca_views_input,
    load_base_comparison_data,
    load_fold_bases,
    load_peer_embeddings_on_base_rows,
    maxvar_ambient_map,
    maxvar_block_overlap,
    maxvar_component_blocks,
    mo,
    np,
    parse_rank_list,
    pd,
    score_maxvar_partial_support,
    select_maxvar_alpha,
    stacked_svd_subspace_mean,
    validate_matching_fold_indices,
):
    mo.stop(not gcca_run_button.value, mo.md("Configure views, then run MAXVAR GCCA."))
    _view_names = list(gcca_views_input.value)
    mo.stop(
        len(_view_names) < 2,
        mo.callout(mo.md("Select at least two GCCA views."), kind="warn"),
    )
    try:
        _dimensions = parse_rank_list(gcca_dimensions_input.value)
        _alphas = parse_gcca_alpha_grid(gcca_alphas_input.value)
        _ranks = {name: int(gcca_rank_inputs.value[name]) for name in _view_names}
    except Exception as _exc:
        mo.stop(True, mo.callout(mo.md(f"GCCA controls: {_exc}"), kind="warn"))
    mo.stop(
        max(_dimensions) > sum(_ranks.values()),
        mo.callout(
            mo.md(
                f"Shared dimension {max(_dimensions)} exceeds the configured union-rank "
                f"upper bound ({sum(_ranks.values())})."
            ),
            kind="warn",
        ),
    )

    _anchor = gcca_run_map[gcca_anchor_input.value]
    try:
        _anchor_data = load_base_comparison_data(_anchor)
    except Exception as _exc:
        mo.stop(True, mo.callout(mo.md(f"Anchor data: {_exc}"), kind="warn"))

    _embeddings_by_view = {}
    _bases_by_view = {}
    _basis_modes = {}
    for _name in _view_names:
        _run = gcca_run_map[_name]
        _rank = _ranks[_name]
        try:
            _embeddings = (
                _anchor_data.X_base
                if _name == _anchor.name
                else load_peer_embeddings_on_base_rows(_anchor, _run, _anchor_data)
            )
            _fold_bases = load_fold_bases(_run.unit_dir, _rank)
            if _run.dataset_key == _anchor.dataset_key:
                _peer_data = load_base_comparison_data(_run)
                if not np.array_equal(_anchor_data.selection, _peer_data.selection):
                    raise ValueError("selected transcript rows differ from the anchor")
                validate_matching_fold_indices(
                    _anchor_data.test_idxs,
                    _peer_data.test_idxs,
                )
                _missing = sorted(
                    set(range(len(_anchor_data.test_idxs))) - set(_fold_bases)
                )
                if _missing:
                    raise ValueError(f"missing fold bases {_missing}")
                _bases_by_view[_name] = _fold_bases
                _basis_modes[_name] = "matching_anchor_fold"
            else:
                _mean_basis, _ = stacked_svd_subspace_mean(
                    [_fold_bases[_fold] for _fold in sorted(_fold_bases)],
                    _rank,
                )
                _bases_by_view[_name] = {
                    _fold: _mean_basis for _fold in range(len(_anchor_data.test_idxs))
                }
                _basis_modes[_name] = "cross_dataset_fold_mean"
            _embeddings_by_view[_name] = np.asarray(_embeddings)
        except Exception as _exc:
            mo.stop(
                True,
                mo.callout(mo.md(f"GCCA view `{_name}`: {_exc}"), kind="warn"),
            )

    _summary_rows, _view_rows, _component_rows, _inner_rows = [], [], [], []
    _artifacts = {}
    _all_rows = np.arange(_anchor_data.word_count, dtype=int)
    for _outer_fold, (_outer_train, _outer_test) in enumerate(
        zip(_anchor_data.train_idxs, _anchor_data.test_idxs)
    ):
        _projected = {
            _name: _embeddings_by_view[_name]
            @ _bases_by_view[_name][_outer_fold]
            for _name in _view_names
        }
        _inner_splits = []
        for _inner_fold, _block in enumerate(_anchor_data.test_idxs):
            if _inner_fold == _outer_fold:
                continue
            _validation = np.intersect1d(_outer_train, _block)
            _training = np.setdiff1d(_outer_train, _validation)
            if len(_validation) >= 2 and len(_training) > max(_dimensions):
                _inner_splits.append((_training, _validation))
        mo.stop(
            not _inner_splits,
            mo.callout(
                mo.md(f"Outer fold {_outer_fold} has no usable blocked inner splits."),
                kind="warn",
            ),
        )
        try:
            _selected, _fold_inner_rows = select_maxvar_alpha(
                _projected,
                _inner_splits,
                _dimensions,
                _alphas,
                support_cumulative_mass=float(gcca_support_mass_input.value),
            )
        except Exception as _exc:
            mo.stop(
                True,
                mo.callout(
                    mo.md(f"Outer fold {_outer_fold} ridge selection: {_exc}"),
                    kind="warn",
                ),
            )
        _inner_rows.extend(
            {"outer_fold": _outer_fold, **_row} for _row in _fold_inner_rows
        )

        _train_views = {
            _name: _projected[_name][_outer_train] for _name in _view_names
        }
        _test_views = {
            _name: _projected[_name][_outer_test] for _name in _view_names
        }
        for _alpha in sorted(set(_selected.values())):
            _alpha_dimensions = [
                _dimension
                for _dimension in _dimensions
                if _selected[_dimension] == _alpha
            ]
            _fit = fit_regularized_maxvar_gcca(
                _train_views,
                max(_alpha_dimensions),
                alpha=_alpha,
            )
            for _dimension in _alpha_dimensions:
                _score = score_maxvar_partial_support(
                    _fit,
                    _train_views,
                    _test_views,
                    n_components=_dimension,
                    cumulative_mass=float(gcca_support_mass_input.value),
                )
                _summary_rows.append(
                    {
                        "outer_fold": _outer_fold,
                        "shared_dimension": _dimension,
                        "selected_alpha": _alpha,
                        "n_train": len(_outer_train),
                        "n_test": len(_outer_test),
                        "mean_loo_corr": _score["mean_loo_corr"],
                        "mean_pairwise_corr": _score["mean_pairwise_corr"],
                        "mean_partial_support_corr": _score["mean_partial_support_corr"],
                        "mean_component_support": float(
                            np.mean(_fit.eigenvalues[:_dimension] / len(_view_names))
                        ),
                    }
                )
                for _name in _view_names:
                    _corrs = _score["per_view_corrs"][_name]
                    _view_rows.append(
                        {
                            "outer_fold": _outer_fold,
                            "shared_dimension": _dimension,
                            "view": _name,
                            "rank": _ranks[_name],
                            "basis_mode": _basis_modes[_name],
                            "mean_loo_corr": float(np.nanmean(_corrs)),
                        }
                    )
                    for _component, _corr in enumerate(_corrs, start=1):
                        _component_rows.append(
                            {
                                "outer_fold": _outer_fold,
                                "shared_dimension": _dimension,
                                "component": _component,
                                "view": _name,
                                "loo_corr": float(_corr),
                                "selected_support": _name in _score["support_views"][_component - 1],
                                "support_view_count": int(_score["support_counts"][_component - 1]),
                                "support_view_names": "|".join(_score["support_views"][_component - 1]),
                                "selected_view_corr": float(_score["selected_view_corrs"][_name][_component - 1]),
                                "selected_consensus_corr": float(_score["selected_consensus_corrs"][_name][_component - 1]),
                                "maxvar_eigenvalue": float(
                                    _fit.eigenvalues[_component - 1]
                                ),
                                "compound_singular_value": float(_fit.singular_values[_component - 1]),
                                "view_support": float(
                                    _fit.view_support[_name][_component - 1]
                                ),
                                "objective_contribution": float(_fit.objective_contributions[_name][_component - 1]),
                                "contribution_share": float(_fit.contribution_shares[_name][_component - 1]),
                            }
                        )
                    _ambient, _intercept = maxvar_ambient_map(
                        _fit,
                        _name,
                        _bases_by_view[_name][_outer_fold],
                        n_components=_dimension,
                    )
                    _artifacts[(_outer_fold, _dimension, _name)] = {
                        "ambient_map": _ambient,
                        "intercept": _intercept,
                        "basis": _bases_by_view[_name][_outer_fold],
                        "rotation": _fit.rotations[_name][:, :_dimension],
                        "test_shared_scores": _score["test_scores"][_name],
                        "all_anchor_scores": _embeddings_by_view[_name] @ _ambient + _intercept,
                    }
                _artifacts[(_outer_fold, _dimension, "__shared__")] = {
                    "train_shared_scores": _fit.shared_scores[:, :_dimension],
                    "eigenvalues": _fit.eigenvalues[:_dimension],
                    "singular_values": _fit.singular_values[:_dimension],
                    "outer_train_idxs": np.asarray(_outer_train),
                    "outer_test_idxs": np.asarray(_outer_test),
                }

    gcca_summary_table = pd.DataFrame(_summary_rows)
    gcca_view_table = pd.DataFrame(_view_rows)
    gcca_component_table = pd.DataFrame(_component_rows)
    gcca_inner_table = pd.DataFrame(_inner_rows)
    _stability_rows, _block_rows = [], []
    for _dimension in _dimensions:
        _folds = sorted(
            _fold for (_fold, _dim, _name) in _artifacts
            if _dim == _dimension and _name == "__shared__"
        )
        _anchor_scores = {}
        _eigenvalues = {}
        for _fold in _folds:
            _scores = np.mean(
                np.stack([
                    _artifacts[(_fold, _dimension, _name)]["all_anchor_scores"]
                    for _name in _view_names
                ]), axis=0,
            )
            _anchor_scores[_fold] = _scores
            _eigenvalues[_fold] = _artifacts[(_fold, _dimension, "__shared__")]["eigenvalues"]
            _artifacts[(_fold, _dimension, "__shared__")]["all_anchor_consensus_scores"] = _scores
        if len(_folds) < 2:
            continue
        _alignment = align_maxvar_folds(
            _anchor_scores, _eigenvalues,
            match_threshold=float(gcca_match_threshold_input.value),
            eigengap_threshold=float(gcca_eigengap_threshold_input.value),
        )
        _stable_indices = np.flatnonzero(_alignment["stable"])
        if _stable_indices.size:
            _aligned_stable = []
            for _fold in _folds:
                _matched = _alignment["assignments"][_fold][_stable_indices]
                _signed = (
                    _anchor_scores[_fold][:, _matched]
                    * _alignment["signs"][_fold][_stable_indices]
                )
                _aligned_stable.append(_signed)
            _artifacts[(_alignment["medoid_fold"], _dimension, "__shared__")].update({
                "stable_component_indices": _stable_indices + 1,
                "aggregate_stable_anchor_scores": np.mean(np.stack(_aligned_stable), axis=0),
            })
        for _fold in _folds:
            for _component in range(_dimension):
                _stability_rows.append({
                    "shared_dimension": _dimension,
                    "component": _component + 1,
                    "outer_fold": _fold,
                    "medoid_fold": _alignment["medoid_fold"],
                    "matched_component": int(_alignment["assignments"][_fold][_component]) + 1,
                    "match_correlation": float(_alignment["match_correlations"][_fold][_component]),
                    "sign": float(_alignment["signs"][_fold][_component]),
                    "median_absolute_match_correlation": float(_alignment["median_match_correlation"][_component]),
                    "relative_neighboring_eigengap": float(_alignment["relative_neighboring_eigengap"][_component]),
                    "stable_axis": bool(_alignment["stable"][_component]),
                })
        _blocks = maxvar_component_blocks(
            _alignment["stable"], _alignment["relative_neighboring_eigengap"],
            eigengap_threshold=float(gcca_eigengap_threshold_input.value),
        )
        for _block_index, _block in enumerate(_blocks, start=1):
            _medoid_scores = _anchor_scores[_alignment["medoid_fold"]][:, _block]
            _overlaps = []
            for _fold in _folds:
                _matched = _alignment["assignments"][_fold][list(_block)]
                _overlaps.extend(maxvar_block_overlap(_medoid_scores, _anchor_scores[_fold][:, _matched]))
            for _name in _view_names:
                _rows = gcca_component_table[
                    (gcca_component_table["shared_dimension"] == _dimension)
                    & (gcca_component_table["outer_fold"] == _alignment["medoid_fold"])
                    & (gcca_component_table["view"] == _name)
                    & (gcca_component_table["component"].isin([value + 1 for value in _block]))
                ]
                _support = float(_rows["view_support"].sum())
                _block_rows.append({
                    "shared_dimension": _dimension,
                    "block": _block_index,
                    "component_indices": "|".join(str(value + 1) for value in _block),
                    "block_size": len(_block), "view": _name,
                    "summed_view_support": _support,
                    "block_contribution_share": float(_support / max(gcca_component_table[(gcca_component_table["shared_dimension"] == _dimension) & (gcca_component_table["outer_fold"] == _alignment["medoid_fold"]) & (gcca_component_table["component"].isin([value + 1 for value in _block]))]["objective_contribution"].sum(), np.finfo(float).eps)),
                    "mean_canonical_correlation": float(np.mean(_overlaps)),
                })
    gcca_stability_table = pd.DataFrame(_stability_rows)
    gcca_blocks_table = pd.DataFrame(_block_rows)
    gcca_artifacts = _artifacts
    gcca_analysis_metadata = {
        "anchor": _anchor.name,
        "views": _view_names,
        "view_ranks": _ranks,
        "basis_modes": _basis_modes,
        "shared_dimensions": _dimensions,
        "ridge_alphas": _alphas,
        "ridge_selection": "blocked_inner_cv_leave_one_view_out_per_dimension",
        "support_cumulative_mass": float(gcca_support_mass_input.value),
        "axis_match_correlation_threshold": float(gcca_match_threshold_input.value),
        "relative_neighboring_eigengap_threshold": float(gcca_eigengap_threshold_input.value),
        "view_weights": "equal",
    }
    return (
        gcca_analysis_metadata,
        gcca_artifacts,
        gcca_blocks_table,
        gcca_component_table,
        gcca_inner_table,
        gcca_stability_table,
        gcca_summary_table,
        gcca_view_table,
    )


@app.cell(hide_code=True)
def _(
    gcca_blocks_table,
    gcca_component_table,
    gcca_inner_table,
    gcca_stability_table,
    gcca_summary_table,
    gcca_view_table,
    mo,
    plt,
):
    _fold_summary = (
        gcca_summary_table.groupby("shared_dimension", as_index=False)[
            ["mean_loo_corr", "mean_pairwise_corr", "selected_alpha", "mean_component_support"]
        ]
        .mean()
        .sort_values("shared_dimension")
    )
    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4.2))
    _axes[0].plot(
        _fold_summary["shared_dimension"],
        _fold_summary["mean_loo_corr"],
        marker="o",
        label="leave-one-view-out",
    )
    _axes[0].plot(
        _fold_summary["shared_dimension"],
        _fold_summary["mean_pairwise_corr"],
        marker="o",
        label="pairwise",
    )
    _axes[0].set_ylabel("Held-out correlation")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(
        _fold_summary["shared_dimension"],
        _fold_summary["selected_alpha"],
        marker="o",
    )
    _axes[1].set_yscale("symlog", linthresh=1e-7)
    _axes[1].set_ylabel("Mean selected alpha")
    _axes[2].plot(
        _fold_summary["shared_dimension"],
        _fold_summary["mean_component_support"],
        marker="o",
    )
    _axes[2].set_ylabel("Mean MAXVAR support / view")
    for _ax in _axes:
        _ax.set_xlabel("Shared dimension")
        _ax.grid(True, alpha=0.25)
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md("### Outer-fold summary"),
            mo.ui.table(gcca_summary_table, selection=None),
            mo.md("### Per-view held-out agreement"),
            mo.ui.table(gcca_view_table, selection=None),
            mo.md("### Component diagnostics"),
            mo.ui.table(gcca_component_table, selection=None),
            mo.md("### Fold-aligned axis stability"),
            mo.ui.table(gcca_stability_table, selection=None),
            mo.md("### Rotation-invariant component blocks"),
            mo.ui.table(gcca_blocks_table, selection=None),
            mo.md("### Inner regularization grid"),
            mo.ui.table(gcca_inner_table, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    gcca_output_input = mo.ui.text(
        value="results/maxvar_gcca",
        label="GCCA artifact directory",
        full_width=True,
    )
    gcca_save_button = mo.ui.run_button(label="Save GCCA artifacts")
    mo.vstack([gcca_output_input, gcca_save_button])
    return gcca_output_input, gcca_save_button


@app.cell(hide_code=True)
def _(
    Path,
    gcca_analysis_metadata,
    gcca_artifacts,
    gcca_blocks_table,
    gcca_component_table,
    gcca_inner_table,
    gcca_output_input,
    gcca_run_map,
    gcca_save_button,
    gcca_stability_table,
    gcca_summary_table,
    gcca_view_table,
    mo,
    np,
    run_metadata,
    sanitize_name,
):
    mo.stop(not gcca_save_button.value, mo.md(""))
    import json as _json

    _output = Path(gcca_output_input.value)
    _output.mkdir(parents=True, exist_ok=True)
    gcca_summary_table.to_csv(_output / "summary.csv", index=False)
    gcca_view_table.to_csv(_output / "view_metrics.csv", index=False)
    gcca_component_table.to_csv(_output / "component_metrics.csv", index=False)
    gcca_component_table.to_csv(_output / "view_component_support.csv", index=False)
    gcca_stability_table.to_csv(_output / "component_stability.csv", index=False)
    gcca_blocks_table.to_csv(_output / "component_blocks.csv", index=False)
    gcca_inner_table.to_csv(_output / "inner_ridge_grid.csv", index=False)
    _metadata = {
        **gcca_analysis_metadata,
        "anchor_run": run_metadata(
            gcca_run_map[gcca_analysis_metadata["anchor"]]
        ),
        "runs": {
            _name: run_metadata(gcca_run_map[_name])
            for _name in gcca_analysis_metadata["views"]
        },
    }
    (_output / "metadata.json").write_text(
        _json.dumps(_metadata, indent=2, sort_keys=True)
    )
    for (_fold, _dimension, _name), _artifact in gcca_artifacts.items():
        _folder = _output / f"fold_{_fold}" / f"shared_dim_{_dimension}"
        _folder.mkdir(parents=True, exist_ok=True)
        _filename = (
            "shared.npz"
            if _name == "__shared__"
            else f"view_{sanitize_name(_name)}.npz"
        )
        np.savez_compressed(_folder / _filename, **_artifact)
    mo.callout(mo.md(f"Saved GCCA artifacts to `{_output}`."), kind="success")
    return


@app.cell(hide_code=True)
def _(Path, mo, pd, plt):
    import json
    import math

    def _run_key(run):
        """Identify the underlying embedding run independently of RRR/PCA kind."""
        dataset_key = run.get("dataset_key") or [run.get("dataset", "unknown")]
        model_config = run.get("per_model_config") or {}
        return (
            "/".join(str(value) for value in dataset_key),
            str(model_config.get("model_dir", run.get("model", "unknown"))),
            str(model_config.get("layer", "?")),
            str(run.get("offset", "?")),
        )

    def _run_label(key):
        dataset, model, layer, offset = key
        stimulus = dataset.rsplit("/", 1)[-1]
        offset_suffix = "" if offset == "0" else f" o{offset}"
        return f"{stimulus} L{layer}{offset_suffix}"

    def _method_label(base, peer):
        base_kind = str(base.get("kind", "unknown")).upper()
        peer_kind = str(peer.get("kind", "unknown")).upper()
        return base_kind if base_kind == peer_kind else f"{base_kind} → {peer_kind}"

    def _load_batch_summaries(root):
        rows = {"cca": [], "rsa": []}
        for metadata_path in sorted(root.rglob("comparison_metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            base = metadata.get("base", {})
            peer = metadata.get("peer", {})
            base_key, peer_key = _run_key(base), _run_key(peer)
            common = {
                "comparison": f"{_run_label(base_key)} → {_run_label(peer_key)}",
                "method": _method_label(base, peer),
                "base_name": base.get("name", "unknown"),
                "peer_name": peer.get("name", "unknown"),
                "result_dir": str(metadata_path.parent),
            }
            for analysis in rows:
                summary_path = metadata_path.parent / f"{analysis}_summary.csv"
                if not summary_path.exists():
                    continue
                summary = pd.read_csv(summary_path)
                rows[analysis].append(summary.assign(**common))
        return {
            analysis: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            for analysis, frames in rows.items()
        }

    def _plot_pairwise_grid(frame, metric, ylabel, title):
        def _reciprocal_sort_key(comparison):
            base, separator, peer = comparison.partition(" → ")
            if not separator:
                return (comparison, comparison, 0)
            first, second = sorted((base, peer))
            direction = 0 if (base, peer) == (first, second) else 1
            return (first, second, direction)

        # Two columns form one comparison unit: A → B on the left and its
        # reciprocal B → A on the right whenever both directions exist.
        comparisons = sorted(
            frame["comparison"].unique(), key=_reciprocal_sort_key
        )
        ncols = min(2, len(comparisons))
        nrows = math.ceil(len(comparisons) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.2 * ncols, 4.0 * nrows),
            squeeze=False,
            sharey=True,
        )
        for ax, comparison in zip(axes.flat, comparisons):
            subset = frame[frame["comparison"] == comparison]
            for method, method_rows in subset.groupby("method", sort=True):
                # A repeated batch run can leave duplicate summaries; retain the
                # last discovered value for each rank.
                method_rows = method_rows.sort_values("result_dir").drop_duplicates(
                    "rank", keep="last"
                )
                ax.plot(
                    method_rows["rank"],
                    method_rows[metric],
                    marker="o",
                    linewidth=1.5,
                    label=method,
                )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Subspace rank")
            ax.set_title(comparison, fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        for ax in axes.flat[len(comparisons) :]:
            ax.set_visible(False)
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        return fig

    _results_root = Path("results")
    _summaries = _load_batch_summaries(_results_root)
    _cca = _summaries["cca"]
    _rsa = _summaries["rsa"]
    mo.stop(
        _cca.empty and _rsa.empty,
        mo.callout(
            mo.md(
                "No batch comparison summaries were found under `results/`. "
                "Run `scripts/rrr/compare_subspaces.py` first."
            ),
            kind="warn",
        ),
    )

    _items = [mo.md("## Batch pairwise comparison summary")]
    if not _cca.empty:
        _items.extend(
            [
                mo.md(
                    "### CCA\n\nEach panel is an ordered pairwise comparison; "
                    "RRR and PCA summaries are overlaid when both are available."
                ),
                _plot_pairwise_grid(
                    _cca,
                    "mean_all_test_corr",
                    "Mean held-out canonical correlation",
                    "",
                ),
            ]
        )
    if not _rsa.empty:
        _items.extend(
            [
                mo.md("### RSA"),
                _plot_pairwise_grid(
                    _rsa,
                    "mean_spearman",
                    "Mean Spearman correlation",
                    "RSA over projected word embeddings",
                ),
            ]
        )
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
