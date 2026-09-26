"""Batch export of every figure in notebooks/final_plots.py."""

from __future__ import annotations

import argparse
from pathlib import Path


def _layer(value):
    if value.lower() == "none":
        return None
    try:
        layer = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("layer must be an integer or 'none'") from error
    if layer < 0:
        raise argparse.ArgumentTypeError("layer must be nonnegative or 'none'")
    return layer


def available_pairs(quantiles, *, loso=False):
    """Use the same eligibility criteria as the notebook's example dropdowns."""
    from src.utils.plotting.final_data import CONTROL_MODELS, DATASETS

    methods = ("RRR baseline", "RRR transfer", "PCA baseline") if loso else ("rrr", "pca")
    present = {(key[0], key[1], key[2], key[-1]) for key in quantiles}
    return sorted(
        {(key[1], key[2]) for key in quantiles
         if key[1] not in CONTROL_MODELS and all((dataset, key[1], key[2], method) in present
                for dataset in DATASETS for method in methods)},
        key=lambda pair: (pair[0], pair[1] is not None,
                          pair[1] if pair[1] is not None else -1),
    )


def export_figures(data, output_dir, formats, example_pair, loso_pair):
    """Save and close figures one at a time to keep export memory bounded."""
    import matplotlib.pyplot as plt
    from src.utils.plotting import final_figures as figures

    (internal, loso, cross, quantiles, loso_quantiles, full_rows,
     significant, decoding, gaps) = data
    control_internal = internal
    from src.utils.plotting.final_data import primary_analysis_data

    (internal, loso, cross, quantiles, loso_quantiles, decoding) = primary_analysis_data(
        internal, loso, cross, quantiles, loso_quantiles, decoding
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order = figures.model_order(full_rows)
    jobs = [
        ("rrr_internal_aggregate", lambda: figures.rrr_aggregate(
            internal, loso, quantiles, loso_quantiles, full_rows, example_pair,
            mode="internal")),
        ("rrr_loso_aggregate", lambda: figures.rrr_aggregate(
            internal, loso, quantiles, loso_quantiles, full_rows, loso_pair,
            mode="loso")),
        ("rrr_cross_dataset_aggregate", lambda: figures.cross_aggregate(cross)),
        ("rrr_cross_dataset_auc_gap", lambda: figures.cross_gap_scatter(cross)),
        ("language_decoding_aggregate", lambda: figures.decoding_aggregate(decoding)),
        ("encoding_performance", lambda: figures.encoding_grid(full_rows, order)),
        ("significant_targets", lambda: figures.significance_grid(significant, order)),
        ("rrr_pca_target_quantiles", lambda: figures.quantile_grid(quantiles, example_pair, internal)),
        ("mean_normalized_quantiles", lambda: figures.average_quantile_grid(quantiles)),
        ("saturation_sensitivity_fraction", lambda: figures.sensitivity_grid(
            internal, (.5, .75, .9, .95, .99, 1.0))),
        ("loso_saturation_sensitivity_fraction", lambda: figures.sensitivity_grid(
            loso, (.5, .75, .9, .95, .99, 1.0), mode="loso")),
        ("appendix_random_gpt2_saturation", lambda: figures.control_saturation_grid(
            control_internal, "random-gpt2")),
        ("appendix_glove_saturation", lambda: figures.control_saturation_grid(
            control_internal, "glove-300")),
    ]
    paths = []
    for index, (name, build) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] Building {name}...", flush=True)
        figure = build()
        try:
            for extension in dict.fromkeys(formats):
                path = output_dir / f"{name}.{extension}"
                print(f"Saving {path}...", flush=True)
                figure.savefig(path, dpi=300)
                paths.append(path)
                print(f"Saved {path}", flush=True)
        finally:
            plt.close(figure)
    table_path = output_dir / "significant_targets.tex"
    print(f"Saving {table_path}...", flush=True)
    table_path.write_text(figures.significance_latex(significant, order) + "\n")
    paths.append(table_path)
    gaps_path = output_dir / "result_gaps.txt"
    print(f"Saving {gaps_path}...", flush=True)
    gaps_path.write_text("\n".join(gaps) + ("\n" if gaps else ""))
    paths.append(gaps_path)
    print(f"Export complete: {len(paths)} files in {output_dir}", flush=True)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for exported figures (existing filenames are replaced)")
    parser.add_argument("--format", nargs="+", choices=("pdf", "png", "svg"),
                        default=["pdf", "png", "svg"], help="Output formats (default: pdf png svg)")
    parser.add_argument("--example-model", default="gpt2")
    parser.add_argument("--example-layer", type=_layer, default=0,
                        help="Example layer, or 'none' for a layerless model (default: 0)")
    parser.add_argument("--loso-example-model", help="Override the LOSO example model")
    parser.add_argument("--loso-example-layer", type=_layer, default=argparse.SUPPRESS,
                        help="Override the LOSO example layer, or 'none'")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    from src.utils.plotting.final_data import load_final_data

    results_root = Path(__file__).resolve().parents[3] / "results"
    print(f"Loading result summaries from {results_root} (using the cache when valid)...", flush=True)
    data = load_final_data(results_root)
    print("Result summaries loaded. Validating example model/layers...", flush=True)
    example_pair = (args.example_model, args.example_layer)
    loso_pair = (args.loso_example_model or args.example_model,
                 getattr(args, "loso_example_layer", args.example_layer))
    for label, pair, quantiles, is_loso in (
        ("internal", example_pair, data[3], False),
        ("LOSO", loso_pair, data[4], True),
    ):
        pairs = available_pairs(quantiles, loso=is_loso)
        if pair not in pairs:
            available = ", ".join(f"{model}/{'none' if layer is None else layer}"
                                  for model, layer in pairs) or "none"
            parser.error(f"Unavailable {label} example {pair}. Available model/layer pairs: {available}")
    print(f"Examples: internal={example_pair}, LOSO={loso_pair}", flush=True)
    print(f"Missing or incomplete results: {len(data[8])}; exporting available results.", flush=True)
    return export_figures(data, args.output_dir, args.format, example_pair, loso_pair)
