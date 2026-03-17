import numpy as np
import pandas as pd


def _make_context(words_arr, idx, window):
    """
    Build a string context window around a target word.

    Returns a space-joined string of words from [idx-window, idx+window],
    clamped to the array bounds. The target word at `idx` is wrapped in
    **bold** markers.

    Args:
        words_arr (array-like): Sequence of word strings.
        idx (int): Index of the target word.
        window (int): Number of words to include on each side.

    Returns:
        str: Context string with the target word bolded.
    """
    lo = max(0, idx - window)
    hi = min(len(words_arr), idx + window + 1)
    parts = []
    for i in range(lo, hi):
        w = words_arr[i]
        if i == idx:
            parts.append(f"**{w}**")
        else:
            parts.append(w)
    return " ".join(parts)


def _sample(activations, words, num_bins, samples_per_bin, context_window, seed):
    """
    Sample words from bins across the activation distribution.

    Partitions activations into equal-width bins and randomly draws up to
    `samples_per_bin` words from each non-empty bin.

    Args:
        activations (np.ndarray): 1-D array of activation values.
        words (array-like): Sequence of word strings, aligned with activations.
        num_bins (int): Number of equal-width bins.
        samples_per_bin (int): Maximum samples to draw per bin.
        context_window (int): Words on each side for _make_context.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Rows with columns bin, range, activation, word, context,
        count_in_bin. Only non-empty bins produce rows.
    """
    rng = np.random.default_rng(seed)
    words_arr = np.array(words)
    bin_edges = np.linspace(activations.min(), activations.max(), num_bins + 1)
    bin_edges[-1] += 1e-9

    rows = []
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (activations >= lo) & (activations < hi)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        chosen = rng.choice(
            indices, size=min(samples_per_bin, len(indices)), replace=False
        )
        for idx in chosen:
            rows.append({
                "bin": i + 1,
                "range": f"[{lo:.3f}, {hi:.3f})",
                "activation": float(activations[idx]),
                "word": words_arr[idx],
                "context": _make_context(words_arr, idx, context_window),
                "count_in_bin": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def get_example_display_controls():
    """
    Create marimo UI sliders for the activation explorer.

    Returns:
        Tuple of (num_bins_slider, samples_per_bin_slider, context_slider, seed_input).
    """
    import marimo as mo

    num_bins_slider = mo.ui.slider(
        start=3, stop=20, value=8, step=1, label="Number of bins"
    )
    samples_per_bin_slider = mo.ui.slider(
        start=1, stop=30, value=5, step=1, label="Samples per bin"
    )
    context_slider = mo.ui.slider(
        start=1, stop=30, value=8, step=1, label="Context window (words each side)"
    )
    seed_input = mo.ui.number(value=0, label="Random seed")

    return num_bins_slider, samples_per_bin_slider, context_slider, seed_input


def generate_example_table(
    curr_activations,
    df_word,
    epochs,
    num_bins_slider,
    samples_per_bin_slider,
    context_slider,
    seed_input,
):
    """
    Generate a marimo-formatted table of example words per activation bin.

    Args:
        curr_activations: 1-D activation array.
        df_word: DataFrame with 'word' column.
        epochs: MNE Epochs-like object with .selection attribute.
        num_bins_slider: marimo slider for bin count.
        samples_per_bin_slider: marimo slider for samples per bin.
        context_slider: marimo slider for context window size.
        seed_input: marimo number input for random seed.

    Returns:
        marimo UI element displaying the binned examples.
    """
    import marimo as mo

    df = _sample(
        curr_activations,
        df_word.iloc[epochs.selection]["word"],
        num_bins=num_bins_slider.value,
        samples_per_bin=samples_per_bin_slider.value,
        context_window=context_slider.value,
        seed=int(seed_input.value),
    )

    if df.empty:
        return mo.callout(mo.md("No data — check your inputs."), kind="warn")

    bin_tables = []
    for bin_idx, bin_group in df.groupby("bin"):
        rng_label = bin_group["range"].iloc[0]
        count = bin_group["count_in_bin"].iloc[0]
        pct = 100 * count / len(curr_activations)
        header = mo.md(
            f"### Bin {bin_idx} &nbsp; `{rng_label}` "
            f"&nbsp; <small>({count} words, {pct:.1f}%)</small>"
        )
        row_html = ""
        for _, row in bin_group.sort_values("activation").iterrows():
            context_html = row["context"].replace(
                f"**{row['word']}**",
                f'<mark style="background:#fde68a;padding:0 2px;border-radius:3px;">{row["word"]}</mark>',
            )
            row_html += f"""
            <tr>
              <td style="font-family:monospace;white-space:nowrap;padding:6px 12px;border-bottom:1px solid #e5e7eb;color:#6b7280;">{row['activation']:.4f}</td>
              <td style="font-family:monospace;white-space:nowrap;padding:6px 12px;border-bottom:1px solid #e5e7eb;font-weight:600;">{row['word']}</td>
              <td style="padding:6px 12px;border-bottom:1px solid #e5e7eb;line-height:1.6;">{context_html}</td>
            </tr>"""
        table_html = mo.Html(f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
          <thead>
            <tr style="background:#f3f4f6;text-align:left;">
              <th style="padding:8px 12px;border-bottom:2px solid #d1d5db;white-space:nowrap;">Activation</th>
              <th style="padding:8px 12px;border-bottom:2px solid #d1d5db;white-space:nowrap;">Word</th>
              <th style="padding:8px 12px;border-bottom:2px solid #d1d5db;">Context</th>
            </tr>
          </thead>
          <tbody>{row_html}</tbody>
        </table>""")
        bin_tables.append(mo.vstack([header, table_html]))

    return mo.vstack(bin_tables, gap="1.5rem")


def get_mo_display_activations(
    curr_activations,
    df_word,
    epochs,
    num_bins_slider,
    samples_per_bin_slider,
    context_slider,
    seed_input,
):
    """
    Build the full marimo activation distribution explorer widget.

    Returns:
        marimo vstack with title, description, and the example table.
    """
    import marimo as mo

    return mo.vstack([
        mo.md("# Activation Distribution Explorer"),
        mo.md(
            "Samples words from across the activation distribution. "
            "The target word is **bolded** in its context window."
        ),
        generate_example_table(
            curr_activations,
            df_word,
            epochs,
            num_bins_slider,
            samples_per_bin_slider,
            context_slider,
            seed_input,
        ),
    ])
