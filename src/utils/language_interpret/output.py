"""Direct Artifact writes, with a file sink for standalone decoding utilities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_selector(task, subspace, rank, fold):
    selector = {"task": task, "subspace": subspace, "fold": int(fold)}
    if rank is not None:
        selector["rank"] = int(rank)
    return selector


class DecodingOutput:
    def __init__(self, directory, artifact=None, refresh=False):
        self.directory = Path(directory)
        self.artifact = artifact
        self.refresh = refresh

    def save(self, field, value, selector=None):
        if self.artifact is not None:
            self.artifact.save_field(field, value, selector=selector, overwrite=True)
            return
        directory = self.directory
        if selector and "subspace" in selector:
            directory /= f"tasks/{selector['task']}/{selector['subspace']}"
            if "rank" in selector:
                directory /= f"k{selector['rank']}"
            directory /= f"fold_{selector['fold']}"
        elif selector and field == "target_diagnostics":
            directory /= f"target_diagnostics/{selector['task']}"
        directory.mkdir(parents=True, exist_ok=True)
        name = {
            "fit_metrics": "metrics", "decoder_weights": "weights",
        }.get(field, field)
        if field == "decoder_weights" and (selector or {}).get("name"):
            name = "classifier_weights"
        if field == "target_diagnostics":
            name = selector["name"]
        if isinstance(value, pd.DataFrame):
            if field == "fit_metrics":
                (directory / f"{name}.json").write_text(json.dumps(value.iloc[0].to_dict(), allow_nan=True))
            else:
                value.to_csv(directory / f"{name}.csv", index=False)
        elif isinstance(value, dict) and any(isinstance(v, np.ndarray) for v in value.values()):
            np.savez_compressed(directory / f"{name}.npz", **value)
        else:
            (directory / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=True))

    def save_fit(self, metrics, weights, predictions):
        selector = fit_selector(metrics["task"], metrics["subspace"], metrics["k"], metrics["fold"])
        self.save("decoder_weights", weights, selector)
        self.save("heldout_predictions", predictions, selector)
        # Metrics commit last; an interrupted fit is not considered cached.
        self.save("fit_metrics", pd.DataFrame([metrics]), selector)

    def cached_fit(self, task, subspace, rank, fold):
        selector = fit_selector(task, subspace, rank, fold)
        if self.refresh or self.artifact is None:
            return None
        if not all(self.artifact.has_field(field, selector=selector) for field in
                   ("fit_metrics", "decoder_weights", "heldout_predictions")):
            return None
        return self.artifact.load_field("fit_metrics", selector=selector).iloc[0].to_dict()

    def weights(self, task, subspace, rank, fold):
        return self.artifact.load_field("decoder_weights", selector=fit_selector(task, subspace, rank, fold))

    def save_plot(self, path):
        if self.artifact is not None:
            self.artifact.save_field("plots", Path(path).read_bytes(),
                                     selector={"name": "rank_analysis"}, overwrite=True)
