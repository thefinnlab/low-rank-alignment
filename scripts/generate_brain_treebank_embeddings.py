"""Generate missing embeddings for all movies in the cross-stimulus cohort.

This is an orchestration entrypoint around the repository's existing embedding
generator. It discovers movies from Brain Treebank metadata, then runs each
configured model once per distinct movie while leaving complete outputs alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import h5py

# Keep the documented ``uv run python scripts/...`` invocation independent of
# whether the caller has exported PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import GenerateEmbeddingsConfig
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.rrr.cross_stimulus import discover_subject_movies
from src.utils.shared.config import load_config, parse_override_args
from scripts.utilities.generate_embeddings import _auto_output_name, generate


def _requested_layers(config: GenerateEmbeddingsConfig, metadata: dict) -> list[int]:
    layers = config.layers
    if layers == "all" or layers is None:
        layers = metadata.get("layers_saved", [])
    return [int(layer) for layer in layers]


def embedding_output_dir(config: GenerateEmbeddingsConfig, movie: str) -> Path:
    output_name = config.output_name or _auto_output_name(config.hf_model_id)
    context = resolve_dataset_context(
        dataset="brain_treebank",
        subject_ids=[],
        per_dataset_config={"movie": movie},
        data_root=config.data_root,
    )
    adapter = get_dataset_adapter("brain_treebank")
    return Path(adapter.get_embedding_dir(config.data_root, context, output_name))


def embedding_is_complete(config: GenerateEmbeddingsConfig, movie: str) -> bool:
    """Check the complete generated-embedding artifact for one movie."""
    output_dir = embedding_output_dir(config, movie)
    features_path = output_dir / "features.hdf5"
    transcript_path = output_dir / "transcript.tsv"
    metadata_path = output_dir / "metadata.json"
    if not all(path.is_file() for path in (features_path, transcript_path, metadata_path)):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset") != "brain_treebank":
            return False
        if metadata.get("stimulus_id") != movie:
            return False
        raw_saved_layers = metadata.get("layers_saved")
        if not isinstance(raw_saved_layers, list) or not raw_saved_layers:
            return False
        layers = _requested_layers(config, metadata)
        saved_layers = {int(layer) for layer in raw_saved_layers}
        if not set(layers).issubset(saved_layers):
            return False
        with h5py.File(features_path, "r") as handle:
            return all(f"layer-{layer}" in handle for layer in layers)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def movies_for_config(config: GenerateEmbeddingsConfig) -> list[str]:
    discovered = discover_subject_movies(config.data_root, min_movies=2)
    return list(dict.fromkeys(movie for movies in discovered.values() for movie in movies))


def generate_missing(config: GenerateEmbeddingsConfig) -> tuple[list[str], list[str]]:
    if config.dataset != "brain_treebank":
        raise ValueError("Brain Treebank embedding generation requires dataset=brain_treebank.")
    movies = movies_for_config(config)
    generated: list[str] = []
    skipped: list[str] = []
    for movie in movies:
        if embedding_is_complete(config, movie):
            print(f"Skipping {movie}: embeddings are complete")
            skipped.append(movie)
            continue
        movie_config = deepcopy(config)
        movie_config.per_dataset_config = dict(movie_config.per_dataset_config)
        movie_config.per_dataset_config["movie"] = movie
        print(f"Generating embeddings for {movie}")
        generate(movie_config)
        generated.append(movie)
    return generated, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    args, unknown = parser.parse_known_args()
    overrides = parse_override_args(unknown)
    config_paths = args.config or [None]
    for path in config_paths:
        config = load_config(path, overrides, GenerateEmbeddingsConfig)
        generated, skipped = generate_missing(config)
        print(f"Finished {path or '<defaults>'}: generated={len(generated)}, skipped={len(skipped)}")


if __name__ == "__main__":
    main()
