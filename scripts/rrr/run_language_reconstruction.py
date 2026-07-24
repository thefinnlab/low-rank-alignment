"""Predict saved RRR/PCA embedding subspaces from language-task variables."""

from src.core.config import LanguageReconstructionConfig
from src.utils.config_utils import (
    expand_per_model_layer_configs,
    load_config,
    parse_args,
)
from src.utils.language_reconstruction import run_language_reconstruction


def main():
    args, overrides = parse_args(
        "Reconstruct RRR/PCA subspaces from language-task variables"
    )
    config = load_config(args.config, overrides, LanguageReconstructionConfig)
    configs = expand_per_model_layer_configs(config)
    for layer_config in configs:
        layer = layer_config.encoding.per_model_config.get("layer")
        print(f"\nLanguage reconstruction for layer {layer}")
        output_dir = run_language_reconstruction(layer_config)
        print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
