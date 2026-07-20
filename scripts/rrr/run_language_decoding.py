"""Decode word-aligned language features from saved RRR input subspaces."""

from src.core.config import LanguageDecodingConfig
from src.utils.config_utils import (
    expand_per_model_layer_configs,
    load_config,
    parse_args,
)
from src.utils.language_decoding import run_language_decoding


def main():
    args, overrides = parse_args("Decode linguistic features from RRR subspaces")
    config = load_config(args.config, overrides, LanguageDecodingConfig)
    configs = expand_per_model_layer_configs(config)
    for layer_config in configs:
        layer = layer_config.encoding.per_model_config.get("layer")
        print(f"\nLanguage decoding for layer {layer}")
        output_dir = run_language_decoding(layer_config)
        print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
