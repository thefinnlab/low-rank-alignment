"""Encode neural responses from saved language-decoder task axes."""

from src.core.config import LanguageAxisEncodingConfig
from src.utils.config_utils import load_config, parse_args
from src.utils.language_axis_encoding import run_language_axis_encoding


def main():
    args, overrides = parse_args(
        "Test neural sufficiency and removal of saved language-decoder axes"
    )
    config = load_config(args.config, overrides, LanguageAxisEncodingConfig)
    output_dir = run_language_axis_encoding(config)
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
