"""Run downstream encoding from saved fold-specific GCCA artifacts."""

from src.core.config import GCCAEncodingConfig
from src.utils.config_utils import load_config, parse_args
from src.utils.gcca_encoding_runner import run_gcca_encoding


def main():
    args, overrides = parse_args("Evaluate saved GCCA scores as neural encodings")
    output = run_gcca_encoding(load_config(args.config, overrides, GCCAEncodingConfig))
    print(f"Saved GCCA encoding results to {output}")


if __name__ == "__main__":
    main()
