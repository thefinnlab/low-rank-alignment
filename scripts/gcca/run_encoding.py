"""Evaluate calibrated GCCA embeddings and compare them with saved RRR results."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT != _ROOT.parent:
    if (_ROOT / "src").is_dir():
        break
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.config import GCCAEncodingConfig
from src.utils.config_utils import load_config, parse_args
from src.utils.gcca_encoding_runner import run_gcca_encoding


def main():
    args, overrides = parse_args("Evaluate GCCA encodings and compare saved RRR results")
    output = run_gcca_encoding(load_config(args.config, overrides, GCCAEncodingConfig))
    print(f"Saved GCCA encoding results to {output}")


if __name__ == "__main__":
    main()
