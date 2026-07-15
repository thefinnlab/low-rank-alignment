"""Run config-driven partial-support MAXVAR GCCA over saved RRR/PCA subspaces."""

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

from src.core.config import MaxvarGCCAConfig
from src.utils.config_utils import load_config, parse_args
from src.utils.gcca import run_gcca


def main() -> None:
    args, overrides = parse_args("Fit anchor-scoped partial-support MAXVAR GCCA")
    config = load_config(args.config, overrides, MaxvarGCCAConfig)
    output = run_gcca(config)
    print(f"Saved MAXVAR GCCA results to {output}")


if __name__ == "__main__":
    main()
