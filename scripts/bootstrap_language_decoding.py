#!/usr/bin/env python
"""Run paired rank-wise language-decoding bootstrap inference for one dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fmri.bootstrap_language_decoding import main


if __name__ == "__main__":
    main()
