"""Importable facade for the language-decoding Artifact client."""

from importlib import import_module
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_implementation = import_module("scripts.language-interpret.run_language_decoding")
Artifact = _implementation.Artifact
run_config = _implementation.run_config
main = _implementation.main


if __name__ == "__main__":
    main()
