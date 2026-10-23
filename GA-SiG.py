"""Compatibility entry point of GA-SiG version 0.1.

Version 0.2 moved the code into the ``gasig`` package. This script runs the
sensitivity stage from a JSON configuration file:

    python GA-SiG.py config.json

The equivalent package command is ``gasig run --stage sensitivity config.json``.
Refer to README.md for the configuration keys.
"""
import sys

from gasig.cli import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python GA-SiG.py config.json", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(["run", "--stage", "sensitivity", sys.argv[1]]))
