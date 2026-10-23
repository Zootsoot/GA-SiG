"""Compatibility entry point of GA-SiG version 0.1.

Version 0.2 moved the code into the ``gasig`` package. This script runs the
ranking stage from a JSON configuration file, using the ``r_AB_<mode>.npy``
files of an earlier sensitivity run:

    python graph_search_to_importance.py config.json

The equivalent package command is ``gasig rank config.json``.
"""
import sys

from gasig.cli import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python graph_search_to_importance.py config.json", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(["rank", sys.argv[1]]))
