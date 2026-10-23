"""Command-line interface: ``gasig`` or ``python -m gasig``."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, template_config


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run

    run(args.config, device=args.device, stage=args.stage)
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    from .pipeline import run

    run(args.config, device=args.device, stage="rank")
    return 0


def _cmd_reduce(args: argparse.Namespace) -> int:
    from .reduce import run_reduce

    run_reduce(args.config)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .report import run_report

    run_report(args.config)
    return 0


def _cmd_template(args: argparse.Namespace) -> int:
    text = json.dumps(template_config(), indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Template written to {args.output}")
    else:
        print(text)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    import numpy
    import cantera

    print(f"gasig   {__version__}")
    print(f"python  {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"numpy   {numpy.__version__}")
    print(f"cantera {cantera.__version__}")
    try:
        import matplotlib

        print(f"matplotlib {matplotlib.__version__} (figures of 'gasig report')")
    except ImportError:
        print("matplotlib not installed; 'gasig report' writes tables only")
    try:
        import torch
    except ImportError:
        print("torch   not installed; 'run' and 'rank' need it, 'reduce' and 'report' do not")
        return 0
    print(f"torch   {torch.__version__} (CUDA build: {torch.version.cuda})")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"cuda:{i}  {props.name}, {props.total_memory / 2**30:.1f} GiB")
    else:
        print("cuda    not available; runs use the CPU")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gasig",
        description="GA-SiG: sensitivity-based species ranking for kinetic mechanism reduction.",
    )
    parser.add_argument("--version", action="version", version=f"gasig {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the sensitivity integration and the ranking from a JSON configuration.")
    p_run.add_argument("config", help="Path of the JSON configuration file.")
    p_run.add_argument("--device", default=None, help="Override the 'device' key: auto, cpu, cuda, cuda:N.")
    p_run.add_argument("--stage", choices=["all", "sensitivity", "rank"], default="all",
                       help="Run every stage (default), only the sensitivity stage, or only the ranking.")
    p_run.set_defaults(func=_cmd_run)

    p_rank = sub.add_parser("rank", help="Rank species again from the r_AB files of an earlier run.")
    p_rank.add_argument("config", help="Path of the JSON configuration file.")
    p_rank.add_argument("--device", default=None, help=argparse.SUPPRESS)
    p_rank.set_defaults(func=_cmd_rank)

    p_red = sub.add_parser("reduce", help="Remove species in rank order and report the ignition-delay error per size.")
    p_red.add_argument("config", help="Path of the JSON configuration file of an earlier run.")
    p_red.set_defaults(func=_cmd_reduce)

    p_rep = sub.add_parser("report", help="Compare the selected reduced mechanism with the detailed one: tables and figures.")
    p_rep.add_argument("config", help="Path of the JSON configuration file of the run with a reduction sweep.")
    p_rep.set_defaults(func=_cmd_report)

    p_tpl = sub.add_parser("template", help="Print a configuration template with every default.")
    p_tpl.add_argument("-o", "--output", default=None, help="Write the template to this file instead of stdout.")
    p_tpl.set_defaults(func=_cmd_template)

    p_info = sub.add_parser("info", help="Show package, library and device information.")
    p_info.set_defaults(func=_cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Exit codes: 0 success, 2 configuration or file error, 3 run-time error.

    Set the environment variable ``GASIG_DEBUG=1`` to get the full traceback
    of a run-time error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # Console output must not fail on characters outside the console code page.
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, ImportError) as exc:
        if os.environ.get("GASIG_DEBUG"):
            raise
        print(f"Error: {str(exc).strip()}", file=sys.stderr)
        print("Set GASIG_DEBUG=1 for the full traceback.", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
