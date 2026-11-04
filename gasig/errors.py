"""Error classes of the package and the record of non-fatal failures.

Every error that the package raises on purpose is a :class:`GasigError`. Each
class also subclasses the built-in exception that the package raised before
this module existed (``ValueError``, ``RuntimeError`` or
``FileNotFoundError``), so callers that catch the built-in type keep working.

Classes, with the exit code of the ``gasig`` command:

* :class:`ConfigError` (2): the configuration file is invalid.
* :class:`MissingInputError` (2): an input file or folder does not exist.
* :class:`InputError` (3): an input file or an input array has wrong content.
* :class:`MechanismError` (3): the mechanism does not fit the request, for
  example a species that is not in the mechanism.
* :class:`SolverError` (3): a Cantera or Zero-RK solver failed.
  :class:`IgnitionSolverError` is the reactor, :class:`FlameSolverError` the
  flame, :class:`SolverTimeout` a flame solve over its time limit.
* :class:`IntegrationError` (3): the sensitivity or the linearised-knockout
  integration gave non-finite values.
* :class:`ReductionError` (3): the reduction sweep cannot continue, for
  example because the detailed mechanism does not ignite.

An error carries the ``stage`` (``sensitivity``, ``rank``, ``lk``,
``reduce``, ``report``) and a ``context`` dictionary, for example the
condition and the mechanism size. :meth:`GasigError.describe` gives both as
one JSON-ready dictionary.

Non-fatal failures
------------------
The sweep and the report continue after a solver failure at one condition.
:func:`record_failure` writes each such failure as one JSON line to the
failure file of the running command (``<log name>.failures.jsonl`` beside
the log file, opened by :func:`gasig.logs.setup_logging`). A fatal error is
recorded with ``"fatal": true``. Without an open failure file the functions
only count. This module must stay free of Cantera and torch imports.
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


class GasigError(Exception):
    """Base class of the errors of the package."""

    category = "error"
    exit_code = 3

    def __init__(self, message: str = "", *, stage: str | None = None, **context: Any):
        super().__init__(message)
        self.stage = stage
        self.context = {k: v for k, v in context.items() if v is not None}

    def describe(self) -> dict:
        """The category, the class, the message, the stage and the context as one dictionary."""
        out = {"category": self.category, "type": type(self).__name__, "message": str(self)}
        if self.stage:
            out["stage"] = self.stage
        if self.context:
            out["context"] = _json_ready(self.context)
        return out


class ConfigError(GasigError, ValueError):
    """The configuration file is invalid."""

    category = "configuration"
    exit_code = 2


class InputError(GasigError, ValueError):
    """An input file or an input array has wrong content."""

    category = "input"


class MissingInputError(InputError, FileNotFoundError):
    """An input file or folder does not exist."""

    category = "missing input"
    exit_code = 2


class MechanismError(GasigError, ValueError):
    """The mechanism does not fit the request."""

    category = "mechanism"


class MechanismCheckError(MechanismError, RuntimeError):
    """The torch rate law does not reproduce Cantera; a ``RuntimeError`` before this module existed."""


class SolverError(GasigError, RuntimeError):
    """A Cantera or Zero-RK solver failed."""

    category = "solver"


class IgnitionSolverError(SolverError):
    """The reactor integration failed."""

    category = "ignition solver"


class FlameSolverError(SolverError):
    """A flame solver failed, is missing, or gave no flame."""

    category = "flame solver"


class SolverTimeout(FlameSolverError):
    """A flame solve took longer than its time limit."""

    category = "solver timeout"


class IntegrationError(GasigError, RuntimeError):
    """The sensitivity or the linearised-knockout integration gave non-finite values."""

    category = "integration"


class ReductionError(GasigError, RuntimeError):
    """The reduction sweep cannot continue."""

    category = "reduction"


def describe_exception(exc: BaseException) -> dict:
    """:meth:`GasigError.describe` for a package error; the class and the message for any other error."""
    if isinstance(exc, GasigError):
        return exc.describe()
    return {"category": "unexpected", "type": type(exc).__name__, "message": str(exc)}


def exit_code(exc: BaseException) -> int:
    """Exit code of the ``gasig`` command for ``exc``: 2 for input the user must fix, else 3."""
    if isinstance(exc, GasigError):
        return int(exc.exit_code)
    if isinstance(exc, FileNotFoundError):
        return 2
    return 3


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):          # NumPy scalar
        try:
            return _json_ready(obj.item())
        except (TypeError, ValueError):
            pass
    return str(obj)


class FailureLog:
    """Counts failures and, with a path, writes each one as a JSON line."""

    def __init__(self, path: Path | None = None, mode: str = "a"):
        self.path = None if path is None else Path(path)
        self.counts: Counter = Counter()
        self._lock = threading.Lock()
        self._fh = None
        if self.path is not None:
            self._fh = self.path.open(mode, encoding="utf-8")

    def record(self, exc: BaseException, *, stage: str | None = None, fatal: bool = False, **context: Any) -> dict:
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "fatal": bool(fatal), **describe_exception(exc)}
        if stage and "stage" not in entry:
            entry["stage"] = stage
        if context:
            entry["context"] = {**entry.get("context", {}), **_json_ready(context)}
        with self._lock:
            self.counts[entry["category"]] += 1
            if self._fh is not None:
                try:
                    self._fh.write(json.dumps(entry, allow_nan=False) + "\n")
                    self._fh.flush()
                except (OSError, ValueError):
                    pass
        return entry

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


_ACTIVE = FailureLog()


def open_failure_log(path: Path | None, mode: str = "a") -> FailureLog:
    """Make a new failure log the active one; ``path`` None counts without a file."""
    global _ACTIVE
    _ACTIVE.close()
    _ACTIVE = FailureLog(path, mode)
    return _ACTIVE


def close_failure_log() -> None:
    """Close the file of the active failure log; the counts stay until the next open."""
    _ACTIVE.close()


def active_failure_log() -> FailureLog:
    return _ACTIVE


def record_failure(exc: BaseException, *, stage: str | None = None, fatal: bool = False, **context: Any) -> dict:
    """Record ``exc`` in the active failure log; returns the entry."""
    return _ACTIVE.record(exc, stage=stage, fatal=fatal, **context)


def failure_counts() -> dict[str, int]:
    """Failures per category since the active log was opened."""
    return dict(_ACTIVE.counts)
