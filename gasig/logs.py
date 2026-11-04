"""Console and file logging for the ``gasig`` logger tree.

This module must stay free of torch imports: the solvers and ``gasig reduce``
log through it without PyTorch.
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from .errors import close_failure_log, describe_exception, open_failure_log, record_failure

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_PREVIOUS = "_gasig_previous_state"


def teardown_logging() -> None:
    """Remove and close the handlers of :func:`setup_logging`; restore the logger state."""
    close_failure_log()
    logger = logging.getLogger("gasig")
    for handler in list(logger.handlers):
        if getattr(handler, "_gasig", False):
            logger.removeHandler(handler)
            handler.close()
    previous = getattr(logger, _PREVIOUS, None)
    if previous is not None:
        logger.propagate, level = previous
        logger.setLevel(level)
        delattr(logger, _PREVIOUS)


def setup_logging(output_dir: Path, level: str, file_name: str = "gasig.log", mode: str = "a") -> None:
    """Console and file logging for the 'gasig' logger tree.

    ``mode`` is the file mode of the log file: ``"a"`` appends to an existing
    file (the sensitivity and the ranking stage share one ``gasig.log``),
    ``"w"`` starts a new file. The failure file ``<stem>.failures.jsonl`` of
    :mod:`gasig.errors` opens beside the log file with the same mode; it
    holds one JSON line per recorded failure.
    """
    teardown_logging()
    open_failure_log(Path(output_dir) / f"{Path(file_name).stem}.failures.jsonl", mode)
    logger = logging.getLogger("gasig")
    setattr(logger, _PREVIOUS, (logger.propagate, logger.level))
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT, "%H:%M:%S"))
    console._gasig = True  # type: ignore[attr-defined]
    logger.addHandler(console)
    file_handler = logging.FileHandler(Path(output_dir) / file_name, mode=mode, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler._gasig = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)
    logger.propagate = False


def log_failure(logger: logging.Logger, message: str) -> None:
    """Record the current exception: one ERROR line everywhere, the traceback in the log file only.

    The exception also goes to the failure file as a fatal failure, with its
    category, its stage and its context.
    """
    exc = sys.exc_info()[1]
    if exc is None:
        logger.error("%s", message)
        return
    logger.error("%s (%s error): %s", message, describe_exception(exc)["category"], exc)
    record_failure(exc, fatal=True, during=message)
    text = traceback.format_exc()
    for handler in logging.getLogger("gasig").handlers:
        if getattr(handler, "_gasig", False) and isinstance(handler, logging.FileHandler):
            handler.acquire()
            try:
                handler.stream.write(text)
                handler.flush()
            finally:
                handler.release()
