"""Write a Cantera Solution as a YAML mechanism file.

Cantera 3.2 cannot write a text field that holds only blanks and at least one
line break: ``Solution.write_yaml`` stops with ``IndexError: basic_string::substr:
__pos (which is 18446744073709551615) > this->size()``. The YAML emitter of
Cantera (``emitString`` in ``src/base/AnyMap.cpp``) trims the blanks of a
multi-line text and finds no character to keep. ``ck2yaml`` makes such a field
from a CHEMKIN comment that holds only blanks, for example a reaction line that
ends with ``! `` or with ``!`` and tabs: the reaction gets the note ``"\\n"``.

:func:`drop_blank_text` removes these fields in place from the user data of
the header, the species (with their thermo and transport data) and the
reactions. A field with text stays, and the elements, the phase options and
the state of the Solution do not change. :func:`write_yaml` calls it before it
writes. The module needs Cantera only; it does not import torch.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cantera as ct

from .errors import MechanismError

log = logging.getLogger(__name__)


def _is_blank_text(value) -> bool:
    return isinstance(value, str) and "\n" in value and not value.strip()


def blank_text_fields(data, path: str = "") -> list[str]:
    """Paths (``/note``, ``/thermo/note``) of the text fields in ``data`` that Cantera cannot write."""
    if isinstance(data, dict):
        return [p for key, value in data.items() for p in blank_text_fields(value, f"{path}/{key}")]
    if isinstance(data, (list, tuple)):
        return [p for i, value in enumerate(data) for p in blank_text_fields(value, f"{path}[{i}]")]
    return [path or "/"] if _is_blank_text(data) else []


def _without_blank_text(data):
    if isinstance(data, dict):
        return {key: _without_blank_text(value) for key, value in data.items() if not _is_blank_text(value)}
    if isinstance(data, (list, tuple)):
        return [_without_blank_text(value) for value in data if not _is_blank_text(value)]
    return data


def unwritable_fields(gas: ct.Solution) -> list[str]:
    """One label per text field of ``gas`` that Cantera cannot write, for example ``reaction 344 (A + B <=> C)/note``.

    Reactions are counted from 1, as in the YAML comments of Cantera.
    """
    out = [f"header{p}" for p in blank_text_fields(dict(gas.input_header))]
    for sp in gas.species():
        out += [f"species {sp.name}{p}" for p in blank_text_fields(sp.input_data)]
    for i, rx in enumerate(gas.reactions()):
        out += [f"reaction {i + 1} ({rx.equation}){p}" for p in blank_text_fields(rx.input_data)]
    return out


def _drop_user_blank(obj) -> None:
    """Remove the blank text fields from the user data of one Cantera object; keep every other user field."""
    if obj is None or not hasattr(obj, "clear_user_data"):
        return
    data = dict(obj.input_data)
    if not blank_text_fields(data):
        return
    obj.clear_user_data()
    core = obj.input_data
    user = _without_blank_text({key: value for key, value in data.items() if key not in core})
    if user:
        obj.update_user_data(user)


def drop_blank_text(gas: ct.Solution, *, stage: str | None = None) -> list[str]:
    """Remove the text fields of ``gas`` that Cantera cannot write, in place; returns their labels.

    The Species and Reaction objects change in place, so a Solution that
    shares them (a reduced mechanism of :func:`gasig.reduce.reduced_solution`)
    changes too. A blank field that is not user data raises a
    :class:`MechanismError`.
    """
    labels = unwritable_fields(gas)
    if not labels:
        return []
    header = dict(gas.input_header)
    if blank_text_fields(header):
        gas.clear_user_header()
        gas.update_user_header(_without_blank_text(header))
    for sp in gas.species():
        _drop_user_blank(sp.thermo)
        _drop_user_blank(getattr(sp, "transport", None))
        _drop_user_blank(sp)
    for rx in gas.reactions():
        _drop_user_blank(rx)
    left = unwritable_fields(gas)
    if left:
        raise MechanismError(f"the mechanism has {len(left)} text field(s) that Cantera cannot write and that are "
                             f"not user data ({left[0]}); remove them from the mechanism file.", stage=stage,
                             fields=left[:20])
    return labels


def write_yaml(gas: ct.Solution, path: str | Path, *, stage: str | None = None, quiet: bool = False) -> list[str]:
    """Write ``gas`` to ``path`` as a Cantera YAML file; returns the labels of the dropped text fields.

    A text field that holds only blanks and a line break is dropped in place
    (:func:`drop_blank_text`), with one warning (a debug line with ``quiet``).
    A file at ``path`` is replaced. A write that fails raises a
    :class:`MechanismError` and leaves no file at ``path``.
    """
    path = Path(path)
    dropped = drop_blank_text(gas, stage=stage)
    if dropped:
        log.log(logging.DEBUG if quiet else logging.WARNING,
                "%s: %d text field(s) hold only blanks and a line break, which Cantera cannot write; they are "
                "left out: %s%s", path.name, len(dropped), ", ".join(dropped[:5]),
                f" and {len(dropped) - 5} more" if len(dropped) > 5 else "")
    try:
        if path.is_file():
            path.unlink()                   # Cantera 3.2 reports no error when it cannot open the file
        gas.write_yaml(str(path))
    except (ct.CanteraError, IndexError, ValueError, OSError) as exc:
        if path.is_file():
            path.unlink()                   # a failed write can leave an empty file
        raise MechanismError(f"Cantera cannot write the {gas.n_species}-species mechanism to {path}: "
                             f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ''}",
                             stage=stage, path=str(path)) from exc
    if not path.is_file() or path.stat().st_size == 0:
        if path.is_file():
            path.unlink()
        raise MechanismError(f"Cantera wrote no file at {path}; the folder must exist and be writable, and the "
                             f"path must not be a folder.", stage=stage, path=str(path))
    return dropped
