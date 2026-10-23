"""Shared helpers of the reduce and report tests: rank files and configurations for USC Mech II."""
import csv
from pathlib import Path

from gasig.config import validate_config

TARGETS = ["CH4", "O2", "OH", "H2O", "CO2"]
# targets + fuel + oxidizer + reference species, with the default keep list
PROTECTED = {"CH4", "CO2", "H2O", "N2", "O2", "OH"}
IDT = {"fuel": {"CH4": 1.0}, "oxidizer": {"O2": 0.21, "N2": 0.79},
       "T_K": [1500.0], "P_atm": [1.0], "phi": [1.0], "end_time_s": 2.0e-3}


def rank_order(gas) -> list[str]:
    """Protected species first: an unprotected sweep would remove them in the first batch."""
    return ["N2", "OH", "O2", "CH4", "H2O", "CO2"] + [s for s in reversed(gas.species_names) if s not in PROTECTED]


def write_rank(folder: Path, order: list[str], importance=None) -> Path:
    """Write ``ranked_species.csv``; targets get importance 1, the others 0 unless ``importance`` is given."""
    path = folder / "ranked_species.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank_from_least_important", "species", "importance"])
        for i, s in enumerate(order, start=1):
            value = importance[s] if importance else (1.0 if s in TARGETS else 0.0)
            writer.writerow([i, s, value])
    return path


def make_config(tmp_path, mech_path, reduction=None, targets=TARGETS, fs=None, report=None, end_time_s=None) -> dict:
    solver = dict(IDT)
    if end_time_s is not None:
        solver["end_time_s"] = end_time_s
    raw = {"mechanism": mech_path, "output_dir": "out",
           "idt": {"enabled": True, "solver": solver},
           "ranking": {"target_species": list(targets)},
           "reduction": {"step": 50, "min_species": 61, **(reduction or {})}}
    if fs is not None:
        raw["fs"] = fs
    if report is not None:
        raw["report"] = report
    return validate_config(raw, tmp_path)
