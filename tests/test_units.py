"""Chain-unit detection: the link rule on a synthetic extent matrix and on a methane trajectory."""
import numpy as np
import pytest

ct = pytest.importorskip("cantera")

from helpers_reduction import IDT, PROTECTED  # noqa: E402
from gasig.config import ConfigError, validate_config  # noqa: E402
from gasig.units import chain_links, detect_units, join_links, reaction_extents  # noqa: E402

NAMES = ["a", "b", "c", "d", "e", "f"]


def synthetic_X():
    """Species change per reaction (rows: species, columns: reactions r0 to r6).

    a -> b -> c is a chain: r0 makes a, r1 turns a into b, r2 turns b into c, r3 uses c. a also gets 0.5 % of its
    production from r6, below the share. d has two significant producers. e is a link alone. f is never produced.
    """
    X = np.zeros((6, 7))
    X[0, 0], X[0, 6], X[0, 1] = 1.0, 0.005, -1.005
    X[1, 1], X[1, 2] = 1.0, -1.0
    X[2, 2], X[2, 3] = 1.0, -1.0
    X[3, 4], X[3, 5], X[3, 3] = 0.5, 0.5, -1.0
    X[4, 5], X[4, 6] = 1.0, -1.0
    X[5, 3] = -1.0
    return X


def test_links_need_one_significant_producer_and_one_consumer():
    links = chain_links(synthetic_X(), list(range(6)), 0.01)
    assert links == {0: (0, 1), 1: (1, 2), 2: (2, 3), 4: (5, 6)}
    assert 0 not in chain_links(synthetic_X(), list(range(6)), 0.001)      # r6 counts at a smaller share


def test_links_join_into_units_in_chain_order():
    links = chain_links(synthetic_X(), list(range(6)), 0.01)
    assert join_links(links, NAMES) == [[0, 1, 2]]                          # e alone is no unit
    shuffled = {2: links[2], 0: links[0], 1: links[1]}
    assert join_links(shuffled, NAMES) == [[0, 1, 2]]
    assert join_links(chain_links(synthetic_X(), [1, 2, 4], 0.01), NAMES) == [[1, 2]]   # candidates limit the chain


@pytest.fixture(scope="module")
def methane(mech_path):
    from gasig.solvers.idt import run_idt

    gas = ct.Solution(mech_path)
    cfg = dict(IDT, reactor="constant_volume", reference_species="OH", ignition_criterion="reference_peak",
               rtol=1e-6, atol=1e-12, max_steps=200000)
    traj = run_idt("", None, 1500.0, ct.one_atm, 1.0, cfg, label="idt-0", gas=gas)
    return gas, traj


def test_reaction_extents_close_the_species_balance(methane):
    gas, traj = methane
    xi = reaction_extents(gas, traj)
    nu = gas.product_stoich_coeffs - gas.reactant_stoich_coeffs
    W = gas.molecular_weights
    dn = traj.Y[traj.end_index] / W - traj.Y[0] / W
    sig = np.abs(dn) > 1e-3 * np.abs(dn).max()
    assert np.max(np.abs((nu @ xi)[sig] - dn[sig]) / np.abs(dn[sig])) < 0.05     # trapezoidal rule on the samples


def test_detected_units_on_methane_follow_the_rule(methane):
    gas, traj = methane
    names = gas.species_names
    removable = set(names) - PROTECTED
    res = detect_units(gas, [traj], removable, 0.01)
    assert res["scope"] == "all" and res["n_candidates"] == len(removable) and res["n_trajectories"] == 1
    members = [s for u in res["units"] for s in u]
    assert len(members) == len(set(members)) and set(members) <= removable
    assert all(len(u) >= 2 for u in res["units"]) and set(members) <= set(res["links"])
    for unit in res["units"]:
        for up, down in zip(unit, unit[1:]):
            assert res["links"][up]["consumer"] == res["links"][down]["producer"]
    importance = {s: float(i) for i, s in enumerate(names)}
    tail = detect_units(gas, [traj], removable, 0.01, importance)
    assert tail["scope"] == "tail" and tail["n_candidates"] < res["n_candidates"]
    assert set(s for u in tail["units"] for s in u) <= {s for s in removable if importance[s] < tail["tail_median_importance"]}


def test_units_config_checks(tmp_path, mech_path):
    base = {"mechanism": mech_path, "idt": {"enabled": True, "solver": dict(IDT)}}
    cfg = validate_config(base, tmp_path)
    assert cfg["units"] == {"detect": False, "scope": "all", "share": 0.01, "output_file": "chain_units.json"}
    assert cfg["reduction"]["search"]["units"] is None
    with pytest.raises(ConfigError, match="units.share"):
        validate_config({**base, "units": {"share": 1.5}}, tmp_path)
    with pytest.raises(ConfigError, match="idt mode"):
        validate_config({"mechanism": mech_path, "fs": {"enabled": True, "solver": {
            "fuel": {"CH4": 1.0}, "oxidizer": {"O2": 1.0}, "P_atm": [1.0], "phi": [1.0]}},
            "units": {"detect": True}}, tmp_path)
