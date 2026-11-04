"""Write a mechanism whose text fields hold only blanks and a line break (gasig.mechanism).

Cantera 3.2 stops in ``Solution.write_yaml`` on such a field with
``IndexError: basic_string::substr``. ``ck2yaml`` makes one from a CHEMKIN
comment that holds only blanks (``! `` after a reaction line).
"""
import logging

import cantera as ct
import numpy as np
import pytest

from gasig import mechanism
from gasig import reduce as reduce_mod
from gasig.errors import MechanismError
from gasig.reduce import load_mechanism, write_reduced_mechanism


def _with_notes(gas: ct.Solution) -> ct.Solution:
    """A copy of ``gas`` with blank notes on reactions 1 and 3 and on the thermo of H, and a real note on reaction 2.

    Reaction 1 also has a user field with text. The changed objects are new;
    every other object is shared with ``gas``.
    """
    reactions = gas.reactions()
    reactions[0] = ct.Reaction.from_dict({**reactions[0].input_data, "note": "\n", "source": "k1"}, gas)
    reactions[1] = ct.Reaction.from_dict({**reactions[1].input_data, "note": "[k74] from the source"}, gas)
    reactions[2] = ct.Reaction.from_dict({**reactions[2].input_data, "note": "\t\t\n"}, gas)
    species = gas.species()
    i_h = gas.species_index("H")
    data = species[i_h].input_data
    data["thermo"] = {**data["thermo"], "note": " \n"}
    species[i_h] = ct.Species.from_dict(data)
    return ct.Solution(thermo="ideal-gas", kinetics="gas", transport_model="mixture-averaged",
                       species=species, reactions=reactions)


def _rates(gas: ct.Solution, T: float, P: float) -> np.ndarray:
    gas.TPX = T, P, "CH4:0.05, O2:0.2, N2:0.7, H2O:0.05, H:1e-3, OH:1e-3"
    return np.concatenate([gas.forward_rate_constants, gas.reverse_rate_constants, gas.mix_diff_coeffs,
                           [gas.cp_mass]])


@pytest.fixture(scope="module")
def gas_full(mech_path):
    return ct.Solution(mech_path)


@pytest.fixture
def gas_notes(gas_full):
    return _with_notes(gas_full)        # a new copy per test: drop_blank_text changes it in place


def test_blank_fields_are_found_with_their_place(gas_notes):
    labels = mechanism.unwritable_fields(gas_notes)
    assert len(labels) == 3
    assert labels[0] == "species H/thermo/note"
    assert labels[1].startswith("reaction 1 (") and labels[1].endswith(")/note")
    assert labels[2].startswith("reaction 3 (")
    assert mechanism.blank_text_fields({"a": ["x", "\n"], "b": {"c": " \n "}, "d": "", "e": " "}) == ["/a[1]", "/b/c"]


def test_a_writable_mechanism_is_written_as_it_is(gas_full, tmp_path):
    assert mechanism.write_yaml(gas_full, tmp_path / "full.yaml") == []
    assert ct.Solution(str(tmp_path / "full.yaml")).n_reactions == gas_full.n_reactions


def test_write_yaml_drops_the_blank_fields_in_place_and_keeps_the_kinetics(gas_notes, tmp_path, caplog):
    states = ((900.0, ct.one_atm), (1600.0, 20 * ct.one_atm))
    before = [_rates(gas_notes, T, P) for T, P in states]
    path = tmp_path / "notes.yaml"
    with caplog.at_level(logging.WARNING, logger="gasig.mechanism"):
        dropped = mechanism.write_yaml(gas_notes, path)
    assert len(dropped) == 3 and "3 text field(s)" in caplog.text
    assert mechanism.unwritable_fields(gas_notes) == []                     # in place
    assert gas_notes.reaction(0).input_data["source"] == "k1"              # other user data stays
    assert gas_notes.reaction(1).input_data["note"] == "[k74] from the source"
    back = ct.Solution(str(path))
    assert mechanism.unwritable_fields(back) == []
    assert back.species_names == gas_notes.species_names and back.n_reactions == gas_notes.n_reactions
    assert "note" not in back.reaction(0).input_data and back.reaction(0).input_data["source"] == "k1"
    assert back.reaction(1).input_data["note"] == "[k74] from the source"
    assert "note" not in back.species("H").input_data["thermo"]
    for (T, P), ref in zip(states, before):
        np.testing.assert_array_equal(_rates(gas_notes, T, P), ref)         # the objects did not change
        np.testing.assert_allclose(_rates(back, T, P), ref, rtol=1e-12)


def test_elements_header_and_phase_stay(tmp_path):
    """A custom atomic weight, the header and the state survive; only the blank fields go."""
    gas = ct.Solution(yaml="""
units: {length: cm, quantity: mol, activation-energy: cal/mol}
elements:
- symbol: Ar
  atomic-weight: 40.5
phases:
- name: gas
  thermo: ideal-gas
  elements: [O, H, Ar]
  species: [{h2o2.yaml/species: [H2, H, O, O2, OH, H2O, HO2, H2O2, AR]}]
  kinetics: gas
  skip-undeclared-third-bodies: true
  reactions: [{h2o2.yaml/reactions: declared-species}]
  state: {T: 300.0, P: 1 atm}
""")
    gas.update_user_header({"description": "\n", "origin": "test"})
    gas.reaction(0).update_user_data({"note": "\n"})
    gas.TPX = 1234.0, 2 * ct.one_atm, "H2:2, O2:1, AR:5"
    dropped = mechanism.write_yaml(gas, tmp_path / "ar.yaml")
    assert dropped == ["header/description", f"reaction 1 ({gas.reaction(0).equation})/note"]
    assert gas.input_header.get("origin") == "test" and "description" not in gas.input_header
    back = ct.Solution(str(tmp_path / "ar.yaml"))
    assert back.atomic_weight("Ar") == pytest.approx(40.5)
    assert back.molecular_weights[back.species_index("AR")] == pytest.approx(40.5)
    assert back.T == pytest.approx(1234.0)


def test_reduced_mechanism_with_blank_notes_is_written(gas_notes, tmp_path):
    keep = gas_notes.species_names[:40]
    path = write_reduced_mechanism(gas_notes, keep, tmp_path / "reduced_40sp.yaml", stage="reduce")
    back = ct.Solution(str(path))
    assert back.n_species == 40 and mechanism.unwritable_fields(back) == []
    assert mechanism.unwritable_fields(gas_notes) == []                     # the objects are shared


def test_a_path_without_a_written_file_is_a_mechanism_error(gas_full, tmp_path):
    with pytest.raises(MechanismError) as info:
        mechanism.write_yaml(gas_full, tmp_path, stage="reduce")            # the path is a folder
    assert info.value.stage == "reduce" and info.value.exit_code == 3
    assert "wrote no file" in str(info.value)


def test_an_old_file_is_replaced(gas_full, tmp_path):
    path = tmp_path / "reduced_9sp.yaml"
    path.write_text("STALE")
    mechanism.write_yaml(gas_full, path)
    assert ct.Solution(str(path)).n_species == gas_full.n_species


def test_a_cantera_write_exception_is_a_mechanism_error_and_leaves_no_file(tmp_path, monkeypatch):
    class Broken:
        n_species = 3

        def write_yaml(self, path):
            open(path, "w").close()                                         # Cantera leaves an empty file
            raise IndexError("basic_string::substr: __pos (which is 18446744073709551615) > this->size() (which is 1)")

    monkeypatch.setattr(mechanism, "drop_blank_text", lambda gas, stage=None: [])
    path = tmp_path / "x.yaml"
    with pytest.raises(MechanismError, match="IndexError: basic_string::substr") as info:
        mechanism.write_yaml(Broken(), path, stage="report")
    assert info.value.stage == "report" and isinstance(info.value.__cause__, IndexError)
    assert not path.exists()


def test_ck2yaml_blank_comment_round_trip(gas_full, tmp_path, caplog):
    """The chain of the 313-species case: CHEMKIN with ``! `` and ``!<tab>`` comments, ck2yaml, load, write."""
    from cantera import ck2yaml, yaml2ck

    names = set(gas_full.species_names[:12])
    small = ct.Solution(thermo="ideal-gas", kinetics="gas", transport_model="mixture-averaged",
                        species=gas_full.species()[:12],
                        reactions=[rx for rx in gas_full.reactions() if set(rx.reactants) | set(rx.products) <= names])
    ck = tmp_path / "ck"
    ck.mkdir()
    yaml2ck.convert(small, mechanism_path=ck / "mech.inp", thermo_path=ck / "therm.dat",
                    transport_path=ck / "tran.dat", overwrite=True)
    lines = (ck / "mech.inp").read_text().splitlines()
    rows = [i for i, ln in enumerate(lines) if "=" in ln and not ln.lstrip().startswith("!")]
    first, second = rows[0], next(i for i in rows if i > rows[0] + 1)
    lines[first] = lines[first].split("!")[0].rstrip() + "   ! "
    lines[second] = lines[second].split("!")[0].rstrip() + "   !\t\t"
    (ck / "mech.inp").write_text("\n".join(lines) + "\n")
    out = tmp_path / "from_ck.yaml"
    ck2yaml.convert(input_file=ck / "mech.inp", thermo_file=ck / "therm.dat", transport_file=ck / "tran.dat",
                    out_name=out, quiet=True)
    raw = ct.Solution(str(out))
    assert len(mechanism.unwritable_fields(raw)) == 2                       # ck2yaml made two notes "\n"
    with caplog.at_level(logging.WARNING, logger="gasig.reduce"):
        gas = load_mechanism(str(out), stage="reduce")
    assert "2 text field(s)" in caplog.text
    assert mechanism.unwritable_fields(gas) == []                           # load_mechanism drops them
    assert mechanism.write_yaml(gas, tmp_path / "rewritten.yaml") == []
    assert ct.Solution(str(tmp_path / "rewritten.yaml")).n_reactions == gas.n_reactions


def test_zerork_conversion_takes_blank_notes_quietly(gas_notes, tmp_path, caplog):
    from gasig.solvers import zerork

    zerork._CHEMKIN_CACHE.clear()                                           # no cache hit from another test
    with caplog.at_level(logging.DEBUG, logger="gasig.mechanism"):
        files = zerork.chemkin_files(gas_notes, tmp_path / "ck_notes")
    assert files.mechanism.is_file() and files.thermo.is_file() and files.transport.is_file()
    records = [r for r in caplog.records if r.name == "gasig.mechanism" and "text field(s)" in r.getMessage()]
    assert len(records) == 1 and records[0].levelno == logging.DEBUG       # quiet: a sweep converts many mechanisms


def test_a_mechanism_error_in_a_flame_stops_the_sweep(gas_full, monkeypatch):
    def fail(*args, **kwargs):
        raise MechanismError("cannot write", stage="reduce")

    monkeypatch.setattr(reduce_mod, "run_flame", fail)
    with pytest.raises(MechanismError):
        reduce_mod._fs_values(gas_full, [(ct.one_atm, 1.0)], {}, "t")      # not a failed condition (inf)
