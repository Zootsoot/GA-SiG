import pathlib

import pytest

try:
    import cantera as _ct

    _ct.suppress_thermo_warnings()
except ImportError:  # pragma: no cover
    pass

REPO = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example"
MECH = EXAMPLE / "USCMechII.yaml"


@pytest.fixture(scope="session")
def mech_path() -> str:
    assert MECH.is_file(), f"missing test mechanism {MECH}"
    return str(MECH)


@pytest.fixture(scope="session")
def example_dir() -> pathlib.Path:
    return EXAMPLE


@pytest.fixture(scope="session")
def device() -> str:
    torch = pytest.importorskip("torch")
    return "cuda" if torch.cuda.is_available() else "cpu"
