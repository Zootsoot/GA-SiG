import pathlib
import sys

import pytest

try:
    import cantera as _ct

    _ct.suppress_thermo_warnings()
except ImportError:  # pragma: no cover
    pass

# The shared helpers (helpers_reduction.py) import as a plain module from this folder.
TESTS_DIR = pathlib.Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

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
