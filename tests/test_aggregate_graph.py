import numpy as np
import pytest

from gasig.aggregate import aggregate, normalize_rows, species_coupling
from gasig.graph import combine_matrices, overall_importance, rank_species, resolve_targets

# Two species, three reactions: species 0 takes part in reactions 0 and 2, species 1 in 1 and 2.
S = np.array([[1.0, -2.0, 0.5],
              [0.0, 3.0, -1.0]])
NU = np.array([[1.0, 0.0, -1.0],
               [0.0, 1.0, 1.0]])


def test_species_coupling_abs_and_signed():
    S_AB = species_coupling(S, NU, abs_S=True)
    assert S_AB == pytest.approx(np.array([[1.5, 2.5], [1.0, 4.0]]))
    S_AB_signed = species_coupling(S, NU, abs_S=False)
    assert S_AB_signed == pytest.approx(np.array([[1.5, -1.5], [-1.0, 2.0]]))


def test_aggregate_manuscript_definition_rows_sum_to_one():
    agg = aggregate([S, 0.5 * S], NU)
    assert agg.n_trajectories == 2
    assert agg.S_AB == pytest.approx(np.array([[1.5, 2.5], [1.0, 4.0]]))
    assert agg.r_AB.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert agg.r_AB == pytest.approx(np.array([[1.5 / 4.0, 2.5 / 4.0], [1.0 / 5.0, 4.0 / 5.0]]))


def test_aggregate_published_normalization():
    agg = aggregate([S], NU, normalization="S_row_sum")
    row_sum_abs_S = np.abs(S).sum(axis=1)          # [3.5, 4.0]
    expected = np.array([[1.5, 2.5], [1.0, 4.0]]) / row_sum_abs_S[:, None]
    assert agg.r_AB == pytest.approx(expected)
    assert np.all(agg.r_AB <= 1.0 + 1e-12)
    assert agg.r_AB.sum(axis=1)[1] > 1.0        # rows are not stochastic in this form


def test_normalize_rows_zero_rows_stay_zero():
    r = normalize_rows(np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([0.0, 4.0]))
    assert r == pytest.approx(np.array([[0.0, 0.0], [0.5, 0.5]]))


def test_aggregate_rejects_bad_input():
    with pytest.raises(ValueError):
        aggregate([], NU)
    with pytest.raises(ValueError):
        aggregate([S], NU, normalization="other")
    with pytest.raises(ValueError):
        species_coupling(S, NU[:, :2])


def test_overall_importance_shortest_paths():
    R = np.array([[0.0, 0.5, 0.1, 0.0],
                  [0.0, 0.0, 0.8, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])
    imp = overall_importance(R, [0])
    assert imp[0] == pytest.approx(1.0)
    assert imp[1] == pytest.approx(0.5)
    assert imp[2] == pytest.approx(0.5 * 0.8)      # the two-hop path beats the direct 0.1 edge
    assert imp[3] == 0.0                           # unreachable
    imp2 = overall_importance(R, [0, 3])
    assert imp2[3] == pytest.approx(1.0)
    assert overall_importance(R, []).sum() == 0.0


def test_combine_matrices_modes():
    a = np.eye(2)
    b = np.ones((2, 2))
    R, used = combine_matrices(a, b, "auto", 0.5)
    assert used == "hybrid" and R == pytest.approx(0.5 * a + 0.5 * b)
    R, used = combine_matrices(a, None, "auto", 0.5)
    assert used == "idt" and R == pytest.approx(a)
    R, used = combine_matrices(a, b, "hybrid", 0.25)
    assert R == pytest.approx(0.25 * a + 0.75 * b)
    with pytest.raises(ValueError):
        combine_matrices(None, b, "idt", 0.5)
    with pytest.raises(ValueError):
        combine_matrices(a, None, "hybrid", 0.5)
    with pytest.raises(ValueError, match="same mechanism"):
        combine_matrices(np.eye(2), np.eye(3), "hybrid", 0.5)
    with pytest.raises(ValueError, match="square"):
        combine_matrices(np.ones((2, 3)), None, "idt", 0.5)


def test_signed_sum_with_row_normalization_warns(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="gasig.aggregate"):
        agg = aggregate([S], NU, abs_S=False, normalization="S_AB_row_sum")
    assert "signed row sums" in caplog.text
    assert np.all(np.isfinite(agg.r_AB))


def test_rank_species_and_targets():
    ranked = rank_species(["A", "B", "C"], np.array([0.2, 0.0, 1.0]))
    assert [n for n, _ in ranked] == ["B", "A", "C"]
    assert resolve_targets(["A", "B"], ["B", "ZZ", "A"]) == [1, 0]


def test_graph_stage_reproduces_the_shipped_example(example_dir):
    """The ranking of the repo's example files is recomputed exactly by the new graph module."""
    ct = pytest.importorskip("cantera")
    r_AB = np.load(example_dir / "r_AB.npy")
    ranked_ref = np.load(example_dir / "ranked_sp.npy")
    gas = ct.Solution(str(example_dir / "USCMechII.yaml"))
    names = list(gas.species_names)
    targets = resolve_targets(names, ["C2H4", "O2", "OH", "H2O", "HO2", "CO2"])
    importance = overall_importance(r_AB, targets, eps_edge=1e-16)
    ref = {row[0]: float(row[1]) for row in ranked_ref}
    for i, name in enumerate(names):
        assert importance[i] == pytest.approx(ref[name], rel=1e-9, abs=1e-300), name
    ranked = rank_species(names, importance)
    assert [n for n, _ in ranked] == [row[0] for row in ranked_ref]
