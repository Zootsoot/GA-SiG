import cantera as ct
import numpy as np
import heapq
from typing import List, Dict, Tuple, Set, Optional, Sequence
ct.suppress_deprecation_warnings()
ct.suppress_thermo_warnings()

def overall_importance(
    R: np.ndarray,
    target_indices: List[int],
    eps_edge: float = 1e-16,
) -> np.ndarray:

    nsp = R.shape[0]
    importance = np.zeros(nsp, dtype=float)

    # Build adjacency lists with weights w_ij = -log(r_ij)
    adj = [[] for _ in range(nsp)]
    for i in range(nsp):
        for j in range(nsp):
            rij = R[i, j]
            if rij > eps_edge:
                w_ij = -np.log(rij)
                adj[i].append((j, w_ij))

    INF = 1e300

    for T in target_indices:
        # Dijkstra from target T
        dist = np.full(nsp, INF, dtype=float)
        dist[T] = 0.0
        heap = [(0.0, T)]

        while heap:
            d_u, u = heapq.heappop(heap)
            if d_u > dist[u]:
                continue
            for v, w_uv in adj[u]:
                nd = d_u + w_uv
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        # Convert distances to path coefficients R_TA
        with np.errstate(over="ignore", under="ignore"):
            R_TA = np.where(dist < INF, np.exp(-dist), 0.0)

        # Combine over targets: I_A = max_T R_TA
        importance = np.maximum(importance, R_TA)

    return importance

def sensitivity_mechanism_rank_from_rAB(
    mech_yaml: str,
    r_AB_global: np.ndarray,
    target_species: List[str]
):
    
    gas = ct.Solution(mech_yaml)
    species_names = gas.species_names
    nsp = gas.n_species

    r_AB_global = np.asarray(r_AB_global, dtype=float)
    if r_AB_global.shape != (nsp, nsp):
        raise ValueError(
            f"r_AB_global shape {r_AB_global.shape} does not match n_species={nsp}."
        )

    # Map target species to indices
    target_indices = []
    for name in target_species:
        if name in species_names:
            target_indices.append(gas.species_index(name))
        else:
            print(f"[WARN] Target species '{name}' not in mechanism; ignoring.")

    print("=== Sensitivity-based r_AB ranking ===")
    print(f"Mechanism: {mech_yaml}")
    print(f"Number of species: {nsp}")
    print(f"Targets: {target_species}")
    print(f"Valid target indices: {target_indices}")

    if not target_indices:
        importance = np.zeros(nsp, dtype=float)
    else:
        importance = overall_importance(r_AB_global, target_indices)

    ranked_species = [
        (name, float(importance[i]))
        for i, name in enumerate(species_names)
    ]
    ranked_species.sort(key=lambda x: x[1])  # least -> most important

    print("\nTop 15 most important species (by sensitivity-based importance):")
    for name, val in ranked_species[-15:]:
        print(f"  {name:15s}  I = {val:.4e}")

    return gas, species_names, importance, ranked_species

if __name__ == '__main__': 

    mech = "YOUR MECHANISM.yaml" # Cantera input file 
    r_AB_global = np.load('YOUR r_AB MATRIX.npy') # python array (N_s, N_s) 
    target_species = ["YOUR TARGET SPECIES"] # e.g. ['H2O', 'CO2', 'CO']

    # === Sensitivity-based ranking ===
    gas_sens, species_names_sens, global_imp_sens, ranked_species_sens = \
        sensitivity_mechanism_rank_from_rAB(
            mech_yaml=mech, 
            r_AB_global=r_AB_global,
            target_species=target_species
        )
