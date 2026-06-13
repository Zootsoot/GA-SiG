# GA-SiG
This is the source code of GPU-Aided Sensitivity-in-Graph (GA-SiG)

## Introduction to this approach 
### Design 
This tool package aims at utilizing the ability of sensitivity analysis to reduce as much redundant information as possible. 
The sensitivity index, specified as $\partial A/\partial \theta$, where $A$ and $\theta$ refer to an integrated species source term and a perturbation parameter, respectively, is used to evaluate how important a species is under a given set of conditions. 
### Computational environment
GA-SiG was constructed and tested under the following software environment:

| Package / Software | Version |
|--------------------|---------|
| Python             | 3.11.9  |
| NumPy              | 1.26.4  |
| Cantera            | 3.2.0   |
| PyTorch            | 2.8.0+cu126 |

### Operation 
1. The code GA-SiG.py is used to obtain the interaction coefficient between any two species. Multiple trjectories (N_traj) with various time steps (Nt) could be considered together to obtain an overall sensitivity-based adjacency matrix. 
The required input files include:
-Detailed kinetic mechanism file: Cantera format (.yaml) is recommended. The number of species (Ns) and reactions (Nr) contained will be used to check the structures of the following inputs. 
-Trajectories of temperatures and pressures: two dictionaries (.npy) structured as {N_traj: (Nt,)}. 
-Trajectories of species mass fraction: a dictionary (.npy) structured as {N_traj: (Nt, Ns)}. 
-Trajectories of time stepping: a dictionary (.npy) structured as {N_traj: (Nt,)}.
-Termination of integration for each trajectory: an array (.npy) structured as (N_traj).
In the code example, all the inputs above are obtained and structured using Cantera's Python version. 

2. The code graph_search_to_importance.py evaluates the species importance indices via the r_AB given by GA-SiG.py. 
The required input files include:
-Detailed kinetic mechanism file: Cantera format (.yaml).
-The sensitivity-based adjacency matrix r_AB (.npy) obtained from step 1. 
