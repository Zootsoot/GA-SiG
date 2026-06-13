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
| Cantera            | 3.2.0   |
| PyTorch            | 2.8.0+cu126 |

### Operation 
The code GA-SiG.py is used to obtain the interaction coefficient between any two species. 

The code graph_search_to_importance.py evaluates the species importance indices via the r_AB given by GA-SiG.py. 
