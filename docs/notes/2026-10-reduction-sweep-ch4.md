# Reduction sweep on methane: rate laws and normalizations

Date: 2026-10-23. Author: Claude for Yun-Jui Wu. Status: final.

## Purpose

This note records the first performance test of the four method versions of GA-SiG 0.2 on methane with USC Mech II. The test answers two questions. Does the corrected rate law change the reduction performance? Does the normalization change it?

## Facts

- Case: USC Mech II (111 species, 784 reactions), methane and air. The configurations are in `example/sweeps/`; the file names give the rate law and the normalization (`ch4_<original|corrected>_<SAB|Srow>.json`). The outputs are local run folders (`results_*/`, not in git); the tables below hold the numbers.
- Sensitivity data: 12 constant-volume ignition cases (1200, 1400 and 1600 K; 1 and 10 atm; phi 0.5 and 1.0) and 3 flames (300 K, 1 atm, phi 0.8, 1.0 and 1.2). The rank uses the hybrid matrix with the targets CH4, O2, OH, H2O and CO2.
- Evaluation: `gasig reduce` removes one species per step down to 15 species. It computes the maximum relative error of the ignition delay time (IDT) over the 12 cases. The flame speed is not evaluated.
- The four versions are the two rate laws (`original`, `corrected`) and the two normalizations (`S_AB_row_sum`, `S_row_sum`). The corrected law runs with the step-size guard (`substep_tol` 0.01, `max_substeps` 64).

Table 1. Smallest mechanism within an IDT error limit

| Version | Within 5 % | Within 10 % | Within 20 % | Reactions at the 5 % size |
|---|---|---|---|---|
| original, `S_AB_row_sum` | 27 species | 27 | 27 | 137 |
| original, `S_row_sum` | 26 species | 26 | 26 | 127 |
| corrected, `S_AB_row_sum` | 25 species | 25 | 25 | 132 |
| corrected, `S_row_sum` | 25 species | 25 | 25 | 132 |

Table 2. Maximum IDT error in percent against the number of species

| Species | original, `S_AB_row_sum` | original, `S_row_sum` | corrected, `S_AB_row_sum` | corrected, `S_row_sum` |
|---|---|---|---|---|
| 90 | 0.10 | 0.10 | 0.24 | 0.16 |
| 60 | 0.02 | 0.02 | 0.03 | 0.18 |
| 45 | 0.31 | 0.31 | 0.23 | 0.13 |
| 35 | 2.60 | 0.74 | 0.45 | 0.14 |
| 30 | 1.93 | 1.93 | 1.69 | 1.69 |
| 27 | 1.33 | 1.33 | 1.33 | 1.33 |
| 26 | 26.05 | 1.33 | 1.33 | 1.33 |
| 25 | 26.05 | 26.05 | 1.36 | 1.36 |
| 24 | 26.05 | 26.05 | 26.05 | 26.05 |
| 20 | 31.80 | 36.07 | 29.82 | 29.82 |
| 15 | no ignition in 3 cases | no ignition in 3 cases | 88.19 | 88.19 |

- Every curve breaks at the removal of C2H4: the error jumps from 1.3 % to 26 %. The versions differ in the species removed before C2H4, not in the break itself.
- Species kept at the 5 % size by the corrected law (25): N2, H, O, OH, HO2, H2, H2O, H2O2, O2, CH, CH2, CH2*, CH3, CH4, HCO, CH2O, CH3O, CO, CO2, C2H3, C2H4, C2H5, C2H6, CH3CHOCH2, CH3CH2CHO. The original law with `S_AB_row_sum` keeps C3H6 and CH3COCH3 in addition (27); with `S_row_sum` it keeps C3H6 and CH3COCH3 but not CH (26).
- Rank differences: the normalization changes the rank of the original law with a rank correlation of 1.000 (largest shift 4 places). For the corrected law the correlation is 0.996 (largest shift 10). The rate law changes the rank with a correlation of 0.90. 42 to 46 species move more than 10 places, and the 30 least important species agree in 21 to 23 cases. The corrected law puts HO2 above CH3 and brings H2O2 and CO into the 15 most important species.
- Cost: the corrected law with the guard needed 136838 solves for 3374 ignition steps and 23619 solves for 533 flame steps; the guard reached the cap of 64 sub-steps on 37 ignition steps. The original law needs one solve per step.
- The error values of up to 0.3 % above 37 species are solver noise. The IDT is the time of a solver sample. The step at ignition is up to 0.28 % of the IDT (1600 K, 1 atm, phi 0.5).
- The sweeps ran with the code before the review of 2026-10-23. The review changed the reactor path (clone of a shared Solution, composition set before the state). The ranks and the sweeps were not repeated after the review.

## Decision

On methane the corrected rate law gives a more compact mechanism within 5 % IDT error (25 species against 26 or 27). It also gives lower errors between 40 and 28 species. The normalization changes the result by at most one species. This is one fuel and one target property, so it does not decide the default. The default stays `original` until the four mechanisms of the paper are tested (milestone M5).

## Next steps

1. Run the same sweep on C2H4, n-C7H16, n-C10H22 and Gevo AtJ-SPK with the conditions of the paper (M5). Use a GPU with more than 4 GB for the AtJ mechanism.
2. Add the flame-speed error to the methane sweep with `reduction.evaluate_fs` on a longer run.
3. Raise `max_substeps` or lower `substep_tol` for the corrected law and check whether the 37 capped steps change the rank.
4. Interpolate the peak of the reference species to lower the IDT noise below 0.1 %.
