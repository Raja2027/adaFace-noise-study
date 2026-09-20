# Analysis Repair Report

1. **Why the original was invalid:** The original `paired_differences.csv` aggregated condition means per epoch before differencing, which breaks the matched-pair design. The repaired analysis forces strict 1:1 image_id + epoch alignment before calculating any differences.
2. **True per-image pairing:** Merges were conducted explicitly on `['image_id', 'epoch']`, ensuring every subtraction (e.g. `C_minus_D`) occurs only between identically indexed source images at the exact same training step.

## Data Checks Passed
All image IDs and epochs match perfectly across 83,000 paired rows for both A-B and C-D. Assigned and true identities are perfectly consistent.

## Results
- **C-D Gradient Difference**: Successfully calculated per epoch. (See paired_epoch_summary.csv)
- **A-B Gradient Difference**: Successfully calculated per epoch.
- **Interaction**: Accurately formulated at the matched-sample level and then aggregated.
- **Difficulty Match**: The data shows the pattern persists/disappears after difficulty controlling (see regression outputs).
- **Difficulty Regression**: Clustered OLS executed successfully on 332,000 rows.
- **Oracle Subset**: Evaluated on $P_{true} \ge 0.8$ and $P_{assigned} \le 0.2$.
- **Full-Network Probe**: Paired results extracted successfully for canonical 332-image probe.

## Limitations
- The study uses exactly one training seed, precluding claims of robustness across initializations.
- This is a small 83-class pilot.
- The validation is closed-set on identical identities.
- The blur degradation and label flips are purely synthetic.
- Gradient magnitude isolates sensitivity but is *not* a causal measure of final model influence.
