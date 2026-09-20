# AdaFace Four-Condition Pilot Analysis Report

## Methodology
Four experiments (A=Clean/High, B=Clean/Low, C=Noisy/High, D=Noisy/Low) were evaluated using exact matching datasets. Data integrity passed all checks (83k training rows, 1328 probe rows).

## Key Summary Table
| Metric | A | B | C | D | A-B | C-D | Interaction |
|--------|---|---|---|---|-----|-----|-------------|
| raw_norm | 22.5339 | 22.5296 | 22.5418 | 22.5130 | 0.0042 | 0.0288 | 0.0246 |
| adaface_quality | 0.1246 | 0.0266 | 0.1357 | 0.0177 | 0.0980 | 0.1180 | 0.0199 |
| difficulty | 0.0227 | 0.3067 | 0.1755 | 0.6670 | -0.2839 | -0.4915 | -0.2076 |
| per_sample_loss | 0.0293 | 0.8322 | 0.3931 | 3.6991 | -0.8029 | -3.3060 | -2.5031 |
| head_gradient_norm | 0.5039 | 7.6109 | 4.1185 | 17.1485 | -7.1070 | -13.0300 | -5.9230 |

*(Please see CSV artifacts for complete data, distributions, and intermediate epochs.)*

## Factual Conclusion
The data has been successfully processed and plotted. We observed strong quality effects and a distinct gradient response pattern to identity noise. We do not claim population robustness or extrapolate the mechanism, as per the strict analytical constraints.
