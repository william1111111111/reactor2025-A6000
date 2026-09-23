# REACT2024 inference-time spread scaling

This is a validation-only post-processing diagnostic for the frozen
Distance-KD student. It does not change training or checkpoints.

For each frame, let `mu` be the mean of the ten predicted slots. The scaled
prediction is `mu + alpha * (prediction - mu)`. AU, VA, and expression domains
are then projected exactly as in the existing evaluation path, followed by AU
rounding. One global `alpha` is used for every validation context; no
ground-truth-based per-context selection is performed.

Full official React2024 VAL (1,124 directions):

| alpha | FRC | S_MSE | FRVar | FRDvs |
|---:|---:|---:|---:|---:|
| 1.00 | 0.483342 | 0.107258 | 0.070031 | 0.159101 |
| 1.50 | 0.457099 | 0.148280 | 0.069689 | 0.158709 |
| 1.60 | 0.452758 | 0.153923 | 0.069628 | 0.158695 |
| 1.65 | 0.450797 | 0.156561 | 0.069607 | 0.158717 |
| 1.75 | 0.446792 | 0.161544 | 0.069584 | 0.158797 |
| 2.00 | 0.438160 | 0.172689 | 0.069606 | 0.159173 |

The coefficient raises within-set S_MSE, but trades away FRC. Around `alpha=1.6`
the student remains near the teacher's FRC while reaching S_MSE about `0.154`;
around `alpha=1.75`, S_MSE reaches about `0.162` but FRC falls below the
teacher. The effect is amplitude expansion of existing output differences, not
evidence that new behavioral modes were learned. `FRDvs` is the native
React2024 metric; this diagnostic did not compute temporal DTW-FRD.

The reported values are VAL diagnostics and must not be presented as held-out
TEST results.
