> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Learned inference-safe common-VA correction

REACT2025 full VAL571, all-session processed-GT extension. The corrector was fitted only on TRAIN1660 labels; no listener GT, T0 model, metric, oracle, or tuned threshold is available at inference.

| Policy | FRC ↑ | exact FRD ↓ | FRDiv | FRVar | Raw-GT FRC ↑ | Gate rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1.072308 | 96.808993 | 0.205423 | 0.068098 | 1.033339 | 0.000000 |
| ungated | 1.074465 | 96.503191 | 0.205423 | 0.068120 | 1.035171 | 1.000000 |
| gate050 | 1.073816 | 96.518769 | 0.205423 | 0.068109 | 1.034693 | 0.322242 |
| gate080 **(primary)** | 1.073422 | 96.569532 | 0.205423 | 0.068104 | 1.034386 | 0.122592 |
| gate090 | 1.072766 | 96.686329 | 0.205423 | 0.068101 | 1.033772 | 0.049037 |

## Prespecified decision

Primary strict success: **True**. Delta FRC +0.001114; exact FRD -0.239462; FRDiv +0.000000000; raw-GT FRC +0.001047.

Threshold0.80 and all training hyperparameters were frozen before label generation. Other policies are reported diagnostics, not replacements chosen after observing VAL.

The primary gate applies on 70/571 contexts (12.26%). Its paired session-bootstrap 95% intervals exclude zero in the favorable direction: FRC `[+0.000427,+0.002139]`, exact FRD `[-0.569083,-0.081455]`, and raw-GT FRC `[+0.000443,+0.001893]`. FRC and FRD are jointly non-worse than baseline on 97.02% of all contexts. Against the post-hoc VAL oracle gate (diagnostic only), the fixed0.80 gate has 85.71% precision and 22.99% recall.

Ungated correction has the largest mean gains (`ΔFRC +0.002157`, `ΔFRD -0.305803`), but only 51.14% of contexts are jointly non-worse. The conservative gate therefore trades gain magnitude for substantially better context-level safety.

## Scope

TRAIN finite-oracle labels: `{"gate_positive_fraction": 0.36927711963653564, "mean_FRC_delta": 0.017728941005727607, "mean_exact_FRD_delta": -0.32041714372752483, "selection_counts": {"baseline": 1047, "oracle_025": 425, "oracle_050": 143, "oracle_100": 4, "quality_centroid": 41}}`.

Corrector parameters: 111,933; training time 2.01min; cached-input latency 2.385ms/context. Pairwise geometry maximum numerical change 3.32e-08; AU/expression are unchanged and FRDiv is structurally invariant.

Checkpoint SHA256: `0ab2a415aeb564aadbeb4057a0b23816c46414155bbe297a35340b6ee5aee100`. Commit at execution: `c421f509f8cb7997e9341592cae2eb1e721b0e8e`.

This one-seed VAL result is exploratory, not an independent TEST claim. Full context ledgers, per-session metrics and session-bootstrap intervals are retained in the local artifact.

This correction is a teacher-quality calibration result, not a diversity mechanism: preserving FRDiv exactly is intentional. It can be composed with the diverse teacher ensemble, but it cannot by itself close the GT diversity ceiling.
