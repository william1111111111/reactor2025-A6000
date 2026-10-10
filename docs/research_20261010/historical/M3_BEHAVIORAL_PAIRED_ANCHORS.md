> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Ten anchors with representative behavioral targets

Status: all ten 50-epoch models and deterministic full VAL571 evaluation complete.

## FRC-only checkpoint diagnostic

Same deterministic VAL64 subset (seed1234), center750 source, paired+9 GT,
official AU rounding and seeded official Processor at every checkpoint.
No FRD was computed. Raw per-context/per-rank results:
`reports/anchor_behavioral_pairs_full50_seed1/frc_curve_val64.json`.

| epoch | 10 | 20 | 30 | 40 | 50 |
|---|---:|---:|---:|---:|---:|
| ensemble FRC | 0.774278 | 0.779779 | 0.773807 | 0.750112 | 0.738111 |
| rank0 contribution | 0.094097 | 0.095436 | 0.095196 | 0.088800 | 0.087340 |

All ten individual models have lower VAL64 FRC at epoch50 than epoch20.
The aggregate drops 0.041668 (5.34%) between those checkpoints despite rising
training CCC. This does not support simply extending training to improve FRC;
it suggests a generalization/objective mismatch or overfitting, not a proven
causal diagnosis. Epoch20 is best among the five tested checkpoints on VAL64,
not yet established as best on full VAL571 or an independent test set. These
VAL64 values must not be compared directly with full VAL571 means above.

The TRAIN dataset contains 3,320 source records including both dyadic directions.
Every source retains its same-name opposite-role paired GT in position zero.
Positions 1–9 are distinct opposite-role GTs from the same session, the legal
candidate pool defined by ReactionDataset.

Nine global KMeans prototypes are fitted to standardized 602D descriptors of
3,320 TRAIN trajectories only, with seed 1 and 20 initializations. Mode IDs are
canonicalized by their nearest real trajectory indices. For each context,
minimum-cost bipartite assignment selects nine distinct alternatives for the
nine prototypes. Cost is mean squared descriptor distance plus 0.1 times local
five-neighbor distance, favoring supported representatives over isolated GTs.
There is no equal-capacity constraint and no VAL clustering or target selection.

The selected GT's nearest prototype agrees with its assigned model role in
65.58% of assignments. This is a soft role assignment: not every session supports
all nine global behaviors, and distinct-target constraints can require fallback
representatives. Dataset membership establishes eligibility, not independently
verified psychological appropriateness. Selection does not guarantee an increase
in official FRDiv after training.

All ten models are freshly trained using the existing Mam ConditionalREGNN
configuration: 3,720,129 parameters each, seed 1, 50 epochs, batch32, BF16,
AdamW lr1e-4/wd1e-4, and MSE + (1-CCC) + 0.05 velocity loss. Source/crop schedule
and diffusion target crop/tail-fill behavior are unchanged. Target mappings stay
fixed throughout training. Manifest SHA256 is saved with every checkpoint.

Local artifacts are under
`coreset_reactor/reports/anchor_behavioral_pairs_full50_seed1/`:
`targets.json`, source/data provenance, per-model logs and checkpoints, and
`full_val571.json` / `full_val571.md`.
GPU4 rejected CUDA allocation; both queues run on GPU5, with recovery provenance
recording the change. The original odd-rank queue continues independently.

Full VAL571 uses the identical paired-plus-nine deterministic evaluation GTs and
exact rolling FRD as the matched random-fixed-target experiment.

| method | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | TLCC | GT coverage ↓ |
|---|---:|---:|---:|---:|---:|---:|
| random fixed nine | 0.806542 | 95.141523 | 0.066710 | 0.054319 | 49.0 | 1.147389 |
| behavioral representative nine | 0.780959 | 94.669544 | 0.118325 | 0.045690 | 49.0 | 1.095427 |
| frozen M1 | 0.810111 | 82.480135 | 0.013611 | 0.039200 | n/a | 1.095160 |

Versus random fixed targets, behavioral selection changes FRC by -0.025583,
exact FRD by -0.471979, FRDiv by +0.051615 (1.77x), FRVar by -0.008629,
and GT coverage by -0.051962. Candidate utilization rises from 0.533800 to
0.722592 and prediction validity improves from 1.008993 to 0.972053.

Behavioral target selection therefore produces substantially broader and better
covered outputs than random fixed targets. It does not preserve quality: FRC is
lower than both references, and exact FRD remains 12.189409 worse than M1. This
is a diversity frontier result rather than a quality-safe operating point.

TLCC is retained with the existing first-prediction limitation. Data, target
manifests and checkpoints remain local.

Validation before launch: four paired-target/crop tests passed, including
manifest duplicate rejection; both first GPU jobs produced finite losses.
