# Paper Claim Audit Report

**Date**: 2026-09-02  
**Auditor**: GPT-5.5 xhigh, fresh zero-context reviewer  
**Input**: `PROGRESSIVE_ABLATION_CLAIMS_INPUT_20260902.tex`

## Overall Verdict: FAIL

The Mam-Reactor endpoint and nearly all ablation metrics can be traced to raw
full-test files. The table is not publication-ready because it contains two
numeric errors, presents incompatible training protocols as one progressive
ablation, and marks VA refinement absent in three runs where it was active.
The Huang et al. and MReactor comparisons were not backed by raw evidence in
the declared audit set.

## Claim Counts

| Status | Count |
|---|---:|
| exact match | 6 |
| valid rounding | 29 |
| missing evidence | 6 |
| config mismatch | 6 |
| number mismatch | 2 |
| scope overclaim | 1 |
| ambiguous mapping | 1 |
| **Total** | **51** |

## Numeric Provenance

| Row | Raw FRC | Raw FRD | Raw FRDiv | Raw FRVar | Display verdict |
|---|---:|---:|---:|---:|---|
| without-P0 P1 | 0.1388087592 | 129.4226295430 | **0.0019831082** | 0.0055727721 | FRDiv must be **0.0020**, not 0.0198 |
| without-P0 P2 | 0.1111104322 | 129.8669736312 | 0.0147709483 | 0.0044198534 | valid rounding |
| without-P0 P3 | 0.1586350596 | 131.0624727803 | 0.0088274144 | 0.0256684758 | valid rounding |
| P0 Anchor | 1.0816580328 | 131.8084665318 | 0.0000000000 | 0.0562575646 | valid rounding |
| with-P0 P1 | 0.9565051233 | 141.0981427742 | 0.0420465022 | 0.0547948778 | valid rounding |
| with-P0 P2 | 0.8063625821 | 187.3755268050 | 0.1560553163 | 0.0588330217 | valid rounding |
| with-P0 P3 | 0.8525824264 | 154.3268892112 | 0.1530033201 | 0.0600871146 | valid rounding |
| with-P0 P4 | 0.8539534367 | 151.6207084705 | 0.1521245241 | **0.0600496866** | FRVar rounds to **0.0600**, not 0.0601 |

## Material Issues

### FAIL: with-P0 P2 to P3 is not a controlled transition

The P1/P2 branch uses 30 epochs, learning rate `2e-4`, 3,320 training
examples, no emotion-query warm-start, `target_support` compatibility and a
uniform mixture. The P3/P4 branch uses 3 epochs, learning rate `2e-6`, 2,146
`B_fit` examples, an emotion-query warm-start, `legacy_relative_softmax` and
a predicted-support mixture, with several additional objectives. Therefore
the change from `0.8064/187.38` to `0.8526/154.33` cannot be attributed only
to adding set alignment.

### FAIL: no-anchor rows contain VA refinement

All three no-anchor configs set `isolated_va_calibration=true`; their epoch-3
training records show `isolated_va_epoch_active=true` and 134 VA refinement
steps. Their VA column cannot be marked absent. The block should also be
called “without Stage-1 anchor pretraining,” because it still contains a
randomly initialized, jointly trained anchor and an anchor objective.

### WARN/FAIL: main-result baseline evidence is incomplete

Mam-Reactor is fully supported by the raw endpoint
`FRC=0.8539534367`, `FRD=151.6207084705`, `FRDiv=0.1521245241`, and
`FRVar=0.0600496866`. No raw Huang et al. evidence, MReactor evidence, or
complete main-result table was included in the audit inputs, so “strongest,”
the Huang comparison, and the MReactor comparison are not independently
verified by this audit.

## Required Corrections

1. Replace no-anchor P1 FRDiv `0.0198` with `0.0020`.
2. Replace P4 FRVar `0.0601` with `0.0600` under standard rounding.
3. Do not use the current P2 and P3 as a one-factor set-alignment ablation.
4. Either mark VA refinement enabled in all no-anchor rows or rerun them with
   isolated VA disabled.
5. Rename the group to “without Stage-1 anchor pretraining.”
6. Attach raw/protocol-matched evidence for Huang et al. and MReactor before
   retaining the main-result comparison language.
7. Qualify “without collapsing” unless a comparison threshold or statistical
   criterion is stated.

The authoritative machine-readable artifact is
`paper/PAPER_CLAIM_AUDIT.json`; the forensic reviewer trace is under
`.aris/traces/paper-claim-audit/2026-09-02_run01/`.
