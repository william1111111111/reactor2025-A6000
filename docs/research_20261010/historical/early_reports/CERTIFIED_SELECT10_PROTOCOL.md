> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Frozen finite-pool Select-10: bound experiment

## Scope locked before results

- No new training or checkpoint modification. No GitHub upload.
- Baseline: the current **Basic-KD student**, not historical CoReSet M1.
- Primary pool R30: ten original outputs each from Basic-KD, Distance-KD and
  Centered-direction-KD, all seed1/10epoch students with 3,803,201 parameters.
- Secondary pool E60: R30 plus each model's scale1.5 residual-expanded outputs.
  All expansion is in soft trajectories before domain projection/AU rounding.
- Pool settings are fixed globally, not selected per context using GT.
- Exact duplicate official trajectories merge provenance, except the original
  baseline's ten instances are retained so it remains a feasible set even
  when baseline trajectories repeat. Actual distinct counts are recorded.
- Deterministic pilot: sorted NumPy default_rng(1234).choice(571,24,False).
  Selection is independent of metrics; these 24 are not claimed to be full VAL.
- Every selected context retains **all** its same-session opposite-role GTs,
  the same center750 lengths, postprocessor cache/seed and AU rounding as the
  prior cached full-VAL evaluations. Only source contexts are subsampled.
- Zero policy quality slack **per context**: FRC >= baseline, exact FRD <=
  baseline. This is stronger than a dataset-average quality constraint; the
  resulting upper bounds do not upper-bound that looser aggregate problem.

## Objective and analytical upper bound

Normalize each flattened official trajectory by sqrt(T*25) to obtain u_i.
Let G_ij = u_i^T u_j and n_i=||u_i||^2. With sum(z)=K=10:

    F(z) = (2K n^T z - 2 z^T G z) / (K(K-1)).

For integer selections F is exactly official FRDiv in real arithmetic. Its
Hessian is -4G/(K(K-1)), negative semidefinite. The box relaxation with fixed
cardinality and affine quality constraints is therefore a convex optimization
problem (maximizing a concave objective).

Crucially, an approximate feasible solution of this relaxation is NOT an
upper bound on its maximum. We do not report it as one.

For any tangent point x, concavity gives F(z) <= F(x)+g^T(z-x), g=grad F(x).
Write scaled quality constraints A z <= b and cardinality 1^T z=K. For ANY
lambda>=0 and any scalar nu, the following quantity is an upper bound:

    U = F(x)-g^T x + lambda^T b + nu K
        + sum_i max(0, g_i-(A^T lambda)_i-nu).

SLSQP supplies useful tangent points, not a certificate. HiGHS LP supplies
candidate dual multipliers; signs are projected to lambda>=0 and the above
expression is independently recomputed. A cardinality-only tangent upper
bound is always available if LP fails. We take the smallest valid bound.

This implementation is FLOAT64 with a recorded allowance of
1e-8*max(1,abs(U)). It is an analytical certificate evaluated numerically,
checked against brute-force instances and actual official metrics; it is not
a formal interval-arithmetic or arbitrary-precision proof certificate.

## Feasible lower bound and finite search

- Always initialize with the actual baseline ten, which is feasible.
- Deterministic one-swap plus bounded two-swap improvement.
- An additional exact binary formulation with pair-product variables is
  explored by a time-limited MILP solver (5 seconds per pool/context).
- A MILP timeout is allowed. Only integer incumbents whose recomputed quality
  passes the strict check may improve the lower bound.
- MIP dual bounds are logged, but the headline upper bound is the independently
  evaluated concave-tangent/LP-dual bound.
- Quality numeric tolerances are FRC1e-10 and FRD1e-8, not percentage slack.
- Wide gaps are explicitly unresolved; do not infer generator insufficiency
  from a low feasible value when the upper bound remains high.

The relaxation is a diagnostic of THIS finite pool, not of all possible
generator outputs, and no inference-time GT-free quality guarantee follows.
Convex max-sum diversification is established prior art, not a claimed new
theorem: https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SoCG.2016.26

## Artifacts and reproducibility

Command from repository root:

```bash
PYTHONPATH=mam_reactor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
$LOCAL_PATH/python -u -m coreset_reactor.run_certified_oracle \
  --output coreset_reactor/reports/fixed_teacher_distillation/certified_select10_val24_seed1234_v2 \
  --contexts 24 --workers 8 --mip-seconds 5
```

`--resume` explicitly reuses completed contexts and per-candidate metric
caches, after comparing config and code/input hashes. Without it, existing
output directories are not overwritten.

- `config.json`: exact source subset, checkpoint/prediction/source-code hashes.
- `quality_*.npz`: vectorized FRC and Mam-style exact rolling-DTW FRD per candidate.
- `context_*.json`: selected indices/origins, numeric thresholds, LB/UB,
  relaxation status, tangent/dual multipliers, MIP status and wall times.
- `summary.json`: per-context results, aggregate means and distributions.
- `REPORT.md`: live human-readable pilot results.

No new neural inference is timed here because stored predictions are reused.
Training temporal-loss budget is zero. Report selector and metric timing
separately; neither is an end-to-end deployed generator benchmark.

The initial output directory without `_v2` stopped before writing a completed
context because a NumPy boolean in solver metadata was not JSON serializable.
This serialization issue was fixed and covered by a regression assertion;
that diagnostic directory is retained, and the actual pilot runs in `_v2`.

## Verification before pilot launch

55 tests passed. New tests include gradient and squared-distance identities,
five random tiny problems checked by exhaustive enumeration, an infeasible
tangent-point certificate, a collapsed/duplicate pool and an inadmissible
high-diversity candidate. First real context checks vectorized FRC against the
scalar implementation; every selected set checks FRDiv against official code.

## Next decision, not an automatic next job

Finish all24 and inspect gaps. If bounds are loose, improve optimization before
claiming a generator ceiling. If strong strict-quality gains are present,
confirm on a larger frozen subset/full VAL and only then design a GT-free
quality scorer plus independent calibration. Neither learned selection nor
32/64-candidate training is launched by this pilot.

## Authorized expansion: complete VAL571

The user approved expansion after all24 finished. The full evaluation keeps
the same baseline, candidates, all-session GT, numerical solver and strict
per-context quality constraints. Three disjoint contiguous shards cover
0:190, 190:380 and 380:571. Each shard has eight exact-FRD workers and the
unchanged five-second MILP budget per pool/context. No new model is trained.

```bash
PYTHONPATH=mam_reactor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
$LOCAL_PATH/python -u -m coreset_reactor.run_certified_fullval \
  --output coreset_reactor/reports/fixed_teacher_distillation/certified_select10_val571_seed1234 \
  --reuse-dir coreset_reactor/reports/fixed_teacher_distillation/certified_select10_val24_seed1234_v2 \
  --shards 3 --workers-per-shard 8 --mip-seconds 5
```

The24 pilot contexts are reused only after checking checkpoint/prediction
hashes, numerical-source hashes, GT cache tensor hashes and candidate origins.
Their FRC, selection feasibility, FRDiv and saved upper-bound expressions are
recomputed before import. Original pilot artifacts remain untouched. Only
the orchestration wrapper is allowed to differ from the pilot source hashes.

The coordinator checks shard assignments and protocols, rejects duplicate
contexts, and updates the root `REPORT.md`, `summary.json` and `status.json`.
Progress counts both validated reused and newly evaluated contexts. Partial
means must never be reported as finished full-VAL results. Child errors are
recorded without terminating unaffected shards; an incomplete run does not
receive a successful completion marker. An explicit `--resume` can continue
under the same frozen launch/config hashes.

All prediction tensors already exist, so this phase uses CPU exact-DTW and
optimization rather than repeating GPU inference. No weights are overwritten.
