# Release validation

Validated on 2026-11-22 with Python 3.10.12, PyTorch 2.1.0+cu121,
NumPy 1.26.4, SciPy 1.15.3 and scikit-learn 1.7.2. The checks below ran
on CPU; the complete paper training was not repeated for this release.

| Check | Observed result |
| --- | --- |
| Unit/contract tests | 12 passed |
| Synthetic training pipeline | All six stages passed: reference organization, paired teacher, target-specific teacher, ten-output cache, centered-difference student KD, export |
| Main checkpoint export | All parameter names and tensor values identical to the original A0 seed-1 checkpoint |
| Real source, full long sequence | 958 frames, two windows; released and original soft outputs bitwise identical on the same CPU runtime |
| Real source, short sequence prefix | 127 frames, one padded window; released and original soft outputs bitwise identical |
| Evaluation output | AU-rounded and query-0-postprocessed outputs identical for both parity cases |
| Feature dataset parity | Four examples, including stratified views and both source roles; returned tensors and metadata identical to the original loader |
| Frozen reference identity | All 3,320 mappings identical to the archived paper target manifest |
| Independent CLI | Main weight produces `[10,32,25]` finite outputs on synthetic features |
| Packaging | Wheel built, installed into a separate temporary directory and ran the CLI outside the repository |

The pipeline smoke test uses synthetic feature files and small models with
one optimizer update per teacher/student. It verifies the executable training
path without claiming benchmark convergence. The real-source parity checks
use the released 3,803,201-parameter model and the archived main-result weight.
The quantitative benchmark row in `MODEL_CARD.md` is retained from its frozen
full-VAL evaluation rather than recomputed by these smoke checks.

The two historical q0 assets (student initializer and teacher-cache q0)
have different tensors and are recorded separately. Their maximum absolute
parameter difference is 0.0018344643. This distinction does not affect the
verified identity of the final released A0 weight or its inference parity.

To repeat the repository checks:

```bash
python -m pytest -q
python scripts/smoke_test.py
python -m coreset_reactor.inference --demo --device cpu --output demo.npz
```
