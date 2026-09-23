# Quality-gated teacher pilot

This is the short matched 100-step pilot for the TRAIN quality-gated target
frontier. It uses the same ConditionalREGNN 25-D model, initialization, source
schedule, optimizer, BF16 precision, `relative_time_masked` alignment and
evaluation contract as the frozen teacher experiment. Only the fixed target
manifest changes.

Arms were trained with 10 ranks each:

```text
T0, DESC_Q50, DESC_Q70, DESC_Q80, T1
```

The evaluation is VAL64 with center750 and all same-session opposite-role GT.
It is a pilot diagnostic, not the final full-VAL result.

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ | validity ↓ | utilization ↑ | slot-pair FRDiv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 0.473121 | 90.826744 | 0.020280 | 0.012589 | 1.211186 | 1.054795 | 0.334375 | 0.020280 |
| DESC_Q50 | **0.812996** | 93.165099 | 0.082605 | 0.028422 | **1.172519** | 1.078178 | **0.559375** | 0.082605 |
| DESC_Q70 | 0.654682 | 94.804934 | 0.096315 | 0.025591 | 1.195919 | 1.146287 | 0.418750 | 0.096316 |
| DESC_Q80 | 0.660603 | 96.296029 | 0.107741 | 0.025885 | 1.199486 | 1.177352 | 0.445312 | 0.107742 |
| T1 | 0.649133 | 99.966683 | 0.150155 | 0.033545 | 1.198728 | 1.243317 | 0.510938 | 0.150156 |

The short pilot shows a monotone diversity/specialization response to the
descriptor gate: FRDiv rises from `0.020280` to `0.150155`, while descriptor
centroid distance rises from `0.039981` to `0.153489`. All 50 checkpoints were
written successfully. FRC is not monotone at 100 steps, so the pilot is not
used to select a final quality-safe operating point; the next test is the
matched 600-step run.

The raw checkpoints and per-context evaluation JSON are local-only. The
formal-run provenance will record the committed source SHA, target-manifest
hashes, initialization hash, source-schedule hash, and GPU assignment.
