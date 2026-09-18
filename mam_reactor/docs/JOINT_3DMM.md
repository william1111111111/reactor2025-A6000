# Listener 3DMM trial (2026-09-17)

Run `bash train_joint_3dmm.sh` from this directory. The default is GPU 5,
three epochs on B_fit, frozen pretrained 25-D anchor and legacy EQR warm-start.
This is a second-stage extension, not a retraining of the first-stage anchor.
The original `train.sh` still defaults to the 25-D-only model.

The added linear head predicts normalized 58-D listener coefficients from
shared Mamba hidden states. `prediction` remains [B,10,T,25];
`prediction_3dmm` is [B,9,T,58], corresponding exactly to prediction[:,1:].
There is no 58-D prediction for the fixed q0 anchor. Do not interpret this as
a fully pretrained 83-D anchor or a validated ten-query 3DMM evaluation model.

Listener coefficient files have the same relative paths, frame counts and
crop starts as their emotion targets. Missing/misaligned/nonfinite coefficients
raise errors. FaceVerse mean_face.npy/std_face.npy normalize the coefficients;
recover raw values as pred * std_face + mean_face. Padding and unavailable
targets are excluded from both position and velocity supervision.

Added loss: 1.0 * (normalized coefficient MSE + 0.1 * velocity MSE).
A detached balanced Sinkhorn plan matches the nine learned queries to available
targets using 25-D emotion MSE plus normalized 58-D coefficient MSE. This plan
is used for the added geometry loss; the existing 25-D objective and its own
matching remain unchanged. There is no ground-truth listener input to the model.
All trainable parameters use the inherited learning-rate schedule (2e-6 base).
This short trial checks trainability, not final geometry quality.

Configuration includes listener_3dmm; checkpoints include to_3dmm weights.
Metrics include listener_3dmm_mse and listener_3dmm_velocity. The ordinary
inference/export scripts still export 25-D results; 58-D output is exposed by
the model forward method and needs explicit inverse normalization when exported.
Only full offline sequence-set training is supported by this extension.

Original source backups: local_backups/pre_joint_3dmm_20260917/.
The archive MANIFEST.sha256 describes the original archive, so modified source
files will no longer match those original hashes.

Validation: explicit vs vectorized masked loss; padding invariance; finite
gradients; real ten-target crop alignment; real CUDA forward/backward with
nonzero new-head/shared-trunk gradients and no anchor gradients.
