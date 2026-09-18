# Joint first-stage anchor, 2026-09-17

`bash train_joint_anchor.sh` trains a new anchor from random initialization.
No original anchor or second-stage warm-start is loaded. Full original train
split (3,320 records), 750-frame crops, batch 32, 50 epochs, seed 1, AdamW
learning rate 1e-4 and cosine decay. Every epoch is checkpointed. Neither val
nor test is used for optimization. This follows the archived first-stage
training scope; it is not the second-stage B_fit-only three-epoch schedule.

Shared condition fusion and Transformer features feed two heads: the original
25-D facial-attribute/graph branch and a linear normalized 58-D listener
coefficient head. All parameters train jointly. Model forward returns
`prediction` [B,T,25] and `prediction_3dmm` [B,T,58]. Keeping the 25-D field
separate preserves existing official metric interfaces.

Loss = original emotion MSE + CCC loss + 0.05 * emotion velocity MSE
     + 1.0 * normalized listener 3DMM MSE + 0.1 * 3DMM velocity MSE.
Paired target emotion and coefficients use identical frame count and crop.
Both geometry terms exclude padding; velocity needs two valid adjacent frames.
Normalize with external/FaceVerse/{mean_face,std_face}.npy; inverse transform
is prediction_3dmm * std_face + mean_face.

ConditionalREGNNConfig.listener_3dmm defaults to false for old checkpoints.
New checkpoints store it as true and include the geometry head in state_dict.
Second-stage training has NOT automatically been rerun or switched to this
new anchor; existing stage-two checkpoints remain their original experiments.

Validation passed: real paired crop alignment, geometry gradients into shared
backbone, gradients in both heads, optimizer update, strict new checkpoint
roundtrip, old checkpoint loading, and padding-invariant geometry loss.
Original modified-file backups: local_backups/pre_joint_anchor_20260917/.
