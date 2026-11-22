# REACT2025 feature dataset

`regnn.conditional_data.ConditionalReactionDataset` reads pre-extracted
features rather than raw audio or video. Prepare the REACT2025/MARS release
and use its official extraction/preprocessing pipeline to obtain:

```text
react2025/
  train/
    audio-features/{speaker,listener}/sessionN/clip_name.npy
    facial-attributes/{speaker,listener}/sessionN/clip_name.npy
    coefficients/{speaker,listener}/sessionN/clip_name.npy
  val/
    audio-features/{speaker,listener}/sessionN/clip_name.npy
    facial-attributes/{speaker,listener}/sessionN/clip_name.npy
    coefficients/{speaker,listener}/sessionN/clip_name.npy
```

The same basename across roles identifies the native pair. Every modality
uses the same source-relative path. A session reference pool contains all
opposite-role facial-attribute files from that session. Both roles are used
as sources. The paper TRAIN feature release has 3,320 records in total;
complete VAL has 1,142 (571 in each direction).

| Feature | Shape and processing |
| --- | --- |
| Audio | Float `[T,768]`, already extracted and aligned to facial frames |
| Facial attributes | Float `[T,25]`: 15 AUs, valence, arousal, eight expression probabilities |
| Coefficients | Float `[T,58]` or `[T,1,58]`, normalized using bundled mean/std |

All arrays must be finite and stored in standard NumPy `.npy` format. The
available source duration is the minimum length of its three feature streams.
Use the same feature definitions and temporal alignment as the original
dataset; raw waveforms and RGB frames are not direct model inputs.

## Loader example

```python
from regnn.conditional_data import ConditionalReactionDataset

dataset = ConditionalReactionDataset(
    root_dir="/path/to/react2025", split="train", clip_length=750,
    training=True, target_mode="paired", seed=1,
)
sample = dataset[0]
# speaker_audio: [750,768]
# speaker_emotion: [750,25]
# speaker_3dmm: [750,58], already normalized
# target: [750,25]
# length: scalar valid-prefix length
# clip_id, start, crop_view_id: identifiers for input/target reconstruction
```

`set_epoch(epoch)` changes deterministic training crops. Teacher targets
use the frozen expert assignment. q0 uses the true paired target at the
source offset; other teachers resample their designated cross-recording
reference to the valid source duration (nearest interpolation for AUs,
linear interpolation for VA/expression and expression renormalization).
Pointwise velocity supervision is disabled for these unpaired references.
References from a session are not assumed to be frame-synchronized.

The teacher cache uses a fixed epoch and three stratified views for long
recordings. `TeacherCacheDataset` reconstructs each view and checks its
sample ID, view ID, start and length before returning a training example.
Padding is excluded from all supervised losses.

The release bundles the exact 58-D normalization arrays under
`regnn/assets/`. Set `CORESET_NORMALIZATION_DIR` to a directory containing
`mean_face.npy` and `std_face.npy` only when intentionally using another
compatible normalization. Inference uses the statistics embedded in the
selected student checkpoint. Avoid normalizing the same coefficients twice.

The frozen reference assignment contains dataset-relative file identities,
not facial feature values. Training features and video/audio recordings
must be obtained separately from the dataset providers.
