# Data Preparation

The public University-1652 dataset and the trained feature extractor must be
obtained from their respective sources. This repository does not redistribute
images, labels, descriptors, or model weights.

The evaluation MAT file must contain:

| Variable | Shape | Description |
| --- | --- | --- |
| `query_features` | `(37855, 768)` | Drone-query descriptors |
| `gallery_features` | `(951, 768)` | Original satellite descriptors |
| `query_labels` | `(37855,)` | Query identities |
| `gallery_labels` | `(951,)` | Gallery identities |

The training files `train_drone_features.npy` and
`train_satellite_features.npy` must be two-dimensional float32 arrays with 768
columns. All evaluation and training descriptors must be L2-normalized.

Generated galleries, rankings, indexes, and run outputs belong under
`workspace/` and must not be committed.
