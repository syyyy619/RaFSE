# University-1652 Protocol

## Workload

- Direction: drone query to satellite gallery.
- Queries: all 37,855 query descriptors.
- Descriptor: 768-dimensional float32, L2-normalized.
- Gallery sizes: 100,000 and 1,000,000.
- Original gallery: 951 satellite descriptors.
- Distractors: sampled with replacement from training-drone followed by
  training-satellite descriptors.
- Gallery random seed: 42.

## Retrieval and metrics

Every query returns exactly 200 gallery IDs. Junk labels (`-1`) are removed
before evaluation. R@1 and R@5 use the first positive after junk filtering.
AP@200 follows the trapezoidal University-1652 evaluator and is normalized by
the number of non-junk positives for the query identity in the full gallery.
Positives outside the returned top 200 contribute zero.

## Dynamic-K

The ambiguity margin is `g = d2 - d1`. Thresholds are stored in ascending
order `tau1 < tau2 < tau3`:

- `g >= tau3`: K=20
- `tau2 <= g < tau3`: K=50
- `tau1 <= g < tau2`: K=100
- `g < tau1`: K=200

The 100K and 1M thresholds are scale-specific and are distributed under
`configs/thresholds/`.

## Timing

Workstation timing uses 16 Faiss threads. It includes retrieval and candidate
refinement, and excludes feature extraction, file loading, index training,
index construction, and gallery construction.
