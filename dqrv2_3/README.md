# BQR-DN V2.3: norm-constrained residual fusion

This package is a clean BQR-DN V2 ablation. V2's four sampling points per
encoder level, query-only attention, projection, sigmoid gate, losses,
initialization and training-only DN-query path are unchanged. The only change
caps each final BQR correction to `residual_ratio * ||query||`, with
`residual_ratio=0.5`.

The isolated method key is `bqr_dn_v2_3`, so checkpoints are written to
`artifacts/bqr_dn_v2_3/seed_42`.
