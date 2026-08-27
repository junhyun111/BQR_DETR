# BQR-DN V2.2: half-strength fusion

This package is a clean ablation of BQR-DN V2. It keeps V2's four sampling
points per encoder level, query-only attention, gate, loss, initialization and
training-only DN-query path unchanged. Only the final residual correction is
scaled from `alpha * projected` to `0.5 * alpha * projected`.

The isolated method key is `bqr_dn_v2_2`, so its checkpoints are written to
`artifacts/bqr_dn_v2_2/seed_42` and cannot overwrite the original V2 run.
