# BQR-DN V2.1

V2.1 keeps the BQR-DN V2 noisy-box sampling and gated DN-query fusion, but
scores each sampled encoder feature using both the DN-query spatial prior and
query-feature content compatibility. It adds no loss, decoder, head, inference
operation or reference-box correction.

The importable package is `dqr_v2_1`; the enclosing directory keeps the
requested experiment name `dqr-v2.1`.
