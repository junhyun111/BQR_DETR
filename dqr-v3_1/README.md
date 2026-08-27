# BQR-DN V3.1

V3.1 is the weak-scale-prior ablation of BQR-DN V3. It preserves V3's
five-point sampler, geometric prior, fusion path, initialization and
diagnostics, while changing the configured attention logits from

`A[l, k] = Q[l, k] + 1.0 * S[l]`

to

`A[l, k] = Q[l, k] + 0.5 * S[l]`.

The package deliberately reuses the V3 implementation so the scale multiplier
is the only implementation-level difference. Runs and checkpoints remain
isolated under the `bqr_dn_v3_1` method key.
