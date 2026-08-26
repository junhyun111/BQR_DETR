# BQR-DN V3

V3 keeps the official DINO R50 4-scale detector and the BQR-DN V2 training-only
query-fusion path. It changes only region aggregation for valid denoising
queries:

- five learnable sampling points per encoder level, initialized at the noisy-box
  centre and four quarter-box corners;
- a parameter-free geometric level prior computed from the noisy box footprint
  in valid feature cells;
- one softmax over all 20 sampled points after adding query and scale logits.

The decoder, reference boxes, normal matching queries, detection losses and
inference path are unchanged. V2.1 content-aware attention is intentionally not
part of V3 so the 5-point and scale-prior effects remain isolated.
