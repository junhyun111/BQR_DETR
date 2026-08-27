# DINO R50 + GT-guided localization research on VOC2007

The research comparison keeps the official DINO R50 4-scale detector identical
between methods. `gt_guided_aux` adds only a training-time query-aligned spatial
sampling and auxiliary box-regression path. `bqr_dn_v2` instead enriches the
official noisy DN query content with noisy-box region evidence before the
shared decoder. `bqr_dn_v2_1` retains V2 and makes its region attention depend
on both the DN query and the actually sampled encoder features. Inference uses
the normal DINO path for every method. `bqr_dn_v2_2` is a clean V2 ablation
that halves only the final gated residual fusion strength. `bqr_dn_v3` instead extends V2 with a
learnable centre-plus-corners five-point pattern and a geometric scale-aware
prior over the unchanged four encoder levels. `bqr_dn_v3_1` keeps V3 intact
except for reducing the scale-prior multiplier from 1.0 to 0.5.

## Environment

This project targets regular CPython 3.14 (not free-threaded `3.14t`) and an RTX
3060. From PowerShell in the project directory:

```powershell
uv python install 3.14
uv sync --python 3.14
uv run python -m ipykernel install --user --name gt-guided-dino --display-name "GT-guided DINO (Python 3.14)"
uv run jupyter lab
```

The official compiled deformable-attention operator is optional. When it is not
installed, the model automatically uses the differentiable PyTorch fallback.

## Notebooks

- `notebooks/01_train.ipynb`: train or resume either method.
- `notebooks/01-1_trainV2.ipynb`: train or resume BQR-DN V2.
- `notebooks/01-2_train.ipynb`: train or resume content-aware BQR-DN V2.1.
- `notebooks/01-3_train.ipynb`: train or resume scale-aware five-point BQR-DN V3.
- `notebooks/01-4.ipynb`: train or resume BQR-DN V3.1 with scale weight 0.5.
- `notebooks/01-5_train.ipynb`: train or resume BQR-DN V2.2 with fusion weight 0.5.
- `notebooks/02_evaluate.ipynb`: VOC2007 11-point mAP@0.5 evaluation and plots.
- `notebooks/03_compare.ipynb`: compare AP/AP50/AP75, object-size AP/AR,
  per-class AP, loss curves and run cost.
- `notebooks/03-1_compare_localization.ipynb`: class-score-free validation
  localization comparison (matched IoU/GIoU, L1, centre and size errors).

The method keys accepted by the shared Python pipeline are `baseline`,
`gt_guided_aux`, `bqr_dn_v2`, `bqr_dn_v2_1`, `bqr_dn_v2_2`, `bqr_dn_v3`, and
`bqr_dn_v3_1`.
BQR-DN implementations live under their matching `dqr-v*` directories; V3.1
checkpoints are written to `artifacts/bqr_dn_v3_1/seed_42`.

The default recipe uses 1,000 deterministic images from the official VOC2007
train split, the full 2,510-image validation split, 12 epochs, and an LR drop at
epoch 11. All implementation logic lives in Python modules; notebooks only
orchestrate runs.

## Provenance

The official DINO source is vendored at commit
`d84a491d41898b3befd8294d1cf2614661fc0953`. See `THIRD_PARTY_NOTICES.md`.
