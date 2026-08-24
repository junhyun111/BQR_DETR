# DINO R50 + GT-guided auxiliary localization on VOC2007

The research comparison keeps the official DINO R50 4-scale detector identical
between methods. `gt_guided_aux` adds only a training-time query-aligned spatial
sampling and auxiliary box-regression path. Inference uses the normal DINO path.

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
- `notebooks/02_evaluate.ipynb`: VOC2007 11-point mAP@0.5 evaluation and plots.

The default recipe uses 1,000 deterministic images from the official VOC2007
train split, the full 2,510-image validation split, 12 epochs, and an LR drop at
epoch 11. All implementation logic lives in Python modules; notebooks only
orchestrate runs.

## Provenance

The official DINO source is vendored at commit
`d84a491d41898b3befd8294d1cf2614661fc0953`. See `THIRD_PARTY_NOTICES.md`.
