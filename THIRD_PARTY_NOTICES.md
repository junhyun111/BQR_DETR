# Third-party notices

`third_party/dino` contains source from IDEA-Research/DINO at commit
`d84a491d41898b3befd8294d1cf2614661fc0953`. It is distributed under the
Apache License 2.0; see `third_party/dino/LICENSE`.

Local compatibility changes are limited to modern torchvision weight loading,
device-neutral denoising tensors, and a differentiable PyTorch fallback for the
optional multi-scale deformable-attention extension.
