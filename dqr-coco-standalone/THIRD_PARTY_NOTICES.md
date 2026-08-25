# Third-party notices

`third_party/dino` contains source from IDEA-Research/DINO at commit
`d84a491d41898b3befd8294d1cf2614661fc0953`. It is distributed under the
Apache License 2.0; see `third_party/dino/LICENSE`.

Local compatibility changes include modern torchvision weight loading,
device-neutral denoising tensors, a differentiable PyTorch fallback for the
optional multi-scale deformable-attention extension, FP16-to-FP32 casting at
the compiled attention boundary, forced CUDA compilation during container
builds, pinned/non-blocking `NestedTensor` transfers, and an optional
loss-normalizer override used to emulate the official global batch under
gradient accumulation. The default official DINO code path is unchanged when
that override is omitted.
