# Third-party notices

This standalone package vendors the official DN-DETR repository at commit
`ff3902a20d521ead052d1243ff249b19bc1ce531` under `third_party/dn_detr`.
DN-DETR is distributed under the Apache License 2.0; its original license is
preserved at `third_party/dn_detr/LICENSE`.

Small compatibility patches are applied to the vendored target model for
device-safe DN loss construction, current torchvision ResNet weights, optional
PyTorch MSDeformAttn fallback in local tests, AMP-safe CUDA dispatch, and
container-time CUDA extension compilation. The exact patched source is included
in each experiment fingerprint.

The vendored implementation also contains notices inherited from DAB-DETR,
Deformable DETR, and DETR in the corresponding source files.
