ARG PYTORCH_IMAGE=pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
FROM ${PYTORCH_IMAGE}

ARG TORCH_CUDA_ARCH_LIST=8.9

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/workspace/torch-cache \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTHONPATH=/workspace/project:/workspace/project/third_party/dn_detr

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        fonts-dejavu-core \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/project

COPY requirements.txt /tmp/dn-bqr-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/dn-bqr-requirements.txt

COPY third_party/dn_detr ./third_party/dn_detr
COPY bqr_dn ./bqr_dn
COPY scripts ./scripts
COPY tests ./tests
COPY train.py evaluate.py compare.py prepare.py requirements.txt THIRD_PARTY_NOTICES.md ./

RUN cd third_party/dn_detr/models/dn_dab_deformable_detr/ops \
    && FORCE_CUDA=1 python -m pip install --no-cache-dir --no-build-isolation .

RUN chmod +x scripts/*.sh \
    && python -c "from models.dn_dab_deformable_detr.ops.functions import HAS_MSDA; assert HAS_MSDA" \
    && python -m pytest -q tests \
    && python -m compileall -q bqr_dn train.py evaluate.py compare.py prepare.py

CMD ["python", "train.py", "--help"]
