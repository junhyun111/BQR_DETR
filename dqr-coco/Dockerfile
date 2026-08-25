ARG PYTORCH_IMAGE=pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
FROM ${PYTORCH_IMAGE}

ARG TORCH_CUDA_ARCH_LIST=8.9
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/workspace/torch-cache \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTHONPATH=/workspace/project/dqr-coco:/workspace/project/third_party/dino

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential fonts-dejavu-core libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/project
COPY dqr-coco/requirements.txt /tmp/dqr-coco-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/dqr-coco-requirements.txt

COPY third_party/dino ./third_party/dino
COPY dqr-coco ./dqr-coco
COPY THIRD_PARTY_NOTICES.md ./THIRD_PARTY_NOTICES.md

RUN cd third_party/dino/models/dino/ops \
    && FORCE_CUDA=1 python -m pip install --no-cache-dir --no-build-isolation .

RUN chmod +x dqr-coco/scripts/*.sh \
    && python -c "from models.dino.ops.functions import HAS_MSDA; assert HAS_MSDA" \
    && python -m pytest -q dqr-coco/tests \
    && python -m compileall -q dqr-coco/dqr_coco dqr-coco/train.py dqr-coco/evaluate.py dqr-coco/compare.py

CMD ["python", "dqr-coco/train.py", "--help"]
