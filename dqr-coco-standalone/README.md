# DQR-COCO Standalone

이 폴더 하나만 GPU 서버로 복사하면 BQR-DN V2와 공식 DINO baseline을 COCO 2017에서 학습·평가할 수 있습니다. 필요한 공식 DINO 소스와 프로젝트 패치가 `third_party/dino`에 포함되어 있으므로 상위 저장소는 필요하지 않습니다.

## 포함된 학습 설정

```text
GPU: RTX 4090 2장, DDP
Precision: FP16 AMP
Batch: GPU당 2 × GPU 2 × accumulation 4 = effective batch 16
Schedule: 12 epochs, LR drop at epoch 11
Model: official DINO R50 4-scale, 900 queries, DN 100
Evaluation: 매 epoch 전체 COCO val2017
Checkpoints: latest.pt / best.pt / final.pt
```

baseline과 BQR-DN V2는 동일한 DINO 초기화와 학습 설정을 사용하며, 차이는 BQR 모듈의 유무뿐입니다. 첫 세 accumulation step에는 DDP `no_sync()`가 적용됩니다. MSDeformAttn CUDA extension은 Docker 이미지 빌드 중 RTX 4090의 compute capability 8.9 대상으로 컴파일되며, 확장을 불러오지 못하면 빌드가 실패하도록 되어 있습니다.

## 서버에 복사할 폴더

```text
dqr-coco-standalone/
├── dqr_coco/             # 학습/평가 엔진과 BQR-DN V2
├── third_party/dino/     # 포함된 공식 DINO 및 호환 패치
├── scripts/
├── tests/
├── Dockerfile
├── compose.yaml
├── train.py
├── evaluate.py
├── compare.py
└── .env.example
```

COCO 이미지와 annotation은 용량이 크므로 폴더에 넣지 않고 서버 NVMe에서 읽기 전용으로 mount합니다.

```text
/mnt/nvme/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## 1. 환경 설정과 이미지 빌드

서버에는 NVIDIA driver, Docker, NVIDIA Container Toolkit, Docker Compose plugin이 설치되어 있어야 합니다.

```bash
cd dqr-coco-standalone
cp .env.example .env
nano .env
docker compose --env-file .env build
```

`.env`에서 최소한 다음 두 경로를 실제 서버 경로로 수정합니다.

```dotenv
COCO_ROOT=/mnt/nvme/coco
OUTPUT_ROOT=/mnt/nvme/experiments/dqr-coco
```

## 2. 2-GPU smoke test

본 학습 전에 32개 train 이미지와 16개 val 이미지로 1 epoch를 확인합니다.

```bash
docker compose --env-file .env run --rm train bash scripts/smoke_2gpu.sh
```

로그에 다음 조건이 표시되어야 합니다.

```text
world_size=2 batch_per_gpu=2 accumulation=4 effective_batch=16 precision=fp16
```

## 3. BQR-DN V2 학습

기본 method가 `bqr_dn_v2`이므로 다음 명령으로 시작합니다.

```bash
docker compose --env-file .env up train
```

중단 후 같은 명령을 다시 실행하면 기본값 `RESUME=auto`에 의해 동일 run의 `latest.pt`를 탐색해 재개합니다. 별도 run 이름이 필요하면 `.env`의 `RUN_NAME`을 지정합니다.

## 4. DINO baseline 학습

V2와 완전히 동일한 recipe에서 BQR만 제거한 baseline입니다.

```bash
METHOD=baseline docker compose --env-file .env run --rm train
```

## 5. 평가

`.env`의 `CHECKPOINT`에 컨테이너 내부 경로를 설정합니다. 호스트의 `OUTPUT_ROOT`는 컨테이너에서 `/workspace/artifacts`로 보입니다.

```dotenv
CHECKPOINT=/workspace/artifacts/bqr_dn_v2/seed_42/<run>/checkpoints/best.pt
```

```bash
docker compose --env-file .env --profile evaluation run --rm evaluate
```

전체 val2017 평가는 run 폴더의 `evaluation.json`에 저장됩니다.

## 6. baseline/V2 비교 검증과 그래프

두 checkpoint가 동일한 DINO 초기화와 학습 조건을 썼는지 먼저 확인할 수 있습니다.

```bash
docker compose --env-file .env run --rm train python scripts/validate_experiment_pair.py \
  /workspace/artifacts/baseline/seed_42/<run>/checkpoints/final.pt \
  /workspace/artifacts/bqr_dn_v2/seed_42/<run>/checkpoints/final.pt
```

비교 대시보드는 다음처럼 생성합니다.

```bash
docker compose --env-file .env run --rm train python compare.py \
  /workspace/artifacts/baseline/seed_42/<run>/history.csv \
  /workspace/artifacts/bqr_dn_v2/seed_42/<run>/history.csv \
  --output-dir /workspace/artifacts/comparison
```

결과는 `OUTPUT_ROOT` 아래에 남고, checkpoint는 `latest.pt`, `best.pt`, `final.pt`만 유지됩니다.
