# DN-Deformable DETR + clean-GT BQR COCO PoC

공식 DN-DETR의 `dn_dab_deformable_detr`를 baseline으로 사용하고, 학습 중
DN content query에만 clean GT box 기반 encoder region evidence를 더하는
standalone 실험 패키지입니다. 상위 저장소 없이 이 폴더 하나만 GPU 서버로
복사해 Docker에서 실행할 수 있습니다.

## 고정된 비교 조건

- upstream: `IDEA-Research/DN-DETR` commit
  `ff3902a20d521ead052d1243ff249b19bc1ce531`
- detector: R50, encoder 6층, decoder 6층, 4 feature levels, 300 queries
- DN: scalar 5, label noise 0.2, box noise 0.4
- train: COCO train2017에서 seed 42로 고정 추출한 10,000장
- validation: COCO val2017 전체 5,000장
- schedule: 20 epochs, LR drop epoch 16
- evaluation: epochs 5, 10, 15, 20
- effective batch: 16
- default hardware: RTX 4090 2대, FP16 AMP

첫 실행에서 다음 subset manifest를 생성합니다.

```text
artifacts/_shared/subsets/coco_train2017_n10000_seed42.json
```

manifest에는 image ID 10,000개, annotation SHA-256, category 및 object-size
분포가 들어갑니다. Baseline과 BQR는 같은 manifest와 같은 detector 초기화를
검증한 뒤 학습합니다.

## BQR 경로

```text
clean augmented GT box
  -> each of four encoder levels: fixed 2x2 sampling
  -> average points and levels
  -> LayerNorm + Linear
  -> sigmoid gate([DN query, region feature])
  -> DN query + 0.25 * gate * region feature
  -> unchanged official decoder and DN attention mask
```

matching query는 수정하지 않습니다. 별도 auxiliary loss, learned offset,
scale attention, gate schedule도 사용하지 않습니다. 평가와 inference에는 DN
prefix 자체가 없으므로 BQR branch가 실행되지 않습니다.

## COCO 디렉터리

```text
/mnt/nvme/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## Docker 준비

서버에 NVIDIA driver, Docker, NVIDIA Container Toolkit, Docker Compose plugin이
설치되어 있어야 합니다.

```bash
cd dn-bqr-coco-standalone
cp .env.example .env
nano .env
docker compose --env-file .env build
```

`.env`에서 최소한 다음 경로와 GPU 번호를 서버에 맞게 수정합니다.

```dotenv
COCO_ROOT=/mnt/nvme/coco
OUTPUT_ROOT=/mnt/nvme/experiments/dn-bqr-coco
TORCH_CACHE=/mnt/nvme/cache/torch
GPU0_ID=0
GPU1_ID=1
```

Docker build 중 RTX 4090의 compute capability 8.9용 MSDeformAttn CUDA
extension을 컴파일하고 unit tests를 실행합니다.

## 2-GPU smoke

Baseline과 BQR를 각각 32 train images / 16 val images로 실행합니다. 각
모델은 2회의 optimizer update를 수행합니다.

```bash
docker compose --env-file .env --profile smoke run --rm smoke
```

smoke가 확인하는 항목은 다음과 같습니다.

- 2-rank NCCL/DDP 실행
- FP16 forward/backward 및 MSDeformAttn CUDA operator
- baseline/BQR 공통 초기화와 subset 일치
- BQR gradient와 진단 지표
- NaN/Inf 방지
- COCO evaluation과 checkpoint 생성
- baseline/BQR pair fingerprint 검증 및 비교 그래프 생성

## 권장 PoC: 먼저 epoch 10까지

2개 GPU를 한 모델에 사용하는 가장 보수적인 비교입니다. Baseline이 끝난 뒤
BQR가 실행됩니다.

```bash
docker compose --env-file .env run --rm train bash scripts/train_pair.sh
docker compose --env-file .env run --rm train bash scripts/compare.sh
```

기본 `.env`의 `STOP_AFTER_EPOCH=10` 때문에 20-epoch scheduler를 그대로
유지하면서 epoch 10 checkpoint에서 멈춥니다.

계속 진행할 때 `.env`를 아래처럼 바꾸고 같은 명령을 다시 실행합니다.

```dotenv
STOP_AFTER_EPOCH=20
RESUME=auto
```

```bash
docker compose --env-file .env run --rm train bash scripts/train_pair.sh
docker compose --env-file .env run --rm train bash scripts/compare.sh
```

## 선택 사항: GPU별 한 모델 동시 실행

4090 두 장 사이의 DDP 통신을 피하고 baseline과 BQR를 동시에 실행하려면:

```bash
docker compose --env-file .env run --rm train bash scripts/train_pair_parallel.sh
```

각 프로세스는 GPU 한 장과 accumulation 8을 사용하므로 effective batch는
동일하게 16입니다. 두 작업이 COCO 파일을 동시에 읽기 때문에 로컬 NVMe를
권장합니다. 가장 엄격한 비교는 위의 순차 2-GPU DDP 명령입니다.

## 개별 실행

```bash
docker compose --env-file .env run --rm train bash scripts/train_baseline.sh
docker compose --env-file .env run --rm train bash scripts/train_bqr.sh
```

compose 기본 command로 BQR 하나만 실행할 수도 있습니다.

```bash
docker compose --env-file .env up train
```

## 결과

```text
artifacts/
├── _shared/
│   ├── initialization/
│   └── subsets/
├── baseline/seed_42/poc_10k_20e/
│   ├── checkpoints/{latest,best,epoch_05,...}.pt
│   ├── config.json
│   ├── environment.json
│   ├── history.csv
│   └── metrics.jsonl
├── bqr/seed_42/poc_10k_20e/
└── comparison/poc_10k_20e/
    ├── comparison.csv
    ├── comparison.json
    └── comparison.png
```

COCO 결과에는 `AP`, `AP50`, `AP75`, `AP_S/M/L`, `AR100`이 포함됩니다.
학습 로그에는 matching loss와 `tgt_loss_ce`, `tgt_loss_bbox`,
`tgt_loss_giou` 및 BQR gate, region norm, fusion delta, gradient norm이
기록됩니다.

## 별도 checkpoint 평가

`.env`에 container 내부 checkpoint 경로를 지정합니다.

```dotenv
CHECKPOINT=/workspace/artifacts/bqr/seed_42/poc_10k_20e/checkpoints/best.pt
```

```bash
docker compose --env-file .env --profile evaluation run --rm evaluate
```

## 구현상 의도적인 제한

- 실제 10k PoC에서는 query 수, encoder/decoder 층, 학습 해상도 및 공식 COCO
  augmentation을 줄이지 않습니다.
- `torch.compile`은 variable-resolution input과 custom CUDA extension의
  재컴파일 위험 때문에 사용하지 않습니다.
- activation checkpointing은 4090 24GB에서 필요할 때만 별도로 검토합니다.
- full val 평가는 매 epoch가 아니라 5/10/15/20에서만 수행합니다.
