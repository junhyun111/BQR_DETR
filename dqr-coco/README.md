# BQR-DN V2 on COCO 2017

이 폴더는 official DINO R50 4-scale 12-epoch recipe에 BQR-DN V2만 추가해
2×RTX 4090에서 학습하기 위한 Docker/DDP 실행 패키지입니다. 기본 method는
`bqr_dn_v2`이며 `--method baseline`은 BQR 모듈 자체를 생성하지 않습니다.

## 고정된 메인 recipe

```text
2 GPUs × 2 images/GPU × 4 accumulation = effective global batch 16
FP16 AMP
AdamW: main 1e-4 / backbone 1e-5 / weight decay 1e-4
gradient clipping 0.1
12 epochs / LR drop 11
900 object queries / DN 100 / box noise 0.4 / label noise 0.5
COCO val2017 전체를 매 epoch 평가
```

앞의 세 accumulation micro-step은 DDP `no_sync()`로 gradient 통신을 생략하고
네 번째 micro-step에서만 동기화합니다. accumulation window 전체를 8개의
virtual rank로 간주해 official DINO criterion의 box normalizer를 계산합니다.

GPU당 batch 4는 effective batch를 유지하더라도 각 process의 GT 최대 개수에 따라
결정되는 DINO DN group 구성을 바꿀 수 있으므로 메인 비교 recipe에는 사용하지
않습니다.

## COCO 데이터 구조

NVMe에 다음 구조로 준비합니다.

```text
coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## Docker 준비

서버에는 NVIDIA driver, Docker, NVIDIA Container Toolkit이 필요합니다. 이
저장소 루트에서 환경 파일을 만든 뒤 NVMe 경로와 GPU 번호를 수정합니다.

```bash
cp dqr-coco/.env.example dqr-coco/.env
docker compose --env-file dqr-coco/.env -f dqr-coco/compose.yaml build
```

이미지는 `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel`을 기본으로 사용하고,
RTX 4090의 compute capability `8.9`로 official MSDeformAttn CUDA extension을
빌드합니다. 학습 시 compiled extension을 찾지 못하면 Python fallback으로
진행하지 않고 즉시 중단합니다.

## 먼저 2-GPU smoke test

```bash
docker compose --env-file dqr-coco/.env -f dqr-coco/compose.yaml run --rm \
  train bash dqr-coco/scripts/smoke_2gpu.sh
```

32장 train, 16장 val, 1 epoch를 실행합니다. 로그 첫 줄에서 다음을 확인합니다.

```text
world_size=2 batch_per_gpu=2 accumulation=4 effective_batch=16 precision=fp16
```

## V2 전체 학습

`.env`의 기본 `METHOD=bqr_dn_v2` 상태에서 실행합니다.

```bash
docker compose --env-file dqr-coco/.env -f dqr-coco/compose.yaml up train
```

또는 컨테이너 안에서 다음 스크립트를 사용할 수 있습니다.

```bash
bash dqr-coco/scripts/train_v2.sh
```

`RESUME=auto`는 동일 recipe의 `latest.pt`가 있으면 optimizer, scheduler,
GradScaler와 rank별 RNG 상태를 복원합니다.

## Baseline 학습

V2가 끝난 뒤 method만 바꿉니다. 나머지 recipe와 공통 detector 초기화는 같습니다.

```bash
METHOD=baseline docker compose --env-file dqr-coco/.env \
  -f dqr-coco/compose.yaml run --rm train
```

두 최종 checkpoint가 동일 조건인지 확인합니다.

```bash
python dqr-coco/scripts/validate_experiment_pair.py \
  /workspace/artifacts/baseline/seed_42/<run>/checkpoints/final.pt \
  /workspace/artifacts/bqr_dn_v2/seed_42/<run>/checkpoints/final.pt
```

## 평가

`.env`의 `CHECKPOINT`를 컨테이너 내부 경로로 지정합니다.

```bash
docker compose --env-file dqr-coco/.env -f dqr-coco/compose.yaml \
  --profile evaluation run --rm evaluate
```

제한 평가는 `evaluation_val<N>.json`, 전체 val2017 평가는 `evaluation.json`에
저장되어 서로 덮어쓰지 않습니다.

## 결과

체크포인트는 디스크 사용량을 줄이기 위해 세 종류만 유지합니다.

```text
artifacts/<method>/seed_42/<run>/
├── config.json
├── environment.json
├── history.csv
├── metrics.jsonl
├── evaluation.json
└── checkpoints/
    ├── latest.pt
    ├── best.pt
    └── final.pt
```

`history.csv`에는 AP/AP50/AP75/AP_S/AP_M/AP_L/AR, final main 및 DN loss,
전체 weighted loss, epoch 시간, validation 시간, peak GPU memory가 저장됩니다.
BQR 진단은 기본적으로 100 optimizer step마다 마지막 micro-batch에서만 계산하며,
gate, entropy, offset, fusion delta, size별 level attention mass와 positive/negative
DN sample의 GT-inside ratio를 기록합니다.

두 학습이 끝나면 한 장의 비교 그래프와 수렴 요약을 생성할 수 있습니다.

```bash
python dqr-coco/compare.py \
  /workspace/artifacts/baseline/seed_42/<run>/history.csv \
  /workspace/artifacts/bqr_dn_v2/seed_42/<run>/history.csv \
  --output-dir /workspace/artifacts/comparison
```
