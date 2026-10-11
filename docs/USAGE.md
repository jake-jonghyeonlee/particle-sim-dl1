# psdl1 사용 안내 — 절차, 입력 형식, 출력 형식

DL1은 새 제품과 **원본 형상·전극 연결도**가 비슷한 기존 제품을 찾는다. unit 크기가 달라도 찾을 수 있게 unit 전체를 같은 격자로 맞춘 뒤 전극 × layer 단위로 묶어 비교한다. 결과는 비교군 후보 목록이며 원인 판정이나 위험 예측이 아니다.

- 입력은 원본 layout 형상과 연결 정보뿐이다. hotspot, CA, fab 값은 받지 않는다(있으면 오류).
- 실제 자료 검증은 폐쇄망에서 사용자가 한다. 여기서 검증한 것은 합성 자료뿐이다.
- 세부 규칙의 근거는 [data_contract.md](data_contract.md), [training.md](training.md), [validation.md](validation.md)에 있다.

## 1. 전체 절차

| 순서 | 하는 일 | 명령 | 만들어지는 것 |
|---|---|---|---|
| 0 | 설치 | `uv pip install` | `.venv` |
| 1 | 입력 파일 준비 | (사용자) | manifest, mask, split, relations |
| 2 | 입력 검증과 격자 확인 | `python -m psdl1.data` | 격자 보고서 JSON |
| 3 | 학습 | `python -m psdl1.train` | `dl1.pt`, `split.json`, `history.json` |
| 4 | 검색 | `python -m psdl1.search` | 검색 결과 JSON |
| 5 | 평가(정답이 있을 때) | `python -m psdl1.evaluate` | 평가 JSON |

## 2. 설치

Python 3.12 기준이다. 네트워크 내려받기는 쓰지 않는다. 사내 PyPI나 wheel 폴더에서 설치한다.

```bash
uv venv --python 3.12 .venv
uv pip install --no-index --find-links /approved/wheels -r requirements.lock
uv pip install --no-index --find-links /approved/wheels -e .
.venv/bin/python -m pytest -p no:warnings          # 설치 확인
```

`requirements.lock`은 개발 기기(macOS)에 설치된 버전 기록이다. 폐쇄망 OS에 맞는 wheel로 같은 버전을 설치한다.

## 3. 입력

### 3.1 폴더 구성 예

```text
data/
  manifest.json          제품 목록 (필수)
  split.json             학습/검증/시험 분할 (학습·평가에 필수)
  relations.csv          제품 쌍 정답 (선택; 없으면 --unlabelled-negatives)
  hard-negatives.json    어려운 반례 쌍 (선택, 평가용)
  P1/valid.npy           관측 범위
  P1/gate-L1.npy         전극 × layer 하나의 형상
  P1/via-1.npy           연결(CNT/VIA) 형상
```

### 3.2 mask 파일

| 항목 | 규칙 |
|---|---|
| 형식 | `.npy`, 2차원 배열 |
| 값 | 0/1(bool 포함) 또는 0/255. 0이 아닌 값이 형상이다 |
| 거부 | 확률값, 음수, NaN, 1과 255 혼합, 3차원 |
| 크기 | 그 제품의 `frame.shape_yx`와 같아야 한다. 읽을 때 크기를 바꾸지 않는다 |
| 경로 | manifest 폴더 기준 상대 경로. 절대 경로, `..`, 폴더 밖 symlink는 거부 |

이 reader에는 극성 선택이 없다. **검정이 형상인 이미지는 0/1을 뒤집어 저장한 뒤 넣어야 한다.** (layout 이미지의 극성은 아직 확인되지 않은 항목이다.)

### 3.3 `manifest.json` — schema `dl1_shape_v1_draft`

```json
{
  "schema_version": "dl1_shape_v1_draft",
  "candidate_grid": 256,
  "products": [{
    "product_id": "P1",
    "lineage_id": "family-P",
    "design_version": "v1",
    "unit_size_um": [100.0, 100.0],
    "frame": {
      "frame_id": "P1-unit", "length_unit": "um", "axis_order": "yx",
      "origin_xy_um": [0.0, 0.0], "pixel_pitch_um": 0.05,
      "shape_yx": [2000, 2000]
    },
    "valid": "P1/valid.npy",
    "elements": [
      {"element_id": "gate-L1", "electrode_id": "gate", "role": "gate",
       "actual_layer": "L1", "observation": "observed", "mask": "P1/gate-L1.npy"},
      {"element_id": "data-L2", "electrode_id": "data", "role": "data",
       "actual_layer": "L2", "observation": "observed", "mask": "P1/data-L2.npy"}
    ],
    "connectivity_observation": "complete",
    "connections": [{
      "connection_id": "via-1", "kind": "VIA", "observation": "observed",
      "mask": "P1/via-1.npy",
      "endpoints": [
        {"element_id": "gate-L1", "xy_um": [12.5, 40.0]},
        {"element_id": "data-L2", "xy_um": [12.5, 40.0]}
      ]
    }]
  }]
}
```

위 숫자는 형식을 보이는 예시이며 권장값이 아니다.

| 필드 | 뜻 | 규칙 |
|---|---|---|
| `candidate_grid` | unit 전체를 맞출 격자 한 변의 칸 수 | 필수. 기본값이 없고 추정하지 않는다. 가장 가는 선이 약 2칸 이상 남도록 사용자가 정한다 |
| `product_id` | 제품 이름 | manifest 안에서 유일 |
| `lineage_id` | 같은 계열을 묶는 이름 | 같은 계열의 변형(크기만 다른 것 등)은 같은 값. 분할 누수 방지에 쓴다 |
| `unit_size_um` | unit 크기 [가로, 세로] | `shape_yx × pixel_pitch_um`과 같아야 한다 |
| `frame.pixel_pitch_um` | 1픽셀의 µm | manifest 안 모든 제품이 같아야 한다 |
| `frame.shape_yx` | [세로, 가로] 픽셀 수 | 제품마다 달라도 된다 |
| `frame.origin_xy_um` | 왼쪽 위 모서리 좌표 | |
| `valid` | 관측된 범위 mask | 관측 안 된 곳은 0 |
| `elements[]` | 전극 × 실제 layer 하나씩 | 한 제품 안에서 role × layer 조합은 하나만 |
| `elements[].observation` | `observed`(mask 경로) 또는 `missing`(`mask: null`) | 빈 형상(관측된 0)과 누락을 구별한다. missing이 있으면 그 제품은 계산에서 빠진다 |
| `connectivity_observation` | `complete` / `partial` / `unavailable` | 연결 정보가 없으면 `connections: []`와 `unavailable` |
| `connections[]` | CNT 또는 VIA 하나 | `kind`는 `CNT` 또는 `VIA`. 끝점 2개, 서로 다른 element |
| `endpoints[].xy_um` | 연결 위치(원본 좌표, µm) | 해당 전극 형상과 연결 mask 안에 있어야 한다 |
| `cnt` (선택) | 제품 전체 CNT mask | 끝점 정보를 대신하지 못한다. 이것만 있으면 연결 정보는 unavailable로 남는다 |

`hotspot`, `CA`, `fab`이라는 이름의 필드, 알 수 없는 필드, 중복 key는 모두 오류다.

### 3.4 `split.json` — schema `dl1_split_v1`

```json
{
  "schema_version": "dl1_split_v1",
  "train": ["P1", "P2", "P3"],
  "validation": ["P4"],
  "test": ["P5", "P6"]
}
```

모든 제품이 정확히 한 번 들어가야 하고, 같은 `lineage_id`의 제품은 한 구획에만 있어야 한다. `train`은 비어 있으면 안 된다.

### 3.5 `relations.csv` — 제품 쌍 정답 (선택)

머리글 없는 CSV다. 형식은 아직 초안이며 최종 형식은 확정되지 않았다.

```text
P1,P2,similar
P1,P3,different
```

- 적지 않은 쌍은 "모름"으로 두고 학습 손실에서 뺀다.
- 같은 쌍에 similar와 different가 함께 있으면 오류다.
- different 쌍이 하나도 없으면 학습을 시작하지 않는다(모든 제품이 한 점으로 모이는 것을 막는다).
- 정답이 없으면 `--unlabelled-negatives`로 "적지 않은 쌍은 다르다"고 가정할 수 있다. 실제로 비슷한 제품을 서로 밀어낼 수 있는 실험 옵션이다. 같은 계열끼리는 이 경우에도 "모름"으로 둔다.

### 3.6 `hard-negatives.json` — 어려운 반례 (선택, 평가용)

형상은 같고 연결 끝점만 다른 제품 쌍 목록이다. 시험 구획의 제품 id를 쓴다.

```json
[["P5", "P6"]]
```

### 3.7 검색 입력

`psdl1.search`는 manifest 두 개를 받는다. 형식은 3.3과 같다.

- `--query-manifest`: 찾으려는 새 제품이 든 manifest. `--query`로 제품 id를 고른다.
- `--candidates-manifest`: 후보 제품들이 든 manifest.

## 4. 명령

```bash
# 2) 입력 검증과 격자 확인 (학습 전에 반드시)
.venv/bin/python -m psdl1.data data/manifest.json --report grid-report.json

# 3) 학습 (설정 파일 + 명령줄 덮어쓰기)
.venv/bin/python -m psdl1.train --config configs/train.example.json
.venv/bin/python -m psdl1.train --manifest data/manifest.json --split data/split.json \
  --relations data/relations.csv --out runs/dl1 --epochs 100 --batch-size 8 --lr 0.0003

# 이어서 학습 (--epochs는 총 목표 횟수)
.venv/bin/python -m psdl1.train --config configs/train.example.json \
  --resume runs/dl1/dl1.pt --epochs 150

# 4) 검색
.venv/bin/python -m psdl1.search --ckpt runs/dl1/dl1.pt \
  --query-manifest data/query.json --query NEW-PRODUCT \
  --candidates-manifest data/manifest.json --topk 5 --out search.json

# 5) 평가 (정답이 있을 때만)
.venv/bin/python -m psdl1.evaluate --ckpt runs/dl1/dl1.pt \
  --manifest data/manifest.json --split data/split.json --relations data/relations.csv \
  --hard-negatives data/hard-negatives.json --topk 5 --out evaluation.json
```

### 학습 설정 (`configs/train.example.json`의 key = 명령줄 옵션)

| key | 뜻 | 비고 |
|---|---|---|
| `manifest`, `split`, `out` | 입력 manifest, 분할, 출력 폴더 | 필수 |
| `relations` | 정답 CSV | 없으면 `unlabelled_negatives: true` 필요 |
| `epochs`, `batch_size`, `lr`, `temperature`, `seed` | 학습 값 | 예시 값은 권장값이 아니다 |
| `grid` | 격자 | 생략하면 manifest의 `candidate_grid`. 다른 값이면 `allow_grid_override` 필요 |
| `allow_grid_check_failure` | 가는 선·연결 끊김 경고를 무시하고 진행 | 기본 false. 켜면 checkpoint에 기록된다 |
| `use_unit_size` | unit 크기를 모델에 조건으로 넣기 | 기본 false. 켠 것과 끈 것을 비교하는 용도 |
| `backbone` | `tiny`(기본) 또는 timm 이름 | 내려받기 없음 |
| `weights` | 로컬 가중치 파일 | 기본 없음. 공개 ConvNeXt V2 파일은 검증하지 않았다 |
| `augmentation` | `none`(기본) 또는 `intensity` | 자르기·이동은 지원하지 않는다 |
| `device` | `cpu`(기본) 또는 `cuda` | |
| `resume` | 이어서 학습할 checkpoint | 설정·입력·분할이 바뀌면 거부 |

## 5. 출력

### 5.1 격자 보고서 (`psdl1.data --report`)

제품마다 격자 확인 한 건이 들어간다.

| 필드 | 뜻 |
|---|---|
| `product_id`, `grid`, `source_shape_yx`, `scale` | 어느 제품을 어떤 격자로 얼마나 줄였는지 |
| `elements[]` | 형상마다 줄이기 전후의 픽셀 수, 덩어리 수, 가장 짧은 선 길이와 격자 기준 추정값 |
| `elements[].thin_line_warning` | 가장 짧은 선이 약 2칸 미만 |
| `elements[].bridge_or_component_loss` | 줄인 뒤 덩어리 수가 바뀜(끊김, 사라짐, 붙음) |
| `reasons` | 경고 사유 목록 |
| `connectivity_preserving` | 항상 false. 이 보고서는 연결 보존을 보증하지 않는다 |

경고가 있으면 학습이 기본으로 막힌다. 격자를 키우거나, 확인 후 `--allow-grid-check-failure`로 진행한다.

### 5.2 학습 출력 (`--out` 폴더)

| 파일 | 내용 |
|---|---|
| `dl1.pt` | checkpoint. 매 epoch 끝에 원자적으로 교체된다. 모델, optimizer, 난수 상태, 설정 fingerprint, 입력·분할·코드 hash, 격자 근거, 학습 쌍 통계 |
| `split.json` | 실제로 쓴 분할 |
| `history.json` | epoch별 손실 |

표준 출력에는 마지막 손실 등 요약 JSON 한 줄이 나온다.

### 5.3 검색 결과 (`psdl1.search --out`)

```json
{
  "query_id": "NEW-PRODUCT",
  "status": "available",
  "reason": null,
  "rankings": [
    {"product_id": "P2", "score": 0.912345,
     "connectivity_status": "unavailable", "input_identity": "e3cf…"}
  ],
  "excluded": [{"product_id": "P7", "reason": "grid_check_failed"}],
  "candidate_count": 12, "eligible_count": 11,
  "grid": 256,
  "grid_selection": {"selected_grid": 256, "manifest_candidate_grids": [256],
                     "source": "checkpoint", "override": false, "override_requested": false},
  "grid_gate": {"policy": "strict_default", "bypass_requested": false, "products": []},
  "fingerprint": "c983…",
  "query_connectivity": "unavailable",
  "score_policy": "canonical_single_product_float64_cosine_round6_then_id",
  "approval": {"input_method": "user_approved_2026_10_09", "real_grid_linewidth": "pending",
               "endpoint_source_format": "pending", "G6": "pending"}
}
```

| 필드 | 뜻 |
|---|---|
| `status` | `available` 또는 `unavailable` |
| `reason` | unavailable일 때 사유. 예: `no_candidates`, `no_eligible_candidates`, `grid_check_failed`, 학습에 없던 role/layer |
| `rankings[]` | 유사도 높은 순. `score`는 cosine(−1~1)을 소수 6자리로 반올림한 값. 같은 점수는 제품 id 순 |
| `rankings[].connectivity_status` | 그 제품의 연결 정보 상태(`complete`/`partial`/`unavailable`) |
| `rankings[].input_identity` | 입력 형상·연결의 fingerprint |
| `excluded[]` | 후보에서 뺀 제품과 사유 |
| `grid`, `grid_selection`, `grid_gate` | 실제로 쓴 격자와 제품별 격자 확인 근거 |
| `fingerprint` | 모델 설정 fingerprint. 수치 저장소에서 결과를 읽을 때 대조한다 |
| `approval` | 무엇이 확정되고 무엇이 미정인지 |

후보가 없다는 것은 "비슷한 제품이 없다"는 뜻일 뿐 위험이 0이라는 뜻이 아니다.

### 5.4 평가 결과 (`psdl1.evaluate --out`) — schema `dl1_evaluation_v1`

| 필드 | 뜻 |
|---|---|
| `recall_at_k`, `baseline_recall_at_k` | 모델과 기준선(정규화 이미지 거리)의 Recall@k 평균. 계산할 query가 없으면 null |
| `evaluated_queries`, `unavailable_queries` | 계산된 query 수와 빠진 수 |
| `queries[]` | query별 결과: `metric_status`, `metric_reason`, `known_positive_candidates`, `recall_at_k`, `baseline_recall_at_k`, 모델 순위(`model`), 기준선 순위(`baseline`) |
| `hard_negative_evaluation` | 어려운 반례의 순위·점수·정답 대비 여유. 목록이 없으면 `unavailable`/`none_supplied` |
| `performance_acceptance` | 항상 `not_established`. 수락 기준은 아직 정해지지 않았다 |
| `checkpoint_weights_hash`, `input_hash`, `code_hash`, `split_hash` | 재현 근거 |

평가는 시험 구획의 제품만 쓴다. 학습하지 않은 모델의 순위는 성능 근거가 아니며 평가기가 거부한다.

## 6. 폐쇄망에서 정할 것

- 실제 격자 크기(`candidate_grid`): 가장 가는 선폭 기준
- CNT/VIA 끝점 정보가 들어오는 실제 형식
- layout 이미지의 극성과 파일 형식(지금은 `.npy`, 0이 아닌 값이 형상)
- 정답 형식과 수락 기준
- 공개 가중치를 쓸지 여부와 licence

## 7. 수치 저장소와의 연결

검색 결과 JSON은 `particle-sim-compare`의 `simcompare.dl_results`가 파일로 읽어 원본 CA·line/pixel 값 옆에 붙인다. DL 점수는 수치 값을 바꾸지 않는다. 절차는 그 저장소의 `docs/handoff/commands.md`에 있다.
