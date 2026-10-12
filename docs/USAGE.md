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

이 NPY reader에는 극성 선택이 없다. 검정 원천 PNG는 아래 PNG importer를 통해 검정=1인 bool NPY로 변환한다. PNG의 극성은 검정 형상 / 흰 배경으로 확정되어 있다.

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
| `unit_size_um` | 전체 frame 범위 [가로, 세로] (PNG에서 단일 unit 추정 안 함) | `shape_yx × pixel_pitch_um`과 같아야 한다 |
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
- 실제 PNG writer/mode 및 관측 범위 확인(검정 형상/흰 배경, 0.1 µm/pixel은 확정; 내부 `.npy`는 1이 형상)
- 정답 형식과 수락 기준
- 공개 가중치를 쓸지 여부와 licence

## 7. 수치 저장소와의 연결

검색 결과 JSON은 `particle-sim-compare`의 `simcompare.dl_results`가 파일로 읽어 원본 CA·line/pixel 값 옆에 붙인다. DL 점수는 수치 값을 바꾸지 않는다. 절차는 그 저장소의 `docs/handoff/commands.md`에 있다.

## 8. 실제 PNG 폴더 입력 — 검사 → 명시적 export

PNG 원천은 `<root>/<model>/<electrode>/<seq>-<layer>.png`, CNT는
`<root>/<model>/cno/<seq>_<layer>.png`다. 숫자는 모델 공통 위→아래 stack이며
zero padding은 같은 숫자로 해석한다. 첫 `-`/`_` 뒤 전체 문자열이 정확한 layer명이다.
검정=형상(1), 흰색=배경(0), 원본 pitch=**0.1 µm/pixel**이다. 위 NPY manifest의
0.05 예시는 과거의 임의 합성 NPY 예시이며 실제 PNG pitch를 뜻하지 않는다.
원점은 모델별 로컬 `[0,0]`, 전체 frame 범위는 `[W*0.1,H*0.1]` µm다.
기존 `unit_size_um`에도 이 **전체 frame 범위**를 넣는다. 단일 unit 크기를 추정하지 않는다.
중앙 core `[500,500,W-500,H-500]`는 참고 사각형이며 실제 주변 패턴을 제거하지 않는다.
모델 입력은 전체 frame의 기존 공통 resize+pad다. 자동 crop/타일/별칭 mapping은 없다.

PNG 기능은 optional `png` dependency를 쓴다. 승인된 폐쇄망 wheelhouse에 **Pillow
12.3.0**의 해당 OS/Python wheel이 있어야 하며, package build를 한다면 hatchling의
wheel도 필요하다(현재 runtime lock에 build wheel이 포함되었다는 뜻이 아니다).
기존 NPY 경로는 Pillow 없이도 작동한다. importer는 실제 실행 중 Pillow 버전을
보고서에 기록하며, 12.3.0과 다르면 경고한다. dependency pin을 runtime에서 강제
교체하거나 네트워크로 설치하지 않는다.

회색/색상/반투명 픽셀, grey/RGB의 모든 tRNS, 사용하지 않는 항목도 포함한
non-255 palette alpha, IEND 뒤 바이트, 모든 chunk의 잘못된 CRC, APNG chunk
(단일 frame APNG도 포함), 비정상 방향 metadata는 거부된다. 이런 실제 파일은
**plain opaque black/white PNG로 다시 저장**한 뒤 검사한다. importer가 몰래 고치지 않는다.

아래 경로와 변수는 사용자가 실제 값으로 정한다. grid 기본값/권장 숫자는 없다.
출력은 입력 tree 밖의 **새 폴더**여야 한다. 기존 출력은 빈 폴더여도 덮어쓰지 않는다.

```bash
# lineage/grid 없이 원본 해상도 검사, 연결 후보/미해결/자원/이름 inventory와 preview
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$INSPECT_OUT"

# 사용자가 정한 여러 grid를 비교만 한다: 자동 선택/승인이 아님
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$GRID_INSPECT_OUT" \
  --inspect-grids "$GRID_A" "$GRID_B"

# 명시적 metadata + 후보 grid로 기존 v1 manifest와 bool NPY를 자동 생성
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$EXPORT_OUT" \
  --export --metadata "$METADATA_JSON" --candidate-grid "$CANDIDATE_GRID"

# 원천 PNG가 없는 복사 bundle에서도 파일 hash / manifest-report-asset 연결 검사
.venv/bin/python -m psdl1.import_png --verify-bundle "$EXPORT_OUT"
.venv/bin/python -m psdl1.data "$EXPORT_OUT/manifest.json" --report "$VALIDATOR_REPORT"
```

metadata는 `metadata_schema_version: dl1_png_metadata_v1`과 `models` mapping을
가진다. 각 정확한 모델 폴더명에 사용자가 `lineage_id`, `design_version`을 공급한다.
[예시](../configs/png-metadata.example.json)의 값은 눈에 띄게 **합성 예시**이며
실제 가족/버전 정보가 아니다. 폴더명에서 추정하지 않는다. 검사 중 부분 mapping은
pending으로 보고한다. export에 metadata/grid가 빠지거나 한 모델이라도 입력 오류가
있으면 진단 보고서/유효 모델 preview만 쓰고 nonzero로 종료한다. `manifest.json`이나
부분 학습 pool은 생성하지 않는다. 미해결 CNT 관계 자체는 입력 오류가 아니며 global
CNT에는 미해결 형상도 남는다. Label/split은 자동으로 만들거나 승인하지 않는다.

`import-report.json`은 `dl1_png_import_v2` sidecar이며 core manifest에 임의 필드를
추가하지 않는다. `bundle-integrity.json`은 보고서/manifest/NPY/preview의 파일 SHA-256을
묶는다. 검사는 전기적 진실/서명 인증이 아닌 bundle 손상 탐지다. PNG 원천은 한 번
읽은 compressed snapshot으로 검증·decode·hash하므로 import 중 파일 교체가 현재
snapshot의 mask/hash를 바꾸지 않는다. 게시 전 sibling `.staging-*`에서 검증하고 새
폴더로 native no-replace 게시를 시도한다. 지원되는 플랫폼에서는 directory rename이 원자적이다.
Native API가 없거나 ENOSYS/EINVAL이면 목적지를 exclusive mkdir로 먼저 claim한 뒤
manifest를 마지막에 쓴다. 이 fallback은 directory 전체가 한 번에 보이는 방식은 아니지만
기존 목적지를 대체하지 않고, 실패 시 consumable manifest를 남기지 않는다.
실패한 staging은 자동 삭제하지 않고 stderr 경로로
남긴다. 중단된 staging은 model-ready 최종 bundle로 취급하지 말고 새 출력 경로로 재실행한다.

보고서의 모델별 `resource_estimate`와 정확한 role/layer 목록은 grid 평가/export
크기의 배열을 만들기 전에 저장하고 stdout에 출력한다. N/E/K/R, 생성 mask 수,
bool NPY payload/header 합계, loader의 retained float32 mask bytes 및 K개 모두 해결
시나리오를 본다. 이는 peak RAM/runtime 예측이 아니다. 기존 v1의 관계당 전체 frame
mask, grid tensors/검증 복사, decoder 임시 RGBA/boolean 배열(합성 관측 약 17–18
bytes/pixel), compressed snapshot, **모든 모델의 모든 source bool mask 보관** 비용이
더해진다. 실제 자원 적합성은 사용자가 판단한다; 자동 수치 limit를 추정하지 않는다.

role은 전극 폴더명이고 layer는 suffix 그대로다. checkpoint 학습 vocabulary에 없는
query 이름은 unavailable, 후보는 이유를 붙여 excluded, 직접 embedding은 명확한
오류가 된다. 같은 이름은 embedding symbol을 공유하지만 물리/전기적 동등성이
검증되었다는 뜻이 아니다. 이름 union/unique 목록만으로 checkpoint coverage를
인증할 수 없다. 동일 mask/모델 geometry 경고는 내부 symlink alias나 중복 제품
가능성을 보여주며 자동 merge/drop하지 않는다. NFC/NFD 이름도 정확하게 구분한다.

세 상태의 뜻을 구분한다: `extractor_run: ran`은 extractor가 실행됨, extraction
coverage `complete/partial/unavailable`은 규칙에 따라 해결한 범위, model connectivity
status `available/partial/unavailable`은 기존 모델 status 입력이다.
**complete means extraction coverage only; electrical verification not performed.**
이 coverage는 실제 모델의 세 가지 connectivity embedding 입력에 반영된다. 같은 mask라도
partial→complete이면 모델 입력/identity가 바뀔 수 있다. skipped-layer
`needs_confirmation`은 preview/sidecar의 물리 확인 요청이며 추가 model status가 아니다.
4-neighbour/nearest-overlap skipped-layer 규칙은 구현 해석이고 실제 전기 검증이 아니다.

`preview.html`은 인터넷 없이 연다. 전체 frame 축(x 오른쪽/y 아래), 레이어 토글,
reference-only core, 원본 bbox offset의 component display crop, 해결/미해결/확인 필요
표시와 후보/건너뛴 전극·cno 목록을 보여준다. **Display crop은 사람이 보는 용도이며
model input은 항상 전체 frame**이다. Inspector는 연결을 전기적으로 승인하는 control이 아니다.
실제 데이터 호환성과 검색 품질/G6는 폐쇄망 결과로 별도 평가한다. 5–10개 다양한 제품
pilot은 import/연결 검사 pilot일 뿐 학습 자료 수나 검색 성능의 승인 기준이 아니다.


### 공통 transform과 사람이 보는 context

검사/export/validator grid 보고서와 train 끝점 feature는 같은 `frame_transform`을 쓴다.
원본 H×W를 G에 맞춰 반올림한 nh×nw로 area resize하고 중앙 pad한다. 실제
`sx=nw/W`, `sy=nh/H`는 반올림 때문에 서로 다를 수 있다. 보고서에는 shape,
padding LTRB, 실제 XY scale, content의 effective pitch가 함께 있다. Pixel centre는
`((x+0.5)*sx+left,(y+0.5)*sy+top)`로 연속 좌표 변환하며 inverse도 제공한다.
이는 좌표의 round trip이고 이미지 축소를 복원한다는 뜻이 아니다. 전체 원본 frame의
self/rest/CNT/valid/관계 mask에 같은 변환을 적용하고, 기존 float32 feature 산술을 유지한다.

Preview는 각 CNT bbox에 64px 주변 맥락을 더해 frame 경계에 맞춰 보여준다.
원본 context image는 1 source pixel/image pixel이며 작은 화면은 정수 배율
nearest-neighbour 확대 파일을 따로 제공한다. bbox/offset/배율은 보고서에 기록된다.
파랑=선택 위쪽 전극, rose=선택 아래쪽 전극(같은 선택 layer의 후보 모두), 회색=기타 전극,
검정=CNT outline, 노랑=공통 교집합 J, 녹색 crosshair=공유 끝점이다. 미해결 이유와
위→아래 layer 목록, 건너뛴 전극·cno 목록을 함께 확인한다. Crop/확대는 사람이 보는
표시 용도이고 모델 입력은 그대로 전체 frame이다. 원본 해상도에서 해결한 관계가
작은 grid에서 사라져도 원본 보고서에 남으며 train/search의 기존 gate가 차단한다.


### R-B 수정: 파일 identity, report v2 및 publication 제한

출력의 기존 ancestor와 입력 root를 `os.path.samefile`로 비교하여 case/NFC/NFD와
symlink를 다르게 적어도 같은 tree 안에 쓸 수 없게 한다. 입력이 출력 tree 안에
있어도 거부한다. Argument와 source root를 검증하기 전에 새 parent/staging을 만들지
않으며, invalid model/metadata처럼 진단 bundle을 요청한 경우에만 검사 후 출력한다.
실패/KeyboardInterrupt 후 staging의 manifest는 `manifest.pending.json` 등으로 되돌리고
보고서에는 `failed/interrupted`를 남긴다. 남은 `.writing` 파일이 원래 오류를 가리지 않는다.

새 sidecar는 `dl1_png_import_v2`, integrity는 `dl1_png_bundle_v2`다. Strict
`dl1_shape_v1_draft` manifest, model input/checkpoint version은 바뀌지 않는다. 새 verifier는
이전 v1 sidecar/integrity를 unsupported로 명시적으로 거부한다. 이전 결과를 삭제하거나
덮어쓰지 말고 필요하면 새 출력 경로로 import한다. Integrity는 파일뿐 아니라 정확한
디렉터리 목록도 묶으므로 빈 디렉터리 추가도 검출한다. 잘못되거나 불완전한 보고서는
traceback 대신 nonzero diagnostic으로 종료한다. 완전히 일관되게 다시 봉인한 파일의
진실성/서명 인증은 여전히 제공하지 않는다.

보고서 모델별 `electrode_layers`와 `cno_layers`는 분리되어 있다. Cross-model 목록도
`electrode_layer_union/unique_electrode_layers`와 `cno_layer_union/unique_cno_layers`로
구분한다. **모델 vocabulary에 들어가는 것은 electrode layer 이름뿐**이다.
`all_source_layers/all_source_layer_union/all_source_unique_layers`는 전체 source 감사용이다.
`extractor_run_status: ran/blocked/not_run`은 실행 상태이며 더 이상 실행을 `complete`로
표기하지 않는다. Coverage의 `complete/partial/unavailable` 및 기존 model status와
분리된다. Invalid metadata 진단의 path는 `metadata`이며 absolute location을 report에 넣지 않는다.

`publication_method`는 실제 사용한 방법을 기록한다: macOS의 `darwin_rename_excl`,
Linux의 `linux_renameat2_noreplace`, 또는 `exclusive_directory_claim_manifest_last`.
실제 host에서 실행한 것은 **macOS APFS만**이다. Linux/Windows 실제 publication은
미검증이며 fallback은 macOS에서 native API 부재/ENOSYS/EINVAL을 monkeypatch해 검증했다.
Fallback 동안에는 불완전한 directory가 보일 수 있으므로 manifest 존재만 보고 읽지 말고
완료된 bundle에 `--verify-bundle`을 먼저 실행한다. 실패한 claim directory/staging은 증거로
보존되며 새 출력 경로로 재실행한다. **Exported does not mean the grid is accepted.**
Warnings가 있는 후보 grid도 export할 수 있으나 train/search의 기존 gate는 기본적으로
차단하며 grid 승인이나 전기적 검증이 되지 않는다.
