# psdl1 사용 안내 — PNG 폴더에서 검색까지

DL1은 새 제품과 **원본 형상·전극 연결**이 비슷한 기존 제품을 찾는다. 결과는 비교군 후보 목록이며 원인 판정이나 위험 예측이 아니다.

이 문서는 실행 순서대로 쓰여 있다. 처음이면 1~6장을 차례로 따라 하면 된다. 세부 규칙의 근거는 [data_contract.md](data_contract.md), [training.md](training.md), [validation.md](validation.md)에 있다.

먼저 알아 둘 것:

- 입력은 원본 layout 형상과 CNT 이미지뿐이다. hotspot, CA, fab 값은 받지 않는다(있으면 오류).
- 지금까지 검증한 것은 **합성 PNG뿐**이다. 실제 자료 호환성, 검색 품질, 전기적 연결은 검증되지 않았다.
- 실제로 실행해 본 운영체제는 **macOS뿐**이다. Windows와 Linux에서는 한 번도 실행되지 않았다. Windows에서는 2장의 설치 확인(테스트 실행)을 먼저 한다.

## 1. 전체 순서

| 순서 | 하는 일 | 명령 | 필요한 것 | 만들어지는 것 |
|---|---|---|---|---|
| 0 | 설치와 확인 | `pip install`, `pytest` | Python 3.12, wheel 폴더 | `.venv` |
| 1 | PNG 폴더 준비 | (사용자) | 모델별 전극·CNT PNG | 입력 폴더 |
| 2 | 검사 | `psdl1.import_png` | 입력 폴더 | 보고서, preview |
| 3 | 연결 확인 | (사용자) `preview.html` | 2의 결과 | 규칙이 맞는지 판단 |
| 4 | grid 고르기 | `psdl1.import_png --inspect-grids` | 후보 grid 숫자 | grid별 경고 |
| 5 | 내보내기 | `psdl1.import_png --export` | 계보 파일, grid | `manifest.json`, NPY |
| 6 | 묶음 확인 | `--verify-bundle`, `psdl1.data` | 5의 결과 | 확인 결과, 격자 보고서 |
| 7 | 학습 | `psdl1.train` | manifest, 분할, (정답) | `dl1.pt` |
| 8 | 검색 | `psdl1.search` | checkpoint, manifest 2개 | 검색 결과 JSON |
| 9 | 평가 | `psdl1.evaluate` | 정답이 있을 때 | 평가 JSON |

연결만 확인하려면 2~3까지만 하면 된다. 2~4는 계보 정보와 grid 없이 실행된다.

## 2. 설치와 확인

Python 3.12 기준이다. 네트워크 내려받기는 쓰지 않는다. 승인된 wheel 폴더에서 설치한다.

필요한 패키지:

| 패키지 | 용도 | 비고 |
|---|---|---|
| numpy, torch | 필수 | `requirements.lock`에 개발 기기(macOS)의 버전이 적혀 있다. 대상 OS용 wheel로 같은 버전을 설치한다 |
| Pillow **12.3.0** | PNG 입력 | 디코더 검증이 이 버전에서만 이루어졌다. 다른 버전이면 실행은 되지만 보고서에 경고가 남는다 |
| pytest | 설치 확인 | |
| hatchling | `pip install -e .`을 쓸 때만 | 아래 "설치 없이 실행"을 쓰면 필요 없다 |

### Windows (명령 프롬프트)

```bat
cd D:\work\particle-sim-dl1
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --no-index --find-links D:\wheels numpy torch Pillow==12.3.0 pytest

rem 패키지를 설치하지 않고 실행 (hatchling 불필요)
set PYTHONPATH=%CD%\src
set PYTHONDONTWRITEBYTECODE=1

rem 설치 확인: 실제 자료를 넣기 전에 반드시 한 번 실행
.venv\Scripts\python -m pytest -p no:warnings
```

`set PYTHONPATH=...`은 창을 새로 열 때마다 다시 입력한다. 이후 명령의 `PY`는 `.venv\Scripts\python`을 뜻한다.

### macOS / Linux

```bash
cd ~/work/particle-sim-dl1
uv venv --python 3.12 .venv
uv pip install --no-index --find-links /approved/wheels -r requirements.lock
uv pip install --no-index --find-links /approved/wheels -e ".[png]"
.venv/bin/python -m pytest -p no:warnings
```

이후 명령의 `PY`는 `.venv/bin/python`을 뜻한다.

### 테스트 결과 보는 법

개발 기기에서는 `484 passed, 1 skipped`가 나온다(약 80초). 건너뛴 1개는 선택형 학습 테스트다. Windows에서 실패가 나오면 실패한 테스트 이름과 메시지를 그대로 전달한다. 실제 자료를 넣기 전에 고칠 수 있다.

## 3. PNG 폴더 준비

### 폴더 규칙

```text
<입력 폴더>\
  <모델명>\
    <전극명>\
      1-pxl.png
      2-SD2.png
    <전극명>\
      4-GAT.png
    cno\
      006_OCNT.png
```

| 항목 | 규칙 |
|---|---|
| 모델명 폴더 | 그대로 제품 이름(`product_id`)이 된다 |
| 전극명 폴더 | 그대로 전극 이름이자 역할 이름(`role`)이 된다 |
| `cno` 폴더 | CNT 전용. 소문자 `cno`여야 한다. `CNO`처럼 대소문자가 다르면 오류다 |
| 파일명 | `<순번>-<layer>.png` 또는 `<순번>_<layer>.png`. 숫자 뒤 첫 `-`/`_`가 구분자이고 나머지 전체가 layer 이름이다 |
| 순번 | 모델 전체에서 공통인 위→아래 순서. 1이 가장 위. `006`과 `6`은 같은 숫자다 |
| 같은 순번 | 한 모델 안에서 같은 순번은 같은 layer 이름이어야 한다 |
| 이름 비교 | 글자 그대로 비교한다. 대소문자, 한글 조합 방식(NFC/NFD)이 다르면 다른 이름이다. 별칭을 자동으로 합치지 않는다 |
| 그 밖의 폴더 | 입력 폴더 바로 아래의 모든 폴더는 모델로 본다. 모델이 아닌 폴더가 있으면 전체가 오류다 |

### 이미지 규칙

| 항목 | 규칙 |
|---|---|
| 색 | 검정 = 패턴, 흰색 = 배경. 정확히 검정/흰색만 허용한다 |
| 크기 | 한 모델 안의 모든 PNG가 같은 가로·세로여야 한다. 가로·세로 모두 1000픽셀보다 커야 한다 |
| 해상도 | 0.1 µm/픽셀로 고정 |
| 범위 | 2×2 영역 + 상하좌우 500픽셀(실제 패턴). 전체를 그대로 쓴다. 자르지 않는다 |
| 큰 이미지 | 약 8,947만 픽셀(약 9459×9459)을 넘으면 `--png-max-image-pixels <숫자>`를 직접 줘야 한다 |

### 거부되는 PNG와 조치

아래 파일은 파일별 오류로 거부된다. 프로그램이 몰래 고치지 않는다. **단순한 불투명 흑백 PNG로 다시 저장**하면 된다.

- 회색이나 색이 있는 픽셀
- 투명도(알파 채널, tRNS, 255가 아닌 팔레트 알파)
- 16비트 이미지
- APNG(프레임이 하나여도)
- 파일 끝(IEND) 뒤에 남은 바이트, 잘린 파일, CRC 오류
- 방향 정보(EXIF/XMP orientation)가 기본값이 아닌 파일

## 4. 검사 — 연결 확인과 grid 고르기

### 4.1 검사 실행

계보 정보와 grid 없이 실행된다. 출력 폴더는 **입력 폴더 밖의 새 폴더**여야 한다. 이미 있는 폴더는 비어 있어도 거부한다.

```bat
PY -m psdl1.import_png --root D:\png_root --out D:\out\inspect1
```

Windows에서는 출력 경로를 짧게 잡는다. 생성되는 파일 이름이 70자 이상이라 경로가 깊으면 경로 길이 제한(260자)에 걸릴 수 있다(추정이며 실제로 확인되지는 않았다).

종료 코드:

| 코드 | 뜻 |
|---|---|
| 0 | 성공 |
| 2 | 입력 오류, 내보내기 조건 미충족, 출력 경로 거부, 실패 |

화면에는 JSON이 두 줄 나온다. 첫 줄(`resource_inventory`)은 용량 추정과 이름 목록, 둘째 줄(`completed`)은 결과다.

### 4.2 만들어지는 파일

```text
D:\out\inspect1\
  import-report.json       보고서 (dl1_png_import_v2)
  preview.html             연결 확인 화면 (인터넷 없이 열림)
  preview\                 preview가 쓰는 이미지
  bundle-integrity.json    위 파일들의 SHA-256 목록
```

### 4.3 preview로 연결 확인

`preview.html`을 브라우저로 연다.

전체 그림의 표시:

| 표시 | 뜻 |
|---|---|
| 초록 원 | 연결이 정해짐 |
| 주황 사각형+원 | 연결은 정해졌지만 **확인 필요**(중간 layer를 건너뜀) |
| 빨간 X | 미해결 |
| 보라 사각형 | 중앙 2×2 참고 영역. 참고 표시일 뿐 자르는 데 쓰지 않는다 |

CNT 덩어리별 확대 그림(주변 64픽셀 포함)의 색:

| 색 | 뜻 |
|---|---|
| 파랑 | 선택된 위쪽 전극 |
| 분홍 | 선택된 아래쪽 전극(미해결이면 그 layer의 후보 전부) |
| 회색 | 그 밖의 전극 |
| 검정 테두리 | CNT 덩어리 |
| 노랑 | CNT·위 전극·아래 전극이 모두 겹치는 영역 |
| 초록 십자 | 연결 위치(끝점) |

확대 그림은 사람이 보는 용도다. 모델 입력은 그대로 전체 frame이다.

### 4.4 연결을 정하는 규칙

CNT 덩어리 하나마다 다음 순서로 정한다.

1. CNT와 겹치는 전극을 찾는다.
2. CNT 순번보다 작은 쪽에서 가장 가까운 layer가 위쪽, 큰 쪽에서 가장 가까운 layer가 아래쪽이다. 더 먼 layer는 쓰지 않는다.
3. 위·아래 각각 전극이 정확히 하나여야 한다.
4. CNT·위 전극·아래 전극이 같은 위치에서 겹쳐야 한다.
5. 겹친 영역 안의 한 픽셀을 연결 위치로 쓴다.

덩어리는 상하좌우로 붙은 픽셀끼리 묶는다(대각선은 별개).

미해결 사유(`unresolved_reasons`)와 뜻:

| 사유 | 뜻 |
|---|---|
| `missing_upper_overlap` / `missing_lower_overlap` | 위쪽/아래쪽에 겹치는 전극이 없음 |
| `multiple_upper_electrodes_on_selected_layer` / `multiple_lower_...` | 선택된 layer에 겹치는 전극이 둘 이상 |
| `same_bound_element` | 위와 아래가 같은 전극·layer |
| `equal_stack_overlap` | CNT와 같은 순번의 전극이 겹침 |
| `empty_common_intersection` | 위·아래 전극이 CNT의 서로 다른 위치에서만 겹침 |
| `disconnected_common_intersection_possible_multicontact` | 겹친 영역이 둘 이상으로 떨어져 있음(CNT 하나에 연결이 여러 개일 가능성) |

미해결 덩어리는 연결 관계를 만들지 않는다. 형상은 전체 CNT 이미지에 그대로 남는다.

아직 사용자 확정이 아닌 해석:

- **중간 layer를 건너뛴 연결**을 관계로 인정하는 것. preview에서 주황으로 표시된 연결이 실제로 맞는지 확인이 필요하다.
- **같은 layer에 후보 전극이 여럿일 때** 미해결로 두는 것.

자동 추출은 전기적 연결 검증이 아니다. 보고서의 `electrical_verification`은 항상 `not_performed`다.

### 4.5 보고서에서 볼 곳 (`import-report.json`)

| 위치 | 내용 |
|---|---|
| `export_status` | `not_requested`(검사만), `blocked`(조건 미충족), `exported`, `failed`, `interrupted` |
| `pending_prerequisites` | 내보내기를 막은 사유. 예: `explicit_metadata_missing` |
| `inventory.models[].errors` / `warnings` | 파일별 오류와 경고(무시된 파일, 빈 전극 폴더 등) |
| `inventory.models[].stack_table` | 모델의 순번–layer 표 |
| `inventory.models[].resource_estimate` | 용량 추정. 아래 4.6 참고 |
| `inventory.role_union`, `electrode_layer_union` | 전체 모델의 전극 이름, 전극 layer 이름 |
| `inventory.unique_roles`, `unique_electrode_layers` | 한 모델에만 있는 이름 |
| `duplicate_geometry` | 형상이 완전히 같은 파일/모델(중복 제품 가능성). 자동으로 합치지 않는다 |
| `extraction[].summary` | 덩어리 수, 해결 수, 미해결 수, 확인 필요 수 |
| `extraction[].connectivity_observation` | 추출 범위: `complete`(전부 해결), `partial`, `unavailable`(해결 없음) |
| `extraction[].components[]` | 덩어리별 위치(`bbox_xyxy_px`), 후보, 선택, 미해결 사유, `relation` |
| `relation.representative_xy_px` | 연결 위치(원본 픽셀) |
| `relation.needs_confirmation` | 건너뛴 layer가 있으면 true |
| `relation.skipped_upper_layers` / `skipped_lower_layers` | 건너뛴 전극 layer와 CNT layer |
| `grid_evidence` | grid별 확인 결과(4.7) |
| `publication_method` | 묶음을 게시한 방식(8장) |

**`complete`는 추출 범위만 뜻한다. 전기적 검증이 아니다**(보고서 문구: complete means extraction coverage only; electrical verification not performed). 이 값은 모델 입력에도 들어가므로, 같은 형상이라도 `partial`과 `complete`는 모델에 다른 입력이 된다.

### 4.6 용량 추정

기존 manifest 형식은 연결 하나마다 전체 크기의 mask 파일이 하나씩 필요하다. CNT가 많으면 파일과 메모리가 그만큼 늘어난다.

| 필드 | 뜻 |
|---|---|
| `frame_pixels` | 이미지 픽셀 수 |
| `electrode_asset_count`, `cnt_component_count`, `resolved_relation_count` | 전극 파일 수, CNT 덩어리 수, 해결된 연결 수 |
| `generated_mask_asset_count` | 내보낼 mask 파일 수 |
| `projected_bool_npy_export_bytes` | 내보낼 NPY 전체 크기(바이트) |
| `projected_load_manifest_retained_float32_mask_bytes` | 읽을 때 메모리에 남는 mask 크기(바이트) |
| `all_components_resolved` | 덩어리가 모두 해결됐을 때의 값 |

이 값은 최대 메모리나 실행 시간 예측이 아니다. 디코딩 중에는 픽셀당 약 17~18바이트가 일시적으로 더 쓰인다. 자원이 충분한지는 이 숫자를 보고 사용자가 판단한다. 프로그램이 정한 한도는 없다.

### 4.7 grid 고르기

grid는 모델에 넣기 전에 전체 이미지를 줄이는 정사각형 한 변의 칸 수다. 모든 전극·CNT 이미지가 같은 비율로 줄고 남는 부분은 빈칸으로 채워진다.

- 작으면 가는 선이 사라지거나 떨어진 패턴이 붙는다. 대신 빠르다.
- 크면 형상이 보존되지만 메모리와 시간이 늘어난다.
- 연결 추출은 항상 원본 해상도에서 하므로 grid와 무관하다.
- 학습한 checkpoint는 그 grid에 묶인다. 검색할 제품도 같은 grid로 내보낸다.

기준: 가장 가는 선이 줄어든 뒤에도 2칸 이상 남아야 한다.

```text
grid ≥ 2 × (이미지 긴 변 픽셀 수) ÷ (가장 가는 선폭 픽셀 수)
```

후보를 직접 넣어 비교한다. 기본값은 없고, 프로그램이 골라 주지 않는다.

```bat
PY -m psdl1.import_png --root D:\png_root --out D:\out\grid1 --inspect-grids 256 512 1024
```

`grid_evidence`에서 grid별로 다음을 본다.

| 필드 | 뜻 |
|---|---|
| `thin_line_warning` | 가장 짧은 선이 약 2칸 미만 |
| `bridge_or_component_loss` | 줄인 뒤 덩어리 수가 바뀜(끊김, 사라짐, 붙음) |
| `reasons` | 경고 사유 목록 |
| `transform` | 실제 줄인 크기, 여백, 배율, 줄인 뒤의 µm/칸 |

경고가 없는 가장 작은 grid가 출발점이다. 경고가 있는 grid도 내보낼 수는 있지만 학습·검색에서 기본으로 막힌다. **내보냈다고 grid가 승인된 것은 아니다.**

## 5. 내보내기 — manifest 자동 생성

### 5.1 계보 파일 만들기

모델마다 계열 이름(`lineage_id`)과 설계 버전(`design_version`)을 사용자가 적는다. 폴더명에서 추정하지 않는다. 같은 계열의 변형은 같은 `lineage_id`를 쓴다(학습 분할에서 누수를 막는 데 쓰인다).

`D:\meta\meta.json`:

```json
{
  "metadata_schema_version": "dl1_png_metadata_v1",
  "models": {
    "MODEL-A": {"lineage_id": "FAMILY-1", "design_version": "v1"},
    "MODEL-B": {"lineage_id": "FAMILY-2", "design_version": "v1"}
  }
}
```

`models`의 키는 모델 폴더명과 글자 그대로 같아야 한다. 입력 폴더의 모든 모델이 들어 있어야 하고, 없는 모델 이름이 있으면 오류다. [예시 파일](../configs/png-metadata.example.json)의 값은 합성 예시다.

### 5.2 내보내기 실행

```bat
PY -m psdl1.import_png --root D:\png_root --out D:\out\bundle1 --export --metadata D:\meta\meta.json --candidate-grid 512
```

- 계보 파일이나 grid가 빠지면 종료 코드 2로 끝나고 `manifest.json`을 만들지 않는다. 보고서와 preview만 남는다.
- 모델 하나라도 입력 오류가 있으면 전체를 내보내지 않는다. 일부만 내보내지 않는다.
- 미해결 CNT는 오류가 아니다. 그 모델은 `partial` 또는 `unavailable`로 내보낸다.

만들어지는 것:

```text
D:\out\bundle1\
  manifest.json            기존 형식(dl1_shape_v1_draft). 그대로 학습·검색에 쓴다
  assets\*.npy             전극, 관측 범위, 전체 CNT, 연결별 mask (bool)
  import-report.json
  preview.html, preview\
  bundle-integrity.json
```

manifest의 `unit_size_um`에는 **전체 이미지 범위**(가로×0.1, 세로×0.1 µm)가 들어간다. 단일 unit 크기가 아니다.

### 5.3 묶음 확인

```bat
PY -m psdl1.import_png --verify-bundle D:\out\bundle1
PY -m psdl1.data D:\out\bundle1\manifest.json --report D:\out\grid-report1.json
```

`--verify-bundle`은 원본 PNG 없이도 파일이 바뀌거나 빠지거나 추가됐는지 확인한다. 묶음을 복사한 뒤에도 실행할 수 있다. 손상 탐지일 뿐, 내용이 옳다는 인증은 아니다. 탐색기가 만든 숨김 파일이 들어가도 실패로 나온다.

## 6. 학습, 검색, 평가

### 6.1 분할 파일 (`split.json`)

```json
{
  "schema_version": "dl1_split_v1",
  "train": ["MODEL-A", "MODEL-B", "MODEL-C"],
  "validation": ["MODEL-D"],
  "test": ["MODEL-E", "MODEL-F"]
}
```

모든 제품이 정확히 한 번 들어가야 하고, 같은 `lineage_id`의 제품은 한 구획에만 있어야 한다. `train`은 비어 있으면 안 된다. 자동으로 만들어 주지 않는다.

### 6.2 정답 파일 (`relations.csv`, 선택)

머리글 없는 CSV다. 형식은 아직 초안이다.

```text
MODEL-A,MODEL-B,similar
MODEL-A,MODEL-C,different
```

- 적지 않은 쌍은 "모름"으로 두고 학습에서 뺀다.
- 같은 쌍에 similar와 different가 함께 있으면 오류다.
- different 쌍이 하나도 없으면 학습을 시작하지 않는다.
- 정답이 없으면 `--unlabelled-negatives`로 "적지 않은 쌍은 다르다"고 가정할 수 있다. 실제로 비슷한 제품을 밀어낼 수 있는 실험 옵션이다. 같은 계열끼리는 이 경우에도 "모름"이다.

### 6.3 학습

```bat
PY -m psdl1.train --manifest D:\out\bundle1\manifest.json --split D:\data\split.json --relations D:\data\relations.csv --out D:\runs\dl1 --epochs 100 --batch-size 8 --lr 0.0003

rem 이어서 학습 (--epochs는 총 목표 횟수)
PY -m psdl1.train --manifest D:\out\bundle1\manifest.json --split D:\data\split.json --relations D:\data\relations.csv --out D:\runs\dl1 --resume D:\runs\dl1\dl1.pt --epochs 150
```

설정 파일(`--config configs/train.example.json`)로도 줄 수 있다. 명령줄 값이 우선한다.

| 옵션 | 뜻 | 비고 |
|---|---|---|
| `--manifest`, `--split`, `--out` | 입력 manifest, 분할, 출력 폴더 | 필수 |
| `--relations` | 정답 CSV | 없으면 `--unlabelled-negatives` 필요 |
| `--epochs`, `--batch-size`, `--lr`, `--temperature`, `--seed` | 학습 값 | 예시 값은 권장값이 아니다 |
| `--grid` | 격자 | 생략하면 manifest의 `candidate_grid`. 다른 값이면 `--allow-grid-override` 필요 |
| `--allow-grid-check-failure` | 가는 선·끊김 경고를 무시하고 진행 | 기본 꺼짐. 켜면 checkpoint에 기록된다 |
| `--use-unit-size` | 이미지 범위 크기를 모델에 조건으로 넣기 | 기본 꺼짐 |
| `--backbone` | `tiny`(기본) 또는 timm 이름 | 내려받기 없음 |
| `--weights` | 로컬 가중치 파일 | 기본 없음. 공개 ConvNeXt V2 파일은 검증하지 않았다 |
| `--augmentation` | `none`(기본) 또는 `intensity` | 자르기·이동은 지원하지 않는다 |
| `--device` | `cpu`(기본) 또는 `cuda` | |
| `--resume` | 이어서 학습할 checkpoint | 설정·입력·분할·코드가 바뀌면 거부 |

5–10개 제품은 가져오기와 연결 확인을 위한 시험 규모다. 학습에 충분한 수나 검색 품질의 근거가 아니다.

### 6.4 검색

manifest 두 개를 받는다. 둘 다 같은 grid로 내보낸 것이어야 한다.

```bat
PY -m psdl1.search --ckpt D:\runs\dl1\dl1.pt --query-manifest D:\out\query1\manifest.json --query NEW-MODEL --candidates-manifest D:\out\bundle1\manifest.json --topk 5 --out D:\out\search1.json
```

**이름 제한**: 전극 폴더명이 그대로 역할 이름이 된다. 학습에 없던 전극 이름이나 layer 이름을 가진 제품은 검색할 수 없다. 찾으려는 제품이 그러면 결과가 `unavailable`이 되고, 후보가 그러면 사유와 함께 `excluded`에 들어간다. 같은 이름은 같은 것으로 취급되지만, 두 전극이 실제로 같은 기능이라는 뜻은 아니다. 별칭을 맞추는 기능은 없다.

### 6.5 평가 (정답이 있을 때)

```bat
PY -m psdl1.evaluate --ckpt D:\runs\dl1\dl1.pt --manifest D:\out\bundle1\manifest.json --split D:\data\split.json --relations D:\data\relations.csv --topk 5 --out D:\out\evaluation1.json
```

`--hard-negatives <파일>`로 형상은 같고 연결 위치만 다른 제품 쌍 목록(`[["MODEL-E", "MODEL-F"]]`)을 줄 수 있다.

## 7. 출력 형식

### 7.1 격자 보고서 (`psdl1.data --report`)

| 필드 | 뜻 |
|---|---|
| `product_id`, `grid`, `source_shape_yx`, `scale` | 어느 제품을 어떤 격자로 얼마나 줄였는지 |
| `transform` | 실제 줄인 크기, 여백, 가로·세로 배율, 줄인 뒤의 µm/칸 |
| `elements[]` | 형상마다 줄이기 전후의 픽셀 수, 덩어리 수, 가장 짧은 선 길이와 격자 기준 추정값 |
| `elements[].thin_line_warning` | 가장 짧은 선이 약 2칸 미만 |
| `elements[].bridge_or_component_loss` | 줄인 뒤 덩어리 수가 바뀜 |
| `reasons` | 경고 사유 목록 |
| `connectivity_preserving` | 항상 false. 이 보고서는 연결 보존을 보증하지 않는다 |

경고가 있으면 학습이 기본으로 막힌다. grid를 키우거나, 확인 후 `--allow-grid-check-failure`로 진행한다.

### 7.2 학습 출력 (`--out` 폴더)

| 파일 | 내용 |
|---|---|
| `dl1.pt` | checkpoint. 매 epoch 끝에 교체된다. 모델, optimizer, 난수 상태, 설정 fingerprint, 입력·분할·코드 hash, 격자 근거, 학습 쌍 통계 |
| `split.json` | 실제로 쓴 분할 |
| `history.json` | epoch별 손실 |

### 7.3 검색 결과 (`psdl1.search --out`)

```json
{
  "query_id": "NEW-MODEL",
  "status": "available",
  "reason": null,
  "rankings": [
    {"product_id": "MODEL-B", "score": 0.912345,
     "connectivity_status": "partial", "input_identity": "e3cf…"}
  ],
  "excluded": [{"product_id": "MODEL-G", "reason": "grid_check_failed"}],
  "candidate_count": 12, "eligible_count": 11,
  "grid": 512
}
```

| 필드 | 뜻 |
|---|---|
| `status` | `available` 또는 `unavailable` |
| `reason` | unavailable일 때 사유. 예: `no_candidates`, `no_eligible_candidates`, `grid_check_failed`, 학습에 없던 role/layer |
| `rankings[]` | 유사도 높은 순. `score`는 cosine(−1~1)을 소수 6자리로 반올림한 값. 같은 점수는 제품 이름 순 |
| `rankings[].connectivity_status` | 그 제품의 연결 정보 상태(`available`/`partial`/`unavailable`) |
| `rankings[].input_identity` | 입력 형상·연결의 fingerprint |
| `excluded[]` | 후보에서 뺀 제품과 사유 |
| `grid`, `grid_selection`, `grid_gate` | 실제로 쓴 격자와 제품별 격자 확인 근거 |
| `fingerprint` | 모델 설정 fingerprint. 수치 저장소에서 결과를 읽을 때 대조한다 |
| `approval` | 무엇이 확정되고 무엇이 미정인지 |

후보가 없다는 것은 "비슷한 제품이 없다"는 뜻일 뿐 위험이 0이라는 뜻이 아니다. 전체 제품 유사도이며, 제품 사이에서 대응하는 위치를 찾아 주지 않는다.

### 7.4 평가 결과 (`psdl1.evaluate --out`)

| 필드 | 뜻 |
|---|---|
| `recall_at_k`, `baseline_recall_at_k` | 모델과 기준선(정규화 이미지 거리)의 Recall@k 평균. 계산할 query가 없으면 null |
| `evaluated_queries`, `unavailable_queries` | 계산된 query 수와 빠진 수 |
| `queries[]` | query별 결과와 모델·기준선 순위 |
| `hard_negative_evaluation` | 어려운 반례의 순위·점수. 목록이 없으면 `unavailable`/`none_supplied` |
| `performance_acceptance` | 항상 `not_established`. 수락 기준은 아직 정해지지 않았다 |

평가는 시험 구획의 제품만 쓴다.

## 8. 문제가 생겼을 때

| 증상 | 원인과 조치 |
|---|---|
| `output must be outside the input tree` | 출력 폴더가 입력 폴더 안이다(대소문자만 다른 경로 포함). 입력 폴더 밖 경로를 준다 |
| 출력 폴더가 이미 있다는 오류 | 기존 폴더는 덮어쓰지 않는다. 새 이름을 준다 |
| `pixels must be exactly black or white` | 회색·색 픽셀이 있다. 흑백으로 다시 저장한다 |
| `nonopaque PNG pixels`, tRNS 오류 | 투명도가 있다. 불투명으로 다시 저장한다 |
| `APNG is unsupported` | APNG 청크가 있다. 일반 PNG로 다시 저장한다 |
| `shape mismatch` | 한 모델 안에 크기가 다른 PNG가 있다 |
| `conflicting stack` | 같은 순번에 다른 layer 이름이 있다 |
| `duplicate normalized binding` | 같은 전극·layer 파일이 둘이다(예: `6-X.png`와 `006_X.png`) |
| `unsupported directory layout` | 모델/전극 폴더 구조가 아닌 폴더가 있다. 이전 실행의 출력이 입력 폴더 안에 남은 경우도 이렇게 나온다 |
| DecompressionBomb 오류 | 이미지가 Pillow 기본 한도를 넘는다. `--png-max-image-pixels <픽셀 수>`를 직접 준다 |
| `explicit_metadata_missing` | 계보 파일이 없거나 빠진 모델이 있다(5.1) |
| Pillow가 없다는 오류 | Pillow 12.3.0을 wheel 폴더에서 설치한다. 기존 NPY 경로는 Pillow 없이 동작한다 |
| 검색 결과가 `unavailable`, 사유가 role/layer | 학습에 없던 전극·layer 이름이다(6.4) |
| 학습이 grid 경고로 막힘 | grid를 키워 다시 내보내거나, 확인 후 `--allow-grid-check-failure` |
| 실행 중 중단됨 | 출력 폴더 옆에 `.<이름>.staging-…` 폴더가 남는다. 최종 결과가 아니다. 새 출력 경로로 다시 실행한다 |
| `--verify-bundle` 실패 | 파일이 바뀌었거나 추가됐다. 숨김 파일(`.DS_Store` 등)도 해당한다 |

### 묶음 게시와 운영체제

마지막에 임시 폴더를 최종 폴더로 옮기는 방식은 보고서의 `publication_method`에 기록된다.

| 값 | 뜻 |
|---|---|
| `darwin_rename_excl` | macOS. 한 번에 옮긴다. **실제로 실행된 유일한 방식** |
| `linux_renameat2_noreplace` | Linux(glibc 2.28 이상). 실제 호스트에서 실행된 적 없음 |
| `exclusive_directory_claim_manifest_last` | 그 밖(Windows, 오래된 Linux). 최종 폴더를 먼저 만들고 `manifest.json`을 마지막에 쓴다. 모의 테스트만 통과했고 실제 호스트에서 실행된 적 없음 |

마지막 방식에서는 쓰는 도중 폴더가 잠시 불완전하게 보일 수 있다. `manifest.json`이 있다는 것만 보고 쓰지 말고 `--verify-bundle`을 먼저 실행한다. 이 방식으로 성공하면 임시 폴더가 그대로 남으므로 확인 후 지워도 된다.

알려진 제한: 프로세스를 강제 종료(작업 관리자, SIGTERM)하면 게시 전 임시 폴더에 `manifest.json`이 남을 수 있다. 임시 폴더는 최종 결과가 아니므로 쓰지 않는다.

## 9. 아직 정해지지 않은 것

| 항목 | 상태 |
|---|---|
| 중간 layer를 건너뛴 연결을 인정할지 | 구현 해석. preview에서 확인 필요 |
| 같은 layer에 후보 전극이 여럿일 때의 규칙 | 미정. 지금은 미해결로 둔다 |
| 실제 grid와 최소 선폭 | 폐쇄망에서 4.7로 정한다 |
| 계열·버전 정보, 역할/layer 별칭 대응 | 사용자가 공급. 별칭 대응 기능 없음 |
| 정답 형식과 수락 기준(G6) | 미정 |
| 실제 PNG의 모드·크기, 관측되지 않은 영역 유무 | 폐쇄망에서 확인. 지금은 전체가 관측됐다고 가정한다 |
| Windows/Linux 실행 | 미검증 |
| 공개 가중치 사용 여부와 licence | 미정 |

## 10. NPY manifest를 직접 쓰는 경우

PNG 가져오기를 쓰지 않고 mask를 `.npy`로 직접 준비할 수도 있다. 이 경로는 Pillow가 필요 없다.

### mask 파일

| 항목 | 규칙 |
|---|---|
| 형식 | `.npy`, 2차원 배열 |
| 값 | 0/1(bool 포함) 또는 0/255. 0이 아닌 값이 형상이다(PNG와 반대로 **1이 형상**) |
| 거부 | 확률값, 음수, NaN, 1과 255 혼합, 3차원 |
| 크기 | 그 제품의 `frame.shape_yx`와 같아야 한다 |
| 경로 | manifest 폴더 기준 상대 경로. 절대 경로, `..`, 폴더 밖 symlink는 거부 |

### `manifest.json` — `dl1_shape_v1_draft`

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

위 숫자는 형식을 보이는 합성 예시다. 0.05는 예시 값이며 실제 PNG 해상도(0.1)와 무관하다.

| 필드 | 뜻 | 규칙 |
|---|---|---|
| `candidate_grid` | 격자 한 변의 칸 수 | 필수. 기본값이 없다 |
| `product_id` | 제품 이름 | manifest 안에서 유일 |
| `lineage_id` | 같은 계열을 묶는 이름 | 분할 누수 방지에 쓴다 |
| `unit_size_um` | 전체 frame 범위 [가로, 세로] | `shape_yx × pixel_pitch_um`과 같아야 한다 |
| `frame.pixel_pitch_um` | 1픽셀의 µm | manifest 안 모든 제품이 같아야 한다 |
| `frame.shape_yx` | [세로, 가로] 픽셀 수 | 제품마다 달라도 된다 |
| `valid` | 관측된 범위 mask | 관측 안 된 곳은 0 |
| `elements[]` | 전극 × 실제 layer 하나씩 | 한 제품 안에서 role × layer 조합은 하나만 |
| `elements[].observation` | `observed`(mask 경로) 또는 `missing`(`mask: null`) | missing이 있으면 그 제품은 계산에서 빠진다 |
| `connectivity_observation` | `complete` / `partial` / `unavailable` | 연결 정보가 없으면 `connections: []`와 `unavailable` |
| `connections[]` | CNT 또는 VIA 하나 | 끝점 2개, 서로 다른 element |
| `endpoints[].xy_um` | 연결 위치(원본 좌표, µm) | 해당 전극 형상과 연결 mask 안에 있어야 한다 |
| `cnt` (선택) | 제품 전체 CNT mask | 끝점 정보를 대신하지 못한다 |

`hotspot`, `CA`, `fab`이라는 이름의 필드, 알 수 없는 필드, 중복 key는 모두 오류다.

## 11. 수치 저장소와의 연결

검색 결과 JSON은 `particle-sim-compare`의 `simcompare.dl_results`가 파일로 읽어 원본 CA·line/pixel 값 옆에 붙인다. DL 점수는 수치 값을 바꾸지 않는다. 절차는 그 저장소의 `docs/handoff/commands.md`에 있다.
