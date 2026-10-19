> **Windows에서 uv 네트워크 연결이 안 되면:** [Windows 폐쇄망 설치](WINDOWS_OFFLINE.md)의 준비된 CPU wheel 묶음과 명시적 `python.exe` 명령을 사용하십시오.

# psdl1 사용 안내 — PNG 폴더에서 검색까지

DL1은 새 제품과 **원본 형상·전극 연결**이 비슷한 기존 제품을 찾는다. 결과는 비교군 후보 목록이며 원인 판정이나 위험 예측이 아니다.

이 문서는 실행 순서대로 쓰여 있다. 처음이면 1~6장을 차례로 따라 하면 된다. 기본 경로는 **image_only**이며 순서 파일을 쓰지 않는다. 선택적 relations 모드는 11장에서 별도로 준비한다. 세부 규칙의 근거는 [data_contract.md](data_contract.md), [training.md](training.md), [validation.md](validation.md)에 있다.

먼저 알아 둘 것:

- 순서 파일 없이 CNT를 이미지 채널로만 사용한다. 모든 CNT foreground의 합집합을 `cnt`로 내보내며 `connections: []`, `connectivity_observation: "unavailable"`이다. 연결 위치나 상하 위치를 추정하지 않는다.
- 입력은 원본 layout 형상과 CNT 이미지뿐이다. hotspot, CA, fab 값은 받지 않는다(있으면 오류).
- 지금까지 검증한 것은 **합성 PNG뿐**이다. 실제 자료 호환성, 검색 품질, 전기적 연결은 검증되지 않았다.
- 실제로 실행해 본 운영체제는 **macOS뿐**이다. Windows와 Linux에서는 한 번도 실행되지 않았다. Windows에서는 2장의 설치 확인(테스트 실행)을 먼저 한다.

## 1. 전체 순서

| 순서 | 하는 일 | 명령 | 필요한 것 | 만들어지는 것 |
|---|---|---|---|---|
| 0 | 설치와 확인 | `py -3.12 -I install_windows_bundle.py` | Python 3.12 x64, 동봉 wheel | `.venv-cmd` |
| 1 | PNG 폴더 준비 | (사용자) | 모델별 전극·CNT PNG | 입력 폴더 |
| 2 | 검사 | `psdl1.import_png` | 입력 폴더 | 보고서, preview |
| 3 | 형상 확인(image_only) | (사용자) `preview.html` | 2의 결과 | CNT 형상·여백 확인 |
| 4 | grid 고르기 | `psdl1.import_png --inspect-grids` | 후보 grid 숫자 | grid별 경고 |
| 5 | 계보 파일 준비 | (사용자) metadata JSON | 모든 모델의 계열·설계 버전 | 계보 파일 |
| 6 | 내보내기 | `psdl1.import_png --export` | 계보 파일, grid | `manifest.json`, NPY |
| 7 | 묶음 확인 | `--verify-bundle`, `psdl1.data` | 6의 결과 | 확인 결과, 격자 보고서 |
| 8 | 학습 | `psdl1.train` | manifest, 분할, (정답) | `dl1.pt` |
| 9 | 검색 | `psdl1.search` | checkpoint, manifest 2개 | 검색 결과 JSON |
| 10 | 평가 | `psdl1.evaluate` | 정답이 있을 때 | 평가 JSON |

형상만 확인하려면 2~3까지만 하면 된다. 연결 추출이 필요할 때만 11장의 선택적 경로를 쓴다. 2~3은 계보 정보와 grid 없이 실행된다. 4의 grid 비교에는 사용자가 후보 숫자를 직접 준다.

## 2. 설치와 확인

Windows 설치는 **Git Bash 또는 CMD(명령 프롬프트)**에서 설치된 Python을 직접 실행합니다.
`.cmd` 파일 실행이나 가상환경 활성화는 필요하지 않습니다. Python 3.12 x64,
pip/venv와 VC++ x64 런타임은 사전 설치되어 있어야 합니다. 관리자 권한은 필요하지 않습니다.
자세한 사전 조건과 수동 설치는 [CMD 폐쇄망 가이드](WINDOWS_OFFLINE.md)를 따릅니다.

### 2.1 Git 분할본 재결합 → 압축 해제

받은 저장소의 **루트 폴더**(README.md와 bundles 폴더가 있는 곳)에서 실행합니다.
`bundles\windows_cp312_win_amd64.parts`는 ZIP 분할본이며, part001~part004와
`parts.json`, `reassemble.py` **총 6개 파일 모두** 필요합니다.

**Git Bash:**

```bash
py -3.12 scripts/transfer_windows_bundle.py --version
py -3.12 scripts/transfer_windows_bundle.py join bundles/windows_cp312_win_amd64.parts --output outputs/dl1-cmd-v2.zip
py -3.12 -m zipfile -e outputs/dl1-cmd-v2.zip "$HOME/dl1-cmd"
cd "$HOME/dl1-cmd/windows_cp312_win_amd64"
```

**CMD:**

```bat
rem 버전이 psdl1-transfer 2인지 먼저 확인
py -3.12 scripts\transfer_windows_bundle.py --version
rem 1. 분할본의 목록/개별 hash를 확인하고 ZIP 재결합
py -3.12 scripts\transfer_windows_bundle.py join bundles\windows_cp312_win_amd64.parts --output outputs\dl1-cmd-v2.zip
rem 2. 쓰기 가능한 사용자 폴더에 압축 해제
py -3.12 -m zipfile -e outputs\dl1-cmd-v2.zip "%USERPROFILE%\dl1-cmd"
rem 3. 압축 해제된 설치 폴더로 이동
cd /d "%USERPROFILE%\dl1-cmd\windows_cp312_win_amd64"
```

`outputs\dl1-cmd-v2.zip`과 `%USERPROFILE%\dl1-cmd`는 **아직 없는 새 이름**을 사용합니다.
이미 있으면 위 명령의 ZIP/폴더 이름을 함께 바꾸십시오. 재결합기는 기존 ZIP을 덮어쓰지 않고,
성공 시 크기와 최종 SHA256을 출력합니다. 버전 출력은
`psdl1-transfer 2 (verified LF/CRLF/BOM)`이어야 합니다. 오류가 나면 다음 단계로 진행하지 마십시오.

`reassembly script hash differs`는 Git의 줄바꿈 변환으로 생길 수 있습니다.
[v2 재결합기 확인 및 오류 안내](WINDOWS_OFFLINE.md#git-반입-중-reassembly-script-hash-differs가-발생할-때)를
따르십시오. v2는 원래 해시와 일치하는 CRLF/BOM 차이를 자동 처리합니다.
해시 검사를 지우거나 parts.json의 기대 해시를 바꾸지 마십시오.
완성된 ZIP을 별도로 받은 경우에는 재결합을 생략하고 그 ZIP 경로로 압축 해제합니다.

### 2.2 설치 → help 확인 → 선택 smoke

위에서 압축 해제된 `windows_cp312_win_amd64` 폴더로 이동한 뒤 실행합니다.
설치 명령은 **Git Bash와 CMD에서 동일**합니다.

```bat
py -3.12 -I install_windows_bundle.py
```

설치 성공 후에는 **설치된 venv의 Python**으로 실행합니다. `py -3.12 -m psdl1...`로 바꾸면
base Python을 선택할 수 있으므로 아래의 명시적 venv 실행 파일을 사용하십시오.

**Git Bash — help 확인 및 선택 smoke:**

```bash
./.venv-cmd/Scripts/python.exe -I -m psdl1.import_png --help
# 선택: 새 출력 폴더에서 DL1 수치 smoke (설치/help와 별도)
./.venv-cmd/Scripts/python.exe -I smoke_windows_bundle.py --out "../dl1-smoke-new"
```

**CMD — help 확인 및 선택 smoke:**

```bat
.venv-cmd\Scripts\python.exe -I -m psdl1.import_png --help
rem 선택: 새 출력 폴더에서 DL1 수치 smoke (설치/help와 별도)
.venv-cmd\Scripts\python.exe -I smoke_windows_bundle.py --out "..\dl1-smoke-new"
```

설치기는 `.venv-cmd` 생성 → hash/inventory 검증 → 동봉 DL1 전용 wheel만 해시 고정 설치 →
pip check → CPU import/help를 확인합니다. 기존 venv는 자동 삭제·업데이트하지 않습니다.
재사용은 `py -3.12 -I install_windows_bundle.py --reuse-venv`로 정확 일치를 검사합니다.
수동 venv/verify/pip 설치 순서도 [CMD 가이드](WINDOWS_OFFLINE.md)에 있습니다.
Bash/PowerShell 설치기, uv, 온라인 설치, source checkout/PYTHONPATH, editable install은 쓰지 않습니다.
pytest와 hatchling은 반입 PC 설치에 필요하지 않습니다. macOS `requirements.lock`를 Windows에 적용하지 않습니다.

이후 Windows CMD 예시는 이 설치 폴더에서 `.venv-cmd\Scripts\python.exe -I`로 실행합니다.
Git Bash에서는 실행 파일을 `./.venv-cmd/Scripts/python.exe -I`로 쓰고, 데이터 경로는
`"D:/입력 폴더/manifest.json"`처럼 큰따옴표와 `/`를 사용하십시오.
다른 경로에서 실행할 때는 Python 실행 파일의 절대 경로를 큰따옴표로 감싸십시오.
POSIX 예시는 미리 준비된 개발 환경의 `.venv/bin/python`을 뜻하며 Windows 설치 절차가 아닙니다.
출력 경로는 매번 새 이름을 씁니다.

compare에서 사용자 실제 성공은 CMD 설치/pip check/help 범위뿐입니다. 이는 DL1 설치·수치 smoke·
실자료 성공의 근거가 아닙니다. DL1의 v2 재결합 단계는 사용자에게서 진행 성공을 확인했습니다.
DL1 설치/import/help·수치 smoke·실자료 처리의 Windows 성공 여부는 아직 별도 확인이 필요합니다.
이 호스트의 DL1 실행 검증은 macOS 합성 자료입니다.
선택 smoke 결과와 상세 검증 범위는 [검증 기록](WINDOWS_OFFLINE_VALIDATION.md)에 있습니다.

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
      1-OCNT.png
```

| 항목 | 규칙 |
|---|---|
| 모델명 폴더 | 그대로 제품 이름(`product_id`)이 된다 |
| 전극명 폴더 | 그대로 전극 이름이자 역할 이름(`role`)이 된다 |
| `cno` 폴더 | CNT 전용. 소문자 `cno`여야 한다. `CNO`처럼 대소문자가 다르면 오류다 |
| 파일명 | `<번호>-<layer>.png` 또는 `<번호>_<layer>.png`. 숫자 뒤 첫 `-`/`_`가 구분자이고 나머지 전체가 layer 이름이다. 확장자는 대소문자 구분 없이 `.png`/`.PNG`/`.Png`를 받으며 원본 파일명은 보존한다 |
| 파일명 숫자 | 전극·cno 모두 양의 정수와 파일명 문법만 검사한다. **기록용(provenance only)**이며 순서·identity에 쓰지 않는다. 원본 파일을 번호에 맞춰 **이름 변경하지 않는다** |
| 번호 재사용 | 서로 다른 layer 이름에 같은 번호를 써도 된다. 다른 전극 폴더의 같은 layer가 다른 번호여도 된다 |
| 중복 전극 × layer | 한 전극 폴더 안에서 같은 layer 파일이 둘이면 오류이며 두 경로를 보고한다 |
| 같은 CNT 이름 | 서로 다른 mask의 여러 cno 파일은 source asset과 component를 각각 보존한다. 이미지 채널에서 전체 CNT를 합친다. **image_only**에서 같은 이름·같은 decoded full mask(모두 흰색 포함)는 허용하고 `duplicate_cnt_mask_same_name` 경고를 낸다. 하나의 content identity에 모든 파일의 `provenance_files`를 남긴다. **relations**에서는 `ambiguous_cnt_asset_identity` 오류로 거부한다 |
| CNT 이름 종류 | 지금까지 알려진 OCNT/GCNT/PCNT는 whitelist가 아니다. 다른 유효한 이름도 받는다 |
| 이름 비교 | identity·manifest·vocabulary는 원본 철자를 보존한다. 선택적 순서 파일 매칭만 `casefold`로 대소문자를 무시한다. image_only는 한 모델 안 또는 모델 간 대소문자만 다른 layer 이름을 `layer_name_case_variants` 경고로 보고하고 각각의 vocabulary 철자를 보존한다. 순서 파일을 주면 한 모델 안의 충돌은 오류다. NFC/NFD는 정규화하지 않는다 |
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

## 4. 검사 — CNT 형상 확인과 grid 고르기

### 4.1 검사 실행

계보 정보와 grid 없이 실행된다. 출력 폴더는 **입력 폴더 밖의 새 폴더**여야 한다. 이미 있는 폴더는 비어 있어도 거부한다.

```bash
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$INSPECT_OUT"
```

Windows (명령 프롬프트):

```bat
".venv-cmd\Scripts\python.exe" -I -m psdl1.import_png --root "D:\png_root" --out "D:\out\inspect1"
```

Windows에서는 출력 경로를 짧게 잡는다. 생성되는 파일 이름이 70자 이상이라 경로가 깊으면 경로 길이 제한(260자)에 걸릴 수 있다(추정이며 실제로 확인되지는 않았다).

종료 코드:

| 코드 | 뜻 |
|---|---|
| 0 | 성공 |
| 2 | 입력 오류, 내보내기 조건 미충족, 출력 경로 거부, 실패 |

표준 출력(stdout)에는 JSON이 두 줄 나온다. 첫 줄(`resource_inventory`)은 용량 추정과 이름 목록, 둘째 줄(`completed`)은 결과다. 진행 상황은 별도의 stderr JSON 줄(`event: "progress"`)로 즉시 출력된다. 읽기, 모델, grid, preview, 내보내기, 검증 단계와 첫 파일·100개 단위·마지막 파일 수를 확인할 수 있다. 라이브러리 호출은 `progress_sink`를 지정할 때만 진행 상황을 전달한다. 진행 줄은 `event == "progress"`인 JSON 객체다. 오류·경고·argparse 메시지 등 다른 stderr 줄은 일반 텍스트이므로, 전체 stderr를 JSON으로 해석하지 말고 먼저 걸러낸다.

```bash
# stderr 전체 기록에서 진행 JSON 줄만 추출한 뒤 JSON으로 해석한다.
grep '"event": "progress"' stderr.log > progress.jsonl
```


### 4.2 만들어지는 파일

```text
D:\out\inspect1\
  import-report.json       보고서 (dl1_png_import_v3)
  preview.html             CNT 형상 확인 화면 (인터넷 없이 열림)
  preview\                 preview가 쓰는 이미지
  bundle-integrity.json    위 파일들의 SHA-256 목록
```

`--no-preview`를 지정하면 `preview.html`과 `preview/`를 만들지 않고, 보고서의 `preview`는 빈 목록이 된다. `diagnostics`에는 `preview_skipped_by_request: --no-preview`가 정확히 한 번 기록된다. 원본·추출·grid 근거 및 내보내는 mask는 모두 유지된다. 기본값은 preview 생성이다. 이 옵션은 `--verify-bundle`과 함께 사용할 수 없다.

Linux/macOS 예시 (출력 경로는 새 경로로 지정):

```bash
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$NEW_INSPECT_OUT" \
  --inspect-grids 32 64 --no-preview > result.jsonl 2> progress.jsonl
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$NEW_EXPORT_OUT" \
  --export --metadata "$METADATA_JSON" --candidate-grid 64 --no-preview \
  > export-result.jsonl 2> export-progress.jsonl
```

Windows CMD 예시:

```bat
".venv-cmd\Scripts\python.exe" -I -m psdl1.import_png --root "D:\png_root" --out "D:\out\inspect-new" --inspect-grids 32 64 --no-preview 1> result.jsonl 2> progress.jsonl
".venv-cmd\Scripts\python.exe" -I -m psdl1.import_png --root "D:\png_root" --out "D:\out\export-new" --export --metadata "D:\metadata.json" --candidate-grid 64 --no-preview 1> export-result.jsonl 2> export-progress.jsonl
```

중단/실패하면 표시된 staging 경로의 진단 근거를 보존하고 **새 `--out` 경로**로 다시 실행한다. 기존 출력이나 실패 stage를 자동 재사용·덮어쓰지 않는다. 읽기 패스 사이에 원본(병합 CNT의 모든 provenance 파일 포함)이 바뀌거나 없어지거나 루트 밖 링크로 바뀌면 출판을 중단한다.

### 4.3 preview로 CNT 형상 확인

`preview.html`을 브라우저로 연다. 화면과 CLI `completed` 이벤트의 **CNT mode: image_only**를 확인한다.
CNT 파일별 이름, bbox, 픽셀 수, `cnt_image_only_no_relations` 사유가 보인다.
회색 사각형은 덩어리의 표시용 중심이며 연결 위치가 아니다. 검정 테두리는 CNT 형상,
회색 전극은 주변 형상이고 보라 사각형은 중앙 2×2 참고 영역이다.
상하 순서, 선택된 위/아래 전극, 연결 끝점은 생성하지 않는다.

확대 그림은 사람이 보는 용도다. **모델 입력은 그대로 전체 frame**이다.
원본 여백도 모델 입력과 전체 CNT 이미지에 그대로 남는다.

### 4.4 이미지 채널과 연결 정보의 구분

모델의 모든 CNT foreground를 합친 기존 `cnt` mask는 이미지 채널이다.
같은 이름의 CNT 파일도 각각 검사하고 component geometry를 따로 기록하지만,
채널에는 전체 합집합을 사용한다. CNT가 보인다는 사실을 endpoint 관계로 바꾸지 않는다.
`connections: []`, `connectivity_observation: "unavailable"`로 내보낸다.

4-neighbour(상하좌우; 대각선만 붙은 픽셀은 별개)는 사용자가 확인한 규칙이다.
이것은 전기적 연결 검증이 아니다. 연결 추출은 11장의 선택적 relations 모드에서만 한다.

### 4.5 보고서에서 볼 곳 (`import-report.json`)

| 위치 | 내용 |
|---|---|
| `export_status` | `not_requested`(검사만), `blocked`(조건 미충족), `exported`, `failed`, `interrupted` |
| `pending_prerequisites` | 내보내기를 막은 사유. 예: `explicit_metadata_missing` |
| `inventory.models[].errors` / `warnings` | 파일별 오류와 경고(무시된 파일, 빈 전극 폴더 등) |
| `cnt_mode`, `inventory.models[].cnt_mode`, `extraction[].cnt_mode` | `image_only` 또는 `relations`; 모드가 일치해야 한다 |
| `extractor_run_status` | image_only는 `not_run_image_only`(geometry inventory만 수행), relations는 `ran`/`partial`/`blocked`; coverage와 별개 |
| `inventory.models[].assets[].filename_number` | 모든 파일의 원본 숫자. provenance only; 순서·identity가 아니다 |
| `stack_order` | 선택적 순서 파일의 schema/source/raw-byte `file_sha256`/`document`/`errors`; image_only는 파일 evidence 없음 |
| `inventory.models[].stack_order` | image_only는 `status: not_used`, 비어 있는 order 및 null digest; relations는 `resolved`/`invalid`, `selected_list`, `resolved_order`, `ignored_names`, `matched_pairs`, `resolved_order_sha256`, `errors` |
| `resolved_order[]` | 0부터 연속인 `order_index`, source spelling `name`, `kind`, 모든 별도 source의 `asset_ids`; 파일 번호 필드 없음 |
| `inventory.models[].resource_estimate` | 용량 추정. 아래 4.6 참고 |
| `inventory.role_union`, `electrode_layer_union` | 전체 모델의 전극 이름, 전극 layer 이름 |
| `inventory.unique_roles`, `unique_electrode_layers` | 한 모델에만 있는 이름 |
| `duplicate_geometry` | 형상이 완전히 같은 파일/모델(중복 제품 가능성). 자동으로 합치지 않는다 |
| `extraction[].summary` | image_only는 geometry 덩어리 K/관계 0이며 실패한 접촉을 뜻하지 않음; relations는 해결·미해결·확인 필요 수 |
| `extraction[].connectivity_observation` | 추출 범위: `complete`(전부 해결), `partial`, `unavailable`(해결 없음) |
| `extraction[].summary` | image_only는 `geometry_only_component_count`로 형상 수를 기록하고 `unresolved_component_count: 0`; 미해결 접촉을 뜻하지 않는다. 이전 v3 묶음의 geometry용 unresolved count도 reader가 호환 검증한다 |
| `extraction[].components[]` | geometry bbox/픽셀 수/source path/CNT 이름; image_only는 `cnt_order_index: null`, 후보 없음, `relation: null`; `cnt_filename_number`는 provenance |
| `relation.representative_xy_px` | 연결 위치(원본 픽셀) |
| `relation.needs_confirmation` | 건너뛴 layer가 있으면 true |
| `relation.skipped_upper_layers` / `skipped_lower_layers` | 건너뛴 전극 layer와 CNT layer |
| `grid_evidence` | grid별 확인 결과(4.7) |
| `publication_method` | 묶음을 게시한 방식(8장) |

**`complete`는 추출 범위만 뜻한다. 전기적 검증이 아니다**(보고서 문구: complete means extraction coverage only; electrical verification not performed). 이 값은 모델 입력에도 들어가므로, 같은 형상이라도 `partial`과 `complete`는 모델에 다른 입력이 된다.

### 4.6 용량 추정

image_only는 **R=0**이므로 연결별 mask를 만들지 않는다. 선택적 relations 모드의 기존 manifest 형식은 연결 하나마다 전체 크기의 mask 파일이 하나씩 필요하다. CNT가 많으면 파일과 메모리가 그만큼 늘어난다.

| 필드 | 뜻 |
|---|---|
| `frame_pixels` | 이미지 픽셀 수 |
| `electrode_asset_count`, `cnt_component_count`, `resolved_relation_count` | 전극 파일 수, CNT 덩어리 수, 해결된 연결 수 |
| `generated_mask_asset_count` | 내보낼 mask 파일 수 |
| `projected_bool_npy_export_bytes` | 내보낼 NPY 전체 크기(바이트) |
| `projected_load_manifest_retained_float32_mask_bytes` | 읽을 때 메모리에 남는 mask 크기(바이트) |
| `all_components_resolved` | image_only는 null, relations에서만 모든 덩어리 해결 시 비용 시나리오 |
| `all_components_resolved_applicability` | image_only는 `not_applicable_image_only`; 이것을 실행 계획으로 해석하지 않는다 |

이 값은 최대 메모리나 실행 시간 예측이 아니다. 디코딩 중에는 픽셀당 약 17~18바이트가 일시적으로 더 쓰인다. 자원이 충분한지는 이 숫자를 보고 사용자가 판단한다. 프로그램이 정한 한도는 없다.

검사/내보내기는 원본 mask를 **모델 하나씩** 읽고 추출한 뒤, 보고서에는 배열 없는 기록만 보관한다. 전체 용량 추정은 grid/Product 할당 전에 기록된다. 이후 필요한 모델만 다시 읽어 PNG 바이트 해시·mode·shape·형상 해시를 검사하고 추출 결과를 대조한다. 모델의 각 관측 mask에서 원본 폭·성분 수·활성 픽셀 수를 한 번 계산해 후보 grid 사이에 재사용하며, 후보별 변환과 결과 검사는 그대로 수행한다. 내보내기는 별도 패스로 다시 읽는다. 반복 디코딩/추출 때문에 CPU·디스크 읽기 시간이 늘 수 있다.

이 한도는 고정 RAM 한도가 아니라 **가장 큰 모델 기준**이다. 연결마다 전체 frame mask가 필요하며, 내보낸 NPY 검증은 한 모델의 Product를 float32(픽셀·mask당 약 4바이트)로 보관한다. 예를 들어 1500×1500 mask 900개는 **한 모델만 약 8.1 GB**이고 여기에 디코더·검증 임시 배열 등이 더해진다. 모델 하나가 너무 크면 여전히 메모리가 부족할 수 있다. 보고서·manifest·HTML 텍스트는 모델 수에 비례한다. 학습 및 공개 `data.load_manifest`는 모든 Product를 메모리에 유지하는 eager API다. 이 변경으로 실제 18,000 PNG 데이터 처리 성공을 확인한 것은 아니다.


### 4.7 grid 고르기

grid는 모델에 넣기 전에 전체 이미지를 줄이는 정사각형 한 변의 칸 수다. 모든 전극·CNT 이미지가 같은 비율로 줄고 남는 부분은 빈칸으로 채워진다.

- 작으면 가는 선이 사라지거나 떨어진 패턴이 붙는다. 대신 빠르다.
- 크면 형상이 보존되지만 메모리와 시간이 늘어난다.
- image_only의 grid 검사는 전극·valid·전체 CNT 형상을 본다. 선택적 relations 추출은 원본 해상도에서 하므로 grid와 무관하다.
- 학습한 checkpoint는 그 grid에 묶인다. 검색할 제품도 같은 grid로 내보낸다.

기준: 가장 가는 선이 줄어든 뒤에도 2칸 이상 남아야 한다.

```text
grid ≥ 2 × (이미지 긴 변 픽셀 수) ÷ (가장 가는 선폭 픽셀 수)
```

후보를 직접 넣어 비교한다. 기본값은 없고, 프로그램이 골라 주지 않는다.

```bash
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$GRID_INSPECT_OUT" --inspect-grids "$GRID_A" "$GRID_B"
```

Windows (명령 프롬프트):

```bat
".venv-cmd\Scripts\python.exe" -I -m psdl1.import_png --root "D:\png_root" --out "D:\out\grid1" --inspect-grids 256 512 1024
```

`grid_evidence`에서 grid별로 다음을 본다.

| 필드 | 뜻 |
|---|---|
| `thin_line_warning` | 가장 짧은 선이 약 2칸 미만 |
| `bridge_or_component_loss` | 줄인 뒤 덩어리 수가 바뀜(끊김, 사라짐, 붙음) |
| `reasons` | 경고 사유 목록 |
| `transform` | 실제 줄인 크기, 여백, 배율, 줄인 뒤의 µm/칸 |

경고가 없는 가장 작은 grid가 출발점이다. 경고가 있는 grid도 내보낼 수는 있지만 학습·검색에서 기본으로 막힌다. **내보냈다고 grid가 승인된 것은 아니다.**

#### 모델 × grid 진단 PNG

**기존 Windows ZIP 설치 후 새 그림이 안 보일 때:** `git pull`만으로 `.venv-cmd`에 설치된
wheel은 바뀌지 않습니다. [grid 진단 업데이트](../bundles/updates/grid-diagnostics-f74a718/README.md)의
폴더 전체를 기존 `.venv-cmd` 옆으로 옮긴 뒤, 안내된 오프라인 `pip --force-reinstall` 명령으로
psdl1만 교체합니다(기존 numpy/torch/Pillow 재설치 없음). 설치 버전은 같은 `0.1.0`이므로
버전 번호만으로 적용 여부를 구분하지 말고 안내된 import 확인 명령을 실행하세요.
기존 PNG는 바뀌지 않으므로 **새 출력 폴더**로 `--inspect-grids`를 다시 실행해야 합니다.
기존 전체 ZIP의 wheelhouse 파일을 직접 교체하면 기존 manifest 검증이 실패하므로 별도 업데이트 폴더를 사용합니다.
업데이트 후 전체 ZIP 기준 설치 검증(`--installed-prefix`, `--reuse-venv`)은 `installed psdl1 source differs`로
실패하는 것이 예상 동작입니다([Windows 설치 안내](WINDOWS_OFFLINE.md)의 grid 진단 업데이트 항목).

```sh
python -m psdl1.import_png --root ./png-source --out ./inspect-new \
  --inspect-grids 32 64 --no-preview
python -m psdl1.import_png --verify-bundle ./inspect-new
```

각 유효 모델과 중복 제거한 inspect grid마다 `diagnostics/model_<sha256>/grid-<g>.png`와
근거 JSONL이 자동 생성됩니다. 모델별 실제 PNG 경로는 `import-report.json`의
`grid_evidence[].diagnostic.png`에서 찾습니다. PNG는 1100×1040 고정 크기입니다. `--no-preview`는 기존 전체
미리보기/HTML만 생략하며 이 진단은 유지합니다. `--candidate-grid`만 지정하면 진단은 생성하지 않습니다.

상단 큰 문구로 `SHAPE CHANGE DETECTED`(형상변화 검출), `THIN AREAS: REFERENCE ONLY`
(얇은 구간 참고만), `NO SHAPE CHANGE DETECTED`(검출 변화 없음), `INSPECTION INCOMPLETE`
(검사 미완료)를 구분합니다. 미완료가 우선 표시됩니다.

- 전체 화면은 실제 축소 결과를 하나의 정합된 768px 화면에 겹칩니다. 전극은 **층 이름별 고정 색**을
  반투명(50%)으로 누적합니다. 색은 Product 순서에서 처음 나온 층 순서로 정해지므로 같은 제품의
  모든 grid에서 같은 층은 같은 색입니다(다른 제품 간 색 일치는 보장하지 않음). 우하단 범례에
  최대 14개 층과 `+N more`가 표시됩니다. CNT는 모든 전극 뒤에 반투명 초록으로 그리고,
  전극∩CNT는 색을 지우지 않는 성긴 진한 해치로 표시합니다. 겹침 순서는 **표시 합성 순서일 뿐
  물리적 적층 순서가 아닙니다**. 회색은 패딩, 회색 해치는 미관측 영역입니다.
- 노랑은 원본 축방향 run의 `길이 × grid/max(H,W) < 2` 참고 영역입니다(대상 2칸 미만 투영).
  대각선의 1px run도 포함하므로 노랑만으로 형상 손상이나 실제 최소 선폭을 판단하지 않습니다.
  오른쪽 아래 `Yellow thin references`에 grid당 최대 2개 예시(Y1, Y2; 관측당 1개, grid 결과에
  남은 칸 우선 후 최단 run)를 `original | grid <g>` 제목이 붙은 두 열로 나란히 보여주고 `horizontal run 24px`,
  `x 64/1203`, `-> 1.28 cells`, `(< 2: reference)`처럼 같은 식의 실제 수치를 적습니다.
  해당 run은 원본 crop에 테두리로 표시하며, 예시의 층 이름 옆 색 견본과 crop 색은 전체 화면의
  같은 층 색입니다(CNT는 초록). 노랑 예시는 빨강 상세/`insets_*` 개수에 포함되지 않습니다.
- 빨강은 성분 소실·분리·합쳐짐 의심의 실제 witness 위치입니다. 모든 층의 검출 마커를 유지합니다.
  하단 `Observations`는 종류별 해당 **관측 수**이며 독립 물리결함 개수가 아닙니다. 같은 변화가
  global CNT와 source CNT 등에 겹쳐 나타날 수 있습니다. 정확한 단선 픽셀을 뜻하지 않습니다.
- 오른쪽 빨강 상세는 최대 3개(R1–R3, 전체 화면에도 같은 번호)이며 `more`는 생략된 이벤트 상세
  수입니다. 각 상세는 해당 관측(층) 하나만 분리해 `original | grid <g> | difference`를 같은 grid ROI
  창(좌표 표기)과 grid 칸 경계선으로 보여줍니다. 제목 첫 줄은 번호·종류·관측 ID, 둘째 줄은 전체
  화면과 같은 색 견본과 층 이름이며 crop도 그 층 색으로 그립니다. LOSS 창은 witness 주변 국소 창
  (`local`)입니다. MERGE에서 이벤트에 이미 기록된 source bounds 중 앞쪽 최대 8개 안에 한 축으로
  떨어져 있고 다른 축 범위가 겹치는 상자 쌍이 있으면, 가장 가까운 틈 중심에 일반 상세의 절반
  크기(최소 2칸) 국소 창(`local join context`)을 두어 원본의 두 조각 가장자리와 틈이 확대로 보이고
  grid 열에서는 이어진 모습이 보입니다. 표시 전용 위치이며 물리적 연결을 증명하지 않습니다.
  그런 쌍이 없는(겹치거나 불규칙한) MERGE와 SPLIT 창은 source/target bounds를 1칸 여유로 감싼
  정사각형(`event bounds+1`, grid 크기로 제한)입니다. 창이 커지면 `src OR 1:k`로 축소되며, 2px 간격처럼 축소비보다 작은 틈은 원본 열에서도 보이지 않을
  수 있습니다(JSONL 좌표 확인). 창 계산은 표시 전용이고 수치 근거를 다시 계산하지 않습니다. 원본
  footprint와 축소된 grid 결과는 타일(타일당 64K 픽셀 이하, 아주 넓은 행은 가로로도 분할)로 훑으므로 큰 창도 원본 크기 배열을 만들지 않습니다.
  difference는 회색=양쪽, 자홍=원본만, 청록=grid만.
  원본은 최근접 표본 + 원본 전경 픽셀 OR로만 그리며 팽창(MaxFilter/stamp)이 없습니다. 창의 원본
  footprint가 90px 이하면 `src x1.6 nearest`(블록 확대), 넘으면 `src OR 1:k`로 표기합니다. 확대는
  표시 전용이며 분석이나 source mask를 바꾸지 않습니다. 1px 결함은 확대 후에도 작게 보일 수 있습니다.
  모든 좌표, 원본 경로, 중복 provenance와 노랑/빨강 근거는 함께 생성되는 JSONL에 남습니다.
  JSONL·PNG 크기·metadata(v1) 계약은 변경되지 않았습니다.

기존 legacy status와 학습/검색 차단 조건은 그대로이며 `import-report.json`에서 확인합니다.
축방향 run 경고와 새 형상 판단은 별개입니다. 변화 미검출도 안전/승인이 아니며 경계 이동,
연결을 유지한 침식, 구멍 변화 등은 놓칠 수 있습니다.

대응 계산의 source 좌표/정렬 임시배열은 최대 65,536 입력 픽셀의 row band로 처리합니다
(한 행이 더 넓으면 한 행). source/target labels, 성분별 통계와 모든 고유 대응 edge는 여전히
크기에 비례합니다. 모델별 streaming은 유지되지만 전체 import RAM 상한을 보장하지 않습니다.
1500² 합성 단일 source→384 grid의 대응 계산 peak는 밀집 320.3→8.3 MiB,
3% noise + 보존되는 두 영역을 넣은 sparse는 11.2→3.7 MiB였습니다. 3회 중앙 실행시간은
각각 131→161 ms, 11.6→24.5 ms로
시간 비용이 있습니다. 전체 모델 import 또는 실제 18,000 PNG 성공 측정은 아닙니다.
`python scripts/benchmark_png_correspondence.py --case dense` (`sparse`도 가능)로 재현하며
`--baseline-source <이전 png_diagnostic.py>`를 별도 프로세스에 주면 이전 구현과 비교합니다.

재현 가능한 합성 예제(실제 데이터 성능/안전성 검증이 아님):

```sh
python scripts/demo_png_diagnostic.py --out ./fresh-synthetic-diagnostic
```

기존 경로를 덮어쓰지 않습니다. 지정한 경로 안에만 합성 입력과 grid64/128 검증 bundle을 만들고
다층 겹침·소실/분리/합쳐짐 의심 모델과 노랑만 있는 모델의 PNG 경로 4개를 출력합니다. report v3와 기존 필드는 유지되며 새 중첩 diagnostic v1만 추가됩니다.

호환성 주의: 기존 v3 reader는 legacy 근거를 읽을 수 있지만 새 diagnostic의 schema/개수/namespace를
검증하지 않습니다. 실제 b8c8a22 verifier도 seal된 새 bundle을 받아들이므로, 그 `verified` 결과를
새 진단 근거의 검증으로 해석하면 안 됩니다. 진단 검증에는 새 verifier를 사용하십시오.


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

```bash
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$EXPORT_OUT" --export --metadata "$METADATA_JSON" --candidate-grid "$CANDIDATE_GRID"
```

Windows (명령 프롬프트):

```bat
".venv-cmd\Scripts\python.exe" -I -m psdl1.import_png --root "D:\png_root" --out "D:\out\bundle1" --export --metadata "D:\meta\meta.json" --candidate-grid 512
```

- 계보 파일이나 grid가 빠지면 종료 코드 2로 끝나고 `manifest.json`을 만들지 않는다. 보고서와 preview만 남는다.
- 모델 하나라도 입력 오류가 있으면 전체를 내보내지 않는다. 일부만 내보내지 않는다.
- 순서 파일은 기본 경로에 필요 없다. cno가 있어도 image_only로 내보낸다. 모든 CNT 형상은 남지만 관계는 없고 연결 상태는 `unavailable`이다. 선택적 relations의 미해결 CNT도 형상을 보존한다.

만들어지는 것:

```text
D:\out\bundle1\
  manifest.json            기존 형식(dl1_shape_v1_draft). 그대로 학습·검색에 쓴다
  assets\*.npy             전극, 관측 범위, 전체 CNT (bool); relations에서만 연결별 mask 추가
  import-report.json
  preview.html, preview\
  bundle-integrity.json
```

manifest의 `unit_size_um`에는 **전체 이미지 범위**(가로×0.1, 세로×0.1 µm)가 들어간다. 단일 unit 크기가 아니다.

### 5.3 묶음 확인

```bash
.venv/bin/python -m psdl1.import_png --verify-bundle "$EXPORT_OUT"
.venv/bin/python -m psdl1.data "$EXPORT_OUT/manifest.json" --report "$VALIDATOR_REPORT"
```

Windows (명령 프롬프트):

```bat
".venv-cmd\Scripts\python.exe" -I -m psdl1.import_png --verify-bundle "D:\out\bundle1"
".venv-cmd\Scripts\python.exe" -I -m psdl1.data "D:\out\bundle1\manifest.json" --report "D:\out\grid-report1.json"
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

머리글 없는 CSV다. 형식은 아직 초안이다. 이것은 제품 쌍의 학습 정답이며 선택적 CNT relations 모드와 별개다.

```text
MODEL-A,MODEL-B,similar
MODEL-A,MODEL-C,different
```

- 적지 않은 쌍은 "모름"으로 두고 학습에서 뺀다.
- 같은 쌍에 similar와 different가 함께 있으면 오류다.
- different 쌍이 하나도 없으면 학습을 시작하지 않는다.
- 정답이 없으면 `--unlabelled-negatives`로 "적지 않은 쌍은 다르다"고 가정할 수 있다. 실제로 비슷한 제품을 밀어낼 수 있는 실험 옵션이다. 같은 계열끼리는 이 경우에도 "모름"이다.

### 6.2a `relations.csv` 초안 만들기 (`psdl1.relations_draft`)

같은 계열(`lineage_id`)은 `similar`, 다른 계열은 `different` 후보로 쓰는 초안을 만든다. 다른 계열이라도 실제로 닮은 설계는 `different`로 넣기 어렵다. 그런 쌍은 **이미지 거리가 가장 가까운 순서대로 일부를 모름으로 남긴다**(`--hold-out-nearest`). 검토 목록(`*.review.csv`)을 보고 직접 판단한다.

```bash
.venv/bin/python -m psdl1.relations_draft --manifest "$EXPORT_OUT/manifest.json" --split "$SPLIT_JSON" \
  --out relations.csv --partitions train --hold-out-nearest 0.3
```

- `relations.csv`는 머리글 없는 3열이라 `psdl1.train --relations`에 그대로 쓴다. 검토 목록에는 쌍마다 구획, 관계, 이미지 거리, 계열, 메모가 들어간다. 기존 파일은 덮어쓰지 않는다.
- 쌍은 같은 구획 안에서만 만든다. 기본은 `train`이고, 평가에 쓰려면 `--partitions train test`처럼 지정한다.
- 이미지 거리는 role/layer로 맞춘 낮은 해상도(`--grid`, 기본 64) 마스크의 평균 절대 차이다. 학습된 값이 아니라 판단을 돕는 지표이고 정답이 아니다. "다른 계열은 다른 설계"라는 가정이 맞는지는 사용자가 확인한다.
- 출력 JSON의 `products_without_different_pair`에 `different` 쌍이 하나도 없는 제품이 나온다. 가능하면 비워 둔다.

### 6.3 학습

```bash
.venv/bin/python -m psdl1.train --manifest "$EXPORT_OUT/manifest.json" --split "$SPLIT_JSON" --relations "$RELATIONS_CSV" --out "$TRAIN_OUT" --epochs 100 --batch-size 8 --lr 0.0003
.venv/bin/python -m psdl1.train --manifest "$EXPORT_OUT/manifest.json" --split "$SPLIT_JSON" --relations "$RELATIONS_CSV" --out "$TRAIN_OUT" --resume "$TRAIN_OUT/dl1.pt" --epochs 150
```

Windows (명령 프롬프트):

```bat
".venv-cmd\Scripts\python.exe" -I -m psdl1.train --manifest "D:\out\bundle1\manifest.json" --split "D:\data\split.json" --relations "D:\data\relations.csv" --out "D:\runs\dl1" --epochs 100 --batch-size 8 --lr 0.0003

rem 이어서 학습 (--epochs는 총 목표 횟수)
".venv-cmd\Scripts\python.exe" -I -m psdl1.train --manifest "D:\out\bundle1\manifest.json" --split "D:\data\split.json" --relations "D:\data\relations.csv" --out "D:\runs\dl1" --resume "D:\runs\dl1\dl1.pt" --epochs 150
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

### 6.3a 백본 사전학습 (`psdl1.pretrain`, 선택)

라벨 없이 train 구획의 요소 이미지에서 가린 패치를 복원해 백본만 학습한다(FCMAE 방식의 dense 근사, sparse convolution 아님). 결과 `backbone.pt`를 `psdl1.train --weights`에 넣는다.

```bash
.venv/bin/python -m psdl1.pretrain --manifest "$EXPORT_OUT/manifest.json" --split "$SPLIT_JSON" --out "$PRETRAIN_OUT" \
  --device cuda --batch-size 2 --epochs 100
.venv/bin/python -m psdl1.train ... --weights "$PRETRAIN_OUT/backbone.pt" --backbone tiny
```

- train과 validation 제품만 쓴다. test는 캐시·학습·해시에 쓰지 않는다(manifest 검증 때문에 시작할 때 읽기만 한다). 같은 `split.json`을 본 학습에도 써야 한다.
- 시작할 때 요소 이미지를 `--out/cache/`에 float16으로 저장하고, 에폭 중에는 배치마다 읽는다. 제품이 같으면 다음 실행에서 재사용한다. 제품당 요소 60~80장, grid 1024면 제품당 약 0.4GB다.
- 진행 상황은 stderr에 JSON 줄(`event: "progress"`)로 나온다. `--no-progress`로 끈다. `psdl1.train`도 같은 형식이다.
- 백본은 `tiny`와 timm 이름을 지원한다. 마스크 패치 크기는 백본의 stride(`tiny` 8, ConvNeXt 32)에 맞춘다.
- `history.json`의 validation `loss`가 `baseline_loss`보다 낮고 `masked_foreground_iou`가 오르는지 본다. 사전학습이 본 학습 성능을 올리는지는 별도로 검증해야 한다(같은 split과 라벨로 `--weights` 없이/있이 학습해 validation에서 비교).
- `--resume`은 설정·입력·split·코드가 같을 때만 허용한다. 학습 코드가 바뀌면 이전 체크포인트는 거부된다.

### 6.4 검색

manifest 두 개를 받는다. 둘 다 같은 grid로 내보낸 것이어야 한다.

```bash
.venv/bin/python -m psdl1.search --ckpt "$TRAIN_OUT/dl1.pt" --query-manifest "$QUERY_EXPORT_OUT/manifest.json" --query "$QUERY_ID" --candidates-manifest "$EXPORT_OUT/manifest.json" --topk 5 --out "$SEARCH_OUT"
```

Windows (명령 프롬프트):

```bat
".venv-cmd\Scripts\python.exe" -I -m psdl1.search --ckpt "D:\runs\dl1\dl1.pt" --query-manifest "D:\out\query1\manifest.json" --query NEW-MODEL --candidates-manifest "D:\out\bundle1\manifest.json" --topk 5 --out "D:\out\search1.json"
```

**이름 제한**: 전극 폴더명이 그대로 역할 이름이 된다. 학습에 없던 전극 이름이나 layer 이름을 가진 제품은 검색할 수 없다. 찾으려는 제품이 그러면 결과가 `unavailable`이 되고, 후보가 그러면 사유와 함께 `excluded`에 들어간다. 같은 이름은 같은 것으로 취급되지만, 두 전극이 실제로 같은 기능이라는 뜻은 아니다. 별칭을 맞추는 기능은 없다.

### 6.5 평가 (정답이 있을 때)

```bash
.venv/bin/python -m psdl1.evaluate --ckpt "$TRAIN_OUT/dl1.pt" --manifest "$EXPORT_OUT/manifest.json" --split "$SPLIT_JSON" --relations "$RELATIONS_CSV" --topk 5 --out "$EVALUATION_OUT"
```

Windows (명령 프롬프트):

```bat
".venv-cmd\Scripts\python.exe" -I -m psdl1.evaluate --ckpt "D:\runs\dl1\dl1.pt" --manifest "D:\out\bundle1\manifest.json" --split "D:\data\split.json" --relations "D:\data\relations.csv" --topk 5 --out "D:\out\evaluation1.json"
```

`--hard-negatives <파일>`로 형상은 같고 연결 위치만 다른 제품 쌍 목록(`[["MODEL-E", "MODEL-F"]]`)을 줄 수 있다.

### 6.6 임베딩 거리 보고서 (`psdl1.embed_plot`)

학습된 체크포인트가 제품 사이 거리를 어떻게 만드는지 오프라인 HTML로 본다(외부 라이브러리·네트워크 없음).

```bash
.venv/bin/python -m psdl1.embed_plot --ckpt "$TRAIN_OUT/dl1.pt" --manifest "$EXPORT_OUT/manifest.json" --out "$REPORT_OUT" --device cpu
```

- 새 폴더에 `embedding_map.html`(2D 투영 두 종류, 코사인 거리 히트맵, 표), `embeddings.csv`, `distances.csv`, `summary.json`이 생긴다.
- 색과 모양은 구획(train/validation/test)이고, 같은 계열은 가는 선으로 잇는다. 2D 투영은 거리를 왜곡할 수 있으니 정확한 값은 히트맵과 표를 본다.
- `summary.json`과 화면 위쪽에 같은 계열 안/밖의 평균 거리와 "가장 가까운 제품이 같은 계열인 비율"이 나온다. 모든 거리가 0.02 미만이면 임베딩이 한 점으로 몰렸을 수 있다는 경고가 나온다.
- 학습에서 못 본 role/layer가 있는 제품은 제외되고 사유가 표시된다. 체크포인트 학습 입력과 다른 제품은 `summary.json`의 `inputs_differing_from_checkpoint`에 나온다.
- 제품 하나의 요소 전체를 한 번에 통과시키므로 grid 1024에서는 GPU 메모리가 부족할 수 있다. 기본은 `--device cpu`다.

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
| `layer_name_case_variants` | image_only 경고이며 내보내기를 막지 않는다. 모델 안/모델 간 대소문자 차이가 별도 vocabulary로 남는다. 원본 자료를 확인하고 번호로 해결하지 않는다 |
| `stack_order_case_collision` | 선택적 relations에서 순서 파일 또는 한 모델의 이름이 대소문자만 다르다. 어느 원본 이름이 맞는지 확인한다. 번호를 바꿔 해결하지 않는다 |
| `duplicate_cnt_mask_same_name` | image_only 경고이며 내보내기를 막지 않는다. 같은 이름·같은 decoded mask의 모든 파일을 확인한다. 하나의 content asset에 provenance를 모으며 CNT union은 변하지 않는다 |
| `ambiguous_cnt_asset_identity` | relations 오류. 같은 CNT 이름의 decoded full mask가 같다. 보고된 모든 파일의 중복 여부를 확인하고 안정된 source 식별 정책을 요청한다. 번호 변경으로 해결하지 않는다 |
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
| 같은 전극 연결 규칙 | 사용자가 확인했지만 relations에는 아직 미적용(`same_electrode_required: false`); 다른 전극 간 관계도 나올 수 있는 알려진 gap |
| 동일 이름·동일 CNT mask | image_only는 경고와 전체 파일 provenance로 허용; relations는 계속 거부하며 별도 stable source ID 계약이 필요. prefix 변경은 해법이 아님 |
| CNT 이름 whitelist | OCNT/GCNT/PCNT만으로 제한하지 않음; 선택적 순서 파일에 있는 다른 이름도 허용 |
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

## 11. 선택 사항 — 순서 파일로 relations 모드 사용

기본 image_only 경로에는 이 장이 필요 없다. 원본 좌표의 연결 위치 후보를 검토할 때만
`--stack-order PATH`를 준다. 이때 `cnt_mode: relations`가 된다. 잘못된 파일을 주면
종료 코드 2로 끝나며 image_only로 자동 전환하지 않는다.

**알려진 gap:** 사용자는 "한 CNT segment가 같은 전극의 layer 조각끼리 연결된다"고 확인했다.
그러나 relations 모드는 아직 이 규칙을 적용하지 않는다(`same_electrode_required: false`, inactive).
따라서 서로 다른 전극 폴더 사이의 관계도 만들 수 있다. 이 규칙을 구현·검토하기 전에는
relations 결과를 실제 연결 판단에 의존하지 않는다. image_only는 이 관계 추출을 실행하지 않는다.

### 11.1 순서 파일 준비 — 검사 전에 하는 별도 준비

[configs/stack-order.json](../configs/stack-order.json)은 **2026-10-13 사용자가 공급한 13-name 참조**다.
해당 자료에 대한 참조이며 다른 자료의 물리 순서나 보편적인 CNT 위치를 주장하지 않는다.
파일을 PNG 입력 tree 밖에 둔다. 프로그램에는 CNT 이름별 위치가 하드코딩되어 있지 않다.

아래는 **문법만 보이는 합성 예시(syntax only)**다. 실제 위치를 지정하는 예제가 아니다.

```json
{
  "stack_order_schema_version": "dl1_png_stack_order_v1",
  "order": ["SYNTHETIC_TOP", "SYNTHETIC_CNT", "SYNTHETIC_BOTTOM"]
}
```

- 한 JSON의 `order` 목록을 모든 모델에 공통으로 적용한다. 위→아래로 전극 layer 이름과 CNT 이름을 섞어 적는다.
- 모델에 없는 이름은 정상이며 `ignored_names`에 남는다. 모델에 있는 모든 전극 layer/CNT 이름은 목록에 있어야 한다(흰색 CNT 파일도 포함).
- 이름 매칭은 `casefold`로 대소문자를 무시한다. PXL ↔ pxl 매칭은 `matched_pairs`에 기록되지만 manifest·identity·vocabulary에는 원본 pxl 철자가 남는다.
- 한 모델의 두 source 이름 또는 order의 두 항목이 대소문자만 다르면 오류다. NFC/NFD는 합치지 않는다.
- 같은 CNT 이름의 여러 파일은 하나의 order 위치를 공유하지만 source asset/component는 별개다. 관계 추출에서 파일을 합치지 않는다.
- 전극·cno의 숫자는 여기서도 provenance only다. 순서에 맞춰 원본 파일을 이름 변경하지 않는다.
- UTF-8(LF/CRLF, 한 개의 선행 BOM 허용)로 저장한다. UTF-16/UTF-32, NUL, 주석, trailing JSON, 중복 key, NaN/Infinity는 거부된다.
- root key는 `stack_order_schema_version`과 `order` 두 개뿐이다. `models`/`default` override는 없다. 이름은 비어 있거나 공백·control·unpaired surrogate를 포함하면 안 된다.
- 같은 전극 폴더의 같은 layer 중복은 여전히 오류다. 서로 다른 폴더의 같은 layer는 번호가 달라도 같은 이름 위치에 놓인다.
- 이미지 겹침, 파일 번호, CNT 이름 자체에서 상하 순서를 추정하지 않는다.

### 11.2 선택적 검사·내보내기·확인

POSIX에서는 아래 경로 변수를 사용자 값으로 설정한다. `STACK_ORDER_JSON`은 실제 입력의 참조이고
`RELATIONS_INSPECT_OUT`/`RELATIONS_EXPORT_OUT`은 입력 밖의 새 폴더다. grid 값은 합성 문법 예시일 뿐 권장값이 아니다.

```bash
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$RELATIONS_INSPECT_OUT" --stack-order "$STACK_ORDER_JSON"
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$RELATIONS_EXPORT_OUT" --stack-order "$STACK_ORDER_JSON" --export --metadata "$METADATA_JSON" --candidate-grid "$CANDIDATE_GRID"
.venv/bin/python -m psdl1.import_png --verify-bundle "$RELATIONS_EXPORT_OUT"
```

Windows (명령 프롬프트; 경로에 공백/한글이 있어도 한 인수로 전달):

```bat
".venv-cmd\Scripts\python.exe" -I -m psdl1.import_png --root "D:\png_root" --out "D:\out\relations-inspect1" --stack-order "D:\meta\stack-order.json"
".venv-cmd\Scripts\python.exe" -I -m psdl1.import_png --root "D:\png_root" --out "D:\out\relations-bundle1" --stack-order "D:\meta\stack-order.json" --export --metadata "D:\meta\meta.json" --candidate-grid 512
".venv-cmd\Scripts\python.exe" -I -m psdl1.import_png --verify-bundle "D:\out\relations-bundle1"
```

`--verify-bundle`은 단독 모드다. 다른 입력 flag나 `--stack-order`를 같이 주지 않는다.
원본 PNG와 순서 파일이 없어도 bundle에 저장된 document·hash·order evidence로 확인한다.
Windows 형태의 argv, native 경로의 공백/한글, CRLF/BOM/snapshot은 합성 테스트로 확인하지만
실제 Windows 호스트에서 실행됐다는 의미는 아니다. 기존 게시 방식과 Windows 제한은 8장 그대로다.

### 11.3 선택적 관계 규칙과 preview

preview의 **CNT mode: relations**와 `Resolved top-to-bottom order (user_supplied_order_file)`을 먼저 본다.
표시되는 0-based 위치는 모델에 실제 있는 이름으로 필터한 order이며 번호가 아니다.

1. 각 CNT 파일을 따로 4-neighbour component로 나눈다(사용자 확인됨).
2. component와 겹치는 전극 중 order_index가 가장 가까운 위/아래 layer를 먼저 선택한다.
3. 선택된 각 layer에서 전극이 하나여야 한다. 공통 교집합이 없거나 떨어진 영역이 여러 개면 미해결이다.
4. CNT·위·아래 전극의 공통 교집합 내부 픽셀을 공유 끝점으로 쓴다(centroid 거리, y/x 동률 순).
5. 가까운 pair가 실패해도 더 먼 pair로 넘어가지 않는다. CNT component당 관계는 최대 하나다.

건너뛴 layer는 `skipped_upper_layers`/`skipped_lower_layers`에서 전극과 cno를 나눠 기록한다.
같은 이름의 CNT 파일은 source 참조가 여러 개여도 한 plane으로 센다. 모델에 없는 참조 이름은
건너뛴 layer가 아니다. 건너뜀의 물리적 해석과 같은 layer의 다중 전극 해소 규칙은 아직 미정이다.

| preview 표시 | 뜻 |
|---|---|
| 초록 원/십자 | 추출된 공통 위치; 전기적 검증 아님 |
| 주황 사각형+원 | 건너뛴 layer가 있어 확인 필요 |
| 빨간 X | 미해결; 관계를 만들지 않음 |
| 파랑 / 분홍 | 선택된 위 / 아래 전극 |
| 회색 / 검정 / 노랑 | 다른 전극 / CNT 테두리 / 공통 교집합 |

| component 사유 | 뜻 |
|---|---|
| `missing_upper_overlap`, `missing_lower_overlap` | 해당 쪽에 겹치는 전극 없음 |
| `multiple_upper_electrodes_on_selected_layer`, `multiple_lower_electrodes_on_selected_layer` | 선택된 layer에 후보가 여럿 |
| `same_bound_element` | 동일 element를 양 끝점으로 사용할 수 없음 |
| `empty_common_intersection` | 두 전극이 CNT의 서로 다른 위치에서만 겹침 |
| `disconnected_common_intersection_possible_multicontact` | 공통 영역이 여러 개라 한 접촉으로 확정하지 않음 |
| `invalid_stack_order` | 순서 오류로 관계 추출이 막힘; 구체적 errors를 확인 |

미해결 foreground도 전체 CNT 이미지에 남는다. `complete`는 추출 범위이며 전기적 검증이 아니다.
이 규칙은 image_only에 적용되지 않으며 image_only의 geometry-only reason은 `cnt_image_only_no_relations`다.

### 11.4 순서·migration 오류와 조치

| 코드/진단 | 원인 | 조치 |
|---|---|---|
| `stack_order_read_error` | 파일 없음·directory·권한 오류 | 실제 파일과 접근 권한 확인; 절대 경로가 보고서에 복사되지 않음 |
| `stack_order_encoding` | UTF-8이 아님 또는 raw NUL | UTF-8로 저장(선행 BOM 한 개 허용) |
| `stack_order_json` | 주석·trailing content·잘못된 JSON | 표시된 line/column에서 strict JSON 문법 수정 |
| `stack_order_duplicate_key` | 중복 JSON key | 중복 제거; nested key도 해당 |
| `stack_order_nonfinite` | NaN/Infinity 또는 nonfinite 숫자 | 비유한 숫자 제거 |
| `stack_order_schema` | root가 object가 아님 | 두 key를 가진 object 사용 |
| `stack_order_fields` | key 누락/unknown(`models`/`default` 포함) | schema와 order 두 key만 남김 |
| `stack_order_version` | version/type이 다름 | 정확한 `dl1_png_stack_order_v1` 사용 |
| `stack_order_list_type` | order가 배열이 아님 | 이름 배열로 수정 |
| `stack_order_identifier` | 빈 이름, 양끝 공백, control, surrogate, 잘못된 type | 원본 layer 이름의 유효한 문자열 사용 |
| `stack_order_duplicate_name` | 목록의 exact 중복(모델에 없는 이름도 검사) | 항목 하나만 유지 |
| `stack_order_case_collision` | order/source의 casefold 이름 충돌 | 보고된 두 이름과 경로를 확인; source spelling을 임의로 병합하지 않음 |
| `stack_order_name_kind_collision` | 한 이름이 전극 layer와 cno 이름 모두로 쓰임 | 이름만으로 위치를 구분할 수 없으므로 원천 계약 확인 |
| `stack_order_missing_electrode_names` | 모델의 전극 layer가 목록에 없음 | 해당 이름을 실제 위치에 추가; 위치를 추측하지 않음 |
| `stack_order_missing_cnt_names` | 모델의 CNT 이름이 목록에 없음 | 흰색 파일도 이름을 추가; 임의 생략 금지 |
| `ambiguous_cnt_asset_identity` | 같은 CNT 이름·동일 decoded full mask | 모든 파일의 중복 여부와 stable source ID 정책을 확인; 번호 변경으로 해결 안 됨 |
| `invalid_stack_order` | 위 순서 오류가 있음 | 개별 errors 수정 후 새 output으로 다시 실행; 자동 image_only fallback 없음 |
| `unsupported PNG report_schema_version` | v1/v2 또는 unknown sidecar | 원본 PNG에서 현재 importer로 다시 생성 |
| `unsupported PNG algorithm_id/algorithm_version` | 현재 algorithm "2"가 아님 | 기존 endpoint를 재사용하지 말고 다시 생성 |
| `invalid PNG stack order evidence` | mode/matched_pairs/index/digest 등 불일치 | bundle를 수정·재봉인하지 말고 원본에서 다시 생성 |
| `unsupported bundle integrity version/algorithm` | v1/v2 또는 unknown integrity | 현재 v3 bundle을 새 directory에 생성 |

### 11.5 Migration과 모드 일관성

현재 sidecar는 **dl1_png_import_v3**, integrity는 **dl1_png_bundle_v3**, algorithm은 **"2"**다.
예전 v1/v2 sidecar와 번호를 모델 공통 위치로 본 설명은 **superseded(폐기된 과거 규칙)**다.
번호 비교 기반 동률·역전 검사와 순서 파일 없을 때의 export block도 현재 규칙이 아니다.

기존 sidecar의 version label만 바꾸거나 endpoint를 그대로 재사용하지 않는다.
원본 PNG·metadata·선택한 grid에서 새 output directory로 다시 가져오고 검증한다.
image_only가 기본이며, 관계가 필요하면 실제 order 파일을 명시해서 다시 생성한다.
이전 파일은 역사적 evidence로 남긴다. 바뀐 Product identity의 캐시/embedding은 재구축한다.

같은 source의 image_only와 관계가 생성된 relations export는 **Product input identities가 다르다**.
한 checkpoint/search 안에서 두 모드를 섞지 않는다. source geometry identity와 파일 번호 변경의
불변성은 서로 다른 모드의 모델 입력이 같다는 뜻이 아니다. 관계가 없는 모델은 strict 입력이
같을 수 있지만 checkpoint/search의 모드 정책은 일관되게 유지한다.

strict `dl1_shape_v1_draft`, `SCHEMA_VERSION`, `MODEL_INPUT_VERSION` 및 NPY/checkpoint 숫자 계산은 그대로다.
따라서 기존 manifest를 따로 떼어 generic NPY loader로 읽는 것은 여전히 가능하다.
이미 생성된 static HTML도 열릴 수 있다. 새 PNG verifier/preview **reader**가 과거 sidecar를 거부하는
것이지, 기존 NPY loader나 이미 열린 HTML을 소급해서 막는 기능은 아니다.
resume의 코드·입력 provenance 검사는 계속 적용되며 hash/version bypass는 승인하지 않는다.

## 12. 수치 저장소와의 연결

검색 결과 JSON은 `particle-sim-compare`의 `simcompare.dl_results`가 파일로 읽어 원본 CA·line/pixel 값 옆에 붙인다. DL 점수는 수치 값을 바꾸지 않는다. 절차는 그 저장소의 `docs/handoff/commands.md`에 있다.
