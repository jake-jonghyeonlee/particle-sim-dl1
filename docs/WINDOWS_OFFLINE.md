# Windows 폐쇄망 설치: CMD + 설치된 Python + 로컬 wheel

Windows에서는 **명령 프롬프트(CMD)**를 사용합니다. 사용자 폴더에 ZIP을 풀고 사전 설치된
CPython 3.12 x64로 venv를 만든 뒤 동봉 wheel만 설치합니다. PowerShell 설치기는 제거했습니다.
`uv`, Bash 설치 스크립트, 인터넷 다운로드, 관리자 권한은 이 설치 과정에 필요하지 않습니다.
연결 환경의 배포 준비용 builder만 외부 wheel을 다운로드합니다.

compare에서 사용자가 실제 확인한 범위는 이 CMD 순서의 설치 / pip check / help까지입니다.
그 결과는 **DL1 설치·실행, 수치 smoke, 실제 자료 성공의 근거가 아닙니다.**
DL1은 자체 psdl1 wheel과 NumPy/PyTorch/Pillow 의존성 및 별도 검증을 사용합니다.
참고 저장소는 변경하지 않습니다. 기존 참고 snapshot은 `rewrite/line-pixel-rate`,
`d417ad847244ac0cd986f6195e23c4ba8d88d08c`입니다.

## 사전 조건과 압축 해제

- Windows 10/11 x64와 **CPython 3.12.x x64** 정식 설치(pip/venv/ensurepip 포함)가 필요합니다.
  Python 자체는 ZIP에 없습니다. x86/ARM64, 다른 Python, embeddable Python은 지원하지 않습니다.
- Microsoft Visual C++ v14 Redistributable **x64**를 별도로 반입하여 사전 설치하십시오.
  [Microsoft 안내](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist),
  [PyTorch DLL FAQ](https://docs.pytorch.org/docs/main/notes/windows.html).
  DLL 로딩 실패 시 온라인 pip 업데이트나 임의 DLL 다운로드를 하지 마십시오.
- 탐색기로 ZIP을 **쓰기 가능한 사용자 폴더**에 풀고 그 안의 `windows_cp312_win_amd64`로 이동합니다.
  `Program Files`, Windows 시스템 폴더, 읽기 전용 공유 폴더는 피하십시오.
  관리자 CMD나 실행 정책 변경은 필요하지 않습니다. 가능한 짧은 로컬 경로를 쓰십시오.
- ZIP과 venv 등의 공간으로 최소 3GB 여유를 준비하십시오(입출력·학습 저장 공간 별도).
  16GB RAM으로 18,000 PNG 전체 처리/학습이 가능하다는 뜻은 아닙니다.
  작은 대표 제품으로 inventory/peak RAM부터 확인하십시오. 제품 내 mask가 많으면 커지고,
  `load_manifest`/training은 전체 제품을 eager 로드합니다.

ZIP만 반입한 경우 CMD에서 다음 4줄로 압축 해제·설치·help 확인을 수행합니다.
`D:\반입\windows_cp312_win_amd64.zip`은 실제 ZIP 경로로 바꾸고, 출력은 새 경로를 사용하십시오.

```bat
py -3.12 -m zipfile -e "D:\반입\windows_cp312_win_amd64.zip" "%USERPROFILE%\dl1-offline"
cd /d "%USERPROFILE%\dl1-offline\windows_cp312_win_amd64"
install_windows_bundle.cmd
.venv-cmd\Scripts\python.exe -I -m psdl1.import_png --help
```

`py.exe`가 없으면 위 `py -3.12` 대신 설치된 `"C:\Python312\python.exe"`를 사용하고,
설치는 `"C:\Python312\python.exe" -I install_windows_bundle.py`로 실행하십시오.
CMD wrapper는 같은 Python 설치기를 실행하며 다운로드하지 않습니다. 다른 폴더에서 wrapper를
실행해도 bundle 경로를 기준으로 설치합니다. 공백/한글 경로는 반드시 큰따옴표로 감싸십시오.

## 수동 설치: 사용자 성공 순서와 동일한 방식

ZIP 안의 폴더를 현재 CMD 경로로 한 뒤 실행합니다. **`.venv-cmd`가 이미 있으면 아래 venv 생성과
pip 설치를 다시 실행하지 마십시오.** 새 압축 해제 경로를 쓰거나 아래 명시적 재사용 검사를 쓰십시오.
기존 환경을 삭제하거나 덮어쓰지 않습니다. `requirements.lock`는 macOS snapshot이므로 쓰지 않습니다.

```bat
set "PIP_CONFIG_FILE=nul"
set "PIP_NO_INDEX=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
py -3.12 -m venv .venv-cmd
.venv-cmd\Scripts\python.exe -I verify_windows_bundle.py .
.venv-cmd\Scripts\python.exe -I -m pip --isolated --disable-pip-version-check install --no-index --no-cache-dir --only-binary=:all: --require-hashes --find-links=wheelhouse -r requirements-windows.txt
.venv-cmd\Scripts\python.exe -I -m pip --disable-pip-version-check check
.venv-cmd\Scripts\python.exe -I verify_windows_bundle.py . --installed-prefix .venv-cmd
.venv-cmd\Scripts\python.exe -I -m psdl1.import_png --help
```

자동 설치기도 이 순서로 venv 생성 → 전체 hash/inventory/Windows target 의존성 검증 → 로컬 wheel 설치
→ 설치 버전·격리·DL1 코드 검사 → pip check → CPU import/help 확인을 합니다. 숫자 smoke는 기본 실행하지
않습니다. 네트워크 fallback, source checkout/PYTHONPATH 주입, editable install은 사용하지 않습니다.
검증기는 **현재 실행 중인 격리 venv `.venv-cmd` 한 폴더만** payload 목록에서 제외합니다.
다른 파일·빈 디렉터리·임의 venv는 계속 거절합니다. venv 밖의 payload는 수정하지 마십시오.

기존 환경 명시적 재사용(설치/업데이트하지 않고 정확 일치 검사만):

```bat
install_windows_bundle.cmd --reuse-venv
```

다른 venv 경로는 `--venv "D:\내 환경\dl1"`로 지정할 수 있지만 bundle 밖이어야 합니다.
실패한 환경과 로그는 보존됩니다. 검증 실패를 우회하지 말고 원인을 고친 뒤 새 압축 해제 경로에서
다시 설치하십시오. 수동으로 venv를 만들었다면 자동 설치에는 `--reuse-venv`를 지정합니다.

## 선택: DL1 수치 smoke와 실제 PNG 실행

설치/import/help 성공과 수치 smoke 성공은 별도입니다. 설치 후 다음 명령으로 합성 PNG 2개 모델의
2개 grid 진단 PNG·JSONL → verify-bundle → metadata/candidate-grid export → manifest 검증 →
tiny CPU 1-step 학습·검색을 검사할 수 있습니다. 새 출력 폴더에 모든 증거를 보존합니다.

```bat
.venv-cmd\Scripts\python.exe -I smoke_windows_bundle.py --out "..\dl1-smoke-new"
```

자동 설치에 함께 요청하려면 `install_windows_bundle.cmd --smoke`, 기존 정확 일치 환경이면
`install_windows_bundle.cmd --reuse-venv --smoke`를 씁니다. 기본 smoke 출력은 bundle 부모의 고유 폴더입니다.
이 smoke도 실자료 품질·18,000장 규모 성공이나 실제 grid 승인은 아닙니다.

```bat
.venv-cmd\Scripts\python.exe -I -m psdl1.import_png --root "D:\원본 PNG" --out "D:\결과\새 inventory" --no-preview
.venv-cmd\Scripts\python.exe -I -m psdl1.import_png --root "D:\원본 PNG" --out "D:\결과\새 grids" --inspect-grids 64 128 --no-preview 2> "grid-progress.jsonl"
.venv-cmd\Scripts\python.exe -I -m psdl1.import_png --verify-bundle "D:\결과\새 grids"
.venv-cmd\Scripts\python.exe -I -m psdl1.import_png --root "D:\원본 PNG" --out "D:\결과\새 export" --export --metadata "D:\설정\metadata.json" --candidate-grid 128 --no-preview
.venv-cmd\Scripts\python.exe -I -m psdl1.import_png --verify-bundle "D:\결과\새 export"
.venv-cmd\Scripts\python.exe -I -m psdl1.data "D:\결과\새 export\manifest.json" --report "D:\결과\validation.json"
```

64/128은 예제이며 실자료 grid 승인이 아닙니다. 실제 grid는 별도로 확인하십시오.

metadata는 실제 모델 이름/lineage/design으로 작성하십시오. stderr에는 JSON progress와 일반 경고가
혼합될 수 있으며 `--no-preview`도 grid diagnostic PNG/JSONL은 생성합니다. 기본 image_only는 CNT
영상만 보존하고 연결을 추론하지 않습니다. 선택 relations의 물리 의미·미확정 사항은 `docs/USAGE.md`를
따르십시오. CPU core의 tiny train/search/evaluate를 포함하며 **timm, torchvision, safetensors,
pretrained weights, GPU/CUDA는 제외**합니다. 이 옵션이 필요한 config/checkpoint는 별도 준비가 필요합니다.

**실제 DL1 Windows 호스트에서 설치·CMD·DLL 로딩·수치 smoke·실자료 처리를 검증하지 못했습니다.**
Mac 검증 결과와 compare의 사용자 성공을 Windows DL1 성공으로 표현하지 않습니다.

## 설치 이후 CMD 실행: 검사·학습·재개·검색·평가

같은 CMD 창의 bundle 폴더에서 실행합니다. 아래 변수는 설치된 Python의 절대 경로입니다.
실제 입력/설정/출력 경로와 제품 ID로 예제를 바꾸십시오. 학습 config/labels/split과 평가 자료는
사용자가 준비하며 bundle에 실제 자료가 포함되어 있지 않습니다. config의 상대 경로는 현재 작업 폴더
기준이므로 다른 폴더에서 실행할 때는 config 내부 데이터 경로도 확인하십시오.

```bat
set "DL1PY=%CD%\.venv-cmd\Scripts\python.exe"
"%DL1PY%" -I -m psdl1.import_png --root "D:\원본 PNG" --out "D:\결과\새 검사" --inspect-grids 64 128 --no-preview
"%DL1PY%" -I -m psdl1.import_png --verify-bundle "D:\결과\새 검사"
"%DL1PY%" -I -m psdl1.data "D:\결과\새 export\manifest.json" --report "D:\결과\validation.json"
"%DL1PY%" -I -m psdl1.train --config "D:\설정\train.json"
"%DL1PY%" -I -m psdl1.train --config "D:\설정\train.json" --resume "D:\학습\dl1.pt"
"%DL1PY%" -I -m psdl1.search --ckpt "D:\학습\dl1.pt" --query-manifest "D:\입력\query.json" --query "실제-제품-ID" --candidates-manifest "D:\입력\candidates.json" --topk 5 --batch-size 32 --out "D:\결과\search.json"
"%DL1PY%" -I -m psdl1.evaluate --ckpt "D:\학습\dl1.pt" --manifest "D:\입력\manifest.json" --split "D:\설정\split.json" --relations "D:\설정\relations.csv" --hard-negatives "D:\설정\hard-negatives.json" --topk 5 --batch-size 32 --out "D:\결과\evaluation.json"
```

첫 검사 예제의 grid는 승인값이 아닙니다. 학습은 제공한 metadata/labels/split 및 grid gate를 요구하며,
경고 gate를 자동 우회하지 않습니다. 검색·평가는 준비된 checkpoint와 일치하는 입력/설정을 사용합니다.
평가에 hard-negative 목록이 없으면 해당 인자를 생략할 수 있으며 결과는 unavailable 사유를 보존합니다.
CMD의 환경 변수는 `%DL1PY%`, 경로 인용은 큰따옴표입니다. 위 명령은 한 줄씩 실행합니다.
학습 config의 tiny CPU 범위만 이 배포에 포함됩니다. 자세한 계약은 저장소 `docs/training.md`,
`docs/validation.md`와 `docs/USAGE.md`를 따르십시오.

## 고정 버전과 출처

Windows wheel 12개 + 자체 `psdl1[png]==0.1.0` wheel 1개입니다.

| 패키지 | Windows 고정 버전 | macOS snapshot과 차이 |
|---|---|---|
| numpy | 2.5.3 | 동일 버전, Windows cp312 wheel |
| Pillow | 12.3.0 | 계약 유지; PyPI에 실제 cp312-win_amd64 wheel 있음 |
| torch | 2.14.1+cpu | 2.14.1과 같은 release, 공식 Windows CPU 전용 build |
| filelock | 4.2.0 | snapshot 4.0.12 대신 Windows 해석 결과를 별도 고정 |
| fsspec | 2026.9.0 | 동일 |
| Jinja2 / MarkupSafe | 3.1.6 / 3.0.4 | 동일; MarkupSafe Windows wheel |
| mpmath / networkx | 1.3.0 / 3.7 | 동일 |
| setuptools / sympy | 84.0.0 / 1.14.0 | 동일; torch의 Windows 전이 의존성 |
| typing_extensions | 4.16.0 | 동일 |

선택 timm/HuggingFace/torchvision와 개발용 pytest 등 macOS snapshot의 나머지는 이 CPU/PNG
범위에서 제외했습니다. 공개 index에 없는 버전을 임의로 대체하지 않았습니다. runtime pins는
`scripts/windows-runtime.in`, 실제 platform 해석 결과/hash는 bundle의
`build-evidence/resolved-windows.txt`, 최종 설치 hash는 `requirements-windows.txt`입니다.
PyPI JSON과 [공식 PyTorch CPU index](https://download.pytorch.org/whl/cpu/torch/)의
파일별 원본 URL/hash를 `manifest.json`의 wheels/source에 기록합니다. 런타임 lock은 URL을 포함하지 않습니다.
빌드 시 hatchling **1.27.0**을 pyproject와 build constraints에 고정했습니다. 폐쇄망에서는 빌드하지 않습니다.

## 연결 환경 배포 준비 및 Git 전달

기존 경로는 보존하므로 재생성 시 새 `--output`/`--parts`를 지정합니다. 연결된 빌더에
Python+pip와 uv **0.11.23**이 필요합니다. 인터넷 다운로드는 이 빌더에서만 허용됩니다.

```bash
python3 scripts/build_windows_bundle.py \
  --output outputs/windows_cp312_win_amd64 \
  --parts bundles/windows_cp312_win_amd64.parts
python3 scripts/verify_windows_bundle.py outputs/windows_cp312_win_amd64
python3 scripts/transfer_windows_bundle.py join bundles/windows_cp312_win_amd64.parts \
  --output outputs/rejoined-windows.zip
```

빌더는 target `x86_64-pc-windows-msvc`, Python 3.12로 해석하고 pip download에
`--platform win_amd64 --implementation cp --python-version 3.12 --abi cp312 --abi abi3 --abi none`
을 명시합니다. 해시가 확인된 전이 deps 전체만 복사한 뒤 자체 wheel 소스와 src를 대조합니다.
아티팩트에는 build 시각/환경이 기록되므로 ZIP hash가 매번 같다는 뜻의 byte 재현성은 보장하지 않습니다.

GitHub 단일 파일 100MB 제한 때문에 ZIP을 직접 add/commit하지 않습니다. outputs와 *.zip은
ignore 대상입니다. Git 전달이 필요하면 준비한 `bundles/windows_cp312_win_amd64.parts/`와
스크립트/문서를 검토해 함께 add/commit하십시오(이번 수정에서는 commit/push하지 않음).
각 part는 최대 **48,000,000 bytes**, parts.json에는 순서/개별 hash/최종 ZIP hash가 있습니다.
재결합기는 누락·추가·변조 파일, 순서/이름 오류를 거절하고 기존 ZIP을 덮어쓰지 않습니다.
분할로 전송 총량이 줄지는 않습니다. 저장소 clone 크기가 우려되면 ZIP을 별도 로컬 매체로 반입하십시오.
