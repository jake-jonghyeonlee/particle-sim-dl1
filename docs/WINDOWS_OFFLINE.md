# Windows 폐쇄망 설치: CPython 3.12 x64 / CPU

이 경로는 `uv sync`, `uv run`, 인터넷 연결을 사용하지 않습니다. 연결된 Mac에서 준비한
Windows wheel과 DL1 자체 wheel을 반입하여 `pip --no-index --only-binary=:all: --require-hashes`로 설치합니다.
`requirements.lock`는 macOS arm64 snapshot이며 Windows 설치에 사용하지 않습니다.
참고 UX: particle-sim-compare `rewrite/line-pixel-rate`, commit
`d417ad847244ac0cd986f6195e23c4ba8d88d08c`의 build/verify/PowerShell 설치기·tests·usage·ZIP.
DL1 application baseline: `11e3e4b2728e8754d2ab5b42d960ed5988678b1a` (diagnostic 보완 포함).

## 사전 준비

- Windows 10/11 **x64**, CPython **3.12.x x64**의 정식 설치가 필요합니다. x86, ARM64,
  다른 Python 버전은 설치기가 거절합니다. Python 자체와 OS 설치기는 이 묶음에 없습니다.
  Python 설치 시 pip/venv(ensurepip)를 포함하십시오. `py.exe` Launcher가 없으면 설치기에
  `-PythonExe 'C:\Python312\python.exe'`를 명시합니다. embeddable ZIP Python은 지원하지 않습니다.
- 네이티브 64비트 Windows PowerShell **5.1 이상**에서 실행합니다. 관리자 권한은 설치기에
  필요하지 않지만 Python/VC runtime 사전 설치나 조직 정책은 관리자 처리가 필요할 수 있습니다.
- **Microsoft Visual C++ v14 Redistributable x64**를 연결 환경에서 별도로 확보하여 사전 설치합니다.
  [Microsoft 공식 안내](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist),
  [PyTorch Windows DLL FAQ](https://docs.pytorch.org/docs/main/notes/windows.html).
  공식 CPU wheel의 DLL도 이 OS runtime을 필요로 할 수 있습니다. DLL 오류 시 임의 DLL 다운로드나
  온라인 pip 업데이트 대신 사전 조건과 동봉 torch/lib를 확인하십시오. CUDA/컴파일러는 필요하지 않습니다.
- Windows 호스트는 여기서 접근하지 못했습니다. **실제 Windows / PowerShell 5.1 / DLL 로딩은 미검증**입니다.
  Mac에서 Windows wheel tag·메타데이터 전이 의존성·공개 배포 hash·자체 wheel 소스·전달 무결성을 검증합니다.
- bundle wheel 압축 해제와 venv에 약 1GB 이상의 여유가 필요하고, 전달 ZIP/분할본/임시 파일을
  함께 보관하므로 **최소 3GB 여유**를 준비하십시오(입력·출력·학습 저장 공간 별도).
  16GB RAM은 18,000 PNG 전체의 처리/학습 성공을 보장하지 않습니다. 원본 18,000장을 바로
  export/train하지 말고 작은 대표 제품으로 inventory와 peak RAM을 먼저 확인하십시오.
  import는 한 모델씩 처리하지만 제품 내 mask 수에 따라 커지며 `load_manifest`/training은 전체 제품을 eager 로드합니다.

## 반입 후 4줄: 설치와 첫 실행

저장소 루트, 새 `outputs\offline` 경로 기준입니다. 기존 출력이나 venv는 자동 삭제하지 않습니다.
`bundles\windows_cp312_win_amd64.parts`의 **모든 파일**을 함께 반입합니다.

```powershell
py -3.12 .\bundles\windows_cp312_win_amd64.parts\reassemble.py join .\bundles\windows_cp312_win_amd64.parts --output .\outputs\windows_cp312_win_amd64.zip
Expand-Archive -LiteralPath '.\outputs\windows_cp312_win_amd64.zip' -DestinationPath '.\outputs\offline'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\outputs\offline\windows_cp312_win_amd64\install_and_smoke.ps1' -VenvPath '.\outputs\psdl1-venv'
& '.\outputs\psdl1-venv\Scripts\python.exe' -I -m psdl1.import_png --help
```

로컬 ZIP만 반입해도 됩니다. 그 경우 1줄 대신 신뢰한 SHA256과
`Get-FileHash -Algorithm SHA256 -LiteralPath '.\outputs\windows_cp312_win_amd64.zip'` 결과를 비교합니다.
ZIP은 저장소 `outputs/windows_cp312_win_amd64.zip`, 독립 hash는 같은 이름의 `.zip.sha256`입니다.
ZIP 또는 parts.json까지 공격자가 함께 바꾸면 자체 hash만으로 출처 인증은 할 수 없습니다.
최종 전달 SHA256을 별도 신뢰 경로로 보관하십시오.

설치기는 immutable bundle 전체의 파일/디렉터리 목록과 hash, Windows target wheel tag,
Python 조건, 전이 의존성을 확인한 뒤 **새** venv를 만듭니다. 패키지 저장소 설정을 차단하고,
해시가 고정된 동봉 local wheel만 설치합니다. pip check, CPU import, 설치된 소스 일치 검증과
smoke까지 통과해야 성공을 표시합니다. bundle 안에 venv/smoke 출력을 만들지 마십시오.
한글·공백 경로를 인자로 전달하며 활성화 스크립트는 필요 없습니다. 가능한 짧은 로컬 경로(예: C:\dl1)를 사용하십시오.
  긴 경로 오류가 있으면 조직 정책에 따라 Windows/Python의 long-path 지원도 사전 설정하십시오.
기본 smoke 폴더는 bundle의 부모에 고유 이름으로 생성되고 PNG/JSONL/로그/checkpoint/search 결과가 보존됩니다.
`-SmokeOut 'D:\검증 결과\새 폴더'`로 지정할 수도 있습니다.

기존 venv가 있으면 기본적으로 설치를 중단합니다. 새 경로를 권장합니다. 의도적 재사용은
같은 명령 뒤에 `-ReuseVenv`를 붙입니다. 이때 **설치/업데이트하지 않고** 격리 venv인지,
모든 런타임의 정확 버전과 추가 패키지 유무, psdl1 코드 hash를 검사하고 smoke만 다시 실행합니다.
불일치는 중단하며 기존 파일을 삭제하지 않습니다. 실패 후에도 기존 환경과 smoke 증거는 보존됩니다.
수정한 동일 버전 의존성 wheel을 가져오는 용도가 아니므로 필요한 경우 새 venv를 쓰십시오.

## 실제 PNG 실행 (uv 대신 명시적 python.exe)

```powershell
& '.\outputs\psdl1-venv\Scripts\python.exe' -I -m psdl1.import_png --root 'D:\원본 PNG' --out 'D:\결과\새 inventory' --no-preview
& '.\outputs\psdl1-venv\Scripts\python.exe' -I -m psdl1.import_png --root 'D:\원본 PNG' --out 'D:\결과\새 grids' --inspect-grids 64 128 --no-preview 2> '.\grid-progress.jsonl'
& '.\outputs\psdl1-venv\Scripts\python.exe' -I -m psdl1.import_png --verify-bundle 'D:\결과\새 grids'
& '.\outputs\psdl1-venv\Scripts\python.exe' -I -m psdl1.import_png --root 'D:\원본 PNG' --out 'D:\결과\새 export' --export --metadata 'D:\설정\metadata.json' --candidate-grid 128 --no-preview
& '.\outputs\psdl1-venv\Scripts\python.exe' -I -m psdl1.import_png --verify-bundle 'D:\결과\새 export'
& '.\outputs\psdl1-venv\Scripts\python.exe' -I -m psdl1.data 'D:\결과\새 export\manifest.json' --report 'D:\결과\validation.json'
```

64/128은 **예제**이며 실자료의 승인된 grid가 아닙니다. stderr에는 JSON progress와 일반 경고가
혼합될 수 있습니다. `--no-preview`여도 grid diagnostic PNG/JSONL은 생성됩니다.
metadata는 실제 모델 이름/lineage/design으로 작성하며 합성 smoke 값을 실자료에 복사하지 마십시오.
기본 image_only는 CNT 영상만 보존하고 연결을 추론하지 않습니다. 선택 `--stack-order`와
relations의 물리 의미·미확정 사항은 `docs/USAGE.md`를 따르십시오. export가 실제 grid/전기 연결의 승인은 아닙니다.

CPU core는 `psdl1.train`, `psdl1.search`, `psdl1.evaluate`와 기본 tiny backbone을 포함합니다.
명시한 config/labels/split/checkpoint로 같은 python.exe를 사용합니다. CPU 학습 1-step smoke는
설치 통합 증거이며 실자료 검색 품질·18,000장 규모/속도/RAM 검증이 아닙니다.
**timm, torchvision, safetensors, 사전학습 weights, GPU/CUDA는 이 묶음에 없습니다.**
이 옵션이 필요한 config/checkpoint는 그대로 실행할 수 없습니다. 별도 승인된 platform-specific
wheel/weights 전달과 검증이 필요합니다. 자동 pretrained 다운로드나 온라인 fallback은 사용하지 않습니다.

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

## 연결 환경 재생성 및 Git 전달

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
스크립트/문서를 검토해 함께 add/commit하십시오(이번 작업은 commit/push하지 않음).
각 part는 최대 **48,000,000 bytes**, parts.json에는 순서/개별 hash/최종 ZIP hash가 있습니다.
재결합기는 누락·추가·변조 파일, 순서/이름 오류를 거절하고 기존 ZIP을 덮어쓰지 않습니다.
분할로 전송 총량이 줄지는 않습니다. 저장소 clone 크기가 우려되면 ZIP을 별도 로컬 매체로 반입하십시오.
