# DL1 Windows CMD offline bundle 검증 기록

현재 설치 및 설치 이후 실행 안내는 [CMD 가이드](WINDOWS_OFFLINE.md)를 따릅니다.

## 사용자 확인 범위와 DL1 범위

사용자가 실제 확인한 것은 compare의 Windows CMD 설치 / pip check / help까지입니다.
그 사실을 DL1 설치·수치 smoke·실자료 성공의 증거로 사용하지 않습니다.
DL1은 자체 psdl1 wheel, NumPy/PyTorch/Pillow와 Windows 전이 의존성 13개 wheel을 사용합니다.
compare용 pandas/pyarrow/zarr는 포함하지 않았고 compare 저장소도 수정하지 않았습니다.

이번 시작점은 `feat/png-input`의 `78a2f91` (이전 offline bundle 작업을 포함한 clean 상태)입니다.
원래 runtime baseline `11e3e4b2728e8754d2ab5b42d960ed5988678b1a`의 `src/psdl1`은
이번 수정에서도 바꾸지 않았습니다. 기존 커밋·환경·자료는 보존했습니다.
역사적 참고 snapshot은 compare `rewrite/line-pixel-rate`의
`d417ad847244ac0cd986f6195e23c4ba8d88d08c`입니다. 최신 compare 수정 상태를 복제한 것은 아닙니다.

## 최종 산출물

- ZIP: `outputs/windows_cp312_win_amd64.zip`, **150,903,435 bytes**.
- SHA256: `9d677ad8a5745422838f07fcd1045aa65dfc543d151020920f5c950546a7abdf`.
- 자체 wheel SHA256: `baaacaaa633bec279bb21594aa705e4777b9ddb3e219175d07f68a266bd8a46f`.
- Git 전달용: `bundles/windows_cp312_win_amd64.parts/`의 4개 part(각 최대 48,000,000 bytes),
  `parts.json`, `reassemble.py`. 최종 ZIP을 실제 재결합하여 검증했습니다.
- runtime 고정 버전은 이전 묶음과 같습니다: numpy 2.5.3, Pillow 12.3.0, torch 2.14.1+cpu,
  filelock 4.2.0, fsspec 2026.9.0, Jinja2 3.1.6, MarkupSafe 3.0.4, mpmath 1.3.0,
  networkx 3.7, setuptools 84.0.0, sympy 1.14.0, typing_extensions 4.16.0, psdl1[png] 0.1.0.
  hatchling 1.27.0은 연결된 빌더에서만 사용합니다. macOS requirements.lock는 재사용하지 않았습니다.
- 배포 ZIP에는 CMD/Python 설치기, verifier, 선택 smoke, local wheels, hash requirements,
  README와 docs의 usage/training/validation/data contract/CMD 가이드를 동봉했습니다.
  PowerShell/Bash 설치기는 포함하지 않습니다. 외부 다운로드는 연결된 배포 준비 빌더에만 있습니다.

## 변경 및 검증 결과

1. Windows 설치 진입점은 `install_windows_bundle.cmd`와 `install_windows_bundle.py`입니다.
   preinstalled CPython으로 venv 생성 → payload hash/inventory/target/closure 검증 → local wheels만
   해시 고정 설치 → 설치 버전/격리/DL1 source 검사 → pip check → CPU imports/help 순서입니다.
   기존 venv는 거절하고 `--reuse-venv`일 때 설치/업데이트 없이 정확 일치만 검사합니다.
   `--smoke`는 선택이며 설치/help와 수치 smoke 결과를 구분합니다.
2. 검증기는 현재 실행 중인 격리 `.venv-cmd`만 목록에서 제외하여 사용자 성공 순서인
   venv 생성 후 verify를 지원합니다. 미사용/임의 venv, system-site-packages, payload의
   누락/변조/추가 파일·디렉터리는 계속 거절합니다. 실제 최종 추출본에서 Mac CPython 3.12 venv를
   `.venv-cmd`로 만든 뒤 그 Python으로 verifier 실행을 통과했습니다.
3. README/usage/training/validation과 ZIP의 Windows 실행 예시는 모두 CMD입니다.
   입력 검사, grids/export/verify, 학습·재개, 검색, 평가를 명시적 venv python.exe와
   큰따옴표 경로로 안내합니다. `PY` 같은 launcher로 오인할 수 있는 placeholder도 제거했습니다.
   기존 POSIX 개발 명령과 역사적 검증 기록은 Windows 설치 절차와 구분합니다.
4. `.venv/bin/python -m pytest`: **845 passed, 1 skipped, 41 warnings**, 189.57초.
   skip은 기존 opt-in 40-epoch training test입니다.
   `tests/test_windows_bundle.py`: **58 passed**. 신규/재사용/실패 시 환경 보존,
   CPython 버전·아키텍처 거절, offline flags/실행 순서, active venv 예외와 strict payload 검사 등을 포함합니다.
5. Windows target wheel tag, Python 조건, 전이 closure와 공개 index의 원본 파일/hash,
   자체 wheel의 전체 src 내용, requirements hash와 파일 inventory 검증을 통과했습니다.
   네트워크 socket을 거절한 Windows target pip dry-run의 대상 13개는 모두 local `file:` URL입니다.
   재결합 ZIP hash/CRC, 추출 inventory, 최종 installer 파일 일치 및 실제 Python 설치기 변조 거절도 확인했습니다.
6. 별도 Mac CPython 3.12.13 venv에 **전달본과 같은 자체 wheel**을 설치했습니다.
   pip check 및 import_png/data/train/search/evaluate 5개 module의 help가 통과했습니다.
   source checkout/PYTHONPATH 없이 `python -I`와 network guard를 사용한 합성 smoke도 통과했습니다:
   PNG 2개 모델 × 2개 grid의 diagnostic PNG/JSONL 4세트 → verify-bundle →
   metadata/candidate-grid export → manifest validator → tiny CPU 1-step 학습 → checkpoint 검색.
   Mac용 torch 2.14.1/numpy 2.5.3/Pillow 12.3.0을 사용했고 pretrained 다운로드는 없었습니다.

증거는 `outputs/cmd-final-build.log`, `cmd-full-suite.log`, `cmd-regression.log`,
`cmd-doc-audit.json`, `cmd-pip-dry-run.{json,log}`, `cmd-final-roundtrip.log`,
`cmd-final-active-venv-check.log`, `cmd-help-checks.json`, `cmd-mac-smoke.log`,
`cmd installed smoke 한글/smoke-result.json`에 보존했습니다. outputs는 Git ignore 대상입니다.
이전 배포본은 outputs의 보관 폴더로 옮겼고, 전달용 ZIP과 Git 분할본은 CMD 버전으로 갱신했습니다.
이번 수정은 commit/push하지 않았습니다.

## 실제 호스트에서 남은 검증

Windows DL1 설치, CMD wrapper의 실제 실행/quoting, Microsoft VC++ DLL 로딩,
수치 smoke 및 실자료 처리는 이 Mac에서 실행할 수 없었습니다. Python 설치기의 orchestration은
단위 테스트로 검증한 것이며 실제 Windows 성공을 뜻하지 않습니다.
16GB/18,000 PNG 처리시간·peak RAM·검색 품질도 미검증입니다. timm/torchvision/safetensors,
pretrained weights와 GPU/CUDA는 이 CPU/PNG 배포 범위에서 제외합니다.

## Git CRLF 전달 오류 후속 수정

`reassembly script hash differs`의 한 원인으로 Windows Git `core.autocrlf=true`의
LF→CRLF 변환을 실제 Git checkout으로 재현했습니다. LF 복구 후 원래 manifest hash와
정확히 일치했습니다. `.gitattributes`로 분할 폴더의 모든 파일에 `-text`를 적용하고,
helper 원본에는 `text eol=lf`를 적용했습니다. 무결성 검사는 그대로 유지합니다.

회귀 테스트는 보호 설정이 없을 때 같은 오류를 재현하고, 보호 설정이 있을 때 동일 Git 설정으로
checkout한 전달 파일이 원본과 바이트 단위로 같고 재결합까지 통과함을 검증합니다.
`tests/test_windows_bundle.py`: **60 passed** (0.93초).
기존 분할본·최종 ZIP hash는 변경하지 않았습니다. 이미 변환된 파일의 해시 확인 후 LF 복구 명령은
[CMD 가이드](WINDOWS_OFFLINE.md#git-반입-중-reassembly-script-hash-differs가-발생할-때)에 있습니다.
이 재현은 Git의 파일 변환 동작 검증이며 실제 Windows DL1 실행 검증은 아닙니다.
