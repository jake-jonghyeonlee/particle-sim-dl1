# Windows CPU offline bundle 검증 기록

2026-10-15, 인터넷 가능한 macOS arm64에서 Codex가 직접 생성·검증했습니다.
Windows 호스트 접근, 외부 업로드/배포, commit/push는 수행하지 않았습니다.
설치·반입 명령과 버전 차이는 [Windows 폐쇄망 가이드](WINDOWS_OFFLINE.md)에 있습니다.

## 기준과 산출물

- DL1: `feat/png-input`, baseline `11e3e4b2728e8754d2ab5b42d960ed5988678b1a`, 시작 상태 clean.
  runtime source는 변경하지 않았습니다. 제공된 diagnostic 보완을 포함한 `src/psdl1` 전체와 자체 wheel 내용을 대조했습니다.
- 참고 저장소(읽기만): particle-sim-compare `rewrite/line-pixel-rate`,
  `d417ad847244ac0cd986f6195e23c4ba8d88d08c`. build/verify/install/runtime/build constraints/tests/usage와
  실제 reference ZIP(60,631,353 bytes, 16 wheels)을 확인했습니다.
- ZIP: `outputs/windows_cp312_win_amd64.zip`, **150,827,776 bytes** (약 143.84 MiB).
- ZIP SHA256: `a1fe46839ef0cb8c234dd916a8b0a2e3affe6f706ded84d401d346a85f1c750e`.
- 자체 wheel: `psdl1-0.1.0-py3-none-any.whl`, SHA256
  `345a6dcb527c117e99a4ff9f4f25dd26d47c11428c5790c19d3b88f590b29ff5`.
- 총 13 wheels: psdl1[png], numpy 2.5.3, Pillow 12.3.0, torch 2.14.1+cpu,
  filelock 4.2.0, fsspec 2026.9.0, Jinja2 3.1.6, MarkupSafe 3.0.4, mpmath 1.3.0,
  networkx 3.7, setuptools 84.0.0, sympy 1.14.0, typing_extensions 4.16.0.
- wheelhouse 등 ZIP 내부 payload는 154,304,630 bytes; wheels 전체 비압축 파일 합계는
  573,737,549 bytes입니다. venv/pip/bytecode/입출력 및 중복 전달 파일 공간은 추가로 필요합니다.
- Git 반입용 `bundles/windows_cp312_win_amd64.parts/`: 48,000,000 bytes × 3개 +
  6,827,776 bytes × 1개, `parts.json`, `reassemble.py`.
  각 part hash와 최종 hash는 parts.json에 있습니다. ZIP/임시환경은 ignore 처리했고 분할본은 아직 untracked입니다.

## 통과한 검증

1. 연결 환경의 uv 0.11.23으로 Windows CPython 3.12 x64 전용 runtime을 해석했습니다.
   PyPI JSON / 공식 PyTorch CPU index의 파일명·SHA256과 실제 다운로드를 비교했습니다.
   Pillow 12.3.0의 cp312-win_amd64 wheel이 존재하여 계약 pin 변경은 없었습니다.
   build backend hatchling은 1.27.0으로 고정했습니다.
2. Windows wheel tag, Requires-Python, Windows 환경 marker 및 PNG extra를 포함한 전이 의존성
   closure, requirements의 정확 pin/hash와 전체 파일·디렉터리 목록을 확인했습니다.
3. Python socket connect/connect_ex/create_connection/getaddrinfo를 거절한 상태에서 Windows target
   `pip install --dry-run --ignore-installed --no-index --no-cache-dir --only-binary=:all: --require-hashes`
   를 실행했습니다. report의 설치 대상 13개 모두 `file:` URL이며 온라인 해석은 없었습니다.
   이는 cross-target 설치 해석 검증이며 Windows 바이너리 실행은 아닙니다.
4. 실제 bundle의 hardlink 복제본에서 wheel 누락, 설치기 변조, 예상 밖 파일·빈 디렉터리를
   각각 거절했습니다. 원본 아티팩트는 수정하지 않았습니다.
5. 4개 분할 파일로 실제 재결합한 ZIP의 최종 SHA256·CRC를 확인하고, 새 폴더에 압축 해제한 뒤
   strict inventory/hash/closure를 다시 통과했습니다. 추출된 설치기·verify·smoke가 저장소 최종 파일과 같습니다.
6. 별도 **새 macOS CPython 3.12.13 venv**에 전달본과 바이트가 동일한 자체 wheel을 설치했습니다.
   이 호스트의 numpy 2.5.3 / Pillow 12.3.0 / torch 2.14.1(macOS wheel)로 `pip check` 및
   `python -I` 설치 smoke를 통과했습니다. 기존 프로젝트 .venv 의존성은 변경하지 않았습니다.
   smoke는 source checkout 경로 주입 없이 site-packages의 비편집 설치를 확인하며 socket을 차단합니다.
   합성 PNG 2개 모델 × 2개 grid의 diagnostic PNG/JSONL 4세트 → verify-bundle →
   metadata/candidate-grid export → verify-bundle 및 manifest validator →
   tiny CPU 1-epoch/1-step 학습 → checkpoint 검색(eligible 1)을 검증했습니다.
   실제 pretrained weights나 다운로드는 사용하지 않았습니다.
7. `.venv/bin/python -m pytest`: **831 passed, 1 skipped**, 189.85초.
   skip은 기존 명시적 opt-in 40-epoch training smoke입니다. 새 acceptance matrix 7개는 이후 별도 실행에 포함했습니다.
   `.venv/bin/python -m pytest tests/test_windows_bundle.py`: **51 passed**.
   `git diff --check` 통과. 테스트의 기존 grid-warning 41개는 실패가 아닙니다.

빌드 및 상세 증거는 로컬 `outputs/windows-final-build.log`, `outputs/windows_cp312_win_amd64/manifest.json`,
`outputs/windows-pip-dry-run.{log,json}`, `outputs/actual-artifact-negative-checks.json`,
`outputs/full-suite.log`, `outputs/windows-bundle-tests.log`, `outputs/final-mac-smoke.log`,
`outputs/final installed smoke 한글/smoke-result.json`에 보존했습니다. outputs는 Git으로 자동 전달되지 않습니다.

## 남은 실제 환경 검증

Windows CPython의 wheel 설치, Microsoft VC++ DLL 로딩, 실제 PowerShell 5.1 및 Windows의
경로/파일 게시 동작은 실행하지 못했습니다. 설치기에 PS7 전용 문법 없이 ASCII 스크립트와 인자 경로 전달을
사용하고, 합성 smoke는 한글·공백 경로를 포함합니다. 이 사실은 Windows 성공의 증거가 아닙니다.
Windows에서는 문서의 install_and_smoke.ps1을 실행하여 해당 호스트의 증거를 생성해야 합니다.

CPU PNG 검사/진단/export 및 tiny core train/search 범위입니다. timm/torchvision/safetensors,
사전학습 weights, GPU/CUDA는 제외했습니다. 16GB/18,000 PNG의 peak RAM·처리시간·검색 품질을
검증하지 않았으며 전체 eager training/loader에는 별도 pilot과 자원 검토가 필요합니다.
기존 venv는 자동 삭제/업데이트하지 않으며 재사용은 명시적 -ReuseVenv 일치 검사만 지원합니다.
