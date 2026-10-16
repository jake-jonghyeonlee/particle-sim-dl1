# grid 진단 이미지 업데이트 — f74a718

기존 Windows 오프라인 ZIP으로 설치한 환경에 최신 psdl1 코드만 교체합니다.
Python 3.12 x64와 기존 numpy/CPU torch/Pillow 의존성이 설치된 `.venv-cmd`가 필요합니다.
새 환경 설치용 전체 ZIP이 아닙니다. 버전 표시는 여전히 0.1.0이므로 `--force-reinstall`이 필요합니다.
원본 데이터·기존 출력·venv를 삭제하지 않습니다. Windows에서 직접 실행 검증하지 않았습니다.

1. 이 `grid-diagnostics-f74a718` 폴더 전체를 기존 `.venv-cmd`가 있는 폴더에 복사합니다.
2. Git Bash를 그 기존 폴더에서 열고 아래 명령을 실행합니다. CMD에서도 실행 파일 경로는
   `".venv-cmd\Scripts\python.exe"`로 바꾸고 나머지는 같은 명령을 사용합니다.

```bash
./.venv-cmd/Scripts/python.exe -I -m pip --disable-pip-version-check install --no-index --only-binary=:all: --require-hashes --no-deps --force-reinstall --find-links="./grid-diagnostics-f74a718" -r "./grid-diagnostics-f74a718/requirements-update.txt"
./.venv-cmd/Scripts/python.exe -I -m pip --disable-pip-version-check check
./.venv-cmd/Scripts/python.exe -I -c "from psdl1._png_diagnostic_render import _join_seam; print('grid diagnostics update OK')"
```

인터넷 조회와 의존성 재설치를 하지 않으며 wheel 해시를 검사합니다.
`pip check`가 실패하면 그 오류를 해결한 뒤 입력 검사를 진행하세요.

3. 기존 검사 명령을 **새 출력 폴더**로 다시 실행합니다. 입력 위치와 grid 값은 실제 값으로 바꿉니다.

```bash
./.venv-cmd/Scripts/python.exe -I -m psdl1.import_png --root "D:/png_root" --out "D:/out/grid-updated-1" --inspect-grids 256 512 1024 --no-preview
./.venv-cmd/Scripts/python.exe -I -m psdl1.import_png --verify-bundle "D:/out/grid-updated-1"
```

새 출력의 `diagnostics/model_…/grid-숫자.png`를 봅니다. 기존 PNG는 자동 갱신되지 않습니다.
층별 색/반투명 합성, CNT/겹침 해치, 노랑 수치 예시, 빨강 original/grid/difference가 표시됩니다.
노랑은 축 방향 길이 참고이며 실제 최소 선폭·손상 증거가 아닙니다.
원래 Windows 전체 ZIP을 다시 설치하면 옛 wheel로 돌아갈 수 있으니 이 업데이트를 다시 적용하세요.
기존 전체 ZIP/wheelhouse의 파일을 직접 교체하지 마세요(기존 무결성 manifest와 해시가 달라집니다).
