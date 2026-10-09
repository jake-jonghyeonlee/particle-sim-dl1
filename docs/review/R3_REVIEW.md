# R3 독립 코드 검토 — DL1 (T8 + T9)

- 검토자: 새 세션의 Claude Opus 5.5 (effort high). 이 코드와 계획을 작성하지 않았다.
- 저장소 / 브랜치: `particle-sim-dl1` / `rewrite/shape-connectivity-v1`
- 후보: `7e52d67156f42b86c03fc934c8039c8d400e4c3e` (작업 트리 clean 확인)
- 검토 범위: `git diff 44a13a6 7e52d67` (T8 = `81d1f57`, T9 = `7e52d67`), 27개 파일
- 계약: `particle-sim-compare/docs/plan/PLAN_v2.md` §2.1, §3(G3·G5·G6), §4.2, §6 T8·T9, §8 R3, 그리고 `implementation_memory.json`의 2026-10-07 / 10-08 / 10-09 사용자 결정
- 검토일: 2026-10-09

## 판정: **FAIL** (blocking 1건)

blocking은 B1 하나이고 범위가 좁다. 나머지 점검 항목(§4.2 결함 수정, rest 채널, 경로 안전, 손실, sampler, split, checkpoint·resume, offline)은 아래 증거대로 확인됐다. B1을 고친 뒤에는 그 변경과 영향 시험만 다시 검토하면 된다.

이 판정은 합성 입력에 대한 구현 검토다. 실자료 성능, 실제 연결성 보존, 실제 grid 적합성은 검증 대상이 아니었고 검증되지 않았다.

## Blocking

### B1. 학습 CLI가 미확정인 grid를 256으로 추정해 쓰고, manifest의 `candidate_grid`를 조용히 덮어쓴다

- 위치: `src/psdl1/train.py:213` (`--grid` 기본값 256), `src/psdl1/train.py:235` (`load_manifest(a.manifest, grid=a.grid)`), `src/psdl1/train.py:112` (`train(..., grid=256, ...)`). manifest 쪽은 `src/psdl1/data.py:257-261`에서 `candidate_grid`를 필수로 요구한다.
- 재현: `candidate_grid: 4`인 manifest로 `python -m psdl1.train --manifest m.json --split split.json --out o`를 `--grid` 없이 실행했다(학습 함수는 대체해 인자만 확인). 결과는 `grid=256`, 제품의 `grid_check["grid"]`도 256이었다. manifest에 적은 값은 어디에도 쓰이지 않았다.
- 왜 blocking인가: 실제 최소 선폭 / grid 크기는 G5 안에서 아직 답이 없는 항목이고, "pending으로 보여야 하며 추정 기본값이면 안 된다"는 것이 이번 검토의 명시 조건이다. 출력 metadata에는 `real_grid_linewidth: pending`이 찍히지만 동시에 256이 기본으로 적용된다. 약 0.05 µm/px에서 unit이 1000 px 이상이면 256 grid는 4 px 이하 선을 1 px 미만으로 줄이는데, 이때 나오는 것은 Python 경고와 checkpoint 안의 기록뿐이다.
- 기대 동작: grid를 명시적으로 받거나(없으면 오류) manifest의 `candidate_grid`를 쓴다. 둘 다 있고 값이 다르면 오류를 내거나 어느 쪽을 썼는지 결과에 남긴다. `train()` API의 기본값도 같은 원칙을 따른다.

## Non-blocking (중요도 순)

### N1. 같은 lineage의 크기 변형이 `--unlabelled-negatives`에서 negative가 된다

- 위치: `src/psdl1/losses.py:26`, 문서 `docs/training.md:17`.
- 재현: `relation_matrix(["comb-40","comb-64","bus-40"], unlabelled_negatives=True)` → `comb-40`/`comb-64` 칸이 0(NEGATIVE).
- 영향: opt-in 실험이지만, 라벨이 없는 같은 lineage의 40/64/96 변형을 서로 밀어내 DL1 목적(크기가 달라도 같은 형상·연결 찾기)과 정반대로 학습한다. 문서는 "닮은 제품을 밀어낼 수 있다"는 일반 위험만 적었다.
- 기대 동작: 같은 `lineage_id` 쌍은 opt-in에서도 UNKNOWN으로 두거나, 최소한 문서와 coverage 기록에 이 경우를 명시한다.

### N2. 동점 정렬이 실제 모델에서는 입력 순서·batch에 따라 바뀐다

- 위치: `src/psdl1/search.py:91-94`, 문서 `docs/validation.md:61` ("Exact score ties sort by product ID, independent of input ordering"), 시험 `tests/test_search.py:21-34`.
- 재현: 원소 2개인 동일 제품을 ID `A`, `Z`로 두 번, 원소 3개인 제품 `M`을 후보로 넣고 `batch_size=2`로 검색. 후보 순서 `[Z, M, A]`에서는 A(0.99999994) > Z(0.99999988), 순서 `[A, M, Z]`에서는 Z > A. 같은 입력인데 원소 수가 더 많은 제품과 한 batch에 들어가면 padding 때문에 점수가 약 6e-8 달라진다.
- 시험이 놓치는 이유: 시험은 `embed`를 상수로 바꿔 정렬 키만 확인한다. 실제 encoder 경로는 지나지 않는다.
- 기대 동작: 점수를 일정 자릿수로 반올림한 뒤 ID로 정렬하거나, 문서의 주장을 "bit 단위로 같은 점수에 한함"으로 낮춘다.

### N3. endpoint만 바꾼 hard negative는 선언만 되고 평가 값이 없다

- 위치: `src/psdl1/evaluate.py:51-55` (검증), `:80` (목록 출력). 문서 `README.md:15`, `docs/validation.md:31,41`.
- 재현: 1 epoch 합성 checkpoint로 `evaluate(..., hard_negative_pairs=[("q","neg")])` 실행. 결과에서 hard negative와 관련된 키는 `hard_negative_pairs` 목록 하나뿐이다. 쌍의 점수, 양성 대비 순위, 통과 여부는 없다.
- 기대 동작: 쌍별 점수와 "같은 query의 known positive보다 아래에 있는가"를 항목으로 낸다. 기준값은 G6 pending으로 둔다. 지금 문서의 "held-out evaluation item"은 "선언·검증만 하고 지표는 아직 없다"로 고쳐야 정확하다.

### N4. 로컬 가중치 경로는 공개 ConvNeXt V2 가중치를 읽지 못한다

- 위치: `src/psdl1/model.py:44,58,85`. `README.md:48`에 4채널 조건은 적혀 있다.
- 재현: timm `convnextv2_atto` 기본 구성(3채널, 분류 head)의 state dict를 저장해 `ProductEncoder(..., "convnextv2_atto", weights_file)`에 주면 `Unexpected key(s) head.fc.*`와 `stem.0.weight` 크기 불일치로 `RuntimeError`.
- 영향: 실패는 명시적이고 온라인 fallback도 없어 offline 계약은 지킨다. 다만 "가중치를 파일로 반입해 로드"라는 사용자 전제는 같은 4채널·head 없는 구조에서 저장한 파일에만 성립한다. 변환 절차가 없다.
- 기대 동작: 문서에 이 한계를 한 줄 더 분명히 적거나, stem 채널 확장과 head 제거를 하는 명시적 변환을 제공한다.

### N5. 직사각형 unit에서 endpoint 좌표 특징이 이미지 위치와 어긋난다

- 위치: `src/psdl1/train.py:89-90` (축별로 W, H로 나눔), 이미지 쪽 `src/psdl1/data.py:32-36` (등방 축소 후 가운데 pad), 문서 `docs/data_contract.md:58` (pad를 더하는 변환을 설명).
- 재현: 4×8 px unit, endpoint가 오른쪽 아래 칸. grid 8에서 연결 mask 중심은 (x, y) = (0.9375, 0.6875), endpoint 특징은 (0.9375, 0.875).
- 기대 동작: 문서의 변환과 같은 좌표계를 쓰거나, 특징이 축별 정규화라는 점을 문서에 맞춰 적는다. 정사각형 unit에는 영향이 없다.

### N6. 승인 상태 표기가 서로 다르고, 오래된 문장이 남아 있다

- 과소 표기: `src/psdl1/data.py:121,231,312`, `src/psdl1/schema.py:1`, `src/psdl1/data.py:1`, `pyproject.toml:4`, `docs/data_contract.md:3`이 whole-unit resize + pad 전체를 `G5_pending`으로 적는다. 사용자는 2026-10-09에 이 방식을 승인했다. 같은 checkpoint 안에서 `approval.input_method`는 `user_approved_2026_10_09`, `grid_checks[].approval`은 `G5_pending`으로 나온다(실행으로 확인). `tests/test_input_contract.py:212`가 이 값을 고정한다.
- T9 이후 틀린 문장: `docs/data_contract.md:60,76,82` ("model이 아직 valid/endpoint를 쓰지 않는다", "model.py 등 변경 없음"), `docs/data_contract.md:86`과 `docs/validation.md:74` ("Changes are left uncommitted").
- 기대 동작: grid 보고의 승인 필드를 train/search와 같은 4항목 구조로 맞추고, T8 기록 절은 "당시 기록"임을 표시하거나 갱신한다. 과장 방향의 표기는 찾지 못했다.

### N7. grid 경고는 학습을 막지 않는다

- 위치: `src/psdl1/data.py:157-158` (경고만), `src/psdl1/train.py:152` (기록만).
- 영향: bridge 손실이나 2 px 미만 선이 감지돼도 학습은 진행된다. 문서의 시험 명령은 `-p no:warnings`라 경고가 보이지 않는다. T8 Change의 "경고·gate" 중 gate에 해당하는 것이 없다.
- 기대 동작: 실제 기준이 정해지기 전이라도 경고가 있으면 멈추는 명시적 선택지(예: strict 옵션)를 두고, 기본 동작을 결과 JSON에 남긴다.

### N8. 틀린 구현에서도 통과할 수 있는 시험

- `tests/test_search.py:96-117`: 후보 2개에 `topk=2`이고 대체한 검색이 topk를 무시해 전부 돌려준다. Recall은 순위와 무관하게 1이다. 내가 실제 검색으로 돌린 경우도 양성 1개가 1위라 topk 1/3/50 모두 1.0이었고, recall이 1보다 작은 경우는 어디서도 확인되지 않았다.
- `tests/test_relations.py:9-22`: 손실의 절대값 기대치가 없다(전후 동일성만 확인). 손으로 계산한 값과 대조한 결과는 아래 "실행한 것"에 있다. 구현은 맞다.
- `train(resume=...)` 전체 경로 시험이 없다. 직접 fixture만 있다. 아래 실험으로 동등성을 확인했다.
- `tests/conftest.py:22-23`: 모든 family의 CNT가 여전히 같은 위치다(§4.2 지적). endpoint 반례는 `manifest_case` 기반 시험으로 따로 추가돼 있다.

### N9. 작은 견고성 문제

- `src/psdl1/checkpoint.py:112`: `config`가 없는 checkpoint는 `ValueError`가 아니라 `KeyError: 'config'`.
- `src/psdl1/losses.py:42`: 대각선이 POSITIVE인지 강제하지 않는다. 대각선이 0인 행렬도 받아들인다(확인). 학습 경로는 `relation_matrix`만 쓰므로 실제로는 닿지 않는다.
- `src/psdl1/data.py:131-136,294`: `unit_size_um`이 `[2, 2]`와 `[2.0, 2.0]`이면 `input_identity`가 달라진다(확인). manifest를 다시 직렬화하면 resume / evaluate가 입력 변경으로 거부할 수 있다.
- `src/psdl1/train.py:174-175`: 매 step 두 번 `prepare_product`를 다시 계산한다(검사·hash 포함, cache 없음). 2000 px, 원소 3개 제품에서 호출당 약 0.37 s. `load_manifest`는 모든 mask를 float32로 한꺼번에 올린다. 정확성 문제는 아니고 실자료 규모에서의 비용이다.
- `src/psdl1/model.py:47-48`: `.safetensors` 경로가 선언되지 않은 `safetensors`에 의존한다(timm과 함께만 설치됨).
- `src/psdl1/data.py:230`: `augment_with_check`는 채널 0만 본다. CNT 채널이 잘려 나가도 보고되지 않는다. 학습에서는 기하 증강이 거부되므로 현재는 닿지 않는다.

## 확인된 항목

| 점검 대상 | 결과 | 근거 |
| --- | --- | --- |
| rest 채널 겹침 | 확인 | `data.py:165-166`. 두 원소가 겹치는 칸에서 self `[1,1]`, rest `[1,1]` (직접 실행). 손으로 쓴 pixel oracle `tests/test_data.py:91-100` |
| 원소 공통 변환 | 확인 | 전체 frame 기준 한 번의 축소·pad, 원소별 중심 정렬 없음 (`data.py:27-37`, `tests/test_input_contract.py:238-246`) |
| manifest 경로 안전 | 확인 | 절대경로, `..`, 밖을 가리키는 파일·디렉터리 symlink, 빈 값, `null`, 숫자, `~`, `file://`, 비 `.npy`, pickle 객체, 이름만 바꾼 npz 모두 거부. root 안을 가리키는 symlink와 `./x.npy`는 허용. hardlink는 구별 불가(파일시스템 한계) |
| hotspot / CA / fab 차단 | 확인 | 5개 위치 × 7개 키 변형(`Hotspot_mask`, `ca_um2`, `critical_area`, `fab_density` 등) 모두 거부. 실제 방어선은 금지어 목록이 아니라 모든 단계의 필드 allowlist(`schema.py:65-70`). 값 수준(역할 이름이나 파일 이름이 `hotspot`)은 막을 수 없다 |
| role-layer binding | 확인 | binding을 바꾸면 `input_identity`와 embedding이 달라지고, 원소 순서 변경은 4.5e-8 이내, endpoint 순서 변경은 차이 0 |
| bridge / 좁은 선 검사 | 확인(진단 한정) | 빈 mask는 0, 1 px bridge 손실 보고, `connectivity_preserving`은 항상 false. 원소 사이 간격 변화는 검사 대상이 아니다 |
| 손실 | 확인 | UNKNOWN은 분자·분모·gradient에서 제외, 충돌 라벨 거부, negative 없는 행 제외·집계, 온도·형태·유한성 검증 |
| sparse negative sampler | 확인 | 모든 negative edge가 epoch마다 배치됨. 실행 기록에서 `covered_negative_pairs 12 / required 12` |
| lineage split | 확인 | 같은 lineage가 partition을 넘으면 거부. vocabulary는 train만 사용 |
| checkpoint / resume | 확인 | 버전·fingerprint·가중치 hash 거부, 원자적 교체. lr / batch / relations / seed / 입력 mask / config 변경 시 resume 거부 |
| 검색 | 확인(N2 제외) | 독립 query·후보 manifest, batch 단위 encode, topk 검증, 사유 필드 |
| offline | 확인 | `src/`에 네트워크 호출 없음. timm은 `pretrained=False`만 사용, 없는 파일·비유한 가중치는 fallback 없이 실패 |
| 증강 | 확인 | crop·기하 증강은 학습 전에 거부(`train.py:68-69,119-120`). `intensity`는 기하를 건드리지 않는다. 연결이 바뀐 view를 positive로 학습할 경로는 없다 |
| §4.2 결함 목록 | 수정됨 | `data.py` 재작성, `losses.py` 충돌 거부·검증, `train.py` 재작성, `search.py` fingerprint·batch·사유, `model.py` valid/endpoint 입력·검증, BatchNorm 주장 삭제. 남은 차이는 N6·N8에 적었다 |
| pending 표기 | B1 제외 확인 | endpoint 원천 형식과 G6(라벨·split id·수락 기준)는 모든 출력과 문서에서 pending이다. `performance_acceptance`는 `not_established` |

## 실행한 것

저장소 안에서는 이 보고서 외에 아무것도 쓰지 않았다. 실험은 저장소 밖 임시 폴더에서 했고 네트워크는 쓰지 않았다(socket 차단 상태).

| 명령 / 실험 | 결과 |
| --- | --- |
| `.venv/bin/python -m pytest -p no:warnings` | 163 passed, 1 skipped |
| `.venv/bin/python -m pytest -p no:warnings --run-training` | 164 passed |
| 손실 손계산 대조 | 계약 반례: 0.3133 = ln(1 + e⁻¹) = 0.31326. 양성 2개 행 포함: 0.610434 = 손계산 0.610434 |
| 경로·금지 필드·binding·동점 실험 | 위 표와 N2 |
| resume 동등성 (합성 6개 제품, grid 16) | 3 epoch 연속 실행과 1 epoch 후 전역 RNG를 흩뜨리고 resume한 실행의 history와 가중치 hash가 동일 |
| 평가 실행 (합성 6개 제품, 1 epoch) | N3, N8 |
| CLI grid 기본값 확인 (학습 함수 대체) | B1 |
| 공개 형태 timm 가중치 로드 | N4 |
| 2000 px 준비 시간 측정 | N9 |

합성 학습을 직접 돌린 것은 resume 동등성(3 + 1 + 2 epoch)과 평가용 1 epoch뿐이며, 둘 다 grid 16의 합성 fixture다. 이 실행들은 구현 동작만 보여 주며 성능 근거가 아니다.

## 실행하지 않은 것

- 실자료, 실제 라벨, 실제 학습, CUDA 경로, 반입 pretrained 가중치.
- G6 수락 기준에 대한 어떤 판정. 기준 자체가 아직 없다.

---

## Re-review of dba0fb5

- 검토자: 또 다른 새 세션의 Claude Opus 5.5 (effort high). 코드, 계획, 위의 1차 검토 어느 것도 작성하지 않았다.
- 후보: `dba0fb5370bf8244b2eb204c0a54da926a89f925` (작업 트리 clean 확인)
- 검토 범위: `git diff 7e52d67 dba0fb5` 중 이 보고서를 추가한 `63a3636`을 뺀 부분 (소스 9개, 시험 8개, 문서 4개, `pyproject.toml`, `configs/train.example.json`)
- 계약: `PLAN_v2.md` T8·T9, `implementation_memory.json`의 `gate_G5_answer_2026_10_09` (whole-unit resize + pad 승인. 실제 grid / 최소 선폭, CNT / VIA endpoint 원천 형식, G6 전체는 미결)
- 검토일: 2026-10-09

### 판정: **PASS-WITH-NOTES** (DL1 범위 T8–T9, 후보 `dba0fb5`)

B1은 해결됐다. 추정 grid를 적용하는 경로는 학습 CLI, Python API, resume 어디에도 남아 있지 않다. N1, N2, N3, N5, N6, N7, N8, N9는 고쳐졌고 N4는 문서로만 다뤘다. 수정으로 생긴 blocking 결함은 찾지 못했다. 아래 R1–R6은 non-blocking이며, 그중 R1과 R2는 실자료를 넣기 전에 정리하는 편이 좋다.

이 판정은 합성 입력에 대한 구현 검토다. 실자료 성능, 실제 연결성 보존, 실제 grid 적합성은 검증되지 않았고, 이 PASS는 `dba0fb5`에만 유효하다.

### B1: 해결됨

| 확인 항목 | 실행 | 결과 |
| --- | --- | --- |
| `--grid` 없이 실행하면 manifest 값을 쓴다 | `candidate_grid: 4` manifest로 `psdl1.train` CLI 실행 | checkpoint `config.grid` 4, `grid_selection.source` `manifest_candidate`, 준비 단계 grid도 4 |
| 다른 `--grid`는 거부된다 | `--grid 2` | `ValueError: grid 2 conflicts with manifest candidate_grid 4`. 출력 폴더와 checkpoint가 생기지 않는다 |
| 설정 파일로 들어온 값도 같다 | `--config`에 `"grid": 256` | 같은 오류로 거부. `configs/train.example.json`에서 `grid` 키가 빠졌다 |
| 명시 override는 기록된다 | `--grid 2 --allow-grid-override` | `config.grid` 2, `grid_selection` = `{manifest_candidate_grid: 4, selected_grid: 2, source: explicit_override, override: true, override_requested: true}` |
| override 플래그 형식 | 설정 파일의 `"allow_grid_override": "false"`, API의 `allow_grid_override=1` | 둘 다 `must be boolean`으로 거부 |
| Python API | manifest 없이 만든 제품으로 `train()` | grid 없으면 `No grid supplied`, `grid=4`를 주면 `source: api_explicit`로 기록 |
| Python API + manifest | manifest 제품에 `grid=8`, `load_manifest(grid=2)` 뒤 `train(grid=2)` | 둘 다 override 플래그 없이는 거부 |
| resume | `--grid` 없이 1 → 2 epoch | 진행되고 `grid_selection`이 그대로 유지된다 |
| resume에서 grid 변경 | `--grid 2`, `--grid 2 --allow-grid-override`, manifest를 4 → 2로 바꾼 뒤 재개, 그 상태에서 `--grid 4 --allow-grid-override` | 차례로 conflict, `resume config differs`, `resume config differs`, `contract differs`로 모두 거부. 거부 뒤 기존 checkpoint는 바뀌지 않았다 |
| resume에서 플래그만 변경 | `--allow-grid-override` 또는 `--allow-grid-check-failure`만 추가 | `contract differs`로 거부. 두 플래그가 resume 계약 hash에 들어 있다 |
| search / evaluate | grid 4 checkpoint에 `candidate_grid: 8` manifest | 추정값은 쓰지 않고 checkpoint의 grid 4를 쓴다. 두 CLI에 `--grid` 옵션은 없다. 기록 방식의 빈틈은 R1 |

1차 검토의 `7e52d67` checkpoint는 `MODEL_INPUT_VERSION`이 `dl1_structured_v2_padded_endpoints`로 바뀌어 읽는 단계에서 거부된다. 이 항목은 `checkpoint.py:110`을 읽어 확인했고 실행하지는 않았다.

### 1차 지적 사항 상태

| 항목 | 상태 | 확인한 내용 |
| --- | --- | --- |
| N1 | 수정됨 | `--unlabelled-negatives`에서 같은 lineage 쌍은 UNKNOWN, 다른 lineage만 NEGATIVE다. `comb-40/64/96 + bus-40`에서 sampler의 negative edge는 bus와의 3쌍뿐이고, 학습 기록은 `negative_pairs 3 / unknown_pairs 3`이다. `similar`로 라벨한 같은 lineage 쌍은 POSITIVE로 남는다. opt-in에는 `lineage_ids`가 필수다 |
| N2 | 수정됨 | 실제 encoder로 후보 9개를 40번 섞고 batch 크기 1·2·3·4·5·7·32를 바꿔 가며 검색했다. 순위와 점수 조합은 1가지뿐이다. `embed` 결과는 batch 크기와 thread 수(1 / 4)에 대해 bit 단위로 같다. 동점인 `A`/`Z`, `M`/`X2`는 ID 순서다 |
| N3 | 수정됨 | 평가 CLI를 실제 검색 경로로 돌렸다. `hard_negative_evaluation`에 방향별 점수, 순위, known positive의 점수·순위, `below_all_known_positives`, `margin_to_lowest_positive`가 나온다. positive가 없는 방향은 `comparison_status: unavailable`과 사유가 나오고 기준값은 `G6_pending`이다. topk 1에서도 순위 2인 hard negative가 채점된다 |
| N4 | 문서만 | loader는 그대로다. `README.md:50`, `docs/training.md:64`가 공개 ConvNeXt V2 가중치는 검증되지 않았고 변환이 없다고 적는다. 3채널 분류 state를 조용히 받아들이지 않는다는 시험이 추가됐다 |
| N5 | 수정됨 | 아래 "직사각형 unit의 endpoint 좌표" |
| N6 | 수정됨 (R4 잔여) | 모든 보고가 `approval_status()` 한 곳의 4항목을 쓴다. 학습 checkpoint의 `approval`과 `grid_checks[].approval`, 검색, 평가, validator 보고, 증강 보고에서 같은 값을 확인했다. `G5_pending` 문자열은 소스와 현재 문서에 남아 있지 않다. T8 기록 절은 "Historical"로 표시됐다 |
| N7 | 수정됨 (R2 범위 한계) | 아래 "grid gate" |
| N8 | 수정됨 | 평가 시험이 recall 0과 1을 가진 query로 평균 0.5를 확인한다. 손실 절대값 0.31326168751822286 대조, `train(resume=...)` 전체 경로 동등성 시험, family별로 다른 CNT 위치가 추가됐다 |
| N9 | 수정됨 (1건은 문서화) | `config` 없는 checkpoint는 `ValueError`, 대각선 0인 관계 행렬 거부, `[2, 2]`와 `[2.0, 2.0]`의 `input_identity` 동일, 제품당 준비 1회, `safetensors` 선택 의존성 선언, 증강 검사의 전 채널 확인. mask를 한꺼번에 올리는 점은 고치지 않고 `docs/validation.md`에 적었다 |

### 수정으로 생긴 문제 점검

**grid gate와 우회 플래그.** 8×8 unit의 1 px 선을 grid 4로 학습하면 `grid check failed before training`으로 멈추고 출력 폴더가 생기지 않는다. `--allow-grid-check-failure`를 주면 진행되고 checkpoint에 `grid_gate = {policy: strict_default, bypass_requested: true, status: bypassed, failures: [...]}`가 남는다. 문자열 `"yes"`는 거부된다. 우회 플래그는 grid override 플래그와 분리돼 있고 둘 다 resume 계약에 들어 있다. gate는 모델 생성과 seed 설정 전에 평가된다. 범위 한계는 R2.

**unlabelled negative에서의 같은 lineage 규칙.** 위 N1대로 동작한다. opt-in 상태에서 같은 lineage 쌍을 `different`로 라벨하면 `same-lineage negative forbidden`으로 거부되고, opt-in이 아닐 때는 그 라벨이 NEGATIVE로 받아들여진다. 이 차이는 `docs/training.md:17`에 적혀 있어 결함으로 보지 않는다. 전부 같은 lineage인 학습 집합은 negative edge가 없어 학습 전에 거부된다.

**순서·batch와 무관한 검색 점수.** 위 N2대로 성립한다. 방법은 제품 하나씩 forward한 뒤 float64 cosine을 소수 6자리로 반올림하고 ID로 정렬하는 것이며, 결과의 `score_policy`에 적힌다. 비용은 R5.

**직사각형 unit의 endpoint 좌표.** 11가지 경우(4×8, 8×4, 8×8, 5×8, 3×8, 8×16, 16×8, 6×10, 10×6, grid 7·8·16)에서 endpoint 특징과 준비된 연결 mask의 무게중심을 비교했다. 축소가 없는 7가지는 차이 0이다. 축소가 있는 4가지는 차이가 0.024–0.031로 grid 한 칸(0.125–0.143)의 1/4 이하다. 특징은 칸 안의 실제 위치이고 mask 쪽은 칸 중심이라 생기는 차이이며 같은 칸을 가리킨다. 반올림한 축소 크기와 pad 몫은 `normalize_unit`과 같은 식을 쓴다.

**약해진 시험.** 찾지 못했다. 기존 시험에서 바뀐 것은 다음뿐이다.

- `relation_matrix(..., unlabelled_negatives=True)` 호출 3곳에 `lineage_ids` 추가. 기대 행렬은 그대로다.
- `train()` 호출 4곳에 `grid=32` 추가(이제 필수). 인자 오류 시험들은 grid gate보다 앞선 검증에서 걸리므로 의도한 오류를 여전히 확인한다.
- 평가 시험은 `topk` 2 → 1, recall 기대값 1 → 0.5 / 0으로 더 엄격해졌다.
- validator CLI 시험의 `approval` 기대값이 `G5_pending`에서 4항목 객체로 바뀌었다(N6의 의도).
- opt-in 합성 학습 시험에 `allow_grid_check_failure=True` 추가. 단정문은 그대로다. 영향은 R6.

### 새 non-blocking 사항

**R1. 검색 결과에 grid 숫자가 없고, manifest의 `candidate_grid`와 checkpoint grid가 달라도 알리지 않는다.**
- 위치: `src/psdl1/search.py:122,125`, `src/psdl1/evaluate.py:127`.
- 재현: grid 4 checkpoint와 `candidate_grid: 8` manifest로 search CLI 실행. 결과는 `status: available`이고, 출력 키는 `approval, candidate_count, eligible_count, excluded, fingerprint, query_connectivity, query_id, rankings, reason, score_policy, status`다. grid는 `fingerprint` hash 안에만 있다. 평가 출력에는 `model_config.grid`가 있지만 manifest 값과 다르다는 표시는 없다.
- blocking이 아닌 이유: 적용된 grid는 추정값이 아니라 학습 때 기록된 값이고, 모델은 그 grid로만 쓸 수 있다.
- 기대 동작: 검색·평가 출력에 적용 grid와 manifest의 `candidate_grid`를 함께 적고, 다르면 표시하거나 거부한다.

**R2. grid gate는 학습 partition만 본다.**
- 위치: `src/psdl1/train.py:174-179`.
- 재현: 1 px 선 제품을 test partition에만 두고 grid 4로 학습하면 `grid_gate.status: passed`, `failures: []`다. 같은 제품을 query로 검색하면 `status: available`이고 결과에 grid 검사 정보가 없다(제품의 `grid_check.status`는 `warning`).
- 영향: 선이 사라진 held-out 제품이 평가 지표와 검색 순위에 조용히 들어간다. 문서의 시험 명령은 `-p no:warnings`라 경고도 보이지 않는다.
- 기대 동작: 검색·평가 결과에 query와 후보의 grid 검사 상태를 싣거나, 같은 strict 선택지를 둔다.

**R3. 손으로 만든 `grid_check`가 manifest 출처로 기록된다.**
- 위치: `src/psdl1/train.py:128`.
- 재현: manifest 없이 `replace(p, grid_check=grid_check(p, 2))`로 만든 제품을 `train()`에 grid 없이 주면 `source: manifest_candidate`, `manifest_candidate_grid: 2`로 기록된다. manifest는 없었다.
- 영향: Python API에서만 생긴다. CLI는 항상 `load_manifest`를 거치므로 닿지 않는다.

**R4. 문서 잔여.**
- `docs/validation.md:92`: "Changes below remain uncommitted". `dba0fb5`로 commit된 상태라 맞지 않는다(N6과 같은 종류).
- `docs/data_contract.md:69`: validator 예시 명령이 `--grid 256`이다. 진단 도구의 예시이고 보고에 값이 기록되지만, 미결인 grid의 권장값처럼 읽힐 수 있다.

**R5. 검색이 제품 하나씩 forward한다.** `src/psdl1/search.py:56-57`. `batch_size`는 이제 묶음 크기만 정하고 forward 수는 후보 수와 같다. 학습 때의 batch forward와는 embedding이 최대 1.5e-7 다르다(측정). 정확성 문제는 아니고 실자료 규모의 처리량 비용이다. `docs/validation.md:61`에 적혀 있다.

**R6. 실제 CNN을 학습하는 유일한 시험이 gate를 우회한다.** `tests/test_model_and_training.py:78`. 합성 fixture의 CNT가 40 px unit에서 2 px이라 grid 32에서 1.6 px이 되어 strict gate에 걸린다. gate 통과 경로는 2-parameter 대체 모델을 쓰는 시험(`test_b1_override_recorded_in_checkpoint` 등)과 위 B1 실험으로만 확인됐다.

### 실행한 것

저장소 안에서는 이 보고서 외에 아무것도 쓰지 않았다(`git status`는 이 파일만 수정됨). 실험은 저장소 밖 임시 폴더에서 했고 socket 연결을 막은 상태였다.

| 명령 / 실험 | 결과 |
| --- | --- |
| `.venv/bin/python -m pytest -p no:warnings` | 184 passed, 1 skipped |
| `.venv/bin/python -m pytest -p no:warnings --run-training` | 185 passed |
| 학습 CLI·API·resume의 grid 선택 26가지 경우 | 위 B1 표, R3 |
| search / evaluate CLI (grid 4 checkpoint, `candidate_grid: 8` manifest) | R1, N3 |
| grid gate (1 px 선, 우회, 플래그 형식, test partition 전용 제품, 검색) | "grid gate", R2 |
| lineage 규칙 (행렬, sampler edge, 학습 기록) | N1 |
| 실제 encoder 검색 40회 (순서·batch 크기 변경), thread 1 / 4 | N2, R5 |
| 직사각형 endpoint 좌표 11가지 | N5 |
| checkpoint `config` 제거, 대각선 0 행렬 | N9 |

학습 경로 실험은 저장소 시험과 같은 방식으로 encoder를 2-parameter 대체 모델로 바꿔 제어 흐름과 checkpoint 기록만 확인했다. 검색·평가 실험의 checkpoint는 학습하지 않은 실제 encoder를 저장한 것이다. 실제 CNN 학습은 `--run-training`의 합성 시험 한 번뿐이다.

### 실행하지 않은 것

- 실자료, 실제 라벨, 실제 학습, CUDA 경로, 반입 pretrained 가중치, `.safetensors` 로드.
- `7e52d67` checkpoint를 실제로 만들어 거부되는지 확인하는 실험(코드로만 확인).
- G6 수락 기준에 대한 판정. 기준이 아직 없다.
