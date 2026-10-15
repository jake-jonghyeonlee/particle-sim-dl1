# psdl1 — DL1 product similarity

> 한국어 사용 안내: 전체 절차와 입력·출력 형식은 **[docs/USAGE.md](docs/USAGE.md)** 에 정리되어 있습니다.

DL1 aims to find products with similar original shapes and electrode connectivity even when unit sizes differ. The user approved **whole-unit isotropic resize + pad and electrode × actual-layer set aggregation on 2026-10-09**, prioritizing shape/connectivity matching over preserving absolute linewidth. Unit-size conditioning is optional and **off by default** so the user can compare it on and off in the closed network. No second original-µm view is implemented.

The real minimum linewidth/grid choice and the source CNT/VIA endpoint metadata format remain open within G5. G6 label-format and closed-network acceptance criteria remain open. The exact encoder, endpoint feature representation, and training recipe are candidates; no real search performance or connectivity preservation has been established. G3 repository naming remains pending.

Inputs contain only original shape, valid coverage, and supplied connectivity metadata; hotspot, CA, and fab fields are rejected. PNG products use **0.1 µm/pixel**, have different pixel counts, and share a full frame across their layers. Historical NPY examples retain their explicit pitch. For PNG export, `unit_size_um` means `[W*0.1,H*0.1]` µm of full frame extent, including the four real margins; no single unit or simulation-domain size is inferred. See [docs/data_contract.md](docs/data_contract.md) for the accepted T8 draft schema. All reports distinguish the approved input method from pending real grid/linewidth, source endpoint format, and G6.

The T9 model consumes T8 spatial valid coverage as a fourth image channel and source relations as features bound to the corresponding element representations. Connection features retain each endpoint's normalized padded-image coordinate, CNT/VIA kind, observation flag, and mask coverage. Connectivity availability is separate from image values. A global CNT image without source endpoint metadata remains **unavailable**, never CNN-verified. No layer is independently centred. The fixed-grid image is a candidate representation and cannot certify connectivity.

Labels are explicit and unknown pairs are excluded from both loss numerator and denominator. Conflicting similar/different labels are rejected. Unlabelled negatives require `--unlabelled-negatives`; this experiment can push truly similar products of different lineages apart when labels are incomplete. Same-lineage variants remain UNKNOWN or explicitly positive; they are never experimental negatives. The relation sampler covers every supplied negative edge each epoch and reports effective pairs, excluded rows, and unsampled products. Training vocabulary/statistics use only the validated training partition; unknown query/candidate role or layer is rejected with a reason.

Training uses an explicit lineage-safe split and the manifest’s `candidate_grid`, with no guessed grid default. A conflicting `--grid` requires `--allow-grid-override`. Thin-line or bridge failures block training unless `--allow-grid-check-failure` is passed. Both flags and their evidence are checkpointed. Cropping and geometric augmentation are disabled; default augmentation is `none`, with an optional `intensity` ablation that applies one scalar to all product image channels and keeps connection geometry. Checkpoints store version/fingerprint, vocabulary/statistics, split, code/input/weights hashes, seed, optimizer state, RNG states, grid evidence, and relation coverage, with atomic replacement after each epoch. Resume rejects changed contracts and restores state.

See [docs/training.md](docs/training.md) for the complete configuration and resume protocol, and [docs/validation.md](docs/validation.md) for held-out evaluation and limits. Endpoint-only hard negatives are scored as separate held-out items, with full-corpus rank, known-positive comparisons and margins even outside the requested top-k; an absent list is reported unavailable with `none_supplied`. Its ranking by an untrained encoder is not performance evidence; the evaluator rejects such validation.

```bash
# Input validation only; original masks must satisfy the versioned draft contract.
.venv/bin/python -m psdl1.data data/manifest.json --report grid-report.json

# User-run closed-network training; configure grid, labels and split first.
.venv/bin/python -m psdl1.train --config configs/train.example.json

# Independent query and candidate manifests; JSON output with availability/reasons.
.venv/bin/python -m psdl1.search --ckpt runs/dl1/dl1.pt \
  --query-manifest data/query.json --query new-product \
  --candidates-manifest data/candidates.json --topk 5 --batch-size 32 --out search.json

# Held-out model vs normalized-image baseline; no real acceptance threshold yet.
.venv/bin/python -m psdl1.evaluate --ckpt runs/dl1/dl1.pt \
  --manifest data/manifest.json --split data/split.json --relations data/relations.csv \
  --hard-negatives data/hard-negatives.json --topk 5 --out evaluation.json
```

`configs/` contains illustrative filenames/IDs and unapproved example hyperparameters, except for the explicitly user-supplied `stack-order.json` reference of 2026-10-13. Both manifests, labels, and split must be supplied with actual provenance before running user training. A 2×2 simulation domain is not automatically converted to a unit.

The checked runtime is Python 3.12. Windows users must follow the
[CMD offline installation guide](docs/WINDOWS_OFFLINE.md): unpack the prepared ZIP
into a writable user folder, create `.venv-cmd` with preinstalled CPython, then
install the bundled wheels with hash checks and `pip --no-index`. There is no
PowerShell or online installer. The macOS `requirements.lock` snapshot is retained
for provenance and is not a Windows installation lock. Building/downloading the
distribution happens only on the connected preparation machine.

Developer regression checks use the existing prepared development environment:

```bash
.venv/bin/python -m pytest -p no:warnings
.venv/bin/python -m pytest -p no:warnings --run-training
```

Default tests include bounded synthetic forward/loss/optimizer and one-step importer/training-driver integration fixtures; the existing 40-epoch dataset smoke stays opt-in. Socket network calls fail in tests. `pretrained=False` is always used for backbone construction; optional weights load from a local `.pt`, `.pth`, or `.safetensors` state dict with strict shape/finite checks and no online fallback. For the product encoder, local backbone files must match its four input channels. Record an imported artifact's exact hash, source and license before use; none was accepted here.

The local checkpoint path (`--ckpt` / `--resume`) loads this project's own versioned checkpoints. `--weights` accepts compatible backbone-only project/timm-format files, with four input channels and no classifier head for the product encoder. These paths have **not been verified against public ConvNeXt V2 weight files**; no conversion is performed. Pretrained remains off by default. Optional safetensors support is declared in the `local-weights` dependency group and must be obtained through the approved offline wheel process.

Search processes bounded chunks with canonical single-product forwards so padding neighbours and requested chunk size cannot change scores. Cosines use float64 accumulation and are rounded to six decimals before product-ID tie sorting; this favours deterministic results over multi-product forward throughput.

Search and evaluation report the numeric checkpoint `grid` and `grid_selection` with manifest candidate grids. Any mismatch requires `--allow-grid-override`; inference keeps the checkpoint grid and records the exception. Their default grid gate blocks thin-line/vanishing-bridge queries and excludes failing candidates, including held-out evaluation products and the baseline. Only `--allow-grid-check-failure` bypasses it, with per-product evidence in JSON. Both flags are separate, default off, and do not approve a real grid or minimum linewidth.

Canonical single-product forwards preserve order/chunk-size reproducibility. `batch_size` controls candidate chunks rather than forward batch size; evaluation caches each product embedding once per command to avoid repeated forwards across queries, trading O(N × embedding dimension) memory for fewer forwards. Multi-product forward acceleration remains deferred because it changes arithmetic. The opt-in 40-epoch synthetic CNN smoke now uses an explicitly chosen synthetic grid 64, passes the strict grid gate with no bypass, and retains all loss/ranking/separation assertions; this fixture choice is not a real-data grid recommendation.

## Windows 폐쇄망 설치와 실행 (CMD)

Windows CPython 3.12 x64 CPU용 준비된 wheel 묶음과 CMD 설치는
[Windows 폐쇄망 가이드](docs/WINDOWS_OFFLINE.md)를 먼저 따르십시오.
`bundles/windows_cp312_win_amd64.parts`는 Git 전달용 분할 ZIP이며,
기존 macOS `requirements.lock`는 Windows용 lock가 아닙니다.

ZIP을 사용자 폴더에 풀고 CMD에서 `install_windows_bundle.cmd`를 실행합니다.
검사·학습·검색·평가도 같은 설치 Python을 사용합니다(실제 경로/ID로 변경).

```bat
set "DL1PY=%CD%\.venv-cmd\Scripts\python.exe"
"%DL1PY%" -I -m psdl1.import_png --root "D:\PNG 입력" --out "D:\결과\새 검사" --no-preview
"%DL1PY%" -I -m psdl1.data "D:\입력\manifest.json" --report "D:\결과\validation.json"
"%DL1PY%" -I -m psdl1.train --config "D:\설정\train.json"
"%DL1PY%" -I -m psdl1.search --ckpt "D:\학습\dl1.pt" --query-manifest "D:\입력\query.json" --query "실제-제품-ID" --candidates-manifest "D:\입력\candidates.json" --out "D:\결과\search.json"
"%DL1PY%" -I -m psdl1.evaluate --ckpt "D:\학습\dl1.pt" --manifest "D:\입력\manifest.json" --split "D:\설정\split.json" --relations "D:\설정\relations.csv" --out "D:\결과\evaluation.json"
```

소스 checkout이나 활성화는 필요 없습니다. 아래 `.venv/bin/python` 예시는 POSIX 개발 환경용입니다.

## PNG workflow, offline

Use `psdl1[png]` with approved local **Pillow 12.3.0** wheels (and hatchling/build
requirements when building the package). The optional PNG dependency is imported
lazily: existing NPY/checkpoint inference needs no Pillow. `requirements.lock` is
unchanged and does not guarantee a portable wheelhouse; no online fallback exists.

```bash
# Paths and candidate grids must be supplied by the user; destinations must be new.
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$INSPECT_OUT"
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$GRID_INSPECT_OUT" \
  --inspect-grids "$GRID_A" "$GRID_B" --no-preview 2> "$PROGRESS_JSONL"
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$EXPORT_OUT" \
  --export --metadata "$METADATA_JSON" --candidate-grid "$CANDIDATE_GRID"
.venv/bin/python -m psdl1.import_png --verify-bundle "$EXPORT_OUT"
.venv/bin/python -m psdl1.data "$EXPORT_OUT/manifest.json" --report "$VALIDATOR_REPORT"
```

CLI progress lines are JSON objects with `event == "progress"`, flushed on stderr.
Other stderr lines (errors, warnings and argparse messages) are plain text; filter them
before parsing, for example `grep '"event": "progress"' stderr.log > progress.jsonl`.
Stdout remains exactly the
`resource_inventory` and `completed` result events. `--no-preview` skips HTML/preview
rasters only, recording `preview_skipped_by_request: --no-preview`; all geometry,
extraction, grids and export assets remain. After interruption, retain failure staging
and retry with a fresh `--out`. No 18,000-PNG production success is claimed.

Inspection writes a versioned report and offline full-frame preview without metadata
or a grid. Explicit grid inspection compares candidates without approving/selecting
one. Export requires exact per-model lineage/design metadata (see the conspicuously
synthetic example in `configs/png-metadata.example.json`) and an explicit candidate
grid; a bad model blocks the entire export pool. No labels, split, aliases or grid
are inferred. Without --stack-order the default cnt_mode is image_only: full-frame masks and the union of every CNT foreground survive export, with connections: [] and connectivity_observation: unavailable. No relation, endpoint or vertical placement is inferred. No model-input crop/tiling is performed; context
windows in the preview are display aids only.

Resource and name inventories precede heavy export/grid allocations. Review N/E/K/R,
bool export bytes and retained float32 loader bytes before use. Image-only uses R=0 and all_components_resolved=null (not_applicable_image_only); that upper scenario applies only to relations mode. The importer retains original rasters for one model at a time, verifies source hashes on
reload, and validates staged Products sequentially. This trades repeated decoding/extraction
for lower retention. Decoder buffers, per-relation full-frame masks and one float32 Product
can still be large; 900 masks of 1500×1500 need about 8.1 GB in one float32 Product alone.
Training and the public `data.load_manifest` API remain eager across all products; estimates
do not certify peak RAM or runtime. Exact role
and layer names must be in the checkpoint's training vocabulary: unknown queries are
unavailable, candidates excluded, and direct embedding fails with the missing names.
A shared symbol does not establish physical equivalence.

**complete means extraction coverage only; electrical verification not performed.**
Coverage reaches the model's connectivity embedding (`unavailable/partial/available`)
and affects input identity. 4-neighbour components are user accepted; skipped-layer physics and multi-electrode ambiguity resolution remain unconfirmed. Real grid/minimum linewidth, source-format compatibility and G6
labels/quality acceptance remain pending. A **5–10 diverse product pilot** is an
integration/connection-inspection pilot, never a required training corpus size or
retrieval-quality acceptance. See USAGE for strict PNG re-save remedies and preview
layer/endpoint interpretation.


The current PNG report/integrity versions are `dl1_png_import_v3` and
`dl1_png_bundle_v3`, algorithm `"2"`; v1/v2 sidecars and shared-number stack assumptions are superseded; strict v1 manifests and model/checkpoint versions stay unchanged.
Electrode layer vocabulary and CNT-plane names are separate report lists; combined
`all_source_*` lists are audit evidence. `cnt_mode` is recorded at top/model/extraction levels and in the completed event. Image-only records `not_run_image_only` and order `not_used`; `ran` applies to optional relation extraction, while coverage and model status remain separate. The verifier rejects
unsupported old sidecars and added files/directories with clean diagnostics.


Every electrode/cno filename number is positive syntax and provenance only; never
renumber source files to encode placement. Electrode bindings use exact model/folder/layer
names with no number. Same-name CNT files with different masks stay separate content-bound source assets;
only the image channel uses their global union. Identical same-name decoded masks are
accepted in image_only with a duplicate_cnt_mask_same_name warning and all files recorded
as provenance of one content asset; relations still rejects them. OCNT/GCNT/PCNT are known names, not a hard-coded whitelist.

Optional **relations** mode is documented in [USAGE §11](docs/USAGE.md#11-선택-사항--순서-파일로-relations-모드-사용):
pass --stack-order with a common reference filtered per model, covering every present
name. Matching uses casefold, collisions reject, and source spelling is preserved in
IDs/manifest/vocabulary; matched_pairs records the spelling pair. The user's reference
of 2026-10-13 is [configs/stack-order.json](configs/stack-order.json); it does not prescribe
universal CNT positions. Other inline/test orders are syntax-only synthetic examples.
**Known gap:** the user confirmed same-electrode contact semantics, but relations still
has same_electrode_required=false and can emit cross-electrode pairs. Do not rely on it
for physical connectivity before that gap is closed and reviewed. Invalid supplied order
files fail; they never fall back to image_only. Keep one mode consistently within a
checkpoint/search: emitted connections change Product input identities.

Migration: re-import original PNGs with metadata/grid into a new output, optionally
supplying a correct order for relations; verify and inspect before use. Do not relabel
old sidecars or reuse old endpoints. Rebuild identity-bound caches. The generic strict
NPY loader and existing static HTML are not retroactively blocked by PNG reader version
rejection; code/identity resume checks still apply. See USAGE §11 for every order error.

Output containment compares existing ancestor file identities, including case/Unicode
spellings and symlinks. Publication uses native atomic no-replace rename when available,
otherwise an exclusive directory claim with manifest written last and disabled on any
failure/interruption. The fallback can expose a partial directory; verify a completed
bundle before consumption. `publication_method` records the route. Only macOS APFS was
exercised on a real host; Linux/Windows publication remains untested on real hosts.
**Exported does not mean the grid is accepted**; warnings and default train/search gates
still apply. Failed stages retain pending manifests and failed/interrupted reports.
