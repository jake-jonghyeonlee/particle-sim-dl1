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

`configs/` contains illustrative filenames/IDs, not real data or approved real hyperparameters. Both manifests, labels, and split must be supplied with actual provenance before running user training. A 2×2 simulation domain is not automatically converted to a unit.

The checked runtime is Python 3.12. The T8 `requirements.lock` records exact installed versions for the macOS runtime, including optional timm/test dependencies; it is not a portable wheel/build lock. Use the closed-network package process to obtain approved distributions and the build backend; no dependency updates or downloads were performed here.

```bash
uv pip install --no-index --find-links /approved/wheels -r requirements.lock
uv pip install --no-index --find-links /approved/wheels -e .

.venv/bin/python -m pytest -p no:warnings
# Explicit opt-in to the existing synthetic training/search smoke:
.venv/bin/python -m pytest -p no:warnings --run-training
```

Default tests include bounded synthetic forward/loss/optimizer and one-step importer/training-driver integration fixtures; the existing 40-epoch dataset smoke stays opt-in. Socket network calls fail in tests. `pretrained=False` is always used for backbone construction; optional weights load from a local `.pt`, `.pth`, or `.safetensors` state dict with strict shape/finite checks and no online fallback. For the product encoder, local backbone files must match its four input channels. Record an imported artifact's exact hash, source and license before use; none was accepted here.

The local checkpoint path (`--ckpt` / `--resume`) loads this project's own versioned checkpoints. `--weights` accepts compatible backbone-only project/timm-format files, with four input channels and no classifier head for the product encoder. These paths have **not been verified against public ConvNeXt V2 weight files**; no conversion is performed. Pretrained remains off by default. Optional safetensors support is declared in the `local-weights` dependency group and must be obtained through the approved offline wheel process.

Search processes bounded chunks with canonical single-product forwards so padding neighbours and requested chunk size cannot change scores. Cosines use float64 accumulation and are rounded to six decimals before product-ID tie sorting; this favours deterministic results over multi-product forward throughput.

Search and evaluation report the numeric checkpoint `grid` and `grid_selection` with manifest candidate grids. Any mismatch requires `--allow-grid-override`; inference keeps the checkpoint grid and records the exception. Their default grid gate blocks thin-line/vanishing-bridge queries and excludes failing candidates, including held-out evaluation products and the baseline. Only `--allow-grid-check-failure` bypasses it, with per-product evidence in JSON. Both flags are separate, default off, and do not approve a real grid or minimum linewidth.

Canonical single-product forwards preserve order/chunk-size reproducibility. `batch_size` controls candidate chunks rather than forward batch size; evaluation caches each product embedding once per command to avoid repeated forwards across queries, trading O(N × embedding dimension) memory for fewer forwards. Multi-product forward acceleration remains deferred because it changes arithmetic. The opt-in 40-epoch synthetic CNN smoke now uses an explicitly chosen synthetic grid 64, passes the strict grid gate with no bypass, and retains all loss/ranking/separation assertions; this fixture choice is not a real-data grid recommendation.

## PNG workflow, offline

Use `psdl1[png]` with approved local **Pillow 12.3.0** wheels (and hatchling/build
requirements when building the package). The optional PNG dependency is imported
lazily: existing NPY/checkpoint inference needs no Pillow. `requirements.lock` is
unchanged and does not guarantee a portable wheelhouse; no online fallback exists.

```bash
# Paths and candidate grids must be supplied by the user; destinations must be new.
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$INSPECT_OUT"
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$GRID_INSPECT_OUT" \
  --inspect-grids "$GRID_A" "$GRID_B"
.venv/bin/python -m psdl1.import_png --root "$PNG_ROOT" --out "$EXPORT_OUT" \
  --export --metadata "$METADATA_JSON" --candidate-grid "$CANDIDATE_GRID"
.venv/bin/python -m psdl1.import_png --verify-bundle "$EXPORT_OUT"
.venv/bin/python -m psdl1.data "$EXPORT_OUT/manifest.json" --report "$VALIDATOR_REPORT"
```

Inspection writes a versioned report and offline full-frame preview without metadata
or a grid. Explicit grid inspection compares candidates without approving/selecting
one. Export requires exact per-model lineage/design metadata (see the conspicuously
synthetic example in `configs/png-metadata.example.json`) and an explicit candidate
grid; a bad model blocks the entire export pool. No labels, split, aliases or grid
are inferred. Full-frame masks, original-resolution CNT relations and all unresolved
CNT foreground survive export. No model-input crop/tiling is performed; context
windows in the preview are display aids only.

Resource and name inventories precede heavy export/grid allocations. Review N/E/K/R,
bool export bytes, retained float32 loader bytes and the all-components-resolved
scenario before use. Decoder temporaries, all retained source masks and per-relation
full-frame masks add memory; estimates do not certify peak RAM or runtime. Exact role
and layer names must be in the checkpoint's training vocabulary: unknown queries are
unavailable, candidates excluded, and direct embedding fails with the missing names.
A shared symbol does not establish physical equivalence.

**complete means extraction coverage only; electrical verification not performed.**
Coverage reaches the model's connectivity embedding (`unavailable/partial/available`)
and affects input identity. 4-neighbour and skipped-layer rules are implementation
interpretations. Real grid/minimum linewidth, source-format compatibility and G6
labels/quality acceptance remain pending. A **5–10 diverse product pilot** is an
integration/connection-inspection pilot, never a required training corpus size or
retrieval-quality acceptance. See USAGE for strict PNG re-save remedies and preview
layer/endpoint interpretation.


The current PNG report/integrity versions are `dl1_png_import_v2` and
`dl1_png_bundle_v2`; strict v1 manifests and model/checkpoint versions stay unchanged.
Electrode layer vocabulary and CNT-plane names are separate report lists; combined
`all_source_*` lists are audit evidence. `extractor_run_status: ran` says only that
extraction ran, while coverage and model status remain separate. The verifier rejects
unsupported old sidecars and added files/directories with clean diagnostics.

Output containment compares existing ancestor file identities, including case/Unicode
spellings and symlinks. Publication uses native atomic no-replace rename when available,
otherwise an exclusive directory claim with manifest written last and disabled on any
failure/interruption. The fallback can expose a partial directory; verify a completed
bundle before consumption. `publication_method` records the route. Only macOS APFS was
exercised on a real host; Linux/Windows publication remains untested on real hosts.
**Exported does not mean the grid is accepted**; warnings and default train/search gates
still apply. Failed stages retain pending manifests and failed/interrupted reports.
