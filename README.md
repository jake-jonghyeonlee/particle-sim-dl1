# psdl1 — DL1 product similarity

> 한국어 사용 안내: 전체 절차와 입력·출력 형식은 **[docs/USAGE.md](docs/USAGE.md)** 에 정리되어 있습니다.

DL1 aims to find products with similar original shapes and electrode connectivity even when unit sizes differ. The user approved **whole-unit isotropic resize + pad and electrode × actual-layer set aggregation on 2026-10-09**, prioritizing shape/connectivity matching over preserving absolute linewidth. Unit-size conditioning is optional and **off by default** so the user can compare it on and off in the closed network. No second original-µm view is implemented.

The real minimum linewidth/grid choice and the source CNT/VIA endpoint metadata format remain open within G5. G6 label-format and closed-network acceptance criteria remain open. The exact encoder, endpoint feature representation, and training recipe are candidates; no real search performance or connectivity preservation has been established. G3 repository naming remains pending.

Inputs contain only original shape, valid coverage, and supplied connectivity metadata; hotspot, CA, and fab fields are rejected. Products share physical pixel pitch (about 0.05 µm in the user data), have different pixel counts, and share a frame across their layers. See [docs/data_contract.md](docs/data_contract.md) for the accepted T8 draft schema. All reports distinguish the approved input method from pending real grid/linewidth, source endpoint format, and G6.

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

Default tests use direct forward/loss/optimizer-state fixtures and skip dataset training. Socket network calls fail in tests. `pretrained=False` is always used for backbone construction; optional weights load from a local `.pt`, `.pth`, or `.safetensors` state dict with strict shape/finite checks and no online fallback. For the product encoder, local backbone files must match its four input channels. Record an imported artifact's exact hash, source and license before use; none was accepted here.

The local checkpoint path (`--ckpt` / `--resume`) loads this project's own versioned checkpoints. `--weights` accepts compatible backbone-only project/timm-format files, with four input channels and no classifier head for the product encoder. These paths have **not been verified against public ConvNeXt V2 weight files**; no conversion is performed. Pretrained remains off by default. Optional safetensors support is declared in the `local-weights` dependency group and must be obtained through the approved offline wheel process.

Search processes bounded chunks with canonical single-product forwards so padding neighbours and requested chunk size cannot change scores. Cosines use float64 accumulation and are rounded to six decimals before product-ID tie sorting; this favours deterministic results over multi-product forward throughput.

Search and evaluation report the numeric checkpoint `grid` and `grid_selection` with manifest candidate grids. Any mismatch requires `--allow-grid-override`; inference keeps the checkpoint grid and records the exception. Their default grid gate blocks thin-line/vanishing-bridge queries and excludes failing candidates, including held-out evaluation products and the baseline. Only `--allow-grid-check-failure` bypasses it, with per-product evidence in JSON. Both flags are separate, default off, and do not approve a real grid or minimum linewidth.

Canonical single-product forwards preserve order/chunk-size reproducibility. `batch_size` controls candidate chunks rather than forward batch size; evaluation caches each product embedding once per command to avoid repeated forwards across queries, trading O(N × embedding dimension) memory for fewer forwards. Multi-product forward acceleration remains deferred because it changes arithmetic. The opt-in 40-epoch synthetic CNN smoke now uses an explicitly chosen synthetic grid 64, passes the strict grid gate with no bypass, and retains all loss/ranking/separation assertions; this fixture choice is not a real-data grid recommendation.
