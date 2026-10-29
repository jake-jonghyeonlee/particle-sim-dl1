# DL1 training and recovery

The user approved whole-unit isotropic resize + pad and electrode × actual-layer set aggregation on 2026-10-09. Shape/connectivity matching has priority over absolute linewidth. `use_unit_size` is off by default and remains an ablation switch. There is no second original-µm input. Real minimum linewidth/grid and the source CNT/VIA endpoint format remain G5 pending; all G6 label and acceptance criteria remain pending. This implementation is a model/training candidate, not a real-data performance claim.

## Windows offline execution (CMD)

Use the prepared [CMD offline bundle](WINDOWS_OFFLINE.md). From its extracted folder,
run the installed interpreter directly; no shell activation or source checkout is needed.
Replace paths with the actual CPU/tiny config, labels, split and checkpoint. Relative
paths inside config files resolve from the current working directory.

```bat
set "DL1PY=%CD%\.venv-cmd\Scripts\python.exe"
"%DL1PY%" -I -m psdl1.train --config "D:\DL1 config\train.json"
"%DL1PY%" -I -m psdl1.train --config "D:\DL1 config\train.json" --use-unit-size --out "D:\DL1 runs\unit-size"
"%DL1PY%" -I -m psdl1.train --config "D:\DL1 config\train.json" --augmentation intensity --out "D:\DL1 runs\intensity"
"%DL1PY%" -I -m psdl1.train --config "D:\DL1 config\train.json" --resume "D:\DL1 runs\dl1.pt"
```

These are CMD commands. POSIX examples below describe the prepared developer environment.
The bundle excludes timm, optional weights and GPU dependencies. Installation/help is
separate from numerical smoke and real-data validation; Windows DL1 execution remains unverified.

## Inputs and splits

Use the T8 `dl1_shape_v1_draft` manifest and explicit `lineage_id` values. Build the split JSON with `schema_version: dl1_split_v1` and `train`, `validation`, `test` lists of unique product IDs. Every product must occur exactly once, the training list must be nonempty, and all variants of a lineage must be in one partition. Size variants, patches and particle conditions are not independent products. `configs/split.example.json` shows the format with illustrative IDs.

`psdl1.split.lineage_split` can create a reproducible whole-lineage allocation; small fractions may produce empty validation/test partitions and must be reviewed before claiming an evaluation split. The CLI requires an explicit split. The Python `train` API's omitted split produces an explicit training-only manifest for the synthetic smoke, with empty held-out lists and no generalization claim.

Role/layer vocabulary and statistics are fitted only on `train`. IDs start at 1; 0 is padding, never an unknown bucket. A role or layer unseen during training makes a query unavailable or excludes that candidate with its role/layer names. This may reduce held-out coverage and is reported; it is never silently mapped onto a known role. Metadata includes original input identity/lineage/design records for all declared products solely to validate provenance, not to fit test statistics.

Relations are a draft CSV with no header: `product_a,product_b,similar|different`. IDs must reference products in the manifest; self pairs and empty IDs are rejected. Pair order is irrelevant. Duplicates of one label are harmless; any pair in both label sets is rejected. Cross-partition relations remain in the recorded contract but cannot affect training losses. The final source label format is still G6 pending.

Without an explicit relation, a pair is UNKNOWN. Positive self relations supply a second view of the same product; UNKNOWN entries do not contribute to the numerator, denominator, or gradients. Rows with no known negative are excluded and counted. A run with no usable negative edge is rejected before model training. A batch must contain at least two unique products; singleton inference/forward is supported by LayerNorm. No claim that BatchNorm always prevents collapse remains.

`unlabelled_negatives: true` explicitly fills otherwise UNKNOWN off-diagonal entries as experimental negatives, even with a partial label file. This can repel truly similar products of different lineages and must be evaluated separately. Same-lineage variants are UNKNOWN unless explicitly similar; a conflicting same-lineage negative under this experiment is rejected. Lineage IDs are required for this policy. Default is false. The sampler anchors batches on known negative edges, adds explicit positives when room permits, and guarantees every negative edge is covered each epoch. It reports observed positive/negative pair coverage, effective directed pairs (excluding self), positive-only rows and products that could not be scheduled. Very small batches may omit positive edges or isolated products; these omissions are visible and are not called effective samples.

## Model and grid

`prepare_product` preserves the original T8 channel and feature contract. Images retain the self/CNT-VIA/rest candidate channels and a fourth spatial-valid coverage channel. The encoder binds role/layer to each element. Source relations use the two bound element representations plus normalized padded-image endpoint XY, kind, observation flag and normalized mask mean coverage, pooled symmetrically across endpoint and connection order. An unavailable/partial/source-complete indicator preserves missing connectivity metadata. This is a candidate feature design; availability is never a CNN connectivity verdict.

Training takes `candidate_grid` from the manifest. Neither CLI nor Python API guesses a default. `--grid` may match the candidate; a different value is rejected unless `--allow-grid-override` is explicitly supplied. The checkpoint records manifest/selected grids, source and the override flag. Direct API callers without manifest reports must supply a grid explicitly.

Training stores each product's grid report and caches preparation once per training call. A shortest-axis-run estimate below 2 candidate pixels or any bridge/component change **blocks training before model construction by default**. `--allow-grid-check-failure` is the sole explicit bypass and records failures, bypass request and strict-default policy in checkpoint metadata and the resume contract. The flag acknowledges heuristic warnings; it does not approve a real grid or certify connectivity. Valid-hole/missing coverage evidence is also retained. Review all evidence before real use. Reports do not certify preservation; the threshold/component heuristic may miss topology changes. Grid/linewidth approval is still required for real use. Missing element observations cannot be encoded as zero. Missing connection masks retain source relation coordinates and observation state.

Geometric/crop augmentation is rejected. `augmentation: none` is the default. `intensity` is an optional ablation: multiply all elements and image channels of a product by one scalar in [0.9,1.0], leaving spatial validity and source endpoint coordinates unchanged. This does not crop, translate or disconnect original geometry. Resize itself remains subject to the grid warnings; this policy does not claim a resized tensor is electrically certified.

## Complete configuration

`configs/train.example.json` is accepted by the training CLI. Explicit command-line flags override JSON defaults. Relative paths resolve from the working directory. It contains no grid default; the manifest must supply `candidate_grid`. Batch size 8, lr 0.0003, temperature 0.1, epochs 100 and seed 0 are illustrative values, not approved closed-network hyperparameters. Replace data filenames and validate the split/relations before use.

```bash
.venv/bin/python -m psdl1.train --config configs/train.example.json
# Unit-size ablation (default is off):
.venv/bin/python -m psdl1.train --config configs/train.example.json \
  --use-unit-size --out runs/dl1-with-unit
# Geometry-preserving image-intensity ablation:
.venv/bin/python -m psdl1.train --config configs/train.example.json \
  --augmentation intensity --out runs/dl1-intensity
```

Pretrained is off. The tiny encoder is the default. Optional timm construction always uses `pretrained=False`; local file loading occurs afterwards, using weights-only torch loading or local safetensors, strict shape matching and finite-state checks. There is no fallback on missing, malformed or incompatible files. Product backbones require four input channels. Source images, imported files and their licenses must be approved in the closed network; no imported pretrained artifact was used in T9.

## Checkpoints and resume

After each completed epoch, an atomic temporary-file replacement writes `dl1.pt`; it is the authoritative recovery artifact. `split.json` and `history.json` are human-readable companions. The checkpoint records:

- `dl1_checkpoint_v1`, original schema and model-input version, canonical config fingerprint, state-dict content hash.
- Training vocabulary/statistics, full split, input identity/lineage/design records and combined input hash, code hash, optional initial local-file weights hash, seed, label/policy contract hash and training grid checks.
- Epoch/history, effective training pairs and coverage, optimizer state, Python/NumPy/CPU torch and sampler-generator RNG, and CUDA RNG when available.

Resume uses the same total target epoch count semantics: `--epochs 150` continues a saved epoch 100 through epoch 150. No reseeding of the resumed sampler occurs after restoration. Config, input, split/relation/policy and code mismatches are rejected before restoring optimizer/RNG. The two grid flags must also match the saved contract on resume. Change of total target epochs and output location is permitted; changes of learning rate, batch size, temperature, augmentation, labels, split, initial input or model configuration require a new run.

```bash
.venv/bin/python -m psdl1.train --config configs/train.example.json \
  --resume runs/dl1/dl1.pt --epochs 150
```

Recovery is at the last completed epoch. Work from an interrupted partial epoch is repeated using the saved RNG/optimizer; no partial-step resume is claimed. Checkpoint reads use `weights_only=True`, reject incompatible versions/fingerprints, and verify state contents. Defaults use CPU; CUDA training is supported with its RNG states. Bitwise equality across different hardware/backends is not promised.

### Warm start (`--init-from`)

`--init-from <parent dl1.pt>` starts a new run from a parent checkpoint's weights, for [incremental rounds](USAGE.md#68-증분-라운드-새-제품만-더하는-학습). It is exclusive with `--resume` and `--weights`; the combination is rejected before any training step. `--round N` records the round number in metadata. `--new-edge-repeat k` (default 1) needs a parent: with neither `--init-from` nor a resumed warm-start checkpoint it is rejected.

- The parent must match the new run in `backbone`, `use_unit_size`, `grid`, `model_kwargs` and `unknown_vocabulary`. A difference is rejected (`init-from parent differs: <field>`) before the optimizer is created.
- The vocabulary is extended, not rebuilt. Every parent role/layer keeps its ID. Names that are new in the current training split are sorted and appended with the next IDs. Parent names that are no longer used stay and are listed as `unused_parent_names`.
- Only the `role.weight` and `layer.weight` embeddings grow. Parent rows are copied. New rows keep the current model's seeded initialisation, and the helper consumes no RNG. Every other tensor is the parent's.
- Optimizer, grad scaler and RNG start fresh, and epochs count from 0. `--epochs` is the total for the new run, not an extension of the parent.
- Metadata adds `parent` (path, `weights_hash`, `contract_hash`, `input_hash`, `fingerprint`, `code_hash`, `code_hash_matches`, `epoch`, `round`), `round`, `warm_start` (parent/total counts and `new_names` per vocabulary) and `delta_negative_pairs`. A parent `code_hash` that differs from the current code is recorded as `code_hash_matches: false`; it is not rejected.
- Delta negative pairs are the current training negative edges the parent never trained. They are rebuilt from the parent's metadata (its split, relations, `unlabelled_negatives` and lineage). `--new-edge-repeat k` schedules each delta edge `k` times per epoch. Every negative edge is still covered at least once per epoch, and the loss is unchanged. Per-epoch statistics add `delta_negative_pairs`, `new_edge_repeat` and `delta_coverage_min`.
- The contract hash covers `new_edge_repeat`, `init_from` (parent `weights_hash`, `contract_hash`, `input_hash`, `fingerprint`) and `delta_negative_pairs_hash`. Runs without a parent store `init_from: null` and `new_edge_repeat: 1`, so their `contract_hash` differs from checkpoints made by older code. Resume already requires an identical code hash, so such old checkpoints cannot be resumed anyway.
- Resume of a warm-start run uses `--resume` with the same `--epochs`, `--lr`, `--new-edge-repeat` and other training arguments, without `--init-from`. The vocabulary, parent record and delta pairs come from the saved checkpoint. The parent file is not read again and may be deleted. A changed `--new-edge-repeat` fails with the contract mismatch. A `--round` value, if given, must equal the saved one.

```bash
.venv/bin/python -m psdl1.train --config configs/train.example.json \
  --init-from runs/r01/dl1.pt --round 2 --epochs 30 --lr 1e-4 --new-edge-repeat 2 --out runs/r02
# interrupted: same arguments, --resume instead of --init-from
.venv/bin/python -m psdl1.train --config configs/train.example.json \
  --resume runs/r02/dl1.pt --round 2 --epochs 30 --lr 1e-4 --new-edge-repeat 2 --out runs/r02
```

Default tests include bounded synthetic forward/loss/optimizer fixtures and a one-step PNG training-driver integration with explicit synthetic metadata, labels and split. The existing 40-epoch synthetic dataset smoke is explicit `--run-training`; its size-variant assertions are not lineage-held-out evidence. Real training and closed-network acceptance remain user-owned.

The checkpoint loader (`--ckpt` / `--resume`) accepts this project's own versioned checkpoints. Optional `--weights` accepts compatible backbone-only project/timm-format states with four input channels and no classifier head. **Loading public ConvNeXt V2 weight files has not been verified**; no stem expansion or classifier conversion is performed, and pretrained remains off by default. The optional `local-weights` dependency group explicitly declares safetensors; use approved local wheels without online fallback.

Endpoint coordinates use the same rounded resize dimensions and central padding as the image, then divide by the grid size. For a 4×8 unit at grid 8, the bottom-right cell centre maps from (7.5,3.5) to (7.5,5.5), yielding (0.9375,0.6875). `MODEL_INPUT_VERSION=dl1_structured_v2_padded_endpoints` identifies this corrected coordinate contract; earlier endpoint-model checkpoints are rejected rather than silently interpreted with changed coordinates.

Preparation caching removes repeated validation/hash/resize for both views and subsequent epochs in one training call. The cache is a snapshot of the validated products, not a global cache across runs. Original masks are still eagerly loaded by the manifest reader; very large input pools remain a memory limitation and no streaming loader is claimed.

Search and evaluation report the numeric checkpoint `grid` and `grid_selection` with manifest candidate grids. Any mismatch requires `--allow-grid-override`; inference keeps the checkpoint grid and records the exception. Their default grid gate blocks thin-line/vanishing-bridge queries and excludes failing candidates, including held-out evaluation products and the baseline. Only `--allow-grid-check-failure` bypasses it, with per-product evidence in JSON. Both flags are separate, default off, and do not approve a real grid or minimum linewidth.

Canonical single-product forwards preserve order/chunk-size reproducibility. `batch_size` controls candidate chunks rather than forward batch size; evaluation caches each product embedding once per command to avoid repeated forwards across queries, trading O(N × embedding dimension) memory for fewer forwards. Multi-product forward acceleration remains deferred because it changes arithmetic. The opt-in 40-epoch synthetic CNN smoke now uses an explicitly chosen synthetic grid 64, passes the strict grid gate with no bypass, and retains all loss/ranking/separation assertions; this fixture choice is not a real-data grid recommendation.


## Imported PNG bundles

Follow [USAGE](USAGE.md#8-실제-png-폴더-입력--검사--명시적-export) to inspect, explicitly
export and validate the unchanged `dl1_shape_v1_draft` manifest. Optional Pillow 12.3.0
and package build wheels must be locally approved; no downloads or online pretrained
fallback occur. PNGs use 0.1 µm/pixel and `unit_size_um` means full frame extent.
No core crop/tiling or label/split/grid inference occurs. Review resource N/E/K/R and
full-frame relation-mask/retained float32 load estimates before loading a pool.

Image preparation, endpoint features and importer reports now call one shared rounded
full-frame transform; original relations/coordinates are preserved before resampling.
Grid gates remain strict by default. Train vocabulary uses byte-exact electrode/layer
names only; unseen query/candidate names are diagnosed and never aliased.
**complete means extraction coverage only; electrical verification not performed.**
That coverage enters the connectivity embedding and identity; confirmation flags are
sidecar inspection evidence, not a new model status. Real linewidth/grid, source-format
validation and G6 labels/quality remain pending. Start with the 5–10 diverse product
integration/connection-inspection pilot before deciding real-data training/quality gates;
that pilot is not a training sample-size or retrieval-quality claim.

Baseline-format checkpoints with the unchanged model-input version still support
inference. Resume requires exact code hash, so a checkpoint produced before the PNG
implementation/shared helper is rejected for resume; begin a new explicitly configured
run rather than bypassing provenance.
