# DL1 training and recovery

The user approved whole-unit isotropic resize + pad and electrode × actual-layer set aggregation on 2026-10-09. Shape/connectivity matching has priority over absolute linewidth. `use_unit_size` is off by default and remains an ablation switch. There is no second original-µm input. Real minimum linewidth/grid and the source CNT/VIA endpoint format remain G5 pending; all G6 label and acceptance criteria remain pending. This implementation is a model/training candidate, not a real-data performance claim.

## Inputs and splits

Use the T8 `dl1_shape_v1_draft` manifest and explicit `lineage_id` values. Build the split JSON with `schema_version: dl1_split_v1` and `train`, `validation`, `test` lists of unique product IDs. Every product must occur exactly once, the training list must be nonempty, and all variants of a lineage must be in one partition. Size variants, patches and particle conditions are not independent products. `configs/split.example.json` shows the format with illustrative IDs.

`psdl1.split.lineage_split` can create a reproducible whole-lineage allocation; small fractions may produce empty validation/test partitions and must be reviewed before claiming an evaluation split. The CLI requires an explicit split. The Python `train` API's omitted split produces an explicit training-only manifest for the synthetic smoke, with empty held-out lists and no generalization claim.

Role/layer vocabulary and statistics are fitted only on `train`. IDs start at 1; 0 is padding, never an unknown bucket. A role or layer unseen during training makes a query unavailable or excludes that candidate with its role/layer names. This may reduce held-out coverage and is reported; it is never silently mapped onto a known role. Metadata includes original input identity/lineage/design records for all declared products solely to validate provenance, not to fit test statistics.

Relations are a draft CSV with no header: `product_a,product_b,similar|different`. IDs must reference products in the manifest; self pairs and empty IDs are rejected. Pair order is irrelevant. Duplicates of one label are harmless; any pair in both label sets is rejected. Cross-partition relations remain in the recorded contract but cannot affect training losses. The final source label format is still G6 pending.

Without an explicit relation, a pair is UNKNOWN. Positive self relations supply a second view of the same product; UNKNOWN entries do not contribute to the numerator, denominator, or gradients. Rows with no known negative are excluded and counted. A run with no usable negative edge is rejected before model training. A batch must contain at least two unique products; singleton inference/forward is supported by LayerNorm. No claim that BatchNorm always prevents collapse remains.

`unlabelled_negatives: true` explicitly fills otherwise UNKNOWN off-diagonal entries as experimental negatives, even with a partial label file. This can repel truly similar products and must be evaluated separately. Default is false. The sampler anchors batches on known negative edges, adds explicit positives when room permits, and guarantees every negative edge is covered each epoch. It reports observed positive/negative pair coverage, effective directed pairs (excluding self), positive-only rows and products that could not be scheduled. Very small batches may omit positive edges or isolated products; these omissions are visible and are not called effective samples.

## Model and grid

`prepare_product` is consumed without changing T8 files. Images retain the self/CNT-VIA/rest candidate channels and a fourth spatial-valid coverage channel. The encoder binds role/layer to each element. Source relations use the two bound element representations plus whole-unit-relative endpoint XY, kind, observation flag and normalized mask mean coverage, pooled symmetrically across endpoint and connection order. An unavailable/partial/source-complete indicator preserves missing connectivity metadata. This is a candidate feature design; availability is never a CNN connectivity verdict.

Training stores each training product's T8 grid report. Review thin-line, missing-observation and bridge/component warnings before selecting a real grid. Reports do not certify preservation; the threshold/component heuristic may miss topology changes. Grid/linewidth approval is still required for real use. Missing element observations cannot be encoded as zero. Missing connection masks retain source relation coordinates and observation state.

Geometric/crop augmentation is rejected. `augmentation: none` is the default. `intensity` is an optional ablation: multiply all elements and image channels of a product by one scalar in [0.9,1.0], leaving spatial validity and source endpoint coordinates unchanged. This does not crop, translate or disconnect original geometry. Resize itself remains subject to the grid warnings; this policy does not claim a resized tensor is electrically certified.

## Complete configuration

`configs/train.example.json` is accepted by the training CLI. Explicit command-line flags override JSON defaults. Relative paths resolve from the working directory. Its grid 256, batch size 8, lr 0.0003, temperature 0.1, epochs 100 and seed 0 are illustrative values, not approved closed-network hyperparameters. Replace data filenames and validate the split/relations before use.

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

Resume uses the same total target epoch count semantics: `--epochs 150` continues a saved epoch 100 through epoch 150. No reseeding of the resumed sampler occurs after restoration. Config, input, split/relation/policy and code mismatches are rejected before restoring optimizer/RNG. Change of total target epochs and output location is permitted; changes of learning rate, batch size, temperature, augmentation, labels, split, initial input or model configuration require a new run.

```bash
.venv/bin/python -m psdl1.train --config configs/train.example.json \
  --resume runs/dl1/dl1.pt --epochs 150
```

Recovery is at the last completed epoch. Work from an interrupted partial epoch is repeated using the saved RNG/optimizer; no partial-step resume is claimed. Checkpoint reads use `weights_only=True`, reject incompatible versions/fingerprints, and verify state contents. Defaults use CPU; CUDA training is supported with its RNG states. Bitwise equality across different hardware/backends is not promised.

Default unit tests perform direct forward/loss and optimizer-state fixtures only. The existing 40-epoch synthetic dataset smoke is explicit `--run-training`; its size-variant assertions are not lineage-held-out evidence. Real training and closed-network acceptance remain user-owned.
