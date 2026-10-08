# DL1 validation and held-out evaluation

The user approved whole-unit isotropic resize + pad and electrode × actual-layer set aggregation on 2026-10-09, with shape/connectivity matching prioritized over absolute linewidth. Unit-size conditioning remains optional and off by default; compare both settings in the closed network. No second original-µm view is implemented. G5 real linewidth/grid and source endpoint metadata format, G6 labels/acceptance thresholds, and G3 naming remain open. The model/evaluation harness has no real-data performance PASS.

## Default contract checks

```bash
.venv/bin/python -m pytest -p no:warnings \
  tests/test_relations.py tests/test_split.py tests/test_checkpoint.py \
  tests/test_search.py tests/test_offline.py
.venv/bin/python -m pytest -p no:warnings
```

Default tests perform direct model forward/loss and optimizer-state fixtures, not dataset training. The inherited synthetic dataset smoke is skipped unless explicitly opted in:

```bash
.venv/bin/python -m pytest -p no:warnings --run-training
```

The smoke checks loss reduction and same-family search at another unit size. Its training sizes 40/64 and query size 96 share source families, so the result is a synthetic implementation check, **not a lineage-held-out generalization result**. A separate counterexample rejects assigning one lineage's 40/64/96 variants to train and test. No real training, pretrained artifact or network is used by these checks.

Required counterexamples and evidence:

| Concern | Independent expected behavior |
| --- | --- |
| UNKNOWN B raised to similarity logit `1e20` against A | A/C loss unchanged within `1e-8`; unknown-logit gradient exactly zero; B's positive-only row excluded |
| A-B labelled similar and different, in either order | Error for both sets and CSV rows |
| Sparse negatives outside ordinary random batches | Every labelled negative edge scheduled each epoch, tested with 11 products and batch size 2 |
| Same lineage at sizes 40/64/96 across train/test | Split validation fails with lineage leakage |
| Role/layer absent from training | Query unavailable or candidate excluded with explicit names; no unknown-to-known mapping |
| Endpoint-only hard negative with identical XY images | T8 image tensors and image baseline are equal, bound endpoint features differ; item is wholly held out and untrained ranking validation is rejected |
| Checkpoint round trip | Identical embeddings with equal batching; schema/format/fingerprint/state-hash mismatch rejected |
| Resume | Optimizer state and Python/NumPy/torch/sampler RNG restored exactly in direct fixtures; changed input/contract/code rejected |
| Offline/local weights | Socket calls fail; tiny and local tiny/timm state files load with `pretrained=False`; missing or nonfinite weights fail without fallback |
| Small batches/invalid model inputs | Singleton forward supported; all-padding, nonfinite images, invalid dimensions/head counts and invalid trainer arguments rejected |

T8 overlap, narrow-line/bridge, valid hole, binary/frame/path and binding tests still run in the full suite. The T9 structured-input test demonstrates that endpoint/validity changes reach model computations; it does not use a random model's ranking as connectivity evidence. The endpoint example comes from a synthetic four-pixel source frame and is not invented real electrical geometry.

## Evaluation protocol

Supply the exact full input pool and split used in the checkpoint, preserving product IDs, lineage, frame, shape and source relation identity. The harness verifies all source input/lineage hashes and the exact split against checkpoint provenance. Product IDs and lineages are disjoint across partitions; every endpoint-only hard negative must have a `different` relation and both IDs inside the test partition. It is never used to select training hyperparameters or to prove an untrained ranking.

The evaluator requires a checkpoint recording at least one completed training epoch. This is a necessary provenance check, not a sufficient performance criterion. Default unit tests test metric arithmetic with fixed ranking fixtures and never assert quality from a random encoder. The final G6 real labels, held-out families, goal Recall@k, allowed omissions, and runtime/resource limits must be supplied by the user.

```bash
.venv/bin/python -m psdl1.evaluate \
  --ckpt runs/dl1/dl1.pt --manifest data/manifest.json \
  --split data/split.json --relations data/relations.csv \
  --hard-negatives data/hard-negatives.json \
  --topk 5 --batch-size 32 --out evaluation.json
```

`configs/hard-negatives.example.json` illustrates the ID-pair list only; it is not a real label set. The model and baseline rank the same eligible held-out candidates for each eligible held-out query. The simple baseline aligns original-shape images by bound `(role, actual_layer)`, normalizes the whole unit to the checkpoint grid, masks invalid coverage, substitutes zeros for absent bindings, and ranks by mean absolute image difference. It has no learned weights and intentionally cannot distinguish equal XY shapes with different endpoint metadata. Missing shape observations and unseen vocabulary are excluded consistently with the model policy.

For each query with at least one known positive candidate, `Recall@k = retrieved_known_positive_count / eligible_known_positive_count`. UNKNOWN candidates are not turned into negatives and may occupy ranking positions; the known/unknown counts are reported. Queries with no known positive or unavailable input have `null` metrics with explicit reasons. Aggregate means cover only available query metrics; `evaluated_queries` and `unavailable_queries` expose the denominator. A report with no evaluable positives stays unavailable, not zero performance.

JSON contains model/baseline rankings, per-query and aggregate Recall@k, eligible/excluded/unknown coverage, connectivity metadata availability, separately scored hard negatives (full rank, known-positive scores and score margins), split/input/code/weights provenance and the model configuration (including unit-size conditioning). `performance_acceptance` stays `not_established`; synthetic checks do not assign a real PASS. The real grid/linewidth and source endpoint format remain explicitly pending alongside G6.

## Search contract

Queries and candidates are separate manifests; the query need not have been in the training or candidate pool. Checkpoint versions/config fingerprints/state hashes are checked before loading. All source observations must satisfy T8 validation. Candidate encoding is bounded by `batch_size`; only scores are retained for the full corpus. Canonical single-product forward shapes remove padding/batch-context variation. Cosines accumulate in float64 and are rounded to six decimals before product-ID tie sorting. The checked encoder path produces identical rankings/scores across input order and chunk sizes; multi-product forward throughput is traded for reproducibility. `topk` and batch size must be positive integers, and duplicate candidate IDs are rejected.

JSON returns unavailable reasons for no candidates, no eligible candidates, and unseen training vocabulary, plus candidate exclusions. Query ID matches are excluded as self results. Connectivity availability is reported independently of similarity scores; unavailable metadata is never presented as CNN-verified connectivity.

```bash
.venv/bin/python -m psdl1.search --ckpt runs/dl1/dl1.pt \
  --query-manifest data/query.json --query new-product \
  --candidates-manifest data/candidates.json \
  --topk 5 --batch-size 32 --out search.json
```

## Historical T9 execution record (before R3 review)

Baseline: clean `rewrite/shape-connectivity-v1` at `81d1f57b0c058460205f26aa4ea44d601b443ce0`. T8 `data.py`, `schema.py`, `docs/data_contract.md`, `pyproject.toml` and dependency lock remained unchanged in T9; no minimal T8 interface adjustments were needed then. `particle-sim-compare` was read only. T9 was subsequently committed as `7e52d67`. Independent R3 review found B1 and N1–N9; its original report remains unchanged in `docs/review/R3_REVIEW.md`.

Final checks on 2026-10-09:

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -p no:warnings tests/test_relations.py tests/test_split.py tests/test_checkpoint.py tests/test_search.py tests/test_offline.py` | 53 passed, 0.98 s |
| `.venv/bin/python -m pytest -p no:warnings` | 163 passed, 1 skipped, 1.30 s |
| `.venv/bin/python -m pytest -p no:warnings --run-training` | 164 passed, 6.21 s |

The one default skip is `tests/test_model_and_training.py::test_training_lowers_the_loss_and_search_finds_the_same_layout_at_another_unit_size`; it passed in the explicitly authorized synthetic smoke run. `git diff --check` passed. No real layouts/labels, training run, imported pretrained artifact, or network request was used. Minimal T8 interface adjustments: none.

## Hard-negative scoring

`hard_negative_evaluation` is a separate report item. With no supplied pairs it is `unavailable` with reason `none_supplied`. Each declared pair is checked as different and wholly held out, then scored in both query directions against the full eligible test candidate list, regardless of requested top-k. The report includes its score/rank, known-positive scores/ranks, whether it lies below all known positives, and the margin to the lowest positive. No-positive or unavailable scores produce explicit unavailable comparison reasons. Threshold acceptance remains G6 pending; a descriptive ordering result is not a real performance PASS. The CLI fixture tests a hard negative at rank 2 with topk 1, and hand-computed margins +0.3 and -0.3.

## R3 finding response — fix candidate, 2026-10-09

Baseline: clean `rewrite/shape-connectivity-v1` at `63a36369597547996818af7606b1400b694b2b25` (the original R3 report reviews `7e52d67`). Changes below remain uncommitted for independent re-review; this implementation record is not an independent PASS. `particle-sim-compare` and the original review report were not modified. The coordinator explicitly extended ownership to the two finding-related `pyproject.toml` edits: approval wording and optional safetensors declaration; no install/download occurred.

| Finding | Status and concrete change | Counterexample/evidence |
| --- | --- | --- |
| B1 | Fixed: no trainer/API guessed grid; manifest default, conflict rejection, explicit recorded override | `test_b1_*`: CLI candidate 4 stays 4 without --grid; grid 2 is rejected unless allowed and is stored in checkpoint metadata |
| N1 | Fixed: experimental negatives require lineage IDs and never negate same-lineage variants | Hand-written 40/64/96 matrix has UNKNOWN within comb and NEGATIVE only against bus; explicit same-lineage negative is rejected in this mode |
| N2 | Fixed: canonical single-product inference, float64 cosine accumulation, six-decimal scores and ID tie sorting | Real encoder tests identical two-element A/Z mixed with three-element M, across three orders and batch sizes 1/2/3/8, asserting identical complete ranking rows |
| N3 | Fixed: hard-negative scores/ranks and positive comparisons are separate report items; absence and unavailable scores have reasons | CLI fixture scores rank-2 negative with topk 1, hand-computed positive/negative margins ±0.3, none-supplied and unseen-role availability checks |
| N4 | Loader not changed, as instructed; documentation fixed to describe own versioned checkpoints and compatible backbone-only/timm-format files, unverified against public ConvNeXt V2 | Existing offline local-file checks preserved; added test explicitly rejects three-channel classifier state without conversion |
| N5 | Fixed: normalized endpoint coordinates include rounded resize and common padding offsets | Hand-computed 4×8 unit/grid 8 bottom-right centre yields (0.9375,0.6875), matching connection image cell [5,7] |
| N6 | Fixed: one shared four-field approval object; current docs updated and prior T8/T9 execution records explicitly historical | Approval checked in input CLI, grid/common-transform/search reports and trainer checkpoint; no obsolete whole-input pending enum remains in current source/docs/metadata |
| N7 | Fixed: thin-line or bridge/component failure blocks before model construction; only an explicit recorded bypass allows it | Separate thin-line and disappearing-bridge tests; checkpoint bypass/status/failure assertions; strict default tested without relying on Python warnings |
| N8 | Fixed: retrieval miss arithmetic, absolute loss oracle, full resume driver fixture and distinct family CNT geometry | Recall 0 and 1 query examples aggregate to 0.5; loss ln(1+exp(-1)) = 0.3132616875; continuous/resumed driver history and weights hashes identical after RNG disruption |
| N9 | Fixed: incomplete checkpoint ValueError, POSITIVE diagonal, numeric canonical identities, one preparation per product/run, optional safetensors declaration, checks for every common-transform channel | Integer/float unit serialization has identical identity; preparation-count assertion; CNT-only crop reports 1→0 components despite empty unchanged self channel |

Original masks remain eagerly loaded; the preparation cache removes repeated per-view/per-step work without claiming a streaming input loader. No CUDA or real-input performance claim is added. Endpoint coordinate semantics are now versioned as `dl1_structured_v2_padded_endpoints` so older coordinate-model checkpoints are rejected explicitly.

Changed existing expectations and fixtures (no assertions weakened):

- `tests/test_input_contract.py` now expects the complete approved-input/pending-real-grid/pending-endpoint-format/pending-G6 object instead of the obsolete whole-input pending string (N6).
- The fixed-ranking evaluation fixture now honours topk and uses k=1: model recalls 0 and 1 average to 0.5 and tied-image baseline recall is 0, replacing the old trivial all-candidates recalls of 1 (N8).
- Opt-in negative matrix tests supply explicit distinct lineage IDs; their existing cross-lineage expected negative values are unchanged (N1).
- Crop/empty-split/invalid-argument tests supply an explicit valid grid so they still test their intended rejection rather than the new missing-grid error (B1).
- The 40-epoch synthetic smoke explicitly records `allow_grid_check_failure=True` for its deliberately small grid 32 (its 40-pixel CNT run estimates below 2 target pixels). Existing loss reduction, same-family rankings and >0.1 score-separation assertions remain unchanged; the bypass never approves a real grid (N7).
- Family fixtures place CNT at distinct synthetic relative positions; the original same-family size comparison remains a smoke check, not a held-out connectivity result (N8).

Final validation on the fix candidate:

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -p no:warnings` | 184 passed, 1 skipped, 1.86 s |
| `.venv/bin/python -m pytest -p no:warnings --run-training` | 185 passed, 4.60 s |
| `git diff --check` | Passed |

Default tests use direct forward/loss/optimizer fixtures, including a two-parameter fixture through the trainer/resume driver. Only the explicitly opted-in existing synthetic smoke trains the CNN; no real training, pretrained artifact or network was used. The remaining decisions are real grid/minimum linewidth, source endpoint format, G6 acceptance and independent re-review of this exact candidate.
