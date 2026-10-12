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

Queries and candidates are separate manifests; the query need not have been in the training or candidate pool. Checkpoint versions/config fingerprints/state hashes are checked before loading. All source observations must satisfy T8 validation. Candidate encoding is bounded by `batch_size`; only scores are retained for the full corpus. Canonical single-product forward shapes remove padding/batch-context variation. Evaluation caches one embedding per eligible product for the duration of the command, avoiding repeated forwards across queries; this uses O(N × embedding dimension) memory. `batch_size` bounds chunks, not the number of products in a forward. Single-query search still retains only scores across candidate chunks. Cosines accumulate in float64 and are rounded to six decimals before product-ID tie sorting. The checked encoder path produces identical rankings/scores across input order and chunk sizes; multi-product forward throughput is traded for reproducibility. `topk` and batch size must be positive integers, and duplicate candidate IDs are rejected.

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

Baseline: clean `rewrite/shape-connectivity-v1` at `63a36369597547996818af7606b1400b694b2b25` (the original R3 report reviews `7e52d67`). The changes below were committed as `dba0fb5` and independently reviewed as PASS-WITH-NOTES; this historical implementation record is not itself an independent PASS. `particle-sim-compare` and the original review report were not modified. The coordinator explicitly extended ownership to the two finding-related `pyproject.toml` edits: approval wording and optional safetensors declaration; no install/download occurred.

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

## R3 re-review notes response — 2026-10-09

Baseline: clean `rewrite/shape-connectivity-v1` at `b45bd77d7cd370b84b231ee134f43ab1b01f0977`. This is an implementation record for R1–R6 from the independent `dba0fb5` re-review, not an independent review verdict. The review report and `particle-sim-compare` are untouched. No guessed grid, network access, imported weights or real-data training was added.

| Note | Response | Counterexample evidence |
| --- | --- | --- |
| R1 | Fixed: search/evaluation report numeric applied `grid`, manifest candidate grids and explicit selection provenance. A mismatch errors unless `--allow-grid-override`, which records the conflict while retaining the checkpoint grid | Both real manifest CLI paths reject candidate 8/checkpoint 4 before output creation and accept only the recorded override |
| R2 | Fixed: query/candidate and held-out grid checks recomputed at checkpoint grid. Thin-line/bridge failures block queries and exclude candidates by default; baseline uses the same exclusion. Separate `--allow-grid-check-failure` records per-product bypass evidence | CLI checks for both commands; hand-built 1 px line in 8 px unit at grid 4 has run estimate 0.5; bridge loses components; failing held-out products yield null metrics and empty baseline rankings |
| R3 | Fixed: only the loader's explicit `manifest_candidate_grid` field supplies manifest provenance; a standalone hand-written grid report cannot supply an implicit training grid | API stops without an explicit grid, then reports `api_explicit` and null manifest candidate for explicit 64 |
| R4 | Fixed: historical R3 status reflects committed `dba0fb5`; validator example uses manifest grid, with no literal recommended grid | Documentation regression test; actual grid/minimum width remain pending |
| R5 | Mitigated and made explicit: evaluation caches embeddings once per product per command, reducing repeated encoder forwards from N × N to N; JSON/docs state single-product forward semantics and chunk meaning | Actual encoder held-out fixture requires exactly two forwards for two queries rather than four, with unchanged scores. Single-query multi-product forward acceleration is intentionally unchanged to preserve verified batch-size-independent arithmetic; throughput cost remains documented |
| R6 | Fixed: the existing opt-in 40-epoch real CNN synthetic smoke uses explicit synthetic grid 64 and strict gate without bypass | Checkpoint asserts `passed`, no failures and `bypass_requested: false`; original loss decrease, same-family top-two and >0.1 score separation assertions retained |

Changed expectations and fixtures: search tests now choose the fixture manifest grid 4 when present, or explicit synthetic 64 instead of 16 to keep thin-line rejection from hiding tie/vocabulary assertions. The real-encoder N2 fixture uses explicit synthetic 64 instead of 16 for the same reason; all exact invariance and tie assertions are unchanged. The opt-in smoke changes only its explicit synthetic grid 32 to 64 and removes the bypass, adding the strict-checkpoint assertion; epochs, learning rate and all existing performance assertions are unchanged. No real grid recommendation follows from these fixture choices. No tests were weakened.

Caching is scoped to one evaluation command and requires O(N × embedding dimension) memory; single-query search continues to bound candidate embedding memory by chunks. Original source masks remain eagerly loaded. Per-product evidence is retained even when Python warnings are suppressed. A grid-check pass never certifies connectivity preservation or establishes real performance. Input-method approval remains 2026-10-09; real grid/linewidth, source endpoint format and G6 acceptance remain pending.

| Required validation | Result |
| --- | --- |
| `.venv/bin/python -m pytest -p no:warnings` | 195 passed, 1 skipped in 2.38s |
| `.venv/bin/python -m pytest -p no:warnings --run-training` | 196 passed in 7.13s |
| `git diff --check` | Passed |

The default suite uses encoder/optimizer fixtures only; the existing synthetic CNN smoke remains opt-in. All tests retain the socket-network ban. Changes are left in the working tree without a commit; independent review of this candidate remains the coordinator's responsibility.

## PNG T4 integration candidate — 2026-10-12

Baseline: accepted T3 `e2d329333d6e240dc4af0abb1af324b96efa379f` on
`feat/png-input`. This is implementation evidence for independent R-B review, not an
independent PASS or real-data/quality claim. All earlier validation records above are
historical and retained. No real layout, supplied label corpus, remote request,
package install or pretrained artifact was used. The existing 40-epoch smoke was not
opted into; its default skip remains. The coordinator explicitly added preview/context
files to T4 ownership; core schema/model/checkpoint/search/evaluation/split/loss files
and `requirements.lock` remained unchanged.

| Exact command | Result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/test_png_integration.py tests/test_png_transform.py tests/test_png_compatibility.py tests/test_checkpoint.py tests/test_search.py tests/test_offline.py` | 46 passed, 0 failed, 42.68 s |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/test_png_integration.py::test_grid_loss_gates_and_original_relations_survive tests/test_png_transform.py::test_shared_transform_matches_float32_endpoint_features_at_corners tests/test_png_preview.py tests/test_png_import.py::test_explicit_inspect_grid_evidence_without_export` | 7 passed, 0 failed, 3.91 s |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -p no:warnings` | 446 passed, 1 existing skip, 0 failed, 78.11 s |
| `git diff --check` | Passed |

S17/S18 retain independently specified patterns in all four 500px margins, overlapping
self/rest masks, unresolved global CNT and source relation masks. Hand-computed H=1001,
W=1203/G=32 gives 27×32 content and padding `[0,2,0,3]`; corner/interior continuous
round trips use absolute float64 tolerance 2e-13 and float32 endpoint comparison 1e-7.
S26 inspect/export/validator transform records equal the shared helper's record and
structured endpoint coordinates agree with the same rounded dimensions/pad.

S19 exports two >1000px-axis synthetic products through the actual importer CLI and
validates them through unchanged `load_manifest`/`validate_product` plus the actual
`python -m psdl1.data` CLI. A tiny forward/loss/optimizer fixture and one-step training
driver use explicit synthetic lineage/design IDs, a labelled synthetic negative and a
lineage-safe split; the saved checkpoint records that split, a passing strict grid gate
and exactly one optimizer step. Actual `python -m psdl1.search` and real-encoder API
search return available finite results, stable across candidate order/chunk size.
No assertion treats those rankings as quality/generalization evidence. S21 resolves a
one-pixel CNT at original coordinate `(601,601)`, then proves the small-grid training/
search gates block by default and record an explicit bypass. S24 tests unseen query,
candidate and direct-embed role/layer diagnostics on a passing grid. S27 hooks the real
connectivity embedding and observes `[0,1,2]` for unavailable/partial/complete coverage;
identity changes with coverage, while electrical verification remains not performed.

S20 snapshots **03bf3e33c7107af75297aa298fa323e24018f28d** with `git archive | tar -x`
into test temp storage, without a worktree/checkout/branch. Baseline and candidate run
in separate subprocesses, cwd outside the repositories and explicit PYTHONPATH, and
assert `psdl1.data.__file__` under the expected source tree. Both use seed 47, identical
7×11 bool NPY fixtures, 0.05 synthetic legacy pitch, origin `[1,-2]`, grid 16 and a
16-dimensional tiny encoder with 8-dimensional output. Content is 10×16 with top/bottom
pad 3; hand-specified endpoint features start `[0,1,0.5,0.5,0.5,0.5,0,1]`.
Every normalized image, collated tensor, valid/relation tensor, endpoint feature and
embedding matches **byte-for-byte**, with equal input identity/config. Checkpoint format
`dl1_checkpoint_v1`, schema `dl1_shape_v1_draft` and model input
`dl1_structured_v2_padded_endpoints` stay unchanged; wrong schema/model-input/fingerprint
rejects, baseline checkpoint inference matches exactly, and changed-code resume rejects
before RNG restoration. Snapshot/checkpoint/golden outputs stay in `tmp_path`, not Git.

S22 blocks Pillow imports in a subprocess while running NPY load/checkpoint inference,
then asserts PNG fails with optional-dependency guidance; documentation keeps offline
Pillow/build-wheel prerequisites, 0.1 PNG pitch, full frame extent, no crop, exact
vocabulary and pending G6/pilot limits. Tests retain the socket-network ban.

Preview evidence: `/tmp/dl1-png-preview-evidence/t4-inspection/preview.html`, plus its
sealed report/integrity manifest and local Pillow rasters. A 1401×1603 frame contains
80px crossing traces, 32px CNT pads, direct/margin resolved contacts, skipped-layer
confirmation and ambiguous same-layer lower candidates. The 64px padded context bboxes
are `[0,0,120,120]`, `[520,520,680,680]`, `[720,720,880,880]`,
`[920,920,1080,1080]`; native context keeps one source pixel/image pixel and display
scales are `[4,3,3,3]` with nearest-neighbour pixels verified exactly. A visual montage
at `/tmp/dl1-png-preview-evidence/t4-visual-check.png` was inspected for axes, reference
core, margin visibility, semantic upper/lower colours, CNT outlines/J and multi-pixel
endpoint markers. Bundle verification returned `verified` for all 23 listed files.
The earlier draft preview is preserved at `t4-inspection-initial`; no existing bundle
was overwritten. Display context never enters model input.

Complete means extraction coverage only; electrical verification not performed.
Real grid/minimum linewidth, source compatibility, labels/splits, electrical truth,
resources and retrieval-quality acceptance require closed-network decisions/evidence.
The 5–10 diverse product pilot is strictly integration/connection inspection. Independent
R-B review of the final exact candidate remains the coordinator's next gate.

## R-B finding response candidate — 2026-10-12

Baseline: clean `feat/png-input@01240f9f61541fa328976a5e543c60276cc1fff7`, which the
independent R-B review marked FAIL for the file-identity containment bypass B1.
This is the implementation response to B1 and N1–N8/N11, not an independent review
verdict. Historical validation above is retained. Strict manifest/model/checkpoint
versions, NPY load/preparation/endpoint numerics, extraction semantics and emitted
identities/hashes are unchanged; read-only core files and dependency lock are untouched.
The report/integrity shape changes are explicitly `dl1_png_import_v2` and
`dl1_png_bundle_v2`; older sidecars are cleanly rejected rather than silently reinterpreted.

| Exact command | Result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/test_png_import.py tests/test_png_preview.py tests/test_png_integration.py tests/test_png_source.py` | 249 passed, 0 failed, 65.74 s |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/test_png_import.py::test_publish_fallback_claim_and_manifest_last tests/test_png_import.py::test_publish_fallback_failure_never_leaves_consumable_manifest tests/test_png_import.py::test_fallback_raced_claim_preserves_destination tests/test_png_import.py::test_rb_S15_multi_stage_failure_and_pending_manifest tests/test_png_import.py::test_publication_interruption_and_no_overwrite` | 14 passed, 0 failed, 6.34 s |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -p no:warnings` | 484 passed, 1 existing opt-in skip, 0 failed, 76.00 s |
| `git diff --check` | Passed |

Only macOS/APFS was exercised on a real host. The case and Unicode-normalisation tests
**all ran; none skipped** on this filesystem. Every review reproduction is now exit 2
and the complete surrounding file/directory snapshot (including source bytes and file
mtime) is unchanged before/after the refused invocation:

| Review B1 row | Observed refusal / new entries |
| --- | --- |
| root `Src`, out `SRC/M1/o6` | Refused, 0 entries created |
| root `src`, out `Src/M1/o8` | Refused, 0 entries created |
| root NFC `입력`, out NFD spelling of `입력/M1/o` | Refused, 0 entries created |
| root `Src`, out `src/o5` | Refused, 0 entries created |
| root `Src`, out `SRC/M1/A/o7` | Refused, 0 entries created |

Exact-case root/model/electrode paths, `Src/../Src`, a relative destination from inside
the root, source/output symlink spellings, `out == root` and reverse nesting are also
refused with an unchanged snapshot. A missing/file source root or invalid candidate-grid
argument creates no new output parents or stage. Invalid models/metadata still produce
the authorized diagnostic bundle after inspection.

Fallback tests force the native symbol away, ENOSYS and EINVAL, and a simulated Windows
platform where libc must not be loaded. All select and seal
`exclusive_directory_claim_manifest_last`. Exclusive claim refuses existing/raced
outputs without changing their contents; copy ordering proves manifest is last.
A fault or KeyboardInterrupt immediately after copying the final manifest removes its
consumable name from destination and stage and records failed/interrupted status.
Native Darwin `darwin_rename_excl` was exercised normally. Linux/Windows actual-host
publication remains untested; the fallback exposes a partially copied directory and is
not atomic directory visibility, as USAGE now states.

S15 injects faults at validation, seal, verify, publish, KeyboardInterrupt during seal
and a report write leaving `.writing`. Every failure stage has a pending manifest and
failed/interrupted report; the abandoned writing file does not mask the original error.
S14 detects unlisted/missing files, extra empty/nested directories and a re-sealed report
whose manifest hash link is wrong. Malformed reports/manifests produce clean CLI nonzero
diagnostics without traceback, including omission of publication/algorithm/run fields;
verification requires the complete frozen v2 top-level field set. Electrode/CNT layer unions and unique lists are separate,
raw run status is `ran/blocked/not_run`, metadata errors are location independent, and
exported lineage/design values are exactly the explicit metadata values.

S16 now independently specifies asymmetric bbox `[130,710,137,719]`, crop offset
`[130,710]`, endpoint local `(3,4)`, context offset `[66,646]` and crosshair `(67,68)`.
Overview green marker pixels lie in x `[94,105]`, y `[529,540]`, independently computed
from the source centre `(133,714)` and actual full-frame display scales. Hostile names
include a resolved CNT component and an invalid model; invalid headings, selected
candidate lines and stack text must all remain escaped. No production preview coordinate
or extraction change was required.

Required mutations were each applied only in isolated copies outside both repositories.
The final-code rerun command was `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python
/tmp/dl1-rb-mutations.py`; the driver invokes the installed repository venv with each
scratch `src` on explicit PYTHONPATH, asserts its import path, then runs
`-m pytest -p no:cacheprovider -p no:warnings` with the targeted test below. Every case
exited 1 with exactly one failed test (no collection/import error). Evidence and individual
outputs: `/tmp/dl1-rb-fix-mutants.ju29ta_c/results.json` and each mutant's `result.txt`.

| Mutant | Test that fails |
| --- | --- |
| Removed unlisted-files check | `test_rb_S14_unlisted_missing_entries_and_manifest_report_hash_link[extra-file]` |
| Removed manifest/report hash link | `test_rb_S14_unlisted_missing_entries_and_manifest_report_hash_link[manifest-report-link]` |
| Manifest final name before validation | `test_rb_S15_multi_stage_failure_and_pending_manifest[seal]` |
| Swapped x/y crop offset | `test_rb_S16_asymmetric_offsets_marker_endpoint_and_crosshair` |
| Swapped x/y overview marker | Same asymmetric test |
| Swapped x/y crop endpoint | Same asymmetric test |
| Crosshair shifted 10 px | Same asymmetric test |
| Removed invalid-model heading escape | `test_rb_S16_escapes_invalid_model_candidates_and_stack` |
| Removed selected-candidate line escape | Same escaping test |
| Removed stack-line escape | Same escaping test |
| Folder name used as lineage_id | `test_auto_manifest_export_and_pending_prerequisites` |

The `imported` integration fixture is now module-scoped and new corruption/publication
checks reuse a single immutable CLI export instead of rebuilding it per parameter.
The suite gained 38 tests while elapsed full-suite time was 76.00 s versus the recorded
T4 78.11 s on this machine; this is a local observation, not a portable timing promise.
No assertion was weakened and no opt-in training smoke, install or download ran.
Independent refreshed R-B review and real-data/electrical/G6 decisions remain pending.


A preliminary full run passed 481 tests with one existing skip in 74.49 s. The final
field-completeness guard added three missing-field cases afterwards, so the required
command was rerun on the final code (484 passed above). The preceding targeted run
was 246 passed in 67.39 s; the final targeted rerun was 249 passed as recorded above.
This repeat was due to the new guard/tests, not an unchanged-candidate repetition.
Environment: macOS 26.5.2 arm64 / Python 3.12.13 / Pillow 12.3.0 / NumPy 2.5.3.
