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

JSON contains model/baseline rankings, per-query and aggregate Recall@k, eligible/excluded/unknown coverage, connectivity metadata availability, declared hard negatives, split/input/code/weights provenance and the model configuration (including unit-size conditioning). `performance_acceptance` stays `not_established`; synthetic checks do not assign a real PASS. The real grid/linewidth and source endpoint format remain explicitly pending alongside G6.

## Search contract

Queries and candidates are separate manifests; the query need not have been in the training or candidate pool. Checkpoint versions/config fingerprints/state hashes are checked before loading. All source observations must satisfy T8 validation. Candidate encoding is bounded by `batch_size`; only scores are retained for the full corpus. Exact score ties sort by product ID, independent of input ordering. `topk` and batch size must be positive integers, and duplicate candidate IDs are rejected.

JSON returns unavailable reasons for no candidates, no eligible candidates, and unseen training vocabulary, plus candidate exclusions. Query ID matches are excluded as self results. Connectivity availability is reported independently of similarity scores; unavailable metadata is never presented as CNN-verified connectivity.

```bash
.venv/bin/python -m psdl1.search --ckpt runs/dl1/dl1.pt \
  --query-manifest data/query.json --query new-product \
  --candidates-manifest data/candidates.json \
  --topk 5 --batch-size 32 --out search.json
```

## Candidate execution record

Baseline: clean `rewrite/shape-connectivity-v1` at `81d1f57b0c058460205f26aa4ea44d601b443ce0`. T8 `data.py`, `schema.py`, `docs/data_contract.md`, `pyproject.toml` and dependency lock remain unchanged; no minimal T8 interface adjustments were required. `particle-sim-compare` was read only. Changes are left uncommitted. Independent R3 review remains required and has not been issued by the implementation worker.

Final checks on 2026-10-09:

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -p no:warnings tests/test_relations.py tests/test_split.py tests/test_checkpoint.py tests/test_search.py tests/test_offline.py` | 53 passed, 0.98 s |
| `.venv/bin/python -m pytest -p no:warnings` | 163 passed, 1 skipped, 1.30 s |
| `.venv/bin/python -m pytest -p no:warnings --run-training` | 164 passed, 6.21 s |

The one default skip is `tests/test_model_and_training.py::test_training_lowers_the_loss_and_search_finds_the_same_layout_at_another_unit_size`; it passed in the explicitly authorized synthetic smoke run. `git diff --check` passed. No real layouts/labels, training run, imported pretrained artifact, or network request was used. Minimal T8 interface adjustments: none.
