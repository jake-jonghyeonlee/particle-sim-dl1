# DL1 original-shape contract — `dl1_shape_v1_draft`

The whole-unit isotropic resize + pad and electrode × actual-layer set input method was user-approved on 2026-10-09. G3 repository naming, real linewidth/grid, the source CNT/VIA endpoint metadata format and G6 acceptance remain pending. Encoding details remain an internal draft, and synthetic schema checks do not prove electrical connectivity or search performance. Every report uses the same four-field approval object: approved input method, pending real grid/linewidth, pending endpoint source format, and pending G6.

## Manifest

Asset paths are relative to the manifest directory. The example uses one 4×4 unit with pitch 0.05 µm; every referenced mask is 4×4 and every example endpoint must lie in both its original electrode and connection mask. These are synthetic example identifiers and coordinates, not real geometry or labels.

```json
{
  "schema_version": "dl1_shape_v1_draft",
  "candidate_grid": 4,
  "products": [{
    "product_id": "P", "lineage_id": "family-P", "design_version": "v1",
    "unit_size_um": [0.2, 0.2],
    "frame": {
      "frame_id": "P-unit", "length_unit": "um", "axis_order": "yx",
      "origin_xy_um": [0.0, 0.0], "pixel_pitch_um": 0.05,
      "shape_yx": [4, 4]
    },
    "valid": "P/valid.npy",
    "elements": [
      {"element_id": "A-L1", "electrode_id": "electrode-A", "role": "A",
       "actual_layer": "L1", "observation": "observed", "mask": "P/A-L1.npy"},
      {"element_id": "B-L2", "electrode_id": "electrode-B", "role": "B",
       "actual_layer": "L2", "observation": "observed", "mask": "P/B-L2.npy"}
    ],
    "connectivity_observation": "complete",
    "connections": [{
      "connection_id": "via-1", "kind": "VIA", "observation": "observed",
      "mask": "P/via-1.npy",
      "endpoints": [
        {"element_id": "A-L1", "xy_um": [0.025, 0.025]},
        {"element_id": "B-L2", "xy_um": [0.025, 0.025]}
      ]
    }]
  }]
}
```

A product may also supply optional `cnt` as a global original-frame CNT mask. It is not a replacement for endpoint relations. With no supplied relations use `connections: []` and `connectivity_observation: "unavailable"`. `complete` with an empty relation list means an explicitly observed absence of connections; do not use it when metadata is absent.

## Validation and observations

- Product, lineage, design, frame, electrode, role, actual-layer, element, and connection IDs are nonempty trimmed strings. Product IDs are unique within the manifest; element IDs, connection IDs, and role-layer bindings are unique within each product. Lineage may be shared by related products. The draft represents one element per role × actual layer; finer source segmentation requires an explicit contract revision.
- All masks are nonempty, finite, numeric 2-D `.npy` arrays. Binary 0/1 (including bool) or binary 0/255 are accepted and converted to 0/1; negatives, probabilities, NaN, infinities, mixed 1/255 encoding, and intensity masks are rejected. Empty foreground is an observed zero mask, not a missing file.
- A frame uses `um`, `yx`, isotropic positive pixel pitch, positive integer `shape_yx`, and a finite `origin_xy_um`. The origin is the upper-left cell edge; the domain is half open. Size order is width,height and shape order is height,width. `unit_size_um` equals shape × pitch within `atol=1e-9, rtol=1e-9`. Products within a manifest share pitch at the same tolerances; the user value of about 0.05 is not hard-coded. Original layers, CNT/VIA masks, and valid masks must match the frame; the loader never resizes mismatched assets.
- Every asset must exist as a root-relative `.npy` file. Absolute paths (including absolute paths inside the root), parent traversal, Windows absolute paths, backslash separators, and symlinks escaping the root are rejected. Duplicate JSON keys, unknown fields, and missing required fields are rejected.
- `observation` is `observed` with a mask path or `missing` with `mask: null`. Missing elements and connection images remain explicit; they are never called an observed zero. A missing element prevents tensor preparation. Missing connection masks retain their relation and observation status in the structured result; its image slot is zero only as a placeholder alongside that status. Valid holes retain the unknown coverage and mask out shape channels; observed endpoint pixels cannot be in holes.
- A `CNT` or `VIA` relation has exactly two endpoints bound to distinct element IDs and coordinates in the original shared frame. Coordinates must be finite and inside the half-open frame. Observed endpoints must occupy valid electrode and connection pixels. Missing observations still require known endpoint references and valid in-frame coordinates; this draft cannot accept unknown endpoint geometry. It does not infer endpoint coordinates, intermediate layers, electrical labels, or wiring from a mask.
- `connectivity_observation` is `complete`, `partial`, or `unavailable`. Supplied relations require `complete` or `partial`. A complete source declaration with all masks observed yields metadata availability, not verified electrical connectivity; partial declarations or missing assets yield partial availability. A global CNT mask without endpoint metadata leaves availability unavailable.
- Fields named `hotspot`, `CA`, or `fab` are rejected recursively, case-insensitively. The strict field allowlist prevents additional analysis payloads from entering the tensor builder. The builder accepts the typed original-shape product only.

## Candidate preparation and identity

`load_manifest(path, grid=None)` validates the versioned draft and stores `Product.grid_check` for the manifest's required `candidate_grid` or an explicit override. `prepare_product(product, grid, roles, layers)` returns candidate images, bound role/layer IDs, spatial valid coverage, original-frame source relations, normalized connection masks, input identity, metadata availability, and grid evidence. Product identity, lineage, design, unit size, and original frame remain available on the source `Product`; they are not image channels.

A single full-frame isotropic scale and pad recipe is applied to every shape, valid mask, and CNT/VIA mask. It uses area interpolation and rounds resized dimensions to integers; the small rounding discrepancy is recorded through source shape, grid, and scale. The mask foreground is never used to choose a separate centre. Endpoint coordinates remain in the original frame so their physical meaning survives resizing. To locate an endpoint on the candidate grid, convert `(xy-origin)/pitch` to cell-edge coordinates, multiply each axis by its rounded resized size divided by original size, and add the shared pad. This derived position is diagnostic; thresholded occupancy may change under resizing.

The current compatibility images have channels `[self, CNT/VIA union, rest]`. For element i, `rest_i = clip(sum(normalized_element_j for j != i), 0, 1)`. At an XY pixel occupied by two different elements, both rest values are exactly 1 at an unchanged grid. Excluding self **before** composition prevents overlap erasure. Rest loses the layer identity of its contributors; individual elements and their role-layer bindings remain separate. The T9 encoder additionally consumes spatial validity as its fourth image channel and bound endpoint features in the padded-image coordinate frame.

`input_identity` fingerprints original masks, coverage, frame, unit, each bound element tuple, and source endpoint relations independent of element/connection ordering. It is an input fingerprint, not a learned embedding or a lineage split key. A product with bindings A/L1,B/L2 differs from A/L2,B/L1 even with identical XY masks and equal role and layer sets. Element order permutes candidate rows together with their bound IDs and does not change this identity.

## Grid and transform evidence

The validator CLI writes a durable JSON report with the version, current approval status, and one grid check per product at the manifest candidate grid:

```bash
.venv/bin/python -m psdl1.data manifest.json --report grid-report.json
```

Each mask's evidence includes source/target foreground pixel counts, 4-neighbour component counts at a stated target coverage threshold of 0.5, the shortest horizontal/vertical active run, its scaled estimate, and a bridge/component loss flag. Empty foreground has run 0. Axis runs below about 2 candidate pixels produce a warning; this is an auxiliary heuristic, sensitive to corners/diagonals and unrelated to a proven physical minimum linewidth. Real minimum-linewidth acceptance remains G5 pending.

Changing a component count flags bridge splitting, complete erasure, or unintended merging. The checks are conservative diagnostics: equal component counts may still conceal changed paths, endpoints, or topology. Therefore `connectivity_preserving` is always false (no certification), including warning-free cases. The report does not use area similarity to establish connectivity. It records missing observations and valid holes separately.

`augment_with_check` applies one affine transform to every element and channel, returns component evidence and the common theta, and always leaves `safe_for_positive_label: false`. A deliberately removed 1-pixel bridge in the tests changes only `8/384 ≈ 2.08%` of image area yet splits one component into two. Shrink and common-crop tests both report that loss with hand-written expected counts. Every image channel, including CNT/VIA, contributes to the common-transform loss report. The trainer rejects geometric augmentation and uses none or intensity-only views; it blocks thin-line/bridge grid failures unless an explicit recorded bypass is supplied.

## Validation evidence and limits

Synthetic tests compare the same shape at pixel sizes 40,80,120,160 (pitch 0.05 µm, differing unit dimensions) using grids 16,32,64. Mean absolute normalized-mask coverage error must be below 0.04. Exact unchanged-grid overlap values and role-layer ID arrays provide independent hand-written oracles. Tests also exercise valid holes, source endpoints, missing assets, input identity, units, frame extents, and path rejection.

The default full suite excludes the existing 40-epoch T9 training/search test unless explicitly enabled with `--run-training`; no training, pretrained weights, or network are required for this contract. The historical T8-only implementation did not change the model, losses, training or search; T9 now consumes this structured contract. Real electrical/connectivity labels, held-out search quality, the source metadata adapter, and final G5 policy acceptance remain pending.

## Historical T8 execution record — 2026-10-09 (before T9 and R3 fixes)

Baseline: `rewrite/shape-connectivity-v1`, commit `44a13a671d80f41ce495a7f1bf2da2cca2e16264`, initially clean. T8 was subsequently committed as `81d1f57`; this section records its initial validation only. T9 implementation files and `particle-sim-compare` were not modified; no import/signature adjustments to T9 files were needed.

| Command | Observed result |
| --- | --- |
| `.venv/bin/python -m pytest -p no:warnings tests/test_data.py tests/test_input_contract.py` | 90 passed, 0.21 s |
| `.venv/bin/python -m pytest -p no:warnings` | 99 passed, 1 skipped, 1.34 s |

The skipped test is `tests/test_model_and_training.py::test_training_lowers_the_loss_and_search_finds_the_same_layout_at_another_unit_size`, owned by T9 `train.py` and `search.py`; running it would violate T8's no-training scope. All other collected tests passed with socket connections blocked.

The bridge example's stored check uses source shape `[16,24]`, grid 6, scale 0.25, coverage threshold 0.5, and 4-neighbour components. Its source contains 136 active pixels and 1 component; the thresholded normalized mask contains 8 active pixels and 2 components. The shortest axis run is 1 pixel (estimated 0.25 at the chosen grid), and both thin-line and bridge/component-loss warnings are set, with connectivity unavailable and preservation uncertified. This evidence is synthetic and does not approve a real grid. Independent R3 review subsequently identified findings recorded in `docs/review/R3_REVIEW.md`; the fix candidate requires re-review.

## Current training grid policy

`candidate_grid` is required and is not inferred from physical pitch or an example configuration. Training uses that manifest value by default. A conflicting explicit grid requires `--allow-grid-override`; a thin-line/bridge failure requires the separate `--allow-grid-check-failure` flag. Both decisions and the evidence are recorded in checkpoint metadata. Reports remain diagnostics, not electrical preservation certificates. The input identity canonicalizes numeric metadata so integer/float JSON serialization of the same physical values has the same fingerprint.

Inference checks the checkpoint grid against every manifest candidate grid. A mismatch requires `--allow-grid-override`, which retains the checkpoint grid and records both values; it does not approve the physical grid. Search and evaluation recompute thin-line and bridge diagnostics for query/candidate/held-out products at the applied grid. A failed query is unavailable and a failed candidate is excluded by default. `--allow-grid-check-failure` records a per-product bypass; the image baseline follows the same eligibility policy. Hand-computed grid reports without the loader's explicit `manifest_candidate_grid` field are diagnostics, not manifest provenance, and cannot supply an implicit trainer grid.

## PNG source adapter contract (T1)

The separate `png_source` adapter discovers `<root>/<model>/<electrode>/<seq>-<layer>.png`
and `<root>/<model>/cno/<seq>_<layer>.png`. The first hyphen or underscore after a
positive integer separates stack from the exact layer suffix; subsequent separators
belong to the name. Zero padding normalizes the numeric stack, while original filenames
and relative paths remain provenance. Stack numbers increase downwards and are model-wide.
Same-stack conflicting names, duplicate electrode/layer bindings (including repeated
bindings at another stack), malformed PNG names, shape mismatches, unsupported directory
layouts and escaping symlinks invalidate the model. Unrelated files are inventory warnings.
A repeated layer name at distinct stacks is reported and preserved, never merged; a
binding collision blocks use in the existing v1 export. `cno` is reserved for CNT.

Decoding requires optional `psdl1[png]` (Pillow 12.3.0, already in the lock snapshot);
Pillow is imported lazily and the existing NPY path remains independent. Use approved
offline wheels; this dependency declaration does not provide or download wheels.
Single-frame modes 1/L/RGB/LA/RGBA/P are accepted only when decoded RGB is exactly black
or white and alpha is fully opaque. Black becomes bool foreground 1; white becomes 0.
Palette and transparency expansion is exact. Grey, colour, transparent pixels,
16-bit/unsupported modes, nonidentity EXIF orientation, animated or corrupt PNG fail
with file paths. No threshold, gamma adjustment, rotation, registration or resampling
is performed. Empty observed foreground is valid.

The installed Pillow finite pixel guard is retained. Its warning and hard error are
both file errors, recording dimensions/pixels when available, exception class, Pillow
version and effective limit. `png_max_image_pixels` is an explicit positive integer
override, recorded together with the prior default and restored in `finally`; it never
disables the guard. Decoding sessions are serial while the global override is active.
An externally disabled Pillow guard requires an explicit finite override. Resource
suitability is pending; no threshold here approves real data.

The source frame uses shape `[H,W]`, pitch **0.1 µm/pixel**, local edge origin `[0,0]`,
full extent `[W*0.1,H*0.1]` and centre coordinates `[(x+0.5)*0.1,(y+0.5)*0.1]`.
Both W and H must exceed 1000. The reference-only core bbox is the half-open rectangle
`[500,500,W-500,H-500]`, with core layout `[2,2]` and unknown single-unit extent.
All real margin patterns remain in input. Alignment is a source assertion, not measured
registration. `unit_size_um` in a future v1 export will carry the **full frame extent**,
without claiming a single physical unit. No core crop or tiling is performed.

Exact sorted role/layer sets are recorded per model, across-model unions and names unique
to one model; repeated assets count once per model. Roles are electrode folder names.
These inventories cannot certify checkpoint vocabulary coverage. Unknown symbols retain
the existing embedding/search diagnostics; equal names share embedding symbols without
proving electrical or functional equivalence. No aliases are inferred.

Canonical identities use sorted UTF-8 JSON plus SHA-256, independent of source root and
enumeration order. Asset binding identity contains model/kind/electrode/numeric stack/layer;
frame identity contains the local geometry. Binary-mask hashes include shape and bool
pixels, separately from source-file byte hashes. Component identity uses original bbox,
frame shape and local binary geometry, with no full-frame component allocation. Original
filename changes/re-encoding may change provenance without changing logical geometry.

Metadata is a strict JSON object with exactly `metadata_schema_version` and `models`:

```json
{"metadata_schema_version":"dl1_png_metadata_v1","models":{"MODEL":{"lineage_id":"USER_FAMILY","design_version":"USER_VERSION"}}}
```

Each mapping entry contains exactly the two nonempty trimmed strings shown. Duplicate
keys, unknown discovered-model names, wrong versions/types, extra fields and invalid
strings fail. Inspection permits partial metadata and explicitly lists pending models;
export requires all discovered models mapped. Nothing is inferred from folder names.

`png_contract` freezes source, metadata, report, CLI-options and resource records. The
sidecar version is `report_schema_version: dl1_png_import_v1`; readers reject other
versions. Resource estimates begin unknown with reasons until extraction fills N/E/K/R,
CNT indicator U, generated-mask count, bool payload/actual NPY-header/total bytes,
retained float32 loader bytes and the all-components-resolved scenario. They exclude
manifest/report/preview bytes and do not predict peak RAM or runtime. Invalid inputs
never receive reassuring zero estimates. Electrical and real-data verification remain
`not_performed`. The 4-neighbour/skipped-layer policy is an implementation interpretation,
not an additional user-confirmed physical fact. T1 emits no connections.

The frozen CLI spelling for subsequent T3 implementation is
`python -m psdl1.import_png --root ROOT --out OUTPUT` for inspection,
`--inspect-grids G [G ...]` for explicit optional grid evidence,
`--png-max-image-pixels N` for the explicit finite decoder limit, and
`--export --metadata METADATA_JSON --candidate-grid G` for model-ready export.
`--verify-bundle BUNDLE` verifies a published bundle without sources. No grid is chosen
by default. T1 supplies contracts/APIs; the importer CLI and export/preview are T3 work.
Existing schema/model-input versions, NPY loader and requirements lock are unchanged.

## Original-resolution CNT extraction (T2)

`png_connections.extract_model` operates without lineage/design metadata or grids.
It labels each CNT source plane independently using horizontal-run union/find with
4-neighbour edge overlap. The single int32 source label array is released after each
file; bbox/pixel counts derive from runs, without a full-frame scan per component.
Each component retains a bbox-local bool mask. Candidate intersections, common-contact
checks, geometry hashing and centroid selection receive only bbox slices. Large or
widely spanning bboxes may still be expensive; this is no universal sublinear guarantee.

For CNT stack s, full overlapping electrode candidates are recorded above, below and
at equal stack. The maximal overlapping upper stack and minimal overlapping lower stack
are selected **before** their common intersection is tested. Multiple electrodes on a
selected stack, missing side, same endpoint element, equal-stack contact, empty common
intersection or disconnected common-contact regions leave the component unresolved with
named reasons and no relation. A failed nearest pair never falls through to a deeper
layer. Other lower overlaps retain the explicit reason
`deeper_than_nearest_overlapping_lower`. Equal-stack electrode/cno overlap is reachable
only with the same stack/name; conflicting names already invalidate source inventory.

A uniquely selected pair may skip modeled nonoverlapping layers and still emit one
relation. Its `skipped_upper_layers` and `skipped_lower_layers` separately group actual
electrode and cno stack/layer entries with asset/path references; numeric gaps do not
invent layers. `needs_confirmation` and directional summary counts expose this rule.
The nearest-overlap/skipped-layer and 4-neighbour rules are implementation interpretations,
not independently established physical facts. A connected common intersection cannot
prove one physical contact; disconnected common regions flag possible multicontact and
are never heuristically split.

The representative is an actual common-intersection pixel closest to its centroid.
Squared distances compare Python integer numerators, avoiding floating-point ties and
integer overflow; ties sort by y, then x. Bbox offsets restore original coordinates.
Both endpoint references share `[(x+0.5)*0.1,(y+0.5)*0.1]` µm. Component/connection IDs
bind canonical original geometry and source/element identities, independent of traversal,
root path and PNG re-encoding. Duplicate/colliding emitted identities fail explicitly.

All resolved components yield at most one observed CNT relation each. All/some/zero
resolved components produce `complete`/`partial`/`unavailable`; missing cno, present cno
without PNGs and observed white-only CNT have distinct reasons. The global CNT mask
preserves **all** source foreground, including unresolved components. A skipped-layer
confirmation flag alone does not force partial coverage. **complete means extraction
coverage only; electrical verification not performed.** Both electrical and real-data
validation remain `not_performed`; no absent/white image certifies electrical absence.

`extract_inventory` returns resource-populated source records plus compact extraction
records for sidecar serialization. `iter_connections` materializes one full-frame bool
component mask at a time for export; callers must release consumed masks.
`to_product` is the explicit existing-schema compatibility boundary: it retains full-frame
connection masks, creates full valid/global-CNT masks, preserves original frames and
coordinates, and runs `validate_product`. It keeps lineage/design absent unless supplied
as explicit metadata. Callers must surface estimates before that potentially heavy step.
The legacy loader/validator still adds float32 copies and further allocations.

Estimates use N=H*W, E=electrode assets, K=nonempty CNT components, R=resolved relations,
and U=1 when decoded CNT source files exist (including white-only), otherwise 0.
Generated mask count M=E+1+U+R includes valid, optional global CNT and connection masks.
Bool payload is N*M bytes; projected retained loader masks use 4*N*M bytes. Actual NumPy
v1 bool header encoding is generated in memory, without creating any export-sized array,
and its M-fold overhead is included in total projected NPY bytes. The upper scenario
replaces R by K. Invalid source estimates remain unknown with a reason. Estimates exclude
manifest/report/preview files and decoder/label/temporary/grid/Python/validation overhead;
they predict neither peak RAM nor runtime and do not approve real resource suitability.

S23 measured labeling, extraction and estimate generation on a synthetic 2000×2000 frame
with 300 disjoint 2×2 components, one 200×200 square and one 3×500 elongated component.
One small component lacks upper overlap: K=302 and R=301. Hand-specified centres include
`[1099,1099]` for the even square and `[1149,1501]` for the elongated region. Structural
instrumentation requires precisely one source labeling pass, 604 bbox overlap checks,
301 bbox common intersections/representatives and 905 bbox hash operations; full-frame
connection-mask materialization is forbidden during the measured operation.

Measured elapsed time: **0.055192 s** on **Mac16,11 / Apple M4 Pro**, macOS 26.5.2 arm64,
Python 3.12.13, NumPy 2.5.3. The coordinator reviewed the initial 10-second proposal and
requested **`SCALE_REGRESSION_SECONDS=2.0`**, additional larger components and re-measurement;
the revised measurement is below its 0.2-second re-ask threshold and the guard exceeds
10× measured time (about 36×). This committed constant is solely a deterministic synthetic
regression guard, not a production limit; changes require evidence/coordinator review.
Timing excludes PNG decoding/source hashing, disk export, full legacy loading, model
inference and network. No real source or hundreds of full-frame masks were exported.

For this fixture M=305, projected bool payload=1,220,000,000 bytes, headers=39,040 bytes,
NPY total=1,220,039,040 bytes and retained float32 masks=4,880,000,000 bytes. The K scenario
has M=306, NPY total=1,224,039,168 bytes and float32 masks=4,896,000,000 bytes. A single
synthetic bool NPY under test `tmp_path` independently checks the 128-byte header.
