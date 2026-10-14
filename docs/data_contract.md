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
- A frame uses `um`, `yx`, isotropic positive pixel pitch, positive integer `shape_yx`, and a finite `origin_xy_um`. The origin is the upper-left cell edge; the domain is half open. Size order is width,height and shape order is height,width. `unit_size_um` equals shape × pitch within `atol=1e-9, rtol=1e-9`. Products within a manifest share pitch at the same tolerances; legacy synthetic pitches such as 0.05 are not hard-coded; PNG import explicitly uses 0.1 µm/pixel. Original layers, CNT/VIA masks, and valid masks must match the frame; the loader never resizes mismatched assets.
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

## Current PNG source adapter contract — v3 / algorithm 2 (2026-10-13)

The separate `png_source` adapter discovers `<root>/<model>/<electrode>/<number>-<layer>.png`
and `<root>/<model>/cno/<number>_<layer>.png`. The first separator after a positive
integer separates the provenance number from the exact layer suffix. **Every number
is provenance only**, for both electrodes and cno; it never supplies ordering, identity
or a conflict rule. Do not renumber files. The original filename and relative path stay
in evidence. Different names may reuse a number; the same layer in different electrode
folders may have different numbers. Duplicate electrode × layer inside one folder is
an error naming both files. Same-name cno files with different decoded full masks stay
separate assets/components; only the global CNT input channel unions them. Identical
same-name full masks are accepted in image_only with warning `duplicate_cnt_mask_same_name`
naming all files: one content asset keeps every decoded file in `provenance_files`
(relative_path, original_filename, filename_number, file_sha256, decoded_mode).
No filename number/path enters identity. Components describe that one geometry, not
failed contacts; cnt_file_count still counts all observed files. In relations the
`ambiguous_cnt_asset_identity` error stays, with no merging or number/path tie-breaking. `cno` is reserved, case-sensitive folder spelling; its layer names
are not limited to OCNT/GCNT/PCNT. Without an order file, source layer case variants warn with `layer_name_case_variants`
per model and at inventory level across models, preserving separate exact-spelling
vocabulary entries. With an order file, per-model casefold collisions still reject.
The .png extension alone is case-insensitive (.PNG, .Png); other filename syntax and
strict decoding are unchanged, including unsupported-depth PNG classification.
Malformed names, shape mismatch, unsupported layout and escaping symlinks still invalidate
source; unrelated files warn. Earlier shared-stack-number statements are superseded.

Without --stack-order, cnt_mode=image_only: order status is not_used, geometry inventory
has no positions/candidates/relations and reason cnt_image_only_no_relations. Export keeps
all decoded CNT foreground in the optional global cnt mask, connections=[], connectivity
unavailable. Missing/empty cno does not require an order file. Inspect grids works for
full-frame elements, valid and global CNT. Metadata and candidate grid remain required
for export; any source-invalid model blocks the whole pool.

With --stack-order, cnt_mode=relations. A strict two-key UTF-8 JSON document has
stack_order_schema_version=dl1_png_stack_order_v1 and one top-to-bottom order array of
mixed electrode/CNT names. The single reference is filtered to each model's present
names (absent entries become ignored_names); every present name is required, including
white-only CNT. Matching is casefold only; source spelling stays in IDs/manifest/vocabulary,
matched_pairs records order spelling -> source spelling, and casefold collisions in
source/reference reject. NFC/NFD are not normalized. No per-model/default overrides,
number comparisons or overlap-derived order exist. No-cno models also validate any
explicit file. Invalid supplied files never fall back to image_only.

The user's reference of 2026-10-13 is [configs/stack-order.json](../configs/stack-order.json).
It is input data for that user's optional workflow, not hard-coded general CNT physics.
Other examples are syntax-only synthetic references. See [USAGE §11](USAGE.md) for strict
JSON/identifier diagnostics, LF/CRLF/BOM behavior, quoted Windows commands and remedies.
Bytes are read once, hashed before UTF-8-sig decoding, and that snapshot supplies parsing.

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

Canonical identities use UTF-8 canonical JSON plus SHA-256 and do not depend on root,
enumeration, filename numbers or order-file bytes. Electrode asset/element binding is
`[model,"electrode",electrode,layer]`; CNT asset binding is
`[model,"cno",null,CNT_name,decoded_full_mask_sha256]`. Component payload is
`[model,CNT_name,cnt_binding,shape_yx,bbox_xyxy_px,local_mask_sha256]`; connection payload
is `[component_id,upper_element_id,lower_element_id,original_xy_px]` under its namespace.
Model source identity includes the new bindings and mask hashes in a number-free sort.
The semantic resolved-order digest hashes `[[kind,source_name],...]`; neither positions'
file numbers nor raw byte hash enter that digest. Geometry identities survive prefix
renaming/re-encoding, while provenance changes; a pixel change changes the content-bound
CNT identity. Product input identities still come from unchanged data.input_identity:
emitted relations/coverage change them, so modes must be consistent within checkpoint/search.
No order or cnt_mode field is injected into the strict manifest. If no relations are emitted,
strict numerical inputs can coincide across modes; this does not remove the mode policy.

Metadata is a strict JSON object with exactly `metadata_schema_version` and `models`:

```json
{"metadata_schema_version":"dl1_png_metadata_v1","models":{"MODEL":{"lineage_id":"USER_FAMILY","design_version":"USER_VERSION"}}}
```

Each mapping entry contains exactly the two nonempty trimmed strings shown. Duplicate
keys, unknown discovered-model names, wrong versions/types, extra fields and invalid
strings fail. Inspection permits partial metadata and explicitly lists pending models;
export requires all discovered models mapped. Nothing is inferred from folder names.

Current sidecar/algorithm versions are `dl1_png_import_v3`, `dl1_png_bundle_v3`, and
nearest_overlap_common_intersection / `"2"`. Historical T1 v1 and R-B v2 sidecars are
superseded, not supported by current readers. cnt_mode appears at top/model/extraction
levels and in completed CLI events/previews. Raw extractor_run_status is not_run_image_only
for valid image-only inventory; relations uses ran, blocked, or aggregate partial.
This is distinct from connectivity_observation and coverage.
New image_only summaries record geometry_only_component_count equal to component_count
and unresolved_component_count=0; the components keep cnt_image_only_no_relations as
geometry evidence. Relations summaries are unchanged. Readers validate these counts,
accept earlier v3 image_only summaries where unresolved_component_count held the
geometry count, and validate grouped duplicate-file provenance. Versions stay unchanged.

Top stack_order stores schema/source/raw file_sha256/validated document/errors, never
an absolute source path. Each model stores status (not_used/resolved/invalid), source,
selected_list, resolved_order, ignored_names, matched_pairs, semantic digest and errors.
Resolved entries have order_index/kind/name/asset_ids only, covering all distinct same-name
assets. Asset filename_number is provenance and order_index is null without resolution.
Component cnt_order_index is null in image_only, cnt_filename_number is provenance;
equal-number candidate fields and numeric stack tables are removed. Shared preview and
bundle readers validate mode, algorithm, filtering, matched pairs, contiguous positions,
coverage, digest, and no relation/endpoint/position on inactive modes; resealing corrupt
evidence does not bypass these checks. Image-only manifest relations are rejected too.

Resource estimates begin unknown with reasons. Image-only uses R=0, no per-relation mask,
all_components_resolved=null and all_components_resolved_applicability=not_applicable_image_only.
Relations retains its all-components-resolved cost scenario. Invalid source costs remain
unknown; these are not peak RAM/runtime predictions. Electrical/real-data verification
remain not_performed. User accepted 4-neighbour; skipped-layer physics remains unconfirmed.

CLI: `python -m psdl1.import_png --root ROOT --out NEW_OUTPUT` is image-only inspection;
`--inspect-grids G [G ...]` adds explicit grid evidence; `--export --metadata FILE
--candidate-grid G` exports image-only; --stack-order FILE optionally enables relations.
`--verify-bundle BUNDLE` is exclusive of all input options and needs no source/order file.
No grid, metadata, label, split or alias is inferred. Decoder finite-limit override and
publication safeguards remain unchanged; existing schema/model-input/NPY/lock versions
are unchanged.

## Optional original-resolution CNT relations — current v3

`png_connections.extract_model` operates without lineage/design metadata or grids.
It labels each CNT source plane independently using horizontal-run union/find with
4-neighbour edge overlap. The single int32 source label array is released after each
file; bbox/pixel counts derive from runs, without a full-frame scan per component.
Each component retains a bbox-local bool mask. Candidate intersections, common-contact
checks, geometry hashing and centroid selection receive only bbox slices. Large or
widely spanning bboxes may still be expensive; this is no universal sublinear guarantee.

Only relations mode evaluates overlapping electrode candidates using resolved order_index.
Select the maximal upper index and minimal lower index **before** common-intersection
checks. Multiple candidates on a selected layer, missing side, same element, empty or
disconnected common intersection leave it unresolved. A failed nearest pair never falls
through to a deeper pair. Other lower overlaps retain deeper_than_nearest_overlapping_lower.
There is no equal-position electrode/CNT branch; a shared name across kinds is rejected.

Skipped upper/lower evidence groups actual order planes separately as electrode/cno,
with all asset IDs and source paths. Same-name CNT files count as one skipped plane;
absent reference names create no position or skipped entry. Numeric gaps have no meaning.
needs_confirmation exposes skipped-layer interpretation, which remains physically
unconfirmed; multi-electrode ambiguity resolution remains open. 4-neighbour components
were accepted by the user. Connected common intersection still cannot prove one physical
contact, while disconnected common regions are never heuristically split.

**Known gap:** the user confirmed that one CNT segment joins layers of the same electrode,
but relations mode still records same_electrode_required=false and may emit cross-electrode
pairs. It has not been reviewed against that confirmed rule; do not rely on it for physical
connectivity before implementation/review closes this gap. Image-only does not infer pairs.

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
and its M-fold overhead is included in total projected NPY bytes. In relations only, the upper scenario replaces R by K. In image_only R=0 and the upper scenario is not applicable. Invalid source estimates remain unknown with a reason. Estimates exclude
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

## R-A strict decoder and cost corrections (FA)

The independent R-A review of T1+T2 identified malformed streams that Pillow 12.3.0
accepted, sometimes fabricating black foreground. The decoder now verifies PNG structure
independently of Pillow before trusting pixels: signature, IHDR-first/unique, legal
bit-depth/colour combinations, compression/filter methods 0, interlace 0 or Adam7,
PLTE-before-IDAT where applicable, contiguous IDAT, critical chunk order, chunk CRCs,
IEND-last and no trailing bytes. Unknown critical chunks fail. Concatenated IDAT is
stream-inflated in bounded blocks; its output must have **exactly** the IHDR scanline
byte count, including each nonempty Adam7 pass's filter bytes and packed row sizes.
Short, long, incomplete or extra zlib streams and invalid scanline filter types fail
with per-file paths. The check retains bounded blocks and row buffers, not several
full-file or full-inflated-image copies. Pillow's pixel guard is checked before
inflation and remains finite under the previously documented explicit override policy.

Palette PNG requires PLTE and every actual pixel index must reference one of its entries;
packed samples are checked in their original bit depth after exact PNG unfiltering.
Padding bits at the end of a packed row are not pixels. Every non-palette tRNS chunk
(including grayscale and RGB) is unsupported and rejected, even if its sample is unused.
Palette tRNS is accepted only when **every supplied alpha is 255**, including unused
entries; missing alpha entries imply 255. This intentionally conservative rule avoids
Pillow's 2/4-bit grayscale transparency scaling ambiguity. Ordinary explicit-alpha
LA/RGBA still requires every decoded pixel to be fully opaque. No valid opaque black/white
mask geometry, identity or source file is changed.

An empty electrode folder now gives a warning instead of disappearing silently.
Case-insensitive variants of the reserved CNT directory, such as `CNO` or `Cno`, are
errors naming exact `cno`; no case alias is created. Every directory directly under
source root is treated as a model, so non-model directories such as `__MACOSX`, `.hidden`
or `.ipynb_checkpoints` invalidate the inventory if they do not meet the source layout.
Unrelated root files remain warnings. Names are compared byte-exactly: NFC and NFD
spellings differ and no Unicode normalisation is applied. Non-NFC model/electrode/layer
names produce path-specific warnings while their original names/identities are preserved;
a move between filesystems may therefore change identities if names are rewritten.

S23 now instruments `component_slice` itself: 302 comparisons must receive bbox-sized
label windows sharing the source label array's memory, before any `==` allocation.
Bypassing that helper for a full-frame `labels == label` fails the counter, and handing
the helper a full-frame array fails its shape check. The diagonal component test includes
both mirrored orientations. Each component's local binary mask is hashed **once** and
its hash is reused in the existing canonical component ID; the resulting bytes/IDs are
unchanged. S23's updated hash count is 603 (302 component hashes plus 301 common-contact
hashes); the original T2 measurement/counts above are historical evidence.

Representative-pixel selection uses NumPy int64 arithmetic only when Python-integer
bounds prove that each numerator and the sum of squared terms fit in int64. Otherwise
it uses exact Python integers over coordinate arrays without allocating a per-pixel
Python tuple list. Both paths preserve y/x tie breaking and are compared on 300 seeded
random masks against an independent `fractions.Fraction` centroid-distance reference.
Extraction semantics and the 2-second S23 test-only guard are unchanged.

FA validation evidence: all five short-stream B1 reproductions (63/32/1/0 of 64 one-bit
rows and 3 of 4 eight-bit rows) reject; B2 white/black tRNS at depths 1/2/4/8 reject;
B3 missing PLTE and undefined index 3 at depths 2/4/8 reject. All eight N5 cases, including
trailing/concatenated PNG bytes, missing CRC/IEND, wrong IDAT CRC, invalid compression/
filter/interlace methods and overlong inflated streams, reject. Valid opaque fixtures,
exact Adam7 passes, palette PNG filters 0–4 and split IDAT streams pass.

Both requested mutation checks ran only in scratch package copies with verified import
paths (`/tmp/dl1-ra-fa-mutants.vkvl_2b9`), without modifying repository files:
`local = (labels == box.label)[window]` made S23 fail (0 rather than 302 slice calls),
and `previous[k][0] <= x1` made the mirrored diagonal test fail (one component instead
of two); each isolated pytest invocation exited 1 with exactly one failed test.
An archived source snapshot of `105acea9c475678cc160b85f398d68928729f90d` and the fixed
candidate separately decoded/extracted the same synthetic ring plus unresolved component;
asset/frame/component/relation/endpoint/mask hashes and Product input identity matched
exactly (`/tmp/dl1-ra-fa-identity.qy9oc0i9`). Final S23 measurement after these changes was
0.043456 seconds on the same documented machine/environment, under the unchanged 2-second
guard. Temporary scratch outputs are review evidence, not committed input assets.

FA acceptance commands: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p
no:cacheprovider tests/test_png_source.py tests/test_png_connections.py
tests/test_png_scale.py` → **155 passed**; `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python
-m pytest -p no:cacheprovider -p no:warnings` → **350 passed, 1 existing skip**.
The unchanged default suite remains offline; no package installation was performed.

## R-A2 APNG, arithmetic and validation corrections (FA2)

Intentional strictness may reject real files: **any tRNS on grayscale/RGB**, any non-255
palette alpha (even unused entries), bytes after IEND, a bad CRC on any chunk (including
ancillary chunks), any APNG `acTL`/`fcTL`/`fdAT` chunk, and non-identity orientation
metadata. The remedy is to re-save the source as a **plain opaque black/white PNG**.
The adapter never silently strips these features, thresholds pixels or rewrites sources.

The structure validator now runs before `Image.open`, `getexif` or `load`; PNG `getexif`
can trigger decoding, so the previous ordering was too late. The finite Pillow pixel
threshold is checked from validated IHDR dimensions before IDAT inflation using the
same warning/error thresholds and exception classes; no decoder pixel guard is disabled.
Every APNG chunk produces the validator reason `APNG is unsupported`, including valid
single-frame APNG and malformed frame regions. This prevents Pillow from interpreting
only an fcTL subregion and fabricating black pixels in the rest of an otherwise valid
IHDR/IDAT frame. B1–B3, N5 and critical-order tests now assert the validator's actual
reason field; the missing-PLTE test uses a neutral filename and exact reason.

Palette validation skips sample reconstruction/range scanning when PLTE contains every
possible index (`2**bit_depth` entries), since no encoded index can be undefined. Smaller
palettes use NumPy extraction on unfiltered packed row bytes, respecting MSB-first order
and excluding padding bits. Tests cover non-symmetric depth-1/2/4 bytes, set padding bits,
Paeth ties where left and upper-left differ, and all seven nonempty Adam7 passes at 9×9
and 13×11 for every supported colour type/bit depth. Ordinary Pillow-written opaque
1/L/RGB/LA/RGBA/P files, text/EXIF/physical-resolution ancillary chunks, multiple IDAT and
non-APNG interlaced fixtures remain accepted with exact expected masks.

Representative distances now centre coordinates on the integer centroid mean. For
count n, integer mean q and remainder r, compare `n*(dy²+dx²)-2*ry*dy-2*rx*dx`; omitted
constant remainder squares cannot affect ordering. A Python-integer bound on the sum
of absolute terms proves all int64 intermediates and scores safe before NumPy runs.
The exact fallback converts coordinates into separate integer lists (no per-pixel tuple
list), compares the same scores with Python integers and retains y/x ties. The Fraction
reference comparison still covers both paths; an injected sparse coordinate case places
the proof bound just 1,505,827,713 above int64 maximum and forbids the unsafe NumPy path.
A separate hand-specified component fixture ties its stored hash/ID to its actual
nonrectangular mask; complementing or altering that mask must change the hash and ID.

Before/after timings on the same Mac16,11 / Apple M4 Pro, macOS 26.5.2 arm64,
Python 3.12.13 / NumPy 2.5.3 / Pillow 12.3.0:

| Operation / synthetic fixture | Before FA2 | After FA2 |
| --- | ---: | ---: |
| Validator only, 6000×6000 1-bit indexed PNG with full two-entry palette | 3.376316 s | 0.003168 s |
| Representative pixel, full 2000×2000 intersection | 1.716412 s | 0.037479 s |
| Forced exact fallback, same intersection | — | 0.588391 s |
| Archived 105acea representative implementation, same intersection | 1.223155 s | — |

All representative results were exactly `[999,999]`. These are one-run synthetic
measurements, not real resource acceptance limits; S23's separate 2-second guard is
unchanged. Timing inputs/results live in `/tmp/dl1-ra2-timing.k5pf1d1e`.

Six requested mutants ran solely in verified scratch package imports at
`/tmp/dl1-ra2-fa2-mutants.kcyzcoir`: Adam7 pass-2 column 4→5 caused 5 failures;
reversed packed shifts caused 3; swapped Paeth ties, removed PLTE requirement,
hashing `~local`, and int64 proof replaced by `True` each caused 1. Every mutant pytest
invocation exited 1; repository files were never mutated for these checks. The APNG
matrix includes smaller/offset fcTL regions, full-size single-frame APNG, zero-frame
acTL with fcTL/fdAT, duplicate acTL and acTL after IDAT, plus isolated fcTL/fdAT;
each rejects with the exact APNG reason before any Pillow open.

FA2 acceptance: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p
no:cacheprovider tests/test_png_source.py tests/test_png_connections.py
tests/test_png_scale.py` → **212 passed**; `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python
-m pytest -p no:cacheprovider -p no:warnings` → **407 passed, 1 existing skip**.
The mixed resolved/unresolved fixture still matches the archived 105acea asset/frame/
component/relation/endpoint/mask and Product input identities byte-for-byte; evidence
is `fa2-identity.json` under the earlier FA identity scratch directory.

## Inspect/export bundles (T3)

`python -m psdl1.import_png` now implements the frozen inspect/export/verify CLI.
Inspection needs no metadata or grid, writes `import-report.json` plus `preview.html`
for valid models, and lists all source warnings/errors, exact per-model/union/unique
vocabulary, duplicate geometry and resource estimates. These records are saved and
surfaced before full-frame grid/Product allocations or export writes. Explicit
`--inspect-grids` only compares supplied candidates; there is no automatic grid choice.
Rounded transform records match the existing full-frame recipe exactly.

Export requires complete explicit metadata and a candidate grid. One invalid model blocks
the entire export pool; missing/invalid metadata/grid yields a diagnostic-only published
bundle and nonzero exit. Valid unresolved CNT components remain in the global CNT union.
Every generated NPY is bool, and full frame extent goes into the unchanged strict v1
manifest's compatibility size field. Connection masks are materialized/written one at
a time. The shared strict manifest iterator validates one staged float32 Product at a
time and collects scalar input identities/grid evidence. It must be exhausted before
manifest promotion: late duplicate IDs, pitch differences and bad final assets still
fail publication. Public `data.load_manifest` remains an eager list API for training.

Inventory discovery/extraction streams one model, attaches its exact resource estimate,
then privately detaches every source and bbox-component mask (`mask=None`, never empty
observed data). No arrays enter retained reports. The shared scanner preserves global
identity claims, exact role/layer vocabulary, notices and CNT provenance/merge rules.
After the complete initial resource report, a verified per-model reload supports grids
and preview; a separate verified reload supports export only after all global gates pass.
Every recorded file, including all merged CNT provenance files, is checked using the
single immutable `_decode` snapshot's file hash, mode, shape and geometry hash; replayed
extraction/resource evidence must equal pass one. Source changes abort publication;
new unrelated files are not added after the discovery snapshot. Metadata/order snapshots
are not reread. Skip reloads when they have no work. Repeated decoding/extraction costs
CPU/I/O in return for bounded cross-model raster retention.

Grid-independent source width, 4-neighbour components and active pixels are computed
once per logical observed mask on `binary_mask(mask)*valid` within one Product evaluation.
Only scalars are cached, with no mutable cross-call cache; missing masks remain missing.
Area resize, frame mapping, target topology/counts and warning order remain unchanged.
Completed grid records are yielded incrementally so a later failure preserves earlier evidence.
Preview renders one model into scalar rows/HTML strings; final HTML uses all final warnings.

`--no-preview` omits preview HTML/rasters only, sets `preview=[]`, and adds exactly one
`preview_skipped_by_request: --no-preview` diagnostic after grid diagnostics. Verification
rejects inconsistent skipped-preview evidence and unknown report fields. No field or version
is added: report/bundle remain v3 and algorithm remains version 2; existing v3 bundles work.
CLI progress is flushed stderr JSON lines (`event=progress`, phase/stage/state, model/grid/file
counts when known), separate from unchanged stdout/event_sink resource_inventory/completed. Inventory reading emits started for every model, even when no files can be decoded.
In inventory progress, files_done/files_total count completed decode attempts: contained
non-directory PNG paths in supported immediate electrode/CNT folders whose filename passed
parse_png_name, including attempts rejected by the decoder. Unsupported layouts, escaping/
broken paths, non-PNGs and malformed PNG filenames are not counted. files_done is cumulative
per model; inventory files_total is known/reported only at reading completion. Reload
progress counts every recorded provenance file (including each merged CNT file), with
files_total known before reading starts and files_done incremented after each verified
snapshot; a failed snapshot aborts before its completion increment.
Library `run_import` accepts a keyword-only `progress_sink` and otherwise stays quiet.

The bound is the largest model, not fixed RAM or a per-file guarantee. Source/component
arrays, relation masks and a staged float32 Product can still be large: 900 masks at
1500×1500×4 bytes require about 8.1 GB for ONE Product before validation temporaries.
Report/manifest/HTML metadata scales with model count, and public/training load_manifest
still holds all Products. The weakref regression measures unique retained array backing
bytes at source/extraction/Product/loader boundaries, not total RSS or decoder transient
buffers. The opt-in `scripts/benchmark_png_import.py --models 2 8 --modes image_only relations
--preview both --operation both` separately reports tracemalloc-instrumented time/peak and
process-lifetime peak RSS; synthetic results do not prove production-scale success.

Output must be new and outside the source tree. A sibling `.staging-*` receives the
report, local previews and assets; a pending manifest becomes `manifest.json` only after
staged validation. Native atomic no-replace directory rename publishes the complete
bundle (Darwin RENAME_EXCL/Linux RENAME_NOREPLACE/Windows refusing existing destinations).
Even a concurrently created empty destination is refused. Failures preserve staging
and report its location. Retry with a fresh `--out`; the importer never resumes/reuses
a failed stage or removes/replaces an existing destination.

`bundle-integrity.json` (current `dl1_png_bundle_v3`; historical T3 v1 is superseded) binds hashes of report/manifest/NPY/HTML/raster
files; the report also binds manifest/assets and records Product input identities.
`--verify-bundle` validates contained paths, file hashes, supported versions, exact listed
files, manifest/report/asset links and bool mask headers without source PNGs or a full
legacy load. It is corruption detection, not signed authenticity or electrical truth.
A copied bundle remains verifiable offline. Strict sidecar fields stay outside the core
manifest; the existing loader continues rejecting them if injected into it.

The coordinator-authorized R-A3 C1 fix captures each PNG's compressed bytes once, then
validates, decodes via BytesIO and hashes that immutable snapshot. Replacing a source
file after validation cannot change its decoded geometry/provenance hash. Resource notes
now separately disclose observed transient decoder cost (~17–18 bytes/pixel), compressed
snapshot storage and a largest-model raster bound in the streaming importer. Eager
compatibility discovery/extraction retains all requested arrays. These notes do not
bound peak RAM or approve real resources. Duplicate mask/model geometry produces explicit
possible-alias/duplicate warnings; no source name or product is merged or silently omitted.
Actual runtime Pillow version is recorded and a different version from pinned/tested
12.3.0 is warned, without runtime replacement. Approved wheelhouse/build constraints and
intentional PNG strictness/re-save remedy appear in USAGE.

`status_summary` explicitly distinguishes extractor run (`ran/blocked`), extraction
coverage (`complete/partial/unavailable`), model connectivity status (`available/partial/
unavailable`) and unevaluated checkpoint vocabulary eligibility. Legacy per-model
`extraction_status: complete` means the extractor ran, never coverage or certification.
Electrical and real-data validation stay `not_performed`. This status distinction and
all warnings appear in preview as well as report.

Pillow raster previews provide a downscaled **full-frame** overview with actual XY display
scale, local toggleable source layers, a reference-only core rectangle and resolved/
unresolved/needs-confirmation markers. Every component has a bbox-local original-resolution
image with original offset and candidate/skipped-layer evidence; these display crops never
enter model input. CSS magnification and browser scrolling are display aids only. Source
names are HTML-escaped; images and scripts are bundle-local, with no CDN/fetch/font assets.

T3 validation: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider
 tests/test_png_import.py tests/test_png_preview.py tests/test_png_source.py
 tests/test_png_connections.py` → **233 passed**; full default suite
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -p no:warnings`
→ **430 passed, 1 existing skip**. The real importer CLI exported >1000-pixel-axis synthetic
frames; unchanged `load_manifest` and `python -m psdl1.data MANIFEST --report REPORT` passed.
Every generated mask was bool and independently hand-specified payload/header bytes matched.
Tests also covered publication interruption, existing/empty destination refusal, metadata
pool completeness, source relocation, hash tampering, explicit optional grids and failed
grid diagnostics, source warning/runtime-version display and immutable compressed snapshots.

Synthetic manual preview evidence: `/tmp/dl1-png-preview-evidence/t3-inspection/preview.html`,
with sealed `import-report.json`, `bundle-integrity.json` and Pillow rasters beside it.
The frame is 1401×1603 and contains margin/direct resolved, skipped-layer needs-confirmation
and ambiguous components. Full-frame marker/core placement and magnified source crops were
visually inspected; original offsets/polarity/endpoint centres are correct. The montage
`/tmp/dl1-png-preview-evidence/t3-visual-check.png` is outside the sealed bundle and solely
inspection evidence. `--verify-bundle` passed for the preview bundle. C1/C3 decoder/test
file changes were explicitly authorized by the coordinator's T3 scope addition; no core
schema/model/NPY-loader/lock change or installation occurred.


## Shared full-frame transform and T4 integration — 2026-10-12

`data.frame_transform` centralizes the existing rounded area resize and central pad:
`nh=round(H*G/max(H,W))`, `nw=round(W*G/max(H,W))`, each at least 1;
`left=(G-nw)//2`, `top=(G-nh)//2`, `sx=nw/W`, `sy=nh/H`.
The full original frame, including real margins, feeds self/rest/global CNT/valid and
connection-mask channels with this single transform. Rounding can make `sx != sy`.
Pixel index `(x,y)` maps continuously to the grid edge coordinate
`((x+0.5)*sx+left,(y+0.5)*sy+top)`; its inverse subtracts the pad, divides by
actual XY scale, then subtracts 0.5. This round trip concerns coordinates, never
reversible image downsampling. Synthetic float64 coordinate round trips use absolute
tolerance 2e-13; independent float32 feature comparisons use absolute tolerance 1e-7
(and baseline compatibility compares exact bytes). Physical endpoints first subtract original edge origin
and divide by original pitch, then apply rounded content dimensions/pad and divide by G.
The legacy multiply/divide operation order is retained for bit-identical float32 model
features. `MODEL_INPUT_VERSION` and `SCHEMA_VERSION` remain unchanged.

Every validator/grid/inspect/export report exposes the same transform record: source
and rounded shapes, padding LTRB, actual XY scale and effective content pitch
`[0.1/sx,0.1/sy]` µm for PNGs. Padding has no physical extent. For H=1001,W=1203,G=32,
content is 27×32 and padding is `[0,2,0,3]`; no nominal-only scale or core crop is used.
Candidate G values must be explicit; grid checks retain their heuristic gates and never
certify connectivity. Existing checkpoint inference remains compatible with baseline
03bf3e33c7107af75297aa298fa323e24018f28d, while changed-code resume is rejected before
optimizer/RNG restoration. This compatibility evidence is synthetic and specific to the
unchanged schema/model-input contract, not a real-data guarantee.

Preview context views add a 64-source-pixel display margin around each CNT bbox,
clamped to the original frame. Reports preserve the tight bbox crop plus context bbox,
source offset/shape and a separate integer nearest-neighbour display scale. Native
context images keep one source pixel per image pixel. Selected upper electrodes are
blue, selected lower rose, others gray; black outlines CNT, yellow shows J, and a
visible green crosshair marks the shared endpoint. Unresolved rows show every candidate
on each selected layer and its reason. Top-to-bottom layer text marks skipped electrode
and cno layers separately. These windows/upscales never become model input. The offline
HTML retains full-frame overview, margin components, core reference and confirmation
status; fixed semantic colours do not depend on asset enumeration.

For closed-network use, supply approved local Pillow and build wheels and inspect
resource estimates before allocating a bundle. PNG pitch is 0.1 µm/pixel and exported
`unit_size_um` is full frame extent, not an inferred unit. Exact vocabulary names are
required by each checkpoint; no alias mapping exists. Complete is extraction coverage
only and electrical verification is not performed, although coverage feeds the existing
connectivity embedding. Real grid/linewidth, real writer/source compatibility, labels,
G6 quality thresholds and resource approval remain pending. The 5–10 diverse product
pilot is solely for integration/connection inspection.

T4 validation on 2026-10-12: required targeted integration/compatibility command →
**46 passed**, focused updated grid/preview/inspect-transform checks → **7 passed**,
full default suite → **446 passed, 1 existing skip**. Exact commands, fixture recipe,
baseline checkpoint/tensor evidence and pending review limits are preserved as a new
entry in [validation.md](validation.md). Synthetic offline context preview is
`/tmp/dl1-png-preview-evidence/t4-inspection/preview.html`; its 23-file bundle hash
verification passed. The montage beside the bundle was visually checked. No real-data,
electrical verification or retrieval-quality claim follows.


## Historical R-B response: publication and superseded sidecar v2 — 2026-10-12

The publication/containment fixes below remain in force. Version/run-field descriptions
are historical; current v3 mode/order/identity rules above supersede them. Historical
validation counts apply only to their original candidate, not the present code.

Containment now compares `os.path.samefile` for every existing output ancestor against
the input root, and rejects the reverse nesting too, before any directory creation.
This closes the case/NFC/NFD path-spelling bypass on default APFS. Source/argument
validation happens before making missing parent/staging directories where a diagnostic
bundle is not yet needed. Invalid inventory/metadata still gets the authorized diagnostic
bundle. No source files or model geometry are changed.

Native atomic no-replace directory publication remains on macOS/Linux when available.
A missing symbol or ENOSYS/EINVAL selects the exclusive-directory-claim fallback (also
used on Windows). The fallback refuses every existing destination, copies exclusively,
writes manifest last, verifies the complete bundle, and disables the manifest on failure
or KeyboardInterrupt. Unlike native rename it does not hide the partially copied
directory. Consumers must wait for completion and verify the bundle. Failed claimed
outputs and stages remain evidence; no existing destination is deleted or overwritten.
The actual method is sealed into `publication_method`. Only macOS APFS was exercised on
a real host; native Linux and Windows publication are not verified on real hosts.

After any failure/interruption in the staged build, `manifest.json` is renamed back to
pending evidence and the report says `failed/interrupted`. A distinct failure-report
writing temporary avoids collisions with an abandoned `.writing` file and preserves the
original exception. The handler also catches KeyboardInterrupt. Staging never becomes
a consumable model bundle merely because validation, seal, verify or publication failed.

At that historical candidate report shape changes were **dl1_png_import_v2** (superseded by v3). The per-model
and top-level raw run field is `extractor_run_status` with `ran/blocked/not_run`, never
`complete` for a run. Coverage `complete` remains extraction coverage only, not electrical
verification. Report vocabulary separates `electrode_layers` from `cno_layers`, their
unions and unique-to-one-model lists; only electrode layers enter the model vocabulary.
Combined names remain clearly labelled `all_source_*` audit fields. Internal extraction
records/semantics and emitted IDs/hashes are unchanged. Invalid metadata report diagnostics
use the location-independent `metadata` label and omit absolute file paths.

At that historical candidate integrity was **dl1_png_bundle_v2** (superseded by v3) and lists exact directories as well as files. Added
empty/nested directories and unlisted/non-file/missing entries reject. Missing/malformed
report or manifest fields produce a clean diagnostic. V1 sidecar/integrity versions are
explicitly unsupported by the new verifier; re-import to a new destination if needed.
This does not bump `dl1_shape_v1_draft`, model input or checkpoint versions or change NPY
load/preparation arithmetic. Consistent malicious re-sealing is still outside corruption
verification's claim. Export with grid warnings remains allowed by plan: exported does
not mean the grid is accepted; training/search retain their strict default gates.

R-B fix validation: required targeted command **249 passed (65.74 s)**, final publication
follow-up **14 passed (6.34 s)**, full default suite **484 passed, 1 existing skip
(76.00 s)**. All five case/NFC/NFD containment reproductions ran on APFS with zero entries
created. All eleven required scratch mutants failed their specified tests; exact commands,
rows, fixture coordinates and evidence path are recorded in the new dated
[validation record](validation.md). Independent refreshed review remains pending.


## PNG v3 migration

Re-import original PNGs with explicit metadata/grid into a **new output directory**, using
image_only by default or an actual reference file for optional relations. Verify and inspect
mode/order/geometry; never relabel old v1/v2 sidecars or reuse number-based endpoints. Retain
old artifacts as historical evidence, rebuild identity-bound caches/embeddings, and keep one
CNT mode consistently in checkpoint/search. Existing code/identity resume gates stay active.
Current verifier rejects old bundle/report versions, algorithm 1, and malformed v3 evidence
with the documented explicit diagnostics. Generic detached v1 NPY manifests and previously
rendered static HTML still open; they are not retroactively blocked or certified by PNG readers.
All compatibility/quality evidence is synthetic, not Windows-host, real-data, electrical or
retrieval-performance validation. The strict manifest and core numerical versions are unchanged.
