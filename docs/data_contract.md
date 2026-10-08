# DL1 original-shape contract — `dl1_shape_v1_draft`

This is the T8 internal draft, validated with synthetic masks. G3 repository names and G5 whole-unit isotropic resize + pad, electrode × actual-layer set aggregation, real linewidth/grid choices, and the source CNT/VIA endpoint metadata format are **pending approval**. The three-channel set input is an implementation candidate. Schema acceptance does not resolve those gates, establish real labels, or prove a CNN can distinguish connectivity.

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

The current compatibility images have channels `[self, CNT/VIA union, rest]`. For element i, `rest_i = clip(sum(normalized_element_j for j != i), 0, 1)`. At an XY pixel occupied by two different elements, both rest values are exactly 1 at an unchanged grid. Excluding self **before** composition prevents overlap erasure. Rest loses the layer identity of its contributors; individual elements and their role-layer bindings remain separate. The legacy three-channel model does not consume the structured valid/endpoints yet; that work belongs to T9.

`input_identity` fingerprints original masks, coverage, frame, unit, each bound element tuple, and source endpoint relations independent of element/connection ordering. It is an input fingerprint, not a learned embedding or a lineage split key. A product with bindings A/L1,B/L2 differs from A/L2,B/L1 even with identical XY masks and equal role and layer sets. Element order permutes candidate rows together with their bound IDs and does not change this identity.

## Grid and transform evidence

The validator CLI writes a durable JSON report with the version, pending approval, and one grid check per product:

```bash
.venv/bin/python -m psdl1.data manifest.json --grid 256 --report grid-report.json
```

Each mask's evidence includes source/target foreground pixel counts, 4-neighbour component counts at a stated target coverage threshold of 0.5, the shortest horizontal/vertical active run, its scaled estimate, and a bridge/component loss flag. Empty foreground has run 0. Axis runs below about 2 candidate pixels produce a warning; this is an auxiliary heuristic, sensitive to corners/diagonals and unrelated to a proven physical minimum linewidth. Real minimum-linewidth acceptance remains G5 pending.

Changing a component count flags bridge splitting, complete erasure, or unintended merging. The checks are conservative diagnostics: equal component counts may still conceal changed paths, endpoints, or topology. Therefore `connectivity_preserving` is always false (no certification), including warning-free cases. The report does not use area similarity to establish connectivity. It records missing observations and valid holes separately.

`augment_with_check` applies one affine transform to every element and channel, returns component evidence and the common theta, and always leaves `safe_for_positive_label: false`. A deliberately removed 1-pixel bridge in the tests changes only `8/384 ≈ 2.08%` of image area yet splits one component into two. Shrink and common-crop tests both report that loss with hand-written expected counts. Endpoint/coverage-aware augmentation approval and positive-label gating must be implemented by T9; the legacy training consumer does not enforce these reports automatically.

## Validation evidence and limits

Synthetic tests compare the same shape at pixel sizes 40,80,120,160 (pitch 0.05 µm, differing unit dimensions) using grids 16,32,64. Mean absolute normalized-mask coverage error must be below 0.04. Exact unchanged-grid overlap values and role-layer ID arrays provide independent hand-written oracles. Tests also exercise valid holes, source endpoints, missing assets, input identity, units, frame extents, and path rejection.

The default full suite excludes the existing 40-epoch T9 training/search test unless explicitly enabled with `--run-training`; no training, pretrained weights, or network are required for this contract. No model.py, losses.py, train.py, or search.py changes are included. Real electrical/connectivity labels, held-out search quality, the source metadata adapter, and final G5 policy acceptance remain pending.

## T8 execution record — 2026-10-09

Baseline: `rewrite/shape-connectivity-v1`, commit `44a13a671d80f41ce495a7f1bf2da2cca2e16264`, initially clean. Changes are left uncommitted in the assigned repository. T9 implementation files and `particle-sim-compare` were not modified; no import/signature adjustments to T9 files were needed.

| Command | Observed result |
| --- | --- |
| `.venv/bin/python -m pytest -p no:warnings tests/test_data.py tests/test_input_contract.py` | 90 passed, 0.21 s |
| `.venv/bin/python -m pytest -p no:warnings` | 99 passed, 1 skipped, 1.34 s |

The skipped test is `tests/test_model_and_training.py::test_training_lowers_the_loss_and_search_finds_the_same_layout_at_another_unit_size`, owned by T9 `train.py` and `search.py`; running it would violate T8's no-training scope. All other collected tests passed with socket connections blocked.

The bridge example's stored check uses source shape `[16,24]`, grid 6, scale 0.25, coverage threshold 0.5, and 4-neighbour components. Its source contains 136 active pixels and 1 component; the thresholded normalized mask contains 8 active pixels and 2 components. The shortest axis run is 1 pixel (estimated 0.25 at the chosen grid), and both thin-line and bridge/component-loss warnings are set, with connectivity unavailable and preservation uncertified. This evidence is synthetic and does not approve a real grid. Independent review is still required under the adopted R3 route.
