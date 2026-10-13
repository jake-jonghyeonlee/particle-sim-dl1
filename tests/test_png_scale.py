"""S23 deterministic scale regression: extraction only, never production limits."""
import platform
import time

import numpy as np
import pytest

from png_helpers import write_model, write_stack_order
import psdl1.png_connections as extraction
import psdl1.png_contract as contract
from psdl1.png_source import discover_source

# Test-only guard, selected by coordinator; measured timing recorded in docs.
# It must remain >=10x the measured fixture time; never silently loosen a failure.
SCALE_REGRESSION_SECONDS = 2.0


def test_bbox_extraction_scale_and_resource_estimates(tmp_path, monkeypatch):
    # 300 distinct 2x2 contacts, a 200x200 square and a 3x500 elongated contact.
    # Last SMALL contact lacks upper overlap: 302 components, 301 relations.
    origins = [(100 + 20 * row, 150 + 20 * col) for row in range(15) for col in range(20)]
    small = tuple((y + dy, x + dx) for y, x in origins for dy in range(2) for dx in range(2))
    large = tuple((y, x) for y in range(1000, 1200) for x in range(1000, 1200))
    elongated = tuple((y, x) for y in range(1500, 1503) for x in range(900, 1400))
    pixels = small + large + elongated
    upper_pixels = small[:-4] + large + elongated
    write_model(tmp_path, [("A", "1-U.png", upper_pixels), ("B", "8-L.png", pixels),
                           ("cno", "6_C.png", pixels)], shape=(2000, 2000))
    model, = discover_source(tmp_path, stack_order=write_stack_order(tmp_path.parent / (tmp_path.name + "-order.json"), ["U", "C", "L"])).models
    assert model.valid
    operations = {"slice": 0, "overlap": 0, "joint": 0, "representative": 0, "hash": 0, "full_labels": 0}
    original_overlap = extraction.overlap_pixels
    original_joint = extraction.common_intersection
    original_rep = extraction.representative_pixel
    original_hash = contract.mask_hash
    original_labels = extraction.label_components
    original_slice = extraction.component_slice
    source_labels = []
    def local_only(mask):
        assert mask.shape in {(2, 2), (200, 200), (3, 500)}
    def check_overlap(c, e):
        local_only(c); local_only(e)
        assert any(np.shares_memory(e, a.mask) for a in model.electrodes)
        operations["overlap"] += 1
        return original_overlap(c, e)
    def check_joint(c, u, l):
        local_only(c); local_only(u); local_only(l)
        operations["joint"] += 1
        return original_joint(c, u, l)
    def check_rep(j):
        local_only(j); operations["representative"] += 1
        return original_rep(j)
    def check_hash(m):
        local_only(m); operations["hash"] += 1
        return original_hash(m)
    def check_labels(m):
        if m.shape == (2000, 2000):
            operations["full_labels"] += 1
            assert m is model.cnt_assets[0].mask
        else:
            local_only(m)
        labels, boxes = original_labels(m)
        if m.shape == (2000, 2000):
            source_labels.append(labels)
        return labels, boxes
    def check_slice(label_window, label):
        local_only(label_window)
        assert np.shares_memory(label_window, source_labels[0])
        operations["slice"] += 1
        return original_slice(label_window, label)
    def forbidden_full_mask(c):
        pytest.fail("scale inspection materialized a full-frame connection mask")
    monkeypatch.setattr(extraction, "overlap_pixels", check_overlap)
    monkeypatch.setattr(extraction, "common_intersection", check_joint)
    monkeypatch.setattr(extraction, "representative_pixel", check_rep)
    monkeypatch.setattr(extraction, "mask_hash", check_hash)
    monkeypatch.setattr(contract, "mask_hash", check_hash)
    monkeypatch.setattr(extraction, "label_components", check_labels)
    monkeypatch.setattr(extraction, "component_mask", forbidden_full_mask)
    monkeypatch.setattr(extraction, "component_slice", check_slice)
    started = time.perf_counter()
    result = extraction.extract_model(model)
    elapsed = time.perf_counter() - started
    print(f"S23 measured={elapsed:.6f}s guard={SCALE_REGRESSION_SECONDS}s "
          f"Python={platform.python_version()} system={platform.platform()} machine={platform.machine()} "
          f"NumPy={np.__version__}")
    assert elapsed < SCALE_REGRESSION_SECONDS
    assert result.summary.component_count == 302 and result.summary.resolved_relation_count == 301
    assert result.summary.unresolved_component_count == 1 and result.connectivity_observation == "partial"
    assert operations == {"slice": 302, "overlap": 604, "joint": 301, "representative": 301, "hash": 603, "full_labels": 1}
    expected_origins = origins[:-1] + [(1099, 1099), (1501, 1149)]
    assert [r.representative_xy_px for r in result.relations] == [(x, y) for y, x in expected_origins]
    for r, (y, x) in zip(result.relations, expected_origins):
        assert [e.element_id for e in r.endpoints] == [
            "element_469897965578e2a39abd4379749385548f08d9c49a77a78808f75490cdb2e8aa",
            "element_72a338a8f9aef26c9341f65684b9c7cfad243741ebbd7aeb9cad28b54f01a685"]
        assert r.endpoints[0].xy_um == r.endpoints[1].xy_um == pytest.approx(((x + 0.5) * 0.1, (y + 0.5) * 0.1))
    assert result.components[299].unresolved_reasons == ("missing_upper_overlap",)
    assert all(c.mask.shape == (2, 2) and c.pixel_count == 4 for c in result.components[:300])
    assert [(c.bbox_xyxy_px, c.mask.shape, c.pixel_count) for c in result.components[300:]] == [
        ((1000, 1000, 1200, 1200), (200, 200), 40_000),
        ((900, 1500, 1400, 1503), (3, 500), 1_500)]
    estimate = result.resource_estimate
    assert (estimate.shape_yx, estimate.frame_pixels, estimate.electrode_asset_count,
            estimate.cnt_file_count, estimate.cnt_component_count, estimate.resolved_relation_count,
            estimate.global_cnt_indicator, estimate.generated_mask_asset_count) == ((2000, 2000), 4_000_000, 2, 1, 302, 301, 1, 305)
    assert estimate.bool_payload_bytes == 1_220_000_000
    assert estimate.npy_header_bytes == 39_040  # 305 actual v1 NPY headers of 128 bytes
    assert estimate.projected_bool_npy_export_bytes == 1_220_039_040
    assert estimate.projected_load_manifest_retained_float32_mask_bytes == 4_880_000_000
    scenario = estimate.all_components_resolved
    assert (scenario.resolved_relation_count, scenario.generated_mask_asset_count) == (302, 306)
    assert scenario.bool_payload_bytes == 1_224_000_000
    assert scenario.npy_header_bytes == 39_168
    assert scenario.projected_bool_npy_export_bytes == 1_224_039_168
    assert scenario.projected_load_manifest_retained_float32_mask_bytes == 4_896_000_000
    assert estimate.unknown_reasons == () and estimate.excludes == ("manifest", "report", "preview")
    # Write just ONE bool mask under tmp_path to independently verify real np.save
    # encoding; never export/load the hundreds of full-frame relation masks.
    example = tmp_path / "one-mask.npy"
    np.save(example, model.cnt_assets[0].mask, allow_pickle=False)
    assert example.stat().st_size == 4_000_128


def test_resource_estimate_without_cnt_and_unknown_invalid_source(tmp_path):
    write_model(tmp_path, [("A", "1-U.png", ((10, 20),))])
    model, = discover_source(tmp_path, stack_order=write_stack_order(tmp_path.parent / (tmp_path.name + "-order.json"), ["U", "C", "L"])).models
    result = extraction.extract_model(model)
    e = result.resource_estimate
    assert (e.global_cnt_indicator, e.cnt_file_count, e.cnt_component_count, e.resolved_relation_count) == (0, 0, 0, 0)
    assert e.generated_mask_asset_count == 2
    assert e.bool_payload_bytes == 2_408_406
    assert e.npy_header_bytes == 256
    assert e.projected_bool_npy_export_bytes == 2_408_662
    assert e.projected_load_manifest_retained_float32_mask_bytes == 9_633_624
    assert e.all_components_resolved.projected_bool_npy_export_bytes == 2_408_662
