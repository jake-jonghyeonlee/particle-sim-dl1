"""Original-coordinate connection counterexamples S06-S12; synthetic only."""
from dataclasses import replace
import hashlib
import json
import shutil

import numpy as np
import pytest

from png_helpers import write_model, write_png
from psdl1.data import input_identity
from psdl1.png_connections import (component_mask, extract_inventory, extract_model, iter_connections,
                                   label_components, representative_pixel, to_product)
from psdl1.png_contract import ImportReport, ModelMetadata
from psdl1.png_source import discover_source, load_metadata, report_payload
from psdl1.schema import validate_product


def expected_element(electrode, stack, layer, model="M"):
    # Independent specification: names and stacks below are hand-specified, never
    # obtained from extractor/discovery identities or enumerated candidate results.
    payload = ["element", [model, "electrode", electrode, stack, layer]]
    return "element_" + hashlib.sha256(json.dumps(payload, separators=(",", ":"),
                                                ensure_ascii=False).encode()).hexdigest()


def run_model(tmp_path, assets):
    write_model(tmp_path, assets)
    model, = discover_source(tmp_path).models
    assert model.valid
    return model, extract_model(model)


@pytest.mark.parametrize("joint_at_nearest", [True, False])
def test_nearest_overlapping_lower_only(tmp_path, joint_at_nearest):
    model, result = run_model(tmp_path, [
        ("A", "1-U.png", ((10, 20),)),
        ("N", "7-N.png", ((500, 500),)),
        ("B", "8-L.png", ((10, 20 if joint_at_nearest else 22),)),
        ("D", "10-D.png", ((10, 20),)),
        ("cno", "006_OCNT.png", ((10, 20), (10, 21), (10, 22)))])
    c, = result.components
    assert [(v.electrode, v.stack) for v in c.below_candidates] == [("B", 8), ("D", 10)]
    assert [(v.electrode, v.stack) for v in c.selected_lower_candidates] == [("B", 8)]
    assert [(v.candidate.electrode, v.reason) for v in c.ignored_lower_candidates] == [
        ("D", "deeper_than_nearest_overlapping_lower")]
    if joint_at_nearest:
        relation, = result.relations
        assert [e.element_id for e in relation.endpoints] == [expected_element("A", 1, "U"),
                                                             expected_element("B", 8, "L")]
        assert relation.representative_xy_px == (20, 10)
        assert relation.endpoints[0].xy_um == pytest.approx((2.05, 1.05))
        assert result.connectivity_observation == "complete"
        validate_product(to_product(model, result))
    else:
        assert c.unresolved_reasons == ("empty_common_intersection",)
        assert result.relations == () and result.connectivity_observation == "unavailable"


def test_nearest_upper_and_symmetric_skipped_layers(tmp_path):
    model, result = run_model(tmp_path, [
        ("A", "1-U.png", ((10, 20),)),
        ("F", "2-F.png", ((10, 20),)),  # nearest overlapping upper is 2, not 1
        ("N", "3-N.png", ((700, 700),)),
        ("cno", "4_C4.png", ()),
        ("cno", "6_C6.png", ((10, 20),)),
        ("cno", "7_C7.png", ()),
        ("B", "8-B.png", ((700, 700),)),
        ("B2", "8-B.png", ((700, 701),)),  # one layer entry with two asset references
        ("L", "9-L.png", ((10, 20),))])
    r, = result.relations
    assert [e.element_id for e in r.endpoints] == [expected_element("F", 2, "F"), expected_element("L", 9, "L")]
    assert [(l.stack, l.layer, l.source_paths) for l in r.skipped_upper_layers.electrode_layers] == [
        (3, "N", ("M/N/3-N.png",))]
    assert [(l.stack, l.layer) for l in r.skipped_upper_layers.cno_layers] == [(4, "C4")]
    assert [(l.stack, l.layer, l.source_paths) for l in r.skipped_lower_layers.electrode_layers] == [
        (8, "B", ("M/B/8-B.png", "M/B2/8-B.png"))]
    assert [(l.stack, l.layer) for l in r.skipped_lower_layers.cno_layers] == [(7, "C7")]
    assert r.needs_confirmation and result.connectivity_observation == "complete"
    s = result.summary
    assert (s.needs_confirmation_relations, s.skipped_upper_relations, s.skipped_lower_relations,
            s.skipped_upper_electrode_layers, s.skipped_upper_cno_layers,
            s.skipped_lower_electrode_layers, s.skipped_lower_cno_layers) == (1, 1, 1, 1, 1, 1, 1)
    assert result.electrical_verification == "not_performed"
    assert "implementation_interpretation" in result.rule_evidence
    assert to_product(model, result).connectivity_observation == "complete"


def test_skipped_numeric_gaps_are_not_invented_layers(tmp_path):
    _, result = run_model(tmp_path, [("A", "1-U.png", ((10, 20),)),
                                    ("B", "20-L.png", ((10, 20),)),
                                    ("cno", "6_C.png", ((10, 20),))])
    r, = result.relations
    assert not r.needs_confirmation
    assert not r.skipped_upper_layers.present and not r.skipped_lower_layers.present


@pytest.mark.parametrize("missing", ["upper", "lower", "both", "equal"])
def test_equal_stack_and_missing_side_unresolved(tmp_path, missing):
    items = [("cno", "6_C.png", ((10, 20),))]
    if missing not in {"upper", "both"}: items.append(("A", "1-U.png", ((10, 20),)))
    if missing not in {"lower", "both"}: items.append(("B", "8-L.png", ((10, 20),)))
    if missing == "equal": items.append(("E", "6-C.png", ((10, 20),)))
    if missing == "both": items.append(("N", "1-U.png", ((500, 500),)))
    _, result = run_model(tmp_path, items)
    c, = result.components
    expected = {"upper": ("missing_upper_overlap",), "lower": ("missing_lower_overlap",),
                "both": ("missing_upper_overlap", "missing_lower_overlap"), "equal": ("equal_stack_overlap",)}
    assert c.unresolved_reasons == expected[missing] and not result.relations
    if missing == "equal":
        assert [(v.stack, v.layer) for v in c.equal_stack_candidates] == [(6, "C")]


def test_disjoint_upper_lower_contact_unresolved(tmp_path):
    _, result = run_model(tmp_path, [("A", "1-U.png", ((10, 20),)),
                                    ("B", "8-L.png", ((10, 22),)),
                                    ("cno", "6_C.png", ((10, 20), (10, 21), (10, 22)))])
    c, = result.components
    assert len(c.above_candidates) == len(c.below_candidates) == 1
    assert c.common_intersection_pixels == 0 and c.common_intersection_regions == 0
    assert c.unresolved_reasons == ("empty_common_intersection",) and not result.relations


@pytest.mark.parametrize("side", ["upper", "lower"])
@pytest.mark.parametrize("reverse", [False, True])
def test_same_layer_multiple_candidates_unresolved(tmp_path, side, reverse):
    items = [("A", "1-U.png", ((10, 20), (10, 21))),
             ("B", "8-L.png", ((10, 20), (10, 21))),
             ("cno", "6_C.png", ((10, 20), (10, 21))),
             ("small", "1-U.png" if side == "upper" else "8-L.png", ((10, 21),)),
             ("deeper", "10-D.png", ((10, 20), (10, 21)))]
    model, result = run_model(tmp_path, list(reversed(items)) if reverse else items)
    if reverse: result = extract_model(replace(model, assets=tuple(reversed(model.assets))))
    c, = result.components
    assert c.unresolved_reasons == (f"multiple_{side}_electrodes_on_selected_layer",)
    assert result.relations == ()
    selected = c.selected_upper_candidates if side == "upper" else c.selected_lower_candidates
    assert {v.electrode for v in selected} == ({"A", "small"} if side == "upper" else {"B", "small"})


def test_component_rule_and_multicontact_pending(tmp_path):
    mask = np.array([[True, False, False], [False, True, True], [False, False, False]])
    labels, boxes = label_components(mask)
    assert labels.tolist() == [[1, 0, 0], [0, 2, 2], [0, 0, 0]]
    assert [(b.bbox_xyxy_px, b.pixel_count) for b in boxes] == [((0, 0, 1, 1), 1), ((1, 1, 3, 2), 2)]
    mirrored = np.array([[False, True], [True, False]])
    mirrored_labels, mirrored_boxes = label_components(mirrored)
    assert mirrored_labels.tolist() == [[0, 1], [2, 0]]
    assert [(b.bbox_xyxy_px, b.pixel_count) for b in mirrored_boxes] == [
        ((1, 0, 2, 1), 1), ((0, 1, 1, 2), 1)]
    _, result = run_model(tmp_path, [
        ("A", "1-U.png", ((10, 20), (10, 22))),
        ("B", "8-L.png", ((10, 20), (10, 22))),
        ("cno", "6_C.png", ((10, 20), (10, 21), (10, 22)))])
    c, = result.components
    assert c.pixel_count == 3 and c.common_intersection_pixels == 2 and c.common_intersection_regions == 2
    assert c.unresolved_reasons == ("disconnected_common_intersection_possible_multicontact",)
    assert not result.relations
    assert "cannot_prove_single_physical_contact" in result.residual_limitations


def test_same_bound_element_guard():
    # Real inventory prevents this binding; exercise the conservative guard with
    # an injected selected-pair identity alias, without allowing a relation.
    from psdl1.png_connections import _selected_pair_reasons
    from psdl1.png_contract import OverlapCandidate
    u = OverlapCandidate("same", "u", "A", 1, "U", "M/A/1-U.png", 1)
    l = OverlapCandidate("same", "l", "B", 8, "L", "M/B/8-L.png", 1)
    assert _selected_pair_reasons((u,), (l,), ()) == ("same_bound_element",)


def test_representative_pixel_and_one_connection(tmp_path):
    # Hollow square J: centroid lies in the empty middle, four tied closest
    # pixels resolve to y=0,x=1. Local shape is offset in the original frame.
    local = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert representative_pixel(local) == (1, 0)
    assert representative_pixel(np.ones((2, 2), dtype=bool)) == (0, 0)
    pixels = ((20, 30), (20, 31), (20, 32), (21, 30), (21, 32), (22, 30), (22, 31), (22, 32))
    model, result = run_model(tmp_path, [("A", "1-U.png", pixels), ("B", "8-L.png", pixels),
                                       ("cno", "6_C.png", pixels)])
    c, = result.components
    r, = result.relations
    assert r.representative_xy_px == (31, 20)
    assert r.common_intersection_pixels == 8
    assert r.endpoints[0].xy_um == r.endpoints[1].xy_um == pytest.approx((3.15, 2.05))
    assert c.bbox_xyxy_px == (30, 20, 33, 23) and c.mask.tolist() == local.tolist()
    connection, = iter_connections(result)
    assert connection.kind == "CNT" and connection.observation == "observed"
    assert np.argwhere(connection.mask).tolist() == [list(p) for p in pixels]
    assert np.array_equal(connection.mask, model.cnt_assets[0].mask)
    assert c.pixel_count == 8 and result.summary.resolved_relation_count == 1
    assert to_product(model, result).lineage_id is None
    explicit = ModelMetadata("M", "synthetic_family", "synthetic_v1")
    product = to_product(model, result, metadata=explicit)
    assert product.lineage_id == "synthetic_family" and product.design_version == "synthetic_v1"
    assert validate_product(product)[0] == (1001, 1203)


@pytest.mark.parametrize("status", ["complete", "partial", "unavailable", "absent", "empty", "no_assets"])
def test_extraction_coverage_and_cnt_source_reasons(tmp_path, status):
    upper = ((10, 20), (30, 40)) if status == "complete" else ((10, 20),)
    lower = ((10, 20), (30, 40)) if status in {"complete", "unavailable"} else ((10, 20),)
    if status == "unavailable": upper = ((500, 500),)
    items = [("A", "1-U.png", upper), ("B", "8-L.png", lower)]
    if status not in {"absent", "no_assets"}:
        items.append(("cno", "6_C.png", () if status == "empty" else ((10, 20), (30, 40))))
    write_model(tmp_path, items)
    if status == "no_assets": (tmp_path / "M/cno").mkdir()
    model, = discover_source(tmp_path).models
    result = extract_model(model)
    expected = status if status in {"complete", "partial", "unavailable"} else "unavailable"
    assert result.connectivity_observation == expected
    assert result.summary.resolved_relation_count == {"complete": 2, "partial": 1}.get(status, 0)
    assert result.electrical_verification == result.real_data_validation == "not_performed"
    assert result.source_reasons == {"absent": ("missing_cno_directory",),
                                     "empty": ("observed_cnt_empty_foreground",),
                                     "no_assets": ("present_cno_without_png_assets",)}.get(status, ())
    product = to_product(model, result)
    if status in {"partial", "unavailable"}: assert product.cnt[30, 40]
    if status in {"absent", "no_assets"}: assert product.cnt is None
    if status == "empty": assert product.cnt is not None and product.cnt.sum() == 0
    if status == "complete": assert result.summary.resolved_relation_count <= result.summary.component_count


def test_components_on_distinct_cnt_planes_are_not_unioned(tmp_path):
    model, result = run_model(tmp_path, [("A", "1-U.png", ((10, 20),)),
                                       ("B", "8-L.png", ((10, 20),)),
                                       ("cno", "4_C4.png", ((10, 20),)),
                                       ("cno", "6_C6.png", ((10, 20),))])
    assert result.summary.component_count == result.summary.resolved_relation_count == 2
    assert len({c.component_id for c in result.components}) == 2
    assert [c.cnt_stack for c in result.components] == [4, 6]
    assert len(to_product(model, result).connections) == 2


def test_enumeration_and_root_relocation_invariance(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    pixels = ((10, 20), (30, 40))
    write_model(first, [("A", "1-U.png", pixels), ("B", "8-L.png", pixels),
                        ("cno", "6_C.png", pixels)])
    model, = discover_source(first).models
    before = extract_model(model)
    shutil.copytree(first, second)
    (second / "M/cno/6_C.png").rename(second / "M/cno/006-C.png")
    write_png(second, "M/A/1-U.png", pixels=pixels, compress_level=0)
    from pathlib import Path
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda p: iter(reversed(list(original(p)))))
    moved, = discover_source(second).models
    moved = replace(moved, assets=tuple(reversed(moved.assets)))
    import psdl1.png_connections as module
    original_labeler = module.label_components
    def reverse_components(mask):
        labels, boxes = original_labeler(mask)
        return labels, tuple(reversed(boxes))
    monkeypatch.setattr(module, "label_components", reverse_components)
    after = extract_model(moved)
    assert [c.component_id for c in before.components] == [c.component_id for c in after.components]
    assert [r.connection_id for r in before.relations] == [r.connection_id for r in after.relations]
    assert [r.representative_xy_px for r in after.relations] == [(20, 10), (40, 30)]
    assert input_identity(to_product(model, before)) == input_identity(to_product(moved, after))
    changed = replace(moved, assets=tuple(replace(a, electrode="renamed", role="renamed")
                                          if a.electrode == "B" else a for a in moved.assets))
    changed_result = extract_model(changed)
    assert [r.connection_id for r in changed_result.relations] != [r.connection_id for r in before.relations]
    assert input_identity(to_product(changed, changed_result)) != input_identity(to_product(model, before))
    inv, results = extract_inventory(discover_source(first))
    payload = report_payload(ImportReport(inv, load_metadata(None, ["M"]), extraction=results,
                                         extraction_status="complete"))
    assert payload["extraction"][0]["summary"]["component_count"] == 2
    assert "mask" not in payload["extraction"][0]["components"][0]
    assert payload["inventory"]["models"][0]["resource_estimate"]["frame_pixels"] == 1_204_203
    json.dumps(payload, allow_nan=False)


def test_invalid_source_resource_estimate_remains_unknown(tmp_path):
    write_model(tmp_path, [("A", "1-U.png", ((10, 20),))], shape=(1000, 1203))
    model, = discover_source(tmp_path).models
    result = extract_model(model)
    assert result.extraction_status == "blocked"
    assert result.resource_estimate.frame_pixels is None
    assert result.resource_estimate.projected_bool_npy_export_bytes is None
    assert result.resource_estimate.unknown_reasons == ("invalid_or_undecodable_source",)
    with pytest.raises(ValueError): to_product(model, result)


def test_label_run_unions_bbox_and_canonical_first_pixel_order():
    mask = np.array([[1, 0, 1, 0, 0, 1],
                     [1, 1, 1, 0, 0, 1],
                     [1, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 1]], dtype=bool)
    labels, boxes = label_components(mask)
    assert labels.tolist() == [[1, 0, 1, 0, 0, 2], [1, 1, 1, 0, 0, 2],
                               [1, 0, 1, 0, 0, 0], [0, 0, 0, 3, 0, 4]]
    assert [(b.label, b.bbox_xyxy_px, b.pixel_count, b.first_yx) for b in boxes] == [
        (1, (0, 0, 3, 3), 7, (0, 0)), (2, (5, 0, 6, 2), 2, (0, 5)),
        (3, (3, 3, 4, 4), 1, (3, 3)), (4, (5, 3, 6, 4), 1, (3, 5))]
    labels, boxes = label_components(np.zeros((2, 3), dtype=bool))
    assert labels.tolist() == [[0, 0, 0], [0, 0, 0]] and boxes == ()


def test_representative_pixel_matches_fraction_reference_randomized(monkeypatch):
    from fractions import Fraction
    import psdl1.png_connections as module
    rng = np.random.default_rng(482)
    def reference(mask):
        points = [(x, y) for y in range(mask.shape[0]) for x in range(mask.shape[1]) if mask[y, x]]
        cx = Fraction(sum(p[0] for p in points), len(points))
        cy = Fraction(sum(p[1] for p in points), len(points))
        return min(points, key=lambda p: ((p[0] - cx) ** 2 + (p[1] - cy) ** 2, p[1], p[0]))
    masks = []
    for _ in range(300):
        shape = tuple(int(x) for x in rng.integers(1, 16, size=2))
        mask = rng.random(shape) < rng.uniform(0.05, 0.95)
        mask[0, 0] = True
        masks.append(mask)
    expected = [reference(mask) for mask in masks]
    assert [representative_pixel(mask) for mask in masks] == expected
    # Force the exact-integer fallback at small sizes, without allocating an
    # enormous image: lower the proof bound only, never modify coordinate values.
    class SmallInt64:
        max = 1
    monkeypatch.setattr(module.np, "iinfo", lambda dtype: SmallInt64())
    assert [representative_pixel(mask) for mask in masks] == expected


def test_component_id_precomputed_hash_preserves_identity():
    from psdl1.png_contract import component_id, mask_hash
    mask = np.array([[True, False], [True, True]])
    args = ("M", 6, "C", ("M", "cno", None, 6, "C"), (1001, 1203), (20, 30, 22, 32), mask)
    # Both call routes must retain the exact published pre-FA canonical identity.
    expected = "component_" + hashlib.sha256(json.dumps([
        "component", ["M", 6, "C", ["M", "cno", None, 6, "C"], [1001, 1203], [20, 30, 22, 32],
                      mask_hash(mask)]], separators=(",", ":")).encode()).hexdigest()
    assert component_id(*args) == component_id(*args, local_mask_sha256=mask_hash(mask)) == expected
