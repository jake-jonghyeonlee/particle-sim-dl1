"""Original-coordinate connection counterexamples S06-S12; synthetic only."""
from dataclasses import replace
import hashlib
import json
import shutil

import numpy as np
import pytest

from png_helpers import write_model, write_png, write_stack_order
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


def run_model(tmp_path, assets, order=None):
    write_model(tmp_path, assets)
    model, = discover_source(tmp_path, stack_order=write_stack_order(tmp_path.parent / (tmp_path.name + "-order.json"), order if order is not None else ["U", "C", "L", "D"])).models
    assert model.valid
    return model, extract_model(model)


@pytest.mark.parametrize("joint_at_nearest", [True, False])
def test_nearest_overlapping_lower_only(tmp_path, joint_at_nearest):
    model, result = run_model(tmp_path, [
        ("A", "1-U.png", ((10, 20),)),
        ("N", "7-N.png", ((500, 500),)),
        ("B", "8-L.png", ((10, 20 if joint_at_nearest else 22),)),
        ("D", "10-D.png", ((10, 20),)),
        ("cno", "006_OCNT.png", ((10, 20), (10, 21), (10, 22)))], ["U", "OCNT", "N", "L", "D"])
    c, = result.components
    assert [(v.electrode, v.electrode_stack) for v in c.below_candidates] == [("B", 8), ("D", 10)]
    assert [(v.electrode, v.electrode_stack) for v in c.selected_lower_candidates] == [("B", 8)]
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
        ("L", "9-L.png", ((10, 20),))], ["U", "F", "N", "C4", "C6", "C7", "B", "L"])
    r, = result.relations
    assert [e.element_id for e in r.endpoints] == [expected_element("F", 2, "F"), expected_element("L", 9, "L")]
    assert [(l.electrode_stack, l.layer, l.source_paths) for l in r.skipped_upper_layers.electrode_layers] == [
        (3, "N", ("M/N/3-N.png",))]
    assert [(l.order_index, l.layer) for l in r.skipped_upper_layers.cno_layers] == [(3, "C4")]
    assert [(l.electrode_stack, l.layer, l.source_paths) for l in r.skipped_lower_layers.electrode_layers] == [
        (8, "B", ("M/B/8-B.png", "M/B2/8-B.png"))]
    assert [(l.order_index, l.layer) for l in r.skipped_lower_layers.cno_layers] == [(5, "C7")]
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


@pytest.mark.parametrize("missing", ["upper", "lower", "both"])
def test_missing_side_unresolved(tmp_path, missing):
    items = [("cno", "6_C.png", ((10, 20),))]
    if missing not in {"upper", "both"}: items.append(("A", "1-U.png", ((10, 20),)))
    if missing not in {"lower", "both"}: items.append(("B", "8-L.png", ((10, 20),)))
    if missing == "both": items.append(("N", "1-U.png", ((500, 500),)))
    _, result = run_model(tmp_path, items)
    c, = result.components
    expected = {"upper": ("missing_upper_overlap",), "lower": ("missing_lower_overlap",),
                "both": ("missing_upper_overlap", "missing_lower_overlap")}
    assert c.unresolved_reasons == expected[missing] and not result.relations


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
    u = OverlapCandidate("same", "u", "A", 0, 1, "U", "M/A/1-U.png", 1)
    l = OverlapCandidate("same", "l", "B", 2, 8, "L", "M/B/8-L.png", 1)
    assert _selected_pair_reasons((u,), (l,)) == ("same_bound_element",)


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
    model, = discover_source(tmp_path, stack_order=write_stack_order(tmp_path.parent / (tmp_path.name + "-order.json"), ["U", "C", "L"])).models
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
                                       ("cno", "6_C6.png", ((10, 20),))], ["U", "C4", "C6", "L"])
    assert result.summary.component_count == result.summary.resolved_relation_count == 2
    assert len({c.component_id for c in result.components}) == 2
    assert [c.cnt_order_index for c in result.components] == [1, 2]
    assert len(to_product(model, result).connections) == 2


def test_enumeration_and_root_relocation_invariance(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    pixels = ((10, 20), (30, 40))
    write_model(first, [("A", "1-U.png", pixels), ("B", "8-L.png", pixels),
                        ("cno", "6_C.png", pixels)])
    model, = discover_source(first, stack_order=write_stack_order(tmp_path / "order.json", ["U", "C", "L"])).models
    before = extract_model(model)
    shutil.copytree(first, second)
    (second / "M/cno/6_C.png").rename(second / "M/cno/006-C.png")
    write_png(second, "M/A/1-U.png", pixels=pixels, compress_level=0)
    from pathlib import Path
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda p: iter(reversed(list(original(p)))))
    moved, = discover_source(second, stack_order=tmp_path / "order.json").models
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
    inv, results = extract_inventory(discover_source(first, stack_order=write_stack_order(tmp_path / "order.json", ["U", "C", "L"])))
    payload = report_payload(ImportReport(inv, load_metadata(None, ["M"]), extraction=results,
                                         extraction_status="complete"))
    assert payload["extraction"][0]["summary"]["component_count"] == 2
    assert "mask" not in payload["extraction"][0]["components"][0]
    assert payload["inventory"]["models"][0]["resource_estimate"]["frame_pixels"] == 1_204_203
    json.dumps(payload, allow_nan=False)


def test_invalid_source_resource_estimate_remains_unknown(tmp_path):
    write_model(tmp_path, [("A", "1-U.png", ((10, 20),))], shape=(1000, 1203))
    model, = discover_source(tmp_path, stack_order=write_stack_order(tmp_path.parent / (tmp_path.name + "-order.json"), ["U", "C", "L"])).models
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
    args = ("M", "C", ("M", "cno", None, "C", "full-mask-hash"), (1001, 1203), (20, 30, 22, 32), mask)
    # Both call routes follow the declared number-free component payload.
    expected = "component_" + hashlib.sha256(json.dumps([
        "component", ["M", "C", ["M", "cno", None, "C", "full-mask-hash"], [1001, 1203], [20, 30, 22, 32],
                      mask_hash(mask)]], separators=(",", ":")).encode()).hexdigest()
    assert component_id(*args) == component_id(*args, local_mask_sha256=mask_hash(mask)) == expected


def test_extracted_component_hash_and_id_describe_actual_mask(tmp_path):
    # Nonrectangular component: replacing its hash input by ~local must fail.
    pixels = ((10, 20), (10, 21), (11, 20))
    _, result = run_model(tmp_path, [("A", "1-U.png", pixels), ("B", "8-L.png", pixels),
                                   ("cno", "6_C.png", pixels)])
    c, = result.components
    actual = np.array([[True, True], [True, False]])
    assert np.array_equal(c.mask, actual)
    def independent_hash(mask):
        h = hashlib.sha256(b'{"shape_yx":[2,2]}')
        h.update(np.asarray(mask, dtype=bool).tobytes())
        return h.hexdigest()
    expected_hash = independent_hash(actual)
    assert c.mask_sha256 == expected_hash
    full = np.zeros((1001, 1203), dtype=bool)
    for y, x in pixels: full[y, x] = True
    full_digest = hashlib.sha256(b'{"shape_yx":[1001,1203]}')
    full_digest.update(full.tobytes())
    payload = ["component", ["M", "C", ["M", "cno", None, "C", full_digest.hexdigest()], [1001, 1203],
                             [20, 10, 22, 12], expected_hash]]
    expected_id = "component_" + hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    assert c.component_id == expected_id
    assert independent_hash(~actual) != c.mask_sha256
    altered = actual.copy(); altered[0, 1] = False
    assert independent_hash(altered) != c.mask_sha256
    from psdl1.png_contract import component_id
    assert component_id("M", "C", ("M", "cno", None, "C", "full-mask-hash"), (1001, 1203),
                        (20, 10, 22, 12), ~actual) != c.component_id


def test_int64_proof_near_overflow_takes_exact_fallback(monkeypatch):
    from fractions import Fraction
    import psdl1.png_connections as module
    # Sparse coordinates injected at the arithmetic boundary avoid a multi-GB
    # image allocation. Expanded score bound is just beyond signed int64.
    ys = np.array([0, 2_630_119_584, 2_630_119_585], dtype=np.int64)
    xs = np.array([0, 0, 0], dtype=np.int64)
    mean_y, rem_y = divmod(sum(map(int, ys)), 3)
    max_delta = max(abs(int(y) - mean_y) for y in ys)
    bound = 3 * max_delta ** 2 + 2 * rem_y * max_delta
    assert np.iinfo(np.int64).max < bound < np.iinfo(np.int64).max + 100_000_000_000
    cy = Fraction(sum(map(int, ys)), 3)
    expected_y = min(map(int, ys), key=lambda y: ((Fraction(y) - cy) ** 2, y))
    assert expected_y == 2_630_119_584
    monkeypatch.setattr(module.np, "nonzero", lambda mask: (ys, xs))
    def forbidden_argmin(scores):
        pytest.fail("unsafe near-overflow coordinates took int64 path")
    monkeypatch.setattr(module.np, "argmin", forbidden_argmin)
    assert representative_pixel(np.ones((1, 1), dtype=bool)) == (0, expected_y)


# O01–O12/O18: literal synthetic orders are syntax-only, not physical prescriptions.
def ordered_source(root, order, assets=None):
    if assets is not None:
        write_model(root, assets)
    reference = write_stack_order(root.parent / (root.name + '-order.json'), order)
    model, = discover_source(root, stack_order=reference).models
    return model, extract_model(model), reference


def three_layers(cnts=None):
    p = ((10, 20),)
    return [('A', '1-U.png', p), ('B', '2-M.png', p), ('D', '3-L.png', p)] + (
        cnts if cnts is not None else [('cno', '1-OCNT.png', p)])


def endpoint_ids(result):
    return [[e.element_id for e in r.endpoints] for r in result.relations]


def test_cno_123_numbers_do_not_conflict_with_electrode_stacks(tmp_path):
    root = tmp_path / 'source'
    model, result, order = ordered_source(root, ['U', 'OCNT', 'M', 'GCNT', 'L', 'PCNT'],
        three_layers([('cno', '1-OCNT.png', ((10, 20),)), ('cno', '2-GCNT.png', ((10, 20),)),
                      ('cno', '3-PCNT.png', ((10, 20),))]))
    assert model.valid and all(a.stack is None for a in model.cnt_assets)
    assert endpoint_ids(result) == [[expected_element('A', 1, 'U'), expected_element('B', 2, 'M')],
                                    [expected_element('B', 2, 'M'), expected_element('D', 3, 'L')]]
    assert result.components[2].unresolved_reasons == ('missing_lower_overlap',)
    for name in ['2-GCNT.png', '3-PCNT.png']:
        (root / 'M/cno' / name).rename(root / 'M/cno' / ('1-' + name.split('-', 1)[1]))
    renamed, = discover_source(root, stack_order=order).models
    changed = extract_model(renamed)
    assert [a.asset_id for a in model.cnt_assets] == [a.asset_id for a in renamed.cnt_assets]
    assert [r.connection_id for r in result.relations] == [r.connection_id for r in changed.relations]


def test_same_geometry_two_orders_changes_relations(tmp_path):
    root = tmp_path / 'source'
    first, result, _ = ordered_source(root, ['U', 'OCNT', 'M', 'L'], three_layers())
    second, changed, _ = ordered_source(root, ['U', 'M', 'OCNT', 'L'])
    assert endpoint_ids(result) == [[expected_element('A', 1, 'U'), expected_element('B', 2, 'M')]]
    assert endpoint_ids(changed) == [[expected_element('B', 2, 'M'), expected_element('D', 3, 'L')]]
    for r in result.relations + changed.relations:
        assert all(e.xy_um == pytest.approx((2.05, 1.05)) for e in r.endpoints)
    assert first.cnt_assets[0].asset_id == second.cnt_assets[0].asset_id
    assert result.components[0].component_id == changed.components[0].component_id
    assert result.relations[0].connection_id != changed.relations[0].connection_id
    assert first.stack_order.resolved_order_sha256 != second.stack_order.resolved_order_sha256


def test_stack_order_rejects_reversed_electrode_numbers(tmp_path):
    model, result, order = ordered_source(tmp_path / 'source', ['M', 'OCNT', 'U', 'L'], three_layers())
    assert model.stack_order.errors == ('stack_order_electrode_order_conflict: model "M", listed [["M", 2], ["U", 1], ["L", 3]], expected [["U", 1], ["M", 2], ["L", 3]]',)
    assert result.extraction_status == 'blocked' and result.relations == ()
    from psdl1.import_png import run_import
    from psdl1.png_contract import ImportOptions
    out = tmp_path / 'out'
    assert run_import(ImportOptions(root=tmp_path / 'source', out=out, export=True, stack_order=order))[1] == 2
    assert not (out / 'manifest.json').exists() and not (out / 'assets').exists()


@pytest.mark.parametrize('white', [False, True])
def test_stack_order_requires_all_present_names(tmp_path, white):
    assets = three_layers([('cno', '1-OCNT.png', () if white else ((10, 20),))])
    model, result, _ = ordered_source(tmp_path / 'source', ['U', 'M', 'L'], assets)
    assert model.stack_order.errors == ('stack_order_missing_cnt_names: model "M", missing ["OCNT"]',)
    assert not result.relations
    model, result, _ = ordered_source(tmp_path / 'source', ['U', 'OCNT', 'L'])
    assert model.stack_order.errors == ('stack_order_missing_electrode_names: model "M", missing ["M"]',)
    assert not result.relations


def test_reference_order_filters_absent_layers_for_all_models(tmp_path):
    root = tmp_path / 'source'
    write_model(root, three_layers(), model='A')
    write_model(root, [('A', '1-U.png', ((10, 20),)), ('D', '3-L.png', ((10, 20),)),
                       ('cno', '1-GCNT.png', ((10, 20),))], model='B')
    write_model(root, [('A', '1-X.png', ((10, 20),)), ('D', '3-Y.png', ((10, 20),)),
                       ('cno', '1-Z.png', ((10, 20),))], model='C')
    order = write_stack_order(tmp_path / 'order.json', ['U', 'OCNT', 'M', 'GCNT', 'L', 'PCNT'])
    a, b, c = discover_source(root, stack_order=order).models
    assert [e.name for e in a.stack_order.resolved_order] == ['U', 'OCNT', 'M', 'L']
    assert a.stack_order.ignored_names == ('GCNT', 'PCNT')
    assert [e.name for e in b.stack_order.resolved_order] == ['U', 'GCNT', 'L']
    assert b.stack_order.ignored_names == ('OCNT', 'M', 'PCNT')
    assert c.stack_order.errors == ('stack_order_missing_electrode_names: model "C", missing ["X", "Y"]',
                                    'stack_order_missing_cnt_names: model "C", missing ["Z"]')
    assert not extract_model(c).relations
    from psdl1.png_source import load_stack_order
    order.write_text('{"stack_order_schema_version":"dl1_png_stack_order_v1","order":[],"models":{}}')
    assert load_stack_order(order).errors == ('stack_order_fields: required ["order", "stack_order_schema_version"]; missing []; unknown ["models"]',)


def test_same_name_cnt_assets_remain_separate_and_ambiguities_fail(tmp_path):
    root = tmp_path / 'source'
    assets = [('A', '1-U.png', ((10, 20), (10, 21), (10, 22))),
              ('B', '3-L.png', ((10, 20), (10, 21), (10, 22))),
              ('cno', '1-OCNT.png', ((10, 20), (10, 21))), ('cno', '9-OCNT.png', ((10, 21), (10, 22)))]
    model, result, order = ordered_source(root, ['U', 'OCNT', 'L'], assets)
    assert model.valid and len(model.cnt_assets) == 2
    assert result.summary.component_count == result.summary.resolved_relation_count == 2
    assert {c.cnt_order_index for c in result.components} == {1}
    assert {c.pixel_count for c in result.components} == {2}
    assert model.stack_order.resolved_order[1].asset_ids == tuple(sorted(a.asset_id for a in model.cnt_assets))
    original = sorted(a.asset_id for a in model.cnt_assets)
    (root / 'M/cno/1-OCNT.png').rename(root / 'M/cno/88_OCNT.png')
    (root / 'M/cno/9-OCNT.png').rename(root / 'M/cno/2-OCNT.png')
    renamed, = discover_source(root, stack_order=order).models
    assert sorted(a.asset_id for a in renamed.cnt_assets) == original
    write_png(root, 'M/cno/2-OCNT.png', pixels=((10, 20), (10, 21)), compress_level=0)
    invalid, = discover_source(root, stack_order=order).models
    reason, = [e.reason for e in invalid.errors if e.reason.startswith('ambiguous_cnt_asset_identity:')]
    assert '"M/cno/2-OCNT.png", "M/cno/88_OCNT.png"' in reason
    assert not invalid.valid and not extract_model(invalid).relations
    for name, fixture, expected in [
        ('ambiguous', [('A', '1-X.png', ((10, 20),)), ('B', '3-X.png', ((10, 20),)), ('cno', '1-C.png', ((10, 20),))], 'stack_order_ambiguous_electrode_name'),
        ('collision', [('A', '1-OCNT.png', ((10, 20),)), ('cno', '1-OCNT.png', ((10, 20),))], 'stack_order_name_kind_collision')]:
        m, r, _ = ordered_source(tmp_path / name, ['X', 'C', 'OCNT'], fixture)
        assert m.stack_order.errors[0].startswith(expected + ':') and not r.relations
    m, _, _ = ordered_source(tmp_path / 'shared', ['U', 'OCNT', 'L'],
        [('A', '1-U.png', ((10, 20),)), ('B', '1-U.png', ((10, 20),)), ('D', '3-L.png', ((10, 20),)), ('cno', '1-OCNT.png', ((10, 20),))])
    assert m.valid and m.stack_order.status == 'resolved'


def test_cno_renumbering_preserves_semantic_identities(tmp_path, monkeypatch):
    from psdl1.png_contract import model_identity
    root = tmp_path / 'source'
    model, result, order = ordered_source(root, ['U', 'OCNT', 'M', 'GCNT', 'L'],
        three_layers([('cno', '1-OCNT.png', ((10, 20),)), ('cno', '9-OCNT.png', ((10, 20), (10, 21))),
                      ('cno', '2-GCNT.png', ((10, 20),))]))
    moved = tmp_path / 'moved'; shutil.copytree(root, moved)
    (moved / 'M/cno/1-OCNT.png').rename(moved / 'M/cno/099_OCNT.png')
    (moved / 'M/cno/2-GCNT.png').rename(moved / 'M/cno/1-GCNT.png')
    from pathlib import Path
    iterdir = Path.iterdir
    monkeypatch.setattr(Path, 'iterdir', lambda p: iter(reversed(list(iterdir(p)))))
    after, = discover_source(moved, stack_order=order).models
    changed = extract_model(after)
    assert model_identity(model) == model_identity(after)
    assert [c.component_id for c in result.components] == [c.component_id for c in changed.components]
    assert [r.connection_id for r in result.relations] == [r.connection_id for r in changed.relations]
    assert input_identity(to_product(model, result)) == input_identity(to_product(after, changed))
    for a in model.cnt_assets:
        payload = ['asset', ['M', 'cno', None, a.layer, a.mask_sha256]]
        assert a.asset_id == 'asset_' + hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()
        c = next(c for c in result.components if c.source_asset_id == a.asset_id)
        payload = ['component', ['M', a.layer, ['M', 'cno', None, a.layer, a.mask_sha256],
                                 [1001, 1203], list(c.bbox_xyxy_px), c.mask_sha256]]
        assert c.component_id == 'component_' + hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()
    from psdl1.import_png import run_import
    from psdl1.png_contract import ImportOptions
    from psdl1.data import load_manifest
    mapping = tmp_path / 'metadata.json'
    mapping.write_text(json.dumps({'metadata_schema_version': 'dl1_png_metadata_v1', 'models': {
        'M': {'lineage_id': 'SYNTHETIC', 'design_version': 'SYNTHETIC-V1'}}}))
    loaded_ids = []
    for source, destination in [(root, tmp_path / 'before-export'), (moved, tmp_path / 'after-export')]:
        assert run_import(ImportOptions(root=source, out=destination, export=True, metadata=mapping,
                                       candidate_grid=16, stack_order=order))[1] == 0
        loaded_ids.append(input_identity(load_manifest(destination / 'manifest.json')[0]))
    assert loaded_ids[0] == loaded_ids[1]
    assert next(a for a in after.cnt_assets if a.original_filename == '099_OCNT.png').filename_number == 99
    write_png(moved, 'M/cno/099_OCNT.png', pixels=((11, 20),))
    pixel_changed, = discover_source(moved, stack_order=order).models
    assert {a.asset_id for a in pixel_changed.cnt_assets} != {a.asset_id for a in after.cnt_assets}
    (moved / 'M/cno/099_OCNT.png').rename(moved / 'M/cno/099_OTHER.png')
    name_changed, = discover_source(moved, stack_order=write_stack_order(tmp_path / 'other-order.json', ['U', 'OCNT', 'OTHER', 'M', 'GCNT', 'L'])).models
    assert {a.asset_id for a in name_changed.cnt_assets} != {a.asset_id for a in pixel_changed.cnt_assets}


def test_order_skipped_layers_separate_electrode_and_cno(tmp_path):
    root = tmp_path / 'source'
    model, result, _ = ordered_source(root, ['U', 'E', 'GCNT', 'OCNT', 'F', 'PCNT', 'L'],
        [('A', '1-U.png', ((10, 20),)), ('B', '9-L.png', ((10, 20),)),
         ('E', '3-E.png', ((900, 900),)), ('F', '7-F.png', ((900, 900),)),
         ('cno', '88-OCNT.png', ((10, 20),)), ('cno', '1-GCNT.png', ()), ('cno', '2-PCNT.png', ())])
    r, = result.relations
    assert result.components[0].cnt_order_index == 3
    assert [(e.order_index, e.layer, e.electrode_stack) for e in r.skipped_upper_layers.electrode_layers] == [(1, 'E', 3)]
    assert [(e.order_index, e.layer, e.electrode_stack) for e in r.skipped_upper_layers.cno_layers] == [(2, 'GCNT', None)]
    assert [(e.order_index, e.layer, e.electrode_stack) for e in r.skipped_lower_layers.electrode_layers] == [(4, 'F', 7)]
    assert [(e.order_index, e.layer, e.electrode_stack) for e in r.skipped_lower_layers.cno_layers] == [(5, 'PCNT', None)]
    s = result.summary
    assert (s.skipped_upper_electrode_layers, s.skipped_upper_cno_layers, s.skipped_lower_electrode_layers, s.skipped_lower_cno_layers) == (1, 1, 1, 1)
    assert r.needs_confirmation
    write_png(root, 'M/cno/47-GCNT.png', pixels=((800, 800),))
    model, changed, _ = ordered_source(root, ['ABSENT', 'U', 'E', 'GCNT', 'OCNT', 'F', 'PCNT', 'L'])
    r, = changed.relations
    assert len(r.skipped_upper_layers.cno_layers) == 1
    assert r.skipped_upper_layers.cno_layers[0].source_paths == ('M/cno/1-GCNT.png', 'M/cno/47-GCNT.png')
    assert model.stack_order.ignored_names == ('ABSENT',)
    assert changed.summary.component_count == 2 and changed.summary.resolved_relation_count == 1


def test_nearest_pair_guards_and_bbox_cost_with_explicit_order(tmp_path, monkeypatch):
    from psdl1 import png_connections as module
    operands = []
    original = module.overlap_pixels
    def inspect(c, a):
        operands.append((c.shape, a.shape))
        return original(c, a)
    monkeypatch.setattr(module, 'overlap_pixels', inspect)
    model, result, _ = ordered_source(tmp_path / 'source', ['OCNT', 'U', 'M', 'L'], three_layers())
    assert result.components[0].unresolved_reasons == ('missing_upper_overlap',)
    assert not result.relations and operands == [((1, 1), (1, 1))] * 3
    # Existing tests above retain no-fallthrough, common-intersection, selected
    # layer ambiguity, 4-neighbour, centroid ties and disconnected-region guards.
    assert result.resource_estimate.cnt_component_count == 1
    assert result.resource_estimate.resolved_relation_count == 0


def test_same_electrode_rule_stays_off_until_confirmed(tmp_path):
    model, result, _ = ordered_source(tmp_path / 'source', ['U', 'OCNT', 'L'],
        [('A', '1-U.png', ((10, 20),)), ('B', '3-L.png', ((10, 20),)), ('cno', '1-OCNT.png', ((10, 20),))])
    assert endpoint_ids(result) == [[expected_element('A', 1, 'U'), expected_element('B', 3, 'L')]]
    assert result.same_electrode_required is False
    assert ImportReport.__dataclass_fields__['same_electrode_required'].default is False
    _, changed, _ = ordered_source(tmp_path / 'ambiguous', ['U', 'OCNT', 'L'],
        [('A', '1-U.png', ((10, 20),)), ('B', '1-U.png', ((10, 20),)),
         ('A', '3-L.png', ((10, 20),)), ('cno', '1-OCNT.png', ((10, 20),))])
    assert not changed.relations
    assert changed.components[0].unresolved_reasons == ('multiple_upper_electrodes_on_selected_layer',)
