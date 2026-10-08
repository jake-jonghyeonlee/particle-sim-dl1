"""Draft contract rejection and input identity counterexamples, no training."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from psdl1.data import (build_vocab, input_identity, load_manifest, prepare_product,
                        product_tensors, grid_check, main)
from psdl1.schema import Element, Product, Endpoint, Connection, Frame, binary_mask, validate_product


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-1]]),
                                np.array([[0.25]]), np.array([[2]]), np.zeros((2, 2, 1)), np.zeros((0, 2)),
                                np.array([["1"]]), np.array([[1, 255]])])
def test_bad_binary_masks_are_rejected(bad):
    with pytest.raises(ValueError):
        binary_mask(bad)


@pytest.mark.parametrize("good", [np.array([[0, 1]]), np.array([[0, 255]], dtype=np.uint8), np.array([[False, True]])])
def test_binary_encodings_have_explicit_values(good):
    assert binary_mask(good).tolist() == [[0, 1]]


@pytest.mark.parametrize("field,value", [("product_id", ""), ("lineage_id", " "), ("design_version", ""),
                                          ("unit_size_um", [0, 0.2]), ("unit_size_um", [0.2, float("nan")]),
                                          ("unit_size_um", [0.3, 0.2])])
def test_product_identity_and_unit_rejections(manifest_case, field, value):
    doc, write, _ = manifest_case
    doc["products"][0][field] = value
    with pytest.raises(ValueError):
        load_manifest(write())


@pytest.mark.parametrize("field,value", [("origin_xy_um", [0, float("inf")]), ("origin_xy_um", [0]),
                                          ("pixel_pitch_um", 0), ("length_unit", "nm"),
                                          ("axis_order", "xy"), ("shape_yx", [4, 5]),
                                          ("shape_yx", [True, 4]), ("frame_id", "")])
def test_frame_rejections(manifest_case, field, value):
    doc, write, _ = manifest_case
    doc["products"][0]["frame"][field] = value
    with pytest.raises(ValueError):
        load_manifest(write())


@pytest.mark.parametrize("asset", ["cnt", "valid", "connection", "element"])
def test_all_assets_share_frame(manifest_case, asset):
    doc, write, root = manifest_case
    np.save(root / "bad.npy", np.zeros((3, 4)))
    p = doc["products"][0]
    if asset == "connection":
        p["connections"][0]["mask"] = "bad.npy"
    elif asset == "element":
        p["elements"][0]["mask"] = "bad.npy"
    else:
        p[asset] = "bad.npy"
    with pytest.raises(ValueError, match="grid|frame"):
        load_manifest(write())


@pytest.mark.parametrize("path", ["/tmp/out.npy", "../outside.npy", "sub/../../outside.npy", "C:\\inside\\shape.npy"])
def test_absolute_and_escape_paths(manifest_case, path):
    doc, write, _ = manifest_case
    doc["products"][0]["elements"][0]["mask"] = path
    with pytest.raises(ValueError, match="path"):
        load_manifest(write())


def test_absolute_path_inside_root_still_rejected(manifest_case):
    doc, write, root = manifest_case
    doc["products"][0]["valid"] = str(root / "valid.npy")
    with pytest.raises(ValueError, match="absolute"):
        load_manifest(write())


def test_symlink_escape_rejected(manifest_case, tmp_path):
    doc, write, root = manifest_case
    outside = root.parent / f"{root.name}-outside.npy"
    np.save(outside, np.ones((4, 4)))
    (root / "link.npy").symlink_to(outside)
    doc["products"][0]["valid"] = "link.npy"
    with pytest.raises(ValueError, match="폴더 밖"):
        load_manifest(write())


@pytest.mark.parametrize("field", ["hotspot", "CA", "fab", "Fab", "ca"])
def test_forbidden_fields_cannot_enter_tensor_path(manifest_case, field):
    doc, write, _ = manifest_case
    doc["products"][0]["elements"][0][field] = "shape.npy"
    with pytest.raises(ValueError, match="forbidden"):
        load_manifest(write())


@pytest.mark.parametrize("mutation", ["schema", "duplicate_product", "duplicate_element", "duplicate_binding", "duplicate_connection", "unknown_field"])
def test_version_and_id_rejections(manifest_case, mutation):
    doc, write, _ = manifest_case
    p = doc["products"][0]
    if mutation == "schema":
        del doc["schema_version"]
    elif mutation == "duplicate_product":
        doc["products"].append(p)
    elif mutation == "duplicate_element":
        p["elements"][1]["element_id"] = "A-L1"
    elif mutation == "duplicate_binding":
        p["elements"][1]["role"] = "A"
        p["elements"][1]["actual_layer"] = "L1"
    elif mutation == "duplicate_connection":
        p["connections"].append(p["connections"][0])
    else:
        p["unclear_payload"] = 42
    with pytest.raises(ValueError):
        load_manifest(write())


@pytest.mark.parametrize("mutation", ["unknown", "one", "same_element", "outside", "not_on_shape", "invalid_region", "missing_observed_element"])
def test_endpoint_rejections(manifest_case, mutation):
    doc, write, root = manifest_case
    p = doc["products"][0]
    ep = p["connections"][0]["endpoints"][0]
    if mutation == "unknown":
        ep["element_id"] = "unknown"
    elif mutation == "one":
        p["connections"][0]["endpoints"].pop()
    elif mutation == "same_element":
        ep["element_id"] = "B-L2"
    elif mutation == "outside":
        ep["xy_um"] = [1.2, -2.0]  # half-open upper edge
    elif mutation == "not_on_shape":
        np.save(root / "shape.npy", np.zeros((4, 4)))
    elif mutation == "invalid_region":
        m = np.ones((4, 4)); m[0, 0] = 0
        np.save(root / "valid.npy", m)
    else:
        p["elements"][0].update(mask=None, observation="missing")
    with pytest.raises(ValueError, match="endpoint"):
        load_manifest(write())


def test_missing_observation_is_not_zero(manifest_case):
    doc, write, _ = manifest_case
    p = doc["products"][0]
    p["elements"][0].update(mask=None, observation="missing")
    p["connections"][0].update(mask=None, observation="missing")
    p["connectivity_observation"] = "partial"
    product, = load_manifest(write())
    assert product.elements[0].mask is None
    assert product.connectivity_status == "partial"
    assert product.grid_check["elements"][0] == {"id": "A-L1", "observation": "missing"}
    with pytest.raises(ValueError, match="missing"):
        product_tensors(product, 4, *build_vocab([product]))


def test_no_endpoint_metadata_means_connectivity_unavailable(manifest_case):
    doc, write, _ = manifest_case
    p = doc["products"][0]
    p.update(connections=[], connectivity_observation="unavailable")
    product, = load_manifest(write())
    assert product.cnt.sum() == 16  # CNT image alone gives no endpoint relation
    assert product.connectivity_status == "unavailable"
    assert not product.grid_check["connectivity_preserving"]


def test_valid_hole_preserved_and_masked_without_recentering(manifest_case):
    doc, write, root = manifest_case
    valid = np.ones((4, 4)); valid[2, 3] = 0
    np.save(root / "valid.npy", valid)
    product, = load_manifest(write())
    item = prepare_product(product, 4, *build_vocab([product]))
    assert item.valid.tolist() == [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
    assert item.images[:, :, 2, 3].sum() == 0
    assert item.connections[0].endpoints[0].xy_um == (1.025, -1.975)
    assert item.connection_masks[0, 2, 3] == 0


def test_element_order_same_representation_but_binding_swap_differs(manifest_case):
    _, write, _ = manifest_case
    p, = load_manifest(write())
    reverse = replace(p, elements=p.elements[::-1])
    assert input_identity(p) == input_identity(reverse)
    vocab = build_vocab([p])
    a, b = prepare_product(p, 4, *vocab), prepare_product(reverse, 4, *vocab)
    assert torch.equal(a.images, b.images.flip(0))
    assert torch.equal(a.role_ids, b.role_ids.flip(0))
    assert torch.equal(a.layer_ids, b.layer_ids.flip(0))
    # Same XY masks, same role set {A,B}, same layer set {L1,L2}, different bindings.
    swapped = replace(p, elements=(replace(p.elements[0], layer="L2"), replace(p.elements[1], layer="L1")))
    assert {e.role for e in p.elements} == {e.role for e in swapped.elements} == {"A", "B"}
    assert {e.layer for e in p.elements} == {e.layer for e in swapped.elements} == {"L1", "L2"}
    assert input_identity(p) != input_identity(swapped)
    c = prepare_product(swapped, 4, *vocab)
    assert torch.equal(a.images, c.images)
    assert a.role_ids.tolist() == c.role_ids.tolist() == [1, 2]
    assert a.layer_ids.tolist() == [1, 2] and c.layer_ids.tolist() == [2, 1]


def test_endpoint_binding_is_part_of_identity(manifest_case):
    _, write, _ = manifest_case
    p, = load_manifest(write())
    c = p.connections[0]
    moved = replace(p, connections=(replace(c, endpoints=(replace(c.endpoints[0], xy_um=(1.075, -1.975)), c.endpoints[1])),))
    assert input_identity(p) != input_identity(moved)


def test_cli_stores_per_product_evidence(manifest_case, monkeypatch):
    _, write, root = manifest_case
    report = root / "report.json"
    monkeypatch.setattr("sys.argv", ["psdl1-validate", str(write()), "--grid", "2", "--report", str(report)])
    main()
    result = json.loads(report.read_text())
    assert result["approval"] == "G5_pending"
    assert result["products"][0]["product_id"] == "P"
    assert result["products"][0]["grid"] == 2
    assert result["products"][0]["elements"][0]["source_components"] == 1


def test_products_can_have_different_pixel_counts_at_same_pitch(manifest_case):
    from copy import deepcopy
    doc, write, root = manifest_case
    second = deepcopy(doc["products"][0])
    second.update(product_id="Q", unit_size_um=[0.3, 0.3])
    second["frame"].update(shape_yx=[6, 6], frame_id="Q-unit")
    np.save(root / "larger.npy", np.ones((6, 6)))
    second["valid"] = second["cnt"] = "larger.npy"
    for element in second["elements"]:
        element["mask"] = "larger.npy"
    second["connections"][0]["mask"] = "larger.npy"
    doc["products"].append(second)
    products = load_manifest(write())
    assert [p.frame.shape_yx for p in products] == [(4, 4), (6, 6)]
    second["frame"]["pixel_pitch_um"] = 0.1
    second["unit_size_um"] = [0.6, 0.6]
    with pytest.raises(ValueError, match="same pixel pitch"):
        load_manifest(write())


def test_all_layers_use_shared_frame_without_individual_centering():
    left = np.zeros((4, 8), dtype=np.float32); left[:, 0] = 1
    right = np.zeros((4, 8), dtype=np.float32); right[:, 7] = 1
    p = Product("offset", (0.4, 0.2), (Element("A", "L1", left), Element("B", "L2", right)))
    item = prepare_product(p, 8, *build_vocab([p]))
    assert item.images[0, 0, 2:6, 0].tolist() == [1, 1, 1, 1]
    assert item.images[1, 0, 2:6, 7].tolist() == [1, 1, 1, 1]
    assert item.images[0, 0, :, 7].sum() == 0
    assert item.images[1, 0, :, 0].sum() == 0


@pytest.mark.parametrize("field,value", [("shape_yx", None), ("origin_xy_um", None)])
def test_malformed_frame_lists_rejected(manifest_case, field, value):
    doc, write, _ = manifest_case
    doc["products"][0]["frame"][field] = value
    with pytest.raises(ValueError):
        load_manifest(write())


def test_malformed_endpoint_coords_rejected(manifest_case):
    doc, write, _ = manifest_case
    doc["products"][0]["connections"][0]["endpoints"][0]["xy_um"] = None
    with pytest.raises(ValueError):
        load_manifest(write())
