import json

import numpy as np
import pytest
import torch

from psdl1.data import augment, build_vocab, collate, load_manifest, narrowest_run_px, normalize_unit, product_tensors, unit_feature

from conftest import make_product


def test_unit_is_resized_to_the_grid_keeping_the_aspect_ratio():
    m = np.ones((20, 40), dtype=np.float32)
    out = normalize_unit(m, 16)
    assert out.shape == (16, 16)
    assert out[4:12].min() == 1.0 and out[:4].max() == 0.0 and out[12:].max() == 0.0


def test_same_layout_at_two_unit_sizes_gives_nearly_the_same_image():
    a = normalize_unit(make_product("comb", 40).elements[0].mask, 32)
    b = normalize_unit(make_product("comb", 80).elements[0].mask, 32)
    other = normalize_unit(make_product("ring", 80).elements[0].mask, 32)
    same, different = np.abs(a - b).mean(), np.abs(a - other).mean()
    assert same < 0.04 and same < 0.2 * different  # 남는 차이는 줄일 때 생기는 가장자리 번짐


def test_shrinking_keeps_partial_coverage_instead_of_dropping_thin_lines():
    m = np.zeros((64, 64), dtype=np.float32)
    m[:, 31] = 1.0  # 1픽셀 선
    out = normalize_unit(m, 16)
    assert out.max() == pytest.approx(0.25) and out.sum() == pytest.approx(64 / 16)


def test_narrowest_run_reports_the_thinnest_line():
    m = np.zeros((10, 10), dtype=bool)
    m[2:8, 3:5] = True   # 폭 2, 길이 6
    m[0, 0:9] = True     # 높이 1
    assert narrowest_run_px(m) == 1
    assert narrowest_run_px(m[2:]) == 2


def test_product_tensors_have_target_cnt_and_rest_channels(products):
    roles, layers = build_vocab(products)
    images, role_ids, layer_ids = product_tensors(products[0], 32, roles, layers)
    assert images.shape == (3, 3, 32, 32)
    assert role_ids.tolist() == [roles["gate"], roles["data"], roles["pixel"]] and min(roles.values()) == 1
    assert torch.allclose(images[0, 1], images[1, 1])  # CNT는 원소마다 같음
    assert (images[0, 0] * images[0, 2]).sum() < images[0, 0].sum()  # 나머지 채널에는 자기 자신이 빠짐


def test_unknown_role_is_an_error(products):
    roles, layers = build_vocab(products)
    with pytest.raises(ValueError, match="어휘에 없는"):
        product_tensors(products[0], 32, {"gate": 1}, layers)


def test_collate_pads_products_with_fewer_elements(products):
    roles, layers = build_vocab(products)
    full = product_tensors(products[0], 16, roles, layers)
    short = tuple(t[:2] for t in full)
    images, role_ids, _, valid, unit = collate([(*full, unit_feature((40.0, 40.0))), (*short, unit_feature((64.0, 64.0)))])
    assert images.shape == (2, 3, 3, 16, 16) and valid.tolist() == [[True, True, True], [True, True, False]]
    assert role_ids[1, 2] == 0 and unit.shape == (2, 2)


def test_augment_moves_all_elements_of_a_product_together(products):
    roles, layers = build_vocab(products)
    images, _, _ = product_tensors(products[0], 32, roles, layers)
    out = augment(images, torch.Generator().manual_seed(1))
    assert out.shape == images.shape and not torch.allclose(out, images)
    assert torch.allclose(out[0, 1], out[1, 1], atol=1e-6)  # 같은 변형이라 CNT 채널이 그대로 같음


def test_manifest_reads_npy_masks_and_rejects_paths_outside_its_folder(tmp_path):
    (tmp_path / "A").mkdir()
    np.save(tmp_path / "A" / "gate.npy", np.array([[0, 255], [255, 0]], dtype=np.uint8))
    np.save(tmp_path / "A" / "data.npy", np.zeros((2, 2), dtype=np.uint8))
    doc = {"products": [{"product_id": "A", "unit_size_um": [60, 60], "elements": [
        {"role": "gate", "layer": "layer1", "mask": "A/gate.npy"}, {"role": "data", "layer": "layer2", "mask": "A/data.npy"}]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    (product,) = load_manifest(tmp_path / "manifest.json")
    assert product.elements[0].mask.tolist() == [[0.0, 1.0], [1.0, 0.0]] and product.cnt is None
    doc["products"][0]["elements"][0]["mask"] = "../outside.npy"
    (tmp_path / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="폴더 밖"):
        load_manifest(tmp_path / "manifest.json")
