import pytest
import torch

from psdl1.data import build_vocab, collate, product_tensors, unit_feature
from psdl1.losses import NEGATIVE, POSITIVE, UNKNOWN, masked_contrastive_loss, relation_matrix
from psdl1.model import ProductEncoder, build_backbone
from psdl1.search import load_model, rank
from psdl1.train import train

from conftest import make_product


def batch_of(products, grid=32):
    roles, layers = build_vocab(products)
    items = [(*product_tensors(p, grid, roles, layers), unit_feature(p.unit_size_um)) for p in products]
    return roles, layers, collate(items)


def test_product_feature_has_unit_length_and_ignores_element_order(products):
    roles, layers, (images, role_ids, layer_ids, valid, unit) = batch_of(products[:2])
    model = ProductEncoder(len(roles), len(layers)).eval()
    z = model(images, role_ids, layer_ids, valid, unit)
    assert z.shape == (2, 128) and torch.allclose(z.norm(dim=1), torch.ones(2), atol=1e-5)
    perm = torch.tensor([2, 0, 1])
    z2 = model(images[:, perm], role_ids[:, perm], layer_ids[:, perm], valid[:, perm], unit)
    assert torch.allclose(z, z2, atol=1e-5)


def test_padding_slots_do_not_change_the_feature(products):
    roles, layers, (images, role_ids, layer_ids, valid, unit) = batch_of(products[:1])
    model = ProductEncoder(len(roles), len(layers)).eval()
    z = model(images, role_ids, layer_ids, valid, unit)
    pad_img = torch.cat([images, torch.rand(1, 2, *images.shape[2:])], dim=1)
    pad = lambda t: torch.cat([t, torch.zeros(1, 2, dtype=t.dtype)], dim=1)
    z2 = model(pad_img, pad(role_ids), pad(layer_ids), pad(valid), unit)
    assert torch.allclose(z, z2, atol=1e-5)


def test_unit_size_condition_is_optional_and_required_when_enabled(products):
    roles, layers, (images, role_ids, layer_ids, valid, unit) = batch_of(products[:2])
    model = ProductEncoder(len(roles), len(layers), use_unit_size=True).eval()
    assert model(images, role_ids, layer_ids, valid, unit).shape == (2, 128)
    with pytest.raises(ValueError, match="unit 크기"):
        model(images, role_ids, layer_ids, valid, None)


def test_relation_matrix_without_and_with_labels():
    ids = ["A", "B", "C"]
    assert relation_matrix(ids).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    rel = relation_matrix(ids, similar={frozenset(("A", "B"))}, different={frozenset(("A", "C"))})
    assert rel[0].tolist() == [POSITIVE, POSITIVE, NEGATIVE]
    assert rel[1].tolist() == [POSITIVE, POSITIVE, UNKNOWN]


def test_loss_is_low_when_aligned_and_ignores_unknown_pairs():
    z = torch.nn.functional.normalize(torch.eye(3), dim=1)
    rel = relation_matrix(["A", "B", "C"])
    assert masked_contrastive_loss(z, z, rel)[0] < 0.01
    shuffled = z[[1, 2, 0]]
    loss, rows = masked_contrastive_loss(z, shuffled, rel)
    assert loss > 1.0 and rows == 3
    unknown = rel.clone()
    unknown[rel == NEGATIVE] = UNKNOWN
    loss, rows = masked_contrastive_loss(z, shuffled, unknown)
    assert loss == 0.0 and rows == 0  # 구분할 쌍이 없는 행은 계산에서 뺌
    partly = rel.clone()
    partly[0, 1] = UNKNOWN  # 0행은 2열과만 구분, 1열은 모름
    full_rows = masked_contrastive_loss(z, shuffled, rel)[0]
    assert masked_contrastive_loss(z, shuffled, partly)[1] == 3 and masked_contrastive_loss(z, shuffled, partly)[0] != full_rows
    with pytest.raises(ValueError, match="가까워야 하는 쌍"):
        masked_contrastive_loss(z, z, torch.zeros(3, 3, dtype=torch.long))


def test_training_lowers_the_loss_and_search_finds_the_same_layout_at_another_unit_size(products, tmp_path):
    families = ("bus", "comb", "ring")
    similar = {frozenset((f"{f}-40", f"{f}-64")) for f in families}
    different = {frozenset((f"{f}-{a}", f"{g}-{b}")) for f in families for g in families if f != g for a in (40, 64) for b in (40, 64)}
    history = train(products, tmp_path, epochs=40, grid=32, batch_size=6, lr=3e-4, relations=(similar, different))
    assert history[-1] < history[0]
    model, cfg = load_model(tmp_path / "dl1.pt")
    query = make_product("comb", 96, product_id="new")       # 학습에 없던 unit 크기
    ranked = rank(model, cfg, query, products)
    assert {ranked[0][0], ranked[1][0]} == {"comb-40", "comb-64"}
    assert ranked[1][1] > ranked[2][1] + 0.1  # 다른 배치와는 분명히 떨어짐 (모두 같은 특징으로 쏠리지 않음)


def test_labels_without_any_different_pair_are_rejected_instead_of_collapsing(products, tmp_path):
    similar = {frozenset(("bus-40", "bus-64"))}
    with pytest.raises(ValueError, match="구분할 쌍"):
        train(products, tmp_path, epochs=1, grid=32, relations=(similar, set()))


def test_training_needs_at_least_two_products(products, tmp_path):
    with pytest.raises(ValueError, match="둘 이상"):
        train(products[:1], tmp_path, epochs=1)


def test_tiny_backbone_has_no_pretrained_weights():
    with pytest.raises(ValueError, match="사전학습 가중치가 없다"):
        build_backbone("tiny", weights_file="x.pth")


def test_timm_backbone_builds_without_network_when_installed():
    pytest.importorskip("timm")
    net = build_backbone("convnextv2_atto")
    assert net(torch.zeros(1, 3, 64, 64)).shape == (1, net.num_features)
