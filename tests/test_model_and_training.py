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
    assert relation_matrix(ids, unlabelled_negatives=True, lineage_ids=ids).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    rel = relation_matrix(ids, similar={frozenset(("A", "B"))}, different={frozenset(("A", "C"))})
    assert rel[0].tolist() == [POSITIVE, POSITIVE, NEGATIVE]
    assert rel[1].tolist() == [POSITIVE, POSITIVE, UNKNOWN]


def test_loss_is_low_when_aligned_and_ignores_unknown_pairs():
    z = torch.nn.functional.normalize(torch.eye(3), dim=1)
    rel = relation_matrix(["A", "B", "C"], unlabelled_negatives=True, lineage_ids=["A", "B", "C"])
    assert masked_contrastive_loss(z, z, rel)[0] < 0.01
    shuffled = z[[1, 2, 0]]
    loss, rows = masked_contrastive_loss(z, shuffled, rel)
    assert loss > 1.0 and rows == 3
    unknown = rel.clone()
    unknown[rel == NEGATIVE] = UNKNOWN
    loss, rows = masked_contrastive_loss(z, shuffled, unknown)
    assert loss == 0.0 and rows == 0  # 구분할 쌍이 없는 행은 계산에서 뺌
    partly = rel.clone()
    partly[0, 1] = partly[1, 0] = UNKNOWN  # 0행은 2열과만 구분, 1열은 모름
    full_rows = masked_contrastive_loss(z, shuffled, rel)[0]
    assert masked_contrastive_loss(z, shuffled, partly)[1] == 3 and masked_contrastive_loss(z, shuffled, partly)[0] != full_rows
    with pytest.raises(ValueError, match="가까워야 하는 쌍"):
        masked_contrastive_loss(z, z, torch.zeros(3, 3, dtype=torch.long))


def test_training_lowers_the_loss_and_search_finds_the_same_layout_at_another_unit_size(products, tmp_path):
    families = ("bus", "comb", "ring")
    similar = {frozenset((f"{f}-40", f"{f}-64")) for f in families}
    different = {frozenset((f"{f}-{a}", f"{g}-{b}")) for f in families for g in families if f != g for a in (40, 64) for b in (40, 64)}
    history = train(products, tmp_path, epochs=40, grid=64, batch_size=6, lr=3e-4, relations=(similar, different))
    assert history[-1] < history[0]
    from psdl1.checkpoint import read_checkpoint
    assert read_checkpoint(tmp_path / "dl1.pt")["metadata"]["grid_gate"] == {"policy": "strict_default", "bypass_requested": False, "status": "passed", "failures": []}
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
    with pytest.raises(ValueError, match="local weights file"):
        build_backbone("tiny", weights_file="x.pth")


def test_timm_backbone_builds_without_network_when_installed():
    pytest.importorskip("timm")
    net = build_backbone("convnextv2_atto")
    assert net(torch.zeros(1, 3, 64, 64)).shape == (1, net.num_features)



def test_singleton_training_forward_supported_and_invalid_inputs_rejected(products):
    roles, layers, batch = batch_of(products[:1])
    model = ProductEncoder(len(roles), len(layers)).train()
    assert model(*batch).shape == (1, 128)
    bad = list(batch); bad[3] = torch.zeros_like(batch[3])
    with pytest.raises(ValueError, match="valid element"):
        model(*bad)
    bad = list(batch); bad[0] = torch.full_like(batch[0], float("nan"))
    with pytest.raises(ValueError, match="finite"):
        model(*bad)
    with pytest.raises(ValueError, match="dimensions"):
        ProductEncoder(3, 2, dim=7, heads=2)


def test_structured_validity_and_endpoint_inputs_are_consumed(manifest_case):
    from dataclasses import replace
    from psdl1.data import load_manifest
    from psdl1.train import encode_products
    _, write, _ = manifest_case
    p, = load_manifest(write())
    roles, layers = build_vocab([p])
    torch.manual_seed(3)
    model = ProductEncoder(len(roles), len(layers)).eval()
    before = encode_products(model, [p], 8, roles, layers)
    c = p.connections[0]
    moved = replace(p, connections=(replace(c, endpoints=(replace(c.endpoints[0], xy_um=(1.175, -1.975)), c.endpoints[1])),))
    after = encode_products(model, [moved], 8, roles, layers)
    assert not torch.allclose(before, after, atol=1e-7)  # mechanics, never ranking/performance
    valid = p.valid.copy(); valid[3, 3] = 0
    hole = replace(p, valid=valid)
    assert not torch.allclose(before, encode_products(model, [hole], 8, roles, layers), atol=1e-7)


def test_crop_augmentation_is_rejected_before_any_training(products, tmp_path):
    with pytest.raises(ValueError, match="augmentation"):
        train(products, tmp_path, epochs=1, grid=32, augmentation="crop")


@pytest.mark.parametrize("field,value", [("epochs", 0), ("batch_size", 1), ("grid", -1),
                                          ("lr", float("nan")), ("temperature", 0), ("seed", -1)])
def test_training_argument_errors_precede_any_optimizer_step(products, tmp_path, field, value):
    with pytest.raises(ValueError):
        train(products, tmp_path, relations=(set(), {frozenset(("bus-40", "comb-40"))}), **{**{"grid": 32}, field: value})


def test_json_config_and_cli_override_without_training(manifest_case, tmp_path, monkeypatch, capsys):
    import json
    from psdl1.train import main
    from psdl1.split import SPLIT_VERSION
    _, write, _ = manifest_case
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"schema_version": SPLIT_VERSION, "train": ["P"], "validation": [], "test": []}))
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps({"manifest": str(write()), "split": str(split_path), "out": str(tmp_path / "out"), "epochs": 5, "grid": 4, "use_unit_size": False}))
    received = {}
    def fake_train(*args, **kwargs):
        received["epochs"] = args[2]
        received["unit"] = args[8]
        received.update(kwargs)
        return [0.5]
    monkeypatch.setattr("psdl1.train.train", fake_train)
    main(["--config", str(config_path), "--epochs", "7", "--use-unit-size"])
    assert received["epochs"] == 7 and received["unit"] is True
    assert received["split_manifest"]["train"] == ["P"]
    assert json.loads(capsys.readouterr().out)["last_loss"] == 0.5


def test_explicit_empty_split_cannot_fall_back_to_training_all_products(products, tmp_path):
    with pytest.raises(ValueError, match="split schema"):
        train(products, tmp_path, grid=32, split_manifest={}, relations=(set(), {frozenset(("bus-40", "comb-40"))}))


def test_train_reports_prepare_and_epoch_progress(products, tmp_path):
    events = []
    similar = {frozenset(("bus-40", "bus-64"))}
    different = {frozenset(("bus-40", "comb-40")), frozenset(("bus-64", "ring-64")), frozenset(("comb-64", "ring-40"))}
    train(products, tmp_path, epochs=2, grid=64, batch_size=3, relations=(similar, different), progress_sink=events.append, progress_interval=0)
    phases = [(e["phase"], e["state"]) for e in events]
    assert phases[0] == ("prepare", "started") and phases.count(("prepare", "product_prepared")) == len(products)
    assert ("train", "started") in phases and phases.count(("train", "epoch_completed")) == 2
    running = [e for e in events if e["state"] == "running"]
    assert running and running[0]["epoch"] == 1 and running[-1]["epoch"] == 2 and running[-1]["step"] == running[-1]["steps"]
    train(products, tmp_path / "silent", epochs=1, grid=64, batch_size=3, relations=(similar, different))  # sink is optional
