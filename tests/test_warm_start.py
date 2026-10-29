import copy
import dataclasses
import json
import random

import numpy as np
import pytest
import torch

from psdl1 import train as train_module
from psdl1.checkpoint import (extend_vocab, init_from_checkpoint, json_hash, read_checkpoint, validate_config, warm_start_state)
from psdl1.data import Element, build_vocab
from psdl1.model import ProductEncoder
from psdl1.train import RelationBatchSampler, delta_edges, encode_products, model_config, parent_delta_pairs, train

SMALL = {"dim": 32, "out_dim": 16, "heads": 2, "blocks": 1}
GRID = 64


def relations_for(products):
    similar, different = set(), set()
    for a in products:
        for b in products:
            if a.product_id < b.product_id:
                (similar if a.lineage_id == b.lineage_id else different).add(frozenset((a.product_id, b.product_id)))
    return similar, different


def with_extra(product):
    """새 role/layer 이름을 가진 element 하나를 더한다(어휘 확장용)."""
    mask = product.elements[0].mask.copy()
    return dataclasses.replace(product, elements=product.elements + (Element("extra", "layer3", mask),))


@pytest.fixture
def families(products):
    old = [p for p in products if p.lineage_id in {"bus", "comb"}]
    new = [with_extra(p) if p.lineage_id == "ring" else p for p in products]
    return old, new


def fit(tmp_path, name, items, **kw):
    kw = {"epochs": 1, "grid": GRID, "batch_size": 3, "relations": relations_for(items), "model_kwargs": SMALL, **kw}
    train(items, tmp_path / name, **kw)
    return tmp_path / name / "dl1.pt"


def test_extend_vocab_keeps_parent_ids_and_appends_sorted_new_names():
    parent = {"B": 1, "A": 2}
    assert extend_vocab(parent, ["A", "C", "B", "D"]) == {"B": 1, "A": 2, "C": 3, "D": 4}
    assert extend_vocab(parent, ["D", "C"]) == {"B": 1, "A": 2, "C": 3, "D": 4}
    assert extend_vocab(parent, ["A"]) == parent and parent == {"B": 1, "A": 2}  # 부모에만 있는 이름 유지, 입력 불변
    validate_config(model_config(8, extend_vocab(parent, ["C"]), {"L": 1}))


def test_warm_start_copies_parent_rows_and_keeps_every_other_tensor(families, tmp_path):
    old, new = families
    path = fit(tmp_path, "parent", old)
    parent = read_checkpoint(path)
    roles, layers = build_vocab(new)
    config = model_config(GRID, extend_vocab(parent["config"]["roles"], roles), extend_vocab(parent["config"]["layers"], layers), model_kwargs=SMALL)
    assert len(config["roles"]) > len(parent["config"]["roles"]) and len(config["layers"]) > len(parent["config"]["layers"])
    torch.manual_seed(5)
    model = ProductEncoder(len(config["roles"]), len(config["layers"]), **SMALL)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = (torch.get_rng_state(), random.getstate(), np.random.get_state()[1].copy())
    generator = torch.Generator().manual_seed(9)
    gen_state = generator.get_state()
    info = init_from_checkpoint(path, model, expected_config=config)
    # astra:A1 — 새 행은 호출 전 현재 모델의 행과 같고 RNG 는 소비되지 않는다.
    after = model.state_dict()
    for key, field in (("role.weight", "roles"), ("layer.weight", "layers")):
        n = len(parent["config"][field]) + 1
        assert after[key].shape[0] == len(config[field]) + 1 and torch.isfinite(after[key]).all()
        assert torch.equal(after[key][:n], parent["state_dict"][key])
        assert torch.equal(after[key][n:], before[key][n:])
    assert all(torch.equal(after[k], v) for k, v in parent["state_dict"].items() if k not in {"role.weight", "layer.weight"})
    assert torch.equal(rng[0], torch.get_rng_state()) and rng[1] == random.getstate() and np.array_equal(rng[2], np.random.get_state()[1])
    assert torch.equal(gen_state, generator.get_state())
    assert info["warm_start"]["roles"]["new_names"] == ["extra"] and info["warm_start"]["layers"]["parent"] == len(parent["config"]["layers"])
    # 부모 state 는 변하지 않는다(사본).
    assert read_checkpoint(path)["weights_hash"] == parent["weights_hash"]


def test_warm_start_state_rejects_a_vocabulary_that_does_not_extend_the_parent(families, tmp_path):
    old, new = families
    parent = read_checkpoint(fit(tmp_path, "parent", old))
    roles, layers = build_vocab(new)
    bad = model_config(GRID, {n: len(roles) + 1 - i for i, n in enumerate(roles, 1)}, extend_vocab(parent["config"]["layers"], layers), model_kwargs=SMALL)
    model = ProductEncoder(len(bad["roles"]), len(bad["layers"]), **SMALL)
    with pytest.raises(ValueError, match="init-from parent differs: roles"):
        warm_start_state(parent["state_dict"], parent["config"], bad, model.state_dict())


def test_old_product_embeddings_equal_parent_before_any_step(families, tmp_path):
    old, new = families
    path = fit(tmp_path, "parent", old)
    parent = read_checkpoint(path)
    roles, layers = build_vocab(new)
    config = model_config(GRID, extend_vocab(parent["config"]["roles"], roles), extend_vocab(parent["config"]["layers"], layers), model_kwargs=SMALL)
    child = ProductEncoder(len(config["roles"]), len(config["layers"]), **SMALL)
    init_from_checkpoint(path, child, expected_config=config)
    base = ProductEncoder(len(parent["config"]["roles"]), len(parent["config"]["layers"]), **SMALL)
    base.load_state_dict(parent["state_dict"])
    child.eval(), base.eval()
    with torch.no_grad():
        a = encode_products(child, old, GRID, config["roles"], config["layers"])
        b = encode_products(base, old, GRID, parent["config"]["roles"], parent["config"]["layers"])
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("case", ["resume", "weights", "backbone", "use_unit_size", "model_kwargs", "grid"])
def test_init_from_is_exclusive_and_checks_parent_compatibility(families, tmp_path, case):
    old, new = families
    path = fit(tmp_path, "parent", old)
    extra = {"resume": {"resume": str(path)}, "weights": {"weights_file": str(path)}, "backbone": {"backbone": "other"},
             "use_unit_size": {"use_unit_size": True}, "model_kwargs": {"model_kwargs": {**SMALL, "dim": 64}}, "grid": {"grid": 96}}[case]
    message = {"resume": "--init-from and --resume are exclusive", "weights": "--init-from and --weights are exclusive",
               "backbone": "init-from parent differs: backbone", "use_unit_size": "init-from parent differs: use_unit_size",
               "model_kwargs": "init-from parent differs: model_kwargs", "grid": "init-from parent differs: grid"}[case]
    out, events = tmp_path / "child", []
    with pytest.raises(ValueError) as error:
        train(new, out, epochs=1, progress_sink=events.append,
              **{"grid": GRID, "batch_size": 3, "relations": relations_for(new), "model_kwargs": SMALL, "init_from": path, **extra})
    assert str(error.value) == message
    assert not (out / "dl1.pt").exists()
    assert not any(e["phase"] == "prepare" for e in events)  # 제품 준비 전에 거부된다


def test_warm_started_run_resumes_with_the_same_arguments(families, tmp_path):
    old, new = families
    parent_path = fit(tmp_path, "parent", old)
    parent = read_checkpoint(parent_path)
    args = {"relations": relations_for(new), "model_kwargs": SMALL, "grid": GRID, "batch_size": 3, "new_edge_repeat": 2, "round_index": 2}
    path = fit(tmp_path, "warm", new, init_from=parent_path, **args)
    first = read_checkpoint(path)
    assert first["metadata"]["parent"]["weights_hash"] == parent["weights_hash"] and first["metadata"]["round"] == 2
    with pytest.raises(ValueError, match="contract differs"):
        train(new, tmp_path / "warm", epochs=2, resume=str(path), **{**args, "new_edge_repeat": 3})
    train(new, tmp_path / "warm", epochs=2, resume=str(path), **args)
    saved = read_checkpoint(path)
    assert saved["epoch"] == 2 and saved["metadata"]["parent"] == first["metadata"]["parent"] and saved["metadata"]["round"] == 2
    assert saved["metadata"]["contract"]["new_edge_repeat"] == 2 and saved["metadata"]["delta_negative_pairs"] == first["metadata"]["delta_negative_pairs"]


def reference_batches(sampler, generator):
    """d8001a7 의 RelationBatchSampler.batches 그대로."""
    edges = [sampler.edges[i] for i in torch.randperm(len(sampler.edges), generator=generator).tolist()]
    remaining = set(edges)
    result = []
    for edge in edges:
        if edge not in remaining:
            continue
        batch = list(edge)
        order = torch.randperm(len(sampler.ids), generator=generator).tolist()
        positives = [i for i in order if i not in batch and any(frozenset((sampler.ids[i], sampler.ids[j])) in sampler.similar for j in batch)]
        batch.extend((positives + [i for i in order if i not in batch and i not in positives])[:sampler.batch_size - len(batch)])
        members = set(batch)
        remaining = {e for e in remaining if not set(e) <= members}
        result.append(batch)
    return result


def sampler_for(products, batch_size=3, **kw):
    similar, different = relations_for(products)
    return RelationBatchSampler([p.product_id for p in products], batch_size, similar, different, **kw)


def test_sampler_with_no_repeat_matches_previous_batches(products):
    sampler = sampler_for(products)
    for seed in range(5):
        assert sampler.batches(torch.Generator().manual_seed(seed)) == reference_batches(sampler, torch.Generator().manual_seed(seed))
    explicit = sampler_for(products, repeat={})
    assert explicit.batches(torch.Generator().manual_seed(1)) == reference_batches(sampler, torch.Generator().manual_seed(1))


def test_sampler_repeat_validates_edges_and_values(products):
    sampler = sampler_for(products)
    for bad in ({(0, 0): 2}, {sampler.edges[0]: 0}, {sampler.edges[0]: True}, {sampler.edges[0]: 1.5}):
        with pytest.raises(ValueError, match="repeat"):
            sampler.repeat = bad


def test_new_edge_repeat_two_covers_each_delta_edge_twice(products, families, tmp_path):
    sampler = sampler_for(products)
    edges = list(sampler.edges)
    sampler.repeat = {edges[0]: 2, edges[-1]: 3}
    batches = sampler.batches(torch.Generator().manual_seed(0))
    count = lambda e: sum(1 for b in batches if e[0] in b and e[1] in b)
    assert count(edges[0]) >= 2 and count(edges[-1]) >= 3 and all(count(e) >= 1 for e in edges)
    assert sampler.edges == edges  # 유일 집합은 그대로
    old, new = families
    parent_path = fit(tmp_path, "parent", old)
    path = fit(tmp_path, "child", new, init_from=parent_path, new_edge_repeat=2)
    stats = read_checkpoint(path)["metadata"]["epochs"][0]
    assert stats["delta_negative_pairs"] == 8 and stats["new_edge_repeat"] == 2
    assert stats["delta_coverage_min"] >= 2 and stats["covered_negative_pairs"] == stats["required_negative_pairs"] == 12


def meta(train_ids, different, *, similar=(), unlabelled=False):
    return {"contract": {"similar": [list(p) for p in similar], "different": [list(p) for p in different], "unlabelled_negatives": unlabelled},
            "split": {"train": train_ids}, "input_records": {i: {"lineage": f"L{i}"} for i in train_ids}}


def test_delta_edges_are_pairs_the_parent_never_trained():
    ids = ["A", "B", "C"]
    different = {frozenset(p) for p in (("A", "B"), ("A", "C"), ("B", "C"))}
    sampler = RelationBatchSampler(ids, 3, set(), different)
    assert parent_delta_pairs(sampler, meta(["A", "B"], [("A", "B")])) == [["A", "C"], ["B", "C"]]
    # 옛 쌍에 새로 붙은 different 라벨도 delta.
    assert parent_delta_pairs(sampler, meta(["A", "B"], [])) == [["A", "B"], ["A", "C"], ["B", "C"]]
    # 부모 contract 에 train 밖 쌍이 있어도 부모 train 으로만 재구성한다.
    assert parent_delta_pairs(sampler, meta(["A", "B"], [("A", "B"), ("B", "Z")])) == [["A", "C"], ["B", "C"]]
    with pytest.raises(ValueError, match="parent metadata incomplete"):
        parent_delta_pairs(sampler, {"contract": {}})


@pytest.mark.parametrize("pairs,message", [([["A", "C"], ["C", "A"]], "duplicates"), ([["A", "Z"]], "unknown product"),
                                           ([["A", "A"]], "not a current negative edge"), ("A-C", "must be a list"), ([["A"]], "product ID pairs")])
def test_delta_edges_reject_malformed_pairs(pairs, message):
    sampler = RelationBatchSampler(["A", "B", "C"], 3, set(), {frozenset(("A", "C")), frozenset(("B", "C"))})
    with pytest.raises(ValueError, match=message):
        delta_edges(sampler, pairs)
    with pytest.raises(ValueError, match="not a current negative edge"):
        delta_edges(sampler, [["A", "B"]])  # 현재 negative 가 아닌 쌍
    assert delta_edges(sampler, [["C", "A"]]) == {(0, 2)}


def test_new_edge_repeat_requires_a_parent(families, tmp_path):
    old, new = families
    with pytest.raises(ValueError, match="needs a parent"):
        fit(tmp_path, "x", new, new_edge_repeat=2)
    path = fit(tmp_path, "plain", new)
    assert read_checkpoint(path)["metadata"]["parent"] is None  # parent 없는 체크포인트
    with pytest.raises(ValueError, match="needs a parent"):
        train(new, tmp_path / "plain", epochs=2, grid=GRID, batch_size=3, relations=relations_for(new), model_kwargs=SMALL, resume=str(path), new_edge_repeat=2)
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            fit(tmp_path, "y", new, new_edge_repeat=bad)


def test_parent_code_hash_mismatch_is_recorded_not_rejected(families, tmp_path):
    old, new = families
    path = fit(tmp_path, "parent", old)
    saved = read_checkpoint(path)
    assert fit(tmp_path, "same", new, init_from=path) and read_checkpoint(tmp_path / "same" / "dl1.pt")["metadata"]["parent"]["code_hash_matches"] is True
    saved["metadata"]["code_hash"] = "other"
    torch.save(saved, path)
    child = read_checkpoint(fit(tmp_path, "child", new, init_from=path))["metadata"]
    assert child["parent"]["code_hash_matches"] is False and child["parent"]["code_hash"] == "other"


def test_cli_init_from_round_and_repeat(families, tmp_path, monkeypatch, capsys):
    old, new = families
    parent_path = fit(tmp_path, "parent", old, model_kwargs=None)
    similar, different = relations_for(new)
    rel = tmp_path / "relations.csv"
    rel.write_text("".join(f"{sorted(p)[0]},{sorted(p)[1]},similar\n" for p in similar) + "".join(f"{sorted(p)[0]},{sorted(p)[1]},different\n" for p in different))
    split = tmp_path / "split.json"
    split.write_text(json.dumps(train_module.training_only_split(new)))
    monkeypatch.setattr(train_module, "load_manifest", lambda path: new)
    out = tmp_path / "cli"
    train_module.main(["--manifest", "m.json", "--split", str(split), "--out", str(out), "--epochs", "1", "--grid", str(GRID), "--batch-size", "3", "--relations", str(rel),
                       "--no-progress", "--init-from", str(parent_path), "--round", "2", "--new-edge-repeat", "2"])
    saved = read_checkpoint(out / "dl1.pt")
    assert saved["metadata"]["round"] == 2 and saved["metadata"]["contract"]["new_edge_repeat"] == 2
    assert saved["metadata"]["parent"]["weights_hash"] == read_checkpoint(parent_path)["weights_hash"]
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["checkpoint"] == str(out / "dl1.pt")


def tampered(path, tmp_path, mutate):
    saved = read_checkpoint(path)
    mutate(saved["metadata"])
    target = tmp_path / "tampered"
    target.mkdir(exist_ok=True)
    torch.save(saved, target / "dl1.pt")
    return target / "dl1.pt"


def test_deleted_parent_resume_matches_a_single_two_epoch_run(families, tmp_path, monkeypatch):
    """astra:X1 — 부모 파일을 지워도 resume 은 한 번에 2 epoch 학습한 결과와 batch·coverage·weights_hash 가 같다."""
    old, new = families
    parent_path = fit(tmp_path, "parent", old)
    recorded = []
    original = RelationBatchSampler.batches

    def spy(self, generator):
        result = original(self, generator)
        recorded.append(copy.deepcopy(result))
        return result

    monkeypatch.setattr(RelationBatchSampler, "batches", spy)
    args = {"relations": relations_for(new), "model_kwargs": SMALL, "grid": GRID, "batch_size": 3, "new_edge_repeat": 2, "round_index": 2}
    once = fit(tmp_path, "once", new, epochs=2, init_from=parent_path, **args)
    single, recorded[:] = list(recorded), []
    half = fit(tmp_path, "half", new, epochs=1, init_from=parent_path, **args)
    parent_path.unlink()
    train(new, tmp_path / "half", epochs=2, resume=str(half), **args)
    assert len(single) == 2 and recorded == single
    a, b = read_checkpoint(once), read_checkpoint(half)
    assert a["metadata"]["epochs"] == b["metadata"]["epochs"] and a["history"] == b["history"]
    assert a["weights_hash"] == b["weights_hash"] and a["metadata"]["contract_hash"] == b["metadata"]["contract_hash"]
    pairs = a["metadata"]["delta_negative_pairs"]
    assert pairs == sorted(sorted(p) for p in pairs) and all(isinstance(v, str) for p in pairs for v in p) and len(pairs) == 8
    assert a["metadata"]["contract"]["delta_negative_pairs_hash"] == json_hash(pairs)


@pytest.mark.parametrize("case", ["duplicate", "unknown", "not_negative", "other_subset"])
def test_resume_rejects_tampered_delta_pairs(families, tmp_path, case):
    old, new = families
    parent_path = fit(tmp_path, "parent", old)
    args = {"relations": relations_for(new), "model_kwargs": SMALL, "grid": GRID, "batch_size": 3, "new_edge_repeat": 2}
    path = fit(tmp_path, "warm", new, init_from=parent_path, **args)
    saved = read_checkpoint(path)["metadata"]["delta_negative_pairs"]
    change = {"duplicate": lambda m: m.update(delta_negative_pairs=saved + [saved[0]]),
              "unknown": lambda m: m.update(delta_negative_pairs=saved + [["nope-1", "nope-2"]]),
              "not_negative": lambda m: m.update(delta_negative_pairs=saved + [["bus-40", "bus-64"]]),
              "other_subset": lambda m: m.update(delta_negative_pairs=saved[:-1])}[case]
    with pytest.raises(ValueError, match="delta_negative_pairs|contract differs"):
        train(new, tmp_path / "out2", epochs=2, resume=str(tampered(path, tmp_path, change)), **args)
