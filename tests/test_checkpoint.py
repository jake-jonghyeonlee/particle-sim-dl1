import random
import numpy as np
import pytest
import torch

from psdl1.checkpoint import read_checkpoint, resume_checkpoint, save_checkpoint
from psdl1.data import build_vocab
from psdl1.model import ProductEncoder
from psdl1.search import embed, load_model
from psdl1.train import model_config


def fixture_model(products):
    roles, layers = build_vocab(products)
    config = model_config(16, roles, layers, model_kwargs={"dim": 32, "out_dim": 16, "heads": 2, "blocks": 1})
    model = ProductEncoder(len(roles), len(layers), **config["model_kwargs"])
    return model, config


def test_round_trip_identical_embeddings(products, tmp_path):
    model, config = fixture_model(products)
    model.eval()
    before = embed(model, config, products, batch_size=2)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, config)
    loaded, cfg = load_model(path)
    after = embed(loaded, cfg, products, batch_size=2)
    assert torch.equal(before, after)
    assert read_checkpoint(path)["weights_hash"]


@pytest.mark.parametrize("field,value", [("schema_version", "old"), ("format_version", "old"),
                                          ("model_input_version", "old"), ("fingerprint", "wrong"), ("weights_hash", "wrong")])
def test_incompatible_schema_and_fingerprint_rejected(products, tmp_path, field, value):
    model, config = fixture_model(products)
    path = tmp_path / "model.pt"
    saved = save_checkpoint(path, model, config)
    saved[field] = value
    torch.save(saved, path)
    with pytest.raises(ValueError, match="incompatible|hash"):
        read_checkpoint(path)


def test_resume_restores_optimizer_and_all_rng_states(products, tmp_path):
    model, config = fixture_model(products)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # Direct optimizer-state fixture, not a dataset training run.
    loss = sum(p.square().mean() for p in model.parameters())
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    generator = torch.Generator().manual_seed(123)
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    path = tmp_path / "resume.pt"
    saved = save_checkpoint(path, model, config, optimizer=optimizer, generator=generator, epoch=1,
                            metadata={"input_hash": "fixture"})
    expected = (torch.rand(3), torch.rand(3, generator=generator), random.random(), np.random.rand(3))
    other, _ = fixture_model(products)
    other_opt = torch.optim.AdamW(other.parameters(), lr=9)
    other_gen = torch.Generator().manual_seed(9)
    resume_checkpoint(path, other, other_opt, other_gen, expected_config=config, expected_input_hash="fixture")
    actual = (torch.rand(3), torch.rand(3, generator=other_gen), random.random(), np.random.rand(3))
    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])
    assert actual[2] == expected[2] and np.array_equal(actual[3], expected[3])
    assert other_opt.param_groups[0]["lr"] == 0.001
    for key, state in saved["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(other_opt.state_dict()["state"][key][name], value)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), other.parameters()))


def test_resume_rejects_changed_input(products, tmp_path):
    model, config = fixture_model(products)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, config, optimizer=opt, metadata={"input_hash": "before"})
    with pytest.raises(ValueError, match="input hash"):
        resume_checkpoint(path, model, opt, torch.Generator(), expected_input_hash="after")


@pytest.mark.parametrize("expected", ["contract", "code"])
def test_resume_rejects_changed_contract_before_rng_restoration(products, tmp_path, expected):
    model, config = fixture_model(products)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, config, optimizer=opt, metadata={"contract_hash": "before", "code_hash": "before"})
    before = torch.get_rng_state().clone()
    kwargs = {"expected_contract_hash" if expected == "contract" else "expected_code_hash": "after"}
    with pytest.raises(ValueError, match="differs"):
        resume_checkpoint(path, model, opt, torch.Generator(), **kwargs)
    assert torch.equal(before, torch.get_rng_state())
