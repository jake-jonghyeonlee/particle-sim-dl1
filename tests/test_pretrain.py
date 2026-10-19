import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from conftest import FAMILIES, make_product
from psdl1.data import build_vocab, load_manifest, prepare_product
from psdl1.model import ProductEncoder, build_backbone
from psdl1.pretrain import IN_CHANS, MaskedPretrainer, element_images, patch_mask, pretrain, reconstruction_loss
from psdl1.schema import SCHEMA_VERSION
from psdl1.split import SPLIT_VERSION


def write_manifest(root, products, grid):
    docs = []
    for p in products:
        size = next(iter(p.elements)).mask.shape[0]
        np.save(root / f"{p.product_id}-valid.npy", np.ones((size, size), dtype=np.uint8))
        np.save(root / f"{p.product_id}-cnt.npy", p.cnt.astype(np.uint8))
        elements = []
        for e in p.elements:
            np.save(root / f"{p.product_id}-{e.role}.npy", e.mask.astype(np.uint8))
            elements.append({"element_id": f"{e.role}@{e.layer}", "electrode_id": e.role, "role": e.role, "actual_layer": e.layer,
                             "observation": "observed", "mask": f"{p.product_id}-{e.role}.npy"})
        docs.append({"product_id": p.product_id, "lineage_id": p.lineage_id, "design_version": "v1", "unit_size_um": list(p.unit_size_um),
                     "frame": {"frame_id": "F", "length_unit": "um", "origin_xy_um": [0.0, 0.0], "pixel_pitch_um": 0.05,
                               "shape_yx": [size, size], "axis_order": "yx"},
                     "valid": f"{p.product_id}-valid.npy", "cnt": f"{p.product_id}-cnt.npy", "elements": elements,
                     "connectivity_observation": "unavailable", "connections": []})
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "candidate_grid": grid, "products": docs}))
    return path


@pytest.fixture
def case(tmp_path):
    products = [make_product(f, s) for f in FAMILIES for s in (40, 64)]
    manifest = write_manifest(tmp_path, products, 16)
    split = {"schema_version": SPLIT_VERSION, "train": ["bus-40", "bus-64"], "validation": ["comb-40", "comb-64"], "test": ["ring-40", "ring-64"]}
    return manifest, split, tmp_path / "out"


def test_element_images_match_the_training_input(case):
    manifest, _, _ = case
    for product in load_manifest(manifest):
        roles, layers = build_vocab([product])
        expected = prepare_product(product, 16, roles, layers)
        images, valid = element_images(product, 16)
        assert np.allclose(images, expected.images.numpy(), atol=1e-5)
        assert np.allclose(valid, expected.valid.numpy())


def test_patch_mask_ratio_alignment_and_reproducibility():
    a = patch_mask(5, 32, 8, 0.5, torch.Generator().manual_seed(1))
    assert a.shape == (5, 1, 32, 32) and set(a.unique().tolist()) <= {0.0, 1.0}
    blocks = a.view(5, 4, 8, 4, 8).permute(0, 1, 3, 2, 4).reshape(5, 16, 64)
    assert ((blocks.min(-1).values == blocks.max(-1).values)).all()  # whole patches only
    assert (blocks[:, :, 0].sum(1) == 8).all()  # half of 16 patches
    assert torch.equal(a, patch_mask(5, 32, 8, 0.5, torch.Generator().manual_seed(1)))
    assert (patch_mask(3, 16, 8, 0.99).flatten(1).amin(1) == 0).all()  # a visible patch always remains
    with pytest.raises(ValueError):
        patch_mask(1, 8, 8, 0.5)


def test_loss_only_counts_masked_observed_pixels():
    target = torch.zeros(1, 3, 8, 8)
    target[..., :4] = 1
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., 4:] = 1
    valid = torch.ones(1, 1, 8, 8)
    wrong_on_visible = torch.where(target > 0, torch.tensor(-9.0), torch.tensor(9.0))
    right_on_masked = wrong_on_visible.clone()
    right_on_masked[..., 4:] = -9.0
    assert reconstruction_loss(right_on_masked, target, mask, valid, 5.0) < 1e-3
    valid[..., 4:] = 0
    assert reconstruction_loss(wrong_on_visible, target, mask, valid, 5.0) == 0


def test_pretraining_exports_backbone_weights_without_touching_test(case):
    manifest, split, out = case
    history = pretrain(manifest, split, out, epochs=6, batch_size=3, lr=1e-3, mask_ratio=0.5, seed=0)
    assert len(history) == 6 and all(np.isfinite(history)) and history[-1] < history[0]
    index = json.loads((out / "cache" / "index.json").read_text())
    assert set(index["products"]) == {"bus-40", "bus-64", "comb-40", "comb-64"}  # no test product cached
    assert len(list((out / "cache").glob("*.npy"))) == 8
    record = json.loads((out / "history.json").read_text())[-1]
    assert record["validation"]["elements"] == 6 and record["validation"]["loss"] > 0
    meta = json.loads((out / "pretrain.json").read_text())
    assert meta["split"] == split and "ring-40" not in json.dumps(meta["config"]) and meta["config"]["stride"] == 8
    weights = torch.load(out / "backbone.pt", weights_only=True)
    encoder = ProductEncoder(3, 2, "tiny", weights_file=str(out / "backbone.pt"))  # strict load, as psdl1.train --weights does
    for key, value in weights.items():
        assert torch.equal(encoder.backbone.state_dict()[key], value)


def test_resume_matches_an_uninterrupted_run_and_rejects_changes(case, tmp_path):
    manifest, split, out = case
    full = pretrain(manifest, split, tmp_path / "full", epochs=3, batch_size=3, mask_ratio=0.5, seed=2)
    pretrain(manifest, split, out, epochs=2, batch_size=3, mask_ratio=0.5, seed=2)
    resumed = pretrain(manifest, split, out, epochs=3, batch_size=3, mask_ratio=0.5, seed=2, resume=out / "pretrain.pt")
    assert np.allclose(full, resumed, rtol=1e-5)
    with pytest.raises(ValueError, match="resume"):
        pretrain(manifest, split, out, epochs=4, batch_size=2, mask_ratio=0.5, seed=2, resume=out / "pretrain.pt")


def test_lineage_leakage_and_grid_conflict_rejected(case):
    manifest, split, out = case
    leaking = dict(split, train=["bus-40"], validation=["bus-64", "comb-40", "comb-64"])
    with pytest.raises(ValueError, match="lineage leakage"):
        pretrain(manifest, leaking, out, epochs=1)
    with pytest.raises(ValueError, match="allow-grid-override"):
        pretrain(manifest, split, out, epochs=1, grid=32)


def test_patch_must_fit_backbone_stride_and_grid(case):
    manifest, split, out = case
    with pytest.raises(ValueError, match="stride"):
        pretrain(manifest, split, out, epochs=1, patch=4)


def test_timm_backbone_uses_its_own_stride():
    pytest.importorskip("timm")
    net = MaskedPretrainer(build_backbone("convnextv2_atto", IN_CHANS), 32)
    assert net.stride == 32
    images, valid = torch.rand(2, 3, 64, 64), torch.ones(2, 1, 64, 64)
    mask = patch_mask(2, 64, 32, 0.5)
    assert net(images, valid, mask).shape == (2, 3, 64, 64)
