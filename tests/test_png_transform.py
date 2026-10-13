"""S17/S18/S26: full-frame channels and one continuous rounded transform."""
import numpy as np
import pytest
import torch

from psdl1.data import frame_transform, normalize_unit, prepare_product, build_vocab
from psdl1.train import structured_batch
from psdl1.png_contract import ModelMetadata
from png_helpers import write_stack_order
from psdl1.png_source import discover_source
from psdl1.png_connections import extract_model, to_product
from png_helpers import write_wide_import_fixture


def test_original_grid_original_roundtrip_with_rounding():
    t = frame_transform((1001, 1203), 32, 0.1)
    assert t.resized_shape_yx == (27, 32)
    assert t.padding_ltrb == (0, 2, 0, 3)
    assert t.scale_xy == (32 / 1203, 27 / 1001)
    assert t.to_record()["effective_content_pitch_xy_um"] == [0.1 * 1203 / 32, 0.1 / (27 / 1001)]
    for x, y in [(0, 0), (1202, 1000), (613.25, 509.125)]:
        gx, gy = t.pixel_to_grid(x, y)
        assert gx == (x + 0.5) * 32 / 1203
        assert gy == pytest.approx((y + 0.5) * 27 / 1001 + 2)
        assert t.grid_to_pixel(gx, gy) == pytest.approx((x, y), abs=2e-13)
    image = normalize_unit(np.ones((1001, 1203), bool), 32)
    assert np.array_equal(image[:2], np.zeros((2, 32)))
    assert np.array_equal(image[2:29], np.ones((27, 32)))
    assert np.array_equal(image[29:], np.zeros((3, 32)))


def test_S17_margin_elements_cnt_connections_and_valid_share_full_frame(tmp_path):
    write_wide_import_fixture(tmp_path)
    model, = discover_source(tmp_path, stack_order=write_stack_order(tmp_path.parent / (tmp_path.name + "-order.json"), ["U", "N", "C4", "U5", "C6", "L"])).models
    extraction = extract_model(model)
    p = to_product(model, extraction, metadata=ModelMetadata(model.model_id, "SYNTHETIC-FAMILY", "SYNTHETIC-V1"))
    roles, layers = build_vocab([p])
    batch, extra, prepared = structured_batch([p], 128, roles, layers)
    item, = prepared
    assert p.frame.shape_yx == (1401, 1603) and p.unit_size_um == pytest.approx((160.3, 140.1))
    assert p.cnt.sum() == 4096 and len(p.connections) == 3
    assert p.cnt[999, 999]  # unresolved component survives global CNT
    # Hand-computed rounded content height=112; padding top=bottom=8.
    assert frame_transform(p.frame.shape_yx, 128, .1).padding_ltrb == (0, 8, 0, 8)
    assert torch.equal(item.valid[:8], torch.zeros((8, 128), dtype=torch.bool))
    assert item.valid[8:120].all() and not item.valid[120:].any()
    # Margin contact centre (39,39) is far outside reference core but remains.
    assert item.images[:, 1, 10:12, 2:4].sum() > 0
    assert item.images[:, 0, 8:14, :18].sum() > 0
    assert item.connection_masks[:, 10:12, 2:4].sum() > 0
    expected = torch.tensor([(39.5 * 128 / 1603) / 128, (39.5 * 112 / 1401 + 8) / 128])
    assert torch.equal(extra["connection_features"][0, 0, 2:4], expected)
    assert torch.equal(extra["connection_features"][0, 0, 2:4], extra["connection_features"][0, 0, 4:6])
    assert item.images.shape == (5, 3, 128, 128)


def test_real_margin_patterns_reach_all_channels(tmp_path):
    from PIL import Image
    shape = (1401, 1603)
    # All four 500px margins, separated from each other and from the core.
    boxes = [(600, 20, 680, 100), (20, 600, 100, 680),
             (1500, 600, 1580, 680), (600, 1300, 680, 1380)]
    electrode = np.zeros(shape, bool); cnt = np.zeros(shape, bool)
    for x0, y0, x1, y1 in boxes:
        electrode[y0:y1, x0:x1] = True
        cnt[y0+24:y0+56, x0+24:x0+56] = True
    for folder, filename, mask in [("A", "1-U.png", electrode), ("B", "7-L.png", electrode), ("cno", "6_C.png", cnt)]:
        path = tmp_path / "M" / folder / filename; path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(path)
    source, = discover_source(tmp_path, stack_order=write_stack_order(tmp_path.parent / (tmp_path.name + "-order.json"), ["U", "C", "L"])).models
    extracted = extract_model(source)
    p = to_product(source, extracted, metadata=ModelMetadata("M", "SYNTHETIC-MARGIN", "SYNTHETIC-V1"))
    assert len(p.connections) == 4 and p.connectivity_observation == "complete"
    assert p.cnt.sum() == 4096
    assert [c.relation.representative_xy_px for c in extracted.components] == [(639, 59), (59, 639), (1539, 639), (639, 1339)]
    roles, layers = build_vocab([p]); item = prepare_product(p, 128, roles, layers)
    assert item.grid_check["reasons"] == []
    # Independent expected target cells from rounded 112×128 content/top 8.
    for y, x in [(12, 51), (59, 4), (59, 122), (115, 51)]:
        assert item.images[:, 0, y, x].sum() > 0  # self
        assert item.images[:, 2, y, x].sum() > 0  # overlapping rest survives exclude-self
        assert item.images[:, 1, y, x].sum() > 0  # global CNT
        assert item.valid[y, x]
        assert item.connection_masks[:, y, x].sum() > 0
    assert item.images[:, :, 64:70, 64:70].sum() == 0  # no centring onto the core


def test_shared_transform_matches_float32_endpoint_features_at_corners():
    from psdl1.schema import Product, Element, Connection, Endpoint, Frame
    mask = np.ones((1001, 1203), bool)
    elements = (Element("A", "U", mask, "A", "upper"), Element("B", "L", mask, "B", "lower"))
    points = [(0, 0), (1202, 1000), (613.25, 509.125)]
    connections = tuple(Connection(str(i), "CNT", (Endpoint("upper", ((x+.5)*.1, (y+.5)*.1)),
                                                   Endpoint("lower", ((x+.5)*.1, (y+.5)*.1))), mask)
                        for i, (x,y) in enumerate(points))
    p = Product("SYNTHETIC-CORNERS", (120.3, 100.1), elements, mask,
                "SYNTHETIC-FAMILY", "SYNTHETIC-V1", Frame("F", (1001, 1203), .1), mask, connections, "complete")
    roles, layers = build_vocab([p])
    for g, nh, top in [(16, 13, 1), (32, 27, 2)]:
        _, extra, prepared = structured_batch([p], g, roles, layers)
        assert prepared[0].grid_check["transform"] == frame_transform((1001, 1203), g, .1).to_record()
        hand = np.array([[(x+.5)/1203, ((y+.5)*nh/1001+top)/g] for x,y in points], np.float32)
        # Independent float32 model tolerance; float64 inverse above is 2e-13.
        np.testing.assert_allclose(extra["connection_features"][0, :, 2:4].numpy(), hand, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(extra["connection_features"][0, :, 2:4], extra["connection_features"][0, :, 4:6])
