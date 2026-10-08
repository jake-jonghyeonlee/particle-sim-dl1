"""합성 제품: 같은 배치(family)를 unit 크기만 달리해 만든다."""
import numpy as np
import pytest

from psdl1.data import Element, Product

# family마다 (역할, layer, 사각형들[x0, y0, x1, y1] — unit을 1로 본 비율)
FAMILIES = {
    "bus": [("gate", "layer1", [(0.0, 0.45, 1.0, 0.55)]), ("data", "layer2", [(0.45, 0.0, 0.55, 1.0)]), ("pixel", "layer2", [(0.1, 0.1, 0.4, 0.4)])],
    "comb": [("gate", "layer1", [(0.0, 0.1, 1.0, 0.2), (0.2, 0.2, 0.3, 0.9), (0.6, 0.2, 0.7, 0.9)]), ("data", "layer2", [(0.0, 0.7, 1.0, 0.8)]), ("pixel", "layer2", [(0.4, 0.3, 0.55, 0.6)])],
    "ring": [("gate", "layer1", [(0.1, 0.1, 0.9, 0.2), (0.1, 0.8, 0.9, 0.9), (0.1, 0.1, 0.2, 0.9), (0.8, 0.1, 0.9, 0.9)]), ("data", "layer2", [(0.45, 0.3, 0.55, 0.7)]), ("pixel", "layer2", [(0.3, 0.45, 0.7, 0.55)])],
}


def make_product(family: str, size_px: int, um_per_px: float = 0.05, product_id: str | None = None) -> Product:
    elements = []
    for role, layer, rects in FAMILIES[family]:
        m = np.zeros((size_px, size_px), dtype=np.float32)
        for x0, y0, x1, y1 in rects:
            m[round(y0 * size_px):round(y1 * size_px), round(x0 * size_px):round(x1 * size_px)] = 1.0
        elements.append(Element(role, layer, m))
    cnt = np.zeros((size_px, size_px), dtype=np.float32)
    cnt[round(0.48 * size_px):round(0.52 * size_px), round(0.48 * size_px):round(0.52 * size_px)] = 1.0
    unit = size_px * um_per_px
    return Product(product_id or f"{family}-{size_px}", (unit, unit), tuple(elements), cnt, lineage_id=family, design_version="synthetic-v1")


@pytest.fixture
def products():
    return [make_product(f, s) for f in FAMILIES for s in (40, 64)]


def pytest_addoption(parser):
    parser.addoption("--run-training", action="store_true", default=False,
                     help="Explicitly opt into T9 training tests (not part of T8)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-training"):
        for item in items:
            if item.name == "test_training_lowers_the_loss_and_search_finds_the_same_layout_at_another_unit_size":
                item.add_marker(pytest.mark.skip(reason="T9 train.py/search.py: training requires explicit --run-training; T8 forbids training"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def denied(*args, **kwargs):
        raise AssertionError("network is forbidden during input-contract validation")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def manifest_case(tmp_path):
    import json
    from psdl1.schema import SCHEMA_VERSION
    mask = np.ones((4, 4), dtype=np.uint8)
    np.save(tmp_path / "shape.npy", mask)
    np.save(tmp_path / "valid.npy", mask)
    doc = {"schema_version": SCHEMA_VERSION, "candidate_grid": 4, "products": [{
        "product_id": "P", "lineage_id": "family-P", "design_version": "v1",
        "unit_size_um": [0.2, 0.2], "frame": {"frame_id": "F", "length_unit": "um",
        "origin_xy_um": [1.0, -2.0], "pixel_pitch_um": 0.05, "shape_yx": [4, 4], "axis_order": "yx"},
        "valid": "valid.npy", "cnt": "shape.npy",
        "elements": [{"element_id": "A-L1", "electrode_id": "A", "role": "A", "actual_layer": "L1", "mask": "shape.npy", "observation": "observed"},
                     {"element_id": "B-L2", "electrode_id": "B", "role": "B", "actual_layer": "L2", "mask": "shape.npy", "observation": "observed"}],
        "connectivity_observation": "complete", "connections": [{"connection_id": "V1", "kind": "VIA",
        "observation": "observed", "mask": "shape.npy", "endpoints": [
        {"element_id": "A-L1", "xy_um": [1.025, -1.975]}, {"element_id": "B-L2", "xy_um": [1.025, -1.975]}]}]}]}
    path = tmp_path / "manifest.json"
    def write():
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path
    return doc, write, tmp_path


@pytest.fixture
def bridge_mask():
    # Two 8x8 blocks joined by exactly one 1-pixel-thick bridge.
    mask = np.zeros((16, 24), dtype=np.float32)
    mask[4:12, :8] = 1
    mask[4:12, 16:] = 1
    mask[7, 8:16] = 1
    return mask
