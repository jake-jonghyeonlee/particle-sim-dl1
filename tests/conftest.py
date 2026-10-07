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


def make_product(family: str, size_px: int, um_per_px: float = 1.0, product_id: str | None = None) -> Product:
    elements = []
    for role, layer, rects in FAMILIES[family]:
        m = np.zeros((size_px, size_px), dtype=np.float32)
        for x0, y0, x1, y1 in rects:
            m[round(y0 * size_px):round(y1 * size_px), round(x0 * size_px):round(x1 * size_px)] = 1.0
        elements.append(Element(role, layer, m))
    cnt = np.zeros((size_px, size_px), dtype=np.float32)
    cnt[round(0.48 * size_px):round(0.52 * size_px), round(0.48 * size_px):round(0.52 * size_px)] = 1.0
    unit = size_px * um_per_px
    return Product(product_id or f"{family}-{size_px}", (unit, unit), tuple(elements), cnt)


@pytest.fixture
def products():
    return [make_product(f, s) for f in FAMILIES for s in (40, 64)]
