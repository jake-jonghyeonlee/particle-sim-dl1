"""relations.csv 초안 생성. 같은 계열은 similar, 다른 계열은 different 후보로 두되
이미지 거리가 가장 가까운(판단이 어려운) 쌍은 빼서 모름으로 남기고 검토 목록에 적는다.

이미지 거리는 role/layer로 맞춘 낮은 해상도 마스크의 평균 절대 차이다. 학습된 값이 아니라 판단을 돕는 지표일 뿐이다."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .data import _iter_manifest_products
from .losses import validate_relations
from .pretrain import check_split_ids, element_images
from .split import validate_split

PARTITIONS = ("train", "validation", "test")
REVIEW_FIELDS = ("product_a", "product_b", "partition", "relation", "image_distance", "lineage_a", "lineage_b", "note")


def signature(product, grid):
    """(role, layer) → 요소 마스크 [g,g]. 요소 자신의 채널만 쓴다."""
    images, _ = element_images(product, grid)
    return {(e.role, e.layer): images[i, 0] for i, e in enumerate(product.elements)}


def image_distance(a, b):
    zero = np.zeros_like(next(iter(a.values())))
    return float(np.mean([np.abs(a.get(k, zero) - b.get(k, zero)).mean() for k in sorted(a.keys() | b.keys())]))


def draft_relations(manifest, split, *, partitions=("train",), grid=64, hold_out_nearest=0.0):
    if not 0 <= hold_out_nearest < 1 or type(grid) is not int or grid <= 0:
        raise ValueError("hold_out_nearest must be in [0,1) and grid a positive integer")
    if not partitions or any(p not in PARTITIONS for p in partitions):
        raise ValueError(f"partitions must be chosen from {PARTITIONS}")
    document = json.loads(Path(manifest).read_text(encoding="utf-8"))
    check_split_ids(split, [p["product_id"] for p in document["products"]])
    wanted = {pid for part in partitions for pid in split[part]}
    stubs, signatures, lineage = [], {}, {}
    for product in _iter_manifest_products(manifest, grid=grid):  # 끝까지 읽어 manifest 전체를 검증한다
        stubs.append(SimpleNamespace(product_id=product.product_id, lineage_id=product.lineage_id))
        lineage[product.product_id] = product.lineage_id
        if product.product_id in wanted:
            signatures[product.product_id] = signature(product, grid)
    validate_split(stubs, split)  # 같은 계열이 두 구획에 걸치면 여기서 막는다
    rows = []
    for part in partitions:
        ids = sorted(split[part])
        candidates = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                row = {"product_a": a, "product_b": b, "partition": part, "lineage_a": lineage[a], "lineage_b": lineage[b],
                       "image_distance": round(image_distance(signatures[a], signatures[b]), 6), "note": ""}
                if lineage[a] == lineage[b]:
                    rows.append({**row, "relation": "similar"})
                else:
                    candidates.append(row)
        candidates.sort(key=lambda r: (r["image_distance"], r["product_a"], r["product_b"]))
        held = math.floor(hold_out_nearest * len(candidates))
        for k, row in enumerate(candidates):
            rows.append({**row, "relation": "unknown" if k < held else "different", "note": "held_out_nearest" if k < held else ""})
    rows.sort(key=lambda r: (r["partition"], r["product_a"], r["product_b"]))
    similar = {frozenset((r["product_a"], r["product_b"])) for r in rows if r["relation"] == "similar"}
    different = {frozenset((r["product_a"], r["product_b"])) for r in rows if r["relation"] == "different"}
    validate_relations(similar, different, product_ids=list(lineage))
    return rows


def summarize(rows, split, partitions):
    counts = {r: sum(row["relation"] == r for row in rows) for r in ("similar", "different", "unknown")}
    uncovered = {}
    for part in partitions:
        with_different = {pid for row in rows if row["partition"] == part and row["relation"] == "different" for pid in (row["product_a"], row["product_b"])}
        uncovered[part] = sorted(set(split[part]) - with_different)
    distances = [row["image_distance"] for row in rows if row["relation"] in {"different", "unknown"}]
    return {"pairs": counts, "products_without_different_pair": uncovered,
            "cross_lineage_distance": None if not distances else {"min": min(distances), "median": float(np.median(distances)), "max": max(distances)}}


def write_outputs(rows, out, review):
    out, review = Path(out), Path(review)
    for path in (out, review):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}; choose a new path")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([r["product_a"], r["product_b"], r["relation"]] for r in rows if r["relation"] in {"similar", "different"})
    with open(review, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["relation"] == "similar", r["image_distance"])))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draft relations.csv from lineage; ambiguous cross-lineage pairs stay unknown. Review the draft before training.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, help="new relations.csv (header-free, as psdl1.train expects)")
    ap.add_argument("--review-out", help="new review CSV with distances (default: <out>.review.csv)")
    ap.add_argument("--partitions", nargs="+", default=["train"], choices=PARTITIONS, help="pairs are made inside each partition only")
    ap.add_argument("--grid", type=int, default=64, help="low resolution used for the image distance")
    ap.add_argument("--hold-out-nearest", type=float, default=0.0, help="fraction of the closest cross-lineage pairs per partition left unknown (e.g. 0.3)")
    a = ap.parse_args(argv)
    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    rows = draft_relations(a.manifest, split, partitions=tuple(a.partitions), grid=a.grid, hold_out_nearest=a.hold_out_nearest)
    review = a.review_out or str(a.out) + ".review.csv"
    write_outputs(rows, a.out, review)
    print(json.dumps({"relations": a.out, "review": review, **summarize(rows, split, a.partitions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
