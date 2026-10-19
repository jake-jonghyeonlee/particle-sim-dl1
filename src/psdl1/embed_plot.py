"""학습된 임베딩 사이의 거리를 보여주는 오프라인 HTML 보고서(외부 라이브러리·네트워크 없음).

색은 구획(train/validation/test)을 뜻하고 모양을 함께 써서 색만으로 구분하지 않는다. 같은 계열은 가는 선으로 잇는다.
2D 투영은 거리를 왜곡할 수 있으므로 코사인 거리 표와 히트맵이 기준이다."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from .data import _iter_manifest_products, input_identity
from .search import embed, load_model, vocabulary_reason

PARTITIONS = ("train", "validation", "test")
SHAPES = {"train": "circle", "validation": "square", "test": "triangle"}


def collect_embeddings(model, cfg, manifest, split, partitions=PARTITIONS):
    """제품을 하나씩 읽어 임베딩한다(메모리 절약). 학습에서 못 본 role/layer가 있는 제품은 건너뛰고 사유를 남긴다."""
    part_of = {pid: part for part in PARTITIONS for pid in split.get(part, [])}
    records = model.training_metadata.get("input_records", {})
    rows, skipped = [], []
    for product in _iter_manifest_products(manifest, grid=cfg["grid"]):
        part = part_of.get(product.product_id, "unassigned")
        if part not in partitions:
            continue
        reason = vocabulary_reason(product, cfg)
        if reason:
            skipped.append({"product_id": product.product_id, "partition": part, **reason})
            continue
        z = embed(model, cfg, [product])[0].numpy().astype(np.float64)
        trained = records.get(product.product_id, {}).get("identity")
        rows.append({"product_id": product.product_id, "lineage": product.lineage_id, "partition": part, "z": z,
                     "matches_checkpoint": None if trained is None else trained == input_identity(product)})
    return rows, skipped


def cosine_distances(z):
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    return np.clip(1.0 - z @ z.T, 0.0, 2.0)


def pca_2d(z):
    centred = z - z.mean(0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    total = float((s ** 2).sum())
    return coords, [float(v) / total if total else 0.0 for v in (s[:2] ** 2)] + [0.0] * (2 - len(s[:2]))


def mds_2d(d):
    """거리 행렬을 가장 잘 보존하는 2D 배치(고전적 MDS)."""
    n = len(d)
    j = np.eye(n) - np.ones((n, n)) / n
    values, vectors = np.linalg.eigh(-0.5 * j @ (d ** 2) @ j)
    top = np.argsort(values)[::-1][:2]
    coords = vectors[:, top] * np.sqrt(np.clip(values[top], 0, None))
    return np.pad(coords, ((0, 0), (0, 2 - coords.shape[1]))) if coords.shape[1] < 2 else coords


def separation_stats(rows, d):
    """같은 계열 안과 계열 사이의 평균 거리, 가장 가까운 이웃이 같은 계열인 비율."""
    lineage = [r["lineage"] for r in rows]
    same = [d[i, j] for i in range(len(rows)) for j in range(i + 1, len(rows)) if lineage[i] == lineage[j]]
    other = [d[i, j] for i in range(len(rows)) for j in range(i + 1, len(rows)) if lineage[i] != lineage[j]]
    nearest, hits = [], 0
    for i in range(len(rows)):
        peers = [j for j in range(len(rows)) if j != i]
        j = min(peers, key=lambda k: d[i, k]) if peers else None
        nearest.append(None if j is None else {"product_id": rows[j]["product_id"], "distance": float(d[i, j])})
        if j is not None and any(lineage[k] == lineage[i] for k in peers):
            hits += lineage[j] == lineage[i]
    eligible = sum(any(lineage[k] == lineage[i] for k in range(len(rows)) if k != i) for i in range(len(rows)))
    return {"mean_distance_same_lineage": float(np.mean(same)) if same else None,
            "mean_distance_other_lineage": float(np.mean(other)) if other else None,
            "nearest_neighbour_same_lineage_rate": hits / eligible if eligible else None,
            "products_with_a_same_lineage_peer": eligible}, nearest


# ---- HTML -------------------------------------------------------------------------------------------------------
# 구획 3색은 skill의 검증된 범주 슬롯 1-3(전 쌍 검증 통과). 히트맵은 한 색(파랑) 순차 단계.
CSS = """
.viz-root{color-scheme:light;--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--grid:#e3e2de;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
--h0:#cde2fb;--h1:#9ec5f4;--h2:#6da7ec;--h3:#3987e5;--h4:#256abf;--h5:#184f95;--h6:#0d366b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;--grid:#2e2e2c;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--h0:#0d366b;--h1:#184f95;--h2:#256abf;--h3:#3987e5;--h4:#6da7ec;--h5:#9ec5f4;--h6:#cde2fb}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;--grid:#2e2e2c;--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--h0:#0d366b;--h1:#184f95;--h2:#256abf;--h3:#3987e5;--h4:#6da7ec;--h5:#9ec5f4;--h6:#cde2fb}
body{margin:0}.viz-root{background:var(--surface);color:var(--text);font:14px/1.5 system-ui,'Segoe UI','Malgun Gothic',sans-serif;padding:24px;min-height:100vh}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 6px}p.sub{color:var(--text2);margin:0 0 16px;max-width:900px}
.stats{display:flex;gap:32px;flex-wrap:wrap;margin:12px 0}.stat b{display:block;font-size:24px;font-weight:600}.stat span{color:var(--text2);font-size:12px}
.legend{display:flex;gap:18px;margin:8px 0;color:var(--text2);font-size:13px}.legend svg{vertical-align:-3px;margin-right:6px}
.panels{display:flex;gap:24px;flex-wrap:wrap}svg text{fill:var(--text2);font-size:11px}svg .lab{fill:var(--text)}
.p1{fill:var(--s1)}.p2{fill:var(--s2)}.p3{fill:var(--s3)}.mark{stroke:var(--surface);stroke-width:2}.link{stroke:var(--text2);stroke-width:1;opacity:.45;fill:none}
.axis{stroke:var(--grid);stroke-width:1}.cell{stroke:var(--surface);stroke-width:2}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--text);border:1px solid var(--grid);border-radius:6px;padding:6px 10px;font-size:12px;display:none;box-shadow:0 2px 8px rgba(0,0,0,.18);z-index:9}
details{margin-top:24px}table{border-collapse:collapse;font-size:12px}td,th{padding:3px 10px;border-bottom:1px solid var(--grid);text-align:left}th{color:var(--text2);font-weight:500}td.n{text-align:right;font-variant-numeric:tabular-nums}
.warn{color:var(--text2);border-left:3px solid var(--s2);padding-left:10px;margin:12px 0}
"""
JS = """const t=document.getElementById('tip');document.querySelectorAll('[data-tip]').forEach(e=>{e.addEventListener('mousemove',v=>{t.textContent=e.dataset.tip;t.style.display='block';t.style.left=(v.clientX+14)+'px';t.style.top=(v.clientY+14)+'px'});e.addEventListener('mouseleave',()=>{t.style.display='none'})});"""
PCLASS = {"train": "p1", "validation": "p2", "test": "p3", "unassigned": "p1"}


def marker(kind, x, y, cls, r=6):
    if kind == "square":
        return f'<rect class="mark {cls}" x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="2"/>'
    if kind == "triangle":
        return f'<polygon class="mark {cls}" points="{x:.1f},{y - r - 1:.1f} {x + r + 1:.1f},{y + r:.1f} {x - r - 1:.1f},{y + r:.1f}"/>'
    return f'<circle class="mark {cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'


def scatter_svg(rows, coords, title, xlabel, ylabel, width=560, height=440, pad=54):
    xs, ys = coords[:, 0], coords[:, 1]
    span = max(float(np.ptp(xs)), float(np.ptp(ys)), 1e-9)
    scale = min(width - 2 * pad, height - 2 * pad) / span  # 한 축척: 두 축의 거리 비율을 유지
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    px = lambda v: width / 2 + (v - cx) * scale
    py = lambda v: height / 2 - (v - cy) * scale
    pts = [(px(c[0]), py(c[1])) for c in coords]
    out = [f'<svg width="{width}" height="{height}" role="img" aria-label="{html.escape(title)}"><text class="lab" x="0" y="14" style="font-size:13px;font-weight:600">{html.escape(title)}</text>',
           f'<line class="axis" x1="{pad}" y1="{height - 22}" x2="{width - pad}" y2="{height - 22}"/><line class="axis" x1="{pad - 20}" y1="{pad}" x2="{pad - 20}" y2="{height - 36}"/>',
           f'<text x="{width / 2}" y="{height - 6}" text-anchor="middle">{html.escape(xlabel)}</text>',
           f'<text transform="translate(12 {height / 2}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>']
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["lineage"], []).append(i)
    for members in groups.values():  # 같은 계열: 가는 선
        for a, b in zip(members, members[1:]):
            out.append(f'<line class="link" x1="{pts[a][0]:.1f}" y1="{pts[a][1]:.1f}" x2="{pts[b][0]:.1f}" y2="{pts[b][1]:.1f}"/>')
    placed = []
    for i, r in enumerate(rows):
        x, y = pts[i]
        w = 6.3 * len(r["product_id"])
        for dx, dy in ((9, -8), (9, 14), (-9 - w, -8), (-9 - w, 14), (-w / 2, -14), (-w / 2, 22)):
            box = (x + dx, y + dy - 10, x + dx + w, y + dy + 3)
            if not any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3] for o in placed):
                break
        placed.append(box)
        out.append(f'<text class="lab" x="{x + dx:.1f}" y="{y + dy:.1f}">{html.escape(r["product_id"])}</text>')
    for i, r in enumerate(rows):
        x, y = pts[i]
        tip = f'{r["product_id"]} · {r["partition"]} · lineage {r["lineage"]}'
        out.append(marker(SHAPES.get(r["partition"], "circle"), x, y, PCLASS[r["partition"]]))
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="transparent" data-tip="{html.escape(tip)}"/>')  # 마크보다 큰 hover 영역
    out.append("</svg>")
    return "".join(out)


def heatmap_svg(rows, d, order):
    n = len(order)
    cell = max(10, min(26, 640 // max(n, 1)))
    left, top = 130, 120
    top_value = max(float(d.max()), 1e-9)
    out = [f'<svg width="{left + n * cell + 90}" height="{top + n * cell + 10}" role="img" aria-label="cosine distance heatmap">']
    for k, i in enumerate(order):
        out.append(f'<text class="lab" x="{left - 6}" y="{top + k * cell + cell * 0.7:.1f}" text-anchor="end">{html.escape(rows[i]["product_id"])}</text>')
        out.append(f'<text class="lab" transform="translate({left + k * cell + cell * 0.65:.1f} {top - 6}) rotate(-60)">{html.escape(rows[i]["product_id"])}</text>')
    for a, i in enumerate(order):
        for b, j in enumerate(order):
            level = min(6, int(d[i, j] / top_value * 7))
            tip = f'{rows[i]["product_id"]} ↔ {rows[j]["product_id"]}: {d[i, j]:.3f}' + (" (같은 계열)" if rows[i]["lineage"] == rows[j]["lineage"] and i != j else "")
            out.append(f'<rect class="cell" x="{left + b * cell}" y="{top + a * cell}" width="{cell}" height="{cell}" rx="2" style="fill:var(--h{level})" data-tip="{html.escape(tip)}"/>')
    out.append("</svg>")
    return "".join(out)


def legend_html(partitions):
    items = []
    for part in PARTITIONS:
        if part in partitions:
            items.append(f'<span><svg width="16" height="16">{marker(SHAPES[part], 8, 8, PCLASS[part], 5)}</svg>{part}</span>')
    return '<div class="legend">' + "".join(items) + '<span>— 같은 계열</span></div>'


def fmt(value, digits=3):
    """거리는 매우 작을 수 있어(임베딩이 한 점으로 몰릴 때) 유효숫자로 표시한다. 비율은 소수 자리로 표시한다."""
    if value is None:
        return "–"
    return f"{value:.{digits}f}" if digits == 2 else f"{value:.3g}"


def render_html(rows, d, pca, explained, mds, stats, nearest, skipped, meta):
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["lineage"], rows[i]["product_id"]))
    present = {r["partition"] for r in rows}
    table_rows = "".join(
        f'<tr><td>{html.escape(r["product_id"])}</td><td>{html.escape(r["lineage"])}</td><td>{r["partition"]}</td>'
        f'<td>{html.escape(nearest[i]["product_id"]) if nearest[i] else "–"}</td><td class="n">{fmt(nearest[i]["distance"]) if nearest[i] else "–"}</td>'
        f'<td>{"예" if r["matches_checkpoint"] else "아니오" if r["matches_checkpoint"] is False else "–"}</td></tr>' for i, r in enumerate(rows))
    matrix = ("<table><tr><th></th>" + "".join(f'<th>{html.escape(rows[j]["product_id"])}</th>' for j in order) + "</tr>" +
              "".join(f'<tr><th>{html.escape(rows[i]["product_id"])}</th>' + "".join(f'<td class="n">{d[i, j]:.3f}</td>' for j in order) + "</tr>" for i in order) + "</table>")
    collapse_html = "" if float(d.max()) >= 0.02 else ('<p class="warn">모든 코사인 거리가 0.02 미만입니다. 임베딩이 거의 한 점으로 몰려 있을 수 있어(학습 부족 또는 collapse), '
                                                      '아래 히트맵 색은 이 아주 작은 차이를 최대값 기준으로 늘려 보여줍니다. 거리 값을 확인하세요.</p>')
    skipped_html = "" if not skipped else '<p class="warn">학습에서 못 본 role/layer 때문에 제외된 제품: ' + html.escape(", ".join(f'{s["product_id"]}({s["reason"]})' for s in skipped)) + "</p>"
    stat = lambda label, value: f'<div class="stat"><b>{value}</b><span>{label}</span></div>'
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>DL1 임베딩 거리</title><style>{CSS}</style></head>
<body><div class="viz-root"><h1>DL1 임베딩 거리</h1>
<p class="sub">체크포인트 에폭 {meta['epoch']} · 제품 {len(rows)}개. 거리는 단위 벡터 사이 코사인 거리(0=같은 방향)입니다. 아래 2D 그림은 {meta['dim']}차원을 줄인 것이라 거리를 왜곡할 수 있으니, 정확한 값은 히트맵과 표를 보세요.</p>
<div class="stats">{stat("같은 계열 평균 거리", fmt(stats["mean_distance_same_lineage"]))}{stat("다른 계열 평균 거리", fmt(stats["mean_distance_other_lineage"]))}
{stat(f"가장 가까운 제품이 같은 계열인 비율 (대상 {stats['products_with_a_same_lineage_peer']}개)", fmt(stats["nearest_neighbour_same_lineage_rate"], 2))}</div>
<p class="sub">같은 계열이 다른 계열보다 가까우면 학습이 설계 단위로 모양을 구분한다는 신호입니다. 다만 test 제품끼리만 보면 계열이 한 제품뿐이라 비율이 비어 있을 수 있습니다.</p>
{collapse_html}{skipped_html}{legend_html(present)}
<div class="panels">{scatter_svg(rows, pca, "PCA 투영", f"PC1 ({explained[0] * 100:.0f}%)", f"PC2 ({explained[1] * 100:.0f}%)")}{scatter_svg(rows, mds, "거리 보존 투영 (MDS)", "MDS 1", "MDS 2")}</div>
<h2>코사인 거리 히트맵 (계열 순서)</h2><p class="sub">색이 밝을수록(다크 모드에서는 어두울수록) 가깝습니다. 마우스를 올리면 값이 보입니다.</p>{heatmap_svg(rows, d, order)}
<details><summary>표로 보기</summary><h2>제품별 가장 가까운 제품</h2><table><tr><th>제품</th><th>계열</th><th>구획</th><th>가장 가까운 제품</th><th>거리</th><th>체크포인트 입력과 일치</th></tr>{table_rows}</table>
<h2>전체 거리 행렬</h2>{matrix}</details><div id="tip"></div></div><script>{JS}</script></body></html>"""


def build_report(ckpt, manifest, out, *, partitions=PARTITIONS, device="cpu", allow_grid_override=False):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}; choose a new folder")
    model, cfg = load_model(ckpt)
    candidate = json.loads(Path(manifest).read_text(encoding="utf-8"))["candidate_grid"]
    if candidate != cfg["grid"] and not allow_grid_override:
        raise ValueError(f"checkpoint grid {cfg['grid']} conflicts with manifest candidate_grid {candidate}; --allow-grid-override required")
    model.to(device)
    split = model.training_metadata.get("split")
    if not isinstance(split, dict):
        raise ValueError("checkpoint has no split provenance")
    rows, skipped = collect_embeddings(model, cfg, manifest, split, partitions)
    if len(rows) < 2:
        raise ValueError("at least two embedded products are required")
    z = np.stack([r["z"] for r in rows])
    d = cosine_distances(z)
    pca, explained = pca_2d(z)
    stats, nearest = separation_stats(rows, d)
    meta = {"epoch": model.trained_epochs, "dim": z.shape[1]}
    out.mkdir(parents=True)
    (out / "embedding_map.html").write_text(render_html(rows, d, pca, explained, mds_2d(d), stats, nearest, skipped, meta), encoding="utf-8")
    with open(out / "embeddings.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["product_id", "lineage_id", "partition", *[f"z{i}" for i in range(z.shape[1])]])
        writer.writerows([r["product_id"], r["lineage"], r["partition"], *[f"{v:.8g}" for v in r["z"]]] for r in rows)
    with open(out / "distances.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["product_id", *[r["product_id"] for r in rows]])
        writer.writerows([r["product_id"], *[f"{v:.6f}" for v in d[i]]] for i, r in enumerate(rows))
    summary = {**stats, **meta, "products": len(rows), "skipped": skipped, "pca_explained_variance": explained[:2],
               "inputs_differing_from_checkpoint": [r["product_id"] for r in rows if r["matches_checkpoint"] is False]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Offline HTML report of embedding distances for a trained DL1 checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="new folder: embedding_map.html, embeddings.csv, distances.csv, summary.json")
    ap.add_argument("--partitions", nargs="+", default=list(PARTITIONS), choices=PARTITIONS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-grid-override", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(build_report(a.ckpt, a.manifest, a.out, partitions=tuple(a.partitions), device=a.device, allow_grid_override=a.allow_grid_override), ensure_ascii=False))


if __name__ == "__main__":
    main()
