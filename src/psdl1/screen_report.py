"""스크리닝 검토 보고서(review.html) 렌더링. 판정·정렬·지표는 screen() 이 이미 계산한 값을 읽기만 하고 다시 계산하지 않는다.

맨 위에 지표별 분포 그림(인라인 SVG)과 검토 대상 전체 표를 놓고, 카드마다 두 정렬(기존 CNT/전극, Layer)의 근거, A·B 전체 프레임 썸네일,
모든 binding 의 지표 표, 정렬 후 Layer 층별 A | B | 차이 타일을 보여 준다. 외부 URL 이 없는 한 파일이고 이미지는 data URI 다.
카드는 하나씩 만들고 버린다(이미지를 쌍 수만큼 모아 두지 않는다)."""
from __future__ import annotations

import base64
import html
import io

import numpy as np

from . import screen_relations as sr

TILE = 168
THUMB = 256
DENSE_POINTS = 2000  # 이보다 점이 많으면 strip 대신 히스토그램
HIST_BINS = 20
DIST_METRICS = ("similarity", "similarity_layer", "usage_jaccard", "usage_jaccard_electrodes", "electrode_similarity", "pattern_similarity", "cnt_similarity")
PRIMARY_PRIORITY = ("pattern_similarity", "electrode_similarity", "similarity_layer", "similarity", "cnt_similarity", "usage_jaccard_electrodes", "usage_jaccard")  # AUC 동률일 때 앞선 것
COLOR_A, COLOR_B, COLOR_BOTH = (57, 135, 229, 255), (235, 104, 52, 255), (122, 122, 118, 255)
LAYER_COLORS = ((57, 135, 229), (235, 104, 52), (42, 157, 110), (192, 80, 160), (200, 160, 30), (40, 170, 190), (140, 100, 220), (210, 90, 90), (110, 150, 60), (160, 110, 70))
STATUS_TEXT = {"both": "양쪽 사용", "only_a": "A만 사용", "only_b": "B만 사용"}

REVIEW_CSS = """
:root{color-scheme:light dark;--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--grid:#e3e2de;--a:#3987e5;--b:#eb6834;--both:#7a7a76;--same:#2a9d6e}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;--grid:#2e2e2c}}
body{margin:0;background:var(--surface);color:var(--text);font:14px/1.5 system-ui,'Segoe UI','Malgun Gothic',sans-serif;padding:24px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:14px 0 6px}h3{font-size:13px;margin:10px 0 4px;color:var(--text2);font-weight:600}p.sub{color:var(--text2);max-width:900px;margin:0 0 12px}
.legend{display:flex;gap:18px;color:var(--text2);margin:8px 0 18px;flex-wrap:wrap}.legend i{display:inline-block;width:12px;height:12px;margin-right:6px;border-radius:2px;vertical-align:-1px}
.card{border:1px solid var(--grid);border-radius:8px;padding:14px;margin:0 0 16px}.card header{display:flex;gap:8px 12px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}
.badge{color:var(--text2);font-size:12px;border:1px solid var(--grid);border-radius:10px;padding:1px 8px}.line{display:flex;gap:6px 8px;flex-wrap:wrap;margin:2px 0 6px;align-items:baseline}.line b{font-size:12px;min-width:84px}
.thumbs{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0}.thumb{display:block;border:1px solid var(--grid);background:var(--surface);image-rendering:pixelated;max-width:100%}
.thumbs figure{margin:0}.thumbs figcaption{font-size:11px;color:var(--text2)}.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 3px 0 8px;vertical-align:-1px}
.tiles{display:flex;gap:10px;flex-wrap:wrap}.tile{width:168px}.tile img{display:block;width:168px;height:168px;background:var(--surface);border:1px solid var(--grid);image-rendering:pixelated}
.tile span{display:block;font-size:11px;color:var(--text2);word-break:break-all}table{border-collapse:collapse;font-size:12px;margin-bottom:12px}
td,th{padding:3px 10px;border-bottom:1px solid var(--grid);text-align:left}th{color:var(--text2);font-weight:500}td.n{text-align:right;font-variant-numeric:tabular-nums}
.bar{display:inline-block;width:60px;height:8px;background:var(--grid);border-radius:2px;vertical-align:middle;margin-left:6px}.bar i{display:block;height:8px;background:var(--a);border-radius:2px}
.triple{display:flex;gap:4px;margin:0 0 8px;align-items:flex-start;flex-wrap:wrap}.triple .tile{width:168px}.triples{display:flex;gap:18px;flex-wrap:wrap}.triples .group{max-width:100%}
ol.key{margin:4px 0 8px;padding-left:20px}ol.key li{margin:1px 0}ol.key li.warn{color:#d33;font-weight:600}ol.key b{display:inline-block;min-width:64px;color:var(--text2);font-weight:600}p.primary b{color:var(--same)}
details{margin:6px 0}summary{cursor:pointer;color:var(--text2);font-size:12px}
svg.dist{max-width:100%;height:auto}svg.dist text{fill:var(--text);font:11px system-ui,sans-serif}svg.dist .muted{fill:var(--text2)}svg.dist line.grid{stroke:var(--grid)}
svg.dist .pt-same{fill:var(--same);fill-opacity:.7}svg.dist .pt-other{fill:var(--both);fill-opacity:.55}svg.dist .floor{stroke:#d33;stroke-width:1.5}svg.dist .floor-group{stroke:#d33;stroke-width:1;stroke-dasharray:3 3}
"""
REASON_TEXT = {"close_to_same_lineage": "다른 계열인데 같은 계열 쌍만큼 비슷함: 같은 회로일 수 있음", "near_boundary": "경계 부근: 판단 필요",
               "weakest_same_lineage": "같은 계열 쌍 중 유사도가 가장 낮은 쌍: 정말 비슷한지 확인", "different_near_boundary": "자동으로 different가 됐지만 기준에 가까움: 정말 다른지 확인",
               "ambiguous_alignment": "정렬 후보가 여러 개(반복 패턴 등): 유사도를 덜 믿을 것", "alignment_unreliable": "기준 마스크가 잘 맞지 않음: 유사도를 덜 믿을 것",
               "low_overlap": "두 프레임이 겹치는 영역이 작음: 유사도를 덜 믿을 것", "no_anchor": "정렬 기준이 없어 프레임 중심으로 맞춤"}
TIER_TEXT = {"similar": "similar", "different": "different", "unknown": "모름"}


def esc(value):
    return html.escape(str(value))


def num(value, digits=3):
    return "–" if value is None else f"{value:.{digits}f}"


def data_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def tile_uri(rgba, size=TILE):
    """RGBA 배열을 정사각형 캔버스(왼쪽 위 정렬, 남는 곳 투명)에 담는다. size 보다 작으면 원래 해상도 그대로 두고 CSS 가 키운다(파일이 작다)."""
    from PIL import Image
    image = Image.fromarray(rgba, "RGBA")
    height, width = rgba.shape[:2]
    if max(height, width) > size:
        scale = size / max(height, width)
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.NEAREST)
    side = max(image.size) if max(height, width) <= size else size
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(image, (0, 0))
    return data_uri(canvas)


def solid_rgba(mask, color):
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[mask] = color
    return rgba


def diff_rgba(a, b):
    """공통 회색, A만 파랑, B만 주황(범례와 같은 색)."""
    rgba = np.zeros(a.shape + (4,), dtype=np.uint8)
    rgba[a & ~b], rgba[b & ~a], rgba[a & b] = COLOR_A, COLOR_B, COLOR_BOTH
    return rgba


def overlay_png(a, b, size=TILE):
    """겹친 영역(ROI)을 비율을 유지한 채 size 안에 그린다. 남는 부분은 투명하다."""
    return tile_uri(diff_rgba(a, b), size)


def layer_stack_rgba(masks, colors):
    """층 마스크를 색으로 겹친다(나중 층이 위). masks: [(이름, bool 배열)], colors: {이름: (r, g, b)}."""
    rgba = np.zeros(masks[0][1].shape + (4,), dtype=np.uint8)
    for name, mask in masks:
        rgba[mask] = colors[name] + (255,)
    return rgba


def thumb_html(rgba, box, label):
    """전체 프레임 썸네일(≤256 px). box 는 선택 정렬에서 두 프레임이 겹치는 ROI 의 래스터 좌표 (y0, y1, x0, x1) 이고 None 이면 그리지 않는다."""
    from PIL import Image, ImageDraw
    height, width = rgba.shape[:2]
    scale = min(1.0, THUMB / max(height, width))
    image = Image.fromarray(rgba, "RGBA")
    if scale < 1:
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.NEAREST)
    if box is not None:
        y0, y1, x0, x1 = box
        ImageDraw.Draw(image).rectangle([x0 * scale, y0 * scale, max(x0 * scale, x1 * scale - 1), max(y0 * scale, y1 * scale - 1)], outline=(220, 40, 60, 255))
    zoom = max(1, THUMB // max(image.size))
    return (f'<figure><img class="thumb" alt="{esc(label)}" src="{data_uri(image)}" width="{image.size[0] * zoom}" height="{image.size[1] * zoom}">'
            f'<figcaption>{esc(label)}</figcaption></figure>')


def selected_shift(p, sa):
    """metrics_anchor 위치의 래스터 이동. LAYER 면 scores.csv 에 쓰이는 µm 값을 래스터 칸으로 되돌린다(정수 이동 × um_per_px 였다)."""
    if p["metrics_anchor"] == "LAYER" and p["layer_shift_y_um"] is not None:
        return round(p["layer_shift_y_um"] / sa["um_per_px"]), round(p["layer_shift_x_um"] / sa["um_per_px"])
    return p["result"]["shift"]


def pair_layers(sa, sb):
    """두 제품 모두 Layer 가 있으면 두 제품의 층 이름 합집합(정렬), 아니면 None."""
    la, lb = sa.get("layers"), sb.get("layers")
    return None if not la or not lb else sorted(la.keys() | lb.keys())


def layer_mask(sig, name, view, valid):
    layer = (sig.get("layers") or {}).get(name)
    return np.zeros(valid.shape, bool) if layer is None else layer["mask"][view] & valid


def dice_of(a, b):
    total = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    return 1.0 if total == 0 else 2 * int(np.count_nonzero(a & b)) / total


def triple_html(caption, a, b):
    images = [(solid_rgba(a, COLOR_A), "A"), (solid_rgba(b, COLOR_B), "B"), (diff_rgba(a, b), "차이")]
    tiles = "".join(f'<div class="tile"><img alt="{esc(caption)} {label}" src="{tile_uri(rgba)}"><span>{label}</span></div>' for rgba, label in images)
    return f'<div class="group"><h3>{esc(caption)}</h3><div class="triple">{tiles}</div></div>'


def layer_items(sa, sb, names, shift):
    """정렬 후 층별 (차이=1 − 래스터 Dice, 이름, A 마스크, B 마스크), 차이가 큰 순. 두 프레임이 겹치지 않으면 None."""
    views = sr.roi_views(sa, sb, shift[0], shift[1])
    if views is None:
        return None
    _, a_slice, b_slice, valid = views
    items = []
    for name in names:
        a, b = layer_mask(sa, name, a_slice, valid), layer_mask(sb, name, b_slice, valid)
        if a.any() or b.any():
            items.append((1.0 - dice_of(a, b), name, a, b))
    items.sort(key=lambda item: (-item[0], item[1]))
    return items


def layer_triples(p, items, top_layers):
    """정렬 후 층별 A | B | 차이. 차이(1 − 래스터 Dice)가 큰 층 top_layers 개는 펼치고 나머지는 접는다."""
    if items is None:
        return "<p class=\"sub\">선택한 정렬에서 두 프레임이 겹치지 않아 층별 비교를 그리지 않습니다.</p>"
    caption = lambda dist, name: f"{name} · Dice {1 - dist:.2f}"
    shown = "".join(triple_html(caption(d, n), a, b) for d, n, a, b in items[:top_layers])
    rest = "".join(triple_html(caption(d, n), a, b) for d, n, a, b in items[top_layers:])
    folded = f'<details><summary>나머지 {len(items) - top_layers}층 펼치기</summary><div class="triples">{rest}</div></details>' if len(items) > top_layers else ""
    where = "Layer 정렬" if p["metrics_anchor"] == "LAYER" else f"{p['metrics_anchor']} 정렬(Layer 이동 미사용)"
    return (f'<h3>정렬 후 Layer 층별 A | B | 차이 — {esc(where)} 위치, 차이가 큰 층 순</h3><div class="triples">{shown or "<p class=sub>비교할 층이 없습니다.</p>"}</div>{folded}')


def binding_triples(p, sa, sb, shift, top_layers):
    """Layer 가 없을 때 binding 단위 A | B | 차이(compare 거리가 큰 순 top_layers 개)."""
    views = sr.roi_views(sa, sb, shift[0], shift[1])
    if views is None:
        return ""
    _, a_slice, b_slice, valid = views
    parts = sorted(p["result"]["bindings"], key=lambda item: (-item[2], item[0]))[:top_layers]

    def roi(sig, key, view):
        return np.zeros(valid.shape, bool) if key not in sig["rasters"] else sig["rasters"][key]["mask"][view] & valid
    shown = "".join(triple_html(f"{k} · 거리 {d:.2f}", roi(sa, k, a_slice), roi(sb, k, b_slice)) for k, s, d in parts)
    return (f'<h3>정렬 후 binding별 A | B | 차이 — Layer 가 없어 binding 단위, {esc(p["metrics_anchor"])} 정렬 위치, 거리가 큰 순</h3>'
            f'<div class="triples">{shown or "<p class=sub>비교할 binding이 없습니다.</p>"}</div>')


def frame_thumbs(p, sa, sb, names, shift):
    """A·B 전체 프레임 썸네일 둘. Layer 가 있으면 층 색 겹침, 없으면 전극 합집합(회색)과 CNT(색). 선택 정렬의 ROI 박스를 그린다."""
    box = sr.roi_box(sa["shape"], sb["shape"], shift[0], shift[1])
    box_b = None if box is None else (box[0] - shift[0], box[1] - shift[0], box[2] - shift[1], box[3] - shift[1])
    if names is not None:
        colors = {name: LAYER_COLORS[i % len(LAYER_COLORS)] for i, name in enumerate(names)}

        def paint(sig):
            present = [(n, sig["layers"][n]["mask"]) for n in names if n in sig["layers"]]
            return layer_stack_rgba(present, colors) if present else np.zeros(sig["shape"] + (4,), np.uint8)
        legend = "".join(f'<span class="swatch" style="background:rgb{colors[n]}"></span>{esc(n)}' for n in names)
        note = "Layer 색 겹침"
    else:
        def paint(sig):
            rgba = solid_rgba(sig["electrodes"], COLOR_BOTH)
            rgba[sig["rasters"][sr.CNT]["mask"]] = COLOR_A
            return rgba
        legend = '<span class="swatch" style="background:rgb(122,122,118)"></span>전극 합집합<span class="swatch" style="background:rgb(57,135,229)"></span>CNT'
        note = "전극 합집합 + CNT"
    label_a, label_b = f"A {p['product_a']} · {note}", f"B {p['product_b']} · {note}"
    return (f'<div class="thumbs">{thumb_html(paint(sa), box, label_a)}{thumb_html(paint(sb), box_b, label_b)}</div>'
            f'<div class="legend">{legend}<span style="color:rgb(220,40,60)">▭ 선택 정렬의 겹침(ROI)</span></div>')


def binding_table(p):
    rows = []
    for r in p["metric_rows"]:
        if r["kind"] not in {"binding", "cnt"}:
            continue
        bar = "" if r["boundary_f1"] is None else f'<span class="bar"><i style="width:{max(0.0, min(1.0, r["boundary_f1"])) * 100:.0f}%"></i></span>'
        rows.append(f'<tr><td>{esc(r["key"])}</td><td>{STATUS_TEXT[r["status"]]}</td><td class="n">{r["area_a_px"]}</td><td class="n">{r["area_b_px"]}</td>'
                    f'<td class="n">{num(r["area_ratio"], 2)}</td><td class="n">{num(r["dice"])}</td><td class="n">{num(r["boundary_f1"])}{bar}</td>'
                    f'<td class="n">{num(r["boundary_precision"])}</td><td class="n">{num(r["boundary_recall"])}</td><td class="n">{num(r["fill_f1"])}</td></tr>')
    head = ('<tr><th>binding</th><th>사용</th><th>면적 A(px)</th><th>면적 B(px)</th><th>면적 비율</th><th>Dice</th><th>경계 F1</th><th>경계 P</th><th>경계 R</th><th>fill F1</th></tr>')
    return f'<h3>binding별 지표(metrics_anchor={esc(p["metrics_anchor"])} 위치, 원본 해상도 면적)</h3><table>{head}{"".join(rows) or "<tr><td colspan=10>binding 지표 없음(겹침 없음)</td></tr>"}</table>'


def alignment_lines(p):
    flags = "".join(f'<span class="badge">{esc(f)}</span>' for f in p["alignment_flag"].split(";") if f)
    old = (f'<div class="line"><b>기존 정렬</b><span class="badge">tier 근거: {esc(p["anchor"])}</span><span class="badge">정렬 기준 {esc(p["anchor"])} · 이동 ({num(p["shift_y_um"], 1)}, {num(p["shift_x_um"], 1)})µm</span>'
           f'<span class="badge">alignment_score {num(p["alignment_score"])}</span><span class="badge">겹침 {p["roi_overlap"]:.0%}</span>{flags}</div>')
    layer_flags = "".join(f'<span class="badge">{esc(f)}</span>' for f in p["layer_align_flag"].split(";") if f)
    if p["layer_shift_y_um"] is None:
        new = f'<div class="line"><b>Layer 정렬</b><span class="badge">Layer 이동 없음</span>{layer_flags}<span class="badge">metrics_anchor {esc(p["metrics_anchor"])}</span></div>'
    else:
        new = (f'<div class="line"><b>Layer 정렬</b><span class="badge">이동 ({num(p["layer_shift_y_um"], 1)}, {num(p["layer_shift_x_um"], 1)})µm</span>'
               f'<span class="badge">L1 Dice(첫 층) {num(p["layer_align_dice"])}</span>'
               f'<span class="badge">결정 층 Dice {num(p["layer_align_decided_dice"])} ({esc(p["layer_align_decided"])})</span>'
               f'<span class="badge">2순위 Dice {num(p["layer_align_second_dice"])}</span><span class="badge">불일치 층 {p["layer_align_disagreeing"]}</span>'
               f'{layer_flags}<span class="badge">metrics_anchor {esc(p["metrics_anchor"])}</span></div>')
    return old + new


def primary_metric(result):
    """이번 데이터의 주 지표: summary 의 AUC 가 가장 높은 지표(동률이면 PRIMARY_PRIORITY 순). 값이 있는 쌍이 전체의 절반 미만인 지표는 후보에서 뺀다.
    같은 계열 쌍이 없어 AUC 가 하나도 없으면 similarity. summary 에 쓸 수 있는 dict 를 돌려준다."""
    pairs, auc = result["pairs"], result["summary"]["separation_auc_same_vs_other_lineage"]
    counts = {name: sum(p[name] is not None for p in pairs) for name in PRIMARY_PRIORITY}
    candidates = [(auc.get(name), name) for name in PRIMARY_PRIORITY if auc.get(name) is not None and counts[name] * 2 >= len(pairs)]
    if candidates:
        best = max(value for value, _ in candidates)
        name = next(name for value, name in candidates if value == best)
        return {"name": name, "auc": auc[name], "n": counts[name], "reason": "highest_auc"}
    return {"name": "similarity", "auc": None, "n": counts["similarity"], "reason": "no_auc"}


def primary_context(result):
    """주 지표와 같은 계열 범위(min–max)·다른 계열 중앙값(카드 ②줄 용)."""
    primary = primary_metric(result)
    same = [p[primary["name"]] for p in result["pairs"] if p["same_lineage"] and p[primary["name"]] is not None]
    other = [p[primary["name"]] for p in result["pairs"] if not p["same_lineage"] and p[primary["name"]] is not None]
    return {**primary, "same_min": min(same) if same else None, "same_max": max(same) if same else None, "other_median": float(np.median(other)) if other else None}


def primary_line(primary):
    if primary["reason"] == "highest_auc":
        why = "같은 계열 쌍과 다른 계열 쌍을 가장 잘 가르는(AUC 최고, 동률이면 고정 우선순위) 지표"
    else:
        why = "같은 계열 쌍이 없어 AUC 를 못 구해 기본 지표를 씁니다"
    return f'<p class="sub primary">이번 데이터의 주 지표: <b>{esc(primary["name"])}</b> (AUC {num(primary["auc"])}, n={primary["n"]}) — {esc(why)}.</p>'


def key_lines(p, primary, diff_names):
    """카드 머리 핵심 3줄: ① 정렬 신뢰 ② 주 지표 ③ 차이 큰 층."""
    layer_flags = [f for f in p["layer_align_flag"].split(";") if f]
    if p["metrics_anchor"] == "LAYER":
        first = f'Layer 정렬 · L1 Dice {num(p["layer_align_dice"])} · ' + (f'플래그 {esc(", ".join(layer_flags))}' if layer_flags else "플래그 없음")
    else:
        first = f'기존({esc(p["metrics_anchor"])}) 정렬 · ' + (esc(", ".join(layer_flags)) if layer_flags else "Layer 이동 미사용")
    warn = ' class="warn"' if layer_flags else ""
    name = primary["name"]
    low, high = primary["same_min"], primary["same_max"]
    second = (f'{esc(name)} = {num(p[name])} (같은 계열 {num(low)}–{num(high)}, 다른 계열 중앙값 {num(primary["other_median"])}) · '
              f'현재 {TIER_TEXT[p["tier"]]} · 기준 {p["floor"]:.3f}')
    third = ", ".join(esc(n) for n in diff_names) or "비교할 층 없음"
    return (f'<ol class="key"><li{warn}><b>정렬 신뢰</b> {first}</li><li><b>주 지표</b> {second}</li><li><b>차이 큰 층</b> {third}</li></ol>')


def card_html(p, sig, top_layers, primary):
    sa, sb = sig[p["product_a"]], sig[p["product_b"]]
    shift = selected_shift(p, sa)
    names = pair_layers(sa, sb)
    reasons = "".join(f'<li>{esc(REASON_TEXT[r])}</li>' for r in p["review"])
    if names is not None:
        items = layer_items(sa, sb, names, shift)
        body, diff_names = layer_triples(p, items, top_layers), [item[1] for item in (items or [])[:3]]
    else:
        body = binding_triples(p, sa, sb, shift, top_layers)
        diff_names = [k for k, _, _ in sorted(p["result"]["bindings"], key=lambda item: (-item[2], item[0]))[:3]]
    group = "" if p["group_a"] == "-" and p["group_b"] == "-" else f'<span class="badge">그룹 {esc(p["group_a"])}/{esc(p["group_b"])}</span>'
    badges = (f'<div class="line"><b>유사도</b><span class="badge">유사도 {p["similarity"]:.3f}</span><span class="badge">Layer 유사도 {num(p["similarity_layer"])}</span>'
              f'<span class="badge">기준 {p["floor"]:.3f}</span><span class="badge">현재: {TIER_TEXT[p["tier"]]}</span>{group}</div>'
              f'<div class="line"><b>사용 일치</b><span class="badge" title="전극 합집합의 사용 일치(Jaccard)">usage_jaccard {p["usage_jaccard"]:.2f}</span>'
              f'<span class="badge" title="CNT 를 빼고 사용 문턱을 적용한 binding 의 사용 일치">usage_jaccard_electrodes {num(p["usage_jaccard_electrodes"], 2)}</span></div>')
    folded = f'<details class="all"><summary>모든 값 보기</summary>{badges}{alignment_lines(p)}{binding_table(p)}</details>'
    return (f'<section class="card"><header><h2>A {esc(p["product_a"])} ↔ B {esc(p["product_b"])}</h2></header>'
            f'<ul class="sub">{reasons}</ul>{key_lines(p, primary, diff_names)}{frame_thumbs(p, sa, sb, names, shift)}{body}{folded}</section>')


def distribution_svg(result):
    """지표별 같은 계열(초록) vs 다른 계열(회색) 분포. 점이 DENSE_POINTS 를 넘으면 20-bin 히스토그램. similarity 에는 floor 선."""
    pairs, summary = result["pairs"], result["summary"]
    auc = summary["separation_auc_same_vs_other_lineage"]
    left, width, lane, gap = 170, 520, 26, 10
    height = len(DIST_METRICS) * (2 * lane + gap) + 30
    out = [f'<svg class="dist" viewBox="0 0 {left + width + 130} {height}" width="{left + width + 130}" role="img" aria-label="지표별 같은 계열 대 다른 계열 분포">']
    for i in range(0, 11):
        x = left + width * i / 10
        out.append(f'<line class="grid" x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="{height - 14}"/><text class="muted" x="{x:.1f}" y="10" text-anchor="middle">{i / 10:.1f}</text>')
    for row, name in enumerate(DIST_METRICS):
        top = 20 + row * (2 * lane + gap)
        groups = [[p[name] for p in pairs if p["same_lineage"] and p[name] is not None], [p[name] for p in pairs if not p["same_lineage"] and p[name] is not None]]
        n = len(groups[0]) + len(groups[1])
        value_auc = auc.get(name)
        out.append(f'<g data-metric="{name}"><text x="4" y="{top + lane}">{esc(name)}</text>'
                   f'<text class="muted" x="{left + width + 8}" y="{top + lane - 2}">AUC {num(value_auc)}</text><text class="muted" x="{left + width + 8}" y="{top + lane + 12}">n={n}'
                   f'{" (히스토그램)" if n > DENSE_POINTS else ""}</text>')
        for lane_index, (values, css) in enumerate(zip(groups, ("pt-same", "pt-other"))):
            base = top + lane_index * lane
            if n > DENSE_POINTS:
                counts = np.histogram(np.clip(values, 0, 1), bins=HIST_BINS, range=(0, 1))[0] if values else np.zeros(HIST_BINS, int)
                peak = max(int(counts.max()), 1)
                for k, c in enumerate(counts):
                    h = (lane - 4) * int(c) / peak
                    out.append(f'<rect class="{css}" x="{left + width * k / HIST_BINS:.1f}" y="{base + lane - 2 - h:.1f}" width="{width / HIST_BINS - 1:.1f}" height="{h:.1f}"/>')
            else:
                for k, v in enumerate(values):
                    jitter = ((k * 0.6180339887) % 1.0) * (lane - 8)
                    out.append(f'<circle class="{css}" cx="{left + width * min(max(v, 0.0), 1.0):.1f}" cy="{base + 4 + jitter:.1f}" r="2.2"/>')
        if name == "similarity":
            floor = summary["floor"]
            out.append(f'<line class="floor" x1="{left + width * floor["global"]:.1f}" y1="{top - 4}" x2="{left + width * floor["global"]:.1f}" y2="{top + 2 * lane}"><title>floor(global) {floor["global"]:.3f}</title></line>')
            for group, value in sorted(floor["per_group"].items()):
                x = left + width * value
                out.append(f'<line class="floor-group" x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{top + 2 * lane}"><title>floor {esc(group)} {value:.3f}</title></line>')
        out.append("</g>")
    out.append("</svg>")
    return "".join(out)


def render_review(result, top_pairs=60, top_bindings=6, top_layers=4):
    """검토 대상 쌍(p["review"]가 비어 있지 않은 모든 쌍)을 위쪽 표에 전부 나열하고, 경계에 가까운 순으로 top_pairs쌍은 카드로 보여 준다.

    top_bindings 는 옛 인수 호환용으로 받기만 한다: 옛 '거리 상위 binding 겹침 타일'은 카드의 binding 지표 표와 층별(없으면 binding별) A | B | 차이 타일이 대신한다."""
    pairs = result["pairs"]
    chosen = sorted((p for p in pairs if p["review"]), key=lambda p: (abs(p["similarity"] - p["floor"]), p["product_a"], p["product_b"]))
    summary = result["summary"]
    counts = "".join(f'<li>{esc(r)}: {n}쌍 — {esc(REASON_TEXT[r])}</li>' for r, n in summary["review"]["by_reason"].items())
    table = "".join(f'<tr><td>{i}</td><td>{esc(p["product_a"])} ↔ {esc(p["product_b"])}</td><td>{TIER_TEXT[p["tier"]]}</td><td class="n">{p["similarity"]:.3f}</td>'
                    f'<td class="n">{p["floor"]:.3f}</td><td>{esc(", ".join(p["review"]))}</td><td>{"아래 카드" if i <= top_pairs else "scores.csv"}</td></tr>' for i, p in enumerate(chosen, 1))
    primary = primary_context(result)
    summary["review"]["primary_metric"] = {key: primary[key] for key in ("name", "auc", "n", "reason")}  # 새 키만 추가한다
    cards = "".join(card_html(p, result["signatures"], top_layers, primary) for p in chosen[:top_pairs])
    omitted = len(chosen) - min(len(chosen), top_pairs)
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>DL1 스크리닝 검토</title><style>{REVIEW_CSS}</style></head><body>'
            f'<h1>DL1 스크리닝 검토</h1><p class="sub">검토 대상 <b>{len(chosen)}쌍</b>입니다(경계에 가까운 순). 유사도 = 1 − binding별 거리의 평균이며, 한쪽에서만 쓰이는 전극은 거리 1입니다. '
            f'이미지는 B를 A에 정렬한 뒤 두 제품이 겹치는 영역(ROI)만 그린 것입니다. 확정할 쌍은 relations.csv에 similar/different로 직접 추가하고, 틀린 자동 줄은 지우세요.</p>'
            f'{primary_line(primary)}'
            f'<p class="sub">용어: L1 Dice(첫 층) = 첫 층만의 Dice(정렬 신뢰 판정 기준), 결정 층 Dice = 정렬을 정한 층의 Dice, 2순위 Dice 는 결정 층 기준입니다.</p>'
            f'<h2>지표별 분포: 같은 계열(초록) vs 다른 계열(회색)</h2><p class="sub">AUC 1에 가까울수록 두 집단이 잘 갈립니다. 빨간 선은 similarity 의 floor 입니다(실선=global, 점선=그룹). 빈칸 값은 그리지 않고 n 에서 뺍니다.</p>'
            f'{distribution_svg(result)}'
            f'<ul class="sub">{counts or "<li>검토 대상이 없습니다.</li>"}</ul>'
            f'<div class="legend"><span><i style="background:var(--a)"></i>A만</span><span><i style="background:var(--b)"></i>B만</span><span><i style="background:var(--both)"></i>둘 다</span>'
            f'<span>기준(같은 계열 최저 유사도) {summary["floor"]["global"]:.3f}</span></div>'
            f'<h2>검토 대상 전체</h2><table><tr><th>#</th><th>쌍</th><th>현재 판정</th><th>유사도</th><th>기준</th><th>이유</th><th>이미지</th></tr>{table}</table>'
            + (f'<p class="sub">카드는 {top_pairs}쌍까지만 그렸습니다. 나머지 {omitted}쌍은 위 표와 scores.csv로 보세요(--review-pairs로 늘릴 수 있음).</p>' if omitted else "")
            + f'<h2>카드</h2>{cards or "<p>검토할 쌍이 없습니다.</p>"}</body></html>')
