"""검색 결과(JSON)를 단일 HTML 보고서로 바꾼다. 네트워크·외부 파일 없이 열린다."""
from __future__ import annotations

import base64
import html
import io

import numpy as np

PREVIEW_MAX = 240  # 모든 미리보기는 같은 규칙: 원본 mask를 긴 변 240px에 맞춰 NEAREST로 확대/축소(비율 유지)
BACKGROUND = (245, 245, 242)
LAYER_ALPHA = 0.5
CNT_COLOR = (20, 20, 20)
CNT_ALPHA = 0.85
UNKNOWN_COLORS = ((205, 205, 205), (150, 150, 150))  # valid=false(관측 불가) 사선 무늬의 두 회색
UNKNOWN_LABEL = "관측 불가(valid=false)"
# Okabe-Ito 색약 친화 팔레트(검정 제외) 7색 + 손으로 고른 5색. layer 이름을 정렬한 순서대로 배정한다.
PALETTE = ((0, 114, 178), (230, 159, 0), (0, 158, 115), (213, 94, 0), (86, 180, 233), (204, 121, 167), (240, 228, 66),
           (120, 94, 240), (220, 50, 32), (110, 70, 20), (150, 200, 40), (190, 0, 130))
GOLDEN = 0.6180339887498949


SATURATIONS = (0.65, 0.8, 0.5)
VALUES = (0.9, 0.7, 0.8)


def overflow_color(k):
    """팔레트를 넘는 k번째(0부터) layer 색: golden-angle 색상 + 채도/명도 순환. 재시도 없이 닫힌 식으로 정한다(고유성은 보장하지 않음)."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((k * GOLDEN) % 1.0, SATURATIONS[k % 3], VALUES[(k // 3) % 3])
    return round(r * 255), round(g * 255), round(b * 255)


def layer_colors(products):
    """layer 이름 정렬 순서 → 색. 팔레트를 넘는 layer는 정렬 순번만으로 결정되는 색을 O(n)에 받으며, 서로 비슷할 수 있다."""
    layers = sorted({e.layer for p in products for e in p.elements})
    return {layer: PALETTE[i] if i < len(PALETTE) else overflow_color(i - len(PALETTE)) for i, layer in enumerate(layers)}


def preview_size(shape):
    h, w = shape
    scale = PREVIEW_MAX / max(h, w)
    return max(1, round(w * scale)), max(1, round(h * scale))


def _small_mask(mask, size):
    """원본 mask를 미리보기 크기로 NEAREST 리샘플(원본 크기의 RGB/float 캔버스는 만들지 않는다)."""
    from PIL import Image
    m = np.asarray(mask) > 0
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0


def _blend(canvas, mask, color, alpha):
    canvas[mask] = canvas[mask] * (1 - alpha) + np.array(color, dtype=np.float64) * alpha


def observed_cnt_sources(product):
    """data.py/screen_relations.py와 같은 CNT 구성: 전역 cnt + 마스크가 있는 원본 connection. 관측되지 않은 connection은 뺀다."""
    sources = [c.mask for c in product.connections if c.mask is not None and c.observation == "observed"]
    if product.cnt is not None:
        sources.append(product.cnt)
    return sources


def preview_rgb(product, colors):
    """제품의 관측된 element mask를 layer 색으로 반투명 합성한 미리보기 크기 RGB. 그릴 mask가 없으면 None.

    valid=false 픽셀은 material/CNT로 그리지 않고 회색 사선 무늬(관측 불가)로 덮는다."""
    observed = [e for e in product.elements if e.mask is not None]
    cnt_sources = observed_cnt_sources(product)
    ref = [np.asarray(e.mask) for e in observed] + [np.asarray(m) for m in cnt_sources]
    if not ref:
        return None
    size = preview_size(ref[0].shape)
    valid = None if product.valid is None else _small_mask(product.valid, size)
    canvas = np.empty((size[1], size[0], 3), dtype=np.float64)
    canvas[:] = BACKGROUND

    def visible(mask):
        m = _small_mask(mask, size)
        return m if valid is None else m & valid
    for layer in sorted({e.layer for e in observed}):
        union = np.zeros(canvas.shape[:2], dtype=bool)
        for e in observed:
            if e.layer == layer:
                union |= visible(e.mask)
        _blend(canvas, union, colors[layer], LAYER_ALPHA)
    if cnt_sources:
        union = np.zeros(canvas.shape[:2], dtype=bool)
        for m in cnt_sources:
            union |= visible(m)
        _blend(canvas, union, CNT_COLOR, CNT_ALPHA)
    if valid is not None and not valid.all():
        yy, xx = np.indices(canvas.shape[:2])
        stripe = ((xx + yy) // 3) % 2
        for i, color in enumerate(UNKNOWN_COLORS):
            canvas[(~valid) & (stripe == i)] = color
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def preview_uri(product, colors):
    from PIL import Image
    rgb = preview_rgb(product, colors)
    if rgb is None:
        return None
    buffer = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _num(v):
    return f"{v:.10g}"


def size_info(p):
    w, h = p.unit_size_um
    lines = [f"unit {_num(w)} × {_num(h)} µm"]
    if p.frame is not None:
        lines.append(f"frame {p.frame.shape_yx[0]} × {p.frame.shape_yx[1]} px (y×x)")
        lines.append(f"pitch {_num(p.frame.pixel_pitch_um)} µm/px")
    else:
        lines.append("frame n/a")
        lines.append("pitch n/a")
    return lines


def size_differs(p, q):
    """데이터 계약(schema.py)과 같은 허용오차(atol=1e-9, rtol=1e-9, numpy isclose)로 물리 크기를 비교한다."""
    same_unit = all(bool(np.isclose(a, b, atol=1e-9, rtol=1e-9)) for a, b in zip(p.unit_size_um, q.unit_size_um))
    pa = None if p.frame is None else p.frame.pixel_pitch_um
    pb = None if q.frame is None else q.frame.pixel_pitch_um
    same_pitch = (pa is None and pb is None) or (pa is not None and pb is not None and bool(np.isclose(pa, pb, atol=1e-9, rtol=1e-9)))
    return not (same_unit and same_pitch)


CSS = """
body{margin:0;background:#fcfcfb;color:#0b0b0b;font:14px/1.5 system-ui,'Segoe UI','Malgun Gothic',sans-serif;padding:24px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:18px 0 8px}
.caveat{color:#52514e;margin:0 0 12px}.meta{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;margin:0 0 12px}.meta dt{color:#52514e}.meta dd{margin:0;word-break:break-all}
.legend{display:flex;gap:18px;flex-wrap:wrap;color:#52514e;margin:8px 0 16px}.legend i{display:inline-block;width:12px;height:12px;margin-right:6px;border-radius:2px;vertical-align:-1px;border:1px solid #888}
.card{display:flex;gap:16px;border:1px solid #e3e2de;border-radius:8px;padding:12px;margin:0 0 12px}.card img{display:block;border:1px solid #e3e2de;background:#f5f5f2;flex:none}
.body{flex:1;min-width:260px}.body h3{margin:0 0 4px;font-size:15px}.sub{color:#52514e;font-size:12px}
.badge{font-size:12px;border:1px solid #e3e2de;border-radius:10px;padding:1px 8px;margin-left:6px;color:#52514e}.warn{color:#8a4b00;border-color:#d99a3a;background:#fff5e0}
.bar{position:relative;height:14px;background:#eceae5;border-radius:3px;margin:6px 0;max-width:420px}.bar b{position:absolute;left:0;top:0;bottom:0;background:#0072b2;border-radius:3px}.bar u{position:absolute;left:50%;top:-2px;bottom:-2px;border-left:1px solid #0b0b0b}
.noimg{width:120px;height:120px;display:flex;align-items:center;justify-content:center;color:#52514e;border:1px dashed #bbb;flex:none}
table{border-collapse:collapse}td,th{border:1px solid #e3e2de;padding:4px 10px;text-align:left}
.unavail{border:2px solid #d99a3a;background:#fff5e0;border-radius:8px;padding:12px 16px;margin:12px 0;font-size:16px}
"""


def _e(v):
    return html.escape(str(v), quote=True)


def _img(uri, label):
    return f'<img src="{uri}" alt="{_e(label)}">' if uri else '<div class="noimg">mask 없음</div>'


def _reason_text(reason):
    if isinstance(reason, dict):
        return ", ".join(f"{k}={v}" for k, v in reason.items())
    return str(reason)


def render_report(result, query_product, candidate_products_by_id):
    shown = [query_product] + [candidate_products_by_id[r["product_id"]] for r in result["rankings"] if r["product_id"] in candidate_products_by_id]
    colors = layer_colors(shown)
    out = [f"<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><title>검색 보고서 {_e(result['query_id'])}</title><style>{CSS}</style></head><body>"]
    out.append(f"<h1>검색 보고서 — {_e(result['query_id'])}</h1>")
    out.append("<p class=\"caveat\">제품 전체 유사도이며 대응 위치를 찾아주지 않음; 실제 검색 성능은 미검증</p>")
    meta = [("상태", result["status"]), ("사유", "—" if result["reason"] is None else _reason_text(result["reason"])), ("grid", result["grid"]),
            ("후보 수 / 검색 가능", f"{result['candidate_count']} / {result['eligible_count']}"), ("체크포인트 fingerprint", result["fingerprint"])]
    out.append("<dl class=\"meta\">" + "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in meta) + "</dl>")
    legend = "".join(f"<span><i style=\"background:rgb{c}\"></i>{_e(layer)}</span>" for layer, c in colors.items())
    legend += f"<span><i style=\"background:rgb{CNT_COLOR}\"></i>CNT</span>"
    if any(p.valid is not None for p in shown):
        a, b = UNKNOWN_COLORS
        legend += f"<span><i style=\"background:repeating-linear-gradient(45deg,rgb{a} 0 3px,rgb{b} 3px 6px)\"></i>{UNKNOWN_LABEL}</span>"
    out.append("<div class=\"legend\">" + legend + "</div>")
    if len(colors) > len(PALETTE):
        out.append(f"<p class=\"sub warn\">layer가 {len(colors)}개로 손으로 고른 팔레트({len(PALETTE)}색)를 넘습니다. 팔레트를 넘는 layer의 색은 서로 비슷해 구분하기 어려울 수 있음(범례의 색을 참조).</p>")
    out.append("<p class=\"sub\">미리보기: 원본 mask를 layer별 반투명(α0.5)으로 겹쳐 그림. 모든 미리보기는 긴 변 240px에 맞춰 비율 유지 확대/축소(NEAREST)하므로 실제 크기 비교는 아래 크기 정보를 볼 것.</p>")
    out.append("<h2>질의</h2><div class=\"card\">" + _img(preview_uri(query_product, colors), query_product.product_id) + "<div class=\"body\">"
               f"<h3>{_e(query_product.product_id)}</h3><div class=\"sub\">연결 상태: {_e(result['query_connectivity'])}</div>"
               + "".join(f"<div>{_e(s)}</div>" for s in size_info(query_product)) + "</div></div>")
    if result["status"] != "available":
        out.append(f"<div class=\"unavail\"><b>검색 불가</b> — 사유: {_e(_reason_text(result['reason']))}</div>")
    else:
        out.append(f"<h2>상위 {len(result['rankings'])}개 후보</h2><p class=\"sub\">막대 눈금: cosine −1 ~ 1 (세로선 = 0)</p>")
        for rank, row in enumerate(result["rankings"], 1):
            p = candidate_products_by_id.get(row["product_id"])
            pct = max(0.0, min(100.0, (row["score"] + 1) / 2 * 100))
            badge = " <span class=\"badge warn\">크기 다름</span>" if p is not None and size_differs(p, query_product) else ""
            info = size_info(p) if p is not None else ["unit n/a", "frame n/a", "pitch n/a"]
            out.append(f"<div class=\"card\">{_img(preview_uri(p, colors) if p is not None else None, row['product_id'])}<div class=\"body\">"
                       f"<h3>#{rank} {_e(row['product_id'])}{badge}</h3>"
                       f"<div class=\"bar\"><b style=\"width:{pct:.2f}%\"></b><u></u></div><div>cosine {row['score']:.6f}</div>"
                       f"<div class=\"sub\">연결 상태: {_e(row['connectivity_status'])}</div>" + "".join(f"<div>{_e(s)}</div>" for s in info) + "</div></div>")
    out.append("<h2>제외된 후보</h2>")
    if result["excluded"]:
        out.append("<table><tr><th>product_id</th><th>사유</th></tr>" + "".join(
            f"<tr><td>{_e(r['product_id'])}</td><td>{_e(_reason_text({k: v for k, v in r.items() if k != 'product_id'}))}</td></tr>" for r in result["excluded"]) + "</table>")
    else:
        out.append("<p class=\"sub\">없음</p>")
    out.append("</body></html>\n")
    return "".join(out)
