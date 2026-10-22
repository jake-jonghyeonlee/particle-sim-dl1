import base64
import io
import json
import os
import re
import stat
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from conftest import make_product
from psdl1.data import Element
from psdl1.search import main, search
from psdl1.search_report import layer_colors, preview_rgb, render_report
from test_search import setup_model


def decode_images(page):
    return [np.asarray(Image.open(io.BytesIO(base64.b64decode(m))).convert("RGB"))
            for m in re.findall(r'src="data:image/png;base64,([^"]+)"', page)]


@pytest.fixture
def scene(products):
    model, cfg = setup_model(products)
    query = replace(products[0], product_id="query")
    cands = [replace(products[i], product_id=f"c{i}") for i in range(1, 5)]
    cands[1] = replace(cands[1], unit_size_um=(9.0, 9.0))
    result = search(model, cfg, query, cands, topk=3)
    return model, cfg, query, cands, result


def test_page_is_self_contained_and_ordered(scene):
    _, _, query, cands, result = scene
    page = render_report(result, query, {p.product_id: p for p in cands})
    assert not re.search(r'(src|href)="https?://', page) and "http://" not in page and "https://" not in page
    ids = [r["product_id"] for r in result["rankings"]]
    order = [page.index(f"#{i} {pid}") for i, pid in enumerate(ids, 1)]
    assert order == sorted(order)
    assert f"{result['rankings'][0]['score']:.6f}" in page
    assert "제품 전체 유사도이며 대응 위치를 찾아주지 않음; 실제 검색 성능은 미검증" in page
    assert result["fingerprint"] in page and "후보 수" in page
    assert "unit " in page and "µm" in page and "frame n/a" in page and "pitch n/a" in page


def test_size_info_and_mismatch_badge(scene):
    _, _, query, cands, result = scene
    big = replace(cands[0], product_id="big", unit_size_um=(9.0, 9.5),
                  frame=__import__("psdl1.schema", fromlist=["Frame"]).Frame("F", (7, 8), 0.25))
    fake = dict(result, rankings=[{"product_id": "big", "score": 0.5, "connectivity_status": "unavailable", "input_identity": "x"}])
    page = render_report(fake, query, {"big": big})
    assert "unit 9 × 9.5 µm" in page and "frame 7 × 8 px" in page and "pitch 0.25 µm/px" in page
    assert "크기 다름" in page
    same = replace(query, product_id="same")
    page = render_report(dict(fake, rankings=[dict(fake["rankings"][0], product_id="same")]), query, {"same": same})
    assert "크기 다름" not in page


def test_layer_colors_distinct_and_shared(scene):
    _, _, query, cands, result = scene
    page = render_report(result, query, {p.product_id: p for p in cands})
    colors = layer_colors([query, *cands])
    assert len(set(colors.values())) == len(colors) >= 2
    for c in colors.values():
        assert f"rgb{c}" in page
    # 같은 제품을 질의/후보로 그리면 픽셀이 동일하다(같은 layer→색 매핑)
    same = replace(query, product_id="twin")
    a = preview_rgb(query, colors)
    b = preview_rgb(same, colors)
    assert (a == b).all()


def test_overlap_keeps_both_layers_visible():
    m1 = np.zeros((10, 10), np.float32); m1[:, :6] = 1
    m2 = np.zeros((10, 10), np.float32); m2[:, 4:] = 1
    p = replace(make_product("bus", 10), cnt=None, elements=(Element("a", "L1", m1), Element("b", "L2", m2)))
    colors = layer_colors([p])
    both = preview_rgb(p, colors)[120, 132]
    only1 = preview_rgb(p, colors)[120, 12]
    only2 = preview_rgb(p, colors)[120, 228]
    assert not (both == only1).all() and not (both == only2).all() and not (only1 == only2).all()


def test_unavailable_shows_reason_and_excluded_table(products):
    model, cfg = setup_model(products)
    unknown = replace(products[0], product_id="bad", elements=(replace(products[0].elements[0], layer="never-trained"), *products[0].elements[1:]))
    result = search(model, cfg, products[0], [unknown])
    assert result["status"] == "unavailable"
    page = render_report(result, products[0], {"bad": unknown})
    assert "검색 불가" in page and "no_eligible_candidates" in page
    assert "제외된 후보" in page and "unknown_training_vocabulary" in page and "<td>bad</td>" in page
    assert "상위" not in page


def test_html_escaping_of_hostile_ids(scene):
    _, _, query, cands, result = scene
    evil = '<script>alert(1)</script>'
    q = replace(query, product_id=evil,
                elements=tuple(replace(e, layer=e.layer) for e in query.elements))
    c = replace(cands[0], product_id=evil + "c", elements=(replace(cands[0].elements[0], layer='"><b>x'), *cands[0].elements[1:]))
    res = dict(result, query_id=evil, rankings=[{"product_id": c.product_id, "score": 0.1, "connectivity_status": "x<i>", "input_identity": "i"}],
               excluded=[{"product_id": evil, "reason": "<img src=x onerror=1>"}])
    page = render_report(res, q, {c.product_id: c})
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "<img src=x" not in page and "<b>x" not in page


def run_cli(manifest_case, tmp_path, monkeypatch, extra, capsys=None):
    from psdl1.data import load_manifest
    _, write, _ = manifest_case
    q, = load_manifest(write())
    cand = replace(q, product_id="candidate")
    q = replace(q, product_id="qq")
    model, cfg = setup_model([q])
    monkeypatch.setattr("psdl1.search.load_model", lambda path: (model, cfg))
    monkeypatch.setattr("psdl1.search.load_manifest", lambda path, **k: [q] if path == "q.json" else [cand])
    base = ["--ckpt", "x.pt", "--query-manifest", "q.json", "--candidates-manifest", "c.json", "--query", "qq"]
    main(base + extra)


def test_cli_json_unchanged_with_html_and_works_without_out(manifest_case, tmp_path, monkeypatch, capsys):
    plain, withhtml = tmp_path / "a.json", tmp_path / "b.json"
    run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(plain)])
    page = tmp_path / "r.html"
    run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(withhtml), "--html", str(page)])
    assert plain.read_bytes() == withhtml.read_bytes()
    text = page.read_text(encoding="utf-8")
    assert "candidate" in text and "qq" in text and not (tmp_path / "r.html.tmp").exists()
    run_cli(manifest_case, tmp_path, monkeypatch, [])
    stdout_plain = capsys.readouterr().out
    run_cli(manifest_case, tmp_path, monkeypatch, ["--html", str(tmp_path / "s.html")])
    assert capsys.readouterr().out == stdout_plain == plain.read_text()
    assert (tmp_path / "s.html").exists()


def test_render_failure_leaves_no_file(manifest_case, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("render")
    monkeypatch.setattr("psdl1.search_report.render_report", boom)
    page = tmp_path / "r.html"
    with pytest.raises(RuntimeError):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(tmp_path / "o.json"), "--html", str(page)])
    assert not page.exists() and not (tmp_path / "r.html.tmp").exists()


# ---- 디코딩 기반 검증 -------------------------------------------------------

def _prod(h, w, elements, **kw):
    from psdl1.schema import Product
    return Product("p", (w * 0.05, h * 0.05), tuple(elements), **kw)


def _rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.float32)
    m[y0:y1, x0:x1] = 1
    return m


def test_embedded_previews_are_decoded_and_counted(scene):
    _, _, query, cands, result = scene
    page = render_report(result, query, {p.product_id: p for p in cands})
    images = decode_images(page)
    assert len(images) == 1 + len(result["rankings"])
    assert all(max(i.shape[:2]) == 240 for i in images)


def test_known_mask_pixel_colors_and_nonsquare_aspect():
    p = _prod(4, 6, [Element("a", "L1", _rect(4, 6, 0, 2, 0, 3))])
    colors = layer_colors([p])
    page = render_report(dict(_FAKE, rankings=[]), p, {})
    img = decode_images(page)[0]
    assert img.shape == (160, 240, 3)  # 4×6 → 240×160 (높이×너비)
    expect = np.rint(np.array((245, 245, 242)) * .5 + np.array(colors["L1"]) * .5).astype(np.uint8)
    assert (img[10, 10] == expect).all() and (img[150, 230] == (245, 245, 242)).all()
    assert (img[10, 130] == (245, 245, 242)).all()  # x가 3px 이상인 쪽은 비어 있음


_FAKE = {"query_id": "p", "status": "available", "reason": None, "grid": 64, "candidate_count": 0, "eligible_count": 0,
         "fingerprint": "f", "query_connectivity": "unavailable", "rankings": [], "excluded": []}


def test_valid_holes_are_hatched_not_drawn_as_material():
    valid = np.ones((4, 6), np.float32)
    valid[0:2, 0:2] = 0
    p = _prod(4, 6, [Element("a", "L1", np.ones((4, 6), np.float32))], cnt=np.ones((4, 6), np.float32), valid=valid)
    colors = layer_colors([p])
    rgb = preview_rgb(p, colors)
    hole, known = rgb[10, 10], rgb[150, 230]
    assert tuple(hole) in {(205, 205, 205), (150, 150, 150)} and tuple(known) not in {(205, 205, 205), (150, 150, 150)}
    hole_px = rgb[:80, :80]
    assert {tuple(v) for v in hole_px.reshape(-1, 3)} == {(205, 205, 205), (150, 150, 150)}
    page = render_report(dict(_FAKE), p, {})
    assert "관측 불가(valid=false)" in page
    no_valid = render_report(dict(_FAKE), replace(p, valid=None), {})
    assert "관측 불가" not in no_valid


def test_observed_connection_cnt_is_drawn_and_unobserved_skipped():
    from psdl1.schema import Connection, Endpoint
    el = [Element("a", "L1", _rect(4, 6, 0, 1, 0, 1)), Element("b", "L2", _rect(4, 6, 0, 1, 5, 6))]
    ep = (Endpoint("a", (0.0, 0.0)), Endpoint("b", (0.1, 0.0)))
    conn = Connection("c", "CNT", ep, _rect(4, 6, 3, 4, 0, 6))
    base = _prod(4, 6, el, cnt=None)
    colors = layer_colors([base])
    plain = preview_rgb(base, colors)
    with_c = preview_rgb(replace(base, connections=(conn,)), colors)
    skipped = preview_rgb(replace(base, connections=(replace(conn, observation="missing"),)), colors)
    assert (plain == skipped).all() and not (plain == with_c).all()
    assert tuple(with_c[150, 120]) == (int(round(245 * .15 + 20 * .85)), int(round(245 * .15 + 20 * .85)), int(round(242 * .15 + 20 * .85)))


def test_more_than_palette_layers_have_distinct_colors_and_legend():
    els = [Element(f"r{i}", f"L{i:02d}", _rect(4, 6, 0, 1, 0, 1)) for i in range(20)]
    p = _prod(4, 6, els)
    colors = layer_colors([p])
    assert len(set(colors.values())) == 20 == len(colors)
    page = render_report(dict(_FAKE), p, {})
    for c in colors.values():
        assert f"rgb{c}" in page
    assert "구분하기 어려울 수 있음" in page
    assert "구분하기 어려울 수 있음" not in render_report(dict(_FAKE), _prod(4, 6, els[:3]), {})
    # L0 만 / L7 만 있는 같은 기하의 제품은 이제 다른 픽셀을 만든다
    a = _prod(4, 6, [Element("x", "L00", _rect(4, 6, 0, 2, 0, 2))] + [Element("y", f"L{i:02d}", np.zeros((4, 6), np.float32)) for i in range(1, 8)])
    b = _prod(4, 6, [Element("x", "L00", np.zeros((4, 6), np.float32))] + [Element("y", f"L{i:02d}", _rect(4, 6, 0, 2, 0, 2) if i == 7 else np.zeros((4, 6), np.float32)) for i in range(1, 8)])
    cs = layer_colors([a])
    assert not (preview_rgb(a, cs) == preview_rgb(b, cs)).all()


def test_size_tolerance_both_directions():
    from psdl1.schema import Frame
    from psdl1.search_report import size_differs, size_info

    def prod(pitch):
        return replace(_prod(1, 1, [Element("a", "L1", np.ones((1, 1), np.float32))]), unit_size_um=(pitch, pitch), frame=Frame("F", (1, 1), pitch))
    assert size_differs(prod(0.1), prod(0.10000005))      # 상대 5e-7 > 1e-9: 다름
    assert size_info(prod(0.1))[2] == "pitch 0.1 µm/px" and "0.10000005" in size_info(prod(0.10000005))[2]
    assert not size_differs(prod(0.00001), prod(0.0000100005))  # 절대 5e-10 ≤ atol: 같음
    assert not size_differs(prod(0.1), prod(0.1 * (1 + 1e-10)))


# ---- CLI 안전성 -------------------------------------------------------------

def test_preexisting_tmp_and_symlink_untouched(manifest_case, tmp_path, monkeypatch):
    page = tmp_path / "r.html"
    (tmp_path / "r.html.tmp").write_text("USER DATA")
    victim = tmp_path / "victim.txt"
    victim.write_text("KEEP")
    run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(tmp_path / "o.json"), "--html", str(page)])
    assert (tmp_path / "r.html.tmp").read_text() == "USER DATA" and "<html" in page.read_text(encoding="utf-8")
    page2 = tmp_path / "s.html"
    (tmp_path / "s.html.tmp").symlink_to(victim)
    run_cli(manifest_case, tmp_path, monkeypatch, ["--html", str(page2)])
    assert victim.read_text() == "KEEP" and (tmp_path / "s.html.tmp").is_symlink() and not page2.is_symlink()
    leftovers = [f.name for f in tmp_path.iterdir() if f.name.endswith(".tmp") and f.name not in ("r.html.tmp", "s.html.tmp")]
    assert leftovers == []


def test_unique_temp_names_do_not_collide(tmp_path, monkeypatch):
    from psdl1.search import _write_html
    seen = []
    real = __import__("os").replace
    monkeypatch.setattr("os.replace", lambda s, d: (seen.append(str(s)), real(s, d))[1])
    _write_html(str(tmp_path / "x.html"), "A")
    _write_html(str(tmp_path / "x.html"), "B")
    assert len(set(seen)) == 2 and all(s.endswith(".tmp") for s in seen)
    assert (tmp_path / "x.html").read_text() == "B" and [f.name for f in tmp_path.iterdir()] == ["x.html"]


def test_out_equal_html_rejected_without_clobber(manifest_case, tmp_path, monkeypatch):
    same = tmp_path / "same.json"
    same.write_text("ORIGINAL")
    with pytest.raises(ValueError, match="같은 파일"):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(same), "--html", str(same)])
    assert same.read_text() == "ORIGINAL"
    (tmp_path / "sub").mkdir()
    alias = str(tmp_path / "sub" / ".." / "same.json")
    with pytest.raises(ValueError, match="같은 파일"):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(same), "--html", alias])
    assert same.read_text() == "ORIGINAL"
    link = tmp_path / "link.html"
    link.symlink_to(same)
    with pytest.raises(ValueError, match="같은 파일"):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(same), "--html", str(link)])
    assert same.read_text() == "ORIGINAL"


def test_missing_pillow_fails_before_any_output(manifest_case, tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no PIL")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    out = tmp_path / "o.json"
    with pytest.raises(ValueError, match=r"Pillow.*png"):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(out), "--html", str(tmp_path / "r.html")])
    assert not out.exists() and not (tmp_path / "r.html").exists()


# ---- 출력 게시(stage → os.replace) ---------------------------------------------

def _leftovers(d):
    return sorted(f.name for f in d.iterdir() if f.name.endswith(".tmp"))


def _case_insensitive(d):
    probe = d / "CaseProbe"
    probe.write_text("x")
    try:
        return (d / "caseprobe").exists()
    finally:
        probe.unlink()


def test_case_alias_rejected_before_anything_written(manifest_case, tmp_path, monkeypatch):
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem")
    with pytest.raises(ValueError, match="같은 파일"):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(tmp_path / "new-case.json"), "--html", str(tmp_path / "NEW-CASE.JSON")])
    assert not [f for f in tmp_path.iterdir() if "case" in f.name.lower()]


def test_nfc_nfd_alias_rejected_before_anything_written(manifest_case, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="같은 파일"):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(tmp_path / "new-\u00e9.json"), "--html", str(tmp_path / "new-e\u0301.json")])
    assert not [f for f in tmp_path.iterdir() if f.name.startswith("new-")]


def test_existing_private_html_stays_private_and_new_follow_umask(manifest_case, tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX permission bits")
    old = os.umask(0o022)
    try:
        page = tmp_path / "r.html"
        page.write_text("OLD")
        page.chmod(0o600)
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(tmp_path / "o.json"), "--html", str(page)])
        assert "<html" in page.read_text(encoding="utf-8") and stat.S_IMODE(page.stat().st_mode) == 0o600
        assert stat.S_IMODE((tmp_path / "o.json").stat().st_mode) == 0o644
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(tmp_path / "o2.json"), "--html", str(tmp_path / "new.html")])
        assert stat.S_IMODE((tmp_path / "new.html").stat().st_mode) == 0o644
        assert os.umask(0o022) == 0o022  # 프로세스 umask를 건드리지 않았다
    finally:
        os.umask(old)


def test_html_failure_leaves_existing_json_untouched(manifest_case, tmp_path, monkeypatch):
    out = tmp_path / "o.json"
    out.write_bytes(b"OLD JSON")
    with pytest.raises(FileNotFoundError):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(out), "--html", str(tmp_path / "missing" / "r.html")])
    assert out.read_bytes() == b"OLD JSON" and _leftovers(tmp_path) == []


def test_symlinked_out_is_replaced_not_followed(manifest_case, tmp_path, monkeypatch):
    html = tmp_path / "real.html"
    html.write_text("HTML MUST SURVIVE")
    out = tmp_path / "o.json"
    out.symlink_to(html)
    # 사전 검사는 링크 별칭을 잡으므로, 다른 HTML과 함께 쓰면 링크가 일반 파일로 교체되어야 한다
    run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(out), "--html", str(tmp_path / "other.html")])
    assert not out.is_symlink() and html.read_text() == "HTML MUST SURVIVE"
    assert json.loads(out.read_text())["query_id"] == "qq" and _leftovers(tmp_path) == []


def test_samefile_final_guard_keeps_published_json(tmp_path, monkeypatch):
    from psdl1.search import _publish_both
    out, html = tmp_path / "o.json", tmp_path / "r.html"
    html.write_text("OLD HTML")
    monkeypatch.setattr("psdl1.search._same_file", lambda a, b: True)
    with pytest.raises(ValueError, match="같은 파일"):
        _publish_both(str(out), "{}", str(html), "NEW HTML")
    assert out.read_text() == "{}" and html.read_text() == "OLD HTML" and _leftovers(tmp_path) == []


def test_many_layers_with_missing_masks_finish_quickly(scene):
    import time
    _, _, query, cands, result = scene
    elements = tuple(Element(f"e{i}", f"L{i:05d}", None) for i in range(2500))
    big = replace(query, elements=elements)
    t = time.monotonic()
    colors = layer_colors([big])
    page = render_report(dict(result, status="unavailable", reason="x", rankings=[]), big, {})
    assert time.monotonic() - t < 5 and len(colors) == 2500
    assert "비슷해" in page and "모두 서로 다름" not in page
    assert layer_colors([big]) == colors  # 결정적


# ---- 라운드 4: 디렉터리 사전 거부, 임시 파일 기밀성, JSON 바이트 동일성 --------------

def test_existing_html_directory_rejected_old_json_survives(manifest_case, tmp_path, monkeypatch):
    out, d = tmp_path / "o.json", tmp_path / "r.html"
    out.write_bytes(b"OLD JSON")
    d.mkdir()
    (d / "child").write_text("KEEP")
    with pytest.raises(ValueError, match="디렉터리"):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(out), "--html", str(d)])
    assert out.read_bytes() == b"OLD JSON" and (d / "child").read_text() == "KEEP" and _leftovers(tmp_path) == []


def test_existing_out_directory_rejected_old_html_survives(manifest_case, tmp_path, monkeypatch):
    out, page = tmp_path / "o.json", tmp_path / "r.html"
    out.mkdir()
    (out / "child").write_text("KEEP")
    page.write_bytes(b"OLD HTML")
    with pytest.raises(ValueError, match="디렉터리"):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(out), "--html", str(page)])
    assert page.read_bytes() == b"OLD HTML" and (out / "child").read_text() == "KEEP" and _leftovers(tmp_path) == []


def test_staged_bytes_never_broader_than_existing_destination(manifest_case, tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX permission bits")
    real_fdopen = os.fdopen
    seen = []

    class Spy:
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            self.f.__enter__()
            return self

        def __exit__(self, *a):
            return self.f.__exit__(*a)

        def write(self, data):
            r = self.f.write(data)
            self.f.flush()
            seen.append(stat.S_IMODE(os.fstat(self.f.fileno()).st_mode))  # 내용이 쓰인 시점의 임시 파일 모드
            return r

        def __getattr__(self, name):
            return getattr(self.f, name)

    monkeypatch.setattr("psdl1.search.os.fdopen", lambda *a, **k: Spy(real_fdopen(*a, **k)))
    old = os.umask(0o022)
    try:
        page, out = tmp_path / "r.html", tmp_path / "o.json"
        for f in (page, out):
            f.write_text("OLD")
            f.chmod(0o600)
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(out), "--html", str(page)])
    finally:
        os.umask(old)
    assert len(seen) == 2 and all(m & 0o077 == 0 for m in seen), [oct(m) for m in seen]
    assert stat.S_IMODE(page.stat().st_mode) == stat.S_IMODE(out.stat().st_mode) == 0o600


def test_json_bytes_identical_with_and_without_html_under_translating_newlines(manifest_case, tmp_path, monkeypatch):
    # Windows를 모사: 텍스트 모드 쓰기에서 newline 기본값(None)을 "\r\n"으로 번역하는 opener로 두 경로를 모두 지나게 한다.
    real_open, real_fdopen = io.open, os.fdopen

    def win_newline(mode, newline):
        return "\r\n" if "w" in mode and "b" not in mode and newline is None else newline

    def fake_open(f, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *a, **k):
        return real_open(f, mode, buffering, encoding, errors, win_newline(mode, newline), *a, **k)

    def fake_fdopen(fd, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *a, **k):
        return real_fdopen(fd, mode, buffering, encoding=encoding, errors=errors, newline=win_newline(mode, newline), *a, **k)

    monkeypatch.setattr(io, "open", fake_open)
    monkeypatch.setattr("psdl1.search.os.fdopen", fake_fdopen)
    plain, withhtml = tmp_path / "a.json", tmp_path / "b.json"
    run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(plain)])
    run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(withhtml), "--html", str(tmp_path / "r.html")])
    assert b"\r\n" in plain.read_bytes()  # 모사가 실제로 적용됐다
    assert plain.read_bytes() == withhtml.read_bytes()


# ---- 라운드 5: 끝 구분자 사전 거부, stdout은 준비 성공 뒤에만 ---------------------

@pytest.mark.parametrize("final", ["absent", "file"])
@pytest.mark.parametrize("sep", ["/", os.sep])
def test_html_trailing_separator_rejected_old_json_survives(manifest_case, tmp_path, monkeypatch, final, sep):
    out, page = tmp_path / "o.json", tmp_path / "r.html"
    out.write_bytes(b"OLD JSON")
    if final == "file":
        page.write_bytes(b"OLD HTML")
    with pytest.raises(ValueError, match="디렉터리 형태"):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(out), "--html", str(page) + sep])
    assert out.read_bytes() == b"OLD JSON" and _leftovers(tmp_path) == []
    assert (page.read_bytes() == b"OLD HTML") if final == "file" else not page.exists()


@pytest.mark.parametrize("final", ["absent", "file"])
@pytest.mark.parametrize("sep", ["/", os.sep])
def test_out_trailing_separator_rejected_old_html_survives(manifest_case, tmp_path, monkeypatch, final, sep):
    out, page = tmp_path / "o.json", tmp_path / "r.html"
    page.write_bytes(b"OLD HTML")
    if final == "file":
        out.write_bytes(b"OLD JSON")
    with pytest.raises(ValueError, match="디렉터리 형태"):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(out) + sep, "--html", str(page)])
    assert page.read_bytes() == b"OLD HTML" and _leftovers(tmp_path) == []
    assert (out.read_bytes() == b"OLD JSON") if final == "file" else not out.exists()


def test_stdout_empty_when_html_staging_fails(manifest_case, tmp_path, monkeypatch, capsys):
    with pytest.raises(FileNotFoundError):
        run_cli(manifest_case, tmp_path, monkeypatch, ["--html", str(tmp_path / "missing" / "r.html")])
    assert capsys.readouterr().out == "" and _leftovers(tmp_path) == []


# ---- 라운드 6: 원본 경로 해석은 스테이징에서, 불변 플래그는 사전 거부 ----------------

@pytest.mark.parametrize("arg", ["missing/../r.html", "missing/.", "missing/..", "file/../r.html"])
@pytest.mark.parametrize("use_out", [True, False])
def test_invalid_original_traversal_fails_before_any_publish(manifest_case, tmp_path, monkeypatch, capsys, arg, use_out):
    out, page = tmp_path / "out.json", tmp_path / "r.html"
    (tmp_path / "file").write_text("F")
    page.write_bytes(b"OLD HTML")
    extra = ["--html", f"{tmp_path}/{arg}"]
    if use_out:
        out.write_bytes(b"OLD JSON")
        extra = ["--out", str(out)] + extra
    with pytest.raises((OSError, ValueError)):
        run_cli(manifest_case, tmp_path, monkeypatch, extra)
    assert page.read_bytes() == b"OLD HTML" and capsys.readouterr().out == ""
    assert (out.read_bytes() == b"OLD JSON") if use_out else not out.exists()
    assert _leftovers(tmp_path) == []


@pytest.mark.skipif(not hasattr(os, "chflags") or not hasattr(stat, "UF_IMMUTABLE"), reason="no os.chflags")
@pytest.mark.parametrize("which", ["html", "out"])
def test_immutable_destination_rejected_before_any_publish(manifest_case, tmp_path, monkeypatch, which):
    out, page = tmp_path / "out.json", tmp_path / "r.html"
    out.write_bytes(b"OLD JSON")
    page.write_bytes(b"OLD HTML")
    locked = page if which == "html" else out
    os.chflags(locked, stat.UF_IMMUTABLE)
    def unlock():
        try:
            os.chflags(locked, 0)
        except OSError:
            pass
    try:
        with pytest.raises(ValueError, match="불변"):
            run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(out), "--html", str(page)])
        assert out.read_bytes() == b"OLD JSON" and page.read_bytes() == b"OLD HTML" and _leftovers(tmp_path) == []
    finally:
        unlock()


# ---- 라운드 7: 심볼릭 링크 뒤의 '..' 별칭은 원본 문자열을 realpath로 먼저 해석해 잡는다 ----------

def _symlink_tree(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX symlinks")
    d = tmp_path / "D"
    (d / "sub" / "deep").mkdir(parents=True)
    os.symlink("sub/deep", d / "link")
    return d


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("out_first", [True, False])
def test_symlink_dotdot_alias_rejected_before_anything_written(manifest_case, tmp_path, monkeypatch, capsys, relative, out_first):
    d = _symlink_tree(tmp_path)
    alias, real = "D/link/../same.json", "D/sub/same.json"
    if not relative:
        alias, real = str(tmp_path / alias), str(tmp_path / real)
    monkeypatch.chdir(tmp_path)
    opts = [("--out", alias), ("--html", real)]
    extra = [x for pair in (opts if out_first else opts[::-1]) for x in pair]
    with pytest.raises(ValueError, match="같은 파일"):
        run_cli(manifest_case, tmp_path, monkeypatch, extra)
    assert not (d / "sub" / "same.json").exists() and not (d / "same.json").exists()
    assert capsys.readouterr().out == ""
    assert [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp") or p.name == "same.json"] == []


def test_distinct_paths_through_symlinked_parent_succeed(manifest_case, tmp_path, monkeypatch):
    d = _symlink_tree(tmp_path)
    run_cli(manifest_case, tmp_path, monkeypatch, ["--out", str(d / "link" / ".." / "a.json"), "--html", str(d / "link" / "b.html")])
    assert (d / "sub" / "a.json").is_file() and (d / "sub" / "deep" / "b.html").is_file()
    assert [p for p in tmp_path.rglob("*.tmp")] == []
