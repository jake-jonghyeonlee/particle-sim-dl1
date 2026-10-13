"""Offline preview S16: full-frame overview and original-resolution display crops."""
from html.parser import HTMLParser
import json

from PIL import Image
import pytest

from png_helpers import write_stack_order, write_import_fixture, write_png
from psdl1.import_png import run_import
from psdl1.png_contract import ImportOptions
from psdl1.png_preview import require_preview_report


class References(HTMLParser):
    def __init__(self):
        super().__init__(); self.references = []
    def handle_starttag(self, tag, attrs):
        self.references.extend(value for key, value in attrs if key in {"src", "href"})


def test_preview_full_frame_and_unresolved_evidence(tmp_path):
    root = tmp_path / "source"; write_import_fixture(root)
    out = tmp_path / "out"
    assert run_import(ImportOptions(root, out, stack_order=write_stack_order(tmp_path / "stack-order.json", ['U', 'N', 'C4', 'U5', 'C6', 'L', 'X', 'Y', 'Z'])))[1] == 0
    doc = json.loads((out / "import-report.json").read_text())
    record, = doc["preview"]
    assert record["source_shape_yx"] == [1001, 1203]
    assert record["overview_shape_yx"] == [749, 900]
    assert record["display_scale_xy"] == pytest.approx([900 / 1203, 749 / 1001])
    assert record["comparison_scope"] == "full_frame" and record["core_bbox_use"] == "reference_only"
    crops = record["component_crops"]
    assert len(crops) == 4
    assert [c["source_offset_xy_px"] for c in crops] == [[20, 20], [600, 600], [700, 700], [800, 800]]
    assert [c["state"] for c in crops] == ["resolved", "resolved", "needs_confirmation", "unresolved"]
    for crop in crops:
        assert crop["source_pixels_per_image_pixel"] == 1 and crop["model_input"] is False
        assert crop["shape_yx"] == [3, 3]
        with Image.open(out / crop["image"]) as image: assert image.size == (3, 3)
    with Image.open(out / record["overview"]) as image:
        assert image.size == (900, 749)
        assert image.getpixel((400, 300)) == (255, 255, 255)
        # Thick source patterns and markers preserve both margin and core views.
        assert image.getpixel((16, 16)) != (255, 255, 255)
    html = (out / "preview.html").read_text()
    assert "automatic extraction; electrical verification not performed" in html
    assert "not model input" in html and "full frame and real margins" in html
    assert "x→right, y→down" in html and "source offset 20,20" in html
    assert "needs confirmation" in html and "multiple_lower_electrodes_on_selected_layer" in html
    assert "Skipped upper" in html and "electrode layers:" in html and "cno layers:" in html
    assert "1:N" in html and "2:C4" in html and "3:U5" in html
    assert "model connectivity status: partial" in html and "extraction coverage: partial" in html
    assert html.count('data-layer="') == 7 and html.count('<article class=') == 4
    references = References(); references.feed(html)
    assert references.references
    assert all(not value.startswith(("http:", "https:", "//", "data:")) and (out / value).is_file()
               for value in references.references)
    assert "fetch(" not in html and "@import" not in html
    require_preview_report(doc)
    with pytest.raises(ValueError): require_preview_report({"report_schema_version": "unsupported"})


def test_preview_escapes_source_names_and_surfaces_duplicate_geometry(tmp_path):
    root = tmp_path / "source"
    write_png(root, 'M<&"/A<&"/1-L<&".png')
    # Two internal aliases are valid source names but reported as possible duplicates.
    (root / "COPY").symlink_to(root / 'M<&"')
    out = tmp_path / "out"
    assert run_import(ImportOptions(root, out))[1] == 0
    report = json.loads((out / "import-report.json").read_text())
    assert any(e["kind"] == "identical_model_geometry" for e in report["duplicate_geometry"])
    assert any(e["kind"] == "identical_asset_geometry" for e in report["duplicate_geometry"])
    html = (out / "preview.html").read_text()
    assert 'M<&"' not in html and 'A<&"' not in html and 'L<&"' not in html
    assert "M&lt;&amp;&quot;" in html and "Possible alias/duplicate geometry" in html


def test_source_warnings_and_runtime_version_are_visible(tmp_path, monkeypatch):
    import psdl1.png_source as source
    root = tmp_path / "source"; write_png(root, "M/A/1-X.png")
    (root / "M/B").mkdir(); (root / "notes.txt").write_text("ignored")
    original = source._pillow
    def different_version():
        image, _ = original(); return image, "99.0.TEST"
    monkeypatch.setattr(source, "_pillow", different_version)
    out = tmp_path / "out"
    assert run_import(ImportOptions(root, out))[1] == 0
    doc = json.loads((out / "import-report.json").read_text())
    assert doc["inventory"]["decoder"]["pillow_version"] == "99.0.TEST"
    assert any("differs from tested" in issue["reason"] for issue in doc["diagnostics"])
    html = (out / "preview.html").read_text()
    assert "99.0.TEST" in html and "notes.txt" in html and "electrode folder contains no PNG" in html
    assert "17–18 bytes/pixel" in doc["inventory"]["models"][0]["resource_estimate"]["overhead_note"]


def test_preview_context_semantics_and_nearest_display_scale(tmp_path):
    from png_helpers import write_wide_import_fixture
    import numpy as np
    root = tmp_path / "source"; write_wide_import_fixture(root)
    out = tmp_path / "out"
    assert run_import(ImportOptions(root, out, stack_order=write_stack_order(tmp_path / "stack-order.json", ['U', 'N', 'C4', 'U5', 'C6', 'L', 'X', 'Y', 'Z'])))[1] == 0
    doc = json.loads((out / "import-report.json").read_text())
    crops = doc["preview"][0]["component_crops"]
    assert [c["context_bbox_xyxy_px"] for c in crops] == [[0, 0, 120, 120], [520, 520, 680, 680],
                                                           [720, 720, 880, 880], [920, 920, 1080, 1080]]
    assert [c["context_source_offset_xy_px"] for c in crops] == [[0, 0], [520, 520], [720, 720], [920, 920]]
    assert [c["display_integer_scale"] for c in crops] == [4, 3, 3, 3]
    for c in crops:
        assert c["context_margin_px"] == 64 and c["context_source_pixels_per_image_pixel"] == 1
        assert c["display_resampling"] == "nearest" and c["model_input"] is False
        with Image.open(out / c["context_image"]) as original, Image.open(out / c["display_image"]) as enlarged:
            assert original.size == tuple(reversed(c["context_shape_yx"]))
            scale = c["display_integer_scale"]
            np.testing.assert_array_equal(np.asarray(enlarged), np.repeat(np.repeat(np.asarray(original), scale, axis=0), scale, axis=1))
            # Shared endpoint is a visible multi-pixel marker, never a lone cell.
            if c["state"] != "unresolved":
                green = np.all(np.asarray(original) == [8, 123, 76], axis=-1)
                assert green.sum() >= 40
    with Image.open(out / crops[1]["context_image"]) as direct:
        assert direct.getpixel((10, 80)) == (117, 168, 207)  # upper-only blue
        assert direct.getpixel((80, 10)) == (205, 148, 162)  # lower-only rose
    html = (out / "preview.html").read_text()
    assert "blue=selected upper electrodes" in html and "rose=selected lower electrodes" in html
    assert "all candidates on each selected layer" in html and "gray=other electrodes" in html
    assert "black=CNT outline" in html and "yellow=common intersection J" in html and "green crosshair=shared endpoint" in html
    assert "Involved layers, top to bottom: position 0:U / A → position 1:N / N [skipped, non-overlapping] → position 2:C4 / cno [skipped, non-overlapping] → position 3:U5 / A [skipped, non-overlapping] → position 4:C6 / cno → position 5:L / B" in html
    assert "lower: B@5:L, D@5:L" in html and "multiple_lower_electrodes_on_selected_layer" in html


def test_rb_S16_asymmetric_offsets_marker_endpoint_and_crosshair(tmp_path):
    import numpy as np
    from png_helpers import write_model
    # Literal bbox [130,710,137,719] => J centre (133,714), local (3,4).
    pixels = tuple((y, x) for y in range(710, 719) for x in range(130, 137))
    root = tmp_path / "source"
    write_model(root, [("A", "1-U.png", pixels), ("B", "7-L.png", pixels), ("cno", "6_C.png", pixels)])
    out = tmp_path / "out"
    assert run_import(ImportOptions(root, out, stack_order=write_stack_order(tmp_path / "stack-order.json", ['U', 'C', 'L'])))[1] == 0
    record, = json.loads((out / "import-report.json").read_text())["preview"]
    crop, = record["component_crops"]
    assert crop["bbox_xyxy_px"] == [130, 710, 137, 719]
    assert crop["source_offset_xy_px"] == [130, 710] and crop["shape_yx"] == [9, 7]
    assert crop["context_bbox_xyxy_px"] == [66, 646, 201, 783]
    assert crop["context_source_offset_xy_px"] == [66, 646]
    with Image.open(out / crop["image"]) as image:
        assert image.size == (7, 9)
        assert image.getpixel((3, 4)) == (8, 123, 76)
        assert image.getpixel((4, 3)) != (8, 123, 76)
    with Image.open(out / crop["context_image"]) as image:
        assert image.size == (135, 137)
        assert image.getpixel((67, 68)) == (8, 123, 76)
        assert image.getpixel((67, 61)) == (8, 123, 76)
        assert image.getpixel((60, 68)) == (8, 123, 76)
        assert image.getpixel((77, 78)) != (8, 123, 76)
    annotation, = (out / "preview").glob("*-annotations.png")
    with Image.open(annotation) as image:
        green = np.all(np.asarray(image)[:, :, :3] == [8, 123, 76], axis=-1)
        yy, xx = np.nonzero(green)
        assert len(xx) >= 20
        # Hand-computed centre ~[99.88,534.64] after full-frame overview scaling.
        assert 94 <= xx.min() <= xx.max() <= 105
        assert 529 <= yy.min() <= yy.max() <= 540
    html = (out / "preview.html").read_text()
    assert "source offset 66,646" in html and "Source bbox XYXY: (130, 710, 137, 719)" in html


def test_rb_S16_escapes_invalid_model_candidates_and_stack(tmp_path):
    from png_helpers import write_model
    root = tmp_path / "source"
    pixels = ((710, 130), (710, 131), (711, 130), (711, 131))
    write_model(root, [('A<&"', '1-U<&".png', pixels), ('B<&"', '7-L<&".png', pixels),
                       ('cno', '6_C<&".png', pixels)], model='M<&"')
    bad = write_png(root, 'BAD<&"/A/1-X.png'); Image.new("L", (1203, 1001), 127).save(bad)
    out = tmp_path / "out"
    assert run_import(ImportOptions(root, out, stack_order=write_stack_order(tmp_path / "stack-order.json", ['U<&"', 'C<&"', 'L<&"'])))[1] == 2
    html = (out / "preview.html").read_text()
    assert '<h2>BAD&lt;&amp;&quot;</h2><p>Invalid source' in html
    assert 'Selected upper: A&lt;&amp;&quot;@0:U&lt;&amp;&quot;; lower: B&lt;&amp;&quot;@2:L&lt;&amp;&quot;' in html
    assert 'Involved layers, top to bottom: position 0:U&lt;&amp;&quot; / A&lt;&amp;&quot; → position 1:C&lt;&amp;&quot; / cno → position 2:L&lt;&amp;&quot; / B&lt;&amp;&quot;' in html
    assert all(name not in html for name in ['BAD<&"', 'M<&"', 'A<&"', 'B<&"', 'U<&"', 'L<&"', 'C<&"'])


def test_preview_lists_order_without_invented_cnt_position(tmp_path):
    from png_helpers import write_model
    root = tmp_path / 'source'; p,q,r = (10,20),(10,21),(10,22)
    write_model(root,[('A','1-U<&".png',(p,q,r)),('B','1-L<&".png',(p,q,r)),
        ('E','1-E<&".png',((900,900),)),('F','1-F<&".png',((900,900),)),
        ('cno','1-CNT<&".png',(p,q)),('cno','2-CNT<&".png',(q,r)),('cno','1-OTHER.png',())])
    # Syntax-only synthetic positions, never derived from file prefixes.
    order = write_stack_order(tmp_path/'order.json',['U<&"','E<&"','OTHER','CNT<&"','F<&"','L<&"'])
    related = tmp_path/'relations'
    assert run_import(ImportOptions(root=root,out=related,stack_order=order))[1] == 0
    doc = json.loads((related/'import-report.json').read_text())
    assert doc['extraction'][0]['summary']['resolved_relation_count'] == 2
    assert doc['extraction'][0]['summary']['needs_confirmation_relations'] == 2
    assert doc['extraction'][0]['summary']['skipped_upper_electrode_layers'] == 2
    assert doc['extraction'][0]['summary']['skipped_upper_cno_layers'] == 2
    html = (related/'preview.html').read_text()
    expected = ['position 0: electrode / U&lt;&amp;&quot;', 'position 1: electrode / E&lt;&amp;&quot;',
        'position 2: cno / OTHER','position 3: cno / CNT&lt;&amp;&quot;',
        'position 4: electrode / F&lt;&amp;&quot;','position 5: electrode / L&lt;&amp;&quot;']
    offsets = [html.index(text) for text in expected]
    assert offsets == sorted(offsets)
    assert 'Selected upper: A@0:U&lt;&amp;&quot;; lower: B@5:L&lt;&amp;&quot;' in html
    assert '1:E&lt;&amp;&quot;' in html and '2:OTHER' in html and '4:F&lt;&amp;&quot;' in html
    assert 'U<&"' not in html and 'CNT<&"' not in html
    image = tmp_path/'image_only'
    assert run_import(ImportOptions(root=root,out=image))[1] == 0
    doc = json.loads((image/'import-report.json').read_text())
    html = (image/'preview.html').read_text()
    assert 'CNT mode: image_only' in html and 'Geometry only:' in html
    assert 'Resolved top-to-bottom order' not in html and 'Selected upper:' not in html and 'Endpoint source XY' not in html
    assert len(doc['preview'][0]['component_crops']) == 2
    assert all(c['cnt_order_index'] is None and c['relation'] is None for c in doc['extraction'][0]['components'])
