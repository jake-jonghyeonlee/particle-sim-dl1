"""Offline preview S16: full-frame overview and original-resolution display crops."""
from html.parser import HTMLParser
import json

from PIL import Image
import pytest

from png_helpers import write_import_fixture, write_png
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
    assert run_import(ImportOptions(root, out))[1] == 0
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
    assert "3:N" in html and "4:C4" in html and "5:U5" in html
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
    assert run_import(ImportOptions(root, out))[1] == 0
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
    assert "Involved layers, top to bottom: 1:U / A → 3:N / N [skipped, non-overlapping] → 4:C4 / cno [skipped, non-overlapping] → 5:U5 / A [skipped, non-overlapping] → 6:C6 / cno → 7:L / B" in html
    assert "lower: B@7:L, D@7:L" in html and "multiple_lower_electrodes_on_selected_layer" in html
