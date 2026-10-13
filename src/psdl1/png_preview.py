"""Bundle-local Pillow raster previews; display crops never become model input."""
from html import escape
from pathlib import Path

import numpy as np

from .png_contract import canonical_id, require_report_version

# Display palette and overview size are UI choices, never a model grid/alias map.
DISPLAY_MAX_EDGE = 900
CONTEXT_MARGIN_PX = 64
SEMANTIC_COLOURS = {"upper": (33, 115, 178), "lower": (175, 83, 105),
                    "other": (140, 140, 140), "intersection": (255, 210, 40), "cnt": (20, 20, 20)}
COLOURS = [(33, 115, 178), (115, 78, 172), (27, 145, 108), (175, 83, 105), (122, 103, 49)]


def _layer_image(mask, shape_xy, colour):
    from PIL import Image
    alpha = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize(shape_xy, Image.Resampling.BOX)
    image = Image.new("RGBA", shape_xy, colour + (0,))
    image.putalpha(alpha.point(lambda value: round(value * 0.62)))
    return image


def _draw_marker(draw, xy, state, index, radius=5):
    x, y = xy
    if state == "unresolved":
        draw.line((x - radius, y - radius, x + radius, y + radius), fill="#c32836", width=2)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill="#c32836", width=2)
    else:
        colour = "#bb7100" if state == "needs_confirmation" else "#087b4c"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour, width=2)
        if state == "needs_confirmation":
            draw.rectangle((x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), outline=colour)
    draw.text((x + radius + 2, y - radius), str(index), fill="#17242e")


def _skipped_text(skipped):
    electrode = "; ".join(f"{l.order_index}:{l.layer} [{', '.join(l.source_paths)}]" for l in skipped.electrode_layers) or "none"
    cno = "; ".join(f"{l.order_index}:{l.layer} [{', '.join(l.source_paths)}]" for l in skipped.cno_layers) or "none"
    return escape("electrode layers: " + electrode + " | cno layers: " + cno)


def _context_image(model, component):
    """Display-only source window; semantic colours never depend on enumeration."""
    from PIL import Image, ImageDraw
    h, w = model.frame.shape_yx
    x0, y0, x1, y1 = component.bbox_xyxy_px
    box = (max(0, x0 - CONTEXT_MARGIN_PX), max(0, y0 - CONTEXT_MARGIN_PX),
           min(w, x1 + CONTEXT_MARGIN_PX), min(h, y1 + CONTEXT_MARGIN_PX))
    a, b, z, d = box
    upper = {c.asset_id for c in component.selected_upper_candidates}
    lower = {c.asset_id for c in component.selected_lower_candidates}
    image = Image.new("RGBA", (z - a, d - b), "white")
    for asset in model.electrodes:
        key = "upper" if asset.asset_id in upper else "lower" if asset.asset_id in lower else "other"
        image.alpha_composite(_layer_image(asset.mask[b:d, a:z], image.size, SEMANTIC_COLOURS[key]))
    # J and outline are computed only in the component's bbox, then placed in
    # this display context. No padded display mask is sent to the model.
    local = component.mask
    if len(upper) == len(lower) == 1:
        assets = {asset.asset_id: asset for asset in model.electrodes}
        j = local & assets[next(iter(upper))].mask[y0:y1, x0:x1] & assets[next(iter(lower))].mask[y0:y1, x0:x1]
        patch = _layer_image(j, (x1 - x0, y1 - y0), SEMANTIC_COLOURS["intersection"])
        image.alpha_composite(patch, (x0 - a, y0 - b))
    padded = np.pad(local, 1)
    interior = padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    outline = Image.new("RGBA", (x1 - x0, y1 - y0), SEMANTIC_COLOURS["cnt"] + (0,))
    outline.putalpha(Image.fromarray((local & ~interior).astype(np.uint8) * 255))
    image.alpha_composite(outline, (x0 - a, y0 - b))
    if component.relation is not None:
        x, y = component.relation.representative_xy_px
        marker = ImageDraw.Draw(image)
        marker.ellipse((x-a-5, y-b-5, x-a+5, y-b+5), fill="white", outline="#087b4c", width=2)
        marker.line((x-a-7, y-b, x-a+7, y-b), fill="#087b4c", width=2)
        marker.line((x-a, y-b-7, x-a, y-b+7), fill="#087b4c", width=2)
    return image.convert("RGB"), box


def _stack_text(model, component):
    if model.stack_order.status != "resolved":
        return '<p>Unordered CNT inventory: ' + escape(component.cnt_layer +
            f" / filename number (provenance only) {component.cnt_filename_number} / " + component.source_path) + '</p>'
    r = component.relation
    skipped = set()
    if r is not None:
        for side in (r.skipped_upper_layers, r.skipped_lower_layers):
            for layer in side.electrode_layers + side.cno_layers:
                skipped.update(layer.asset_ids)
    involved = {c.asset_id for c in component.above_candidates + component.below_candidates}
    involved.add(component.source_asset_id)
    involved.update(skipped)
    lines = [f"position {model.order_index(a)}:{a.layer} / {a.electrode or 'cno'}" + (" [skipped, non-overlapping]" if a.asset_id in skipped else "")
             for a in sorted(model.assets, key=lambda a: (model.order_index(a) if model.order_index(a) is not None else -1, a.layer, a.electrode or 'cno')) if a.asset_id in involved]
    return '<p>Involved layers, top to bottom: ' + escape(' → '.join(lines)) + '</p>'


def write_preview(stage, inventory, extraction, *, report=None):
    from PIL import Image, ImageDraw
    directory = stage / "preview"
    directory.mkdir()
    rows, sections = [], []
    warnings = []
    warnings.extend(f"{w.path}: {w.reason}" for w in inventory.warnings)
    if report is not None:
        warnings.extend(f"{w.path}: {w.reason}" for w in report.diagnostics)
        warnings.extend("Possible alias/duplicate geometry: " + str(entry) for entry in report.duplicate_geometry)
    for model, result in zip(inventory.models, extraction):
        warnings.extend(f"{w.path}: {w.reason}" for w in model.warnings)
        warnings.extend(f"{e.path}: {e.reason}" for e in model.errors)
        if not model.valid:
            sections.append(f"<section><h2>{escape(model.model_id)}</h2><p>Invalid source; no raster preview.</p></section>")
            continue
        h, w = model.frame.shape_yx
        scale = min(1.0, DISPLAY_MAX_EDGE / max(h, w))
        ow, oh = max(1, round(w * scale)), max(1, round(h * scale))
        sx, sy = ow / w, oh / h
        model_key = canonical_id("preview", model.model_id)
        base_relative = "preview/" + model_key + "-overview.png"
        overview = Image.new("RGB", (ow, oh), "white")
        overlays, toggles = [], []
        display_assets = sorted(model.assets, key=lambda a: (
            model.order_index(a) if model.order_index(a) is not None else -1, a.layer, a.electrode or "", a.mask_sha256))
        for index, asset in enumerate(display_assets):
            colour = (30, 30, 30) if asset.kind == "cno" else COLOURS[index % len(COLOURS)]
            image = _layer_image(asset.mask, (ow, oh), colour)
            overview.paste(image, (0, 0), image)
            layer_relative = "preview/" + asset.asset_id + "-layer.png"
            image.save(stage / layer_relative)
            target = "layer-" + asset.asset_id
            overlays.append(f'<img class="overlay" id="{target}" src="{layer_relative}" alt="{escape(asset.relative_path, quote=True)}">')
            toggles.append(f'<label><input type="checkbox" checked data-layer="{target}"> {escape(asset.electrode or "cno")} / {escape(asset.layer)} / filename number (provenance only) {asset.filename_number}</label>')
        annotation = Image.new("RGBA", (ow, oh), (255, 255, 255, 0))
        draw = ImageDraw.Draw(annotation)
        bbox = model.frame.core_bbox_xyxy_px
        draw.rectangle((bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy), outline="#6648aa", width=2)
        draw.text((bbox[0] * sx + 3, bbox[1] * sy + 3), "core: reference only", fill="#6648aa")
        component_sections, crops = [], []
        for number, component in enumerate(result.components, 1):
            r = component.relation
            state = "unresolved" if r is None else "needs_confirmation" if r.needs_confirmation else "resolved"
            x0, y0, x1, y1 = component.bbox_xyxy_px
            position = r.representative_xy_px if r is not None else ((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2)
            _draw_marker(draw, ((position[0] + 0.5) * sx, (position[1] + 0.5) * sy), state, number)
            # Display-only crop, exactly one source pixel per image pixel. Include
            # electrode overlays within the original bbox, without full C raster.
            crop = Image.new("RGBA", (x1 - x0, y1 - y0), "white")
            for index, asset in enumerate(model.electrodes):
                layer = _layer_image(asset.mask[y0:y1, x0:x1], crop.size, COLOURS[index % len(COLOURS)])
                crop.alpha_composite(layer)
            black = Image.new("RGBA", crop.size, (20, 20, 20, 0))
            black.putalpha(Image.fromarray(component.mask.astype(np.uint8) * 180))
            crop.alpha_composite(black)
            if r is not None:
                # A 1-pixel endpoint indicates source cell, preserving crop size.
                cx, cy = r.representative_xy_px[0] - x0, r.representative_xy_px[1] - y0
                ImageDraw.Draw(crop).point((cx, cy), fill="#bb7100" if r.needs_confirmation else "#087b4c")
            crop_relative = "preview/" + component.component_id + "-crop.png"
            crop.convert("RGB").save(stage / crop_relative)
            context, context_box = _context_image(model, component)
            context_relative = "preview/" + component.component_id + "-context.png"
            context.save(stage / context_relative)
            display_factor = max(1, min(4, 512 // max(context.size)))
            enlarged_relative = "preview/" + component.component_id + "-context-display.png"
            context.resize((context.width * display_factor, context.height * display_factor), Image.Resampling.NEAREST).save(stage / enlarged_relative)
            anchor = "component-" + component.component_id
            above = ", ".join(f"{c.electrode}@{c.order_index}:{c.layer}" for c in component.selected_upper_candidates) or "none"
            below = ", ".join(f"{c.electrode}@{c.order_index}:{c.layer}" for c in component.selected_lower_candidates) or "none"
            reasons = "; ".join(component.unresolved_reasons) or "resolved under implementation rule"
            details = f'<p>Selected upper: {escape(above)}; lower: {escape(below)}. {escape(reasons)}</p>'
            if r is not None:
                details += f'<p>Endpoint source XY: {escape(str(r.representative_xy_px))}; µm: {escape(str(r.endpoints[0].xy_um))}; both endpoints share this centre.</p>'
                details += '<p>Skipped upper — ' + _skipped_text(r.skipped_upper_layers) + '</p>'
                details += '<p>Skipped lower — ' + _skipped_text(r.skipped_lower_layers) + '</p>'
            details += _stack_text(model, component)
            ignored = "; ".join(f"{c.candidate.electrode}@{c.candidate.order_index}:{c.candidate.layer}: {c.reason}" for c in component.ignored_lower_candidates) or "none"
            details += f'<p>Ignored lower candidates: {escape(ignored)}</p>'
            details += '<details><summary>All overlapping source candidates</summary><p>' + escape(str([
                (c.electrode, c.order_index, c.layer, c.overlap_pixels) for c in
                component.above_candidates + component.below_candidates])) + '</p></details>'
            component_sections.append(f'<article class="{state}" id="{anchor}"><h3>{number}. {state.replace("_", " ")}</h3>'
                                      f'<code>{component.component_id}</code><p>Source bbox XYXY: {component.bbox_xyxy_px}; '
                                      f'crop offset XY=({x0},{y0}); 1 source pixel / display image pixel.</p>'
                                      f'<p>Context window XYXY={context_box}; source offset {context_box[0]},{context_box[1]}; nearest-neighbour display ×{display_factor}.</p>'
                                      f'<a href="{context_relative}"><img class="crop" src="{enlarged_relative}" '
                                      f'alt="Original-resolution component {number} at source offset {x0},{y0}"></a>{details}</article>')
            crops.append({"component_id": component.component_id, "image": crop_relative, "bbox_xyxy_px": list(component.bbox_xyxy_px),
                          "source_offset_xy_px": [x0, y0], "shape_yx": [y1 - y0, x1 - x0], "source_pixels_per_image_pixel": 1,
                          "state": state, "model_input": False,
                          "context_image": context_relative, "context_bbox_xyxy_px": list(context_box),
                          "context_source_offset_xy_px": list(context_box[:2]),
                          "context_shape_yx": [context.height, context.width], "context_margin_px": CONTEXT_MARGIN_PX,
                          "context_source_pixels_per_image_pixel": 1, "display_image": enlarged_relative,
                          "display_integer_scale": display_factor, "display_resampling": "nearest"})
        annotation_relative = "preview/" + model_key + "-annotations.png"
        annotation.save(stage / annotation_relative)
        overview.paste(annotation, (0, 0), annotation)
        overview.save(stage / base_relative)
        # Layer controls operate on separate raster layers above a plain white
        # canvas; downloadable overview is the composited full-frame evidence.
        view = (f'<div class="viewport"><div class="stack" style="width:{ow}px;height:{oh}px">' +
                ''.join(overlays) + f'<img class="overlay" src="{annotation_relative}" alt="Core reference and source-coordinate markers"></div></div>')
        model_status = "available" if result.connectivity_observation == "complete" else result.connectivity_observation
        if model.stack_order.status == "resolved":
            order_html = '<p>Resolved top-to-bottom order (user_supplied_order_file)</p><ol start="0">' + ''.join(
                '<li>' + escape(f"position {e.order_index}: {e.kind} / {e.name}" +
                                " / assets " + ', '.join(e.asset_ids)) + '</li>'
                for e in model.stack_order.resolved_order) + '</ol>'
        elif model.stack_order.status == "required":
            order_html = '<p><strong>Stack order required — supply --stack-order; CNT relations were not extracted.</strong></p><p>Source overlays are an unordered inventory.</p>'
        else:
            order_html = '<p>Unordered source inventory. ' + escape('; '.join(model.stack_order.errors)) + '</p>'
        sections.append(f'<section><h2>{escape(model.model_id)}</h2>' + order_html + f'<p>Extractor run: {"ran" if result.extraction_status == "complete" else "blocked"}; '
                        f'extraction coverage: {result.connectivity_observation}; model connectivity status: {model_status}.</p>'
                        f'<p>Full source frame H×W={h}×{w}; origin (0,0), x→right, y→down; x=0…{w-1}, y=0…{h-1}; '
                        f'pitch 0.1 µm/pixel. Overview scale XY=({sx:.8f},{sy:.8f}).</p><p><a href="{base_relative}">Download full-frame overview</a></p>'
                        '<div class="controls">' + ''.join(toggles) + '</div>' + view +
                        '<p>Pan with the scrollbars; browser zoom or the source crops provides magnification. Green circle=resolved; '
                        'amber square/circle=needs confirmation; red X=unresolved; purple rectangle=reference-only core.</p>' +
                        ''.join(component_sections) + '</section>')
        rows.append({"model_id": model.model_id, "overview": base_relative, "source_shape_yx": [h, w],
                     "overview_shape_yx": [oh, ow], "display_scale_xy": [sx, sy], "comparison_scope": "full_frame",
                     "core_bbox_use": "reference_only", "component_crops": crops})
    warnings_html = ''.join('<li>' + escape(text) + '</li>' for text in warnings) or '<li>None recorded.</li>'
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>DL1 PNG source inspection</title><style>body{font:16px system-ui;margin:0;background:#f1f4f7;color:#17242e}'
            'header,main{max-width:1120px;margin:auto;padding:24px}header{background:#17242e;color:white}section{background:white;padding:24px;margin:24px 0;border-radius:12px}'
            'article{padding:16px;margin:16px 0;border-left:5px solid #087b4c;background:#f6f8fa}article.unresolved{border-color:#c32836}'
            'article.needs_confirmation{border-color:#bb7100;background:#fff9ed}code{overflow-wrap:anywhere;font-size:12px}'
            '.viewport{overflow:auto;max-height:700px;border:1px solid #bbc5ce;background:white}.stack{position:relative}.overlay{position:absolute;inset:0;width:100%;height:100%}'
            '.controls{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}.crop{image-rendering:pixelated;min-width:100px;max-width:100%;max-height:400px;object-fit:contain;object-position:left;border:1px solid #aab5bf}'
            'p{line-height:1.5}h2{margin-top:0}a{color:#185e9f}</style><header><h1>DL1 PNG source inspection</h1>'
            '<p><strong>automatic extraction; electrical verification not performed</strong></p>'
            '<p>Display crops are for inspection only, not model input. Model input always retains the full frame and real margins.</p>'
            '<p>Context legend: blue=selected upper electrodes; rose=selected lower electrodes (all candidates on each selected layer); gray=other electrodes; black=CNT outline; yellow=common intersection J; green crosshair=shared endpoint.</p><p>complete means extraction coverage only; electrical verification not performed. 4-neighbour components are user accepted; skipped-layer physical interpretation remains unconfirmed.</p></header>'
            '<main><details open><summary>Source warnings and possible duplicates</summary><ul>' + warnings_html + '</ul></details>' + ''.join(sections) + '</main>'
            '<script>document.querySelectorAll("[data-layer]").forEach(x=>x.addEventListener("change",()=>{document.getElementById(x.dataset.layer).hidden=!x.checked;}));</script></html>')
    (stage / "preview.html").write_text(html, encoding="utf-8")
    return tuple(rows)


def require_preview_report(report):
    """Preview readers must reject unsupported sidecar versions."""
    require_report_version(report)
