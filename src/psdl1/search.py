"""Independent query/candidate search with explicit availability and stable ties."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import unicodedata
from pathlib import Path

import torch

from .checkpoint import read_checkpoint, validate_config
from .data import load_manifest, input_identity, grid_check
from .model import ProductEncoder
from .schema import approval_status
from .train import encode_products


def load_model(ckpt):
    saved = read_checkpoint(ckpt)
    cfg = saved["config"]
    model = ProductEncoder(len(cfg["roles"]), len(cfg["layers"]), cfg["backbone"], None,
                           use_unit_size=cfg["use_unit_size"], **cfg["model_kwargs"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.config_fingerprint = saved["fingerprint"]
    model.training_metadata = saved["metadata"]
    model.trained_epochs = saved["epoch"]
    model.weights_hash = saved["weights_hash"]
    return model.eval(), cfg


def vocabulary_reason(product, cfg):
    roles = sorted({e.role for e in product.elements} - cfg["roles"].keys())
    layers = sorted({e.layer for e in product.elements} - cfg["layers"].keys())
    if roles or layers:
        return {"reason": "unknown_training_vocabulary", "roles": roles, "layers": layers}
    if any(e.mask is None for e in product.elements):
        return {"reason": "missing_shape_observation"}
    return None


def inference_grid_selection(products, cfg, *, allow_grid_override=False):
    if type(allow_grid_override) is not bool:
        raise ValueError("allow_grid_override must be boolean")
    candidates = sorted({p.grid_check["manifest_candidate_grid"] for p in products
                         if p.grid_check is not None and "manifest_candidate_grid" in p.grid_check})
    conflicts = [value for value in candidates if value != cfg["grid"]]
    if conflicts and not allow_grid_override:
        raise ValueError(f"checkpoint grid {cfg['grid']} conflicts with manifest candidate_grid {conflicts}; --allow-grid-override required")
    return {"selected_grid": cfg["grid"], "manifest_candidate_grids": candidates,
            "source": "checkpoint", "override": bool(conflicts), "override_requested": allow_grid_override}


def inference_grid_checks(products, grid, *, allow_grid_check_failure=False):
    if type(allow_grid_check_failure) is not bool:
        raise ValueError("allow_grid_check_failure must be boolean")
    checks = []
    for p in products:
        report = grid_check(p, grid)
        failed = any(e.get("thin_line_warning") or e.get("bridge_or_component_loss") for e in report["elements"])
        checks.append({**report, "gate_status": "bypassed" if failed and allow_grid_check_failure else "blocked" if failed else "passed"})
    return {"policy": "strict_default", "bypass_requested": allow_grid_check_failure, "products": checks}


@torch.no_grad()
def embed(model, cfg, products, *, batch_size=32, cache=None):
    fingerprint = validate_config(cfg)
    if getattr(model, "config_fingerprint", fingerprint) != fingerprint:
        raise ValueError("model/config fingerprint mismatch")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for p in products:
        reason = vocabulary_reason(p, cfg)
        if reason:
            raise ValueError(f"{p.product_id}: {reason}")
    model.eval()
    if not products:
        return torch.empty(0, model.head[-1].out_features)
    # Canonical inference shape: neither padding neighbours nor requested chunk size
    # changes the model's arithmetic. Chunks bound memory; forwards are per product.
    cache = {} if cache is None else cache
    vectors = []
    for start in range(0, len(products), batch_size):
        for p in products[start:start + batch_size]:
            key = (id(model), fingerprint, p.product_id, input_identity(p))
            if key not in cache:
                cache[key] = encode_products(model, [p], cfg["grid"], cfg["roles"], cfg["layers"]).cpu()
            vectors.append(cache[key])
    z = torch.cat(vectors)
    if not torch.isfinite(z).all():
        raise ValueError("nonfinite search embeddings")
    return z


@torch.no_grad()
def search(model, cfg, query, candidates, *, topk=5, batch_size=32, allow_grid_override=False, allow_grid_check_failure=False, embedding_cache=None):
    if type(topk) is not int or topk <= 0:
        raise ValueError("topk must be positive")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    fingerprint = validate_config(cfg)
    if getattr(model, "config_fingerprint", fingerprint) != fingerprint:
        raise ValueError("model/config fingerprint mismatch")
    if len({p.product_id for p in candidates}) != len(candidates):
        raise ValueError("candidate product IDs must be unique")
    selection = inference_grid_selection([query, *candidates], cfg, allow_grid_override=allow_grid_override)
    gate = inference_grid_checks([query, *sorted(candidates, key=lambda p: p.product_id)], cfg["grid"], allow_grid_check_failure=allow_grid_check_failure)
    blocked = {r["product_id"] for r in gate["products"] if r["gate_status"] == "blocked"}
    result = {"grid": cfg["grid"], "grid_selection": selection, "grid_gate": gate,
              "inference_policy": "one_forward_per_product_cached_within_evaluation; batch_size_is_chunk_bound",
              "query_id": query.product_id, "approval": approval_status(), "status": "unavailable", "reason": None,
              "rankings": [], "excluded": [], "candidate_count": len(candidates), "eligible_count": 0,
              "score_policy": "canonical_single_product_float64_cosine_round6_then_id",
              "fingerprint": validate_config(cfg), "query_connectivity": query.connectivity_status}
    reason = vocabulary_reason(query, cfg)
    if reason or query.product_id in blocked:
        result["reason"] = reason or {"reason": "grid_check_failed"}
        return result
    eligible = []
    for p in candidates:
        reason = {"reason": "query_itself"} if p.product_id == query.product_id else vocabulary_reason(p, cfg)
        reason = reason or ({"reason": "grid_check_failed"} if p.product_id in blocked else None)
        if reason:
            result["excluded"].append({"product_id": p.product_id, **reason})
        else:
            eligible.append(p)
    result["excluded"].sort(key=lambda row: row["product_id"])
    result["eligible_count"] = len(eligible)
    if not eligible:
        result["reason"] = "no_candidates" if not candidates else "no_eligible_candidates"
        return result
    cache_kwargs = {} if embedding_cache is None else {"cache": embedding_cache}
    qz = embed(model, cfg, [query], batch_size=batch_size, **cache_kwargs)[0]
    # Single-query search retains scores only; evaluation supplies a per-command embedding cache.
    scores = []
    for start in range(0, len(eligible), batch_size):
        chunk = eligible[start:start + batch_size]
        vectors = embed(model, cfg, chunk, batch_size=batch_size, **cache_kwargs).to(torch.float64)
        scores.extend(round(float((vector * qz.to(torch.float64)).sum()), 6) for vector in vectors)
    ranked = sorted(zip(eligible, scores), key=lambda pair: (-pair[1], pair[0].product_id))[:topk]
    result.update(status="available", rankings=[{"product_id": p.product_id, "score": score,
                    "connectivity_status": p.connectivity_status, "input_identity": input_identity(p)} for p, score in ranked])
    return result


def rank(model, cfg, query, candidates, *, topk=None, batch_size=32):
    result = search(model, cfg, query, candidates, topk=max(1, len(candidates)) if topk is None else topk, batch_size=batch_size)
    return [(row["product_id"], row["score"]) for row in result["rankings"]]


def _identity(path):
    """대소문자/유니코드 정규화를 무시한 절대 경로 키(아직 없는 경로의 별칭도 잡는다)."""
    p = os.path.normcase(os.path.realpath(path))
    return unicodedata.normalize("NFC", p).casefold()


def _same_file(a, b):
    if _identity(a) == _identity(b):
        return True
    return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)


def _check_replaceable(opt, path):
    """기존 목적지가 os.replace로 바뀔 수 없게 표시돼 있으면(BSD/macOS 불변·추가전용 플래그, Windows 읽기 전용) 게시 전에 거부한다."""
    st = os.stat(path)
    restricted = 0
    for name in ("UF_IMMUTABLE", "UF_APPEND", "SF_IMMUTABLE", "SF_APPEND"):
        restricted |= getattr(stat, name, 0)
    if getattr(st, "st_flags", 0) & restricted:
        raise ValueError(f"기존 출력 파일이 불변/추가 전용 플래그로 보호돼 있어 교체할 수 없습니다({opt}): {path}")
    if getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_READONLY", 0):
        raise ValueError(f"기존 출력 파일이 읽기 전용입니다({opt}): {path}")


def _check_html_options(a):
    """--html 사전 검사: 어떤 출력도 쓰기 전에 실패시킨다."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise ValueError("--html 미리보기에는 Pillow가 필요합니다(NPY 입력이어도 마찬가지): pip install 'psdl1[png]'") from None
    if a.out and _same_file(a.out, a.html):
        raise ValueError(f"--html과 --out이 같은 파일입니다: {a.html}")
    seps = {"/", os.sep} | ({os.altsep} if os.altsep else set())
    for opt, path in (("--out", a.out), ("--html", a.html)):
        if not path:
            continue
        if path.endswith(tuple(seps)) or not os.path.basename(os.path.normpath(path)):
            raise ValueError(f"출력 경로가 디렉터리 형태입니다({opt}, 파일 경로여야 합니다): {path}")
        if os.path.basename(path) in (".", ".."):
            raise ValueError(f"출력 경로의 마지막 요소가 '.' 또는 '..'입니다({opt}, 파일 경로여야 합니다): {path}")
        if os.path.isdir(path):
            raise ValueError(f"{opt}가 기존 디렉터리입니다(파일 경로여야 합니다): {path}")
        if os.path.exists(path):
            _check_replaceable(opt, path)


# 비-HTML 경로의 Path.write_text(payload)와 같은 텍스트 모드 기본값(플랫폼 줄바꿈 변환 포함). JSON은 항상 이것으로 쓴다.
_JSON_TEXT = {}
_HTML_TEXT = {"encoding": "utf-8", "newline": ""}


def _stage(dest, data, text=_JSON_TEXT):
    """dest와 같은 폴더에 고유한 임시 파일을 배타 생성(O_EXCL)하고 내용을 쓴다. 임시 경로를 돌려준다.

    임시 파일의 내용이 최종 목적지보다 넓은 principal에게 읽히는 일이 없도록, 목적지가 이미 있으면 0600으로 만들어
    내용을 쓴 뒤 닫기 전에 fchmod로 목적지의 기존 모드를 적용한다(없으면 0666에 umask)."""
    # 사용자가 준 문자열 그대로의 부모(없으면 ".")에 만든다: missing/.. 같은 경로는 OS가 직접 해석하다 여기서(게시 전에) 실패한다.
    parent = os.path.dirname(dest) or "."
    name = os.path.basename(dest)
    final_mode = stat.S_IMODE(os.stat(dest).st_mode) if os.path.exists(dest) else None
    while True:
        tmp = os.path.join(parent, f".{name}.{secrets.token_hex(8)}.tmp")
        try:
            fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666 if final_mode is None else 0o600)
            break
        except FileExistsError:
            continue
    try:
        with os.fdopen(fd, "w", **text) as f:
            f.write(data)
            if final_mode is not None:
                f.flush()
                if hasattr(os, "fchmod"):
                    os.fchmod(f.fileno(), final_mode)
                else:
                    os.chmod(tmp, final_mode)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return tmp


def _publish_both(out_path, payload, html_path, page):
    """두 출력을 먼저 임시 파일로 만든 뒤(실패해도 목적지는 그대로) os.replace로 JSON → HTML 순서로 게시한다.

    os.replace는 디렉터리 항목을 바꾸므로 목적지의 심볼릭 링크를 따라가 잘라내지 않는다. 보장: 게시 전 실패는 아무것도 바꾸지 않고,
    링크를 통해 다른 파일이 잘리지 않는다. 검사와 게시 사이에 적대적 행위자가 경로를 바꾸는 경쟁은 범위 밖이다(단일 사용자 CLI)."""
    staged = []
    try:
        json_tmp = _stage(out_path, payload)
        staged.append(json_tmp)
        html_tmp = _stage(html_path, page, _HTML_TEXT)
        staged.append(html_tmp)
        os.replace(json_tmp, out_path)
        staged.remove(json_tmp)
        if _same_file(out_path, html_path):
            raise ValueError(f"--html과 --out이 같은 파일입니다: {html_path}")
        os.replace(html_tmp, html_path)
        staged.remove(html_tmp)
    finally:
        for tmp in staged:
            if os.path.lexists(tmp):
                os.unlink(tmp)


def _write_html(html_path, page, before_publish=None):
    """HTML을 먼저 임시 파일로 만들고, before_publish()(예: stdout JSON 출력)를 부른 뒤 게시한다."""
    tmp = _stage(html_path, page, _HTML_TEXT)
    try:
        if before_publish is not None:
            before_publish()
        os.replace(tmp, html_path)
        tmp = None
    finally:
        if tmp and os.path.lexists(tmp):
            os.unlink(tmp)


def main(argv=None):
    ap = argparse.ArgumentParser(description="DL1 candidate search JSON")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--candidates-manifest", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--allow-grid-override", action="store_true")
    ap.add_argument("--allow-grid-check-failure", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--html", help="자체 완결 HTML 보고서 경로(JSON 출력과 별개로 추가)")
    a = ap.parse_args(argv)
    if a.html:
        _check_html_options(a)
    model, cfg = load_model(a.ckpt)
    queries = {p.product_id: p for p in load_manifest(a.query_manifest, grid=cfg["grid"])}
    if a.query not in queries:
        raise ValueError("query ID absent from independent query manifest")
    candidates = load_manifest(a.candidates_manifest, grid=cfg["grid"])
    result = search(model, cfg, queries[a.query], candidates, topk=a.topk, batch_size=a.batch_size, allow_grid_override=a.allow_grid_override, allow_grid_check_failure=a.allow_grid_check_failure)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    page = None
    if a.html:
        from .search_report import render_report
        page = render_report(result, queries[a.query], {p.product_id: p for p in candidates})
    if page is None:
        if a.out:
            Path(a.out).write_text(payload)
        else:
            print(payload, end="")
    elif a.out:
        _publish_both(a.out, payload, a.html, page)
    else:
        _write_html(a.html, page, lambda: print(payload, end=""))

if __name__ == "__main__":
    main()
