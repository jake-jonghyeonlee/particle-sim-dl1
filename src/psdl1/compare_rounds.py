"""라운드 비교: 이전 라운드와 현재 라운드의 evaluation.json을 쿼리 단위로 맞춰 recall 변화를 보고한다.

r(k-1) 체크포인트를 rk 데이터에 다시 평가하지 않는다(evaluate()는 split·input_hash가 체크포인트와 같아야 한다). 두 평가는 후보 풀이 달라서
recall 하락이 전부 망각은 아니다. 그래서 쿼리마다 후보 수를 같이 적고 풀이 커졌으면 경고한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA_VERSION = "dl1_round_comparison_v1"
EVALUATION_VERSION = "dl1_evaluation_v1"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_evaluation(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != EVALUATION_VERSION or not isinstance(document.get("queries"), list):
        raise ValueError(f"{path} is not a {EVALUATION_VERSION} file")
    for q in document["queries"]:
        if not isinstance(q, dict) or any(type(q.get(key)) is not int for key in ("known_candidate_pairs", "unknown_candidate_pairs")) or "recall_at_k" not in q:
            raise ValueError(f"{path} has a query without query_id/recall_at_k/known_candidate_pairs/unknown_candidate_pairs: {q.get('query_id') if isinstance(q, dict) else q!r}")
    ids = [q.get("query_id") for q in document["queries"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} lists a query_id twice")
    return document


def side(path, document):
    return {"path": str(path), "file_sha256": file_sha256(path), "checkpoint_weights_hash": document.get("checkpoint_weights_hash"), "input_hash": document.get("input_hash"),
            "split_hash": document.get("split_hash"), "recall_at_k": document.get("recall_at_k"), "evaluated_queries": document.get("evaluated_queries")}


def candidates(query):
    return query["known_candidate_pairs"] + query["unknown_candidate_pairs"]  # 후보 풀 크기: 쿼리가 평가에서 본 eligible 후보 수


def mean(values):
    return sum(values) / len(values) if values else None


def compare(previous_path, current_path, *, drop_tolerance=0.0):
    if isinstance(drop_tolerance, bool) or not isinstance(drop_tolerance, (int, float)) or not math.isfinite(drop_tolerance) or drop_tolerance < 0:
        raise ValueError("drop_tolerance must be a finite number >= 0")
    previous, current = load_evaluation(previous_path), load_evaluation(current_path)
    if previous.get("topk") != current.get("topk"):
        raise ValueError(f"topk differs between rounds: {previous.get('topk')} vs {current.get('topk')}")
    if previous.get("checkpoint_weights_hash") is None or previous.get("checkpoint_weights_hash") == current.get("checkpoint_weights_hash"):
        raise ValueError("both evaluations come from the same checkpoint (or a checkpoint without weights hash); compare two different rounds")
    before, after = ({q["query_id"]: q for q in d["queries"]} for d in (previous, current))
    common, regressed, improved, lost, pairs_prev, pairs_cur = [], 0, 0, 0, [], []
    for qid in sorted(before.keys() & after.keys()):
        p, c = before[qid], after[qid]
        pr, cr = p["recall_at_k"], c["recall_at_k"]
        if pr is None or cr is None:
            status, delta = ("lost_metric" if pr is not None else "unavailable"), None
            lost += status == "lost_metric"
        else:
            delta = cr - pr
            status = "regressed" if cr < pr - drop_tolerance else "improved" if cr > pr + drop_tolerance else "unchanged"
            pairs_prev.append(pr)
            pairs_cur.append(cr)
        regressed += status == "regressed"
        improved += status == "improved"
        common.append({"query_id": qid, "lineage_id": c.get("lineage_id"), "previous_recall": pr, "current_recall": cr, "delta": delta,
                       "previous_candidates": candidates(p), "current_candidates": candidates(c), "status": status})
    new = [{"query_id": qid, "lineage_id": after[qid].get("lineage_id"), "recall_at_k": after[qid]["recall_at_k"], "metric_status": after[qid]["metric_status"]}
           for qid in sorted(after.keys() - before.keys())]
    mean_previous, mean_current = mean(pairs_prev), mean(pairs_cur)
    common_delta = None if mean_previous is None else mean_current - mean_previous
    summary = {"common_mean_previous": mean_previous, "common_mean_current": mean_current, "common_delta": common_delta, "regressed_queries": regressed,
               "improved_queries": improved, "new_mean": mean([q["recall_at_k"] for q in new if q["recall_at_k"] is not None])}
    warnings = []
    if common_delta is not None and common_delta < 0:
        warnings.append(f"possible_forgetting: mean recall of {len(pairs_prev)} common queries fell from {mean_previous:.4f} to {mean_current:.4f}")
    if any(q["current_candidates"] > q["previous_candidates"] for q in common):
        warnings.append("candidate_pool_grew: recall is not directly comparable")
    if previous.get("input_hash") == current.get("input_hash"):
        warnings.append("same_input_hash: both rounds were trained on the same data, so this is not a new round")
    if lost:
        warnings.append(f"queries_lost_metric: {lost}")
    missing = [{"query_id": qid, "lineage_id": before[qid].get("lineage_id")} for qid in sorted(before.keys() - after.keys())]
    if missing:
        warnings.append(f"queries_missing_from_current: {len(missing)}")
    return {"schema_version": SCHEMA_VERSION, "topk": current.get("topk"), "drop_tolerance": drop_tolerance, "previous": side(previous_path, previous), "current": side(current_path, current),
            "common_queries": common, "new_queries": new, "missing_queries": missing, "summary": summary, "warnings": warnings}


def comparison_inputs_match(out_path, previous_path, current_path):
    """이미 있는 round-comparison.json이 지금 두 평가 파일로 만든 것인지 본다. 맞으면 None, 아니면 이유 문자열(실행기가 skip 여부를 정할 때 쓴다)."""
    try:
        document = json.loads(Path(out_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"{out_path} cannot be read: {error}"
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        return f"{out_path} is not a {SCHEMA_VERSION} file"
    for name, path in (("previous", previous_path), ("current", current_path)):
        recorded = (document.get(name) or {}).get("file_sha256") if isinstance(document.get(name), dict) else None
        if recorded is None:
            return f"{out_path} does not record the {name} evaluation file hash"
        try:
            actual = file_sha256(path)
        except OSError as error:
            return f"{path} cannot be read: {error}"
        if recorded != actual:
            return f"{name} evaluation file differs from the one {out_path} was made from"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare two rounds' evaluation.json per query (recall change on common queries, list of new queries)")
    ap.add_argument("--previous", required=True, help="evaluation.json of the previous round")
    ap.add_argument("--current", required=True, help="evaluation.json of this round")
    ap.add_argument("--out", required=True, help="new round-comparison.json (an existing file is never overwritten)")
    ap.add_argument("--drop-tolerance", type=float, default=0.0, help="a common query counts as regressed only when its recall fell by more than this")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}; choose a new path")
    result = compare(a.previous, a.current, drop_tolerance=a.drop_tolerance)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"comparison": str(out), **result["summary"], "warnings": result["warnings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
