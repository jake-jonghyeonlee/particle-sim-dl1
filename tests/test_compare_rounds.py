import json

import pytest

from psdl1.compare_rounds import comparison_inputs_match, compare, main


def query(qid, recall, known=4, unknown=2, lineage=None):
    return {"query_id": qid, "lineage_id": lineage or qid[0], "metric_status": "available" if recall is not None else "unavailable",
            "known_candidate_pairs": known, "unknown_candidate_pairs": unknown, "recall_at_k": recall}


def evaluation(path, queries, *, weights="w1", topk=5, input_hash="in1"):
    available = [q["recall_at_k"] for q in queries if q.get("recall_at_k") is not None]
    path.write_text(json.dumps({"schema_version": "dl1_evaluation_v1", "topk": topk, "checkpoint_weights_hash": weights, "input_hash": input_hash, "split_hash": "s",
                                "evaluated_queries": len(available), "recall_at_k": sum(available) / len(available) if available else None, "queries": queries}, indent=2) + "\n")
    return path


@pytest.fixture
def rounds(tmp_path):
    previous = evaluation(tmp_path / "prev.json", [query("A1", 1.0), query("B1", 0.5), query("C1", 0.5), query("D1", None)])
    current = evaluation(tmp_path / "cur.json", [query("A1", 0.5, known=6), query("B1", 1.0), query("C1", 0.5), query("D1", 1.0), query("E1", 0.75), query("F1", None)],
                         weights="w2", input_hash="in2")
    return previous, current, tmp_path


def test_compare_marks_regressed_common_queries_and_lists_new_ones(rounds):
    previous, current, _ = rounds
    result = compare(previous, current)
    by = {q["query_id"]: q for q in result["common_queries"]}
    assert {k: v["status"] for k, v in by.items()} == {"A1": "regressed", "B1": "improved", "C1": "unchanged", "D1": "unavailable"}
    assert by["A1"]["delta"] == -0.5 and by["A1"]["previous_candidates"] == 6 and by["A1"]["current_candidates"] == 8 and by["D1"]["delta"] is None
    assert [q["query_id"] for q in result["new_queries"]] == ["E1", "F1"] and result["new_queries"][1]["metric_status"] == "unavailable"
    summary = result["summary"]
    assert summary["regressed_queries"] == 1 and summary["improved_queries"] == 1 and summary["new_mean"] == 0.75
    assert summary["common_mean_previous"] == pytest.approx(2 / 3) and summary["common_mean_current"] == pytest.approx(2 / 3) and summary["common_delta"] == pytest.approx(0.0)
    assert result["warnings"] == ["candidate_pool_grew: recall is not directly comparable"]
    assert compare(previous, current, drop_tolerance=0.5)["summary"]["regressed_queries"] == 0


def test_compare_warns_on_forgetting_and_same_input_hash(tmp_path):
    previous = evaluation(tmp_path / "p.json", [query("A1", 1.0), query("B1", 1.0)])
    current = evaluation(tmp_path / "c.json", [query("A1", 0.5), query("B1", 0.5)], weights="w2")
    warnings = compare(previous, current)["warnings"]
    assert warnings[0].startswith("possible_forgetting:") and any(w.startswith("same_input_hash") for w in warnings)


def test_compare_reports_lost_metric_and_queries_missing_from_current(tmp_path):
    previous = evaluation(tmp_path / "p.json", [query("A1", 1.0), query("B1", 0.5), query("G1", 0.5, lineage="LG")])
    current = evaluation(tmp_path / "c.json", [query("A1", None), query("B1", 0.5)], weights="w2", input_hash="in2")
    result = compare(previous, current)
    assert {q["query_id"]: q["status"] for q in result["common_queries"]} == {"A1": "lost_metric", "B1": "unchanged"}
    assert result["missing_queries"] == [{"query_id": "G1", "lineage_id": "LG"}]
    assert "queries_lost_metric: 1" in result["warnings"] and "queries_missing_from_current: 1" in result["warnings"]


@pytest.mark.parametrize("field", ["known_candidate_pairs", "unknown_candidate_pairs", "recall_at_k"])
def test_compare_rejects_queries_missing_required_fields(tmp_path, field):
    bad = query("A1", 1.0)
    del bad[field]
    previous = evaluation(tmp_path / "p.json", [bad])
    current = evaluation(tmp_path / "c.json", [query("A1", 1.0)], weights="w2")
    with pytest.raises(ValueError, match="without query_id"):
        compare(previous, current)
    with pytest.raises(ValueError, match="without query_id"):
        compare(current, previous)


def test_compare_rejects_mismatched_topk_and_same_checkpoint(rounds, tmp_path):
    previous, current, _ = rounds
    with pytest.raises(ValueError, match="topk differs"):
        compare(previous, evaluation(tmp_path / "k.json", [query("A1", 1.0)], weights="w2", topk=3))
    with pytest.raises(ValueError, match="same checkpoint"):
        compare(previous, evaluation(tmp_path / "s.json", [query("A1", 1.0)], weights="w1"))
    (tmp_path / "round.json").write_text("{}")
    with pytest.raises(ValueError, match="not a dl1_evaluation_v1"):
        compare(previous, tmp_path / "round.json")
    with pytest.raises(ValueError, match="drop_tolerance"):
        compare(previous, current, drop_tolerance=-0.1)


def test_compare_cli_writes_json_and_refuses_overwrite(rounds, capsys):
    previous, current, root = rounds
    out = root / "round-comparison.json"
    main(["--previous", str(previous), "--current", str(current), "--out", str(out)])
    assert json.loads(capsys.readouterr().out)["regressed_queries"] == 1
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["schema_version"] == "dl1_round_comparison_v1" and document["previous"]["checkpoint_weights_hash"] == "w1" and document["current"]["checkpoint_weights_hash"] == "w2"
    with pytest.raises(FileExistsError):
        main(["--previous", str(previous), "--current", str(current), "--out", str(out)])


def test_comparison_records_input_file_hashes_and_detects_replaced_inputs(rounds):
    previous, current, root = rounds
    out = root / "round-comparison.json"
    main(["--previous", str(previous), "--current", str(current), "--out", str(out)])
    assert comparison_inputs_match(out, previous, current) is None
    # same checkpoint weights hash, different recall: the comparison no longer matches the file
    replaced = evaluation(root / "cur.json.new", [query("A1", 1.0), query("B1", 0.25), query("C1", 0.5), query("D1", 1.0), query("E1", 0.75), query("F1", None)],
                          weights="w2", input_hash="in2")
    assert "current evaluation file differs" in comparison_inputs_match(out, previous, replaced)
    assert "previous evaluation file differs" in comparison_inputs_match(out, replaced, current)
    assert "cannot be read" in comparison_inputs_match(root / "missing.json", previous, current)
    old = json.loads(out.read_text(encoding="utf-8"))
    del old["current"]["file_sha256"]
    out.write_text(json.dumps(old))
    assert "does not record" in comparison_inputs_match(out, previous, current)
