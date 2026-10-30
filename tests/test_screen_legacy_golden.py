"""레거시 스크리닝 판정의 골든 고정. 기준 commit 7794f37 의 출력과 현재 코드의 출력을 비교한다.

scores.csv 는 기존 18개 컬럼만, summary.json 은 기존 키만 비교하므로 이후 Task 가 컬럼·키를 추가해도 통과한다.
골든 재생성: PSDL1_REGEN_SCREEN_GOLDEN=1 (판정식이 의도적으로 바뀔 때만).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from psdl1.screen_relations import main
from test_screen_relations import FACTOR, TOL, case, incremental_inputs  # noqa: F401  (case is a fixture)

GOLDEN = Path(__file__).parent / "golden" / "screen_legacy"
REGEN = os.environ.get("PSDL1_REGEN_SCREEN_GOLDEN") == "1"
LEGACY_SCORE_COLUMNS = ["product_a", "product_b", "partition", "group_a", "group_b", "lineage_a", "lineage_b", "similarity", "usage_jaccard",
                        "geometry_distance", "anchor", "shift_y_um", "shift_x_um", "alignment_score", "roi_overlap", "alignment_flag", "tier", "reason"]
BASE_PLACEHOLDER = "<BASE_RELATIONS>"


def legacy_summary(summary):
    """비교 대상 키만 고른다. 새 키는 허용한다."""
    return {"pairs_by_tier": summary["pairs_by_tier"], "unknown_by_reason": summary["unknown_by_reason"],
            "floor": {k: summary["floor"][k] for k in ("global", "per_group", "margin")},
            "separation_auc_same_vs_other_lineage": {k: summary["separation_auc_same_vs_other_lineage"][k] for k in ("similarity", "usage_jaccard", "geometry")},
            "alignment": {"anchors": summary["alignment"]["anchors"], "flags": summary["alignment"]["flags"]},
            "review": {"by_reason": summary["review"]["by_reason"]},
            "products_without_different_pair": summary["products_without_different_pair"]}


def run_cli(root, name, manifest, split, args):
    split_path = root / f"split-{name}.json"
    split_path.write_text(json.dumps(split))
    out = root / f"out-{name}" / "relations.csv"
    main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--factor", str(FACTOR), "--tolerance", str(TOL), *args])
    summary = json.loads(Path(str(out) + ".summary.json").read_text(encoding="utf-8"))
    if "base" in summary:
        summary["base"]["relations"] = BASE_PLACEHOLDER  # 임시 경로가 골든에 들어가지 않게 한다
    return {"relations.csv": out.read_bytes(), "scores.csv": Path(str(out) + ".scores.csv").read_bytes(),
            "summary.json": (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")}


def score_rows(data):
    rows = list(csv.DictReader(data.decode("utf-8").splitlines()))
    return [[row[c] for c in LEGACY_SCORE_COLUMNS] for row in rows]


def check(name, produced):
    folder = GOLDEN / name
    if REGEN:
        folder.mkdir(parents=True, exist_ok=True)
        for filename, data in produced.items():
            (folder / filename).write_bytes(data)
    golden = {filename: (folder / filename).read_bytes() for filename in produced}
    assert produced["relations.csv"] == golden["relations.csv"]
    assert score_rows(produced["scores.csv"]) == score_rows(golden["scores.csv"])
    assert legacy_summary(json.loads(produced["summary.json"])) == legacy_summary(json.loads(golden["summary.json"]))


def test_train_partition_defaults(case):
    manifest, split, root, _ = case
    produced = run_cli(root, "a", manifest, split, [])
    check("a_train", produced)


def test_test_partition_with_the_train_floor(case):
    manifest, split, root, _ = case
    floor = json.loads((GOLDEN / "a_train" / "summary.json").read_text(encoding="utf-8"))["floor"]["global"]
    check("b_test_fixed_floor", run_cli(root, "b", manifest, split, ["--partitions", "test", "--floor", repr(floor)]))


def test_groups_cap_cross_group_pairs(case):
    manifest, split, root, _ = case
    groups = root / "groups.json"
    groups.write_text(json.dumps({"BUS_R00": "mobile", "BUS_T00": "mobile", "COMB_R00": "it", "COMB_T00": "it", "RING_R00": "auto", "RING_T00": "auto"}))
    check("d_groups", run_cli(root, "d", manifest, split, ["--groups", str(groups), "--partitions", "train", "test"]))


def incremental_args(root, split, groups=None):
    _, base_split_path, base_relations, floor_json = incremental_inputs(root, split)
    args = ["--partitions", "train", "test", "--base-split", str(base_split_path), "--base-relations", str(base_relations), "--floor-json", str(floor_json)]
    return args if groups is None else [*args, "--groups", str(groups)]


def test_incremental_delta(case):
    manifest, split, root, _ = case
    check("c_incremental", run_cli(root, "c", manifest, split, incremental_args(root, split)))


def test_incremental_delta_keeps_the_group_floor(case):
    manifest, split, root, _ = case
    groups = root / "groups.json"
    groups.write_text(json.dumps({pid: "x" for part in ("train", "test") for pid in split[part]}))
    produced = run_cli(root, "e", manifest, split, incremental_args(root, split, groups))
    assert json.loads(produced["summary.json"])["floor"]["per_group"] == {"x": 0.8}
    check("e_incremental_group_floor", produced)


def test_golden_files_exist_for_every_scenario():
    names = {"a_train", "b_test_fixed_floor", "c_incremental", "d_groups", "e_incremental_group_floor"}
    assert {p.name for p in GOLDEN.iterdir() if p.is_dir()} == names
    for name in names:
        assert sorted(p.name for p in (GOLDEN / name).iterdir()) == ["relations.csv", "scores.csv", "summary.json"]
