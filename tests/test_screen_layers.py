"""screen-metrics/T4: screen() 가 Layer 사이드카로 정렬 컬럼을 더하되 기존 판정 경로는 바이트까지 그대로 둔다."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from psdl1.png_contract import mask_hash
from psdl1.screen_relations import load_floor_spec, load_screening_layers, main, screen, write_csvs
from test_screen_align import blobs, put
from test_screen_relations import FACTOR, case, incremental_inputs  # noqa: F401  (case is a fixture)

LEGACY_COLUMNS = ["product_a", "product_b", "partition", "group_a", "group_b", "lineage_a", "lineage_b", "similarity", "usage_jaccard",
                  "geometry_distance", "anchor", "shift_y_um", "shift_x_um", "alignment_score", "roi_overlap", "alignment_flag", "tier", "reason"]
NEW_COLUMNS = ["metrics_anchor", "similarity_layer", "layer_shift_y_um", "layer_shift_x_um", "layer_align_dice", "layer_align_second_dice",
                "layer_align_decided", "layer_align_disagreeing", "layer_align_flag"]
FRAME = (96, 96)
NAMES = ("Metal", "via", "Poly")


def layer_masks(origin):
    """같은 설계(층 3장)를 프레임 안 origin(전 해상도 px) 에 놓은 Layer 마스크."""
    return {name: put(FRAME, blobs((40, 40), 200 + k, count=7, low=4, high=12), origin) for k, name in enumerate(NAMES)}


def write_sidecar(root, per_product, frame_id="F", mutate=None):
    """사이드카를 manifest 폴더에 쓴다. per_product: {product_id: {층 이름: bool 마스크}}. mutate(doc) 로 JSON 을 망가뜨릴 수 있다."""
    (root / "screening_layers").mkdir(exist_ok=True)
    doc = {"schema_version": "dl1_screening_layers_v1", "products": {}}
    for pid, masks in per_product.items():
        layers = []
        for number, (name, mask) in enumerate(masks.items(), 1):
            relative = f"screening_layers/{pid}-{number}.npy"
            np.save(root / relative, mask)
            layers.append({"layer": name, "filename_number": number, "relative_path": f"{pid}/Layer/{number}-{name}.png", "mask": relative, "mask_sha256": mask_hash(mask)})
        doc["products"][pid] = {"frame_id": frame_id, "layers": layers}
    if mutate is not None:
        mutate(doc)
    (root / "screening-layers.json").write_text(json.dumps(doc))
    return doc


def scores_rows(result, root, name):
    out, scores = root / f"{name}.csv", root / f"{name}.csv.scores.csv"
    write_csvs(result, out, scores)
    return out.read_bytes(), list(csv.DictReader(scores.read_text(encoding="utf-8").splitlines()))


def pair(rows, a, b):
    return next(r for r in rows if r["product_a"] == a and r["product_b"] == b)


def test_layer_alignment_adds_columns_but_leaves_the_existing_judgement_byte_identical(case):
    manifest, split, root, _ = case
    without = screen(manifest, split, factor=FACTOR, margin=0.0)
    relations_without, rows_without = scores_rows(without, root, "without")
    write_sidecar(root, {"BUS_R00": layer_masks((30, 30)), "BUS_T00": layer_masks((10, 20))})  # CNT 정렬은 (-8, -6), Layer 정렬은 (10, 5) 를 고른다
    with_layers = screen(manifest, split, factor=FACTOR, margin=0.0)
    relations_with, rows_with = scores_rows(with_layers, root, "with")
    assert relations_with == relations_without
    assert [[r[c] for c in LEGACY_COLUMNS] for r in rows_with] == [[r[c] for c in LEGACY_COLUMNS] for r in rows_without]
    for key in ("pairs_by_tier", "unknown_by_reason", "floor", "review", "products_without_different_pair"):
        assert with_layers["summary"][key] == without["summary"][key]
    for key in ("similarity", "usage_jaccard", "geometry"):  # 새 쌍 지표 AUC 는 Layer 위치에서 계산하므로 달라질 수 있다
        assert with_layers["summary"]["separation_auc_same_vs_other_lineage"][key] == without["summary"]["separation_auc_same_vs_other_lineage"][key]
    assert with_layers["summary"]["alignment"]["anchors"] == without["summary"]["alignment"]["anchors"]
    plain, layered = pair(rows_without, "BUS_R00", "BUS_T00"), pair(rows_with, "BUS_R00", "BUS_T00")
    assert [layered[c] for c in LEGACY_COLUMNS] == [plain[c] for c in LEGACY_COLUMNS] and layered["anchor"] == "CNT" and layered["shift_y_um"] == "-0.8"
    assert layered["metrics_anchor"] == "LAYER" and plain["metrics_anchor"] == "CNT" and plain["layer_align_flag"] == "layer_missing"
    assert float(layered["similarity_layer"]) != float(layered["similarity"]) and float(layered["similarity_layer"]) < float(layered["similarity"])
    assert (layered["layer_shift_y_um"], layered["layer_shift_x_um"]) == ("1.0", "0.5") and layered["layer_align_dice"] == "1.0"
    assert layered["layer_align_flag"] == "" and layered["layer_align_disagreeing"] == "0" and layered["layer_align_decided"] in {n.casefold() for n in NAMES}
    assert layered["layer_align_second_dice"] != "" and float(layered["layer_align_second_dice"]) < 1
    for row in rows_without + [r for r in rows_with if (r["product_a"], r["product_b"]) != ("BUS_R00", "BUS_T00")]:
        assert row["layer_align_flag"] == "layer_missing" and row["metrics_anchor"] == row["anchor"] and row["similarity_layer"] == row["similarity"]
        assert all(row[c] == "" for c in NEW_COLUMNS[2:-1])
    assert list(rows_with[0])[:18] == LEGACY_COLUMNS and list(rows_with[0])[18:27] == NEW_COLUMNS
    assert list(rows_with[0])[27:] == ["usage_jaccard_electrodes", "electrode_similarity", "pattern_similarity", "cnt_similarity"]
    summary = with_layers["summary"]
    assert summary["alignment"]["metrics_anchors"] == {"CNT": 5, "LAYER": 1} and without["summary"]["alignment"]["metrics_anchors"] == {"CNT": 6}
    assert summary["layer_alignment"]["flags"]["layer_missing"] == 5 and summary["layer_alignment"]["flags"]["ambiguous_alignment"] == 0
    assert summary["layer_alignment"]["excluded_layers"] == {"empty": 0, "covers_valid": 0, "insufficient_support": 0, "no_candidate_support": 0}


def test_layer_rows_and_pattern_similarity_exist_only_where_the_layer_shift_is_used(case):
    manifest, split, root, _ = case
    write_sidecar(root, {"BUS_R00": layer_masks((30, 30)), "BUS_T00": layer_masks((10, 20))})
    result = screen(manifest, split, factor=FACTOR, margin=0.0)
    by = {(p["product_a"], p["product_b"]): p for p in result["pairs"]}
    layered = by[("BUS_R00", "BUS_T00")]
    assert layered["metrics_anchor"] == "LAYER" and layered["pattern_similarity"] == 1.0  # 복제 Layer 를 Layer 이동에서 비교
    rows = [r for r in layered["metric_rows"] if r["kind"] == "layer"]
    assert [r["key"] for r in rows] == sorted(n.casefold() for n in NAMES) and all(r["status"] == "both" and r["boundary_f1"] == 1.0 for r in rows)
    assert all(r["area_a_px"] == r["area_b_px"] for r in rows)
    for key, p in by.items():
        if key != ("BUS_R00", "BUS_T00"):
            assert p["pattern_similarity"] is None and all(r["kind"] != "layer" for r in p["metric_rows"])
    summary = result["summary"]["separation_auc_same_vs_other_lineage"]
    assert summary["n_pairs_with_value"]["pattern_similarity"] == 1 and summary["n_pairs_with_value"]["similarity_layer"] == 6


def test_incremental_old_pairs_are_skipped_and_a_missing_layer_falls_back(case, tmp_path):
    manifest, split, root, _ = case
    base_split = {**split, "train": ["BUS_R00", "BUS_T00"], "validation": [], "test": []}
    base_relations = root / "base-relations.csv"
    base_relations.write_text("")
    groups = root / "groups.json"
    groups.write_text(json.dumps({pid: "x" for pid in split["train"]}))
    floor_json = root / "floor.json"
    floor_json.write_text(json.dumps({"floor": {"global": 0.9, "per_group": {"x": 0.8}}}))
    kwargs = dict(factor=FACTOR, margin=0.0, groups=groups, base_split=base_split, base_relations=base_relations, floor_spec=load_floor_spec(floor_json))
    without = screen(manifest, split, **kwargs)
    write_sidecar(root, {"COMB_R00": layer_masks((36, 36)), "COMB_T00": layer_masks((14, 40))})  # 새×새 쌍만 양쪽에 Layer 가 있다. CNT 정렬은 (-8, 12)
    layered = screen(manifest, split, **kwargs)
    keys = {(p["product_a"], p["product_b"]) for p in layered["pairs"]}
    assert keys == {("BUS_R00", "COMB_R00"), ("BUS_R00", "COMB_T00"), ("BUS_T00", "COMB_R00"), ("BUS_T00", "COMB_T00"), ("COMB_R00", "COMB_T00")}  # old x old 없음
    assert layered["summary"]["floor"]["per_group"] == {"x": 0.8} == without["summary"]["floor"]["per_group"] and layered["summary"]["floor"]["global"] == 0.9
    assert scores_rows(layered, root, "inc-layer")[0] == scores_rows(without, root, "inc-plain")[0]
    assert [(p["tier"], p["reason"], p["similarity"], p["floor"]) for p in layered["pairs"]] == [(p["tier"], p["reason"], p["similarity"], p["floor"]) for p in without["pairs"]]
    by = {(p["product_a"], p["product_b"]): p for p in layered["pairs"]}
    assert by[("COMB_R00", "COMB_T00")]["metrics_anchor"] == "LAYER" and by[("COMB_R00", "COMB_T00")]["layer_align_flag"] == ""
    assert (by[("COMB_R00", "COMB_T00")]["layer_shift_y_um"], by[("COMB_R00", "COMB_T00")]["layer_shift_x_um"]) == (1.1, -0.2)
    for key, p in by.items():
        if key != ("COMB_R00", "COMB_T00"):
            assert p["layer_align_flag"] == "layer_missing" and p["metrics_anchor"] == p["anchor"] and p["similarity_layer"] == p["similarity"]
    assert layered["summary"]["alignment"]["metrics_anchors"] == {"CNT": 4, "LAYER": 1}


def test_cli_columns_flags_and_arguments_without_nan_or_inf(case, capsys):
    manifest, split, root, _ = case
    write_sidecar(root, {"BUS_R00": layer_masks((30, 30)), "BUS_T00": layer_masks((10, 20))})
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))

    def cli(name, *extra):
        out = root / f"{name}.csv"
        main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--factor", str(FACTOR), "--margin", "0.0", *extra])
        capsys.readouterr()
        text = (root / f"{name}.csv.scores.csv").read_text(encoding="utf-8")
        summary_text = (root / f"{name}.csv.summary.json").read_text(encoding="utf-8")
        for cell in (c.strip().lower() for line in csv.reader(text.splitlines()) for c in line):
            assert cell not in {"nan", "inf", "-inf", "+inf"}
        json.loads(summary_text, parse_constant=lambda token: pytest.fail(f"non-finite JSON constant {token}"))  # NaN / Infinity 는 JSON 에 쓸 수 없다
        return list(csv.DictReader(text.splitlines())), json.loads(summary_text)
    rows, summary = cli("default")
    assert pair(rows, "BUS_R00", "BUS_T00")["metrics_anchor"] == "LAYER" and summary["alignment"]["metrics_anchors"] == {"CNT": 5, "LAYER": 1}
    rows, summary = cli("strict", "--align-min-support", "100000")
    assert pair(rows, "BUS_R00", "BUS_T00")["layer_align_flag"] == "layer_no_common" and pair(rows, "BUS_R00", "BUS_T00")["metrics_anchor"] == "CNT"
    assert summary["layer_alignment"]["flags"]["layer_no_common"] == 1 and summary["layer_alignment"]["excluded_layers"]["insufficient_support"] == 3
    assert all(pair(rows, "BUS_R00", "BUS_T00")[c] == "" for c in NEW_COLUMNS[2:-1])
    rows, _ = cli("overlap", "--align-min-overlap", "0.99")  # 진짜 이동 (10, 5) 는 겹침 0.71 이라 후보가 아니다. 남은 후보 (0, 0) 은 Dice 가 낮다
    assert pair(rows, "BUS_R00", "BUS_T00")["layer_shift_y_um"] != "1.0" and pair(rows, "BUS_R00", "BUS_T00")["metrics_anchor"] == "CNT"
    rows, _ = cli("dice", "--align-min-dice", "1.0")  # 완전 복제의 Dice 1.0 은 문턱 미만이 아니다
    assert pair(rows, "BUS_R00", "BUS_T00")["metrics_anchor"] == "LAYER"
    for bad in ("--align-min-support", "0"), ("--align-min-dice", "2"), ("--align-min-overlap", "-0.1"):
        with pytest.raises(ValueError, match="align_min"):
            main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(root / "bad.csv"), "--factor", str(FACTOR), *bad])


def test_search_bound_and_unreliable_alignments_are_not_applied_but_their_values_are_kept(case):
    manifest, split, root, _ = case
    write_sidecar(root, {"BUS_R00": layer_masks((30, 30)), "BUS_T00": layer_masks((10, 20))})
    bound = pair(scores_rows(screen(manifest, split, factor=FACTOR, margin=0.0, max_shift=0.2), root, "bound")[1], "BUS_R00", "BUS_T00")  # 반경 10 칸: 이동 10 은 경계다
    assert bound["layer_align_flag"] == "alignment_at_search_bound" and bound["metrics_anchor"] == "CNT" and bound["similarity_layer"] == bound["similarity"]
    assert bound["layer_shift_y_um"] == "1.0" and bound["layer_align_dice"] == "1.0"
    masks_b = layer_masks((10, 20))
    rng = np.random.RandomState(1)
    noisy = {n: m & (rng.rand(*m.shape) > 0.85) for n, m in masks_b.items()}  # 2x2 any-reduction keeps about half of the raster pixels
    write_sidecar(root, {"BUS_R00": layer_masks((30, 30)), "BUS_T00": noisy})
    unreliable = pair(scores_rows(screen(manifest, split, factor=FACTOR, margin=0.0, align_min_dice=0.95), root, "unreliable")[1], "BUS_R00", "BUS_T00")
    assert unreliable["layer_align_flag"] == "alignment_unreliable" and unreliable["metrics_anchor"] == "CNT" and 0.3 < float(unreliable["layer_align_dice"]) < 0.95


def test_ambiguous_periodic_layers_are_applied_with_the_flag(case):
    manifest, split, root, _ = case
    stripes = np.zeros(FRAME, bool)
    for x in range(4, 96, 16):
        stripes[:, x:x + 6] = True
    write_sidecar(root, {"BUS_R00": {"grating": stripes}, "BUS_T00": {"grating": stripes.copy()}})
    row = pair(scores_rows(screen(manifest, split, factor=FACTOR, margin=0.0), root, "periodic")[1], "BUS_R00", "BUS_T00")
    assert row["layer_align_flag"] == "ambiguous_alignment" and row["metrics_anchor"] == "LAYER"


# ---- 사이드카: 없음은 정상, 손상은 오류 -------------------------------------------------------------------------------------
def test_missing_sidecar_means_no_layers_and_corruption_is_an_error(case):
    manifest, split, root, _ = case
    needed = set(split["train"])
    assert load_screening_layers(manifest, needed, FACTOR) == {}
    (root / "screening_layers").mkdir()  # 폴더만 있고 사이드카 JSON 이 없어도 Layer 없음이다
    assert load_screening_layers(manifest, needed, FACTOR) == {}
    write_sidecar(root, {"BUS_R00": layer_masks((30, 30))})
    loaded = load_screening_layers(manifest, needed, FACTOR)
    assert set(loaded) == {"BUS_R00"} and set(loaded["BUS_R00"]) == {n.casefold() for n in NAMES}
    assert all(r.dtype == np.bool_ and r.shape == (48, 48) for r in loaded["BUS_R00"].values())  # block_reduce(any, 2)
    assert load_screening_layers(manifest, {"COMB_R00"}, FACTOR) == {}  # 필요한 제품만 읽는다


@pytest.mark.parametrize("damage,message", [
    ("hash", "mask_sha256"), ("bytes", "mask_sha256"), ("dtype", "dtype bool"), ("shape", "dtype bool"), ("missing_file", "No such file"),
    ("unknown_product", "not in the manifest"), ("frame_id", "frame_id"), ("duplicate_name", "duplicate layer name"), ("escape", "contained bundle path"),
    ("outside_dir", "mask path"), ("extra_field", "layer fields"), ("schema", "schema_version"), ("not_json", "corrupt|Expecting"), ("empty_layers", "frame_id"),
])
def test_corrupt_sidecars_are_errors_and_not_silently_treated_as_missing(case, damage, message):
    manifest, split, root, _ = case
    per_product = {"BUS_R00": layer_masks((30, 30)), "BUS_T00": layer_masks((10, 20))}

    def mutate(doc):
        layers = doc["products"]["BUS_R00"]["layers"]
        if damage == "hash":
            layers[0]["mask_sha256"] = "0" * 64
        elif damage == "unknown_product":
            doc["products"]["GHOST"] = doc["products"]["BUS_R00"]
        elif damage == "frame_id":
            doc["products"]["BUS_R00"]["frame_id"] = "OTHER"
        elif damage == "duplicate_name":
            layers[1]["layer"] = layers[0]["layer"].upper()
        elif damage == "escape":
            layers[0]["mask"] = "screening_layers/../manifest.json"
        elif damage == "outside_dir":
            layers[0]["mask"] = "assets/a.npy"
        elif damage == "extra_field":
            layers[0]["note"] = "x"
        elif damage == "schema":
            doc["schema_version"] = "dl1_screening_layers_v0"
        elif damage == "empty_layers":
            doc["products"]["BUS_R00"]["layers"] = []
    doc = write_sidecar(root, per_product, mutate=mutate)
    first = root / doc["products"]["BUS_R00"]["layers"][0]["mask"] if damage in {"bytes", "dtype", "shape", "missing_file"} else None
    if damage == "bytes":
        flipped = np.load(first)
        flipped[0, 0] = ~flipped[0, 0]
        np.save(first, flipped)
    elif damage == "dtype":
        np.save(first, np.load(first).astype(np.uint8))
    elif damage == "shape":
        np.save(first, np.zeros((95, 96), bool))
    elif damage == "missing_file":
        first.unlink()
    elif damage == "not_json":
        (root / "screening-layers.json").write_text("{not json")
    with pytest.raises(ValueError, match=message):
        screen(manifest, split, factor=FACTOR)


def test_a_damaged_sidecar_of_a_product_that_is_not_screened_is_not_read(case):
    manifest, split, root, _ = case
    write_sidecar(root, {"BUS_R00": layer_masks((30, 30)), "BUS_T00": layer_masks((10, 20)), "RING_R00": layer_masks((30, 30))})
    (root / "screening_layers" / "RING_R00-1.npy").write_bytes(b"garbage")
    result = screen(manifest, split, factor=FACTOR, margin=0.0)  # train 구획만 본다. RING 은 test 구획이다
    assert len(result["pairs"]) == 6
    with pytest.raises(ValueError, match="RING_R00-1.npy"):
        screen(manifest, split, factor=FACTOR, partitions=("train", "test"))


def test_options_are_validated(case):
    manifest, split, *_ = case
    for bad in ({"align_min_support": 0}, {"align_min_support": 1.5}, {"align_min_dice": 1.5}, {"align_min_overlap": -1}):
        with pytest.raises(ValueError, match="align_min"):
            screen(manifest, split, factor=FACTOR, **bad)
