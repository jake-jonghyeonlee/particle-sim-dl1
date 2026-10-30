import json

import pytest

import psdl1.screen_relations as sr
from conftest import make_product
from psdl1.screen_relations import main, screen
from psdl1.split import SPLIT_VERSION
from test_pretrain import write_manifest
from test_screen_relations import FACTOR, embed

NAMES = ["A1", "A2", "A3", "A4", "B1", "B2"]  # sorted pair order: (A1,A2) (A1,A3) (A1,A4) (A1,B1) (A1,B2) (A2,A3) ... (B1,B2)
# call index -> similarity of the same-lineage pairs; every other pair is far below.
SIMILARITY = {0: 0.50, 1: 0.90, 2: 0.60, 5: 0.80, 6: 0.70, 9: 0.95, 14: 0.85}  # A-lineage: 0 1 2 5 6 9, B-lineage: 14
# sorted same-lineage values: .50 .60 .70 .80 .85 .90 .95 (n=7); A-lineage (group g1): .50 .60 .70 .80 .90 .95 (n=6)


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    offsets = [(10, 10), (30, 22), (20, 40), (40, 8), (14, 30), (36, 36)]
    products = [embed(make_product("bus", 48), n, n[0], 96, o) for n, o in zip(NAMES, offsets)]
    manifest = write_manifest(tmp_path, products, 64)
    split = {"schema_version": SPLIT_VERSION, "train": NAMES, "validation": [], "test": []}
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({n: "g1" if n[0] == "A" else "g2" for n in NAMES}))
    real, calls = sr.compare, []

    def fake(*args, **kwargs):
        result = dict(real(*args, **kwargs))
        result["similarity"] = SIMILARITY.get(len(calls), 0.1)
        calls.append(1)
        return result

    monkeypatch.setattr(sr, "compare", fake)

    def run(**kwargs):
        calls.clear()
        return screen(manifest, split, groups=groups, factor=FACTOR, margin=0.0, **kwargs)
    return run, manifest, split, tmp_path


def test_default_floor_is_the_minimum_and_summary_has_no_quantile_keys(fixed):
    run, *_ = fixed
    result = run()
    assert result["summary"]["floor"]["global"] == 0.5 and result["summary"]["floor"]["per_group"] == {"g1": 0.5}
    assert "quantile" not in result["summary"]["floor"] and "quantile_method" not in result["summary"]["floor"]


def test_quantile_zero_equals_the_minimum(fixed):
    run, *_ = fixed
    base = run()["summary"]["floor"]
    zero = run(floor_quantile=0)["summary"]["floor"]
    assert zero["global"] == base["global"] and zero["per_group"] == base["per_group"]
    assert zero["quantile"] == 0 and zero["quantile_method"] == "lower_index"


def test_quantile_uses_the_lower_index_value(fixed):
    run, *_ = fixed
    floor = run(floor_quantile=0.2)["summary"]["floor"]  # index floor(0.2*6)=1 -> .60 (ceil would give .70)
    assert floor["global"] == 0.6 and floor["quantile"] == 0.2 and floor["quantile_method"] == "lower_index"
    assert run(floor_quantile=1)["summary"]["floor"]["global"] == 0.95


def test_per_group_floor_uses_the_same_quantile(fixed):
    run, *_ = fixed
    result = run(floor_quantile=0.5)
    # global: floor(0.5*6)=3 -> .80; g1: floor(0.5*5)=2 -> .70; g2 has one pair (< min_group_pairs), so no group floor
    assert result["summary"]["floor"]["global"] == 0.8 and result["summary"]["floor"]["per_group"] == {"g1": 0.7}
    assert result["floor"]["groups"] == {"g1": 0.7}


@pytest.mark.parametrize("extra", [{"floor": 0.5}, {"floor_spec": {"global": 0.5, "groups": {}}}])
def test_quantile_is_exclusive_with_floor_and_floor_spec(fixed, extra):
    run, *_ = fixed
    with pytest.raises(ValueError, match="exclusive"):
        run(floor_quantile=0.2, **extra)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), "0.2", True])
def test_quantile_out_of_range_is_rejected(fixed, bad):
    run, *_ = fixed
    with pytest.raises(ValueError, match="floor_quantile"):
        run(floor_quantile=bad)


def test_cli_quantile_writes_summary_and_rejects_conflicts(fixed, capsys):
    run, manifest, split, root = fixed
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))
    base = ["--manifest", str(manifest), "--split", str(split_path), "--factor", str(FACTOR), "--margin", "0.0"]
    main([*base, "--out", str(root / "q.csv"), "--floor-quantile", "0.2"])
    assert json.loads(capsys.readouterr().out)["floor"]["quantile"] == 0.2
    summary = json.loads((root / "q.csv.summary.json").read_text(encoding="utf-8"))
    assert summary["floor"]["quantile"] == 0.2 and summary["floor"]["quantile_method"] == "lower_index"
    floor_json = root / "q.csv.summary.json"
    for n, extra in enumerate((["--floor", "0.5"], ["--floor-json", str(floor_json)], ["--floor-quantile", "1.5"], ["--floor-quantile", "nan"])):
        with pytest.raises(SystemExit) as stopped:
            main([*base, "--out", str(root / f"bad{n}.csv"), "--floor-quantile", "0.2", *extra] if n < 2 else [*base, "--out", str(root / f"bad{n}.csv"), *extra])
        assert stopped.value.code == 2
