from dataclasses import replace
import pytest

from conftest import make_product
from psdl1.data import build_vocab
from psdl1.split import SPLIT_VERSION, extend_split, lineage_split, validate_split
from psdl1.train import structured_batch


def test_resized_40_64_96_variants_split_across_train_test_are_leakage():
    products = [make_product("comb", size) for size in (40, 64, 96)]
    split = {"schema_version": SPLIT_VERSION, "train": [products[0].product_id], "validation": [],
             "test": [p.product_id for p in products[1:]]}
    with pytest.raises(ValueError, match="lineage leakage"):
        validate_split(products, split)


def test_lineage_split_reproducible_and_no_leakage():
    products = [replace(make_product("comb", size), product_id=f"{i}-{size}", lineage_id=f"family-{i}") for i in range(10) for size in (40, 64, 96)]
    split = lineage_split(products, seed=3)
    assert split == lineage_split(products, seed=3)
    parts = validate_split(products, split)
    assert len(parts["test"]) == len(parts["validation"]) == 6
    assert len(parts["train"]) == 18


def family_products(first, last, sizes=(40, 64)):
    return [replace(make_product("comb", size), product_id=f"{i}-{size}", lineage_id=f"family-{i}")
            for i in range(first, last) for size in sizes]


def test_extend_split_keeps_previous_assignments_and_lineages_together():
    old = family_products(0, 10)
    previous = lineage_split(old, seed=3)
    # family-0 gains a third variant; ten brand-new families arrive.
    grown = old + [replace(make_product("comb", 96), product_id="0-96", lineage_id="family-0")] + family_products(10, 20)
    split = extend_split(previous, grown, seed=3)
    parts = validate_split(grown, split)
    for key in ("train", "validation", "test"):
        assert set(previous[key]) <= set(split[key])
    owner = next(key for key in ("train", "validation", "test") if "0-40" in previous[key])
    assert "0-96" in split[owner]
    assert split["policy"] == "lineage_grouped_incremental" and split["schema_version"] == SPLIT_VERSION
    assert sum(len(v) for v in parts.values()) == len(grown)


def test_extend_split_assigns_new_lineages_by_target_fractions_deterministically():
    old = family_products(0, 10)
    previous = lineage_split(old, seed=3)
    grown = old + family_products(10, 30)
    split = extend_split(previous, grown, seed=5)
    assert split == extend_split(previous, grown, seed=5)
    lineages = {key: {p.lineage_id for p in validate_split(grown, split)[key]} for key in ("train", "validation", "test")}
    # 30 lineages * 0.2 = 6 per held-out partition, counting the 2 already held out.
    assert len(lineages["validation"]) == len(lineages["test"]) == 6
    assert len(lineages["train"]) == 18
    # Already above target: no new lineage goes to validation/test.
    assert extend_split(previous, old + family_products(10, 11), seed=5)["train"] == sorted(
        previous["train"] + ["10-40", "10-64"])
    # Without new products the split is unchanged.
    same = extend_split(previous, old, seed=9)
    assert all(same[key] == previous[key] for key in ("train", "validation", "test"))


def test_extend_split_rejects_missing_previous_products_and_bad_schema():
    old = family_products(0, 5)
    previous = lineage_split(old, seed=1)
    with pytest.raises(ValueError, match="previous split product missing"):
        extend_split(previous, old[1:] + family_products(5, 6))
    with pytest.raises(ValueError, match="incompatible split schema"):
        extend_split({**previous, "schema_version": "other"}, old)
    with pytest.raises(ValueError, match="incompatible split schema"):
        extend_split({**previous, "base": "x"}, old)
    with pytest.raises(ValueError, match="unique lists"):
        extend_split({**previous, "train": previous["train"] * 2}, old)
    with pytest.raises(ValueError, match="invalid split fractions"):
        extend_split(previous, old, validation_fraction=0.6, test_fraction=0.5)


def test_vocab_only_uses_training_partition_and_unknown_role_rejected():
    train = make_product("bus", 40)
    heldout = make_product("ring", 64)
    heldout = replace(heldout, elements=(replace(heldout.elements[0], role="new-role"), *heldout.elements[1:]))
    vocab = build_vocab([train])
    assert "new-role" not in vocab[0]
    with pytest.raises(ValueError, match="어휘에 없는"):
        structured_batch([heldout], 16, *vocab)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "no_lineage"])
def test_invalid_splits(products, mutation):
    split = {"schema_version": SPLIT_VERSION, "train": [p.product_id for p in products], "validation": [], "test": []}
    if mutation == "missing":
        split["train"].pop()
    elif mutation == "duplicate":
        split["test"].append(products[0].product_id)
    elif mutation == "unknown":
        split["train"].append("unknown")
    else:
        products[0] = replace(products[0], lineage_id=None)
    with pytest.raises(ValueError):
        validate_split(products, split)
