from dataclasses import replace
import pytest

from conftest import make_product
from psdl1.data import build_vocab
from psdl1.split import SPLIT_VERSION, lineage_split, validate_split
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
