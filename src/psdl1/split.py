"""Explicit product/lineage split contracts, never split variants independently."""
from __future__ import annotations

import random

SPLIT_VERSION = "dl1_split_v1"
PARTITIONS = ("train", "validation", "test")


def validate_split(products, manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SPLIT_VERSION or set(manifest) - {"schema_version", *PARTITIONS, "policy", "seed"}:
        raise ValueError("incompatible split schema")
    by_id = {p.product_id: p for p in products}
    if not by_id or len(by_id) != len(products) or any(not p.lineage_id or not p.lineage_id.strip() for p in products):
        raise ValueError("unique product IDs and explicit lineage IDs required")
    seen, owners = set(), {}
    for partition in PARTITIONS:
        ids = manifest.get(partition)
        if not isinstance(ids, list) or len(set(ids)) != len(ids):
            raise ValueError("partition IDs must be unique lists")
        for pid in ids:
            if pid not in by_id or pid in seen:
                raise ValueError("unknown or repeated product in split")
            seen.add(pid)
            lineage = by_id[pid].lineage_id
            if lineage in owners and owners[lineage] != partition:
                raise ValueError(f"lineage leakage: {lineage}")
            owners[lineage] = partition
    if seen != set(by_id) or not manifest["train"]:
        raise ValueError("split must cover all products and contain training products")
    return {key: [by_id[pid] for pid in manifest[key]] for key in PARTITIONS}


def training_only_split(products):
    manifest = {"schema_version": SPLIT_VERSION, "train": [p.product_id for p in products],
                "validation": [], "test": [], "policy": "training_only_no_heldout_claim"}
    validate_split(products, manifest)
    return manifest


def lineage_split(products, *, seed=0, validation_fraction=0.2, test_fraction=0.2):
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("invalid split fractions")
    groups = sorted({p.lineage_id for p in products})
    if not groups or any(not g for g in groups):
        raise ValueError("lineage IDs required")
    random.Random(seed).shuffle(groups)
    nv, nt = int(len(groups) * validation_fraction), int(len(groups) * test_fraction)
    assigned = {g: "validation" if i < nv else "test" if i < nv + nt else "train" for i, g in enumerate(groups)}
    manifest = {"schema_version": SPLIT_VERSION, "seed": seed, "policy": "lineage_grouped",
                **{key: sorted(p.product_id for p in products if assigned[p.lineage_id] == key) for key in PARTITIONS}}
    validate_split(products, manifest)
    return manifest
