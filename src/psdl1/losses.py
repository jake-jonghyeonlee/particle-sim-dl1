"""Label-aware contrastive loss; unlabelled negatives require explicit opt-in."""
from __future__ import annotations

import math
import torch

POSITIVE, NEGATIVE, UNKNOWN = 1, 0, -1


def validate_relations(similar=None, different=None, *, product_ids=None):
    similar, different = set(similar or ()), set(different or ())
    for pair in similar | different:
        if not isinstance(pair, frozenset) or len(pair) != 2 or any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in pair):
            raise ValueError("relations require two distinct trimmed product IDs")
        if product_ids is not None and not pair <= set(product_ids):
            raise ValueError(f"relation references unknown product: {sorted(pair)}")
    if similar & different:
        raise ValueError(f"conflicting similar/different relation: {sorted(map(sorted, similar & different))}")
    return similar, different


def relation_matrix(product_ids, similar=None, different=None, *, unlabelled_negatives=False):
    if not product_ids or len(set(product_ids)) != len(product_ids):
        raise ValueError("batch needs unique product IDs")
    similar, different = validate_relations(similar, different)
    rel = torch.full((len(product_ids), len(product_ids)), NEGATIVE if unlabelled_negatives else UNKNOWN, dtype=torch.long)
    for i, a in enumerate(product_ids):
        for j, b in enumerate(product_ids):
            key = frozenset((a, b))
            if i == j or key in similar:
                rel[i, j] = POSITIVE
            elif key in different:
                rel[i, j] = NEGATIVE
    return rel


def validate_matrix(relation, size):
    if relation.shape != (size, size) or not torch.isfinite(relation).all() or not torch.isin(relation, torch.tensor([UNKNOWN, NEGATIVE, POSITIVE], device=relation.device)).all():
        raise ValueError("relation matrix requires square batch shape and values -1,0,1")
    if not torch.equal(relation, relation.T):
        raise ValueError("relation matrix must be symmetric")
    if not (relation == POSITIVE).any(dim=1).all():
        raise ValueError("행마다 가까워야 하는 쌍이 하나는 있어야 한다")


def relation_coverage(relation):
    validate_matrix(relation, len(relation))
    upper = torch.triu(torch.ones_like(relation, dtype=torch.bool), diagonal=1)
    usable = (relation == NEGATIVE).any(dim=1)
    return {"positive_pairs": int(((relation == POSITIVE) & upper).sum()),
            "negative_pairs": int(((relation == NEGATIVE) & upper).sum()),
            "unknown_pairs": int(((relation == UNKNOWN) & upper).sum()),
            "usable_rows": int(usable.sum()), "excluded_positive_only_rows": int((~usable).sum()),
            "effective_pairs": int((((relation != UNKNOWN) & usable[:, None]) & ~torch.eye(len(relation), dtype=torch.bool, device=relation.device)).sum())}


def loss_from_logits(logits, relation):
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or not logits.numel() or not torch.isfinite(logits).all():
        raise ValueError("finite nonempty square logits required")
    validate_matrix(relation, logits.shape[0])
    relation = relation.to(logits.device)
    usable = (relation == NEGATIVE).any(dim=1)
    if not usable.any():
        return logits[0, 0] * 0, 0
    positive = relation == POSITIVE
    masked = logits.masked_fill(relation == UNKNOWN, float("-inf"))
    log_prob = masked - torch.logsumexp(masked, dim=1, keepdim=True)
    per_row = -log_prob.masked_fill(~positive, 0).sum(dim=1) / positive.sum(dim=1)
    return per_row[usable].mean(), int(usable.sum())


def masked_contrastive_loss(z1, z2, relation, temperature=0.1):
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if z1.ndim != 2 or z1.shape != z2.shape or not z1.numel() or not torch.isfinite(z1).all() or not torch.isfinite(z2).all():
        raise ValueError("embeddings must have equal nonempty [B,D] shape and finite values")
    return loss_from_logits(z1 @ z2.T / temperature, relation)
