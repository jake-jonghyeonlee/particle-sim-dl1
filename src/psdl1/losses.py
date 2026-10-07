"""대조 학습 손실. 관계를 모르는 쌍은 계산에서 뺀다."""
from __future__ import annotations

import torch

POSITIVE, NEGATIVE, UNKNOWN = 1, 0, -1


def relation_matrix(product_ids: list[str], similar: set[frozenset] | None = None, different: set[frozenset] | None = None) -> torch.Tensor:
    """제품 쌍의 관계 [B,B]. 자기 자신은 가깝게(1).

    정답이 없으면(similar, different 모두 None) 다른 제품은 모두 구분(0)으로 둔다.
    정답이 있으면 적힌 쌍만 쓰고, 적히지 않은 쌍은 모름(-1)으로 둔다.
    """
    b = len(product_ids)
    labelled = similar is not None or different is not None
    rel = torch.full((b, b), UNKNOWN if labelled else NEGATIVE)
    for i in range(b):
        for j in range(b):
            key = frozenset((product_ids[i], product_ids[j]))
            if product_ids[i] == product_ids[j] or (similar and key in similar):
                rel[i, j] = POSITIVE
            elif different and key in different:
                rel[i, j] = NEGATIVE
    return rel


def masked_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, relation: torch.Tensor, temperature: float = 0.1) -> tuple[torch.Tensor, int]:
    """z1[i]와 z2[j]의 관계가 relation[i,j]. 가까워야 하는 쌍을 그 행의 알려진 쌍들 가운데서 고르게 한다.

    모름(-1)은 분자에도 분모에도 넣지 않는다. 모르는 쌍을 다른 쌍으로 밀어내지 않기 위해서다.
    구분할 쌍이 하나도 없는 행은 뺀다. 그런 행만 있으면 모든 특징을 같게 만드는 것이 답이 되기 때문이다.
    (손실, 계산에 쓴 행 수)를 돌려준다.
    """
    logits = z1 @ z2.T / temperature
    known = relation != UNKNOWN
    positive = relation == POSITIVE
    if not positive.any(dim=1).all():
        raise ValueError("행마다 가까워야 하는 쌍이 하나는 있어야 한다")
    usable = (relation == NEGATIVE).any(dim=1)
    if not usable.any():
        return logits.sum() * 0.0, 0
    logits = logits.masked_fill(~known, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_row = -(log_prob.masked_fill(~positive, 0.0)).sum(dim=1) / positive.sum(dim=1)
    return per_row[usable].mean(), int(usable.sum())
