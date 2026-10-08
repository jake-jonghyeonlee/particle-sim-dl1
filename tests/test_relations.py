import pytest
import torch

from psdl1.losses import (NEGATIVE, POSITIVE, UNKNOWN, loss_from_logits, masked_contrastive_loss,
                          relation_coverage, relation_matrix)
from psdl1.train import RelationBatchSampler, load_relations


def test_extreme_unknown_b_logit_does_not_change_a_loss_or_gradient():
    # A-C is different; B is unknown to A/C and its positive-only row is excluded.
    relation = torch.tensor([[1, -1, 0], [-1, 1, -1], [0, -1, 1]])
    logits = torch.tensor([[2., 3., 1.], [3., 2., 0.], [1., 0., 2.]], requires_grad=True)
    before, rows = loss_from_logits(logits, relation)
    extreme = logits.detach().clone()
    extreme[0, 1] = extreme[1, 0] = 1e20
    after, _ = loss_from_logits(extreme, relation)
    assert before.item() == pytest.approx(0.31326168751822286, abs=1e-7)  # ln(1 + exp(-1)), hand-computed
    assert rows == 2 and before.item() == pytest.approx(after.item(), abs=1e-8)
    before.backward()
    assert logits.grad[0, 1] == 0 and logits.grad[1, 0] == 0
    coverage = relation_coverage(relation)
    assert coverage["excluded_positive_only_rows"] == 1
    assert coverage["negative_pairs"] == 1 and coverage["unknown_pairs"] == 2


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_relation_rejected_regardless_of_order(tmp_path, reverse):
    a, b = ("B", "A") if reverse else ("A", "B")
    pair = frozenset((a, b))
    with pytest.raises(ValueError, match="conflicting"):
        relation_matrix(["A", "B"], {pair}, {pair})
    file = tmp_path / "relations.csv"
    file.write_text(f"{a},{b},similar\n{b},{a},different\n")
    with pytest.raises(ValueError, match="conflicting"):
        load_relations(file)


def test_unlabelled_negatives_require_explicit_opt_in():
    assert relation_matrix(["A", "B"]).tolist() == [[1, -1], [-1, 1]]
    assert relation_matrix(["A", "B"], unlabelled_negatives=True, lineage_ids=["family-A", "family-B"]).tolist() == [[1, 0], [0, 1]]


@pytest.mark.parametrize("temp", [0, -1, float("nan"), float("inf"), True])
def test_invalid_temperature(temp):
    with pytest.raises(ValueError, match="temperature"):
        masked_contrastive_loss(torch.eye(2), torch.eye(2), relation_matrix(["A", "B"]), temp)


@pytest.mark.parametrize("relation", [torch.ones(2, 3), torch.tensor([[1, 2], [2, 1]]),
                                      torch.tensor([[1, -1], [0, 1]]), torch.tensor([[1., float("nan")], [float("nan"), 1.]])])
def test_invalid_relation_matrix(relation):
    with pytest.raises(ValueError):
        masked_contrastive_loss(torch.eye(2), torch.eye(2), relation)


def test_nonfinite_or_mismatched_embeddings():
    with pytest.raises(ValueError, match="embeddings"):
        masked_contrastive_loss(torch.tensor([[float("nan"), 1.]]), torch.ones(1, 2), torch.ones(1, 1))
    with pytest.raises(ValueError, match="embeddings"):
        masked_contrastive_loss(torch.ones(2, 2), torch.ones(3, 2), torch.ones(2, 2))


def test_sampler_covers_every_sparse_negative_in_each_epoch():
    ids = [f"P{i}" for i in range(11)]
    different = {frozenset(("P0", "P10")), frozenset(("P2", "P8")), frozenset(("P1", "P7"))}
    sampler = RelationBatchSampler(ids, 2, set(), different)
    for seed in (0, 1, 2):
        batches = sampler.batches(torch.Generator().manual_seed(seed))
        covered = {frozenset(ids[i] for i in batch) for batch in batches}
        assert covered == different
        assert all(len(set(batch)) == len(batch) == 2 for batch in batches)


@pytest.mark.parametrize("size", [0, 1, True])
def test_sampler_rejects_unusable_batch_size(size):
    with pytest.raises(ValueError, match="batch_size"):
        RelationBatchSampler(["A", "B"], size, different={frozenset(("A", "B"))})
