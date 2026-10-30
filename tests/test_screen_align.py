"""screen-metrics/T4: Layer Dice 정렬(screen_align). numpy 만 쓴다 — torch·conftest 없이 Python 3.9 에서도 돈다."""
from __future__ import annotations

import numpy as np
import pytest

from psdl1.screen_align import align_layers, fast_len, search_axes, window_scores

LIMIT = (17, 17)


def blobs(shape, seed, count=6, low=3, high=9):
    """겹침이 조금 있는 임의 직사각형 합. 주기가 없다."""
    rng = np.random.RandomState(seed)
    mask = np.zeros(shape, dtype=bool)
    for _ in range(count):
        h, w = rng.randint(low, high, size=2)
        y, x = rng.randint(0, shape[0] - h), rng.randint(0, shape[1] - w)
        mask[y:y + h, x:x + w] = True
    return mask


def put(shape, patch, origin):
    out = np.zeros(shape, dtype=bool)
    out[origin[0]:origin[0] + patch.shape[0], origin[1]:origin[1] + patch.shape[1]] = patch
    return out


def pair_with_shift(names=("p", "q", "r"), shape_a=(64, 64), shape_b=(64, 64), origin_a=(20, 18), origin_b=(8, 10), size=(30, 30)):
    """B 의 층이 A 의 같은 층을 (origin_a - origin_b) 만큼 옮긴 복제다: B[y] 가 A[y + (origin_a - origin_b)] 에 놓인다."""
    layers_a, layers_b = {}, {}
    for k, name in enumerate(names):
        patch = blobs(size, 100 + k)
        layers_a[name], layers_b[name] = put(shape_a, patch, origin_a), put(shape_b, patch, origin_b)
    return layers_a, layers_b, np.ones(shape_a, bool), np.ones(shape_b, bool)


def run(layers_a, layers_b, va, vb, **kw):
    kw.setdefault("center", ((va.shape[0] - vb.shape[0]) // 2, (va.shape[1] - vb.shape[1]) // 2))
    kw.setdefault("limit", LIMIT)
    return align_layers(layers_a, layers_b, va, vb, **kw)


# ---- 정확한 이동 복원 -----------------------------------------------------------------------------------------------
def test_parallel_copy_recovers_the_exact_shift_for_every_layer():
    la, lb, va, vb = pair_with_shift()
    out = run(la, lb, va, vb)
    assert out["shift"] == (12, 8) and out["applied"] and out["flags"] == [] and out["dice"] == 1.0
    assert out["disagreeing"] == 0 and out["excluded"] == {} and sorted(out["order"]) == ["p", "q", "r"]
    assert out["second_dice"] is not None and 0 <= out["second_dice"] < 1


def test_parallel_copy_with_different_frame_sizes_and_a_non_zero_start():
    la, lb, va, vb = pair_with_shift(shape_a=(70, 80), shape_b=(50, 44), origin_a=(30, 40), origin_b=(10, 4), size=(25, 30))
    out = run(la, lb, va, vb, limit=(24, 24))  # the default start is (10, 18), so the answer is 18 columns away
    assert out["shift"] == (20, 36) and out["applied"] and out["dice"] == 1.0
    assert run(la, lb, va, vb, center=(20, 36))["shift"] == (20, 36)  # the start itself may be the answer


def test_the_layer_that_cannot_decide_hands_over_to_the_next_one():
    shape = (64, 64)
    stripes = np.zeros(shape, bool)
    for x in range(2, 64, 8):
        stripes[:, x:x + 3] = True
    patch = blobs((18, 18), 7)
    la = {"a_stripes": stripes, "b_blob": put(shape, patch, (20, 28))}
    lb = {"a_stripes": stripes.copy(), "b_blob": put(shape, patch, (20, 20))}
    out = run(la, lb, np.ones(shape, bool), np.ones(shape, bool))
    assert out["shift"] == (0, 8) and out["decided"] == "b_blob" and out["flags"] == [] and out["applied"]
    assert out["order"][0] == "a_stripes" and out["disagreeing"] == 1  # the stripes alone would have chosen (0, 0)


# ---- 주기 패턴은 모호하지만 적용한다 ----------------------------------------------------------------------------------
def test_periodic_pattern_is_ambiguous_yet_applied_at_the_tie_key_choice():
    shape = (64, 64)
    stripes = np.zeros(shape, bool)
    for x in range(2, 64, 8):
        stripes[:, x:x + 3] = True
    v = np.ones(shape, bool)
    out = run({"s": stripes}, {"s": stripes.copy()}, v, v)
    assert out["flags"] == ["ambiguous_alignment"] and out["applied"] and out["shift"] == (0, 0) and out["dice"] == 1.0
    near = run({"s": stripes}, {"s": stripes.copy()}, v, v, center=(0, 8))
    assert near["shift"] == (0, 8) and near["flags"] == ["ambiguous_alignment"]  # nearest to the start wins


# ---- 상태표 --------------------------------------------------------------------------------------------------------------
def test_layer_missing_when_either_product_has_no_sidecar_entry():
    la, lb, va, vb = pair_with_shift()
    for a, b in ((None, lb), (la, None), (None, None)):
        out = run(a, b, va, vb)
        assert out["flags"] == ["layer_missing"] and not out["applied"] and out["shift"] is None and out["dice"] is None and out["decided"] is None


def test_layer_no_common_without_a_shared_name_or_without_an_informative_layer():
    la, lb, va, vb = pair_with_shift()
    out = run({"x": la["p"]}, {"y": lb["p"]}, va, vb)
    assert out["flags"] == ["layer_no_common"] and not out["applied"] and out["shift"] is None and out["excluded"] == {}
    empty = run({"p": np.zeros_like(la["p"])}, {"p": lb["p"]}, va, vb)
    assert empty["flags"] == ["layer_no_common"] and empty["excluded"] == {"p": "empty"}
    few = np.zeros_like(la["p"])
    few[0, :10] = True
    assert run({"p": few}, {"p": few}, va, vb)["excluded"] == {"p": "insufficient_support"}
    assert run({"p": few}, {"p": few}, va, vb, min_support=10)["flags"] != ["layer_no_common"]


def test_uninformative_layers_are_left_out_and_the_cause_is_recorded():
    la, lb, va, vb = pair_with_shift(names=("ok",))
    full = np.ones(va.shape, bool)
    nearly = np.ones(va.shape, bool)
    nearly[:3, :] = False                       # 95% of the valid area
    la.update({"empty": np.zeros(va.shape, bool), "full": full, "nearly": nearly})
    lb.update({"empty": lb["ok"].copy(), "full": full.copy(), "nearly": nearly.copy()})
    out = run(la, lb, va, vb)
    assert out["excluded"] == {"empty": "empty", "full": "covers_valid", "nearly": "covers_valid"}
    assert out["order"] == ["ok"] and out["shift"] == (12, 8) and out["applied"]
    inside_valid = np.zeros(va.shape, bool)    # covered pixels that lie outside valid do not count
    inside_valid[:50, :] = True
    holed = va.copy()
    holed[50:, :] = False
    out = run({"w": inside_valid}, {"w": inside_valid.copy()}, holed, holed)
    assert out["excluded"] == {"w": "covers_valid"}


def test_no_candidate_when_the_support_inside_every_overlap_is_too_small():
    shape_a, shape_b = (64, 64), (64, 32)
    a = np.zeros(shape_a, bool)
    a[10:30, 56:64] = True                     # plenty of pixels in A, none where B can land
    b = blobs(shape_b, 3, count=8)
    va, vb = np.ones(shape_a, bool), np.ones(shape_b, bool)
    out = run({"p": a}, {"p": b}, va, vb, center=(0, 16), limit=(2, 2))
    assert out["flags"] == ["layer_no_candidate"] and not out["applied"] and out["shift"] is None and out["excluded"] == {"p": "no_candidate_support"}
    assert run({"p": a}, {"p": b}, va, vb, center=(0, 32), limit=(2, 2))["flags"] != ["layer_no_candidate"]  # B reaches the pixels


def test_no_candidate_when_the_valid_areas_never_overlap_enough():
    shape = (64, 64)
    va, vb = np.zeros(shape, bool), np.zeros(shape, bool)
    va[:, :10], vb[:, 54:] = True, True
    a, b = np.zeros(shape, bool), np.zeros(shape, bool)
    a[:, :10], b[:, 54:] = blobs((64, 10), 1, count=10), blobs((64, 10), 2, count=10)
    out = run({"p": a}, {"p": b}, va, vb, center=(0, 0), limit=(2, 2))
    assert out["flags"] == ["layer_no_candidate"] and not out["applied"] and out["shift"] is None
    wide = np.zeros(shape, bool)               # a shift exists, but only 3 of B's 10 valid columns land on A's valid area
    wide[:, :20] = True
    a_wide = np.zeros(shape, bool)
    a_wide[:, :20] = blobs((64, 20), 1, count=20)
    out = run({"p": a_wide}, {"p": b}, wide, vb, center=(0, -37), limit=(0, 0))
    assert out["flags"] == ["layer_no_candidate"] and not out["applied"]


def test_unreliable_when_the_deciding_dice_is_below_the_minimum():
    la, lb, va, vb = pair_with_shift(names=("p",))
    rng = np.random.RandomState(0)
    lb["p"] = lb["p"] & (rng.rand(*lb["p"].shape) > 0.4)   # the copy lost about 40% of its pixels
    out = run(la, lb, va, vb, min_dice=0.9)
    assert out["flags"] == ["alignment_unreliable"] and not out["applied"] and out["shift"] == (12, 8) and 0.5 < out["dice"] < 0.9
    assert run(la, lb, va, vb, min_dice=0.5)["applied"]


def test_search_bound_hit_is_not_applied_but_recorded():
    la, lb, va, vb = pair_with_shift()
    out = run(la, lb, va, vb, limit=(13, 13))
    assert out["flags"] == ["alignment_at_search_bound"] and not out["applied"] and out["shift"] == (12, 8) and out["dice"] == 1.0
    assert run(la, lb, va, vb, limit=(14, 14))["applied"]


def test_several_problems_are_all_reported_and_one_is_enough_to_block():
    shape = (64, 64)
    stripes = np.zeros(shape, bool)
    for x in range(2, 64, 8):
        stripes[:, x:x + 3] = True
    v = np.ones(shape, bool)
    out = run({"s": stripes}, {"s": stripes.copy()}, v, v, limit=(1, 1), min_dice=1.01)
    assert out["flags"] == ["alignment_unreliable", "alignment_at_search_bound", "ambiguous_alignment"] and not out["applied"]
    assert out["shift"] == (0, 0) and out["dice"] == 1.0  # the values stay recorded


# ---- FFT 점수 == 직접 합 ---------------------------------------------------------------------------------------------------
def reduce_any(mask, factor):
    h, w = mask.shape
    mask = np.pad(mask, ((0, -h % factor), (0, -w % factor)))
    return mask.reshape(mask.shape[0] // factor, factor, mask.shape[1] // factor, factor).any(axis=(1, 3))


def direct(a, b, va, vb, dy, dx):
    """정의대로 한 이동의 (|Ω|, a, b, I) 를 센다. B[y, x] 가 A[y + dy, x + dx] 에 놓인다."""
    omega = sa = sb = inter = 0
    for y in range(b.shape[0]):
        for x in range(b.shape[1]):
            ya, xa = y + dy, x + dx
            if 0 <= ya < a.shape[0] and 0 <= xa < a.shape[1] and va[ya, xa] and vb[y, x]:
                omega += 1
                sa += int(a[ya, xa])
                sb += int(b[y, x])
                inter += int(a[ya, xa] and b[y, x])
    return omega, sa, sb, inter


def check_against_direct(a, b, va, vb, center=(0, 0), limit=(40, 40)):
    dys, dxs = search_axes(va.shape, vb.shape, center, limit)
    overlap, scores = window_scores({"p": a}, {"p": b}, va, vb, dys, dxs)
    sa, sb, inter = scores["p"]
    assert overlap.dtype == np.int64 and overlap.shape == (len(dys), len(dxs))
    for i, dy in enumerate(dys):
        for j, dx in enumerate(dxs):
            assert (overlap[i, j], sa[i, j], sb[i, j], inter[i, j]) == direct(a, b, va, vb, int(dy), int(dx)), (dy, dx)
    return len(dys) * len(dxs)


@pytest.mark.parametrize("seed", range(4))
def test_fft_scores_equal_the_direct_sum_for_random_inputs(seed):
    rng = np.random.RandomState(seed)
    a, b = rng.rand(9, 11) > 0.5, rng.rand(7, 13) > 0.4
    assert check_against_direct(a, b, np.ones(a.shape, bool), np.ones(b.shape, bool)) == (9 + 7 - 1) * (11 + 13 - 1)


def test_fft_scores_equal_the_direct_sum_with_valid_holes_and_empty_valid():
    rng = np.random.RandomState(5)
    a, b = rng.rand(12, 10) > 0.4, rng.rand(8, 14) > 0.4
    va, vb = rng.rand(*a.shape) > 0.3, rng.rand(*b.shape) > 0.3   # holes everywhere: not rectangles
    va[2:6, 3:8] = False
    check_against_direct(a, b, va, vb)
    check_against_direct(a, b, np.zeros(a.shape, bool), vb)
    check_against_direct(a, b, va, np.zeros(b.shape, bool))
    check_against_direct(a, b, np.zeros(a.shape, bool), np.zeros(b.shape, bool))


def test_fft_scores_equal_the_direct_sum_for_frames_that_the_factor_does_not_divide():
    rng = np.random.RandomState(9)
    for factor, shape_a, shape_b in ((4, (37, 29), (22, 31)), (3, (20, 17), (26, 11)), (8, (50, 41), (50, 41))):
        a, b = rng.rand(*shape_a) > 0.5, rng.rand(*shape_b) > 0.5
        va, vb = rng.rand(*shape_a) > 0.1, rng.rand(*shape_b) > 0.1
        ra, rb, rva, rvb = (reduce_any(m, factor) for m in (a, b, va, vb))
        assert ra.shape == (-(-shape_a[0] // factor), -(-shape_a[1] // factor))
        check_against_direct(ra & rva, rb & rvb, rva, rvb, center=(1, -1), limit=(6, 5))


def test_the_dice_uses_the_overlap_region_not_the_whole_mask():
    # B covers only the left half of A's valid area. Over that half the masks are identical, so the Dice there is 1 even though A is twice as large.
    a = np.zeros((20, 20), bool)
    a[:, :10] = blobs((20, 10), 4, count=10)
    a[:, 10:] = blobs((20, 10), 5, count=10)
    b = a[:, :10].copy()
    va, vb = np.ones(a.shape, bool), np.ones(b.shape, bool)
    out = run({"p": a}, {"p": b}, va, vb, center=(0, 0), limit=(3, 3), min_support=16)
    assert out["shift"] == (0, 0) and out["dice"] == 1.0


def test_fast_len_is_a_five_smooth_number_not_below_the_request():
    for n in (1, 7, 11, 49, 97, 1001):
        m = fast_len(n)
        assert m >= n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        assert m == 1


# ---- T5 지표 헬퍼(numpy 전용) -------------------------------------------------------------------------------------------
def test_inner_boundary_treats_everything_outside_omega_as_foreground():
    from psdl1.screen_align import inner_boundary
    omega = np.ones((12, 12), bool)
    omega[3:9, 3:9] = False  # valid 구멍
    assert not inner_boundary(omega, omega).any()  # Ω 를 꽉 채운 마스크: 배열 가장자리·구멍 둘레에 경계가 없다
    square = np.zeros((12, 12), bool)
    square[2:10, 2:10] = True
    assert inner_boundary(square, np.ones((12, 12), bool)).sum() == 28 and not inner_boundary(square, np.ones((12, 12), bool))[3:9, 3:9].any()
    touching = np.zeros((12, 12), bool)
    touching[:, :6] = True  # 배열 가장자리에 닿은 쪽에는 경계가 없다
    assert set(np.flatnonzero(inner_boundary(touching, np.ones((12, 12), bool)).any(axis=0))) == {5}


def test_original_window_pack_and_block_sum():
    from psdl1.screen_align import block_sum, original_window, pack_mask, unpack_mask
    assert original_window((10, 10), (6, 8), 3, -2) == ((slice(3, 9), slice(0, 6)), (slice(0, 6), slice(2, 8)))
    assert original_window((10, 10), (6, 8), 10, 0) is None
    mask = blobs((13, 17), 5)
    assert np.array_equal(unpack_mask(pack_mask(mask), mask.shape), mask)
    sums = block_sum(mask, 4)
    assert sums.shape == (4, 5) and sums.sum() == mask.sum() and sums[0, 0] == mask[:4, :4].sum() and sums[3, 4] == mask[12:, 16:].sum()


def test_results_contain_only_finite_plain_numbers():
    la, lb, va, vb = pair_with_shift()
    out = run(la, lb, va, vb)
    for key in ("dice", "second_dice"):
        assert isinstance(out[key], float) and np.isfinite(out[key])
    assert all(type(v) is int for v in out["shift"]) and type(out["disagreeing"]) is int
