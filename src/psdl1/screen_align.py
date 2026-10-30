"""Layer 마스크의 Dice 로 제품 쌍을 정렬한다(스크리닝 전용, numpy 만 쓴다).

모든 계산은 `screen_relations.product_signature` 와 같은 물리 래스터에서 한다. 이동 (dy, dx) 는 compare 와 같은 약속이다:
B[y, x] 가 A[y + dy, x + dx] 에 놓인다. 래스터마다 다음 정수를 FFT 상관으로 구하고 반올림한다(적분 영상은 쓰지 않는다. valid 가 직사각형이 아니면 틀린다).

    Va, Vb      valid 래스터            Ωδ = Va ∩ shift(Vb, δ)
    aδ = Σ A·Ωδ   bδ = Σ shift(B, δ)·Ωδ   Iδ = Σ A·shift(B, δ)·Ωδ     Dice = 2Iδ / (aδ + bδ)

층(Layer 이름, casefold) 순서는 층별 최고 Dice 내림차순이다. 첫 층의 최고 Dice 와 ε 안쪽인 후보를 남기고, 다음 층으로 계속 좁힌다
(particle-sim-compare `layer_align.choose_dice_first` 와 같은 규칙, Dice 는 정수 분수로 비교한다).
결과 상태와 플래그는 PLAN_v1.2 X3 의 표를 따른다.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

ALIGN_MIN_OVERLAP = 0.5
ALIGN_MIN_SUPPORT = 16
ALIGN_MIN_DICE = 0.3
EPSILON = Fraction(1, 100)
COVER_LIMIT = 0.9  # 이 비율을 넘게 valid 를 덮는 층은 위치 정보가 없다(screen_relations.anchor_layers 의 informative 와 같은 기준)
FLAGS = ("layer_missing", "layer_no_common", "layer_no_candidate", "alignment_unreliable", "alignment_at_search_bound", "ambiguous_alignment")
EXCLUDE_CAUSES = ("empty", "covers_valid", "insufficient_support", "no_candidate_support")


def fast_len(n):
    """n 이상인 가장 작은 2·3·5 의 곱(FFT 가 빠른 길이)."""
    n = max(int(n), 1)
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def search_axes(shape_a, shape_b, center, limit):
    """탐색할 이동 범위: center 주변 limit 칸 안이면서 두 프레임이 한 칸이라도 겹치는 이동."""
    dys = np.arange(max(-(shape_b[0] - 1), center[0] - limit[0]), min(shape_a[0] - 1, center[0] + limit[0]) + 1)
    dxs = np.arange(max(-(shape_b[1] - 1), center[1] - limit[1]), min(shape_a[1] - 1, center[1] + limit[1]) + 1)
    return dys, dxs


def window_scores(layers_a, layers_b, va, vb, dys, dxs):
    """이동 창 (dys × dxs) 의 정수 점수. 반환 (overlap, {이름: (a, b, inter)}), 모두 int64 이고 모양은 (len(dys), len(dxs))."""
    ha, wa = va.shape
    hb, wb = vb.shape
    size = (fast_len(ha + hb), fast_len(wa + wb))  # 0 이하 이동까지 wrap 없이 담는다
    rows, cols = np.ix_(dys % size[0], dxs % size[1])

    def spectrum(mask):
        return np.fft.rfft2(np.asarray(mask, dtype=np.float64), s=size)

    def correlate(sx, sy):  # 이동 d 마다 Σ_q X[q + d]·Y[q]
        return np.rint(np.fft.irfft2(sx * np.conj(sy), s=size)[rows, cols]).astype(np.int64)
    fva, fvb = spectrum(va), spectrum(vb)
    overlap = correlate(fva, fvb)
    scores = {}
    for name in layers_a:
        fa, fb = spectrum(layers_a[name] & va), spectrum(layers_b[name] & vb)
        scores[name] = (correlate(fa, fvb), correlate(fva, fb), correlate(fa, fb))
    return overlap, scores


def inner_boundary(mask, omega):
    """∂A = A ∧ ¬erode₁(A) (3×3 침식). 침식할 때 Ω 밖과 배열 밖은 전부 전경으로 본다: ROI 가장자리·valid 구멍 둘레에 인공 경계가 생기지 않는다.
    mask 는 omega 안의 bool 배열이다."""
    mask, omega = np.asarray(mask, dtype=bool), np.asarray(omega, dtype=bool)
    height, width = mask.shape
    foreground = np.pad(mask | ~omega, 1, constant_values=True)
    eroded = np.ones(mask.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= foreground[dy:dy + height, dx:dx + width]
    return mask & ~eroded


def original_window(shape_a, shape_b, dy, dx):
    """원본 해상도 이동 (dy, dx) 에서 B[y, x] 가 A[y + dy, x + dx] 에 놓일 때의 겹침. (A 슬라이스, B 슬라이스) 이고 겹침이 없으면 None."""
    y0, y1 = max(0, dy), min(shape_a[0], shape_b[0] + dy)
    x0, x1 = max(0, dx), min(shape_a[1], shape_b[1] + dx)
    if y1 <= y0 or x1 <= x0:
        return None
    return (slice(y0, y1), slice(x0, x1)), (slice(y0 - dy, y1 - dy), slice(x0 - dx, x1 - dx))


def pack_mask(mask):
    """bool 마스크를 비트로 묶어 보관한다(원본 해상도 마스크를 제품마다 여럿 들고 있어도 메모리가 작다)."""
    return np.packbits(np.asarray(mask, dtype=bool).ravel())


def unpack_mask(packed, shape):
    return np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)


def block_sum(mask, factor):
    """factor×factor 칸마다 참인 픽셀 수(래스터 크기는 block_reduce 와 같다)."""
    height, width = mask.shape
    padded = np.pad(mask, ((0, -height % factor), (0, -width % factor)))
    return padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor).sum(axis=(1, 3), dtype=np.int64)


def exclusion_cause(mask, valid, min_support):
    """층이 정렬 키로 쓸모 있는지. 쓸모 없으면 원인, 있으면 None."""
    count = int(np.count_nonzero(mask & valid))
    if count == 0:
        return "empty"
    if count > COVER_LIMIT * max(int(valid.sum()), 1):
        return "covers_valid"
    if count < min_support:
        return "insufficient_support"
    return None


def _best(num, den, indices):
    """flat 색인 목록 중 가장 큰 분수 num/den. 반환 (분자, 분모). 서로 다른 분수는 1e-11 이상 벌어져 float 나눗셈 순서가 정확하다."""
    pick = indices[int(np.argmax(num[indices] / den[indices]))]
    return int(num[pick]), int(den[pick])


def _within(num, den, best, epsilon):
    """num/den + ε >= best 를 정수로 비교한다(den, best 분모는 양수)."""
    bn, bd = best
    p, q = epsilon.numerator, epsilon.denominator
    return q * num * bd + p * den * bd >= q * bn * den


def align_layers(layers_a, layers_b, va, vb, *, center, limit, min_overlap=ALIGN_MIN_OVERLAP, min_support=ALIGN_MIN_SUPPORT,
                 min_dice=ALIGN_MIN_DICE, epsilon=EPSILON):
    """Layer 래스터 {casefold 이름: bool} 두 개로 B 를 A 에 정렬한다. layers_a/b 가 None 이면 그 제품은 사이드카 항목이 없다.

    반환 dict: flags(FLAGS 순서), applied(Layer 이동을 쓰는가), shift (dy, dx) 또는 None, dice(결정 층의 선택 이동 Dice), second_dice,
    decided(결정 층 이름), disagreeing(선택 이동과 최고점이 다른 층 수), order(층 순서), excluded({이름: 원인})."""
    out = {"flags": [], "applied": False, "shift": None, "dice": None, "second_dice": None, "decided": None, "disagreeing": None,
           "order": [], "excluded": {}}
    if layers_a is None or layers_b is None:
        out["flags"] = ["layer_missing"]
        return out
    va, vb = np.asarray(va, dtype=bool), np.asarray(vb, dtype=bool)
    usable = {}
    for name in sorted(set(layers_a) & set(layers_b)):
        a, b = np.asarray(layers_a[name], dtype=bool), np.asarray(layers_b[name], dtype=bool)
        if a.shape != va.shape or b.shape != vb.shape:
            raise ValueError(f"layer {name!r} shape differs from the valid raster")
        causes = {c for c in (exclusion_cause(a, va, min_support), exclusion_cause(b, vb, min_support)) if c is not None}
        if causes:
            out["excluded"][name] = next(c for c in EXCLUDE_CAUSES if c in causes)
        else:
            usable[name] = (a, b)
    if not usable:
        out["flags"] = ["layer_no_common"]
        return out
    dys, dxs = search_axes(va.shape, vb.shape, center, limit)
    if dys.size == 0 or dxs.size == 0:
        out["flags"] = ["layer_no_candidate"]
        return out
    overlap, scores = window_scores({n: m[0] for n, m in usable.items()}, {n: m[1] for n, m in usable.items()}, va, vb, dys, dxs)
    candidate = (overlap > 0) & (overlap >= min_overlap * min(int(va.sum()), int(vb.sum())))
    ry, rx = np.meshgrid(dys - center[0], dxs - center[1], indexing="ij")
    ry, rx = ry.ravel(), rx.ravel()
    tie = np.lexsort((rx, ry, ry * ry + rx * rx))  # 시작점에 가까운 순, 같으면 (dy, dx) 작은 순(compare 의 tie-key)
    rank = np.empty(tie.size, dtype=np.int64)
    rank[tie] = np.arange(tie.size)
    tables = {}
    for name, (a, b, inter) in scores.items():
        ok = (candidate & (a >= min_support) & (b >= min_support) & (a + b > 0)).ravel()
        if not ok.any():
            out["excluded"][name] = "no_candidate_support"
            continue
        num, den = 2 * inter.ravel(), (a + b).ravel()
        indices = np.flatnonzero(ok)
        best = _best(num, den, indices)
        top = indices[num[indices] * best[1] == best[0] * den[indices]]
        tables[name] = {"ok": ok, "num": num, "den": den, "best": Fraction(*best), "best_pair": best, "argmax": int(top[np.argmin(rank[top])])}
    if not tables:
        out["flags"] = ["layer_no_candidate"]
        return out
    order = sorted(tables, key=lambda n: (-tables[n]["best"], n))
    out["order"] = order
    first = tables[order[0]]
    keep = first["ok"] & _within(first["num"], first["den"], first["best_pair"], epsilon)
    survivors = np.flatnonzero(keep)
    decided = order[0]
    for name in order[1:]:
        if survivors.size == 1:
            break
        t = tables[name]
        decided = name
        usable_here = survivors[t["ok"][survivors]]
        if usable_here.size == 0:
            continue  # 이 층은 남은 후보를 구별하지 못한다
        best = _best(t["num"], t["den"], usable_here)
        survivors = survivors[t["ok"][survivors] & _within(t["num"][survivors], t["den"][survivors], best, epsilon)]
    flags = []
    if survivors.size == 1:
        chosen = int(survivors[0])
    else:
        diameter = max(int(np.ptp(ry[survivors])), int(np.ptp(rx[survivors])))  # 체비쇼프 지름
        if diameter >= 2:
            flags.append("ambiguous_alignment")
            chosen = int(survivors[np.argmin(rank[survivors])])
        else:
            def key(index):
                dice = tuple(-(Fraction(int(tables[n]["num"][index]), int(tables[n]["den"][index])) if tables[n]["ok"][index] else Fraction(-1)) for n in order)
                return dice + (int(rank[index]),)
            chosen = min((int(i) for i in survivors), key=key)
    for name in reversed(order[:order.index(decided) + 1]):  # 선택 이동에서 점수가 있는 가장 나중 층을 결정 층으로 본다
        if tables[name]["ok"][chosen]:
            decided = name
            break
    t = tables[decided]
    dice = Fraction(int(t["num"][chosen]), int(t["den"][chosen]))
    if float(dice) < min_dice:
        flags.append("alignment_unreliable")
    if abs(int(ry[chosen])) >= limit[0] - 1 or abs(int(rx[chosen])) >= limit[1] - 1:
        flags.append("alignment_at_search_bound")
    far = t["ok"] & ((np.abs(ry - ry[chosen]) >= 2) | (np.abs(rx - rx[chosen]) >= 2))  # 선택 이동과 체비쇼프 거리 2 이상인 후보
    second = None
    if far.any():
        indices = np.flatnonzero(far)
        second = round(float(Fraction(*_best(t["num"], t["den"], indices))), 6)
    out.update({"flags": [f for f in FLAGS if f in flags], "shift": (int(center[0] + ry[chosen]), int(center[1] + rx[chosen])),
                "dice": round(float(dice), 6), "second_dice": second, "decided": decided,
                "disagreeing": sum(tables[n]["argmax"] != chosen for n in order)})
    out["applied"] = not ({"alignment_unreliable", "alignment_at_search_bound"} & set(flags))
    return out
