"""Behavior tests for tac.score_aware_weight_requant (Task #69).

NO-FAKE discipline: every test verifies BEHAVIOR (the re-quant actually changes
q-bytes, lowers entropy, crushes harder at fewer levels; the byte-maps round-trip
exactly; the allocator ranks by sensitivity and crushes low-sensitivity tensors
harder; replacing the ranking with random FAILS the entropy advantage). None of
these pass if the functions return canonical constants.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.score_aware_weight_requant import (
    TensorRequantPlan,
    allocate_bits_by_sensitivity,
    byte_map_roundtrip_is_identity,
    contest_score_from_components,
    decode_byte_map_u8,
    effective_levels_for_bits,
    encode_byte_map_u8,
    q_byte_entropy_bits,
    requant_signed_q,
    score_delta_components,
)


# ---------------------------------------------------------------------------
# Byte-map exact inverse (grammar preservation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("byte_map", ["zig", "twos", "off"])
def test_byte_map_roundtrip_is_exact_identity(byte_map):
    # encode∘decode must be the identity over the full uint8 range, or re-packing
    # a re-quantized tensor into the grammar would corrupt unrelated bytes.
    u8 = np.arange(256, dtype=np.uint8)
    signed = decode_byte_map_u8(u8, byte_map)
    back = encode_byte_map_u8(signed, byte_map)
    assert np.array_equal(back, u8)
    assert byte_map_roundtrip_is_identity(byte_map)


def test_negzig_roundtrip_bijective_except_int8_overflow_sentinel():
    # negzig inherits the upstream decode_mapped_u8 quirk: u8=255 decodes to the
    # int8 value -128 (zigzag of 255 is -128, negated to +128 which wraps to -128
    # under int8). The frontier encoder never emits 255 for negzig tensors, so the
    # re-quant only needs the decode->encode->decode FIXED-POINT to be stable, which
    # it is for every q-value in [-127, 127] (the legal stored range).
    u8 = np.arange(255, dtype=np.uint8)  # exclude the 255 overflow sentinel
    signed = decode_byte_map_u8(u8, "negzig")
    back = encode_byte_map_u8(signed, "negzig")
    assert np.array_equal(back, u8)
    # u8=255 is the lone non-round-trippable sentinel (decodes to int8 -128, which
    # is outside the legal stored q-range [-127,127]); the frontier encoder never
    # emits it for negzig tensors. Document the quirk so re-quant never produces it:
    # requant_signed_q clips to [-127,127], so the re-quantized signed values are
    # always legal and re-encode cleanly under negzig.
    legal_signed = np.arange(-127, 128, dtype=np.int8)
    legal_u8 = encode_byte_map_u8(legal_signed, "negzig")
    assert np.array_equal(decode_byte_map_u8(legal_u8, "negzig"), legal_signed)


def test_byte_map_decode_matches_upstream_codec_semantics():
    # Spot-check against the documented PR #101 semantics.
    # zig: 0->0, 1->-1, 2->1, 3->-2
    assert list(decode_byte_map_u8(np.array([0, 1, 2, 3], np.uint8), "zig")) == [0, -1, 1, -2]
    # off: subtract 128
    assert list(decode_byte_map_u8(np.array([128, 129, 127], np.uint8), "off")) == [0, 1, -1]
    # twos: reinterpret
    assert list(decode_byte_map_u8(np.array([255, 1], np.uint8), "twos")) == [-1, 1]


def test_unknown_byte_map_raises():
    with pytest.raises(ValueError):
        decode_byte_map_u8(np.array([0], np.uint8), "bogus")
    with pytest.raises(ValueError):
        encode_byte_map_u8(np.array([0], np.int8), "bogus")


# ---------------------------------------------------------------------------
# effective_levels_for_bits
# ---------------------------------------------------------------------------


def test_effective_levels_for_bits_ladder():
    assert effective_levels_for_bits(8) == 256
    assert effective_levels_for_bits(4) == 16
    assert effective_levels_for_bits(3) == 8
    assert effective_levels_for_bits(2) == 4
    assert effective_levels_for_bits(1) == 2
    assert effective_levels_for_bits(9) == 256  # clamps to full int8
    with pytest.raises(ValueError):
        effective_levels_for_bits(0)


# ---------------------------------------------------------------------------
# requant_signed_q — the load-bearing transform
# ---------------------------------------------------------------------------


def test_requant_levels_256_is_identity():
    rng = np.random.default_rng(0)
    q = rng.integers(-127, 128, size=5000).astype(np.int8)
    out = requant_signed_q(q, 256)
    assert np.array_equal(out, q)


def test_requant_actually_changes_bytes_class1_no_op_guard():
    # The #1 NO-FAKE class: the re-quant must ACTUALLY change the q-values.
    rng = np.random.default_rng(1)
    q = rng.integers(-127, 128, size=20000).astype(np.int8)
    out4 = requant_signed_q(q, 16)
    # A meaningful fraction of values must change (int8 -> int4 is a real crush).
    changed = int(np.count_nonzero(out4 != q))
    assert changed > q.size // 4, f"only {changed}/{q.size} changed — re-quant is a near-no-op"


def test_requant_fewer_levels_collapse_support_set():
    rng = np.random.default_rng(2)
    q = rng.integers(-127, 128, size=50000).astype(np.int8)
    uniq8 = len(np.unique(q))
    uniq4 = len(np.unique(requant_signed_q(q, 16)))
    uniq3 = len(np.unique(requant_signed_q(q, 8)))
    uniq2 = len(np.unique(requant_signed_q(q, 4)))
    # Fewer levels => strictly smaller support set => monotone collapse.
    assert uniq2 <= uniq3 <= uniq4 < uniq8
    assert uniq4 <= 16
    assert uniq3 <= 8
    assert uniq2 <= 4


def test_requant_lowers_entropy_monotonically_after_bytemap():
    # The win mechanism: fewer levels -> lower q-byte ENTROPY -> fewer ctx bytes.
    rng = np.random.default_rng(3)
    # Use a realistic bell-ish distribution (decoder q-values cluster near 0).
    q = np.clip(rng.normal(0, 30, size=80000), -127, 127).astype(np.int8)
    H_full = q_byte_entropy_bits(encode_byte_map_u8(q, "zig"))
    H4 = q_byte_entropy_bits(encode_byte_map_u8(requant_signed_q(q, 16), "zig"))
    H3 = q_byte_entropy_bits(encode_byte_map_u8(requant_signed_q(q, 8), "zig"))
    H2 = q_byte_entropy_bits(encode_byte_map_u8(requant_signed_q(q, 4), "zig"))
    assert H2 < H3 < H4 < H_full


def test_requant_error_bounded_by_grid_step():
    # Lossy-on-pixels but BOUNDED: dequant error <= half a coarse-grid step.
    rng = np.random.default_rng(4)
    q = rng.integers(-127, 128, size=10000).astype(np.int8)
    for levels in (16, 8, 4):
        step = 255.0 / (levels - 1)
        err = np.abs(requant_signed_q(q, levels).astype(np.float64) - q.astype(np.float64))
        # Bound = half a grid step + 1 (reconstruction points are stored as int8,
        # so they sit up to 0.5 off the ideal grid; the q-value rounding adds <=0.5).
        assert err.max() <= step / 2 + 1.0


def test_requant_invalid_levels_raises():
    with pytest.raises(ValueError):
        requant_signed_q(np.zeros(4, np.int8), 1)


def test_requant_roundtrip_through_grammar_preserves_distinctness():
    # encode(requant(decode(u8))) must be valid uint8 AND differ from original.
    rng = np.random.default_rng(5)
    u8 = rng.integers(0, 256, size=10000).astype(np.uint8)
    signed = decode_byte_map_u8(u8, "zig")
    rq = requant_signed_q(signed, 16)
    out_u8 = encode_byte_map_u8(rq, "zig")
    assert out_u8.dtype == np.uint8
    assert not np.array_equal(out_u8, u8)  # real change in stored bytes
    # And decoding the new bytes recovers exactly the re-quantized q-values.
    assert np.array_equal(decode_byte_map_u8(out_u8, "zig"), rq)


# ---------------------------------------------------------------------------
# Sensitivity-driven bit allocation — the ORIGINAL method (class 7)
# ---------------------------------------------------------------------------


def _toy_sensitivity_inputs():
    # 5 tensors: idx 0 hugely sensitive, idx 4 zero-sensitive.
    sensitivities = {0: 1.0, 1: 0.5, 2: 0.1, 3: 0.01, 4: 0.0}
    numels = {0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000}
    names = {i: f"t{i}" for i in range(5)}
    return sensitivities, numels, names


def test_allocator_keeps_high_sensitivity_full_and_crushes_low():
    sens, numels, names = _toy_sensitivity_inputs()
    plans = allocate_bits_by_sensitivity(
        sensitivities=sens, numels=numels, names=names,
        sensitivity_threshold=0.2,
    )
    # Highest-sensitivity tensor stays full precision.
    assert plans[0].levels == 256
    assert plans[1].levels == 256  # 0.5 >= threshold 0.2
    # Below-threshold tensors get crushed.
    assert plans[2].levels < 256
    assert plans[3].levels < 256
    assert plans[4].levels < 256
    # Lower sensitivity => deeper crush (fewer levels).
    assert plans[4].levels <= plans[3].levels <= plans[2].levels


def test_allocator_protect_top_k_overrides_threshold():
    sens, numels, names = _toy_sensitivity_inputs()
    # With threshold ABOVE every sensitivity, only protect_top_k keeps tensors
    # full; the rest crush. protect_top_k=2 must protect the two MOST sensitive.
    plans = allocate_bits_by_sensitivity(
        sensitivities=sens, numels=numels, names=names,
        sensitivity_threshold=2.0, protect_top_k=2,
    )
    assert plans[0].levels == 256  # top-1 sensitive, protected
    assert plans[1].levels == 256  # top-2 sensitive, protected
    assert plans[2].levels < 256  # not in top-2 and below threshold -> crushed
    assert plans[3].levels < 256
    assert plans[4].levels < 256


def test_allocator_ranking_beats_random_on_protected_sensitivity_mass():
    # The CORE claim: a plan built from the TRUE per-tensor sensitivity keeps the
    # high-sensitivity mass at full precision, whereas a plan built from a RANDOM
    # (sensitivity-blind) ranking does not. We compare the TRUE sensitivity mass
    # left at full precision under each. With many tensors and a fixed protect
    # budget, the true-ranked plan must (in expectation, decisively) protect more
    # true-sensitivity mass than the random one.
    rng = np.random.default_rng(7)
    n = 24  # realistic decoder tensor count
    # Skewed sensitivities: a few tensors carry almost all the score sensitivity.
    true_sens = {i: float(rng.random() ** 4) for i in range(n)}
    numels = {i: 1000 for i in range(n)}
    names = {i: f"t{i}" for i in range(n)}

    def true_mass_at_full(plans):
        return sum(true_sens[i] for i, p in plans.items() if p.levels == 256)

    # True-ranked plan: protect top-6 by REAL sensitivity, crush the rest.
    ranked = allocate_bits_by_sensitivity(
        sensitivities=true_sens, numels=numels, names=names,
        sensitivity_threshold=1.0, protect_top_k=6,  # threshold > max -> only top-k full
    )
    ranked_mass = true_mass_at_full(ranked)

    # Random-ranked plans: protect top-6 by a RANDOM key (sensitivity-blind).
    wins_for_random = 0
    n_trials = 100
    for _ in range(n_trials):
        fake = {i: float(rng.random()) for i in range(n)}
        rnd = allocate_bits_by_sensitivity(
            sensitivities=fake, numels=numels, names=names,
            sensitivity_threshold=1.0, protect_top_k=6,
        )
        # Score the RANDOM plan by the TRUE sensitivity it protects.
        rnd_mass = sum(
            true_sens[i] for i, p in rnd.items() if p.levels == 256
        )
        if rnd_mass >= ranked_mass:
            wins_for_random += 1
    # The true-ranked plan should protect strictly more true mass than nearly
    # every random plan (it protects the genuine top-6 by construction).
    assert wins_for_random <= 2, (
        f"sensitivity-blind ranking matched/beat true ranking "
        f"{wins_for_random}/{n_trials} times — allocator is not using sensitivity"
    )


def test_allocator_empty_sensitivities_raises():
    with pytest.raises(ValueError):
        allocate_bits_by_sensitivity(
            sensitivities={}, numels={}, names={}, sensitivity_threshold=0.1,
        )


def test_tensor_requant_plan_effective_bits():
    p = TensorRequantPlan(storage_index=0, name="t0", numel=100, levels=16)
    assert abs(p.effective_bits - 4.0) < 1e-9


# ---------------------------------------------------------------------------
# Contest score (recompute-from-components, class 8 authority discipline)
# ---------------------------------------------------------------------------


def test_contest_score_matches_frontier_anchor():
    # Frontier: d_seg=0.00055978, d_pose=2.942e-05, 177169 B -> 0.19109982.
    out = contest_score_from_components(
        d_seg=0.00055978, d_pose=0.00002942, archive_zip_size=177169,
    )
    assert abs(out["score"] - 0.19109982) < 1e-6
    # Rate term dominates (decoder weights = 91% of bytes).
    assert out["rate_term"] > out["seg_term"]
    assert out["rate_term"] > out["pose_term"]


def test_score_delta_decomposition_rate_only_when_distortion_held():
    base = contest_score_from_components(
        d_seg=0.00055978, d_pose=0.00002942, archive_zip_size=177169,
    )
    # Cut 5000 bytes, distortion held => the win is rate-only and negative.
    cand = contest_score_from_components(
        d_seg=0.00055978, d_pose=0.00002942, archive_zip_size=172169,
    )
    delta = score_delta_components(base=base, cand=cand)
    assert delta["d_score"] < 0
    assert abs(delta["d_seg_term"]) < 1e-12
    assert abs(delta["d_pose_term"]) < 1e-12
    assert delta["d_rate_term"] < 0


def test_score_increases_when_distortion_moves_more_than_rate_saved():
    base = contest_score_from_components(
        d_seg=0.00055978, d_pose=0.00002942, archive_zip_size=177169,
    )
    # Crushed too hard: saved 3000 bytes but d_seg doubled -> net worse.
    cand = contest_score_from_components(
        d_seg=0.00111956, d_pose=0.00002942, archive_zip_size=174169,
    )
    delta = score_delta_components(base=base, cand=cand)
    assert delta["d_seg_term"] > 0
    assert delta["d_score"] > 0  # the seg penalty dominates the rate saving
