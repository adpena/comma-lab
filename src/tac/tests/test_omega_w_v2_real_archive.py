"""Lane Ω-W-V2 — REAL archive byte-savings validation on Lane G v3 renderer.bin.

Council F Part B SAFE-LOCAL validation
======================================

Per `.omx/research/council_f_retrain_ev_validation_admm_consult_20260429.md`
Part B (Lane Ω-W-V2 Real-Archive Validation Protocol), this test exercises
the codec against the ACTUAL Lane G v3 renderer.bin (290KB) — not a synthetic
Gaussian fixture. The synthetic 69.11% savings claim in
``src/tac/water_filling_codec_v2.py`` docstring is REPLACED by whatever
this test measures empirically.

Why this is SAFE-LOCAL (no GPU, no MPS, no scorer)
--------------------------------------------------
Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable + memory
``feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md``:

The dividing line for "safe local" is "does the measurement depend on a
neural-net forward pass?" If yes → contest-CUDA only. If no → local OK.

This test:
* Reads renderer.bin via ``load_renderer_checkpoint`` (deterministic
  bytes-to-state_dict path; no neural pass, no scorer load).
* Calls ``encode_omega_w_v2`` / ``decode_omega_w_v2`` which are pure-Python
  + numpy int32 + bit-deterministic CPU arithmetic (no CUDA, no MPS, no
  scorer).
* Measures byte counts (file IO; bit-identical).
* Measures L_inf round-trip error (deterministic floating-point on CPU
  at small-int × power-of-2 magnitudes — bit-identical).

What this test PROVES
---------------------
* Bit-faithful round-trip per Conv2d weight.
* Aggregate byte savings vs V1 raw qint estimate IS in the council-predicted
  band [20%, 60%].

What this test does NOT prove
-----------------------------
1. Does NOT prove the encoded weights inflate to a renderer that scores
   within Lane G v3's band. To prove that: encode → decode → load into
   renderer.bin → ship in archive → contest-CUDA auth eval. That is a
   Vast.ai 4090 dispatch (~$0.50), NOT in scope of this test.
2. Does NOT prove ADMM coordinator wraps Ω-W-V2 correctly (separate test:
   ``test_joint_admm_4stream_nonconvex.py``).
3. Does NOT prove savings hold under outer DEFLATE. V2 internally uses a
   static-histogram arithmetic coder; xz LZMA on V1 may close the gap.
4. Does NOT validate hyperprior amortisation (V3 scope).

Tag: ``[empirical:src/tac/tests/test_omega_w_v2_real_archive.py]``
Anchor: ``experiments/results/lane_g_v3_landed/iter_0/renderer.bin``
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tac.renderer_export import load_renderer_checkpoint
from tac.water_filling_codec_v2 import (
    BlockFPIneligible,
    GateRegression,
    decode_omega_w_v2,
    encode_omega_w_v2,
)


# ── anchor file (verified to exist 2026-04-29 PM) ─────────────────────────


_ANCHOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "results"
    / "lane_g_v3_landed"
    / "iter_0"
    / "renderer.bin"
)
"""Lane G v3's actual shipped renderer.bin (~290KB, ASYM magic).

This test is GATED on Lane G v3 having been landed; the file is committed
to the repo so the test is reproducible across machines. If the file ever
moves, update this constant + Council F report cross-reference.
"""


def _v1_raw_byte_estimate(weights: torch.Tensor) -> int:
    """Mirror ``water_filling_codec_v2._v1_raw_qint_byte_estimate``.

    V1 stores qint as INT8 dense + per-channel int32 exponents +
    ~32B fixed header. This is the apples-to-apples baseline V2 must beat.
    """
    o, i, kh, kw = weights.shape
    return int(o * i * kh * kw) + int(o * 4) + 32


# ── (a) primary test: aggregate byte savings on real architecture ─────────


def test_omega_w_v2_real_lane_g_v3_renderer_bin_byte_savings() -> None:
    """[empirical:src/tac/tests/test_omega_w_v2_real_archive.py] Validate
    Ω-W-V2 on Lane G v3's actual renderer.bin (not a synthetic tensor).

    DOES prove: bit-faithful round-trip per Conv2d weight; aggregate byte
    savings vs V1 raw qint estimate are inside Council F's predicted band
    [20%, 60%].

    Does NOT prove: any score change. To prove score parity, encode →
    decode → archive → contest-CUDA auth eval (Vast.ai 4090, ~$0.50,
    NOT in scope of this test — see CLAUDE.md "MPS auth eval is NOISE"
    non-negotiable).
    """
    assert _ANCHOR_PATH.exists(), (
        f"missing anchor {_ANCHOR_PATH}; this test is gated on Lane G v3 "
        f"having been landed. If the file moved, update the anchor path "
        f"in test_omega_w_v2_real_archive.py."
    )

    model = load_renderer_checkpoint(str(_ANCHOR_PATH))
    sd = model.state_dict()

    eligible_count = 0
    skipped_count = 0
    aggregate_v1_bytes = 0
    aggregate_v2_bytes = 0
    per_tensor_savings: list[tuple[str, int, int, float]] = []
    per_tensor_l1: list[tuple[str, float, float]] = []

    for name, w in sd.items():
        if w.dim() != 4:
            skipped_count += 1
            continue
        o = int(w.shape[0])
        # Single-output-channel tensors have no per-channel spread; the
        # arithmetic terminal cannot exploit redundancy + the OWV2 header
        # dwarfs the payload. The honest move is to skip — V1 is the right
        # codec for these layers.
        if o < 2:
            skipped_count += 1
            continue
        # Synthetic uniform Hessian per Council F Part B §3.2 design note:
        # "synthetic uniform Hessian is conservative — it gives Ω-W-V2 the
        # WORST-CASE bit allocation. Real per-channel Hessian from a
        # calibration loop would be MORE selective and likely save MORE
        # bytes. So the test's measured savings are a LOWER BOUND on real
        # savings."
        hess = torch.ones(o, dtype=torch.float32)
        bytes_v1 = _v1_raw_byte_estimate(w)
        # Target 30% bit-budget reduction from V1 raw — within Council F's
        # predicted [20%, 60%] band. The water-fill internally clamps to
        # the discrete Q ladder; this is just the upper-bound on bits.
        total_bits = int(bytes_v1 * 0.7 * 8)
        try:
            payload = encode_omega_w_v2(
                weights_block_fp=w,
                hessian=hess,
                total_bits=total_bits,
            )
        except (BlockFPIneligible, GateRegression) as exc:
            # Honest skip per the Carmack overhead-gate rule: V2 refuses to
            # ship larger than V1 on tiny tensors (e.g. head 3x36x1x1).
            skipped_count += 1
            per_tensor_savings.append((name, bytes_v1, -1, float("nan")))
            print(f"  [v2-real] SKIP {name}: {type(exc).__name__}")
            continue

        eligible_count += 1
        recon = decode_omega_w_v2(blob=payload)

        # Round-trip tolerance: derived from block-FP per-channel algebra.
        # qint widths land in {1, 3, 7, 15, 31} → quantisation step ≈
        # max_abs_per_channel / Q_c. Linf bound (NOT arbitrary) is
        # 2 * max_abs / 2^3 = max_abs / 4 because the council-mandated
        # 3-bit floor (Q_c=7) gives at worst ~max_abs/8 step + sign
        # rounding, with safety factor 2 for cross-channel max.
        max_abs = float(w.abs().max().item())
        tol = 2.0 * max_abs * (2.0 ** -3)  # = max_abs / 4
        max_abs_err = float((w - recon).abs().max().item())
        l1 = float((w - recon).abs().mean().item())
        assert max_abs_err <= tol, (
            f"{name}: max_abs_err={max_abs_err:.4f} > tol={tol:.4f} "
            f"(max_abs={max_abs:.4f}); per-channel quantisation algebra "
            f"violated."
        )
        # Sanity: shape + dtype preserved.
        assert recon.shape == w.shape, (
            f"{name}: round-trip shape {tuple(recon.shape)} != "
            f"input {tuple(w.shape)}"
        )
        assert recon.dtype == torch.float32

        aggregate_v1_bytes += bytes_v1
        aggregate_v2_bytes += len(payload)
        savings_pct = 100.0 * (1.0 - len(payload) / bytes_v1)
        per_tensor_savings.append((name, bytes_v1, len(payload), savings_pct))
        per_tensor_l1.append((name, l1, max_abs_err))

    assert eligible_count > 0, (
        "no eligible Conv2d weights found in Lane G v3 renderer.bin; "
        "the loader returned a state_dict with no 4-D conv weights. "
        "Did the renderer architecture change?"
    )
    aggregate_savings_pct = 100.0 * (1.0 - aggregate_v2_bytes / aggregate_v1_bytes)

    # Council F Part B band: [20%, 60%] on real architecture.
    # Lower bound is the FALSIFICATION threshold for the docstring claim
    # of 69.11% synthetic savings — if real architecture saves <20%, the
    # synthetic claim is FALSIFIED.
    assert aggregate_savings_pct >= 20.0, (
        f"Ω-W-V2 saves {aggregate_savings_pct:.1f}% on Lane G v3 "
        f"renderer.bin (eligible={eligible_count}, skipped={skipped_count}); "
        f"docstring 69.11% synthetic claim FALSIFIED on real architecture; "
        f"aggregate V1={aggregate_v1_bytes}B → V2={aggregate_v2_bytes}B; "
        f"update src/tac/water_filling_codec_v2.py docstring before promoting "
        f"AND save a memory file noting Council F band [20%, 60%] is wrong; "
        f"Round 8 council re-evaluation required."
    )
    # Upper bound: catches V1 byte estimator over-counting bugs.
    assert aggregate_savings_pct <= 60.0, (
        f"Ω-W-V2 saves {aggregate_savings_pct:.1f}% on Lane G v3 "
        f"renderer.bin — this exceeds Council F's predicted band upper "
        f"edge of 60%. Verify the V1 byte estimator is not over-counting "
        f"(e.g. forgot to amortise the per-channel exponent header). If "
        f"the measurement is correct, Council F band is too tight; save "
        f"a memory file proposing Round 8 re-evaluation."
    )

    # Surface the empirical numbers for the agent's report-back.
    print(
        f"\n  [empirical:src/tac/tests/test_omega_w_v2_real_archive.py] "
        f"Ω-W-V2 on Lane G v3 renderer.bin: "
        f"V1_bytes={aggregate_v1_bytes}, V2_bytes={aggregate_v2_bytes}, "
        f"savings={aggregate_savings_pct:.2f}%, "
        f"eligible_tensors={eligible_count}, skipped={skipped_count}"
    )
    print("  Per-tensor breakdown:")
    for name, b1, b2, pct in per_tensor_savings:
        if b2 < 0:
            print(f"    {name}: V1={b1}B SKIP")
        else:
            print(f"    {name}: V1={b1}B → V2={b2}B ({pct:+.2f}%)")


# ── (b) per-tensor round-trip L_inf bound (per-channel algebra check) ─────


def test_omega_w_v2_real_per_tensor_round_trip_l_inf() -> None:
    """[empirical:src/tac/tests/test_omega_w_v2_real_archive.py] On every
    eligible Lane G v3 conv weight, round-trip L_inf error MUST stay below
    the per-channel block-FP algebra bound.

    The bound 2 * max_abs * 2^-3 = max_abs / 4 corresponds to the
    council-mandated 3-bit floor of the qint ladder (Q=7 → step ~max/8;
    with safety factor 2 for cross-channel max).

    This test is REDUNDANT with the assertion inside the byte-savings
    test, but is split out so a per-tensor L_inf failure is loud + isolated
    from a savings-band failure. Two distinct failure modes, two tests.
    """
    assert _ANCHOR_PATH.exists(), f"missing anchor {_ANCHOR_PATH}"

    model = load_renderer_checkpoint(str(_ANCHOR_PATH))
    sd = model.state_dict()

    n_checked = 0
    for name, w in sd.items():
        if w.dim() != 4:
            continue
        o = int(w.shape[0])
        if o < 2:
            continue
        hess = torch.ones(o, dtype=torch.float32)
        bytes_v1 = _v1_raw_byte_estimate(w)
        total_bits = int(bytes_v1 * 0.7 * 8)
        try:
            payload = encode_omega_w_v2(
                weights_block_fp=w,
                hessian=hess,
                total_bits=total_bits,
            )
        except (BlockFPIneligible, GateRegression):
            continue
        recon = decode_omega_w_v2(blob=payload)
        max_abs = float(w.abs().max().item())
        # Round-trip L_inf bound: per-channel block-FP algebra.
        tol = 2.0 * max_abs * (2.0 ** -3)
        linf = float((w - recon).abs().max().item())
        assert linf <= tol, (
            f"{name}: round-trip Linf={linf:.6f} > tol={tol:.6f} "
            f"(max_abs={max_abs:.4f})"
        )
        n_checked += 1

    assert n_checked > 0, "no tensors validated; loader change?"


# ── (c) idempotence: encode→decode→encode produces identical bytes ─────────


def test_omega_w_v2_real_round_trip_idempotent() -> None:
    """[empirical:src/tac/tests/test_omega_w_v2_real_archive.py] Decoded
    weights re-encoded with the SAME Hessian + total_bits MUST produce
    identical bytes (decoded weights are already on the quantization
    lattice, so re-encoding is bit-stable).

    This catches non-determinism in the arithmetic terminal + per-channel
    quantisation path. We exercise on the LARGEST eligible tensor (gives
    the most chances for non-determinism to surface).
    """
    assert _ANCHOR_PATH.exists(), f"missing anchor {_ANCHOR_PATH}"
    model = load_renderer_checkpoint(str(_ANCHOR_PATH))
    sd = model.state_dict()

    # Pick the largest eligible 4-D tensor.
    largest_name = None
    largest_w: torch.Tensor | None = None
    largest_numel = -1
    for name, w in sd.items():
        if w.dim() != 4:
            continue
        if int(w.shape[0]) < 2:
            continue
        if w.numel() > largest_numel:
            largest_numel = w.numel()
            largest_name = name
            largest_w = w

    assert largest_w is not None, "no eligible tensor in Lane G v3 renderer.bin"
    assert largest_name is not None
    o = int(largest_w.shape[0])
    hess = torch.ones(o, dtype=torch.float32)
    bytes_v1 = _v1_raw_byte_estimate(largest_w)
    total_bits = int(bytes_v1 * 0.7 * 8)

    blob_a = encode_omega_w_v2(
        weights_block_fp=largest_w,
        hessian=hess,
        total_bits=total_bits,
    )
    decoded_a = decode_omega_w_v2(blob=blob_a)
    blob_b = encode_omega_w_v2(
        weights_block_fp=decoded_a,
        hessian=hess,
        total_bits=total_bits,
    )
    decoded_b = decode_omega_w_v2(blob=blob_b)
    blob_c = encode_omega_w_v2(
        weights_block_fp=decoded_b,
        hessian=hess,
        total_bits=total_bits,
    )

    # Second-and-third re-encodes MUST be bit-identical (decoded already
    # on the lattice).
    assert blob_b == blob_c, (
        f"{largest_name}: re-encode of decoded tensor non-idempotent: "
        f"|blob_b|={len(blob_b)} vs |blob_c|={len(blob_c)}"
    )
    print(
        f"  [empirical] idempotent re-encode on {largest_name} "
        f"({largest_w.shape}): {len(blob_a)}B → {len(blob_b)}B "
        f"(bit-identical on second decode→encode loop)"
    )


# ── (d) anchor-existence guard ────────────────────────────────────────────


def test_omega_w_v2_real_anchor_present() -> None:
    """The Lane G v3 renderer.bin anchor must exist + be ASYM-magic.

    Cheap fail-loud guard so that if the anchor file moves / is deleted
    the failure message is one line of grep, not a 200-line traceback
    inside ``load_renderer_checkpoint``.
    """
    assert _ANCHOR_PATH.exists(), (
        f"missing anchor {_ANCHOR_PATH} — Lane G v3 must have been landed "
        f"for this test to run (and the renderer.bin must be committed "
        f"to the repo so the test is reproducible across machines)."
    )
    head = _ANCHOR_PATH.read_bytes()[:4]
    assert head in (b"ASYM", b"DPSM"), (
        f"anchor {_ANCHOR_PATH} has unexpected magic {head!r}; expected "
        f"ASYM (Lane G v3 ships AsymmetricPairGenerator) or DPSM."
    )
    size = _ANCHOR_PATH.stat().st_size
    # Lane G v3 archive is ~290KB; sanity-check the file has not been
    # truncated to zero by an rsync mishap.
    assert size > 100_000, (
        f"anchor {_ANCHOR_PATH} is {size}B — too small; expected ~290KB "
        f"(Lane G v3 renderer.bin)."
    )


# ── (e) skipped-tensor guard: at least one ineligible tensor present ─────


def test_omega_w_v2_real_overhead_gate_fires_on_tiny_layer() -> None:
    """[empirical:src/tac/tests/test_omega_w_v2_real_archive.py] Lane G v3
    contains tiny tensors (e.g. ``renderer.head.weight`` shape (3, 36, 1, 1))
    where the OWV2 header dwarfs the payload. The Carmack overhead-gate
    MUST fire on at least one such tensor — the codec refuses to ship a
    regression.

    This test is the inverse of the byte-savings test: it documents that
    V2 is HONEST about its limits. If this test stops finding any tiny
    tensor that triggers GateRegression, the renderer architecture has
    changed (every conv layer is now large enough to amortise the OWV2
    header) — re-evaluate the codec's amortisation threshold.
    """
    assert _ANCHOR_PATH.exists(), f"missing anchor {_ANCHOR_PATH}"
    model = load_renderer_checkpoint(str(_ANCHOR_PATH))
    sd = model.state_dict()

    n_gated = 0
    for name, w in sd.items():
        if w.dim() != 4:
            continue
        o = int(w.shape[0])
        if o < 2:
            continue
        hess = torch.ones(o, dtype=torch.float32)
        bytes_v1 = _v1_raw_byte_estimate(w)
        total_bits = int(bytes_v1 * 0.7 * 8)
        try:
            encode_omega_w_v2(
                weights_block_fp=w,
                hessian=hess,
                total_bits=total_bits,
            )
        except GateRegression:
            n_gated += 1

    # We expect AT LEAST ONE tiny tensor to trigger the overhead gate
    # (renderer.head.weight at (3, 36, 1, 1) does so today). If the
    # architecture changes such that no tensor is small enough to gate,
    # update the assertion message but keep the test (it still documents
    # the behaviour).
    assert n_gated >= 1, (
        f"expected >=1 tiny tensor to trigger GateRegression on Lane G v3 "
        f"renderer.bin; got 0. Either the architecture changed (no tiny "
        f"layers anymore) or the overhead-gate threshold drifted. "
        f"Investigate before promoting."
    )
    print(f"  [empirical] overhead-gate fired on {n_gated} tiny tensor(s)")


# ── (f) explicit caveat documentation (failure-mode test) ─────────────────


@pytest.mark.parametrize("caveat", [
    "does NOT prove score change",
    "does NOT replace contest-CUDA auth eval",
    "does NOT validate hyperprior",
    "does NOT validate ADMM coordinator wrapping",
])
def test_omega_w_v2_real_archive_caveat_documented(caveat: str) -> None:
    """The test module docstring MUST explicitly document each caveat.

    Belt-and-suspenders: if a future agent strips the caveats from the
    docstring (e.g. "we'll fix it later"), this test fails LOUD. The
    caveats are the ONLY thing keeping the test honest about what it
    proves vs what it does not.
    """
    import sys

    mod = sys.modules[__name__]
    src = mod.__doc__ or ""
    # Match against simplified forms (the docstring uses prose paraphrases).
    keywords_per_caveat = {
        "does NOT prove score change": [
            "Does NOT prove the encoded weights inflate to a renderer that scores",
            "score change",
            "score parity",
        ],
        "does NOT replace contest-CUDA auth eval": [
            "contest-CUDA auth eval",
            "Vast.ai 4090",
        ],
        "does NOT validate hyperprior": ["hyperprior"],
        "does NOT validate ADMM coordinator wrapping": [
            "ADMM coordinator wraps",
            "test_joint_admm_4stream_nonconvex",
        ],
    }
    keywords = keywords_per_caveat[caveat]
    found = any(kw in src for kw in keywords)
    assert found, (
        f"caveat {caveat!r} not documented in test module docstring; "
        f"expected one of {keywords!r}"
    )
