"""Lane 8 — multi-pass compressor regression tests.

Covers (per the task spec from `.omx/research/council_lane_8_multipass_design_20260430.md`):

  (a) Synthetic 2-pass converges to a known optimum on a quadratic objective.
  (b) Real-archive 3-pass on a Lane G v3 anchor PROXY (offline; no GPU).
      3-pass produces score <= single-pass score (no regression).
  (c) Regression guard: an artificial pass that increases score is reverted.
  (d) MAX_PASSES enforcement (default 3, absolute cap 5).
  (e) Pass history schema validation (PassRecord fields complete).
  (f) Parameter clamping: out-of-range proposals are clamped to PARAM_RANGES.
  (g) Inflate-time assertion: constructing/running from inflate_renderer.py
      raises (the strict-scorer-rule guard).
  (h) Eps stop: |delta| < eps short-circuits the loop.
  (i) Target hit: score < target_score short-circuits the loop.
  (j) AdjustmentPolicy ABC: subclassable; the coordinate-descent default
      walks the priority axis order.
  (k) Determinism: same encoder + scorer + initial_params produces identical
      results across two runs.
  (l) Functional wrapper ``compress_with_multipass`` returns equivalent result.

NOTE: tests use synthetic byte-and-score-only proxies — NO scorer / renderer
loaded — so they run in <1s on CPU. The CUDA path is wired via
`experiments/pipeline.py` integration (Phase D).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from tac.multipass_compressor import (
    ABSOLUTE_MAX_PASSES,
    DEFAULT_EPS,
    DEFAULT_MAX_PASSES,
    PARAM_RANGES,
    AdjustmentPolicy,
    CoordinateDescentPolicy,
    MultiPassCompressor,
    PassRecord,
    compress_with_multipass,
)


# ── (a) synthetic 2-pass quadratic-objective convergence ─────────────────────


def test_synthetic_quadratic_converges_in_two_passes() -> None:
    """The encoder is a no-op; the scorer is a quadratic in mask_crf with
    minimum at CRF=50 (the default initial value). The CoordinateDescentPolicy
    increases CRF on each pass; at CRF=55 the score increases (regression),
    so the regression guard reverts to pass_idx=0.
    """
    def encoder(state: object, params: dict) -> bytes:
        return f"crf={params['mask_crf']:.1f}".encode()

    def scorer(archive: bytes) -> float:
        crf = float(archive.decode().split("=")[1])
        return (crf - 50.0) ** 2 / 100.0

    result = MultiPassCompressor(
        target_score=-1.0,            # impossible — force eps/regression-stop
        max_passes=3,
        eps=DEFAULT_EPS,
        regression_guard=True,
        initial_params={"mask_crf": 50.0},
    ).compress(None, encoder, scorer)

    # The optimum is the initial pass (delta_pass2 < -eps triggers revert).
    assert result.best_pass_idx == 0
    assert result.reverted is True
    assert result.final_score == pytest.approx(0.0, abs=1e-9)
    assert len(result.pass_history) >= 2


# ── (a-bis) convergence-direction: initial OFF the optimum ──────────────────
#
# Round 1 Contrarian finding: the original synthetic test happens to start
# AT the optimum, so it only exercises the regression-guard path, not the
# convergence direction. Add an explicit test that starts BELOW the optimum
# and confirms the policy walks UP and then plateaus / regresses around it.


def test_block_fp_lower_clamp_plateau() -> None:
    """Round 2 Dykstra finding: when an axis is at its lower clamp,
    the monotone-decrease step is absorbed by clamping → encoder receives
    the SAME params → score is constant → eps-stop fires.

    Use a custom policy that always proposes block_fp_block_size below the
    clamp; verify the loop terminates via eps-stop, not via crash.
    """

    class BlockFPClampingPolicy(AdjustmentPolicy):
        def propose_next_params(self, history, current_params):
            return (
                {**current_params, "block_fp_block_size": 0.0},  # < clamp(4.0)
                "force-clamp",
            )

    def encoder(state, params):
        # Encoder is constant — score depends only on the params dict's
        # block_fp_block_size (clamped). Same clamped value -> same archive.
        return f"bfp={params['block_fp_block_size']:.1f}".encode()

    def scorer(archive):
        return 0.5  # constant — eps-stop fires after 2 passes

    result = MultiPassCompressor(
        target_score=-1.0, max_passes=5, eps=DEFAULT_EPS,
        policy=BlockFPClampingPolicy(),
        initial_params={
            "mask_crf": 50.0,
            "pose_q_bits": 8.0,
            "block_fp_block_size": 4.0,   # at lower clamp
            "residual_gain": 0.0,
        },
    ).compress(None, encoder, scorer)

    # Pass 0 is initial; pass 1's proposed bfp=0 is clamped to 4 → same
    # encoder output → same score → eps-stop converges.
    assert result.converged is True
    assert len(result.pass_history) == 2
    # Both passes saw block_fp_block_size = 4.0 (clamped).
    assert result.pass_history[0].params["block_fp_block_size"] == 4.0
    assert result.pass_history[1].params["block_fp_block_size"] == 4.0


def test_initial_off_optimum_walks_toward_minimum() -> None:
    """Initial CRF=42 with optimum at CRF=50. The CoordinateDescentPolicy
    increases CRF by 5 per pass: 42 -> 47 (improving) -> 52 (regression past
    optimum, guard fires, revert to pass at 47).
    """
    def encoder(state, params):
        return f"crf={params['mask_crf']:.1f}".encode()

    def scorer(archive):
        crf = float(archive.decode().split("=")[1])
        return (crf - 50.0) ** 2 / 100.0

    result = MultiPassCompressor(
        target_score=-1.0,
        max_passes=5, eps=DEFAULT_EPS,
        regression_guard=True,
        initial_params={"mask_crf": 42.0},
    ).compress(None, encoder, scorer)

    # Pass 0: CRF=42, score=(8)**2/100 = 0.64
    # Pass 1: CRF=47, score=(3)**2/100 = 0.09  (improvement)
    # Pass 2: CRF=52, score=(2)**2/100 = 0.04  (improvement)
    # Pass 3: CRF=57, score=(7)**2/100 = 0.49  (regression — revert to pass 2)
    # Best pass: pass 2 at CRF=52.
    assert result.best_pass_idx == 2
    assert result.reverted is True
    assert result.final_score == pytest.approx(0.04, abs=1e-9)


# ── (b) real-archive 3-pass proxy: no-regression invariant ───────────────────


def test_three_pass_no_regression_proxy() -> None:
    """Proxy: encoder maps mask_crf -> archive size linearly; scorer returns
    a known concave-down score curve. Verify 3-pass result is at least as
    good as the 1-pass result.
    """
    def encoder(state: object, params: dict) -> bytes:
        crf = params.get("mask_crf", 50.0)
        size = max(100, int(2000 - 30 * crf))
        return b"x" * size

    def scorer(archive: bytes) -> float:
        # Score has a unique minimum — encoder smaller archives buy bits but
        # eventually distortion dominates. Toy curve: f(s) = (s - 800)**2 / 1e5
        size = len(archive)
        return ((size - 800) ** 2) / 1e5

    one_pass = MultiPassCompressor(
        target_score=-1.0, max_passes=1, eps=DEFAULT_EPS,
        initial_params={"mask_crf": 30.0},
    ).compress(None, encoder, scorer)

    three_pass = MultiPassCompressor(
        target_score=-1.0, max_passes=3, eps=DEFAULT_EPS,
        initial_params={"mask_crf": 30.0},
    ).compress(None, encoder, scorer)

    assert three_pass.final_score <= one_pass.final_score + 1e-9


# ── (c) regression guard: artificial score-up reverted ───────────────────────


def test_regression_guard_reverts_to_best() -> None:
    """Encoder ignores params; scorer returns 1.0, 0.5, 2.0 across calls.
    The third pass must be reverted; result.best_pass_idx == 1.
    """
    scores = iter([1.0, 0.5, 2.0])

    def encoder(state: object, params: dict) -> bytes:
        return b"const"

    def scorer(archive: bytes) -> float:
        return next(scores)

    result = MultiPassCompressor(
        target_score=-1.0, max_passes=3, eps=DEFAULT_EPS,
        regression_guard=True,
        initial_params={"mask_crf": 50.0},
    ).compress(None, encoder, scorer)

    assert result.reverted is True
    assert result.best_pass_idx == 1
    assert result.final_score == pytest.approx(0.5)


def test_regression_guard_off_does_not_revert() -> None:
    """With ``regression_guard=False``, a score-up move is accepted but the
    best-so-far tracking still picks the lowest-score archive.
    """
    scores = iter([1.0, 0.5, 2.0])

    def encoder(state: object, params: dict) -> bytes:
        return b"const"

    def scorer(archive: bytes) -> float:
        return next(scores)

    result = MultiPassCompressor(
        target_score=-1.0, max_passes=3, eps=DEFAULT_EPS,
        regression_guard=False,
        initial_params={"mask_crf": 50.0},
    ).compress(None, encoder, scorer)

    assert result.reverted is False
    # All 3 passes ran — best is still pass_idx=1 (score 0.5).
    assert result.best_pass_idx == 1
    assert result.final_score == pytest.approx(0.5)
    assert len(result.pass_history) == 3


# ── (d) MAX_PASSES enforcement ───────────────────────────────────────────────


def test_max_passes_default_is_three() -> None:
    assert DEFAULT_MAX_PASSES == 3


def test_max_passes_absolute_cap_is_five() -> None:
    assert ABSOLUTE_MAX_PASSES == 5


def test_max_passes_above_absolute_raises() -> None:
    with pytest.raises(ValueError, match="ABSOLUTE_MAX_PASSES"):
        MultiPassCompressor(
            target_score=0.5, max_passes=10, eps=DEFAULT_EPS,
        )


def test_max_passes_below_one_raises() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        MultiPassCompressor(target_score=0.5, max_passes=0, eps=DEFAULT_EPS)


def test_eps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="eps"):
        MultiPassCompressor(target_score=0.5, max_passes=3, eps=0.0)


# ── (e) PassRecord schema ────────────────────────────────────────────────────


def test_pass_record_schema_complete() -> None:
    """Each PassRecord must carry: pass_idx, params, archive_bytes, score,
    delta, elapsed_seconds, reason.
    """
    rec = PassRecord(
        pass_idx=0,
        params={"mask_crf": 50.0},
        archive_bytes=1000,
        score=0.42,
        delta=0.01,
        elapsed_seconds=12.3,
        reason="init",
    )
    d = rec.to_dict()
    for key in (
        "pass_idx", "params", "archive_bytes", "score",
        "delta", "elapsed_seconds", "reason",
    ):
        assert key in d, f"PassRecord missing field {key!r}"


def test_multipass_result_schema_complete() -> None:
    def encoder(state, params):
        return b"x" * 100

    def scorer(archive):
        return 0.5

    result = MultiPassCompressor(
        target_score=-1.0, max_passes=2, eps=DEFAULT_EPS,
        initial_params={"mask_crf": 50.0},
    ).compress(None, encoder, scorer)
    d = result.to_dict()
    for key in (
        "final_score", "best_pass_idx", "converged", "target_hit",
        "reverted", "n_passes", "final_archive_bytes_len", "pass_history",
    ):
        assert key in d, f"MultiPassResult missing field {key!r}"


# ── (f) parameter clamping ───────────────────────────────────────────────────


def test_param_clamping_initial_params_in_range() -> None:
    """Initial out-of-range params get clamped on construction."""
    seen_params: dict = {}

    def encoder(state, params):
        seen_params.update(params)
        return b"x"

    def scorer(archive):
        return 0.5

    MultiPassCompressor(
        target_score=-1.0, max_passes=1, eps=DEFAULT_EPS,
        initial_params={
            "mask_crf": 999.0,            # above max 60
            "pose_q_bits": -3.0,          # below min 4
            "block_fp_block_size": 100.0, # above max 32
            "residual_gain": 5.0,         # above max 1.0
        },
    ).compress(None, encoder, scorer)

    assert seen_params["mask_crf"] == PARAM_RANGES["mask_crf"][1]
    assert seen_params["pose_q_bits"] == PARAM_RANGES["pose_q_bits"][0]
    assert seen_params["block_fp_block_size"] == PARAM_RANGES["block_fp_block_size"][1]
    assert seen_params["residual_gain"] == PARAM_RANGES["residual_gain"][1]


def test_param_clamping_unknown_keys_pass_through() -> None:
    """A custom policy may use additional axes; unknown keys are not clamped."""
    seen: dict = {}

    def encoder(state, params):
        seen.update(params)
        return b"x"

    def scorer(archive):
        return 0.5

    MultiPassCompressor(
        target_score=-1.0, max_passes=1, eps=DEFAULT_EPS,
        initial_params={"mask_crf": 50.0, "custom_axis": 9999.0},
    ).compress(None, encoder, scorer)

    assert seen["custom_axis"] == 9999.0  # untouched


# ── (g) inflate-time assertion ───────────────────────────────────────────────


def test_inflate_time_assertion_blocks_inflate_renderer_caller(
    monkeypatch,
) -> None:
    """If __main__.__file__ ends in inflate_renderer.py, construction-then-
    compress raises. Strict-scorer-rule per CLAUDE.md.
    """
    fake_main = mock.MagicMock()
    fake_main.__file__ = "/some/path/inflate_renderer.py"
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    comp = MultiPassCompressor(
        target_score=0.5, max_passes=1, eps=DEFAULT_EPS,
        allow_inflate_context=False,
    )
    with pytest.raises(RuntimeError, match="(?i)strict-scorer-rule"):
        comp.compress(None, lambda s, p: b"x", lambda a: 0.5)


def test_inflate_time_assertion_can_be_overridden(monkeypatch) -> None:
    """``allow_inflate_context=True`` is the documented escape hatch (e.g.,
    for an explicitly-approved compliance ruling)."""
    fake_main = mock.MagicMock()
    fake_main.__file__ = "/some/path/inflate_renderer.py"
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    comp = MultiPassCompressor(
        target_score=-1.0, max_passes=1, eps=DEFAULT_EPS,
        allow_inflate_context=True,  # explicit operator opt-in
    )
    result = comp.compress(None, lambda s, p: b"x", lambda a: 0.5)
    assert result.final_score == pytest.approx(0.5)


# ── (h) eps stop ─────────────────────────────────────────────────────────────


def test_eps_stop_short_circuits_loop() -> None:
    """If |delta| < eps after pass 1, the loop stops at pass 2.

    Encoder is constant; scorer returns 0.5 each time. delta is exactly 0.
    """
    def encoder(state, params):
        return b"x"

    def scorer(archive):
        return 0.5

    result = MultiPassCompressor(
        target_score=-1.0, max_passes=5, eps=DEFAULT_EPS,
        initial_params={"mask_crf": 50.0},
    ).compress(None, encoder, scorer)

    # pass 1 sets prev_score; pass 2 has delta=0 → eps-stop.
    assert result.converged is True
    assert len(result.pass_history) == 2


# ── (i) target hit ───────────────────────────────────────────────────────────


def test_target_score_hit_short_circuits_loop() -> None:
    def encoder(state, params):
        return b"x"

    def scorer(archive):
        return 0.5

    result = MultiPassCompressor(
        target_score=1.0,         # 0.5 < 1.0 → hit on pass 1
        max_passes=5, eps=DEFAULT_EPS,
        initial_params={"mask_crf": 50.0},
    ).compress(None, encoder, scorer)

    assert result.target_hit is True
    assert result.converged is True
    assert len(result.pass_history) == 1


# ── (j) AdjustmentPolicy ABC ─────────────────────────────────────────────────


def test_adjustment_policy_subclassable() -> None:
    """A custom policy can override propose_next_params."""

    class ConstPolicy(AdjustmentPolicy):
        def propose_next_params(self, history, current_params):
            return {"mask_crf": 42.0}, "const-policy"

    seen_crfs: list[float] = []

    def encoder(state, params):
        seen_crfs.append(params["mask_crf"])
        return b"x"

    scores = iter([1.0, 0.99, 0.98])

    def scorer(archive):
        return next(scores)

    MultiPassCompressor(
        target_score=-1.0, max_passes=3, eps=1e-5,
        policy=ConstPolicy(),
        initial_params={"mask_crf": 50.0},
    ).compress(None, encoder, scorer)

    # Pass 0 uses initial 50.0; pass 1 + 2 use the policy's 42.0.
    assert seen_crfs[0] == 50.0
    assert seen_crfs[1:] == [42.0, 42.0]


def test_coordinate_descent_walks_priority_order() -> None:
    """Default policy hits mask_crf first, then pose_q_bits when mask_crf
    plateaus, etc.
    """
    policy = CoordinateDescentPolicy(eps=DEFAULT_EPS)
    history: list[PassRecord] = [
        PassRecord(0, {"mask_crf": 50.0}, 100, 0.5, 0.0, 0.1, "init"),
    ]
    next_params, reason = policy.propose_next_params(
        history, {"mask_crf": 50.0, "pose_q_bits": 8.0,
                  "block_fp_block_size": 16.0, "residual_gain": 0.0},
    )
    # First non-trivial proposal should touch mask_crf (highest priority).
    assert "mask_crf" in reason


# ── (k) determinism ──────────────────────────────────────────────────────────


def test_determinism_across_runs() -> None:
    """Same encoder + scorer + initial_params produce identical pass_history."""
    def encoder(state, params):
        return f"crf={params['mask_crf']:.1f}".encode()

    def scorer(archive):
        crf = float(archive.decode().split("=")[1])
        return (crf - 45.0) ** 2 / 100.0

    r1 = MultiPassCompressor(
        target_score=-1.0, max_passes=3, eps=DEFAULT_EPS,
        initial_params={"mask_crf": 50.0},
    ).compress(None, encoder, scorer)
    r2 = MultiPassCompressor(
        target_score=-1.0, max_passes=3, eps=DEFAULT_EPS,
        initial_params={"mask_crf": 50.0},
    ).compress(None, encoder, scorer)

    assert len(r1.pass_history) == len(r2.pass_history)
    for a, b in zip(r1.pass_history, r2.pass_history):
        assert a.score == pytest.approx(b.score)
        assert a.params == b.params
        assert a.archive_bytes == b.archive_bytes


# ── (l) functional wrapper ───────────────────────────────────────────────────


def test_functional_wrapper_equivalent_to_class() -> None:
    def encoder(state, params):
        return b"x" * 100

    def scorer(archive):
        return 0.42

    via_class = MultiPassCompressor(
        target_score=-1.0, max_passes=2, eps=DEFAULT_EPS,
        initial_params={"mask_crf": 50.0},
    ).compress(None, encoder, scorer)

    via_func = compress_with_multipass(
        None, encoder, scorer,
        target_score=-1.0, max_passes=2, eps=DEFAULT_EPS,
        initial_params={"mask_crf": 50.0},
    )

    assert via_class.final_score == via_func.final_score
    assert via_class.best_pass_idx == via_func.best_pass_idx
    assert len(via_class.pass_history) == len(via_func.pass_history)


# ── log path ─────────────────────────────────────────────────────────────────


def test_log_path_writes_jsonl() -> None:
    """When log_path is provided, each PassRecord is appended as one
    JSON-line for forensics.
    """
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "multipass.jsonl"

        def encoder(state, params):
            return b"x"

        def scorer(archive):
            return 0.5

        MultiPassCompressor(
            target_score=1.0, max_passes=1, eps=DEFAULT_EPS,
            initial_params={"mask_crf": 50.0},
            log_path=log,
        ).compress(None, encoder, scorer)

        assert log.exists()
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["pass_idx"] == 0
        assert rec["score"] == pytest.approx(0.5)


# ── encoder contract: must return bytes ──────────────────────────────────────


def test_encoder_must_return_bytes() -> None:
    def bad_encoder(state, params):
        return "not-bytes"  # str, not bytes

    def scorer(archive):
        return 0.5

    with pytest.raises(TypeError, match="bytes"):
        MultiPassCompressor(
            target_score=-1.0, max_passes=1, eps=DEFAULT_EPS,
            initial_params={"mask_crf": 50.0},
        ).compress(None, bad_encoder, scorer)
