# SPDX-License-Identifier: MIT
"""ddm_dt1 (#903) — regression tests for the R-operator determinism floor + its cure.

MEASURED 2026-08-03 (memo ``.omx/research/ddm_dt1_determinism_floor_20260803.md``):
the TR1 trainer was NOT run-to-run bit-reproducible on MLX-GPU — 40 of 41 checkpoint
arrays differed between two runs at identical seed/config/inputs. Bisected to the
**upsample VJP** inside the R operator (a scatter whose GPU accumulation order varies;
the downsample VJP is clean, and MLX-CPU is clean). ~1 ULP in the gradient, amplified to
a full parameter step because Adam's first update is essentially ``sign(g)``.

These tests pin the three things that must not silently regress:

  1. the fused Metal R VJP is bit-identical across repeats (the cure actually cures);
  2. the fused R FORWARD is bit-identical to the reference (the cure does not silently
     change the vehicle / d_seg);
  3. ``--deterministic-r`` is NOT an inert flag — parsing it and running the trainer's
     enable path actually flips ``fused_r_kernel_enabled()`` (the config-orphan class),
     and the flag defaults OFF so an absent flag is byte-identical to every prior run.

GPU-gated: (1) and (2) need a Metal device and SKIP elsewhere. (3) is pure wiring and
runs everywhere — so the anti-inert-flag guard is never vacuous on CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_RENDER_HW = (384, 512)
REAL_CAMERA_HW = (874, 1164)


def _mx():
    return pytest.importorskip("mlx.core")


def _metal_available() -> bool:
    try:
        mx = importlib.import_module("mlx.core")
        from tac.local_acceleration.metal_fused_r_operator import metal_fused_r_available
    except Exception:  # noqa: BLE001 - absent MLX/Metal is a skip, not a failure
        return False
    mx.set_default_device(mx.gpu)
    return bool(metal_fused_r_available())


requires_metal = pytest.mark.skipif(
    not _metal_available(), reason="fused Metal R kernel requires a Metal GPU default device")


@requires_metal
def test_fused_r_vjp_is_bit_identical_across_repeats_at_real_geometry():
    """The cure cures: repeated VJPs of the fused R agree to the BIT (max|delta| == 0).

    Runs at the REAL trainer geometry (384x512 render -> 874x1164 camera -> 384x512
    scorer), not a toy one -- a determinism guarantee proven only at (48,64) would not
    cover the shape the trainer actually dispatches.
    """
    from tac.local_acceleration.metal_fused_r_operator import assert_metal_vjp_deterministic

    result = assert_metal_vjp_deterministic(
        in_hw=REAL_RENDER_HW, camera_hw=REAL_CAMERA_HW, output_hw=REAL_RENDER_HW,
        batch=2, repeats=4)
    assert result["deterministic_bit_identical"] is True
    assert result["max_run_to_run_abs_delta"] == 0.0
    assert result["repeats"] == 4  # denominator, never a bare pass


@requires_metal
def test_default_r_backward_is_the_nondeterministic_one_and_fused_is_not():
    """POSITIVE CONTROL for this test module.

    A determinism test that has never been shown to FIRE is untrusted, so this asserts
    the contrast rather than only the good case: with the fused kernel OFF, two identical
    ``mx.grad`` calls on the R op disagree; with it ON, they agree bitwise. If the default
    path ever becomes deterministic upstream (an MLX fix), THIS test fails loudly and the
    memo's mechanism claim must be re-derived rather than silently inherited.
    """
    mx = _mx()
    mx.set_default_device(mx.gpu)
    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np

    import experiments.train_witness_realized_through_R_mlx as T

    rng = np.random.default_rng(11)
    x = mx.array((rng.random((2, *REAL_RENDER_HW, 3)) * 255.0).astype(np.float32))

    def grad_once():
        g = mx.grad(lambda z: mx.mean(T._apply_R(z) ** 2))(x)
        mx.eval(g)
        return np.array(g, copy=True)

    was = T.fused_r_kernel_enabled()
    try:
        T.set_fused_r_kernel(False)
        ref_fwd = np.array(T._apply_R(x))
        a, b = grad_once(), grad_once()
        default_repeats_agree = a.tobytes() == b.tobytes()

        T.set_fused_r_kernel(True)
        fused_fwd = np.array(T._apply_R(x))
        c, d = grad_once(), grad_once()
        fused_repeats_agree = c.tobytes() == d.tobytes()
    finally:
        T.set_fused_r_kernel(was)

    assert fused_repeats_agree, "fused R VJP must be bit-identical across repeats"
    assert not default_repeats_agree, (
        "the default mx.vjp scatter backward was expected to be NON-deterministic "
        "(the #903 mechanism). It agreed -- re-derive the memo's mechanism claim.")
    # The cure must not move the FORWARD: d_seg is read off the forward, so a forward
    # change would silently make every historical realized-d_seg row a different vehicle.
    assert ref_fwd.tobytes() == fused_fwd.tobytes(), (
        "fused R FORWARD must be bit-identical to the reference forward")


def test_deterministic_r_flag_exists_defaults_off_and_is_not_inert():
    """Anti-config-orphan guard: the flag parses, defaults OFF, and reaches a real switch.

    Runs on every platform (no GPU needed) so this guard is never vacuous. It checks
    three distinct failure modes: flag absent from argparse; flag present but defaulting
    ON (which would silently change every run); and flag present but wired to nothing
    (the inert-flag class -- ``set_fused_r_kernel`` must exist and actually toggle
    ``fused_r_kernel_enabled``).
    """
    sys.path.insert(0, str(REPO_ROOT))
    pytest.importorskip("mlx.core")
    from experiments.train_tr1_partition_renderer_mlx import build_argparser
    from experiments.train_witness_realized_through_R_mlx import (
        fused_r_kernel_enabled,
        set_fused_r_kernel,
    )

    ap = build_argparser()
    dests = {a.dest for a in ap._actions}
    assert "deterministic_r" in dests, "--deterministic-r must exist on the TR1 argparser"

    base = ["--variant", "lotto", "--out-dir", "/dev/null"]
    assert ap.parse_args(base).deterministic_r is False, "flag must DEFAULT OFF"
    assert ap.parse_args([*base, "--deterministic-r"]).deterministic_r is True

    was = fused_r_kernel_enabled()
    try:
        set_fused_r_kernel(True)
        assert fused_r_kernel_enabled() is True, "the switch the flag drives is inert"
        set_fused_r_kernel(False)
        assert fused_r_kernel_enabled() is False
    finally:
        set_fused_r_kernel(was)


def test_deterministic_r_flag_is_actually_wired_into_main():
    """The gap the previous test does NOT close: flag parses + switch works, but does
    ``main()`` still CALL the switch?

    Found by adversarial self-review of this very module: every assertion above would
    still pass if the ``if args.deterministic_r:`` block were deleted from ``main`` --
    which is precisely the config-orphan / inert-flag failure class. So this walks the
    trainer's AST, finds the branch guarded by ``args.deterministic_r``, and asserts
    ``set_fused_r_kernel`` is invoked INSIDE it. Structural, not a substring grep, so a
    call sitting in a comment or in an unrelated function cannot satisfy it.
    """
    import ast

    src = (REPO_ROOT / "experiments" / "train_tr1_partition_renderer_mlx.py").read_text()
    tree = ast.parse(src)

    def guards_on_deterministic_r(node: ast.expr) -> bool:
        return any(isinstance(n, ast.Attribute) and n.attr == "deterministic_r"
                   for n in ast.walk(node))

    branches = [n for n in ast.walk(tree)
                if isinstance(n, ast.If) and guards_on_deterministic_r(n.test)]
    assert branches, "no `if args.deterministic_r:` branch found in the trainer"

    called = set()
    for br in branches:
        for stmt in br.body:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    called.add(n.func.id)
    assert "set_fused_r_kernel" in called, (
        "the --deterministic-r branch does not call set_fused_r_kernel -- the flag is "
        f"ORPHANED (calls found in the branch: {sorted(called)})")
    assert "metal_fused_r_available" in called, (
        "the --deterministic-r branch must fail-closed on a non-Metal device rather than "
        "silently running the non-deterministic scatter backward")


def test_comparison_harness_positive_control_passes():
    """The comparator that produced the #903 verdict must fire on a known difference.

    Guards the instrument itself: identical -> IDENTICAL, 1-ULP -> DIFFER, key-asymmetry
    -> ASYMMETRIC, empty scope -> VACUOUS (never a bare pass). If this regresses, no
    determinism verdict from that tool is admissible.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from ddm_dt1_compare_run_determinism import self_check

    ok, lines = self_check()
    assert ok, "comparator positive control FAILED:\n" + "\n".join(lines)
    assert len(lines) == 5, f"expected 5 control fixtures, got {len(lines)}"
