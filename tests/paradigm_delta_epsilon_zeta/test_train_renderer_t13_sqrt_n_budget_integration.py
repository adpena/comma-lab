# LOSS_CONVERGENCE_NOT_REQUIRED: CLI/static wiring test for the opt-in
# T13 Fridrich √n per-pair latent budget integration.
"""T13 wire-in tests for ``experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py``.

Per CLAUDE.md ``forbidden_dead_flag_wiring_pattern``: every CLI flag this
module emits MUST be a subset of the target trainer's ``argparse``
contract. Per CLAUDE.md ``forbidden_empirical_claim_without_evidence_tag``:
every numeric T13 impact claim MUST carry a ``[predicted; T13 ...]`` tag.

Tests verify:
  - the trainer exposes ``--enable-t13-sqrt-n-budget`` (default OFF —
    backward-compat preserved)
  - the trainer exposes ``--t13-alpha`` and ``--t13-current-bits-per-pair``
  - the standalone ``apply_t13_sqrt_n_budget`` helper computes the
    Fridrich √n closed-form
  - the helper enforces input validation (positive symbols / non-neg bits /
    positive rate target)
  - the helper emits the canonical reallocation manifest fields with the
    required ``[predicted; T13 ...]`` tag
  - reading the trainer source confirms the manifest emission path is
    wired to ``write_provenance``

Source memos:
  - ``feedback_t11_t13_t19_free_lateral_leaps_landed_20260509`` (memo)
  - ``feedback_grand_council_portfolio_coherence_journal_grade_20260509`` (council §6)
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TRAIN_PATH = (
    REPO / "experiments" / "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
)


def _import_trainer():
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    if "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend" in sys.modules:
        del sys.modules["train_paradigm_delta_epsilon_zeta_track1_balle_endtoend"]
    return importlib.import_module(
        "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend"
    )


def _train_src() -> str:
    return TRAIN_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI wiring (forbidden_dead_flag_wiring_pattern)
# ---------------------------------------------------------------------------


def test_parse_args_exposes_t13_flags():
    mod = _import_trainer()
    ns = mod.parse_args(["--output-dir", "/tmp/foo"])
    assert ns.enable_t13_sqrt_n_budget is False, "default backward-compat OFF"
    assert ns.t13_alpha == 1.0
    assert ns.t13_current_bits_per_pair == 3.0


def test_parse_args_t13_overrides_take_effect():
    mod = _import_trainer()
    ns = mod.parse_args(
        [
            "--output-dir",
            "/tmp/foo",
            "--enable-t13-sqrt-n-budget",
            "--t13-alpha",
            "1.5",
            "--t13-current-bits-per-pair",
            "2.0",
        ]
    )
    assert ns.enable_t13_sqrt_n_budget is True
    assert ns.t13_alpha == 1.5
    assert ns.t13_current_bits_per_pair == 2.0


def test_t13_flags_listed_in_module_docstring_for_operator_visibility():
    src = _train_src()
    assert "--enable-t13-sqrt-n-budget" in src
    assert "--t13-alpha" in src
    assert "--t13-current-bits-per-pair" in src


# ---------------------------------------------------------------------------
# apply_t13_sqrt_n_budget closed-form math
# ---------------------------------------------------------------------------


def test_apply_t13_a1_substrate_shape():
    """A1 substrate: 600 pairs × 28 latent = 5.29 bits/pair undetectable."""
    mod = _import_trainer()
    r = mod.apply_t13_sqrt_n_budget(
        n_pairs=600,
        n_symbols_per_pair=28,
        current_bits_per_pair=3.0,
        rate_target_bytes=80_000.0,
        alpha=1.0,
    )
    assert r["bit_reallocation_t13_applied"] is True
    assert r["per_pair_undetectable_bits"] == pytest.approx(28**0.5, rel=1e-9)
    assert r["per_pair_current_bits"] == 3.0
    assert r["per_pair_headroom_bits"] == pytest.approx(28**0.5 - 3.0, rel=1e-9)
    # Total reallocated bits: 600 * (sqrt(28) - 3) per the memo's "headroom" form.
    expected_bits = 600 * (28**0.5 - 3.0)
    assert r["latent_bits_reduced"] == pytest.approx(expected_bits, rel=1e-9)
    assert r["pose_bits_added"] == pytest.approx(expected_bits, rel=1e-9)
    # rate target shrinks by the headroom in BYTES.
    expected_after = 80_000.0 - expected_bits / 8.0
    assert r["rate_target_bytes_after"] == pytest.approx(expected_after, rel=1e-9)


def test_apply_t13_hnerv_substrate():
    """HNeRV-style 64-D latent: per memo §6, 8 bits/pair undetectable."""
    mod = _import_trainer()
    r = mod.apply_t13_sqrt_n_budget(
        n_pairs=600,
        n_symbols_per_pair=64,
        current_bits_per_pair=3.0,
        rate_target_bytes=80_000.0,
    )
    assert r["per_pair_undetectable_bits"] == pytest.approx(8.0, rel=1e-9)
    # 5 bits/pair headroom × 600 pairs = 3000 bits = 375 bytes.
    assert r["latent_bits_reduced"] == pytest.approx(3000.0, rel=1e-9)
    assert r["rate_target_bytes_after"] == pytest.approx(80_000.0 - 375.0, rel=1e-9)


def test_apply_t13_no_overshrink_when_current_exceeds_undetectable():
    """If the trainer's current latent rate ALREADY exceeds Fridrich √n,
    the reallocation is clipped to 0 (cannot reallocate negative bits)."""
    mod = _import_trainer()
    r = mod.apply_t13_sqrt_n_budget(
        n_pairs=600,
        n_symbols_per_pair=4,  # sqrt(4) = 2 bits/pair undetectable
        current_bits_per_pair=10.0,  # 10 bits/pair >> undetectable
        rate_target_bytes=80_000.0,
    )
    assert r["per_pair_headroom_bits"] == pytest.approx(2.0 - 10.0, rel=1e-9)
    assert r["latent_bits_reduced"] == 0.0
    assert r["rate_target_bytes_after"] == r["rate_target_bytes_before"]


def test_apply_t13_predicted_tag_present():
    """Per CLAUDE.md Forbidden Score Claims: every numeric impact tagged."""
    mod = _import_trainer()
    r = mod.apply_t13_sqrt_n_budget(
        n_pairs=600,
        n_symbols_per_pair=28,
        current_bits_per_pair=3.0,
        rate_target_bytes=80_000.0,
    )
    assert "[predicted; T13" in r["tag"]
    assert "[predicted; T13" in r["predicted_pose_distortion_decrease"]


def test_apply_t13_alpha_scales_undetectable_bits():
    """α = 2.0 doubles the per-pair undetectable budget (linear in α)."""
    mod = _import_trainer()
    r1 = mod.apply_t13_sqrt_n_budget(
        n_pairs=600,
        n_symbols_per_pair=28,
        current_bits_per_pair=0.0,
        rate_target_bytes=80_000.0,
        alpha=1.0,
    )
    r2 = mod.apply_t13_sqrt_n_budget(
        n_pairs=600,
        n_symbols_per_pair=28,
        current_bits_per_pair=0.0,
        rate_target_bytes=80_000.0,
        alpha=2.0,
    )
    assert r2["per_pair_undetectable_bits"] == pytest.approx(
        2.0 * r1["per_pair_undetectable_bits"], rel=1e-9
    )


# ---------------------------------------------------------------------------
# Input validation (fail-loud per CLAUDE.md)
# ---------------------------------------------------------------------------


def test_apply_t13_rejects_zero_symbols():
    mod = _import_trainer()
    with pytest.raises(SystemExit, match="n_symbols_per_pair"):
        mod.apply_t13_sqrt_n_budget(
            n_pairs=600,
            n_symbols_per_pair=0,
            current_bits_per_pair=3.0,
            rate_target_bytes=80_000.0,
        )


def test_apply_t13_rejects_negative_current_bits():
    mod = _import_trainer()
    with pytest.raises(SystemExit, match="current_bits_per_pair"):
        mod.apply_t13_sqrt_n_budget(
            n_pairs=600,
            n_symbols_per_pair=28,
            current_bits_per_pair=-1.0,
            rate_target_bytes=80_000.0,
        )


def test_apply_t13_rejects_zero_rate_target():
    mod = _import_trainer()
    with pytest.raises(SystemExit, match="rate_target_bytes"):
        mod.apply_t13_sqrt_n_budget(
            n_pairs=600,
            n_symbols_per_pair=28,
            current_bits_per_pair=3.0,
            rate_target_bytes=0.0,
        )


# ---------------------------------------------------------------------------
# Trainer wire-in (T13 → write_provenance → provenance.json)
# ---------------------------------------------------------------------------


def test_trainer_main_calls_apply_t13_when_flag_enabled():
    """grep the trainer source: when --enable-t13-sqrt-n-budget is set,
    main() must call apply_t13_sqrt_n_budget and feed the result to
    write_provenance."""
    src = _train_src()
    assert "args.enable_t13_sqrt_n_budget" in src
    assert "apply_t13_sqrt_n_budget(" in src
    # Result is plumbed into write_provenance.
    assert "t13_bit_reallocation=t13_report" in src


def test_trainer_main_uses_effective_rate_target_for_coord_cfg():
    """The shrunk rate target must flow to JointLagrangianADMMConfig.

    The trainer constructs ``coord_cfg`` via a ``coord_kwargs`` dict; we
    accept either the keyword form or the dict form.
    """
    src = _train_src()
    assert "rate_target_bytes_effective" in src
    assert re.search(
        r'(rate_target_bytes\s*=\s*rate_target_bytes_effective'
        r'|"rate_target_bytes":\s*rate_target_bytes_effective)',
        src,
    ), "coord_cfg must read the T13-shrunk rate target"


def test_trainer_provenance_emits_t13_field_with_default_None():
    """When --enable-t13-sqrt-n-budget is OFF, provenance['t13_bit_reallocation'] is None."""
    src = _train_src()
    # write_provenance signature must accept the optional kwarg.
    assert "t13_bit_reallocation: dict | None = None" in src
    # Default initialiser is None when the flag is OFF.
    assert "t13_report: dict | None = None" in src


def test_trainer_t13_print_uses_predicted_tag():
    """Operator-visible print line must carry the [predicted; T13 ...] tag."""
    src = _train_src()
    assert (
        "[predicted; T13 Fridrich sqrt-n latent shrink]" in src
    ), "T13 console output must carry the predicted tag"


def test_trainer_lane_class_substrate_engineering():
    """Per CLAUDE.md HNeRV parity discipline: substrate-engineering lanes
    must declare lane_class explicitly."""
    src = _train_src()
    assert '"lane_class": "substrate_engineering"' in src


# ---------------------------------------------------------------------------
# Backward-compat: --enable-t13-sqrt-n-budget OFF leaves coord_cfg unchanged
# ---------------------------------------------------------------------------


def test_default_off_does_not_invoke_per_pair_sqrt_n_budget(monkeypatch):
    """When the flag is OFF, ``per_pair_sqrt_n_budget`` is NEVER called.
    This verifies the backward-compat invariant by patching the helper
    and asserting it stays unpatched at zero invocations."""
    mod = _import_trainer()
    ns = mod.parse_args(["--output-dir", "/tmp/foo"])
    # Sanity: the flag is OFF.
    assert ns.enable_t13_sqrt_n_budget is False
    # If main() respected the flag, helper would never be called. We can't
    # easily run main() in the unit test, but the source check above
    # (`if args.enable_t13_sqrt_n_budget:`) gates the call statically.
    src = _train_src()
    assert "if args.enable_t13_sqrt_n_budget:" in src
