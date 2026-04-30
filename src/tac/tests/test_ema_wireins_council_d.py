"""Tests for Council D EMA wire-ins (Check 88) — 2026-04-29 PM.

Council D audit at .omx/research/council_ema_audit_20260429.md identified 5
training scripts missing EMA + 1 buggy duplicate-class wiring. This test
suite asserts:

  1. The 5 wire-in target scripts now import EMA from tac.training, build
     an EMA(model, decay=…) instance with the Quantizr-canonical 0.997
     default (or thread it via --ema-decay), and call ema.update(...) in
     their training loops.
  2. The train_joint_pair.py duplicate `class EMA` was removed and the
     local 0.9995 default was promoted to 0.997 to match Quantizr.
  3. Preflight Check 88 (`check_training_paths_use_ema_correctly`) passes
     STRICT on the live codebase (0 violations).
  4. Negative test: a synthetic training script with optimizer.step() but
     no EMA is detected by Check 88.

Per CLAUDE.md "EMA — NON-NEGOTIABLE" + the audit's recommended fix list
(§7).
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

# Repo-relative paths so the tests run from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Per-script EMA presence assertions ───────────────────────────────────────


def _read_script(rel: str) -> str:
    p = REPO_ROOT / rel
    assert p.exists(), f"target script missing: {p}"
    return p.read_text()


def _assert_ema_wireins_present(rel: str, text: str) -> None:
    """Shared EMA-wirein assertions: EMA construction + update + decay."""
    assert "EMA(" in text, (
        f"{rel}: missing `EMA(...)` construction. Per CLAUDE.md \"EMA — "
        f"NON-NEGOTIABLE\", every training path must instantiate EMA."
    )
    assert "ema.update(" in text, (
        f"{rel}: missing `ema.update(...)` call. EMA must be updated "
        f"AFTER every optimizer.step()."
    )
    # Either a literal 0.997 default OR an --ema-decay CLI flag.
    has_literal = "decay=0.997" in text or "ema_decay: float = 0.997" in text or 'default=0.997' in text
    has_flag = "--ema-decay" in text or "ema_decay" in text
    assert has_literal or has_flag, (
        f"{rel}: needs either literal `decay=0.997` (Quantizr canonical) "
        f"or an `--ema-decay` CLI flag with sensible default."
    )


def test_train_szabolcs_has_ema() -> None:
    """experiments/train_szabolcs.py — Selfcomp clone, audit §3.4 priority #1."""
    text = _read_script("experiments/train_szabolcs.py")
    assert "from tac.training import EMA" in text, (
        "train_szabolcs.py must import EMA from tac.training (canonical class)."
    )
    _assert_ema_wireins_present("experiments/train_szabolcs.py", text)
    # And the saved best checkpoint must reference the EMA shadow.
    assert "ema.state_dict()" in text, (
        "train_szabolcs.py must save the EMA shadow as the inference "
        "checkpoint (CLAUDE.md non-negotiable)."
    )


def test_qat_finetune_has_ema() -> None:
    """experiments/qat_finetune.py — Quantizr full QAT pipeline (audit §3.2)."""
    text = _read_script("experiments/qat_finetune.py")
    _assert_ema_wireins_present("experiments/qat_finetune.py", text)
    # And the QATConfig must have ema_decay field.
    assert "ema_decay: float = 0.997" in text, (
        "qat_finetune.py QATConfig must expose ema_decay field defaulted "
        "to 0.997 (Quantizr canonical)."
    )
    # Best-state captured from EMA shadow.
    assert "raw_state = ema.state_dict()" in text, (
        "qat_finetune.py best_state must be captured from ema.state_dict() "
        "not model.state_dict() — inference bytes come from EMA shadow."
    )


def test_qat_omega_lagrangian_has_ema() -> None:
    """experiments/qat_omega_lagrangian.py — Lane Ω-V2 Lagrangian QAT."""
    text = _read_script("experiments/qat_omega_lagrangian.py")
    _assert_ema_wireins_present("experiments/qat_omega_lagrangian.py", text)


def test_quantize_distilled_has_ema() -> None:
    """experiments/quantize_distilled.py — post-training FP4 quantization."""
    text = _read_script("experiments/quantize_distilled.py")
    _assert_ema_wireins_present("experiments/quantize_distilled.py", text)


def test_train_imp_cycle_has_ema() -> None:
    """experiments/train_imp_cycle.py — IMP per-cycle fine-tune."""
    text = _read_script("experiments/train_imp_cycle.py")
    _assert_ema_wireins_present("experiments/train_imp_cycle.py", text)


def test_train_joint_pair_uses_canonical_ema() -> None:
    """experiments/train_joint_pair.py — duplicate `class EMA` REMOVED.

    Council D audit §3.6 bug 1: duplicate local class diverged from canonical
    (missing the float-buffer guard + late-bound module guard at
    src/tac/training.py L356-358 and L359-364).

    Council D audit §3.6 bug 2: default decay 0.9995 → 0.997 (Quantizr).
    """
    text = _read_script("experiments/train_joint_pair.py")
    # Bug 1: duplicate class is gone.
    assert "from tac.training import EMA" in text, (
        "train_joint_pair.py must import canonical EMA from tac.training."
    )
    # Should NOT contain a local `class EMA:` definition.
    # (The text "class EMA" might appear in a comment; restrict to the
    # exact `class EMA:` declaration form.)
    assert "class EMA:" not in text, (
        "train_joint_pair.py must NOT redefine class EMA locally — that "
        "diverges from canonical guards. Audit §3.6 bug 1."
    )
    # Bug 2: 0.9995 → 0.997.
    assert "ema_decay: float = 0.997" in text, (
        "train_joint_pair.py JointPairConfig.ema_decay must be 0.997 "
        "(Quantizr canonical). Audit §3.6 bug 2."
    )
    # Sanity: 0.9995 should not appear as the new default (a profile may
    # still sweep it, but the class default is 0.997).
    assert "ema_decay: float = 0.9995" not in text, (
        "train_joint_pair.py default decay must be 0.997, not 0.9995."
    )


# ── Check 88 preflight gate ──────────────────────────────────────────────────


def test_check_88_passes_strict_on_live_codebase() -> None:
    """Check 88 (training-needs-ema) MUST be clean on the current codebase.

    This is the gate that lets Check 88 ship STRICT in preflight_all().
    Will raise MetaBugViolation on any violation.
    """
    from tac.preflight import (
        MetaBugViolation,
        check_training_paths_use_ema_correctly,
    )
    try:
        violations = check_training_paths_use_ema_correctly(
            strict=True, verbose=False,
        )
    except MetaBugViolation as e:
        pytest.fail(
            f"Check 88 STRICT failed on live codebase. Council D EMA "
            f"audit wire-ins must be complete:\n{e}"
        )
    assert violations == [], (
        f"Check 88 reported {len(violations)} violations:\n  "
        + "\n  ".join(violations)
    )


def test_check_88_detects_missing_ema_in_synthetic_script(tmp_path: Path) -> None:
    """Negative test: a fresh training script with optimizer.step() but no
    EMA must be flagged by Check 88."""
    from tac.preflight import (
        MetaBugViolation,
        check_training_paths_use_ema_correctly,
    )
    # Build a synthetic repo with one offender.
    exp = tmp_path / "experiments"
    exp.mkdir()
    (exp / "train_synthetic_offender.py").write_text(
        '"""Synthetic training script with no EMA — should be flagged."""\n'
        'import torch\n'
        '\n'
        'def main() -> None:\n'
        '    model = torch.nn.Linear(10, 1)\n'
        '    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n'
        '    for _ in range(10):\n'
        '        optimizer.zero_grad()\n'
        '        loss = (model(torch.randn(4, 10)) ** 2).mean()\n'
        '        loss.backward()\n'
        '        optimizer.step()\n'
        '    torch.save(model.state_dict(), "renderer.pt")\n'
    )
    violations = check_training_paths_use_ema_correctly(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1, (
        f"expected exactly 1 violation, got {len(violations)}: {violations}"
    )
    assert "train_synthetic_offender.py" in violations[0]
    assert "EMA" in violations[0]


def test_check_88_respects_head_marker_waiver(tmp_path: Path) -> None:
    """A `# EMA_WAIVED:` head marker (within first 5 lines) must suppress
    the violation. Operators can use this to skip codec-calibration scripts."""
    from tac.preflight import check_training_paths_use_ema_correctly
    exp = tmp_path / "experiments"
    exp.mkdir()
    (exp / "train_waived_synthetic.py").write_text(
        '"""Synthetic — waived because not in submission path."""\n'
        '# EMA_WAIVED: codec-calibration script, no submission checkpoint\n'
        'import torch\n'
        '\n'
        'def main() -> None:\n'
        '    model = torch.nn.Linear(10, 1)\n'
        '    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n'
        '    optimizer.step()\n'
    )
    violations = check_training_paths_use_ema_correctly(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == [], (
        f"head-marker waiver must suppress violation, got: {violations}"
    )


def test_check_88_respects_exempt_basename(tmp_path: Path) -> None:
    """Scripts in `_EMA_EXEMPT_TRAINING_SCRIPTS` must be skipped automatically."""
    from tac.preflight import check_training_paths_use_ema_correctly
    exp = tmp_path / "experiments"
    exp.mkdir()
    # train_mini_scorer.py is in the exempt list per audit §3.5.
    (exp / "train_mini_scorer.py").write_text(
        '"""Synthetic mini scorer — exempt by basename allowlist."""\n'
        'import torch\n'
        '\n'
        'def main() -> None:\n'
        '    model = torch.nn.Linear(10, 1)\n'
        '    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n'
        '    optimizer.step()\n'
    )
    violations = check_training_paths_use_ema_correctly(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == [], (
        f"exempt basename must skip the script, got: {violations}"
    )


# ── EMA class behavior smoke tests (audit §A) ────────────────────────────────


def test_canonical_ema_decay_default_is_quantizr() -> None:
    """`tac.training.EMA` default decay must be 0.997 (Quantizr canonical).

    CLAUDE.md "EMA — NON-NEGOTIABLE" pins this default. If anyone changes it
    away from 0.997, this test trips and forces a council audit.
    """
    from tac.training import EMA
    sig = inspect.signature(EMA.__init__)
    decay_default = sig.parameters["decay"].default
    assert decay_default == 0.997, (
        f"EMA default decay must be 0.997 (Quantizr canonical), got "
        f"{decay_default}. CLAUDE.md \"EMA — NON-NEGOTIABLE\"."
    )


def test_ema_does_not_mutate_live_model_on_update() -> None:
    """Antipattern guard: ema.update(model) writes to ema.shadow ONLY, never
    to the live model. The DARTS-S freeze hypothesis (audit §6) was
    that EMA back-shadowed the live model — this test rules that out.
    """
    import torch
    from tac.training import EMA
    model = torch.nn.Linear(4, 4)
    pre_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema = EMA(model, decay=0.997)
    # Pretend an optimizer.step() bumped weights.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
    ema.update(model)
    # Live model still has the bumped weights — EMA.update did NOT
    # squash them back to the shadow.
    for k, v in model.state_dict().items():
        delta = (v - pre_state[k]).abs().mean().item()
        if v.is_floating_point():
            assert delta > 0.4, (
                f"EMA.update() must not mutate the live model. {k} delta={delta}"
            )
