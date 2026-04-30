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

import ast
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


# ── Round 7 Defect #4: AST-level ordering + decay assertions ────────────────
#
# Council Round 7 §6.4 noted the existing wire-in tests are TEXT-grep
# checks (inspect.getsource(m) + substring match for "EMA" / "ema.update" /
# "decay=0.997"). A subagent could add `# EMA = "Exponential Moving Average"`
# as a comment + the tests pass vacuously. Strengthen by walking the AST:
#
#   - Test #1 verifies ema.update(model) is called AFTER optimizer.step()
#     in source order, NOT before (a future operator inserting
#     ema.update(model) before optimizer.step would not be caught by the
#     text-grep tests). Skips train_lora_tto.py (LoRA may have a different
#     ordering pattern that the AST scanner would mis-flag).
#   - Test #2 verifies the EMA(model, decay=...) constructor is called
#     with a decay value that is exactly 0.997 OR a config attribute named
#     ``ema_decay`` (the canonical Quantizr default and the config-driven
#     pattern). Catches a future regression where someone hard-codes
#     decay=0.999 / 0.999.


# Scripts that Check 88 wires in. Skip _lora_tto until LoRA ordering audit.
_AST_TEST_TARGETS = [
    "experiments/train_szabolcs.py",
    "experiments/qat_finetune.py",
    "experiments/qat_omega_lagrangian.py",
    "experiments/quantize_distilled.py",
    "experiments/train_imp_cycle.py",
    "experiments/train_postfilter_on_renderer.py",
    "experiments/train_joint_pair.py",
]
# train_lora_tto.py uses LoRA-specific ema timing (apply at end-of-epoch
# vs per-step). Skip for AST ordering assertion until a LoRA-aware audit
# lands.
_AST_TEST_SKIP = {
    "experiments/train_lora_tto.py",
}


def _find_optimizer_step_then_ema_update_order(
    tree: "ast.Module",
) -> tuple[bool, str]:
    """AST walk: for each function body containing both `optimizer.step()`
    (or `*.step()` with a recognised receiver name) and `ema.update(...)`,
    verify the step call appears BEFORE the update call in source line
    order.

    Returns (ok, message). ok=True if every function body honours the
    ordering OR if no function body contains both. ok=False otherwise.
    """
    import ast

    OPT_NAMES = {"optimizer", "optim", "opt", "_opt"}

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Find all .step() and ema.update() calls inside this function.
        step_lines: list[int] = []
        ema_update_lines: list[int] = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "step":
                v = node.func.value
                if isinstance(v, ast.Name) and v.id in OPT_NAMES:
                    step_lines.append(node.lineno)
                elif isinstance(v, ast.Attribute) and v.attr in OPT_NAMES:
                    step_lines.append(node.lineno)
            elif node.func.attr == "update":
                v = node.func.value
                # Match `ema.update(...)` or `self.ema.update(...)`.
                if isinstance(v, ast.Name) and v.id == "ema":
                    ema_update_lines.append(node.lineno)
                elif isinstance(v, ast.Attribute) and v.attr == "ema":
                    ema_update_lines.append(node.lineno)
        if step_lines and ema_update_lines:
            # The CANONICAL pattern: every ema.update should appear AFTER
            # at least one optimizer.step that itself appears before it.
            # Equivalently: max(step_lines) > min(ema_update_lines) would
            # be a violation. We assert: there exists at least one step
            # call before the FIRST ema.update.
            first_update = min(ema_update_lines)
            steps_before_first_update = [s for s in step_lines if s < first_update]
            if not steps_before_first_update:
                return False, (
                    f"function {fn.name} (line {fn.lineno}): first ema.update "
                    f"at line {first_update} appears BEFORE any "
                    f"optimizer.step (step lines: {step_lines}). EMA "
                    f"update must come AFTER optimizer.step."
                )
    return True, ""


def test_ema_update_called_after_optimizer_step_via_ast() -> None:
    """Round 7 Defect #4 strengthening: AST-verify EMA ordering.

    For every Check 88 target script, walk every function body. If both
    optimizer.step() and ema.update(...) exist, assert the FIRST ema.update
    line is after at least one optimizer.step line.
    """
    import ast

    failures: list[str] = []
    for rel in _AST_TEST_TARGETS:
        if rel in _AST_TEST_SKIP:
            continue
        text = _read_script(rel)
        tree = ast.parse(text, filename=rel)
        ok, msg = _find_optimizer_step_then_ema_update_order(tree)
        if not ok:
            failures.append(f"{rel}: {msg}")
    assert not failures, (
        f"EMA ordering violations (must be optimizer.step → ema.update):\n  "
        + "\n  ".join(failures)
    )


def _find_ema_constructor_decay_arg(tree: "ast.Module") -> list[tuple[int, str]]:
    """Walk AST for `EMA(model, decay=<X>)` constructor calls. Return list
    of (lineno, decay_repr) for each EMA(...) call, where decay_repr is
    either:
      - "0.997" / "0.9995" / etc. (numeric literal value as str)
      - "<attr>.<name>"  (attribute access — typically `cfg.ema_decay`,
                          `args.ema_decay`)
      - "<name>"         (bare Name reference — `decay`)
      - "<missing>"      (decay kwarg not present — uses default)
      - "<unknown>"      (some other expression form)
    """
    import ast

    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # EMA(...) — Name form
        if isinstance(node.func, ast.Name) and node.func.id == "EMA":
            decay_kw = next(
                (kw for kw in node.keywords if kw.arg == "decay"), None
            )
            out.append((node.lineno, _decay_repr(decay_kw)))
        # tac.training.EMA(...) — Attribute form
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "EMA"
        ):
            decay_kw = next(
                (kw for kw in node.keywords if kw.arg == "decay"), None
            )
            out.append((node.lineno, _decay_repr(decay_kw)))
    return out


def _decay_repr(decay_kw) -> str:
    """Reduce an AST keyword node to a string we can pattern-match against."""
    import ast

    if decay_kw is None:
        return "<missing>"
    v = decay_kw.value
    return _decay_value_repr(v)


def _decay_value_repr(v) -> str:
    """Helper: render an AST value node to a comparison-friendly string.
    Handles Constant, Attribute, Name, and Call(arg) (e.g. float(args.ema_decay)).
    """
    import ast

    # Constant float / int
    if isinstance(v, ast.Constant):
        return repr(v.value)
    # Attribute access: cfg.ema_decay / args.ema_decay
    if isinstance(v, ast.Attribute):
        receiver = "?"
        if isinstance(v.value, ast.Name):
            receiver = v.value.id
        elif isinstance(v.value, ast.Attribute):
            receiver = "<attr>"
        return f"{receiver}.{v.attr}"
    # Bare Name: decay
    if isinstance(v, ast.Name):
        return v.id
    # Call wrappers: float(args.ema_decay), float(cfg.ema_decay), etc.
    # Recurse into the SOLE positional argument.
    if isinstance(v, ast.Call) and len(v.args) == 1 and not v.keywords:
        return _decay_value_repr(v.args[0])
    return "<unknown>"


def test_ema_decay_is_quantizr_canonical_via_ast() -> None:
    """Round 7 Defect #4 strengthening: AST-verify EMA decay value.

    For every Check 88 target script, walk every EMA(model, decay=...)
    constructor call. The decay value MUST be either:
      - the literal 0.997 (Quantizr canonical), OR
      - an attribute access ending in .ema_decay (config-driven, where
        the config default is asserted by test_canonical_ema_decay_default_is_quantizr).

    Forbids: literal != 0.997, opaque expressions, missing decay kwarg.
    """
    failures: list[str] = []
    import ast

    for rel in _AST_TEST_TARGETS:
        text = _read_script(rel)
        tree = ast.parse(text, filename=rel)
        for lineno, repr_ in _find_ema_constructor_decay_arg(tree):
            ok = False
            # Literal 0.997.
            if repr_ == "0.997":
                ok = True
            # Attribute access ending in .ema_decay.
            elif repr_.endswith(".ema_decay"):
                ok = True
            if not ok:
                failures.append(
                    f"{rel}:{lineno}: EMA constructor decay={repr_} is not "
                    f"the Quantizr canonical 0.997 nor a config-attribute "
                    f"named .ema_decay."
                )
    assert not failures, (
        f"EMA decay constructor violations:\n  " + "\n  ".join(failures)
    )
