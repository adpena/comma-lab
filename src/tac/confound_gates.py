"""Confound-immune-system L2 STRICT preflight gates (Catalog #397-#402).

Source: ``.omx/research/confound_hunt_synthesis_20260705.md`` — the fresh-eyes
adversarial confound hunt on the level-set witness trainer + measurement
apparatus (operator: "that confound is poison"). Six independent hunters found
18 confounds (C1-C18) sharing the signature
**DEFAULT-HARMFUL x SILENT x MEASUREMENT-CORRUPTING**. The synthesis prescribes a
3-layer immune system; this module is **Layer 2 — STRICT preflight gates that
refuse the CODE anti-pattern** (Layer 1 = runtime alarms in the trainer, owned by
the trainer-fixer sibling; Layer 3 = the CLAUDE.md verdict-clearance
non-negotiable).

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against": every
adversarial-review finding gets TWO landings — the fix (sibling trainer/launcher
commits) AND a STRICT preflight check that refuses re-introduction of the bug
class. This module is the second landing for the confound family.

The original six gates landed **WARN-ONLY** per the CLAUDE.md "Strict-flip
atomicity rule": their sibling fixes were not in the same commit. The dated
2026-07-15 follow-on gates at the end of this module land atomically with their
fixes at live-count zero and are therefore wired STRICT in
``tac.preflight.preflight_all``.

Each gate:
  * scans the repo for the anti-pattern signature,
  * allows a same-line (or in-scope) ``# <CLASS>_OK:<rationale>`` waiver with a
    NON-placeholder rationale (Catalog #287 sister discipline — the docstring
    example ``<rationale>`` / ``<reason>`` cannot self-waive),
  * raises ``PreflightError`` in strict mode, warns otherwise,
  * returns ``list[str]`` of violation strings.

Catalog map:
  #397 check_no_spike_guard_defaults_to_deadlock_mode              (C1)
  #398 check_reject_filter_updates_reference_from_accepted_only_has_rearm (C1/C17 structural)
  #399 check_no_duplicate_long_flags_in_launch                     (C13)
  #400 check_resume_palliative_flags_imply_warm_start              (C8)
  #401 check_verdict_pairs_default_is_n600                         (C12)
  #402 check_telemetry_verdict_rows_carry_liveness                 (C6)
  #403 check_launch_config_authored_in_dsl                         (req V, #353)
  2026-07-14 check_codex_retry_preserves_original_sandbox_authority
  2026-07-14 check_codex_nonisolated_writer_cap
  2026-07-14 check_codex_drain_timeout_uses_liveness
  2026-07-15 check_consolidation_debt_monitor_observability_and_cadence
  2026-07-15 check_witness_trainers_emit_partial_freeze_alarm
  2026-07-15 check_witness_verdict_rows_carry_dseg_descent_canary
  2026-07-15 check_verdict_live_gap_defaults_on_during_ema_warmup
  2026-07-18 check_no_duplicate_canonical_spec_across_refs (NAME-ANCHORED-SEARCH
             / duplicate-SoT class; sister tool tools/canonical_doc_registry.py)
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tac.witness_dsl.curriculum_dsl import TRAINER_REL

# Repo root: src/tac/confound_gates.py -> parents[2] == repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Canonical witness trainer surfaces the confound hunt covered. Both are scanned
# by the source-level gates; only those that exist on disk are opened.
_TRAINER_FILES = (
    TRAINER_REL,  # canonical single-source: curriculum_dsl.TRAINER_REL (levelset entry point)
    "experiments/train_witness_realized_through_R_mlx.py",  # base trainer (no canonical constant yet)
)

# Placeholder rationale rejection (Catalog #287 sister discipline): a waiver whose
# rationale is a bare ``<reason>`` / ``<rationale>`` (the docstring example) does
# NOT self-waive.
_PLACEHOLDER_RATIONALE_RE = re.compile(
    r"^<\s*(?:reason|rationale)\s*>$", re.IGNORECASE
)


def _rationale_ok(rationale: str) -> bool:
    """True iff ``rationale`` is a real, non-placeholder string (>=3 chars)."""
    r = rationale.strip()
    if len(r) < 3:
        return False
    if _PLACEHOLDER_RATIONALE_RE.match(r):
        return False
    return True


def _waiver_present(text: str, marker: str) -> bool:
    """True iff ``# <marker>:<non-placeholder rationale>`` appears in ``text``."""
    rx = re.compile(r"#[ \t]*" + re.escape(marker) + r":[ \t]*(\S.*)")
    for m in rx.finditer(text):
        if _rationale_ok(m.group(1)):
            return True
    return False


def _existing_trainers(root: Path) -> list[Path]:
    out: list[Path] = []
    for rel in _TRAINER_FILES:
        p = root / rel
        if p.is_file():
            out.append(p)
    return out


def _launch_files(root: Path) -> list[Path]:
    base = root / "experiments" / "results"
    if not base.is_dir():
        return []
    return sorted(base.glob("**/launch.sh"))


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _strip_comments(text: str) -> str:
    """Drop each line's ``#...`` tail (heuristic — no in-string ``#`` awareness).

    Used so a WAIVER-MARKER comment (e.g. ``# REJECT_FILTER_REARM_OK:...`` whose
    name contains "rearm") cannot masquerade as a real code re-arm token."""
    out = []
    for ln in text.splitlines():
        i = ln.find("#")
        out.append(ln if i < 0 else ln[:i])
    return "\n".join(out)


def _finish(
    *,
    name: str,
    tag: str,
    violations: list[str],
    strict: bool,
    verbose: bool,
    ok_detail: str,
) -> list[str]:
    """Raise in strict / warn in verbose. Mirrors the canonical preflight gate
    epilogue so callers behave identically to ``tac.preflight.check_*``."""
    if violations and strict:
        # Lazy import avoids the confound_gates<->preflight circular import.
        from tac.preflight import PreflightError

        msg = (
            f"{name}: {len(violations)} violation(s):\n  "
            + "\n  ".join(violations[:5])
        )
        raise PreflightError(msg)
    if verbose:
        if violations:
            print(f"  [{tag}] WARN: {len(violations)} violation(s) (strict={strict})")
        else:
            print(f"  [{tag}] OK ({ok_detail})")
    return violations


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _call_is_add_argument(node: ast.Call) -> bool:
    fn = node.func
    return isinstance(fn, ast.Attribute) and fn.attr == "add_argument"


def _first_str_positional(node: ast.Call) -> str | None:
    for a in node.args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return a.value
    return None


def _kw_constant(node: ast.Call, name: str):
    for kw in node.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return _MISSING


class _MISSING:  # sentinel
    pass


def _span_source(lines: list[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None) or start
    if not start:
        return ""
    return "\n".join(lines[start - 1 : end])


def _func_defs(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(node)
    return out


# ===========================================================================
# Catalog #397 — C1: spike/median guard actuator default must not be a
# documented deadlock/degrade mode (the ``--spike-guard-mode default="legacy"``
# class that froze BOTH the v5 and v6 n600 runs).
# ===========================================================================

# Mode-string defaults known to be deadlock/degrade modes for a spike/median
# guard actuator. ``legacy`` == skip-with-frozen-median (the absorbing-median
# freeze). ``skip``/``freeze``/``deadlock``/``degrade``/``off`` are conservative
# additions so a rename cannot dodge the gate.
_DEADLOCK_MODE_DEFAULTS = frozenset(
    {"legacy", "skip", "freeze", "frozen", "deadlock", "degrade", "off"}
)
_SPIKE_GUARD_FLAG_RE = re.compile(r"^--.*(?:spike|median).*guard.*mode$|^--spike-guard-mode$")


def check_no_spike_guard_defaults_to_deadlock_mode(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catalog #397 (C1) — a spike/median-guard mode actuator's argparse
    ``default=`` must not be a documented deadlock/degrade mode.

    Confound C1 (``.omx/research/confound_hunt_synthesis_20260705.md``): the
    ``--spike-guard-mode`` argparse ``default="legacy"`` (levelset trainer:6433).
    ``legacy`` == skip-with-frozen-median; on a sustained spike the running median
    freezes (it is appended only from accepted batches — see Catalog #398) and the
    guard skips 100% of batches -> the optimizer freezes. BOTH the v5 and v6 n600
    runs shipped ``legacy`` (neither passed the built-but-never-defaulted
    ``rollback`` cure) and froze at ep114/ep103 with a pinned eikonal artifact.
    The "viscosity NO-GO" verdict this session rests on that frozen state ->
    poisoned.

    Signature refused: ``add_argument("--spike-guard-mode", ..., default=<m>)``
    (or any flag matching ``--*guard*mode``) where ``<m>`` is a deadlock/degrade
    mode string (``legacy``/``skip``/``freeze``/...).

    Same-line / in-call waiver: ``# SPIKE_GUARD_DEFAULT_OK:<rationale>`` (a real
    rationale — e.g. a byte-identity A/B baseline that autoconfig overrides).

    STRICT-FLIP: DONE (2026-07-06). The levelset trainer now defaults
    ``--spike-guard-mode="rollback"`` (train_levelset...:7003), so live-count is 0
    -> ``preflight_all`` now calls this gate ``strict=True`` per the Strict-flip
    atomicity rule (the function default stays ``strict=False`` so direct/test calls
    remain warn-only).
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    for path in _existing_trainers(root):
        text = _read(path)
        if not text:
            continue
        scanned += 1
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _call_is_add_argument(node):
                continue
            flag = _first_str_positional(node)
            if not flag or not _SPIKE_GUARD_FLAG_RE.match(flag):
                continue
            default = _kw_constant(node, "default")
            if default is _MISSING or not isinstance(default, str):
                continue
            if default.lower() not in _DEADLOCK_MODE_DEFAULTS:
                continue
            if _waiver_present(_span_source(lines, node), "SPIKE_GUARD_DEFAULT_OK"):
                continue
            violations.append(
                f"{rel}:{node.lineno}: spike-guard actuator {flag!r} defaults to "
                f"deadlock/degrade mode {default!r} (C1 absorbing-median freeze). "
                f"Flip default -> a non-deadlock cure mode (e.g. 'rollback'), or add "
                f"a `# SPIKE_GUARD_DEFAULT_OK:<rationale>` waiver."
            )
    return _finish(
        name="check_no_spike_guard_defaults_to_deadlock_mode",
        tag="spike-guard-default-deadlock",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{scanned} trainer(s) scanned",
    )


# ===========================================================================
# Catalog #398 — C1/C17 STRUCTURAL (highest-value): a running-median/quantile
# reject filter whose reference window is appended ONLY inside the
# accepted/non-skip branch MUST have a re-arm escape hatch in the same function.
# ===========================================================================

# Names that identify a running-median / quantile reference window.
_REFERENCE_WINDOW_NAME_TOKENS = ("recent", "median_window", "ref_window", "refwindow")
# Tokens in an ``if`` TEST that mark it as the spike/skip discriminator.
_SKIP_TEST_TOKENS = ("spike", "spiked", "skip", "nonfinite", "non_finite")
# Re-arm escape-hatch tokens, split by SPECIFICITY to shrink the false-negative (MISS) surface:
#  * SPECIFIC tokens are unambiguous re-arm intent (a median re-anchor, a spike-guard rollback mode,
#    an explicit clear/decay of the reference window) -> any presence in the enclosing function clears
#    the gate.
#  * GENERIC tokens (``.clear(`` / ``reset(`` / ``rollback``) are ambiguous -- an unrelated
#    ``some_other_list.clear()`` / ``optimizer.reset()`` / a ``rollback`` mentioned far away would
#    FALSELY clear the gate (a miss of the real absorbing-median deadlock). They count as a re-arm ONLY
#    when they appear NEAR an accepted-only append (same block / within ``_REARM_PROXIMITY_LINES``).
_REARM_TOKENS_SPECIFIC = (
    "rearm",
    "re-arm",
    "re_arm",
    "reanchor",
    "re-anchor",
    "re_anchor",
    "reset_median",
    "spike_guard_mode",
    "resume_clear_spike_guard",
    "clear_spike_guard",
    "quantile_decay",
    "quantile-decay",
    "decay_median",
    "re_arm_median",
)
_REARM_TOKENS_GENERIC = (".clear(", "reset(", "rollback")
# How close (lines, either side) a GENERIC re-arm token must sit to an accepted-only append to count.
_REARM_PROXIMITY_LINES = 10
# Full union preserved for callers/tests that enumerate every recognised re-arm token.
_REARM_TOKENS = _REARM_TOKENS_SPECIFIC + _REARM_TOKENS_GENERIC


def _name_is_reference_window(target: ast.AST) -> bool:
    """True iff ``target`` is a Name/Attribute whose identifier contains a
    reference-window token."""
    ident = ""
    if isinstance(target, ast.Name):
        ident = target.id
    elif isinstance(target, ast.Attribute):
        ident = target.attr
    ident = ident.lower()
    return any(tok in ident for tok in _REFERENCE_WINDOW_NAME_TOKENS)


def _append_calls_to_reference_window(node: ast.AST) -> list[ast.Call]:
    """All ``<ref>.append(...)`` calls under ``node`` whose receiver is a
    reference-window name."""
    out: list[ast.Call] = []
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "append"
            and _name_is_reference_window(n.func.value)
        ):
            out.append(n)
    return out


def check_reject_filter_updates_reference_from_accepted_only_has_rearm(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catalog #398 (C1/C17 STRUCTURAL) — the generalized signature of the whole
    confound: a running-median/quantile reject filter whose reference window is
    appended ONLY inside the accepted/non-skip branch must carry a re-arm escape
    hatch (clear / re-anchor / rollback / quantile-decay) in the SAME function.

    Confound C1 root + C17 sibling: the spike guard skips a batch when
    ``batch_loss > k * median``, and the ``median`` is computed over a window
    (``recent`` / ``recent_losses``) that is appended to ONLY from accepted
    (non-spiked) batches. On a sustained spike, NO accepted batch arrives, the
    window never updates, the median stays frozen at the pre-spike level, and
    EVERY subsequent batch trips the guard -> an absorbing deadlock. Without a
    re-arm (re-anchor the median on sustained all-skip, a rollback mode, a
    quantile decay, or a window clear), the filter can never escape. The base
    trainer still has this exact accepted-only append with NO rollback cure (C17,
    "6-7x spread").

    Signature refused: a function containing ``<ref>.append(...)`` (``<ref>`` a
    reference-window name) that is lexically nested inside an ``if`` whose test
    references spike/skip tokens, WITHOUT any re-arm token elsewhere in the same
    function.

    Same-line / in-function waiver:
    ``# REJECT_FILTER_REARM_OK:<rationale>``.

    STRICT-FLIP CONDITION: flip to ``strict=True`` once the base trainer's
    accepted-only median append is given a rollback/re-arm cure (or the base loop
    is deprecated), i.e. live-count 0. Warn-only until then.
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    for path in _existing_trainers(root):
        text = _read(path)
        if not text:
            continue
        scanned += 1
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        rel = path.relative_to(root).as_posix()
        for fn in _func_defs(tree):
            fn_src = _span_source(lines, fn)
            # Find accepted-only appends: an append to a reference window that is
            # lexically under an `if` whose test references skip/spike tokens.
            accepted_only: list[ast.Call] = []
            for sub in ast.walk(fn):
                if not isinstance(sub, ast.If):
                    continue
                test_src = _span_source(lines, sub.test).lower()
                if not any(tok in test_src for tok in _SKIP_TEST_TOKENS):
                    continue
                # appends anywhere under this if (body or orelse) — both shapes
                # (`if not spiked: append` and `if skip: ... else: append`) are
                # accepted-only w.r.t. the spike discriminator.
                for branch in (sub.body, sub.orelse):
                    for stmt in branch:
                        accepted_only.extend(
                            _append_calls_to_reference_window(stmt)
                        )
            if not accepted_only:
                continue
            # SPECIFIC escape hatch present anywhere in the function? Scan CODE only
            # (comments stripped) so the waiver-marker name (which contains
            # "rearm") cannot self-satisfy the re-arm requirement.
            fn_code_l = _strip_comments(fn_src).lower()
            if any(tok in fn_code_l for tok in _REARM_TOKENS_SPECIFIC):
                continue
            # GENERIC escape hatch (.clear( / reset( / rollback) counts ONLY when it is NEAR an
            # accepted-only append (within _REARM_PROXIMITY_LINES) -- an unrelated clear/reset/rollback
            # elsewhere in the function must NOT falsely clear the median-freeze deadlock (MISS surface).
            near_parts: list[str] = []
            for _ap in accepted_only:
                lo = max(0, _ap.lineno - 1 - _REARM_PROXIMITY_LINES)
                hi = min(len(lines), _ap.lineno + _REARM_PROXIMITY_LINES)
                near_parts.append("\n".join(lines[lo:hi]))
            near_code_l = _strip_comments("\n".join(near_parts)).lower()
            if any(tok in near_code_l for tok in _REARM_TOKENS_GENERIC):
                continue
            # Waiver in the function scope?
            if _waiver_present(fn_src, "REJECT_FILTER_REARM_OK"):
                continue
            first = accepted_only[0]
            violations.append(
                f"{rel}:{first.lineno}: reference-window append is accepted-only "
                f"(guarded by a spike/skip test) inside function "
                f"{getattr(fn, 'name', '?')!r} with NO re-arm escape hatch "
                f"(clear / re-anchor / rollback / quantile-decay) -> absorbing "
                f"median-freeze deadlock (C1/C17). Add a re-arm, or a "
                f"`# REJECT_FILTER_REARM_OK:<rationale>` waiver."
            )
    return _finish(
        name="check_reject_filter_updates_reference_from_accepted_only_has_rearm",
        tag="reject-filter-accepted-only-no-rearm",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{scanned} trainer(s) scanned",
    )


# ===========================================================================
# Catalog #399 — C13: a witness launch.sh must not contain duplicate --long-flags
# (argparse last-wins SILENTLY shifts schedules).
# ===========================================================================

_LONG_FLAG_RE = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]+)")


def check_no_duplicate_long_flags_in_launch(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catalog #399 (C13) — an emitted witness ``launch.sh`` argv must not contain
    duplicate ``--long-flags``.

    Confound C13: argparse is last-wins on a duplicated flag, SILENTLY. Five
    duplicate flags in the v5/v6 launches shifted schedules ~100 epochs and
    flattened the eikonal-weight anneal (``--eikonal-weight-end 0.1`` then
    ``0.05`` -> anneal DEAD). Reading the top-of-config value is then wrong; the
    header lies. The launcher must assert no duplicate long-flags before emit.

    Signature refused: the same ``--flag`` token appearing 2+ times in a
    ``launch.sh``.

    Per-file waiver: ``# DUP_FLAG_OK:<rationale>`` anywhere in the file (for a
    genuine ``action="append"`` flag, or a grandfathered historical launch).

    STRICT-FLIP CONDITION: flip to ``strict=True`` once the launcher-fixer sibling
    lands the no-dup assertion AND existing historical ``launch.sh`` artifacts are
    waived/cleaned to live-count 0. Warn-only until then (historical launch.sh are
    append-only provenance this builder does not rewrite).
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    for path in _launch_files(root):
        text = _read(path)
        if text is None:
            continue
        scanned += 1
        if _waiver_present(text, "DUP_FLAG_OK"):
            continue
        # Count long-flags across the whole script. Comment lines are excluded so
        # a documented example flag in a comment does not create a false dup.
        code_lines = [
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        ]
        flags = _LONG_FLAG_RE.findall("\n".join(code_lines))
        counts = Counter(flags)
        dups = sorted(f for f, c in counts.items() if c >= 2)
        if not dups:
            continue
        rel = path.relative_to(root).as_posix()
        detail = ", ".join(f"{f}(x{counts[f]})" for f in dups[:6])
        violations.append(
            f"{rel}: duplicate long-flag(s) [{detail}] — argparse last-wins "
            f"silently shifts schedules (C13). De-duplicate, or add a "
            f"`# DUP_FLAG_OK:<rationale>` waiver."
        )
    return _finish(
        name="check_no_duplicate_long_flags_in_launch",
        tag="launch-duplicate-long-flags",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{scanned} launch.sh scanned",
    )


# ===========================================================================
# Catalog #403 — the base-vs-levelset hosc anneal-default divergence
# (T3 council 2026-07-05). The base trainer auto-anneals hosc β (1->4) when
# endpoints are unset, but the LEVELSET trainer's _hosc_beta_for_epoch returns
# None (CONSTANT β) when --hosc-beta-end is unset -> a fixed β=4 = tanh(β·sin)
# saturation = vanishing grad (CLAUDE.md Capstone caveat: NEVER fixed β=4). The
# base's safety default is silently dropped by the levelset launch path;
# proven_base shipped fixed-β=4 (two council benches traced levelset:1982/2436/5895).
# ===========================================================================
def check_levelset_hosc_requires_beta_end(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catalog #403 — an emitted witness ``launch.sh`` using ``--activation hosc``
    MUST also set ``--hosc-beta-end`` (annealed) OR use ``--activation step_basis``;
    else the levelset trainer runs a CONSTANT ``--hosc-beta`` (tanh saturation,
    vanishing gradient — the CLAUDE.md-forbidden fixed-β=4 divergence config).

    Signature refused: a launch.sh with ``--activation hosc`` but NO
    ``--hosc-beta-end``. Per-file waiver: ``# FIXED_BETA_OK:<rationale>``.
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    for path in _launch_files(root):
        text = _read(path)
        if text is None:
            continue
        scanned += 1
        if _waiver_present(text, "FIXED_BETA_OK"):
            continue
        code = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        )
        if "--activation hosc" not in code:
            continue
        if "--hosc-beta-end" in code:
            continue
        rel = path.relative_to(root).as_posix()
        violations.append(
            f"{rel}: --activation hosc WITHOUT --hosc-beta-end -> the levelset "
            f"trainer runs a CONSTANT --hosc-beta (tanh-saturation / vanishing "
            f"gradient; CLAUDE.md-forbidden fixed-β=4). Add --hosc-beta-end (anneal) "
            f"or use --activation step_basis, or a `# FIXED_BETA_OK:<rationale>` waiver."
        )
    return _finish(
        name="check_levelset_hosc_requires_beta_end",
        tag="levelset-hosc-fixed-beta",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{scanned} launch.sh scanned",
    )


# ===========================================================================
# Catalog #400 — C8: a launch with resume palliative flags
# (--resume-clear-spike-guard / --resume-allow-lever-drift) that restores
# optimizer moments (a resume) but lacks --warm-start-weights-only is refused.
# ===========================================================================

_RESUME_PALLIATIVE_FLAGS = ("--resume-clear-spike-guard", "--resume-allow-lever-drift")
_RESUME_RESTORE_FLAGS = ("--resume-from", "--resume ", "--resume=", "--resume\n", "--resume\t")
_WARM_START_WEIGHTS_ONLY = "--warm-start-weights-only"


def check_resume_palliative_flags_imply_warm_start(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catalog #400 (C8) — a resume launch that sets a palliative flag
    (``--resume-clear-spike-guard`` / ``--resume-allow-lever-drift``) while
    restoring optimizer moments MUST also pass ``--warm-start-weights-only``.

    Confound C8: the v5/v6 launches used warm-start's COSMETIC side-effects
    (clear-spike-guard + allow-lever-drift) while KEEPING the poison — the stale
    ep100 optimizer moments — because they did NOT pass
    ``--warm-start-weights-only``. The restored moments were fit to the ep100 loss
    geometry, but the resume argv changes that geometry (adds boundary-distance
    0.2 + eikonal-viscosity at full weight), so the momentum drives the exploding
    term. The palliative flags SILENCE the drift guard; the poison rides through.

    Signature refused: a ``launch.sh`` containing any palliative flag AND a
    ``--resume``/``--resume-from`` (opt-moment restore) but NOT
    ``--warm-start-weights-only``.

    Per-file waiver: ``# RESUME_PALLIATIVE_OK:<rationale>``.

    STRICT-FLIP CONDITION: flip to ``strict=True`` once the launcher makes the
    palliative flags imply weights-only (or refuses the combination) AND existing
    launch.sh reach live-count 0. Warn-only until then.
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    for path in _launch_files(root):
        text = _read(path)
        if text is None:
            continue
        scanned += 1
        has_palliative = any(f in text for f in _RESUME_PALLIATIVE_FLAGS)
        if not has_palliative:
            continue
        # Restoring moments == a resume that is NOT weights-only.
        has_resume = any(f in text for f in _RESUME_RESTORE_FLAGS)
        if not has_resume:
            continue
        if _WARM_START_WEIGHTS_ONLY in text:
            continue
        if _waiver_present(text, "RESUME_PALLIATIVE_OK"):
            continue
        which = ", ".join(f for f in _RESUME_PALLIATIVE_FLAGS if f in text)
        rel = path.relative_to(root).as_posix()
        violations.append(
            f"{rel}: resume palliative flag(s) [{which}] set with an optimizer-"
            f"moment restore but WITHOUT {_WARM_START_WEIGHTS_ONLY} — stale "
            f"moments ride into a drifted geometry (C8). Add "
            f"{_WARM_START_WEIGHTS_ONLY}, or a `# RESUME_PALLIATIVE_OK:<rationale>` "
            f"waiver."
        )
    return _finish(
        name="check_resume_palliative_flags_imply_warm_start",
        tag="resume-palliative-implies-warm-start",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{scanned} launch.sh scanned",
    )


# ===========================================================================
# Catalog #401 — C12: the --verdict-pairs argparse default must be 0 (all/n600),
# not a subset, so the DEFAULT measurement is n600.
# ===========================================================================


def check_verdict_pairs_default_is_n600(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catalog #401 (C12) — the ``--verdict-pairs`` argparse ``default=`` must be
    ``0`` (== all pairs / n600), never a non-zero subset.

    Confound C12: ``--verdict-pairs`` defaults to ``24`` (levelset trainer:6470) —
    the DEFAULT best-checkpoint selection + ALL d_seg telemetry + the closed-loop
    classifier run on 24/600 pairs, violating the CLAUDE.md n600 non-negotiable at
    the exact number that DEFINES the goal. This session's launch correctly passed
    ``--verdict-pairs 0`` but the default is the trap for the next launch.

    Signature refused: ``add_argument("--verdict-pairs", ..., default=N)`` with
    ``N != 0``.

    Same-line / in-call waiver: ``# VERDICT_PAIRS_DEFAULT_OK:<rationale>``.

    STRICT-FLIP: DONE (2026-07-06). The levelset trainer now defaults
    ``--verdict-pairs 0`` (train_levelset...:7044), so live-count is 0 ->
    ``preflight_all`` now calls this gate ``strict=True`` per the Strict-flip
    atomicity rule (the function default stays ``strict=False`` so direct/test calls
    remain warn-only).
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    for path in _existing_trainers(root):
        text = _read(path)
        if not text:
            continue
        scanned += 1
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _call_is_add_argument(node):
                continue
            flag = _first_str_positional(node)
            if flag != "--verdict-pairs":
                continue
            default = _kw_constant(node, "default")
            if default is _MISSING:
                continue
            try:
                as_int = int(default)
            except (TypeError, ValueError):
                continue
            if as_int == 0:
                continue
            if _waiver_present(_span_source(lines, node), "VERDICT_PAIRS_DEFAULT_OK"):
                continue
            violations.append(
                f"{rel}:{node.lineno}: --verdict-pairs default={default!r} is a "
                f"non-n600 subset (C12). Set default=0 (all pairs), or add a "
                f"`# VERDICT_PAIRS_DEFAULT_OK:<rationale>` waiver."
            )
    return _finish(
        name="check_verdict_pairs_default_is_n600",
        tag="verdict-pairs-default-n600",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{scanned} trainer(s) scanned",
    )


# ===========================================================================
# Catalog #402 — C6: telemetry verdict / loss_terms emitters must carry a
# liveness field (accepted_frac / weights_stepped / frozen_epoch / spike_skipped)
# so no reader mistakes a FROZEN run for a converging one.
# ===========================================================================

# Liveness tokens: any one on an emitter row lets a reader distinguish a stepped
# from a frozen epoch.
_LIVENESS_TOKENS = (
    "accepted_frac",
    "accepted_batches",
    "accepted_fraction",
    "weights_stepped",
    "frozen_epoch",
    "frozen:",
    '"frozen"',
    "spike_skipped",
    "n_skips",
    "skip_frac",
    "ema_updates_since",
)
# Emitter stages that make a load-bearing claim about training state.
_STATEFUL_EMITTER_STAGES = ("verdict", "loss_terms")
_EMITTER_WINDOW_LINES = 16


def check_telemetry_verdict_rows_carry_liveness(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catalog #402 (C6, best-effort static) — every ``{"stage": "verdict"}`` and
    ``{"stage": "loss_terms"}`` telemetry emitter must carry a LIVENESS field
    (``accepted_frac`` / ``weights_stepped`` / ``frozen_epoch`` / ``spike_skipped``
    / ...) within the emitter's dict-construction window.

    Confound C6 (the #1 self-protect of the whole hunt): ALL telemetry / verdict /
    closed-loop rows were emitted on FROZEN state with NO liveness stamp. The
    closed-loop controller certified a frozen run "converging"; ``ep_loss:0.0``
    read as "converged to zero"; the 0.025 gold was sampled on frozen epochs;
    beta-anneal-on-frozen-weights made the eikonal "creep" look like physics. A
    single per-row accepted-batch-fraction (liveness) makes frozen indistinguish-
    able from converging IMPOSSIBLE.

    This is a light static PRESENCE check (grep the emitter within a
    16-line window), not a full dataflow proof — the dict may be built across
    several lines. It complements the Layer-1 runtime liveness stamp the trainer
    sibling adds.

    Signature warned: a stateful emitter (``stage`` in {verdict, loss_terms})
    with no liveness token in its construction window.

    Same-line / in-window waiver: ``# TELEMETRY_LIVENESS_OK:<rationale>``.

    STRICT-FLIP CONDITION: flip to ``strict=True`` once the trainer sibling stamps
    accepted-batch-fraction (liveness) onto every verdict/loss_terms row
    (live-count 0). Warn-only until then.
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    stage_re = re.compile(r'["\']stage["\']\s*:\s*["\'](verdict|loss_terms)["\']')
    for path in _existing_trainers(root):
        text = _read(path)
        if not text:
            continue
        scanned += 1
        lines = text.splitlines()
        rel = path.relative_to(root).as_posix()
        for idx, raw in enumerate(lines):
            m = stage_re.search(raw)
            if not m:
                continue
            stage = m.group(1)
            if stage not in _STATEFUL_EMITTER_STAGES:
                continue
            window = "\n".join(lines[idx : idx + _EMITTER_WINDOW_LINES])
            if any(tok in window for tok in _LIVENESS_TOKENS):
                continue
            if _waiver_present(window, "TELEMETRY_LIVENESS_OK"):
                continue
            violations.append(
                f"{rel}:{idx + 1}: {stage!r} telemetry emitter has no liveness "
                f"field (accepted_frac / weights_stepped / frozen_epoch / "
                f"spike_skipped) in its construction window (C6) — a reader cannot "
                f"tell a FROZEN epoch from a converging one. Stamp liveness, or add "
                f"a `# TELEMETRY_LIVENESS_OK:<rationale>` waiver."
            )
    return _finish(
        name="check_telemetry_verdict_rows_carry_liveness",
        tag="telemetry-rows-carry-liveness",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{scanned} trainer(s) scanned",
    )


# ===========================================================================
# Catalog #403 — req V (#353): a launch-config-authoring function (a public
# ``derive_*_config`` entry point in the autoconfig seam) must route through the
# typed DSL layer (tac.witness_dsl.typed_config) — no parallel hand-assembly.
# ===========================================================================

# Config-authoring files scanned (OUTSIDE witness_dsl, which is the sanctioned path).
_CONFIG_AUTHORING_FILES = (
    "src/tac/witness_autoconfig.py",
)
# Tokens proving a function routes through the typed DSL authoring/validation layer.
_TYPED_DSL_TOKENS = (
    "typed_config",
    "TypedWitnessConfig",
    "to_program",
    "build_launch_manifest",
    "_attach_dsl_program_manifest",
    "dsl_program_manifest",
)
_DERIVE_CONFIG_RE = re.compile(r"^derive_\w*config$")


def check_launch_config_authored_in_dsl(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catalog #403 (req V, #353) — a public ``derive_*_config`` launch-config entry
    point in the autoconfig seam must route through the typed DSL layer
    (``tac.witness_dsl.typed_config``: ``TypedWitnessConfig.to_program`` +
    ``build_launch_manifest``), never hand-assemble argv on a parallel path.

    Operator 2026-07-08 (verbatim): "The config must be defined in the DSL — no ad
    hoc or hand crafting ... integrate all with apparatus to prevent more dumbass
    bullshit." Every parallel, untyped ``derive_*_config`` is where the PR95 skeleton,
    hardcoded epochs, and bare constants re-entered silently.

    Signature warned: a module-level ``def derive_*_config(...)`` whose body contains
    NO typed-DSL token and NO waiver. The migrated seam (``derive_crucible_v6_config``)
    calls ``_attach_dsl_program_manifest`` (a typed-DSL token) and passes.

    Same-line / in-body waiver: ``# DSL_CONFIG_AUTHORING_OK:<rationale>``.

    STRICT-FLIP CONDITION: flip to ``strict=True`` once the autoconfig migration queue
    is drained (``derive_sealed_205_config`` / ``derive_store_nothing_205_config`` /
    ``derive_fresh_seeded_config`` / ``derive_config`` route through the typed layer or
    carry a migration-queue waiver → live-count 0). WARN-only until then — refusing the
    un-migrated seam now would wedge every non-crucible launch mid-migration.
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    for rel in _CONFIG_AUTHORING_FILES:
        path = root / rel
        text = _read(path)
        if not text:
            continue
        scanned += 1
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        for node in tree.body:  # module-level defs only (the config entry points)
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _DERIVE_CONFIG_RE.match(node.name):
                continue
            span = _span_source(lines, node)
            if any(tok in span for tok in _TYPED_DSL_TOKENS):
                continue
            if _waiver_present(span, "DSL_CONFIG_AUTHORING_OK"):
                continue
            violations.append(
                f"{rel}:{node.lineno}: {node.name}() authors a launch config outside the "
                f"typed DSL layer (no tac.witness_dsl.typed_config routing). Route it through "
                f"TypedWitnessConfig.to_program + build_launch_manifest (see "
                f"derive_crucible_v6_config), or add a `# DSL_CONFIG_AUTHORING_OK:<rationale>` "
                f"migration-queue waiver."
            )
    return _finish(
        name="check_launch_config_authored_in_dsl",
        tag="launch-config-authored-in-dsl",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{scanned} config-authoring file(s) scanned",
    )


# Catalog #404 — the MAGNITUDE-DISMISSAL bug class (operator recurring correction
# 2026-07-08, "we have had this discussion before"). Landing #2 of the self-protect
# pattern: landing #1 = the memory relative-not-absolute-significance-near-goal + the
# live re-audit; the RUNTIME sibling = the Stop hook tools/magnitude_dismissal_detector.py
# (fires the moment the class is INTRODUCED). This STATIC gate scans committed
# .omx/research/*.md memos for a DEFER / DOWNGRADE / ORPHAN / KILL justified by ABSOLUTE
# magnitude (weak/negligible/noise/small-ΔS/not-worth-it) WITHOUT either (a) a
# relative-significance number OR (b) a measured-un-recoverability citation.
#
# ONE classifier SoT: the deterministic predicates live in the hook module (Claude-workflow
# apparatus, not tac — CLAUDE.md tac-cleanliness); this gate OPPORTUNISTICALLY loads them
# (fail-open no-op if the hook is absent). The hook adds the fmtools semantic layer on top;
# this gate is the deterministic-only static surface (preflight stays dependency-light).
_MAGNITUDE_DETECTOR_PATH = REPO_ROOT / "tools" / "magnitude_dismissal_detector.py"


def _load_magnitude_detector():
    """Import the hook module's pure classifier (the SoT). None on any failure — the
    gate then no-ops (fail-open), never crashing preflight over apparatus tooling."""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_magnitude_dismissal_detector_sot", _MAGNITUDE_DETECTOR_PATH)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def check_no_unjustified_magnitude_dismissal(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
    max_report: int = 15,
) -> list[str]:
    """Catalog #404 (the MAGNITUDE-DISMISSAL class) — scan committed .omx/research/*.md
    memos for a lever/finding dismissal (defer/downgrade/orphan/kill) justified purely by
    ABSOLUTE magnitude, WITHOUT a relative-significance number (ΔS / remaining-gap) OR a
    measured-un-recoverability citation (#141 label-noise WITH a measurement).

    The two legitimate dismissals — "measured un-recoverable" and "structurally
    superseded" — are EXEMPT by construction (the classifier's exemption vocabulary), as
    are non-dismissal magnitude usages ("weak supervision", "noise floor", "noise
    injection"). Deliberate exception: same-line ``# MAGNITUDE_DISMISSAL_OK:<rationale>``.

    Deterministic-only (the fmtools semantic refinement lives in the Stop-hook sibling).
    Reuses the hook module's classifier as the SINGLE source of truth (fail-open no-op if
    the hook file is absent, so this gate can never crash preflight over apparatus).

    STRICT-FLIP CONDITION: flip to ``strict=True`` after the historical re-audit sweep
    (memory point 3: every prior absolute-magnitude DEFER/DOWNGRADE/ORPHAN re-opened for
    relative-significance re-ranking) brings live-count to 0. WARN-ONLY until then — the
    .omx/research corpus predates the discipline and will carry historical hits.
    """
    root = Path(repo_root or REPO_ROOT)
    detector = _load_magnitude_detector()
    violations: list[str] = []
    scanned = 0
    if detector is not None and hasattr(detector, "deterministic_flags"):
        research = root / ".omx" / "research"
        for path in sorted(research.glob("*.md")) if research.exists() else []:
            text = _read(path)
            if text is None:
                continue
            scanned += 1
            rel = path.relative_to(root).as_posix()
            lines = text.splitlines()
            for msg in detector.deterministic_flags(lines, source=rel):
                # honor the per-line waiver at the confound-gate surface too (the hook's
                # classifier already skips waived lines, but keep the belt-and-suspenders).
                try:
                    ln_no = int(msg.split(":", 2)[1])
                    if _waiver_present(lines[ln_no - 1], "MAGNITUDE_DISMISSAL_OK"):
                        continue
                except (ValueError, IndexError):
                    pass
                violations.append(msg)
    return _finish(
        name="check_no_unjustified_magnitude_dismissal",
        tag="no-unjustified-magnitude-dismissal",
        violations=violations[:max_report],
        strict=strict,
        verbose=verbose,
        ok_detail=(f"{scanned} research memo(s) scanned"
                   if detector is not None else "classifier absent (fail-open no-op)"),
    )


# ===========================================================================
# EIGHTFOLD DESIGN-PHILOSOPHY GATES (2026-07-09 operator "Encode all")
# Source: `.omx/research/design_philosophies_eightfold_20260709.md` +
# DAG FEED-eightfold-philosophies. The eight brushed-against design philosophies
# (siblings of the same-day geometry-first bindings) get STRUCTURAL apparatus
# where automatable: P1 significance-key canonicalization + P4 meter-canary
# presence are warn-only preflight gates; P2/P5/P6/P7/P8 are fuzzy-by-nature and
# live as crucible SEAL standing checks (see
# `.omx/research/crucible_standing_checks_eightfold_20260709.md`), not static
# gates. Landing memo: `.omx/research/eightfold_apparatus_build_20260709.md`.
# ===========================================================================

# P1 store: the relative-significance JSONL (the ΔS value axis of the DSL).
_SIGNIFICANCE_STORE_REL = ".omx/state/lever_relative_significance.jsonl"


def check_significance_keys_canonical(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """P1 (ONE FACT, ONE STORE, ONE KEY) — every key in the relative-significance
    store must resolve, THROUGH ``tac.witness_dsl.activation_ledger.
    canonicalize_significance_keys``, to a HELD DSL ``Lever`` factory name
    (``lever_registry``).

    Confound class (the duty-to-measure ORPHAN bug): a significance row keyed by a
    human/task-# name (e.g. ``d_seg_aware_taper_121``) that never reconciles onto
    its canonical factory name (``DsegAwareTaper``) makes ``duty_to_measure_ranked``
    compute ``registered = (key in factory_names) == False`` and FALSELY report a
    built+held+wired lever as ``~=unbuilt`` (duty-to-BUILD) instead of
    ``*=never-fired`` (duty-to-MEASURE). RECEIPT 2026-07-09: exactly this — the
    ``canonicalize_significance_keys`` alias map was the point fix; THIS gate is the
    class fix (it refuses a NEW task-#-keyed row that lacks an alias / a build / a
    waiver from silently re-orphaning). Clause A of the same-day geometry-first
    binding, pointed at our own apparatus.

    An unresolved key is a genuine finding UNLESS it is a legitimate not-yet-a-lever
    finding (e.g. a byte-close-tool lever like ``latent_table_truncate_d18_k90``,
    or an A/B finding), in which case the ROW carries an in-notes, JSONL-safe waiver
    ``# SIGNIFICANCE_KEY_OK:<rationale>`` (embedded in the row's ``notes`` string —
    a bare ``<rationale>`` placeholder does not self-waive, Catalog #287 sister).

    Warn-only (Strict-flip atomicity rule). STRICT-FLIP CONDITION: flip once every
    store key resolves to a held factory OR carries the waiver (live-count 0) — this
    requires sibling ``activation_ledger`` alias/build work + the two intentional
    non-factory findings gaining their waiver, so it cannot be guaranteed here.

    Same-line/in-row waiver: ``# SIGNIFICANCE_KEY_OK:<rationale>``.
    """
    root = Path(repo_root or REPO_ROOT)
    store = root / _SIGNIFICANCE_STORE_REL
    violations: list[str] = []
    if not store.is_file():
        return _finish(
            name="check_significance_keys_canonical",
            tag="significance-keys-canonical",
            violations=violations,
            strict=strict,
            verbose=verbose,
            ok_detail="no significance store on disk",
        )
    # Resolve THROUGH the canonical functions the philosophy names (DRY + NO-FAKE:
    # the gate uses the very reconciliation the apparatus uses, never a re-impl).
    try:
        from tac.witness_dsl.activation_ledger import (
            _read_significance,
            canonicalize_significance_keys,
            known_levers,
        )
    except Exception as exc:  # pragma: no cover - import-environment guard
        return _finish(
            name="check_significance_keys_canonical",
            tag="significance-keys-canonical",
            violations=violations,
            strict=strict,
            verbose=verbose,
            ok_detail=f"activation_ledger/lever_registry unavailable ({exc!r}) — fail-open",
        )
    # ddm_rg5 (#825): was ``lever_registry.lever_factories()`` — the SINGLE-MODULE surface. A
    # significance row keyed on an fh1/ph3_s10/ax1 lever was therefore reported as "not a factory
    # — build the lever", when the lever exists. ``known_levers()`` is now the package-wide
    # universe, so this gate resolves against every factory that actually exists.
    factory_names = set(known_levers())
    sig = _read_significance(store)
    canon = canonicalize_significance_keys(sig, factory_names)
    resolved = 0
    for key, row in canon.items():
        if key in factory_names:
            resolved += 1
            continue
        # Waiver lives in the row's parsed ``notes`` string (JSONL has no comments;
        # scanning the raw JSON line would let trailing JSON syntax masquerade as a
        # real rationale — the placeholder-rejection must see the clean field).
        notes = str(row.get("notes", "")) if isinstance(row, dict) else ""
        if _waiver_present(notes, "SIGNIFICANCE_KEY_OK"):
            continue
        hint = (
            "notes name a held factory — add its alias to "
            "activation_ledger._SIGNIFICANCE_LEVER_ALIASES"
            if isinstance(row, dict)
            and any(fn in str(row.get("notes", "")) for fn in factory_names)
            else "build the lever, or (if it is intentionally not a DSL factory) "
            "add a `# SIGNIFICANCE_KEY_OK:<rationale>` marker inside the row notes"
        )
        violations.append(
            f"{_SIGNIFICANCE_STORE_REL}: significance key {key!r} does not resolve "
            f"to a held DSL Lever factory (P1 orphan duty-to-measure signal — a "
            f"task-#-keyed row that never reconciled onto its factory name would be "
            f"FALSELY reported unbuilt). {hint}."
        )
    return _finish(
        name="check_significance_keys_canonical",
        tag="significance-keys-canonical",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{resolved}/{len(canon)} store key(s) resolve to a held factory",
    )


# ── P4 meter-canary gate ─────────────────────────────────────────────────────
_WITNESS_CONTROL_REL = "src/tac/witness_control"
# A CERTAIN measurement surface by NAME. Deliberately EXCLUDES bare ``*Gate``:
# in this codebase ``Gate`` is an ACTUATOR (EventBackstopGate fires treatments;
# GateStep is a frozen value object) — neither is a measurement meter, and P4 is
# about METERS ("every new MEASUREMENT surface ships with a positive control").
# The two named exemplars pass via this set (SigmaMinPlateauDetector -> Detector$,
# VerdictTrendAlarm -> Alarm$). A ``*Gate`` that IS a meter still gets caught by
# the observe/detect/classify method signal below.
_METER_NAME_RE = re.compile(r".*(Detector|Alarm|Trend|Plateau|Monitor|Observer|Meter)$")
# Unambiguous MEASUREMENT verbs (an actuator uses fire/step/update; a meter reads).
_METER_VERB_METHODS = ("observe", "detect", "classify")
# Canary/positive-control presence tokens (module text OR its test-file text).
_CANARY_TOKENS = (
    "canary",
    "positive_control",
    "negative_control",
    "synthetic_control",
    "known_effect",
    "known-effect",
)
_CANARY_SYNTH_RE = re.compile(r"synthetic_\w*control")


def _fm_meter_advisory(class_name: str, class_source: str, timeout: float = 12.0) -> dict | None:
    """ADVISORY on-device FM second opinion for a heuristic-UNCERTAIN class: is it a
    measurement/detector surface (a reading a decision is drawn from) vs an
    actuator/controller/value-object? Mirrors the ``tools/auto_push_main`` fmtools
    firewall (memory ``reference-apple-ondevice-fm-fmtools-classifier-capability``,
    #259): the FM runs in the SEPARATE fmtools venv via SUBPROCESS (the pact venv
    gains ZERO deps); fail-open (absent venv / error / timeout ⇒ ``None``). NEVER a
    sole authority — the caller records the advisory in the finding rationale only,
    and the deterministic heuristic remains the floor. Opt-in (``use_fmtools``) so
    the per-session ``preflight_all`` path pays ZERO cost by default.
    """
    import os
    import subprocess

    fm_py = None
    for cand in (
        os.environ.get("EIGHTFOLD_FM_PYTHON"),
        os.environ.get("DASH_FM_PYTHON"),
        os.path.expanduser("~/Projects/fmtools/.venv/bin/python"),
    ):
        if cand and os.path.exists(cand):
            fm_py = cand
            break
    if not fm_py:
        return None
    script = r'''
import asyncio, json, sys
try:
    import apple_fm_sdk as fm
    from fmtools import local_extract
except Exception:
    print("{}"); raise SystemExit(0)

@fm.generable()
class MeterCheck:
    verdict: str = fm.guide(anyOf=["meter", "not_meter"],
        description="'meter' if the class is a MEASUREMENT/detector surface that produces a reading, classification or verdict a downstream decision is drawn from; 'not_meter' for an actuator/controller/value-object/config.")
    reason: str = fm.guide(description="A short phrase naming why.")

@local_extract(MeterCheck, retries=1, instructions=(
    "You inspect a Python class from a witness-training CONTROL package and decide if it is a MEASUREMENT "
    "surface (a 'meter': it OBSERVES state and emits a reading/classification/verdict that a decision is "
    "drawn from — e.g. a plateau detector, a trend alarm) versus NOT a meter (an actuator that fires a "
    "treatment, a controller, a plain value/config dataclass, an averager). Return 'meter' ONLY when the "
    "class's job is to MEASURE and REPORT. When uncertain, return 'not_meter' (the deterministic name/method "
    "heuristic is the floor; you are a precision second opinion)."))
async def _check(src: str) -> MeterCheck:
    """(instructions above)"""

async def _main():
    try:
        text = sys.stdin.read()[:6000]
    except Exception:
        print("{}"); return
    if not text.strip():
        print("{}"); return
    try:
        r = await _check(text)
        print(json.dumps({
            "is_meter": (str(getattr(r, "verdict", "") or "").lower() == "meter"),
            "reason": str(getattr(r, "reason", "") or ""),
        }))
    except Exception:
        print("{}")

asyncio.run(_main())
'''
    # class_source already includes the ``class <name>:`` header (from _span_source);
    # pass it verbatim (capped for the FM window) — no double-prefix.
    payload = (class_source or f"class {class_name}: ...")[:6000]
    try:
        proc = subprocess.run(
            [fm_py, "-c", script], input=payload,
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return None
        out = json.loads(proc.stdout.strip() or "{}")
        return out if isinstance(out, dict) and out else None
    except Exception:
        return None


def check_witness_control_meters_have_canaries(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
    use_fmtools: bool = False,
) -> list[str]:
    """P4 (NO METER WITHOUT A CANARY) — every MEASUREMENT/detector class in
    ``src/tac/witness_control/*.py`` must ship a canary / positive-control surface
    (a known-effect signal it MUST register + a negative control it must NOT fire
    on) before its readings gate anything — confound-L3 lifted from verdicts to
    build-time law.

    Detection (deterministic floor):
      * CERTAIN meter = class NAME matches ``*(Detector|Alarm|Trend|Plateau|Monitor|
        Observer|Meter)``. A certain meter with no canary token in its module OR its
        ``tests/test_<module>.py`` = a VIOLATION.
      * UNCERTAIN = class defines an ``observe`` / ``detect`` / ``classify`` method
        but the name does not signal a meter (an actuator/controller may also
        observe). These are NOT counted as violations by the heuristic; they are
        listed, and — ONLY when ``use_fmtools=True`` — an on-device FM ADVISORY
        (``_fm_meter_advisory``, #259 firewall: separate venv, subprocess,
        fail-open, NEVER sole authority) is recorded in the finding rationale. Per
        the operator nudge 2026-07-09: heuristic-certain cases decide directly;
        heuristic-uncertain cases get an fmtools advisory recorded in rationale,
        never authority (warn-only gate + advisory classifier = honest composition).
        Default ``use_fmtools=False`` keeps the per-session preflight path at ZERO
        FM cost (documented heuristic-only disposition, the nudge's accepted
        fallback).

    Passing exemplar VERIFIED (source inspection 2026-07-09): ``SigmaMinPlateauDetector``
    (sigma_min_plateau.py ``canary_suite`` = synthetic positive + rising negative).
    NOTE (re-derived, §4): ``VerdictTrendAlarm`` (verdict_trend_alarm.py) carries NO
    canary token in-module or in a test — it is a CURRENT VIOLATOR, not a passing
    exemplar as the build brief supposed; disposition = sibling adds a
    canary_suite-style control OR a ``# METER_CANARY_OK:<rationale>`` waiver.

    Warn-only (Strict-flip atomicity rule). STRICT-FLIP CONDITION: flip once every
    certain meter carries a canary or a waiver (live-count 0).

    Same-line/near-class waiver: ``# METER_CANARY_OK:<rationale>``.
    """
    root = Path(repo_root or REPO_ROOT)
    ctrl = root / _WITNESS_CONTROL_REL
    violations: list[str] = []
    uncertain: list[str] = []
    scanned = 0
    if not ctrl.is_dir():
        return _finish(
            name="check_witness_control_meters_have_canaries",
            tag="meter-canary",
            violations=violations,
            strict=strict,
            verbose=verbose,
            ok_detail="witness_control dir absent",
        )
    for path in sorted(ctrl.glob("*.py")):
        if path.name == "__init__.py":
            continue
        text = _read(path)
        if not text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        scanned += 1
        rel = path.relative_to(root).as_posix()
        test_text = _read(ctrl / "tests" / f"test_{path.stem}.py") or ""
        haystack = text + "\n" + test_text
        has_canary = any(tok in haystack for tok in _CANARY_TOKENS) or bool(
            _CANARY_SYNTH_RE.search(haystack)
        )
        module_waived = _waiver_present(haystack, "METER_CANARY_OK")
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {
                b.name
                for b in node.body
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            name_meter = bool(_METER_NAME_RE.match(node.name))
            verb_meter = bool(methods.intersection(_METER_VERB_METHODS))
            if not (name_meter or verb_meter):
                continue
            if has_canary:
                continue
            span = _span_source(lines, node)
            if module_waived or _waiver_present(span, "METER_CANARY_OK"):
                continue
            if name_meter:
                violations.append(
                    f"{rel}:{node.lineno}: meter class {node.name!r} has no canary / "
                    f"positive-control (P4) in its module or tests/test_{path.stem}.py "
                    f"— a meter must register a known-effect (positive) + not-fire on a "
                    f"negative control before its readings gate anything. Add a "
                    f"canary_suite-style control (see sigma_min_plateau.canary_suite) "
                    f"or a `# METER_CANARY_OK:<rationale>` waiver."
                )
            else:
                advisory = _fm_meter_advisory(node.name, span) if use_fmtools else None
                if advisory and advisory.get("is_meter"):
                    violations.append(
                        f"{rel}:{node.lineno}: class {node.name!r} classified a "
                        f"MEASUREMENT surface by fmtools ADVISORY "
                        f"([{advisory.get('reason', '')}] — advisory only, NOT sole "
                        f"authority) and has no canary (P4). Add a canary or a "
                        f"`# METER_CANARY_OK:<rationale>` waiver."
                    )
                else:
                    uncertain.append(
                        f"{rel}:{node.lineno}: {node.name} "
                        f"(observe/detect/classify; name ambiguous — fmtools advisory "
                        f"available via use_fmtools=True)"
                    )
    if verbose and uncertain:
        print(
            f"  [meter-canary] {len(uncertain)} heuristic-uncertain class(es) "
            f"(advisory-only, not counted): " + "; ".join(uncertain[:5])
        )
    return _finish(
        name="check_witness_control_meters_have_canaries",
        tag="meter-canary",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{scanned} witness_control module(s) scanned",
    )


# ===========================================================================
# Catalog #405 — the #218 ADDITIVE-MARGIN INERT-COMPOSITION confound (the
# binding-vs-inert class, #404). Operator elevation 2026-07-10: HeadGeometry's
# --additive-margin / --head additive-margin arm is a SILENT NO-OP unless
# --margin-field-head-weight>0 (the trainer's margin-field-head target is only
# consumed when mfh_w>0, and the AM base is only non-zero when head==additive-
# margin). An inert arm reads as ON but does NOTHING -> any verdict drawn from a
# run where it was reported-on-but-inert is corrupted (surrogate != authority).
# ONE classifier SoT (:func:`additive_margin_engagement`) is shared by the trainer
# L1 runtime alarm, the DSL `.validate()` L2 fail-closed, and this L2 preflight gate.
# ===========================================================================


def additive_margin_engagement(
    head: str, additive_margin: float, margin_field_head_weight: float,
) -> dict:
    """Pure classifier (the SINGLE SoT) for the #218 additive-margin composition.

    Mirrors the trainer's ACTUAL consumption
    (``train_levelset_witness_realized_through_R_mlx.py``): the per-class margin
    TARGET base ``_mfh_base = additive_margin if head=="additive-margin" else 0.0``
    is applied ONLY when ``margin_field_head_weight > 0`` (the whole
    ``mfh_target_mx`` branch is skipped otherwise). So the AM arm is EFFECTIVE iff
    ``head=="additive-margin" AND mfh_w>0 AND additive_margin!=0``.

    Returns ``{"nominally_set","engaged","inert","reason"}``:
      * ``nominally_set`` — the arm was ASKED for (head is additive-margin OR a
        non-zero additive_margin was passed) — the surface a reader would call "on".
      * ``engaged`` — it will ACTUALLY shape the loss.
      * ``inert`` — ``nominally_set and not engaged`` (the #404 silent no-op).
      * ``reason`` — the precise cause string (for the alarm / violation message).
    """
    head = str(head or "softmax")
    am = float(additive_margin or 0.0)
    mfh = float(margin_field_head_weight or 0.0)
    head_is_am = head == "additive-margin"
    nominally_set = head_is_am or abs(am) > 1e-12
    engaged = head_is_am and mfh > 0.0 and abs(am) > 1e-12
    if engaged:
        reason = "engaged (head=additive-margin, margin-field-head-weight>0, additive-margin!=0)"
    elif not nominally_set:
        reason = "not set (softmax/etf head, additive-margin==0) — no AM arm requested"
    elif head_is_am and mfh <= 0.0:
        reason = ("head=additive-margin but --margin-field-head-weight<=0 -> the margin-field "
                  "target is never built (INERT no-op)")
    elif head_is_am and abs(am) <= 1e-12:
        reason = ("head=additive-margin with additive-margin==0 -> zero hinge base "
                  "(on-but-inert; compose a non-zero --additive-margin)")
    else:  # additive_margin != 0 but head != additive-margin
        reason = (f"--additive-margin={am} set but --head={head!r} (not additive-margin) -> the AM "
                  "base stays 0 (value IGNORED / INERT)")
    return {
        "nominally_set": bool(nominally_set),
        "engaged": bool(engaged),
        "inert": bool(nominally_set and not engaged),
        "reason": reason,
    }


# Real trainer flags (never-invent-flags): the AM composition triple.
_AM_HEAD_FLAG = "--head"
_AM_MARGIN_FLAG = "--additive-margin"
_AM_MFH_FLAG = "--margin-field-head-weight"


def _scalar_flag_value(code: str, flag: str) -> str | None:
    """The last value token following ``flag`` in a launch.sh code body (argparse
    last-wins), or None if absent. Handles ``--flag value`` and ``--flag=value``."""
    val = None
    # --flag=value
    for m in re.finditer(re.escape(flag) + r"=(\S+)", code):
        val = m.group(1)
    # --flag value  (value = next whitespace-delimited token that is not another --flag)
    for m in re.finditer(re.escape(flag) + r"[ \t]+([^\s\\]+)", code):
        tok = m.group(1)
        if not tok.startswith("--"):
            val = tok
    return val


def check_no_inert_additive_margin_composition(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Catalog #405 (#404 binding-vs-inert) — an emitted witness ``launch.sh`` must
    not ask for the #218 additive-margin arm (``--head additive-margin`` or a
    non-zero ``--additive-margin``) while it is INERT (``--margin-field-head-weight``
    absent/<=0, or ``--additive-margin 0`` under ``--head additive-margin``).

    An inert AM arm reads as ON in the launch header but shapes NOTHING in the loss
    (the trainer's margin-field target is built only when ``mfh_w>0``, and the AM
    base is non-zero only when ``head==additive-margin``). A verdict drawn from such
    a run attributes its d_seg to a lever that never engaged — a corrupted
    measurement (surrogate != authority, the #404 confound). REFUSE the inert combo;
    do NOT auto-repair it into activity.

    Signature refused: a ``launch.sh`` whose AM composition
    (:func:`additive_margin_engagement`) is ``inert``.

    Per-file waiver: ``# ADDITIVE_MARGIN_INERT_OK:<rationale>`` (e.g. an intentional
    byte-identical A/B baseline arm that is deliberately off).

    STRICT-FLIP CONDITION: flip to ``strict=True`` once the DSL `.validate()`
    fail-closed (the primary locus) + the trainer L1 alarm land and existing
    launch.sh reach live-count 0. Warn-only until then (historical launch.sh are
    append-only provenance this builder does not rewrite).
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    for path in _launch_files(root):
        text = _read(path)
        if text is None:
            continue
        scanned += 1
        if _waiver_present(text, "ADDITIVE_MARGIN_INERT_OK"):
            continue
        code = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        )
        # The AM arm is only relevant if the launch mentions head/additive-margin at all.
        if _AM_HEAD_FLAG not in code and _AM_MARGIN_FLAG not in code:
            continue
        head = _scalar_flag_value(code, _AM_HEAD_FLAG) or "softmax"
        am = _scalar_flag_value(code, _AM_MARGIN_FLAG) or "0.0"
        mfh = _scalar_flag_value(code, _AM_MFH_FLAG) or "0.0"
        try:
            eng = additive_margin_engagement(head, float(am), float(mfh))
        except (TypeError, ValueError):
            continue
        if not eng["inert"]:
            continue
        rel = path.relative_to(root).as_posix()
        violations.append(
            f"{rel}: #218 additive-margin arm is INERT (#404 binding-vs-inert): "
            f"{eng['reason']}. It reads as ON but shapes no loss -> any verdict from "
            f"this run is corrupted. Compose {_AM_MFH_FLAG}>0 (+ a non-zero "
            f"{_AM_MARGIN_FLAG}) to arm it, drop the AM flags, or add a "
            f"`# ADDITIVE_MARGIN_INERT_OK:<rationale>` waiver."
        )
    return _finish(
        name="check_no_inert_additive_margin_composition",
        tag="additive-margin-inert-composition",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{scanned} launch.sh scanned",
    )


# ===========================================================================
# 2026-07-14/15 apparatus two-landings — workflow confound extinction.
# These gates are behavior probes, not token checks: they import the exact pure
# runtime boundary and execute the diagnosed counterexample plus clean controls.
# ===========================================================================

_CONSOLIDATION_MONITOR_REL = "tools/consolidation_debt.py"
_CONSOLIDATION_SETTINGS_REL = ".claude/settings.json"
_CONSOLIDATION_HOOK_COMMAND = (
    '"$CLAUDE_PROJECT_DIR/.venv/bin/python" '
    '"$CLAUDE_PROJECT_DIR/tools/consolidation_debt.py" --quiet-ok'
)
_CONSOLIDATION_WAIVER = "CONSOLIDATION_DEBT_SIDE_EFFECT_OK"
_CONSOLIDATION_MUTATING_GIT_VERBS = frozenset(
    {
        "add",
        "am",
        "apply",
        "checkout",
        "clean",
        "commit",
        "merge",
        "mv",
        "push",
        "rebase",
        "reset",
        "restore",
        "revert",
        "rm",
        "switch",
        "tag",
    }
)
_CONSOLIDATION_WRITE_METHODS = frozenset(
    {
        "chmod",
        "hardlink_to",
        "mkdir",
        "rename",
        "rmdir",
        "symlink_to",
        "touch",
        "unlink",
        "write_bytes",
        "write_text",
    }
)
_CONSOLIDATION_SUBPROCESS_CALLS = frozenset(
    {
        "os.system",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.run",
    }
)
_CONSOLIDATION_WRITE_CALLS = frozenset(
    {
        "os.makedirs",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.unlink",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
    }
)


def _dotted_call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _call_string_literals(node: ast.Call) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _consolidation_side_effect_waived(node: ast.Call, lines: list[str]) -> bool:
    start = max(0, node.lineno - 1)
    end = min(len(lines), getattr(node, "end_lineno", node.lineno))
    return _waiver_present("\n".join(lines[start:end]), _CONSOLIDATION_WAIVER)


def _open_call_is_writable(node: ast.Call) -> bool:
    mode: str | None = None
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = node.args[1].value if isinstance(node.args[1].value, str) else None
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode = keyword.value.value if isinstance(keyword.value.value, str) else None
    return bool(mode and any(marker in mode for marker in ("w", "a", "x", "+")))


def _mutating_command_reason(node: ast.Call, call_name: str) -> str | None:
    if call_name not in _CONSOLIDATION_SUBPROCESS_CALLS and call_name != "_sh":
        return None
    tokens = _call_string_literals(node)
    lowered = [token.strip().lower() for token in tokens]
    for index, token in enumerate(lowered[:-1]):
        if Path(token).name == "git" and lowered[index + 1] in _CONSOLIDATION_MUTATING_GIT_VERBS:
            return f"mutating git subprocess (`git {lowered[index + 1]}`)"
    joined = " ".join(lowered)
    git_match = re.search(
        r"(?:^|\s)git\s+(" + "|".join(sorted(_CONSOLIDATION_MUTATING_GIT_VERBS)) + r")(?:\s|$)",
        joined,
    )
    if git_match:
        return f"mutating git subprocess (`git {git_match.group(1)}`)"
    for token in lowered:
        basename = Path(token).name
        if (
            basename.startswith("launch")
            or "dispatch" in basename
            or re.search(r"(?:^|[/\s])launch[^\s/]*", token)
            or basename in {"modal", "nohup", "vastai"}
        ):
            return f"launch/dispatch subprocess token ({token!r})"
    return None


def _hook_commands(settings: dict, event: str) -> list[str]:
    commands: list[str] = []
    groups = settings.get("hooks", {}).get(event, [])
    if not isinstance(groups, list):
        return commands
    for group in groups:
        if not isinstance(group, dict):
            continue
        hooks = group.get("hooks", [])
        if not isinstance(hooks, list):
            continue
        for hook in hooks:
            if isinstance(hook, dict) and hook.get("type") == "command":
                command = hook.get("command")
                if isinstance(command, str):
                    commands.append(command)
    return commands


def check_consolidation_debt_monitor_observability_and_cadence(
    *, repo_root: str | Path | None = None, strict: bool = False, verbose: bool = True,
) -> list[str]:
    """Guard the consolidation monitor's read-only contract and proactive cadence.

    The monitor may read Git and durable ledgers, but must not write files, mutate
    Git, or invoke launch/dispatch tooling. Its exact non-blocking ``--quiet-ok``
    command must remain wired into both ``SessionStart`` and ``Stop``. A suspicious
    source call can carry a same-call ``# CONSOLIDATION_DEBT_SIDE_EFFECT_OK:<reason>``
    waiver for a reviewed false positive; placeholder rationales are rejected.

    Wired through ``CONFOUND_GATES`` WARN-ONLY: this is apparatus observability, while
    the codex landing-review gate remains the only Stop-hook blocker.
    """
    root = Path(repo_root or REPO_ROOT)
    monitor = root / _CONSOLIDATION_MONITOR_REL
    settings_path = root / _CONSOLIDATION_SETTINGS_REL
    violations: list[str] = []

    source = _read(monitor)
    if source is None:
        violations.append(
            f"{_CONSOLIDATION_MONITOR_REL}: missing/unreadable — rule chain: proactive "
            "consolidation cadence -> monitor must exist. Restore the monitor."
        )
    else:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            tree = None
            violations.append(
                f"{_CONSOLIDATION_MONITOR_REL}:{exc.lineno or 1}: syntax error — rule "
                "chain: cadence monitor -> executable read-only telemetry. Fix syntax."
            )
        if tree is not None:
            lines = source.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                call_name = _dotted_call_name(node)
                reason: str | None = None
                if call_name in _CONSOLIDATION_WRITE_CALLS:
                    reason = f"filesystem write call ({call_name})"
                elif isinstance(node.func, ast.Attribute) and node.func.attr in _CONSOLIDATION_WRITE_METHODS:
                    reason = f"filesystem write call ({node.func.attr})"
                elif (
                    call_name in {"open", "builtins.open", "Path.open"}
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "open")
                ) and _open_call_is_writable(node):
                    reason = "writable open() mode"
                else:
                    reason = _mutating_command_reason(node, call_name)
                if reason and not _consolidation_side_effect_waived(node, lines):
                    violations.append(
                        f"{_CONSOLIDATION_MONITOR_REL}:{node.lineno}: {reason} — rule "
                        "chain: consolidation monitor -> observability-only -> never "
                        "write/commit/launch. Remove the side effect or add a substantive "
                        f"same-call `# {_CONSOLIDATION_WAIVER}:<reason>` waiver."
                    )

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("top-level JSON value is not an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        settings = {}
        violations.append(
            f"{_CONSOLIDATION_SETTINGS_REL}: unreadable/invalid ({type(exc).__name__}) — "
            "rule chain: proactive cadence -> valid hook config. Restore valid JSON."
        )
    for event in ("SessionStart", "Stop"):
        commands = _hook_commands(settings, event)
        if _CONSOLIDATION_HOOK_COMMAND not in commands:
            violations.append(
                f"{_CONSOLIDATION_SETTINGS_REL}: {event} lost exact non-blocking monitor "
                "wiring — rule chain: regular proactive consolidation -> cadence hook -> "
                f"`{_CONSOLIDATION_HOOK_COMMAND}`. Restore that command; do not add --strict."
            )

    return _finish(
        name="check_consolidation_debt_monitor_observability_and_cadence",
        tag="consolidation-debt-observability-cadence",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail="monitor read-only; SessionStart + Stop --quiet-ok wiring present",
    )


def _load_tool_module(root: Path, filename: str):
    path = root / "tools" / filename
    if not path.is_file():
        return None, f"tools/{filename}: required apparatus helper is missing"
    name = f"_tac_preflight_{filename.replace('.', '_')}_{abs(hash(path))}"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None, f"tools/{filename}: cannot construct import spec"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, None
    except Exception as exc:
        return None, f"tools/{filename}: behavior probe import failed ({type(exc).__name__}: {exc})"


def _main_calls(path: Path, called_name: str) -> bool:
    text = _read(path)
    if text is None:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
            return any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == called_name
                for child in ast.walk(node)
            )
    return False


def check_codex_retry_preserves_original_sandbox_authority(
    *, repo_root: str | Path | None = None, strict: bool = False, verbose: bool = True,
) -> list[str]:
    """STRICT: a same-label relaunch must refuse any sandbox authority downgrade.

    The behavior probe executes the real incident (danger-full-access ->
    workspace-write) and two clean controls.  It also proves ``main`` consumes
    the policy boundary, preventing a correct-but-unwired helper from passing.
    """
    root = Path(repo_root or REPO_ROOT)
    path = root / "tools" / "codex_delegate.py"
    module, error = _load_tool_module(root, "codex_delegate.py")
    violations = [error] if error else []
    policy = getattr(module, "_launch_policy_refusal", None) if module else None
    if module and not callable(policy):
        violations.append("tools/codex_delegate.py: _launch_policy_refusal is missing")
    if callable(policy):
        common = {
            "label": "retry_arm",
            "isolate": True,
            "live_nonisolated_writers": 0,
            "nonisolated_writer_cap": 1,
        }
        downgrade = policy(
            requested_sandbox="workspace-write",
            prior_sandboxes=["danger-full-access"], **common,
        )
        same = policy(
            requested_sandbox="danger-full-access",
            prior_sandboxes=["danger-full-access"], **common,
        )
        upgrade = policy(
            requested_sandbox="danger-full-access",
            prior_sandboxes=["workspace-write"], **common,
        )
        if not downgrade or downgrade[0] == 0:
            violations.append(
                "tools/codex_delegate.py: diagnosed danger-full-access -> workspace-write "
                "same-label relaunch is not refused"
            )
        if same is not None or upgrade is not None:
            violations.append(
                "tools/codex_delegate.py: sandbox preservation gate rejects a same/elevated clean control"
            )
    if not _main_calls(path, "_launch_policy_refusal"):
        violations.append(
            "tools/codex_delegate.py: main() does not call _launch_policy_refusal (unwired guard)"
        )
    retry_helper, retry_error = _load_tool_module(root, "codex_retry_checkpoint.py")
    if retry_error:
        violations.append(retry_error)
    latest = getattr(retry_helper, "latest_resumable_checkpoint", None)
    retry_cap = getattr(module, "_MAX_CAPACITY_RETRIES", None) if module else None
    if not isinstance(retry_cap, int) or isinstance(retry_cap, bool) or not 1 <= retry_cap <= 2:
        violations.append(
            "tools/codex_delegate.py: transient retry cap must be a small positive bound (1..2)"
        )
    if callable(latest):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.jsonl"
            progress.write_text(
                '\n'.join(
                    [
                        '{"parent_id_or_session":"other","status":"in_progress",'
                        '"step":99,"next_action":"wrong"}',
                        '{"parent_id_or_session":"codex_delegate:arm:stamp",'
                        '"status":"complete","step":1,"next_action":""}',
                        '{"parent_id_or_session":"codex_delegate:arm:stamp",'
                        '"status":"in_progress","step":2,"next_action":"continue"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            selected = latest(progress, "codex_delegate:arm:stamp")
            refused = latest(progress, "codex_delegate:missing:stamp")
        if not isinstance(selected, dict) or selected.get("step") != 2 or refused is not None:
            violations.append(
                "tools/codex_retry_checkpoint.py: exact-key resumable checkpoint custody probe failed"
            )
    elif retry_helper:
        violations.append(
            "tools/codex_retry_checkpoint.py: latest_resumable_checkpoint is missing"
        )
    compact = getattr(module, "_write_compact_prompts", None) if module else None
    launcher_writer = getattr(module, "_write_launcher", None) if module else None
    if callable(compact) and callable(launcher_writer):
        old_runs = module.RUNS
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                module.RUNS = tmp_path
                authority = tmp_path / "wrapped.prompt.txt"
                authority.write_text("x" * 200_000, encoding="utf-8")
                entry, resume = compact(
                    label="arm",
                    stamp="stamp",
                    wrapped_prompt=authority,
                    delegation_key="codex_delegate:arm:stamp",
                )
                body = launcher_writer(
                    "arm", "stamp", entry, "gpt-5.6-sol", "high", "read-only",
                    tmp_path / "run.log", tmp_path / "last.txt", tmp_path / "done",
                    False, tmp_path, resume_prompt_file=resume,
                    delegation_key="codex_delegate:arm:stamp",
                ).read_text(encoding="utf-8")
                bounded = entry.stat().st_size < 2_000 and resume.stat().st_size < 2_000
        finally:
            module.RUNS = old_runs
        required = (
            "codex_retry_checkpoint.py",
            "--progress-file \"$WORKDIR/.omx/state/subagent_progress.jsonl\"",
            "RETRY-REFUSED-NO-CHECKPOINT",
            str(resume),
            "ATTEMPT_LOG=",
            'grep -qiE',
            '"$ATTEMPT_LOG"; do',
            ': > "$ATTEMPT_LOG"',
            "2>&1 | tee",
            '"$ATTEMPT_LOG"',
            "tail -c 400",
            "LANDING-REVIEW-REQUIRED",
            "review_required=1",
        )
        attempt_output_isolated = bool(
            re.search(r'grep -qiE [^\n]* "\$ATTEMPT_LOG"; do', body)
            and re.search(r'tee -a [^\n]* "\$ATTEMPT_LOG"', body)
        )
        if (
            not bounded
            or not attempt_output_isolated
            or any(token not in body for token in required)
        ):
            violations.append(
                "tools/codex_delegate.py: compact checkpoint-custodied retry or landing-review guard is missing"
            )
    elif module:
        violations.append(
            "tools/codex_delegate.py: compact prompt/retry launcher behavior is missing"
        )
    return _finish(
        name="check_codex_retry_preserves_original_sandbox_authority",
        tag="codex-retry-sandbox-authority",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail="sandbox preserved; retries bounded, compact, checkpoint-custodied, and review-gated",
    )


def check_codex_nonisolated_writer_cap(
    *, repo_root: str | Path | None = None, strict: bool = False, verbose: bool = True,
) -> list[str]:
    """STRICT: cap live non-isolated writer arms while exempting safe domains."""
    root = Path(repo_root or REPO_ROOT)
    path = root / "tools" / "codex_delegate.py"
    module, error = _load_tool_module(root, "codex_delegate.py")
    violations = [error] if error else []
    policy = getattr(module, "_launch_policy_refusal", None) if module else None
    cap = getattr(module, "_MAX_NONISOLATED_WRITERS", None) if module else None
    if not isinstance(cap, int) or cap < 1 or cap > 2:
        violations.append(
            "tools/codex_delegate.py: _MAX_NONISOLATED_WRITERS must record a small positive cap (1..2)"
        )
    if callable(policy) and isinstance(cap, int):
        common = {
            "label": "writer",
            "prior_sandboxes": [],
            "live_nonisolated_writers": cap,
            "nonisolated_writer_cap": cap,
        }
        writer = policy(requested_sandbox="workspace-write", isolate=False, **common)
        isolated = policy(requested_sandbox="danger-full-access", isolate=True, **common)
        readonly = policy(requested_sandbox="read-only", isolate=False, **common)
        if not writer or writer[0] == 0:
            violations.append(
                "tools/codex_delegate.py: at-cap --no-isolate workspace-write arm is not refused"
            )
        if isolated is not None or readonly is not None:
            violations.append(
                "tools/codex_delegate.py: writer cap does not exempt isolated/read-only clean controls"
            )
    elif module:
        violations.append("tools/codex_delegate.py: callable _launch_policy_refusal is missing")
    if not _main_calls(path, "_launch_policy_refusal"):
        violations.append(
            "tools/codex_delegate.py: main() does not call _launch_policy_refusal (unwired cap)"
        )
    source = _read(path) or ""
    if "shared-tree fallback (NO isolation)" in source or "no shared-tree writer fallback" not in source:
        violations.append(
            "tools/codex_delegate.py: worktree setup may fall back to an uncapped shared-tree writer"
        )
    status_module, status_error = _load_tool_module(root, "codex_status.py")
    if status_error:
        violations.append(status_error)
    is_doomed = getattr(status_module, "_is_strand_doomed", None) if status_module else None
    bucket = getattr(status_module, "_bucket", None) if status_module else None
    if callable(is_doomed) and callable(bucket):
        legacy_writer = {"sandbox": "workspace-write"}
        if not is_doomed(legacy_writer):
            violations.append(
                "tools/codex_status.py: pre-CFL writer without isolate custody is not strand-doomed"
            )
        if is_doomed({"sandbox": "workspace-write", "isolate": True}) or is_doomed(
            {"sandbox": "read-only", "isolate": False}
        ):
            violations.append(
                "tools/codex_status.py: strand-doomed retrofit rejects isolated/read-only controls"
            )
        if bucket(True, None, None, 0.1, strand_doomed=True) != "STRAND_DOOMED":
            violations.append(
                "tools/codex_status.py: live pre-CFL writer is not surfaced as STRAND_DOOMED"
            )
    elif status_module:
        violations.append(
            "tools/codex_status.py: pre-CFL strand-doomed behavior is missing"
        )
    return _finish(
        name="check_codex_nonisolated_writer_cap",
        tag="codex-nonisolated-writer-cap",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"shared-tree writer cap={cap}; pre-CFL writers retrofitted STRAND_DOOMED",
    )


def check_codex_drain_timeout_uses_liveness(
    *, repo_root: str | Path | None = None, strict: bool = False, verbose: bool = True,
) -> list[str]:
    """STRICT: a timed-out but progressing arm must not be classified WEDGED."""
    root = Path(repo_root or REPO_ROOT)
    path = root / "tools" / "codex_drain_detector.py"
    module, error = _load_tool_module(root, "codex_drain_detector.py")
    violations = [error] if error else []
    classify = getattr(module, "classify_timeout", None) if module else None
    exit_code = getattr(module, "exit_code_for_status", None) if module else None
    if callable(classify):
        baseline = {
            "arm_s": {"label": "arm", "log_mtime": 100.0, "progress_cursor": 7}
        }
        fresh_log = {
            "arm_s": {"label": "arm", "log_mtime": 995.0, "progress_cursor": 7}
        }
        advanced_progress = {
            "arm_s": {"label": "arm", "log_mtime": 100.0, "progress_cursor": 8}
        }
        stale = {
            "arm_s": {"label": "arm", "log_mtime": 100.0, "progress_cursor": 7}
        }
        fresh_status, _ = classify(
            baseline, fresh_log, now=1000.0, liveness_window_seconds=60.0
        )
        progress_status, _ = classify(
            baseline, advanced_progress, now=1000.0, liveness_window_seconds=60.0
        )
        stale_status, _ = classify(
            baseline, stale, now=1000.0, liveness_window_seconds=60.0
        )
        timeout_status = getattr(module, "TIMED_OUT", object())
        if fresh_status != timeout_status:
            violations.append(
                "tools/codex_drain_detector.py: recent-log timeout control raises a stuck classification"
            )
        if progress_status != timeout_status:
            violations.append(
                "tools/codex_drain_detector.py: advancing-progress timeout control raises a stuck classification"
            )
        if stale_status != getattr(module, "WEDGED", object()):
            violations.append(
                "tools/codex_drain_detector.py: genuine stale/no-progress control is not WEDGED"
            )
        strand_doomed = {
            "arm_s": {
                "label": "arm", "log_mtime": 995.0, "progress_cursor": 8,
                "strand_doomed": True,
            }
        }
        doomed_status, _ = classify(
            baseline, strand_doomed, now=1000.0, liveness_window_seconds=60.0
        )
        if doomed_status != getattr(module, "WEDGED", object()):
            violations.append(
                "tools/codex_drain_detector.py: pre-CFL strand-doomed control is not WEDGED"
            )
        if not callable(exit_code):
            violations.append("tools/codex_drain_detector.py: exit_code_for_status is missing")
        elif (
            exit_code(fresh_status) != 2
            or exit_code(progress_status) != 2
            or exit_code(stale_status) != 3
        ):
            violations.append(
                "tools/codex_drain_detector.py: TIMED_OUT must exit 2 and WEDGED must exit 3"
            )
    elif module:
        violations.append("tools/codex_drain_detector.py: classify_timeout is missing")
    if not _main_calls(path, "classify_timeout"):
        violations.append(
            "tools/codex_drain_detector.py: main() does not call classify_timeout (unwired liveness gate)"
        )
    if not _main_calls(path, "exit_code_for_status"):
        violations.append(
            "tools/codex_drain_detector.py: main() does not use exit_code_for_status (unwired alarm policy)"
        )
    return _finish(
        name="check_codex_drain_timeout_uses_liveness",
        tag="codex-drain-timeout-liveness",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail="all timeouts exit nonzero; stale/strand-doomed controls are wedged",
    )


# ===========================================================================
# 2026-07-15 follow-on — H1/H2/H3 confound self-protection. All three fixes
# and their gates land atomically at live-count zero, so preflight runs STRICT.
# ===========================================================================


def check_witness_trainers_emit_partial_freeze_alarm(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Refuse a witness trainer with no L1 ``partial_freeze`` open-band alarm."""

    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    for path in _existing_trainers(root):
        code = _strip_comments(_read(path) or "")
        rel = path.relative_to(root).as_posix()
        alarm_pos = code.find('"partial_freeze"')
        if alarm_pos < 0:
            violations.append(f"{rel}: missing typed partial_freeze alarm")
            continue
        window = code[max(0, alarm_pos - 1800): alarm_pos + 1200]
        direct_band = "0.02" in window and "0.5" in window and "accepted_frac" in window
        helper_band = "is_partial_freeze" in window
        if not (direct_band or helper_band):
            violations.append(
                f"{rel}: partial_freeze alarm is not wired to 0.02 < accepted_frac < 0.5"
            )
    helper = root / "src" / "tac" / "confound_observability.py"
    helper_code = _strip_comments(_read(helper) or "")
    if helper.is_file() and not all(
        token in helper_code
        for token in ("PARTIAL_FREEZE_LO = 0.02", "PARTIAL_FREEZE_HI = 0.5", "< frac <")
    ):
        violations.append(f"{helper.relative_to(root).as_posix()}: partial-freeze open band drifted")
    return _finish(
        name="check_witness_trainers_emit_partial_freeze_alarm",
        tag="partial-freeze-alarm",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail="all witness trainers carry the exact open-band alarm",
    )


def check_witness_verdict_rows_carry_dseg_descent_canary(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Refuse missing run-global d_seg canary, row stamps, or L3 clearance."""

    root = Path(repo_root or REPO_ROOT)
    path = root / TRAINER_REL
    violations: list[str] = []
    if path.is_file():
        code = _strip_comments(_read(path) or "")
        rel = path.relative_to(root).as_posix()
        required = (
            "dseg_descent_canary_setup",
            "dseg_descent_canary_passed",
            "dseg_descent_positive_control_registered",
            "dseg_verdict_clearance",
            "canary_suite",
            "verdict_clearance()",
        )
        missing = [token for token in required if token not in code]
        if missing:
            violations.append(f"{rel}: missing d_seg canary wiring tokens: {', '.join(missing)}")
        # Definition + setup row + baseline + async + sync verdict paths.
        if code.count("_dseg_canary_telemetry_fields()") < 5:
            violations.append(
                f"{rel}: d_seg canary stamp does not cover setup, baseline, async, and sync rows"
            )
    return _finish(
        name="check_witness_verdict_rows_carry_dseg_descent_canary",
        tag="dseg-descent-canary",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail="global canary and all verdict row paths are wired",
    )


def check_verdict_live_gap_defaults_on_during_ema_warmup(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Refuse the H2 orphan: live-gap default OFF or no EMA-warmup cadence clock."""

    root = Path(repo_root or REPO_ROOT)
    path = root / TRAINER_REL
    violations: list[str] = []
    if path.is_file():
        text = _read(path) or ""
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            violations.append(f"{rel}: cannot parse trainer: {exc}")
        else:
            defaults: list[object] = []
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and _call_is_add_argument(node)
                    and _first_str_positional(node) == "--verdict-live-gap-every"
                ):
                    default_node = next(
                        (kw.value for kw in node.keywords if kw.arg == "default"), None
                    )
                    try:
                        defaults.append(ast.literal_eval(default_node))
                    except (TypeError, ValueError):
                        defaults.append(_MISSING)
            if defaults != [-1]:
                violations.append(
                    f"{rel}: --verdict-live-gap-every must default to -1 auto-warmup, got {defaults!r}"
                )
        code = _strip_comments(text)
        required = (
            "verdict_live_gap_due(",
            "ema_warmup_updates(",
            '_live["ema_updates"] += 1',
            "VERDICT_LIVE_GAP_AUTO_WARMUP",
        )
        missing = [token for token in required if token not in code]
        if missing:
            violations.append(f"{rel}: missing live-gap warmup wiring: {', '.join(missing)}")
        # Definition + async scheduler + sync verdict path.
        if code.count("_verdict_live_gap_is_due") < 3:
            violations.append(f"{rel}: live-gap warmup predicate is not wired to async and sync verdicts")
    dsl = root / "src" / "tac" / "witness_dsl" / "curriculum_dsl.py"
    dsl_code = _strip_comments(_read(dsl) or "")
    if dsl.is_file() and not all(
        token in dsl_code for token in ("def VerdictLiveGap(", '"--verdict-live-gap-every"')
    ):
        violations.append(f"{dsl.relative_to(root).as_posix()}: VerdictLiveGap DSL lever is missing")
    return _finish(
        name="check_verdict_live_gap_defaults_on_during_ema_warmup",
        tag="verdict-live-gap-warmup",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail="auto-warmup default, accepted-update clock, and both verdict paths are wired",
    )


def _python_source_files(root: Path) -> list[Path]:
    """The DECLARED memory-guard surface, as an explicit denominator.

    ``tools/*.py`` + ``src/tac/**/*.py`` + ``experiments/*.py`` + ``scripts/*.py``, skipping the
    canonical accounting modules, tests, and this gate module itself.

    ddm_gh1 #830: ``experiments/`` and ``scripts/`` were previously OUT of scope silently — a
    future guard written there would have been invisible while the gate still printed ``OK``.
    They are scanned TOP-LEVEL ONLY (deliberately non-recursive): ``experiments/results/**`` and
    ``experiments/**/.venv/**`` hold vendored third-party trees and frozen run bundles that are
    not our guard surface and would swamp the denominator (MEASURED: 20+ vendored numpy/torch
    files match ``virtual_memory``). ``tools/`` stays non-recursive because its only subdirectory
    is ``tools/tests`` (116 files), which the ``/tests/`` filter excludes anyway. The scope is a
    declaration, not an accident — the caller reports ``considered`` next to ``scanned`` so
    "0 violations" can never be confused with "0 scanned".
    """
    out: list[Path] = []
    for relative, pattern in (
        ("tools", "*.py"),
        ("src/tac", "**/*.py"),
        ("experiments", "*.py"),
        ("scripts", "*.py"),
    ):
        directory = root.joinpath(*relative.split("/"))
        if directory.is_dir():
            out.extend(sorted(directory.glob(pattern)))
    skip = {
        (root / "tools" / "mem_basis.py"),
        (root / "tools" / "system_memory_governor.py"),
        (root / "src" / "tac" / "confound_gates.py"),
    }
    return [p for p in out if p not in skip and "/tests/" not in p.as_posix()]


def _vm_safety_attr_nodes(tree: ast.AST) -> list[ast.Attribute]:
    """``psutil.virtual_memory().{available,used,free}`` attribute accesses (the reclaimable-blind
    safety-basis pattern). ``.total`` is EXCLUDED — it is a denominator, never a refuse/admit basis."""
    found: list[ast.Attribute] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr not in {"available", "used", "free"}:
            continue
        base = node.value
        if (
            isinstance(base, ast.Call)
            and isinstance(base.func, ast.Attribute)
            and base.func.attr == "virtual_memory"
        ):
            found.append(node)
    return found


def check_no_raw_virtual_memory_safety_basis(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """CLASS 1 (bug-class sweep 2026-07-17) — refuse ``psutil.virtual_memory().{available,used,free}``
    as a memory safety basis outside the canonical reclaimable-aware helper (``tools/mem_basis.py``)
    / accounting (``tools/system_memory_governor.py``).

    Bug class (operator P0; sisters ``admission_gate_naive_counts_reclaimable_as_committed_20260716``
    + ``tunnel_always_up_supervisor_canonical_20260717``): on macOS ``.available`` = free + inactive
    counts DIRTY ANON parked in the inactive queue as available (measured 57.3 GiB "available" vs
    13.7 GiB truly reclaimable-without-swap next to the live trainer) → refuse/admit guards
    UNDER-protect; ``.used`` counts reclaimable file-cache as committed → over-refuse an idle box.
    Route guards through ``tools.mem_basis.conservative_free_gib`` / ``true_committed_gib``.

    ``.total`` is EXCLUDED (denominator, not a safety basis). Same-line waiver
    ``# RAW_VM_BASIS_OK:<rationale>`` for telemetry-only display / last-resort fallbacks.

    STRICT since 2026-07-31 (ddm_gh1 #830 fix+gate atomic landing). This gate was OWNERLESS and
    RED: declared live-count 0 in its own docstring but MEASURED 6. Re-derived and classified:
    3 in ``tools/launch_ddm_joint_descent.py``'s ``_RSSMonitor`` are TELEMETRY-ONLY (the value
    only ever reaches the ``measured_free_memory_floor_gib`` receipt field — 3 emit sites, never
    compared, never refuses) and are waived with a MEASURED rationale (20 Hz poll; the canonical
    basis costs 45 ms/call vs psutil's 0.014 ms = 3200x, so routing it there would burn ~90% of a
    core beside the live trainer); the other 3 (``remeasure_ddm_e4_ws1_packet``,
    ``run_ddm_j12_receiver_coordinate_custody``, ``run_ddm_ms2r_r3_366box_typed_fisher_g4_waterfill``)
    genuinely REFUSE or publish a ``memory_preflight`` threshold relation and were routed through
    ``tools.mem_basis.conservative_free_gib`` with a fail-closed default. Live evidence for the
    bug class on this box at fix time: psutil ``.available`` 90.55 GiB vs truly reclaimable
    87.29 GiB — a 3.3 GiB over-report in the UNDER-protecting direction. Live count is 0, so it
    flips strict in this landing rather than into warn-only purgatory."""
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    considered = 0
    scanned = 0
    for path in _python_source_files(root):
        considered += 1
        text = _read(path)
        if not text or "virtual_memory" not in text:
            continue
        scanned += 1
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        rel = path.relative_to(root).as_posix()
        for node in _vm_safety_attr_nodes(tree):
            line_text = lines[node.lineno - 1] if 0 <= node.lineno - 1 < len(lines) else ""
            if _waiver_present(line_text, "RAW_VM_BASIS_OK"):
                continue
            violations.append(
                f"{rel}:{node.lineno}: raw psutil.virtual_memory().{node.attr} as a memory safety "
                f"basis (reclaimable-blind on macOS). Route through "
                f"tools.mem_basis.conservative_free_gib / true_committed_gib, or add a same-line "
                f"`# RAW_VM_BASIS_OK:<rationale>` waiver (telemetry / last-resort fallback)."
            )
    return _finish(
        name="check_no_raw_virtual_memory_safety_basis",
        tag="raw-vm-safety-basis",
        violations=violations,
        strict=strict,
        verbose=verbose,
        # DECLARED DENOMINATOR (ddm_gh1 class fix): `considered` is the in-scope surface,
        # `scanned` the subset that mentions the token. Reporting only the latter lets a
        # narrowed scope print a clean OK while scanning almost nothing.
        ok_detail=(
            f"{scanned} of {considered} in-scope source file(s) mention virtual_memory "
            f"(scope: tools/*.py + src/tac/**/*.py + experiments/*.py + scripts/*.py)"
        ),
    )


# Ways a guard enumerates the live process table. `ps` + `splitlines` is the form that made the
# #829 slot guards invisible to this gate; keep this list as the ONE place the surface is declared.
# Each marker is DELIMITED on purpose: the first draft used a bare `"pgrep"` and matched the
# identifier `pgreport` in an unrelated probe — a substring false positive inside the very gate
# written to extinct substring false positives. Delimiters, not bare tokens.
_CLASS2_ENUMERATION_MARKERS = (
    "cmdline", "process_iter", "-axo", "ps -ax", "pgrep ", '"pgrep"', "'pgrep'",
)

_CLASS2_TRAINER_TOKENS = ("train_levelset_witness", "train_witness")
_CLASS2_DECISION_TOKENS = (
    "killpg", "os.kill", "sys.exit", "SystemExit", "return 12", "return 1",
    "refuse", "REFUS", "pkill",
)
_CLASS2_EXCLUSION_MARKERS = (
    "strip_observer_flag_values", "argv_role", "--training-sig", "OBSERVER_ROLE_OK",
    "TRAINING_SIG",
)


def check_process_guard_excludes_observer_flag_values(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """CLASS 2 (bug-class sweep 2026-07-17) — a fail-closed process guard that classifies a live
    process by TRAINER-TOKEN presence in its joined cmdline AND makes a refuse/kill decision must
    exclude OBSERVER flag values (``--training-sig <trainer-name>``) first.

    Bug class (sisters #406 rc=8 supervisor false-refuse + p0_512 same-outdir guard): observers carry
    the trainer NAME as a flag VALUE, so ``trainer_token in " ".join(cmdline)`` misclassifies a
    monitor as a launch → refuses the always-on tunnel / a legit launch. Cure = structural
    ``tools.argv_role.strip_observer_flag_values`` before the token test (raw trainer argv still
    caught). FM role-classification, where used, is an advisory tiebreaker ON TOP of this — never a
    replacement.

    Function-scoped: flags a FunctionDef whose body (a) enumerates cmdlines (``process_iter`` or
    ``cmdline`` + ``.join``), (b) tests a trainer-token literal, (c) makes a decision
    (kill/exit/refuse), and (d) carries no exclusion marker (``strip_observer_flag_values`` /
    ``argv_role`` / ``--training-sig`` handling / ``TRAINING_SIG`` / ``# OBSERVER_ROLE_OK:<why>``).

    WARN-ONLY at landing (live-count 0 after the #512 fix; not in ``_CONFOUND_STRICT``)."""
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    tools = root / "tools"
    files = sorted(tools.glob("*.py")) if tools.is_dir() else []
    for path in files:
        if path.name in {"argv_role.py"}:
            continue
        text = _read(path)
        # ddm_gh1 #829 SCOPE FIX: the prefilter previously required `cmdline` or `process_iter`,
        # so the whole `subprocess.run(["ps", "-axo", "command"])` enumeration family was skipped
        # at the FILE level — the gate printed OK while two live slot guards (ru1, sb1) carried
        # exactly this bug class. A gate that silently scans a fraction of its intended surface is
        # the same defect one level out.
        if not text or not any(marker in text for marker in _CLASS2_ENUMERATION_MARKERS):
            continue
        if not any(tok in text for tok in _CLASS2_TRAINER_TOKENS):
            continue
        scanned += 1
        file_hit = False
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            span = "\n".join(lines[node.lineno - 1: (node.end_lineno or node.lineno)])
            classifies = (
                ("process_iter" in span)
                or ("cmdline" in span and ".join" in span)
                # ddm_gh1 #829: the `ps -axo command` form. `splitlines()` over ps output is the
                # same act of enumerating the process table by text.
                or ('"ps"' in span and "splitlines" in span)
                or ("'ps'" in span and "splitlines" in span)
            )
            if not classifies:
                continue
            if not any(tok in span for tok in _CLASS2_TRAINER_TOKENS):
                continue
            if not any(d in span for d in _CLASS2_DECISION_TOKENS):
                continue
            if any(e in span for e in _CLASS2_EXCLUSION_MARKERS):
                continue
            violations.append(
                f"{rel}:{node.lineno}: {node.name}() classifies a process by trainer-token in its "
                f"joined cmdline AND makes a refuse/kill decision without excluding observer flag "
                f"values. Strip via tools.argv_role.strip_observer_flag_values (mirrors #406/#512), "
                f"or add a `# OBSERVER_ROLE_OK:<rationale>` waiver."
            )
            file_hit = True
        # ddm_gh1 #829 SPLIT-ACROSS-FUNCTIONS fallback. The function-scoped predicate requires
        # enumeration + token + decision in ONE body. The measured ru1/sb1 defect split them: a
        # `slot_is_live()` helper enumerated and token-tested, a separate caller refused, and the
        # token tuple lived at module scope — so all three legs were individually invisible. If a
        # candidate file carries all three legs anywhere and no exclusion marker, flag the FILE.
        if (
            not file_hit
            and not any(e in text for e in _CLASS2_EXCLUSION_MARKERS)
            and any(d in text for d in _CLASS2_DECISION_TOKENS)
        ):
            violations.append(
                f"{rel}: enumerates the process table and refuses on trainer-token presence, "
                f"with the enumeration, the token test, and the decision SPLIT across "
                f"functions/module scope. Route the classification through "
                f"tools.argv_role (cmdline_names_entrypoint / "
                f"process_table_entrypoint_holders), or add a "
                f"`# OBSERVER_ROLE_OK:<rationale>` waiver."
            )
    return _finish(
        name="check_process_guard_excludes_observer_flag_values",
        tag="observer-flag-exclusion",
        violations=violations,
        strict=strict,
        verbose=verbose,
        # DECLARED DENOMINATOR (ddm_gh1 class fix).
        ok_detail=(
            f"{scanned} of {len(files)} in-scope tools/*.py enumerate the process table and name "
            f"a trainer token"
        ),
    )


# ===========================================================================
# 2026-07-18 — NAME-ANCHORED-SEARCH / duplicate-SoT gate
# ===========================================================================

_SPEC_NAME_RE = re.compile(r"(?i)^spec[_\-. ]?v(\d[\d.]*[a-z]?\d*)")
# Heading identity requires SPEC and the vehicle token to be ADJACENT (either
# order, separators only) — '# SPEC_v10 — ...' / '# SPEC — v7.5 ...' / '# v8
# SPEC' claim to BE the vehicle spec; a qualified title like '# v7.5
# OPTIMAL-FORM ACTUATION SPEC' (a companion checklist, measured false-positive
# 2026-07-18) does not.
_SPEC_HEADING_ADJ_RE = re.compile(
    r"(?i)\bspec\b[\s:_\-—–.]*v(\d[\d.]*[a-z]?\d*)"
    r"|v(\d[\d.]*[a-z]?\d*)[\s:_\-—–.]*\bspec\b"
)
_DUP_SOT_WAIVER = "DUPLICATE_SOT_OK"
_DUP_SOT_MAX_BLOB_BYTES = 1_000_000  # skip huge md (e.g. the 2.9MB DAG) in blob fetch


def _norm_vehicle_key(raw: str) -> str:
    """Normalize a vehicle token so 'v7.5' == 'v75' == '7.5' -> '75'."""
    return re.sub(r"[^0-9a-z]", "", raw.lower().lstrip("v"))


def _dup_git(args: list[str], cwd: Path) -> str:
    """git plumbing for the duplicate-SoT gate ('' on error/no-match)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _dup_waiver_ok(text: str) -> bool:
    """Markdown-aware ``# DUPLICATE_SOT_OK:<rationale>`` waiver check.

    Docs carry the waiver inside HTML comments (``<!-- # DUPLICATE_SOT_OK:why
    -->``); strip a trailing ``-->`` before the placeholder-rationale test so
    the literal ``<rationale>`` placeholder cannot self-waive."""
    rx = re.compile(r"#[ \t]*" + re.escape(_DUP_SOT_WAIVER) + r":[ \t]*(\S.*)")
    for m in rx.finditer(text):
        rationale = m.group(1)
        if rationale.rstrip().endswith("-->"):
            rationale = rationale.rstrip()[:-3]
        if _rationale_ok(rationale):
            return True
    return False


def _dup_first_heading(blob: str) -> str:
    for ln in blob.splitlines()[:40]:
        if ln.lstrip().startswith("#"):
            return ln
    return ""


def _dup_batch_heading_keys(root: Path, oids: list[str]) -> dict[str, str | None]:
    """oid -> first-heading vehicle key for many blobs via ONE
    ``git cat-file --batch`` stream (blobs over the size cap map to None)."""
    result: dict[str, str | None] = {}
    if not oids:
        return result
    try:
        proc = subprocess.Popen(
            ["git", "cat-file", "--batch"],
            cwd=str(root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        stdin_payload = ("\n".join(oids) + "\n").encode()
        out, _ = proc.communicate(input=stdin_payload, timeout=300)
    except (OSError, subprocess.TimeoutExpired, AssertionError):
        return dict.fromkeys(oids)
    pos = 0
    for oid in oids:
        nl = out.find(b"\n", pos)
        if nl < 0:
            result[oid] = None
            continue
        header = out[pos:nl].decode("utf-8", "replace").split()
        pos = nl + 1
        if len(header) != 3 or header[1] != "blob":
            # "<oid> missing" — no trailing content record.
            result[oid] = None
            continue
        try:
            size = int(header[2])
        except ValueError:
            result[oid] = None
            continue
        body = out[pos: pos + size]
        pos += size + 1  # skip the record's trailing newline
        if size > _DUP_SOT_MAX_BLOB_BYTES:
            result[oid] = None
            continue
        head_text = body[:4096].decode("utf-8", "replace")
        result[oid] = _dup_heading_vehicle_key(_dup_first_heading(head_text))
    return result


def _dup_heading_vehicle_key(heading: str) -> str | None:
    """The vehicle key a spec-shaped FIRST HEADING claims, else None.

    A heading is spec-shaped when the literal token 'SPEC' sits ADJACENT to a
    v<digits> vehicle token (either order) — this is the CONTENT identity test
    that makes the gate immune to the creator's filename choice."""
    m = _SPEC_HEADING_ADJ_RE.search(heading)
    if not m:
        return None
    return _norm_vehicle_key(m.group(1) or m.group(2) or "")


def check_no_duplicate_canonical_spec_across_refs(
    *,
    repo_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
    max_refs: int = 64,
) -> list[str]:
    """NAME-ANCHORED-SEARCH / duplicate-SoT gate (2026-07-18) — refuses a repo
    state that carries a canonical-spec-shaped doc for a vehicle that ALREADY
    has a same-vehicle spec at a DIFFERENT path on ANY git ref or in the
    canonical doc registry.

    ROOT CAUSE this extincts (operator 2026-07-18, verbatim): "You searched,
    but you searched for what you would have named it. You didn't do an
    exhaustive search." An agent globbed for the v10 spec under its OWN naming
    convention (``*optimal_cold_start_capstone*``), missed the canonical
    ``SPEC_v10_capstone_cold_start_seeded_20260717.md`` on the UNMERGED branch
    ``claude/p0_521_spec_v10_capstone_20260717``, and created a duplicate.
    Name-anchored, main-scoped search is structurally blind to same-content
    docs under other names / on other branches. Memory:
    ``vehicle_naming_v9c_warm_lineage_v10_reserved_capstone_20260718.md``.
    Sisters: the config-orphan confound
    (``[[config_orphan_confound_permanent_fix_lever_registry_20260706]]``) +
    velocity-driven orphaning
    (``[[velocity_driven_orphaning_the_deepest_signal_loss_meta_bug]]``).

    Mechanics (NOT name-anchored, NOT main-only — by construction):
      * scope: working-tree ``.omx/research/**/*.md`` whose BASENAME is
        spec-shaped (``SPEC_v<N>...``) — each extracts a normalized vehicle key
        (v7.5 == v75);
      * registry leg: ``.omx/state/canonical_doc_registry.json`` entries with
        the same vehicle key whose ``canonical_path`` differs -> duplicate;
      * all-refs NAME leg: ``git for-each-ref refs/heads refs/remotes`` deduped
        by commit, then ``git ls-tree -r <commit> -- .omx/research`` — a
        spec-named doc with the same key at a different path -> duplicate;
      * all-refs CONTENT-FAMILY leg: EVERY md blob under ``.omx/research`` on
        EVERY ref is inspected (one ``git cat-file --batch`` stream over the
        unique blob OIDs); a doc whose FIRST HEADING claims the same vehicle
        (``SPEC`` + ``v<key>``) at a different path -> duplicate REGARDLESS of
        its filename (this is the leg a name-anchored glob can never perform,
        and it is exhaustive by construction — no grep pattern to get wrong);
      * working-tree leg: two on-disk spec docs sharing one vehicle key ->
        duplicate.

    Waiver: ``# DUPLICATE_SOT_OK:<rationale>`` inside the doc (non-placeholder
    rationale per Catalog #287). Pre-create dedup lives in
    ``tools/canonical_doc_registry.py`` (``check_before_create``); this gate is
    the second landing per "Bugs must be permanently fixed AND self-protected
    against".

    WARN-ONLY at landing per the Strict-flip atomicity rule (live count 0 at
    landing — the strict flip is owed once a full-refs sweep re-confirms 0 on
    the primary checkout)."""
    root = Path(repo_root or REPO_ROOT)
    research = root / ".omx" / "research"
    violations: list[str] = []

    wt_specs: list[tuple[Path, str]] = []
    if research.is_dir():
        for p in sorted(research.rglob("*.md")):
            m = _SPEC_NAME_RE.match(p.name)
            if m:
                wt_specs.append((p, _norm_vehicle_key(m.group(1))))
    if not wt_specs:
        return _finish(
            name="check_no_duplicate_canonical_spec_across_refs",
            tag="duplicate-sot-across-refs",
            violations=violations,
            strict=strict,
            verbose=verbose,
            ok_detail="no spec-shaped docs in working tree",
        )

    reg_path = Path(
        registry_path
        if registry_path is not None
        else root / ".omx" / "state" / "canonical_doc_registry.json"
    )
    reg_entries: list[dict] = []
    if reg_path.is_file():
        try:
            data = json.loads(reg_path.read_text(encoding="utf-8"))
            rows = data.get("entries", []) if isinstance(data, dict) else data
            reg_entries = [r for r in rows if isinstance(r, dict)]
        except (OSError, ValueError):
            reg_entries = []

    # ALL refs (heads + remotes), deduped by commit object — the structural
    # cure for main-scoped search.
    commits: list[tuple[str, str]] = []
    seen_sha: set[str] = set()
    for line in _dup_git(
        [
            "for-each-ref",
            "--format=%(objectname) %(refname:short)",
            "refs/heads",
            "refs/remotes",
        ],
        root,
    ).splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0] not in seen_sha:
            seen_sha.add(parts[0])
            commits.append((parts[0], parts[1]))
            if len(commits) >= max_refs:
                break

    # Enumerate every md doc under .omx/research on every unique ref commit
    # (path + blob oid), then compute each UNIQUE blob's first-heading vehicle
    # key via ONE streamed `git cat-file --batch` pass. This makes the
    # content leg EXHAUSTIVE (every blob inspected — no grep pattern to get
    # silently wrong; an earlier grep-based leg died on a POSIX-ERE '\-'
    # "invalid character range" and was caught by
    # test_content_family_catch_without_spec_filename) while staying fast:
    # identical bytes shared across the 20+ refs are read exactly once.
    ref_entries_by_sha: dict[str, list[tuple[str, str]]] = {}
    for sha, _refname in commits:
        entries: list[tuple[str, str]] = []
        out = _dup_git(["ls-tree", "-r", sha, "--", ".omx/research"], root)
        for ln in out.splitlines():
            # format: <mode> <type> <oid>\t<path>
            meta, _, path = ln.partition("\t")
            if not path.endswith(".md"):
                continue
            parts = meta.split()
            if len(parts) == 3 and parts[1] == "blob":
                entries.append((path, parts[2]))
        ref_entries_by_sha[sha] = entries

    unique_oids: list[str] = []
    seen_oid: set[str] = set()
    for entries in ref_entries_by_sha.values():
        for _path, oid in entries:
            if oid not in seen_oid:
                seen_oid.add(oid)
                unique_oids.append(oid)
    heading_key_by_oid = _dup_batch_heading_keys(root, unique_oids)

    # Working-tree pairwise duplicates (two on-disk specs, one vehicle).
    by_key: dict[str, list[str]] = {}
    for p, key in wt_specs:
        by_key.setdefault(key, []).append(p.relative_to(root).as_posix())
    for key, paths in by_key.items():
        if len(paths) > 1:
            waived = []
            for rp in paths:
                text = _read(root / rp) or ""
                if _dup_waiver_ok(text):
                    waived.append(rp)
            live = [rp for rp in paths if rp not in waived]
            if len(live) > 1:
                violations.append(
                    f"{live[1]}: duplicate canonical spec for vehicle v{key} — "
                    f"{live[0]} already exists in the working tree. Rule chain: "
                    f"NAME-ANCHORED-SEARCH bug class -> one spec per vehicle -> "
                    f"fold into the existing doc (tools/canonical_doc_registry.py "
                    f"check '<name>') or add `# DUPLICATE_SOT_OK:<rationale>`."
                )

    for doc_path, key in wt_specs:
        rel = doc_path.relative_to(root).as_posix()
        text = _read(doc_path) or ""
        if _dup_waiver_ok(text):
            continue
        flagged: set[str] = set()

        # Registry leg.
        for e in reg_entries:
            veh = e.get("vehicle")
            if not veh or e.get("status", "active") != "active":
                continue
            if _norm_vehicle_key(str(veh)) != key:
                continue
            cpath = str(e.get("canonical_path", "")).strip()
            if cpath.startswith("./"):
                cpath = cpath[2:]
            if cpath and cpath != rel and cpath not in flagged:
                flagged.add(cpath)
                violations.append(
                    f"{rel}: duplicate canonical spec for vehicle v{key} — the "
                    f"canonical doc registry names {cpath} (branch "
                    f"{e.get('branch', '?')}) as the SoT. Rule chain: "
                    f"NAME-ANCHORED-SEARCH bug class -> registry is naming-"
                    f"independent -> fold this content into the registered "
                    f"canonical doc on its branch, or supersede it IN the "
                    f"registry, or add `# DUPLICATE_SOT_OK:<rationale>`."
                )

        # All-refs legs.
        for sha, refname in commits:
            for path, oid in ref_entries_by_sha.get(sha, []):
                if path == rel or path in flagged:
                    continue
                base = path.rsplit("/", 1)[-1]
                m = _SPEC_NAME_RE.match(base)
                if m and _norm_vehicle_key(m.group(1)) == key:
                    # NAME leg: spec-named doc, same key, different path.
                    flagged.add(path)
                    violations.append(
                        f"{rel}: duplicate canonical spec for vehicle v{key} — "
                        f"{path} exists on ref {refname}. Rule chain: NAME-"
                        f"ANCHORED-SEARCH bug class -> search ALL refs before "
                        f"creating -> fold into the doc on {refname}, or add "
                        f"`# DUPLICATE_SOT_OK:<rationale>`."
                    )
                    continue
                if m:
                    continue  # spec-named but a different vehicle
                # CONTENT-FAMILY leg: the doc's first heading claims the same
                # vehicle even though its filename never says SPEC_v<key>.
                # Normalized-key equality is the decisive test: '8' != '81'
                # (v8 never absorbs v8.1) while '75' == '75' (v7.5 == v75).
                if heading_key_by_oid.get(oid) != key:
                    continue
                flagged.add(path)
                violations.append(
                    f"{rel}: duplicate canonical spec for vehicle v{key} — "
                    f"{path} on ref {refname} CLAIMS the same vehicle in its "
                    f"spec heading (content-family match; its filename does "
                    f"not say SPEC_v{key}). Rule chain: NAME-ANCHORED-SEARCH "
                    f"bug class -> content/concept search, not filename glob "
                    f"-> fold into the doc on {refname}, or add "
                    f"`# DUPLICATE_SOT_OK:<rationale>`."
                )

    return _finish(
        name="check_no_duplicate_canonical_spec_across_refs",
        tag="duplicate-sot-across-refs",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=(
            f"{len(wt_specs)} working-tree spec doc(s) x {len(commits)} unique "
            f"ref commit(s) searched by name+content"
        ),
    )


# ── DESIGNED-STUB refusal gate (ddm_sb2, task #819) ──────────────────────────
def check_no_stub_lever_factories(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """REFUSE a DSL ``Lever`` factory whose MECHANISM does not exist (the DESIGNED-STUB class).

    Confound class (NO-FAKE forbidden class #1, marker-without-mechanism, at the registry
    layer): a ``Lever`` factory presents as BUILT to every consumer — the DSL, the
    ``lever_registry`` coverage query, the activation ledger, the costate duty queue, launch
    tickets, council deliberations. If its emitted trainer flag does not EXIST on that
    module's trainer argparse, the lever is hollow, yet its activation row reads
    ``never-fired``, byte-identical to a fully-built default-off lever. Nothing anywhere
    could tell the two apart.

    RECEIPT 2026-07-31 (the incident this gate extincts): a strategy argument for a fresh
    from-birth run rested on "the full protection/force stack has never run from ep0"; the
    gc15 convocation then found **5 of 6 of those forces were DESIGNED-STUBs** — wiring, not
    birth, was the blocker. The debt had been reported as a neutral STATUS LABEL for weeks.
    Operator, same day: *"Everything that is designed to stub ... needs to be fully built
    out. No orphan signal is a very important principle."*

    The judgement is STRUCTURAL, never label-based: ``tac.witness_dsl.lever_registry.
    build_completeness`` resolves EACH module's own trainer (a module declaring
    ``TRAINER_RELPATH`` binds to it; others bind to the levelset entry point + base) and
    grades a factory a stub iff it emits a flag that trainer does not declare. So a factory
    that forgot to say "DESIGNED-STUB" is still caught, and one that says so while its flags
    exist is reported as LABEL DRIFT rather than as a stub.

    Warn-only at landing (Strict-flip atomicity rule) — the live count is non-zero and the
    remaining stubs are chartered builds owned by other arms, so flipping now would refuse
    the tree for debt this arm does not own. STRICT-FLIP CONDITION: flip once every factory
    either has its trainer flag or carries the waiver (live count 0).

    Same-line waiver (in the factory's own source): ``# DESIGNED_STUB_OK:<rationale>``.
    A bare ``<rationale>`` placeholder does not self-waive (Catalog #287 sister).
    """
    from tac.witness_dsl.lever_registry import build_completeness

    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    bc = build_completeness()
    for fb in bc.stubs:
        src_path = root / "src" / "tac" / "witness_dsl" / fb.module
        if _factory_waived(src_path, fb.factory):
            continue
        silent = "" if fb.stub_marker else " [SILENT — it does not even announce itself]"
        violations.append(
            f"src/tac/witness_dsl/{fb.module}: Lever factory {fb.factory!r} is a "
            f"DESIGNED-STUB{silent} — it emits {list(fb.missing_flags)} which "
            f"{fb.trainer} does not declare, so the lever presents as BUILT to the DSL, the "
            f"activation ledger and the duty queue while no mechanism exists "
            f"(NO-FAKE marker-without-mechanism). Build the trainer wiring, or add a "
            f"`# DESIGNED_STUB_OK:<rationale>` marker on the factory's def line."
        )
    for fb in bc.label_drift:
        if fb.is_stub:
            continue  # already reported above
        violations.append(
            f"src/tac/witness_dsl/{fb.module}: Lever factory {fb.factory!r} still declares "
            f"itself a DESIGNED-STUB but every flag it emits now exists on {fb.trainer} — "
            f"stale label. Drop the marker so the registry's grade and the source agree."
        )
    return _finish(
        name="check_no_stub_lever_factories",
        tag="no-stub-lever-factories",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=(
            f"{bc.total - len(bc.stubs)}/{bc.total} lever factories across "
            f"{bc.modules_scanned} module(s) have real trainer mechanisms"
        ),
    )


def check_no_legacy_single_module_lever_surface_consumers(
    *,
    repo_root: str | Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """REFUSE a consumer that binds orphan/duty accounting to the SINGLE-MODULE lever surface.

    Sister of :func:`check_no_stub_lever_factories`, sharing its catalog row (see the scope-
    extension note below): that gate refuses a lever that PRESENTS as built with no mechanism;
    this one refuses a CONSUMER that reads a PARTIAL registry and therefore presents a partial
    universe as complete. Same NO-FAKE marker-without-mechanism class, one layer out.

    Live count 0 at landing (ddm_rg5 #825), so STRICT from byte one per the Strict-flip
    atomicity rule — no warn-only purgatory for a class whose whole failure mode is a correct
    surface nobody opts into.
    """
    root = Path(repo_root or REPO_ROOT)
    return _finish(
        name="check_no_legacy_single_module_lever_surface_consumers",
        tag="no-legacy-single-module-lever-surface",
        violations=_legacy_single_module_lever_consumers(root),
        strict=strict,
        verbose=verbose,
        ok_detail="no consumer binds orphan accounting to the single-module lever surface",
    )


# ── CATALOG #351 SCOPE EXTENSION (ddm_lr2, 2026-08-03) ───────────────────────────────────────
# NOT a new catalog number. Catalog #351 is the fake-claim-guard row — CLAUDE.md records it
# refusing "a selected-pose marker without an authenticated receiver/parse-back byte effect",
# i.e. exactly a marker ASSERTED where the work was never done — and it already carries a
# documented scope extension ("This is a Catalog #351 scope extension, not a new gate or number,
# per the post-#400 Catalog #299 consolidation rule"). This is the same class at the packet-IR
# readiness-field surface, so it rides that row rather than claiming a number past the #400 cap.
#
# THE INCIDENT (surfaced by ddm_la1 §6.3 HIT 1, adjudicated + fixed here).
# ``inverse_steganalysis_operation_set_compiler`` emitted
# ``"byte_closed_operation_count": len(operations)`` and
# ``"chosen_operation_sequence_is_permutation": True`` — declaring both properties while the
# 241-line module performed no byte accounting and no permutation comparison at all. Its SIBLING
# producer in ``byte_shaving_campaign`` computes BOTH for real, and — the part that makes this
# load-bearing rather than cosmetic — both producers' rows land in the same
# ``packet_ir_operation_sets`` list and are SUMMED into ``packet_ir_byte_closed_operation_count``,
# a readiness figure. An unchecked value was being added to a total a reader takes as checked,
# across five production importers. NO-FAKE forbidden class 1 (canonical markers without the
# work) AND class 4 (a declared value in a canonical data field), simultaneously.
#
# The judgement is STRUCTURAL and deliberately narrow: for the two readiness keys, an ASSERTION
# FORM as the dict value is refused — a bare ``True``/``False`` literal for the permutation
# predicate, or a bare ``len(...)`` for the byte-closed count. Both say "all of them, by
# construction". A call to a helper, a comparison, a comprehension or a name are all accepted:
# the gate refuses the shape that CANNOT have done the work, and never tries to judge whether a
# real computation is correct. Scope is the packet-IR producer surface (files that reference
# ``PACKET_IR_OPERATION_SET_SCHEMA``), because that is where the summed figure is built.
#
# STRICT FROM BYTE ONE: live count is 0 in this landing (the one violation is fixed in the same
# commit), per the Strict-flip atomicity rule.
#
# Waiver: same-line ``# ASSERTED_READINESS_FIELD_OK:<rationale>`` on the offending line (a bare
# ``<rationale>`` placeholder does not self-waive, Catalog #287 sister).
_ASSERTED_READINESS_KEYS = {
    "byte_closed_operation_count": "a COUNT of operations proven byte-closed",
    "chosen_operation_sequence_is_permutation": "a DECISION that the sequence is a permutation",
}


def _asserted_readiness_violations(root: Path) -> list[str]:
    import ast as _ast

    out: list[str] = []
    for rel_dir in ("src/tac/optimization", "src/tac/packet_compiler"):
        base = root / rel_dir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "PACKET_IR_OPERATION_SET_SCHEMA" not in text:
                continue
            try:
                tree = _ast.parse(text)
            except SyntaxError:
                continue
            lines = text.splitlines()
            for node in _ast.walk(tree):
                if not isinstance(node, _ast.Dict):
                    continue
                for key, value in zip(node.keys, node.values, strict=False):
                    if not (isinstance(key, _ast.Constant) and isinstance(key.value, str)):
                        continue
                    meaning = _ASSERTED_READINESS_KEYS.get(key.value)
                    if meaning is None:
                        continue
                    asserted = (
                        isinstance(value, _ast.Constant) and isinstance(value.value, bool)
                    ) or (
                        isinstance(value, _ast.Call)
                        and isinstance(value.func, _ast.Name)
                        and value.func.id == "len"
                    )
                    if not asserted:
                        continue
                    lineno = getattr(value, "lineno", key.lineno)
                    src_line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
                    if _waiver_present(src_line, "ASSERTED_READINESS_FIELD_OK"):
                        continue
                    form = "a bare literal" if isinstance(value, _ast.Constant) else "bare len(...)"
                    out.append(
                        f"{path.relative_to(root)}:{lineno}: readiness field {key.value!r} is "
                        f"ASSERTED via {form} — the field means {meaning}, and this shape cannot "
                        f"have done that work. Its value is summed with genuinely-checked sibling "
                        f"rows into packet_ir_byte_closed_operation_count, so an unchecked value "
                        f"enters a total readers take as checked (NO-FAKE forbidden classes 1+4). "
                        f"Compute it (see byte_shaving_campaign's sibling producer), or add "
                        f"`# ASSERTED_READINESS_FIELD_OK:<rationale>` on this line."
                    )
    return out


def check_no_asserted_packet_ir_readiness_fields(
    *,
    repo_root: str | Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """REFUSE a packet-IR readiness field DECLARED rather than DECIDED (Catalog #351 scope)."""
    root = Path(repo_root or REPO_ROOT)
    return _finish(
        name="check_no_asserted_packet_ir_readiness_fields",
        tag="no-asserted-packet-ir-readiness-fields",
        violations=_asserted_readiness_violations(root),
        strict=strict,
        verbose=verbose,
        ok_detail="every packet-IR readiness field is computed from the operations it describes",
    )


# ── SCOPE EXTENSION (ddm_lr2, 2026-08-03) ────────────────────────────────────────────────────
# NOT a new catalog number: per CLAUDE.md "Gate consolidation discipline" (#299) a new strict
# gate past #400 must retire or replace one. This is the SAME bug class as
# ``check_no_stub_lever_factories`` one layer BELOW it — that gate refuses a lever whose
# mechanism does not exist; this one refuses a lever whose *stub verdict was decided by a
# trainer binding nobody declared* — so it rides the same catalog row, the third extension on
# it (after ddm_rg5's consumer-side extension above).
#
# THE INCIDENT (MEASURED, ddm_lr2 §1). ``spec_tr1_renderer_20260728`` was the ONLY module in the
# whole package that declared ``TRAINER_RELPATH``. Every other lever module silently inherited
# the RETIRED levelset trainer — including three whose own docstrings name the TR1 vehicle
# ("the v8/v9/v10 forces ADAPTED to the TR1 vehicle"). Their 8 factories were therefore graded
# against a trainer they were never written for, and — the part that cost real signal — no
# TR1-scoped query could surface them at all. That is the concrete mechanism behind ddm_gd1's
# unexplained meta-finding that "nothing forces a never-fired row to be drained": a queue cannot
# drain a lever filed under the wrong vehicle. Re-homing them moved the census from 149/31 to
# 141/39 (MEASURED). It did NOT make them fireable — their flags exist on NEITHER trainer — and
# this gate deliberately does not pretend otherwise; ``check_no_stub_lever_factories`` owns the
# build debt, this gate owns the ATTRIBUTION of it.
#
# WHY THE SCOPE IS NARROW, and why that is a measurement not a preference. The refusable set is
# "graded a stub AND undeclared" rather than "undeclared", because a factory whose flags all
# exist needs no binding argument to be graded — its verdict is the same under either trainer.
# The obvious worry is a blind spot: a TR1-targeted module whose flags happen to exist on the
# retired trainer would grade clean and stay mis-bound. MEASURED on this tree (ddm_lr2 §1): for
# every one of the 11 undeclared factory-bearing modules, its flags sit on the retired trainer
# and essentially none on TR1 (curriculum_dsl 297/300 retired vs 4 TR1; spec_v9c3 65 vs 4;
# spec_v9_cgauge 18 vs 0), consistent with the default they inherit. The blind spot has live
# count 0 today — stated as a scoped negative, not an existential one.
#
# STRICT FROM BYTE ONE: live count is 0 in this landing (the three TR1 modules now declare
# ``TRAINER_RELPATH``; the two genuinely retired modules that still graded stubs —
# ``constants_telemetry_build_wave`` and ``curriculum_dsl`` — now declare the retired pair
# explicitly via ``TRAINER_RELPATHS``, which the registry test pins as behaviour-identical). The
# Strict-flip atomicity rule is satisfied here rather than deferred to warn-only purgatory.
#
# Waiver: same-line ``# UNDECLARED_TRAINER_BINDING_OK:<rationale>`` on the factory's ``def``
# line (a bare ``<rationale>`` placeholder does not self-waive, Catalog #287 sister).
def check_lever_module_declares_its_trainer(
    *,
    repo_root: str | Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """REFUSE a DESIGNED-STUB verdict that rests on an UNDECLARED trainer binding.

    A lever module that does not state its trainer is not "bound to the levelset trainer" — it
    is bound to whatever the default happens to be, and no reader can tell an intentional
    binding from an author who never considered the question. When such a module's factory is
    graded a stub, the grade is an artifact of that unchosen default: the flag may be missing
    only because it was checked against the wrong vehicle.

    A silent default is an orphan generator (CLAUDE.md, "'Off' is a tracked queue, never a
    forgotten default"): the state must be TRACKED, REASONED and SURFACED, never inherited in
    silence. Declaring ``TRAINER_RELPATH = "..."`` (or ``TRAINER_RELPATHS = (...)`` for a
    genuine multi-trainer module) is the whole fix — one line, no behaviour change.
    """
    from tac.witness_dsl.lever_registry import build_completeness

    root = Path(repo_root or REPO_ROOT)
    # Scan the caller's tree when it carries a witness_dsl package (the positive-control fixture
    # path); otherwise the installed package. Without this the gate could only ever be OBSERVED
    # returning zero on a clean tree — indistinguishable from a gate that cannot fire.
    pkg_dir = root / "src" / "tac" / "witness_dsl"
    violations: list[str] = []
    bc = build_completeness(pkg_dir if pkg_dir.is_dir() else None)
    for fb in bc.verdict_relevant_undeclared:
        src_path = pkg_dir / fb.module
        if _factory_waived_for(src_path, fb.factory, "UNDECLARED_TRAINER_BINDING_OK"):
            continue
        violations.append(
            f"src/tac/witness_dsl/{fb.module}: Lever factory {fb.factory!r} is graded a "
            f"DESIGNED-STUB (missing {list(fb.missing_flags)}) against {fb.trainer}, but its "
            f"module never declares which trainer it targets — the grade rests on an inherited "
            f"default nobody chose, so the flag may be 'missing' only because it was checked "
            f"against the wrong vehicle. Add `TRAINER_RELPATH = \"experiments/<trainer>.py\"` "
            f"(or `TRAINER_RELPATHS = (...)` for a real multi-trainer module) at module level, "
            f"or put `# UNDECLARED_TRAINER_BINDING_OK:<rationale>` on the factory's def line."
        )
    return _finish(
        name="check_lever_module_declares_its_trainer",
        tag="lever-module-declares-its-trainer",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=(
            f"every stub-graded lever factory has a DECLARED trainer binding "
            f"({len(bc.undeclared_trainer_factories)}/{bc.total} factories still inherit the "
            f"default, all of them graded BUILT so the binding does not decide their verdict)"
        ),
    )


# ── SCOPE EXTENSION (ddm_rg5, task #825, 2026-07-31) ──────────────────────────────────────────
# NOT a new catalog number: per CLAUDE.md "Gate consolidation discipline" (#299) the catalog is
# past #400 (next=408), so a new strict gate must retire or replace one. This gate is the same
# bug class as ``check_no_stub_lever_factories`` one layer OUT and rides its catalog row — the
# precedent CLAUDE.md records for the 2026-07-20 Catalog #351 extension, and the precedent the
# ``check_v9_fake_claim_guards`` strict call site records inline ("standalone numbered catalog
# row deferred").
#
# THE CLASS. ``check_no_stub_lever_factories`` refuses a lever that PRESENTS as built while no
# mechanism exists. The sibling failure is a CONSUMER that reads a PARTIAL registry and therefore
# presents a partial universe as complete. ddm_sb2 repaired the registry by ADDING the package-wide
# surface but PRESERVING the single-module default, documenting the choice as "the historical
# contract is unchanged" — backward compatibility chosen over correctness, silently, ON THE ORPHAN
# TRACKER. Nothing opted in: the honest superset had ONE grep hit outside its own definition, a
# docstring. MEASURED consequence (ddm_rg5): ``known_levers()`` enumerated 116 of 179 factories, so
# 61 were structurally ineligible for the duty queue — including 9 of the 10 DESIGNED-STUBS this
# very gate reports. The tracker could not see the debt its sister gate was raising.
#
# The judgement is STRUCTURAL: a call to the narrow surface outside the allowlist is refused
# regardless of what the calling module claims. Tests are exempt (asserting on the narrow surface
# is exactly how its contract stays pinned). Waiver: same-line ``# SINGLE_MODULE_LEVER_SURFACE_OK:
# <rationale>`` (bare ``<rationale>`` does not self-waive, Catalog #287 sister).
#
# STRICT FROM BYTE ONE: live count is 0 after the #825 fix (the two definers are allowlisted, the
# two production consumers were flipped to ``known_levers()``), so the Strict-flip atomicity rule
# is satisfied in this landing rather than deferred to a warn-only purgatory.
_LEGACY_LEVER_SURFACE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(lever_factories|curriculum_dsl_known_levers)\s*\(")
# The two modules that legitimately OWN the narrow surface: the registry that computes it and the
# activation ledger that exposes it under an intention-revealing name.
_LEGACY_LEVER_SURFACE_OWNERS = (
    "src/tac/witness_dsl/lever_registry.py",
    "src/tac/witness_dsl/activation_ledger.py",
)
_LEGACY_LEVER_SURFACE_ROOTS = ("src/tac", "tools", "experiments", "scripts")
_LEGACY_LEVER_SURFACE_NAMES = ("lever_factories", "curriculum_dsl_known_levers")
# Directory names pruned during the walk. Without this the four roots hold 62,817 .py files
# (vendored intake clones + nested virtualenvs) and the scan costs 12.7 s — MEASURED. Pruning
# takes it to ~0.4 s. sb2's lesson, binding: a slow gate is a disabled gate, which is how the
# vacuity survived in the first place.
_LEGACY_LEVER_SURFACE_PRUNE_DIRS = frozenset({
    "__pycache__", ".git", ".venv", "venv", "env", "node_modules", "site-packages",
    "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache", "tests",
})
# ``experiments/results`` is ARTIFACT CUSTODY, not source: 15,498 of the 16,328 .py files under
# ``experiments`` live there (run dirs + public-PR intake clones we are forbidden to edit), they
# cost 1.9 s to read, and MEASURED ZERO of them mention either narrow name. Pruning the subtree
# takes the whole scan 3.67 s -> 0.4 s. Keyed by (root-relative) path so a source dir that merely
# happens to be named "results" elsewhere is unaffected.
_LEGACY_LEVER_SURFACE_PRUNE_SUBTREES = ("experiments/results",)
# Vendored / public-PR-intake path markers (mirrors preflight's ``_VENDORED_PATH_MARKERS``):
# those trees are forensic inputs we may not edit, so a hit there is never actionable.
_LEGACY_LEVER_SURFACE_SKIP_MARKERS = (
    "_intake_", "/pr_heads/", "/vendored/", "/leaderboard_intel_",
    "/reverse_engineering_", "/public_runtime_adapters_", "/av1_crf31_bicubic/",
)


def _legacy_single_module_lever_consumers(root: Path) -> list[str]:
    """Call sites binding orphan/duty accounting to the SINGLE-MODULE lever surface.

    The judgement is AST-based (an ``ast.Call`` whose callee resolves to one of the narrow
    names), never a text match: the two live text hits at landing were a module docstring and an
    error-message f-string in ``tools/register_ema_finisher_duty.py`` — PROSE ABOUT the surface,
    not a binding to it. A regex gate would have reported both and taught its readers to ignore
    it, which is the failure mode this whole task is about.
    """
    out: list[str] = []
    for rel_root in _LEGACY_LEVER_SURFACE_ROOTS:
        base = root / rel_root
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            here = Path(dirpath).relative_to(root).as_posix()
            if any(here == s or here.startswith(f"{s}/")
                   for s in _LEGACY_LEVER_SURFACE_PRUNE_SUBTREES):
                dirnames[:] = []
                continue
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in _LEGACY_LEVER_SURFACE_PRUNE_DIRS)
            for fname in sorted(filenames):
                if not fname.endswith(".py") or fname.startswith("test_"):
                    continue
                path = Path(dirpath) / fname
                rel = path.relative_to(root).as_posix()
                if rel in _LEGACY_LEVER_SURFACE_OWNERS:
                    continue
                if any(m in f"/{rel}" for m in _LEGACY_LEVER_SURFACE_SKIP_MARKERS):
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if not any(n in text for n in _LEGACY_LEVER_SURFACE_NAMES):
                    continue  # cheap reject before paying for a parse
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    continue  # unparseable file cannot be bound to anything; fail-open
                lines = text.splitlines()
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    fn = node.func
                    name = (fn.attr if isinstance(fn, ast.Attribute)
                            else fn.id if isinstance(fn, ast.Name) else None)
                    if name not in _LEGACY_LEVER_SURFACE_NAMES:
                        continue
                    lineno = getattr(node, "lineno", 0)
                    line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
                    if _waiver_present(line, "SINGLE_MODULE_LEVER_SURFACE_OK"):
                        continue
                    out.append(
                        f"{rel}:{lineno}: binds to the SINGLE-MODULE lever surface "
                        f"(`{name}`), which ASTs only curriculum_dsl.py and enumerated 116 of "
                        f"179 factories when measured — a consumer of a partial registry "
                        f"presents a partial universe as complete (ddm_rg5 #825; the orphan "
                        f"tracker was blind to 9 of its own 10 designed-stubs). Use "
                        f"`tac.witness_dsl.activation_ledger.known_levers()` (package-wide), or "
                        f"add `# SINGLE_MODULE_LEVER_SURFACE_OK:<rationale>` if the narrow "
                        f"single-module question is genuinely what is being asked."
                    )
    return out


def _factory_waived_for(path: Path, factory: str, marker: str) -> bool:
    """True when the factory's own ``def`` line carries a non-placeholder ``marker`` waiver.

    Generalised from :func:`_factory_waived` (ddm_lr2, 2026-08-03) so sister gates on the same
    catalog row share ONE line-locating implementation rather than each re-deriving "which line
    is this factory's def" — the duplicated-predicate class.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("def ", "async def ")) and f" {factory}(" in f" {stripped}":
            return _waiver_present(line, marker)
    return False


def _factory_waived(path: Path, factory: str) -> bool:
    """True when the factory's own ``def`` line carries a non-placeholder waiver marker."""
    return _factory_waived_for(path, factory, "DESIGNED_STUB_OK")


_FILE_MOVE_CALLS = frozenset({"move", "rename", "replace"})


def _row_contract_exception_names(tree: ast.AST) -> set[str]:
    """Exception classes RAISED inside a ``validate``-named function.

    MECHANISM, not name-matching.  Which exceptions are "row-contract" is
    decided by WHERE THEY ARE RAISED -- inside per-row validation -- so a class
    called ``FooError`` is caught by this gate exactly when validation raises it.
    Keying on the exception's own spelling would repeat the name-keying mistake
    the GT-lineage census measured (seven files, one name, three lineages).
    """

    names: set[str] = set()
    for fn in _func_defs(tree):
        if "validate" not in fn.name.lower():
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            exc = node.exc
            target = exc.func if isinstance(exc, ast.Call) else exc
            if isinstance(target, ast.Attribute):
                names.add(target.attr)
            elif isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _handler_moves_a_file(handler: ast.ExceptHandler) -> str | None:
    """Only ATTRIBUTE form counts: ``shutil.move`` / ``os.replace`` / ``p.rename``.

    A BARE ``replace(...)`` is almost always ``dataclasses.replace`` -- a copy,
    not a move.  The first draft of this gate matched bare names and flagged
    ``pr106_sidecar_packet.py:3965``, where ``replace(packet, pr106_bytes=...)``
    rebuilds a frozen dataclass on a recode fallback.  Reading the site (rather
    than trusting the count) is what caught it; the predicate now requires the
    receiver form, and excludes the one attribute spelling that is still a copy.
    """

    for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        recv = node.func.value
        recv_name = recv.id if isinstance(recv, ast.Name) else getattr(recv, "attr", "")
        if recv_name == "dataclasses":
            continue
        if node.func.attr in _FILE_MOVE_CALLS:
            return f"{recv_name}.{node.func.attr}" if recv_name else node.func.attr
    return None


def check_no_row_contract_error_quarantines_the_ledger(
    *,
    repo_root: str | Path | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Refuse: a ROW-contract violation must not move the whole shared ledger.

    THE INCIDENT (2026-08-17, two sites).  A strict reader caught
    ``json.JSONDecodeError`` and a per-row validation error in the SAME handler
    and then ``shutil.move``d the ledger.  One arm's malformed row deleted
    shared state for the whole fleet -- first in the canonical task-status
    ledger, then in ``codex_to_claude_inbox`` (7 in-module read paths, two of
    them reached from WRITE paths, so an append destroyed the file too).

    The two failure classes are NOT the same and must not share a response:

    * FILE corruption -- the bytes do not parse.  Later offsets are
      untrustworthy; quarantine is proportionate and stays allowed.
    * ROW-CONTRACT violation -- the line IS a well-formed object and one field
      relationship fails.  Every other row is exactly as valid as before.
      Isolate the ROW.

    Population MEASURED before landing, not assumed: an AST sweep of 7,532
    files under ``src/`` found ZERO remaining sites after the cure, so this
    lands STRICT from byte one rather than sitting in warn-only purgatory.

    Same-line waiver ``# ROW_CONTRACT_QUARANTINE_OK:<rationale>`` on the
    ``except`` line, for a reader whose rows genuinely cannot be isolated
    (a per-key state machine where dropping one row orphans its successors).
    """

    repo = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    for path in sorted((repo / "src").rglob("*.py")):
        text = _read(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        scanned += 1
        row_excs = _row_contract_exception_names(tree)
        if not row_excs:
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                mover = _handler_moves_a_file(handler)
                if mover is None:
                    continue
                caught = handler.type
                if caught is None:
                    continue
                parts = caught.elts if isinstance(caught, ast.Tuple) else [caught]
                hit = sorted(
                    {
                        (p.attr if isinstance(p, ast.Attribute) else getattr(p, "id", ""))
                        for p in parts
                    }
                    & row_excs
                )
                if not hit:
                    continue
                line = lines[handler.lineno - 1] if handler.lineno <= len(lines) else ""
                if _waiver_present(line, "ROW_CONTRACT_QUARANTINE_OK"):
                    continue
                violations.append(
                    f"{path.relative_to(repo)}:{handler.lineno}: except {hit} "
                    f"(raised by a validate-* function) calls {mover}() -- a ROW-contract "
                    f"violation must isolate the row, not move the shared ledger"
                )
    return _finish(
        name="check_no_row_contract_error_quarantines_the_ledger",
        tag="row-contract-quarantine",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{scanned} modules examined, no row-contract quarantine paths",
    )


# The two automatable eightfold gates (P1 + P4), for the preflight wire-in + tests.
EIGHTFOLD_GATES = (
    check_significance_keys_canonical,
    check_witness_control_meters_have_canaries,
)


# Convenience: the gates in catalog order, for the preflight wire-in + tests.
CONFOUND_GATES = (
    check_no_row_contract_error_quarantines_the_ledger,
    check_no_spike_guard_defaults_to_deadlock_mode,
    check_reject_filter_updates_reference_from_accepted_only_has_rearm,
    check_no_duplicate_long_flags_in_launch,
    check_resume_palliative_flags_imply_warm_start,
    check_verdict_pairs_default_is_n600,
    check_telemetry_verdict_rows_carry_liveness,
    check_levelset_hosc_requires_beta_end,
    check_launch_config_authored_in_dsl,
    check_no_unjustified_magnitude_dismissal,
    check_no_inert_additive_margin_composition,
    check_codex_retry_preserves_original_sandbox_authority,
    check_codex_nonisolated_writer_cap,
    check_codex_drain_timeout_uses_liveness,
    check_consolidation_debt_monitor_observability_and_cadence,
    check_witness_trainers_emit_partial_freeze_alarm,
    check_witness_verdict_rows_carry_dseg_descent_canary,
    check_verdict_live_gap_defaults_on_during_ema_warmup,
    check_no_raw_virtual_memory_safety_basis,
    check_process_guard_excludes_observer_flag_values,
    check_no_duplicate_canonical_spec_across_refs,
    check_no_stub_lever_factories,
    check_no_legacy_single_module_lever_surface_consumers,
    check_lever_module_declares_its_trainer,
    check_no_asserted_packet_ir_readiness_fields,
)


# ---------------------------------------------------------------------------
# ddm_gh1 CLASS GUARD — a REFUSE-capable gate needs a POSITIVE CONTROL and a
# DECLARED DENOMINATOR.
#
# THE CLASS (measured 2026-07-31, five instances in one day): a gate that can
# REFUSE is trusted precisely because nobody re-derives it. When its detector is
# narrowed — by a prefilter, a glob, a registry that enumerates part of its
# universe — it keeps printing OK over an almost-empty scan and everyone reads
# that as "clean". Instances: this module's own CLASS-2 gate skipped the entire
# `ps -axo command` guard family at the FILE level while two live slot guards
# carried the bug (#829); the raw-vm gate declared live-count 0 while measuring 6
# and silently omitted experiments/ + scripts/ (#830); the lever registry AST'd
# 1 of 171 modules; a findings gate scanned 0 of 1,260 files; a duty queue
# enumerated 116 of 177. The identical reasoning error shows up in prose as a
# false "X does not exist" claim built on a partial search.
#
# TWO STRUCTURAL REQUIREMENTS, both of which make a narrowing LOUD:
#   1. POSITIVE CONTROL — a fixture the gate MUST still flag. Registered here and
#      EXECUTED by the meta-gate, so it is a live assertion rather than a claim.
#      A narrowing that guts the detector fails here instead of printing OK.
#   2. DECLARED DENOMINATOR — the gate reports what it CONSIDERED next to what it
#      scanned, so "0 violations" can never be read as "0 scanned".
#
# Coverage is a TRACKED QUEUE, never a forgotten default: gates without a control
# are NAMED in the ok_detail and the covered count RATCHETS (it may grow, never
# shrink), so a new REFUSE-capable gate cannot quietly land without one.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositiveControl:
    """A fixture a REFUSE-capable gate MUST still flag, plus why it matters."""

    gate: str
    files: Mapping[str, str]
    must_mention: str
    why: str


POSITIVE_CONTROLS: tuple[PositiveControl, ...] = (
    PositiveControl(
        gate="check_no_raw_virtual_memory_safety_basis",
        files={
            "tools/planted.py": (
                "import psutil\n"
                "def guard():\n"
                "    if psutil.virtual_memory().available < 1:\n"
                "        raise SystemExit(7)\n"
            )
        },
        must_mention="planted.py",
        why=(
            "#830: the gate declared live-count 0 while measuring 6, and omitted experiments/ + "
            "scripts/ from its scan. If a future prefilter narrows it, this fires."
        ),
    ),
    PositiveControl(
        gate="check_process_guard_excludes_observer_flag_values",
        files={
            "tools/planted_guard.py": (
                "import psutil\n"
                "def refuse_dup(out_dir):\n"
                "    for p in psutil.process_iter(['cmdline']):\n"
                "        cl = ' '.join(p.info.get('cmdline') or ())\n"
                "        if 'train_levelset_witness' in cl:\n"
                "            raise SystemExit(12)\n"
            )
        },
        must_mention="planted_guard.py",
        why="#406/#512: the original function-scoped observer-flag leg.",
    ),
    PositiveControl(
        gate="check_process_guard_excludes_observer_flag_values",
        files={
            "tools/planted_split.py": (
                'import subprocess\n'
                'SLOT_TOKENS = ("train_levelset_witness", "evaluate.py")\n'
                'def slot_is_live():\n'
                '    out = subprocess.run(["ps", "-axo", "command"], capture_output=True,\n'
                '                         text=True, check=False).stdout\n'
                '    return any(tok in line for line in out.splitlines()\n'
                '               for tok in SLOT_TOKENS)\n'
                'def main():\n'
                '    if slot_is_live():\n'
                '        raise SystemExit("refuse: slot busy")\n'
            )
        },
        must_mention="planted_split.py",
        why=(
            "#829: the EXACT pre-fix ru1/sb1 shape — `ps -axo command` enumeration with the "
            "enumeration, the token test and the decision SPLIT across functions and module "
            "scope. Every leg was individually invisible to the function-scoped predicate, so "
            "the gate printed OK while three live slot guards emitted false refusals."
        ),
    ),
    PositiveControl(
        gate="check_levelset_hosc_requires_beta_end",
        files={
            "experiments/results/planted_run/launch.sh": (
                "#!/bin/bash\npython t.py --activation hosc --hosc-beta 4.0\n"
            )
        },
        must_mention="planted_run",
        why="CLAUDE.md-forbidden fixed-beta hosc (tanh saturation -> vanishing gradient).",
    ),
    PositiveControl(
        gate="check_checkpoint_saves_do_not_silently_drop_optimizer_state",
        files={
            "experiments/planted_trainer.py": (
                "def save_checkpoint(path, *, model, opt_state_flat, epoch):\n"
                "    return None\n"
                "def train(model, optimizer):\n"
                "    save_checkpoint('c.npz', model=model, opt_state_flat={}, epoch=7)\n"
            )
        },
        must_mention="planted_trainer.py",
        why=(
            "ddm_op2 OP2-1: the EXACT pre-fix shape — a bare `opt_state_flat={}` at a "
            "save_checkpoint callsite. All six trainer callsites carried it, so no checkpoint "
            "on disk held optimizer state and every resume was a full Adam moment reset "
            "(#824 arm B, MEASURED 16.167 epochs of re-convergence per boundary => ~218 of a "
            "666-epoch budget). This control is what proves the detector still fires now that "
            "the live count is 0 — the state in which a working gate and a gutted one print "
            "the identical OK."
        ),
    ),
    # -----------------------------------------------------------------------
    # 2026-08-01 (task #831, ddm_gc16). The three gates below are the ORIGINAL
    # immune system from the 2026-07-05 confound hunt — Catalog #397 / #398 /
    # #401 — and they were the most load-bearing entries still sitting in the
    # uncovered queue. Each was STRICT-flipped at live-count 0, which is exactly
    # the state where nothing else proves the detector still fires: a live count
    # of zero looks identical whether the gate is working or gutted. That is the
    # vacuity genus one layer down, on the gates themselves.
    # -----------------------------------------------------------------------
    PositiveControl(
        gate="check_no_spike_guard_defaults_to_deadlock_mode",
        files={
            "experiments/train_witness_realized_through_R_mlx.py": (
                "import argparse\n"
                "def build():\n"
                "    p = argparse.ArgumentParser()\n"
                "    p.add_argument('--spike-guard-mode', default='legacy')\n"
                "    return p\n"
            )
        },
        must_mention="--spike-guard-mode",
        why=(
            "Catalog #397 anchor confound C1: default='legacy' is skip-with-frozen-median. "
            "It froze BOTH the v5 and v6 n600 runs at ep114/ep103 while telemetry kept "
            "advancing, and a whole session's eikonal/viscosity verdicts were drawn from "
            "the frozen weights. This control is what proves the detector still sees a "
            "deadlock-mode default now that the live trainer defaults to 'rollback'."
        ),
    ),
    PositiveControl(
        gate="check_verdict_pairs_default_is_n600",
        files={
            "experiments/train_witness_realized_through_R_mlx.py": (
                "import argparse\n"
                "def build():\n"
                "    p = argparse.ArgumentParser()\n"
                "    p.add_argument('--verdict-pairs', type=int, default=24)\n"
                "    return p\n"
            )
        },
        must_mention="--verdict-pairs",
        why=(
            "Catalog #401 confound C12: a non-zero --verdict-pairs default runs best-checkpoint "
            "selection and ALL d_seg telemetry on a subset, violating the n600 non-negotiable at "
            "the exact number that defines the goal. A subset verdict is a toy, and a toy that "
            "looks like a measurement is the most expensive kind."
        ),
    ),
    PositiveControl(
        gate="check_reject_filter_updates_reference_from_accepted_only_has_rearm",
        files={
            "experiments/train_witness_realized_through_R_mlx.py": (
                "def train_step(loss, median_window):\n"
                "    ref = sorted(median_window)[len(median_window) // 2]\n"
                "    spiked = loss > 3.0 * ref\n"
                "    if spiked:\n"
                "        return True\n"
                "    else:\n"
                "        median_window.append(loss)\n"
                "    return False\n"
            )
        },
        must_mention="train_witness_realized_through_R_mlx.py",
        why=(
            "Catalog #398 is the GENERALIZED structural gate behind C1: a reference window "
            "appended ONLY in the accepted branch can never recover once a sustained spike "
            "starts rejecting, so the guard deadlocks silently. This fixture is the bare "
            "shape — accepted-only append, spike comparison, no re-arm token."
        ),
    ),
    PositiveControl(
        gate="check_no_duplicate_long_flags_in_launch",
        files={
            "experiments/results/planted_dup/launch.sh": (
                "#!/bin/bash\npython t.py --epochs 4 --epochs 9\n"
            )
        },
        must_mention="planted_dup",
        why="argparse last-wins silently discards the earlier value.",
    ),
    PositiveControl(
        gate="check_verdict_surfaces_report_examined_count",
        files={
            "tools/planted_vacuous.py": (
                "from pathlib import Path\n"
                "def audit(root):\n"
                "    bad = [p for p in Path(root).rglob('*.py') if 'x' in p.name]\n"
                "    if not bad:\n"
                "        print('AUDIT PASSED')\n"
                "    return bad\n"
            )
        },
        must_mention="planted_vacuous.py",
        why=(
            "#842 in miniature: enumerate a scope, then emit the verdict as a bare "
            "constant. When rglob matches NOTHING this prints exactly what a clean "
            "full scan prints — vacuity indistinguishable from PASS. If a future "
            "prefilter narrows the detector (e.g. it stops treating rglob as "
            "enumeration, or stops walking tools/), this control stops firing."
        ),
    ),
    PositiveControl(
        gate="check_no_asserted_packet_ir_readiness_fields",
        files={
            "src/tac/optimization/planted_asserted_readiness.py": (
                "PACKET_IR_OPERATION_SET_SCHEMA = 'packet_ir_operation_set.v1'\n"
                "def build(operations):\n"
                "    return {\n"
                "        'schema': PACKET_IR_OPERATION_SET_SCHEMA,\n"
                "        'chosen_operation_sequence_is_permutation': True,\n"
                "        'byte_closed_operation_count': len(operations),\n"
                "    }\n"
            )
        },
        must_mention="planted_asserted_readiness.py",
        why=(
            "The measured NO-FAKE incident in miniature: DECLARE a readiness property instead "
            "of DECIDING it. `len(operations)` says every operation is byte-closed while no "
            "byte accounting exists, and the bare `True` says the sequence is a permutation "
            "while nothing is compared. Both values are SUMMED with a sibling producer's "
            "genuinely-checked rows into packet_ir_byte_closed_operation_count, so an unchecked "
            "value enters a total readers take as checked. If a future change stops walking "
            "src/tac/optimization, stops keying on PACKET_IR_OPERATION_SET_SCHEMA, or starts "
            "accepting bare literals for these keys, this control stops firing."
        ),
    ),
    PositiveControl(
        gate="check_lever_module_declares_its_trainer",
        files={
            "src/tac/witness_dsl/planted_undeclared_lever.py": (
                "from tac.witness_dsl.curriculum_dsl import Lever\n"
                "def PlantedStub() -> Lever:\n"
                "    return Lever('planted', overrides={'--planted-flag-that-does-not-exist': True})\n"
            )
        },
        must_mention="planted_undeclared_lever.py",
        why=(
            "A lever module that never states its trainer, whose factory is then graded a STUB "
            "against whatever the default happens to be. This is the shape that filed eight "
            "TR1-targeted factories under the RETIRED vehicle for a week, where no TR1-scoped "
            "query could surface them for drainage. If a future change makes the gate scan only "
            "the installed package (ignoring repo_root), or stops treating an undeclared "
            "binding as verdict-relevant, this control stops firing."
        ),
    ),
)

# RATCHET FLOOR: the number of DISTINCT gates carrying a positive control at landing. It may only
# grow. Deleting or stranding a control drops coverage below this and the meta-gate refuses.
# 4 -> 5 (vc1, #842) -> 8 (ddm_gc16, #831) -> 9 (ddm_op2: OP2-1) -> 12 (ddm_gb1, #1073).
# ddm_gb1 raised it to the MEASURED live value, not to its own +1: the floor had drifted three
# below actual, so three controls could have been deleted with the guard still printing OK. A floor
# that lags the truth is not a ratchet. Re-measure and raise on every landing that adds a control.
# 12 -> 13 (ddm_pl1): check_no_bulk_write_strands_the_ready_record ships with its control, MEASURED.
MIN_POSITIVE_CONTROL_COVERAGE = 13

# RATCHET CEILING (added 2026-07-31, task #831). The floor above is on the NUMERATOR — the count
# of gates that HAVE controls — so it can only ever fire when a control is REMOVED. Landing a new
# REFUSE-capable gate without a control raises the DENOMINATOR and leaves the numerator untouched,
# so the floor stays satisfied and the guard prints OK. That is exactly the trigger the original
# comment here advertised ("a new gate landing without a control ... the meta-gate refuses") and
# exactly the one the arithmetic could not produce.
#
# MEASURED at the moment this was found: landing check_upstream_pin_no_content_drift took the
# catalog 23 -> 24 and the uncovered set 19 -> 20, and the guard emitted nothing.
#
# This ceiling closes it from the other side and MAY ONLY SHRINK. It is deliberately set to the
# CURRENT uncovered count rather than to 0: refusing all 20 today would be permanently red, and a
# permanently-red gate trains readers to ignore the suite (the #821 lesson). Recording the debt as
# a number that can only go down is what makes the uncovered set a queue instead of a grave —
# lower it as controls land; never raise it to admit a new bare gate.
MAX_UNCOVERED_REFUSE_GATES = 17  # 20 -> 17 (ddm_gc16, #831). Only ever shrinks.


def positive_control_coverage() -> dict[str, object]:
    """The DECLARED DENOMINATOR for control coverage: covered / total, plus the named queue."""
    gates = [fn.__name__ for fn in CONFOUND_GATES]
    covered = sorted({c.gate for c in POSITIVE_CONTROLS})
    return {
        "controls": len(POSITIVE_CONTROLS),
        "covered_gates": covered,
        "covered": len(covered),
        "total_refuse_capable_gates": len(gates),
        "uncovered_gates": sorted(set(gates) - set(covered)),
    }


def check_refusal_gates_have_live_positive_control(
    *,
    repo_root: str | Path | None = None,  # unused: controls run on synthetic fixtures
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """CLASS GUARD (ddm_gh1 2026-07-31) — every registered POSITIVE CONTROL must still fire, and
    control coverage may never regress.

    This EXECUTES each control against its gate on a synthetic tree. It is not a declaration that
    controls exist; it is a live assertion that they still catch what they were written to catch.
    A prefilter, glob, or registry narrowing that guts a detector fails HERE, loudly, instead of
    printing a clean OK over an almost-empty scan.

    Coverage is reported as a declared denominator with the uncovered gates NAMED, and the covered
    count ratchets against :data:`MIN_POSITIVE_CONTROL_COVERAGE`.

    STRICT from byte one: live count is 0 in this landing.
    """
    violations: list[str] = []
    by_name = {fn.__name__: fn for fn in CONFOUND_GATES}
    for control in POSITIVE_CONTROLS:
        gate = by_name.get(control.gate)
        if gate is None:
            violations.append(
                f"positive control names an unregistered gate {control.gate!r} — a stale control "
                f"is indistinguishable from a passing one. Remove it or restore the gate."
            )
            continue
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for relative, text in control.files.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            try:
                found = gate(repo_root=root, strict=False, verbose=False)
            except Exception as exc:  # a raising gate is itself the finding
                violations.append(f"{control.gate}: positive control raised {exc!r}")
                continue
        if not any(control.must_mention in item for item in found):
            violations.append(
                f"{control.gate}: POSITIVE CONTROL NO LONGER FIRES — the planted violation in "
                f"{sorted(control.files)} was not flagged. The detector has been narrowed or "
                f"gutted and the gate is now printing OK over a surface it does not scan. "
                f"Control rationale: {control.why}"
            )
    coverage = positive_control_coverage()
    covered = int(coverage["covered"])
    uncovered = list(coverage["uncovered_gates"])
    if covered < MIN_POSITIVE_CONTROL_COVERAGE:
        violations.append(
            f"positive-control coverage REGRESSED: {covered} gate(s) covered, floor is "
            f"{MIN_POSITIVE_CONTROL_COVERAGE}. A control was removed or stranded "
            f"(uncovered: {uncovered}) — the floor may only ratchet up."
        )
    # The DENOMINATOR-side ratchet. The floor above cannot see a new bare gate (it raises the
    # denominator, not the numerator); this is the leg that does.
    if len(uncovered) > MAX_UNCOVERED_REFUSE_GATES:
        violations.append(
            f"uncovered REFUSE-capable gates GREW to {len(uncovered)}, ceiling is "
            f"{MAX_UNCOVERED_REFUSE_GATES}. A gate that can refuse work landed without a positive "
            f"control, so nothing proves its detector still fires. Add a control for it, or lower "
            f"the ceiling only by covering others — NEVER raise it to admit a bare gate. "
            f"Uncovered: {uncovered}"
        )
    return _finish(
        name="check_refusal_gates_have_live_positive_control",
        tag="refusal-gate-positive-control",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=(
            f"{len(POSITIVE_CONTROLS)} control(s) fired across {covered} of "
            f"{coverage['total_refuse_capable_gates']} refuse-capable gates; "
            f"uncovered queue: {coverage['uncovered_gates']}"
        ),
    )


# The class guard CONSUMES ``CONFOUND_GATES`` to discover the refuse-capable set, so it is defined
# after the catalog tuple and registers itself here rather than inside the literal. It deliberately
# does NOT include itself in its own coverage denominator (it plants no fixture of its own; its
# controls ARE its test).
CONFOUND_GATES = (*CONFOUND_GATES, check_refusal_gates_have_live_positive_control)


# ---------------------------------------------------------------------------
# UPSTREAM-PIN CONTENT DRIFT — the parent repo is STRUCTURALLY BLIND here.
#
# MEASURED INCIDENT (2026-07-31, task #836). ``upstream/`` is a NESTED GIT REPO.
# ``git status --porcelain upstream`` from the repo root returns EMPTY while
# ``git -C upstream status --porcelain`` returned 36 dirty entries. No parent gate,
# hook or preflight scan could see it, and CLAUDE.md declares that snapshot IMMUTABLE
# ("Never edit, patch, monkeypatch, hotfix, or 'temporarily' modify anything inside
# the pinned upstream snapshot"). The rule had ZERO structural enforcement.
#
# What was actually there: 35 of 36 were MODE-ONLY (100755 -> 100644, exec bit stripped
# by a bulk copy) — including evaluate.py / modules.py / frame_utils.py, whose CONTENT
# hashes were byte-identical to the pin, so the scorer authority was intact. The real
# drift was `uv.lock` (+296/-192, additive: charset-normalizer / requests / urllib3 pulled
# in by a stray `uv` invocation with cwd inside upstream) plus 2 lost symlinks.
#
# WHY CONTENT-ONLY (the #821 lesson, applied): a gate that also refused the 35 benign mode
# strips would be PERMANENTLY RED on a clean checkout. A permanently-red gate is not
# protection — it trains readers to ignore the suite, and it is how the lane-smoke gate
# ended up parked behind ``--no-codebase`` where it could never fire. So this gate refuses
# CONTENT drift only, and is explicitly silent about mode.
#
# BINARY-SAFE BY CONSTRUCTION: ``git diff --numstat`` prints ``-`` for binaries, so it
# CANNOT distinguish a mode-only binary from an edited one. This compares the HEAD blob
# bytes against the working-tree bytes directly — exact for text and binary alike.
_UPSTREAM_PIN_WAIVER = "UPSTREAM_PIN_DRIFT_OK"


def _upstream_git_bytes(up: Path, args: list[str]) -> tuple[int, bytes]:
    """``git -C <up> <args>`` -> (returncode, stdout bytes). Never raises."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(up), *args], capture_output=True, timeout=60, check=False
        )
        return proc.returncode, proc.stdout
    except Exception:  # git missing / timeout / permissions — treat as unavailable
        return 1, b""


def check_upstream_pin_no_content_drift(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Refuse CONTENT drift inside the pinned ``upstream/`` snapshot (mode changes exempt).

    The pinned snapshot is the source of truth for scorer behaviour and contest mechanics,
    and CLAUDE.md forbids modifying it without explicit operator approval. Because it is a
    NESTED git repo, the parent repo's gates cannot see inside it — this gate is the only
    structural enforcement of that rule.

    Drift = any of:
      * a tracked file whose working-tree bytes differ from its HEAD blob,
      * a tracked file DELETED from the working tree,
      * an UNTRACKED file (writing into the pin is exactly what the rule forbids;
        gitignored paths never appear in ``status --porcelain`` so they are already exempt).

    NOT drift: a mode change (``100755 -> 100644``) with identical content.

    Fail-open when the nested repo is absent (not every checkout carries ``upstream/.git``),
    but SAY SO when verbose — a guard that silently stops guarding is the confound signature
    this module exists to extinct.

    Waiver: ``# UPSTREAM_PIN_DRIFT_OK:<rationale>`` anywhere in the first 4 KB of
    ``.omx/state/upstream_pin_waiver.txt`` (a file, not a same-line comment, because the
    drifting artifacts are upstream's own and MUST NOT be annotated in place — editing them
    to waive them would itself be the violation).

    STRICT from byte one (2026-07-31): the operator approved reverting the snapshot to its
    pin in the same landing, so live count is 0 and the "Strict-flip atomicity rule" applies.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    up = root / "upstream"
    violations: list[str] = []

    if not (up / ".git").exists():
        if verbose:
            print(
                "  [upstream-pin-drift] SKIP: no nested git repo at upstream/.git — "
                "the pin CANNOT be verified from here (stated, not silently passed)"
            )
        return violations

    waiver_path = root / ".omx" / "state" / "upstream_pin_waiver.txt"
    if waiver_path.is_file():
        try:
            head = waiver_path.read_text(errors="replace")[:4096]
        except OSError:
            head = ""
        marker = f"# {_UPSTREAM_PIN_WAIVER}:"
        idx = head.find(marker)
        if idx >= 0 and len(head[idx + len(marker):].strip().splitlines()[0].strip()) >= 4:
            if verbose:
                print("  [upstream-pin-drift] WAIVED via .omx/state/upstream_pin_waiver.txt")
            return violations

    rc, out = _upstream_git_bytes(up, ["status", "--porcelain"])
    if rc != 0:
        if verbose:
            print("  [upstream-pin-drift] SKIP: `git -C upstream status` unavailable")
        return violations

    # Paths HEAD tracks as executable (mode 100755). MEASURED 2026-07-31: 33 of them, incl.
    # ffmpeg-new, evaluate.sh, and every submissions/*/inflate.sh. Used for the exec-bit-loss leg.
    head_exec: set[str] = set()
    rc_ls, out_ls = _upstream_git_bytes(up, ["ls-files", "-s"])
    if rc_ls == 0:
        for line in out_ls.decode("utf-8", errors="replace").splitlines():
            # "<mode> <oid> <stage>\t<path>"
            if line.startswith("100755\t") or line.startswith("100755 "):
                tab = line.find("\t")
                if tab != -1:
                    head_exec.add(line[tab + 1 :].strip())

    for raw in out.decode("utf-8", errors="replace").splitlines():
        if len(raw) < 4:
            continue
        code, rest = raw[:2], raw[3:].strip()
        # renames render as "old -> new"; the NEW path is what exists on disk.
        rel = rest.split(" -> ")[-1].strip().strip('"')
        if not rel:
            continue
        wt = up / rel

        if code.strip() == "??":
            violations.append(
                f"upstream/{rel}: UNTRACKED file inside the pinned snapshot — something "
                f"wrote into the immutable upstream tree."
            )
            continue

        if not wt.exists():
            violations.append(
                f"upstream/{rel}: tracked file DELETED from the pinned snapshot."
            )
            continue

        blob_rc, blob = _upstream_git_bytes(up, ["show", f"HEAD:{rel}"])
        if blob_rc != 0:
            violations.append(
                f"upstream/{rel}: dirty ({code.strip() or 'M'}) and its HEAD blob could not "
                f"be read — cannot prove it matches the pin."
            )
            continue
        try:
            live = wt.read_bytes()
        except OSError as exc:
            violations.append(f"upstream/{rel}: unreadable ({type(exc).__name__}).")
            continue
        if live != blob:
            violations.append(
                f"upstream/{rel}: CONTENT differs from the pin "
                f"({len(blob)} B at HEAD vs {len(live)} B on disk). The pinned upstream "
                f"snapshot is IMMUTABLE — revert it, or record an operator-approved waiver "
                f"in .omx/state/upstream_pin_waiver.txt."
            )
            continue
        # Content matches. The mode exemption is NOT blanket — see below.
        #
        # CORRECTION 2026-07-31, prompted by the magnitude-dismissal hook and confirmed by
        # MEASUREMENT: "mode-only is benign" was WRONG for one subset. `git ls-files -s` shows
        # upstream tracks 33 files at 100755, and `ffmpeg-new` is a BINARY invoked by bare path
        # (upstream/submissions/*/compress.sh: FFMPEG="${HERE}/ffmpeg-new"). Strip its exec bit
        # and the invocation is "Permission denied" — that is not a small delta, it is a total
        # failure of the run. Dismissing it as noise would have been the eyeball-dismissal the
        # hook exists to catch.
        #
        # The asymmetry is the whole point: GAINING an exec bit is harmless, LOSING one on a file
        # HEAD tracks as executable is potentially catastrophic. So exempt mode changes EXCEPT
        # exec-bit LOSS. Live count is 0 (the revert restored every mode), so this stays a
        # queue-not-grave refinement rather than a permanently-red one.
        if rel in head_exec and not os.access(wt, os.X_OK):
            violations.append(
                f"upstream/{rel}: EXEC BIT LOST (HEAD tracks it 100755, on disk it is not "
                f"executable). Content is identical, so this is not content drift — but upstream "
                f"invokes its executables by bare path, so a stripped exec bit fails the run "
                f"outright rather than perturbing it. Restore the mode "
                f"(git -C upstream checkout -- '{rel}' or chmod +x)."
            )

    return _finish(
        name="check_upstream_pin_no_content_drift",
        tag="upstream-pin-drift",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=(
            f"upstream/ content matches its pin; mode changes exempt EXCEPT exec-bit LOSS "
            f"({len(head_exec)} paths tracked 100755, all still executable)"
        ),
    )


CONFOUND_GATES = (*CONFOUND_GATES, check_upstream_pin_no_content_drift)


# ---------------------------------------------------------------------------
# VACUITY IS INDISTINGUISHABLE FROM PASS — a verdict is a symbol PLUS a
# denominator (ddm_vc1, 2026-08-01, task #842).
#
# THE LAW (memory ``vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801``):
# an instrument that evaluated an EMPTY SCOPE emits the same symbol as one that
# evaluated a full scope cleanly. This is the GENUS of the 2026-08-01 silent
# instruments, not a sibling of the usual DEFAULT-HARMFUL x SILENT x
# MEASUREMENT-CORRUPTING signature: that one gives a WRONG answer, this one
# gives NO answer while looking like a right one. It defeats every check that
# reads the VERDICT rather than the DENOMINATOR.
#
# THREE MEASURED INSTANCES: pytest reports 57 SKIPPED mlx modules as GREEN;
# `--no-codebase` ran 0 of 27 declared preflight gates in 0.52 s and printed
# "PREFLIGHT PASSED" (MEASURED by this landing's probe); a findings scan with an
# `mtime < 3d` window reported CLEAN over 0 of 1,260 files.
#
# This is the SECOND landing per CLAUDE.md "Bugs must be permanently fixed AND
# self-protected against". The first is ``tac.scope_ledger`` plus its wire-in at
# the preflight CLI verdict.
#
# THE REFUSED SIGNATURE is deliberately narrow and mechanical, because the bug's
# fingerprint is structural: a function that ENUMERATES a scope and then emits a
# verdict as a BARE STRING CONSTANT. A bare constant cannot carry a count — the
# denominator is absent BECAUSE the literal has no room for one. So
# ``print(f"ALL {n} CHECKS PASSED")`` passes and ``print("PREFLIGHT PASSED")``
# refuses, with no judgement call in between.
#
# Sister of ``check_refusal_gates_have_live_positive_control`` above: that guard
# asks "does this detector still fire?", this one asks "did this instrument look
# at anything?". Both are denominator questions.
# ---------------------------------------------------------------------------

# Verdict words that assert success. Matched case-insensitively as substrings of
# a print()ed STRING CONSTANT.
_VACUITY_PASS_TOKENS: tuple[str, ...] = (
    "PASSED",
    "ALL CLEAN",
    "NO VIOLATIONS",
    "ALL OK",
    "ALL GREEN",
)

# Calls that ENUMERATE a scope. Their presence is what turns a verdict into a
# claim ABOUT a population, which is what makes a missing denominator a defect.
_VACUITY_ENUM_TOKENS: tuple[str, ...] = (
    ".glob(",
    ".rglob(",
    ".iterdir(",
    ".scandir(",
    "os.walk(",
    ".walk(",
)

_VACUITY_ENUM_NAME_RE = re.compile(
    r"\b(_existing_\w+|\w+_files|list_\w+|collect_\w+|discover_\w+|scan_\w+"
    r"|iter_\w+|find_\w+|_staged\w*)\s*\("
)

#: Canonical verdict surfaces that MUST route through ``tac.scope_ledger``.
#: Keyed by repo-relative path -> (function name, why). Skipped when the file is
#: absent so the synthetic positive-control tree does not trip this leg.
_VACUITY_LEDGER_SURFACES: tuple[tuple[str, str, str], ...] = (
    (
        "src/tac/preflight.py",
        "_preflight_cli_main",
        "task #842: `--no-codebase` executed 0 of 27 declared gates and printed "
        "a bare PREFLIGHT PASSED. This is the regression the ledger cures.",
    ),
)


def _vacuity_bare_pass_prints(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[int, str]]:
    """``print()`` calls in ``func`` whose argument is a BARE pass-word constant.

    An f-string / ``%`` / ``.format`` verdict is accepted unconditionally: it has
    somewhere to put a count, which is all this gate can mechanically ask for.
    """
    out: list[tuple[int, str]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name):
            name = fn.id
        elif isinstance(fn, ast.Attribute):
            name = fn.attr
        else:
            continue
        if name != "print":
            continue
        for arg in node.args:
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                continue
            upper = arg.value.upper()
            if any(tok in upper for tok in _VACUITY_PASS_TOKENS):
                out.append((node.lineno, arg.value.strip()[:80]))
    return out


def _vacuity_scan_files(root: Path) -> list[Path]:
    """The scanned population: operator-facing tools plus top-level tac modules."""
    files: list[Path] = []
    tools_dir = root / "tools"
    if tools_dir.is_dir():
        files.extend(sorted(tools_dir.glob("*.py")))
    tac_dir = root / "src" / "tac"
    if tac_dir.is_dir():
        files.extend(sorted(tac_dir.glob("*.py")))
    return sorted(set(files))


def check_verdict_surfaces_report_examined_count(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """2026-08-01 (ddm_vc1, task #842) — a verdict emitted over an ENUMERATED
    scope must be able to report how many items it examined.

    Two legs:

    **Leg A — the scanner.** Refuses a function that enumerates a scope (glob /
    rglob / iterdir / scandir / walk, or a ``collect_*`` / ``*_files`` / ``scan_*``
    style call) and emits a success verdict as a BARE STRING CONSTANT. A bare
    constant physically cannot carry a denominator, so "0 violations" and
    "0 scanned" print identically. Same-line or in-function waiver:
    ``# VACUITY_LEDGER_OK:<rationale>``.

    **Leg B — the canonical surfaces.** Named verdict surfaces must reference
    ``tac.scope_ledger``. Currently just the preflight CLI, whose bare
    ``print("PREFLIGHT PASSED")`` over an empty scope is the measured anchor for
    the whole class. Absent files are skipped, so the synthetic positive-control
    tree exercises Leg A cleanly.

    What this gate does NOT claim: it is not a proof that every verdict in the
    repo carries a denominator. It refuses ONE mechanical signature over ONE
    named population, and it reports that population. Surfaces outside
    ``tools/*.py`` and ``src/tac/*.py``, verdicts that are not ``print`` calls,
    and prose ``ok_detail`` strings that state no count are all NOT covered —
    see the census in
    ``.omx/research/ddm_vc1_vacuity_denominator_cure_and_census_20260801.md``.

    STRICT from byte one: live count is 0 in this landing, after the two
    same-batch fixes (the preflight CLI ledger wire-in and ``review_tracker``'s
    ``cmd_selftest`` verdict).
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    enumerating = 0

    for path in _vacuity_scan_files(root):
        text = _read(path)
        if not text:
            continue
        scanned += 1
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        rel = path.relative_to(root).as_posix()
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Whole LINES, not ast.get_source_segment: a node's end_col_offset
            # stops at the last token, so a waiver comment trailing the final
            # statement falls OUTSIDE the segment and would be invisible. That
            # is the shape of the very bug this module guards — a detector
            # narrowed just enough to miss the thing it was written to see.
            segment = _span_source(lines, node)
            if not segment:
                continue
            body = _strip_comments(segment)
            enumerates = any(
                tok in body for tok in _VACUITY_ENUM_TOKENS
            ) or bool(_VACUITY_ENUM_NAME_RE.search(body))
            if not enumerates:
                continue
            enumerating += 1
            if _waiver_present(segment, "VACUITY_LEDGER_OK"):
                continue
            for lineno, literal in _vacuity_bare_pass_prints(node):
                violations.append(
                    f"{rel}:{lineno}: {node.name}() enumerates a scope but emits its "
                    f"verdict as a bare constant {literal!r} — no denominator, so an "
                    f"EMPTY scope prints exactly what a clean full scope prints. "
                    f"Report the count (e.g. f\"... {{len(items)}} examined ...\") or "
                    f"route through tac.scope_ledger.ScopeLedger. Waiver: "
                    f"`# VACUITY_LEDGER_OK:<rationale>`."
                )

    surfaces_checked = 0
    for rel, func_name, why in _VACUITY_LEDGER_SURFACES:
        path = root / rel
        if not path.is_file():
            continue
        text = _read(path)
        if not text:
            continue
        surfaces_checked += 1
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        target = None
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func_name
            ):
                target = node
                break
        if target is None:
            violations.append(
                f"{rel}: canonical verdict surface {func_name}() is GONE. Either it "
                f"was renamed (update _VACUITY_LEDGER_SURFACES) or the ledger "
                f"wire-in was deleted. Why it is registered: {why}"
            )
            continue
        segment = _span_source(text.splitlines(), target)
        if "ScopeLedger" not in segment:
            violations.append(
                f"{rel}:{target.lineno}: canonical verdict surface {func_name}() no "
                f"longer references ScopeLedger, so its verdict can again be emitted "
                f"over an unmeasured scope. Why it is registered: {why}"
            )

    return _finish(
        name="check_verdict_surfaces_report_examined_count",
        tag="verdict-reports-examined-count",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=(
            f"{scanned} file(s) scanned, {enumerating} scope-enumerating function(s) "
            f"considered, {surfaces_checked} of {len(_VACUITY_LEDGER_SURFACES)} "
            f"canonical ledger surface(s) present"
        ),
    )


CONFOUND_GATES = (*CONFOUND_GATES, check_verdict_surfaces_report_examined_count)


# ---------------------------------------------------------------------------
# ddm_op2 (OP2-1) — A CHECKPOINT MAY NOT SILENTLY DROP OPTIMIZER STATE.
#
# MEASURED INCIDENT (2026-08-02/03). `experiments/train_tr1_partition_renderer_mlx.py` had SIX
# `save_checkpoint(...)` callsites and every one of them passed the bare literal
# `opt_state_flat={}`. No checkpoint on disk therefore carried optimizer state, so every
# `--resume-from` constructed a fresh `optim.Adam` with both moments zeroed. That is the
# pre-registered `#824` reset-operator ARM B, and the trainer's own `optimizer_arm` telemetry row
# ships its price: `boundary_impulse_epochs_per_reset = 16.167`.
#
# WHAT IT COST, MEASURED: `ddm_gd5` §3.6 watched window_02's LIVE training signal jump 1.912 ->
# 14.846 across a boundary and take ~17 epochs to return -- against the 16.167 prediction, from a
# completely different channel. At ~46 epochs per 30-minute window a 666-epoch run pays that at
# ~13.5 boundaries => ~218 epochs (33%) re-converging a deliberately reset optimizer, leaving ~450
# effective epochs against an incumbent lineage that reached ep945.
#
# WHY A GATE AND NOT JUST THE FIX: `#824` closed arms A/C as "a BUILD, not a port" precisely
# BECAUSE nothing read or wrote `opt_flat` -- the absence justified itself. Six identical bare
# literals are what an omission looks like when it is never named at any callsite. The cure is not
# "always persist" (the default must stay OFF so a sealed live chain keeps byte-identity); it is
# that a callsite must SAY which it is. `no_opt_state("<reason>")` returns the same `{}` and
# carries the rationale, so "none" is a decision instead of a lapse.
#
# DELIBERATELY NOT REFUSED: passing `{}` through a resolver, a variable, or `no_opt_state(...)`.
# The gate refuses the BARE LITERAL only -- the shape an omission actually takes.
# ---------------------------------------------------------------------------

#: Roots scanned for `save_checkpoint(...)` callsites. Deliberately broad: the trainer is not the
#: only place a checkpoint can be written, and a gate that scanned one file would print OK over a
#: near-empty universe the moment the writer moved (the vacuity genus).
_OPT_STATE_SCAN_ROOTS = ("experiments", "src/tac", "tools", "scripts")
_OPT_STATE_KWARG = "opt_state_flat"
_OPT_STATE_WAIVER = "OPT_STATE_DROP_OK"


def _opt_state_candidate_files(root: Path) -> list[Path]:
    """Every .py under the scan roots that mentions the kwarg at all (the DENOMINATOR)."""
    out: list[Path] = []
    for rel in _OPT_STATE_SCAN_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if ".venv" in path.parts or "_intake_" in str(path):
                continue
            text = _read(path)
            if text is not None and _OPT_STATE_KWARG in text:
                out.append(path)
    return out


def check_checkpoint_saves_do_not_silently_drop_optimizer_state(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """ddm_op2 (OP2-1) — a ``save_checkpoint(..., opt_state_flat={})`` BARE LITERAL is refused.

    A checkpoint that drops optimizer state turns every resume into a full Adam moment reset
    (#824 arm B, MEASURED at 16.167 epochs of re-convergence per boundary). Dropping it may be
    the RIGHT call -- the default is off precisely so a sealed live chain keeps byte-identity --
    but it must be a STATED one. Pass the run's resolver, or ``no_opt_state("<rationale>")``,
    which returns the same empty mapping and records why.

    Per-line or per-file waiver: ``# OPT_STATE_DROP_OK:<rationale>`` (placeholder rejected).

    STRICT from byte one: live count is 0 at landing (all six trainer callsites converted).
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    scanned = 0
    callsites = 0
    for path in _opt_state_candidate_files(root):
        text = _read(path)
        if text is None:
            continue
        scanned += 1
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        file_waived = _waiver_present(text, _OPT_STATE_WAIVER)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != _OPT_STATE_KWARG:
                    continue
                callsites += 1
                # The bare-literal shape, and ONLY it: `opt_state_flat={}`.
                if not (isinstance(kw.value, ast.Dict) and not kw.value.keys):
                    continue
                lineno = getattr(kw.value, "lineno", node.lineno)
                line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
                if file_waived or _waiver_present(line, _OPT_STATE_WAIVER):
                    continue
                rel = path.relative_to(root).as_posix()
                violations.append(
                    f"{rel}:{lineno}: `{_OPT_STATE_KWARG}={{}}` bare literal — this checkpoint "
                    f"silently drops optimizer state, so every resume from it is a full Adam "
                    f"moment reset (#824 arm B, MEASURED 16.167 epochs of re-convergence per "
                    f"boundary; ~218 of 666 epochs in 30-minute windows). Pass the run's "
                    f"optimizer-state resolver, or `no_opt_state(\"<rationale>\")` which returns "
                    f"the same empty mapping WITH the reason, or add a "
                    f"`# {_OPT_STATE_WAIVER}:<rationale>` waiver."
                )
    return _finish(
        name="check_checkpoint_saves_do_not_silently_drop_optimizer_state",
        tag="checkpoint-opt-state-not-silently-dropped",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=(f"{scanned} file(s) mentioning {_OPT_STATE_KWARG} scanned, "
                   f"{callsites} keyword callsite(s) considered"),
    )


CONFOUND_GATES = (*CONFOUND_GATES,
                  check_checkpoint_saves_do_not_silently_drop_optimizer_state)


# ---------------------------------------------------------------------------
# ddm_gb1 (2026-08-15) — THE THROTTLE THAT COULD NOT RE-ARM, AND THE ADMISSION
# PATH THAT COUNTED PHANTOM ROWS.
#
# MEASURED INCIDENT. tools/memory_blackbox.py --daemon (pid 8997) SIGSTOPped every
# throttle-eligible python at ~16:53 local and never resumed one: the mp2 differential
# eval, the wc1 decode run, the wd3 W0 warm train, the dashboard and three safe_run
# wrappers sat in state `T` for 75+ minutes on a box with 40.5 GiB free. Receipt in
# .omx/state/memory_blackbox.daemon.log: "GOVERNOR ALERT avail=40.4GiB level=warn".
# Cause: decide_governor_action resumed on `level == "normal"` ALONE, and
# classify_pressure returns "warn" whenever the macOS kern.memorystatus_vm_pressure_level
# reads >= 2 — a STICKY signal. The guard's re-arm reference was the one thing it could
# not observe recovering. Exactly the spike-guard median-freeze genus (#304).
#
# SAME DAY, SECOND HALF. The wd3 relaunch was REFUSED with "active-growth 100.0 GiB" =
# 4 registry rows x UNKNOWN_GROWTH_HEADROOM_GIB (25.0), of which THREE were DEAD phantom
# `running` rows (pids 7506, 8997, 31881) in .omx/state/durable_daemons.json. A manual
# spawn_durable_daemon.reconcile_dead_daemons() converged them and the identical launch
# was admitted (projected 81.6 < ceiling 116.0). witness_memory_preflight (2026-07-09) and
# spawn_durable_daemon._do_start (2026-07-11) both auto-reconcile before projecting.
# safe_run did not. An ASYMMETRY between sibling admission paths is invisible to every
# gate that checks a path in isolation.
#
# Both legs are the same class: a decision that reads a stale reference and cannot tell.
# One gate, two anti-patterns — per the Catalog #299 consolidation discipline, not a
# pure-additive pair.
_THROTTLE_REARM_WAIVER = "THROTTLE_REARM_OK"
_ADMISSION_RECONCILE_WAIVER = "ADMISSION_RECONCILE_OK"

# A function is a THROTTLE RESUME POLICY iff it names the SIGSTOP-throttle's resume actuator or its
# action payload. Deliberately NOT the bare word "resume": checkpoint resume is everywhere in this
# repo (measured: 40+ files) and a detector that matched it would be permanently red, which is how
# a gate gets parked behind a flag where it can never fire (the #821 lesson).
_THROTTLE_RESUME_MARKERS = ("resume_job", "resume_targets")
# ...and it gates that resume on a PRESSURE CLASSIFICATION — the OS-derived signal that went sticky.
_PRESSURE_GATE_MARKERS = (
    "pressure_level", "PRESSURE_NORMAL", "PRESSURE_WARN", "PRESSURE_CRITICAL",
    '"normal"', "'normal'", '"warn"', "'warn'", '"critical"', "'critical'",
)
# The two cures, named. The detector ZEROES ON THE CURE: with both present it reads clean, so it
# cannot be satisfied by anything except the re-arm actually being wired.
_THROTTLE_REARM_CURE_MARKERS = ("resume_free_gib", "max_stop_duration")

# ── LEG C markers (ddm_mb1 2026-08-16): the ACTUATOR's ARMING, not its mechanism ─────────────────
# ddm_gb1 cured the throttle's mechanism (right object / re-arm / exit-resume) but left it ON by a
# hardcoded default, with an auto-start path that passed no opt-out — so the next training launch
# would have silently restarted the un-adjudicated SIGSTOP actuator. Leg C refuses the two shapes
# that make "the actuator is off" a fact about nobody having launched yet rather than about code.
_THROTTLE_ARM_WAIVER = "THROTTLE_ARM_OK"
# The cure: the switch is resolved from the durable arming surface, never hardcoded.
_THROTTLE_ARM_MARKERS = ("throttle_arming", "throttle_armed")
# A function SPAWNS the actuator iff it builds the black-box daemon's argv.
_ACTUATOR_SPAWN_MARKERS = ("memory_blackbox", '"--daemon"')
# ...and forces it on iff that argv carries the force flag.
_ACTUATOR_FORCE_ON_TOKEN = '"--govern"'
# The actuator's own switch parameter.
_ACTUATOR_SWITCH_PARAM = "govern"


def _gb1_surface_files(root: Path) -> list[Path]:
    """DECLARED scope: tools/*.py + src/tac/**/*.py + experiments/*.py + scripts/*.py, minus tests
    and this module. Top-level only for experiments/ + scripts/ (their subtrees are vendored deps
    and frozen run bundles — the #830 denominator lesson). system_memory_governor.py is IN scope
    here: it owns the throttle policy, so excluding it would exclude the incident itself."""
    out: list[Path] = []
    for relative, pattern in (
        ("tools", "*.py"), ("src/tac", "**/*.py"), ("experiments", "*.py"), ("scripts", "*.py"),
    ):
        directory = root.joinpath(*relative.split("/"))
        if directory.is_dir():
            out.extend(sorted(directory.glob(pattern)))
    skip = {root / "src" / "tac" / "confound_gates.py"}
    return [p for p in out if p not in skip and "/tests/" not in p.as_posix()]


def _function_code_lines(node: ast.AST, lines: list[str]) -> list[str]:
    """The function's own source lines with COMMENTS and its DOCSTRING removed.

    Both removals matter. A comment naming the waiver marker (which contains "rearm") or narrating
    the fix would otherwise masquerade as the fix; and a docstring that describes the resume policy
    in prose would otherwise satisfy a code-level marker test. Only executable text votes."""
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    body = getattr(node, "body", None) or []
    doc_range: set[int] = set()
    if body:
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                and isinstance(first.value.value, str):
            doc_range = set(range(getattr(first, "lineno", 0) - 1,
                                  getattr(first, "end_lineno", 0)))
    kept = [ln for i, ln in enumerate(lines[start:end], start=start) if i not in doc_range]
    return _strip_comments("\n".join(kept)).splitlines()


def _call_func_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return ""


def check_throttle_rearms_and_admission_reconciles(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """ddm_gb1 (2026-08-15) — two stale-reference anti-patterns in the memory governor, one gate.

    **LEG A — a SIGSTOP-throttle resume gated on a pressure classification must carry its own
    re-arm.** Any function that names ``resume_job`` / ``resume_targets`` AND branches on a
    pressure class must also name ``resume_free_gib`` (the governor's OWN derived free-GiB
    reference) and ``max_stop_duration`` (the escape hatch that bounds ANY stuck reference). The
    macOS pressure level is sticky at warn; a resume that depends on it alone never fires. MEASURED:
    five jobs frozen 75+ minutes at 40.4 GiB available. Waiver: ``# THROTTLE_REARM_OK:<rationale>``
    anywhere in the function.

    **LEG B — an admission path that reads the durable-daemon registry must converge it first.**
    Any module calling ``live_admission_decision`` must also call ``reconcile_dead_daemons``. A
    daemon killed out-of-band cannot write ``recorded=stopped``, and every phantom ``running`` row
    charges 25 GiB (``UNKNOWN_GROWTH_HEADROOM_GIB``) of active growth. MEASURED: three dead rows =
    100 GiB of phantom growth that refused a live relaunch twice. Module granularity is the right
    unit because ``spawn_durable_daemon`` legitimately reconciles in the CALLER (``_do_start``,
    under the registry lock) rather than in its gate function. Waiver:
    ``# ADMISSION_RECONCILE_OK:<rationale>`` on the call line.

    **LEG C — the SIGSTOP actuator may not be ARMED BY DEFAULT** (ddm_mb1, 2026-08-16). Leg A cures
    the throttle's MECHANISM; Leg C cures its ARMING, which ddm_gb1 left untouched. Two shapes are
    refused: (C1) a function that builds the black-box daemon argv must not pass ``--govern``, which
    FORCES the actuator on — the auto-start path did exactly this by omission, so "the daemon is
    OFF" was true only because nobody had launched training yet; and (C2) a function taking the
    ``govern`` switch must not default it to a hardcoded ``True``, and a ``None`` default must
    actually resolve against ``throttle_arming()`` (an unresolved ``None`` is not a tracked
    default-OFF — it is an actuator no operator can arm, the same orphan class one step over).
    The RECORDER is deliberately out of scope: read-only observability defaults ON. Waiver:
    ``# THROTTLE_ARM_OK:<rationale>`` anywhere in the function.

    WARN-ONLY at landing per the "Strict-flip atomicity rule", though live count IS 0 in this
    landing (measured pre-fix: 3 Leg-A functions in 2 files + 2 Leg-B modules + 1 Leg-C hardcoded
    ``govern=True`` in ``memory_blackbox.run_daemon``; post-fix 0). It stays
    warn-only for one cycle because both legs scan a surface that sibling arms are actively editing
    (the launcher/governor family), and a strict gate that fires on someone else's in-flight commit
    trains readers to bypass the suite. Strict-flip condition: one clean cycle with live count 0.
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    considered = 0
    scanned = 0
    scanned_functions = 0
    admission_modules = 0
    scanned_actuator_spawns = 0
    scanned_actuator_switches = 0
    for path in _gb1_surface_files(root):
        considered += 1
        text = _read(path)
        if not text:
            continue
        # Cheap substring prefilter BEFORE ast.parse (the sister gates' pattern). Parsing all 6474
        # in-scope files measured 18.9 s, which is too slow to sit in preflight_all on every commit
        # — and a gate people route around is not protection. A file that mentions none of these
        # tokens cannot contain either anti-pattern, so skipping it costs no coverage; `considered`
        # is still reported next to `scanned` so the prefilter can never hide a narrowed scan.
        # ddm_mb1: Leg C's tokens join the prefilter. Without this a file that ONLY carries the
        # actuator-arming anti-pattern (no resume_job, no admission call) would be skipped before
        # the AST parse — and the Leg-C positive controls, which are exactly that shape, would
        # never fire. A prefilter that hides a leg is the "vacuity == pass" failure.
        if not any(tok in text for tok in (*_THROTTLE_RESUME_MARKERS, "live_admission_decision",
                                           _ACTUATOR_SWITCH_PARAM)):
            continue
        scanned += 1
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        rel = path.relative_to(root).as_posix()

        # ── LEG A ──
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            code = _function_code_lines(node, lines)
            if not code:
                continue
            blob = "\n".join(code)
            if not any(m in blob for m in _THROTTLE_RESUME_MARKERS):
                continue
            if not any(m in blob for m in _PRESSURE_GATE_MARKERS):
                continue
            scanned_functions += 1
            if all(m in blob for m in _THROTTLE_REARM_CURE_MARKERS):
                continue
            if any(_waiver_present(ln, _THROTTLE_REARM_WAIVER)
                   for ln in lines[node.lineno - 1:getattr(node, "end_lineno", node.lineno)]):
                continue
            missing = [m for m in _THROTTLE_REARM_CURE_MARKERS if m not in blob]
            violations.append(
                f"{rel}:{node.lineno}: {node.name}() resumes a SIGSTOP-throttled job on a PRESSURE "
                f"CLASSIFICATION but is missing {missing} — the macOS pressure level is STICKY at "
                f"warn, so a resume gated on it alone never fires (MEASURED 2026-08-15: five jobs "
                f"SIGSTOPped 75+ minutes at 40.4 GiB available). Pass the governor's own "
                f"`resume_free_gib` reference and a `max_stop_duration_s` escape hatch, or add a "
                f"`# {_THROTTLE_REARM_WAIVER}:<rationale>` waiver."
            )

        # ── LEG C: the SIGSTOP actuator may not be ARMED BY DEFAULT ──
        # Placed BEFORE Leg B on purpose: Leg B short-circuits with `continue` on files with no
        # admission call, which would silently skip every Leg-C check in the same file.
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            code = _function_code_lines(node, lines)
            if not code:
                continue
            blob = "\n".join(code)
            fn_lines = lines[node.lineno - 1:getattr(node, "end_lineno", node.lineno)]
            waived = any(_waiver_present(ln, _THROTTLE_ARM_WAIVER) for ln in fn_lines)

            # C1 — a spawn path must not FORCE the actuator on.
            if all(m in blob for m in _ACTUATOR_SPAWN_MARKERS):
                scanned_actuator_spawns += 1
                if _ACTUATOR_FORCE_ON_TOKEN in blob and not waived:
                    violations.append(
                        f"{rel}:{node.lineno}: {node.name}() spawns the memory-blackbox daemon with "
                        f"{_ACTUATOR_FORCE_ON_TOKEN} — that FORCES the SIGSTOP throttle on and "
                        f"re-creates the silent re-enable (MEASURED 2026-08-15: the throttle "
                        f"SIGSTOPped three live measurements for 75+ minutes on a 40.5-GiB-free "
                        f"box). Pass neither --govern nor --no-govern so the daemon defers to the "
                        f"durable arming surface, or add a `# {_THROTTLE_ARM_WAIVER}:<rationale>`."
                    )

            # C2 — the actuator's own switch must not default to a hardcoded ON, and a None default
            # must actually RESOLVE against the arming surface (else "off" is unreachable-forever,
            # which is just the opposite silent default).
            args = getattr(node, "args", None)
            if args is None:
                continue
            switch_defaults: list[ast.expr] = []
            # POSITIONAL: ``args.defaults`` aligns to the TAIL of posonlyargs + args COMBINED.
            # Splitting those two lists (or ignoring posonlyargs) mis-aligns the offset and can
            # attribute a default to the wrong parameter — a false positive on someone else's arg.
            positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
            pos_offset = len(positional) - len(args.defaults)
            for idx, default in enumerate(args.defaults):
                pos = pos_offset + idx
                if 0 <= pos < len(positional) and positional[pos].arg == _ACTUATOR_SWITCH_PARAM:
                    switch_defaults.append(default)
            # KEYWORD-ONLY: ``kw_defaults`` is 1:1 with kwonlyargs, with None where there is none.
            # strict=True: CPython guarantees these are 1:1; a mismatch would be a corrupt AST and
            # should raise here rather than silently truncate the scan.
            for kwarg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if default is not None and kwarg.arg == _ACTUATOR_SWITCH_PARAM:
                    switch_defaults.append(default)
            if not switch_defaults:
                continue
            scanned_actuator_switches += 1
            if waived:
                continue
            for default in switch_defaults:
                if isinstance(default, ast.Constant) and default.value is True:
                    violations.append(
                        f"{rel}:{node.lineno}: {node.name}() defaults `{_ACTUATOR_SWITCH_PARAM}=True` "
                        f"— the SIGSTOP actuator is ON by a hardcoded default, so every auto-start "
                        f"re-arms it with no operator adjudication and no recorded reason. Default "
                        f"it to None and resolve via {_THROTTLE_ARM_MARKERS[0]}(), or add a "
                        f"`# {_THROTTLE_ARM_WAIVER}:<rationale>` waiver."
                    )
                elif (isinstance(default, ast.Constant) and default.value is None
                      and not any(m in blob for m in _THROTTLE_ARM_MARKERS)):
                    violations.append(
                        f"{rel}:{node.lineno}: {node.name}() defaults "
                        f"`{_ACTUATOR_SWITCH_PARAM}=None` but never resolves it against the arming "
                        f"surface ({' / '.join(_THROTTLE_ARM_MARKERS)}) — an unresolved None is not "
                        f"a tracked default-OFF, it is an actuator no operator can ever arm. Call "
                        f"`{_THROTTLE_ARM_MARKERS[0]}()`, or add a "
                        f"`# {_THROTTLE_ARM_WAIVER}:<rationale>` waiver."
                    )

        # ── LEG B ──
        admission_calls = [n for n in ast.walk(tree)
                           if isinstance(n, ast.Call)
                           and _call_func_name(n) == "live_admission_decision"]
        if not admission_calls:
            continue
        admission_modules += 1
        if any(isinstance(n, ast.Call) and _call_func_name(n) == "reconcile_dead_daemons"
               for n in ast.walk(tree)):
            continue
        for node in admission_calls:
            line = lines[node.lineno - 1] if 0 < node.lineno <= len(lines) else ""
            if _waiver_present(line, _ADMISSION_RECONCILE_WAIVER):
                continue
            violations.append(
                f"{rel}:{node.lineno}: live_admission_decision() without a "
                f"reconcile_dead_daemons() anywhere in this module — every phantom `running` "
                f"registry row charges 25 GiB of active growth (MEASURED 2026-08-15: three dead "
                f"rows = 100.0 GiB of phantom growth REFUSED a live relaunch twice). Call "
                f"`spawn_durable_daemon.reconcile_dead_daemons(verbose=False)` fail-OPEN before "
                f"the decision, or add a `# {_ADMISSION_RECONCILE_WAIVER}:<rationale>` waiver."
            )
    return _finish(
        name="check_throttle_rearms_and_admission_reconciles",
        tag="throttle-rearm-and-admission-reconcile",
        violations=violations,
        strict=strict,
        verbose=verbose,
        # DECLARED DENOMINATOR: a narrowed scope must never print a clean OK over an empty scan.
        ok_detail=(
            f"{scanned_functions} throttle-resume function(s) + {admission_modules} admission "
            f"module(s) + {scanned_actuator_spawns} actuator-spawn path(s) + "
            f"{scanned_actuator_switches} actuator-switch default(s) checked in {scanned} of "
            f"{considered} in-scope source file(s) mentioning a marker "
            f"(scope: tools/*.py + src/tac/**/*.py + experiments/*.py + scripts/*.py)"
        ),
    )


CONFOUND_GATES = (*CONFOUND_GATES, check_throttle_rearms_and_admission_reconciles)

# The #831 ratchet: a REFUSE-capable gate lands WITH its executed positive controls, never bare.
# Both legs get one, because a single control would leave the other leg free to be gutted silently.
POSITIVE_CONTROLS = (
    *POSITIVE_CONTROLS,
    PositiveControl(
        gate="check_throttle_rearms_and_admission_reconciles",
        files={
            "tools/planted_throttle.py": (
                "def decide(level, jobs):\n"
                "    paused = [j for j in jobs if j.paused]\n"
                '    if level == "normal" and paused:\n'
                '        return ("resume", resume_targets(paused))\n'
                "    return None\n"
                "def resume_targets(paused):\n"
                "    return tuple(paused)\n"
            )
        },
        must_mention="planted_throttle.py",
        why=(
            "ddm_gb1 Leg A: the EXACT pre-fix decide_governor_action shape — a throttle resume "
            "keyed on `level == \"normal\"` with no derived-free reference and no escape hatch. "
            "The macOS pressure level is sticky at warn, so this resume never fires."
        ),
    ),
    PositiveControl(
        gate="check_throttle_rearms_and_admission_reconciles",
        files={
            "tools/planted_admission.py": (
                "import system_memory_governor as gov\n"
                "def gate(projected):\n"
                "    ctx = gov.live_admission_decision(projected_new_gib=projected)\n"
                "    if not ctx.decision.admit:\n"
                "        raise SystemExit(5)\n"
            )
        },
        must_mention="planted_admission.py",
        why=(
            "ddm_gb1 Leg B: the EXACT pre-fix safe_run shape — a REFUSING admission gate that "
            "reads the durable-daemon registry with no reconcile, so dead rows charge 25 GiB each."
        ),
    ),
    PositiveControl(
        gate="check_throttle_rearms_and_admission_reconciles",
        files={
            "tools/planted_actuator_spawn.py": (
                "def ensure_blackbox_running():\n"
                "    argv = [\n"
                '        "--label", "memory_blackbox",\n'
                '        "--", "python", "tools/memory_blackbox.py", "--daemon", "--govern",\n'
                "    ]\n"
                "    return spawn(argv)\n"
            )
        },
        must_mention="planted_actuator_spawn.py",
        why=(
            "ddm_mb1 Leg C1: an auto-start path that FORCES the SIGSTOP throttle on. This is the "
            "shape that made 'the daemon is OFF' a fact about nobody having launched training yet "
            "rather than about the code — the next launch would silently re-arm the actuator that "
            "froze three live measurements for 75+ minutes."
        ),
    ),
    PositiveControl(
        gate="check_throttle_rearms_and_admission_reconciles",
        files={
            "tools/planted_actuator_default.py": (
                "def run_daemon(*, interval=2.0, govern=True):\n"
                "    if govern:\n"
                "        pause_job(lowest_priority_target())\n"
                "    return 0\n"
            )
        },
        must_mention="planted_actuator_default.py",
        why=(
            "ddm_mb1 Leg C2: the EXACT pre-fix run_daemon shape — the SIGSTOP actuator ON by a "
            "hardcoded default, so every auto-start re-arms it with no operator adjudication and "
            "no recorded reason. 'Off' must be a tracked, armed state, never a forgotten default."
        ),
    ),
)


# ---------------------------------------------------------------------------
# ddm_pl1 (2026-08-16) -- THE READY RECORD MADE TO WAIT
#
# Sister of `tac.payload_retention_gate.check_no_measure_and_discard_payload`,
# and deliberately a DIFFERENT class.  That gate refuses a script whose only
# persisted artifact is SCALARS while the bytes were in memory -- a defect of
# DESIGN.  This gate refuses a script that persists BOTH, correctly, but ORDERS
# the two writes so that one failure loses both -- a defect of SEQUENCE.
# ddm_lr1/A2 was the second class, not the first, and the first gate could not
# have seen it.
# ---------------------------------------------------------------------------

_BULK_ATTRS = frozenset({"save", "savez", "savez_compressed", "dump", "tofile", "savemat"})
_BULK_RECVS = frozenset({"torch", "np", "numpy", "pickle", "joblib", "cloudpickle", "scipy"})
_RECORD_ATTRS = frozenset({"dump", "safe_dump"})
_RECORD_RECVS = frozenset({"json", "yaml", "toml", "orjson"})
# A dict literal smaller than this is a closure box or an options bag, not a run
# record.  MEASURED: `_tail_cycle_start_epoch = {"v": None}` in the levelset
# trainer produced three false positives at 1 key; the incident's own `result`
# carries 15.  Reading those sites (not trusting the count) set the threshold.
_MIN_RECORD_KEYS = 3
# Textual prefilter for the scan. Must stay a SUPERSET of every spelling
# `_pl1_primitive_role` treats as bulk, or the gate goes quietly blind.
# `test_the_prefilter_is_a_superset_of_the_bulk_predicate` pins that.
_PL1_BULK_TOKENS = (
    ".save(",
    ".savez(",
    ".savez_compressed(",
    ".dump(",
    ".tofile(",
    ".savemat(",
    ".write_bytes(",
)


def _pl1_attr_call(node: ast.AST) -> tuple[str, str] | tuple[None, None]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        recv = node.func.value
        name = recv.id if isinstance(recv, ast.Name) else getattr(recv, "attr", "")
        return name, node.func.attr
    return None, None


def _pl1_primitive_role(node: ast.Call) -> str | None:
    """BULK vs RECORD for one call, by WHAT IT WRITES -- never by its spelling."""
    recv, attr = _pl1_attr_call(node)
    if attr is None:
        return None
    if attr == "write_bytes" or (attr in _BULK_ATTRS and recv in _BULK_RECVS):
        return "bulk"
    # `json.dump(obj, fp)` persists; bare `json.dumps(obj)` only serialises, and
    # `print(json.dumps(result))` is stdout, not an artifact.  Requiring the file
    # argument is what keeps the CURED trainer (which prints its result after
    # saving) out of the violation set.
    if attr == "write_text" or (attr in _RECORD_ATTRS and recv in _RECORD_RECVS and len(node.args) >= 2):
        return "record"
    return None


def _pl1_helper_roles(tree: ast.AST) -> dict[str, str]:
    """Classify each module-local helper by the primitives ITS BODY calls.

    MECHANISM, not name-matching: ``_atomic_torch_save`` is bulk because it
    calls ``torch.save``, and would still be bulk under any other name.  Keying
    on the helper's spelling is what makes a rename silently disarm a gate --
    exactly how the sibling per-module test (which string-matches two literal
    call spellings) would fail open.
    """
    roles: dict[str, str] = {}
    for fn in _func_defs(tree):
        bulk = record = False
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            recv, attr = _pl1_attr_call(node)
            if attr == "write_bytes" or (attr in _BULK_ATTRS and recv in _BULK_RECVS):
                bulk = True
            elif attr in ("write_text", "write") or (attr in _RECORD_ATTRS and recv in _RECORD_RECVS):
                record = True
        if bulk and not record:
            roles[fn.name] = "bulk"
        elif record and not bulk:
            roles[fn.name] = "record"
    return roles


def _pl1_stmt_role(stmt: ast.stmt, roles: Mapping[str, str]) -> str | None:
    """The write role of ONE SIMPLE statement.

    BRANCHING compound statements (``if`` / ``for`` / ``while`` / ``try``) are
    excluded and recursed into as their own blocks: a write under a condition
    is not a sibling of one that always runs.

    ``with`` IS transparent, because it does not branch -- it is straight-line
    code in a resource scope, and ``with open(p, "w") as fh: json.dump(...)`` is
    the ordinary way to write JSON in Python.  My own positive control caught
    this: the first draft excluded ``with`` wholesale and could not see its own
    planted violation, which would have made the gate near-vacuous on real code.
    """
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        for inner in stmt.body:
            role = _pl1_stmt_role(inner, roles)
            if role is not None:
                return role
        return None
    if not isinstance(stmt, (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)):
        return None
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Call):
            continue
        role = _pl1_primitive_role(node)
        if role is not None:
            return role
        if isinstance(node.func, ast.Name):
            local = roles.get(node.func.id)
            if local is not None:
                return local
    return None


def _pl1_serialized_names(stmt: ast.stmt, roles: Mapping[str, str]) -> set[str]:
    """Names inside the ARGUMENTS of this statement's record-persisting calls.

    The record must be the object BEING SERIALISED -- not merely a name that
    appears somewhere in the line.  MEASURED false positive that forced this:
    ``experiments/tests/test_ddm_cx2_trace_evaluate.py`` builds a dict of test
    FIXTURE PATHS and then calls ``deps["frame_utils.py"].write_text(...)``.
    There the dict is the subscript RECEIVER, and nothing about it is a run
    record.  Restricting to the argument subtree drops it and keeps the real
    sites (``profile_fp4_layer_sensitivity`` nests ``metadata`` inside the dict
    literal it serialises, and that is still an argument).
    """
    out: set[str] = set()
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Call):
            continue
        role = _pl1_primitive_role(node)
        if role is None and isinstance(node.func, ast.Name):
            role = roles.get(node.func.id)
        if role != "record":
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            out |= {n.id for n in ast.walk(arg) if isinstance(n, ast.Name)}
    return out


def _pl1_record_names(stmt: ast.stmt) -> set[str]:
    """Names bound HERE to a dict literal of at least :data:`_MIN_RECORD_KEYS`."""
    if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Dict):
        return set()
    if len(stmt.value.keys) < _MIN_RECORD_KEYS:
        return set()
    return {t.id for t in stmt.targets if isinstance(t, ast.Name)}


def _pl1_block_violations(
    body: list[ast.stmt], roles: Mapping[str, str], path_label: str
) -> list[str]:
    """One statement list.  Siblings only -- the ordering is what is at stake."""
    out: list[str] = []
    seq: list[tuple[ast.stmt, str | None]] = []
    for stmt in body:
        # A bulk write already inside try/except|finally cannot strand the
        # record write: the handler runs and the record still lands.  That is
        # the second legal cure, so a guarded save is never a bulk SITE here.
        seq.append((stmt, "guarded" if isinstance(stmt, ast.Try) else _pl1_stmt_role(stmt, roles)))
    for i, (bulk_stmt, bulk_role) in enumerate(seq):
        if bulk_role != "bulk":
            continue
        for j in range(i + 1, len(seq)):
            record_stmt, record_role = seq[j]
            if record_role != "record":
                continue
            used = _pl1_serialized_names(record_stmt, roles)
            for k in range(i):
                built = _pl1_record_names(seq[k][0]) & used
                if not built:
                    continue
                out.append(
                    f"{path_label}:{bulk_stmt.lineno}: bulk payload write runs BEFORE the "
                    f"already-built record {sorted(built)[0]!r} (built line {seq[k][0].lineno}, "
                    f"persisted line {record_stmt.lineno}) -- if this save raises, the run's "
                    f"only readable product is lost. Persist the record first, or guard the save."
                )
                break
            break
    return out


def _pl1_walk(body: list[ast.stmt], roles: Mapping[str, str], label: str) -> list[str]:
    out = _pl1_block_violations(body, roles, label)
    for stmt in body:
        for field in ("body", "orelse", "finalbody"):
            nested = getattr(stmt, field, None)
            if isinstance(nested, list) and nested and isinstance(nested[0], ast.stmt):
                out.extend(_pl1_walk(nested, roles, label))
        for handler in getattr(stmt, "handlers", []) or []:
            out.extend(_pl1_walk(handler.body, roles, label))
    return out


def _pl1_scan(repo: Path) -> tuple[int, list[str], list[str]]:
    """(modules_examined, violations, unparsed). The DENOMINATOR is returned.

    A gate that reports "0 violations" over a scan it never states the size of
    is indistinguishable from a gate that scanned nothing (the vacuity==pass
    class). Both legs travel together -- and a module the parser CHOKED on is
    reported by name rather than folded silently into the cleared count, because
    "could not analyse" is not "clean".
    """
    violations: list[str] = []
    unparsed: list[str] = []
    scanned = 0
    for root in ("src", "tools", "scripts", "experiments"):
        base = repo / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            # `experiments/results` is harvested RUN OUTPUT, not source we
            # maintain; scanning it would quadruple the cost and report defects
            # nobody can fix in place.  Named here so the denominator is honest.
            if {"results", "site-packages", ".venv", "__pycache__"} & set(path.parts):
                continue
            text = _read(path)
            if text is None:
                continue
            scanned += 1
            # Cheap textual prefilter BEFORE the AST parse. A module with no
            # bulk-write spelling anywhere in its bytes cannot contain the
            # pattern, so parsing it is pure cost. It is still EXAMINED and
            # still counted in the denominator -- the prefilter skips the parse,
            # never the population. (Shrinking the reported denominator to buy
            # speed is the same dishonesty as shrinking it to buy a green.)
            if not any(token in text for token in _PL1_BULK_TOKENS):
                continue
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError):
                unparsed.append(str(path.relative_to(repo)))
                continue
            roles = _pl1_helper_roles(tree)
            lines = text.splitlines()
            label = str(path.relative_to(repo))
            for fn in _func_defs(tree):
                for item in _pl1_walk(fn.body, roles, label):
                    lineno = int(item.split(":")[1])
                    line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
                    if _waiver_present(line, "PAYLOAD_WRITE_ORDER_OK"):
                        continue
                    violations.append(item)
    return scanned, violations, unparsed


def payload_write_order_population(repo_root: str | Path | None = None) -> dict[str, object]:
    """The declared denominator for :func:`check_no_bulk_write_strands_the_ready_record`."""
    scanned, violations, unparsed = _pl1_scan(Path(repo_root or REPO_ROOT))
    return {
        "modules_examined": scanned,
        "violations": violations,
        "live_count": len(violations),
        "unparsed": unparsed,
    }


def check_no_bulk_write_strands_the_ready_record(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Refuse: a fragile BULK write scheduled ahead of an already-built RECORD.

    THE INCIDENT (ddm_lr1 arm A2, 2026-08-16).  The run finished 600 steps on
    Metal, re-evaluated the deployment EMA, passed argmax parity, and built its
    whole ``result`` dict -- ``final_seg``, the parity report, the packed byte
    count, the verdict, the full six-row ``history``.  Then it wrote the 1.7 MB
    checkpoint FIRST.  ``os.replace`` hit an existing empty directory at
    ``--save``, raised ``IsADirectoryError``, and the ``result`` write two lines
    later never ran.  379 s of compute, no ``result.json`` at all -- and the
    ``safe_run`` receipt read ``status=ok exit=1``.

    The two artifacts are NOT interchangeable and must not share a fate:

    * the RECORD is cheap and IRREPLACEABLE -- scalars that cost a full final
      evaluation to produce and cannot be recovered without re-running;
    * the BULK payload is expensive but REBUILDABLE -- and it is the one whose
      write is most likely to fail (size, disk, a path that is a directory).

    Ordering the rebuildable-and-fragile write ahead of the
    irreplaceable-and-cheap one inverts ALWAYS KEEP THE PAYLOAD (P0, operator
    2026-08-09).  Either cure passes: persist the record first, or wrap the bulk
    write in ``try``.

    SISTER, not duplicate.  ``check_no_measure_and_discard_payload`` refuses a
    run that persists ONLY scalars while bytes sat in memory -- a defect of
    DESIGN, detected by a measurement with no adjacent write.  A2 persisted both
    by design and still lost its product, so that gate is silent here.  This one
    keys on SEQUENCE.

    Population MEASURED, not assumed, and the DENOMINATOR is stated: 11,016
    modules under ``src``, ``tools``, ``scripts`` and ``experiments``
    (``experiments/results`` excluded -- harvested run output, not maintained
    source).  Live count at landing is **10**, NOT zero, so this lands
    WARN-ONLY.  Narrowing the scan until the count read zero was available and
    refused: a strict gate bought by shrinking its own population is the
    vacuity==pass class wearing a green label.
    STRICT-FLIP CONDITION: the ten sites listed in
    ``.omx/research/ddm_pl1_payload_loss_two_landing_20260816.md`` are cured or
    waived, at which point flip the wire-in in ``tac.preflight``.

    Same-line waiver ``# PAYLOAD_WRITE_ORDER_OK:<rationale>`` on the bulk-write
    line, for a record that genuinely cannot be written until the save returns
    (a manifest that must record the artifact's real sha).
    """

    scanned, violations, unparsed = _pl1_scan(Path(repo_root or REPO_ROOT))
    return _finish(
        name="check_no_bulk_write_strands_the_ready_record",
        tag="payload-write-order",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=(
            f"{scanned} modules examined ({len(unparsed)} unparseable), "
            "no ready record stranded behind a bulk write"
        ),
    )


CONFOUND_GATES = (*CONFOUND_GATES, check_no_bulk_write_strands_the_ready_record)

POSITIVE_CONTROLS = (
    *POSITIVE_CONTROLS,
    PositiveControl(
        gate="check_no_bulk_write_strands_the_ready_record",
        files={
            "src/tac/planted_stranded_record.py": (
                "import json\n"
                "import torch\n"
                "def _atomic_torch_save(payload, path):\n"
                "    torch.save(payload, path)\n"
                "def finalize(model, args, history):\n"
                "    result = {'verdict': 'PASS', 'seg': 1.0, 'history': history}\n"
                "    _atomic_torch_save({'sd': model}, args.save)\n"
                "    with open(args.out, 'w') as fh:\n"
                "        json.dump(result, fh)\n"
                "    return result\n"
            )
        },
        must_mention="planted_stranded_record.py",
        why=(
            "The EXACT ddm_lr1/A2 shape: the run's `result` is fully built, then a bulk "
            "checkpoint save is scheduled ahead of it, then the record is persisted. A2's save "
            "raised IsADirectoryError and 379 s of Metal produced no result.json at all. If a "
            "future change makes the gate classify helpers by NAME instead of by the primitives "
            "they call, stop recursing into nested blocks, or drop the dict-literal record "
            "predicate, this control stops firing."
        ),
    ),
)


# ===========================================================================
# ddm_cd1 (2026-08-17) — a DEAD CONDITIONAL RE-TEST left behind by an early
# return: the fingerprint of an ACCIDENTAL builder truncation.
#
# ANCHOR (live, MEASURED): tools/costate_digest.py build_digest() opened
# `if ddm_live:` at :2200 and closed it with `return lines, data` at :2251.
# Everything from :2253 to :2330 became unreachable whenever ddm_live was true
# — which is the LIVE state. The consumer harm was not a missing key (the
# branch re-provided a total 16-key schema) but FOUR keys carrying plausible
# WRONG values: `verdict_scope` shadowed by an unrelated provenance dict,
# `corpus_recall` hard-bound to `[]` in violation of its own `dict | None`
# contract, `active_convening`/`graph_memory` hard-bound to None. A JSON
# consumer read a plausible dict and a plausible empty list and never learned
# that 7 recall advisories had fired in 14 days.
#
# WHY THIS SIGNATURE, and not "early return truncates an accumulator": that
# broader shape was implemented and MEASURED first — 30 sites across 30 files,
# and on inspection essentially all were legitimate error-cascade guards that
# RECORD why they bailed (`blockers.append(...); return summary, blockers`).
# Neither accumulator shape nor dict-key-set parity separates those from the
# anchor, because the anchor's author deliberately maintained a total schema;
# the defect lived in the VALUES. A gate on the broad shape would be
# permanently amber over benign code, and a gate readers ignore is not
# protection. Verdict on the broad shape: FORMULATION-level negative, not a
# statement that no gateable form exists.
#
# What DOES separate them is provable: commit 7fac2e7475 added the early
# return AND, in the same commit, an `if ddm_live:` at :2304 INSIDE the region
# it had just orphaned. You do not write a branch into code you meant to make
# unreachable. A re-test of a name an earlier early-return already decided is
# machine-provable dead code and is the accidental-truncation fingerprint;
# deliberate early returns do not leave dead re-tests behind.
#
# SIGNATURE refused: a top-level `if <name>:` in a function whose body is
# preceded by another top-level `if <name>:` that (a) returns unconditionally
# (its body's last statement is a `Return`) and (b) has no `else`, where
# `<name>` is neither rebound nor mutated in between. On every reachable path
# the later test is constant-false, so its branch is dead.
#
# DECLARED NARROWING (round-2 review, stated rather than left implicit): both
# tests must be a BARE `ast.Name`. `if not x:` / `if x is None:` / compound
# tests are NOT matched, so a `if not x: return` guard followed by `if not x:`
# is a MISS. That is deliberate — negated and compound forms invert which
# later test is dead vs constant-TRUE, and getting it wrong would put false
# positives into a strict gate. The bare-name form is the anchor's shape and
# the only one this gate claims. An un-declared narrowing is the VACUITY
# failure; a declared one is a scope.
#
# SHAPE NOTE (where a lazy detector misses the anchor): the anchor's early
# return is `return lines, data` — VALUE-returning. A detector keyed on a bare
# `return` (`node.value is None`) reports a clean scan over the very incident
# that motivated it — the ddm_qd1 "the detector's AST shape did not match how
# the code expresses the thing" genus. Returns are matched by POSITION only.
#
# MUTATION NOTE (a real bug caught in this gate's own bring-up): the first
# draft flagged `harvest_cuda_cpu_axis_profile_registry.build_combined_payload
# _from_pair`, where `if blockers: return ...` is followed by more
# `blockers.append(...)` and a second `if blockers:`. That second test is very
# much alive. Clearing the decision only on NAME REBINDING misses method-call
# mutation, so `_name_mutated_between` treats `x.append(...)`-style calls and
# item stores as invalidating too.
#
# Waiver (in-function): ``# DEAD_CONDITIONAL_RETEST_OK:<rationale>`` — for a
# deliberately redundant defense-in-depth branch retained against future
# control-flow changes.
# ===========================================================================

_BARE_NAME_IF_RE = re.compile(r"^[ \t]*if[ \t]+[A-Za-z_]\w*[ \t]*:[ \t]*$", re.M)

_MUTATING_METHODS = (
    "append", "extend", "update", "setdefault", "insert", "add",
    "pop", "remove", "clear", "sort", "reverse", "discard",
)


def _name_mutated_between(stmts: list[ast.stmt], name: str) -> bool:
    """True iff ``name`` is rebound OR mutated in place anywhere under ``stmts``."""
    for st in stmts:
        for node in ast.walk(st):
            if isinstance(node, ast.Name) and node.id == name and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _MUTATING_METHODS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == name
            ):
                return True
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == name
                and isinstance(node.ctx, (ast.Store, ast.Del))
            ):
                return True
            if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
                return True
    return False


def _python_sources_for_dead_retest(root: Path) -> list[Path]:
    """Hand-authored repo Python. ``experiments/results/**`` is GENERATED
    artifact territory (~40k files, +60 s) and is excluded deliberately; the
    gate's declared denominator reports what WAS parsed so the narrowing is
    visible rather than silent (the VACUITY==PASS discipline)."""
    out: list[Path] = []
    for sub in ("tools", "src/tac", "scripts"):
        base = root / sub
        if base.is_dir():
            out.extend(base.rglob("*.py"))
    exp = root / "experiments"
    if exp.is_dir():
        out.extend(exp.glob("*.py"))
    keep: list[Path] = []
    for p in out:
        parts = set(p.parts)
        if ".venv" in parts or "site-packages" in parts:
            continue
        if any("_intake_" in seg for seg in p.parts):
            continue
        keep.append(p)
    return sorted(set(keep))


def check_no_dead_conditional_retest_after_early_return(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Refuse a conditional re-test that an earlier early-return already decided.

    See the block comment above for the anchor incident (costate_digest
    build_digest, 18 statements orphaned, 4 keys poisoned, 8 digest lines lost),
    for why this signature was chosen over the broader
    early-return-truncates-an-accumulator shape (MEASURED: 30 benign sites), and
    for the mutation bug caught during bring-up.

    Waiver (anywhere in the offending function): a real
    ``# DEAD_CONDITIONAL_RETEST_OK:<rationale>``. Placeholder rationales are
    rejected per the Catalog #287 discipline.
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    n_considered = n_files = n_funcs = n_guards = 0
    # PRE-FILTER on a PROVABLY NECESSARY condition, never a heuristic: a
    # violation needs TWO top-level `if <bare-name>:` tests in one function, so a
    # file with fewer than two such lines cannot hold one. Cuts ~10.8k parses to
    # the candidates (28 s -> ~4 s) without narrowing the detector. Both counts
    # ride in the denominator so the filter can never hide a shrunken scope.
    for path in _python_sources_for_dead_retest(root):
        n_considered += 1
        text = _read(path)
        if not text:
            continue
        if len(_BARE_NAME_IF_RE.findall(text)) < 2:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        n_files += 1
        src_lines = text.splitlines()
        rel = path.relative_to(root).as_posix()
        for fn in _func_defs(tree):
            n_funcs += 1
            body = fn.body
            # name -> (lineno of the deciding guard, index of that statement)
            decided: dict[str, tuple[int, int]] = {}
            for i, st in enumerate(body):
                if not isinstance(st, ast.If) or not isinstance(st.test, ast.Name):
                    continue
                name = st.test.id
                prior = decided.get(name)
                if prior is not None and not _name_mutated_between(
                    body[prior[1] + 1 : i], name
                ):
                    if _waiver_present(
                        _span_source(src_lines, fn), "DEAD_CONDITIONAL_RETEST_OK"
                    ):
                        continue
                    violations.append(
                        f"{rel}:{st.lineno}: `if {name}:` is DEAD — the guard at "
                        f"line {prior[0]} in {getattr(fn, 'name', '?')!r} already "
                        f"returns unconditionally when {name} is truthy, and "
                        f"{name} is not rebound or mutated in between, so this "
                        f"branch is unreachable. This is the fingerprint of an "
                        f"early return that silently truncated the rest of the "
                        f"function: make the split an explicit if/else so the "
                        f"shared tail runs on BOTH paths, or add a "
                        f"`# DEAD_CONDITIONAL_RETEST_OK:<why this redundant "
                        f"branch is retained>` waiver."
                    )
                    continue
                if st.body and isinstance(st.body[-1], ast.Return) and not st.orelse:
                    n_guards += 1
                    decided[name] = (st.lineno, i)
    return _finish(
        name="check_no_dead_conditional_retest_after_early_return",
        tag="dead-conditional-retest-after-early-return",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=(
            f"{n_considered} file(s) considered, {n_files} parsed after the "
            f"necessary-condition pre-filter, {n_funcs} function(s), "
            f"{n_guards} unconditional-return guard(s) tracked"
        ),
    )


CONFOUND_GATES = (
    *CONFOUND_GATES,
    check_no_dead_conditional_retest_after_early_return,
)

POSITIVE_CONTROLS = (
    *POSITIVE_CONTROLS,
    PositiveControl(
        gate="check_no_dead_conditional_retest_after_early_return",
        files={
            "tools/planted_truncated_builder.py": (
                "def build_digest(ddm_live):\n"
                "    lines = []\n"
                "    data = {}\n"
                "    lines.append('pointer')\n"
                "    if ddm_live:\n"
                "        data['schedule'] = 'dominated'\n"
                "        lines.append('BOUNDARY')\n"
                "        return lines, data\n"
                "    lines.append('tail')\n"
                "    if ddm_live:\n"
                "        data['costate_organ'] = 'live'\n"
                "    else:\n"
                "        data['costate_organ'] = 'legacy'\n"
                "    return lines, data\n"
            )
        },
        must_mention="planted_truncated_builder.py",
        why=(
            "The EXACT ddm_cd1 anchor shape reproduced from tools/costate_digest.py "
            "commit 7fac2e7475: `if ddm_live:` ends in `return lines, data`, and a "
            "second `if ddm_live:` sits below in the region that return just "
            "orphaned. Note the return is VALUE-returning -- if a future change keys "
            "the detector on a bare `return` (node.value is None), stops requiring "
            "the guard body's LAST statement to be the return, drops the no-`else` "
            "requirement, or narrows the file scope away from tools/, this control "
            "stops firing and the gate reports a clean scan over its own anchor."
        ),
    ),
)
