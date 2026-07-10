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

All six gates land **WARN-ONLY** initially per the CLAUDE.md "Strict-flip
atomicity rule": the trainer/launcher fixes land in *sibling* commits, so this
builder cannot guarantee live-count 0 across files it does not own. Each gate's
docstring names its explicit strict-flip condition. The gates are wired into
``tac.preflight.preflight_all`` (warn-only) so they run every session.

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
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

# Repo root: src/tac/confound_gates.py -> parents[2] == repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Canonical witness trainer surfaces the confound hunt covered. Both are scanned
# by the source-level gates; only those that exist on disk are opened.
_TRAINER_FILES = (
    "experiments/train_levelset_witness_realized_through_R_mlx.py",
    "experiments/train_witness_realized_through_R_mlx.py",
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
        )
        from tac.witness_dsl.lever_registry import lever_factories
    except Exception as exc:  # pragma: no cover - import-environment guard
        return _finish(
            name="check_significance_keys_canonical",
            tag="significance-keys-canonical",
            violations=violations,
            strict=strict,
            verbose=verbose,
            ok_detail=f"activation_ledger/lever_registry unavailable ({exc!r}) — fail-open",
        )
    factory_names = set(lever_factories().keys())
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


# The two automatable eightfold gates (P1 + P4), for the preflight wire-in + tests.
EIGHTFOLD_GATES = (
    check_significance_keys_canonical,
    check_witness_control_meters_have_canaries,
)


# Convenience: the gates in catalog order, for the preflight wire-in + tests.
CONFOUND_GATES = (
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
)
