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
"""

from __future__ import annotations

import ast
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

    STRICT-FLIP CONDITION: flip to ``strict=True`` once the trainer sibling flips
    ``--spike-guard-mode`` default -> ``rollback`` (or autoconfig injects it), i.e.
    live-count reaches 0. Warn-only until then because the trainer file is owned
    by a sibling commit.
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
# Re-arm escape-hatch tokens: any presence in the enclosing function clears the
# gate (the filter can recover from a sustained all-skip freeze).
_REARM_TOKENS = (
    ".clear(",
    "rearm",
    "re-arm",
    "re_arm",
    "reanchor",
    "re-anchor",
    "re_anchor",
    "reset_median",
    "reset(",
    "rollback",
    "spike_guard_mode",
    "resume_clear_spike_guard",
    "clear_spike_guard",
    "quantile_decay",
    "quantile-decay",
    "decay_median",
    "re_arm_median",
)


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
            # Escape hatch present anywhere in the function? Scan CODE only
            # (comments stripped) so the waiver-marker name (which contains
            # "rearm") cannot self-satisfy the re-arm requirement.
            fn_code_l = _strip_comments(fn_src).lower()
            if any(tok in fn_code_l for tok in _REARM_TOKENS):
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

    STRICT-FLIP CONDITION: flip to ``strict=True`` once the trainer sibling flips
    the default -> 0 (live-count 0). Warn-only until then.
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


# Convenience: the six gates in catalog order, for the preflight wire-in + tests.
CONFOUND_GATES = (
    check_no_spike_guard_defaults_to_deadlock_mode,
    check_reject_filter_updates_reference_from_accepted_only_has_rearm,
    check_no_duplicate_long_flags_in_launch,
    check_resume_palliative_flags_imply_warm_start,
    check_verdict_pairs_default_is_n600,
    check_telemetry_verdict_rows_carry_liveness,
)
