"""Live-vehicle lever activation: derive the queue from SOURCE, ingest the trainer's own emission.

Why this module exists (MEASURED 2026-08-17, ``.omx/research/ddm_todo_p0_live_lever_queue_20260817.md``):

``tac.witness_dsl.lever_registry.completeness()`` reports ``describes_live_vehicle: False`` — its
inventory is ``experiments/train_levelset_witness_realized_through_R_mlx.py`` (443 flags), two
vehicle-generations behind the trainer that actually produces the frontier
(``tac.pr130_lift.train_semantic_quantized_resumable``, 38 flags). Its 199-row ``never_fired()``
queue name-maps 34/199 onto the retired trainer and **0/199** onto the live one. Meanwhile
``.omx/state/lever_activation_ledger.jsonl`` has not been written since 2026-07-27.

The live trainer already emits the record we want, once per run::

    [b2e] editability levers ACTIVE: {"F1_weight_perturb_robustness": {"active": false,
      "reason_if_off": "sigma == 0 (default)", ...}, "F2_weight_qat_q3q4": {"active": true, ...}, ...}

It goes to ``run.log`` and nothing ingests it. This module is that one consumer.

**The anti-staleness contract.** The registry went stale because its vehicle was a hardcoded
pointer. So the live lever set here is DERIVED BY AST from the trainer's own ``add_argument``
calls at call time — it cannot describe a vehicle the code does not have. The only hand-declared
thing is :data:`PLUMBING_FLAGS` (paths, caches, device, cadence, seeds), which are not levers in
any vehicle and are listed explicitly so the exclusion is auditable rather than heuristic.

No score claim anywhere in this module. It records activation events only.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from tac.witness_dsl import activation_ledger as _al

__all__ = [
    "LIVE_TRAINER_PATH",
    "PLUMBING_FLAGS",
    "LiveLever",
    "EditabilityEmission",
    "live_levers",
    "parse_editability_emission",
    "ingest_run_log",
    "ingest_launch_manifest",
    "live_never_fired",
]

#: Repo-relative path to the trainer that produces the current frontier.
LIVE_TRAINER_PATH = Path("src/tac/pr130_lift/train_semantic_quantized_resumable.py")

#: Flags that are run plumbing, not levers, in ANY vehicle: filesystem paths, caches, the device,
#: evaluation/checkpoint cadence, batch shapes, and seeds. Declared explicitly (not pattern-matched)
#: so that every exclusion is auditable. A flag NOT in this set is treated as a lever.
#:
#: JUDGMENT CALL, stated so it can be disputed: ``--steps`` is listed here as the run BUDGET, but it
#: is not inert — the curriculum boundaries are FRACTIONS of the run, so changing steps stretches the
#: schedule (measured: `.omx/research/ddm_ce1_allocation_ladder_verdict_20260817.md`). It is excluded
#: because every run must choose a length, so "fired" would be true of all of them and the queue
#: would carry no information. Move it out if that reasoning ever stops holding.
PLUMBING_FLAGS = frozenset(
    {
        "--cache",
        "--challenge-root",
        "--init",
        "--input-cache",
        "--master-cache",
        "--out",
        "--resume-from",
        "--save",
        "--target-cache",
        "--smoke-pairs",
        "--parity-pairs",
        "--device",
        "--disable-tf32",
        "--batch-size",
        "--eval-batch-size",
        "--checkpoint-every",
        "--eval-every",
        "--seed",
        "--lever-seed",
        "--steps",
    }
)

_ADD_ARG = "add_argument"
_EMISSION_PREFIX = "[b2e] editability levers ACTIVE:"
# The emitted keys are "F<N>_<lever_name>"; the numeric prefix is display ordering, not identity.
_FKEY = re.compile(r"^F\d+_(?P<name>.+)$")


@dataclass(frozen=True)
class LiveLever:
    """One score-affecting flag on the live trainer, with the default it sits at."""

    flag: str
    default: str

    @property
    def ledger_name(self) -> str:
        """Canonical ledger key: ``--film-row-dropout`` -> ``film_row_dropout``."""
        return self.flag.removeprefix("--").replace("-", "_")


@dataclass(frozen=True)
class EditabilityEmission:
    """The trainer's own per-run lever record, parsed from ``run.log``."""

    run_ref: str
    active: tuple[str, ...]
    inactive: tuple[str, ...]
    reasons_if_off: dict[str, str]


def _literal(node: ast.AST) -> str:
    """Render a default expression as a short, honest string (never evaluated)."""
    try:
        return repr(ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError):
        return ast.unparse(node)


def live_levers(trainer_path: Path | None = None) -> tuple[LiveLever, ...]:
    """DERIVE the live trainer's score-affecting lever set by AST. Never a hardcoded list.

    Returns every ``add_argument("--flag", ...)`` on the trainer minus :data:`PLUMBING_FLAGS`,
    sorted by flag. If the trainer gains or loses a flag, this result moves with it — which is
    exactly what the retired registry could not do.
    """
    path = trainer_path or LIVE_TRAINER_PATH
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == _ADD_ARG):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        flag = first.value
        if not flag.startswith("--"):
            continue
        default = "REQUIRED"
        for kw in node.keywords:
            if kw.arg == "default":
                default = _literal(kw.value)
            elif kw.arg == "action" and default == "REQUIRED":
                default = "store_true"
        found[flag] = default
    return tuple(
        LiveLever(flag=f, default=d)
        for f, d in sorted(found.items())
        if f not in PLUMBING_FLAGS
    )


def parse_editability_emission(run_log: Path, run_ref: str | None = None) -> EditabilityEmission | None:
    """Parse the trainer's ``[b2e] editability levers ACTIVE:`` line. Returns None if absent.

    The emission may be split across lines by the logger; the JSON object is recovered by brace
    matching from the prefix, so reflowing does not break the parse.
    """
    text = run_log.read_text(encoding="utf-8", errors="replace")
    idx = text.find(_EMISSION_PREFIX)
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth, end = 0, -1
    in_str, esc = False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        blob = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None

    active: list[str] = []
    inactive: list[str] = []
    reasons: dict[str, str] = {}
    for key, rec in blob.items():
        if not isinstance(rec, dict):
            continue
        m = _FKEY.match(key)
        name = m.group("name") if m else key
        if rec.get("active") is True:
            active.append(name)
        else:
            inactive.append(name)
            why = rec.get("reason_if_off") or rec.get("state") or ""
            if why:
                reasons[name] = str(why)
    return EditabilityEmission(
        run_ref=run_ref or str(run_log.parent),
        active=tuple(sorted(active)),
        inactive=tuple(sorted(inactive)),
        reasons_if_off=reasons,
    )


def ingest_run_log(
    run_log: Path,
    *,
    run_ref: str | None = None,
    agent: str = "live_lever_activation",
    path: Path | None = None,
    dry_run: bool = False,
) -> tuple[dict, ...]:
    """Record a ``fired`` event for every lever the run reports ACTIVE. Returns the rows.

    Levers reported INACTIVE are deliberately NOT recorded: ``never_fired()`` derives the queue
    from the ABSENCE of a fired event, so writing anything for an off lever would corrupt the
    very signal this ingester exists to restore. Their ``reason_if_off`` rides along in the fired
    rows' reason text only as run context, never as a state claim.
    """
    emission = parse_editability_emission(run_log, run_ref=run_ref)
    if emission is None:
        return ()
    rows: list[dict] = []
    for name in emission.active:
        reason = f"ACTIVE in the live-vehicle [b2e] emission (ingested from {run_log.name})"
        if dry_run:
            rows.append({"lever": name, "event": _al.EVENT_FIRED, "run_ref": emission.run_ref, "reason": reason})
            continue
        rows.append(
            _al.record_activation(
                name,
                _al.EVENT_FIRED,
                run_ref=emission.run_ref,
                reason=reason,
                agent=agent,
                path=path,
            )
        )
    return tuple(rows)


def ingest_launch_manifest(
    manifest: Path,
    *,
    trainer_path: Path | None = None,
    agent: str = "live_lever_activation",
    path: Path | None = None,
    dry_run: bool = False,
) -> tuple[dict, ...]:
    """Record a ``fired`` event for every lever the run's argv passed OFF ITS DEFAULT.

    The ``[b2e]`` emission covers only the 5-lever editability family; the other live levers have
    no per-run telemetry at all, so for them the launch argv IS the record. A flag passed at its
    own default is NOT a firing — it leaves the config where it already sat — so the default is
    compared and equal values are skipped. Manifests for other trainers are ignored.
    """
    try:
        blob = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    argv = blob.get("effective_argv") or blob.get("argv") or []
    if isinstance(argv, str):
        argv = argv.split()
    argv = [str(a) for a in argv]
    stem = LIVE_TRAINER_PATH.stem
    if not any(stem in a for a in argv):
        return ()

    by_flag = {lev.flag: lev for lev in live_levers(trainer_path)}
    rows: list[dict] = []
    run_ref = str(manifest.parent)
    for i, tok in enumerate(argv):
        lev = by_flag.get(tok)
        if lev is None:
            continue
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        value_taking = lev.default != "store_true"
        if value_taking and (nxt is None or nxt.startswith("--")):
            # A value-taking flag with no value is malformed argv. We cannot know what it was
            # set to, so recording a firing would be a guess. Fail closed and skip.
            continue
        # NOTE: for nargs="*" flags only the first value is captured in the reason string. The
        # FIRING is still correct (the flag was passed off-default); only the display is partial.
        passed = "SET" if nxt is None or nxt.startswith("--") else nxt
        if _same_as_default(passed, lev.default):
            continue
        reason = f"passed {tok} {passed} (default {lev.default}) in the launch argv"
        if dry_run:
            rows.append({"lever": lev.ledger_name, "event": _al.EVENT_FIRED, "run_ref": run_ref, "reason": reason})
            continue
        rows.append(
            _al.record_activation(
                lev.ledger_name,
                _al.EVENT_FIRED,
                run_ref=run_ref,
                reason=reason,
                agent=agent,
                path=path,
            )
        )
    return tuple(rows)


def _same_as_default(passed: str, default: str) -> bool:
    """True when the argv value leaves the lever where it already sat (numeric-aware)."""
    if default in {"REQUIRED", "store_true"}:
        return False  # a store_true flag's presence IS the firing; REQUIRED has no default
    stripped = default.strip("'\"")
    if passed == stripped:
        return True
    try:
        return float(passed) == float(stripped)
    except (TypeError, ValueError):
        return False


def live_never_fired(
    trainer_path: Path | None = None, path: Path | None = None
) -> tuple[str, ...]:
    """The never-fired queue FOR THE LIVE VEHICLE — the duty-to-measure surface that matters.

    Unlike :func:`tac.witness_dsl.activation_ledger.never_fired`, whose default ``known`` set comes
    from the retired-vehicle registry, this derives ``known`` from the live trainer's own argparse.
    """
    known = tuple(lev.ledger_name for lev in live_levers(trainer_path))
    return _al.never_fired(known=known, path=path)
