"""Run-constant hardcode gates — "anything hardcoded related to a run should be DSL".

Operator directive 2026-07-07 (task #340). The bug class: run CONSUMERS (dashboards,
launch wrappers, observers, viz/checkin tools) hardcoding parameters that duplicate a
property of the run's own config — stage boundaries (``--tau 300 --l7 600``), clip
constants, cadences, pair counts. When the curriculum/config evolves, the consumer
silently mislabels the run (the dashboard --tau/--l7 incident: two launchers disagreed
about the same run's l7 boundary — 600 vs 900 — while the live run had NO l7 stage).
The correct derivation sources are:

* ``tac.witness_dsl.schedule_readback.read_schedule`` — stage boundaries per run
  (launch.sh through the REAL trainer argparse + fired-transition evidence),
* ``tac.clip_profile`` — clip constants (resolution / intrinsics / class order),
* the run dir's own ``launch.sh`` / resume sidecar ``__cfg_*`` keys — everything else.

Gate: :func:`check_no_hardcoded_run_constants_in_consumers` (WARN-ONLY, ``strict=False``
— the false-positive risk on a heuristic source scan is real). Same-line waiver::

    ap.add_argument("--tau", type=int, default=300)  # RUN_CONSTANT_OK:<real rationale>

Placeholder rationales (``<rationale>``, ``<reason>``, ``TBD`` …) are REJECTED per the
Catalog #287 sister discipline.

Scope (deliberately tight to keep noise low):

* **Scanned**: ``tools/*.py`` + top-level ``src/tac/*.py`` consumer modules.
* **NOT scanned** (documented known-accepted cases):
  - ``experiments/`` trainers — argparse defaults there are the DSL's COMPILE TARGET
    (the DSL emits trainer argv; a trainer default is not a consumer hardcode);
  - tests (``test_*`` / ``tests/``) — fixtures pin constants deliberately;
  - ``src/tac/witness_dsl/`` — the DSL itself IS the source of truth;
  - ``src/tac/clip_profile.py`` — the canonical home of resolution constants;
  - vendored/intake clones (``_intake_`` path marker) — pristine per Check #109;
  - derive-first fallbacks of the form ``schedule.get("tau_start") or 300``
    (dashboard_trajectory_model) — the accepted derive-with-labeled-fallback pattern
    (no ``=``-assignment literal, so pattern P4 does not match by design);
  - float ``--tau`` flags with different semantics (e.g. a tolerance in
    ``legal_frame_feasibility_smoke``) — P1 requires ``type=int``.

Patterns:

* **P1** ``stage_cli_int_default`` — ``add_argument("--tau"|"--l7", … type=int …
  default=<int literal>)``: the exact incident class. Stage-flag defaults in consumers
  must be ``None`` (= derive via the DSL read-back; explicit value = override only).
* **P2** ``stage_flag_literal_in_string`` — ``--tau <int>`` / ``--l7 <int>`` embedded
  in a string (command constructions / printed operator hints / usage examples): the
  hint form of the same class (an operator copy-pastes the poisoning override).
* **P3** ``resolution_literal_in_display_tool`` — 874/1164 camera-resolution literals
  in display/labeling tools (``tools/dashboard_*``, ``tools/render_*``,
  ``tools/witness_checkin.py``): must come from ``tac.clip_profile``. Live count 0 at
  landing — a pure regression guard. (Build/measurement tools keep their deliberate
  provenance pins and are NOT in P3 scope.)
* **P4** ``stage_key_literal_assignment`` — ``tau_start|l7_start|muon_start = <int>``
  literal assignment outside the DSL.

Wiring note (deferred with a NAMED blocker): at landing 2026-07-07 the sister
review-counter subagent held ~200 uncommitted lines in ``src/tac/preflight.py``, so
wiring this check into ``preflight_all()`` in the same commit would have staged the
sibling's in-flight hunks (the absorbed-hunks class, recurrence 3+). Run standalone::

    .venv/bin/python -m tac.run_constant_gates          # report
    .venv/bin/python -m tac.run_constant_gates --strict # rc=1 on violations

WIRE-IN, RESOLVED (ddm_wt1, task #868, 2026-08-01). The 2026-07-07 blocker above (a sibling's
in-flight ``preflight.py`` hunks) is long gone, but ``preflight_all()`` is STILL the wrong host,
for a NEW and measured reason: ``preflight_all()`` runs at 26.1 / 29.6 / 30.2 s over three
consecutive measurements against its own ``DEFAULT_PREFLIGHT_CLI_TIMEOUT_S = 30.0`` budget — one
of the three already raised ``PreflightTimeoutError`` at HEAD before this module was added. These
two gates cost a measured 1.21 s + 5.19 s = 6.40 s, or 21% of the whole budget, so wiring them
there would put every run over. The consumer is therefore ``tools/all_lanes_preflight.py``
(Gate #38) — a lower-frequency pre-dispatch surface, and one of the wire-in targets CLAUDE.md's
"Operator gates must be wired and used" names explicitly.

The gate is a RATCHET, not a pass/fail on absolute count: existing debt does not block (both
checks are WARN-ONLY by design and the drain is a separate campaign), but any INCREASE fails.
A gate that can never fail is the vacuity genus — it prints the clean symbol forever and trains
its readers to ignore it — so "warn-only" is expressed as a pinned baseline rather than as an
ignored result. Baselines below are MEASURED at HEAD, not chosen.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

__all__ = [
    "RUN_CONSTANT_RATCHET_BASELINE",
    "CanonicalConstantCopyViolation",
    "GuardedConstantFrozenLiteralViolation",
    "GuardedConstantScanScope",
    "RunConstantViolation",
    "check_no_canonical_equation_constant_copied_as_literal",
    "check_no_frozen_literal_where_guarded_derivation_declared",
    "check_no_hardcoded_run_constants_in_consumers",
    "run_constant_ratchet",
    "scan_guarded_constant_frozen_literals_with_scope",
    "scan_repo_for_canonical_constant_copies",
    "scan_repo_for_guarded_constant_frozen_literals",
    "scan_repo_for_hardcoded_run_constants",
    "scan_staged_for_guarded_constant_frozen_literals",
]

# MEASURED live counts at HEAD on 2026-08-01 (ddm_wt1, task #868) — the ratchet's pins, and the
# ONLY thing standing between this module and the strict flip. Lower them as the debt drains;
# raising one is a deliberate act that must be argued for in the commit, never a quiet edit.
# Reproduce with: .venv/bin/python -m tac.run_constant_gates
#: MEASURED at HEAD 2026-08-03 by this module's own scanner, in the landing commit.
#: Reproduce with: .venv/bin/python -m tac.run_constant_gates
#: Pre-migration the scanner measured 1 (exactly the canonical instance,
#: ``margin_floor: float = 0.1`` on ``DirectDescriptionJointDescentMLXModule``,
#: zero false positives).  The SAME landing migrated that site to consume
#: ``guarded_constant_registry.MARGIN_FLOOR_INCUMBENT`` (byte-identical, proven
#: at import), so the count at landing is 0 and the baseline is pinned there per
#: the strict-flip atomicity rule: the RATCHET now refuses any NEW frozen
#: literal at a declared site, which is the strict surface for this class.  The
#: ``strict=`` parameter on the check function stays available for callers that
#: want raise-on-any semantics.
_P6_MEASURED_LIVE_COUNT_AT_LANDING = 0

RUN_CONSTANT_RATCHET_BASELINE: dict[str, int] = {
    "hardcoded_run_constants": 10,
    "canonical_constant_copies": 2,
    # P6 pinned by ddm_gk1 2026-08-03 in the same commit as the gate. See the P6
    # block below for the measured live count and the strict-flip condition.
    "guarded_constant_frozen_literals": _P6_MEASURED_LIVE_COUNT_AT_LANDING,
}

_WAIVER_TOKEN = "RUN_CONSTANT_OK:"
_PLACEHOLDER_RATIONALES = {"<rationale>", "<reason>", "tbd", "todo", "placeholder", ""}

# P1: stage CLI flags whose int-literal default duplicates a run property.
_STAGE_FLAGS = ("--tau", "--l7")
_P1_HEAD = re.compile(r"add_argument\(\s*['\"](--(?:tau|l7))['\"]")
_P1_INT_DEFAULT = re.compile(r"default\s*=\s*(\d+)\b")
_P1_TYPE_INT = re.compile(r"type\s*=\s*int\b")

# P2: literal stage-flag override embedded in a string (hint/command/example).
# Whitespace directly after the flag so "--l7-start-epoch 1001" (a trainer flag,
# not a consumer override) never matches.
_P2 = re.compile(r"--(?:tau|l7)\s+\d+\b")

# P3: camera-resolution literals in display/labeling tools only.
_P3 = re.compile(r"(?<!\d)(?:874|1164)(?!\d)")
_P3_DISPLAY_PREFIXES = ("dashboard_", "render_", "witness_checkin")

# P4: stage-key literal assignment outside the DSL.
_P4 = re.compile(r"\b(?:tau_start|l7_start|muon_start)\s*=\s*\d+\b")

_EXCLUDED_PATH_MARKERS = ("_intake_", "/tests/", "/witness_dsl/")
_EXCLUDED_BASENAMES = ("clip_profile.py",)


@dataclass(frozen=True)
class RunConstantViolation:
    """One hardcoded-run-constant finding."""

    path: str
    line_no: int
    pattern: str  # P1 | P2 | P3 | P4
    line: str

    def describe(self) -> str:
        return (
            f"{self.path}:{self.line_no} [{self.pattern}] {self.line.strip()[:160]} "
            f"-- run-property hardcode in a consumer; derive it (stage map: "
            f"tac.witness_dsl.schedule_readback.read_schedule(run_dir); clip constants: "
            f"tac.clip_profile) or waive same-line with # {_WAIVER_TOKEN}<rationale>"
        )


def _has_valid_waiver(line: str) -> bool:
    idx = line.find(_WAIVER_TOKEN)
    if idx < 0:
        return False
    rationale = line[idx + len(_WAIVER_TOKEN):].strip().strip("'\")")
    return rationale.lower() not in _PLACEHOLDER_RATIONALES and len(rationale) >= 4


def _is_excluded(path: Path) -> bool:
    s = str(path).replace("\\", "/")
    if path.name.startswith("test_") or path.name in _EXCLUDED_BASENAMES:
        return True
    return any(marker in s for marker in _EXCLUDED_PATH_MARKERS)


def _is_display_tool(path: Path) -> bool:
    return path.parent.name == "tools" and path.name.startswith(_P3_DISPLAY_PREFIXES)


def _scan_file(path: Path) -> list[RunConstantViolation]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    out: list[RunConstantViolation] = []
    display_tool = _is_display_tool(path)
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if _has_valid_waiver(line):
            continue
        is_comment = stripped.startswith("#")

        # P1: join a small statement window (add_argument calls wrap across lines),
        # TRUNCATED at the next add_argument so the window never bleeds into the
        # following flag's type=/default= (the false-positive class caught at landing:
        # a fixed default=None line inheriting the NEXT flag's default=2500).
        if not is_comment and _P1_HEAD.search(line):
            window_lines = [line]
            for follow in lines[i:i + 3]:
                if "add_argument" in follow:
                    break
                window_lines.append(follow)
            window = " ".join(window_lines)
            if _P1_TYPE_INT.search(window) and _P1_INT_DEFAULT.search(window):
                if not any(_has_valid_waiver(w) for w in window_lines):
                    out.append(RunConstantViolation(str(path), i, "P1", line))
                continue  # a P1 line should not double-report as P2

        if not is_comment and _P2.search(line):
            out.append(RunConstantViolation(str(path), i, "P2", line))
            continue

        if display_tool and not is_comment and _P3.search(line):
            out.append(RunConstantViolation(str(path), i, "P3", line))
            continue

        if not is_comment and _P4.search(line):
            out.append(RunConstantViolation(str(path), i, "P4", line))
    return out


def scan_repo_for_hardcoded_run_constants(repo_root: str | Path | None = None
                                          ) -> list[RunConstantViolation]:
    """Scan the consumer surfaces (tools/ + top-level src/tac modules) for the class."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    tools = root / "tools"
    if tools.is_dir():
        candidates += sorted(tools.glob("*.py"))
    tac = root / "src" / "tac"
    if tac.is_dir():
        candidates += sorted(tac.glob("*.py"))  # top-level consumer modules only
    findings: list[RunConstantViolation] = []
    for p in candidates:
        if _is_excluded(p) or p.resolve() == Path(__file__).resolve():
            continue
        findings.extend(_scan_file(p))
    return findings


def check_no_hardcoded_run_constants_in_consumers(strict: bool = False,
                                                  repo_root: str | Path | None = None
                                                  ) -> list[RunConstantViolation]:
    """WARN-ONLY gate for the hardcoded-run-constant consumer class (see module doc).

    Rule chain (per the CLAUDE.md failure-message discipline): a run-property constant
    in a CONSUMER violates the triality DSL-single-source-of-truth rule; the fix is to
    derive it from the run's own config (schedule read-back / clip_profile / launch.sh
    parse) with an explicit-override-only flag and a visibly labeled fallback.

    ``strict=True`` raises ``RuntimeError`` listing every finding; default warns via
    the returned list (callers print). Keep strict=False until the live count is 0.
    """
    findings = scan_repo_for_hardcoded_run_constants(repo_root)
    if strict and findings:
        detail = "\n".join(f.describe() for f in findings)
        raise RuntimeError(
            f"check_no_hardcoded_run_constants_in_consumers: {len(findings)} hardcoded "
            f"run constant(s) in consumer surfaces:\n{detail}"
        )
    return findings


# ---------------------------------------------------------------------------
# P5 — the DISGUISED form: a measured quantity copied out of its canonical
# producer into a live consumer as bare float literals.
#
# Scope EXTENSION of this same #340 surface (post-#400 Catalog #299 consolidation
# rule: extend an existing gate rather than claim a new number). Same class —
# "a constant that duplicates a property owned elsewhere" — but the source of
# truth is the canonical-equations registry rather than the run's own config.
#
# Why it needs its own detector: a copied literal LOOKS wired. It passes review,
# it carries a plausible comment citing the memo it came from, and it cannot
# track its source. The regex patterns P1-P4 cannot see it (the value is not a
# stage flag or a resolution), and a stub sweep marks it GREEN because a
# mechanism does exist — in the canonical module it was copied from.
#
# SIGNATURE (measured 2026-08-01 over 5,794 live files / 59,992 float literals):
# a single value collision is coincidence (0.999 is every Adam beta2 in the
# repo, not a copy). The discriminating signature is a COPIED TABLE — >=3
# DISTINCT values of one canonical constant appearing in one file that never
# names the owning module. Single-value collisions: 347 (noise). Copied tables: 3.
_CANONICAL_COPY_WAIVER_TOKEN = "CANONICAL_CONSTANT_COPY_OK:"
_COPIED_TABLE_MIN_DISTINCT_VALUES = 3
#: A 4-significant-digit value is a MEASUREMENT; 3 significant digits is usually a
#: rounded convention. MEASURED 2026-08-01 on the live tree: the bar at 4 keeps every
#: real finding identical (lane_guard 10 values, pool harness 4, costate backtest 3)
#: while cutting the candidate index 425 -> 278 and the files needing an AST parse from
#: 61% to 9% of the tree. At 5 both remaining findings are LOST, so 4 is the boundary.
_MIN_SIGNIFICANT_DIGITS = 4
_P5_EXCLUDED_MARKERS = (
    "/tests/",
    "/test_",
    "_intake_",
    "/__pycache__/",
    "/experiments/results/",
    "/canonical_equations/",
    # Vendored / third-party trees: not ours to fix, and a nested venv alone is
    # ~40% of the byte volume the scan would otherwise read.
    "/.venv",
    "/site-packages/",
    "/node_modules/",
)
#: Scanned subtrees = the LIBRARY + TOOLING consumer surface (where a live consumer
#: reads a measured quantity). ``experiments/`` is deliberately out of scope for the
#: same reason it is out of scope for P1-P4 — a trainer is the DSL's COMPILE TARGET,
#: not a consumer — and because it doubles the wall clock. A gate that costs many
#: seconds is a gate that gets turned off, which is how the vacuous scan survived.
#: MEASURED 2026-08-01: a one-off wider sweep INCLUDING ``experiments/`` (5,794 files,
#: 59,992 float literals) found NO copied table outside these two subtrees.
_P5_SCANNED_SUBTREES = ("src/tac", "tools")


@dataclass(frozen=True)
class CanonicalConstantCopyViolation:
    """One canonical-equation constant copied into a consumer as bare literals."""

    path: str
    lines: tuple[int, ...]
    owner_module: str
    owner_constant: str
    values: tuple[float, ...]

    def describe(self) -> str:
        return (
            f"{self.path}:{','.join(str(n) for n in self.lines)} [P5] "
            f"{len(self.values)} distinct values of "
            f"tac.canonical_equations.{self.owner_module}:{self.owner_constant} "
            f"({', '.join(repr(v) for v in self.values[:6])}"
            f"{', ...' if len(self.values) > 6 else ''}) appear as bare literals in a "
            f"file that never names the producer -- a copied table cannot track its "
            f"source and goes stale in silence when the producer is re-measured. Fix: "
            f"import the constant from tac.canonical_equations.{self.owner_module} and "
            f"derive these values from it, or waive same-line with "
            f"# {_CANONICAL_COPY_WAIVER_TOKEN}<rationale>"
        )


def _significant_digits(value: float) -> int:
    return len(Decimal(repr(abs(float(value)))).normalize().as_tuple().digits)


def _module_level_float_constants(tree: ast.Module) -> list[tuple[str, float]]:
    """(name, value) for every float in a module-level UPPER_CASE/_private assignment."""
    out: list[tuple[str, float]] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names, value = [stmt.target.id], stmt.value
        else:
            continue
        if value is None:
            continue
        for name in names:
            if not (name.isupper() or name.startswith("_")):
                continue
            for node in ast.walk(value):
                if isinstance(node, ast.Constant) and isinstance(node.value, float):
                    out.append((name, float(node.value)))
    return out


def _canonical_constant_index(root: Path) -> dict[float, tuple[str, str]]:
    """value -> (module_stem, constant_name) for DISTINCTIVE single-owner values.

    Distinctive = >= 3 significant digits AND claimed by exactly one canonical
    module. A value many modules share is a common tolerance (1e-4, 0.005), and
    matching on it produces only coincidences.
    """
    canon_dir = root / "src" / "tac" / "canonical_equations"
    owners: dict[float, set[tuple[str, str]]] = {}
    if not canon_dir.is_dir():
        return {}
    for path in sorted(canon_dir.glob("*.py")):
        if path.name.startswith("_") or path.name in {"equation.py", "evaluators.py"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for name, value in _module_level_float_constants(tree):
            if _significant_digits(value) >= _MIN_SIGNIFICANT_DIGITS:
                owners.setdefault(value, set()).add((path.stem, name))
    return {
        value: sorted(own)[0]
        for value, own in owners.items()
        if len({module for module, _ in own}) == 1
    }


def _has_valid_canonical_copy_waiver(line: str) -> bool:
    idx = line.find(_CANONICAL_COPY_WAIVER_TOKEN)
    if idx < 0:
        return False
    rationale = line[idx + len(_CANONICAL_COPY_WAIVER_TOKEN):].strip().strip("'\")")
    return rationale.lower() not in _PLACEHOLDER_RATIONALES and len(rationale) >= 4


def _candidate_digit_regex(index: dict[float, tuple[str, str]]) -> re.Pattern[str]:
    """Cheap SUPERSET prefilter so the AST parse only runs on plausible files.

    A value's repr is NOT its source text (``0.0004203`` is written ``4.203e-4``), so a
    repr-substring prefilter would produce FALSE NEGATIVES. Match the significant-digit
    sequence instead, allowing an optional decimal point between digits — a superset of
    every textual spelling of the value, so no candidate is ever skipped.

    Deliberately a single first-match ``search``. A richer variant that tallied
    distinct matches per owning constant (to apply the copied-table threshold at the
    text level) was built and MEASURED SLOWER by more than an order of magnitude:
    ``finditer`` must walk every match of short digit sequences across ~100 MB, which
    costs far more than the AST parses it saves. Keeping the cheap form.
    """
    patterns = set()
    for value in index:
        digits = "".join(str(d) for d in Decimal(repr(abs(value))).normalize().as_tuple().digits)
        # NOTE the ESCAPED dot: an unescaped "." matches ANY character, which is still
        # a superset (so still correct) but over-matches so badly the scan never ends.
        patterns.add(r"\.?".join(digits))
    return re.compile("|".join(sorted(patterns)))


def scan_repo_for_canonical_constant_copies(
    repo_root: str | Path | None = None,
) -> list[CanonicalConstantCopyViolation]:
    """Scan live source for copied canonical-equation constant tables (P5)."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    index = _canonical_constant_index(root)
    if not index:
        return []
    prefilter = _candidate_digit_regex(index)
    groups: dict[tuple[str, str, str], tuple[set[float], set[int]]] = {}
    for sub in _P5_SCANNED_SUBTREES:
        base = root / sub
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            # Match markers against the REPO-RELATIVE path, never the absolute one:
            # an absolute path can contain "/tests/" or "/test_" upstream of the repo
            # (a pytest tmpdir, a checkout under ~/test_runs/), which would silently
            # empty the scope. An empty scope reports the same symbol as a clean full
            # scope -- vacuity is indistinguishable from PASS.
            relative = "/" + path.relative_to(root).as_posix()
            if any(marker in relative for marker in _P5_EXCLUDED_MARKERS):
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if not prefilter.search(source):
                continue
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError):
                continue
            source_lines = source.splitlines()
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Constant) and isinstance(node.value, float)):
                    continue
                owner = index.get(float(node.value))
                if owner is None:
                    continue
                module, constant = owner
                if module in source:  # linked: the file names its producer
                    continue
                line_no = getattr(node, "lineno", 0)
                line = source_lines[line_no - 1] if 0 < line_no <= len(source_lines) else ""
                if _has_valid_canonical_copy_waiver(line):
                    continue
                key = (str(path.relative_to(root)), module, constant)
                values, lines = groups.setdefault(key, (set(), set()))
                values.add(float(node.value))
                lines.add(line_no)
    return [
        CanonicalConstantCopyViolation(
            path=path,
            lines=tuple(sorted(lines)),
            owner_module=module,
            owner_constant=constant,
            values=tuple(sorted(values, reverse=True)),
        )
        for (path, module, constant), (values, lines) in sorted(groups.items())
        if len(values) >= _COPIED_TABLE_MIN_DISTINCT_VALUES
    ]


def check_no_canonical_equation_constant_copied_as_literal(
    strict: bool = False,
    repo_root: str | Path | None = None,
) -> list[CanonicalConstantCopyViolation]:
    """WARN-ONLY gate for the copied-canonical-constant-table class (P5, see above).

    Rule chain (per the CLAUDE.md failure-message discipline): a measured quantity
    copied out of its canonical producer into a consumer violates the value-provenance
    ladder — it presents as a class-3 ``measured_anchor`` while actually being a
    class-4 bare literal with no re-derivation path. The fix is to name/import the
    constant from its canonical module and derive the consumer's view from it; where
    the live path genuinely cannot, waive same-line with a real rationale naming the
    re-derivation trigger.

    ``strict=True`` raises ``RuntimeError``; keep ``strict=False`` until live count 0.
    """
    findings = scan_repo_for_canonical_constant_copies(repo_root)
    if strict and findings:
        detail = "\n".join(f.describe() for f in findings)
        raise RuntimeError(
            f"check_no_canonical_equation_constant_copied_as_literal: {len(findings)} "
            f"copied canonical-constant table(s):\n{detail}"
        )
    return findings


# ---------------------------------------------------------------------------
# P6: a frozen literal at a site where a GuardedConstant DECLARES a live derivation.
#
# The class (ddm_gk1, operator directive 2026-08-03): a derivation exists in-repo,
# is documented, and is NOT CALLED -- its output was frozen into a literal. The
# value is right TODAY, so nothing fails and no test goes red; if the data moved,
# nobody would find out. Such a constant passes every provenance check while
# running none. Canonical instance: ``derive_margin_floor``
# (``src/tac/optimization/lane_guard.py:547``, documented "data-derived per run,
# never a bare constant") vs the frozen ``margin_floor: float = 0.1`` default on
# ``DirectDescriptionJointDescentMLXModule.__init__``.
#
# DECLARATION-DRIVEN BY DESIGN, and this is the load-bearing design decision.
# ddm_gd5 (task #864) BUILT the auto-derived version of this detector -- "is this
# derivation reachable / is a better successor unwired?" -- and REFUTED it: the
# import-reachability predicate fires on 1229 of 3251 modules, and the
# "measured-better successor" relation exists only in memos with no representation
# in code. So P6 does NOT infer its own targets. It scans ONLY for
# ``<name> = <value>`` where BOTH the identifier name and the exact value are
# declared by a GuardedConstant in ``tac.witness_dsl.guarded_constant_registry``
# (``literal_site_names`` + ``incumbent_literal``). A constant nobody has declared
# is invisible to this gate -- which is the honest scope, and the scope is
# REPORTED (see ``describe``) rather than implied.
#
# Same-line waiver: ``# GUARDED_CONSTANT_OK:<real rationale>``.
# ---------------------------------------------------------------------------
_GUARDED_CONSTANT_WAIVER_TOKEN = "GUARDED_CONSTANT_OK:"
#: A file that imports the registry is already routed through the guard.
_GUARDED_REGISTRY_MODULE = "tac.witness_dsl.guarded_constant_registry"
def _has_valid_guarded_constant_waiver(line: str) -> bool:
    """Same shape as the P1-P4 / P5 waiver helpers (placeholder rationales rejected)."""
    idx = line.find(_GUARDED_CONSTANT_WAIVER_TOKEN)
    if idx < 0:
        return False
    rationale = line[idx + len(_GUARDED_CONSTANT_WAIVER_TOKEN):].strip().strip("'\")")
    return rationale.lower() not in _PLACEHOLDER_RATIONALES and len(rationale) >= 4


@dataclass(frozen=True)
class GuardedConstantFrozenLiteralViolation:
    """One frozen literal at a site a GuardedConstant declares a live derivation for."""

    path: str
    line: int
    constant_id: str
    site_name: str
    value: float
    derivation: str
    registry_attr: str

    def describe(self) -> str:
        return (
            f"{self.path}:{self.line} [P6] `{self.site_name} = {self.value!r}` is the frozen "
            f"output of a LIVE derivation: GuardedConstant {self.constant_id!r} declares "
            f"{self.derivation} with invocation_required=True, and this file never imports "
            f"{_GUARDED_REGISTRY_MODULE}. The derivation therefore never runs here, so the "
            f"value cannot adapt and nothing would go red if the data moved -- it presents "
            f"as a provenanced constant while running no provenance at all. Fix: consume "
            f"{_GUARDED_REGISTRY_MODULE}.{self.registry_attr} and resolve it with the "
            f"caller's own sample, or waive same-line with "
            f"# {_GUARDED_CONSTANT_WAIVER_TOKEN}<rationale naming why the frozen value is "
            f"acceptable at THIS call site>"
        )


@dataclass(frozen=True)
class GuardedConstantScanScope:
    """The DENOMINATOR of a P6 scan.  Reported, never implied.

    P6 can return zero findings for two completely different reasons: the repo is
    clean, or the gate never looked at anything.  At the findings layer those are
    the same empty list, and the second one prints the clean symbol forever --
    the ``vacuity == PASS`` genus this whole arm exists to extinct, reproduced
    inside the guard itself.  This record is what tells them apart, so
    :func:`run_constant_ratchet` can refuse an empty scope instead of passing it.
    """

    declared_constants: int
    declared_site_names: int
    files_scanned: int
    files_also_importing_registry: int
    registry_import_ok: bool

    @property
    def is_vacuous(self) -> bool:
        """True when a zero finding count carries NO information."""
        return (
            not self.registry_import_ok
            or self.declared_site_names == 0
            or self.files_scanned == 0
        )

    def describe(self) -> str:
        if not self.registry_import_ok:
            return (
                f"scope VACUOUS: {_GUARDED_REGISTRY_MODULE} failed to import, so the gate "
                "had ZERO declared targets and could not have found anything. A zero count "
                "here means 'did not look', not 'clean'"
            )
        return (
            f"{self.declared_constants} declared constant(s) -> "
            f"{self.declared_site_names} site name(s); {self.files_scanned} file(s) scanned "
            f"({'+'.join(_P5_SCANNED_SUBTREES)}, excluding "
            f"{', '.join(_EXCLUDED_PATH_MARKERS)} and test_*), "
            f"{self.files_also_importing_registry} also import the registry (COUNTED, not exempted)"
        )


def _module_is_imported(tree: ast.AST, module: str) -> bool:
    """True only for a REAL import of ``module`` (or a submodule of it).

    Deliberately NOT a substring test over the file text: a mention in a comment,
    a docstring or a string literal would otherwise exempt the whole file from the
    gate.  ``mentions-it == is-guarded-by-it`` is the same confusion a bare
    pattern probe makes when it counts its own watchers.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == module or mod.startswith(f"{module}."):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module or alias.name.startswith(f"{module}."):
                    return True
    return False


def _guarded_literal_targets() -> tuple[dict[str, list[tuple[str, float, str]]], bool]:
    """``({site_name: [(constant_id, value, derivation_path), ...]}, registry_import_ok)``.

    The bool is the load-bearing half.  An import failure yields an EMPTY target
    set that is indistinguishable, at the findings layer, from a repo with nothing
    to find -- so it is returned explicitly and surfaced by
    :class:`GuardedConstantScanScope`, never swallowed.
    """
    try:
        from tac.witness_dsl import guarded_constant_registry as _reg

        REGISTRY = _reg.REGISTRY
        # The fix message must name a symbol that EXISTS. Deriving it from the
        # module namespace (rather than upper-casing the constant_id) keeps the
        # message actionable when the two differ, as they do for MARGIN_FLOOR
        # vs constant_id 'seg_margin_hinge_floor'.
        _attr_of = {
            c.constant_id: nm
            for nm, c in vars(_reg).items()
            if nm.isupper() and getattr(c, "constant_id", None)
        }
    except Exception:  # pragma: no cover - import environment
        return {}, False
    targets: dict[str, list[tuple[str, float, str]]] = {}
    for cid, const in REGISTRY.items():
        deriv = getattr(const, "derivation", None)
        if deriv is None or not getattr(deriv, "invocation_required", False):
            continue
        incumbent = getattr(const, "incumbent_literal", None)
        if incumbent is None or isinstance(incumbent, str):
            continue
        for name in getattr(const, "literal_site_names", ()):  # declaration-driven
            targets.setdefault(name, []).append(
                (cid, float(incumbent), deriv.callable_path, _attr_of.get(cid, cid.upper()))
            )
    return targets, True


def _p6_literal_sites(tree: ast.AST):
    """Yield ``(name, value_node)`` for assignments, annotated assignments and
    function-parameter defaults -- the three shapes a frozen default takes."""
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            yield node.target.id, node.value
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and node.value is not None:
                    yield tgt.id, node.value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            pos = list(a.posonlyargs) + list(a.args)
            # strict=True is safe by construction: the slice length equals
            # len(a.defaults), and ast guarantees len(kw_defaults) == len(kwonlyargs).
            for arg, default in zip(pos[len(pos) - len(a.defaults):], a.defaults, strict=True):
                yield arg.arg, default
            for arg, default in zip(a.kwonlyargs, a.kw_defaults, strict=True):
                if default is not None:
                    yield arg.arg, default
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg:
                    yield kw.arg, kw.value


def _p6_scan_one_file(
    rel: str,
    text: str,
    tree: ast.AST,
    targets: dict,
    only_lines: set[int] | None = None,
) -> list[GuardedConstantFrozenLiteralViolation]:
    """The single P6 per-file implementation, shared by the repo and staged scans.

    ``only_lines`` restricts findings to those line numbers (the staged diff's
    ADDED lines).  ``None`` means every line -- NOT "no lines"; the caller that
    could not obtain a diff must say so rather than passing an empty set, which
    would filter out every site and pass silently (the vacuity-equals-pass genus).
    """
    out: list[GuardedConstantFrozenLiteralViolation] = []
    lines = text.splitlines()
    for name, node in _p6_literal_sites(tree):
        cands = targets.get(name)
        if not cands:
            continue
        if not (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            continue
        lineno = getattr(node, "lineno", 0)
        if only_lines is not None and lineno not in only_lines:
            continue
        src_line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if _has_valid_guarded_constant_waiver(src_line):
            continue
        for cid, value, deriv, attr in cands:
            if float(node.value) == value:
                out.append(
                    GuardedConstantFrozenLiteralViolation(
                        path=rel, line=lineno, constant_id=cid, site_name=name,
                        value=value, derivation=deriv, registry_attr=attr,
                    )
                )
                break
    return out


def scan_staged_for_guarded_constant_frozen_literals(
    *,
    repo_root: str | Path | None = None,
    files: list[str] | None = None,
) -> tuple[list[GuardedConstantFrozenLiteralViolation], list[str]]:
    """P6 over the STAGED diff's ADDED lines.  Returns ``(violations, unexamined)``.

    THIS is the surface that actually fires.  A gate registered only inside
    ``preflight_all()`` does not run at commit: ``--no-codebase`` is the hook's
    default and examines 0 of 27 gates (ddm_ss1, measured 2026-08-03), which is
    why two STRICT kill-verdict gates (Catalog #307/#308) have never executed
    there.  The repo-wide scan above remains the debt view; this one is the guard.

    ``unexamined`` names every staged file whose added-line range could not be
    obtained.  It is returned rather than swallowed so a caller can never report
    "clean" over files it did not actually read.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    targets, import_ok = _guarded_literal_targets()
    unexamined: list[str] = []
    if not import_ok:
        return [], [f"{_GUARDED_REGISTRY_MODULE} failed to import — 0 constants declared"]
    if not targets:
        return [], []
    try:
        from tac.subset_selection_gate import added_lines  # reuse, do not reimplement
    except Exception as exc:  # pragma: no cover - import environment
        return [], [f"added-line helper unavailable: {exc}"]

    out: list[GuardedConstantFrozenLiteralViolation] = []
    for rel in files or []:
        if not rel.endswith(".py"):
            continue
        if any(m in f"/{rel}" for m in _EXCLUDED_PATH_MARKERS):
            continue
        if Path(rel).name.startswith("test_"):
            continue
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            unexamined.append(f"{rel} unreadable")
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            unexamined.append(f"{rel} unparseable")
            continue
        # No import-based exemption here either -- see the gauge-self-test note in
        # the repo scan. The site's own AST decides: a Constant is frozen, a Name is not.
        scope = added_lines(root, rel)
        if scope is None:
            unexamined.append(f"{rel} added-line range unavailable (git diff failed)")
            continue
        out.extend(_p6_scan_one_file(rel, text, tree, targets, only_lines=scope))
    return out, unexamined


def scan_guarded_constant_frozen_literals_with_scope(
    repo_root: str | Path | None = None,
) -> tuple[list[GuardedConstantFrozenLiteralViolation], GuardedConstantScanScope]:
    """Scan for declared frozen-literal sites, WITH the denominator that makes a
    zero count readable.  See :class:`GuardedConstantScanScope`."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    targets, import_ok = _guarded_literal_targets()
    declared_constants = len({cid for cands in targets.values() for cid, _, _, _ in cands})
    files_scanned = 0
    files_routed = 0
    out: list[GuardedConstantFrozenLiteralViolation] = []
    for sub in _P5_SCANNED_SUBTREES:
        base = root / sub
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = str(path.relative_to(root))
            if any(m in f"/{rel}" for m in _EXCLUDED_PATH_MARKERS):
                continue
            if path.name.startswith("test_"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            files_scanned += 1
            if _module_is_imported(tree, _GUARDED_REGISTRY_MODULE):
                # COUNTED, NOT EXEMPTED. Gauge self-test (sm1, 2026-08-03): if adding
                # an import made the gate read "cured", the gate would be measuring the
                # knob rather than the condition. It is also unnecessary -- the scan
                # only flags ast.Constant, so a genuinely migrated site (whose default
                # is a Name) is already invisible. A file that imports the registry AND
                # still hardcodes the value is exactly the case worth catching.
                files_routed += 1
            if not targets:
                continue
            out.extend(_p6_scan_one_file(rel, text, tree, targets))
    scope = GuardedConstantScanScope(
        declared_constants=declared_constants,
        declared_site_names=len(targets),
        files_scanned=files_scanned,
        files_also_importing_registry=files_routed,
        registry_import_ok=import_ok,
    )
    return out, scope


def scan_repo_for_guarded_constant_frozen_literals(
    repo_root: str | Path | None = None,
) -> list[GuardedConstantFrozenLiteralViolation]:
    """Findings-only view of :func:`scan_guarded_constant_frozen_literals_with_scope`.

    Kept because a zero-length list is the shape the ratchet's ``live`` map wants;
    every caller that TURNS THAT INTO A VERDICT must use the ``_with_scope`` form
    instead, because this return value alone cannot distinguish clean from unlooked.
    """
    findings, _scope = scan_guarded_constant_frozen_literals_with_scope(repo_root)
    return findings


def check_no_frozen_literal_where_guarded_derivation_declared(
    strict: bool = False,
    repo_root: str | Path | None = None,
) -> list[GuardedConstantFrozenLiteralViolation]:
    """WARN-ONLY gate for the frozen-derivation-output class (P6, see above).

    Rule chain: a GuardedConstant that declares a LIVE derivation
    (``invocation_required=True``) asserts that its value must be re-derived from
    the caller's own data.  A bare literal of the same value, under the same
    identifier, in a file that never imports the registry, silently defeats that
    assertion -- the constant presents as provenanced while running no provenance.
    Fix: consume the registry declaration, or waive same-line with a real rationale.

    ``strict=True`` raises ``RuntimeError`` -- on findings, AND on an empty scope.
    The second half matters as much as the first: a strict check that returns
    clean because it had nothing to look at is making the same false statement as
    one that missed a violation.  Kept WARN-ONLY at landing per the strict-flip
    atomicity rule; the strict-flip condition is live count 0 (see
    ``_P6_MEASURED_LIVE_COUNT_AT_LANDING``).
    """
    findings, scope = scan_guarded_constant_frozen_literals_with_scope(repo_root)
    if strict and scope.is_vacuous:
        raise RuntimeError(
            "check_no_frozen_literal_where_guarded_derivation_declared: refusing to report "
            f"a clean scan from an EMPTY scope -- {scope.describe()}. Zero findings from a "
            "gate that looked at nothing is not a pass. Fix: restore "
            f"{_GUARDED_REGISTRY_MODULE} so it imports and declares at least one constant "
            "with invocation_required=True and literal_site_names."
        )
    if strict and findings:
        detail = "\n".join(f.describe() for f in findings)
        raise RuntimeError(
            f"check_no_frozen_literal_where_guarded_derivation_declared: "
            f"{len(findings)} frozen derivation output(s):\n{detail}"
        )
    return findings


def run_constant_ratchet(
    repo_root: str | Path | None = None,
    baseline: dict[str, int] | None = None,
) -> tuple[bool, str]:
    """Run BOTH gates and report pass/fail against the pinned baseline — the consumer surface.

    Returns ``(ok, report)`` — the exact shape ``tools/all_lanes_preflight.py`` steps use, so the
    wire-in adds no adapter. ``ok`` is False iff either live count EXCEEDS its baseline.

    Rule chain on failure (per the CLAUDE.md preflight failure-message discipline): a NEW hardcoded
    run constant in a consumer violates the triality DSL-single-source-of-truth rule, and a NEW
    copied canonical constant violates the value-provenance ladder. The fix in both cases is to
    derive the value from its producer — schedule read-back / ``tac.clip_profile`` / the canonical
    equation module — or, where the live path genuinely cannot, to waive same-line with a real
    rationale naming the re-derivation trigger. Draining below baseline is welcome; re-pin
    ``RUN_CONSTANT_RATCHET_BASELINE`` in the same commit so the ratchet keeps its teeth.
    """
    pins = RUN_CONSTANT_RATCHET_BASELINE if baseline is None else baseline
    p6_findings, p6_scope = scan_guarded_constant_frozen_literals_with_scope(repo_root)
    live = {
        "hardcoded_run_constants": scan_repo_for_hardcoded_run_constants(repo_root),
        "canonical_constant_copies": scan_repo_for_canonical_constant_copies(repo_root),
        "guarded_constant_frozen_literals": p6_findings,
    }
    lines: list[str] = []
    ok = True
    for key, findings in live.items():
        pin = pins.get(key, 0)
        count = len(findings)
        if count > pin:
            ok = False
            lines.append(f"  ✗ {key}: {count} > baseline {pin} — {count - pin} NEW violation(s)")
            lines.extend(f"      {f.describe()}" for f in findings)
        else:
            verdict = "at baseline" if count == pin else f"BELOW baseline {pin} — re-pin it"
            lines.append(f"  - {key}: {count} ({verdict})")
    # P6's DENOMINATOR is reported unconditionally, and an empty scope REFUSES.
    # A gate pinned at 0 that never looked at anything prints exactly the same
    # symbol as a clean repo; without this the pin would certify the silence.
    lines.append(f"  · guarded_constant_frozen_literals scope: {p6_scope.describe()}")
    if p6_scope.is_vacuous:
        ok = False
        lines.append(
            "  ✗ guarded_constant_frozen_literals: VACUOUS, not PASS — the gate ran with an "
            "empty scope, so its 0 findings carry no information. Fix: restore "
            f"{_GUARDED_REGISTRY_MODULE} (it must import and declare at least one constant "
            "with invocation_required=True and literal_site_names), or remove the P6 pin "
            "rather than leaving a gate that cannot fire."
        )
    head = ("run-constant ratchet: PASS" if ok else "run-constant ratchet: FAIL (new debt)")
    return ok, "\n".join([head, *lines])


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strict", action="store_true", help="rc=1 on any violation")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args(argv)
    findings = check_no_hardcoded_run_constants_in_consumers(strict=False,
                                                             repo_root=args.repo_root)
    for f in findings:
        print(f"WARN {f.describe()}")
    print(f"[run_constant_gates] live count = {len(findings)}")
    copies = check_no_canonical_equation_constant_copied_as_literal(
        strict=False, repo_root=args.repo_root)
    for c in copies:
        print(f"WARN {c.describe()}")
    print(f"[run_constant_gates] P5 canonical-constant-copy live count = {len(copies)}")
    frozen, p6_scope = scan_guarded_constant_frozen_literals_with_scope(args.repo_root)
    for f6 in frozen:
        print(f"WARN {f6.describe()}")
    # The count is printed WITH its denominator: "0" alone cannot distinguish a
    # clean repo from a gate that looked at nothing.
    print(
        f"[run_constant_gates] P6 guarded-constant frozen-literal live count = {len(frozen)} "
        f"[scope: {p6_scope.describe()}]"
    )
    if p6_scope.is_vacuous:
        print("VACUOUS [run_constant_gates] P6 scope is empty — 0 findings carry NO information")
    return 1 if (args.strict and (findings or copies or frozen or p6_scope.is_vacuous)) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
