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

Wire into ``preflight_all(strict=False)`` once the sibling's preflight.py lands.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "RunConstantViolation",
    "check_no_hardcoded_run_constants_in_consumers",
    "scan_repo_for_hardcoded_run_constants",
]

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
    return 1 if (args.strict and findings) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
