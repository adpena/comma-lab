#!/usr/bin/env python3
"""A3 -- the advisory provenance lane over ddm_ql3's value-fingerprint hits.

The gap ql3 measured
--------------------
ddm_ql3 (2026-09-04) ran two instruments over live constants:

1. a **provenance scan** -- AST plus a +/-6-line source window, flagging a
   measured-looking float whose window NAMES a contiguous-prefix cohort
   (``n96`` / ``gt_n96`` / ``96 frames`` / ``contiguous prefix``). 43 hits.
2. a **value-fingerprint sweep** -- grep for the retired dr1 literals themselves.

Instrument (1) missed the worst debt and that miss was the finding: two harnesses
carried the retired n96 value with **no provenance comment at all**, so their
window named no cohort and no restriction. They were invisible to every
provenance grep by construction. ql3's sentence: *"An unlabelled constant is not
a low-provenance constant; it is an unfindable one."*

What this lane adds
-------------------
The value sweep FINDS an unlabelled constant but cannot say what it IS. Deciding
whether a bare float is an n600 re-measurement, a retired prefix value, a
derived quantity, a deliberately-waived literal, or genuinely unknown is a
reading of the surrounding code -- a classification, not a match. That is what
the advisory lane does here.

It is a COLUMN BESIDE the value fingerprint, never a replacement for it. The
fingerprint's verdict (does this literal equal a known retired value?) is exact
and deterministic and always wins. The advisory label only describes what the
site appears to be, to triage the ones the fingerprint cannot adjudicate.

Classes
-------
``measured_n600``   The window ties the value to a full n600 measurement.
``measured_prefix`` The window ties it to a contiguous-prefix / n96 / n24 / n8
                    cohort -- the retired class.
``derived``         Computed from other constants or a formula, not measured.
``waived``          Explicitly quarantined: named RETIRED, waived, superseded,
                    or a deliberate test fixture.
``unknown``         The window says nothing about where the number came from --
                    ql3's "unfindable" class, and the highest-value output here.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.fmtools_advisory import classify_texts  # noqa: E402

SCAN_DIRS = ("src/tac", "tools", "experiments")

#: The retired dr1 literals ql3 swept for. A prefix match is used (not equality)
#: so a site that stores more or fewer decimals of the same retired number is
#: still caught -- truncation is how a retired value hides from an exact grep.
RETIRED_LITERALS = (
    "0.019590163",
    "0.039180326",
    "0.03712034",
    "0.025631957",
    "0.038173675",
    "0.01356075",
)

#: The n600 replacements dr1 measured, for the same reason.
N600_LITERALS = (
    "0.04376363754272461",
    "0.021881818771362305",
)

WINDOW = 6
CLASSES = ("measured_n600", "measured_prefix", "derived", "waived", "unknown")

#: BINARY CONTROL. The five-way task above asks the model to name a provenance.
#: If it does badly, two explanations survive: the model cannot read provenance
#: from a source window, OR the five-way instruction was the problem. This
#: strictly easier question separates them -- it asks only whether ANY provenance
#: is stated, which is the single bit ql3's finding actually turns on ("an
#: unlabelled constant is not a low-provenance constant; it is an unfindable
#: one"). Running both is what makes the five-way result a SCOPED negative
#: instead of a bare one.
BINARY_CLASSES = ("provenance_is_stated", "no_provenance_stated")
BINARY_INSTRUCTION = (
    "You are shown a numeric constant from a research codebase and the source "
    "lines around it. Answer ONE question: do those surrounding lines STATE "
    "where the number came from?\n\n"
    "Return 'provenance_is_stated' only if the code or comments actually say how "
    "the value was obtained -- naming a measurement, a sample size or cohort, a "
    "formula it was computed from, a source document, or a status such as "
    "RETIRED or SUPERSEDED.\n\n"
    "Return 'no_provenance_stated' if the surroundings show only the assignment "
    "and its use. A descriptive variable name is NOT a statement of provenance. "
    "If you would have to guess, the answer is 'no_provenance_stated'."
)

INSTRUCTION = (
    "You are told a numeric constant from a research codebase and shown the "
    "lines of source around it. Say where the number CAME FROM, judging only "
    "from what the surrounding code and comments actually state.\n\n"
    "'measured_n600': the surroundings tie the number to a full 600-sample "
    "(n600) measurement or a full-population measurement.\n\n"
    "'measured_prefix': the surroundings tie it to a partial cohort -- a "
    "contiguous prefix, n96, n24, n8, gt_n96, '96 frames', or a similar subset "
    "of the full population.\n\n"
    "'derived': the number is computed from other quantities, a formula, a "
    "product, or a ratio, rather than measured directly.\n\n"
    "'waived': the number is explicitly quarantined -- named RETIRED, "
    "SUPERSEDED, deprecated, waived, a historical record kept for comparison, "
    "or an obviously synthetic test fixture.\n\n"
    "'unknown': the surroundings do not say where the number came from. Choose "
    "this whenever there is no stated provenance -- do NOT guess from the "
    "variable name alone.\n\n"
    "Absence of evidence is 'unknown', not a guess."
)

_FLOAT_RE = re.compile(r"\d+\.\d{4,}(?:e-?\d+)?")


def fingerprint_hits() -> list[dict]:
    """Re-run ql3's value-fingerprint sweep over live code.

    Returns one row per (file, line, literal) with the deterministic verdict
    already attached, so the advisory label is only ever an extra column.
    """
    rows: list[dict] = []
    for literal, kind in (
        *((lit, "retired") for lit in RETIRED_LITERALS),
        *((lit, "n600_replacement") for lit in N600_LITERALS),
    ):
        proc = subprocess.run(
            ["grep", "-rn", "--include=*.py", "-F", literal, *SCAN_DIRS],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        for line in proc.stdout.splitlines():
            path, _, rest = line.partition(":")
            lineno, _, text = rest.partition(":")
            if not lineno.isdigit():
                continue
            if "/tests/" in path or Path(path).name.startswith("test_"):
                continue
            if path.startswith("experiments/results/"):
                continue
            if Path(path).name == Path(__file__).name:
                # This scanner's own literal tuple is the RULER, not a consumer.
                # Counting it is the self-hit that inflates every value sweep.
                continue
            rows.append(
                {
                    "path": path,
                    "line": int(lineno),
                    "literal": literal,
                    "fingerprint_verdict": kind,
                    "text": text.strip()[:200],
                }
            )
    return rows


def module_level_float_constants(rel: str) -> list[dict]:
    """Module-level ``NAME = <float>`` assignments in one file, via AST.

    AST rather than regex so a float inside a call argument, a docstring, or a
    nested scope is not mistaken for a module-level constant.
    """
    path = REPO_ROOT / rel
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    out: list[dict] = []
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        value = getattr(node, "value", None)
        if not isinstance(value, ast.Constant) or not isinstance(value.value, float):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and _FLOAT_RE.search(repr(value.value)):
                out.append(
                    {
                        "path": rel,
                        "line": node.lineno,
                        "name": target.id,
                        "literal": repr(value.value),
                        "fingerprint_verdict": "not_a_known_literal",
                        "text": f"{target.id} = {value.value!r}",
                    }
                )
    return out


def window_text(rel: str, lineno: int, radius: int = WINDOW) -> str:
    """The +/-radius source window ql3's provenance scan used."""
    try:
        lines = (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    lo = max(0, lineno - 1 - radius)
    hi = min(len(lines), lineno + radius)
    return "\n".join(lines[lo:hi])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=".omx/research/ddm_fm3_constant_provenance_20260904.json",
        help="where to persist every per-constant row",
    )
    parser.add_argument(
        "--also-scan",
        nargs="*",
        default=[
            "src/tac/canonical_equations/evasion_ceiling_fisher_null_20260715.py",
            "src/tac/canonical_equations/ddm_b2b_rowband_flip_mass_20260731.py",
            "src/tac/canonical_equations/lane_dash_residual_root_cause_findings_20260703.py",
            "src/tac/canonical_equations/boundary_distance_calibration_20260705.py",
            "src/tac/canonical_equations/focal_gradient_concentration_20260705.py",
        ],
        help="extra files from ql3's table to sweep for module-level float constants",
    )
    parser.add_argument("--no-fm", action="store_true", help="skip the advisory lane")
    parser.add_argument(
        "--no-binary-control",
        action="store_true",
        help="skip the single-bit control that scopes a five-way negative",
    )
    args = parser.parse_args(argv)

    rows = fingerprint_hits()
    seen = {(r["path"], r["line"]) for r in rows}
    for rel in args.also_scan:
        for row in module_level_float_constants(rel):
            if (row["path"], row["line"]) not in seen:
                rows.append(row)
                seen.add((row["path"], row["line"]))

    print(f"value-fingerprint hits (retired/n600) : "
          f"{sum(1 for r in rows if r['fingerprint_verdict'] != 'not_a_known_literal')}")
    print(f"additional module-level constants      : "
          f"{sum(1 for r in rows if r['fingerprint_verdict'] == 'not_a_known_literal')}")

    payload = {
        f"{r['path']}:{r['line']}": (
            f"CONSTANT: {r.get('name') or r['literal']} = {r['literal']}\n"
            f"FILE: {r['path']} line {r['line']}\n\n--- surrounding source ---\n"
            + window_text(r["path"], r["line"])
        )
        for r in rows
    }
    fm_labels: dict[str, str] = {}
    binary_labels: dict[str, str] = {}
    fm_ran, fm_reason = False, "skipped by --no-fm"
    if payload and not args.no_fm:
        budget = max(300.0, 4.0 * len(payload))
        print(f"advisory lane: classifying {len(payload)} constant(s), "
              f"budget {budget:.0f}s ...")
        verdict = classify_texts(
            payload,
            labels=list(CLASSES),
            instruction=INSTRUCTION,
            max_chars=2500,
            timeout_s=budget,
        )
        fm_ran, fm_reason = verdict.ran, verdict.reason
        fm_labels = dict(verdict.labels)
        if fm_ran and not args.no_binary_control:
            print(f"binary control: asking the single-bit question on "
                  f"{len(payload)} constant(s) ...")
            binary = classify_texts(
                payload,
                labels=list(BINARY_CLASSES),
                instruction=BINARY_INSTRUCTION,
                max_chars=2500,
                timeout_s=budget,
            )
            if binary.ran:
                binary_labels.update(binary.labels)

    out_rows = []
    for r in rows:
        key = f"{r['path']}:{r['line']}"
        out_rows.append(
            {
                **r,
                "fmtools_advisory": fm_labels.get(key, "no_advice"),
                "fmtools_binary_control": binary_labels.get(key, "no_advice"),
            }
        )

    by_class = Counter(r["fmtools_advisory"] for r in out_rows)
    # The row that matters most: a RETIRED literal whose site the advisory lane
    # reads as having no stated provenance -- ql3's "unfindable" class, live.
    unfindable_retired = [
        r
        for r in out_rows
        if r["fingerprint_verdict"] == "retired" and r["fmtools_advisory"] == "unknown"
    ]

    report = {
        "study": "ddm_fm3 A3 -- constant-provenance advisory lane over ql3's fingerprint hits",
        "retired_literals": list(RETIRED_LITERALS),
        "n600_literals": list(N600_LITERALS),
        "scan_dirs": list(SCAN_DIRS),
        "window_lines": WINDOW,
        "fm_ran": fm_ran,
        "fm_reason": fm_reason,
        "n_rows": len(out_rows),
        "by_advisory_class": dict(by_class),
        "by_binary_control": dict(
            Counter(r["fmtools_binary_control"] for r in out_rows)
        ),
        "unfindable_retired": [f"{r['path']}:{r['line']}" for r in unfindable_retired],
        "rows": out_rows,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nadvisory class        rows")
    print("-" * 26)
    for cls in (*CLASSES, "no_advice"):
        if by_class.get(cls):
            print(f"{cls:<20} {by_class[cls]:>5}")
    if binary_labels:
        bc = Counter(r["fmtools_binary_control"] for r in out_rows)
        print("\nbinary control          rows")
        print("-" * 28)
        for cls in (*BINARY_CLASSES, "no_advice"):
            if bc.get(cls):
                print(f"{cls:<22} {bc[cls]:>5}")
    if unfindable_retired:
        print(f"\nRETIRED literal with no stated provenance ({len(unfindable_retired)}):")
        for r in unfindable_retired:
            print(f"  {r['path']}:{r['line']}  {r['literal']}")
    if not fm_ran and payload:
        print(f"\nADVISORY LANE DID NOT RUN ({fm_reason}) -- the value-fingerprint "
              "verdicts stand alone; fmtools confirmation OWED.")
    print(f"\nper-constant rows persisted -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
