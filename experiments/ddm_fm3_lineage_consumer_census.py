#!/usr/bin/env python3
"""A2 -- the GT ``.npz`` lineage-consumer census, and the class-aware widening plan.

The gap
-------
``_GT_LINEAGE_ARTIFACT_PATTERNS`` (src/tac/preflight.py) matches only
``gt_first6*.npy`` and ``gt_cache_*.pt``. The ``.npz`` GT tables are invisible to
it -- including ``gt_n600.npz``, which is the **PyAV** decode lineage (20,671
sites off DALI; same-object pose ceiling 1.69e-5) and which the born trainer
pins as authority. That is exactly the undeclared-lineage class the gate exists
to stop, sitting in the gate's blind spot (ddm_bh1 finding 2).

Why not just add ``.npz`` and flip strict
-----------------------------------------
Because the widened vocabulary lights up a large number of existing sites, and
they are NOT all the same kind of thing. A trainer that fits against a GT table
is choosing a solve objective; a plotting script that reads the same table to
draw a chart is not. Refusing both would either wedge the repo or force a
blanket waiver that destroys the signal. So this census CLASSIFIES each consumer
first, then the widening is scoped to the class that actually carries the risk.

Classes
-------
``authority_consumer``   The file fits, trains, scores, evaluates, or byte-closes
                         against the table -- the table IS its solve objective.
                         Getting the lineage wrong changes the answer. REFUSE.
``continuity_frame``     Reads the table to stay comparable with an earlier
                         measurement (baselines, regression anchors, re-scoring
                         old rows). Needs a lineage LABEL, not a refusal.
``advisory_instrument``  Probes, censuses, plots, diagnostics, dashboards. The
                         output is advisory and never a score. LABEL.
``test_fixture``         Tests and synthetic fixtures. Out of scope.
``historical_memo``      Docs, memos, archived one-shots. Out of scope.

Method
------
The census reuses the gate's OWN predicates (``_gt_artifact_hits_outside_comment``,
``_python_without_docstrings``, ``_python_imports_gt_lineage_registry``,
``_gt_lineage_waiver_rationale``, ``_is_oss_export_mirror_path``) with only the
artifact vocabulary widened, so the counts it reports are the counts the widened
gate would actually produce -- not a re-implementation that could disagree.

Classification is the fmtools ADVISORY lane. It never decides anything on its
own: the deterministic ``--force-authority`` list wins wherever the two differ,
and every disagreement is recorded in the output for a human to read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.fmtools_advisory import classify_texts  # noqa: E402

SCAN_DIRS = ("src", "tools", "experiments", "scripts")

#: The widened vocabulary is IMPORTED from the shipped gate, never restated here.
#: An earlier draft of this file carried its own copy of the pattern; when the
#: gate's pattern was widened to reach ``gt_strided_n200`` the copy silently kept
#: the old form and this census measured a DIFFERENT scope than the code it was
#: supposed to describe. Two statements of one law drift -- so there is one.

CLASSES = (
    "authority_consumer",
    "continuity_frame",
    "advisory_instrument",
    "test_fixture",
    "historical_memo",
)

INSTRUCTION = (
    "You are classifying one Python file from a video-compression research repo "
    "by HOW it uses a ground-truth (GT) data table it loads. The table exists in "
    "two decode lineages that disagree numerically, so using the wrong one "
    "silently changes results for some consumers and not others.\n\n"
    "'authority_consumer': the file FITS, TRAINS, OPTIMISES, SCORES, EVALUATES, "
    "or BYTE-CLOSES against the table. The table is its solve objective or its "
    "correctness oracle. A wrong lineage changes the answer it produces.\n\n"
    "'continuity_frame': the file reads the table to stay comparable with an "
    "earlier measurement -- baselines, regression anchors, re-scoring previously "
    "recorded rows, reproducing a past number.\n\n"
    "'advisory_instrument': the file only INSPECTS or REPORTS -- a probe, "
    "census, audit, plot, dashboard, diagnostic, or summary. Its output is "
    "advisory and is never itself a score.\n\n"
    "'test_fixture': a test, or a script whose GT data is synthetic or "
    "deliberately malformed.\n\n"
    "'historical_memo': a one-shot archived script or a documentation helper "
    "that is no longer part of any live path.\n\n"
    "Judge by what the file DOES with the table, not by its directory."
)

#: Deterministic override. A file matching one of these is authority regardless
#: of what the advisory lane says -- the model is a second opinion, never the
#: decider, and the highest-consequence class must not depend on it.
FORCE_AUTHORITY_RE = re.compile(
    r"(?:^|/)(?:train_|.*_train|.*trainer|byte_close|.*byte_close|contest_auth_eval|"
    r"auth_eval|.*_eval_|evaluate)", re.IGNORECASE
)


def widened_patterns() -> tuple[re.Pattern[str], ...]:
    """The gate's shipped primary + ``.npz`` vocabularies, exactly as shipped."""
    from tac.preflight import (
        _GT_LINEAGE_ARTIFACT_PATTERNS,
        _GT_LINEAGE_NPZ_PATTERNS,
    )

    return (*_GT_LINEAGE_ARTIFACT_PATTERNS, *_GT_LINEAGE_NPZ_PATTERNS)


def census(patterns: tuple[re.Pattern[str], ...]) -> dict[str, list[dict]]:
    """Walk the scan dirs and return ``{relpath: [hit, ...]}``.

    Applies the live gate's exclusions verbatim so the count is the count the
    widened gate would produce.
    """
    from tac.preflight import (
        _gt_artifact_hits_outside_comment,
        _gt_lineage_waiver_rationale,
        _is_oss_export_mirror_path,
        _iter_python_files,
        _python_imports_gt_lineage_registry,
        _python_without_docstrings,
    )
    import tac.preflight as pf

    original = pf._GT_LINEAGE_ARTIFACT_PATTERNS
    pf._GT_LINEAGE_ARTIFACT_PATTERNS = patterns  # widen the shared predicate
    try:
        found: dict[str, list[dict]] = {}
        for path in _iter_python_files(REPO_ROOT, list(SCAN_DIRS)):
            if "__pycache__" in path.parts:
                continue
            try:
                rel = path.resolve().relative_to(REPO_ROOT).as_posix()
            except (ValueError, OSError):
                continue
            if rel.startswith("experiments/results/"):
                continue
            if rel == "src/tac/gt_lineage.py" or rel.startswith("src/tac/preflight"):
                continue
            if rel == "experiments/ddm_gl1_gt_lineage_census.py":
                continue
            if _is_oss_export_mirror_path(path):
                continue
            try:
                text = path.read_text()
            except (UnicodeDecodeError, FileNotFoundError, OSError):
                continue
            if not any(pat.search(text) for pat in patterns):
                continue
            is_test = "/tests/" in rel or path.name.startswith("test_")
            registry_routed = _python_imports_gt_lineage_registry(
                _python_without_docstrings(text)
            )
            hits: list[dict] = []
            for lineno, line in enumerate(
                _python_without_docstrings(text).splitlines(), 1
            ):
                artifacts = _gt_artifact_hits_outside_comment(line)
                if not artifacts:
                    continue
                hits.append(
                    {
                        "line": lineno,
                        "artifacts": sorted(set(artifacts)),
                        "declares_dali": any("dali" in a.lower() for a in artifacts),
                        "waived": _gt_lineage_waiver_rationale(line) is not None,
                        "text": line.strip()[:200],
                    }
                )
            if hits:
                found[rel] = [
                    {
                        **h,
                        "_file_is_test": is_test,
                        "_file_registry_routed": registry_routed,
                    }
                    for h in hits
                ]
        return found
    finally:
        pf._GT_LINEAGE_ARTIFACT_PATTERNS = original


def undeclared(hits: list[dict]) -> list[dict]:
    """The hits the gate would REPORT: not DALI-declared, not routed, not waived."""
    return [
        h
        for h in hits
        if not h["declares_dali"]
        and not h["waived"]
        and not h["_file_registry_routed"]
        and not h["_file_is_test"]
    ]


def file_excerpt(rel: str, hits: list[dict], max_chars: int = 3000) -> str:
    """Head of the file plus its hit lines -- the classification input."""
    try:
        head = (REPO_ROOT / rel).read_text()[:1800]
    except OSError:
        head = ""
    lines = "\n".join(f"  L{h['line']}: {h['text']}" for h in hits[:12])
    return f"FILE: {rel}\n\n--- head ---\n{head}\n\n--- GT lines ---\n{lines}"[:max_chars]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=".omx/research/ddm_fm3_lineage_consumer_census_20260904.json",
        help="where to persist the full per-file, per-line census",
    )
    parser.add_argument("--no-fm", action="store_true", help="skip the advisory lane")
    args = parser.parse_args(argv)

    from tac.preflight import _GT_LINEAGE_ARTIFACT_PATTERNS, _GT_LINEAGE_NPZ_PATTERNS

    baseline = census(_GT_LINEAGE_ARTIFACT_PATTERNS)
    widened = census(widened_patterns())

    baseline_live = sum(len(undeclared(h)) for h in baseline.values())
    widened_live = sum(len(undeclared(h)) for h in widened.values())
    new_files = {k: v for k, v in widened.items() if undeclared(v) and not undeclared(baseline.get(k, []))}
    print(f"baseline live findings : {baseline_live} across {sum(1 for h in baseline.values() if undeclared(h))} file(s)")
    print(f"widened  live findings : {widened_live} across {sum(1 for h in widened.values() if undeclared(h))} file(s)")
    print(f"newly lit files        : {len(new_files)}")

    # --- classify every file that the WIDENED gate would newly report ---
    to_classify = {rel: file_excerpt(rel, undeclared(h)) for rel, h in new_files.items()}
    fm_labels: dict[str, str] = {}
    fm_ran = False
    fm_reason = "skipped by --no-fm"
    if to_classify and not args.no_fm:
        # Budget the batch by item count: the on-device model runs ~1 s/item, so
        # a fixed ceiling would fail open on a large census and silently cost us
        # the whole advisory lane.
        budget_s = max(300.0, 4.0 * len(to_classify))
        print(f"advisory lane: classifying {len(to_classify)} file(s), "
              f"budget {budget_s:.0f}s ...")
        verdict = classify_texts(
            to_classify,
            labels=list(CLASSES),
            instruction=INSTRUCTION,
            max_chars=3000,
            timeout_s=budget_s,
        )
        fm_ran, fm_reason = verdict.ran, verdict.reason
        fm_labels = dict(verdict.labels)
        print(
            f"advisory lane: {'ran' if fm_ran else 'DID NOT RUN'} "
            f"({len(fm_labels)}/{len(to_classify)} labelled)"
            + ("" if fm_ran else f" -- {fm_reason}")
        )

    rows = []
    for rel, hits in sorted(new_files.items()):
        live = undeclared(hits)
        forced = bool(FORCE_AUTHORITY_RE.search(rel))
        advisory = fm_labels.get(rel, "no_advice")
        final = "authority_consumer" if forced else (
            advisory if advisory in CLASSES else "unclassified"
        )
        rows.append(
            {
                "path": rel,
                "live_findings": len(live),
                "first_line": live[0]["line"] if live else None,
                "artifacts": sorted({a for h in live for a in h["artifacts"]}),
                "deterministic_force_authority": forced,
                "fmtools_advisory": advisory,
                "final_class": final,
                "lanes_disagree": forced and advisory not in ("authority_consumer", "no_advice"),
                "lines": [{"line": h["line"], "text": h["text"]} for h in live[:12]],
            }
        )

    by_class = Counter(r["final_class"] for r in rows)
    findings_by_class: Counter[str] = Counter()
    for r in rows:
        findings_by_class[r["final_class"]] += r["live_findings"]

    report = {
        "study": "ddm_fm3 A2 -- GT .npz lineage-consumer census and widening plan",
        "widened_pattern": [p.pattern for p in _GT_LINEAGE_NPZ_PATTERNS],
        "scan_dirs": list(SCAN_DIRS),
        "baseline_live_findings": baseline_live,
        "widened_live_findings": widened_live,
        "newly_lit_files": len(new_files),
        "fm_ran": fm_ran,
        "fm_reason": fm_reason,
        "files_by_class": dict(by_class),
        "findings_by_class": dict(findings_by_class),
        "disagreements": [r["path"] for r in rows if r["lanes_disagree"]],
        "rows": rows,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nclass                    files  findings")
    print("-" * 41)
    for cls in (*CLASSES, "unclassified"):
        if by_class.get(cls):
            print(f"{cls:<24} {by_class[cls]:>5}  {findings_by_class[cls]:>8}")
    if report["disagreements"]:
        print(f"\nlane disagreements (deterministic wins): {report['disagreements']}")
    if not fm_ran and to_classify:
        print(f"\nADVISORY LANE DID NOT RUN ({fm_reason}) -- classes rest on the "
              "deterministic force-list alone; fmtools confirmation OWED.")
    print(f"\nfull census persisted -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
