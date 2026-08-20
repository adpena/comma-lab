#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Load prior vehicle harvests and drain the entropy partition into the ledger.

This arm is scorer-free.  It reads already-landed research, appends typed rows
to the canonical probe-outcomes ledger, and retains every generated payload on
APDataStore plus a small review copy under the ddm_vh2 charter path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.probe_outcomes_ledger import (  # noqa: E402
    EVIDENCE_CLASS_BUILT,
    EVIDENCE_CLASS_DERIVED,
    EVIDENCE_CLASS_DESIGNED_ONLY,
    EVIDENCE_CLASS_MEASURED,
    EVIDENCE_CLASS_MEASURED_DERIVED,
    EVIDENCE_CLASS_MIXED,
    EVIDENCE_CLASS_SOURCE_INSPECTED,
    PROBE_OUTCOMES_LEDGER_PATH,
    ROUTING_STATUS_DEAD_ON_THIS_BASE,
    ROUTING_STATUS_DEFERRED_WITH_BLOCKER,
    ROUTING_STATUS_NEEDS_REMEASURE,
    ROUTING_STATUS_ROUTED_FIRED,
    ROUTING_STATUS_ROUTED_QUEUED,
    coverage,
    partition_vehicle_corpus,
    register_vehicle_routing_findings,
)

AP_ROOT = Path("/Volumes/APDataStore/pact/ddm_vh2_20260810")
REVIEW_ROOT = REPO / ".omx/research/charters/ddm_vh2_20260810"
WL_DOC = REPO / ".omx/research/ddm_wl1_20260805/TRANSFER_TABLE.md"
FH_DOC = REPO / ".omx/research/ddm_fh1_forces_harvest_20260731.md"
VP_DOC = REPO / ".omx/research/ddm_vp1_20260810/VP1_RESCORING_REPORT.md"
WL_ARTIFACT = ".omx/research/ddm_wl1_20260805"
FH_ARTIFACT = ".omx/research/ddm_fh1_forces_harvest_20260731.md"
VP_ARTIFACT = ".omx/research/ddm_vp1_20260810"
SOURCE_HARVEST = "ddm_vh2_vehicle_harvest_routing_ledger_20260810"

WL_RANKS = (1, 2, 3, 5, 8, 10, 11, 12, 14, 16, 29, 30, 31, 32, 33)
WL_QUEUED = frozenset({1, 2, 3, 16, 31, 32, 33})


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    """Hash a directory deterministically by relative path, kind, and bytes."""

    digest = hashlib.sha256()
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        if item.is_symlink():
            kind = b"L"
            payload = os.readlink(item).encode("utf-8")
        elif item.is_file():
            kind = b"F"
            payload = item.read_bytes()
        elif item.is_dir():
            kind = b"D"
            payload = b""
        else:
            kind = b"O"
            payload = b""
        digest.update(kind + b"\0" + relative + b"\0")
        digest.update(str(len(payload)).encode("ascii") + b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_tree(path) if path.is_dir() else _sha256_file(path)


def _relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _parse_markdown_table(path: Path, header_name: str) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("|") and header_name in line)
    headers = [cell.strip() for cell in lines[header_index].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def _evidence_class(text: str) -> str:
    upper = text.upper()
    if "MEASURED" in upper and "DERIVED" in upper:
        return EVIDENCE_CLASS_MEASURED_DERIVED
    if "MEASURED" in upper:
        return EVIDENCE_CLASS_MEASURED
    if "BUILT" in upper:
        return EVIDENCE_CLASS_BUILT
    if "DESIGNED" in upper:
        return EVIDENCE_CLASS_DESIGNED_ONLY
    if "DERIVED" in upper:
        return EVIDENCE_CLASS_DERIVED
    return EVIDENCE_CLASS_MIXED


def _wl_rows() -> list[dict[str, Any]]:
    source_doc = _relative(WL_DOC)
    source_sha = _sha256_file(WL_DOC)
    parsed = {int(row["rank"]): row for row in _parse_markdown_table(WL_DOC, "named TR1/endgame consumer")}
    findings: list[dict[str, Any]] = []
    for rank in WL_RANKS:
        row = parsed[rank]
        status = ROUTING_STATUS_ROUTED_QUEUED if rank in WL_QUEUED else ROUTING_STATUS_ROUTED_FIRED
        findings.append(
            {
                "route_id": f"ddm_vh2_wl1_rank_{rank:02d}_v1",
                "lineage": "wl",
                "source_doc": source_doc,
                "corpus_artifact": WL_ARTIFACT,
                "sha": source_sha,
                "finding": f"{row['element']}: {row['witness verdict with receipt']}",
                "evidence_class": _evidence_class(row["evidence grade"]),
                "vehicle_scope": row["version"],
                "status": status,
                "owner": "MAIN",
                "consumer": row["named TR1/endgame consumer"],
                "fire_order": row["cost / disposition"],
                "source_harvest": "ddm_wl1_20260805",
            }
        )
    return findings


def _vp_rows() -> list[dict[str, Any]]:
    source_doc = _relative(VP_DOC)
    source_sha = _sha256_file(VP_DOC)
    parsed = {int(row["wl1 rank"]): row for row in _parse_markdown_table(VP_DOC, "PR130 disposition")}
    findings: list[dict[str, Any]] = []
    for rank in WL_RANKS:
        row = parsed[rank]
        disposition = row["PR130 disposition"].strip("*")
        if disposition == "STILL-PORTS":
            status = ROUTING_STATUS_ROUTED_FIRED
        elif disposition == "NEEDS-REMEASURE":
            status = ROUTING_STATUS_NEEDS_REMEASURE
        elif disposition == "DEAD-ON-THIS-BASE":
            status = ROUTING_STATUS_DEAD_ON_THIS_BASE
        else:
            raise ValueError(f"unrecognized vp1 disposition: {disposition!r}")
        finding: dict[str, Any] = {
            "route_id": f"ddm_vh2_vp1_rank_{rank:02d}_v1",
            "lineage": "vp",
            "source_doc": source_doc,
            "corpus_artifact": VP_ARTIFACT,
            "sha": source_sha,
            "finding": f"{row['source row']}: {row['PR130 slice and justification']}",
            "evidence_class": EVIDENCE_CLASS_MIXED,
            "vehicle_scope": "PR130 CPR1 only; scorer-free rescore of wl1 evidence",
            "status": status,
            "owner": "MAIN",
            "consumer": (
                "ddm_ai1 and PR130 archive/runtime acceptance" if rank in {5, 12, 14} else "PR130 vehicle route queue"
            ),
            "source_harvest": "ddm_vp1_20260810",
            "supersedes_route_id": f"ddm_vh2_wl1_rank_{rank:02d}_v1",
        }
        if status == ROUTING_STATUS_DEAD_ON_THIS_BASE:
            finding.update(
                blocker=(
                    "#996 coder axis closure, scope-under-review by ddm_rc2; "
                    "the unchanged PR130 base has no contour-residual object"
                ),
                fire_condition=(
                    "ddm_rc2 must reopen #996 for a new receiver-closed representation "
                    "that creates a distinct probability object"
                ),
            )
        else:
            finding["fire_order"] = row["Required measurement or closure"]
        findings.append(finding)
    return findings


FH_ROWS: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "r1_r4",
        "Tie-locus displacement and per-class-pair edge weighting target the placement pool.",
        EVIDENCE_CLASS_MEASURED_DERIVED,
        ROUTING_STATUS_ROUTED_QUEUED,
        "MAIN burn-4 force-stack owner",
        "Fire after the current-base pair field exists; race uniform, flip-mass, and Young sigma after the knee.",
    ),
    (
        "r2",
        "Xi-advected token base may remove ego-motion structure from deltas but is gated by the zero-cost QA90 coherence read.",
        EVIDENCE_CLASS_DERIVED,
        ROUTING_STATUS_DEFERRED_WITH_BLOCKER,
        "MAIN burn-4 rate-axis owner",
        "QA90 temporal-coherence read is absent; fire only if it finds coherent advected delta structure.",
    ),
    (
        "r3",
        "Margin satisficing cap is a small current-vehicle allocator race, not a finisher stage.",
        EVIDENCE_CLASS_DERIVED,
        ROUTING_STATUS_ROUTED_QUEUED,
        "MAIN burn-4 force-stack owner",
        "Re-measure delta_R on current frames, then race the cap against the incumbent inverse allocator.",
    ),
    (
        "r5",
        "Renderer-weight rate force is real but the measured renderer pool was only 3,284 bytes on the scoped TR1 endpoint.",
        EVIDENCE_CLASS_MEASURED,
        ROUTING_STATUS_DEFERRED_WITH_BLOCKER,
        "MAIN rate-axis owner",
        "Measured pool is too small for a burn arm; reopen only if current renderer bytes materially exceed that scoped pool.",
    ),
    (
        "r6",
        "Per-class Lane birth weight has a derived geometry race on an existing never-fired flag.",
        EVIDENCE_CLASS_MEASURED_DERIVED,
        ROUTING_STATUS_ROUTED_QUEUED,
        "MAIN burn-4 seg-axis owner",
        "Race class-weight-lane inside the birth pool before stacking it with KD or seeding.",
    ),
    (
        "r7",
        "From-birth KD must include the annulus attack set and anneal to CE at a derived event.",
        EVIDENCE_CLASS_DERIVED,
        ROUTING_STATUS_ROUTED_FIRED,
        "MAIN burn-4 charter owner",
        "Already consumed as the mechanism-complete specification for the existing matrix entrant.",
    ),
    (
        "r8",
        "Birth plateau must conjunct the loss knee before sharpening and quantization events engage.",
        EVIDENCE_CLASS_MEASURED_DERIVED,
        ROUTING_STATUS_ROUTED_QUEUED,
        "MAIN burn-4 event-graph owner",
        "Fire after the typed birth_completion key exists and the first matched event window is ready.",
    ),
    (
        "r9",
        "The NCDE stage-exit law transfers as a read-only observer over current verdict rows.",
        EVIDENCE_CLASS_DERIVED,
        ROUTING_STATUS_ROUTED_QUEUED,
        "costate digest owner",
        "Port only after the current verdict stream exposes the required level-normalized cadence fields.",
    ),
    (
        "r10",
        "Every adopted force needs per-term loss, gradient norm, stage-boundary caps, and same-pool conflict cosines.",
        EVIDENCE_CLASS_DERIVED,
        ROUTING_STATUS_ROUTED_QUEUED,
        "MAIN launch-review owner",
        "Apply at the next force-bearing charter seal after telemetry exposes the named signals.",
    ),
    (
        "r13",
        "Area theft versus placement is the telemetry split that routes each class residual to the right force pool.",
        EVIDENCE_CLASS_MEASURED_DERIVED,
        ROUTING_STATUS_ROUTED_QUEUED,
        "MAIN telemetry owner",
        "Add at the next current-vehicle verdict-cadence telemetry landing before choosing a seg force.",
    ),
    (
        "r14",
        "ERF birth-context co-adaptation is a speculative rung behind the first birth-telemetry window.",
        EVIDENCE_CLASS_DESIGNED_ONLY,
        ROUTING_STATUS_DEFERRED_WITH_BLOCKER,
        "MAIN post-window research owner",
        "No current birth-context response row; fire only if first-window telemetry shows slow context settling after births.",
    ),
)


def _fh_rows() -> list[dict[str, Any]]:
    source_doc = _relative(FH_DOC)
    source_sha = _sha256_file(FH_DOC)
    findings: list[dict[str, Any]] = []
    for row_id, finding_text, evidence, status, consumer, action in FH_ROWS:
        finding: dict[str, Any] = {
            "route_id": f"ddm_vh2_fh1_{row_id}_v1",
            "lineage": "fh",
            "source_doc": source_doc,
            "corpus_artifact": FH_ARTIFACT,
            "sha": source_sha,
            "finding": finding_text,
            "evidence_class": evidence,
            "vehicle_scope": "TR1 burn-4 adaptation; not a PR130 score or transferred number",
            "status": status,
            "owner": "MAIN",
            "consumer": consumer,
            "source_harvest": "ddm_fh1_20260731",
        }
        if status == ROUTING_STATUS_DEFERRED_WITH_BLOCKER:
            blocker, fire_condition = action.split("; ", maxsplit=1)
            finding["blocker"] = blocker
            finding["fire_condition"] = fire_condition
        else:
            finding["fire_order"] = action
        findings.append(finding)
    return findings


ENTROPY_ARTIFACTS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "config",
        ".omx/research/ddm_entropy_priced_member_n64_603_613_20260722T044916Z.config.json",
        "The exact n64 direct-description six-stream configuration is retained provenance, but its object is absent on PR130.",
        EVIDENCE_CLASS_SOURCE_INSPECTED,
        ROUTING_STATUS_DEAD_ON_THIS_BASE,
    ),
    (
        "payloads",
        ".omx/research/ddm_entropy_priced_member_n64_603_613_20260722T044916Z_artifacts",
        "Exact coder payloads and receiver fixtures are retained and can test a genuinely new probability object without re-encoding the old one.",
        EVIDENCE_CLASS_BUILT,
        ROUTING_STATUS_ROUTED_QUEUED,
    ),
    (
        "outcome",
        ".omx/research/ddm_entropy_priced_member_n64_603_613_20260722T045940Z.md",
        "Exact entropy reduced the old fixed-width payload from 274,664 to 45,369 bytes, but all eight safe-zero subsets were infeasible at all five tolerances on n64.",
        EVIDENCE_CLASS_MEASURED_DERIVED,
        ROUTING_STATUS_DEFERRED_WITH_BLOCKER,
    ),
    (
        "dag_feed",
        ".omx/research/ddm_entropy_priced_member_n64_603_613_DAG_FEED_20260722T045940Z.md",
        "The entropy bracket was already consumed into Task 613 and the sub-0.15 DAG.",
        EVIDENCE_CLASS_DERIVED,
        ROUTING_STATUS_ROUTED_FIRED,
    ),
    (
        "blockers",
        ".omx/research/ddm_entropy_priced_member_n64_603_613_blocker_register_20260722T045940Z.json",
        "The retained blockers are absolute per-stratum feasibility, pre-uint8 access, and n600 validation.",
        EVIDENCE_CLASS_SOURCE_INSPECTED,
        ROUTING_STATUS_DEFERRED_WITH_BLOCKER,
    ),
    (
        "equations",
        ".omx/research/ddm_entropy_priced_member_n64_603_613_canonical_equations_20260722T045940Z.md",
        "The measured rate-distortion bracket was consumed by the direct-description entropy law and structured-carrier registration.",
        EVIDENCE_CLASS_DERIVED,
        ROUTING_STATUS_ROUTED_FIRED,
    ),
    (
        "register",
        ".omx/research/ddm_entropy_priced_member_n64_603_613_register_20260722T045940Z.jsonl",
        "The append-only registration already routed the old-object result to Task 613.",
        EVIDENCE_CLASS_SOURCE_INSPECTED,
        ROUTING_STATUS_ROUTED_FIRED,
    ),
)


def _entropy_rows() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for row_id, artifact, text, evidence, status in ENTROPY_ARTIFACTS:
        source_path = REPO / artifact
        finding: dict[str, Any] = {
            "route_id": f"ddm_vh2_entropy_{row_id}_v1",
            "lineage": "entropy",
            "source_doc": artifact,
            "corpus_artifact": artifact,
            "sha": _sha256_path(source_path),
            "finding": text,
            "evidence_class": evidence,
            "vehicle_scope": "old n64 direct-description six-stream object; not PR130 evidence",
            "status": status,
            "owner": "ddm_rc2" if row_id in {"config", "payloads", "outcome", "blockers"} else "MAIN",
            "consumer": (
                ".omx/state/probe_outcomes.jsonl and ddm_rc2 coder-scope review"
                if row_id in {"config", "payloads", "outcome", "blockers"}
                else "sub-0.15 DAG and canonical research registry"
            ),
            "source_harvest": SOURCE_HARVEST,
        }
        if status in {ROUTING_STATUS_DEAD_ON_THIS_BASE, ROUTING_STATUS_DEFERRED_WITH_BLOCKER}:
            finding["blocker"] = (
                "#996 coder axis closure is scope-under-review by ddm_rc2, and this evidence "
                "is n64 on an old direct-description object with no current PR130 receiver surface"
            )
            finding["fire_condition"] = (
                "ddm_rc2 closes the scope review and a new receiver-closed PR130 representation "
                "creates a distinct probability object with n600 and pre-uint8 access"
            )
        else:
            finding["fire_order"] = (
                "After ddm_rc2 closes #996 scope, reuse the retained exact fixtures only for a "
                "distinct current-base probability object."
                if status == ROUTING_STATUS_ROUTED_QUEUED
                else "Already consumed; retain as provenance and do not re-harvest."
            )
        findings.append(finding)
    return findings


def _partition_table(report: dict[str, Any]) -> str:
    lines = [
        "# ddm_vh2 vehicle-corpus partition",
        "",
        f"Denominator: {report['denominator_definition']}.",
        f"Lineage rule: {report['lineage_definition']}.",
        "",
        "| lineage | artifacts | harvested | routed | un-harvested |",
        "|---|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        report["lineages"].items(),
        key=lambda item: (-item[1]["artifacts"], item[0]),
    )
    for lineage, counts in ordered:
        lines.append(
            f"| {lineage} | {counts['artifacts']} | {counts['harvested']} | "
            f"{counts['routed']} | {counts['un_harvested']} |"
        )
    totals = report["totals"]
    lines.append(
        f"| **TOTAL** | **{totals['artifacts']}** | **{totals['harvested']}** | "
        f"**{totals['routed']}** | **{totals['un_harvested']}** |"
    )
    lines.append("")
    return "\n".join(lines)


def _report_markdown(
    *,
    corpus_before: int,
    markdown_count: int,
    rows: list[dict[str, Any]],
    appended: int,
    already_present: int,
    coverage_report: dict[str, Any],
    pre_sha: str,
    post_sha: str,
) -> str:
    statuses = Counter(row["status"] for row in rows)
    totals = coverage_report["totals"]
    status_lines = "\n".join(f"- `{status}`: {count}" for status, count in sorted(statuses.items()))
    return f"""# ddm_vh2 vehicle harvest routing ledger report

Axis: `[macOS-CPU advisory, scorer-free]`; `score_claim=false`; PR130 CPR1 frontier unchanged.

## Result

- Current `.omx/research/*.md` count at materialization: **{markdown_count}**. The charter's
  7,092 snapshot is one lower and is therefore recorded as a drifted seed, not current authority.
- The charter's `ddm_* artifacts = 3,001` did **not** reproduce under a stated filesystem scope.
- Reproducible vehicle denominator before writes: **{corpus_before}** top-level `ddm_*` entries;
  files and directories each count once, nested run payloads remain content of one artifact.
- Appended **{appended}** rows; **{already_present}** were already present and byte-semantically identical.
- Loaded prior findings: wl1=15, fh1=11, vp1=15. Drained entropy=7/7 artifacts.
- Coverage after append: artifacts={totals["artifacts"]}, harvested={totals["harvested"]},
  routed={totals["routed"]}, un-harvested={totals["un_harvested"]}.
- Canonical ledger SHA-256: `{pre_sha}` before, `{post_sha}` after.

## Typed row dispositions

{status_lines}

No row is UNOWNED. Every queued/fired/remeasure row has owner, consumer, and fire order; every
deferred/dead row has owner, consumer, named blocker, and fire condition.

## Partition-1 choice and finding

Entropy was selected because current PR130 campaign state is rate-dominant while pose is closed by
pk2 and the seg decomposition is blocked. The seven entropy artifacts form a complete, bounded
lineage. Their strongest old-object fact is real: exact entropy reduced a six-stream n64 payload
from 274,664 B to 45,369 B. It does not port as a PR130 win. All eight safe-zero subsets were
infeasible at all five tolerances, the evidence is n64, and PR130 has no matching direct-description
object. The retained payloads remain useful fixtures only after `ddm_rc2` closes the #996 coder-axis
scope review and a new representation creates a distinct probability object.

## Closure boundaries

- `pk2` closes the tested frozen-PR130 pose recode family, not all pose retraining.
- #996 is cited only as **coder axis, scope-under-review by `ddm_rc2`**.
- `113b52fdb1` closes the declared receiver-v7 gauge bank at its 2,000 B trigger.
- #917 closes retired-vehicle instruments as current production routes.
- Decode time was not used as a disqualifier.

## RECALL EVIDENCE

Read the common contract, PROGRAM, operating manual, current hot state, canonical equation index,
sub-0.15 DAG, task-status surfaces, and the complete wl1/fh1/vp1 and entropy source artifacts before
routing. The prior harvest wording and routing were loaded rather than re-derived. Source SHA-256
values are carried on every ledger row; the retained input and manifest carry the complete list.

## Validation boundary

No scorer, trainer, Modal job, upstream file, protected state file, archive evaluator, or exact-score
pointer was touched. This unit builds a means, not goal progress: no exact row moved and the exact
frontier remains unchanged.
"""


def _artifact_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ap-root", type=Path, default=AP_ROOT)
    parser.add_argument("--review-root", type=Path, default=REVIEW_ROOT)
    args = parser.parse_args()

    corpus_before = len(partition_vehicle_corpus())
    markdown_count = sum(1 for _ in (REPO / ".omx/research").glob("*.md"))
    rows = _wl_rows() + _fh_rows() + _vp_rows() + _entropy_rows()
    if len(rows) != 48:
        raise RuntimeError(f"expected 48 routing rows, got {len(rows)}")

    ledger_before = PROBE_OUTCOMES_LEDGER_PATH.read_bytes()
    pre_sha = _sha256_bytes(ledger_before)
    result = register_vehicle_routing_findings(rows)
    ledger_after = PROBE_OUTCOMES_LEDGER_PATH.read_bytes()
    post_sha = _sha256_bytes(ledger_after)
    coverage_report = coverage()
    partition_markdown = _partition_table(coverage_report)
    report = _report_markdown(
        corpus_before=corpus_before,
        markdown_count=markdown_count,
        rows=rows,
        appended=len(result.appended),
        already_present=len(result.already_present),
        coverage_report=coverage_report,
        pre_sha=pre_sha,
        post_sha=post_sha,
    )

    args.ap_root.mkdir(parents=True, exist_ok=True)
    args.review_root.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for root in (args.ap_root, args.review_root):
        outputs = {
            root / "routing_rows_input.json": (json.dumps(rows, indent=2, sort_keys=True) + "\n").encode(),
            root / "coverage.json": (json.dumps(coverage_report, indent=2, sort_keys=True) + "\n").encode(),
            root / "partition_table.md": partition_markdown.encode(),
            root / "ROUTING_LEDGER_REPORT.md": report.encode(),
        }
        for path, payload in outputs.items():
            _atomic_write(path, payload)
            generated.append(path)

    ledger_copy = args.ap_root / "probe_outcomes.after.jsonl"
    _atomic_write(ledger_copy, ledger_after)
    generated.append(ledger_copy)

    source_paths = [WL_DOC, FH_DOC, VP_DOC] + [REPO / item[1] for item in ENTROPY_ARTIFACTS]
    source_manifest = [
        {
            "path": _relative(path),
            "kind": "directory" if path.is_dir() else "file",
            "sha256": _sha256_path(path),
        }
        for path in source_paths
    ]
    source_manifest_path = args.ap_root / "source_manifest.json"
    _write_json(source_manifest_path, source_manifest)
    generated.append(source_manifest_path)

    manifest = {
        "schema": "ddm_vh2_retention_manifest.v1",
        "axis": "[macOS-CPU advisory, scorer-free]",
        "score_claim": False,
        "command": ".venv/bin/python tools/ddm_vh2_vehicle_routing_ledger.py",
        "corpus_before": corpus_before,
        "markdown_count": markdown_count,
        "ledger": {
            "path": str(PROBE_OUTCOMES_LEDGER_PATH),
            "sha256_before": pre_sha,
            "sha256_after": post_sha,
            "bytes_before": len(ledger_before),
            "bytes_after": len(ledger_after),
            "rows_appended": len(result.appended),
            "rows_already_present": len(result.already_present),
        },
        "sources": source_manifest,
        "retained_outputs": [_artifact_receipt(path) for path in generated],
    }
    for root in (args.ap_root, args.review_root):
        _write_json(root / "run_manifest.json", manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
