#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the ddm_vo2 recursive instrument registry.

The registry is intentionally conservative: it imports the adjudicated round-0
ledgers, then adds an overinclusive source-candidate census for live Python
files that contain measurement/verdict vocabulary. Candidate rows are labelled
as candidates, not as proven verdict consumers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO / ".omx/research/ddm_vo2_20260806"
DEFAULT_LAST_GRADED = "2026-08-06T00:00:00Z"

ELEMENT_NAMES = (
    "initialization",
    "proposal_step_rule",
    "stopping_rule",
    "metric_inner_product",
    "subset_sampling",
    "realization",
    "projection_constraint_handling",
    "tie_breaks",
    "seed_determinism",
    "caches_staleness",
)

LIVE_ROOTS = ("experiments", "tools", "src/tac")
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "site-packages",
    "venv",
}

MEASUREMENT_TOKENS = (
    "d_seg",
    "d_pose",
    "score_claim",
    "promotion_eligible",
    "verdict",
    "MEASURED",
    "DERIVED",
    "axis",
    "archive_bytes",
    "archive.zip",
    "eta",
    "ledger",
    "receipt",
    "SegNet",
    "PoseNet",
    "evaluate.py",
    "n600",
)

TOKEN_PATTERNS = {
    token: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])")
    for token in MEASUREMENT_TOKENS
}


@dataclass(frozen=True)
class RegistryBuild:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text).strip("_")
    return slug[:220] or "empty"


def _element(
    name: str,
    grade: str = "UNKNOWN",
    *,
    defect_class: str | None = None,
    receipt: str | None = None,
) -> dict[str, Any]:
    obj: dict[str, Any] = {"name": name, "form_grade": grade}
    if defect_class:
        obj["defect_class"] = defect_class
    if receipt:
        obj["receipt"] = receipt
    return obj


def _elements_for_defects(defects: list[str], receipts: list[str]) -> list[dict[str, Any]]:
    joined = " ".join(defects).lower()
    receipt = receipts[0] if receipts else None
    grades = {name: _element(name) for name in ELEMENT_NAMES}

    if any(tok in joined for tok in ("cap", "stop", "convergen", "budget")):
        grades["stopping_rule"] = _element(
            "stopping_rule",
            "NAIVE-NAMED",
            defect_class="cap_or_stop_receipt_incomplete",
            receipt=receipt,
        )
    if any(tok in joined for tok in ("euclidean", "diagonal", "metric", "fisher", "gram")):
        grades["metric_inner_product"] = _element(
            "metric_inner_product",
            "NAIVE-NAMED",
            defect_class="wrong_or_incomplete_metric",
            receipt=receipt,
        )
    if any(tok in joined for tok in ("prefix", "subset", "top-k", "top24", "aggregate", "sample")):
        grades["subset_sampling"] = _element(
            "subset_sampling",
            "NAIVE-NAMED",
            defect_class="subset_or_denominator_not_population_authority",
            receipt=receipt,
        )
    if any(tok in joined for tok in ("float", "round", "uint8", "lattice", "realiz")):
        grades["realization"] = _element(
            "realization",
            "NAIVE-NAMED",
            defect_class="float_first_or_receiver_lattice_seam",
            receipt=receipt,
        )
    if any(tok in joined for tok in ("project", "null", "constraint", "post-hoc")):
        grades["projection_constraint_handling"] = _element(
            "projection_constraint_handling",
            "NAIVE-NAMED",
            defect_class="project_after_or_constraint_late",
            receipt=receipt,
        )
    if any(tok in joined for tok in ("greedy", "fixed", "finite", "one-shot", "one_shot")):
        grades["proposal_step_rule"] = _element(
            "proposal_step_rule",
            "NAIVE-NAMED",
            defect_class="finite_menu_or_one_shot_proposal",
            receipt=receipt,
        )
        grades["tie_breaks"] = _element(
            "tie_breaks",
            "UNKNOWN",
            defect_class="ordering_or_restart_receipt_absent",
            receipt=receipt,
        )
    if "cache" in joined or "stale" in joined:
        grades["caches_staleness"] = _element(
            "caches_staleness",
            "NAIVE-NAMED",
            defect_class="cache_or_staleness_lineage_incomplete",
            receipt=receipt,
        )
    return [grades[name] for name in ELEMENT_NAMES]


def _source_lineage(sources: list[str]) -> list[dict[str, str]]:
    if not sources:
        return [{"kind": "UNANCHORED", "ref": "no receipt attached in this row"}]
    return [{"kind": "receipt_or_source", "ref": source} for source in sources]


def _row(
    *,
    instrument_id: str,
    path: str,
    elements: list[dict[str, Any]],
    calibration_lineage: list[dict[str, str]],
    verdict_fanout: int,
    cures_available: list[str],
    reopen_refs: list[str],
    last_graded: str,
    provenance_family: str,
    candidate_status: str = "VERDICT_CONSUMER_OR_ROUND0_INPUT",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "schema": "ddm_vo2_instrument_registry.row.v1",
        "instrument_id": instrument_id,
        "path": path,
        "elements": elements,
        "calibration_lineage": calibration_lineage,
        "verdict_fanout": verdict_fanout,
        "cures_available": cures_available,
        "reopen_refs": reopen_refs,
        "last_graded": last_graded,
        "provenance_family": provenance_family,
        "candidate_status": candidate_status,
        "notes": notes,
    }


def _rows_from_vo1(last_graded: str) -> list[dict[str, Any]]:
    fanout_rows = [
        row
        for row in _read_jsonl(REPO / ".omx/research/ddm_vo1_20260806/INSTRUMENT_FANOUT.jsonl")
        if row.get("kind") == "producer_fanout"
    ]
    reopen_rows = [
        row
        for row in _read_jsonl(REPO / ".omx/research/ddm_vo1_20260806/REOPEN_LEDGER.jsonl")
        if row.get("kind") == "reopen"
    ]
    out: list[dict[str, Any]] = []
    for item in fanout_rows:
        sources = [str(s) for s in item.get("sources", [])]
        defects = [str(s) for s in item.get("named_defects", [])]
        reopen_refs = [
            str(r["row_id"])
            for r in reopen_rows
            if any(d in " ".join(r.get("naive_or_suspect_instrument", [])) for d in defects)
        ][:8]
        out.append(
            _row(
                instrument_id=str(item["producer_id"]),
                path=sources[0] if sources else ".omx/research/ddm_vo1_20260806/INSTRUMENT_FANOUT.jsonl",
                elements=_elements_for_defects(defects, sources),
                calibration_lineage=_source_lineage(sources),
                verdict_fanout=int(item.get("fanout_count", 0)),
                cures_available=[str(c) for c in item.get("now_existing_cure", [])],
                reopen_refs=reopen_refs,
                last_graded=last_graded,
                provenance_family="vo1-round0",
                notes=str(item.get("fanout_basis", "")),
            )
        )
    return out


def _rows_from_ca1(last_graded: str) -> list[dict[str, Any]]:
    path = REPO / ".omx/research/ddm_ca1_20260805/ca1_classified_inventory_20260805.json"
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for site in obj.get("sites", []):
        defects: list[str] = []
        if site.get("status") == "silent_cap_default":
            defects.append("silent cap default")
        if str(site.get("ca1_class", "")).startswith("B_"):
            defects.append("load-bearing cap site")
        elements = _elements_for_defects(defects, [str(path.relative_to(REPO))])
        if site.get("status") == "reports_stop_reason":
            elements = [
                _element("stopping_rule", "OPTIMAL-RECEIPT", receipt=str(path.relative_to(REPO)))
                if element["name"] == "stopping_rule"
                else element
                for element in elements
            ]
        rows.append(
            _row(
                instrument_id=(
                    "ca1_cap_site:"
                    f"{_slug(str(site.get('key', 'unknown')))}"
                    f":line{int(site.get('line', 0))}"
                ),
                path=str(site.get("path", "")),
                elements=elements,
                calibration_lineage=_source_lineage([str(path.relative_to(REPO))]),
                verdict_fanout=1 if str(site.get("ca1_class", "")).startswith("B_") else 0,
                cures_available=["CapStopReceipt", "semantic stop receipt", "uncap ladder"],
                reopen_refs=["ca1_class_b_cap_stopped_rows"]
                if str(site.get("ca1_class", "")).startswith("B_")
                else [],
                last_graded=last_graded,
                provenance_family="ca1-round0",
                candidate_status="CAP_SITE_FULL_DENOMINATOR",
                notes=str(site.get("ca1_rationale", "")),
            )
        )
    return rows


def _rows_from_sw1(last_graded: str) -> list[dict[str, Any]]:
    path = REPO / ".omx/research/ddm_sw1_20260806/SEAM_METRIC_LEDGER.jsonl"
    rows: list[dict[str, Any]] = []
    for item in _read_jsonl(path):
        defects = [
            str(item.get("surface_class", "")),
            str(item.get("classification_group", "")),
            str(item.get("surface", "")),
        ]
        rows.append(
            _row(
                instrument_id=f"sw1_seam:{_slug(str(item.get('id', 'unknown')))}",
                path=str(item.get("surface", "")),
                elements=_elements_for_defects(defects, [str(path.relative_to(REPO))]),
                calibration_lineage=_source_lineage([str(path.relative_to(REPO))]),
                verdict_fanout=1,
                cures_available=[str(item.get("replacement", ""))],
                reopen_refs=[str(item.get("next_action", ""))],
                last_graded=last_graded,
                provenance_family="sw1-round0",
                candidate_status=str(item.get("classification_group", "SEAM_LEDGER_ROW")),
                notes="; ".join(str(e) for e in item.get("evidence", [])),
            )
        )
    return rows


def _rows_from_dk1(last_graded: str) -> list[dict[str, Any]]:
    path = REPO / ".omx/research/ddm_dk1_20260806/lattice_realizer_measurement.json"
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    receipt = ".omx/research/ddm_dk1_20260806/RECEIPT.md"
    for method, aggregate in obj.get("aggregate", {}).items():
        grade = "OPTIMAL-RECEIPT" if method == "cvp" else "NAIVE-NAMED"
        defect = None if method == "cvp" else "dominated_integer_realizer"
        elements = [_element(name) for name in ELEMENT_NAMES]
        elements = [
            _element("realization", grade, defect_class=defect, receipt=receipt)
            if element["name"] == "realization"
            else element
            for element in elements
        ]
        rows.append(
            _row(
                instrument_id=f"dk1_realizer:{method}",
                path="src/tac/optimization/lattice_native_pose_null_realizer.py",
                elements=elements,
                calibration_lineage=_source_lineage([receipt, str(path.relative_to(REPO))]),
                verdict_fanout=5 if method == "cvp" else 1,
                cures_available=["CVP/Babai kept-set enum", "Dykstra round-project"]
                if method != "cvp"
                else ["use as current local realizer primitive; no global MIQP claim"],
                reopen_refs=["q3x_q3_convergence_realizer", "sq1_pose_null_constrained_paint"],
                last_graded=last_graded,
                provenance_family="dk1-round0",
                candidate_status="MEASURED_SMALL_N_REALIZER",
                notes=json.dumps(aggregate, sort_keys=True),
            )
        )
    return rows


def _iter_live_python_files() -> list[Path]:
    files: list[Path] = []
    for root_name in LIVE_ROOTS:
        root = REPO / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            parts = path.relative_to(REPO).parts
            if any(part in SKIP_PARTS for part in parts):
                continue
            if len(parts) >= 2 and parts[:2] == ("experiments", "results"):
                continue
            if "tests" in parts or path.name.startswith("test_"):
                continue
            files.append(path)
    return sorted(files)


def _rows_from_source_candidates(last_graded: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = _iter_live_python_files()
    parseable = 0
    token_counts: Counter[str] = Counter()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        parseable += 1
        hits = [tok for tok, pattern in TOKEN_PATTERNS.items() if pattern.search(text)]
        if len(hits) < 2:
            continue
        for hit in hits:
            token_counts[hit] += 1
        rel = _rel(path)
        defects: list[str] = []
        low = text.lower()
        if "add_argument" in text and any(flag in text for flag in ("--steps", "--max-steps", "--iters", "--max-iters")):
            defects.append("possible cap or stop receipt site")
        if "project" in low and ("round" in low or "clip" in low):
            defects.append("possible project-after realization seam")
        if "prefix" in low or "topk" in low or "top_k" in low:
            defects.append("possible subset gate")
        if "euclidean" in low or "diagonal" in low:
            defects.append("possible metric-site")
        rows.append(
            _row(
                instrument_id=f"source_candidate:{_slug(rel)}",
                path=rel,
                elements=_elements_for_defects(defects, [rel]),
                calibration_lineage=_source_lineage([rel]),
                verdict_fanout=len(hits),
                cures_available=["R2 element decomposition required before verdict reuse"],
                reopen_refs=[],
                last_graded=last_graded,
                provenance_family="vo2-new",
                candidate_status="OVERINCLUSIVE_SOURCE_CANDIDATE_NEEDS_R2_CONSUMER_CONFIRMATION",
                notes=f"measurement/verdict tokens={','.join(hits)}",
            )
        )
    summary = {
        "live_python_files_scanned": len(files),
        "live_python_files_readable": parseable,
        "source_candidate_rows": len(rows),
        "token_hit_file_counts": dict(sorted(token_counts.items())),
    }
    return rows, summary


def build_registry(last_graded: str = DEFAULT_LAST_GRADED) -> RegistryBuild:
    rows: list[dict[str, Any]] = []
    row_groups: dict[str, int] = {}
    for name, builder in (
        ("vo1-round0", _rows_from_vo1),
        ("ca1-round0", _rows_from_ca1),
        ("sw1-round0", _rows_from_sw1),
        ("dk1-round0", _rows_from_dk1),
    ):
        new_rows = builder(last_graded)
        rows.extend(new_rows)
        row_groups[name] = len(new_rows)
    source_rows, source_summary = _rows_from_source_candidates(last_graded)
    rows.extend(source_rows)
    row_groups["vo2-new"] = len(source_rows)

    rows = sorted(rows, key=lambda r: (r["provenance_family"], r["instrument_id"]))
    summary = {
        "schema": "ddm_vo2_instrument_registry.summary.v1",
        "generated_at_utc": last_graded,
        "score_claim": False,
        "scorer_free": True,
        "round_reached": "R1",
        "round_dry_trajectory": [
            {"round": "R0", "new_rows": row_groups.get("vo1-round0", 0), "dry": False},
            {"round": "R1", "new_rows": len(rows), "dry": len(rows) == 0},
        ],
        "row_count": len(rows),
        "row_groups": row_groups,
        "source_candidate_denominator": source_summary,
        "registry_hash_scope": "INSTRUMENT_REGISTRY.jsonl rows sorted by provenance_family/instrument_id",
        "last_graded_source": "deterministic audit label; override with --last-graded for an intentional refresh",
    }
    return RegistryBuild(rows=rows, summary=summary)


def write_registry(out_dir: Path, *, last_graded: str = DEFAULT_LAST_GRADED) -> dict[str, str]:
    build = build_registry(last_graded=last_graded)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = out_dir / "INSTRUMENT_REGISTRY.jsonl"
    summary_path = out_dir / "ROUND_SUMMARY.json"
    registry_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in build.rows),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(build.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        _rel(registry_path): _sha256(registry_path),
        _rel(summary_path): _sha256(summary_path),
    }
    manifest_path = out_dir / "MANIFEST.sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--last-graded", default=DEFAULT_LAST_GRADED)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    if args.summary_only:
        build = build_registry(last_graded=args.last_graded)
        print(json.dumps(build.summary, indent=2, sort_keys=True))
        return 0
    manifest = write_registry(args.out_dir, last_graded=args.last_graded)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
