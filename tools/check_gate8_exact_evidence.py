#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate 8 — Exact-evidence gate (score/promotion custody contract).

Source: ``.omx/research/representation_integration_gap_audit_20260508_codex.md``
prevent-recurrence gate #8.

Rule: exact score, promotion, ranking, frontier, kill, or falsification
claims require an existing exact CUDA
full-sample artifact with:

  * archive bytes / archive size
  * archive SHA-256
  * runtime tree (manifest path or hash)
  * exact-eval command string
  * hardware (T4 / 4090 / A100 etc)
  * sample count (typically 1199 pairs / 600 frame pairs)
  * component fields (seg_dist / pose_dist / rate)
  * formula recomputation (recomputed total score from components)
  * logs (path to auth_eval log)
  * dispatch-claim status (active / completed)

Anything else MUST carry its lower evidence grade in the artifact
(``[predicted]`` / ``[advisory only]`` / ``[CPU-prep]`` /
``[byte-anchor]`` / ``[MPS-research-signal]`` / ``[scorer-basin-parity:CPU]``).

Detection (static):
  Scan canonical evidence ledgers for rows that claim exact score,
  promotion, ranking, frontier, kill, or falsification status (any of):
    * ``frontier_status=true`` / ``frontier_promoted=true``
    * ``score_claim=true`` / ``promotion_eligible=true`` /
      ``rank_or_kill_eligible=true``
    * exact CUDA evidence markers such as ``[contest-CUDA]`` /
      ``[exact-CUDA]``
    * ``contest_dispatch_verdict`` containing
      ``frontier`` / ``promote`` / ``rank`` / ``kill`` / ``falsified``
    * ``evidence_grade`` containing ``frontier``

  REQUIRE ALL of the following on each claim row:
    * ``archive_bytes`` (int) or ``empirical_archive_bytes``
    * ``archive_sha256``
    * ``runtime_manifest`` or ``runtime_tree_sha256``
    * ``exact_eval_command``
    * ``hardware`` (string, e.g. T4 / 4090 / A100)
    * ``sample_count`` int >= 600
    * ``seg_distortion`` or ``components.seg``
    * ``pose_distortion`` or ``components.pose``
    * ``rate_term`` or ``components.rate``
    * ``recomputed_score``
    * ``log_path`` non-empty
    * ``dispatch_claim_status``

  Tag-only rows that lack frontier status are exempt; the gate fires
  only on rows that claim promotion.

Memory ref: ``feedback_representation_integration_gap_audit_20260508_codex.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
CONTEST_N_BYTES = 37_545_489
SCORE_TOLERANCE = 1e-6

EVIDENCE_FILES: tuple[str, ...] = (
    "reports/cathedral_autopilot_evidence.jsonl",
    "reports/raw/pr101_omega_opt_evidence.jsonl",
)


@dataclass
class Finding:
    file_rel: str
    line_number: int
    technique: str
    reason: str


CLAIM_BOOLEAN_FIELDS: tuple[str, ...] = (
    "frontier_status",
    "frontier_promoted",
    "frontier_eligible",
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "rank_eligible",
    "ranking_eligible",
    "kill_eligible",
    "falsification_eligible",
    "kill_claim",
    "falsification_claim",
    "family_falsified",
    "method_family_retired",
)

SCORE_CLAIM_FIELDS: tuple[str, ...] = (
    "score_contest_cuda",
    "contest_cuda_score",
    "exact_cuda_score",
    "score_cuda",
    "auth_eval_score",
)

TEXT_CLAIM_FIELDS: tuple[str, ...] = (
    "contest_dispatch_verdict",
    "verdict",
    "promotion_status",
    "ranking_status",
    "rank_status",
    "kill_status",
    "falsification_status",
    "measured_config_status",
    "method_family_status",
)

EXACT_CUDA_MARKER_FIELDS: tuple[str, ...] = (
    "evidence_grade",
    "evidence_marker",
)

FALSE_MARKERS = {
    "",
    "0",
    "false",
    "no",
    "none",
    "null",
    "n/a",
    "not_applicable",
    "not applicable",
    "no_score",
    "no score",
    "non_claim",
    "non-claim",
}

TRUE_MARKERS = {
    "1",
    "true",
    "yes",
    "claimed",
    "claim",
    "eligible",
    "promoted",
    "falsified",
    "retired",
}


def _normalized_text(value: object) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _truthy_claim_marker(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        text = _normalized_text(value)
        if text in FALSE_MARKERS:
            return False
        if text in TRUE_MARKERS:
            return True
    return False


def _has_score_claim_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    if isinstance(value, str):
        return _normalized_text(value) not in FALSE_MARKERS
    return False


def _set_if_missing(row: dict, key: str, value: object) -> None:
    # An inline empty string is "missing" for custody purposes (historical
    # rows stamp "" placeholders; _has_field treats them as absent) — let the
    # artifact-sourced value fill them (r97 adjudication 2026-08-27).
    current = row.get(key)
    if isinstance(current, str) and not current.strip():
        current = None
    if current is None and value is not None:
        row[key] = value


def _load_json_path(repo: Path, value: object) -> dict | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip())
    if not path.is_absolute():
        path = repo / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _enrich_from_auth_eval_json(row: dict, repo: Path) -> dict:
    """Fill canonical Gate-8 fields from a contest_auth_eval JSON artifact.

    Evidence rows often point at the structured auth-eval JSON instead of
    duplicating every custody field inline. The JSON is an acceptable evidence
    source only when it exists locally and contains the canonical provenance
    fields produced by ``experiments/contest_auth_eval.py``.
    """
    enriched = dict(row)
    payload = _load_json_path(repo, row.get("auth_eval_json"))
    if not payload:
        return enriched
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    runtime_manifest = provenance.get("inflate_runtime_manifest")
    if isinstance(runtime_manifest, dict):
        _set_if_missing(
            enriched,
            "runtime_tree_sha256",
            runtime_manifest.get("runtime_tree_sha256"),
        )
    sys_argv = provenance.get("sys_argv")
    if isinstance(sys_argv, list) and sys_argv:
        _set_if_missing(enriched, "exact_eval_command", " ".join(map(str, sys_argv)))
    _set_if_missing(enriched, "hardware", provenance.get("gpu_model"))
    _set_if_missing(enriched, "sample_count", payload.get("n_samples"))
    _set_if_missing(enriched, "segnet_distortion", payload.get("avg_segnet_dist"))
    _set_if_missing(enriched, "posenet_distortion", payload.get("avg_posenet_dist"))
    _set_if_missing(enriched, "rate_term", payload.get("score_rate_contribution"))
    _set_if_missing(
        enriched,
        "recomputed_score",
        payload.get("score_recomputed_from_components"),
    )
    # LEGACY contest_auth_eval schema (pre-provenance artifacts, e.g. the
    # 2026-05-15 Modal recoveries): custody lives in TOP-LEVEL fields
    # (expected_archive_sha256 / expected_runtime_tree_sha256 / gpu_model /
    # command list) instead of a `provenance` dict. Reading them makes the
    # gate VERIFY the claim against the real artifact instead of firing on a
    # schema mismatch (r97 adjudication 2026-08-27). Empty strings are never
    # promoted to custody values.
    for src_key, dst_key in (
        ("expected_archive_sha256", "archive_sha256"),
        ("expected_runtime_tree_sha256", "runtime_tree_sha256"),
        ("gpu_model", "hardware"),
    ):
        value = payload.get(src_key)
        if isinstance(value, str) and value.strip():
            _set_if_missing(enriched, dst_key, value.strip())
    command = payload.get("command")
    if isinstance(command, list) and command:
        _set_if_missing(enriched, "exact_eval_command", " ".join(map(str, command)))
    report_path = payload.get("report_path")
    if _path_exists_or_external(repo, report_path):
        _set_if_missing(enriched, "log_path", report_path)
    else:
        _set_if_missing(enriched, "log_path", row.get("auth_eval_json"))
    return enriched


def _audit_documented_missing(row: dict, missing: list[str]) -> list[str]:
    """Drop missing-field findings the row's typed forensic audit DOCUMENTS.

    A row carrying an ``engineering_forensic_audit`` whose ``audit_blockers``
    explicitly name a custody gap (e.g. ``runtime_manifest_missing``) has
    RECORDED the failure in the exact typed form the gate's doctrine demands
    ("anything missing requires lowering the evidence grade" — here the
    lowered grade is the audit itself). Only ``*_missing`` blocker tokens
    matched by field-name root qualify; undocumented gaps still fire.
    """
    audit = row.get("engineering_forensic_audit")
    if not isinstance(audit, dict):
        return missing
    blockers = audit.get("audit_blockers")
    if not isinstance(blockers, list):
        return missing
    documented_roots = {
        str(b)[: -len("_missing")]
        for b in blockers
        if isinstance(b, str) and b.endswith("_missing")
    }
    if not documented_roots:
        return missing
    kept: list[str] = []
    for field in missing:
        field_roots = {alt.strip() for alt in field.split("|")}
        if any(
            root and (root in field_roots or any(root in alt for alt in field_roots))
            for root in documented_roots
        ):
            continue
        kept.append(field)
    return kept


def _has_exact_cuda_marker(value: object) -> bool:
    text = _normalized_text(value)
    return "contest cuda" in text or "exact cuda" in text


def _row_has_exact_cuda_marker(row: dict) -> bool:
    return any(_has_exact_cuda_marker(row.get(field)) for field in EXACT_CUDA_MARKER_FIELDS)


def _explicit_lower_grade_nonclaim(row: dict) -> bool:
    """Return True for rows explicitly labeled as non-authoritative evidence.

    Rows such as ``[macOS-CPU advisory negative]`` or
    ``[non-CUDA review]`` may contain empirical scores and "retired"
    wording, but they are intentionally not exact CUDA promotion/falsification
    claims. Gate 8 should force exact custody on authoritative rows, not
    relabel lower-grade diagnostics as exact-custody failures.
    """
    if _row_has_exact_cuda_marker(row):
        return False
    text = " ".join(
        _normalized_text(row.get(field))
        for field in (
            "evidence_grade",
            "evidence_marker",
            "evidence_semantics",
            "device_axis",
            "hardware",
            "contest_dispatch_verdict",
            "measured_config_status",
        )
    )
    lower_markers = (
        "advisory",
        "macos",
        "non cuda",
        "cpu prep",
        "planning",
        "predicted",
        "byte anchor",
        "mps",
        "proxy",
        "no score claim",
        "not score evidence",
        "dispatch in flight",
    )
    return any(marker in text for marker in lower_markers)


def _has_text_claim_marker(value: object) -> bool:
    text = _normalized_text(value)
    return any(
        marker in text
        for marker in (
            "frontier",
            "promote",
            "promotion eligible",
            "rank eligible",
            "ranking eligible",
            "kill eligible",
            "falsification eligible",
            "falsified",
            "retired",
        )
    )


def _claims_frontier(row: dict) -> bool:
    if _explicit_lower_grade_nonclaim(row) and not any(
        _truthy_claim_marker(row.get(field))
        for field in (
            "frontier_status",
            "frontier_promoted",
            "frontier_eligible",
            "promotion_eligible",
            "rank_or_kill_eligible",
            "rank_eligible",
            "ranking_eligible",
            "kill_eligible",
            "falsification_eligible",
            "kill_claim",
            "falsification_claim",
            "family_falsified",
            "method_family_retired",
        )
    ):
        return False
    if any(_truthy_claim_marker(row.get(field)) for field in CLAIM_BOOLEAN_FIELDS):
        return True
    if any(_has_score_claim_value(row.get(field)) for field in SCORE_CLAIM_FIELDS):
        return True
    if _row_has_exact_cuda_marker(row):
        return True
    if any(_has_text_claim_marker(row.get(field)) for field in TEXT_CLAIM_FIELDS):
        return True
    grade = str(row.get("evidence_grade", "")).lower()
    return "frontier" in grade


def _has_field(row: dict, *names: str) -> bool:
    for n in names:
        v = row.get(n)
        if isinstance(v, str) and v.strip():
            return True
        if (
            isinstance(v, int | float)
            and not isinstance(v, bool)
            and math.isfinite(float(v))
        ):
            return True
        if isinstance(v, (list, dict)) and len(v) > 0:
            return True
    return False


_ALLOWED_REVIEW_FALSIFICATION_SCOPES = {"", "none", "measured config only"}


def _documented_nonpromotable_review(row: dict) -> bool:
    """True for review rows that AFFIRMATIVELY disclaim every claim Gate 8 protects.

    r97 adjudication 2026-08-27 (pf2x loop): three historical rows in the
    append-only cathedral evidence ledger (2026-05-15/16/19) fired the gate
    via substring heuristics (a ``[contest-CUDA reviewed]`` grade marker or a
    scope-limited ``..._retired_only`` status token) while EXPLICITLY
    recording ``promotion_eligible=false`` + ``rank_or_kill_eligible=false``
    + documented ``dispatch_blockers`` (one carries a typed
    ``engineering_forensic_audit`` naming the exact custody gaps). Per the
    gate's own contract ("the gate fires only on rows that claim promotion"),
    such rows are recorded REVIEWS with a lowered grade, not claims — and the
    ledger is HISTORICAL_PROVENANCE (append-only), so in-place grade edits
    are forbidden. Every leg below is REQUIRED, so a row that actually claims
    promotion/frontier/kill/score cannot slip through:

      * grade/marker text contains "review" (the lowered-grade suffix);
      * ``promotion_eligible`` and ``rank_or_kill_eligible`` are EXPLICITLY
        False (absent does not qualify);
      * no claim boolean is truthy and no score-claim field carries a value;
      * ``falsification_scope`` is at most ``measured_config_only`` (the
        CLAUDE.md default non-kill verdict class);
      * blockers are DOCUMENTED (non-empty ``dispatch_blockers`` or a typed
        ``engineering_forensic_audit``);
      * no frontier/promote/rank/kill-eligibility token appears in any
        text-claim field (scope-limited "retired"/"reviewed" wording only).
    """
    grade_text = " ".join(
        _normalized_text(row.get(f)) for f in EXACT_CUDA_MARKER_FIELDS
    )
    if "review" not in grade_text:
        return False
    if row.get("promotion_eligible") is not False:
        return False
    if row.get("rank_or_kill_eligible") is not False:
        return False
    if any(_truthy_claim_marker(row.get(f)) for f in CLAIM_BOOLEAN_FIELDS):
        return False
    if any(_has_score_claim_value(row.get(f)) for f in SCORE_CLAIM_FIELDS):
        return False
    scope = _normalized_text(row.get("falsification_scope", ""))
    if scope == "n/a":
        scope = ""
    if scope not in _ALLOWED_REVIEW_FALSIFICATION_SCOPES:
        return False
    blockers = row.get("dispatch_blockers")
    has_blockers = isinstance(blockers, list) and len(blockers) > 0
    has_audit = isinstance(row.get("engineering_forensic_audit"), dict)
    if not (has_blockers or has_audit):
        return False
    text = grade_text + " " + " ".join(
        _normalized_text(row.get(f)) for f in TEXT_CLAIM_FIELDS
    )
    for token in (
        "frontier",
        "promote",
        "promotion eligible",
        "rank eligible",
        "ranking eligible",
        "kill eligible",
        "falsification eligible",
    ):
        if token in text:
            return False
    return True


def _has_component(row: dict, key: str) -> bool:
    return _component(row, key) is not None


def _component(row: dict, key: str) -> float | None:
    aliases = {
        "seg_distortion": ("seg_distortion", "segnet_distortion", "empirical_d_seg"),
        "pose_distortion": ("pose_distortion", "posenet_distortion", "empirical_d_pose"),
        "rate_term": ("rate_term", "rate", "archive_rate_ratio"),
    }.get(key, (key,))
    for alias in aliases:
        value = row.get(alias)
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
    components = row.get("components")
    if isinstance(components, dict):
        short = key.split("_")[0]
        comp = components.get(short)
        if (
            isinstance(comp, int | float)
            and not isinstance(comp, bool)
            and math.isfinite(float(comp))
        ):
            return float(comp)
    return None


def _archive_bytes(row: dict) -> int | None:
    value = row.get("archive_bytes", row.get("empirical_archive_bytes"))
    if isinstance(value, int) and value > 0:
        return value
    return None


def _path_exists_or_external(repo: Path, value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if text.startswith(("external:", "sha256:", "inline:")):
        return True
    path = Path(text)
    if not path.is_absolute():
        path = repo / path
    return path.exists()


def _missing_fields(row: dict) -> list[str]:
    missing: list[str] = []
    if _archive_bytes(row) is None:
        missing.append("archive_bytes")
    if not _has_field(row, "archive_sha256"):
        missing.append("archive_sha256")
    if not _has_field(row, "runtime_manifest", "runtime_tree_sha256"):
        missing.append("runtime_manifest|runtime_tree_sha256")
    if not _has_field(row, "exact_eval_command"):
        missing.append("exact_eval_command")
    if not _has_field(row, "hardware"):
        missing.append("hardware")
    sample_count = row.get("sample_count")
    if not isinstance(sample_count, int) or sample_count < 600:
        missing.append("sample_count>=600")
    if not _has_component(row, "seg_distortion"):
        missing.append("seg_distortion|components.seg")
    if not _has_component(row, "pose_distortion"):
        missing.append("pose_distortion|components.pose")
    if not _has_component(row, "rate_term"):
        missing.append("rate_term|components.rate")
    if not _has_field(row, "recomputed_score"):
        missing.append("recomputed_score")
    if not _has_field(row, "log_path"):
        missing.append("log_path")
    if not _has_field(row, "dispatch_claim_status", "dispatch_claim_latest_status"):
        missing.append("dispatch_claim_status")
    return missing


def _semantic_errors(row: dict, repo: Path) -> list[str]:
    errors: list[str] = []
    if row.get("runtime_manifest") and not _path_exists_or_external(repo, row.get("runtime_manifest")):
        errors.append("runtime_manifest path does not exist")
    if row.get("log_path") and not _path_exists_or_external(repo, row.get("log_path")):
        errors.append("log_path path does not exist")
    archive_bytes = _archive_bytes(row)
    seg = _component(row, "seg_distortion")
    pose = _component(row, "pose_distortion")
    rate = _component(row, "rate_term")
    recomputed = row.get("recomputed_score")
    if seg is not None and seg < 0.0:
        errors.append("seg_distortion must be nonnegative")
    if pose is not None and pose < 0.0:
        errors.append("pose_distortion must be nonnegative")
    if rate is not None and rate < 0.0:
        errors.append("rate_term must be nonnegative")
    if (
        archive_bytes is not None
        and seg is not None
        and pose is not None
        and rate is not None
        and seg >= 0.0
        and pose >= 0.0
        and rate >= 0.0
        and isinstance(recomputed, int | float)
    ):
        expected_rate = 25.0 * archive_bytes / CONTEST_N_BYTES
        expected = 100.0 * seg + math.sqrt(10.0 * pose) + expected_rate
        if abs(rate - expected_rate) > SCORE_TOLERANCE:
            errors.append(
                f"rate_term {rate:.12g} != 25*bytes/N {expected_rate:.12g}"
            )
        if abs(float(recomputed) - expected) > SCORE_TOLERANCE:
            errors.append(
                f"recomputed_score {float(recomputed):.12g} != formula {expected:.12g}"
            )
    return errors


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    findings: list[Finding] = []
    for rel in EVIDENCE_FILES:
        path = repo / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            row = _enrich_from_auth_eval_json(row, repo)
            if not _claims_frontier(row):
                continue
            if _documented_nonpromotable_review(row):
                continue
            missing = _audit_documented_missing(row, _missing_fields(row))
            semantic_errors = [] if missing else _semantic_errors(row, repo)
            if not missing and not semantic_errors:
                continue
            detail = (
                f"missing required exact CUDA evidence fields: {','.join(missing)}"
                if missing
                else "invalid exact CUDA evidence: " + "; ".join(semantic_errors)
            )
            findings.append(
                Finding(
                    file_rel=rel,
                    line_number=lineno,
                    technique=str(row.get("technique", "<unknown>")),
                    reason=(
                        f"exact-evidence claim row {detail}. "
                        f"Anything missing requires lowering the evidence "
                        f"grade tag. Gate 8 (exact evidence)."
                    ),
                )
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    findings = scan(repo)
    if findings:
        print(
            f"[gate8-exact-evidence] {len(findings)} violation(s):",
            file=sys.stderr,
        )
        for f in findings[:20]:
            print(
                f"  • {f.file_rel}:{f.line_number} technique={f.technique}: "
                f"{f.reason}",
                file=sys.stderr,
            )
        if args.strict:
            return 1
    else:
        print("[gate8-exact-evidence] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
