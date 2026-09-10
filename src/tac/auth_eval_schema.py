# SPDX-License-Identifier: MIT
"""Normalize contest auth-eval JSON schemas without weakening custody.

The canonical evaluator emits ``canonical_score``, component distances, and
archive custody fields. Some older wrappers emitted a nested
``score_components`` object. Dispatch scripts should not silently lose score
signal when one schema is absent, and they must not mark a result claimable
when canonical fields are missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

FULL_CONTEST_SAMPLE_COUNT = 600
CONTEST_CUDA_EVIDENCE_TAG = "[contest-CUDA]"
CONTEST_AUTH_AXIS_BY_EVIDENCE_GRADE = {
    "contest-CPU": "contest_cpu",
    "contest-CUDA": "contest_cuda",
}
ORIGINAL_VIDEO_BYTES = 37_545_489
DEFAULT_FORMULA_ABS_TOL = 1e-6
# Canonical-score provenance labels that mean "recomputed from components".
# The canonical emitter (experiments/contest_auth_eval.py) stamps the second,
# more precise label; the first is the historical schema literal. Both assert
# the same provenance, and the numeric guard below (score vs the contest
# formula over seg/pose/bytes) independently refuses payloads whose score does
# not actually recompute — the label check is provenance-only, never the sole
# numeric protection.
RECOMPUTED_CANONICAL_SCORE_SOURCES = frozenset(
    {
        "score_recomputed_from_components",
        "report_8dp_components_plus_exact_archive_bytes",
    }
)


def numeric_or_none(value: Any) -> float | None:
    """Return ``value`` as a finite float, excluding bool/null."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def int_or_none(value: Any) -> int | None:
    """Return ``value`` as an int when it is an integer-like scalar."""

    parsed = numeric_or_none(value)
    if parsed is None or int(parsed) != parsed:
        return None
    return int(parsed)


def contest_formula_score(
    *,
    seg_dist: float,
    pose_dist: float,
    archive_bytes: int,
) -> float:
    """Return the official contest score from component distances and bytes."""

    return (
        100.0 * seg_dist
        + math.sqrt(10.0 * pose_dist)
        + 25.0 * archive_bytes / ORIGINAL_VIDEO_BYTES
    )


def first_numeric(*values: Any) -> float | None:
    """Return the first numeric value from ``values``."""

    for value in values:
        parsed = numeric_or_none(value)
        if parsed is not None:
            return parsed
    return None


def eval_metric_summary(eval_data: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize canonical and legacy auth-eval result keys.

    ``score`` always prefers the recomputed canonical score over rounded
    display scores. ``rate`` is the score contribution; ``rate_unscaled`` is the
    raw ``archive_bytes / contest_denominator`` term.
    """

    if not eval_data:
        return {
            "score": None,
            "pose_avg": None,
            "seg_avg": None,
            "rate": None,
            "rate_unscaled": None,
            "archive_size_bytes": None,
            "n_samples": None,
            "canonical_score_source": None,
        }
    sc = eval_data.get("score_components") or {}
    if not isinstance(sc, dict):
        sc = {}
    return {
        "score": first_numeric(
            eval_data.get("canonical_score"),
            eval_data.get("score_recomputed_from_components"),
            eval_data.get("score"),
            eval_data.get("total_score"),
            eval_data.get("final_score"),
        ),
        "pose_avg": first_numeric(
            eval_data.get("avg_posenet_dist"),
            eval_data.get("pose_avg"),
            sc.get("pose"),
            sc.get("pose_avg"),
            sc.get("posenet"),
        ),
        "seg_avg": first_numeric(
            eval_data.get("avg_segnet_dist"),
            eval_data.get("seg_avg"),
            sc.get("seg"),
            sc.get("seg_avg"),
            sc.get("segnet"),
        ),
        "rate": first_numeric(
            eval_data.get("score_rate_contribution"),
            eval_data.get("rate"),
            sc.get("rate"),
            sc.get("rate_term"),
        ),
        "rate_unscaled": first_numeric(
            eval_data.get("rate_unscaled"),
            sc.get("rate_unscaled"),
        ),
        "archive_size_bytes": int_or_none(
            eval_data.get("archive_size_bytes")
            if eval_data.get("archive_size_bytes") is not None
            else eval_data.get("archive_bytes")
        ),
        "n_samples": int_or_none(eval_data.get("n_samples")),
        "canonical_score_source": eval_data.get("canonical_score_source"),
    }


def auth_eval_completion_summary(eval_data: dict[str, Any] | None) -> dict[str, Any]:
    """Return compact authenticated fields for completion logs.

    Remote wrappers should quote this summary instead of inventing labels such
    as ``contest_cuda_score`` from the requested device alone. The evaluator's
    own evidence fields decide whether a result is a score claim.
    """

    metrics = eval_metric_summary(eval_data)
    prov = _provenance(eval_data)
    payload = eval_data if isinstance(eval_data, dict) else {}
    return {
        "score": metrics.get("score"),
        "score_source": metrics.get("canonical_score_source"),
        "archive_size_bytes": metrics.get("archive_size_bytes"),
        "n_samples": metrics.get("n_samples"),
        "evidence_grade": payload.get("evidence_grade"),
        "lane_tag": payload.get("lane_tag"),
        "score_axis": payload.get("score_axis"),
        "evidence_semantics": payload.get("evidence_semantics"),
        "score_claim": payload.get("score_claim") is True,
        "score_claim_valid": payload.get("score_claim_valid") is True,
        "promotion_eligible": payload.get("promotion_eligible") is True,
        "rank_or_kill_eligible": payload.get("rank_or_kill_eligible") is True,
        "device": eval_device(eval_data),
        "gpu_model": prov.get("gpu_model"),
        "gpu_t4_match": prov.get("gpu_t4_match"),
    }


def load_auth_eval_json(path: Path) -> dict[str, Any]:
    """Load an auth-eval JSON object from disk, failing closed on bad shape."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"auth-eval JSON must contain an object, got {type(payload).__name__}")
    return payload


def _print_auth_eval_completion_summary(args: argparse.Namespace) -> int:
    summary = auth_eval_completion_summary(load_auth_eval_json(args.path))
    if args.field:
        value = summary.get(args.field)
        if value is None:
            return 2
        if isinstance(value, str):
            print(value)
        else:
            print(json.dumps(value, sort_keys=True, separators=(",", ":")))
        return 0
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the small CLI used by remote wrappers."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    completion = subparsers.add_parser(
        "completion-summary",
        help="Print compact authenticated completion fields from contest_auth_eval.json.",
    )
    completion.add_argument("path", type=Path)
    completion.add_argument(
        "--field",
        choices=tuple(auth_eval_completion_summary({}).keys()),
        help="Print one summary field instead of the full JSON object.",
    )
    completion.set_defaults(func=_print_auth_eval_completion_summary)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return int(args.func(args))


def required_exact_eval_metric_blockers(
    metrics: dict[str, Any],
    *,
    expected_archive_bytes: int | None = None,
    expected_n_samples: int | None = None,
    formula_abs_tol: float = DEFAULT_FORMULA_ABS_TOL,
) -> list[str]:
    """Return blockers that make an eval JSON non-claimable."""

    blockers: list[str] = []
    for key in ("score", "pose_avg", "seg_avg", "rate_unscaled", "archive_size_bytes"):
        if metrics.get(key) is None:
            blockers.append(f"{key}_missing")
    if metrics.get("canonical_score_source") not in RECOMPUTED_CANONICAL_SCORE_SOURCES:
        blockers.append("canonical_score_source_not_recomputed_from_components")
    score = metrics.get("score")
    pose_avg = metrics.get("pose_avg")
    seg_avg = metrics.get("seg_avg")
    archive_size_bytes = metrics.get("archive_size_bytes")
    if (
        score is not None
        and pose_avg is not None
        and seg_avg is not None
        and archive_size_bytes is not None
    ):
        if pose_avg < 0 or seg_avg < 0 or archive_size_bytes < 0:
            blockers.append("contest_formula_inputs_negative")
        else:
            recomputed = contest_formula_score(
                seg_dist=float(seg_avg),
                pose_dist=float(pose_avg),
                archive_bytes=int(archive_size_bytes),
            )
            if abs(float(score) - recomputed) > formula_abs_tol:
                blockers.append(
                    "score_component_formula_mismatch:"
                    f"score={float(score):.12g}:recomputed={recomputed:.12g}"
                )
    rate_unscaled = metrics.get("rate_unscaled")
    if rate_unscaled is not None and archive_size_bytes is not None:
        expected_rate = int(archive_size_bytes) / ORIGINAL_VIDEO_BYTES
        if abs(float(rate_unscaled) - expected_rate) > formula_abs_tol:
            blockers.append(
                "rate_unscaled_archive_bytes_mismatch:"
                f"rate={float(rate_unscaled):.12g}:expected={expected_rate:.12g}"
            )
    if (
        expected_archive_bytes is not None
        and metrics.get("archive_size_bytes") is not None
        and metrics["archive_size_bytes"] != expected_archive_bytes
    ):
        blockers.append(
            "archive_size_bytes_mismatch:"
            f"manifest={metrics['archive_size_bytes']}:actual={expected_archive_bytes}"
        )
    if (
        expected_n_samples is not None
        and metrics.get("n_samples") is None
    ):
        blockers.append("n_samples_missing")
    elif (
        expected_n_samples is not None
        and metrics["n_samples"] != expected_n_samples
    ):
        blockers.append(f"n_samples_mismatch:manifest={metrics['n_samples']}:expected={expected_n_samples}")
    return blockers


def _truthy(value: Any) -> bool:
    return value is True or (
        isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}
    )


def _provenance(eval_data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(eval_data, dict):
        return {}
    value = eval_data.get("provenance")
    return value if isinstance(value, dict) else {}


def eval_device(eval_data: dict[str, Any] | None) -> str | None:
    """Return the normalized auth-eval device, preferring provenance custody."""

    if not isinstance(eval_data, dict):
        return None
    prov = _provenance(eval_data)
    device = (
        prov.get("actual_device")
        if prov.get("actual_device") is not None
        else eval_data.get("actual_device")
        if eval_data.get("actual_device") is not None
        else prov.get("device")
        if prov.get("device") is not None
        else eval_data.get("device")
    )
    return str(device).strip().lower() if device is not None else None


def contest_cuda_hardware_is_documented(eval_data: dict[str, Any] | None) -> bool:
    """Return True only for T4 or explicitly documented equivalent CUDA hardware."""

    if not isinstance(eval_data, dict):
        return False
    prov = _provenance(eval_data)
    if _truthy(prov.get("gpu_t4_match")) or _truthy(eval_data.get("gpu_t4_match")):
        return True
    equivalent = _truthy(eval_data.get("contest_equivalent_hardware")) or _truthy(
        prov.get("contest_equivalent_hardware")
    )
    note = (
        eval_data.get("contest_equivalent_hardware_note")
        or eval_data.get("hardware_equivalence_note")
        or eval_data.get("contest_equivalent_hardware_source")
        or eval_data.get("hardware_equivalence_source")
        or prov.get("contest_equivalent_hardware_note")
        or prov.get("hardware_equivalence_note")
        or prov.get("contest_equivalent_hardware_source")
        or prov.get("hardware_equivalence_source")
    )
    return equivalent and isinstance(note, str) and bool(note.strip())


def required_contest_cuda_evidence_blockers(
    eval_data: dict[str, Any] | None,
    metrics: dict[str, Any],
    *,
    expected_archive_bytes: int | None = None,
    expected_n_samples: int = FULL_CONTEST_SAMPLE_COUNT,
) -> list[str]:
    """Return blockers for a claimable ``[contest-CUDA]`` exact-eval result."""

    blockers = required_exact_eval_metric_blockers(
        metrics,
        expected_archive_bytes=expected_archive_bytes,
        expected_n_samples=expected_n_samples,
    )
    if eval_device(eval_data) != "cuda":
        blockers.append("device_not_cuda")
    if not contest_cuda_hardware_is_documented(eval_data):
        blockers.append("contest_cuda_hardware_not_t4_or_documented_equivalent")
    if isinstance(eval_data, dict):
        lane_tag = eval_data.get("lane_tag")
        if lane_tag is not None and lane_tag != CONTEST_CUDA_EVIDENCE_TAG:
            blockers.append("evidence_tag_not_contest_cuda")
        score_axis = eval_data.get("score_axis")
        if score_axis is not None and score_axis != "contest_cuda":
            blockers.append("score_axis_not_contest_cuda")
        semantics = eval_data.get("evidence_semantics")
        if semantics is not None and semantics != "contest_cuda_exact_auth_eval":
            blockers.append("evidence_semantics_not_contest_cuda_exact_auth_eval")
        if "score_claim_valid" in eval_data and eval_data.get("score_claim_valid") is not True:
            blockers.append("score_claim_valid_not_true")
    return blockers


def required_contest_cpu_evidence_blockers(
    eval_data: dict[str, Any] | None,
    metrics: dict[str, Any],
    *,
    expected_archive_bytes: int | None = None,
    expected_n_samples: int = FULL_CONTEST_SAMPLE_COUNT,
) -> list[str]:
    """Return blockers for a claimable ``[contest-CPU]`` exact-eval result."""

    blockers = required_exact_eval_metric_blockers(
        metrics,
        expected_archive_bytes=expected_archive_bytes,
        expected_n_samples=expected_n_samples,
    )
    if eval_device(eval_data) != "cpu":
        blockers.append("device_not_cpu")
    prov = _provenance(eval_data)
    if prov.get("platform_system") != "Linux":
        blockers.append("contest_cpu_platform_system_not_linux")
    machine = str(prov.get("platform_machine") or "").lower()
    if machine not in {"x86_64", "amd64"}:
        blockers.append("contest_cpu_platform_machine_not_x86_64")
    if isinstance(eval_data, dict):
        lane_tag = eval_data.get("lane_tag")
        if lane_tag is not None and lane_tag != "[contest-CPU]":  # CUSTODY_VALIDATOR_OK:this_function_IS_custody_validator_creating_blockers_for_contest_cpu_tag_mismatch_per_comprehensive_bug_audit_cascade_20260526
            blockers.append("evidence_tag_not_contest_cpu")
        score_axis = eval_data.get("score_axis")
        if score_axis is not None and score_axis != "contest_cpu":
            blockers.append("score_axis_not_contest_cpu")
        semantics = eval_data.get("evidence_semantics")
        if semantics is not None and semantics != "public_leaderboard_cpu_reproduction":
            blockers.append("evidence_semantics_not_public_leaderboard_cpu_reproduction")
        if "score_claim_valid" in eval_data and eval_data.get("score_claim_valid") is not True:
            blockers.append("score_claim_valid_not_true")
    return blockers


def required_contest_auth_axis_payload_blockers(
    eval_data: dict[str, Any] | None,
    metrics: dict[str, Any],
    *,
    expected_archive_bytes: int | None = None,
    expected_n_samples: int = FULL_CONTEST_SAMPLE_COUNT,
) -> list[str]:
    """Return blockers when a payload is not a strict contest auth-eval axis.

    This is stricter than the older CUDA-only helper because transfer bridges
    use it as the right-hand authority surface. A diagnostic/advisory payload
    with matching numbers must not become a local-acceleration calibration
    target merely by carrying plausible score fields.
    """

    if not isinstance(eval_data, dict):
        return ["auth_eval_payload_missing_or_not_object"]

    blockers: list[str] = []
    diagnostic_blockers = eval_data.get("diagnostic_blockers")
    if isinstance(diagnostic_blockers, list) and diagnostic_blockers:
        blockers.append("diagnostic_blockers_present")
    elif diagnostic_blockers is not None and diagnostic_blockers != []:
        blockers.append("diagnostic_blockers_malformed_or_present")

    grade = eval_data.get("evidence_grade")
    if grade == "contest-CUDA":
        _extend_unique(
            blockers,
            required_contest_cuda_evidence_blockers(
                eval_data,
                metrics,
                expected_archive_bytes=expected_archive_bytes,
                expected_n_samples=expected_n_samples,
            ),
        )
        _require_exact_field(blockers, eval_data, "lane_tag", CONTEST_CUDA_EVIDENCE_TAG)
        _require_exact_field(blockers, eval_data, "score_axis", "contest_cuda")
        _require_exact_field(
            blockers,
            eval_data,
            "evidence_semantics",
            "contest_cuda_exact_auth_eval",
        )
        if eval_data.get("exact_cuda_eval_complete") is not True:
            blockers.append("exact_cuda_eval_complete_not_true")
    elif grade == "contest-CPU":
        _extend_unique(
            blockers,
            required_contest_cpu_evidence_blockers(
                eval_data,
                metrics,
                expected_archive_bytes=expected_archive_bytes,
                expected_n_samples=expected_n_samples,
            ),
        )
        _require_exact_field(blockers, eval_data, "lane_tag", "[contest-CPU]")
        _require_exact_field(blockers, eval_data, "score_axis", "contest_cpu")
        _require_exact_field(
            blockers,
            eval_data,
            "evidence_semantics",
            "public_leaderboard_cpu_reproduction",
        )
        if eval_data.get("cpu_leaderboard_reproduction_eligible") is not True:
            blockers.append("cpu_leaderboard_reproduction_eligible_not_true")
    else:
        blockers.append("auth_eval_evidence_grade_not_contest_cpu_or_cuda")

    if eval_data.get("score_claim") is not True:
        blockers.append("score_claim_not_true")
    if eval_data.get("score_claim_valid") is not True:
        blockers.append("score_claim_valid_not_true")
    if eval_data.get("promotion_eligible") is not False:
        blockers.append("promotion_eligible_missing_or_not_false")
    if eval_data.get("rank_or_kill_eligible") is not False:
        blockers.append("rank_or_kill_eligible_missing_or_not_false")
    return blockers


def _extend_unique(blockers: list[str], additions: list[str]) -> None:
    for blocker in additions:
        if blocker not in blockers:
            blockers.append(blocker)


def _require_exact_field(
    blockers: list[str],
    payload: dict[str, Any],
    key: str,
    expected: str,
) -> None:
    if payload.get(key) != expected:
        blockers.append(f"{key}_not_{expected}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


CPU_AXIS_REFUSAL_SCHEMA = "contest_cpu_axis_refusal.v1"
RUNTIME_FILES_DIGEST_DEFINITION = "contest_auth_eval.runtime_files_sha256.v1"


def runtime_files_digest(manifest: dict[str, Any]) -> str:
    """Hash the evaluator's environment-free files + evaluate.py projection.

    This is the definition in experiments.contest_auth_eval._runtime_dependency_manifest,
    not runtime_tree_sha256 or runtime_content_tree_sha256. Callers must obtain
    the manifest by hashing the current packet; a claimed digest is insufficient.
    """
    import hashlib

    rows = manifest["files"]
    names = [row["relative_path"] for row in rows]
    if not rows or len(names) != len(set(names)):
        raise ValueError("empty or duplicate runtime file rows")
    payload = {
        "files": sorted(
            ({key: row[key] for key in ("relative_path", "bytes", "sha256")} for row in rows),
            key=lambda row: str(row["relative_path"]),
        ),
        "upstream_evaluate_py": manifest["upstream_evaluate_py"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def retained_json_reference(path: Path) -> dict[str, Any]:
    """Identify retained JSON bytes without changing or adjudicating them."""
    import hashlib

    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _read_bound_json(reference: dict[str, Any]) -> dict[str, Any]:
    """Read a strict file reference and refuse stale, malformed or changed bytes."""
    import hashlib

    if set(reference) != {"path", "bytes", "sha256"}:
        raise ValueError("invalid retained reference fields")
    data = Path(reference["path"]).read_bytes()
    if type(reference["bytes"]) is not int or len(data) != reference["bytes"]:
        raise ValueError("retained reference size mismatch")
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("retained reference sha mismatch")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("retained JSON must be an object")
    return payload


def required_contest_cpu_axis_refusal_blockers(
    payload: dict[str, Any],
    *,
    selected_axis: str,
    archive: dict[str, Any],
    submission_dir: Path,
    runtime_manifest: dict[str, Any],
) -> list[str]:
    """Validate a metric-free refusal against retained evidence and live files.

    An adjudication is custody evidence, not a CPU evaluation. The outer remote
    runtime digest can be empty on refusal; the retained provenance, command pin,
    adjudication's receipt reference and independently measured packet must agree.
    No provider call or receiver execution occurs here.
    """
    import ast
    import hashlib

    blockers: list[str] = []
    if selected_axis != "contest_cuda" or payload.get("selected_axis") != "contest_cuda":
        blockers.append("refusal_requires_selected_contest_cuda")
    # Closed fields forbid metric aliases and unexpected nested score containers.
    expected = {"schema", "archive_sha256", "archive_bytes", "runtime_digest_definition",
                "runtime_files_sha256", "receiver", "modal_call_id", "receipt",
                "adjudication", "selected_axis", "cpu_metrics_absent_by_design"}
    if set(payload) != expected or payload.get("schema") != CPU_AXIS_REFUSAL_SCHEMA:
        blockers.append("refusal_schema_fields_invalid_metrics_forbidden")
    if payload.get("cpu_metrics_absent_by_design") is not True:
        blockers.append("cpu_metrics_must_be_absent_by_design")
    try:
        actual_digest = runtime_files_digest(runtime_manifest)
        if (payload.get("runtime_digest_definition") != RUNTIME_FILES_DIGEST_DEFINITION
                or payload.get("runtime_files_sha256") != actual_digest):
            blockers.append("refusal_runtime_digest_mismatch")
        archive_path = submission_dir / "archive.zip"
        actual_archive = {"sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                          "bytes": archive_path.stat().st_size}
        if (archive.get("sha256") != actual_archive["sha256"]
                or payload.get("archive_sha256") != actual_archive["sha256"]):
            blockers.append("refusal_archive_sha_mismatch")
        if (type(payload.get("archive_bytes")) is not int
                or archive.get("bytes") != actual_archive["bytes"]
                or payload.get("archive_bytes") != actual_archive["bytes"]):
            blockers.append("refusal_archive_size_mismatch")
        adjudication = _read_bound_json(payload["adjudication"])
        remote = _read_bound_json(payload["receipt"])
        if (adjudication.get("schema") != "cpu_axis_adjudication.v1"
                or adjudication.get("axis") != "contest_cpu"
                or adjudication.get("verdict") != "REFUSED_BY_DESIGN"
                or adjudication.get("receipt") != payload["receipt"]
                or adjudication.get("archive_sha256") != actual_archive["sha256"]
                or adjudication.get("archive_bytes") != actual_archive["bytes"]
                or adjudication.get("call") != payload["modal_call_id"]
                or not str(payload["modal_call_id"]).startswith("fc-")
                or adjudication.get("cpu_metrics_absent_by_design") is not True
                or adjudication.get("score_claim") is not False
                or type(adjudication.get("returncode")) is not int
                or adjudication.get("returncode") != 1):
            blockers.append("refusal_adjudication_binding_invalid")
        artifacts = remote["artifacts"]
        provenance = artifacts["provenance.json"]
        if isinstance(provenance, str):
            provenance = json.loads(provenance)
        retained_manifest = provenance["inflate_runtime_manifest"]
        if (runtime_files_digest(retained_manifest) != actual_digest
                or retained_manifest.get("runtime_files_sha256") != actual_digest):
            blockers.append("refusal_retained_runtime_mismatch")
        command = remote["command"]
        if (not isinstance(command, list) or not all(isinstance(arg, str) for arg in command)
                or command.count("--device") != 1
                or command.count("--expected-runtime-files-sha256") != 1
                or any(
                    arg.startswith("--") and any(
                        flag.startswith(arg.split("=", 1)[0]) and arg != flag
                        for flag in ("--device", "--expected-runtime-files-sha256")
                    ) for arg in command
                )):
            raise ValueError("ambiguous CPU command binding flags")
        pin = command.index("--expected-runtime-files-sha256")
        device = command.index("--device")
        if command[pin + 1] != actual_digest or command[device + 1] != "cpu":
            blockers.append("refusal_command_binding_invalid")
        if (remote.get("expected_archive_sha256") != actual_archive["sha256"]
                or remote.get("expected_archive_size_bytes") != actual_archive["bytes"]
                or provenance.get("archive_sha256") != actual_archive["sha256"]
                or provenance.get("archive_size_bytes") != actual_archive["bytes"]):
            blockers.append("refusal_retained_archive_mismatch")
        if (remote.get("passed") is not False or type(remote.get("returncode")) is not int
                or remote.get("returncode") != 1
                or remote.get("score_axis") != "contest_cpu"
                or remote.get("score_claim") is not False
                or remote.get("promotion_eligible") is not False
                or provenance.get("device") != "cpu"
                or provenance.get("platform_system") != "Linux"
                or provenance.get("platform_machine") != "x86_64"
                or provenance.get("cuda_available") is not False
                or any("contest_auth_eval.json" in name for name in artifacts)):
            blockers.append("refusal_not_metric_free_cpu_failure")
        metric_names = {"score", "canonical_score", "score_recomputed_from_components", "score_components",
                        "avg_segnet_dist", "avg_posenet_dist", "seg", "pose", "rate", "rate_unscaled",
                        "d_seg", "d_pose", "S", "metrics"}
        if metric_names & (remote.keys() | provenance.keys() | adjudication.keys()):
            blockers.append("refusal_retained_metrics_forbidden")
        outer_tree = remote.get("expected_runtime_tree_sha256")
        if outer_tree and outer_tree != retained_manifest.get("runtime_tree_sha256"):
            blockers.append("refusal_outer_runtime_mismatch")
        receiver = payload["receiver"]
        if set(receiver) != {"target", "launcher_line", "launcher_text", "guard_line", "guard_text", "error"}:
            raise ValueError("invalid receiver fields")
        launcher = (submission_dir / "inflate.sh").read_text().splitlines()
        source_text = (submission_dir / "inflate.py").read_text()
        source = source_text.splitlines()
        guard = next((node for node in ast.walk(ast.parse(source_text))
                      if isinstance(node, ast.If) and node.lineno == receiver["guard_line"]), None)
        if (guard is None or ast.unparse(guard.test) != "not torch.cuda.is_available()"
                or len(guard.body) != 1 or not isinstance(guard.body[0], ast.Raise)
                or not isinstance(guard.body[0].exc, ast.Call)
                or ast.unparse(guard.body[0].exc.func) != "RuntimeError"
                or len(guard.body[0].exc.args) != 1
                or not isinstance(guard.body[0].exc.args[0], ast.Constant)
                or guard.body[0].exc.args[0].value != receiver["error"]):
            blockers.append("refusal_guard_does_not_raise_recorded_error")
        if (receiver["target"] != "linux-nvidia-t4"
                or type(receiver["launcher_line"]) is not int or receiver["launcher_line"] < 1
                or type(receiver["guard_line"]) is not int or receiver["guard_line"] < 1
                or launcher[receiver["launcher_line"] - 1].strip() != receiver["launcher_text"]
                or receiver["launcher_text"] != 'python "$HERE/inflate.py" "$DATA_DIR" "$base" "$OUTPUT_DIR/$base.raw"'
                or source[receiver["guard_line"] - 1].strip() != receiver["guard_text"]
                or receiver["guard_text"] != "if not torch.cuda.is_available():"
                or "linux-nvidia-t4" not in receiver["error"]
                or "RuntimeError: " + receiver["error"] not in artifacts["contest_auth_eval.stderr.log"]
                or 'line ' + str(receiver["guard_line"] + 1) + ', in main' not in artifacts["contest_auth_eval.stderr.log"]):
            blockers.append("refusal_receiver_guard_not_bound")
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, SyntaxError) as exc:
        blockers.append(f"refusal_evidence_invalid:{type(exc).__name__}:{exc}")
    return blockers


SUBMISSION_POLICY_ADJUDICATION_SCHEMA = "submission_policy_adjudication.v1"
RAW_CUDA_POLICY_REVIEW_BLOCKERS = {
    "promotion_blockers": [
        "raw_auth_eval_does_not_verify_submission_policy_gates",
        "cpu_leaderboard_reproduction_not_adjudicated",
        "pre_submission_compliance_check_not_recorded",
    ],
    "rank_or_kill_blockers": [
        "raw_auth_eval_not_rank_or_kill_authority",
        "requires_adjudicated_cuda_cpu_policy_review",
    ],
}
# These remain FAILED in the compliance report and in the review receipt.
# Recording policy review does not grant release, rank, or promotion authority.
POLICY_REVIEW_RELEASE_BLOCKERS = frozenset({
    "submission_runtime_imports_within_allowlist", "hosted_archive_manifest_supplied",
})
RAW_POLICY_CHECK = "auth_eval_raw_promotion_policy_blockers_absent"


def submission_policy_adjudication(
    *,
    archive: dict[str, Any],
    runtime_files_sha256: str,
    cuda_receipt: dict[str, Any],
    cpu_refusal_receipt: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Record this checker's completed CUDA-only review, retaining release debt.

    Only the raw evaluator's five known requests for policy review are resolved.
    Unknown raw blockers or any other packet error fail closed. The caller must
    supply its own current, complete checks, including the validated CPU refusal;
    persisted user-supplied passed-check flags are never an input to the checker.
    """
    try:
        raw = _read_bound_json(cuda_receipt)
        refusal = _read_bound_json(cpu_refusal_receipt)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if ({key: raw.get(key) for key in RAW_CUDA_POLICY_REVIEW_BLOCKERS}
            != RAW_CUDA_POLICY_REVIEW_BLOCKERS
            or raw.get("score_axis") != "contest_cuda"
            or raw.get("exact_cuda_eval_complete") is not True
            or raw.get("score_claim_valid") is not True
            or refusal.get("schema") != CPU_AXIS_REFUSAL_SCHEMA
            or refusal.get("archive_sha256") != archive.get("sha256")
            or refusal.get("archive_bytes") != archive.get("bytes")
            or refusal.get("runtime_files_sha256") != runtime_files_sha256):
        return None
    by_name = {row["name"]: row for row in checks}
    required = {
        "auth_eval_archive_sha_matches", "auth_eval_archive_size_matches",
        "auth_eval_schema_metric_consistency", "auth_eval_explicit_exact_cuda_stamp",
        "auth_eval_selected_axis_matches_submission_gate", "auth_eval_t4_equivalent",
        "contest_cpu_auth_eval_score_parseable", "contest_cpu_auth_eval_archive_sha_matches",
        "contest_cpu_auth_eval_archive_size_matches", "contest_cpu_auth_eval_schema_metric_consistency",
        "contest_cpu_auth_eval_runtime_tree_recorded", "contest_cpu_auth_eval_runtime_tree_matches_cuda",
        "submission_runtime_manifest_computable", "submission_runtime_tree_matches_auth_eval",
    }
    if len(by_name) != len(checks) or any(by_name.get(name, {}).get("passed") is not True for name in required):
        return None
    unresolved = [row for row in checks if not row["passed"] and row["name"] != RAW_POLICY_CHECK]
    if any(row["name"] not in POLICY_REVIEW_RELEASE_BLOCKERS for row in unresolved):
        return None
    return {
        "schema": SUBMISSION_POLICY_ADJUDICATION_SCHEMA,
        "selected_axis": "contest_cuda",
        "archive_sha256": archive["sha256"], "archive_bytes": archive["bytes"],
        "runtime_digest_definition": RUNTIME_FILES_DIGEST_DEFINITION,
        "runtime_files_sha256": runtime_files_sha256,
        "cuda_receipt": cuda_receipt, "cpu_refusal_receipt": cpu_refusal_receipt,
        "basis": "current_packet_checks_and_bound_cpu_axis_refusal",
        "resolved_raw_review_blockers": {key: list(value) for key, value in RAW_CUDA_POLICY_REVIEW_BLOCKERS.items()},
        "passed_checks": [row for row in checks if row["passed"] and row["name"] != RAW_POLICY_CHECK],
        "unresolved_release_checks": unresolved,
        "review_complete": True,
        "release_ready": not unresolved,
        "cpu_metrics_absent_by_design": True,
        "cpu_leaderboard_reproduction_eligible": False,
        "promotion_eligible": False, "rank_or_kill_eligible": False,
        "note": "Records policy review only; raw evaluation flags remain unchanged. Remaining checks still block release.",
    }


def submission_policy_adjudication_matches(
    payload: dict[str, Any], *, current_packet_adjudication: dict[str, Any] | None,
) -> bool:
    """Accept only the exact object derived from this invocation's packet checks."""
    if current_packet_adjudication is None or payload.get("schema") != SUBMISSION_POLICY_ADJUDICATION_SCHEMA:
        return False
    try:
        return json.dumps(payload, sort_keys=True, allow_nan=False) == json.dumps(
            current_packet_adjudication, sort_keys=True, allow_nan=False,
        )
    except (ValueError, TypeError):
        return False
