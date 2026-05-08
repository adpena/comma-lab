"""Tests for representation-integration gates 1-10 (codex audit, 2026-05-08).

Source: ``.omx/research/representation_integration_gap_audit_20260508_codex.md``
prevent-recurrence gates 1-10. Each gate has:
  * a positive test on the live repo (live count <= warn-band, no
    unexpected regressions)
  * a negative test that synthesizes a violating manifest/row in a tmp
    dir and asserts the underlying scanner detects the violation
  * a clean-pass test in tmp showing a properly-tagged row passes

The negative + clean-pass tests load the ``tools/check_gate<N>_*.py``
scanner module directly and call ``scan(repo)`` so they don't need to
copy the scanner script into the synthetic repo root.

Memory ref:
``feedback_representation_integration_gates_landed_20260508.md``.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

from tac.preflight import (
    check_gate1_representation_promotion_card,
    check_gate2_no_naked_bytes,
    check_gate3_parser_section_manifest,
    check_gate4_export_first,
    check_gate5_runtime_closure,
    check_gate6_mask_pose_coupling,
    check_gate7_no_op_provenance,
    check_gate8_exact_evidence,
    check_gate9_blocker_ownership,
    check_gate10_stack_promotion,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_scanner(name: str):
    """Load a tools/check_gate<N>_<name>.py scanner module directly."""
    path = REPO_ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(
        f"_test_pact_scanner_{name}", path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── helpers ───────────────────────────────────────────────────────────────


def _write_evidence_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _write_build_manifest(
    tmp_path: Path, manifest: dict, name: str = "test_lane"
) -> Path:
    d = tmp_path / "experiments" / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "build_manifest.json"
    p.write_text(json.dumps(manifest))
    return p


# ── Gate 1: representation promotion card ────────────────────────────────


def test_gate1_passes_on_live_repo() -> None:
    violations = check_gate1_representation_promotion_card(
        strict=False, verbose=False
    )
    assert violations == [], f"unexpected: {violations}"


def test_gate1_strict_passes_on_live_repo() -> None:
    check_gate1_representation_promotion_card(strict=True, verbose=False)


def test_gate1_negative_card_missing_fields(tmp_path: Path) -> None:
    """Card-shaped manifest missing required fields fires."""
    p = tmp_path / "experiments" / "results" / "lane_x"
    p.mkdir(parents=True)
    (p / "representation_card.json").write_text(
        json.dumps(
            {
                "promotion_card": True,
                "representation_name": "test_repr",
                # missing 11 other required fields
            }
        )
    )
    mod = _load_scanner("check_gate1_representation_promotion_card.py")
    findings = mod.scan(tmp_path)
    assert any(f.representation_name == "test_repr" for f in findings)
    assert any("target_modes" in f.reason for f in findings)


def test_gate1_clean_card_passes(tmp_path: Path) -> None:
    p = tmp_path / "experiments" / "results" / "lane_x"
    p.mkdir(parents=True)
    (p / "representation_card.json").write_text(
        json.dumps(
            {
                "promotion_card": True,
                "representation_name": "test_repr",
                "target_modes": ["contest_exact_eval"],
                "source_artifact": "exp/foo.pt",
                "archive_builder": "tools/build_x.py",
                "inflate_consumer": "submissions/x/inflate.sh",
                "runtime_manifest": "submissions/x/runtime.json",
                "changed_payload_paths": ["renderer.bin"],
                "old_new_sha256s": {
                    "old": "deadbeef",
                    "new": "cafef00d",
                },
                "component_risk_plan": "no pose coupling expected",
                "exact_eval_command": "bash inflate.sh && python evaluate.py",
                "owner": "claude",
                "next_unblock_action": "T4 dispatch",
            }
        )
    )
    mod = _load_scanner("check_gate1_representation_promotion_card.py")
    findings = mod.scan(tmp_path)
    assert findings == []


# ── Gate 2: no naked bytes ────────────────────────────────────────────────


def test_gate2_passes_on_live_repo() -> None:
    violations = check_gate2_no_naked_bytes(strict=False, verbose=False)
    assert violations == [], f"unexpected: {violations}"


def test_gate2_strict_passes_on_live_repo() -> None:
    check_gate2_no_naked_bytes(strict=True, verbose=False)


def test_gate2_negative_score_claim_no_archive(tmp_path: Path) -> None:
    """Row with score_claim=true and no byte-closure proofs fires."""
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "test_naked",
                "empirical_archive_bytes": 100000,
                "score_claim": True,
                # no archive_path/sha256, no inflate_consumer
            }
        ],
    )
    mod = _load_scanner("check_gate2_no_naked_bytes.py")
    findings = mod.scan(tmp_path)
    assert any(f.technique == "test_naked" for f in findings)
    assert any("Gate 2" in f.reason for f in findings)


def test_gate2_proxy_row_passes(tmp_path: Path) -> None:
    """Row with score_claim=false is exempt."""
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "test_proxy",
                "empirical_archive_bytes": 100000,
                "score_claim": False,
            }
        ],
    )
    mod = _load_scanner("check_gate2_no_naked_bytes.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate2_score_claim_with_archive_passes(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "test_with_archive",
                "score_claim": True,
                "archive_path": "exp/archive.zip",
                "archive_sha256": "deadbeef",
            }
        ],
    )
    mod = _load_scanner("check_gate2_no_naked_bytes.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate2_contest_cuda_status_is_not_byte_closure(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "status_only",
                "score_claim": True,
                "measured_config_status": "contest_cuda_positive",
            }
        ],
    )
    mod = _load_scanner("check_gate2_no_naked_bytes.py")
    findings = mod.scan(tmp_path)
    assert any(f.technique == "status_only" for f in findings)


# ── Gate 3: parser-section manifest ──────────────────────────────────────


def test_gate3_passes_on_live_repo() -> None:
    violations = check_gate3_parser_section_manifest(strict=False, verbose=False)
    assert violations == [], f"unexpected: {violations}"


def test_gate3_strict_passes_on_live_repo() -> None:
    check_gate3_parser_section_manifest(strict=True, verbose=False)


def test_gate3_hnerv_manifest_missing_parser(tmp_path: Path) -> None:
    """HNeRV-family manifest missing parser-section fields fires."""
    _write_build_manifest(
        tmp_path,
        {
            "lane_id": "test_hnerv_lane",
            "packet_family": "hnerv_microcodec",
            "archive_bytes": 178000,
        },
        name="lane_hnerv_test",
    )
    mod = _load_scanner("check_gate3_parser_section_manifest.py")
    findings = mod.scan(tmp_path)
    assert any("test_hnerv_lane" in f.manifest_rel or
               "lane_hnerv_test" in f.manifest_rel for f in findings)
    assert any("section_sha256s" in f.reason for f in findings)


def test_gate3_vendored_intake_exempt(tmp_path: Path) -> None:
    """Vendored public-PR intake manifest is exempt."""
    d = tmp_path / "experiments" / "results" / "public_pr101_intake_x"
    d.mkdir(parents=True)
    (d / "build_manifest.json").write_text(
        json.dumps(
            {
                "lane_id": "public_pr101_intake",
                "packet_family": "hnerv_microcodec",
            }
        )
    )
    mod = _load_scanner("check_gate3_parser_section_manifest.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate3_clean_hnerv_passes(tmp_path: Path) -> None:
    _write_build_manifest(
        tmp_path,
        {
            "lane_id": "test_hnerv",
            "packet_family": "hnerv_microcodec",
            "offsets": [0, 1024, 4096],
            "lengths": [1024, 3072, 8192],
            "section_names": ["decoder", "latent", "side"],
            "section_sha256s": ["aa" * 32, "bb" * 32, "cc" * 32],
            "entropy_estimates": [4.5, 2.3, 6.1],
            "old_new_section_boundaries": {
                "decoder": {"old": [0, 1024], "new": [0, 1024]},
                "latent": {"old": [1024, 4096], "new": [1024, 4096]},
                "side": {"old": [4096, 12288], "new": [4096, 12288]},
            },
        },
        name="lane_hnerv_clean",
    )
    mod = _load_scanner("check_gate3_parser_section_manifest.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate3_a2k1_layout_magic_missing_parser_fires(tmp_path: Path) -> None:
    _write_build_manifest(
        tmp_path,
        {
            "lane_id": "test_a2k1",
            "archive_member_manifest": {
                "layout_magic": "A2K1",
            },
        },
        name="lane_a2k1_missing_parser",
    )
    mod = _load_scanner("check_gate3_parser_section_manifest.py")
    findings = mod.scan(tmp_path)
    assert findings
    assert any("section_sha256s" in f.reason for f in findings)


def test_gate3_cplx1_nested_parser_manifest_passes(tmp_path: Path) -> None:
    _write_build_manifest(
        tmp_path,
        {
            "lane_id": "test_cplx1",
            "wire_format": "CPLX1",
            "parser_section_manifest": {
                "offsets": [0, 4, 8],
                "lengths": [4, 4, 12],
                "section_names": ["cplx_magic", "decoder_section_len", "op1_inner_blob"],
                "section_sha256s": ["aa" * 32, "bb" * 32, "cc" * 32],
                "entropy_estimates": [1.0, 2.0, 3.0],
                "old_new_section_boundaries": {
                    "cplx_magic": {"old": [0, 4], "new": [0, 4]},
                    "decoder_section_len": {"old": [4, 8], "new": [4, 8]},
                    "op1_inner_blob": {"old": [8, 20], "new": [8, 20]},
                },
            },
        },
        name="lane_cplx1_clean",
    )
    mod = _load_scanner("check_gate3_parser_section_manifest.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate3_rejects_malformed_parser_schema(tmp_path: Path) -> None:
    _write_build_manifest(
        tmp_path,
        {
            "lane_id": "test_hnerv_bad_schema",
            "packet_family": "hnerv_microcodec",
            "offsets": [0, -1],
            "lengths": [1024],
            "section_names": ["decoder", "latent"],
            "section_sha256s": ["not_hex", "aa" * 32],
            "entropy_estimates": [4.5, "bad"],
            "old_new_section_boundaries": {
                "decoder": {"old": [0, 1024], "new": [0, 1024]},
            },
        },
        name="lane_hnerv_bad_schema",
    )
    mod = _load_scanner("check_gate3_parser_section_manifest.py")
    findings = mod.scan(tmp_path)
    assert any("invalid parser-section schema" in f.reason for f in findings)


# ── Gate 4: export-first ─────────────────────────────────────────────────


def test_gate4_warn_only_on_live_repo() -> None:
    """Live count is 2 warn (lane_12_nerv_mask_codec, lane_alpha_nerv_mask).

    Ships warn-only; flip strict after backfill."""
    violations = check_gate4_export_first(strict=False, verbose=False)
    # We expect the two known violations; assert that the count stays
    # bounded so a regression that adds new violators is caught.
    assert len(violations) <= 4, (
        f"gate4 live count drift: {len(violations)} violations "
        f"(expected <= 4): {violations}"
    )


def test_gate4_negative_lane_without_export_format(tmp_path: Path) -> None:
    """Lane in registry mentioning a learned-codec representation but
    missing export_format AND research_only flags fires."""
    p = tmp_path / ".omx" / "state" / "lane_registry.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lanes": [
                    {
                        "id": "lane_test_coolchic_x",
                        "name": "test coolchic lane",
                        "level": 2,
                    }
                ],
            }
        )
    )
    mod = _load_scanner("check_gate4_export_first.py")
    findings = mod.scan(tmp_path)
    assert any("Gate 4" in f.reason for f in findings)


def test_gate4_research_only_lane_passes(tmp_path: Path) -> None:
    p = tmp_path / ".omx" / "state" / "lane_registry.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lanes": [
                    {
                        "id": "lane_test_nerv",
                        "level": 1,
                        "research_only": True,
                    }
                ],
            }
        )
    )
    mod = _load_scanner("check_gate4_export_first.py")
    findings = mod.scan(tmp_path)
    assert findings == []


# ── Gate 5: runtime closure ──────────────────────────────────────────────


def test_gate5_passes_on_live_repo() -> None:
    violations = check_gate5_runtime_closure(strict=False, verbose=False)
    assert violations == [], f"unexpected: {violations}"


def test_gate5_strict_passes_on_live_repo() -> None:
    check_gate5_runtime_closure(strict=True, verbose=False)


def test_gate5_dispatch_manifest_no_runtime_closure(tmp_path: Path) -> None:
    _write_build_manifest(
        tmp_path,
        {
            "lane_id": "test_dispatch_no_runtime",
            "submission_dir_relpath": "exp/submission",
            "ready_for_exact_eval_dispatch": True,
        },
        name="lane_no_runtime",
    )
    mod = _load_scanner("check_gate5_runtime_closure.py")
    findings = mod.scan(tmp_path)
    assert any(f.technique == "test_dispatch_no_runtime" for f in findings)
    assert any("runtime" in f.reason.lower() for f in findings)


def test_gate5_nonexistent_runtime_manifest_does_not_close_runtime(tmp_path: Path) -> None:
    _write_build_manifest(
        tmp_path,
        {
            "lane_id": "test_dispatch_bad_runtime_path",
            "submission_dir_relpath": "exp/submission",
            "ready_for_exact_eval_dispatch": True,
            "runtime_manifest": "does/not/exist.json",
        },
        name="lane_bad_runtime_path",
    )
    mod = _load_scanner("check_gate5_runtime_closure.py")
    findings = mod.scan(tmp_path)
    assert any(f.technique == "test_dispatch_bad_runtime_path" for f in findings)


def test_gate5_existing_runtime_manifest_passes(tmp_path: Path) -> None:
    runtime_manifest = tmp_path / "experiments" / "results" / "lane_runtime" / "runtime.json"
    runtime_manifest.parent.mkdir(parents=True)
    runtime_manifest.write_text("{}", encoding="utf-8")
    _write_build_manifest(
        tmp_path,
        {
            "lane_id": "test_dispatch_runtime_path",
            "submission_dir_relpath": "exp/submission",
            "ready_for_exact_eval_dispatch": True,
            "runtime_manifest": "experiments/results/lane_runtime/runtime.json",
        },
        name="lane_runtime",
    )
    mod = _load_scanner("check_gate5_runtime_closure.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate5_public_pr_negative_no_failure_class(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "public_pr106_replay",
                "source": "[contest-CUDA] public_pr106 replay log",
                "contest_dispatch_verdict": "negative",
                # no failure_class — must be classified
            }
        ],
    )
    mod = _load_scanner("check_gate5_runtime_closure.py")
    findings = mod.scan(tmp_path)
    assert any("failure_class" in f.reason for f in findings)


def test_gate5_public_pr_negative_classified_passes(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "public_pr106_replay",
                "source": "[contest-CUDA] public_pr106 replay log",
                "contest_dispatch_verdict": "negative",
                "failure_class": "runtime_blocker_dependency_missing",
            }
        ],
    )
    mod = _load_scanner("check_gate5_runtime_closure.py")
    findings = mod.scan(tmp_path)
    assert findings == []


# ── Gate 6: mask/pose coupling ───────────────────────────────────────────


def test_gate6_passes_on_live_repo() -> None:
    violations = check_gate6_mask_pose_coupling(strict=False, verbose=False)
    assert violations == [], f"unexpected: {violations}"


def test_gate6_strict_passes_on_live_repo() -> None:
    check_gate6_mask_pose_coupling(strict=True, verbose=False)


def test_gate6_mask_replacement_missing_coupling_fields(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "nerv_mask_replacement_test",
                "lane_id": "test_nerv_mask_lane",
                "score_claim": True,
                "changed_payload_paths": ["masks.mkv"],
            }
        ],
    )
    mod = _load_scanner("check_gate6_mask_pose_coupling.py")
    findings = mod.scan(tmp_path)
    assert any("Gate 6" in f.reason for f in findings)


def test_gate6_proxy_mask_row_passes(tmp_path: Path) -> None:
    """Mask row without dispatch claim is exempt."""
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "nerv_mask_proxy",
                "score_claim": False,
                "changed_payload_paths": ["masks.mkv"],
            }
        ],
    )
    mod = _load_scanner("check_gate6_mask_pose_coupling.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate6_proxy_mask_build_manifest_passes(tmp_path: Path) -> None:
    _write_build_manifest(
        tmp_path,
        {
            "lane_id": "nerv_mask_proxy_manifest",
            "representation_name": "nerv_mask_proxy",
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        name="lane_mask_proxy",
    )
    mod = _load_scanner("check_gate6_mask_pose_coupling.py")
    findings = mod.scan(tmp_path)
    assert findings == []


# ── Gate 7: no-op + provenance ───────────────────────────────────────────


def test_gate7_passes_on_live_repo() -> None:
    violations = check_gate7_no_op_provenance(strict=False, verbose=False)
    assert violations == [], f"unexpected: {violations}"


def test_gate7_strict_passes_on_live_repo() -> None:
    check_gate7_no_op_provenance(strict=True, verbose=False)


def test_gate7_byte_repack_no_provenance(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "brotli_q_search_test",
                "score_claim": True,
                "transform_kind": "brotli_param",
            }
        ],
    )
    mod = _load_scanner("check_gate7_no_op_provenance.py")
    findings = mod.scan(tmp_path)
    assert any("Gate 7" in f.reason for f in findings)


def test_gate7_byte_repack_with_provenance_passes(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "brotli_q_search_test",
                "score_claim": True,
                "transform_kind": "brotli_param",
                "old_archive_sha256": "deadbeef" * 8,
                "new_archive_sha256": "cafef00d" * 8,
                "payload_change_proof": "exp/diff.log",
                "runtime_consumption_proof": "exp/runtime.log",
            }
        ],
    )
    mod = _load_scanner("check_gate7_no_op_provenance.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate7_byte_delta_claim_needs_provenance_even_without_token(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "neutral_name",
                "score_claim": True,
                "empirical_archive_bytes": 100,
                "baseline_archive_bytes": 120,
            }
        ],
    )
    mod = _load_scanner("check_gate7_no_op_provenance.py")
    findings = mod.scan(tmp_path)
    assert any(f.technique == "neutral_name" for f in findings)


# ── Gate 8: exact evidence ───────────────────────────────────────────────


def test_gate8_passes_on_live_repo() -> None:
    violations = check_gate8_exact_evidence(strict=False, verbose=False)
    assert violations == [], f"unexpected: {violations}"


def test_gate8_strict_passes_on_live_repo() -> None:
    check_gate8_exact_evidence(strict=True, verbose=False)


def test_gate8_frontier_row_missing_fields(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "test_frontier",
                "frontier_status": True,
                "archive_bytes": 178000,
                # missing many other required fields
            }
        ],
    )
    mod = _load_scanner("check_gate8_exact_evidence.py")
    findings = mod.scan(tmp_path)
    assert any("Gate 8" in f.reason for f in findings)


def test_gate8_score_promotion_rank_kill_claims_require_custody(
    tmp_path: Path,
) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "score_claim_marker",
                "score_claim": True,
                "empirical_archive_bytes": 178000,
            },
            {
                "technique": "promotion_eligible_marker",
                "promotion_eligible": True,
                "empirical_archive_bytes": 178000,
            },
            {
                "technique": "rank_or_kill_eligible_marker",
                "rank_or_kill_eligible": True,
                "empirical_archive_bytes": 178000,
            },
            {
                "technique": "exact_cuda_grade_marker",
                "evidence_grade": "[contest-CUDA A-negative]",
                "score_claim": False,
                "empirical_archive_bytes": 178000,
            },
            {
                "technique": "ranking_status_marker",
                "ranking_status": "rank eligible",
                "empirical_archive_bytes": 178000,
            },
            {
                "technique": "falsification_status_marker",
                "falsification_status": "kill eligible",
                "empirical_archive_bytes": 178000,
            },
        ],
    )
    mod = _load_scanner("check_gate8_exact_evidence.py")
    findings = mod.scan(tmp_path)

    techniques = {f.technique for f in findings}
    assert techniques == {
        "score_claim_marker",
        "promotion_eligible_marker",
        "rank_or_kill_eligible_marker",
        "exact_cuda_grade_marker",
        "ranking_status_marker",
        "falsification_status_marker",
    }
    assert all("Gate 8" in f.reason for f in findings)


def test_gate8_complete_frontier_row_passes(tmp_path: Path) -> None:
    runtime_manifest = tmp_path / "runtime.json"
    log_path = tmp_path / "auth_eval.log"
    runtime_manifest.write_text("{}", encoding="utf-8")
    log_path.write_text("ok\n", encoding="utf-8")
    archive_bytes = 178000
    seg = 0.0006
    pose = 0.001
    rate = 25.0 * archive_bytes / 37_545_489
    score = 100.0 * seg + math.sqrt(10.0 * pose) + rate
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "test_complete",
                "frontier_status": True,
                "archive_bytes": archive_bytes,
                "archive_sha256": "deadbeef" * 8,
                "runtime_manifest": str(runtime_manifest),
                "exact_eval_command": "bash inflate.sh && python evaluate.py",
                "hardware": "T4",
                "sample_count": 1199,
                "seg_distortion": seg,
                "pose_distortion": pose,
                "rate_term": rate,
                "recomputed_score": score,
                "log_path": str(log_path),
                "dispatch_claim_status": "completed",
            }
        ],
    )
    mod = _load_scanner("check_gate8_exact_evidence.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate8_rejects_bad_paths_and_score_recompute(tmp_path: Path) -> None:
    archive_bytes = 178000
    seg = 0.0006
    pose = 0.001
    rate = 25.0 * archive_bytes / 37_545_489
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "test_bad_frontier",
                "frontier_status": True,
                "archive_bytes": archive_bytes,
                "archive_sha256": "deadbeef" * 8,
                "runtime_manifest": "missing/runtime.json",
                "exact_eval_command": "bash inflate.sh && python evaluate.py",
                "hardware": "T4",
                "sample_count": 1199,
                "seg_distortion": seg,
                "pose_distortion": pose,
                "rate_term": rate,
                "recomputed_score": 999.0,
                "log_path": "missing/auth_eval.log",
                "dispatch_claim_status": "completed",
            }
        ],
    )
    mod = _load_scanner("check_gate8_exact_evidence.py")
    findings = mod.scan(tmp_path)
    assert any("invalid exact CUDA evidence" in f.reason for f in findings)


def test_gate8_negative_pose_component_fails_closed(tmp_path: Path) -> None:
    """Invalid component domains must be findings, not scanner crashes."""
    runtime_manifest = tmp_path / "runtime.json"
    log_path = tmp_path / "auth_eval.log"
    runtime_manifest.write_text("{}", encoding="utf-8")
    log_path.write_text("ok\n", encoding="utf-8")
    archive_bytes = 178000
    rate = 25.0 * archive_bytes / 37_545_489
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "negative_pose_exact_claim",
                "frontier_status": True,
                "archive_bytes": archive_bytes,
                "archive_sha256": "deadbeef" * 8,
                "runtime_manifest": str(runtime_manifest),
                "exact_eval_command": "bash inflate.sh && python evaluate.py",
                "hardware": "T4",
                "sample_count": 1199,
                "seg_distortion": 0.0006,
                "pose_distortion": -0.001,
                "rate_term": rate,
                "recomputed_score": 0.1,
                "log_path": str(log_path),
                "dispatch_claim_status": "completed",
            }
        ],
    )
    mod = _load_scanner("check_gate8_exact_evidence.py")

    findings = mod.scan(tmp_path)

    assert len(findings) == 1
    assert findings[0].technique == "negative_pose_exact_claim"
    assert "pose_distortion must be nonnegative" in findings[0].reason


def test_gate8_zero_components_are_present_but_formula_checked(
    tmp_path: Path,
) -> None:
    runtime_manifest = tmp_path / "runtime.json"
    log_path = tmp_path / "auth_eval.log"
    runtime_manifest.write_text("{}", encoding="utf-8")
    log_path.write_text("ok\n", encoding="utf-8")
    archive_bytes = 178000
    rate = 25.0 * archive_bytes / 37_545_489
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "zero_component_exact_claim",
                "frontier_status": True,
                "archive_bytes": archive_bytes,
                "archive_sha256": "deadbeef" * 8,
                "runtime_manifest": str(runtime_manifest),
                "exact_eval_command": "bash inflate.sh && python evaluate.py",
                "hardware": "T4",
                "sample_count": 1199,
                "seg_distortion": 0.0,
                "pose_distortion": 0.0,
                "rate_term": rate,
                "recomputed_score": rate,
                "log_path": str(log_path),
                "dispatch_claim_status": "completed",
            }
        ],
    )
    mod = _load_scanner("check_gate8_exact_evidence.py")

    findings = mod.scan(tmp_path)

    assert findings == []


def test_gate8_predicted_row_passes(tmp_path: Path) -> None:
    """Tag-only [predicted] row not claiming frontier is exempt."""
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "test_predicted",
                "evidence_grade": "[predicted]",
                "empirical_archive_bytes": 100000,
            }
        ],
    )
    mod = _load_scanner("check_gate8_exact_evidence.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate8_explicit_non_claim_row_passes(tmp_path: Path) -> None:
    """False claim markers and proxy wording do not require exact custody."""
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "ordinary_proxy_row",
                "evidence_grade": "[CPU-prep]",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "contest_dispatch_verdict": "DEFERRED-pending-research",
                "dispatch_blockers": [
                    "missing_exact_cuda_auth_eval",
                    "requires_exact_cuda_auth_eval_before_any_score_use",
                ],
                "empirical_archive_bytes": 178000,
            }
        ],
    )
    mod = _load_scanner("check_gate8_exact_evidence.py")
    findings = mod.scan(tmp_path)
    assert findings == []


# ── Gate 9: blocker ownership ────────────────────────────────────────────


def test_gate9_passes_on_live_repo() -> None:
    violations = check_gate9_blocker_ownership(strict=False, verbose=False)
    assert violations == [], f"unexpected: {violations}"


def test_gate9_strict_passes_on_live_repo() -> None:
    check_gate9_blocker_ownership(strict=True, verbose=False)


def test_gate9_blocked_lane_no_ownership(tmp_path: Path) -> None:
    p = tmp_path / ".omx" / "state" / "lane_registry.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lanes": [
                    {
                        "id": "lane_test_hnerv_blocked",
                        "level": 1,
                        "blocked": True,
                        "blockers": ["compliance pending"],
                    }
                ],
            }
        )
    )
    mod = _load_scanner("check_gate9_blocker_ownership.py")
    findings = mod.scan(tmp_path)
    assert any("Gate 9" in f.reason for f in findings)


def test_gate9_blocked_lane_with_owner_passes(tmp_path: Path) -> None:
    p = tmp_path / ".omx" / "state" / "lane_registry.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lanes": [
                    {
                        "id": "lane_test_hnerv_owned",
                        "level": 1,
                        "blocked": True,
                        "blockers": ["compliance pending"],
                        "active_owner": "claude",
                        "unblock_experiment": "T4 dispatch",
                    }
                ],
            }
        )
    )
    mod = _load_scanner("check_gate9_blocker_ownership.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate9_terminal_retirement_passes(tmp_path: Path) -> None:
    p = tmp_path / ".omx" / "state" / "lane_registry.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lanes": [
                    {
                        "id": "lane_test_coolchic_retired",
                        "level": 1,
                        "blocked": True,
                        "blockers": ["scoreboard moved past us"],
                        "terminal_retirement_note": (
                            "outclassed by PR101; no reactivation planned"
                        ),
                    }
                ],
            }
        )
    )
    mod = _load_scanner("check_gate9_blocker_ownership.py")
    findings = mod.scan(tmp_path)
    assert findings == []


# ── Gate 10: stack promotion ─────────────────────────────────────────────


def test_gate10_passes_on_live_repo() -> None:
    violations = check_gate10_stack_promotion(strict=False, verbose=False)
    assert violations == [], f"unexpected: {violations}"


def test_gate10_strict_passes_on_live_repo() -> None:
    check_gate10_stack_promotion(strict=True, verbose=False)


def test_gate10_stack_dispatch_missing_fields(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "hstack_vstack_test",
                "score_claim": True,
                # missing all stack-promotion fields
            }
        ],
    )
    mod = _load_scanner("check_gate10_stack_promotion.py")
    findings = mod.scan(tmp_path)
    assert any("Gate 10" in f.reason for f in findings)


def test_gate10_proxy_stack_must_explicitly_disable(tmp_path: Path) -> None:
    """Stack proxy without explicit score_claim=false fails."""
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "hstack_proxy_test",
                # no score_claim or ready_for_exact_eval_dispatch fields
            }
        ],
    )
    mod = _load_scanner("check_gate10_stack_promotion.py")
    findings = mod.scan(tmp_path)
    assert any("Gate 10" in f.reason for f in findings)


def test_gate10_proxy_stack_disabled_passes(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "hstack_proxy_clean",
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
    )
    mod = _load_scanner("check_gate10_stack_promotion.py")
    findings = mod.scan(tmp_path)
    assert findings == []


def test_gate10_proxy_stack_manifest_must_explicitly_disable(tmp_path: Path) -> None:
    _write_build_manifest(
        tmp_path,
        {
            "lane_id": "hstack_proxy_manifest",
        },
        name="lane_hstack_proxy",
    )
    mod = _load_scanner("check_gate10_stack_promotion.py")
    findings = mod.scan(tmp_path)
    assert any("Gate 10" in f.reason for f in findings)


def test_gate10_complete_stack_dispatch_passes(tmp_path: Path) -> None:
    _write_evidence_jsonl(
        tmp_path,
        [
            {
                "technique": "hstack_complete_test",
                "score_claim": True,
                "archive_boundary": {"decoder": [0, 100], "latent": [100, 200]},
                "side_information": "exp/side.json",
                "latent_streams": "exp/latent.bin",
                "k_scale_tables": "exp/k_scale.json",
                "decoder_overhead_bytes": 1024,
                "runtime_consumer": "submissions/x/inflate.sh",
                "exact_eval_plan": "T4 dispatch script ready",
            }
        ],
    )
    mod = _load_scanner("check_gate10_stack_promotion.py")
    findings = mod.scan(tmp_path)
    assert findings == []
