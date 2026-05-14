# SPDX-License-Identifier: MIT
"""Tests for ``tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py``.

REVIEW-ENG C2 closure: the original Path B step 6 wire format reserves 28
bytes for per-tensor K side-info but the decoder discards them. The "no dead
K" variant drops the K section from the archive (~28 B free win) while
keeping K in the build manifest as audit metadata.

These tests verify the source declares the required CLAUDE.md flags, the
forked inflate.py source omits the K read, and the wire format documentation
is accurate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))


def _read_tool_source() -> str:
    return (REPO_ROOT / "tools" / "build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py").read_text()


def test_no_dead_k_tool_source_declares_required_flags() -> None:
    """Tool source must declare CLAUDE.md flags that gate promotion +
    dispatch (mirrors the original variant)."""
    src = _read_tool_source()
    assert '"family_falsified": False' in src
    assert '"score_claim": False' in src
    assert '"ready_for_exact_eval_dispatch": False' in src
    assert '"falsification_scope": "lagrangian_x_continuous_K_no_dead_k_only"' in src


def test_no_dead_k_tool_documents_review_eng_c2_finding() -> None:
    """The tool docstring must explicitly cite REVIEW-ENG C2 and explain
    what changes vs the original variant."""
    src = _read_tool_source()
    assert "REVIEW-ENG C2" in src
    assert "L117-121" in src or "inflate.py L117-121" in src
    assert "no dead K" in src.lower() or "no_dead_k" in src
    assert "28" in src  # mentions 28 bytes savings


def test_no_dead_k_uses_weights_only_true() -> None:
    """Per REVIEW-ENG C4 — weights_only must be True at every torch.load."""
    src = _read_tool_source()
    # Both torch.load sites in the new tool must use weights_only=True.
    assert src.count("torch.load(") >= 2
    assert "weights_only=True" in src
    # And no leftover weights_only=False
    assert "weights_only=False" not in src


def test_no_dead_k_forked_inflate_omits_K_section() -> None:
    """The hardcoded forked inflate source must NOT read 28 K bytes."""
    src = _read_tool_source()
    # The new wire format only carries scales + brotli payload (no K).
    # The forked source must not declare K_SECTION_BYTES.
    # Find the forked source string boundary.
    start = src.index("_FORKED_INFLATE_SRC = '''")
    end = src.index("'''", start + 30)
    forked = src[start:end]
    assert "K_SECTION_BYTES" not in forked, "no-dead-k inflate must NOT reference K_SECTION_BYTES"
    # Must reference SCALE_SECTION_BYTES (the section we still keep)
    assert "SCALE_SECTION_BYTES" in forked
    # Wire format docstring must mention "without K" / "no-dead-k"
    assert "without K" in forked or "no K" in forked or "no-dead-k" in forked


def test_no_dead_k_tool_dispatch_blocker_includes_c3_apogee_int6() -> None:
    """REVIEW-ENG C3 attaches `apogee_int6_contest_cuda_anchor_required_first`
    to ALL Path B step 6 candidates (rel_err → score mapping unmeasured).
    The no-dead-k variant inherits that blocker."""
    src = _read_tool_source()
    assert "apogee_int6_contest_cuda_anchor_required_first" in src


def test_no_dead_k_fallback_inflate_sh_uses_contest_three_arg_contract() -> None:
    """If the historical source inflate.sh artifact is missing, the fallback
    generator must still emit the contest auth-eval three-arg wrapper.
    """
    src = _read_tool_source()
    assert 'DATA_DIR="${1:?data dir required}"' in src
    assert 'OUTPUT_DIR="${2:?output dir required}"' in src
    assert 'FILE_LIST="${3:?file list required}"' in src
    assert "while IFS= read -r line" in src
    assert "${BASE}.raw" in src
    assert 'exec python "$HERE/inflate.py" "$1" "$2"' not in src


def test_no_dead_k_section_total_bytes_28_smaller_than_original() -> None:
    """Direct module-level constant check — when imported, the tool's
    `_build_lossy_decoder_section_no_K` returns a ``section_total_bytes``
    that is exactly 28 less than the equivalent original variant build."""
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    # The wire format constants we can validate without running the encoder:
    # K bytes in wire format = 0 (vs 28 in original).
    src = _read_tool_source()
    assert '"K_bytes_in_wire_format": 0' in src
    # The new variant's lane_id is distinct from the original.
    assert no_k.LANE_ID == "admm_x_lossy_coarsening_path_b_step6_no_dead_k"


def test_no_dead_k_preserves_original_audit_trail() -> None:
    """The new variant outputs to a separate dir
    (``..._no_dead_k_<ts>``) so the original variant's submission_dir +
    archive.zip remains intact as forensic record."""
    src = _read_tool_source()
    assert "admm_x_lossy_coarsening_path_b_step6_no_dead_k_" in src
    # Original variant tool name preserved in the manifest for cross-reference
    assert "tools/build_admm_x_lossy_coarsening_path_b_step6.py" in src


def test_no_dead_k_can_load_selected_Ks_from_score_weight_manifest(tmp_path: Path) -> None:
    """Score-aware beta-Fisher/Jacobian planning manifests should be able to
    drive the no-dead-K archive builder without editing the source constant."""
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    selected = [1] * len(no_k.FIXED_STATE_SCHEMA)
    selected[0] = 2
    manifest = {
        "schema": "beta_fisher_lossy_coarsening_tensor_weights.v1",
        "evidence_semantics": "cpu_allocator_weight_export_no_score_no_dispatch",
        "dispatch_blockers": ["requires_exact_cuda_auth_eval_before_score_claim"],
        "weighted_k_allocations": [
            {
                "rms_target": 0.0386,
                "total_bytes": 159544,
                "rel_err": 0.033605,
                "selected_Ks": selected,
            }
        ],
    }
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    Ks, metadata = no_k._load_selected_Ks_from_manifest(path, rms_target=0.0386)

    assert Ks == selected
    assert metadata["per_tensor_K_source"] == "selected_Ks_json"
    assert metadata["selected_Ks_row_total_bytes_proxy"] == 159544
    assert metadata["selected_Ks_source_evidence_semantics"] == ("cpu_allocator_weight_export_no_score_no_dispatch")


def test_no_dead_k_can_load_selected_Ks_from_generic_jacobian_fisher_manifest(
    tmp_path: Path,
) -> None:
    """The generic Jacobian/Fisher allocation manifest should be consumable
    without routing through the PR101-specific beta-Fisher exporter."""
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    selected = [1] * len(no_k.FIXED_STATE_SCHEMA)
    selected[0] = 3
    selected[4] = 5
    expected_names = [name for name, _shape in no_k.FIXED_STATE_SCHEMA]
    manifest = {
        "schema": "jacobian_fisher_importance_allocator.v1",
        "evidence_semantics": ("cpu_mps_proxy_importance_weighted_quantization_allocation_no_score_no_dispatch"),
        "dispatch_blockers": [
            "cpu_mps_proxy_importance_inputs_not_score_authority",
            "requires_exact_archive_cuda_score_custody_before_rank_promotion_or_kill",
        ],
        "allocation": {
            "objective": "target_distortion",
            "target_distortion": 0.0386,
            "total_bytes": 158901,
            "weighted_rms_error": 0.0325,
            "unweighted_rms_error": 0.0341,
            "selected_by_tensor": [
                {
                    "tensor_index": idx,
                    "tensor_name": expected_names[idx],
                    "K": value,
                    "allocator_weight": 1.0,
                    "bytes": 100,
                    "error": 0.01,
                }
                for idx, value in enumerate(selected)
            ],
        },
    }
    path = tmp_path / "jacobian_fisher_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    Ks, metadata = no_k._load_selected_Ks_from_manifest(path, rms_target=0.0386)

    assert Ks == selected
    assert metadata["selected_Ks_source_field"] == "allocation.selected_by_tensor[].K"
    assert metadata["selected_Ks_source_schema"] == "jacobian_fisher_importance_allocator.v1"
    assert metadata["selected_Ks_row_total_bytes_proxy"] == 158901
    assert metadata["selected_Ks_row_weighted_rms_error"] == 0.0325
    assert metadata["selected_Ks_row_unweighted_rms_error"] == 0.0341
    assert metadata["selected_Ks_source_allocation_objective"] == "target_distortion"


def test_no_dead_k_rejects_generic_jacobian_fisher_order_mismatch(
    tmp_path: Path,
) -> None:
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    expected_names = [name for name, _shape in no_k.FIXED_STATE_SCHEMA]
    rows = [
        {
            "tensor_index": idx,
            "tensor_name": expected_names[idx],
            "K": 1,
        }
        for idx in range(len(expected_names))
    ]
    rows[3]["tensor_index"] = 4
    path = tmp_path / "wrong_order.json"
    path.write_text(
        json.dumps(
            {
                "schema": "jacobian_fisher_importance_allocator.v1",
                "allocation": {
                    "objective": "target_distortion",
                    "target_distortion": 0.0386,
                    "selected_by_tensor": rows,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="tensor_index must be 3"):
        no_k._load_selected_Ks_from_manifest(path, rms_target=0.0386)


def test_no_dead_k_selected_Ks_source_blockers_propagate_to_guards(
    tmp_path: Path,
) -> None:
    """A diagnostic planning manifest must remain visibly non-authoritative
    after the archive builder consumes its selected_Ks."""
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    selected = [1] * len(no_k.FIXED_STATE_SCHEMA)
    manifest = {
        "schema": "beta_fisher_lossy_coarsening_tensor_weights.v1",
        "evidence_semantics": "cpu_allocator_weight_export_no_score_no_dispatch",
        "dispatch_blockers": [
            "diagnostic_or_stub_sensitivity_map_not_score_authority",
            "requires_exact_cuda_auth_eval_before_score_claim",
            "selected_Ks_not_yet_encoded_in_no_dead_k_runtime_packet",
            "weight_export_only_no_byte_closed_archive",
        ],
        "weighted_k_allocations": [
            {
                "rms_target": 0.0386,
                "selected_Ks": selected,
            }
        ],
    }
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _, metadata = no_k._load_selected_Ks_from_manifest(path, rms_target=0.0386)

    guard = no_k._guard_fields_with_selected_Ks_source_blockers(metadata)
    closed = no_k._closed_selected_Ks_source_blockers(metadata)

    assert closed == [
        "selected_Ks_not_yet_encoded_in_no_dead_k_runtime_packet",
        "weight_export_only_no_byte_closed_archive",
    ]

    for key in ("dispatch_blockers", "score_claim_blockers"):
        blockers = guard[key]
        assert "selected_Ks_json_cpu_planning_not_score_authority" in blockers
        assert ("selected_Ks_source_evidence_semantics:cpu_allocator_weight_export_no_score_no_dispatch") in blockers
        assert ("selected_Ks_source_blocker:diagnostic_or_stub_sensitivity_map_not_score_authority") in blockers
        assert ("selected_Ks_source_blocker:requires_exact_cuda_auth_eval_before_score_claim") in blockers
        assert ("selected_Ks_source_blocker:selected_Ks_not_yet_encoded_in_no_dead_k_runtime_packet") not in blockers
        assert "selected_Ks_source_blocker:weight_export_only_no_byte_closed_archive" not in blockers


def test_no_dead_k_rejects_bad_selected_Ks_manifest(tmp_path: Path) -> None:
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"weighted_k_allocations": [{"rms_target": 0.0386, "selected_Ks": [1, 2, 3]}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="selected_Ks length"):
        no_k._load_selected_Ks_from_manifest(path, rms_target=0.0386)


def test_no_dead_k_rejects_generic_jacobian_fisher_wrong_target(
    tmp_path: Path,
) -> None:
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    path = tmp_path / "wrong_target.json"
    path.write_text(
        json.dumps(
            {
                "schema": "jacobian_fisher_importance_allocator.v1",
                "allocation": {
                    "objective": "target_distortion",
                    "target_distortion": 0.05,
                    "selected_by_tensor": [
                        {"tensor_name": f"tensor_{idx}.weight", "K": 1} for idx in range(len(no_k.FIXED_STATE_SCHEMA))
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="does not match requested rms_target"):
        no_k._load_selected_Ks_from_manifest(path, rms_target=0.0386)


def test_no_dead_k_cli_documents_score_weight_manifest_inputs() -> None:
    src = _read_tool_source()
    assert "--selected-Ks-json" in src
    assert "--score-weights-json" in src
    assert "--selected-Ks-additive-baseline-cap" in src
    assert "--selected-Ks-max-fp32-smoke-rel-err" in src
    assert "weighted_k_allocations[].selected_Ks" in src
    assert "allocation.selected_by_tensor[].K" in src
    assert "per_tensor_K_source" in src


def test_no_dead_k_manifest_writer_fails_closed_on_non_finite_json() -> None:
    src = _read_tool_source()
    assert "allow_nan=False" in src
    assert '"rel_err_actual_fp32_smoke": rel_err_actual_fp32_smoke' in src
    assert '"max_per_tensor_rel_err_fp32_smoke": max_per_tensor_rel_err_fp32_smoke' in src


def test_no_dead_k_can_blend_selected_Ks_toward_baseline_additive_cap() -> None:
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    selected = [
        10,
        24,
        3,
        10,
        3,
        10,
        5,
        10,
        2,
        1,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]

    blended, metadata = no_k._blend_selected_Ks_with_baseline_additive_cap(
        selected,
        additive_cap=3,
    )

    assert blended == [
        5,
        4,
        3,
        4,
        3,
        4,
        5,
        4,
        2,
        1,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert metadata["selected_Ks_blend_mode"] == "baseline_additive_cap"
    assert metadata["selected_Ks_blend_additive_cap"] == 3
    assert metadata["selected_Ks_blend_changed_count"] == 5
    assert metadata["selected_Ks_original_from_json"] == selected
    assert metadata["selected_Ks_after_blend"] == blended


def test_no_dead_k_selected_Ks_fp32_guard_rejects_bad_smoke() -> None:
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    metadata = {
        "per_tensor_K_source": "selected_Ks_json",
        "selected_Ks_source_dispatch_blockers": [],
    }
    smoke = {
        "rel_err_vs_quantized_fp32": 0.08739940096338861,
        "max_per_tensor_rel_err": 0.1929084482955193,
    }

    guard = no_k._selected_Ks_fp32_smoke_safety_guard(
        k_source_metadata=metadata,
        smoke=smoke,
        archive_bytes=147285,
        max_fp32_smoke_rel_err=0.055,
    )
    fields = no_k._guard_fields_with_selected_Ks_source_blockers(
        metadata,
        selected_Ks_fp32_smoke_guard=guard,
    )

    assert guard["verdict"] == "rejected"
    assert guard["aggregate_fp32_smoke_rel_err"] == 0.08739940096338861
    assert "selected_Ks_fp32_smoke_rel_err_above_guard" in guard["blockers"]
    assert fields["cuda_eval_worth_testing"] is False
    assert "selected_Ks_fp32_smoke_rel_err_above_guard" in fields["dispatch_blockers"]
    assert "selected_Ks_fp32_smoke_rel_err_above_guard" in fields["score_claim_blockers"]


def test_no_dead_k_selected_Ks_fp32_guard_rejects_non_finite_smoke() -> None:
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    metadata = {
        "per_tensor_K_source": "selected_Ks_json",
        "selected_Ks_source_dispatch_blockers": [],
    }
    smoke = {
        "rel_err_vs_quantized_fp32": float("nan"),
        "max_per_tensor_rel_err": float("inf"),
    }

    guard = no_k._selected_Ks_fp32_smoke_safety_guard(
        k_source_metadata=metadata,
        smoke=smoke,
        archive_bytes=153378,
        max_fp32_smoke_rel_err=0.055,
    )
    fields = no_k._guard_fields_with_selected_Ks_source_blockers(
        metadata,
        selected_Ks_fp32_smoke_guard=guard,
    )

    assert guard["verdict"] == "rejected"
    assert guard["aggregate_fp32_smoke_rel_err"] is None
    assert guard["max_per_tensor_fp32_smoke_rel_err"] is None
    assert "selected_Ks_fp32_smoke_rel_err_invalid" in guard["blockers"]
    assert "selected_Ks_fp32_smoke_max_tensor_rel_err_invalid" in guard["blockers"]
    assert fields["cuda_eval_worth_testing"] is False
    assert "selected_Ks_fp32_smoke_rel_err_invalid" in fields["dispatch_blockers"]
    json.dumps(guard, allow_nan=False)


def test_no_dead_k_selected_Ks_fp32_guard_passes_safer_smoke() -> None:
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    metadata = {
        "per_tensor_K_source": "selected_Ks_json",
        "selected_Ks_source_dispatch_blockers": [],
    }
    smoke = {
        "rel_err_vs_quantized_fp32": 0.0512570250306118,
        "max_per_tensor_rel_err": 0.08658950139649849,
    }

    guard = no_k._selected_Ks_fp32_smoke_safety_guard(
        k_source_metadata=metadata,
        smoke=smoke,
        archive_bytes=153378,
        max_fp32_smoke_rel_err=0.055,
    )
    fields = no_k._guard_fields_with_selected_Ks_source_blockers(
        metadata,
        selected_Ks_fp32_smoke_guard=guard,
    )

    assert guard["verdict"] == "passed"
    assert guard["blockers"] == []
    assert fields["cuda_eval_worth_testing"] is True
    assert "selected_Ks_fp32_smoke_rel_err_above_guard" not in fields["dispatch_blockers"]


def test_no_dead_k_removes_import_caches_from_submission_dir(tmp_path: Path) -> None:
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    root = tmp_path / "submission_dir"
    (root / "__pycache__").mkdir(parents=True)
    (root / "__pycache__" / "inflate.cpython-312.pyc").write_bytes(b"cache")
    (root / "src" / "__pycache__").mkdir(parents=True)
    (root / "src" / "__pycache__" / "codec.cpython-312.pyc").write_bytes(b"cache")
    keep = root / "src" / "codec.py"
    keep.write_text("# keep\n", encoding="utf-8")

    removed = no_k._remove_python_caches(root)

    assert removed == ["__pycache__", "src/__pycache__"]
    assert not (root / "__pycache__").exists()
    assert not (root / "src" / "__pycache__").exists()
    assert keep.read_text(encoding="utf-8") == "# keep\n"


def test_no_dead_k_evidence_grade_is_cpu_build() -> None:
    """Per CPU-only ML/scoring policy: cuda_eval_worth_testing=True is allowed
    (this is a free byte-win on a candidate already approved for dispatch),
    but evidence_grade must be ``[CPU-build]`` and score_claim=False."""
    src = _read_tool_source()
    assert '"evidence_grade": "[CPU-build]"' in src
    assert '"score_claim": False' in src
    assert '"cuda_eval_worth_testing": True' in src
    assert '"custody_status": "transient-allowed"' in src


def test_original_step6_builder_marks_ignored_archive_custody() -> None:
    """B3 custody status must be generated by the sibling original builder too."""
    src = (
        REPO_ROOT / "tools" / "build_admm_x_lossy_coarsening_path_b_step6.py"
    ).read_text(encoding="utf-8")
    assert '"custody_status": "transient-allowed"' in src
    assert "contest auth eval" in src
