from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.cross_paradigm_atoms import (
    COMMON_ATOM_FIELDS,
    CrossParadigmAtomError,
    atoms_from_categorical_openpilot_mask_plan,
    atoms_from_foveation_plan,
    atoms_from_hnerv_rate_recode_profile,
    atoms_from_lapose_plan,
    atoms_from_wr01_wavelet_plan,
    build_cross_paradigm_atom_ledger,
)
from tac.optimization.meta_lagrangian_allocator import build_atom_ledger
from tac.repo_io import sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_all_required_paradigm_adapters_emit_common_fields(tmp_path: Path) -> None:
    archive_manifest = _archive_manifest(tmp_path)
    manifest_sha = sha256_file(archive_manifest)
    hnerv = atoms_from_hnerv_rate_recode_profile(
        {
            "source_label": "PR106x",
            "source_archive_sha256": "a" * 64,
            "variants": [
                {
                    "variant": "brotli-q11",
                    "byte_delta_vs_source_section": -151,
                    "raw_equal": True,
                    "archive_manifest_path": archive_manifest.as_posix(),
                    "archive_manifest_sha256": manifest_sha,
                }
            ],
        }
    )
    wr01 = atoms_from_wr01_wavelet_plan(
        {
            "source_label": "PR106x",
            "source_archive_sha256": "b" * 64,
            "evidence_grade": "empirical_wavelet_plan",
            "sections": [
                {
                    "section_name": "latents_and_sidecar_brotli",
                    "source_section_sha256": "c" * 64,
                    "atoms": [
                        {
                            "raw_offset": 8,
                            "level": 0,
                            "coefficient_index": 4,
                            "estimated_wire_bytes": 7,
                            "expected_seg_dist_delta": -0.00002,
                            "confidence": 0.3,
                        }
                    ],
                }
            ],
        }
    )
    categorical = atoms_from_categorical_openpilot_mask_plan(
        {
            "evidence_grade": "empirical_mask_planning",
            "openpilot_priors": ["ego_curvature"],
            "top_atoms": [
                {
                    "atom_id": "row7",
                    "atom_family": "row_run",
                    "cost_model": {"estimated_charged_bytes": 6},
                    "lagrangian": {"estimated_marginal_score_saved_proxy": 0.02},
                    "pair_indices": [3],
                    "class_ids": [1],
                }
            ],
        }
    )
    lapose = atoms_from_lapose_plan(
        {
            "evidence_grade": "diagnostic_cuda_global_response_allocated",
            "atoms": [
                {
                    "pair_index": 75,
                    "latent_action": [1.0, 0.0],
                    "byte_delta": 12,
                    "expected_seg_dist_delta": -0.00003,
                    "expected_pose_dist_delta": -0.000004,
                    "confidence": 0.6,
                    "hard_pair_support": [75],
                    "openpilot_priors": ["yaw_rate"],
                    "allocation_inference": True,
                }
            ],
        }
    )
    foveation = atoms_from_foveation_plan(
        {
            "ok": True,
            "evidence_grade": "empirical_payload_custody",
            "wire_format": "HFV1",
            "bytes": 48,
            "sha256": "d" * 64,
            "source_archive_sha256": "e" * 64,
            "dispatch_blockers": ["exact_cuda_auth_eval_required_before_score_claim"],
        }
    )

    atoms = [*hnerv, *wr01, *categorical, *lapose, *foveation]
    assert {atom["paradigm"] for atom in atoms} == {
        "hnerv_rate_recode",
        "wr01_wavelet",
        "categorical_openpilot_mask",
        "lapose_planning",
        "foveation_planning",
    }
    for atom in atoms:
        for field in COMMON_ATOM_FIELDS:
            assert field in atom, f"{atom['atom_id']} missing {field}"
        assert atom["score_claim"] is False
        assert atom["ready_for_exact_eval_dispatch"] is False
        assert atom["dispatch_blockers"]
        assert atom["interaction_assumptions"]

    direct_ledger = build_atom_ledger(atoms, base_pose_dist=0.01, source="fixture")
    assert direct_ledger["atom_count"] == len(atoms)
    assert direct_ledger["ready_for_exact_eval_dispatch"] is False


def test_cross_paradigm_ledger_preserves_adapter_assumptions_and_blockers() -> None:
    atoms = [
        *atoms_from_hnerv_rate_recode_profile(
            {
                "source_label": "PR106x",
                "variants": [
                    {"variant": "raw-equal", "byte_delta_vs_source_section": -9, "raw_equal": True}
                ],
            }
        ),
        *atoms_from_lapose_plan(
            {
                "atoms": [
                    {
                        "pair_index": 5,
                        "byte_delta": 20,
                        "expected_seg_dist_delta": -0.0002,
                        "confidence": 0.5,
                        "allocation_inference": True,
                    }
                ],
                "dispatch_blockers": ["planning_only_lapose_motion_atoms"],
            }
        ),
    ]

    ledger = build_cross_paradigm_atom_ledger(atoms, base_pose_dist=0.01, source="fixture")

    assert ledger["tool"] == "tac.optimization.cross_paradigm_atoms.build_cross_paradigm_atom_ledger"
    assert ledger["allocator_tool"] == "tac.optimization.meta_lagrangian_allocator.build_atom_ledger"
    assert ledger["paradigm_counts"] == {"hnerv_rate_recode": 1, "lapose_planning": 1}
    hnerv = next(row for row in ledger["rows"] if row["paradigm"] == "hnerv_rate_recode")
    assert "rate_only_raw_equal_required" in hnerv["interaction_assumptions"]
    assert "hnerv_rate_recode_requires_raw_equal_proof" in hnerv["adapter_dispatch_blockers"]
    lapose = next(row for row in ledger["rows"] if row["paradigm"] == "lapose_planning")
    assert lapose["rankable"] is False
    assert "allocated_global_response_not_rankable" in lapose["dispatch_blockers"]
    assert "planning_only_lapose_motion_atoms" in lapose["source_dispatch_blockers"]
    assert ledger["dispatch_attempted"] is False


def test_adapters_are_deterministic_under_input_order_changes() -> None:
    forward = atoms_from_categorical_openpilot_mask_plan(
        {
            "top_atoms": [
                {"atom_id": "b", "atom_family": "row_run", "byte_delta": 2},
                {"atom_id": "a", "atom_family": "row_run", "byte_delta": 1},
            ]
        }
    )
    reverse = atoms_from_categorical_openpilot_mask_plan(
        {
            "top_atoms": [
                {"atom_id": "a", "atom_family": "row_run", "byte_delta": 1},
                {"atom_id": "b", "atom_family": "row_run", "byte_delta": 2},
            ]
        }
    )

    assert forward == reverse
    assert [atom["atom_id"] for atom in forward] == sorted(atom["atom_id"] for atom in forward)


def test_invalid_confidence_and_duplicate_ids_fail_closed() -> None:
    with pytest.raises(CrossParadigmAtomError, match="confidence"):
        atoms_from_foveation_plan({"ok": True, "bytes": 1, "confidence": 2.0})
    atoms = atoms_from_hnerv_rate_recode_profile(
        {
            "source_label": "PR",
            "variants": [
                {"variant": "same", "byte_delta": -1, "raw_equal": True},
                {"variant": "same", "byte_delta": -2, "raw_equal": True},
            ],
        }
    )
    with pytest.raises(CrossParadigmAtomError, match="duplicate atom_id"):
        build_cross_paradigm_atom_ledger(atoms, base_pose_dist=0.01, source="fixture")


def test_build_cross_paradigm_atom_ledger_cli(tmp_path: Path) -> None:
    hnerv_profile = tmp_path / "hnerv.json"
    wr01_plan = tmp_path / "wr01.json"
    out = tmp_path / "ledger.json"
    hnerv_profile.write_text(
        json.dumps(
            {
                "source_label": "PR106x",
                "variants": [
                    {"variant": "good", "byte_delta_vs_source_section": -151, "raw_equal": True}
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    wr01_plan.write_text(
        json.dumps(
            {
                "source_label": "PR106x",
                "sections": [
                    {
                        "section_name": "latents_and_sidecar_brotli",
                        "atoms": [{"raw_offset": 0, "level": 0, "coefficient_index": 0, "estimated_wire_bytes": 5}],
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_cross_paradigm_atom_ledger.py"),
            "--hnerv-rate-recode-profile",
            str(hnerv_profile),
            "--wr01-wavelet-plan",
            str(wr01_plan),
            "--base-pose-dist",
            "0.01",
            "--source",
            "fixture",
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["atom_count"] == 2
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["tool_run_manifest"]["input_files"][0]["path"] == hnerv_profile.as_posix()


def _archive_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "archive-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "archive": {"bytes": 123, "sha256": "1" * 64},
                "score_claim": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
