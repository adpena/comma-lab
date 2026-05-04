from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_frontier_atom_ledger.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_frontier_atom_ledger_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _exact_payload(*, bytes_: int, pose: float, seg: float, sha: str) -> dict:
    module = _load_module()
    score = (
        100.0 * seg
        + math.sqrt(10.0 * pose)
        + module.RATE_SLOPE_SCORE_PER_BYTE * bytes_
    )
    return {
        "archive_size_bytes": bytes_,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": sha,
            "device": "cuda",
            "gpu_model": "Tesla T4",
            "gpu_t4_match": True,
        },
        "score_recomputed_from_components": score,
    }


def test_exact_transition_computes_rate_and_nonrate_terms(tmp_path: Path) -> None:
    module = _load_module()
    source = _write_json(
        tmp_path / "source" / "contest_auth_eval.adjudicated.json",
        _exact_payload(bytes_=1000, pose=0.004, seg=0.005, sha="a" * 64),
    )
    target = _write_json(
        tmp_path / "target" / "contest_auth_eval.adjudicated.json",
        _exact_payload(bytes_=900, pose=0.004, seg=0.0049, sha="b" * 64),
    )

    ledger = module.build_frontier_atom_ledger(
        root=tmp_path,
        exact_candidates=[("C-044", source), ("C-051", target)],
        packer_paths=[],
        top_anatomy_path=None,
    )

    transition = next(t for t in ledger["exact_transitions"] if t["relation"] == "chain_step")
    assert transition["bytes_saved"] == 100
    assert transition["delta_rate_score"] == -100 * module.RATE_SLOPE_SCORE_PER_BYTE
    assert transition["accepted_by_exact_score"] is True
    assert transition["score_saved_per_byte"] > module.RATE_SLOPE_SCORE_PER_BYTE
    candidate = ledger["exact_candidates"][1]
    assert candidate["score_recompute_abs_error"] < 1e-12


def test_active_frontier_prefers_newest_explicit_label(tmp_path: Path) -> None:
    module = _load_module()
    old = _write_json(
        tmp_path / "old" / "contest_auth_eval.adjudicated.json",
        _exact_payload(bytes_=1000, pose=0.004, seg=0.005, sha="a" * 64),
    )
    new = _write_json(
        tmp_path / "new" / "contest_auth_eval.adjudicated.json",
        _exact_payload(bytes_=900, pose=0.004, seg=0.0049, sha="b" * 64),
    )

    ledger = module.build_frontier_atom_ledger(
        root=tmp_path,
        exact_candidates=[("C-051", old), ("C-063", new)],
        packer_paths=[],
        top_anatomy_path=None,
    )

    assert ledger["active_frontier_label"] == "C-063"
    assert ledger["active_frontier_archive_sha256"] == "b" * 64


def test_packer_and_top_submission_atoms_are_deterministic(tmp_path: Path) -> None:
    module = _load_module()
    packer = _write_json(
        tmp_path / "packed" / "packed_renderer_payload_provenance.json",
        {
            "score_claim": False,
            "evidence_grade": "empirical",
            "source_archive_bytes": 1000,
            "source_archive_sha256": "a" * 64,
            "output_archive": "out/archive.zip",
            "output_archive_bytes": 800,
            "output_archive_sha256": "b" * 64,
            "savings_bytes": 200,
            "formula_only_rate_delta": -0.1,
            "payload_format": "rp2_fixed3",
            "payload_member": "p",
            "payload_raw_bytes": 700,
            "payload_compressed_bytes": 650,
            "pose_codec": "pose_qpose14_col_delta_v1",
            "header": {
                "members": [
                    {"name": "renderer.bin", "bytes": 10, "sha256": "c" * 64, "codec": "raw"},
                    {
                        "name": "optimized_poses.bin",
                        "bytes": 12,
                        "sha256": "d" * 64,
                        "codec": "pose_qpose14_col_delta_v1",
                        "lossy": True,
                        "pose_error_stats": {"rows": 600},
                    },
                ]
            },
        },
    )
    anatomy = _write_json(
        tmp_path / "top" / "archive_anatomy.json",
        {
            "schema_version": 2,
            "score_claim": False,
            "evidence_grade": "external_plus_empirical_byte_anatomy",
            "determinism": {"no_timestamps": True},
            "local_jointframegenerator_reference": {"parameter_count": 87836},
            "items": [
                {
                    "label": "pr67",
                    "pr_number": 67,
                    "archive": {"bytes": 276564, "sha256": "e" * 64},
                    "container": {
                        "member": "p",
                        "bytes": 276464,
                        "sha256": "f" * 64,
                        "segmentation": "fixed",
                        "decoded_segments": [
                            {
                                "name": "model_qzs3_br",
                                "compressed_bytes": 5,
                                "raw_bytes": 10,
                                "compressed_sha256": "1" * 64,
                                "raw_sha256": "2" * 64,
                            }
                        ],
                    },
                }
            ],
        },
    )

    ledger_a = module.build_frontier_atom_ledger(
        root=tmp_path,
        exact_candidates=[],
        packer_paths=[packer],
        top_anatomy_path=anatomy,
    )
    ledger_b = module.build_frontier_atom_ledger(
        root=tmp_path,
        exact_candidates=[],
        packer_paths=[packer],
        top_anatomy_path=anatomy,
    )

    assert ledger_a == ledger_b
    assert ledger_a["packer_atoms"][0]["member_atoms"][1]["lossy"] is True
    segment = ledger_a["top_submission_atoms"]["items"][0]["segments"][0]
    assert segment["compression_ratio"] == 0.5
    assert ledger_a["score_claim"] is False


def test_pr65_pr67_trace_atoms_rank_by_waterfill_utility(tmp_path: Path) -> None:
    module = _load_module()
    anatomy = _write_json(
        tmp_path / "top" / "archive_anatomy.json",
        {
            "schema_version": 2,
            "score_claim": False,
            "evidence_grade": "external_plus_empirical_byte_anatomy",
            "items": [
                {
                    "label": "pr67",
                    "pr_number": 67,
                    "archive": {"bytes": 1000, "sha256": "a" * 64},
                    "container": {
                        "member": "p",
                        "bytes": 900,
                        "sha256": "b" * 64,
                        "decoded_segments": [
                            {"name": "mask_obu_br", "compressed_bytes": 600, "raw_bytes": 900},
                            {"name": "pose_qp1_br", "compressed_bytes": 12, "raw_bytes": 24},
                        ],
                    },
                },
                {
                    "label": "pr65",
                    "pr_number": 65,
                    "archive": {"bytes": 1120, "sha256": "c" * 64},
                    "container": {"member": "x", "bytes": 1020, "sha256": "d" * 64},
                },
            ],
        },
    )
    trace = _write_json(
        tmp_path / "trace" / "c057_vs_pr67_recut_top120.json",
        {
            "schema_version": 1,
            "score_claim": False,
            "evidence_grade": "diagnostic_trace_comparison_same_hardware",
            "allocator_use_allowed": True,
            "hardware_statuses": ["same_hardware_identity"],
            "candidate": {"label": "c057", "n_samples": 6},
            "references": [
                {
                    "reference": {"label": "pr67"},
                    "top_excess_combined_samples": [
                        {
                            "pair_index": 2,
                            "frame_indices": [4, 5],
                            "score_combined_excess_first_order": 0.0009,
                            "score_pose_excess_first_order_at_candidate": 0.0001,
                            "score_seg_excess_exact": 0.0008,
                        }
                    ],
                    "top_excess_pose_samples": [
                        {
                            "pair_index": 1,
                            "frame_indices": [2, 3],
                            "score_combined_excess_first_order": 0.0006,
                            "score_pose_excess_first_order_at_candidate": 0.0006,
                            "score_seg_excess_exact": 0.0,
                        }
                    ],
                    "top_excess_seg_samples": [
                        {
                            "pair_index": 3,
                            "frame_indices": [6, 7],
                            "score_combined_excess_first_order": 0.0007,
                            "score_pose_excess_first_order_at_candidate": -0.0001,
                            "score_seg_excess_exact": 0.0008,
                        }
                    ],
                    "pair_deltas": [],
                }
            ],
        },
    )

    ledger = module.build_frontier_atom_ledger(
        root=tmp_path,
        exact_candidates=[],
        packer_paths=[],
        top_anatomy_path=anatomy,
        trace_comparison_paths=[trace],
    )

    table = ledger["atom_allocation_table"]
    atoms = table["ranked_atoms"]
    assert [atom["rank"] for atom in atoms] == [1, 2, 3]
    assert atoms[0]["family"] == "pose"
    assert atoms[0]["charged_bytes"] == 2
    assert atoms[0]["waterfill_positive_ev"] is True
    assert atoms[1]["family"] == "postprocess"
    assert atoms[1]["charged_bytes"] == 120
    assert atoms[2]["family"] == "mask"
    assert atoms[2]["charged_bytes"] == 100
    assert atoms[2]["interaction_risk"] == "medium"
    assert table["byte_models"]["c057_vs_pr67_recut_top120:vs:pr67"][
        "pr65_minus_pr67_archive_bytes"
    ] == 120
    opportunity_ids = {
        atom["atom_id"] for atom in table["full_pipeline_opportunity_atoms"]
    }
    assert "pipeline:pr65_postprocess_residual_table" in opportunity_ids
    assert "pipeline:qzs3_renderer_quant_groups" in opportunity_ids
    assert "pipeline:rl_bandit_multipass_selector" in opportunity_ids
    gates = {gate["gate"] for gate in table["exact_eval_stack_gate_recommendations"]}
    assert {"canonical_cuda_auth_eval", "component_antagonism"} <= gates
    assert table["score_claim"] is False
    assert table["allocator_use_allowed"] is True


def test_untrusted_trace_comparison_does_not_rank_atoms(tmp_path: Path) -> None:
    module = _load_module()
    trace = _write_json(
        tmp_path / "trace" / "c063_t4_vs_pr67_h100.json",
        {
            "schema_version": 1,
            "score_claim": False,
            "evidence_grade": "diagnostic_trace_comparison_hardware_untrusted",
            "allocator_use_allowed": False,
            "hardware_statuses": ["hardware_mismatch"],
            "candidate": {"label": "c063", "n_samples": 1},
            "references": [
                {
                    "reference": {"label": "pr67"},
                    "top_excess_pose_samples": [
                        {
                            "pair_index": 0,
                            "frame_indices": [0, 1],
                            "score_pose_excess_first_order_at_candidate": 1.0,
                            "score_seg_excess_exact": 0.0,
                            "score_combined_excess_first_order": 1.0,
                        }
                    ],
                    "top_excess_combined_samples": [],
                    "top_excess_seg_samples": [],
                    "pair_deltas": [],
                }
            ],
        },
    )

    ledger = module.build_frontier_atom_ledger(
        root=tmp_path,
        exact_candidates=[],
        packer_paths=[],
        top_anatomy_path=None,
        trace_comparison_paths=[trace],
    )

    table = ledger["atom_allocation_table"]
    assert table["allocator_use_allowed"] is False
    assert table["ranked_atoms"] == []
    assert table["allocator_blocked_untrusted_inputs"][0]["artifact"].endswith(
        "c063_t4_vs_pr67_h100.json"
    )
