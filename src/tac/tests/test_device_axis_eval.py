from __future__ import annotations

from tac.device_axis_eval import (
    cuda_minus_cpu_gaps,
    is_contest_cuda_equivalent_gpu,
    mechanism_class_from_pair,
    raw_output_pairing,
    score_terms,
)


def test_score_terms_and_cuda_minus_cpu_gaps_are_formula_exact() -> None:
    cpu = score_terms(pose=3.46e-5, seg=5.7599e-4, archive_bytes=178_981)
    cuda = score_terms(pose=1.7347e-4, seg=6.7565e-4, archive_bytes=178_981)

    gaps = cuda_minus_cpu_gaps(cpu_terms=cpu, cuda_terms=cuda)

    assert cuda["score"] > cpu["score"]
    assert gaps["score"] == cuda["score"] - cpu["score"]
    assert gaps["pose_term"] == cuda["pose_term"] - cpu["pose_term"]
    assert gaps["seg_term"] == cuda["seg_term"] - cpu["seg_term"]
    assert gaps["rate_term"] == 0.0
    assert gaps["pose_gap_share"] is not None


def test_contest_cuda_equivalent_gpu_accepts_supported_models() -> None:
    assert is_contest_cuda_equivalent_gpu(gpu_model="", gpu_t4_match=True) is True
    assert (
        is_contest_cuda_equivalent_gpu(
            gpu_model="NVIDIA A100-SXM4-40GB",
            gpu_t4_match=False,
        )
        is True
    )
    assert (
        is_contest_cuda_equivalent_gpu(
            gpu_model="NVIDIA GeForce RTX 4090",
            gpu_t4_match=False,
        )
        is True
    )
    assert (
        is_contest_cuda_equivalent_gpu(
            gpu_model="NVIDIA L40S",
            gpu_t4_match=False,
        )
        is True
    )


def test_contest_cuda_equivalent_gpu_rejects_unknown_models_and_string_flags() -> None:
    assert (
        is_contest_cuda_equivalent_gpu(
            gpu_model="NVIDIA RTX 3090",
            gpu_t4_match=False,
        )
        is False
    )
    assert (
        is_contest_cuda_equivalent_gpu(
            gpu_model="",
            gpu_t4_match="true",  # type: ignore[arg-type]
        )
        is False
    )


def test_raw_output_pairing_classifies_same_different_partial_and_missing() -> None:
    cpu = {"aggregate_sha256": "1" * 64}
    cuda_same = {"aggregate_sha256": "1" * 64}
    cuda_different = {"aggregate_sha256": "2" * 64}

    assert raw_output_pairing(cpu_raw=cpu, cuda_raw=cuda_same) == {
        "cpu": cpu,
        "cuda": cuda_same,
        "same_inflated_output_aggregate_sha256": True,
        "raw_output_pairing_status": "same_inflated_outputs",
        "mechanism_blockers": [],
        "mechanism_analysis_complete": True,
    }
    assert raw_output_pairing(cpu_raw=cpu, cuda_raw=cuda_different)[
        "raw_output_pairing_status"
    ] == "different_inflated_outputs"
    assert raw_output_pairing(cpu_raw=cpu, cuda_raw=None)["mechanism_blockers"] == [
        "partial_raw_output_manifest"
    ]
    assert raw_output_pairing(cpu_raw=None, cuda_raw=None)["mechanism_blockers"] == [
        "raw_output_manifest_missing"
    ]


def test_mechanism_class_requires_the_narrowest_supported_claim() -> None:
    assert (
        mechanism_class_from_pair(
            same_inflated_output_aggregate_sha256=True,
            same_runtime_tree_sha256=True,
            same_archive_sha256=True,
        )
        == "same_raw_outputs_scorer_or_loader_drift"
    )
    assert (
        mechanism_class_from_pair(
            same_inflated_output_aggregate_sha256=False,
            same_runtime_tree_sha256=True,
            same_archive_sha256=True,
        )
        == "different_raw_outputs_runtime_or_inflate_drift"
    )
    assert (
        mechanism_class_from_pair(
            same_inflated_output_aggregate_sha256=None,
            same_runtime_tree_sha256=True,
            same_archive_sha256=True,
        )
        == "same_archive_runtime_raw_outputs_unmeasured"
    )
    assert (
        mechanism_class_from_pair(
            same_inflated_output_aggregate_sha256=None,
            same_runtime_tree_sha256=False,
            same_archive_sha256=True,
        )
        == "custody_incomplete"
    )
