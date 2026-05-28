# SPDX-License-Identifier: MIT
# FAKE_LANE_OK_FILE: continual-learning posterior tests reference example
# lane_ids (lane_12_v2 etc.) as illustrative source-rho keys, not
# registered-lane references. Per Check #126 file-level waiver semantics.
"""Tests for tac.continual_learning — posterior orchestration layer.

Covers:
  - posterior_update accepts authoritative anchors only
  - macOS substrate refused unless explicitly allowed
  - duplicate archive_sha256 refused (idempotence)
  - per-track Welford running mean + std
  - source-rho posterior validation
  - load/save round-trip
  - schema mismatch rejection
  - history truncation at 500 entries
  - posterior_query helpers return defaults when track/class missing
  - harvest_anchors_from_iter bulk path
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.continual_learning import (
    AUTHORITATIVE_TAGS,
    CONTINUAL_LEARNING_SCHEMA_VERSION,
    NON_PROMOTABLE_TAGS,
    ContestResult,
    ContinualLearningPosterior,
    PerTrackPosterior,
    SourceRhoPosterior,
    contest_result_from_auth_eval_payload,
    harvest_anchors_from_iter,
    load_posterior,
    posterior_query_source_rho,
    posterior_query_track_correction,
    posterior_update,
    posterior_update_locked_from_auth_eval_json,
    save_posterior,
)

# ── Fixture helpers ────────────────────────────────────────────────────────


def _make_authoritative_result(
    *,
    sha: str = "a" * 64,
    score: float = 0.19284,
    arch: str = "pr106_hnerv_cluster",
    axis: str = "cpu",
    tag: str = "[contest-CPU GHA Linux x86_64]",
    substrate: str = "linux_x86_64_gha_cpu",
    track_obs: dict[str, float] | None = None,
    rho: float | None = None,
    metadata: dict[str, object] | None = None,
) -> ContestResult:
    return ContestResult(
        axis=axis,
        hardware_substrate=substrate,
        architecture_class=arch,
        score_value=score,
        evidence_tag=tag,
        archive_sha256=sha,
        archive_bytes=178262,
        cuda_pose=None,
        cuda_seg=None,
        cpu_pose=None,
        cpu_seg=None,
        source_rho_estimate=rho,
        track_correction_observations=track_obs or {},
        metadata=metadata or {},
    )


# ── Acceptance + refusal policies ──────────────────────────────────────────


def test_authoritative_anchor_accepted():
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result()
    update = posterior_update(posterior, result)
    assert update.accepted is True
    assert update.refusal_reason == ""
    assert posterior.accepted_anchor_count == 1


def test_non_authoritative_evidence_tag_refused():
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result(tag="[advisory only]")
    update = posterior_update(posterior, result)
    assert update.accepted is False
    assert "non-authoritative" in update.refusal_reason
    assert posterior.accepted_anchor_count == 0
    assert posterior.refused_anchor_count == 1


def test_macos_substrate_refused_by_default():
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result(
        substrate="macos_arm64_m5max",
        tag="[contest-CPU GHA Linux x86_64]",  # tag is fine, substrate isn't
    )
    update = posterior_update(posterior, result)
    assert update.accepted is False
    # Custody validator (codex round-2 HIGH 2) flags the macOS substrate.
    assert "macos_arm64_m5max" in update.refusal_reason
    assert "macOS" in update.refusal_reason or "1:1 contest-compliant" in update.refusal_reason


def test_macos_substrate_refused_even_when_legacy_override_requested():
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result(
        substrate="macos_arm64_m5max",
        tag="[contest-CPU GHA Linux x86_64]",
    )
    update = posterior_update(posterior, result, forbid_macos_promotion=False)
    assert update.accepted is False
    assert "macOS" in update.refusal_reason


def test_duplicate_sha256_refused():
    posterior = ContinualLearningPosterior()
    result1 = _make_authoritative_result(sha="b" * 64)
    posterior_update(posterior, result1)
    result2 = _make_authoritative_result(sha="b" * 64)  # same sha
    update = posterior_update(posterior, result2)
    assert update.accepted is False
    assert "duplicate" in update.refusal_reason
    assert posterior.accepted_anchor_count == 1
    assert posterior.refused_anchor_count == 1


def test_same_archive_axis_distinct_runtime_tree_accepted():
    posterior = ContinualLearningPosterior()
    sha = "r" * 64
    first = _make_authoritative_result(
        sha=sha,
        metadata={"runtime_tree_sha256": "1" * 64, "n_samples": 600},
    )
    second = _make_authoritative_result(
        sha=sha,
        score=0.1920089730474962,
        metadata={"runtime_tree_sha256": "2" * 64, "n_samples": 600},
    )

    assert posterior_update(posterior, first).accepted is True
    update = posterior_update(posterior, second)

    assert update.accepted is True
    assert posterior.accepted_anchor_count == 2
    assert {row["runtime_tree_sha256"] for row in posterior.accepted_anchor_history} == {
        "1" * 64,
        "2" * 64,
    }


def test_same_archive_axis_same_runtime_tree_refused_as_duplicate():
    posterior = ContinualLearningPosterior()
    sha = "s" * 64
    metadata = {"runtime_tree_sha256": "3" * 64, "n_samples": 600}

    first = posterior_update(
        posterior,
        _make_authoritative_result(sha=sha, metadata=metadata),
    )
    second = posterior_update(
        posterior,
        _make_authoritative_result(sha=sha, score=0.1920089, metadata=metadata),
    )

    assert first.accepted is True
    assert second.accepted is False
    assert "duplicate exact anchor identity" in second.refusal_reason
    assert posterior.accepted_anchor_count == 1


def test_same_archive_axis_legacy_distinct_score_accepted():
    posterior = ContinualLearningPosterior()
    sha = "t" * 64

    assert posterior_update(
        posterior,
        _make_authoritative_result(sha=sha, score=0.1920099730474962),
    ).accepted is True
    update = posterior_update(
        posterior,
        _make_authoritative_result(sha=sha, score=0.1920089730474962),
    )

    assert update.accepted is True
    assert posterior.accepted_anchor_count == 2


def test_duplicate_sha_different_axis_accepted():
    """Same archive sha on cuda + cpu axes → both accepted (different axis)."""
    posterior = ContinualLearningPosterior()
    posterior_update(posterior, _make_authoritative_result(sha="c" * 64, axis="cuda", tag="[contest-CUDA]", substrate="linux_x86_64_t4"))
    update = posterior_update(posterior, _make_authoritative_result(sha="c" * 64, axis="cpu"))
    assert update.accepted is True
    assert posterior.accepted_anchor_count == 2


def test_non_promotable_tags_explicit_set():
    """Sanity: NON_PROMOTABLE_TAGS includes the canonical macOS / MPS tags."""
    assert "[macOS-CPU advisory only]" in NON_PROMOTABLE_TAGS
    assert "[MPS-PROXY]" in NON_PROMOTABLE_TAGS
    assert "[contest-CPU]" in AUTHORITATIVE_TAGS  # short form accepted


def test_invalid_result_type_raises():
    posterior = ContinualLearningPosterior()
    with pytest.raises(TypeError, match="ContestResult"):
        posterior_update(posterior, {"not": "a result"})


def test_nan_score_value_refused_to_prevent_posterior_corruption():
    """R1-Quantizr: NaN injection would corrupt running mean forever."""
    import math
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result()
    result.score_value = math.nan
    update = posterior_update(posterior, result)
    assert update.accepted is False
    assert "non-finite" in update.refusal_reason
    assert posterior.refused_anchor_count == 1


def test_nan_track_correction_refused():
    """R1-Quantizr: NaN in track_correction_observations would poison Welford."""
    import math
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result(track_obs={"t7": math.nan})
    update = posterior_update(posterior, result)
    assert update.accepted is False
    assert "track_correction_observations" in update.refusal_reason


def test_inf_source_rho_refused():
    """R1-Quantizr: inf source_rho would poison SourceRhoPosterior."""
    import math
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result(rho=math.inf)
    update = posterior_update(posterior, result)
    assert update.accepted is False


def test_nan_cuda_pose_refused():
    """R1-Quantizr: NaN in any optional CUDA/CPU component is refused."""
    import math
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result()
    result.cuda_pose = math.nan
    update = posterior_update(posterior, result)
    assert update.accepted is False
    assert "cuda_pose" in update.refusal_reason


# ── Codex round-2 HIGH 2: custody validator (tag + axis + hardware) ────────


def test_custody_axis_mismatch_cuda_tag_with_cpu_axis_refused():
    """[contest-CUDA] tag with axis='cpu' is a custody mismatch."""
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result(
        tag="[contest-CUDA]",
        axis="cpu",  # MISMATCH
        substrate="linux_x86_64_t4",
    )
    update = posterior_update(posterior, result)
    assert update.accepted is False
    assert "axis mismatch" in update.refusal_reason


def test_custody_axis_mismatch_cpu_tag_with_cuda_axis_refused():
    """[contest-CPU] tag with axis='cuda' is a custody mismatch."""
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result(
        tag="[contest-CPU GHA Linux x86_64]",
        axis="cuda",  # MISMATCH
        substrate="linux_x86_64_gha_cpu",
    )
    update = posterior_update(posterior, result)
    assert update.accepted is False
    assert "axis mismatch" in update.refusal_reason


def test_custody_short_form_contest_cpu_requires_approved_linux_x86_64_substrate():
    """The short-form `[contest-CPU]` accepts approved Linux x86_64 hosts only."""
    posterior = ContinualLearningPosterior()
    # Short-form tag on a generic Linux box — refused.
    result = _make_authoritative_result(
        tag="[contest-CPU]",
        axis="cpu",
        substrate="linux_x86_64_random_vm",  # not an approved contest-CPU substrate
    )
    update = posterior_update(posterior, result)
    assert update.accepted is False
    assert "1:1 contest-compliant" in update.refusal_reason


def test_custody_contest_cuda_on_unknown_substrate_refused():
    """[contest-CUDA] on an unrecognized GPU substrate refused."""
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result(
        tag="[contest-CUDA]",
        axis="cuda",
        substrate="some_cloud_unknown_gpu",  # not in TAG_HARDWARE_REQUIREMENT
    )
    update = posterior_update(posterior, result)
    assert update.accepted is False
    assert "1:1 contest-compliant" in update.refusal_reason


def test_custody_contest_cuda_on_t4_accepted():
    """T4 is in the canonical CUDA-substrate set."""
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result(
        tag="[contest-CUDA]",
        axis="cuda",
        substrate="linux_x86_64_t4",
    )
    update = posterior_update(posterior, result)
    assert update.accepted is True


def test_custody_validate_returns_ok_reason_pair():
    """validate_custody returns (bool, str) — caller can use both."""
    result = _make_authoritative_result()
    ok, reason = result.validate_custody()
    assert ok is True
    assert reason == ""


def test_custody_validate_returns_failure_reason():
    result = _make_authoritative_result(
        tag="[contest-CUDA]", axis="cpu", substrate="linux_x86_64_t4"
    )
    ok, reason = result.validate_custody()
    assert ok is False
    assert "axis mismatch" in reason


# ── Codex round-2 MEDIUM: parallel-safe locked update ──────────────────────


def test_posterior_update_locked_basic_path(tmp_path):
    """posterior_update_locked acquires lock + updates posterior on disk."""
    from tac.continual_learning import (
        load_posterior,
        posterior_update_locked,
    )

    posterior_path = tmp_path / "posterior.json"
    lock_path = tmp_path / "posterior.lock"
    result = _make_authoritative_result(sha="lock_test_" + "a" * 54)

    update = posterior_update_locked(
        result, posterior_path=posterior_path, lock_path=lock_path
    )
    assert update.accepted is True

    # Reload and confirm posterior persisted.
    loaded = load_posterior(posterior_path)
    assert loaded.accepted_anchor_count == 1


def test_posterior_update_locked_serializes_concurrent_writes(tmp_path):
    """Two threads contending on the lock both succeed (no lost update).

    Uses threads rather than multiprocessing to avoid pytest+multiprocessing
    pickling issues. fcntl.flock is per-process so in-process threads will
    both grab the lock sequentially — but the test still validates
    reload-inside-lock semantics."""
    import threading

    from tac.continual_learning import (
        ContestResult,
        load_posterior,
        posterior_update_locked,
    )

    posterior_path = tmp_path / "posterior.json"
    lock_path = tmp_path / "posterior.lock"

    results: list = []
    results_lock = threading.Lock()

    def worker(sha_seed: int):
        result = ContestResult(
            axis="cpu",
            hardware_substrate="linux_x86_64_gha_cpu",
            architecture_class="pr106_hnerv_cluster",
            score_value=0.19284,
            evidence_tag="[contest-CPU GHA Linux x86_64]",
            archive_sha256=f"{sha_seed:064x}",
            archive_bytes=178262,
        )
        update = posterior_update_locked(
            result,
            posterior_path=posterior_path,
            lock_path=lock_path,
        )
        with results_lock:
            results.append(update)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    accepted = sum(1 for r in results if r.accepted)
    assert accepted == 8, f"expected 8 accepts under lock; got {accepted}"

    # Critical: the on-disk posterior must contain ALL 8 anchors (no lost
    # update). Without the reload-inside-lock semantics, two threads could
    # both load the same stale posterior + both update + the second's save
    # would clobber the first's.
    loaded = load_posterior(posterior_path)
    assert loaded.accepted_anchor_count == 8


def test_save_posterior_uses_unique_tmp_file(tmp_path):
    """save_posterior writes to a UNIQUE .tmp.<uuid> path (codex round-2 MEDIUM).
    Verified by checking no fixed `.tmp` path lingers after a successful save."""
    posterior_path = tmp_path / "posterior.json"
    posterior = ContinualLearningPosterior()
    posterior_update(posterior, _make_authoritative_result(sha="m" * 64))
    save_posterior(posterior, posterior_path)
    # No fixed-suffix .tmp file lingers (the prior bug class).
    assert not (posterior_path.with_suffix(".json.tmp")).exists()
    # The UUID-suffixed tmp file is also cleaned up after rename.
    leftover_tmps = list(tmp_path.glob("*.tmp.*"))
    assert leftover_tmps == [], f"leftover tmp files: {leftover_tmps}"


# ── Per-track Welford posterior ────────────────────────────────────────────


def test_per_track_welford_running_mean():
    p = PerTrackPosterior(track_kind="t7_fisher_rao")
    for x in [1.0, 2.0, 3.0]:
        p.update(x)
    assert p.mean_correction == pytest.approx(2.0)
    assert p.n_observations == 3


def test_per_track_welford_variance():
    p = PerTrackPosterior(track_kind="t7_fisher_rao")
    for x in [1.0, 2.0, 3.0]:
        p.update(x)
    # Sample variance of [1, 2, 3] = 1.0 (n-1 denominator)
    assert p.variance() == pytest.approx(1.0)
    assert p.std() == pytest.approx(1.0)


def test_per_track_welford_single_observation_zero_variance():
    p = PerTrackPosterior(track_kind="t7_fisher_rao")
    p.update(5.0)
    assert p.variance() == 0.0
    assert p.std() == 0.0
    assert p.mean_correction == 5.0


def test_track_correction_observations_accumulate():
    posterior = ContinualLearningPosterior()
    posterior_update(posterior, _make_authoritative_result(sha="d" * 64, track_obs={"t7_fisher_rao": 1.05}))
    posterior_update(posterior, _make_authoritative_result(sha="e" * 64, track_obs={"t7_fisher_rao": 0.95}))
    mean, n = posterior_query_track_correction(posterior, "t7_fisher_rao")
    assert n == 2
    assert mean == pytest.approx(1.0)


def test_track_query_default_when_missing():
    posterior = ContinualLearningPosterior()
    mean, n = posterior_query_track_correction(posterior, "t99_does_not_exist", default=0.5)
    assert mean == 0.5
    assert n == 0


# ── Source-rho posterior (T13 consumer) ────────────────────────────────────


def test_source_rho_posterior_update_validates_range():
    p = SourceRhoPosterior(architecture_class="pr106_hnerv")
    p.update(0.5)
    assert p.mean_rho == 0.5
    with pytest.raises(ValueError, match="must be in"):
        p.update(1.5)
    with pytest.raises(ValueError, match="must be in"):
        p.update(-1.0)


def test_source_rho_query_default_when_missing():
    posterior = ContinualLearningPosterior()
    mean, n = posterior_query_source_rho(posterior, "lane_12_v2", default=0.0)
    assert mean == 0.0
    assert n == 0


def test_source_rho_accumulates_across_anchors():
    posterior = ContinualLearningPosterior()
    posterior_update(posterior, _make_authoritative_result(sha="f" * 64, rho=0.4))
    posterior_update(posterior, _make_authoritative_result(sha="g" * 64, rho=0.6))
    mean, n = posterior_query_source_rho(posterior, "pr106_hnerv_cluster")
    assert n == 2
    assert mean == pytest.approx(0.5)


# ── CUDA-CPU drift hand-off ────────────────────────────────────────────────


def test_cuda_cpu_drift_signal_present_when_all_components_supplied():
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result()
    result.cuda_pose = 1.5e-4
    result.cuda_seg = 6.7e-4
    result.cpu_pose = 7.5e-4
    result.cpu_seg = 7.85e-4
    update = posterior_update(posterior, result)
    assert update.cuda_cpu_drift_signal_present is True
    assert update.cuda_cpu_drift_updated is False
    assert any("cuda_cpu" in n for n in update.notes)


def test_cuda_cpu_drift_not_signaled_when_components_missing():
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result()
    update = posterior_update(posterior, result)
    assert update.cuda_cpu_drift_signal_present is False
    assert update.cuda_cpu_drift_updated is False


# ── Auth-eval JSON bridge ─────────────────────────────────────────────────


def _auth_eval_payload(
    *,
    axis: str = "contest_cuda",
    device: str = "cuda",
    hardware: str = "Modal Tesla T4 Linux x86_64",
    gpu_model: str = "Tesla T4",
    gpu_t4_match: bool | None = True,
    platform_system: str = "Linux",
    platform_machine: str = "x86_64",
    evidence_grade: str = "contest-CUDA",
    lane_tag: str = "[contest-CUDA]",
    score_claim_valid: bool = True,
) -> dict:
    provenance = {
        "archive_sha256": "1" * 64,
        "archive_size_bytes": 186822,
        "device": device,
        "hardware": hardware,
        "platform_system": platform_system,
        "platform_machine": platform_machine,
    }
    if gpu_model:
        provenance["gpu_model"] = gpu_model
    if gpu_t4_match is not None:
        provenance["gpu_t4_match"] = gpu_t4_match
    return {
        "score_axis": axis,
        "device": device,
        "canonical_score": 0.20664588545741508,
        "avg_segnet_dist": 0.00064260,
        "avg_posenet_dist": 0.00003236,
        "archive_size_bytes": 186822,
        "n_samples": 600,
        "evidence_grade": evidence_grade,
        "lane_tag": lane_tag,
        "score_claim_valid": score_claim_valid,
        "promotion_eligible": score_claim_valid,
        "provenance": provenance,
    }


def test_contest_result_from_auth_eval_payload_cuda_t4_promotable_shape():
    result = contest_result_from_auth_eval_payload(
        _auth_eval_payload(),
        architecture_class="pr106_latent_sidecar",
        source_path="eval.json",
    )
    assert result.axis == "cuda"
    assert result.hardware_substrate == "linux_x86_64_t4"
    assert result.evidence_tag == "[contest-CUDA]"
    assert result.archive_sha256 == "1" * 64
    assert result.cuda_pose == pytest.approx(0.00003236)
    assert result.cpu_pose is None
    assert result.validate_custody_verdict().accepted is True
    assert result.metadata["source_path"] == "eval.json"


def test_contest_result_from_auth_eval_payload_cpu_gha_accepted_shape():
    payload = _auth_eval_payload(
        axis="contest_cpu",
        device="cpu",
        hardware="github-actions-ubuntu-latest-x86_64",
        gpu_model="",
        gpu_t4_match=None,
        evidence_grade="contest-CPU",
        lane_tag="[contest-CPU GHA Linux x86_64]",
        score_claim_valid=False,
    )
    result = contest_result_from_auth_eval_payload(
        payload,
        architecture_class="pr106_latent_sidecar",
    )
    assert result.axis == "cpu"
    assert result.hardware_substrate == "linux_x86_64_gha_cpu"
    assert result.evidence_tag == "[contest-CPU GHA Linux x86_64]"
    assert result.cpu_pose == pytest.approx(0.00003236)
    assert result.validate_custody_verdict().accepted is True


def test_contest_result_from_auth_eval_payload_modal_cpu_accepted_shape():
    payload = _auth_eval_payload(
        axis="contest_cpu",
        device="cpu",
        hardware="Modal CPU Linux x86_64",
        gpu_model="",
        gpu_t4_match=None,
        evidence_grade="contest-CPU",
        lane_tag="[contest-CPU]",
        score_claim_valid=False,
    )
    result = contest_result_from_auth_eval_payload(
        payload,
        architecture_class="pr106_latent_sidecar",
    )
    verdict = result.validate_custody_verdict()
    assert result.hardware_substrate == "linux_x86_64_modal_cpu"
    assert result.evidence_tag == "[contest-CPU]"
    assert verdict.accepted is True
    assert verdict.refused_class is None


def test_contest_result_from_auth_eval_payload_preserves_runtime_identity():
    payload = _auth_eval_payload(
        axis="contest_cpu",
        device="cpu",
        hardware="Modal CPU Linux x86_64",
        gpu_model="",
        gpu_t4_match=None,
        evidence_grade="contest-CPU",
        lane_tag="[contest-CPU]",
        score_claim_valid=False,
    )
    payload["canonical_score_source"] = "score_recomputed_from_components"
    payload["provenance"]["inflate_script_sha256"] = "a" * 64
    payload["provenance"]["pact_commit"] = "pact-sha"
    payload["provenance"]["upstream_commit"] = "upstream-sha"
    payload["provenance"]["inflate_runtime_manifest"] = {
        "runtime_tree_sha256": "b" * 64,
        "runtime_content_tree_sha256": "c" * 64,
    }
    payload["provenance"]["inflated_output_manifest"] = {
        "sha256": "d" * 64,
        "payload": {"aggregate_sha256": "e" * 64},
    }

    result = contest_result_from_auth_eval_payload(
        payload,
        architecture_class="pr106_latent_sidecar",
    )

    assert result.metadata["runtime_tree_sha256"] == "b" * 64
    assert result.metadata["runtime_content_tree_sha256"] == "c" * 64
    assert result.metadata["inflate_script_sha256"] == "a" * 64
    assert result.metadata["inflated_output_manifest_sha256"] == "d" * 64
    assert result.metadata["inflated_output_aggregate_sha256"] == "e" * 64
    assert result.metadata["canonical_score_source"] == "score_recomputed_from_components"


def test_contest_result_from_auth_eval_payload_accepts_wrapper_expected_archive_fields():
    payload = _auth_eval_payload()
    payload.pop("archive_size_bytes")
    payload["expected_archive_sha256"] = "b" * 64
    payload["expected_archive_size_bytes"] = 186700
    payload["provenance"].pop("archive_sha256")
    payload["provenance"].pop("archive_size_bytes")

    result = contest_result_from_auth_eval_payload(
        payload,
        architecture_class="pr106_latent_sidecar",
    )

    assert result.archive_sha256 == "b" * 64
    assert result.archive_bytes == 186700


def test_posterior_update_locked_from_auth_eval_json_updates_once(tmp_path: Path):
    eval_path = tmp_path / "contest_auth_eval.json"
    eval_path.write_text(json.dumps(_auth_eval_payload()), encoding="utf-8")
    posterior_path = tmp_path / "posterior.json"
    lock_path = tmp_path / "posterior.lock"

    first = posterior_update_locked_from_auth_eval_json(
        eval_path,
        architecture_class="pr106_latent_sidecar",
        posterior_path=posterior_path,
        lock_path=lock_path,
    )
    second = posterior_update_locked_from_auth_eval_json(
        eval_path,
        architecture_class="pr106_latent_sidecar",
        posterior_path=posterior_path,
        lock_path=lock_path,
    )

    assert first.accepted is True
    assert second.accepted is False
    assert "duplicate" in second.refusal_reason
    posterior = load_posterior(posterior_path)
    assert posterior.accepted_anchor_count == 1
    assert posterior.refused_anchor_count == 1


# ── Persistence ────────────────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path: Path):
    p = tmp_path / "posterior.json"
    posterior = ContinualLearningPosterior()
    posterior_update(posterior, _make_authoritative_result(sha="h" * 64, track_obs={"t7": 1.05}, rho=0.5))
    posterior_update(posterior, _make_authoritative_result(sha="i" * 64, track_obs={"t7": 0.95}, rho=0.6))
    save_posterior(posterior, p)

    loaded = load_posterior(p)
    assert loaded.accepted_anchor_count == 2
    assert loaded.schema == CONTINUAL_LEARNING_SCHEMA_VERSION
    mean, n = posterior_query_track_correction(loaded, "t7")
    assert n == 2
    assert mean == pytest.approx(1.0)


def test_load_returns_empty_when_missing(tmp_path: Path):
    p = tmp_path / "missing.json"
    posterior = load_posterior(p)
    assert posterior.accepted_anchor_count == 0
    assert posterior.refused_anchor_count == 0


def test_load_rejects_schema_mismatch(tmp_path: Path):
    p = tmp_path / "stale.json"
    p.write_text(json.dumps({"schema": "stale_v0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        load_posterior(p)


def test_save_atomic_via_tmp_swap(tmp_path: Path):
    """save_posterior writes to a .tmp file then renames atomically."""
    p = tmp_path / "atomic.json"
    posterior = ContinualLearningPosterior()
    save_posterior(posterior, p)
    assert p.is_file()
    assert not (p.with_suffix(p.suffix + ".tmp")).exists()


# ── History truncation + bulk harvest ──────────────────────────────────────


def test_history_truncated_to_500_entries():
    posterior = ContinualLearningPosterior()
    for i in range(550):
        sha = f"{i:064x}"
        posterior_update(posterior, _make_authoritative_result(sha=sha))
    assert len(posterior.accepted_anchor_history) == 500
    assert posterior.accepted_anchor_count == 550


def test_harvest_anchors_from_iter_processes_all():
    posterior = ContinualLearningPosterior()
    results = [
        _make_authoritative_result(sha=f"{i:064x}")
        for i in range(5)
    ]
    updates = harvest_anchors_from_iter(posterior, results)
    assert len(updates) == 5
    assert all(u.accepted for u in updates)
    assert posterior.accepted_anchor_count == 5


def test_harvest_mixed_authoritative_and_advisory():
    posterior = ContinualLearningPosterior()
    results = [
        _make_authoritative_result(sha="x" * 64),
        _make_authoritative_result(sha="y" * 64, tag="[advisory only]"),
        _make_authoritative_result(sha="z" * 64),
    ]
    updates = harvest_anchors_from_iter(posterior, results)
    assert sum(u.accepted for u in updates) == 2
    assert sum(not u.accepted for u in updates) == 1
    assert posterior.accepted_anchor_count == 2
    assert posterior.refused_anchor_count == 1


# ── CustodyVerdict typed taxonomy tests (codex round-2 HIGH 2 directive) ───
#
# Per the round-2 directive: "One test per refused_class".
# CustodyVerdict.refused_class ∈ {None, missing_metadata, advisory_grade,
# macos_substrate, cpu_tag_non_gha_linux, cuda_tag_unknown_substrate,
# tag_axis_mismatch}.


def test_custody_verdict_accepted_authoritative_cpu_gha():
    """Acceptance: CPU GHA Linux x86_64 with cpu axis → accepted, refused_class=None."""
    from tac.continual_learning import CustodyVerdict

    result = _make_authoritative_result(
        axis="cpu",
        tag="[contest-CPU GHA Linux x86_64]",
        substrate="linux_x86_64_gha_cpu",
    )
    verdict = result.validate_custody_verdict()
    assert isinstance(verdict, CustodyVerdict)
    assert verdict.accepted is True
    assert verdict.refused_class is None
    assert verdict.reason == ""


def test_custody_verdict_accepted_authoritative_cuda_t4():
    """Acceptance: CUDA tag with linux_x86_64_t4 substrate → accepted."""
    from tac.continual_learning import CustodyVerdict

    result = _make_authoritative_result(
        axis="cuda",
        tag="[contest-CUDA]",
        substrate="linux_x86_64_t4",
    )
    verdict = result.validate_custody_verdict()
    assert isinstance(verdict, CustodyVerdict)
    assert verdict.accepted is True
    assert verdict.refused_class is None


def test_custody_verdict_accepted_authoritative_cuda_4090():
    """Acceptance: CUDA tag with linux_x86_64_4090 substrate → accepted."""
    result = _make_authoritative_result(
        axis="cuda",
        tag="[contest-CUDA]",
        substrate="linux_x86_64_4090",
    )
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is True
    assert verdict.refused_class is None


def test_custody_verdict_refused_class_missing_metadata():
    """Empty axis / hardware_substrate → refused_class=missing_metadata."""
    result = ContestResult(
        axis="",
        hardware_substrate="linux_x86_64_gha_cpu",
        architecture_class="pr106",
        score_value=0.19,
        evidence_tag="[contest-CPU]",
        archive_sha256="a" * 64,
        archive_bytes=178262,
    )
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is False
    assert verdict.refused_class == "missing_metadata"


def test_custody_verdict_refused_class_missing_metadata_blank_substrate():
    """Whitespace-only hardware_substrate → refused_class=missing_metadata."""
    result = ContestResult(
        axis="cpu",
        hardware_substrate="   ",
        architecture_class="pr106",
        score_value=0.19,
        evidence_tag="[contest-CPU]",
        archive_sha256="a" * 64,
        archive_bytes=178262,
    )
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is False
    assert verdict.refused_class == "missing_metadata"


def test_custody_verdict_refused_class_advisory_grade():
    """[advisory only] tag → refused_class=advisory_grade."""
    result = _make_authoritative_result(tag="[advisory only]")
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is False
    assert verdict.refused_class == "advisory_grade"


def test_custody_verdict_refused_class_advisory_grade_mps():
    """[MPS-PROXY] tag → refused_class=advisory_grade."""
    result = _make_authoritative_result(tag="[MPS-PROXY]")
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is False
    assert verdict.refused_class == "advisory_grade"


def test_custody_verdict_refused_class_macos_substrate_via_substrate_field():
    """macOS substrate with authoritative tag → refused_class=macos_substrate."""
    result = _make_authoritative_result(
        substrate="macos_arm64_m5max",
        tag="[contest-CPU GHA Linux x86_64]",
    )
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is False
    assert verdict.refused_class == "macos_substrate"


def test_custody_verdict_refused_class_macos_substrate_via_macos_tag():
    """[macOS-CPU advisory only] tag → refused_class=macos_substrate."""
    result = _make_authoritative_result(tag="[macOS-CPU advisory only]")
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is False
    assert verdict.refused_class == "macos_substrate"


def test_custody_verdict_refused_class_macos_calibrated_tag():
    """[macOS-CPU calibrated] tag is also refused under macos_substrate."""
    result = _make_authoritative_result(tag="[macOS-CPU calibrated]")
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is False
    assert verdict.refused_class == "macos_substrate"


def test_custody_verdict_refused_class_cpu_tag_non_gha_linux():
    """[contest-CPU] with unknown Linux CPU substrate keeps the legacy refused class."""
    result = _make_authoritative_result(
        axis="cpu",
        tag="[contest-CPU]",
        substrate="linux_x86_64_random_vm",
    )
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is False
    assert verdict.refused_class == "cpu_tag_non_gha_linux"


def test_custody_verdict_refused_class_cuda_tag_unknown_substrate():
    """[contest-CUDA] with unknown CUDA substrate → cuda_tag_unknown_substrate."""
    result = _make_authoritative_result(
        axis="cuda",
        tag="[contest-CUDA]",
        substrate="linux_x86_64_unknown_gpu_v2",
    )
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is False
    assert verdict.refused_class == "cuda_tag_unknown_substrate"


def test_custody_verdict_refused_class_tag_axis_mismatch_cpu_tag_cuda_axis():
    """CPU tag with axis='cuda' → refused_class=tag_axis_mismatch."""
    result = _make_authoritative_result(
        axis="cuda",  # mismatch — CPU tag requires axis="cpu"
        tag="[contest-CPU GHA Linux x86_64]",
        substrate="linux_x86_64_gha_cpu",
    )
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is False
    assert verdict.refused_class == "tag_axis_mismatch"


def test_custody_verdict_refused_class_tag_axis_mismatch_cuda_tag_cpu_axis():
    """CUDA tag with axis='cpu' → refused_class=tag_axis_mismatch."""
    result = _make_authoritative_result(
        axis="cpu",  # mismatch — CUDA tag requires axis="cuda"
        tag="[contest-CUDA]",
        substrate="linux_x86_64_t4",
    )
    verdict = result.validate_custody_verdict()
    assert verdict.accepted is False
    assert verdict.refused_class == "tag_axis_mismatch"


def test_custody_verdict_is_frozen_dataclass():
    """CustodyVerdict is frozen so the verdict cannot be mutated post-validation."""
    import dataclasses

    from tac.continual_learning import CustodyVerdict

    v = CustodyVerdict(accepted=True, reason="", refused_class=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.accepted = False  # type: ignore[misc]


def test_custody_validate_back_compat_returns_tuple():
    """Back-compat: validate_custody() returns (bool, str) tuple."""
    result = _make_authoritative_result()
    ok, reason = result.validate_custody()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    assert ok is True
    assert reason == ""


def test_custody_back_compat_existing_anchors_not_retroactively_revalidated():
    """Read-side: existing accepted entries don't get retroactively re-validated.

    The custody validator only refuses NEW entries on the way IN. Once an
    anchor is in accepted_anchor_history, load_posterior does not re-run the
    validator — the historical record is preserved.
    """
    posterior = ContinualLearningPosterior()
    # Seed the history with an entry that would FAIL today's stricter validator
    # (e.g., short-form [contest-CPU] with an unknown Linux CPU substrate, which
    # gets refused_class=cpu_tag_non_gha_linux).
    posterior.accepted_anchor_history.append({
        "axis": "cpu",
        "architecture_class": "legacy_lane",
        "evidence_tag": "[contest-CPU]",
        "archive_sha256": "legacy_" + "0" * 57,
        "archive_bytes": 200_000,
        "score_value": 0.19,
        "hardware_substrate": "linux_x86_64_random_vm",
        "observed_at_utc": "2026-04-01T00:00:00+00:00",
        "track_updates": [],
        "source_rho_estimate": None,
    })
    posterior.accepted_anchor_count = 1

    # The history survives unchanged (no retroactive re-validation).
    assert posterior.accepted_anchor_count == 1
    assert len(posterior.accepted_anchor_history) == 1
    assert posterior.accepted_anchor_history[0]["hardware_substrate"] == (
        "linux_x86_64_random_vm"
    )

    # But a NEW write with the same shape IS refused going forward.
    new_result = _make_authoritative_result(
        axis="cpu",
        tag="[contest-CPU]",
        substrate="linux_x86_64_random_vm",
        sha="modern_" + "1" * 57,
    )
    update = posterior_update(posterior, new_result)
    assert update.accepted is False
    assert "non-1:1" in update.refusal_reason or "not in 1:1" in update.refusal_reason


# ── True multiprocessing tests for posterior_update_locked (MEDIUM fix) ────


def _multiproc_worker_distinct(args):
    """Worker for true cross-process lock test — must be top-level (picklable)."""
    posterior_path_str, lock_path_str, sha_seed = args
    from pathlib import Path as _P

    from tac.continual_learning import (
        ContestResult,
        posterior_update_locked,
    )

    result = ContestResult(
        axis="cpu",
        hardware_substrate="linux_x86_64_gha_cpu",
        architecture_class="pr106_hnerv_cluster",
        score_value=0.19284,
        evidence_tag="[contest-CPU GHA Linux x86_64]",
        archive_sha256=f"{sha_seed:064x}",
        archive_bytes=178262,
    )
    update = posterior_update_locked(
        result,
        posterior_path=_P(posterior_path_str),
        lock_path=_P(lock_path_str),
    )
    return (sha_seed, update.accepted, update.refusal_reason)


def _multiproc_worker_same_anchor(args):
    """Worker that always sends the SAME anchor — idempotence test across procs."""
    posterior_path_str, lock_path_str, idx = args
    from pathlib import Path as _P

    from tac.continual_learning import (
        ContestResult,
        posterior_update_locked,
    )

    result = ContestResult(
        axis="cpu",
        hardware_substrate="linux_x86_64_gha_cpu",
        architecture_class="pr106_hnerv_cluster",
        score_value=0.19284,
        evidence_tag="[contest-CPU GHA Linux x86_64]",
        archive_sha256="dup_anchor_" + "f" * 53,  # SAME for every worker
        archive_bytes=178262,
    )
    update = posterior_update_locked(
        result,
        posterior_path=_P(posterior_path_str),
        lock_path=_P(lock_path_str),
    )
    return (idx, update.accepted, update.refusal_reason)


def test_multiprocess_distinct_anchors_all_land(tmp_path):
    """True multiprocessing: 4 procs update DISTINCT anchors → all land."""
    import multiprocessing as mp

    posterior_path = tmp_path / "mp_posterior.json"
    lock_path = tmp_path / "mp_posterior.lock"

    # Use spawn context for portability across macOS/Linux defaults.
    ctx = mp.get_context("spawn")
    args = [
        (str(posterior_path), str(lock_path), seed) for seed in range(1, 5)
    ]
    with ctx.Pool(processes=4) as pool:
        results = pool.map(_multiproc_worker_distinct, args)

    accepted = sum(1 for (_seed, ok, _reason) in results if ok)
    assert accepted == 4, f"all 4 distinct anchors should land; got {accepted} (results={results})"

    loaded = load_posterior(posterior_path)
    assert loaded.accepted_anchor_count == 4
    seen_shas = {h["archive_sha256"] for h in loaded.accepted_anchor_history}
    assert len(seen_shas) == 4


def test_multiprocess_same_anchor_idempotent(tmp_path):
    """True multiprocessing: 4 procs update the SAME anchor → 1 accept, 3 idempotent refuse."""
    import multiprocessing as mp

    posterior_path = tmp_path / "mp_posterior_dup.json"
    lock_path = tmp_path / "mp_posterior_dup.lock"

    ctx = mp.get_context("spawn")
    args = [
        (str(posterior_path), str(lock_path), idx) for idx in range(4)
    ]
    with ctx.Pool(processes=4) as pool:
        results = pool.map(_multiproc_worker_same_anchor, args)

    accepted = sum(1 for (_idx, ok, _reason) in results if ok)
    refused = sum(1 for (_idx, ok, _reason) in results if not ok)
    assert accepted == 1, f"exactly 1 should accept; got {accepted}"
    assert refused == 3, f"3 should be idempotent-refused; got {refused}"

    # Refusal reason for the 3 must mention "duplicate".
    for _idx, ok, reason in results:
        if not ok:
            assert "duplicate" in reason

    loaded = load_posterior(posterior_path)
    assert loaded.accepted_anchor_count == 1
    assert loaded.refused_anchor_count == 3


def test_lock_released_after_exception(tmp_path):
    """The lock context manager releases the lock even when the body raises."""
    from tac.continual_learning import _posterior_lock

    lock_path = tmp_path / "raise.lock"
    raised = False
    try:
        with _posterior_lock(lock_path):
            raise RuntimeError("intentional")
    except RuntimeError:
        raised = True
    assert raised

    # A second acquisition succeeds because the prior was released.
    with _posterior_lock(lock_path):
        pass


def test_save_posterior_no_fixed_tmp_path_collision(tmp_path):
    """save_posterior tmp paths are unique enough to never collide.

    Codex round-2 MEDIUM fix: the prior single `.tmp` path could clobber
    a sibling save in flight. Verify the producer uses uuid-suffixed paths
    by checking the implementation has a uuid call at the tmp construction.
    """
    import inspect

    from tac.continual_learning import save_posterior as _save

    src = inspect.getsource(_save)
    # We allow either uuid.uuid4 or os.getpid()/time-based suffix; the key is
    # that the suffix is not just `.tmp`.
    assert (
        "uuid" in src
        or "getpid" in src
        or "monotonic_ns" in src
        or "time.time" in src
    ), "save_posterior tmp file must use unique-per-call suffix"
