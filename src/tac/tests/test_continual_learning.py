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
    harvest_anchors_from_iter,
    load_posterior,
    posterior_query_source_rho,
    posterior_query_track_correction,
    posterior_update,
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
    assert "macOS" in update.refusal_reason


def test_macos_substrate_accepted_when_explicitly_allowed():
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result(
        substrate="macos_arm64_m5max",
        tag="[contest-CPU GHA Linux x86_64]",
    )
    update = posterior_update(posterior, result, forbid_macos_promotion=False)
    assert update.accepted is True


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
    assert update.cuda_cpu_drift_updated is True
    assert any("cuda_cpu" in n for n in update.notes)


def test_cuda_cpu_drift_not_signaled_when_components_missing():
    posterior = ContinualLearningPosterior()
    result = _make_authoritative_result()
    update = posterior_update(posterior, result)
    assert update.cuda_cpu_drift_updated is False


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
