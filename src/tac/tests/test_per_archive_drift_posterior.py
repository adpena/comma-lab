"""Tests for tac.per_archive_drift_posterior — Bayesian per-archive drift posterior.

Covers:
  - PerArchiveObservation custody validation (tag/axis/hardware)
  - macOS substrate refused unless override
  - non-finite score refused before Welford touched
  - duplicate observation refused for idempotence
  - Welford running mean + variance updates correctly
  - per-archive ε estimate = mean_cuda - mean_cpu
  - load/save round-trip
  - schema mismatch rejection
  - posterior is_consistent invariants
  - query_archive_epsilon defaults
  - harvest_observations_from_iter bulk path
  - locked update under multiprocess concurrent contention
  - empty Welford returns variance=0 / std=0
  - PerAxisWelfordStats refuses NaN before update
  - epsilon_estimate returns None when one axis untouched
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp

import pytest

from tac.per_archive_drift_posterior import (
    PER_ARCHIVE_DRIFT_SCHEMA_VERSION,
    DEFAULT_PER_ARCHIVE_LOCK_PATH,
    DEFAULT_PER_ARCHIVE_PATH,
    PerArchiveDriftPosterior,
    PerArchiveObservation,
    PerArchivePosterior,
    PerAxisWelfordStats,
    harvest_observations_from_iter,
    load_per_archive_posterior,
    per_archive_update,
    per_archive_update_locked,
    query_archive_epsilon,
    save_per_archive_posterior,
)


# ── Fixture helpers ────────────────────────────────────────────────────────


SHA_A = "a" * 64
SHA_B = "b" * 64


def _cpu_obs(*, sha: str = SHA_A, score: float = 0.196,
             ts: str = "2026-05-08T12:00:00+00:00") -> PerArchiveObservation:
    return PerArchiveObservation(
        archive_sha256=sha,
        archive_bytes=178262,
        axis="cpu",
        hardware_substrate="linux_x86_64_gha_cpu",
        evidence_tag="[contest-CPU GHA Linux x86_64]",
        score_value=score,
        cpu_pose=0.052,
        cpu_seg=0.0021,
        observed_at_utc=ts,
    )


def _cuda_obs(*, sha: str = SHA_A, score: float = 0.229, substrate: str = "linux_x86_64_t4",
              ts: str = "2026-05-08T13:00:00+00:00") -> PerArchiveObservation:
    return PerArchiveObservation(
        archive_sha256=sha,
        archive_bytes=178262,
        axis="cuda",
        hardware_substrate=substrate,
        evidence_tag="[contest-CUDA]",
        score_value=score,
        cuda_pose=0.011,
        cuda_seg=0.0024,
        observed_at_utc=ts,
    )


# ── PerArchiveObservation.validate_custody ─────────────────────────────────


def test_authoritative_cpu_observation_validated():
    obs = _cpu_obs()
    ok, reason = obs.validate_custody()
    assert ok, reason


def test_authoritative_cuda_observation_validated():
    obs = _cuda_obs()
    ok, reason = obs.validate_custody()
    assert ok, reason


def test_observation_validation_delegates_to_continual_learning(monkeypatch):
    """Per-archive custody must use the canonical ContestResult validator."""
    from tac.continual_learning import ContestResult

    seen = {}

    def fake_validate(self):
        seen["axis"] = self.axis
        seen["hardware_substrate"] = self.hardware_substrate
        seen["evidence_tag"] = self.evidence_tag
        seen["archive_sha256"] = self.archive_sha256
        return False, "delegated sentinel"

    monkeypatch.setattr(ContestResult, "validate_custody", fake_validate)

    ok, reason = _cuda_obs().validate_custody()

    assert not ok
    assert reason == "delegated sentinel"
    assert seen == {
        "axis": "cuda",
        "hardware_substrate": "linux_x86_64_t4",
        "evidence_tag": "[contest-CUDA]",
        "archive_sha256": SHA_A,
    }


def test_macos_substrate_refused():
    obs = _cpu_obs()
    obs.hardware_substrate = "macos_arm64"
    ok, reason = obs.validate_custody()
    assert not ok
    assert "macOS" in reason


def test_axis_mismatch_refused():
    obs = _cpu_obs()
    obs.axis = "cuda"  # axis claims cuda but tag is cpu
    ok, reason = obs.validate_custody()
    assert not ok
    assert "axis" in reason.lower()


def test_unknown_substrate_for_cuda_tag_refused():
    obs = _cuda_obs(substrate="linux_x86_64_unknown_gpu")
    ok, reason = obs.validate_custody()
    assert not ok


def test_advisory_tag_refused():
    obs = _cpu_obs()
    obs.evidence_tag = "[macOS-CPU advisory only]"
    ok, _ = obs.validate_custody()
    assert not ok


def test_blank_metadata_refused():
    obs = _cpu_obs()
    obs.archive_sha256 = ""
    ok, reason = obs.validate_custody()
    assert not ok
    assert "missing" in reason


# ── per_archive_update refusal policy ──────────────────────────────────────


def test_authoritative_observation_accepted():
    p = PerArchiveDriftPosterior()
    r = per_archive_update(p, _cpu_obs())
    assert r.accepted
    assert r.archive_sha256 == SHA_A
    assert p.accepted_observation_count == 1


def test_macos_observation_refused():
    p = PerArchiveDriftPosterior()
    obs = _cpu_obs()
    obs.hardware_substrate = "macos_arm64"
    r = per_archive_update(p, obs)
    assert not r.accepted
    assert p.refused_observation_count == 1


def test_macos_observation_accepted_with_override():
    p = PerArchiveDriftPosterior()
    obs = _cpu_obs()
    obs.hardware_substrate = "macos_arm64"
    r = per_archive_update(p, obs, forbid_macos_promotion=False)
    assert r.accepted
    assert "macOS substrate accepted via override" in " ".join(r.notes)


def test_non_finite_score_refused():
    p = PerArchiveDriftPosterior()
    obs = _cpu_obs(score=float("nan"))
    r = per_archive_update(p, obs)
    assert not r.accepted
    assert "non-finite" in r.refusal_reason


def test_duplicate_observation_refused_for_idempotence():
    p = PerArchiveDriftPosterior()
    obs = _cpu_obs()
    r1 = per_archive_update(p, obs)
    r2 = per_archive_update(p, obs)  # same ts → duplicate
    assert r1.accepted and not r2.accepted
    assert "duplicate" in r2.refusal_reason


def test_two_distinct_observations_both_accepted():
    p = PerArchiveDriftPosterior()
    o1 = _cpu_obs(score=0.196, ts="2026-05-08T12:00:00+00:00")
    o2 = _cpu_obs(score=0.197, ts="2026-05-09T12:00:00+00:00")
    r1 = per_archive_update(p, o1)
    r2 = per_archive_update(p, o2)
    assert r1.accepted and r2.accepted
    pa = p.per_archive[SHA_A]
    assert pa.cpu_axis.n == 2


def test_type_error_on_wrong_input():
    p = PerArchiveDriftPosterior()
    with pytest.raises(TypeError):
        per_archive_update(p, "not an observation")


# ── Welford correctness ────────────────────────────────────────────────────


def test_welford_running_mean():
    s = PerAxisWelfordStats()
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        s.update(v, "linux_x86_64_t4")
    assert s.n == 5
    assert s.mean == pytest.approx(3.0)
    assert s.std() == pytest.approx(math.sqrt(2.5))


def test_welford_refuses_nan():
    s = PerAxisWelfordStats()
    with pytest.raises(ValueError):
        s.update(float("nan"), "linux_x86_64_t4")
    assert s.n == 0  # state untouched


def test_welford_refuses_inf():
    s = PerAxisWelfordStats()
    with pytest.raises(ValueError):
        s.update(float("inf"), "linux_x86_64_t4")
    assert s.n == 0


def test_welford_variance_zero_on_single_observation():
    s = PerAxisWelfordStats()
    s.update(0.5, "linux_x86_64_t4")
    assert s.variance() == 0.0
    assert s.std() == 0.0


def test_welford_zero_observations_returns_zero():
    s = PerAxisWelfordStats()
    assert s.variance() == 0.0
    assert s.std() == 0.0


# ── Per-archive epsilon estimate ───────────────────────────────────────────


def test_epsilon_none_when_one_axis_untouched():
    pa = PerArchivePosterior(archive_sha256=SHA_A)
    pa.cuda_axis.update(0.229, "linux_x86_64_t4")
    assert pa.epsilon_estimate() is None  # no cpu


def test_epsilon_correct_when_both_axes_present():
    pa = PerArchivePosterior(archive_sha256=SHA_A)
    pa.cuda_axis.update(0.229, "linux_x86_64_t4")
    pa.cpu_axis.update(0.196, "linux_x86_64_gha_cpu")
    eps = pa.epsilon_estimate()
    assert eps == pytest.approx(0.229 - 0.196)


def test_epsilon_uncertainty_independence_band():
    pa = PerArchivePosterior(archive_sha256=SHA_A)
    pa.cuda_axis.update(0.229, "linux_x86_64_t4")
    pa.cuda_axis.update(0.230, "linux_x86_64_t4")
    pa.cpu_axis.update(0.196, "linux_x86_64_gha_cpu")
    pa.cpu_axis.update(0.197, "linux_x86_64_gha_cpu")
    u = pa.epsilon_uncertainty()
    assert u is not None and u > 0


def test_epsilon_uncertainty_none_when_too_few_observations():
    pa = PerArchivePosterior(archive_sha256=SHA_A)
    pa.cuda_axis.update(0.229, "linux_x86_64_t4")
    pa.cpu_axis.update(0.196, "linux_x86_64_gha_cpu")
    # only 1 each — variance is 0 but uncertainty should still return 0
    u = pa.epsilon_uncertainty()
    # spec: returns None when both axes have <2 observations
    assert u is None


# ── Posterior consistency ──────────────────────────────────────────────────


def test_consistent_on_empty():
    p = PerArchiveDriftPosterior()
    ok, problems = p.is_consistent()
    assert ok, problems


def test_consistent_after_real_updates():
    p = PerArchiveDriftPosterior()
    per_archive_update(p, _cpu_obs())
    per_archive_update(p, _cuda_obs())
    ok, problems = p.is_consistent()
    assert ok, problems


def test_inconsistent_negative_refused_count():
    p = PerArchiveDriftPosterior()
    p.refused_observation_count = -1
    ok, problems = p.is_consistent()
    assert not ok


def test_inconsistent_short_sha_key():
    p = PerArchiveDriftPosterior()
    p.per_archive["short"] = PerArchivePosterior(archive_sha256="short")
    ok, problems = p.is_consistent()
    assert not ok


# ── Load / save round-trip ─────────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path):
    out = tmp_path / "state.json"
    p = PerArchiveDriftPosterior()
    per_archive_update(p, _cpu_obs())
    per_archive_update(p, _cuda_obs())
    save_per_archive_posterior(p, out)

    p2 = load_per_archive_posterior(out)
    assert p2.accepted_observation_count == 2
    assert SHA_A in p2.per_archive
    assert p2.per_archive[SHA_A].cpu_axis.n == 1
    assert p2.per_archive[SHA_A].cuda_axis.n == 1


def test_load_returns_empty_when_missing(tmp_path):
    out = tmp_path / "missing.json"
    p = load_per_archive_posterior(out)
    assert p.accepted_observation_count == 0
    assert p.per_archive == {}


def test_load_refuses_schema_mismatch(tmp_path):
    out = tmp_path / "bad_schema.json"
    out.write_text(json.dumps({"schema": "wrong_version"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        load_per_archive_posterior(out)


def test_save_uses_unique_tmp(tmp_path):
    out = tmp_path / "state.json"
    p1 = PerArchiveDriftPosterior()
    per_archive_update(p1, _cpu_obs())
    save_per_archive_posterior(p1, out)
    # Second save should also succeed atomically.
    p2 = PerArchiveDriftPosterior()
    per_archive_update(p2, _cpu_obs(score=0.197, ts="2026-05-09T12:00:00+00:00"))
    save_per_archive_posterior(p2, out)
    assert out.is_file()


# ── Query helpers ──────────────────────────────────────────────────────────


def test_query_returns_none_for_unknown_archive():
    p = PerArchiveDriftPosterior()
    eps, u, n = query_archive_epsilon(p, SHA_A)
    assert eps is None and u is None and n == 0


def test_query_returns_epsilon_for_known_archive():
    p = PerArchiveDriftPosterior()
    per_archive_update(p, _cpu_obs(score=0.196))
    per_archive_update(p, _cuda_obs(score=0.229))
    eps, u, n = query_archive_epsilon(p, SHA_A)
    assert eps == pytest.approx(0.229 - 0.196)
    assert n == 2


def test_harvest_bulk_path():
    p = PerArchiveDriftPosterior()
    obs_list = [
        _cpu_obs(score=0.196, ts="2026-05-08T10:00:00+00:00"),
        _cuda_obs(score=0.229, ts="2026-05-08T11:00:00+00:00"),
        _cpu_obs(score=0.197, ts="2026-05-09T10:00:00+00:00"),
    ]
    results = harvest_observations_from_iter(p, obs_list)
    assert all(r.accepted for r in results)
    assert p.accepted_observation_count == 3


# ── Multiprocessing concurrent update via locked update ────────────────────


def _worker_concurrent_update(args):
    state_path, lock_path, sha, score, ts = args
    obs = PerArchiveObservation(
        archive_sha256=sha,
        archive_bytes=178262,
        axis="cpu",
        hardware_substrate="linux_x86_64_gha_cpu",
        evidence_tag="[contest-CPU GHA Linux x86_64]",
        score_value=score,
        observed_at_utc=ts,
    )
    return per_archive_update_locked(
        obs, posterior_path=state_path, lock_path=lock_path
    )


def test_locked_update_concurrent_writes_no_loss(tmp_path):
    state_path = tmp_path / "state.json"
    lock_path = tmp_path / ".lock"
    n_workers = 6
    args = [
        (state_path, lock_path, SHA_A, 0.190 + 0.001 * i,
         f"2026-05-08T12:00:0{i}+00:00")
        for i in range(n_workers)
    ]
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_worker_concurrent_update, args)

    accepted = [r for r in results if r.accepted]
    assert len(accepted) == n_workers, [r.refusal_reason for r in results]

    # Every worker's update should be visible in the final state (no lost
    # updates due to lock-contention).
    p = load_per_archive_posterior(state_path)
    assert p.accepted_observation_count == n_workers
    assert p.per_archive[SHA_A].cpu_axis.n == n_workers


def test_locked_single_worker_smoke(tmp_path):
    state_path = tmp_path / "state.json"
    lock_path = tmp_path / ".lock"
    obs = _cpu_obs()
    r = per_archive_update_locked(
        obs, posterior_path=state_path, lock_path=lock_path
    )
    assert r.accepted


# ── Default paths ──────────────────────────────────────────────────────────


def test_default_paths_under_omx_state():
    assert ".omx/state/" in str(DEFAULT_PER_ARCHIVE_PATH).replace("\\", "/")
    assert ".omx/state/" in str(DEFAULT_PER_ARCHIVE_LOCK_PATH).replace("\\", "/")


def test_schema_constant_stable():
    assert PER_ARCHIVE_DRIFT_SCHEMA_VERSION == "tac_per_archive_drift_posterior_v1"
