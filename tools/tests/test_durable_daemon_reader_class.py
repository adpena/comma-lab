"""Task #525 C2 — the DECLARED reader/control-plane admission class.

The SUM-over-RAM training admission model refused a ~2GB codex reader while the
60GiB trainer ran (measured 2026-07-17, sol_ultra_v10_durable launch). The
``--job-class reader`` lane admits small control-plane jobs on the OOM
free-floor preflight only, fail-closed on a declared envelope
(projected ≤ 4GiB + runtime --rss-cap-mb ≤ 4096), mirroring the #370
control-plane exemption pattern with the reason recorded in the registry.
"""
from __future__ import annotations

import argparse

from tools import spawn_durable_daemon as sdd


def _ns(**kw) -> argparse.Namespace:
    base = dict(job_class="training", projected_peak_gib=None, rss_cap_mb=None,
                skip_admission_gate=False)
    base.update(kw)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# _reader_class_refusal — fail-closed envelope validation
# ---------------------------------------------------------------------------

def test_training_class_is_not_reader_validated():
    assert sdd._reader_class_refusal(_ns()) is None


def test_default_namespace_without_job_class_attr_is_training():
    assert sdd._reader_class_refusal(argparse.Namespace()) is None


def test_reader_requires_declared_projection():
    rc, msg = sdd._reader_class_refusal(_ns(job_class="reader", rss_cap_mb=2048))
    assert rc == 7
    assert "projected-peak-gib" in msg


def test_reader_refuses_projection_over_4gib():
    rc, msg = sdd._reader_class_refusal(
        _ns(job_class="reader", projected_peak_gib=8.0, rss_cap_mb=2048))
    assert rc == 7
    assert "ceiling" in msg
    assert "training class" in msg


def test_reader_requires_runtime_rss_cap():
    rc, msg = sdd._reader_class_refusal(
        _ns(job_class="reader", projected_peak_gib=2.0))
    assert rc == 7
    assert "rss-cap-mb" in msg


def test_reader_refuses_rss_cap_over_ceiling():
    rc, msg = sdd._reader_class_refusal(
        _ns(job_class="reader", projected_peak_gib=2.0, rss_cap_mb=32768))
    assert rc == 7
    assert "4096" in msg


def test_reader_valid_envelope_passes():
    assert sdd._reader_class_refusal(
        _ns(job_class="reader", projected_peak_gib=2.0, rss_cap_mb=2048)) is None


def test_reader_boundary_values_pass():
    assert sdd._reader_class_refusal(
        _ns(job_class="reader",
            projected_peak_gib=sdd._READER_CLASS_MAX_PROJECTED_GIB,
            rss_cap_mb=sdd._READER_CLASS_MAX_RSS_CAP_MB)) is None


# ---------------------------------------------------------------------------
# _system_admission_gate — reader lane skips the SUM-over-RAM model only
# ---------------------------------------------------------------------------

def test_admission_gate_skips_sum_model_for_reader(capsys):
    a = _ns(job_class="reader", projected_peak_gib=2.0, rss_cap_mb=2048)
    out = sdd._system_admission_gate(a, ["python", "reader.py"])
    assert out is None
    captured = capsys.readouterr().out
    assert "job-class=reader" in captured
    assert "preflight-only" in captured


def test_admission_gate_training_class_reaches_governor(monkeypatch):
    """Counterfactual: the default class does NOT take the reader shortcut —
    it proceeds into the governor import path (stubbed here)."""
    consulted = {}

    class _FakeGov:
        @staticmethod
        def live_admission_decision(projected_new_gib):
            consulted["gib"] = projected_new_gib

            class _D:
                class decision:
                    admit = True
                    reason = "stub-admit"
            return _D()

    import sys as _sys
    monkeypatch.setitem(_sys.modules, "system_memory_governor", _FakeGov())
    a = _ns(projected_peak_gib=2.0)
    out = sdd._system_admission_gate(a, ["python", "reader.py"])
    assert out is None
    assert consulted["gib"] == 2.0  # governor WAS consulted for training class


def test_reader_constants_match_charter_envelope():
    assert sdd._READER_CLASS_MAX_PROJECTED_GIB == 4.0
    assert sdd._READER_CLASS_MAX_RSS_CAP_MB == 4096


# ---------------------------------------------------------------------------
# wiring — CLI flag + registry custody + preflight projection override
# ---------------------------------------------------------------------------

def test_cli_exposes_job_class_flag():
    src = open(sdd.__file__, encoding="utf-8").read()
    assert '"--job-class"' in src
    assert 'choices=["training", "reader"]' in src
    assert 'default="training"' in src


def test_do_start_records_reader_class_and_reason_in_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(sdd, "_REGISTRY_PATH", tmp_path / "daemons.json")
    monkeypatch.setattr(sdd, "_REGISTRY_LOCK", tmp_path / ".lock")
    monkeypatch.setattr(sdd, "_mem_preflight", lambda a: None)
    monkeypatch.setattr(sdd, "_maybe_autostart_blackbox", lambda a, c: None)
    monkeypatch.setattr(sdd, "_witness_dsl_compile_hash_gate", lambda a, c: None)
    monkeypatch.setattr(sdd, "_launch_readiness_gate", lambda a, c: None)
    monkeypatch.setattr(sdd, "_maybe_wrap_safe_run", lambda c, a: c)  # no safe_run in unit scope
    a = argparse.Namespace(
        cmd=["--", "/bin/sleep", "1"], log=str(tmp_path / "r.log"), label="reader525",
        job_class="reader", projected_peak_gib=2.0, rss_cap_mb=2048,
        projected_gb=25.0, min_free_gb=30.0, skip_mem_preflight=True,
        walltime_cap_s=None, skip_admission_gate=False, priority=None,
        verify_s=0.5, with_pty=False, skip_readiness_gate=True,
        readiness_override_rationale=None, admission_override_rationale=None,
        skip_blackbox_autostart=True, dsl_compile_hash=None,
    )
    rc = sdd._do_start(a)
    assert rc == 0
    rows = sdd._load_registry(tmp_path / "daemons.json")
    row = next(r for r in rows if r.get("label") == "reader525")
    assert row["job_class"] == "reader"
    assert "preflight-only" in row["admission_class_reason"]
    assert "#370" in row["admission_class_reason"]
    # free-floor preflight projection was overridden to the declared envelope
    assert a.projected_gb == 2.0
    sdd._do_stop("reader525")


def test_do_start_refuses_undeclared_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(sdd, "_REGISTRY_PATH", tmp_path / "daemons.json")
    monkeypatch.setattr(sdd, "_REGISTRY_LOCK", tmp_path / ".lock")
    monkeypatch.setattr(sdd, "_witness_dsl_compile_hash_gate", lambda a, c: None)
    a = argparse.Namespace(
        cmd=["--", "/bin/sleep", "1"], log=str(tmp_path / "r.log"), label="reader525b",
        job_class="reader", projected_peak_gib=None, rss_cap_mb=None,
        projected_gb=25.0, min_free_gb=30.0, skip_mem_preflight=True,
        walltime_cap_s=None, skip_admission_gate=False, priority=None,
        verify_s=0.5, with_pty=False, skip_readiness_gate=True,
        readiness_override_rationale=None, admission_override_rationale=None,
        skip_blackbox_autostart=True, dsl_compile_hash=None,
    )
    rc = sdd._do_start(a)
    assert rc == 7
    assert not (tmp_path / "daemons.json").exists()  # nothing registered/started
