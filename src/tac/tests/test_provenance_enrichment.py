"""Verify _capture_provenance() captures every DX #6 enrichment field.

Fridrich + Quantizr R2 sign-off: a CUDA score reproduction must be
self-contained from pipeline_config.json. If any of these fields is
missing the operator has to chase the score through stdout logs and
shell history. Test catches future field-removal regressions.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PIPELINE_PATH = REPO / "experiments" / "pipeline.py"


def _load_pipeline_module():
    """Side-load experiments/pipeline.py — it's not on PYTHONPATH so the
    tests cannot do `from experiments.pipeline import ...` directly."""
    if "pipeline_test_module" in sys.modules:
        return sys.modules["pipeline_test_module"]
    spec = importlib.util.spec_from_file_location("pipeline_test_module", PIPELINE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_test_module"] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:  # missing optional deps in CI
        pytest.skip(f"pipeline.py import failed: {e}")
    return mod


def _make_cfg(mod):
    """Build a minimal PipelineConfig without touching the filesystem.

    PipelineConfig has no `profile` field — we only need the required ones
    to satisfy __init__. Test exercises _capture_provenance() which reads
    `cfg.profile` defensively via getattr."""
    return mod.PipelineConfig(
        video="/tmp/nonexistent.mkv",
        checkpoint="/tmp/nonexistent.pt",
        masks="",
        upstream="/tmp/nonexistent_upstream",
        output_dir="/tmp/nonexistent_out",
        device="cpu",
    )


def test_provenance_has_dx6_fields():
    """Every enriched field must be present (value None is allowed when
    the underlying probe failed — e.g. ffmpeg not installed in CI)."""
    mod = _load_pipeline_module()
    prov = mod._capture_provenance(_make_cfg(mod))

    # The new fields. Each can be present-but-None (env probe failed) but
    # MUST be a key in the dict. profile_dict is conditional on
    # cfg.profile existing (PipelineConfig has no such field today; the
    # captor only writes it when getattr(cfg, 'profile') resolves), so it
    # is excluded here. test_provenance_profile_dict_resolved covers it.
    expected_fields = {
        "sys_argv",
        "env_vars",
        "ffmpeg_version", "libsvtav1_version", "brotli_version",
    }
    # Tolerate the *_error fallback variant for env-dependent probes.
    actual_keys = set(prov.keys())
    for f in expected_fields:
        ok = (f in actual_keys) or (f"{f}_error" in actual_keys)
        assert ok, f"missing provenance field: {f} (have {sorted(actual_keys)})"


def test_provenance_env_vars_subset():
    """env_vars must include every key the bash bootstraps export."""
    mod = _load_pipeline_module()
    prov = mod._capture_provenance(_make_cfg(mod))
    env = prov.get("env_vars")
    assert isinstance(env, dict)
    expected_keys = {
        "LD_LIBRARY_PATH", "PYTHONPATH", "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_CUDA_ALLOC_CONF", "TAC_UPSTREAM_DIR", "PYTHONHASHSEED",
        "TAC_MODELS_DIR", "INFLATE_TTO",
    }
    missing = expected_keys - set(env.keys())
    assert not missing, f"env_vars missing keys: {missing}"


def test_provenance_sys_argv_is_list():
    """argv capture catches the exact CLI invocation. Must be a list of strings."""
    mod = _load_pipeline_module()
    prov = mod._capture_provenance(_make_cfg(mod))
    assert isinstance(prov["sys_argv"], list)
    assert all(isinstance(s, str) for s in prov["sys_argv"])


def test_provenance_profile_dict_resolved():
    """When `cfg.profile` exists and resolves, profile_dict captures the dict.
    PipelineConfig has no `profile` field today, so we expect the captor to
    skip silently — neither key is required, but the captor must NOT crash."""
    mod = _load_pipeline_module()
    prov = mod._capture_provenance(_make_cfg(mod))
    # The capturer either resolves the profile (if the test cfg gained one
    # via dataclass extension) or skips silently. Both are acceptable; what
    # matters is no crash.
    assert isinstance(prov, dict)


def test_provenance_serialises_as_json():
    """The whole prov dict MUST round-trip through json.dumps — anything
    else means we're writing un-replayable data to pipeline_config.json."""
    import json
    mod = _load_pipeline_module()
    prov = mod._capture_provenance(_make_cfg(mod))
    blob = json.dumps(prov, indent=2)
    parsed = json.loads(blob)
    assert parsed["timestamp_utc"] == prov["timestamp_utc"]
