# SPDX-License-Identifier: MIT
"""Tests for the canonical substrate-trainer skeleton helpers.

Lane: lane_canon_dedup_1_20260513
Memo: feedback_canon_dedup_1_LANDED_20260513.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.substrates._shared import trainer_skeleton as ts

# ---------------------------------------------------------------------------
# REPO_ROOT + constants
# ---------------------------------------------------------------------------


def test_repo_root_resolves_to_pact_repo() -> None:
    assert ts.REPO_ROOT.name == "pact"
    assert (ts.REPO_ROOT / "src" / "tac").is_dir()
    assert (ts.REPO_ROOT / "experiments").is_dir()


def test_eval_hw_is_contest_canonical() -> None:
    assert ts.EVAL_HW == (384, 512)


# ---------------------------------------------------------------------------
# sha256_bytes
# ---------------------------------------------------------------------------


def test_sha256_bytes_known_vector() -> None:
    assert (
        ts.sha256_bytes(b"hello")
        == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_sha256_bytes_empty() -> None:
    assert (
        ts.sha256_bytes(b"")
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


# ---------------------------------------------------------------------------
# torch_version_string
# ---------------------------------------------------------------------------


def test_torch_version_string_nonempty() -> None:
    s = ts.torch_version_string()
    assert isinstance(s, str)
    assert s != ""
    # Either a real version like "2.x.y" or the failure sentinel
    assert s == "<unknown>" or "." in s


# ---------------------------------------------------------------------------
# utc_now_iso
# ---------------------------------------------------------------------------


def test_utc_now_iso_format() -> None:
    s = ts.utc_now_iso()
    # YYYY-MM-DDTHH:MM:SSZ
    assert len(s) == 20
    assert s[4] == "-" and s[7] == "-" and s[10] == "T"
    assert s[13] == ":" and s[16] == ":" and s[19] == "Z"


# ---------------------------------------------------------------------------
# git_head_sha
# ---------------------------------------------------------------------------


def test_git_head_sha_default_root() -> None:
    sha = ts.git_head_sha()
    # Either 40-char hex (real repo) or '<unknown>' (failure)
    assert sha == "<unknown>" or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


def test_git_head_sha_bad_root_returns_unknown(tmp_path: Path) -> None:
    bad = tmp_path / "definitely_not_a_repo"
    bad.mkdir()
    sha = ts.git_head_sha(repo_root=bad)
    assert sha == "<unknown>"


# ---------------------------------------------------------------------------
# pin_seeds
# ---------------------------------------------------------------------------


def test_pin_seeds_makes_random_deterministic() -> None:
    import random

    ts.pin_seeds(42)
    a = random.random()
    ts.pin_seeds(42)
    b = random.random()
    assert a == b


def test_pin_seeds_torch_deterministic() -> None:
    import torch

    ts.pin_seeds(0)
    a = torch.rand(4)
    ts.pin_seeds(0)
    b = torch.rand(4)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# StageLog
# ---------------------------------------------------------------------------


def test_stage_log_starts_empty() -> None:
    sl = ts.StageLog()
    assert sl.entries() == []


def test_stage_log_records_in_order() -> None:
    sl = ts.StageLog()
    sl.stage("first")
    sl.stage("second")
    sl.stage("third")
    names = [e["stage"] for e in sl.entries()]
    assert names == ["first", "second", "third"]


def test_stage_log_entries_carry_utc_at() -> None:
    sl = ts.StageLog()
    sl.stage("only")
    entries = sl.entries()
    assert len(entries) == 1
    assert "at" in entries[0]
    assert entries[0]["at"].endswith("Z")


def test_stage_log_entries_is_shallow_copy() -> None:
    sl = ts.StageLog()
    sl.stage("a")
    snap = sl.entries()
    snap.append({"stage": "spoofed", "at": "now"})
    assert [e["stage"] for e in sl.entries()] == ["a"]


# ---------------------------------------------------------------------------
# device_or_die
# ---------------------------------------------------------------------------


def test_device_or_die_cpu_with_smoke_returns_cpu() -> None:
    import torch

    d = ts.device_or_die("cpu", smoke=True, substrate_tag="testsub")
    assert d == torch.device("cpu")


def test_device_or_die_cpu_without_smoke_raises() -> None:
    with pytest.raises(SystemExit) as exc_info:
        ts.device_or_die("cpu", smoke=False, substrate_tag="testsub")
    assert "testsub" in str(exc_info.value)
    assert "--device cpu" in str(exc_info.value)


def test_device_or_die_cpu_with_explicit_full_cpu_exception_returns_cpu() -> None:
    import torch

    d = ts.device_or_die(
        "cpu",
        smoke=False,
        substrate_tag="testsub",
        allow_full_cpu=True,
    )
    assert d == torch.device("cpu")


def test_device_or_die_full_cpu_exception_is_not_smoke_flag() -> None:
    with pytest.raises(SystemExit) as exc_info:
        ts.device_or_die(
            "cpu",
            smoke=True,
            substrate_tag="testsub",
            allow_full_cpu=True,
        )
    assert "allow_full_cpu=True" in str(exc_info.value)


def test_device_or_die_unknown_raises() -> None:
    with pytest.raises(SystemExit) as exc_info:
        ts.device_or_die("mps", smoke=True, substrate_tag="testsub")
    assert "unknown" in str(exc_info.value)


def test_device_or_die_cuda_without_cuda_raises() -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA actually available; cannot test the no-cuda failure path here")
    with pytest.raises(SystemExit) as exc_info:
        ts.device_or_die("cuda", smoke=False, substrate_tag="testsub")
    assert "cuda not available" in str(exc_info.value)


# ---------------------------------------------------------------------------
# load_upstream_yuv420_to_rgb
# ---------------------------------------------------------------------------


def test_load_upstream_yuv420_to_rgb_default_root() -> None:
    fn = ts.load_upstream_yuv420_to_rgb(substrate_tag="testsub")
    assert callable(fn)


def test_load_upstream_yuv420_to_rgb_bad_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        ts.load_upstream_yuv420_to_rgb(substrate_tag="testsub", repo_root=tmp_path)
    assert "frame_utils.py" in str(exc_info.value)


# ---------------------------------------------------------------------------
# decode_real_pairs
# ---------------------------------------------------------------------------


def test_decode_real_pairs_missing_video_raises(tmp_path: Path) -> None:
    bad = tmp_path / "no_such_video.mkv"
    with pytest.raises(FileNotFoundError) as exc_info:
        ts.decode_real_pairs(bad, n_pairs=1, substrate_tag="testsub")
    assert "real target video not found" in str(exc_info.value)


def test_module_exports_are_stable() -> None:
    """Catch accidental __all__ drift that would break new substrates."""
    expected = {
        "EVAL_HW",
        "OPTIMIZATION_FLAGS_MANIFEST",
        "OptimizedTrainingContext",
        "REPO_ROOT",
        "StageLog",
        "TRAINER_PROXY_AXIS_LABEL",
        "TRAINER_PROXY_PROMOTION_REQUIREMENT",
        "build_optimized_training_context",
        "decode_real_pairs",
        "detect_hardware_substrate",
        "device_or_die",
        "git_head_sha",
        "load_upstream_yuv420_to_rgb",
        "merge_optimization_flags",
        "pin_seeds",
        "require_contest_cuda_auth_eval_claim",
        "sha256_bytes",
        "torch_version_string",
        "utc_now_iso",
    }
    assert set(ts.__all__) == expected


# ---------------------------------------------------------------------------
# require_contest_cuda_auth_eval_claim
# ---------------------------------------------------------------------------


def _component_coherent_payload(**overrides):
    from tac.auth_eval_result import recompute_contest_score_from_payload

    payload = {
        "avg_segnet_dist": 0.001,
        "avg_posenet_dist": 0.0004,
        "archive_size_bytes": 150_000,
        "score_axis": "contest_cuda",
        "lane_tag": "[contest-CUDA]",
        "evidence_grade": "contest-CUDA",
        "exact_cuda_eval_complete": True,
        "score_claim": True,
        "score_claim_valid": True,
    }
    payload.update(overrides)
    payload["canonical_score"] = recompute_contest_score_from_payload(payload)
    return payload


def test_require_contest_cuda_auth_eval_claim_accepts_valid_claim(tmp_path: Path) -> None:
    path = tmp_path / "contest_auth_eval_cuda.json"
    path.write_text(json.dumps(_component_coherent_payload()), encoding="utf-8")

    claim, payload = ts.require_contest_cuda_auth_eval_claim(
        path,
        archive_sha256="abc123",
        substrate_tag="testsub",
    )

    assert claim.score_axis == "contest_cuda"
    assert claim.score == payload["canonical_score"]


def test_require_contest_cuda_auth_eval_claim_accepts_matching_archive_sha(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contest_auth_eval_cuda.json"
    path.write_text(
        json.dumps(_component_coherent_payload(archive_sha256="zipsha")),
        encoding="utf-8",
    )

    claim, _payload = ts.require_contest_cuda_auth_eval_claim(
        path,
        archive_sha256="zipsha",
        substrate_tag="testsub",
    )

    assert claim.score_axis == "contest_cuda"


def test_require_contest_cuda_auth_eval_claim_rejects_wrong_scored_object(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contest_auth_eval_cuda.json"
    path.write_text(
        json.dumps(_component_coherent_payload(archive_sha256="payloadbinsha")),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="wrong scored object"):
        ts.require_contest_cuda_auth_eval_claim(
            path,
            archive_sha256="archivezipsha",
            substrate_tag="testsub",
        )


def test_require_contest_cuda_auth_eval_claim_checks_nested_provenance_sha(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contest_auth_eval_cuda.json"
    path.write_text(
        json.dumps(
            _component_coherent_payload(
                provenance={"archive_sha256": "payloadbinsha"}
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="wrong scored object"):
        ts.require_contest_cuda_auth_eval_claim(
            path,
            archive_sha256="archivezipsha",
            substrate_tag="testsub",
        )


def test_require_contest_cuda_auth_eval_claim_rejects_diagnostic_cuda(tmp_path: Path) -> None:
    path = tmp_path / "diagnostic_auth_eval.json"
    path.write_text(
        json.dumps(
            _component_coherent_payload(
                score_axis="diagnostic_cuda",
                lane_tag="[diagnostic-auth-eval]",
                evidence_grade="B",
                exact_cuda_eval_complete=False,
                score_claim=False,
                score_claim_valid=False,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not a valid \\[contest-CUDA\\] claim"):
        ts.require_contest_cuda_auth_eval_claim(
            path,
            archive_sha256="abc123",
            substrate_tag="testsub",
        )


# ---------------------------------------------------------------------------
# detect_hardware_substrate (SIREN audit 2026-05-13 CRITICAL #1)
# ---------------------------------------------------------------------------


def test_detect_hardware_substrate_cpu_axis_returns_modal_cpu(monkeypatch) -> None:
    """axis='cpu' returns Modal CPU only when Modal env is present."""
    monkeypatch.setattr(ts.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ts.platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("MODAL_TASK_ID", "task")
    assert ts.detect_hardware_substrate(
        axis="cpu", substrate_tag="test",
    ) == "linux_x86_64_modal_cpu"


def test_detect_hardware_substrate_cpu_axis_returns_macos_arm64(monkeypatch) -> None:
    """macOS CPU is advisory and must not be mislabeled as Linux Modal CPU."""
    monkeypatch.setattr(ts.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ts.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    assert ts.detect_hardware_substrate(
        axis="cpu", substrate_tag="test",
    ) == "macos_arm64"


def test_detect_hardware_substrate_cpu_axis_returns_plain_linux(monkeypatch) -> None:
    """Plain Linux x86_64 CPU stays distinct from provider-specific CPU axes."""
    monkeypatch.setattr(ts.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ts.platform, "machine", lambda: "x86_64")
    for name in (
        "MODAL_TASK_ID",
        "MODAL_GPU",
        "GITHUB_ACTIONS",
        "VAST_CONTAINERLABEL",
        "VASTAI_INSTANCE_ID",
        "LIGHTNING_CLOUD_PROJECT_ID",
        "LIGHTNING_USER_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    assert ts.detect_hardware_substrate(
        axis="cpu", substrate_tag="test",
    ) == "linux_x86_64_cpu"


def test_detect_hardware_substrate_unknown_axis_returns_unknown() -> None:
    """Unknown axis returns 'unknown' (defensive default)."""
    assert ts.detect_hardware_substrate(
        axis="mps", substrate_tag="test",
    ) == "unknown"


def test_detect_hardware_substrate_provenance_a100(tmp_path: Path) -> None:
    """provenance.json carrying gpu_name=A100 maps to linux_x86_64_a100."""
    import json
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"gpu_name": "NVIDIA A100-SXM4-40GB"}))
    assert ts.detect_hardware_substrate(
        axis="cuda", substrate_tag="test",
        provenance_path=prov,
    ) == "linux_x86_64_a100"


def test_detect_hardware_substrate_provenance_t4(tmp_path: Path) -> None:
    """provenance.json carrying T4 maps to linux_x86_64_t4."""
    import json
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"gpu_name": "Tesla T4"}))
    assert ts.detect_hardware_substrate(
        axis="cuda", substrate_tag="test",
        provenance_path=prov,
    ) == "linux_x86_64_t4"


def test_detect_hardware_substrate_provenance_4090(tmp_path: Path) -> None:
    """provenance.json carrying 4090 maps to linux_x86_64_4090."""
    import json
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"gpu_name": "NVIDIA GeForce RTX 4090"}))
    assert ts.detect_hardware_substrate(
        axis="cuda", substrate_tag="test",
        provenance_path=prov,
    ) == "linux_x86_64_4090"


def test_detect_hardware_substrate_provenance_h100(tmp_path: Path) -> None:
    """provenance.json carrying H100 maps to linux_x86_64_h100."""
    import json
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"gpu_name": "NVIDIA H100 PCIe"}))
    assert ts.detect_hardware_substrate(
        axis="cuda", substrate_tag="test",
        provenance_path=prov,
    ) == "linux_x86_64_h100"


def test_detect_hardware_substrate_env_var_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    """When provenance.json absent, falls back to env_var_candidates."""
    monkeypatch.setenv("TEST_TRAINER_GPU", "a100-80gb")
    assert ts.detect_hardware_substrate(
        axis="cuda", substrate_tag="test",
        provenance_path=tmp_path / "missing_provenance.json",
        env_var_candidates=("TEST_TRAINER_GPU", "MODAL_GPU"),
    ) == "linux_x86_64_a100"


def test_detect_hardware_substrate_env_var_first_match_wins(
    tmp_path: Path, monkeypatch
) -> None:
    """First env var with a value wins (priority order)."""
    monkeypatch.delenv("FIRST_GPU", raising=False)
    monkeypatch.setenv("SECOND_GPU", "T4")
    assert ts.detect_hardware_substrate(
        axis="cuda", substrate_tag="test",
        provenance_path=tmp_path / "nope.json",
        env_var_candidates=("FIRST_GPU", "SECOND_GPU"),
    ) == "linux_x86_64_t4"


def test_detect_hardware_substrate_unknown_gpu_returns_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    """Unrecognized GPU token falls back to linux_x86_64_unknown_cuda."""
    monkeypatch.setenv("WEIRD_GPU", "ExoticBrand X-99-Z")
    # Clear MODAL_GPU if present to avoid leakage from CI env.
    monkeypatch.delenv("MODAL_GPU", raising=False)
    out = ts.detect_hardware_substrate(
        axis="cuda", substrate_tag="test",
        provenance_path=tmp_path / "absent.json",
        env_var_candidates=("WEIRD_GPU",),
    )
    # On a machine with nvidia-smi available, this could still resolve;
    # accept either the literal fallback or any valid linux_x86_64_* token.
    assert out.startswith("linux_x86_64_"), out


def test_detect_hardware_substrate_corrupt_provenance_falls_through(
    tmp_path: Path, monkeypatch
) -> None:
    """Corrupt provenance.json doesn't crash — falls through to env vars."""
    prov = tmp_path / "provenance.json"
    prov.write_text("{not valid json")
    monkeypatch.setenv("TEST_GPU_FALLBACK", "a100")
    assert ts.detect_hardware_substrate(
        axis="cuda", substrate_tag="test",
        provenance_path=prov,
        env_var_candidates=("TEST_GPU_FALLBACK",),
    ) == "linux_x86_64_a100"


def test_detect_hardware_substrate_a10g_token(tmp_path: Path) -> None:
    """A10G token is recognized."""
    import json
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"gpu_name": "NVIDIA A10G"}))
    assert ts.detect_hardware_substrate(
        axis="cuda", substrate_tag="test",
        provenance_path=prov,
    ) == "linux_x86_64_a10g"


def test_detect_hardware_substrate_l40s_token(tmp_path: Path) -> None:
    """L40S token is recognized."""
    import json
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"gpu_name": "NVIDIA L40S"}))
    assert ts.detect_hardware_substrate(
        axis="cuda", substrate_tag="test",
        provenance_path=prov,
    ) == "linux_x86_64_l40s"
