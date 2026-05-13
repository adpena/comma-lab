"""Tests for the canonical substrate-trainer skeleton helpers.

Lane: lane_canon_dedup_1_20260513
Memo: feedback_canon_dedup_1_LANDED_20260513.md
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tac.substrates._shared import trainer_skeleton as ts

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
        "REPO_ROOT",
        "StageLog",
        "decode_real_pairs",
        "device_or_die",
        "git_head_sha",
        "load_upstream_yuv420_to_rgb",
        "pin_seeds",
        "sha256_bytes",
        "torch_version_string",
        "utc_now_iso",
    }
    assert set(ts.__all__) == expected
