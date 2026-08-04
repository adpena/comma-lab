# SPDX-License-Identifier: MIT
"""Characterization + consolidation test for the levelset evaluate wrapper.

``tools/levelset_byte_close_and_eval.py`` carried a PRIVATE twin of the
upstream-evaluate wrapper.  ``ddm_sub1`` measured **>=6 live twins** of that same
step across ``tools/`` and ``experiments/`` (levelset, contest_eval,
contest_auth_eval, mask_rate_sweep, proxy_eval, plus vendored copies) -- one
authority-path step, six implementations, which is the duplicate-SoT class
(#533) at the surface that decides whether a score is real.

These tests are written CHARACTERIZATION-FIRST: they pin the observable
behaviour of the pre-consolidation wrapper, so that delegating its body to
``tac.submission_chain.run_upstream_evaluate`` is provably behaviour-preserving
rather than hopefully so.  If a future edit changes the contract, these fail.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

_REPORT = """
Evaluation results over 600 samples
Average PoseNet Distortion: 0.0012340000
Average SegNet Distortion: 0.0043117947
Compression Rate: 0.0094235
Final score (100*seg + sqrt(10*pose) + 25*rate) = 0.7910689
"""


@pytest.fixture(scope="module")
def levelset():
    spec = importlib.util.spec_from_file_location(
        "_levelset_bce", _REPO / "tools" / "levelset_byte_close_and_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def packet(tmp_path):
    (tmp_path / "archive.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    inflated = tmp_path / "inflated"
    inflated.mkdir()
    (inflated / "0.raw").write_bytes(b"\x00" * 16)
    (tmp_path / "videos").mkdir()
    (tmp_path / "names.txt").write_text("0\n")
    return tmp_path


def test_wrapper_refuses_mps(levelset, packet):
    """MPS is never a score authority -- the single most important refusal."""
    with pytest.raises(ValueError, match="MPS is NEVER"):
        levelset.run_upstream_evaluate(
            packet, device="mps", uncompressed_dir=packet / "videos",
            video_names_file=packet / "names.txt", archive_bytes=353805, timeout=60,
        )


def test_wrapper_refuses_missing_inflated_raw(levelset, packet):
    """A packet with no decoded output must not be scored."""
    for raw in (packet / "inflated").glob("*.raw"):
        raw.unlink()
    with pytest.raises(FileNotFoundError):
        levelset.run_upstream_evaluate(
            packet, device="cpu", uncompressed_dir=packet / "videos",
            video_names_file=packet / "names.txt", archive_bytes=353805, timeout=60,
        )


def test_wrapper_refuses_missing_archive(levelset, packet):
    (packet / "archive.zip").unlink()
    with pytest.raises(FileNotFoundError):
        levelset.run_upstream_evaluate(
            packet, device="cpu", uncompressed_dir=packet / "videos",
            video_names_file=packet / "names.txt", archive_bytes=353805, timeout=60,
        )


def _fake_run(returncode=0, stdout=_REPORT):
    def _runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")

    return _runner


def test_wrapper_contract_keys_and_values(levelset, packet, monkeypatch):
    """CHARACTERIZATION: the exact keys and values downstream readers consume.

    ``recomputed_S_compute_contest_score`` must be recomputed from COMPONENTS,
    never taken from evaluate.py's rounded printed field.
    """
    monkeypatch.setattr(subprocess, "run", _fake_run())
    out = levelset.run_upstream_evaluate(
        packet, device="cpu", uncompressed_dir=packet / "videos",
        video_names_file=packet / "names.txt", archive_bytes=353805, timeout=60,
    )
    for key in (
        "ran", "device", "evaluate_py_final_score", "d_seg", "d_pose",
        "rate_from_evaluate", "n_samples", "recomputed_S_compute_contest_score",
        "recomputed_vs_evaluate_delta", "archive_bytes_scored", "report_path",
        "score_axis", "authority", "promotion_claim",
    ):
        assert key in out, f"contract key {key!r} disappeared"
    assert out["ran"] is True
    assert out["device"] == "cpu"
    assert out["n_samples"] == 600
    assert out["d_seg"] == pytest.approx(0.0043117947)
    assert out["archive_bytes_scored"] == 353805
    assert out["promotion_claim"] is False

    from tac.contest_score import compute_contest_score

    assert out["recomputed_S_compute_contest_score"] == pytest.approx(
        compute_contest_score(0.0043117947, 0.001234, 353805)
    )


def test_wrapper_refuses_nonzero_evaluate_returncode(levelset, packet, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(returncode=9, stdout="boom"))
    with pytest.raises(RuntimeError):
        levelset.run_upstream_evaluate(
            packet, device="cpu", uncompressed_dir=packet / "videos",
            video_names_file=packet / "names.txt", archive_bytes=353805, timeout=60,
        )


def test_wrapper_refuses_partial_sample_count(levelset, packet, monkeypatch):
    """n600 or it is not evidence: a partial count must never land silently."""
    partial = _REPORT.replace("over 600 samples", "over 120 samples")
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout=partial))
    with pytest.raises(RuntimeError, match="600"):
        levelset.run_upstream_evaluate(
            packet, device="cpu", uncompressed_dir=packet / "videos",
            video_names_file=packet / "names.txt", archive_bytes=353805, timeout=60,
        )


def test_wrapper_refuses_a_report_missing_a_component(levelset, packet, monkeypatch):
    """NO-FAKE: never fabricate a missing score component."""
    broken = _REPORT.replace("Average SegNet Distortion: 0.0043117947", "")
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout=broken))
    with pytest.raises(ValueError, match="d_seg"):
        levelset.run_upstream_evaluate(
            packet, device="cpu", uncompressed_dir=packet / "videos",
            video_names_file=packet / "names.txt", archive_bytes=353805, timeout=60,
        )


def test_canonical_and_levelset_agree_on_the_same_report(levelset, packet, monkeypatch):
    """The consolidation invariant: the canonical wrapper and the tool's wrapper
    must extract the SAME components from the SAME report text.

    This is what makes delegation safe, and what would catch a future drift
    between the two if the delegation is ever unwound.
    """
    from tac.submission_chain import parse_evaluate_report

    monkeypatch.setattr(subprocess, "run", _fake_run())
    out = levelset.run_upstream_evaluate(
        packet, device="cpu", uncompressed_dir=packet / "videos",
        video_names_file=packet / "names.txt", archive_bytes=353805, timeout=60,
    )
    canonical = parse_evaluate_report(_REPORT)
    assert out["d_seg"] == canonical["d_seg"]
    assert out["d_pose"] == canonical["d_pose"]
    assert out["rate_from_evaluate"] == canonical["rate"]
    assert out["evaluate_py_final_score"] == canonical["final_score"]
    assert out["n_samples"] == canonical["n_samples"]
