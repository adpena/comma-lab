"""Regression tests for tools/levelset_byte_close_and_eval pointer-authority gates.

1. #247 CLOSE-side gate (``byte_close_verdict_landed``): the 2026-07-06 independent review caught a
   CRITICAL NO-FAKE bug — the gate read the WRONG report key (``"parity"`` instead of
   ``"parity_on_inflated_frames"``), so it fired unconditionally, including on ``--skip-parity``
   runs, fabricating ``measured`` activation events.
2. Exact-eval axis/authority (``_axis_and_authority``): the 2026-07-06 pointer-authority
   whole-subsystem review caught a CRITICAL provenance bug — the axis was derived from the HOST
   platform only, so a real ``--eval-device cuda`` row on Linux x86_64 persisted score_axis
   ``[contest-CPU]``. CPU and CUDA are SEPARATE evidence spaces (CLAUDE.md apples-to-apples);
   the axis must be computed from the ACTUAL device first.
3. n600 fail-closed (``_require_full_600_samples``): an exact row with a parsed sample count
   != 600 must raise, never land silently (n600 or it is NOT evidence).
4. checkpoint-vs-this-run n600 report semantics (``_this_run_scored_full_600`` +
   ``checkpoint_trained_n600``): the old single ``contest_ready_full_600`` field reflected the
   CHECKPOINT's n_pairs, not what the invocation actually scored (review MED)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def bce_mod():
    for p in (str(REPO / "tools"), str(REPO / "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module("levelset_byte_close_and_eval")


@pytest.fixture(scope="module")
def gate(bce_mod):
    return bce_mod.byte_close_verdict_landed


def test_skip_parity_does_not_record_measured(gate):
    # a --skip-parity run stores {"skipped": True}; NO verdict landed -> must NOT record measured
    assert gate({"parity_on_inflated_frames": {"skipped": True}}) is False


def test_real_verdict_records_measured(gate):
    assert gate({"parity_on_inflated_frames": {"skipped": False, "d_seg_realized_on_inflated": 0.004}}) is True


def test_missing_key_is_failsafe_false(gate):
    # a MISSING key must default to NO-measure (fail-safe: never fabricate)
    assert gate({}) is False


def test_non_dict_parity_is_false(gate):
    assert gate({"parity_on_inflated_frames": None}) is False


def test_old_wrong_key_does_not_trigger(gate):
    # the pre-fix bug: report has a "parity" key (wrong) but not the real one -> must NOT trigger
    assert gate({"parity": {"skipped": False}}) is False


# ---------------------------------------------------------------------------
# _axis_and_authority: axis + authority computed from the ACTUAL --device FIRST.
# The pre-fix bug derived the axis from the host platform only, mislabeling real
# CUDA rows on Linux x86_64 as [contest-CPU] (pointer-authority review CRITICAL).
# ---------------------------------------------------------------------------
_LINUX_X86 = ("Linux", "x86_64")
_LINUX_AMD64 = ("Linux", "amd64")
_LINUX_ARM = ("Linux", "aarch64")
_MACOS = ("Darwin", "arm64")
_WINDOWS = ("Windows", "AMD64")

_AXIS_TABLE = [
    # (device, (system, machine), expected_axis, expected_authority)
    ("cuda", _LINUX_X86, "[contest-CUDA]", "[contest-CUDA]"),
    ("cuda", _LINUX_AMD64, "[contest-CUDA]", "[contest-CUDA]"),
    ("cuda", _MACOS,
     "[non-contest-CUDA advisory] NON-PROMOTABLE", "[non-contest-CUDA advisory] NON-PROMOTABLE"),
    ("cuda", _LINUX_ARM,
     "[non-contest-CUDA advisory] NON-PROMOTABLE", "[non-contest-CUDA advisory] NON-PROMOTABLE"),
    ("cuda:0", _LINUX_X86, "[contest-CUDA]", "[contest-CUDA]"),
    ("cpu", _LINUX_X86, "[contest-CPU]", "[contest-CPU advisory] NON-PROMOTABLE"),
    ("cpu", _LINUX_AMD64, "[contest-CPU]", "[contest-CPU advisory] NON-PROMOTABLE"),
    ("cpu", _MACOS,
     "[macOS-CPU advisory] NON-PROMOTABLE", "[macOS-CPU advisory] NON-PROMOTABLE"),
    # LOW fix: a Linux non-x86_64 host must NOT be labeled "macOS"
    ("cpu", _LINUX_ARM,
     "[Linux-non-x86_64-CPU advisory] NON-PROMOTABLE",
     "[Linux-non-x86_64-CPU advisory] NON-PROMOTABLE"),
    ("cpu", _WINDOWS,
     "[non-contest-CPU advisory] NON-PROMOTABLE", "[non-contest-CPU advisory] NON-PROMOTABLE"),
]


@pytest.mark.parametrize("device,host,expected_axis,expected_authority", _AXIS_TABLE)
def test_axis_and_authority_table(bce_mod, monkeypatch, device, host, expected_axis, expected_authority):
    system, machine = host
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr("platform.machine", lambda: machine)
    axis, authority = bce_mod._axis_and_authority(device)
    assert axis == expected_axis
    assert authority == expected_authority


def test_cuda_row_never_labeled_contest_cpu(bce_mod, monkeypatch):
    # the exact pre-fix CRITICAL: --eval-device cuda on Linux x86_64 must NOT say [contest-CPU]
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    axis, authority = bce_mod._axis_and_authority("cuda")
    assert "contest-CPU" not in axis
    assert "contest-CPU" not in authority
    assert axis == "[contest-CUDA]"


def test_axis_and_authority_refuses_mps(bce_mod):
    with pytest.raises(ValueError, match="MPS"):
        bce_mod._axis_and_authority("mps")


def test_axis_and_authority_refuses_unknown_device(bce_mod):
    with pytest.raises(ValueError, match="unknown device"):
        bce_mod._axis_and_authority("tpu")


# ---------------------------------------------------------------------------
# _require_full_600_samples: exact rows fail CLOSED on partial sample counts.
# ---------------------------------------------------------------------------
def test_n_samples_not_600_raises(bce_mod, tmp_path):
    with pytest.raises(RuntimeError, match=r"n_samples=96 != 600"):
        bce_mod._require_full_600_samples(96, tmp_path / "report.txt")


def test_n_samples_600_passes(bce_mod, tmp_path):
    bce_mod._require_full_600_samples(600, tmp_path / "report.txt")  # must not raise


def test_n_samples_none_passes_documented(bce_mod, tmp_path):
    # None == the report format omitted the samples line (absence-of-field, NOT a partial claim)
    bce_mod._require_full_600_samples(None, tmp_path / "report.txt")  # must not raise


# ---------------------------------------------------------------------------
# checkpoint_trained_n600 vs this_run_scored_full_600 semantics (review MED).
# ---------------------------------------------------------------------------
def test_this_run_scored_full_600_true_only_when_600_pairs_scored(bce_mod):
    assert bce_mod._this_run_scored_full_600({"pairs_scored": 600}) is True
    assert bce_mod._this_run_scored_full_600({"pairs_scored": 96}) is False   # --max-pairs cap
    assert bce_mod._this_run_scored_full_600({"skipped": True}) is False      # --skip-parity
    assert bce_mod._this_run_scored_full_600(None) is False                   # absent -> fail-safe
    assert bce_mod._this_run_scored_full_600({}) is False


def test_report_uses_split_n600_fields_not_conflated_one(bce_mod):
    # pin the report key rename: the old conflated key must not be emitted anymore
    src = Path(bce_mod.__file__).read_text()
    assert '"checkpoint_trained_n600"' in src
    assert '"this_run_scored_full_600"' in src
    assert '"contest_ready_full_600":' not in src  # old dict key (comment mentions are fine)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
