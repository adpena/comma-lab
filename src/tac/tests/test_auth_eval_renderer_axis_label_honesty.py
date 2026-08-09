"""Axis-label honesty for experiments/auth_eval_renderer.py.

THE DEFECT (found 2026-08-09, ddm_ax2): the wrapper stamped ``contest_cuda`` /
``[contest-CUDA]`` whenever the forward pass ran on CUDA, while its ground truth
was pinned to ``AVVideoDataset`` unconditionally. ``upstream/evaluate.py:31-46``
selects ``DaliVideoDataset`` iff ``device.type == "cuda"`` and ``AVVideoDataset``
otherwise -- so the contest axis is a property of the GT DECODER. Upstream
couples device and decoder; this wrapper does not. A CUDA run here therefore
measured against PyAV ground truth while claiming the DALI-ground-truth axis.

That label is consumed as CUDA promotion authority (``tac.exact_eval_custody``,
``tac.auth_eval_result``, ``tac.master_gradient_feasibility``), so the over-claim
propagated into promotion gating. This is the phantom-axis class: the metadata is
the truth, and the label must not lie about the object it measured.

Primary self-protection is STRUCTURAL: ``gt_decoder`` is keyword-only with no
default, so omitting it raises TypeError rather than silently defaulting. These
tests cover the residual DRIFT hole -- the module-level ``GT_DECODER`` pin going
stale if the GT construction ever becomes device-conditional.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "experiments" / "auth_eval_renderer.py"


def _load_module():
    for extra in ("upstream", "src"):
        candidate = str(_REPO_ROOT / extra)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    spec = importlib.util.spec_from_file_location("_aer_axis_test", _MODULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        pytest.skip(f"cannot load {_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment-dependent deps
        pytest.skip(f"auth_eval_renderer import unavailable: {type(exc).__name__}: {exc}")
    return module


def test_cuda_over_av_ground_truth_is_not_labelled_contest_cuda():
    """The exact over-claim that shipped: CUDA execution, PyAV ground truth."""
    module = _load_module()
    axis = module.evidence_axis_for_device("cuda", gt_decoder="av")
    assert axis != "contest_cuda"
    assert module.lane_tag_for_evidence_axis(axis) != "[contest-CUDA]"


def test_contest_cuda_remains_reachable_with_dali_ground_truth():
    """Negative control: the fix must not delete the capability, only condition it."""
    module = _load_module()
    assert module.evidence_axis_for_device("cuda", gt_decoder="dali") == "contest_cuda"


def test_gt_decoder_is_required_not_defaulted():
    """Structural guard: omission must fail loudly, not fall back to a device-only axis."""
    module = _load_module()
    with pytest.raises(TypeError):
        module.evidence_axis_for_device("cuda")  # type: ignore[call-arg]


def test_real_call_site_emits_its_ground_truth_and_does_not_over_claim():
    module = _load_module()
    fields = module.device_evidence_fields(
        requested_device="cuda",
        actual_device="cuda",
        device_fallback_occurred=False,
    )
    assert fields["gt_decoder"] == module.GT_DECODER
    assert fields["evidence_axis"] != "contest_cuda"
    # score_axis mirrors evidence_axis; both feed custody, so both must be honest.
    assert fields["score_axis"] != "contest_cuda"


def test_gt_decoder_pin_matches_the_actual_dataset_construction():
    """Drift guard.

    ``GT_DECODER`` is a hand-maintained mirror of what the module actually builds.
    If the GT construction ever becomes device-conditional (mirroring upstream),
    this fires so the pin is re-derived rather than silently going stale.
    """
    source = _MODULE_PATH.read_text(encoding="utf-8")
    module = _load_module()

    av_sites = re.findall(r"^\s*ds_gt\s*=\s*AVVideoDataset\(", source, flags=re.MULTILINE)
    dali_sites = re.findall(r"^\s*ds_gt\s*=\s*DaliVideoDataset\(", source, flags=re.MULTILINE)

    assert len(av_sites) == 1, (
        "auth_eval_renderer ground-truth construction changed shape; re-derive "
        f"GT_DECODER from upstream/evaluate.py:31-46 (av sites={len(av_sites)})"
    )
    assert not dali_sites, (
        "a DALI ground-truth path appeared: GT_DECODER can no longer be a constant, "
        "it must be resolved from the device that selects the dataset"
    )
    assert module.GT_DECODER == "av"
