# SPDX-License-Identifier: MIT
"""Tests for the contest wall-clock budget predicate (``ddm_wc2`` §7.1) and the harness
receipt hardening that carries it.

The two reference points pinned here are REAL measured rows, not fixtures:

* ``br1``  -- contest-CUDA T4, inflate 1,246.928 s + evaluate 43.181 s, Modal call
  ``fc-01M0DQECXABB3PBMS4REVT5P76`` -> **WARN** (fits only on a warm uv cache).
* ``MC36`` -- contest-CPU, 831.535 s measured on an 8-vCPU box, projected to 1,414-1,913 s on
  the contest's 4 vCPU via ua2's 1.7-2.3x band -> **REFUSE** at both ends.

A test that only exercised synthetic seconds would pass while the window drifted underneath
it, so the pins are the anchor and the synthetic cases only probe the edges around them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tac.contest_budget import (
    AXES,
    CONTEST_CPU,
    CONTEST_CUDA,
    DECODE_PATH_NATIVE_DISPATCHED,
    DECODE_PATH_OTHER,
    DECODE_PATH_PYTHON_FALLBACK,
    DECODE_PATH_UNREPORTED,
    GRADE_MEASURED,
    GRADE_PROJECTION,
    JOB_WALL_SECONDS,
    PASS,
    REFUSE,
    WARN,
    BudgetInput,
    CiStep,
    ResidualWindow,
    UnknownBudgetAxis,
    axis_from_lane_tag,
    budget_verdict_for_receipt,
    classify_decode_path,
    evaluate_budget,
    normalize_axis,
    residual_window,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HARNESS = _REPO_ROOT / "experiments" / "contest_auth_eval.py"

# --- the measured reference rows -----------------------------------------------------------
BR1_INFLATE_S = 1246.928
BR1_EVALUATE_S = 43.181
MC36_4VCPU_LOW_S = 1414.0
MC36_4VCPU_HIGH_S = 1913.0


def _load_harness():
    spec = importlib.util.spec_from_file_location("contest_auth_eval_undertest", _HARNESS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================================================
# 1-3. the three verdicts
# ============================================================================================


def test_pass_when_charge_fits_the_cold_cache_ceiling():
    """PASS means it fits even when the runner's uv cache misses."""
    window = residual_window(CONTEST_CUDA)
    verdict = evaluate_budget(CONTEST_CUDA, window.narrow_end_seconds - 100.0, 0.0)
    assert verdict.verdict == PASS
    assert verdict.margin_vs_narrow_end_seconds > 0
    assert "cold-cache ceiling" in verdict.rationale


def test_warn_between_the_two_ends_names_the_cache_dependency():
    """WARN exists because the answer depends on the uv cache state -- say so."""
    window = residual_window(CONTEST_CUDA)
    midpoint = (window.narrow_end_seconds + window.wide_end_seconds) / 2.0
    verdict = evaluate_budget(CONTEST_CUDA, midpoint, 0.0)
    assert verdict.verdict == WARN
    assert "warm" in verdict.rationale
    assert verdict.margin_vs_narrow_end_seconds < 0 < verdict.margin_vs_wide_end_seconds


def test_refuse_beyond_the_warm_cache_ceiling_cites_the_measured_precedent():
    window = residual_window(CONTEST_CPU)
    verdict = evaluate_budget(CONTEST_CPU, window.wide_end_seconds + 1.0, 0.0)
    assert verdict.verdict == REFUSE
    assert "PROJECTED TIMEOUT" in verdict.rationale
    assert "1,958" in verdict.rationale  # lc2/PR130 rc=1, the measured precedent


def test_verdict_is_three_valued_and_never_a_bool():
    """m52: a bool flag is a UI over a continuum. There must be no boolean 'fits' field."""
    verdict = evaluate_budget(CONTEST_CUDA, BR1_INFLATE_S, BR1_EVALUATE_S)
    payload = verdict.to_dict()
    assert payload["verdict"] in {PASS, WARN, REFUSE}
    assert not isinstance(payload["verdict"], bool)
    assert "fits" not in payload


def test_boundaries_are_inclusive_at_each_ceiling():
    """Exactly-at-the-ceiling is inside it; one second past is not."""
    window = residual_window(CONTEST_CPU)
    assert evaluate_budget(CONTEST_CPU, float(window.narrow_end_seconds), 0.0).verdict == PASS
    assert evaluate_budget(CONTEST_CPU, window.narrow_end_seconds + 1.0, 0.0).verdict == WARN
    assert evaluate_budget(CONTEST_CPU, float(window.wide_end_seconds), 0.0).verdict == WARN
    assert evaluate_budget(CONTEST_CPU, window.wide_end_seconds + 1.0, 0.0).verdict == REFUSE


# ============================================================================================
# 4-5. the measured pins
# ============================================================================================


def test_br1_t4_shipping_row_is_WARN():
    """The shipping T4 row fits ONLY on a warm uv cache. This is the live risk wc2 found."""
    verdict = evaluate_budget(CONTEST_CUDA, BR1_INFLATE_S, BR1_EVALUATE_S)
    assert verdict.verdict == WARN
    assert verdict.charged_seconds == pytest.approx(1290.109, abs=1e-6)
    # Margin against the warm ceiling is ~12 s out of 1800 -- a knife edge, not headroom.
    assert 0 < verdict.margin_vs_wide_end_seconds < 20
    assert verdict.margin_vs_narrow_end_seconds < 0


def test_mc36_projected_on_contest_4_vcpu_is_REFUSE_at_both_ends():
    """MC36's 831.5 s was measured on 8 vCPU; the contest box has 4. Both corners refuse."""
    for projected in (MC36_4VCPU_LOW_S, MC36_4VCPU_HIGH_S):
        verdict = evaluate_budget(CONTEST_CPU, projected, 0.0)
        assert verdict.verdict == REFUSE, projected
        assert verdict.margin_vs_wide_end_seconds < 0


def test_the_windows_are_exactly_the_adjudicated_ones():
    """wc2 §7.1 pins CUDA [822,1302] s and CPU [1044,1332] s. Drift here is a silent regrade."""
    cuda = residual_window(CONTEST_CUDA)
    cpu = residual_window(CONTEST_CPU)
    assert (cuda.narrow_end_seconds, cuda.wide_end_seconds) == (822, 1302)
    assert (cpu.narrow_end_seconds, cpu.wide_end_seconds) == (1044, 1332)


def test_the_full_1800_is_never_the_denominator():
    """The 2.17x-headroom defect divided by the whole job wall. No end may equal it."""
    for axis in AXES:
        window = residual_window(axis)
        assert window.wide_end_seconds < JOB_WALL_SECONDS
        assert window.narrow_end_seconds < window.wide_end_seconds


# ============================================================================================
# 6-9. grade + provenance are not strippable
# ============================================================================================


def test_window_cannot_be_constructed_without_provenance():
    with pytest.raises(ValueError, match="provenance"):
        ResidualWindow(
            axis=CONTEST_CUDA, narrow_end_seconds=822, wide_end_seconds=1302,
            grade=GRADE_PROJECTION, provenance=(),
        )


def test_window_refuses_an_unknown_grade_and_an_inverted_window():
    with pytest.raises(ValueError, match="grade"):
        ResidualWindow(
            axis=CONTEST_CUDA, narrow_end_seconds=822, wide_end_seconds=1302,
            grade="TOTALLY_MEASURED", provenance=("somewhere",),
        )
    with pytest.raises(ValueError, match="narrow end"):
        ResidualWindow(
            axis=CONTEST_CUDA, narrow_end_seconds=1302, wide_end_seconds=822,
            grade=GRADE_PROJECTION, provenance=("somewhere",),
        )


def test_window_is_graded_PROJECTION_and_says_so_everywhere_it_is_read():
    """A naked float pair is the defect. Every read surface must carry the grade with it."""
    window = residual_window(CONTEST_CUDA)
    assert window.grade == GRADE_PROJECTION
    assert window.grade in str(window)          # f-string use cannot hide it
    payload = window.to_dict()
    assert payload["grade"] == GRADE_PROJECTION
    assert payload["provenance"]                 # non-empty
    assert "false-authority" in payload["false_authority_warning"]
    # No __iter__: `lo, hi = window` must not silently yield a bare pair.
    with pytest.raises(TypeError):
        _lo, _hi = window  # type: ignore[misc]


def test_window_inputs_separate_measured_payloads_from_estimated_seconds():
    """Only the job wall and the payload BYTES were measured. Every second is an estimate."""
    window = residual_window(CONTEST_CUDA)
    by_name = {i.name: i for i in window.inputs}
    assert by_name["job_wall_seconds"].grade == GRADE_MEASURED
    assert by_name["job_wall_seconds"].value == JOB_WALL_SECONDS
    assert by_name["uv_sync_group_cu128_payload_bytes"].grade == GRADE_MEASURED
    assert by_name["uv_sync_group_cu128_payload_bytes"].value == 3_190_398_780
    assert by_name["evaluate_py_600_pairs_t4_seconds"].grade != GRADE_MEASURED
    # No CI step's SECONDS may claim MEASURED -- none has been timed on a real runner.
    assert all(step.grade != GRADE_MEASURED for step in window.ci_steps)


def test_step_table_reconciles_with_the_published_window_and_reports_the_gap():
    """ua2 published rounded minutes. The gap against a fresh step sum is PINNED, not hidden.

    Pinned exactly rather than bounded loosely: a tolerance wide enough to cover the known
    -21 s CPU gap would also hide a new one, which is how a reconciliation gap stops being
    visible. CUDA reconciles to 1 s; CPU's narrow end is 21 s STRICTER than a fresh sum of
    ua2's own step table, so the published window is the conservative one at the binding end.
    """
    assert residual_window(CONTEST_CUDA).reconciliation_delta_seconds == (-1.0, -1.0)
    assert residual_window(CONTEST_CPU).reconciliation_delta_seconds == (-21.0, 1.0)
    for axis in AXES:
        # The published narrow (binding) end must never be LOOSER than a fresh derivation.
        assert residual_window(axis).reconciliation_delta_seconds[0] <= 0, axis


def test_budget_input_and_ci_step_refuse_bad_construction():
    with pytest.raises(ValueError, match="source"):
        BudgetInput("x", 1, "s", GRADE_MEASURED, "")
    with pytest.raises(ValueError, match="worst"):
        CiStep("x", typical_seconds=10, worst_seconds=5, grade=GRADE_MEASURED, source="s")


# ============================================================================================
# 10-12. axis labelling
# ============================================================================================


def test_advisory_axes_refuse_rather_than_borrowing_a_contest_window():
    """Mapping [macOS-CPU advisory] onto contest-CPU would manufacture compliance from a label."""
    for advisory in ("[macOS-CPU advisory]", "macOS-CPU advisory", "mps", "diagnostic_cpu",
                     "[CPU advisory]", "[diagnostic-auth-eval]", "", "cpu_advisory"):
        with pytest.raises(UnknownBudgetAxis):
            normalize_axis(advisory)
        assert axis_from_lane_tag(advisory) is None


def test_bare_device_names_refuse_because_args_device_cannot_carry_the_axis():
    """``args.device == 'cpu'`` is the SAME string for contest-CPU and macOS advisory.

    Accepting it would let a macOS run be graded against a contest runner's window -- the axis
    laundering the lane_tag exists to prevent. Callers must pass the graded axis.
    """
    for bare in ("cpu", "cuda", "CPU", "CUDA"):
        with pytest.raises(UnknownBudgetAxis):
            normalize_axis(bare)
        assert axis_from_lane_tag(bare) is None


def test_contest_axis_spellings_normalize_and_carry_their_label():
    assert normalize_axis("[contest-CUDA]") == CONTEST_CUDA
    assert normalize_axis("contest_cuda") == CONTEST_CUDA
    assert normalize_axis("CONTEST-CPU") == CONTEST_CPU
    assert axis_from_lane_tag("[contest-CPU]") == CONTEST_CPU
    assert axis_from_lane_tag(None) is None
    payload = evaluate_budget(CONTEST_CPU, 100.0, 10.0).to_dict()
    assert payload["axis_label"] == "[contest-CPU]"
    assert payload["is_score_claim"] is False


def test_receipt_adapter_grades_a_contest_receipt_and_never_raises_on_others():
    graded = budget_verdict_for_receipt(
        {
            "lane_tag": "[contest-CUDA]",
            "inflate_elapsed_seconds": BR1_INFLATE_S,
            "evaluate_elapsed_seconds": BR1_EVALUATE_S,
        }
    )
    assert graded["verdict"] == WARN
    assert graded["axis"] == CONTEST_CUDA
    json.dumps(graded)  # must be receipt-serializable

    advisory = budget_verdict_for_receipt(
        {"lane_tag": "[macOS-CPU advisory]", "inflate_elapsed_seconds": 500.0}
    )
    assert advisory["verdict"] == "NOT_APPLICABLE"
    assert advisory["axis"] is None

    missing = budget_verdict_for_receipt({"lane_tag": "[contest-CPU]"})
    assert missing["verdict"] == "NOT_APPLICABLE"
    assert "inflate_elapsed_seconds absent" in missing["reason"]

    broken = budget_verdict_for_receipt(
        {"lane_tag": "[contest-CPU]", "inflate_elapsed_seconds": float("nan")}
    )
    assert broken["verdict"] == "ERROR"


# ============================================================================================
# 13-15. decode_path
# ============================================================================================


def test_decode_path_is_classified_without_guessing():
    assert classify_decode_path(None) == DECODE_PATH_UNREPORTED
    assert classify_decode_path("unknown") == DECODE_PATH_UNREPORTED
    assert classify_decode_path("python") == DECODE_PATH_PYTHON_FALLBACK
    assert classify_decode_path("scalar-python") == DECODE_PATH_PYTHON_FALLBACK
    assert classify_decode_path("native-hpac-avx2") == DECODE_PATH_NATIVE_DISPATCHED
    assert classify_decode_path("neon") == DECODE_PATH_NATIVE_DISPATCHED
    assert classify_decode_path("wobble") == DECODE_PATH_OTHER
    # Ambiguity must cost caution, not lose it: a mixed label keeps the native warning.
    assert classify_decode_path("native-hpac with python glue") == DECODE_PATH_NATIVE_DISPATCHED


def test_scalar_c_rung_is_native_not_other():
    """``scalar`` is the intrinsic-free C rung the docstring names -- ddm_rr7 shipped it.

    Before the fix it fell through to ``other``, which reads as "unknown rung" AND drops the
    native-fast-path caution.  The ordering against ``scalar-python`` is the whole subtlety:
    a label that NAMES python must stay a fallback.
    """
    assert classify_decode_path("scalar") == DECODE_PATH_NATIVE_DISPATCHED
    assert classify_decode_path("SCALAR") == DECODE_PATH_NATIVE_DISPATCHED
    assert classify_decode_path(" scalar ") == DECODE_PATH_NATIVE_DISPATCHED
    assert classify_decode_path("scalar-c") == DECODE_PATH_NATIVE_DISPATCHED
    # The x86 twin was the SECOND silent drop: f26_hpac_native.c:732-752 can emit exactly
    # {scalar, neon, avx2, x86-scalar}, and TWO of those four fell through to ``other``.
    assert classify_decode_path("x86-scalar") == DECODE_PATH_NATIVE_DISPATCHED
    # ... but python wins whenever the label says python, in either order.
    assert classify_decode_path("scalar-python") == DECODE_PATH_PYTHON_FALLBACK
    assert classify_decode_path("python-scalar") == DECODE_PATH_PYTHON_FALLBACK
    assert classify_decode_path("scalar fallback") == DECODE_PATH_PYTHON_FALLBACK


def test_scalar_c_rung_carries_the_native_caution_in_the_verdict():
    """The consumer, not just the classifier: ``other`` silently dropped this warning."""
    verdict = evaluate_budget(
        CONTEST_CUDA, BR1_INFLATE_S, BR1_EVALUATE_S, decode_path="scalar"
    )
    joined = " ".join(verdict.notes)
    assert verdict.decode_path_class == DECODE_PATH_NATIVE_DISPATCHED
    assert "DISPATCHED NATIVE" in joined
    assert "did not match a known dispatch rung" not in joined
    # The load-bearing consequence, not just the prose: BR1 charges 1290.109 s, inside the
    # wide end, so the verdict is WARN and the margin now DECLARES its fast-path dependence.
    # Under ``other`` this flag stayed False and the dependence was invisible.
    assert verdict.verdict == WARN
    assert verdict.margin_depends_on_unverified_fast_path is True
    assert (
        evaluate_budget(
            CONTEST_CUDA, BR1_INFLATE_S, BR1_EVALUATE_S, decode_path="python"
        ).margin_depends_on_unverified_fast_path
        is False
    )
    # The label still never overrides the measurement.
    assert verdict.verdict == evaluate_budget(CONTEST_CUDA, BR1_INFLATE_S, BR1_EVALUATE_S).verdict


def test_python_fallback_decode_is_visible_in_the_verdict_rationale():
    """A Python-fallback decode is the case this predicate exists to surface."""
    verdict = evaluate_budget(
        CONTEST_CUDA, BR1_INFLATE_S, BR1_EVALUATE_S, decode_path="python"
    )
    joined = " ".join(verdict.notes)
    assert verdict.decode_path_class == DECODE_PATH_PYTHON_FALLBACK
    assert "PYTHON FALLBACK" in joined
    assert "upper bound" in joined
    # The label never overrides the measurement.
    assert verdict.verdict == evaluate_budget(CONTEST_CUDA, BR1_INFLATE_S, BR1_EVALUATE_S).verdict


def test_native_dispatch_pass_is_flagged_as_depending_on_an_unverified_fast_path():
    """The fallback is fail-closed and therefore SILENT; a native PASS carries that risk."""
    fast = evaluate_budget(CONTEST_CUDA, 400.0, 40.0, decode_path="native-hpac-avx2")
    assert fast.verdict == PASS
    assert fast.margin_depends_on_unverified_fast_path is True
    assert "UNVERIFIED" in " ".join(fast.notes)

    slow = evaluate_budget(CONTEST_CUDA, 400.0, 40.0, decode_path="python")
    assert slow.margin_depends_on_unverified_fast_path is False

    refused = evaluate_budget(CONTEST_CPU, 5000.0, 40.0, decode_path="native-hpac-avx2")
    assert refused.margin_depends_on_unverified_fast_path is False  # nothing to protect


def test_unreported_decode_path_is_called_out_not_treated_as_fine():
    verdict = evaluate_budget(CONTEST_CUDA, BR1_INFLATE_S, BR1_EVALUATE_S)
    assert verdict.decode_path_class == DECODE_PATH_UNREPORTED
    assert "not reported" in " ".join(verdict.notes)


# ============================================================================================
# 16-18. charge semantics + input hygiene
# ============================================================================================


def test_missing_evaluate_term_marks_the_charge_as_a_lower_bound():
    verdict = evaluate_budget(CONTEST_CUDA, BR1_INFLATE_S, None)
    assert verdict.charge_is_lower_bound is True
    assert verdict.charged_seconds == pytest.approx(BR1_INFLATE_S)
    assert "LOWER BOUND" in " ".join(verdict.notes)
    assert evaluate_budget(CONTEST_CUDA, BR1_INFLATE_S, 0.0).charge_is_lower_bound is False


def test_charging_evaluate_is_conservative_and_the_receipt_says_so():
    """The residual already netted out an ESTIMATED evaluate; charging it again is deliberate."""
    verdict = evaluate_budget(CONTEST_CUDA, BR1_INFLATE_S, BR1_EVALUATE_S)
    assert verdict.charged_seconds > verdict.inflate_seconds
    joined = " ".join(verdict.notes)
    assert "double-counts" in joined
    assert "never toward a false PASS" in joined


def test_non_finite_and_negative_seconds_fail_closed():
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            evaluate_budget(CONTEST_CUDA, bad, 0.0)
        with pytest.raises(ValueError):
            evaluate_budget(CONTEST_CUDA, 100.0, bad)


# ============================================================================================
# 19-22. harness receipt hardening (items 2 + 3)
# ============================================================================================


def test_harness_timeout_help_text_names_the_job_wall_not_a_decode_budget():
    """Item 2: the defaults permit 3600 s against an 1800 s JOB wall. Make that explicit."""
    text = _HARNESS.read_text()
    inflate_help = text.split('"--inflate-timeout"', 1)[1].split("parser.add_argument", 1)[0]
    assert "WHOLE CI JOB" in inflate_help
    assert "RESIDUAL" in inflate_help
    evaluate_help = text.split('"--evaluate-timeout"', 1)[1].split("parser.add_argument", 1)[0]
    assert "3600s" in evaluate_help
    assert "TWICE" in evaluate_help


def test_harness_gt_lineage_reuses_the_canonical_vocabulary_and_follows_the_device_fork():
    """Item 3: dg1's cure had ZERO occurrences here. Reuse gl1's labels; never invent one."""
    from tac.gt_lineage import AUTHORITY_LINEAGE, DALI_NVDEC, PYAV_YUV420_TO_RGB

    harness = _load_harness()
    names = _REPO_ROOT / "upstream" / "public_test_video_names.txt"
    upstream = _REPO_ROOT / "upstream"
    if not (upstream / "evaluate.py").is_file() or not names.is_file():
        pytest.skip("pinned upstream snapshot not present")

    cuda_prov: dict = {}
    cuda = harness._record_gt_lineage(
        cuda_prov, upstream, names, argparse.Namespace(device="cuda")
    )
    assert cuda["lineage"] == DALI_NVDEC == AUTHORITY_LINEAGE
    assert cuda["is_authority_lineage"] is True
    assert cuda["runtime_decoder"] == "DaliVideoDataset"
    assert cuda_prov["gt_lineage"] is cuda

    cpu = harness._record_gt_lineage({}, upstream, names, argparse.Namespace(device="cpu"))
    assert cpu["lineage"] == PYAV_YUV420_TO_RGB
    assert cpu["is_authority_lineage"] is False
    assert "CROSS-LINEAGE" in cpu["cross_lineage_note"]
    # is_authority_lineage is a COMPARABILITY flag, never a defect flag: a contest-CPU row is
    # SUPPOSED to be PyAV, because that is what upstream's own device fork selects.
    assert cpu["lineage_is_axis_native"] is True
    assert cuda["lineage_is_axis_native"] is True
    assert "not a defect" in cpu["lineage_is_axis_native_note"]
    # The GT INPUT bytes are pinned too: lineage names the decoder, this names what it decoded.
    assert cpu["gt_video_inputs"]
    assert all(row["sha256"] for row in cpu["gt_video_inputs"] if row["exists"])


def test_harness_records_the_full_et4_instrument_tuple_not_just_torch_version():
    """et4: forward VALUES move with (code, weights, threads, batch, device)."""
    harness = _load_harness()
    upstream = _REPO_ROOT / "upstream"
    if not (upstream / "evaluate.py").is_file():
        pytest.skip("pinned upstream snapshot not present")

    prov = {
        "pact_commit": "deadbeef",
        "torch_version": "2.9.0",
        "inflate_runtime_manifest": {"runtime_files_sha256": "abc123"},
    }
    tup = harness._record_instrument_tuple(
        prov, upstream, argparse.Namespace(device="cpu", inflate_device="auto"),
        decode_path="python",
    )
    assert prov["instrument_tuple"] is tup
    for leg in ("code", "weights", "threads", "batch", "device"):
        assert leg in tup, leg
    assert tup["code"]["runtime_files_sha256"] == "abc123"
    assert tup["decode_path"] == "python"
    assert tup["decode_path_reported"] is True
    weights = {w["name"]: w for w in tup["weights"]}
    assert set(weights) == {"posenet.safetensors", "segnet.safetensors"}
    for w in weights.values():
        if w["exists"]:
            assert len(w["sha256"]) == 64
    json.dumps(tup)


def test_harness_reads_the_upstream_batch_shape_from_source():
    """The harness pins nothing, so the effective batch shape is upstream's defaults -- read them."""
    harness = _load_harness()
    upstream = _REPO_ROOT / "upstream"
    if not (upstream / "evaluate.py").is_file():
        pytest.skip("pinned upstream snapshot not present")
    shape = harness._upstream_scorer_batch_shape(upstream)
    assert shape["harness_pins_batch_shape"] is False
    assert shape["batch_size"] == 16
    assert shape["num_threads"] == 2
    assert shape["seq_len"] == 2
    assert "parse_error" not in shape


def test_harness_decode_path_detection_never_invents_a_label(monkeypatch):
    harness = _load_harness()
    for key in ("F26_TOKEN_DECODER", "PACT_DECODE_PATH"):
        monkeypatch.delenv(key, raising=False)
    assert harness._detect_decode_path(None) is None
    assert harness._detect_decode_path({}) is None
    assert harness._detect_decode_path({"F26_TOKEN_DECODER": "native-hpac"}) == "native-hpac"
    monkeypatch.setenv("PACT_DECODE_PATH", "python")
    assert harness._detect_decode_path(None) == "python"
