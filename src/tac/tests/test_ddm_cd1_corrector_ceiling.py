"""Tests for ddm_cd1's corrector-port arithmetic and its receipt extractor.

Two claims carry the arm's verdict, so both are tested rather than asserted:

1. **The ceiling is subtraction.** A port replaces the corrector; it cannot speed up the
   rest of the loop.  So the k -> infinity floor is ``token - port_scope`` exactly, and a
   target below that floor must report UNREACHABLE rather than a large finite speedup.
   Getting this wrong in the optimistic direction is how a dead port gets built.

2. **The extractor finds the report where it actually lives.**  Every Modal row carries the
   inflate report as TEXT inside a captured stdout string, not as a nested object.  A
   walker that handled only the object shape would return nothing and read as "no
   measurement" -- the vacuity genus, where a silent skip is indistinguishable from a pass.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

from ddm_cd1_corrector_ceiling import (  # noqa: E402
    CeilingError,
    load_inflate_report,
    price,
)

from tac.contest_budget import CONTEST_CUDA, REFUSE, residual_window  # noqa: E402


def _report(
    *,
    token_seconds: float,
    internal_total: float,
    port_scope: float,
    with_breakdown: bool = True,
) -> dict:
    decoder: dict = {"token_codec": "rc64", "decoder_bit_position": 910837}
    if with_breakdown:
        decoder["token_stage_breakdown"] = {
            "schema": "ddm_cd1_token_stage_breakdown.v1",
            "family_seconds": {
                "model": token_seconds - port_scope - 10.0,
                "corrector": port_scope + 4.0,
                "orchestration": 6.0,
            },
            "port_scope_seconds": port_scope,
            "prelude_seconds": 1.0,
            "unattributed_in_loop_seconds": 0.5,
        }
    return {
        "schema": "ddm_f26p_inflate_report.v1",
        "archive_bytes": 180625,
        "stage_seconds": {
            "token_decode_or_checkpoint_load": token_seconds,
            "total_including_raw_sha256": internal_total,
        },
        "token_decoder": decoder,
    }


def test_ceiling_is_subtraction_not_a_ratio():
    """The k -> inf floor removes the port scope and NOTHING else."""
    priced = price(
        _report(token_seconds=1000.0, internal_total=1080.0, port_scope=300.0),
        axis=CONTEST_CUDA,
    )
    ceiling = priced["ceiling"]
    assert ceiling["token_seconds_at_infinite_speedup"] == pytest.approx(700.0)
    assert ceiling["inflate_seconds_at_infinite_speedup"] == pytest.approx(780.0)
    assert ceiling["seconds_a_perfect_port_can_remove"] == pytest.approx(300.0)
    # A 2x port removes HALF the scope, never half the stage.
    two_x = next(r for r in priced["rows"] if r["port_speedup"] == 2.0)
    assert two_x["token_seconds"] == pytest.approx(850.0)
    assert two_x["inflate_seconds"] == pytest.approx(930.0)
    infinite = next(r for r in priced["rows"] if r["port_speedup"] == "inf")
    assert infinite["token_seconds"] == pytest.approx(700.0)


def test_unreachable_target_is_null_not_a_big_number():
    """A target below the floor must REFUSE to name a speedup.

    The failure this forbids is reporting "you need 40x" for a target that no finite
    speedup reaches: it reads as hard-but-possible when it is arithmetically closed.
    """
    window = residual_window(CONTEST_CUDA)
    # port_scope is far too small to reach either end from this inflate.
    priced = price(
        _report(token_seconds=1400.0, internal_total=1500.0, port_scope=50.0),
        axis=CONTEST_CUDA,
    )
    assert priced["break_even_port_speedup"]["frame_a_narrow_end"] is None
    assert priced["break_even_port_speedup"]["frame_a_wide_end"] is None
    assert priced["break_even_port_speedup"]["frame_a_window"] == [
        window.narrow_end_seconds,
        window.wide_end_seconds,
    ]
    # ... and every graded row stays REFUSE, including the infinite one.
    assert {row["frame_a_verdict"] for row in priced["rows"]} == {REFUSE}


def test_break_even_speedup_is_exact_when_reachable():
    """When the floor clears the target, the named speedup must actually hit it."""
    window = residual_window(CONTEST_CUDA)
    target = window.wide_end_seconds
    priced = price(
        _report(token_seconds=1200.0, internal_total=target + 200.0, port_scope=500.0),
        axis=CONTEST_CUDA,
    )
    k = priced["break_even_port_speedup"]["frame_a_wide_end"]
    assert k is not None and math.isfinite(k)
    removed = 500.0 * (1.0 - 1.0 / k)
    assert (target + 200.0) - removed == pytest.approx(target)


def test_harness_inflate_seconds_overrides_the_report_internal_total():
    """jg5 MEASURED 1415.024 inside and 1419.904 outside; the window grades the outside."""
    priced = price(
        _report(token_seconds=1341.540, internal_total=1415.024, port_scope=400.0),
        axis=CONTEST_CUDA,
        inflate_elapsed_seconds=1419.904,
    )
    measured = priced["measured"]
    assert measured["inflate_seconds"] == pytest.approx(1419.904)
    assert measured["inflate_report_internal_total_seconds"] == pytest.approx(1415.024)
    assert measured["harness_minus_report_seconds"] == pytest.approx(4.880)
    # The prelude lands in non-token inflate, where a token-stage port cannot touch it.
    assert measured["non_token_inflate_seconds"] == pytest.approx(1419.904 - 1341.540)
    assert "authoritative" in measured["inflate_seconds_source"]


def test_frame_a_break_even_subtracts_the_measured_evaluate():
    """Frame A charges inflate+evaluate, but the port only shortens INFLATE.

    So the frame-A target for inflate alone is ``window_end - evaluate``.  Forgetting the
    subtraction understates the required speedup -- optimism in the direction that gets a
    dead port built.
    """
    window = residual_window(CONTEST_CUDA)
    evaluate = 51.428
    inflate = window.wide_end_seconds - evaluate + 100.0  # 100 s over the frame-A wide end
    priced = price(
        _report(token_seconds=800.0, internal_total=inflate, port_scope=400.0),
        axis=CONTEST_CUDA,
        evaluate_seconds=evaluate,
        inflate_elapsed_seconds=inflate,
    )
    k = priced["break_even_port_speedup"]["frame_a_wide_end"]
    assert k is not None
    removed = 400.0 * (1.0 - 1.0 / k)
    assert removed == pytest.approx(100.0)
    # And the graded row at that speedup sits exactly on the frame-A wide end.
    assert (inflate - removed) + evaluate == pytest.approx(window.wide_end_seconds)


def test_frame_b_window_is_derived_and_reproduces_the_published_pair():
    """rr7 2 publishes [890.6, 1430.6] for jg5's 51.428 s evaluate.  DERIVE it, never type it.

    The correction is ``+ (estimated evaluate - measured evaluate)``: the canonical window
    already netted out an ESTIMATE, so charging a measurement against it double-charges.
    Reproducing the published pair from ``window.ci_steps`` is what makes frames A and B
    provably one measurement rather than two numbers that happen to disagree.
    """
    priced = price(
        _report(token_seconds=1341.540, internal_total=1415.024, port_scope=400.0),
        axis=CONTEST_CUDA,
        evaluate_seconds=51.428,
        inflate_elapsed_seconds=1419.904,
    )
    frame_b = priced["frame_b_window"]
    assert frame_b["narrow_end_seconds"] == pytest.approx(890.6, abs=0.05)
    assert frame_b["wide_end_seconds"] == pytest.approx(1430.6, abs=0.05)
    assert "evaluate_estimate" in frame_b["derivation"]


def test_frame_b_is_omitted_without_a_measured_evaluate():
    """Frame B is UNDEFINED without a measured evaluate; it must be absent, not guessed."""
    priced = price(
        _report(token_seconds=1341.540, internal_total=1415.024, port_scope=400.0),
        axis=CONTEST_CUDA,
    )
    assert priced["frame_b_window"] is None
    assert priced["break_even_port_speedup"]["frame_b_narrow_end"] is None
    assert all(row["frame_b_margin_vs_wide_end_seconds"] is None for row in priced["rows"])


def test_uninstrumented_receipt_refuses_instead_of_guessing():
    with pytest.raises(CeilingError, match="UNINSTRUMENTED"):
        price(
            _report(
                token_seconds=1341.540,
                internal_total=1415.024,
                port_scope=0.0,
                with_breakdown=False,
            ),
            axis=CONTEST_CUDA,
        )


def test_extractor_reads_the_report_out_of_a_captured_stdout_string(tmp_path: Path):
    """The Modal shape: the report is TEXT inside a string field, not a nested object."""
    report = _report(token_seconds=1000.0, internal_total=1080.0, port_scope=300.0)
    embedded = (
        "rendered carriers 600/600 in 43.7s\n"
        + json.dumps(report, sort_keys=True)
        + "\n[contest_auth_eval] done\n"
    )
    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    receipt.write_text(json.dumps({"result": {"stdout": embedded, "returncode": 0}}))
    found = load_inflate_report(receipt)
    assert found["stage_seconds"]["token_decode_or_checkpoint_load"] == pytest.approx(1000.0)


def test_extractor_refuses_when_two_distinct_reports_are_present(tmp_path: Path):
    """Two runs in one file means the decomposition cannot name which one it prices."""
    first = json.dumps(
        _report(token_seconds=1000.0, internal_total=1080.0, port_scope=300.0),
        sort_keys=True,
    )
    second = json.dumps(
        _report(token_seconds=1546.6, internal_total=1607.6, port_scope=300.0),
        sort_keys=True,
    )
    receipt = tmp_path / "two.json"
    receipt.write_text(json.dumps({"stdout": first + "\n" + second}))
    with pytest.raises(CeilingError, match="DISTINCT"):
        load_inflate_report(receipt)


def test_extractor_tolerates_a_repeated_identical_report(tmp_path: Path):
    """The same report echoed twice is one measurement, not an ambiguity."""
    text = json.dumps(
        _report(token_seconds=1000.0, internal_total=1080.0, port_scope=300.0),
        sort_keys=True,
    )
    receipt = tmp_path / "dup.json"
    receipt.write_text(json.dumps({"stdout": text, "stderr": text}))
    assert load_inflate_report(receipt)["archive_bytes"] == 180625
