"""Positive controls for the unscreened-axis paid-dispatch refusal.

A gate that has never refused anything is not a gate.  These tests execute the
REAL dispatcher loader (``experiments/ddm_qs1_modal_t4_dual_axis.load_sealed_inputs``)
against real files on disk, and assert BOTH directions:

* it REFUSES a ps1u-shaped request (every distortion axis an assertion), and
* it ALLOWS a request whose pose or seg axis is genuinely measured.

The synthetic requests reproduce the exact field shapes read out of the retained
``SEALED_REQUEST.json`` corpus on 2026-08-16; the SSD-backed test at the bottom
runs the same loader over the ACTUAL retained ps1u bytes when they are mounted.

Anchor: ``.omx/research/ddm_ps1u_r2_dual_axis_pose_verdict_20260816.md``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tac.deploy.dispatch_axis_screen import (
    UnscreenedAxisDispatchError,
    assert_distortion_axis_locally_screened,
    census_distortion_axis_screen,
)

INPUT_NAMES = ("candidate_archive.zip", "candidate_runtime.zip", "POSE_SCREEN_RESULT.json")

# The exact placeholder pair ps1u shipped, and the exact seg assertion beside it.
PS1U_SEG_PROVENANCE: dict[str, Any] = {
    "re1t_run_id": "NONE_ps1u_seg_asserted_decode_identical",
    "seg_delta_s_exact_t4_field": 0.0,
    "seg_leg_measured": False,
}
# The ps1u evidence payload: a placeholder pair plus a TARGET that looks numeric.
PS1U_EVIDENCE: dict[str, Any] = {
    "schema": "ddm_ps1u_pose_screen_evidence.v1",
    "local_pose_delta": 0.0,
    "pose_unmeasured": True,
    "pre_registered_admission": {"required_cuda_dpose_after": 6.251198917870592e-06},
}


def _base_request(**overrides: Any) -> dict[str, Any]:
    request = {
        "schema": "ddm_qs1_t4_dual_axis_request.v1",
        "run_id": "test_run_r1",
        "resume_from": "test_run_r1",
        "retain_pose_vectors": True,
        "score_claim": False,
        "promotion_eligible": False,
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
    }
    request.update(overrides)
    return request


# --------------------------------------------------------------------------
# census-level controls (the predicate itself)
# --------------------------------------------------------------------------


def test_refuses_the_exact_ps1u_shape() -> None:
    request = _base_request(seg_leg_provenance=PS1U_SEG_PROVENANCE)
    result = census_distortion_axis_screen(request, PS1U_EVIDENCE)
    assert result.pose_measured is False
    assert result.seg_measured is False
    assert result.refused is True
    with pytest.raises(UnscreenedAxisDispatchError, match="every distortion axis is an assertion"):
        assert_distortion_axis_locally_screened(request, PS1U_EVIDENCE)


def test_a_numeric_target_is_not_a_measurement() -> None:
    """The ps1u evidence carries a finite non-zero pose float that is a TARGET."""
    value = PS1U_EVIDENCE["pre_registered_admission"]["required_cuda_dpose_after"]
    assert isinstance(value, float) and value != 0.0
    assert census_distortion_axis_screen(_base_request(), PS1U_EVIDENCE).pose_measured is False


def test_allows_a_request_whose_pose_is_measured_in_the_request() -> None:
    """The ddm_qs2 shape: local_pose_delta real, pose_unmeasured false."""
    request = _base_request(local_pose_delta=1.126177e-07, pose_unmeasured=False)
    result = census_distortion_axis_screen(request, {"schema": "x"})
    assert result.pose_measured is True and result.refused is False


def test_allows_a_nonzero_local_pose_delta_with_a_stale_flag() -> None:
    """The ddm_qs5 shape: a real value parked next to a stale True flag."""
    request = _base_request(local_pose_delta=-6.657906473377261e-09)
    result = census_distortion_axis_screen(request, None)
    assert result.pose_measured is True and result.refused is False
    assert "stale" in result.pose_basis


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"pose_delta_s": 1.378369737898914e-05}, True),  # ddm_qs4
        ({"conservative_residual_pose_bound_s": 1.7746229678414843e-05}, True),  # ddm_qs1
        ({"local_advisory": {"delta_dpose": -1.4632967835484165e-10}}, True),  # ddm_mc36
        ({"base_sample_dpose": 0.00016211002068381195}, True),  # ddm_pk3
        ({"local_pose_advisory": {"delta_dpose": -6.65e-09}}, True),  # ddm_qs5
        ({"local_pose_screen_delta_s": -1.2e-05}, True),  # forward canonical
        ({"target_dpose": 3.44e-06}, False),  # a target, not a screen
        ({"pose_delta_s_placeholder_not_measurement": 0.0}, False),
        ({"maximum_admissible_candidate_flips_without_pose_measurement": 34969}, False),
        ({"pose_delta_s": 0.0}, False),  # zero is the placeholder value
        ({"pose_delta_s": True}, False),  # a bool is never a measurement
    ],
)
def test_evidence_payload_vocabulary(payload: dict[str, Any], expected: bool) -> None:
    assert census_distortion_axis_screen(_base_request(), payload).pose_measured is expected


def test_allows_a_measured_seg_leg_with_pose_still_unknown() -> None:
    """The ddm_re1 shape: buy the pose leg, backed by a real prior seg run."""
    request = _base_request(
        seg_leg_provenance={
            "re1t_run_id": "ddm_re1_round1_t4_gate_20260813r2",
            "seg_delta_s_exact_t4_field": -1.6954210069444444e-06,
        }
    )
    result = census_distortion_axis_screen(request, PS1U_EVIDENCE)
    assert result.pose_measured is False
    assert result.seg_measured is True
    assert result.refused is False


def test_absent_seg_provenance_and_placeholder_pose_refuses() -> None:
    """Deleting the seg block must not become the bypass."""
    assert census_distortion_axis_screen(_base_request(), None).refused is True


def test_deleting_the_pose_unmeasured_flag_is_not_a_bypass() -> None:
    """The gate needs positive evidence; an absent flag proves nothing.

    Found by second-pass review of this gate's own first draft, which used
    ``pose_unmeasured is not True`` and would have passed a request that simply
    dropped the key — the ps1u defect wearing a hat.
    """
    request = _base_request(seg_leg_provenance=PS1U_SEG_PROVENANCE)
    del request["pose_unmeasured"]
    assert census_distortion_axis_screen(request, PS1U_EVIDENCE).refused is True


def test_deleting_both_pose_fields_is_not_a_bypass() -> None:
    request = _base_request(seg_leg_provenance=PS1U_SEG_PROVENANCE)
    del request["pose_unmeasured"]
    del request["local_pose_delta"]
    assert census_distortion_axis_screen(request, PS1U_EVIDENCE).refused is True


# --------------------------------------------------------------------------
# forward declaration + waiver
# --------------------------------------------------------------------------


def test_explicit_forward_declaration_allows() -> None:
    request = _base_request(
        local_axis_screen={
            "pose": {
                "measured": True,
                "delta_s": -1.2e-05,
                "basis": "frozen CPU-torch PoseNet, n600, pk4 chain, $0 local",
            }
        }
    )
    result = census_distortion_axis_screen(request, None)
    assert result.pose_measured is True and result.refused is False


def test_explicit_declaration_cannot_claim_measured_without_a_number() -> None:
    request = _base_request(
        local_axis_screen={
            "pose": {"measured": True, "basis": "frozen CPU-torch PoseNet n600 local screen"}
        }
    )
    with pytest.raises(UnscreenedAxisDispatchError, match="finite numeric delta_s"):
        census_distortion_axis_screen(request, None)


def test_explicit_declaration_cannot_claim_measured_without_a_basis() -> None:
    request = _base_request(
        local_axis_screen={"pose": {"measured": True, "delta_s": -1.2e-05, "basis": "TBD"}}
    )
    with pytest.raises(UnscreenedAxisDispatchError, match="substantive"):
        census_distortion_axis_screen(request, None)


def test_substantive_waiver_allows_the_unscreened_row() -> None:
    request = _base_request(
        seg_leg_provenance=PS1U_SEG_PROVENANCE,
        unscreened_axis_dispatch_waiver={
            "rationale": (
                "operator-approved transfer probe: this row exists to measure the "
                "CPU-vs-CUDA decode gap, not to admit a candidate"
            )
        },
    )
    result = assert_distortion_axis_locally_screened(request, PS1U_EVIDENCE)
    assert result.waived is True and result.refused is False


@pytest.mark.parametrize("rationale", ["", "<reason>", "<rationale>", "TBD", "pending", "ok", "n/a"])
def test_placeholder_waiver_rationales_are_rejected(rationale: str) -> None:
    request = _base_request(
        seg_leg_provenance=PS1U_SEG_PROVENANCE,
        unscreened_axis_dispatch_waiver={"rationale": rationale},
    )
    with pytest.raises(UnscreenedAxisDispatchError, match="placeholder or too-short"):
        census_distortion_axis_screen(request, PS1U_EVIDENCE)


def test_non_object_waiver_is_rejected() -> None:
    request = _base_request(unscreened_axis_dispatch_waiver="just let me through")
    with pytest.raises(UnscreenedAxisDispatchError, match="must be an object"):
        census_distortion_axis_screen(request, PS1U_EVIDENCE)


# --------------------------------------------------------------------------
# wire-in controls: the REAL dispatcher loader, real files
# --------------------------------------------------------------------------


def _seal(tmp_path: Path, request: dict[str, Any], evidence: dict[str, Any]) -> tuple[Path, Path, str]:
    """Write a loadable sealed request + fire inputs, as a sealer would."""
    input_root = tmp_path / "fire_inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "candidate_archive.zip": b"PK\x05\x06" + b"\0" * 18,
        "candidate_runtime.zip": b"PK\x05\x06" + b"\0" * 18,
        "POSE_SCREEN_RESULT.json": (json.dumps(evidence, sort_keys=True) + "\n").encode(),
    }
    inputs = {}
    for name in INPUT_NAMES:
        (input_root / name).write_bytes(payloads[name])
        inputs[name] = {
            "bytes": len(payloads[name]),
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }
    sealed = dict(request)
    sealed["inputs"] = inputs
    request_path = tmp_path / "SEALED_REQUEST.json"
    request_path.write_bytes((json.dumps(sealed, sort_keys=True, indent=2) + "\n").encode())
    return request_path, input_root, hashlib.sha256(request_path.read_bytes()).hexdigest()


def _loader():
    modal = pytest.importorskip("modal", reason="the QS1 transport imports modal")
    assert modal is not None
    from experiments import ddm_qs1_modal_t4_dual_axis as dispatcher

    return dispatcher


def test_real_loader_refuses_the_ps1u_shape(tmp_path: Path) -> None:
    dispatcher = _loader()
    request_path, input_root, sha = _seal(
        tmp_path, _base_request(seg_leg_provenance=PS1U_SEG_PROVENANCE), PS1U_EVIDENCE
    )
    with pytest.raises(UnscreenedAxisDispatchError, match="coin flip on the axis it targets"):
        dispatcher.load_sealed_inputs(request_path, input_root, sha)


def test_real_loader_allows_a_screened_candidate(tmp_path: Path) -> None:
    dispatcher = _loader()
    request_path, input_root, sha = _seal(
        tmp_path, _base_request(), {"schema": "test", "pose_delta_s": 1.378369737898914e-05}
    )
    payloads, request = dispatcher.load_sealed_inputs(request_path, input_root, sha)
    assert set(payloads) == set(INPUT_NAMES)
    assert request["run_id"] == "test_run_r1"


def test_unparseable_evidence_fails_closed(tmp_path: Path) -> None:
    """A corrupt evidence file must read as UNSCREENED, never as a pass."""
    dispatcher = _loader()
    request_path, input_root, sha = _seal(tmp_path, _base_request(), {"schema": "test"})
    corrupt = b"{not json"
    (input_root / "POSE_SCREEN_RESULT.json").write_bytes(corrupt)
    sealed = json.loads(request_path.read_text())
    sealed["inputs"]["POSE_SCREEN_RESULT.json"] = {
        "bytes": len(corrupt),
        "sha256": hashlib.sha256(corrupt).hexdigest(),
    }
    request_path.write_bytes((json.dumps(sealed, sort_keys=True, indent=2) + "\n").encode())
    sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
    assert dispatcher._parse_pose_screen_evidence(corrupt) is None
    with pytest.raises(UnscreenedAxisDispatchError):
        dispatcher.load_sealed_inputs(request_path, input_root, sha)


# --------------------------------------------------------------------------
# the retained artifact itself, when the SSD tier is mounted
# --------------------------------------------------------------------------

PS1U_STORE = Path("/Volumes/APDataStore/pact/ddm_ps1u_uncapped_pose_20260816/dual_axis_pose")


@pytest.mark.skipif(
    not (PS1U_STORE / "SEALED_REQUEST.json").is_file(),
    reason="the retained ps1u SSD store is not mounted",
)
def test_real_loader_refuses_the_retained_ps1u_bytes() -> None:
    """The exact request that bought the +1.686e-02 S REFUSE cannot fire again."""
    dispatcher = _loader()
    request_path = PS1U_STORE / "SEALED_REQUEST.json"
    sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
    with pytest.raises(UnscreenedAxisDispatchError):
        dispatcher.load_sealed_inputs(request_path, PS1U_STORE / "fire_inputs", sha)
