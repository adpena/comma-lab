"""ddm_pr11's `mode: "t4_direct"` leg contract (candidate_decode_wall_clock.v2) and inheritance
from it: a hash-bound, completed, cold, n600 public-entrypoint T4 decode of the exact archive and
runtime is the timing authority — no local denominator, no ratio; every binding re-hashed."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tac.candidate_seal import SealContractError, measure_archive_identity
from tac.decode_wall_clock import (
    DECODE_WALL_CLOCK_SCHEMA_V2,
    INHERITANCE_SCOPE,
    LIMIT_SECONDS,
    build_t4_direct_leg,
    inherit_decode_wall_clock,
    measure_receiver_digest,
    measure_t4_runtime_digest,
    validate_decode_wall_clock,
)


def _report(**changes) -> str:
    report = {"pair_count": 600, "checkpoint_resume": False, "archive_bytes": 1,
              "token_decoder": {"checkpoint_resumed_from_frame": 0, "decoded_token_sha256": "x"},
              "token_cache": {"status": "DISABLED"}}
    report.update(changes)
    return "[inflate] starting\nDWC1_REPORT " + json.dumps(report) + "\n[inflate] done\n"


def _t4_receipt(runtime: Path, path: Path, *, seconds: float = 900.0, **top) -> Path:
    doc = {
        "passed": True, "returncode": 0, "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "inflate_sh_rel": "inflate.sh",
        "expected_archive_sha256": measure_archive_identity(runtime / "archive.zip").sha256,
        "expected_runtime_tree_sha256": measure_t4_runtime_digest(runtime),
        "artifacts": {
            "contest_auth_eval.json": json.dumps({"inflate_elapsed_seconds": seconds, "evaluate_elapsed_seconds": 45, "n_samples": 600}),
            "modal_cuda_preflight.json": json.dumps({"torch_cuda_device_name": "Tesla T4", "torch_cuda_available": True}),
            "contest_auth_eval.stdout.log": _report(),
        },
    }
    doc.update(top)
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def staged(tmp_path):
    from tac.tests.test_candidate_seal import _stage_candidate
    runtime, archive = _stage_candidate(tmp_path)
    return runtime, archive, tmp_path / "t4"


def test_t4_direct_leg_builds_validates_and_inherits(staged):
    runtime, archive, root = staged
    root.mkdir()
    receipt = _t4_receipt(runtime, root / "t4.json", seconds=900.0)
    leg = build_t4_direct_leg(t4_receipt_path=receipt, runtime_dir=runtime, archive_path=archive)
    assert leg["schema"] == DECODE_WALL_CLOCK_SCHEMA_V2 and leg["mode"] == "t4_direct"
    assert leg["measured_t4_decode_seconds"] == leg["projected_t4_decode_seconds"] == 900.0
    assert leg["receiver_sha256"] == measure_receiver_digest(runtime)
    assert leg["t4_runtime_sha256"] == measure_t4_runtime_digest(runtime)
    for key in ("local_receipt", "calibration_receipt", "cpu_to_t4_ratio"):
        assert key not in leg
    problems, observed = validate_decode_wall_clock(leg, runtime_dir=runtime, archive_path=archive)
    assert problems == [] and observed["mode"] == "t4_direct" and observed["cold_report_pairs"] == 600
    leg_path = root / "leg.json"
    leg_path.write_text(json.dumps(leg))
    # a receiver-identical candidate inherits the direct measurement
    inherited = inherit_decode_wall_clock(source_leg_path=leg_path, runtime_dir=runtime, archive_path=archive,
                                          pointer_archive_sha256=leg["archive_sha256"])
    assert inherited["source_mode"] == "t4_direct" and inherited["projected_t4_decode_seconds"] == 900.0
    assert inherited["inheritance_scope"] == INHERITANCE_SCOPE
    assert inherited["source_t4_runtime_sha256"] == leg["t4_runtime_sha256"]
    assert inherited["candidate_t4_runtime_sha256"] == measure_t4_runtime_digest(runtime)
    problems, _ = validate_decode_wall_clock(inherited, runtime_dir=runtime, archive_path=archive,
                                             pointer_archive_sha256=leg["archive_sha256"])
    assert problems == []
    # inheritance from an inherited leg is refused
    inherited_path = root / "inherited.json"
    inherited_path.write_text(json.dumps(inherited))
    with pytest.raises(SealContractError, match="never to an inherited one"):
        inherit_decode_wall_clock(source_leg_path=inherited_path, runtime_dir=runtime, archive_path=archive,
                                  pointer_archive_sha256=leg["archive_sha256"])


@pytest.mark.parametrize("mutation, message", [
    ({"passed": False, "returncode": 1}, "completed successful"),
    ({"canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cpu"}, "canonical path"),
    ({"inflate_sh_rel": "run.sh"}, "inflate.sh"),
    ({"artifacts": {"modal_cuda_preflight.json": json.dumps({"torch_cuda_device_name": "NVIDIA A100", "torch_cuda_available": True})}}, "T4 hardware"),
    ({"artifacts": {"modal_cuda_preflight.json": json.dumps({"torch_cuda_device_name": "Tesla T4", "torch_cuda_available": False})}}, "CUDA unavailable"),
    ({"artifacts": {"contest_auth_eval.json": json.dumps({"inflate_elapsed_seconds": 900.0, "n_samples": 96})}}, "600-sample"),
    ({"artifacts": {"contest_auth_eval.stdout.log": _report(checkpoint_resume=True)}}, "checkpoint resume"),
    ({"artifacts": {"contest_auth_eval.stdout.log": _report(token_decoder={"checkpoint_resumed_from_frame": 25})}}, "resumed from a checkpoint"),
    ({"artifacts": {"contest_auth_eval.stdout.log": _report(token_cache={"status": "ENABLED"})}}, "token cache"),
    ({"artifacts": {"contest_auth_eval.stdout.log": "no report here\n"}}, "cold receiver report absent"),
    ({"artifacts": {"contest_auth_eval.stdout.log": _report(pair_count=599)}}, "600 pairs"),
])
def test_t4_direct_refuses_every_missing_binding(staged, mutation, message):
    runtime, archive, root = staged
    root.mkdir()
    receipt = _t4_receipt(runtime, root / "t4.json")
    doc = json.loads(receipt.read_text())
    for key, value in mutation.items():
        if key == "artifacts":
            doc["artifacts"].update(value)
        else:
            doc[key] = value
    receipt.write_text(json.dumps(doc))
    with pytest.raises(SealContractError, match=message):
        build_t4_direct_leg(t4_receipt_path=receipt, runtime_dir=runtime, archive_path=archive)


def test_t4_direct_refuses_over_limit_and_drift(staged):
    runtime, archive, root = staged
    root.mkdir()
    slow = _t4_receipt(runtime, root / "slow.json", seconds=LIMIT_SECONDS + 0.5)
    with pytest.raises(SealContractError, match="exceeds"):
        build_t4_direct_leg(t4_receipt_path=slow, runtime_dir=runtime, archive_path=archive)
    good = _t4_receipt(runtime, root / "t4.json", seconds=900.0)
    leg = build_t4_direct_leg(t4_receipt_path=good, runtime_dir=runtime, archive_path=archive)
    # the leg's stored seconds must equal the receipt's; a retyped number is refused
    forged = {**leg, "measured_t4_decode_seconds": 800.0, "projected_t4_decode_seconds": 800.0}
    assert any("must equal the receipt" in p for p in validate_decode_wall_clock(forged, runtime_dir=runtime, archive_path=archive)[0])
    # a local denominator smuggled into a direct leg is a schema error
    smuggled = {**leg, "cpu_to_t4_ratio": 1.0}
    assert any("carry no cpu_to_t4_ratio" in p for p in validate_decode_wall_clock(smuggled, runtime_dir=runtime, archive_path=archive)[0])
    # a v1 schema on a direct leg is refused
    wrong = {**leg, "schema": "candidate_decode_wall_clock.v1"}
    assert any("v2" in p for p in validate_decode_wall_clock(wrong, runtime_dir=runtime, archive_path=archive)[0])
    # receipt drift (sha) is refused on validation
    doc = json.loads(good.read_text())
    doc["modal_elapsed_seconds"] = 1
    good.write_text(json.dumps(doc))
    assert any("receipt drift" in p for p in validate_decode_wall_clock(copy.deepcopy(leg), runtime_dir=runtime, archive_path=archive)[0])


def test_inherit_refuses_a_different_receiver(staged, tmp_path):
    runtime, archive, root = staged
    root.mkdir()
    receipt = _t4_receipt(runtime, root / "t4.json")
    leg = build_t4_direct_leg(t4_receipt_path=receipt, runtime_dir=runtime, archive_path=archive)
    leg_path = root / "leg.json"
    leg_path.write_text(json.dumps(leg))
    from tac.tests.test_candidate_seal import _stage_candidate
    other_runtime, other_archive = _stage_candidate(tmp_path / "other")
    (other_runtime / "inflate.py").write_text((other_runtime / "inflate.py").read_text() + "\n# receiver change\n")
    with pytest.raises(SealContractError, match="receiver code differs"):
        inherit_decode_wall_clock(source_leg_path=leg_path, runtime_dir=other_runtime, archive_path=other_archive,
                                  pointer_archive_sha256=leg["archive_sha256"])
