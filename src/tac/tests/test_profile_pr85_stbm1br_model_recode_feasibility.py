from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

from tac.pr85_bundle import FIXED_V5_LENGTHS, SEGMENT_ORDER, pack_pr85_bundle


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_pr85_stbm1br_model_recode_feasibility.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("profile_pr85_stbm_model_recode_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _write_archive(path: Path, segments: dict[str, bytes]) -> None:
    payload = pack_pr85_bundle(segments, header_mode="v5")
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _segments() -> dict[str, bytes]:
    return {
        "mask": b"STBM1BR\0" + b"m" * 24,
        "model": b"QH0" + b"r" * 32,
        "pose": b"P1D1" + b"p" * 16,
        "post": b"post" * 5,
        "shift": b"shift" * 3,
        "frac": b"frac" * 3,
        "frac2": b"frac2" * 3,
        "frac3": b"frac3" * 3,
        "bias": b"b" * FIXED_V5_LENGTHS["bias"],
        "region": b"r" * FIXED_V5_LENGTHS["region"],
        "randmulti": b"z" * 11,
    }


def test_candidate_archive_audit_rejects_non_model_changes(tmp_path: Path) -> None:
    source_segments = _segments()
    candidate_segments = dict(source_segments)
    candidate_segments["model"] = b"QH0" + b"s" * 32
    candidate_segments["randmulti"] = b"y" * 11
    archive = tmp_path / "candidate.zip"
    _write_archive(archive, candidate_segments)

    audit = module._candidate_archive_audit(
        candidate_archive=archive,
        source_segments=source_segments,
    )

    assert audit["model_changed"] is True
    assert audit["non_model_segments_preserved"] is False
    assert audit["forbidden_changed_segments"] == ["randmulti"]


def test_real_stbm1br_model_recode_profile_blocks_exact_eval(tmp_path: Path) -> None:
    archive = (
        REPO
        / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/"
        "pr90_stbm1br_lossless_pr85_mask_recode/archive.zip"
    )
    replay = (
        REPO
        / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/"
        "replay_submission_stbm/inflate.py"
    )
    pr90_probe = REPO / "experiments/results/public_pr90_intake_20260504_worker/payload_probe.json"
    if not archive.is_file() or not replay.is_file() or not pr90_probe.is_file():
        pytest.skip("PR85_STBM1BR or PR90 local intake artifacts are not present")

    profile = module.build_feasibility_profile(
        archive=archive,
        out_dir=tmp_path / "profile",
        replay_inflate_py=replay,
        qh0_qualities=(0, 10, 11),
        qh0_lgwins=(18, 24),
        qfq4_qrow_policies=("shifted_int8_rows",),
    )

    assert profile["dispatch"] is False
    assert profile["source_archive"]["archive_bytes"] == 229756
    assert profile["source_bundle"]["segment_lengths"]["model"] == 57074
    assert profile["source_bundle"]["segment_lengths"]["mask"] == 152439
    assert profile["owned_surface"] == "model"
    assert set(profile["forbidden_surfaces"]) == set(SEGMENT_ORDER) - {"model"}
    assert profile["model_only_candidate_guard"]["passed"] is True
    assert profile["model_only_candidate_guard"]["candidate_archive_audits"] == []

    qh0 = profile["qh0_qm0_screen"]["best_byte_screen"]
    assert qh0["candidate_id"] == "qh0_canonical_source_passthrough"
    assert qh0["model_delta_bytes_vs_source"] == 0
    assert qh0["decoded_tensor_parity"] is True

    qfq4 = profile["qfq4_screen"]["best_byte_screen"]
    assert qfq4["candidate_id"] == "qfq4_pr85_shifted_int8_rows"
    assert qfq4["model_delta_bytes_vs_source"] == -659
    assert qfq4["projected_archive_bytes_if_components_identical_formula_only"] == 229097
    assert qfq4["decoded_tensor_parity"]["decoded_tensor_parity"] is False
    assert "decoded_model_tensor_parity_failed" in qfq4["build_blockers"]
    assert "pr85_runtime_missing_qfq4_loader" in qfq4["build_blockers"]

    readiness = profile["exact_eval_readiness"]
    assert readiness["ready_after_lane_claim"] is False
    assert "no_byte_closed_model_recode_candidate_built" in readiness["blockers"]
    assert "qfq4:tensor_parity_failed_and_runtime_incompatible" in readiness["blockers"]

    blocker = json.loads(
        (tmp_path / "profile" / "dispatch_blocker.json").read_text(encoding="utf-8")
    )
    assert blocker["dispatch"] is False
    assert blocker["candidate_archive_emitted"] is False
    assert blocker["model_only_segment_gate"]["passed"] is True
    assert blocker["qfq4_contract"]["decision"] == "fail_closed_no_archive_candidate"
    assert blocker["qfq4_contract"]["decoded_tensor_parity_gate"]["decoded_tensor_parity"] is False
    assert blocker["runtime_output_parity_gate"]["passed"] is False
    assert "local_renderer_output_parity_on_source_vs_candidate" in blocker["required_before_dispatch"]
