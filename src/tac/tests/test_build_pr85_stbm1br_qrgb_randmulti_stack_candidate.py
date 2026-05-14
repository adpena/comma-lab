# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


brotli = pytest.importorskip("brotli")

from tac.pr85_bundle import PR85_HEADERLESS_RANDMULTI_SPECS, SEGMENT_ORDER, pack_pr85_bundle, parse_pr85_bundle
from tac.stbm1br_mask_codec import STBM1BR_MAGIC


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_stbm1br_qrgb_randmulti_stack_candidate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_stbm1br_qrgb_stack_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()
qrgb = module.qrgb_builder


def _zip_info(name: str = "x") -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, qrgb.FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_archive(path: Path, *, mask: bytes) -> Path:
    randmulti_rows = [
        [bytearray(qrgb.PAIR_COUNT) for _ in range(row_count)]
        for _lh, _lw, _amp, row_count in PR85_HEADERLESS_RANDMULTI_SPECS
    ]
    randmulti, _meta = qrgb._encode_randmulti_stream(randmulti_rows)
    segments = {
        "mask": mask,
        "model": brotli.compress(b"QH0" + b"R" * 64, quality=5),
        "pose": brotli.compress(b"P1D1" + bytes([1, 0]) + (600).to_bytes(2, "little") + bytes(600), quality=5),
        "post": brotli.compress(bytes(qrgb.PAIR_COUNT * 4), quality=5),
        "shift": brotli.compress(b"SD4" + bytes(qrgb.PAIR_COUNT), quality=5),
        "frac": brotli.compress(b"FH1" + bytes([4]) * qrgb.PAIR_COUNT, quality=5),
        "frac2": brotli.compress(b"FH2" + bytes([4]) * qrgb.PAIR_COUNT, quality=5),
        "frac3": brotli.compress(b"FD3" + bytes(qrgb.PAIR_COUNT), quality=5),
        "bias": b"B" * 223,
        "region": b"R" * 273,
        "randmulti": randmulti,
    }
    assert set(segments) == set(SEGMENT_ORDER)
    raw = pack_pr85_bundle(segments, header_mode="v5")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("x"), raw)
    return path


def _archive_member(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        return zf.read("x")


def _archive_meta(path: Path) -> tuple[dict[str, Any], bytes]:
    return qrgb._read_source_archive(path)


def _basis() -> dict[str, Any]:
    return {
        "target_stream": "randmulti",
        "target_randmulti_group": {
            "group_index": 61,
            "lh": 222,
            "lw": 222,
            "amp": 4,
            "selection_row": 0,
        },
    }


def _candidate(candidate_id: str = module.DEFAULT_CANDIDATE_ID) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "header_mode": "explicit_30",
        "actions": [
            {
                "op": "set",
                "pair_index": 192,
                "stream": "randmulti",
                "source_value": 0,
                "candidate_value": 20,
                "source_artifact_sha256": "a" * 64,
                "source_atom_id": "fixture:pair_0192",
                "rationale": "fixture QRGB transfer action",
            }
        ],
        "score_claim": False,
    }


def _write_qrgb_inputs(tmp_path: Path, pr85_archive: Path) -> tuple[Path, Path]:
    pr85_sha = module._sha256_file(pr85_archive)
    candidate = _candidate()
    action = candidate["actions"][0]
    evidence = {
        "schema": qrgb.PAIR_ACTION_EVIDENCE_SCHEMA,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "candidates": [candidate],
    }
    plan = {
        "schema": qrgb.TRANSFER_PLAN_SCHEMA,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "pair_planning_source": {"source_sha_for_pair_action": "a" * 64},
        "ranked_candidates": [
            {
                "candidate_id": module.DEFAULT_CANDIDATE_ID,
                "rank": 1,
                "can_feed_pair_action_evidence": True,
                "source_archive_bytes": pr85_archive.stat().st_size,
                "source_archive_sha256": pr85_sha,
                "pr90_source_archive_sha256": "b" * 64,
                "pr90_source_evidence_sha256": "c" * 64,
                "source_atom_id": action["source_atom_id"],
                "actions": [dict(action)],
                "basis_action_schema": _basis(),
                "expected_break_even": {"combined_max_charged_bytes_for_zero_net_change": 100.0},
                "qrgb_pair_summary": {"pair_index": 192},
                "score_claim": False,
                "dispatch_unlocked": False,
            }
        ],
    }
    return (
        _write_json(tmp_path / "pair_action_evidence.json", evidence),
        _write_json(tmp_path / "planning.json", plan),
    )


def _write_stbm_manifest(path: Path, *, pr85_meta: dict[str, Any], stbm_meta: dict[str, Any], render_sha: str) -> Path:
    source_mask = parse_pr85_bundle(_archive_member(REPO / pr85_meta["path"] if not Path(pr85_meta["path"]).is_absolute() else Path(pr85_meta["path"]))).segments["mask"]
    candidate_mask = parse_pr85_bundle(_archive_member(REPO / stbm_meta["path"] if not Path(stbm_meta["path"]).is_absolute() else Path(stbm_meta["path"]))).segments["mask"]
    payload = {
        "schema": "fixture_stbm_manifest_v1",
        "candidate_id": "pr90_stbm1br_lossless_pr85_mask_recode",
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "source_archive": pr85_meta,
        "candidate_archive": stbm_meta,
        "segments": {
            "source_mask": {
                "codec": "QMA9",
                "bytes": len(source_mask),
                "sha256": module._sha256_bytes(bytes(source_mask)),
            },
            "candidate_mask": {
                "codec": "STBM1BR_fixture",
                "bytes": len(candidate_mask),
                "sha256": module._sha256_bytes(bytes(candidate_mask)),
            },
        },
        "parity": {
            "decoded_mask_equal": True,
            "diff_pixels": 0,
            "pr85_render_order_sha256": render_sha,
            "candidate_render_order_sha256": render_sha,
        },
        "fail_closed_preflight": {
            "status": "passed",
            "checks": {
                "candidate_archive_byte_positive": True,
                "candidate_mask_byte_positive": True,
                "candidate_non_noop_at_byte_level": True,
                "decoded_mask_equal": True,
                "decoded_render_order_sha_matches_expected": True,
                "no_scorer_load": True,
                "remote_dispatch_not_performed": True,
                "runtime_support_present": True,
                "single_member_x_only": True,
            },
        },
    }
    return _write_json(path, payload)


def _write_qrgb_standalone(path: Path) -> Path:
    payload = {
        "candidate_id": module.DEFAULT_CANDIDATE_ID,
        "candidate_archive": {
            "archive_bytes": module.EXPECTED_QRGB_STANDALONE["archive_bytes"],
            "archive_sha256": module.EXPECTED_QRGB_STANDALONE["archive_sha256"],
        },
        "selected_streams": ["randmulti"],
        "selected_pair_indices": [192],
        "action_proofs": [
            {
                "stream": "randmulti",
                "pair_index": 192,
                "source_value": 0,
                "candidate_value": 20,
            }
        ],
        "fixed_runtime_preflight": {"ready_for_fixed_runtime_exact_eval": True},
    }
    return _write_json(path, payload)


def _ready_preflight(archive: Path, _runtime: Path, **_kwargs) -> dict[str, Any]:
    raw = _archive_member(archive)
    bundle = parse_pr85_bundle(raw)
    mask_sha = module._sha256_bytes(bytes(bundle.segments["mask"]))
    return {
        "ready_for_fixed_runtime_exact_eval": False,
        "readiness_status": "blocked",
        "blockers": [{"code": "pr85_segment_probe_failed", "severity": "blocking", "detail": "fixture STBM mask"}],
        "custody_expectations": {"remaining_blockers": []},
        "atom_edit_guard": {
            "status": "passed",
            "changed_segments": [{"segment": "randmulti"}],
        },
        "fixed_runtime_bridge": {
            "expansion_available": True,
            "remaining_blockers": [],
            "expansion_manifest": {
                "runtime_members": {
                    "masks.qma9": {
                        "sha256": mask_sha,
                        "bytes": len(bundle.segments["mask"]),
                    }
                }
            },
        },
    }


def _runtime_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "inflate_renderer.py").write_text(
        "STBM1BR_MAGIC = b'STBM1BR\\\\0'\n"
        "def _load_masks_from_stbm1br(path, expected_frames=None): return path\n"
        "def _load_masks_from_archive(mask_video_path):\n"
        "    if mask_video_path.suffix.lower() == \".qma9\" and STBM1BR_MAGIC:\n"
        "        return _load_masks_from_stbm1br(mask_video_path)\n"
        "    return mask_video_path\n"
        "# masks.stbm1br\n",
        encoding="utf-8",
    )
    return path


def test_stack_builder_emits_mask_plus_randmulti_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    render_sha = "d" * 64
    pr85_archive = _write_archive(tmp_path / "pr85.zip", mask=b"QMA9" + b"M" * 64)
    pr85_meta, pr85_raw = _archive_meta(pr85_archive)
    stbm_segments = dict(parse_pr85_bundle(pr85_raw).segments)
    stbm_segments["mask"] = STBM1BR_MAGIC + b"S" * 32
    stbm_archive = tmp_path / "stbm.zip"
    with zipfile.ZipFile(stbm_archive, "w") as zf:
        zf.writestr(_zip_info("x"), pack_pr85_bundle(stbm_segments, header_mode="v5"))
    stbm_meta, _stbm_raw = _archive_meta(stbm_archive)
    stbm_manifest = _write_stbm_manifest(
        tmp_path / "stbm_manifest.json",
        pr85_meta=pr85_meta,
        stbm_meta=stbm_meta,
        render_sha=render_sha,
    )
    qrgb_standalone = _write_qrgb_standalone(tmp_path / "qrgb_manifest.json")
    evidence, plan = _write_qrgb_inputs(tmp_path, pr85_archive)
    stbm_report = module._validate_stbm_manifest(
        stbm_manifest_path=stbm_manifest,
        stbm_manifest=json.loads(stbm_manifest.read_text(encoding="utf-8")),
        stbm_archive=stbm_meta,
        pr85_archive=pr85_meta,
        expected_render_order_sha256=render_sha,
    )
    override = module.create_reviewed_stack_override_manifest(
        output_path=tmp_path / "override.json",
        pr85_archive_meta=pr85_meta,
        stbm_archive_meta=stbm_meta,
        stbm_manifest_report=stbm_report,
        qrgb_candidate_id=module.DEFAULT_CANDIDATE_ID,
        reviewed_by="pytest",
    )
    assert override["reviewed"] is True
    monkeypatch.setattr(
        module.qrgb_builder,
        "KNOWN_PR85",
        {"archive_bytes": pr85_archive.stat().st_size, "archive_sha256": module._sha256_file(pr85_archive), "score": 0.0},
    )
    monkeypatch.setattr(module, "EXPECTED_STBM", {"archive_bytes": stbm_archive.stat().st_size, "archive_sha256": module._sha256_file(stbm_archive), "render_order_sha256": render_sha, "diff_pixels": 0})
    monkeypatch.setattr(module.fixed_preflight, "build_preflight", _ready_preflight)

    summary = module.build_pr85_stbm1br_qrgb_randmulti_stack_candidate(
        pr85_archive=pr85_archive,
        stbm_archive=stbm_archive,
        stbm_manifest_path=stbm_manifest,
        qrgb_standalone_manifest_path=qrgb_standalone,
        pair_action_evidence_json=evidence,
        transfer_plan_json=plan,
        reviewed_stack_override_manifest=tmp_path / "override.json",
        out_dir=tmp_path / "out",
        robust_current_dir=_runtime_dir(tmp_path / "runtime"),
        expected_render_order_sha256=render_sha,
    )

    assert summary["candidate_count"] == 1
    assert summary["dispatch_unlocked"] is False
    assert summary["exact_eval_safe_after_standalone_exact_positives"] is True
    manifest = json.loads((tmp_path / "out" / module.STACK_CANDIDATE_ID / "manifest.json").read_text(encoding="utf-8"))
    assert [step["transform_id"] for step in manifest["source_transform_chain"]] == [
        "pr90_stbm1br_lossless_pr85_mask_recode",
        "pr85_qrgb_f2_randglobal_pair_0192",
    ]
    assert manifest["orthogonality"]["status"] == "passed"
    assert manifest["orthogonality"]["stack_vs_stbm_changed_segments"][0]["segment"] == "randmulti"
    archive = REPO / manifest["candidate_archive"]["archive_path"]
    parsed = parse_pr85_bundle(_archive_member(archive))
    assert parsed.segments["mask"] == STBM1BR_MAGIC + b"S" * 32
    groups, _meta = qrgb._decode_randmulti_stream(parsed.segments["randmulti"])
    assert groups[61][0][192] == 20


def test_stack_builder_requires_reviewed_override_for_source_sha_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    render_sha = "e" * 64
    pr85_archive = _write_archive(tmp_path / "pr85.zip", mask=b"QMA9" + b"M" * 64)
    pr85_meta, pr85_raw = _archive_meta(pr85_archive)
    stbm_segments = dict(parse_pr85_bundle(pr85_raw).segments)
    stbm_segments["mask"] = STBM1BR_MAGIC + b"S" * 32
    stbm_archive = tmp_path / "stbm.zip"
    with zipfile.ZipFile(stbm_archive, "w") as zf:
        zf.writestr(_zip_info("x"), pack_pr85_bundle(stbm_segments, header_mode="v5"))
    stbm_meta, _ = _archive_meta(stbm_archive)
    stbm_manifest = _write_stbm_manifest(
        tmp_path / "stbm_manifest.json",
        pr85_meta=pr85_meta,
        stbm_meta=stbm_meta,
        render_sha=render_sha,
    )
    qrgb_standalone = _write_qrgb_standalone(tmp_path / "qrgb_manifest.json")
    evidence, plan = _write_qrgb_inputs(tmp_path, pr85_archive)
    monkeypatch.setattr(
        module.qrgb_builder,
        "KNOWN_PR85",
        {"archive_bytes": pr85_archive.stat().st_size, "archive_sha256": module._sha256_file(pr85_archive), "score": 0.0},
    )
    monkeypatch.setattr(module, "EXPECTED_STBM", {"archive_bytes": stbm_archive.stat().st_size, "archive_sha256": module._sha256_file(stbm_archive), "render_order_sha256": render_sha, "diff_pixels": 0})

    with pytest.raises(module.StackBuildError, match="reviewed stack override manifest is required"):
        module.build_pr85_stbm1br_qrgb_randmulti_stack_candidate(
            pr85_archive=pr85_archive,
            stbm_archive=stbm_archive,
            stbm_manifest_path=stbm_manifest,
            qrgb_standalone_manifest_path=qrgb_standalone,
            pair_action_evidence_json=evidence,
            transfer_plan_json=plan,
            reviewed_stack_override_manifest=None,
            out_dir=tmp_path / "out",
            robust_current_dir=_runtime_dir(tmp_path / "runtime"),
            expected_render_order_sha256=render_sha,
        )
