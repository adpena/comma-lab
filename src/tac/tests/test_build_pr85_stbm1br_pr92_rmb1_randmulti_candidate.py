# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import brotli

from tac.pr85_bundle import (
    FIXED_V5_LENGTHS,
    PR85_HEADERLESS_RANDMULTI_SPECS,
    SEGMENT_ORDER,
    pack_pr85_bundle,
)

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_rmb1_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _sha(data: bytes) -> str:
    return module._sha256_bytes(data)


def _zip_x(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)
    return path


def _zero_randmulti_rows() -> bytes:
    row_count = sum(int(spec[3]) for spec in PR85_HEADERLESS_RANDMULTI_SPECS)
    return b"\x00" * row_count


def _rmb1_zero_randmulti(row_count: int) -> bytes:
    mask = b"\x00" * (row_count * 75)
    mask_br = brotli.compress(mask, quality=11)
    vals_br = brotli.compress(b"", quality=11)
    return b"RMB1" + len(mask_br).to_bytes(2, "little") + mask_br + vals_br


def _segments(*, mask: bytes, randmulti: bytes) -> dict[str, bytes]:
    def br(payload: bytes) -> bytes:
        return brotli.compress(payload, quality=6)

    segments = {
        "mask": mask,
        "model": br(b"QH0-model"),
        "pose": br(b"P1D1-pose"),
        "post": br(b"post"),
        "shift": br(b"shift"),
        "frac": br(b"frac"),
        "frac2": br(b"frac2"),
        "frac3": br(b"frac3"),
        "bias": b"b" * FIXED_V5_LENGTHS["bias"],
        "region": b"r" * FIXED_V5_LENGTHS["region"],
        "randmulti": randmulti,
    }
    assert set(segments) == set(SEGMENT_ORDER)
    return segments


def _write_runtime(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "inflate.sh").write_text("#!/usr/bin/env bash\npython inflate.py\n", encoding="utf-8")
    (path / "inflate.py").write_text(
        "import brotli\n"
        "def load_stbm1br_mask():\n"
        "    return b'STBM1BR'\n"
        "def main():\n"
        "    bundle = {'randmulti': b''}\n"
        "    if False:\n"
        "        raw_n = brotli.decompress(bundle[\"randmulti\"])\n",
        encoding="utf-8",
    )
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_synthetic_pr85_stbm_pr92_rmb1_candidate_builds_without_score_claim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = _zero_randmulti_rows()
    stbm_randmulti = brotli.compress(rows, quality=11)
    rmb1_randmulti = _rmb1_zero_randmulti(len(rows))
    pr85_mask = b"QMA9-pr85-mask"
    stbm_mask = b"QMA9-stbm-mask"

    pr85_payload = pack_pr85_bundle(
        _segments(mask=pr85_mask, randmulti=stbm_randmulti), header_mode="v5"
    )
    stbm_payload = pack_pr85_bundle(
        _segments(mask=stbm_mask, randmulti=stbm_randmulti), header_mode="v5"
    )
    pr92_payload = pack_pr85_bundle(
        _segments(mask=pr85_mask, randmulti=rmb1_randmulti), header_mode="v5"
    )
    pr85_archive = _zip_x(tmp_path / "pr85.zip", pr85_payload)
    stbm_archive = _zip_x(tmp_path / "stbm.zip", stbm_payload)
    pr92_archive = _zip_x(tmp_path / "pr92.zip", pr92_payload)
    stbm_manifest = _write_json(
        tmp_path / "stbm_manifest.json",
        {
            "score_claim": False,
            "dispatch_performed": False,
            "candidate_archive": {
                "archive_sha256": module._sha256_file(stbm_archive),
            },
            "source_archive": {
                "archive_sha256": module._sha256_file(pr85_archive),
            },
            "parity": {
                "decoded_mask_equal": True,
                "diff_pixels": 0,
                "candidate_render_order_sha256": "render-order",
            },
            "exact_eval_runtime_contract": {
                "ready_for_exact_eval_runtime": True,
                "runtime_tree_sha256": "runtime-tree",
            },
            "segments": {
                "candidate_mask": {
                    "sha256": _sha(stbm_mask),
                }
            },
        },
    )
    pr92_profile = _write_json(
        tmp_path / "pr92_profile.json",
        {
            "score_claim": False,
            "promotion_eligible": False,
            "label": "synthetic-pr92",
            "evidence_grade": "external",
            "side_info": {"charged_bytes": len(rmb1_randmulti)},
            "primary_member": {
                "segments": [
                    {
                        "name": "randmulti",
                        "bytes": len(rmb1_randmulti),
                        "sha256": _sha(rmb1_randmulti),
                        "codec": "RMB1_side_info_backed_randmulti",
                    }
                ]
            },
        },
    )
    stbm_exact = _write_json(
        tmp_path / "stbm_exact.json",
        {
            "archive_size_bytes": stbm_archive.stat().st_size,
            "n_samples": 600,
            "canonical_score": 0.33,
            "avg_posenet_dist": 1e-4,
            "avg_segnet_dist": 1e-4,
            "provenance": {
                "archive_sha256": module._sha256_file(stbm_archive),
                "device": "cuda",
                "gpu_t4_match": True,
                "inflate_runtime_manifest": {
                    "runtime_tree_sha256": "runtime-tree",
                },
            },
        },
    )
    monkeypatch.setitem(module.EXPECTED, "pr85_archive_bytes", pr85_archive.stat().st_size)
    monkeypatch.setitem(module.EXPECTED, "pr85_archive_sha256", module._sha256_file(pr85_archive))
    monkeypatch.setitem(module.EXPECTED, "stbm_archive_bytes", stbm_archive.stat().st_size)
    monkeypatch.setitem(module.EXPECTED, "stbm_archive_sha256", module._sha256_file(stbm_archive))
    monkeypatch.setitem(module.EXPECTED, "pr92_archive_bytes", pr92_archive.stat().st_size)
    monkeypatch.setitem(module.EXPECTED, "pr92_archive_sha256", module._sha256_file(pr92_archive))
    monkeypatch.setitem(module.EXPECTED, "stbm_randmulti_bytes", len(stbm_randmulti))
    monkeypatch.setitem(module.EXPECTED, "pr92_rmb1_randmulti_bytes", len(rmb1_randmulti))
    monkeypatch.setitem(module.EXPECTED, "pr92_rmb1_randmulti_sha256", _sha(rmb1_randmulti))
    monkeypatch.setitem(module.EXPECTED, "decoded_randmulti_rows_bytes", len(rows))
    monkeypatch.setitem(module.EXPECTED, "decoded_randmulti_rows_sha256", _sha(rows))
    monkeypatch.setitem(module.EXPECTED, "stbm_mask_sha256", _sha(stbm_mask))

    summary = module.build_candidate(
        pr85_archive=pr85_archive,
        stbm_archive=stbm_archive,
        stbm_manifest=stbm_manifest,
        pr92_archive=pr92_archive,
        pr92_profile=pr92_profile,
        stbm_replay_runtime=_write_runtime(tmp_path / "runtime"),
        stbm_exact_t4_json=stbm_exact,
        out_dir=tmp_path / "out",
    )

    manifest = json.loads(
        (tmp_path / "out" / module.CANDIDATE_ID / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["score_claim"] is False
    assert summary["remote_jobs_dispatched"] is False
    assert manifest["randmulti_decoded_row_parity"]["decoded_rows_match"] is True
    assert manifest["non_noop_byte_change"]["changed_segments_vs_stbm"] == ["randmulti"]
    assert manifest["stbm_mask_preservation"]["unchanged"] is True
    assert manifest["exact_eval_runtime_contract"]["ready_for_exact_eval_runtime"] is True
    assert manifest["dispatch_readiness"]["checks"]["stbm_standalone_exact_t4_positive_present"] is True
    assert manifest["dispatch_readiness"]["lane_id"] == module.LANE_ID
    assert "${PREDICTED_ETA_UTC}" in manifest["dispatch_readiness"]["next_claim_command"]

    first_summary_text = (tmp_path / "out" / "candidate_summary.json").read_text(
        encoding="utf-8"
    )
    first_manifest_text = (
        tmp_path / "out" / module.CANDIDATE_ID / "manifest.json"
    ).read_text(encoding="utf-8")
    repeat_summary = module.build_candidate(
        pr85_archive=pr85_archive,
        stbm_archive=stbm_archive,
        stbm_manifest=stbm_manifest,
        pr92_archive=pr92_archive,
        pr92_profile=pr92_profile,
        stbm_replay_runtime=tmp_path / "runtime",
        stbm_exact_t4_json=stbm_exact,
        out_dir=tmp_path / "out",
    )
    assert repeat_summary == summary
    assert (tmp_path / "out" / "candidate_summary.json").read_text(
        encoding="utf-8"
    ) == first_summary_text
    assert (tmp_path / "out" / module.CANDIDATE_ID / "manifest.json").read_text(
        encoding="utf-8"
    ) == first_manifest_text
