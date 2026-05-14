# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "profile_pr85_hpm1_residual_prefix_trajectory.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "profile_pr85_hpm1_residual_prefix_trajectory_test",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_payload(module: Any) -> SimpleNamespace:
    segment = b"HPM1" + b"s" * 16
    tokens = b"tokn" * 3
    hpac = b"hpac!!"
    return SimpleNamespace(
        archive_path=None,
        segment=segment,
        tokens_blob=tokens,
        hpac_ppmd_blob=hpac,
        contract=SimpleNamespace(
            bytes=len(segment),
            sha256=module._sha256_bytes(segment),
            metadata={
                "N": 4,
                "H": 3,
                "W": 4,
                "P": 2,
                "delta": 1,
                "ch": 4,
                "use_spm": False,
                "hpac_d_film": 2,
                "tokens_len": len(tokens),
                "hpac_len": len(hpac),
                "ppmd_order": 4,
            },
        ),
        archive_report={"path": "synthetic_pr91.zip", "bytes": 123},
        bundle_report={"format": "synthetic"},
    )


def _fake_residual_prototype(module: Any):
    def _call(raw_tokens: np.ndarray, payload: Any, *, max_frames: int, **_: Any) -> dict[str, Any]:
        prefix = np.ascontiguousarray(raw_tokens[:max_frames])
        token_stream_bytes = max_frames * 10 + 4
        segment_bytes = 48 + token_stream_bytes + len(payload.hpac_ppmd_blob)
        prefix_sha = module._sha256_bytes(prefix.tobytes(order="C"))
        return {
            "status": "passed",
            "elapsed_sec": round(max_frames / 1000, 3),
            "input_tokens_sha256": prefix_sha,
            "residual_symbols_sha256": f"residual-{max_frames}",
            "residual_roundtrip_raw_tokens_sha256": prefix_sha,
            "candidate_hpm1_segment": {
                "bytes": segment_bytes,
                "sha256": f"{max_frames:064x}",
                "tokens_len": token_stream_bytes,
                "hpac_len": len(payload.hpac_ppmd_blob),
            },
            "hpm1_encode": {"symbol_context_contract": "synthetic_symbols_nhw"},
        }

    return _call


def test_load_raw_tokens_normalizes_qma9_storage_nwh_to_nhw(tmp_path: Path) -> None:
    module = _load_module()
    storage = (np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3) % 5).astype(np.uint8)
    token_path = tmp_path / "tokens.bin"
    token_path.write_bytes(storage.tobytes(order="C"))

    tokens, report = module.load_raw_tokens(
        token_path,
        shape_text="2,4,3",
        layout="qma9_storage_nwh_to_nhw",
    )

    assert tokens.shape == (2, 3, 4)
    assert np.array_equal(tokens, storage.transpose(0, 2, 1))
    assert report["storage_shape"] == [2, 4, 3]
    assert report["returned_shape"] == [2, 3, 4]
    assert report["normalization"] == "reshape_NWH_transpose_to_NHW"
    assert report["observed_range"] == {"min": 0, "max": 4}


def test_build_report_records_prefix_marginals_and_non_dispatch_flags(tmp_path: Path) -> None:
    module = _load_module()
    raw_tokens = (np.arange(4 * 3 * 4, dtype=np.uint8).reshape(4, 3, 4) % 5).astype(np.uint8)
    raw_source = {
        "path": "synthetic_tokens.bin",
        "bytes": raw_tokens.nbytes,
        "sha256": module._sha256_bytes(raw_tokens.tobytes(order="C")),
        "normalized_nhw_sha256": module._sha256_bytes(raw_tokens.tobytes(order="C")),
    }

    report = module.build_report(
        raw_tokens_nhw=raw_tokens,
        raw_token_source=raw_source,
        source_payload=_fake_payload(module),
        source_archive=tmp_path / "synthetic_pr91.zip",
        frame_counts=[1, 2, 4],
        prototype_fn=_fake_residual_prototype(module),
    )

    assert report["planning_only"] is True
    assert report["score_claim"] is False
    assert report["dispatch_unlocked"] is False
    assert report["gpu_or_remote_work"] is False
    rows = report["trajectory"]
    assert [row["frame_count"] for row in rows] == [1, 2, 4]
    assert [row["token_bytes"] for row in rows] == [12, 24, 48]
    assert [row["candidate_hpm1_segment_bytes"] for row in rows] == [68, 78, 98]
    assert rows[0]["marginal_vs_previous_prefix"] is None
    assert rows[1]["marginal_vs_previous_prefix"]["candidate_hpm1_segment_bytes_delta"] == 10
    assert rows[2]["marginal_vs_previous_prefix"]["candidate_hpm1_segment_bytes_per_added_frame"] == 10
    assert rows[2]["residual_roundtrip_raw_tokens_sha256"] == rows[2]["input_tokens_sha256"]


def test_main_writes_json_and_markdown_with_synthetic_payload(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_module()
    storage = (np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3) % 5).astype(np.uint8)
    token_path = tmp_path / "tokens.bin"
    token_path.write_bytes(storage.tobytes(order="C"))
    out_json = tmp_path / "profile.json"
    out_md = tmp_path / "profile.md"

    monkeypatch.setattr(module, "extract_pr91_hpm1_payload", lambda _archive: _fake_payload(module))
    monkeypatch.setattr(
        module,
        "prototype_reencode_hpm1_residual_from_raw_tokens",
        _fake_residual_prototype(module),
    )

    rc = module.main(
        [
            "--archive",
            str(tmp_path / "pr91.zip"),
            "--raw-token-bin",
            str(token_path),
            "--raw-token-shape",
            "4,4,3",
            "--frame-counts",
            "1,2",
            "--json-out",
            str(out_json),
            "--md-out",
            str(out_md),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "pr85_hpm1_residual_prefix_trajectory_v1"
    assert payload["score_claim"] is False
    assert payload["dispatch_unlocked"] is False
    assert payload["trajectory"][1]["frame_count"] == 2
    assert "PR85 -> HPM1 Residual Prefix Trajectory" in out_md.read_text(encoding="utf-8")
    assert "score_claim=false dispatch_unlocked=false" in capsys.readouterr().out
