"""Pipeline fail-closed tests for the JCSP integration gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_PATH = REPO_ROOT / "experiments" / "pipeline.py"


def _load_pipeline_module(name: str):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_PATH)
    assert spec is not None
    assert spec.loader is not None
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[name] = pipeline
    spec.loader.exec_module(pipeline)
    return pipeline


def test_jcsp_flag_invalidates_cached_fp4_and_requires_marginals(
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline_module("_pipeline_jcsp_missing_marginals")
    output_dir = tmp_path / "out"
    iter_dir = output_dir / "iter_0"
    iter_dir.mkdir(parents=True)
    done = iter_dir / ".done_compress_weights"
    done.write_text(json.dumps({"mode": "fp4"}))
    stale_renderer = iter_dir / "renderer.bin"
    stale_renderer.write_bytes(b"stale-fp4")

    cfg = pipeline.PipelineConfig(
        output_dir=str(output_dir),
        use_joint_codec_stack=True,
        jcsp_score_marginals_path="",
    )

    with pytest.raises(NotImplementedError, match="missing prerequisites"):
        pipeline.step_compress_weights(
            cfg,
            tmp_path / "missing_checkpoint.pt",
            iteration=0,
        )

    assert not done.exists()
    assert stale_renderer.read_bytes() == b"stale-fp4"


def test_jcsp_present_marginals_writes_archive_member_then_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _load_pipeline_module("_pipeline_jcsp_present_marginals")
    artifact = tmp_path / "marginals.json"
    artifact.write_text(
        json.dumps(
            {"score_marginals": {"a.weight": 1e-6}},
            sort_keys=True,
        )
    )
    output_dir = tmp_path / "out"
    logs: list[tuple[str, str]] = []

    def fake_load_renderer_for_jcsp_dry_run(_cfg, _checkpoint_path):
        return {"a.weight": torch.linspace(-1.0, 1.0, 16, dtype=torch.float32)}

    monkeypatch.setattr(
        pipeline,
        "_load_renderer_for_jcsp_dry_run",
        fake_load_renderer_for_jcsp_dry_run,
    )
    monkeypatch.setattr(
        pipeline,
        "_log",
        lambda msg, level="pipeline": logs.append((level, msg)),
    )

    cfg = pipeline.PipelineConfig(
        output_dir=str(output_dir),
        use_joint_codec_stack=True,
        jcsp_score_marginals_path=str(artifact),
    )

    with pytest.raises(NotImplementedError, match="byte-closed JCSP archive member"):
        pipeline.step_compress_weights(
            cfg,
            tmp_path / "checkpoint.pt",
            iteration=0,
        )

    iter_dir = output_dir / "iter_0"
    assert iter_dir.exists()
    archive_path = iter_dir / "jcsp_archive_member.zip"
    manifest_path = iter_dir / "jcsp_archive_member_manifest.json"
    assert sorted(path.name for path in iter_dir.iterdir()) == [
        "jcsp_archive_member.zip",
        "jcsp_archive_member_manifest.json",
    ]
    assert archive_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "jcsp_stream_source_archive_member_contract_v1"
    assert manifest["score_claim"] is False
    assert manifest["ready_for_runtime_loader"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["archive_bytes_written"] is True
    assert manifest["runtime_payloads_encoded"] is True
    assert manifest["stream_count"] == 1
    assert manifest["runtime_loader_parity"]["stream_count"] == 1
    assert manifest["runtime_loader_parity"]["streams"][0]["payload_magic"] == "AQv1"
    assert manifest["stream_archive_byte_reconciliation"][
        "stream_payload_bytes_reconciled"
    ] is True
    assert not (iter_dir / ".done_compress_weights").exists()
    assert any(
        "archive_bytes_written=True" in msg
        and "ready_for_runtime_loader=True" in msg
        and "ready_for_exact_eval_dispatch=False" in msg
        for _level, msg in logs
    )
