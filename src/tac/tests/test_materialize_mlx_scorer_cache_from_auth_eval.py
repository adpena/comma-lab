# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools" / "materialize_mlx_scorer_cache_from_auth_eval.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "materialize_mlx_scorer_cache_from_auth_eval_under_test",
        TOOL_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_custody_accepts_matching_content_tree(monkeypatch, tmp_path: Path) -> None:
    tool = _load_tool()
    inflate_sh = tmp_path / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(tool, "_sha256", lambda path, prefix=0: "s" * 64)
    monkeypatch.setattr(
        tool,
        "_runtime_dependency_manifest",
        lambda inflate, upstream: {
            "runtime_tree_sha256": "l" * 64,
            "runtime_content_tree_sha256": "c" * 64,
            "runtime_file_count": 3,
        },
    )

    result = tool._validate_runtime_custody(
        {
            "provenance": {
                "inflate_script_sha256": "s" * 64,
                "inflate_runtime_manifest": {
                    "runtime_tree_sha256": "r" * 64,
                    "runtime_content_tree_sha256": "c" * 64,
                },
            }
        },
        {"expected_runtime_tree_sha256": "r" * 64},
        inflate_sh=inflate_sh,
        upstream_dir=tmp_path,
    )

    assert result["comparison"] == "runtime_content_tree_sha256"
    assert result["local_runtime_content_tree_sha256"] == "c" * 64


def test_runtime_custody_rejects_content_tree_mismatch(monkeypatch, tmp_path: Path) -> None:
    tool = _load_tool()
    inflate_sh = tmp_path / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(tool, "_sha256", lambda path, prefix=0: "s" * 64)
    monkeypatch.setattr(
        tool,
        "_runtime_dependency_manifest",
        lambda inflate, upstream: {
            "runtime_tree_sha256": "l" * 64,
            "runtime_content_tree_sha256": "d" * 64,
        },
    )

    with pytest.raises(SystemExit, match="runtime content-tree SHA mismatch"):
        tool._validate_runtime_custody(
            {
                "provenance": {
                    "inflate_script_sha256": "s" * 64,
                    "inflate_runtime_manifest": {
                        "runtime_tree_sha256": "r" * 64,
                        "runtime_content_tree_sha256": "c" * 64,
                    },
                }
            },
            {},
            inflate_sh=inflate_sh,
            upstream_dir=tmp_path,
        )


def test_inflate_policy_refuses_nondefault_replay() -> None:
    tool = _load_tool()

    with pytest.raises(SystemExit, match="non-default inflate_device_policy"):
        tool._validate_default_inflate_policy(
            {"provenance": {"inflate_device_policy": "cpu"}}
        )


def test_failed_audit_deletes_materialized_cache(tmp_path: Path) -> None:
    tool = _load_tool()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text("{}", encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"passed": False}), encoding="utf-8")

    manifest = tool._finalize_cache_after_audit(
        output_cache=cache_dir,
        audit_output=audit_path,
        audit={"passed": False, "verdict": "FAIL_CACHE_AUTH_EVAL_IDENTITY"},
        auth_dir=tmp_path / "auth",
        auth_eval_path=tmp_path / "auth" / "contest_auth_eval.json",
    )

    assert manifest == {}
    assert not cache_dir.exists()


def test_passing_audit_stamps_materialized_cache_manifest(monkeypatch, tmp_path: Path) -> None:
    tool = _load_tool()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "mlx_scorer_input_cache.v1"}),
        encoding="utf-8",
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"passed": True}), encoding="utf-8")
    monkeypatch.setattr(tool, "_sha256", lambda path, prefix=0: "a" * 64)

    manifest = tool._finalize_cache_after_audit(
        output_cache=cache_dir,
        audit_output=audit_path,
        audit={
            "schema_version": "mlx_scorer_input_cache_auth_eval_audit.v1",
            "passed": True,
            "verdict": "PASS_CACHE_AUTH_EVAL_IDENTITY",
            "identity_residual": 0,
        },
        auth_dir=tmp_path / "auth",
        auth_eval_path=tmp_path / "auth" / "contest_auth_eval.json",
    )

    assert manifest["eligible_for_local_mlx_transfer_calibration"] is True
    assert manifest["auth_eval_identity_audit"]["verdict"] == "PASS_CACHE_AUTH_EVAL_IDENTITY"
    assert manifest["auth_eval_identity_audit"]["sha256"] == "a" * 64
    persisted = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest
