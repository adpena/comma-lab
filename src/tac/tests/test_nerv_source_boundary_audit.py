# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from tac.analysis.nerv_source_boundary_audit import (
    NERV_SOURCE_BOUNDARY_AUDIT_SCHEMA,
    audit_nerv_source_boundary,
)
from tac.analysis.nerv_witness_readiness_dag import build_nerv_witness_readiness_dag
from tac.repo_io import json_text

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_source_boundary_audit_accepts_compact_eval_source_with_charged_archive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inflate.py"
    source.write_text(
        "from pathlib import Path\n"
        "def inflate(archive_dir, output_dir):\n"
        "    Path(output_dir).mkdir(parents=True, exist_ok=True)\n",
        encoding="utf-8",
    )
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("payload.bin", b"learned-bytes-are-charged")

    report = audit_nerv_source_boundary(
        source_paths=[source],
        archive_zip=archive,
        mode="conservative",
    )

    assert report["schema"] == NERV_SOURCE_BOUNDARY_AUDIT_SCHEMA
    assert report["source_boundary_clean"] is True
    assert report["ready_for_witness_compile"] is True
    assert report["score_claim"] is False
    assert report["archive_zip"]["bytes"] > 0
    assert report["blockers"] == []


def test_source_boundary_audit_blocks_large_uncharged_literal(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inflate.py"
    source.write_text(
        "PAYLOAD = '" + ("a" * 4096) + "'\n",
        encoding="utf-8",
    )

    report = audit_nerv_source_boundary(
        source_paths=[source],
        mode="aggressive",
        large_literal_bytes=1024,
    )

    assert report["source_boundary_clean"] is False
    assert any("large_hex_literal_in_eval_source" in item for item in report["blockers"])
    assert report["ready_for_exact_eval_dispatch"] is False


def test_witness_dag_consumes_source_boundary_audit_report(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inflate.py"
    source.write_text("def main():\n    return 0\n", encoding="utf-8")
    audit = audit_nerv_source_boundary(source_paths=[source], mode="aggressive")
    audit_path = tmp_path / "nerv_source_boundary_audit.json"
    audit_path.write_text(json_text(audit), encoding="utf-8")

    payload = build_nerv_witness_readiness_dag(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "dag_out",
        source_boundary_audit_report=audit_path,
    )

    nodes = {row["node_id"]: row for row in payload["gate_nodes"]}
    assert nodes["shared.source_boundary_compliance_audit"]["status"] == "succeeded"
    assert payload["source_boundary_evidence"]["source_boundary_clean"] is True


def test_witness_dag_keeps_source_boundary_blocked_on_unclean_report(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inflate.py"
    source.write_text("PAYLOAD = '" + ("b" * 4096) + "'\n", encoding="utf-8")
    audit = audit_nerv_source_boundary(
        source_paths=[source],
        mode="aggressive",
        large_literal_bytes=1024,
    )
    audit_path = tmp_path / "nerv_source_boundary_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    payload = build_nerv_witness_readiness_dag(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "dag_out",
        source_boundary_audit_report=audit_path,
    )

    nodes = {row["node_id"]: row for row in payload["gate_nodes"]}
    source_node = nodes["shared.source_boundary_compliance_audit"]
    assert source_node["status"] == "blocked"
    assert any("large_hex_literal_in_eval_source" in item for item in source_node["blockers"])
