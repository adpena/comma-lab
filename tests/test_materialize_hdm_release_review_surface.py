from __future__ import annotations

import importlib.util
import json
import stat
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "materialize_hdm_release_review_surface.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("materialize_hdm_release_review_surface", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_zip(path: Path) -> tuple[int, str, dict[str, object]]:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    payload = b"hdm4-payload"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
    raw = path.read_bytes()
    return (
        len(raw),
        hashlib.sha256(raw).hexdigest(),
        {
            "name": "0.bin",
            "file_size": len(payload),
            "compress_size": len(payload),
            "crc": zipfile.crc32(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )


def _runtime(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    root = tmp_path / "runtime"
    src = root / "src"
    src.mkdir(parents=True)
    files = {
        "inflate.sh": "#!/usr/bin/env bash\nexec python \"$PWD/inflate.py\" \"$@\"\n",
        "inflate.py": "print('inflate')\n",
        "src/codec.py": "MAGIC = 'HDM4'\n",
    }
    rows: list[dict[str, object]] = []
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if rel == "inflate.sh":
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        rows.append({"relative_path": rel, "bytes": path.stat().st_size, "sha256": _sha(path)})
    (root / "report.txt").write_text("old archive_sha256: stale\n", encoding="utf-8")
    return root, rows


def _write_auth_eval(path: Path, *, archive_bytes: int, archive_sha: str, rows: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "archive_size_bytes": archive_bytes,
                "final_score": 0.2064,
                "canonical_score": 0.2064,
                "avg_segnet_dist": 0.001,
                "avg_posenet_dist": 0.002,
                "n_samples": 600,
                "lane_tag": "[contest-CUDA]",
                "score_axis": "contest_cuda",
                "evidence_grade": "A++",
                "exact_cuda_eval_complete": True,
                "score_claim_valid": True,
                "provenance": {
                    "archive_sha256": archive_sha,
                    "archive_size_bytes": archive_bytes,
                    "inflate_runtime_manifest": {
                        "runtime_tree_sha256": "a" * 64,
                        "runtime_content_tree_sha256": "b" * 64,
                        "runtime_file_count": len(rows),
                        "files": [
                            *rows,
                            {
                                "relative_path": "report.txt",
                                "bytes": 5,
                                "sha256": "c" * 64,
                            },
                        ],
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_materializes_surface_with_runtime_verified_against_auth_eval(tmp_path: Path) -> None:
    tool = _load_tool()
    archive_bytes, archive_sha, member = _write_zip(tmp_path / "archive.zip")
    runtime, rows = _runtime(tmp_path)
    auth = tmp_path / "contest_auth_eval.json"
    _write_auth_eval(auth, archive_bytes=archive_bytes, archive_sha=archive_sha, rows=rows)

    manifest = tool.materialize_release_review_surface(
        archive=tmp_path / "archive.zip",
        auth_eval_json=auth,
        source_runtime_dir=runtime,
        output_dir=tmp_path / "surface",
        lane_id="hnerv_hdm4_q_brotli_split_exact_eval",
        job_id="job",
        candidate_label="HDM4",
    )

    surface = tmp_path / "surface"
    assert (surface / "archive.zip").read_bytes() == (tmp_path / "archive.zip").read_bytes()
    assert (surface / "inflate.sh").stat().st_mode & stat.S_IXUSR
    assert (surface / "report.txt").read_text(encoding="utf-8").find(archive_sha) >= 0
    assert (surface / "report.txt").read_text(encoding="utf-8").find(str(archive_bytes)) >= 0
    assert (surface / "pre_submission_compliance.release_review_manifest.json").is_file()
    assert not (surface / "release_review_manifest.json").exists()
    archive_manifest = json.loads((surface / "archive_manifest.json").read_text(encoding="utf-8"))
    assert archive_manifest["archive_sha256"] == archive_sha
    assert archive_manifest["archive_size_bytes"] == archive_bytes
    assert archive_manifest["members"] == [member]
    assert manifest["runtime_files_verified_against_auth_eval"] == [
        {**row, "mode": "0755" if row["relative_path"] == "inflate.sh" else "0644"}
        for row in sorted(rows, key=lambda item: item["relative_path"])
    ]
    assert "--expected-runtime-tree-sha256" in manifest["compliance_command"]


def test_refuses_source_runtime_that_does_not_match_exact_cuda_manifest(tmp_path: Path) -> None:
    tool = _load_tool()
    archive_bytes, archive_sha, _member = _write_zip(tmp_path / "archive.zip")
    runtime, rows = _runtime(tmp_path)
    auth = tmp_path / "contest_auth_eval.json"
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["sha256"] = "0" * 64
    _write_auth_eval(auth, archive_bytes=archive_bytes, archive_sha=archive_sha, rows=tampered_rows)

    try:
        tool.materialize_release_review_surface(
            archive=tmp_path / "archive.zip",
            auth_eval_json=auth,
            source_runtime_dir=runtime,
            output_dir=tmp_path / "surface",
            lane_id="hnerv_hdm4_q_brotli_split_exact_eval",
            job_id="job",
            candidate_label="HDM4",
        )
    except tool.HdmReleaseReviewSurfaceError as exc:
        assert "runtime file mismatch after copy" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected runtime mismatch refusal")
