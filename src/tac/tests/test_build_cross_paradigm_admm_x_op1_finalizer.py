from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_builder_module():
    module_path = REPO_ROOT / "tools" / "build_cross_paradigm_admm_x_op1_finalizer.py"
    spec = importlib.util.spec_from_file_location(
        "build_cross_paradigm_admm_x_op1_finalizer", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_output_root_accepts_repo_relative_paths() -> None:
    builder = _load_builder_module()

    resolved = builder._resolve_output_root(Path("experiments/results/worker_d"))

    assert resolved.is_absolute()
    assert resolved == (REPO_ROOT / "experiments/results/worker_d").resolve()


def test_resolve_output_root_preserves_absolute_paths(tmp_path: Path) -> None:
    builder = _load_builder_module()
    output_root = tmp_path / "worker-d-out"

    assert builder._resolve_output_root(output_root) == output_root.resolve()


def test_forked_inflate_sh_supports_python_override_for_local_advisory_eval() -> None:
    builder = _load_builder_module()

    assert 'if [ -n "${PYTHON:-}" ]; then' in builder.FORKED_INFLATE_SH
    assert '"$PYTHON" "$HERE/inflate.py" "$SRC" "$DST"' in builder.FORKED_INFLATE_SH
    assert '"$UV_BIN" run --no-project "${UV_WITH_INFLATE_DEPS[@]}" python "$HERE/inflate.py" "$SRC" "$DST"' in builder.FORKED_INFLATE_SH


def test_static_release_surface_records_archive_custody(tmp_path: Path) -> None:
    builder = _load_builder_module()
    archive_path = tmp_path / "archive.zip"
    byte_maps = b'{"0":"zig"}'
    op1_blob = b"op1-inner"
    latent = b"l" * builder.LATENT_BLOB_LEN
    sidecar = b"sidecar"
    decoder_section_bytes = 10 + len(byte_maps) + len(op1_blob)
    payload = (
        builder.CPLX_MAGIC
        + decoder_section_bytes.to_bytes(4, "little")
        + len(byte_maps).to_bytes(2, "little")
        + byte_maps
        + op1_blob
        + latent
        + sidecar
    )
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", payload)
    archive_bytes = archive_path.stat().st_size
    archive_sha = builder._sha256(archive_path.read_bytes())
    submission_dir = tmp_path / "submission_dir"
    submission_dir.mkdir()

    surface = builder._write_static_release_surface(
        submission_dir,
        archive_path=archive_path,
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
    )

    staged_archive = submission_dir / "archive.zip"
    manifest = json.loads((submission_dir / "archive_manifest.json").read_text())
    report = (submission_dir / "report.txt").read_text()
    assert builder._sha256(staged_archive.read_bytes()) == archive_sha
    assert surface["score_claim"] is False
    assert surface["ready_for_exact_eval_dispatch"] is False
    assert manifest["archive_sha256"] == archive_sha
    assert manifest["archive_size_bytes"] == archive_bytes
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["custody_status"] == "transient-allowed"
    assert "auth-eval" in manifest["custody_status_reason"]
    assert manifest["archive"]["members"][0]["name"] == "x"
    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser_section_manifest"]["section_names"] == [
        "cplx_magic",
        "decoder_section_len_u32le",
        "byte_maps_json_len_u16le",
        "byte_maps_json",
        "op1_inner_blob",
        "latent_blob",
        "sidecar_blob",
    ]
    assert manifest["parser_section_custody"]["score_claim"] is False
    assert surface["parser_section_gate"]["ready"] is True
    assert archive_sha in report
    assert str(archive_bytes) in report
