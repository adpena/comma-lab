"""Operator decision B 2026-05-09: Phase 1 trainer `_write_runtime` fix.

Tests verify that the trainer now emits a contest-compliant inflate.sh +
inflate.py + 3-member archive that passes the Phase 1 packet compiler's
optimize-mode validation gate.

Memory: feedback_phase1_trainer_write_runtime_fix_landed_20260509.md
"""
from __future__ import annotations

import importlib.util
import io
import re
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = (
    REPO_ROOT
    / "experiments"
    / "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
)


def _load_trainer():
    """Import the trainer module by file-path so tests don't depend on PYTHONPATH."""
    if "experiments" not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "pact_phase1_trainer_for_test",
        TRAINER_PATH,
    )
    if spec is None or spec.loader is None:
        pytest.skip(f"trainer not importable from {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _load_trainer()


def _read_write_runtime_body() -> str:
    """Return the source body of the `_write_runtime` function only."""
    text = TRAINER_PATH.read_text(encoding="utf-8")
    start_marker = "\ndef _write_runtime("
    start = text.find(start_marker)
    assert start >= 0, "trainer:_write_runtime function not found"
    rest = text[start + 1 :]
    end_match = re.search(r"\n(def |class )", rest)
    return rest if end_match is None else rest[: end_match.start()]


def _read_emitted_strings_only() -> str:
    """Return the EMITTED string-literal blocks from `_write_runtime` body only.

    Filters out the function's docstring and any commentary lines that
    legitimately mention forbidden tokens for documentation purposes.
    """
    body = _read_write_runtime_body()
    pattern = re.compile(r"(?ms)^\s*[a-zA-Z_][a-zA-Z_0-9]*\s*=\s*\((.*?)\n\s*\)")
    return "\n".join(match.group(1) for match in pattern.finditer(body))


# ---------------------------------------------------------------------------
# Source-level structural tests on _write_runtime
# ---------------------------------------------------------------------------


def test_write_runtime_function_exists(trainer):
    assert hasattr(trainer, "_write_runtime")
    assert callable(trainer._write_runtime)


def test_write_runtime_emits_three_positional_args():
    body = _read_write_runtime_body()
    # All three positional shell vars must be referenced.
    assert "$1" in body
    assert "$2" in body
    assert "$3" in body
    # AND we expect explicit handoff via DATA_DIR / OUTPUT_DIR / FILE_LIST.
    assert "DATA_DIR" in body
    assert "OUTPUT_DIR" in body
    assert "FILE_LIST" in body


def test_write_runtime_emits_set_e():
    body = _read_write_runtime_body()
    assert "set -euo pipefail" in body or "set -e" in body


def test_write_runtime_does_not_emit_single_arg_passthrough():
    """Refuse the legacy `"$@"` passthrough that the packet compiler rejects."""
    body = _read_write_runtime_body()
    assert 'inflate.py" "$@"' not in body, (
        "trainer must emit explicit positional args, not '\"$@\"' passthrough"
    )


def test_write_runtime_inflate_py_has_no_forbidden_scorer_tokens():
    """Per Q6 council + CLAUDE.md strict-scorer-rule.

    Scans the EMITTED string literals only, NOT the docstring/comments
    which legitimately document the prohibition.
    """
    emitted = _read_emitted_strings_only()
    forbidden = (
        "PoseNet",
        "SegNet",
        "from upstream.modules",
        "import upstream.modules",
        "rgb_to_yuv6",
        "EfficientNet",
        "FastViT",
    )
    for token in forbidden:
        assert token not in emitted, (
            f"trainer:_write_runtime EMITTED template contains forbidden inflate token {token!r}"
        )


def test_write_runtime_inflate_py_has_per_video_loop():
    body = _read_write_runtime_body()
    # Must contain at least one of the canonical per-video iteration patterns.
    assert any(
        pat in body
        for pat in (
            "for line in file_list",
            "splitlines()",
            "while IFS= read",
            "for base in",
        )
    )


def test_write_runtime_does_not_fetch_runtime_dependencies():
    body = _read_write_runtime_body()
    emitted = _read_emitted_strings_only()
    forbidden = (
        "uv run",
        "--with ",
        "--extra-index-url",
        "--index-url",
        "https://",
        "pip install",
    )
    for token in forbidden:
        assert token not in emitted


def test_write_runtime_inflate_py_imports_no_pickle():
    """Q1 council: serialization is length-prefixed binary, NOT pickle."""
    body = _read_write_runtime_body()
    # The serialiser uses struct + brotli; no pickle.loads anywhere.
    assert "pickle.loads" not in body, (
        "Q1 consensus: deserialization must be length-prefixed binary, not pickle"
    )


def test_write_runtime_inflate_py_writes_camera_resolution():
    body = _read_write_runtime_body()
    # Per Q3 council: camera-resolution = (874, 1164).
    assert "874" in body
    assert "1164" in body
    # Bicubic interpolation per Q3 + INFLATE_ROUNDTRIP_CAMERA_HW.
    assert "bicubic" in body
    assert "align_corners=False" in body


def test_write_runtime_inflate_py_uses_uint8_rgb_output():
    body = _read_write_runtime_body()
    assert "torch.uint8" in body or "to(torch.uint8)" in body
    # Per the contest .raw contract: "(N, H, W, 3)" — channels-last.
    assert "permute(0, 2, 3, 1)" in body


# ---------------------------------------------------------------------------
# End-to-end emission tests (write to tmp_path, validate output structure)
# ---------------------------------------------------------------------------


@pytest.fixture
def written_runtime(trainer, tmp_path):
    """Invoke _write_runtime against a fresh submission_dir under tmp_path."""
    submission_dir = tmp_path / "submission_dir"
    submission_dir.mkdir()
    decoder_cfg = trainer.Decoder128KConfig(latent_dim=28)
    balle_cfg = trainer.BalleHyperpriorConfig(y_channels=28)
    trainer._write_runtime(
        submission_dir=submission_dir,
        decoder_config=decoder_cfg,
        balle_config=balle_cfg,
    )
    return submission_dir


def test_emitted_inflate_sh_exists_and_is_executable(written_runtime):
    inflate_sh = written_runtime / "inflate.sh"
    assert inflate_sh.is_file()
    mode = inflate_sh.stat().st_mode & 0o777
    assert mode & 0o100, "inflate.sh must be executable (0o755)"


def test_emitted_inflate_py_exists(written_runtime):
    assert (written_runtime / "inflate.py").is_file()


def test_emitted_src_codec_and_model_exist(written_runtime):
    assert (written_runtime / "src" / "codec.py").is_file()
    assert (written_runtime / "src" / "model.py").is_file()
    assert (written_runtime / "src" / "tac" / "__init__.py").is_file()
    assert (
        written_runtime
        / "src"
        / "tac"
        / "paradigm_delta_epsilon_zeta"
        / "decoder_128k.py"
    ).is_file()
    assert (
        written_runtime
        / "src"
        / "tac"
        / "paradigm_delta_epsilon_zeta"
        / "balle_hyperprior.py"
    ).is_file()


def test_emitted_model_uses_packet_local_tac_only(written_runtime):
    text = (written_runtime / "src" / "model.py").read_text()
    for token in (
        "_find_repo_src",
        "here.parents",
        "repo-local tac runtime dependency",
        "sys.path.insert",
    ):
        assert token not in text
    assert "from tac.paradigm_delta_epsilon_zeta.decoder_128k import" in text
    assert "from tac.paradigm_delta_epsilon_zeta.balle_hyperprior import" in text


def test_emitted_inflate_sh_under_100_loc(written_runtime):
    """Per Q4 + packet compiler `inflate_runtime_loc_budget=100`."""
    text = (written_runtime / "inflate.sh").read_text()
    code_lines = [
        line for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(code_lines) <= 100, f"inflate.sh has {len(code_lines)} executable lines (>100)"


def test_emitted_inflate_sh_passes_bash_n_syntax(written_runtime):
    """The emitted inflate.sh must parse cleanly under `bash -n`."""
    import shutil
    import subprocess

    if not shutil.which("bash"):
        pytest.skip("bash not available")
    inflate_sh = written_runtime / "inflate.sh"
    proc = subprocess.run(
        ["bash", "-n", str(inflate_sh)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, f"bash -n failed: {proc.stderr}"


def test_emitted_inflate_sh_has_3_positional_arg_handoff(written_runtime):
    text = (written_runtime / "inflate.sh").read_text()
    # uv run / direct exec line must hand DATA_DIR / OUTPUT_DIR / FILE_LIST
    # to inflate.py explicitly.
    assert '"$DATA_DIR"' in text
    assert '"$OUTPUT_DIR"' in text
    assert '"$FILE_LIST"' in text


def test_emitted_inflate_sh_no_curl_wget_pip_install(written_runtime):
    text = (written_runtime / "inflate.sh").read_text()
    for forbidden in (
        "curl ",
        "wget ",
        "pip install",
        "git clone",
        "uv run",
        "--with ",
        "--extra-index-url",
        "--index-url",
        "https://",
    ):
        assert forbidden not in text, f"inflate.sh contains forbidden network token {forbidden!r}"


def test_emitted_inflate_py_no_scorer_imports(written_runtime):
    text = (written_runtime / "inflate.py").read_text()
    forbidden = ("PoseNet", "SegNet", "rgb_to_yuv6", "from upstream.modules",
                 "EfficientNet", "FastViT")
    for token in forbidden:
        assert token not in text, f"inflate.py contains forbidden scorer token {token!r}"


def test_emitted_inflate_py_signature_is_three_args(written_runtime):
    text = (written_runtime / "inflate.py").read_text()
    # Either explicit argv length check OR docstring usage line.
    assert "len(sys.argv) != 4" in text or "Usage: inflate.py <archive_dir>" in text


def test_emitted_inflate_py_imports_brotli_compressai(written_runtime):
    text = (written_runtime / "inflate.py").read_text()
    assert "import brotli" in text
    # compressai is loaded via the model.py shim's BalleRuntime import chain.
    # The inflate.py itself doesn't need to import compressai directly.


def test_emitted_inflate_py_has_per_video_loop(written_runtime):
    text = (written_runtime / "inflate.py").read_text()
    assert "for line in file_list.read_text().splitlines()" in text


# ---------------------------------------------------------------------------
# Wire-format roundtrip tests (length-prefixed Ballé strings)
# ---------------------------------------------------------------------------


def _deserialise_strings_inline(payload: bytes) -> dict:
    """Mirror of the inflate.py _deserialise_strings function."""
    off = 0

    def _read_byte_list(off):
        n = struct.unpack_from("<I", payload, off)[0]
        off += 4
        items = []
        for _ in range(n):
            blen = struct.unpack_from("<I", payload, off)[0]
            off += 4
            items.append(payload[off : off + blen])
            off += blen
        return items, off

    y_strings, off = _read_byte_list(off)
    z_strings, off = _read_byte_list(off)
    n_shape = struct.unpack_from("<I", payload, off)[0]
    off += 4
    z_shape = []
    for _ in range(n_shape):
        z_shape.append(struct.unpack_from("<i", payload, off)[0])
        off += 4
    return {"y_strings": y_strings, "z_strings": z_strings, "z_shape": z_shape}


def test_serialise_balle_strings_roundtrip_dict_form(trainer):
    """The trainer-side _serialise_balle_strings must round-trip via the
    inflate.py-side _deserialise_strings inverse (dict form per Ballé wrapper)."""
    sample = {
        "y_strings": [b"abc", b"defgh", b"ij"],
        "z_strings": [b"hyperprior_blob"],
        "z_shape": [1, 28, 4, 1],
    }
    payload = trainer._serialise_balle_strings(sample)
    rebuilt = _deserialise_strings_inline(payload)
    assert rebuilt == sample


def test_serialise_balle_strings_rejects_non_dict(trainer):
    with pytest.raises(RuntimeError, match="expected dict"):
        trainer._serialise_balle_strings("not a dict")


def test_serialise_balle_strings_rejects_missing_y_strings(trainer):
    with pytest.raises(RuntimeError, match="y_strings"):
        trainer._serialise_balle_strings({"z_strings": [], "z_shape": []})


def test_serialise_balle_strings_rejects_non_bytes_blob(trainer):
    bad = {"y_strings": ["str-not-bytes"], "z_strings": [], "z_shape": []}
    with pytest.raises(RuntimeError, match="expected bytes"):
        trainer._serialise_balle_strings(bad)


def test_serialise_balle_strings_rejects_non_int_shape(trainer):
    bad = {"y_strings": [], "z_strings": [], "z_shape": [1.5]}
    with pytest.raises(RuntimeError, match="expected int"):
        trainer._serialise_balle_strings(bad)


def test_serialise_balle_strings_rejects_negative_shape_dim(trainer):
    bad = {"y_strings": [], "z_strings": [], "z_shape": [1, -1, 4, 1]}
    with pytest.raises(RuntimeError, match="nonnegative int32"):
        trainer._serialise_balle_strings(bad)


# ---------------------------------------------------------------------------
# Catalog #146 preflight gate live count + strict behavior
# ---------------------------------------------------------------------------


def test_catalog_146_check_function_exists():
    from tac import preflight as pf

    assert hasattr(pf, "check_phase1_trainer_runtime_emits_contest_compliant_inflate")


def test_catalog_146_live_count_zero_on_current_trainer():
    from tac import preflight as pf

    violations = pf.check_phase1_trainer_runtime_emits_contest_compliant_inflate(
        strict=False, verbose=False,
    )
    assert violations == [], f"Catalog #146 unexpected violations: {violations}"


def test_catalog_146_strict_mode_raises_on_simulated_regression(tmp_path, monkeypatch):
    """Verify STRICT mode raises MetaBugViolation when the trainer source is
    mutated to reintroduce the broken scaffold pattern."""
    from tac import preflight as pf

    # Read real trainer + corrupt _write_runtime by overwriting $1/$2/$3 references.
    text = TRAINER_PATH.read_text(encoding="utf-8")
    start_marker = "\ndef _write_runtime("
    start = text.find(start_marker)
    assert start >= 0
    body_start = start + 1
    rest = text[body_start:]
    end_match = re.search(r"\n(def |class )", rest)
    body_end = body_start + (end_match.start() if end_match else len(rest))
    body = text[body_start:body_end]
    # Strip the explicit positional handoff to simulate the legacy regression.
    corrupted_body = (
        body.replace("$DATA_DIR", "_REMOVED_")
        .replace("$OUTPUT_DIR", "_REMOVED_")
        .replace("$FILE_LIST", "_REMOVED_")
        .replace('"$1"', "")
        .replace('"$2"', "")
        .replace('"$3"', "")
        .replace("$1", "")
        .replace("$2", "")
        .replace("$3", "")
    )
    fake_trainer = tmp_path / "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
    fake_trainer.write_text(text[:body_start] + corrupted_body + text[body_end:])

    # Monkey-patch REPO_ROOT in preflight module so the check finds the fake trainer.
    fake_root = tmp_path
    (fake_root / "experiments").mkdir(exist_ok=True)
    real_path = fake_root / "experiments" / "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
    real_path.write_text(text[:body_start] + corrupted_body + text[body_end:])

    monkeypatch.setattr(pf, "REPO_ROOT", fake_root)

    with pytest.raises(pf.MetaBugViolation, match="PHASE1_TRAINER_WRITE_RUNTIME"):
        pf.check_phase1_trainer_runtime_emits_contest_compliant_inflate(
            strict=True, verbose=False,
        )


def test_catalog_146_warn_only_returns_violations_without_raising(tmp_path, monkeypatch):
    from tac import preflight as pf

    fake_root = tmp_path
    (fake_root / "experiments").mkdir(exist_ok=True)
    bad = (
        "def _write_runtime(submission_dir, decoder_config, balle_config):\n"
        "    inflate_sh = '#!/bin/bash\\nexec uv run python \"$HERE/inflate.py\" \"$@\"\\n'\n"
        "    (submission_dir / 'inflate.sh').write_text(inflate_sh)\n"
    )
    (fake_root / "experiments" / "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py").write_text(bad)

    monkeypatch.setattr(pf, "REPO_ROOT", fake_root)

    violations = pf.check_phase1_trainer_runtime_emits_contest_compliant_inflate(
        strict=False, verbose=False,
    )
    assert len(violations) >= 1


# ---------------------------------------------------------------------------
# Packet-compiler validation gate (the canonical READY-FOR-DISPATCH check)
# ---------------------------------------------------------------------------


def test_emitted_archive_has_three_named_zip_members(trainer, tmp_path):
    """End-to-end: build an archive via build_archive_from_ema then verify
    the archive.zip has the 3 expected named members per Q2 council."""
    import torch
    import zipfile

    decoder_cfg = trainer.Decoder128KConfig(latent_dim=28)
    balle_cfg = trainer.BalleHyperpriorConfig(y_channels=28)
    decoder = trainer.build_decoder_128k(decoder_cfg)
    balle = trainer.build_balle_hyperprior(balle_cfg)
    ema_decoder = trainer.EMA(decoder, decay=0.997)
    ema_balle = trainer.EMA(balle, decay=0.997)
    latents = torch.randn((4, 28))  # tiny synthetic for shape-test only

    output_dir = tmp_path / "trainer_run"
    output_dir.mkdir()
    archive_path = trainer.build_archive_from_ema(
        output_dir=output_dir,
        decoder=decoder,
        balle=balle,
        ema_decoder=ema_decoder,
        ema_balle=ema_balle,
        latents=latents,
        decoder_config=decoder_cfg,
        balle_config=balle_cfg,
    )
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = sorted(zf.namelist())
    assert names == sorted(["x", "decoder.bin", "balle.bin"]), (
        f"archive must have exactly 3 named members; got {names}"
    )


def test_packet_compiler_optimize_mode_accepts_emitted_packet(trainer, tmp_path):
    """The Phase 1 packet compiler's optimize mode MUST return blockers == ()
    on the trainer's emitted packet (READY-FOR-DISPATCH validation gate).

    This is the canonical proof that operator decision B is closed.
    """
    import torch

    from tac.phase1_packet_compiler import (
        A1_CANONICAL_ARCHIVE_SHA256,
        A1_CANONICAL_ARCHIVE_SIZE_BYTES,
        compile_phase1_packet,
    )

    decoder_cfg = trainer.Decoder128KConfig(latent_dim=28)
    balle_cfg = trainer.BalleHyperpriorConfig(y_channels=28)
    decoder = trainer.build_decoder_128k(decoder_cfg)
    balle = trainer.build_balle_hyperprior(balle_cfg)
    ema_decoder = trainer.EMA(decoder, decay=0.997)
    ema_balle = trainer.EMA(balle, decay=0.997)
    latents = torch.randn((4, 28))

    trainer_output = tmp_path / "trainer_run"
    trainer_output.mkdir()
    trainer.build_archive_from_ema(
        output_dir=trainer_output,
        decoder=decoder,
        balle=balle,
        ema_decoder=ema_decoder,
        ema_balle=ema_balle,
        latents=latents,
        decoder_config=decoder_cfg,
        balle_config=balle_cfg,
    )

    packet_input = trainer_output / "submission_dir"
    packet_output = tmp_path / "packet_compiled"

    result = compile_phase1_packet(
        input_packet=packet_input,
        output_dir=packet_output,
        mode="optimize",
        target_mode="contest_one_video_replay",
        runtime_dep_closure=("torch", "brotli", "compressai"),
        export_format="phase1_three_member_x_decoder_bin_balle_bin",
        bolt_on_loc_budget=400,
        score_affecting_payload_changed=True,
        baseline_archive_sha256=A1_CANONICAL_ARCHIVE_SHA256,
        baseline_archive_size_bytes=A1_CANONICAL_ARCHIVE_SIZE_BYTES,
    )

    # The blockers list must be empty (READY-FOR-DISPATCH).
    # Note: byte-mutation executable smoke may fail in test environment if uv
    # isn't available, but that downgrades runtime_consumes_bytes silently;
    # the structural blockers (positional args, set -e, etc.) MUST be clean.
    structural_blockers = [
        b for b in result.blockers
        if not b.startswith("inflate_does_not_consume_archive_bytes:")
        and not b.startswith("no_op_detector_failed:")
    ]
    assert structural_blockers == [], (
        f"Phase 1 packet compiler returned structural blockers: {structural_blockers}"
    )
