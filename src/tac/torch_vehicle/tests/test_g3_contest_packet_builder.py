# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the G3 contest-packet builder (tools/build_torch_vehicle_g3_contest_packet.py).

These prove the builder ACTUALLY:
  * byte-closes a real HNeRV decoder + latents into a ZIP_STORED archive.zip whose member
    is the byte-exact 0.bin (parse-back fixed-point parity, NOT a metadata claim);
  * assembles a self-contained runtime tree (inflate.sh + inflate.py + src/{model,codec});
  * the assembled inflate.sh ACTUALLY RUNS via subprocess and produces a correctly-shaped
    raw uint8 frame dump through the SAME inflate.py the contest evaluate.sh invokes.

If any of these were a no-op (e.g. the builder emitted markers without a real archive, or
wrote an inflate.sh that does not run), the corresponding test would FAIL — this is the
runtime-effect proof, not a constants check.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.torch_vehicle.driver import import_vendored_bundle

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BUILDER = _REPO_ROOT / "tools" / "build_torch_vehicle_g3_contest_packet.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_torch_vehicle_g3_contest_packet", _BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_fake_ckpt(ckpt_dir: Path, *, base_ch: int = 8, n_pairs: int = 4) -> None:
    """Write a real (small) HNeRV decoder + latents + meta as a driver best/ dir."""
    v = import_vendored_bundle()
    dec = v.HNeRVDecoder(latent_dim=28, base_channels=base_ch).eval()
    lat = torch.randn(n_pairs, 28) * 0.1
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dec.state_dict(), ckpt_dir / "best_ema_decoder.pt")
    torch.save(lat, ckpt_dir / "best_ema_latents.pt")
    (ckpt_dir / "best_meta.json").write_text(
        json.dumps({"base_channels": base_ch, "eval_size": [384, 512], "n_pairs": n_pairs})
    )


def test_builder_produces_byte_exact_zip_stored_archive(tmp_path):
    """archive.zip is ZIP_STORED, member '0.bin', byte-identical to the raw 0.bin payload."""
    mod = _load_builder()
    ckpt = tmp_path / "ckpt"
    _write_fake_ckpt(ckpt, base_ch=8, n_pairs=4)
    out = tmp_path / "submission_dir"
    manifest = mod.build_packet(ckpt, out)

    assert manifest["parse_back_parity_ok"] is True
    assert manifest["parity_status"]["weights_fixed_point"] is True
    assert manifest["parity_status"]["latents_fixed_point"] is True

    zip_path = out / "archive.zip"
    bin_path = out / "0.bin"
    assert zip_path.exists() and bin_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert names == ["0.bin"], names
        info = zf.getinfo("0.bin")
        assert info.compress_type == zipfile.ZIP_STORED
        member_bytes = zf.read("0.bin")
    # The zipped member is byte-identical to the standalone 0.bin payload.
    assert member_bytes == bin_path.read_bytes()


def test_builder_assembles_self_contained_runtime_tree(tmp_path):
    """inflate.sh + inflate.py + src/{model.py,codec.py} are all present and self-contained."""
    mod = _load_builder()
    ckpt = tmp_path / "ckpt"
    _write_fake_ckpt(ckpt, base_ch=8, n_pairs=4)
    out = tmp_path / "submission_dir"
    mod.build_packet(ckpt, out)

    for rel in ("inflate.sh", "inflate.py", "src/model.py", "src/codec.py", "README.md"):
        assert (out / rel).exists(), f"missing runtime file {rel}"
    # inflate.sh calls inflate.py DIRECTLY (no fragile `-m submissions.X.inflate`).
    sh = (out / "inflate.sh").read_text()
    assert '"$HERE/inflate.py"' in sh
    assert "submissions." not in sh  # no package-import form


def test_inflate_sh_actually_runs_and_produces_correct_shape_raw(tmp_path):
    """The assembled inflate.sh ACTUALLY runs via subprocess and emits a correctly-shaped raw.

    This is the runtime-closure proof at unit scale: the SAME inflate.sh -> inflate.py chain
    the contest evaluate.sh invokes, on a tiny 4-pair archive, produces (8, 874, 1164, 3) uint8.
    """
    mod = _load_builder()
    ckpt = tmp_path / "ckpt"
    n_pairs = 4
    _write_fake_ckpt(ckpt, base_ch=8, n_pairs=n_pairs)
    out = tmp_path / "submission_dir"
    mod.build_packet(ckpt, out)

    # Mirror evaluate.sh: unzip archive.zip -> archive/, then run inflate.sh.
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(out / "archive.zip") as zf:
        zf.extractall(archive_dir)
    inflated_dir = tmp_path / "inflated"
    names_file = tmp_path / "video_names.txt"
    names_file.write_text("0.mkv\n")

    env = {
        **__import__("os").environ,
        "PYTHON": sys.executable,
        "COMMA_CHALLENGE_ROOT": str((_REPO_ROOT / "upstream").resolve()),
    }
    r = subprocess.run(
        ["bash", str(out / "inflate.sh"), str(archive_dir), str(inflated_dir), str(names_file)],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert r.returncode == 0, f"inflate.sh failed: {r.stderr[-2000:]}"

    raw = inflated_dir / "0.raw"
    assert raw.exists()
    expected_bytes = n_pairs * 2 * 874 * 1164 * 3
    assert raw.stat().st_size == expected_bytes, (raw.stat().st_size, expected_bytes)
    arr = np.frombuffer(raw.read_bytes(), dtype=np.uint8)
    assert arr.size == expected_bytes
    assert int(arr.min()) >= 0 and int(arr.max()) <= 255


def test_builder_refuses_missing_checkpoint(tmp_path):
    """NO-FAKE: a missing checkpoint raises rather than fabricating an archive."""
    mod = _load_builder()
    with pytest.raises((FileNotFoundError, SystemExit, RuntimeError)):
        mod.build_packet(tmp_path / "does_not_exist", tmp_path / "out")


def test_manifest_records_real_custody_fields(tmp_path):
    """Manifest carries real archive sha + size + runtime-tree sha (custody, not placeholders)."""
    mod = _load_builder()
    ckpt = tmp_path / "ckpt"
    _write_fake_ckpt(ckpt, base_ch=8, n_pairs=4)
    out = tmp_path / "submission_dir"
    manifest = mod.build_packet(ckpt, out)

    assert manifest["archive_zip_bytes"] == (out / "archive.zip").stat().st_size
    assert len(manifest["archive_zip_sha256"]) == 64
    assert len(manifest["runtime_tree_sha256"]) == 64
    assert "placeholder" not in json.dumps(manifest).lower()
    assert "pending" not in json.dumps(manifest).lower()
    assert manifest["zip_overhead_bytes"] >= 0
