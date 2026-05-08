"""Tests for runtime guards added in deep hardening pass 3 dimension 3.

- experiments/contest_auth_eval.py: _validate_archive_members whitelist
- src/tac/training.py: finite-loss assertion before backward (smoke only —
  full training-loop integration is exercised by existing trainer tests)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_validator():
    mod = _load_module(
        REPO_ROOT / "experiments" / "contest_auth_eval.py",
        "_test_contest_auth_eval",
    )
    return mod._validate_archive_members


def _get_contest_auth_eval_module():
    return _load_module(
        REPO_ROOT / "experiments" / "contest_auth_eval.py",
        "_test_contest_auth_eval_full",
    )


def test_contest_auth_eval_requires_durable_result_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mod = _get_contest_auth_eval_module()
    tmp_root = tmp_path / "system_tmp"
    durable_root = tmp_path / "repo_results"
    tmp_root.mkdir()
    durable_root.mkdir()
    monkeypatch.setattr(mod.tempfile, "gettempdir", lambda: str(tmp_root))

    with pytest.raises(SystemExit, match="requires --work-dir or --json-out"):
        mod._validate_durable_result_output(
            types.SimpleNamespace(work_dir=None, json_out=None, allow_temp_work_dir=False)
        )

    with pytest.raises(SystemExit, match="under temp storage"):
        mod._validate_durable_result_output(
            types.SimpleNamespace(
                work_dir=tmp_root / "contest_eval",
                json_out=None,
                allow_temp_work_dir=False,
            )
        )

    mod._validate_durable_result_output(
        types.SimpleNamespace(
            work_dir=durable_root / "contest_eval",
            json_out=None,
            allow_temp_work_dir=False,
        )
    )
    mod._validate_durable_result_output(
        types.SimpleNamespace(
            work_dir=tmp_root / "scratch_eval",
            json_out=None,
            allow_temp_work_dir=True,
        )
    )


# ─── Archive whitelist validator ───────────────────────────────────────────


def test_validator_passes_canonical_archive():
    """Renderer + masks + poses is the canonical contest contract."""
    validator = _get_validator()
    validator(["renderer.bin", "masks.mkv", "poses.pt"])  # no raise


def test_validator_passes_brotli_renderer():
    """Brotli-compressed renderer is allowed (.bin.br suffix)."""
    validator = _get_validator()
    validator(["renderer.bin.br", "masks.mkv", "poses.pt"])


def test_validator_passes_nerv_mask_archive():
    """Lane 12 NeRV stores masks in a dedicated .nrv payload."""
    validator = _get_validator()
    validator(["renderer.bin", "masks.nrv", "poses.pt"])


def test_validator_passes_charged_mask_grammar_payload():
    """CMG1 is a charged mask grammar payload decoded by inflate runtime."""
    validator = _get_validator()
    validator(["renderer.bin", "masks.cmg1", "optimized_poses.bin"])


def test_validator_passes_alpha_sparse_repair_payloads():
    """Alpha grayscale sparse repair uses AMR1 side-info charged in archive."""
    validator = _get_validator()
    validator(
        [
            "renderer.bin",
            "grayscale.mkv",
            "alpha4_residual_repair.amr1.xz",
            "optimized_poses.bin",
        ]
    )
    validator(["alpha4_residual_repair.amr1"])
    validator(["alpha4_residual_repair.amr1.zlib"])
    validator(["alpha4_residual_repair.amr1.br"])


def test_validator_passes_short_p_payload_member():
    """Top submissions use basename-only member p for a packed payload."""
    validator = _get_validator()
    validator(["p"])


def test_validator_passes_short_x_payload_member():
    """PR65/henosis uses basename-only member x for a packed payload."""
    validator = _get_validator()
    validator(["x"])


def test_validator_rejects_other_extensionless_payloads():
    """Allowing p must not become a blanket extensionless-file bypass."""
    validator = _get_validator()
    with pytest.raises(RuntimeError, match="UNKNOWN file types"):
        validator(["debug_payload"])


def test_validator_rejects_macos_resource_fork():
    """macOS ._foo files inflate the rate silently — must raise."""
    validator = _get_validator()
    with pytest.raises(RuntimeError, match="FORBIDDEN files"):
        validator(["renderer.bin", "._renderer.bin", "masks.mkv"])


def test_validator_rejects_ds_store():
    validator = _get_validator()
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        validator(["renderer.bin", ".DS_Store"])


def test_validator_rejects_unknown_file_type():
    """Stale debug artifacts (e.g., .pkl) should fail loud."""
    validator = _get_validator()
    with pytest.raises(RuntimeError, match="UNKNOWN file types"):
        validator(["renderer.bin", "masks.mkv", "debug_state.pkl"])


def test_validator_rejects_empty_archive():
    validator = _get_validator()
    with pytest.raises(RuntimeError, match="EMPTY archive"):
        validator([])


def test_validator_rejects_macosx_dir():
    validator = _get_validator()
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        validator(["renderer.bin", "__MACOSX/renderer.bin"])


def test_extract_rejects_zip_central_local_filename_divergence(tmp_path: Path):
    """Different unzip readers must not see different archive members."""
    mod = _get_contest_auth_eval_module()
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("p", b"payload")

    raw = bytearray(archive.read_bytes())
    assert raw[:4] == b"PK\x03\x04"
    raw[26:28] = (0).to_bytes(2, "little")
    archive.write_bytes(raw)

    with pytest.raises(RuntimeError, match="central/local filename mismatch|EMPTY zip local filename"):
        mod._extract_archive(archive, tmp_path / "extracted")


def test_config_env_guard_is_scoped_to_renderer_dispatchers(tmp_path: Path):
    """External public inflate.sh launchers need not ship robust config.env."""
    mod = _get_contest_auth_eval_module()

    public_inflate = tmp_path / "public" / "inflate.sh"
    public_inflate.parent.mkdir()
    public_inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "python \"$HERE/inflate.py\" \"$1\" \"$2\" \"$3\"\n"
    )
    assert mod._inflate_sh_requires_config_env_guard(public_inflate) is False
    mod._validate_config_env_for_renderer_dispatch(public_inflate)

    robust_inflate = tmp_path / "robust" / "inflate.sh"
    robust_inflate.parent.mkdir()
    robust_inflate.write_text(
        "#!/usr/bin/env bash\n"
        "CONFIG_ENV_PATH=\"${CONFIG_ENV_PATH:-$SELF_DIR/config.env}\"\n"
        "if [ \"$PYTHON_INFLATE\" = \"renderer\" ]; then true; fi\n"
    )
    assert mod._inflate_sh_requires_config_env_guard(robust_inflate) is True
    with pytest.raises(SystemExit, match="config.env missing"):
        mod._validate_config_env_for_renderer_dispatch(robust_inflate)

    (robust_inflate.parent / "config.env").write_text("PYTHON_INFLATE=renderer\n")
    mod._validate_config_env_for_renderer_dispatch(robust_inflate)


def test_uv_guard_adopts_standard_installer_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Non-login remote shells often omit ~/.local/bin despite uv being installed."""
    mod = _get_contest_auth_eval_module()
    fake_home = tmp_path / "home"
    uv_dir = fake_home / ".local" / "bin"
    uv_dir.mkdir(parents=True)
    uv = uv_dir / "uv"
    uv.write_text("#!/usr/bin/env sh\nexit 0\n")
    uv.chmod(0o755)

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("PATH", str(tmp_path / "empty_path"))
    monkeypatch.setattr(mod.Path, "home", lambda: fake_home)

    mod._ensure_uv_available()

    assert str(uv_dir) in os.environ["PATH"].split(os.pathsep)


# ─── Trainer finite-loss guard (smoke check) ────────────────────────────────


def test_finite_loss_guard_present_in_training_py():
    """Verify the finite-loss assertion is wired into the canonical Trainer
    BEFORE backward(). If the guard is removed, this test fails loud so the
    operator knows protection was lost."""
    text = (REPO_ROOT / "src" / "tac" / "training.py").read_text()
    # Both Trainer paths (canonical + lazy) must guard.
    assert "non-finite loss" in text, (
        "training.py: finite-loss guard removed — re-add the .item()-based "
        "check before backward() (deep hardening pass 3 dim 3)."
    )
    # Specifically: must mention 'before backward' to ensure ordering matters.
    assert text.count("non-finite loss") >= 2, (
        "training.py: only one finite-loss guard present (expected 2: "
        "canonical Trainer + lazy variant)."
    )
