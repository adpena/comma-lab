"""Lock-in tests for the 2026-04-26 SHIRAZ-class bugs.

Two bug classes burned ~21h of A100 time on 2026-04-26:

1. **Suffix-based pose dispatch.** A wrapper renamed `optimized_poses_partial.pt`
   (a torch.save pickle) to `optimized_poses.bin` (raw fp16 buffer). The auth
   eval loader called `torch.frombuffer(..., dtype=float16).reshape(-1, 6)` on
   a pickle file and crashed with `shape '[-1, 6]' is invalid for input of
   size 7862` after 7 minutes of mask extraction.

2. **Self-matching `pgrep -f TOKEN` wait loop.** A wrapper invoked as
   `bash -c "while pgrep -f train_distill ...; done; bash run_pipeline.sh"`
   matched its OWN argv (because the bash -c string contained the literal
   "train_distill"), looped forever, and a fully-paid A100 sat idle.

These tests pin the permanent prevention so the bugs never come back.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tac.preflight import (
    _scan_bash_text_for_forbidden,
    _scan_python_for_forbidden,
    _scan_text_for_dangerous_patterns,
)
from tac.submission_archive import (
    _looks_like_pickle,
    load_optimized_poses,
    save_poses_binary,
)


# ── Pose loader: content-based format detection ───────────────────────────


def _write_raw_fp16(path: Path, n_pairs: int, pose_dim: int = 6) -> None:
    poses = torch.randn(n_pairs, pose_dim, dtype=torch.float32) * 0.01
    save_poses_binary(poses, path)


def _write_pickle_pt(path: Path, n_pairs: int, pose_dim: int = 6) -> None:
    poses = torch.randn(n_pairs, pose_dim, dtype=torch.float32) * 0.01
    torch.save(poses, str(path))


def test_load_optimized_poses_raw_bin_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "poses.bin"
    _write_raw_fp16(p, 600)
    loaded = load_optimized_poses(p, pose_dim=6, expected_n_pairs=600)
    assert loaded.shape == (600, 6)
    assert loaded.dtype == torch.float32


def test_load_optimized_poses_pickle_pt_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "poses.pt"
    _write_pickle_pt(p, 600)
    loaded = load_optimized_poses(p, pose_dim=6, expected_n_pairs=600)
    assert loaded.shape == (600, 6)


def test_load_optimized_poses_pickle_renamed_to_bin(tmp_path: Path) -> None:
    """The exact 2026-04-26 SHIRAZ scenario: a .pt pickle masquerading as
    .bin. The loader MUST detect by content (pickle magic bytes), load via
    torch.load, and succeed — not crash with frombuffer reshape errors."""
    pkl = tmp_path / "real.pt"
    _write_pickle_pt(pkl, 600)
    bin_path = tmp_path / "renamed.bin"
    bin_path.write_bytes(pkl.read_bytes())  # blind copy
    loaded = load_optimized_poses(bin_path, pose_dim=6, expected_n_pairs=600)
    assert loaded.shape == (600, 6)


def test_load_optimized_poses_partial_caught_by_count(tmp_path: Path) -> None:
    """The 2026-04-26 SHIRAZ run shipped 60 of 600 poses. With expected_n_pairs
    set, the loader MUST refuse rather than silently return a short tensor
    that downstream FiLM conditioning would zero-pad over."""
    p = tmp_path / "poses.bin"
    _write_raw_fp16(p, 60)  # partial
    with pytest.raises(ValueError) as excinfo:
        load_optimized_poses(p, pose_dim=6, expected_n_pairs=600)
    assert "60" in str(excinfo.value)
    assert "600" in str(excinfo.value)


def test_load_optimized_poses_misaligned_buffer(tmp_path: Path) -> None:
    """A buffer whose size isn't a multiple of pose_dim*2 bytes is corrupt.
    The 2026-04-26 SHIRAZ pickle-as-bin had file size 15724 → 7862 fp16
    elements → not divisible by 6. The loader must fail loudly with a
    diagnostic that names the actual sizes, not just `RuntimeError: shape ...
    is invalid for input of size N`."""
    p = tmp_path / "garbage.bin"
    p.write_bytes(b"\x00" * 15724)  # not pickle, not multiple of 12
    with pytest.raises(ValueError) as excinfo:
        load_optimized_poses(p, pose_dim=6)
    msg = str(excinfo.value).lower()
    assert "multiple" in msg
    assert "15724" in str(excinfo.value)


def test_load_optimized_poses_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        load_optimized_poses(p)


def test_load_optimized_poses_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_optimized_poses(tmp_path / "does-not-exist.bin")


def test_pickle_magic_recognises_torch_save_outputs(tmp_path: Path) -> None:
    """torch.save uses ZIP container by default (PK\\x03\\x04). Older protocols
    use raw pickle (\\x80\\x05). Both must be recognised so we don't accidentally
    drop into the raw-buffer branch on a pickle."""
    p = tmp_path / "a.pt"
    torch.save(torch.zeros(10, 6), str(p))
    raw = p.read_bytes()
    assert _looks_like_pickle(raw), f"first 8 bytes = {raw[:8]!r}"

    # Synthesise a legacy pickle protocol 4 buffer too.
    legacy = b"\x80\x04" + b"\x00" * 30
    assert _looks_like_pickle(legacy)


def test_pickle_magic_rejects_raw_fp16_buffer() -> None:
    raw_fp16 = torch.randn(10, 6, dtype=torch.float16).numpy().tobytes()
    assert not _looks_like_pickle(raw_fp16)


# ── Preflight: pgrep -f self-match (the 2026-04-26 deadlock) ──────────────


def test_self_match_pgrep_in_bash_argv_string() -> None:
    """The exact bash -c argv that deadlocked SHIRAZ on 2026-04-26."""
    argv = (
        "while pgrep -f train_distill > /dev/null; do sleep 60; done; "
        "bash run_pipeline.sh && python -u experiments/train_distill.py"
    )
    violations = _scan_text_for_dangerous_patterns(argv, "argv")
    self_match = [v for v in violations if "SELF-MATCH" in v]
    assert len(self_match) == 1, violations
    assert "train_distill" in self_match[0]


def test_self_match_pgrep_in_python_fstring(tmp_path: Path) -> None:
    """The same pattern composed via Python f-string (deploy_vastai-style)."""
    py = tmp_path / "fake_deploy.py"
    py.write_text(
        "import subprocess\n"
        "def launch(host, port):\n"
        "    cmd = f'while pgrep -f train_distill ; do sleep 60; done; "
        "bash run_pipeline.sh && python experiments/train_distill.py'\n"
        "    subprocess.run(['ssh', f'-p{port}', host, cmd])\n"
    )
    violations = _scan_python_for_forbidden(py)
    self_match = [v for v in violations if "SELF-MATCH" in v]
    assert len(self_match) >= 1, violations


def test_auth_eval_mps_python_invocation_caught(tmp_path: Path) -> None:
    py = tmp_path / "auth_eval_mps.py"
    py.write_text(
        "import subprocess\n"
        "subprocess.run([\n"
        "    'python', 'experiments/auth_eval_renderer.py',\n"
        "    '--device', 'mps',\n"
        "])\n"
    )
    violations = _scan_python_for_forbidden(py)
    assert any("auth_eval invocation" in v for v in violations), violations


def test_auth_eval_cuda_python_invocation_passes(tmp_path: Path) -> None:
    py = tmp_path / "auth_eval_cuda.py"
    py.write_text(
        "import subprocess\n"
        "subprocess.run([\n"
        "    'python', 'experiments/auth_eval_renderer.py',\n"
        "    '--device', 'cuda',\n"
        "])\n"
    )
    assert _scan_python_for_forbidden(py) == []


def test_pgrep_unique_token_does_not_false_positive(tmp_path: Path) -> None:
    """A wait loop with a token that appears NOWHERE else in the file is fine
    (the token won't self-match). Don't punish correct usage."""
    sh = tmp_path / "ok.sh"
    sh.write_text(
        "#!/bin/bash\n"
        "while pgrep -f COOKIE_abc12345 > /dev/null; do sleep 5; done\n"
        "echo ready\n"
    )
    violations = _scan_bash_text_for_forbidden(sh)
    assert not [v for v in violations if "SELF-MATCH" in v], violations


# ── Preflight: blind .pt → .bin rename ────────────────────────────────────


def test_blind_rename_cp_caught() -> None:
    text = "cp model.pt model.bin"
    violations = _scan_text_for_dangerous_patterns(text, "test")
    assert any("renames a pickle .pt to raw .bin" in v for v in violations), violations


def test_blind_rename_mv_with_flags_caught() -> None:
    text = "mv -f /tmp/foo.pt /tmp/bar.bin && echo done"
    violations = _scan_text_for_dangerous_patterns(text, "test")
    assert any("renames a pickle .pt to raw .bin" in v for v in violations)


def test_innocent_cp_does_not_false_positive() -> None:
    cases = [
        "cp model.bin model.bin.backup",
        "cp /tmp/a.txt /tmp/b.txt",
        "mv old.pt new.pt",
    ]
    for c in cases:
        v = _scan_text_for_dangerous_patterns(c, "test")
        renames = [x for x in v if "renames a pickle" in x]
        assert not renames, f"false positive on {c!r}: {renames}"


def test_partial_artifact_reference_caught(tmp_path: Path) -> None:
    sh = tmp_path / "wrapper.sh"
    sh.write_text(
        "#!/bin/bash\n"
        "cp /tmp/run/optimized_poses_partial.pt /tmp/run/poses.pt\n"
    )
    violations = _scan_bash_text_for_forbidden(sh)
    assert any("_partial" in v for v in violations), violations


def test_partial_glob_alone_is_fine() -> None:
    """Listing or grepping a partial file is harmless — only flag when the
    partial is being SHIPPED (cp/mv/archive). False-positives on a docstring
    or `--resume` arg blocked unrelated commits."""
    text = "ls /tmp/run/*_partial.pt | head -1"
    violations = _scan_text_for_dangerous_patterns(text, "test")
    assert not any("_partial" in v for v in violations), violations


def test_partial_being_copied_to_archive_caught(tmp_path: Path) -> None:
    """The 2026-04-26 retto wrapper: cp $(ls *_partial.pt) /tmp/.../archive/...
    When the partial is being copied/moved into an archive directory, that's
    the bug pattern."""
    sh = tmp_path / "ship.sh"
    sh.write_text(
        "#!/bin/bash\n"
        "cp /tmp/run/optimized_poses_partial.pt /tmp/run/archive/optimized_poses.bin\n"
    )
    violations = _scan_bash_text_for_forbidden(sh)
    # Should fire on either the rename rule OR the partial-ship rule (or both).
    assert any("_partial" in v or "renames a pickle" in v for v in violations), violations


def test_resume_from_partial_does_not_false_positive(tmp_path: Path) -> None:
    """A producer script that resumes from its own partial is fine — it
    should not be punished by the preflight rule."""
    py = tmp_path / "resume_ok.py"
    py.write_text(
        '"""Some script with --resume <partial> support."""\n'
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--resume', type=str, help='Resume from partial .pt')\n"
        "# Example: --resume experiments/results/x/latent_codes_partial.pt\n"
    )
    violations = _scan_python_for_forbidden(py)
    assert not [v for v in violations if "_partial" in v], violations
