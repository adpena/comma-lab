"""Tests for Catalog #137 — remote dispatch runbooks must NOT default to local CUDA probe.

Defense-in-depth on codex round-3 MEDIUM 1 (2026-05-09): the round-3 fix patched
ONE runbook (`scripts/remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh`).
This META gate refuses ANY future `scripts/remote_lane_*.sh` from introducing
an unguarded local CUDA/NVDEC probe.

Memory: feedback_production_hardening_polish_defense_in_depth_landed_20260509.md.
Cross-ref CLAUDE.md "Remote code parity" + "Vast.ai cost paranoia".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_remote_dispatch_runbooks_no_local_cuda_probe_default,
)


def _make_scripts(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir(parents=True)
    return tmp_path


# ── Catches probe tokens without guards ──────────────────────────────────


def test_137_catches_probe_nvdec_unguarded(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_test_bad.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "probe_nvdec.sh\n"
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "remote_lane_test_bad.sh" in x]
    assert len(matches) == 1
    assert "probe_nvdec.sh" in matches[0]


def test_137_catches_nvidia_smi_unguarded(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_test_bad.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "nvidia-smi --query-gpu=name --format=csv\n"
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert any("remote_lane_test_bad.sh" in x and "nvidia-smi" in x for x in v)


def test_137_catches_nvcc_unguarded(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_test_bad.sh").write_text(
        "#!/bin/bash\n"
        "nvcc --version\n"
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert any("remote_lane_test_bad.sh" in x and "nvcc" in x for x in v)


def test_137_catches_torch_cuda_is_available_unguarded(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_test_bad.sh").write_text(
        "#!/bin/bash\n"
        'python -c "import torch; print(torch.cuda.is_available())"\n'
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert any(
        "remote_lane_test_bad.sh" in x and "torch.cuda.is_available()" in x
        for x in v
    )


# ── Accept guarded probes ────────────────────────────────────────────────


def test_137_accepts_local_cuda_worker_guard(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_ok.sh").write_text(
        "#!/bin/bash\n"
        'if [[ "${LOCAL_CUDA_WORKER:-0}" == "1" ]]; then\n'
        "    nvidia-smi\n"
        "fi\n"
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert all("remote_lane_ok.sh" not in x for x in v)


def test_137_accepts_dry_run_short_circuit(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_ok.sh").write_text(
        "#!/bin/bash\n"
        'if [[ "${DRY_RUN:-0}" != "1" ]]; then\n'
        "    nvidia-smi\n"
        "fi\n"
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert all("remote_lane_ok.sh" not in x for x in v)


def test_137_accepts_remote_execution_wrapper_vastai(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_ok.sh").write_text(
        "#!/bin/bash\n"
        'vastai exec "$INSTANCE_ID" "nvidia-smi"\n'
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert all("remote_lane_ok.sh" not in x for x in v)


def test_137_accepts_remote_execution_wrapper_lightning(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_ok.sh").write_text(
        "#!/bin/bash\n"
        "lightning ssh\n"
        "nvidia-smi\n"
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert all("remote_lane_ok.sh" not in x for x in v)


def test_137_accepts_remote_execution_wrapper_modal(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_ok.sh").write_text(
        "#!/bin/bash\n"
        "modal run my_app::probe_gpu\n"
        "nvidia-smi\n"
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert all("remote_lane_ok.sh" not in x for x in v)


def test_137_accepts_ssh_remote_host(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_ok.sh").write_text(
        "#!/bin/bash\n"
        'ssh "$REMOTE_HOST" "nvidia-smi"\n'
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert all("remote_lane_ok.sh" not in x for x in v)


def test_137_accepts_per_line_waiver(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_waived.sh").write_text(
        "#!/bin/bash\n"
        "nvidia-smi  # LOCAL_CUDA_PROBE_OK: operator workstation has CUDA + DALI installed\n"
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert all("remote_lane_waived.sh" not in x for x in v)


# ── Out-of-scope patterns ────────────────────────────────────────────────


def test_137_ignores_non_remote_lane_scripts(tmp_path):
    """Scripts that aren't `remote_lane_*.sh` are out-of-scope (e.g.
    `remote_archive_only_eval.sh` is a remote-side bootstrap that legitimately
    runs nvidia-smi ON the GPU)."""
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_archive_only_eval.sh").write_text(
        "#!/bin/bash\n"
        "nvidia-smi\n"
    )
    (root / "scripts" / "remote_train_bootstrap.sh").write_text(
        "#!/bin/bash\n"
        "nvidia-smi\n"
    )
    (root / "scripts" / "build_renderer.sh").write_text(
        "#!/bin/bash\n"
        "nvidia-smi\n"
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert all(
        "remote_archive_only_eval.sh" not in x
        and "remote_train_bootstrap.sh" not in x
        and "build_renderer.sh" not in x
        for x in v
    )


def test_137_ignores_comments(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_ok.sh").write_text(
        "#!/bin/bash\n"
        "# Run nvidia-smi after SSH'ing to the GPU\n"
    )
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=root, strict=False, verbose=False
    )
    assert all("remote_lane_ok.sh" not in x for x in v)


# ── Strict-mode round-trip ───────────────────────────────────────────────


def test_137_strict_raises(tmp_path):
    root = _make_scripts(tmp_path)
    (root / "scripts" / "remote_lane_test_bad.sh").write_text(
        "#!/bin/bash\n"
        "nvidia-smi\n"
    )
    with pytest.raises(
        PreflightError,
        match="check_remote_dispatch_runbooks_no_local_cuda_probe_default",
    ):
        check_remote_dispatch_runbooks_no_local_cuda_probe_default(
            repo_root=root, strict=True, verbose=False
        )


def test_137_no_scripts_dir_returns_empty(tmp_path):
    """Tmp dir without `scripts/` returns empty, doesn't crash."""
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert v == []


# ── Live-repo sanity ─────────────────────────────────────────────────────


def test_137_live_repo_clean():
    """Live-repo sanity: catalog #137 must land at 0 violations.

    Round-3's fix already guarded the only known instance
    (`scripts/remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh`).
    """
    v = check_remote_dispatch_runbooks_no_local_cuda_probe_default(
        strict=False, verbose=False
    )
    assert v == [], (
        f"Catalog #137 landed with {len(v)} violations:\n"
        + "\n".join(v[:3])
    )
