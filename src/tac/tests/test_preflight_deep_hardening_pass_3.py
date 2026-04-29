"""Tests for Checks 58, 59, 60, 61 — deep hardening pass 3 dimension 2.

Memory: feedback_deep_hardening_pass_3_patterns_20260428.

Check 58: launcher --max-dph floor (≥ 0.40) so NVDEC_BAD attrition doesn't
          starve the host pool.
Check 59: cmd_phase2_extract MUST destroy on CUDA-probe failure.
Check 60: MEMORY.md size ceiling (250 lines soft, warn-only).
Check 61: canonical bootstrap scripts MUST write provenance.json with
          git_hash + gpu_name fields.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_canonical_bootstraps_write_provenance,
    check_launcher_max_dph_floor,
    check_memory_md_size_under_ceiling,
    check_phase2_extract_destroys_on_failure,
)


def _make_scripts_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for name, body in files.items():
        (scripts_dir / name).write_text(textwrap.dedent(body).lstrip("\n"))
    return tmp_path


# ─── Check 58: launcher --max-dph floor ───────────────────────────────────


def test_check58_passes_above_floor(tmp_path: Path) -> None:
    repo = _make_scripts_repo(tmp_path, {
        "launch_lane_on_vastai.py": """
            #!/usr/bin/env python3
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--max-dph", type=float, default=0.50)
        """,
    })
    violations = check_launcher_max_dph_floor(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], f"expected 0 violations, got: {violations}"


def test_check58_fails_below_floor(tmp_path: Path) -> None:
    repo = _make_scripts_repo(tmp_path, {
        "launch_lane_on_vastai.py": """
            #!/usr/bin/env python3
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--max-dph", type=float, default=0.30)
        """,
    })
    violations = check_launcher_max_dph_floor(
        repo_root=repo, strict=False, verbose=False,
    )
    assert any("0.3" in v for v in violations), f"got: {violations}"
    with pytest.raises(MetaBugViolation, match="--max-dph BELOW FLOOR"):
        check_launcher_max_dph_floor(
            repo_root=repo, strict=True, verbose=False,
        )


def test_check58_waiver_allows_below_floor(tmp_path: Path) -> None:
    repo = _make_scripts_repo(tmp_path, {
        "launch_lane_on_vastai.py": """
            #!/usr/bin/env python3
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--max-dph", type=float, default=0.20)  # MAX_DPH_OK: legacy promo lane
        """,
    })
    violations = check_launcher_max_dph_floor(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], f"expected 0 with waiver, got: {violations}"


def test_check58_skips_comment_lines(tmp_path: Path) -> None:
    repo = _make_scripts_repo(tmp_path, {
        "launch_lane_on_vastai.py": """
            #!/usr/bin/env python3
            # historical: --max-dph 0.10 was tried in 2026-04-15 and starved
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--max-dph", type=float, default=0.50)
        """,
    })
    violations = check_launcher_max_dph_floor(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], f"comment line should be skipped, got: {violations}"


# ─── Check 59: cmd_phase2_extract destroys on failure ──────────────────────


def test_check59_passes_when_both_present(tmp_path: Path) -> None:
    repo = _make_scripts_repo(tmp_path, {
        "launch_lane_on_vastai.py": """
            def cmd_phase2_extract(args) -> int:
                ok, msg = lightweight_nvdec_probe(host, port)
                if not ok:
                    destroy_instance(instance_id)
                    return 1
                return 0
        """,
    })
    violations = check_phase2_extract_destroys_on_failure(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], f"expected 0, got: {violations}"


def test_check59_fails_when_destroy_missing(tmp_path: Path) -> None:
    repo = _make_scripts_repo(tmp_path, {
        "launch_lane_on_vastai.py": """
            def cmd_phase2_extract(args) -> int:
                ok, msg = lightweight_nvdec_probe(host, port)
                if not ok:
                    return 1  # FORGOT to destroy
                return 0
        """,
    })
    violations = check_phase2_extract_destroys_on_failure(
        repo_root=repo, strict=False, verbose=False,
    )
    assert any("destroy_instance" in v for v in violations), f"got: {violations}"
    with pytest.raises(MetaBugViolation, match="MUST AUTO-DESTROY"):
        check_phase2_extract_destroys_on_failure(
            repo_root=repo, strict=True, verbose=False,
        )


def test_check59_fails_when_probe_missing(tmp_path: Path) -> None:
    repo = _make_scripts_repo(tmp_path, {
        "launch_lane_on_vastai.py": """
            def cmd_phase2_extract(args) -> int:
                # no NVDEC probe at all
                destroy_instance(instance_id)
                return 1
        """,
    })
    violations = check_phase2_extract_destroys_on_failure(
        repo_root=repo, strict=False, verbose=False,
    )
    assert any("lightweight_nvdec_probe" in v for v in violations), f"got: {violations}"


# ─── Check 60: MEMORY.md size ceiling ──────────────────────────────────────


def test_check60_passes_under_ceiling(tmp_path: Path, monkeypatch) -> None:
    # Place a small repo-level MEMORY.md
    (tmp_path / "MEMORY.md").write_text("\n".join(f"line {i}" for i in range(50)))
    # No home-level memory file in tmp
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "no-home")
    violations = check_memory_md_size_under_ceiling(
        repo_root=tmp_path, strict=False, verbose=False, ceiling=250,
    )
    assert violations == [], f"50 lines should pass, got: {violations}"


def test_check60_fails_over_ceiling(tmp_path: Path, monkeypatch) -> None:
    big = "\n".join(f"line {i}" for i in range(300))
    (tmp_path / "MEMORY.md").write_text(big)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "no-home")
    violations = check_memory_md_size_under_ceiling(
        repo_root=tmp_path, strict=False, verbose=False, ceiling=250,
    )
    assert len(violations) == 1, f"got: {violations}"
    assert "300 lines" in violations[0]
    with pytest.raises(MetaBugViolation, match="EXCEEDS"):
        check_memory_md_size_under_ceiling(
            repo_root=tmp_path, strict=True, verbose=False, ceiling=250,
        )


# ─── Check 61: bootstrap provenance ────────────────────────────────────────


def test_check61_passes_when_provenance_written(tmp_path: Path) -> None:
    repo = _make_scripts_repo(tmp_path, {
        "remote_train_bootstrap.sh": """
            #!/bin/bash
            cat > provenance.json <<EOF
            {"git_hash": "$GIT_HASH", "gpu_name": "$GPU_NAME"}
            EOF
        """,
        "remote_pose_tto_bootstrap.sh": """
            #!/bin/bash
            python -c "import json; json.dump({'git_hash': h, 'gpu_name': g}, open('provenance.json', 'w'))"
        """,
    })
    violations = check_canonical_bootstraps_write_provenance(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], f"got: {violations}"


def test_check61_fails_when_provenance_missing(tmp_path: Path) -> None:
    repo = _make_scripts_repo(tmp_path, {
        "remote_train_bootstrap.sh": """
            #!/bin/bash
            echo "no provenance written"
        """,
    })
    violations = check_canonical_bootstraps_write_provenance(
        repo_root=repo, strict=False, verbose=False,
    )
    assert any("does not write provenance.json" in v for v in violations), f"got: {violations}"


def test_check61_fails_when_required_field_missing(tmp_path: Path) -> None:
    repo = _make_scripts_repo(tmp_path, {
        "remote_train_bootstrap.sh": """
            #!/bin/bash
            cat > provenance.json <<EOF
            {"git_hash": "$GIT_HASH"}
            EOF
        """,
    })
    violations = check_canonical_bootstraps_write_provenance(
        repo_root=repo, strict=False, verbose=False,
    )
    assert any("gpu_name" in v for v in violations), f"got: {violations}"
