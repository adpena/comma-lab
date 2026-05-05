"""Tests for tools/check_dispatch_wrapper_stages_implemented.py (PCC11).

The CRITICAL test reproduces the council Q5-B4 example: a wrapper with
`# Stage 5: contest-CUDA eval` followed only by `echo "TODO"` must be flagged.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PCC11 = REPO_ROOT / "tools" / "check_dispatch_wrapper_stages_implemented.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_pcc11_test", PCC11)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_repo(tmp_path: Path, name: str, content: str) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / name).write_text(content)
    return tmp_path


def test_pcc11_catches_comment_only_stage(tmp_path: Path) -> None:
    """The canonical Q5-B4 example: stage label + only echo TODO."""
    mod = _load_module()
    repo = _make_repo(tmp_path, "remote_buggy.sh", """#!/usr/bin/env bash
set -euo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }

log "=== Stage 1: real work ==="
python compute.py --input data.pt

# Stage 5: contest-CUDA eval
echo "TODO"
""")
    violations = mod.scan(repo, window=20)
    assert any("Stage 5" in v.stage_label for v in violations), (
        f"expected Stage 5 violation. Got: {[v.stage_label for v in violations]}"
    )


def test_pcc11_does_not_flag_implemented_stage(tmp_path: Path) -> None:
    """Stage with real python invocation must NOT be flagged."""
    mod = _load_module()
    repo = _make_repo(tmp_path, "remote_good.sh", """#!/usr/bin/env bash
set -euo pipefail

# Stage 1: train model
python -u experiments/train.py --epochs 100

# Stage 2: build archive
$PYBIN scripts/build_archive.py --output archive.zip
""")
    violations = mod.scan(repo, window=20)
    assert violations == [], f"unexpected violations: {[v.stage_label for v in violations]}"


def test_pcc11_dedupes_consecutive_stage_labels(tmp_path: Path) -> None:
    """A header comment followed by `log` echo of the same stage is one stage."""
    mod = _load_module()
    repo = _make_repo(tmp_path, "remote_dedup.sh", """#!/usr/bin/env bash
set -euo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }

# Stage 0: GPU presence check
# (implemented via embedded Python below)
log "=== Stage 0: GPU presence check ==="
python -c "import torch; assert torch.cuda.is_available()"

# Stage 1: real work
python compute.py
""")
    violations = mod.scan(repo, window=20)
    assert violations == [], f"unexpected violations after dedup: {[v.stage_label for v in violations]}"


def test_pcc11_strict_exits_nonzero(tmp_path: Path) -> None:
    """--strict mode exits 1 when violations exist.

    Uses padding lines + a body-content stage to bypass the docstring
    heuristic and the SKIP_MARKERS regex (which skips TODO/STUB/etc).
    """
    mod = _load_module()
    repo = _make_repo(tmp_path, "remote_buggy_strict.sh", """#!/usr/bin/env bash
set -euo pipefail
log() { echo "[$(date)] $*"; }

log "=== Stage 1: real implementation ==="
python compute.py --input data.pt
log "Stage 1 complete"

# A bunch of padding to push past docstring detection
A=1
B=2
C=3

# Stage 9: contest-CUDA promotion check
echo "informational only — no real work"
""")
    rc = mod.main(["--repo-root", str(repo), "--strict"])
    assert rc == 1


def test_pcc11_only_scans_dispatch_wrappers(tmp_path: Path) -> None:
    """Files that don't match the dispatch glob are out of scope."""
    mod = _load_module()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    # Random file with stage labels but not a dispatch wrapper
    (scripts / "random_helper.sh").write_text("# Stage 1: stale\necho TODO\n")
    violations = mod.scan(tmp_path, window=20)
    assert violations == [], "random_helper.sh should not be scanned"
