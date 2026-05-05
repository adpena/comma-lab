from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "recovery_quarantine_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("recovery_quarantine_audit_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quarantine_audit_normalizes_recovery_suffixes_before_live_comparison(tmp_path: Path) -> None:
    mod = _load_module()
    live = tmp_path / "scripts" / "remote_lane_x.sh"
    live.parent.mkdir(parents=True)
    live.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")

    quarantine = tmp_path / "q"
    q_path = quarantine / "scripts" / "remote_lane_x.sh.PREFLIGHT_DEBT"
    q_path.parent.mkdir(parents=True)
    q_path.write_text(live.read_text(encoding="utf-8"), encoding="utf-8")

    [record] = mod.iter_records(tmp_path, quarantine)

    assert record.relpath == "scripts/remote_lane_x.sh.PREFLIGHT_DEBT"
    assert record.live_exists is True
    assert record.live_same_sha256 is True
    assert record.live_relpath == "scripts/remote_lane_x.sh"
    assert record.disposition == "duplicate_same_safe_to_delete_after_manifest_commit"
