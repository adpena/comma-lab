# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ACTIVE_STATUSES = {"running", "queued"}
REPUSHABLE_PHASES = {"quota_blocked", "kernel_error", "kernel_cancelled", "kernel_cancel_acknowledged"}


def _counts_as_remote_active(status: str, phase: str) -> bool:
    if status == "running":
        return True
    if status == "queued" and phase in {"kernel_queued"}:
        return True
    return False


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _status_payload(manifest_path: Path) -> dict[str, object]:
    manifest = _read_json(manifest_path)
    status_path = manifest.get("status_path")
    if isinstance(status_path, str) and status_path and Path(status_path).exists():
        return _read_json(Path(status_path))
    return manifest


def select_manifest_for_repush(manifest_paths: list[Path], *, max_active: int = 2) -> Path | None:
    active = 0
    candidates: list[Path] = []
    for manifest_path in manifest_paths:
        payload = _status_payload(manifest_path)
        status = str(payload.get("status", "")).strip()
        phase = str(payload.get("phase", "")).strip()
        if _counts_as_remote_active(status, phase):
            active += 1
            continue
        if phase in REPUSHABLE_PHASES or status in {"paused", "error"}:
            candidates.append(manifest_path)
    if active >= max_active:
        return None
    return candidates[0] if candidates else None


def run_repush_command(
    command: str,
    *,
    command_runner: Callable[[str], object] | None = None,
) -> int:
    if command_runner is None:
        result = subprocess.run(shlex.split(command), shell=False, text=True, capture_output=True)  # subprocess-no-check-OK: returncode propagated to caller via line below
    else:
        result = command_runner(command)
    returncode = int(getattr(result, "returncode", 0))
    stdout = str(getattr(result, "stdout", "") or "")
    stderr = str(getattr(result, "stderr", "") or "")
    combined = f"{stdout}\n{stderr}"
    if "Kernel push error:" in combined:
        return 1
    return returncode


def tick_queue(
    *,
    manifests_dir: Path,
    max_active: int = 2,
    command_runner: Callable[[str], object] | None = None,
) -> dict[str, object]:
    manifests = sorted(manifests_dir.glob("kaggle-*.json"))
    selected = select_manifest_for_repush(manifests, max_active=max_active)
    if selected is None:
        return {"action": "noop", "reason": "no_slot_or_no_candidate"}

    manifest = _read_json(selected)
    command = manifest.get("remote_command")
    if not isinstance(command, str) or not command.strip():
        return {"action": "noop", "reason": "missing_remote_command", "manifest": str(selected)}

    rc = run_repush_command(command, command_runner=command_runner)
    status_path = manifest.get("status_path")
    if isinstance(status_path, str) and status_path:
        path = Path(status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    **_status_payload(selected),
                    "status": "queued" if rc == 0 else "error",
                    "phase": "repush_submitted" if rc == 0 else "repush_failed",
                    "written_at": now_iso(),
                },
                indent=2,
            )
        )
    return {
        "action": "repush",
        "manifest": str(selected),
        "returncode": rc,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Poll the Kaggle queue and repush a waiting manifest when a slot is free.")
    parser.add_argument("--manifests-dir", type=Path, default=Path(".omx/logs/remote_jobs"))
    parser.add_argument("--max-active", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = tick_queue(manifests_dir=args.manifests_dir, max_active=args.max_active)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
