from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json_file(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _canonical_status_text(text: str) -> str:
    return " ".join(text.strip().split())


def _merge_notes(existing: object, update: str | None) -> str | None:
    if not update:
        return None if existing is None else str(existing)
    if existing is None:
        return update
    existing_text = str(existing).strip()
    if not existing_text:
        return update
    if update in existing_text:
        return existing_text
    return f"{existing_text}\n{update}"


def parse_kaggle_status_text(text: str) -> dict[str, object]:
    raw = _canonical_status_text(text)
    lower = raw.lower()

    if "batch gpu session count" in lower or "quota" in lower or "not_pushed" in lower:
        return {
            "status": "paused",
            "phase": "quota_blocked",
            "kernel_status": raw,
            "notes": raw,
        }
    if "kernelworkerstatus.cancel_acknowledged" in lower:
        return {
            "status": "paused",
            "phase": "kernel_cancel_acknowledged",
            "kernel_status": raw,
            "notes": raw,
        }
    if "kernelworkerstatus.cancelled" in lower:
        return {
            "status": "paused",
            "phase": "kernel_cancelled",
            "kernel_status": raw,
            "notes": raw,
        }
    if "kernelworkerstatus.running" in lower or " running" in f" {lower}" or lower == "running":
        return {
            "status": "running",
            "phase": "kernel_running",
            "kernel_status": raw,
        }
    if "kernelworkerstatus.queued" in lower or " queued" in f" {lower}" or lower == "queued":
        return {
            "status": "queued",
            "phase": "kernel_queued",
            "kernel_status": raw,
        }
    if "kernelworkerstatus.paused" in lower or " paused" in f" {lower}" or lower == "paused":
        return {
            "status": "paused",
            "phase": "kernel_paused",
            "kernel_status": raw,
        }
    if "kernelworkerstatus.complete" in lower or " complete" in f" {lower}" or lower == "complete":
        return {
            "status": "complete",
            "phase": "kernel_complete",
            "kernel_status": raw,
        }
    if "kernelworkerstatus.error" in lower or " error" in f" {lower}" or lower == "error":
        return {
            "status": "error",
            "phase": "kernel_error",
            "kernel_status": raw,
        }
    return {
        "status": "unknown",
        "phase": "kernel_status_unknown",
        "kernel_status": raw,
        "notes": raw,
    }


def query_kaggle_status_text(
    kernel_ref: str,
    *,
    command_runner: Callable[[list[str]], object] | None = None,
) -> str:
    if shutil.which("kaggle") is not None:
        command = ["kaggle", "kernels", "status", kernel_ref]
    else:
        command = ["uv", "run", "--with", "kaggle", "kaggle", "kernels", "status", kernel_ref]
    if command_runner is None:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    else:
        result = command_runner(command)
    returncode = getattr(result, "returncode", 0)
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    if returncode != 0:
        raise RuntimeError(f"kaggle kernels status failed for {kernel_ref}: {stderr or stdout}".strip())
    text = stdout if isinstance(stdout, str) and stdout.strip() else stderr
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"kaggle kernels status returned no usable output for {kernel_ref}")
    return text


def build_status_payload(
    manifest: dict[str, object],
    *,
    manifest_path: Path,
    status_text: str,
    output_path: Path | None = None,
) -> dict[str, object]:
    payload = dict(manifest)
    parsed = parse_kaggle_status_text(status_text)
    payload.update(parsed)
    if "manifest_path" not in payload:
        payload["manifest_path"] = str(manifest_path)
    if output_path is not None:
        payload["status_path"] = str(output_path)
    payload["written_at"] = now_iso()
    payload["notes"] = _merge_notes(manifest.get("notes"), parsed.get("notes") if isinstance(parsed.get("notes"), str) else None)
    if payload["notes"] is None:
        payload.pop("notes", None)
    return payload


def sync_kaggle_status(
    manifest_path: str | Path,
    *,
    status_text: str | None = None,
    output_path: str | Path | None = None,
    command_runner: Callable[[list[str]], object] | None = None,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    manifest = _read_json_file(manifest_path)
    kernel_ref = manifest.get("kernel_ref")
    if not isinstance(kernel_ref, str) or not kernel_ref.strip():
        raise ValueError(f"Missing kernel_ref in {manifest_path}")

    resolved_output = Path(output_path) if output_path is not None else None
    if status_text is None:
        status_text = query_kaggle_status_text(kernel_ref.strip(), command_runner=command_runner)
    payload = build_status_payload(
        manifest,
        manifest_path=manifest_path,
        status_text=status_text,
        output_path=resolved_output,
    )
    if resolved_output is None:
        status_path = manifest.get("status_path")
        if isinstance(status_path, str) and status_path.strip():
            resolved_output = Path(status_path)
    if resolved_output is None:
        raise ValueError(f"Missing status_path for {manifest_path}")
    _write_json_file(resolved_output, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync Kaggle status payloads from a manifest and CLI output.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the Kaggle remote-job manifest")
    parser.add_argument(
        "--status-text",
        default=None,
        help="Raw Kaggle status text to parse instead of querying the Kaggle CLI",
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=None,
        help="Override the output status file path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    sync_kaggle_status(
        args.manifest,
        status_text=args.status_text,
        output_path=args.status_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
