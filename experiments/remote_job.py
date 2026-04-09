from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_remote_launch_script(
    *,
    remote_root: str,
    remote_log: str,
    remote_command: str,
) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {remote_root}",
            f"mkdir -p {Path(remote_log).parent.as_posix()}",
            f"nohup {remote_command} > {remote_log} 2>&1 < /dev/null &",
            "echo $!",
        ]
    )


def write_manifest(*, manifest_path: Path, data: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["written_at"] = now_iso()
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a remote-job manifest")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-log", required=True)
    parser.add_argument("--remote-command", required=True)
    parser.add_argument("--remote-pid", type=int, default=None)
    parser.add_argument("--manifest-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    script = render_remote_launch_script(
        remote_root=args.remote_root,
        remote_log=args.remote_log,
        remote_command=args.remote_command,
    )
    write_manifest(
        manifest_path=args.manifest_path,
        data={
            "slug": args.slug,
            "host": args.host,
            "remote_root": args.remote_root,
            "remote_log": args.remote_log,
            "remote_command": args.remote_command,
            "remote_pid": args.remote_pid,
            "remote_launch_script": script,
        },
    )
    print(script)
    print(f"manifest_written {args.manifest_path}")


if __name__ == "__main__":
    main()
