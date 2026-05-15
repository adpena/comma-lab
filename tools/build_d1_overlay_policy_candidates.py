#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize byte-closed D1 overlay policy candidates.

This tool does not score or promote candidates. It rewrites only the D1POLY1
metadata fields that select inflate-time overlay policy, then emits a complete
submission runtime tree and deterministic archive.zip for each policy tuple.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.train_substrate_d1_segnet_margin_polytope import (  # noqa: E402
    _build_archive_zip,
    _write_runtime,
)
from tac.substrates.d1_segnet_margin_polytope import (  # noqa: E402
    parse_archive,
    update_d1poly1_meta,
)
from tac.substrates.d1_segnet_margin_polytope.overlay import (  # noqa: E402
    D1_OVERLAY_AMPLITUDE_SCALES,
    D1_OVERLAY_CHANNEL_POLICIES,
    D1_OVERLAY_SIGN_POLICIES,
    normalize_overlay_amplitude_scale,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _parse_policies(raw: str) -> list[str]:
    policies = [item.strip() for item in raw.split(",") if item.strip()]
    if not policies:
        raise SystemExit("--policies produced an empty policy list")
    allowed = set(D1_OVERLAY_CHANNEL_POLICIES)
    unknown = [policy for policy in policies if policy not in allowed]
    if unknown:
        raise SystemExit(
            f"unsupported policies {unknown}; expected subset of {sorted(allowed)}"
        )
    return policies


def _parse_sign_policies(raw: str) -> list[str]:
    policies = [item.strip() for item in raw.split(",") if item.strip()]
    if not policies:
        raise SystemExit("--sign-policies produced an empty policy list")
    allowed = set(D1_OVERLAY_SIGN_POLICIES)
    unknown = [policy for policy in policies if policy not in allowed]
    if unknown:
        raise SystemExit(
            f"unsupported sign policies {unknown}; expected subset of {sorted(allowed)}"
        )
    return policies


def _parse_amplitude_scales(raw: str) -> list[float]:
    scales: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            scales.append(normalize_overlay_amplitude_scale(float(item)))
        except ValueError as exc:
            raise SystemExit(f"unsupported amplitude scale {item!r}: {exc}") from exc
    if not scales:
        raise SystemExit("--amplitude-scales produced an empty scale list")
    return scales


def _scale_slug(scale: float) -> str:
    text = f"{normalize_overlay_amplitude_scale(scale):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _candidate_id(*, channel_policy: str, amplitude_scale: float, sign_policy: str) -> str:
    return (
        f"d1_overlay_channel_{channel_policy}"
        f"_amp_{_scale_slug(amplitude_scale)}"
        f"_sign_{sign_policy}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build byte-closed D1 overlay policy candidates"
    )
    parser.add_argument("--d1-bin", type=Path, required=True)
    parser.add_argument("--a1-bin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policies",
        default=",".join(D1_OVERLAY_CHANNEL_POLICIES),
        help="Comma-separated channel policies to materialize.",
    )
    parser.add_argument(
        "--amplitude-scales",
        default="1.0",
        help=(
            "Comma-separated overlay attenuation scales in [0,1]. "
            f"Known schedule: {','.join(str(v) for v in D1_OVERLAY_AMPLITUDE_SCALES)}."
        ),
    )
    parser.add_argument(
        "--sign-policies",
        default="payload",
        help="Comma-separated sign policies to materialize.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    d1_bytes = args.d1_bin.read_bytes()
    a1_bytes = args.a1_bin.read_bytes()
    source = parse_archive(d1_bytes)
    a1_sha = _sha256_bytes(a1_bytes)
    if a1_sha[:16] != source.base_archive_sha256_truncated:
        raise SystemExit(
            "A1 base sha mismatch: "
            f"{a1_sha[:16]} != {source.base_archive_sha256_truncated}"
        )

    policies = _parse_policies(args.policies)
    amplitude_scales = _parse_amplitude_scales(args.amplitude_scales)
    sign_policies = _parse_sign_policies(args.sign_policies)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for amplitude_scale in amplitude_scales:
            for sign_policy in sign_policies:
                candidate_id = _candidate_id(
                    channel_policy=policy,
                    amplitude_scale=amplitude_scale,
                    sign_policy=sign_policy,
                )
                candidate_root = args.output_dir / candidate_id
                submission_dir = candidate_root / "submission_dir"
                candidate_root.mkdir(parents=True, exist_ok=True)

                d1_variant = update_d1poly1_meta(
                    d1_bytes,
                    {
                        "overlay_channel_policy": policy,
                        "overlay_amplitude_scale": amplitude_scale,
                        "overlay_sign_policy": sign_policy,
                    },
                )
                parsed_variant = parse_archive(d1_variant)
                _write_runtime(submission_dir)
                (submission_dir / "d1_polytope.bin").write_bytes(d1_variant)
                (submission_dir / "a1.bin").write_bytes(a1_bytes)
                archive_zip = candidate_root / "archive.zip"
                _build_archive_zip(
                    archive_zip,
                    d1_bin_bytes=d1_variant,
                    base_bin_bytes=a1_bytes,
                    base_substrate_id="a1",
                )
                archive_bytes = archive_zip.read_bytes()
                (submission_dir / "archive.zip").write_bytes(archive_bytes)
                row = {
                    "candidate_id": candidate_id,
                    "overlay_channel_policy": policy,
                    "overlay_amplitude_scale": amplitude_scale,
                    "overlay_sign_policy": sign_policy,
                    "candidate_root": candidate_root,
                    "submission_dir": submission_dir,
                    "archive_zip": archive_zip,
                    "archive_bytes": len(archive_bytes),
                    "archive_sha256": _sha256_bytes(archive_bytes),
                    "d1_bin_bytes": len(d1_variant),
                    "d1_bin_sha256": _sha256_bytes(d1_variant),
                    "d1_meta_policy_tuple": {
                        "overlay_channel_policy": parsed_variant.meta.get(
                            "overlay_channel_policy"
                        ),
                        "overlay_amplitude_scale": parsed_variant.meta.get(
                            "overlay_amplitude_scale"
                        ),
                        "overlay_sign_policy": parsed_variant.meta.get(
                            "overlay_sign_policy"
                        ),
                    },
                    "source_d1_bin_sha256": _sha256_bytes(d1_bytes),
                    "a1_bin_sha256": a1_sha,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_blockers": [
                        "not_paired_contest_cpu_cuda_exact_eval",
                        "no_dispatch_claim_for_policy_candidate",
                    ],
                }
                rows.append(row)
                _write_json(candidate_root / "candidate_manifest.json", row)

    summary = {
        "tool": "tools/build_d1_overlay_policy_candidates.py",
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": _git_head(),
        "source_d1_bin": args.d1_bin,
        "source_a1_bin": args.a1_bin,
        "source_d1_bin_sha256": _sha256_bytes(d1_bytes),
        "a1_bin_sha256": a1_sha,
        "channel_policies": policies,
        "amplitude_scales": amplitude_scales,
        "sign_policies": sign_policies,
        "policy_count": len(rows),
        "candidates": rows,
        "score_claim": False,
        "promotion_eligible": False,
    }
    _write_json(args.output_dir / "d1_overlay_policy_candidates_manifest.json", summary)
    print(json.dumps(summary, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
