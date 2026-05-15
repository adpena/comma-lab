#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile PR101 FEC6 selector bytes against byte-closed FEC7 prototypes.

This tool reads an existing PR101/FEC6 archive, reuses the parser-section
manifest and existing PR101 selector packer decoder, then builds no-score FEC7
range/adaptive selector payload prototypes.  It answers one narrow question:
can a byte-closed FEC7 selector replacement save at least the requested charged
bytes versus the current FEC6 selector payload?
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.hnerv_packet_sections import (  # noqa: E402
    PARSER_PR101_FEC6,
    build_packet_section_manifest,
)
from tac.packet_compiler.pr101_fec7_selector import (  # noqa: E402
    build_fec7_candidates,
    profile_selector_encodings,
)
from tac.repo_io import repo_relative, sha256_bytes, write_json  # noqa: E402

DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)
DEFAULT_TARGET_SAVING_BYTES = 79


def _load_pr101_builder() -> Any:
    path = REPO_ROOT / "tools" / "build_pr101_frame_exploit_selector_packet.py"
    spec = importlib.util.spec_from_file_location("build_pr101_frame_exploit_selector_packet", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import PR101 selector builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_single_member_payload(archive: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected one archive member in {archive}, found {len(infos)}")
        name = infos[0].filename
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"unsafe archive member name: {name!r}")
        return name, zf.read(name)


def _extract_selector_payload(wrapper_payload: bytes) -> bytes:
    if len(wrapper_payload) < 10:
        raise ValueError("PR101 FEC6 wrapper truncated")
    if wrapper_payload[:4] != b"FP11":
        raise ValueError(f"PR101 FEC6 wrapper magic mismatch: {wrapper_payload[:4]!r}")
    source_len = struct.unpack_from("<I", wrapper_payload, 4)[0]
    selector_len_offset = 8 + source_len
    if selector_len_offset + 2 > len(wrapper_payload):
        raise ValueError("PR101 FEC6 wrapper truncated before selector length")
    selector_len = struct.unpack_from("<H", wrapper_payload, selector_len_offset)[0]
    selector_start = selector_len_offset + 2
    selector_end = selector_start + selector_len
    selector_payload = wrapper_payload[selector_start:selector_end]
    if len(selector_payload) != selector_len or selector_end != len(wrapper_payload):
        raise ValueError("PR101 FEC6 selector payload length mismatch")
    if not selector_payload.startswith(b"FEC6"):
        raise ValueError(f"expected FEC6 selector payload, got {selector_payload[:4]!r}")
    return selector_payload


def profile_archive(
    archive: Path,
    *,
    target_saving_bytes: int = DEFAULT_TARGET_SAVING_BYTES,
    pairmod_contexts: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    builder = _load_pr101_builder()
    member_name, wrapper_payload = _read_single_member_payload(archive)
    manifest = build_packet_section_manifest(
        archive,
        label="pr101_fec6_selector_entropy_source",
        parser=PARSER_PR101_FEC6,
        repo_root=REPO_ROOT,
    )
    if manifest.get("score_claim") is not False:
        raise ValueError("parser manifest unexpectedly carries a score claim")
    selector_payload = _extract_selector_payload(wrapper_payload)
    _source_payload, selector_codes = builder.unpack_pr101_selector_payload(wrapper_payload)
    if len(selector_codes) != 600:
        raise ValueError(f"expected 600 selector codes, got {len(selector_codes)}")

    contexts = pairmod_contexts if pairmod_contexts is not None else (2, 4, 8, 16, 25, 50, 100)
    profile = profile_selector_encodings(
        selector_codes,
        fec6_selector_payload_bytes=len(selector_payload),
        target_saving_bytes=target_saving_bytes,
        pairmod_contexts=contexts,
    )
    best_payload = min(
        build_fec7_candidates(selector_codes, pairmod_contexts=contexts),
        key=lambda candidate: (candidate.payload_bytes, candidate.name),
    ).payload
    archive_delta_bytes = int(profile["best_charged_candidate"]["payload_bytes"]) - len(
        selector_payload
    )
    profile.update(
        {
            "source_archive": {
                "path": repo_relative(archive, REPO_ROOT),
                "bytes": archive.stat().st_size,
                "sha256": sha256_bytes(archive.read_bytes()),
                "member_name": member_name,
                "member_bytes": len(wrapper_payload),
                "member_sha256": sha256_bytes(wrapper_payload),
            },
            "parser_section_manifest_gate": manifest["parser_section_gate"],
            "fec6_selector": {
                "payload_bytes": len(selector_payload),
                "payload_sha256": sha256_bytes(selector_payload),
                "wire_format": "FEC6_fixed_huffman_k16_archive_charged_compact_palette",
                "charged_archive_bytes_equal_selector_payload_delta": True,
                "reason": (
                    "archive member is stored; replacing the selector payload changes "
                    "the wrapper and ZIP size by the same byte delta"
                ),
            },
            "best_fec7_payload_sha256": sha256_bytes(best_payload),
            "best_charged_candidate_archive_bytes_estimate": archive.stat().st_size
            + archive_delta_bytes,
            "charged_archive_delta_bytes_estimate": archive_delta_bytes,
        }
    )
    return profile


def render_markdown(profile: dict[str, Any]) -> str:
    best = profile["best_charged_candidate"]
    blocker = profile["explicit_blocker"]
    lines = [
        "# PR101 FEC7 Selector Entropy Profile",
        "",
        "- score_claim: `false`",
        "- dispatch_attempted: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        f"- source_archive: `{profile['source_archive']['path']}`",
        f"- FEC6 selector bytes: `{profile['fec6_selector']['payload_bytes']}`",
        f"- target saving bytes: `{profile['target_saving_bytes']}`",
        f"- global entropy floor bytes: `{profile['global_entropy_floor_bytes']}`",
        f"- best charged FEC7 candidate: `{best['name']}`",
        f"- best charged FEC7 payload bytes: `{best['payload_bytes']}`",
        f"- best saving vs FEC6 selector: `{best['saving_vs_fec6_selector_bytes']}`",
        f"- can meet target: `{str(profile['can_meet_target_with_charged_fec7_prototype']).lower()}`",
        "",
        "## Charged Candidates",
        "",
        "| candidate | bytes | saving vs FEC6 | model bytes | range bytes | meets target |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in profile["charged_candidates"]:
        lines.append(
            "| {name} | {payload_bytes} | {saving} | {model} | {stream} | {meets} |".format(
                name=row["name"],
                payload_bytes=row["payload_bytes"],
                saving=row["saving_vs_fec6_selector_bytes"],
                model=row["charged_model_bytes"],
                stream=row["range_stream_bytes"],
                meets=str(row["meets_target_saving"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Blocker",
            "",
            f"- blocked: `{str(blocker['blocked']).lower()}`",
            f"- reason: {blocker['reason']}",
            f"- reactivation_criteria: {blocker['reactivation_criteria']}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--target-saving-bytes", type=int, default=DEFAULT_TARGET_SAVING_BYTES)
    parser.add_argument(
        "--pairmod-contexts",
        default="2,4,8,16,25,50,100",
        help="Comma-separated deterministic pair-index modulo contexts to prototype.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contexts = tuple(
        int(part.strip())
        for part in str(args.pairmod_contexts).split(",")
        if part.strip()
    )
    profile = profile_archive(
        Path(args.archive),
        target_saving_bytes=int(args.target_saving_bytes),
        pairmod_contexts=contexts,
    )
    if args.json_out:
        write_json(Path(args.json_out), profile)
    if args.md_out:
        Path(args.md_out).write_text(render_markdown(profile) + "\n", encoding="utf-8")
    if not args.json_out and not args.md_out:
        json.dump(profile, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
