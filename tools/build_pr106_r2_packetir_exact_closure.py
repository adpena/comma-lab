#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an exact-eval closure record for the PR106/R2 PacketIR candidate."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packetir_exact_closure import (  # noqa: E402
    build_packetir_exact_closure,
    render_packetir_exact_closure_markdown,
)
from tac.repo_io import read_json, write_json  # noqa: E402

DEFAULT_RESULT_DIR = (
    REPO_ROOT / "experiments/results/pr106_r2_packetir_exact_closure_20260513_codex"
)
DEFAULT_CANDIDATE_RESULT = (
    REPO_ROOT / "experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/result.json"
)
DEFAULT_CANDIDATE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/"
    "pr106_r2_pr101_grammar_hnerv_brotli_repack_candidate.zip"
)
DEFAULT_CUDA_EVAL = (
    REPO_ROOT
    / "experiments/results/modal_auth_eval/"
    "pr106_r2_pr101_grammar_lowlevel_repack_cuda_20260513_codex/contest_auth_eval.json"
)
DEFAULT_CPU_EVAL = (
    REPO_ROOT
    / "experiments/results/modal_auth_eval_cpu/"
    "pr106_r2_pr101_grammar_lowlevel_repack_cpu_20260513_codex/contest_auth_eval.json"
)
DEFAULT_SOURCE_CUDA_EVAL = (
    REPO_ROOT
    / "experiments/results/modal_auth_eval/"
    "pr106_latent_sidecar_r2_pr101_grammar_20260511T180000Z/contest_auth_eval.json"
)
DEFAULT_CURRENT_BEST_CUDA_EVAL = (
    REPO_ROOT
    / "experiments/results/modal_auth_eval/"
    "hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513/contest_auth_eval.json"
)
DEFAULT_RECODE_PROFILE = (
    REPO_ROOT
    / "experiments/results/pr106_r2_packetir_recode_refresh_20260513T215443Z/recode_profile.json"
)
DEFAULT_RUNTIME_CONSUMPTION_PROOF = (
    REPO_ROOT
    / "experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/"
    "runtime_consumption.json"
)
DEFAULT_FULL_FRAME_PARITY_PROOF = (
    REPO_ROOT
    / "experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/"
    "same_runtime_full_frame_parity_local_cpu.json"
)


def build_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Load inputs and build the closure manifest."""

    candidate_result = _candidate_result_with_packetir_identity(
        candidate_result=read_json(args.candidate_result),
        packetir_identity=_read_optional(args.packetir_identity),
    )
    input_paths = {
        "candidate_result": _rel(args.candidate_result),
        "candidate_archive": _rel(args.candidate_archive),
        "cuda_eval": _rel(args.cuda_eval),
        "cpu_eval": _rel(args.cpu_eval) if args.cpu_eval is not None else None,
        "source_cuda_eval": _rel(args.source_cuda_eval)
        if args.source_cuda_eval is not None
        else None,
        "current_best_cuda_eval": _rel(args.current_best_cuda_eval)
        if args.current_best_cuda_eval is not None
        else None,
        "runtime_consumption_proof": _rel(args.runtime_consumption_proof),
        "full_frame_parity_proof": _rel(args.full_frame_parity_proof),
        "recode_profile": _rel(args.recode_profile) if args.recode_profile is not None else None,
        "packetir_identity": _rel(args.packetir_identity)
        if args.packetir_identity is not None
        else None,
    }
    return build_packetir_exact_closure(
        lane_id=args.lane_id,
        candidate_result=candidate_result,
        candidate_archive_path=args.candidate_archive,
        cuda_eval=read_json(args.cuda_eval),
        cpu_eval=_read_optional(args.cpu_eval),
        source_cuda_eval=read_json(args.source_cuda_eval),
        current_best_cuda_eval=read_json(args.current_best_cuda_eval),
        runtime_consumption_proof=read_json(args.runtime_consumption_proof),
        full_frame_parity_proof=read_json(args.full_frame_parity_proof),
        recode_profile=_read_optional(args.recode_profile),
        input_paths={key: value for key, value in input_paths.items() if value is not None},
        repo_root=REPO_ROOT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-id", default="pr106_r2_packetir_pr101_grammar_lowlevel_closure")
    parser.add_argument("--candidate-result", type=Path, default=DEFAULT_CANDIDATE_RESULT)
    parser.add_argument("--candidate-archive", type=Path, default=DEFAULT_CANDIDATE_ARCHIVE)
    parser.add_argument("--cuda-eval", type=Path, default=DEFAULT_CUDA_EVAL)
    parser.add_argument("--cpu-eval", type=Path, default=DEFAULT_CPU_EVAL)
    parser.add_argument("--source-cuda-eval", type=Path, default=DEFAULT_SOURCE_CUDA_EVAL)
    parser.add_argument(
        "--current-best-cuda-eval",
        type=Path,
        default=DEFAULT_CURRENT_BEST_CUDA_EVAL,
    )
    parser.add_argument("--recode-profile", type=Path, default=DEFAULT_RECODE_PROFILE)
    parser.add_argument(
        "--runtime-consumption-proof",
        type=Path,
        default=DEFAULT_RUNTIME_CONSUMPTION_PROOF,
    )
    parser.add_argument(
        "--full-frame-parity-proof",
        type=Path,
        default=DEFAULT_FULL_FRAME_PARITY_PROOF,
    )
    parser.add_argument(
        "--packetir-identity",
        type=Path,
        default=None,
        help=(
            "Optional PacketIR identity proof to merge when the candidate result "
            "manifest does not carry packet_ir_consumed_byte_proof directly."
        ),
    )
    parser.add_argument(
        "--no-cpu-eval",
        action="store_true",
        help="Omit the optional contest-CPU axis closure artifact.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RESULT_DIR / "closure.json")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_RESULT_DIR / "closure.md")
    args = parser.parse_args(argv)
    if args.no_cpu_eval:
        args.cpu_eval = None

    closure = build_from_args(args)
    closure["recorded_at_utc"] = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    write_json(args.output_json, closure)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_packetir_exact_closure_markdown(closure), encoding="utf-8")
    print(f"closure_json={_rel(args.output_json)}")
    print(f"closure_md={_rel(args.output_md)}")
    print(f"classification={closure['classification']}")
    print(f"ready_for_exact_eval_dispatch={closure['ready_for_exact_eval_dispatch']}")
    return 0 if not closure.get("blockers") else 2


def _read_optional(path: Path | None) -> Any | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    return read_json(path)


def _candidate_result_with_packetir_identity(
    *,
    candidate_result: dict[str, Any],
    packetir_identity: Any | None,
) -> dict[str, Any]:
    """Normalize supported static PacketIR manifests into closure input shape."""

    result = dict(candidate_result)
    if "candidate_diff_audit" not in result:
        byte_delta = (
            result.get("candidate_archive_byte_delta_vs_source")
            if isinstance(result.get("candidate_archive_byte_delta_vs_source"), int)
            else result.get("candidate_archive_byte_delta")
        )
        blockers = result.get("archive_build_blockers", [])
        if isinstance(byte_delta, int):
            result["candidate_diff_audit"] = {
                "blockers": blockers if isinstance(blockers, list) else [],
                "total_byte_delta": byte_delta,
            }
    if (
        "source_archive_bytes" not in result
        and isinstance(result.get("candidate_archive_bytes"), int)
        and isinstance(result.get("candidate_archive_byte_delta_vs_source"), int)
    ):
        result["source_archive_bytes"] = (
            result["candidate_archive_bytes"]
            - result["candidate_archive_byte_delta_vs_source"]
        )
    if "packet_ir_consumed_byte_proof" not in result and isinstance(
        result.get("candidate_packet_ir_consumed_byte_proof"), dict
    ):
        result["packet_ir_consumed_byte_proof"] = result[
            "candidate_packet_ir_consumed_byte_proof"
        ]
    if "packet_ir_consumed_byte_proof" not in result and isinstance(packetir_identity, dict):
        packet = packetir_identity.get("packet")
        if isinstance(packet, dict):
            proof = packet.get("packet_ir_consumed_byte_proof")
            if isinstance(proof, dict):
                result["packet_ir_consumed_byte_proof"] = proof
    return result


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
