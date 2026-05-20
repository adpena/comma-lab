#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# AUTOCAST_FP16_WAIVED:research-only-local-fixture-export-no-paid-dispatch-no-scorer-training
# TF32_WAIVED:research-only-local-fixture-export-no-cuda-matmul-until-authorized-scorer-training
# TORCH_COMPILE_WAIVED:research-only-local-fixture-export-no-dispatch-training-loop
# NO_GRAD_WAIVED:torch-backed-export-smoke-owns-inference-mode-in-v8-smoke-helper
"""V8 learned-compression Faiss local implementation smoke.

This is the Codex implementation scaffold for
``v1_faiss_v8_learned_compression_faiss_design_20260519``. It now produces a
deterministic byte-closed fixture archive from a local categorical posterior
and scale-hyperprior summary, while remaining fail-closed for paid dispatch,
score claims, archive promotion, and exact-eval readiness.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

# macOS Faiss CPU and torch can both load OpenMP runtimes during same-process
# focused tests; V4 probe evidence already established this guard as required.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from tac.optimization.faiss_ivf_pq_atw_channel import (  # noqa: E402
    DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS,
    estimate_pq_encoding_budget,
)
from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates.v8_learned_compression_faiss.archive import (  # noqa: E402
    build_raw_frame_archive,
    parse_v8_archive,
)
from tac.substrates.v8_learned_compression_faiss.architecture import (  # noqa: E402
    V8CategoricalPosteriorConfig,
    build_scale_hyperprior_from_codewords,
    deterministic_categorical_codewords,
    deterministic_rgb_codebook,
)
from tac.substrates.v8_learned_compression_faiss.score_aware_loss import (  # noqa: E402
    build_score_aware_roundtrip_contract,
)
from tac.substrates.v8_learned_compression_faiss.smoke import (  # noqa: E402
    V8LocalSmokeConfig,
    run_local_cpu_export_smoke,
    write_v8_export_files,
)

LANE_ID = "lane_v8_learned_compression_faiss_scaffold_codex_20260520"
DESIGN_MEMO_PATH = ".omx/research/v1_faiss_v8_learned_compression_faiss_design_20260519.md"
INTAKE_MEMO_PATH = (
    ".omx/research/codex_findings_latest_design_memo_implementation_intake_"
    "20260520T031038Z_codex.md"
)
SUBMISSION_DIR = "submissions/v8_learned_compression_faiss"
OPERATOR_RECIPE_PATH = (
    ".omx/operator_authorize_recipes/"
    "substrate_v8_learned_compression_faiss_modal_a100_smoke.yaml"
)

TIER_1_EXTRA_MOUNT_PATHS: tuple[str, ...] = (
    "submissions/a1/archive.zip",
    "upstream/videos/0.mkv",
)

TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--output-dir": {
        "env": "V8_FAISS_OUTPUT_DIR",
        "rationale": "custody directory for V8 local smoke manifest and future archive export",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "V8_FAISS_UPSTREAM_DIR",
        "rationale": "future scorer-roundtrip training source; smoke does not load scorers",
        "default": "upstream",
        "required_input_file": True,
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--base-archive": {
        "env": "V8_FAISS_BASE_ARCHIVE",
        "rationale": "A1 latent/source archive for future side-info conditioning",
        "default": "submissions/a1/archive.zip",
        "required_input_file": True,
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "V8_FAISS_DEVICE",
        "rationale": "future full training device; smoke is CPU-only and non-promotable",
        "default": "cuda",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--categorical-groups": {
        "env": "V8_FAISS_CATEGORICAL_GROUPS",
        "rationale": "number of small categorical posterior groups",
        "default": "16",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--codebook-size": {
        "env": "V8_FAISS_CODEBOOK_SIZE",
        "rationale": "categorical codebook size per group",
        "default": "128",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--base-archive", type=Path, default=REPO_ROOT / "submissions/a1/archive.zip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-pairs", type=int, default=16)
    parser.add_argument("--categorical-groups", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=28)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--fixture-frames", type=int, default=4)
    parser.add_argument("--fixture-height", type=int, default=8)
    parser.add_argument("--fixture-width", type=int, default=8)
    parser.add_argument("--fixture-payload", type=Path, default=None)
    parser.add_argument(
        "--require-fixture-payload",
        action="store_true",
        help="fail closed unless --fixture-payload supplies exact raw RGB bytes",
    )
    parser.add_argument("--archive-name", default="v8_fixture.v8f")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--write-json", action="store_true", default=True)
    return parser


def _resolve_trainer_mode(args: argparse.Namespace) -> str:
    explicit = os.environ.get("V8_FAISS_TRAINER_MODE", "").strip().lower()
    if explicit in {"smoke", "full"}:
        return explicit
    legacy = os.environ.get("SMOKE_ONLY", "").strip()
    if legacy == "0":
        return "full"
    if legacy == "1":
        return "smoke"
    if args.smoke:
        return "smoke"
    return "smoke"


def default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "experiments" / "results" / f"lane_v8_learned_compression_faiss_smoke_{stamp}"


def _archive_budget_estimate(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_pairs <= 0:
        raise ValueError("--num-pairs must be positive")
    if args.categorical_groups <= 0:
        raise ValueError("--categorical-groups must be positive")
    if args.codebook_size <= 1:
        raise ValueError("--codebook-size must be greater than 1")
    if args.hidden_dim <= 0 or args.latent_dim <= 0:
        raise ValueError("--hidden-dim and --latent-dim must be positive")
    if args.fixture_frames <= 0 or args.fixture_height <= 0 or args.fixture_width <= 0:
        raise ValueError("--fixture-frames/height/width must be positive")

    bits_per_code = (args.codebook_size - 1).bit_length()
    codeword_stream_bits = args.num_pairs * args.categorical_groups * bits_per_code
    codeword_stream_bytes = (codeword_stream_bits + 7) // 8
    encoder_weight_bytes_int8 = (
        args.latent_dim * args.hidden_dim
        + args.hidden_dim * args.categorical_groups * args.codebook_size
    )
    scale_hyperprior_bytes = args.categorical_groups * 2
    header_bytes = 64
    total = (
        header_bytes
        + encoder_weight_bytes_int8
        + codeword_stream_bytes
        + scale_hyperprior_bytes
    )
    return {
        "header_bytes": header_bytes,
        "encoder_weight_bytes_int8": encoder_weight_bytes_int8,
        "categorical_codeword_stream_bits": codeword_stream_bits,
        "categorical_codeword_stream_bytes": codeword_stream_bytes,
        "scale_hyperprior_bytes": scale_hyperprior_bytes,
        "total_archive_contribution_bytes_estimate": total,
        "contest_rate_cost_estimate": 25.0 * total / 37_545_489.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _fixture_payload_from_args(
    args: argparse.Namespace,
    *,
    codewords: bytes,
    codebook: tuple[tuple[int, int, int], ...],
) -> tuple[bytes, dict[str, Any]]:
    pixels_per_frame = args.fixture_height * args.fixture_width
    expected_bytes = args.fixture_frames * pixels_per_frame * 3
    if args.require_fixture_payload and args.fixture_payload is None:
        raise FileNotFoundError(
            "--require-fixture-payload set but --fixture-payload was not supplied"
        )
    if args.fixture_payload is not None:
        if not args.fixture_payload.is_file():
            raise FileNotFoundError(f"V8 fixture payload not found: {args.fixture_payload}")
        payload = args.fixture_payload.read_bytes()
        if len(payload) != expected_bytes:
            raise ValueError(
                "V8 fixture payload byte length mismatch: "
                f"{len(payload)} != {expected_bytes}"
            )
        return payload, {
            "kind": "supplied_raw_rgb_fixture",
            "path": str(args.fixture_payload),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "score_claim": False,
            "promotion_eligible": False,
        }

    payload = bytearray()
    for code in codewords:
        payload.extend(bytes(codebook[int(code)]) * pixels_per_frame)
    payload_bytes = bytes(payload)
    return payload_bytes, {
        "kind": "synthetic_categorical_rgb_fixture",
        "bytes": len(payload_bytes),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _build_local_fixture_archive(args: argparse.Namespace) -> dict[str, Any]:
    config = V8CategoricalPosteriorConfig(
        categorical_groups=args.categorical_groups,
        codebook_size=args.codebook_size,
        seed=args.seed,
    )
    codewords = deterministic_categorical_codewords(args.fixture_frames, config)
    scale_hyperprior = build_scale_hyperprior_from_codewords(
        codewords,
        categorical_groups=args.categorical_groups,
    )
    codebook = deterministic_rgb_codebook(args.codebook_size)
    payload_bytes, payload_source = _fixture_payload_from_args(
        args,
        codewords=codewords,
        codebook=codebook,
    )
    archive_bytes = build_raw_frame_archive(
        payload_bytes,
        frame_count=args.fixture_frames,
        height=args.fixture_height,
        width=args.fixture_width,
        channels=3,
    )
    header, decoded_payload = parse_v8_archive(archive_bytes)
    if decoded_payload != payload_bytes:
        raise RuntimeError("V8 local fixture archive failed byte roundtrip")
    return {
        "archive_bytes": archive_bytes,
        "raw_payload": payload_bytes,
        "raw_payload_source": payload_source,
        "header": header.as_dict(),
        "categorical_codewords_sha256": hashlib.sha256(codewords).hexdigest(),
        "categorical_codeword_count": len(codewords),
        "categorical_groups": args.categorical_groups,
        "codebook_size": args.codebook_size,
        "scale_hyperprior": [
            {"group": idx, "mean_code": mean, "scale_code": scale}
            for idx, (mean, scale) in enumerate(scale_hyperprior)
        ],
        "implemented_local_blockers": [
            "deterministic_categorical_posterior_codewords",
            "scale_hyperprior_summary",
            "byte_closed_export_and_inflate_contract",
            "score_aware_eval_roundtrip_contract_declared",
        ],
        "remaining_promotion_blockers": [
            "real_contest_video_scorer_training_not_run",
            "exact_cuda_auth_eval_missing",
            "catalog_324_tier_c_validation_missing",
            "modal_dispatch_recipe_remains_research_only_and_disabled",
        ],
    }


def build_smoke_manifest(args: argparse.Namespace, *, observed_at_utc: str | None = None) -> dict[str, Any]:
    observed = observed_at_utc or utc_now()
    pq_budget = estimate_pq_encoding_budget(
        variant_id="v8_reference_v4_midpoint_for_budget_context",
        n_regions=args.categorical_groups,
        nlist=32,
        m_subq=2,
        nbits=7,
        top_k_regions=min(3, args.categorical_groups),
        total_pairs=args.num_pairs,
    )
    architecture_tuple = {
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "categorical_groups": args.categorical_groups,
        "codebook_size": args.codebook_size,
        "beta": args.beta,
        "temperature_start": args.temperature_start,
        "temperature_end": args.temperature_end,
        "seed": args.seed,
    }
    arch_hash = sha256_text(json.dumps(architecture_tuple, sort_keys=True))
    return {
        "schema": "v8_learned_compression_faiss_smoke_v1",
        "observed_at_utc": observed,
        "lane_id": LANE_ID,
        "design_memo": DESIGN_MEMO_PATH,
        "intake_memo": INTAKE_MEMO_PATH,
        "research_only": True,
        "dispatch_enabled": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_label": "[diagnostic-CPU scaffold smoke; no scorer load]",
        "base_archive": str(args.base_archive),
        "upstream_dir": str(args.upstream_dir),
        "submission_dir": SUBMISSION_DIR,
        "operator_recipe": OPERATOR_RECIPE_PATH,
        "architecture_tuple": architecture_tuple,
        "encoder_architecture_hash": arch_hash,
        "decoder_architecture_hash": arch_hash,
        "archive_budget_estimate": _archive_budget_estimate(args),
        "pq_budget_context": pq_budget.as_dict(),
        "meaningful_mi_threshold_bits": DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS,
        "canonical_helpers": {
            "pq_codec": "tac.optimization.faiss_ivf_pq_atw_channel",
            "pq_mi": "tac.optimization.faiss_ivf_pq_atw_channel.compute_pq_mi_verdict",
            "scorer": "tac.scorer.load_default_scorers",
            "score_aware": "tac.substrates.score_aware_common.score_pair_components_dispatch",
            "auth_eval_gate": "tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call",
            "inflate_device": "tac.substrates._shared.inflate_runtime.select_inflate_device",
            "hardware_substrate": "tac.substrates._shared.trainer_skeleton.detect_hardware_substrate",
            "modal_call_id": "tac.deploy.modal.call_id_ledger.register_dispatched_call_id_fail_closed",
            "cost_band": "tac.cost_band_calibration.append_anchor",
            "probe_outcome": "tac.probe_outcomes_ledger.register_probe_outcome",
            "wyner_ziv_deliverability": (
                "tac.wyner_ziv_deliverability.proof_builder."
                "build_deliverability_proof_from_wyner_ziv_classification"
            ),
            "provenance_archive_member": "tac.provenance.build_provenance_for_archive_member",
        },
        "corrected_design_premises": [
            "score-aware helper path is tac.substrates.score_aware_common, not tac.substrates._shared.score_aware_common",
            "compute_pq_mi_verdict is now a tac.optimization helper, not a probe-only API",
            "canonical Provenance archive-member builder is build_provenance_for_archive_member",
        ],
        "local_implementation_status": "byte_closed_fixture_export_available",
        "score_aware_roundtrip_contract": build_score_aware_roundtrip_contract(),
        "remaining_promotion_blockers": [
            "real_contest_video_scorer_training_not_run",
            "exact_cuda_auth_eval_missing",
            "catalog_324_tier_c_validation_missing",
            "modal_dispatch_recipe_remains_research_only_and_disabled",
        ],
    }


def _write_local_outputs(args: argparse.Namespace, *, mode: str) -> tuple[Path, dict[str, Any]]:
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = _build_local_fixture_archive(args)
    learned_payload, learned_header = run_local_cpu_export_smoke(
        V8LocalSmokeConfig(
            num_pairs=max(1, args.num_pairs),
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            categorical_groups=args.categorical_groups,
            codebook_size=args.codebook_size,
            beta=args.beta,
            temperature_start=args.temperature_start,
            temperature_end=args.temperature_end,
            seed=args.seed,
            eval_roundtrip_resize=False,
        )
    )
    learned_export = write_v8_export_files(
        output_dir,
        payload=learned_payload,
        header=learned_header,
    )
    archive_path = output_dir / args.archive_name
    raw_path = archive_path.with_suffix(".raw")
    archive_path.write_bytes(fixture["archive_bytes"])
    raw_path.write_bytes(fixture["raw_payload"])
    manifest = build_smoke_manifest(args)
    manifest.update(
        {
            "trainer_mode": mode,
            "hardware_substrate": _canon_detect_hardware_substrate(
                axis="cpu",
                substrate_tag="v8_learned_compression_faiss_local_fixture",
            ),
            "local_byte_closed_export": {
                "archive_path": str(archive_path),
                "archive_bytes": len(fixture["archive_bytes"]),
                "archive_sha256": hashlib.sha256(fixture["archive_bytes"]).hexdigest(),
                "raw_fixture_path": str(raw_path),
                "raw_fixture_bytes": len(fixture["raw_payload"]),
                "raw_fixture_sha256": hashlib.sha256(fixture["raw_payload"]).hexdigest(),
                "raw_fixture_source": fixture["raw_payload_source"],
                "header": fixture["header"],
                "categorical_codewords_sha256": fixture["categorical_codewords_sha256"],
                "categorical_codeword_count": fixture["categorical_codeword_count"],
                "categorical_groups": fixture["categorical_groups"],
                "codebook_size": fixture["codebook_size"],
                "scale_hyperprior": fixture["scale_hyperprior"],
            },
            "learned_compression_export": learned_export,
            "learned_compression_export_header": learned_header,
            "implemented_local_blockers": fixture["implemented_local_blockers"],
            "remaining_promotion_blockers": fixture["remaining_promotion_blockers"],
        }
    )
    output_json = output_dir / "v8_smoke_results.json"
    output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_json, manifest


def _smoke_main(args: argparse.Namespace) -> int:
    output_json, manifest = _write_local_outputs(args, mode="smoke")
    print("[v8-faiss] SMOKE MODE")
    print(f"[v8-faiss] lane={LANE_ID}")
    print("[v8-faiss] local_byte_closed_export=true")
    print("[v8-faiss] research_only=true dispatch_enabled=false score_claim=false")
    print(f"[v8-faiss] manifest={output_json}")
    print(f"[v8-faiss] archive={manifest['local_byte_closed_export']['archive_path']}")
    print(
        "[v8-faiss] estimated archive contribution="
        f"{manifest['archive_budget_estimate']['total_archive_contribution_bytes_estimate']} bytes"
    )
    return 0


def _full_main(args: argparse.Namespace) -> int:
    output_json, manifest = _write_local_outputs(args, mode="full_local_fixture")
    print("[v8-faiss] FULL MODE LOCAL FIXTURE")
    print("[v8-faiss] no scorer load, no provider call, no score claim")
    print(f"[v8-faiss] manifest={output_json}")
    print(f"[v8-faiss] archive={manifest['local_byte_closed_export']['archive_path']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    mode = _resolve_trainer_mode(args)
    print(f"[v8-faiss] resolved trainer mode: {mode}")
    if mode == "smoke":
        return _smoke_main(args)
    if mode == "full":
        return _full_main(args)
    raise RuntimeError(f"unknown trainer mode: {mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
