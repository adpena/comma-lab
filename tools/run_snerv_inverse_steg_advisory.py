#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""$0 macOS-CPU advisory CLI for the SNeRV inverse-steganalysis carrier.

Runs the complete byte-closed SNeRV stack on REAL ``upstream/videos/0.mkv`` frames
and emits a NON-PROMOTABLE ``[macOS-CPU advisory]`` JSON (Catalog #341/#192/#127/
#323). Reports the achieved rate term (bytes), advisory d_seg/d_pose, the
Z8-falsification verdict, the G3 DWT-adjoint exactness number, and the L-inf-vs-L2
score delta.

NO paid dispatch, NO cloud GPU, NO PR, NO MPS-as-authority. The scorer is the
bit-exact CPU mirror (offline oracle + advisory re-measure only); it never crosses
the receiver boundary.

Usage:
    .venv/bin/python tools/run_snerv_inverse_steg_advisory.py \
        --n-pairs 4 --levels 3 --bits-per-coeff 2.5 \
        --out .omx/research/snerv_inverse_steg_advisory_<utc>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.snerv_inverse_steg_carrier.advisory import (  # noqa: E402  (sys.path bootstrap above)
    CONTEST_BYTE_PRICE,
    run_snerv_advisory,
)
from tac.substrates.snerv_inverse_steg_carrier.archive_candidate import (  # noqa: E402
    export_snerv_archive_bound_candidate_package,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (  # noqa: E402
    SNERV_OFFICIAL_DEFAULT_DEC_STRDS,
    SNERV_OFFICIAL_DEFAULT_ENC_STRDS,
    SNERV_SPECTRA_PRESERVING_ADAPTER,
)
from tac.substrates.snerv_inverse_steg_carrier.trained_ladder_bridge import (  # noqa: E402
    build_snerv_trained_ladder_row_from_advisory,
)


def _default_out() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f".omx/research/snerv_inverse_steg_advisory_{stamp}.json"


def _sha256_file(path: Path) -> str:
    h = __import__("hashlib").sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float_attr(value: object, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(value, name))
    except (TypeError, ValueError, AttributeError):
        return float(default)


def _int_attr(value: object, name: str, default: int = 0) -> int:
    try:
        return int(getattr(value, name))
    except (TypeError, ValueError, AttributeError):
        return int(default)


def _charge_packaged_archive_rate(
    *,
    payload: dict[str, object],
    advisory_result: object,
    archive_zip_path: Path,
) -> None:
    """Make packaged archive.zip bytes the charged rate numerator.

    The raw SNAR1 packet remains useful diagnostic telemetry, but contest rate
    is paid on the packaged archive. This helper is intentionally CLI-local
    because ``run_snerv_advisory`` can also be used without packaging.
    """

    archive_bytes = archive_zip_path.stat().st_size
    archive_sha256 = _sha256_file(archive_zip_path)
    packet_bytes = _int_attr(advisory_result, "archive_bytes_total")
    packet_rate = _float_attr(advisory_result, "rate_term")
    packet_score_linf = _float_attr(advisory_result, "score_linf")
    nonrate_linf = packet_score_linf - packet_rate
    charged_rate = float(CONTEST_BYTE_PRICE * archive_bytes)
    charged_score_linf = float(nonrate_linf + charged_rate)
    frontier_bytes = _int_attr(advisory_result, "pr101_frontier_bytes")

    payload["receiver_snar_packet_rate_accounting"] = {
        "schema": "snerv_receiver_snar_packet_rate_accounting.v1",
        "archive_path_kind": "receiver_snar_packet",
        "archive_bytes_total": packet_bytes,
        "rate_term": packet_rate,
        "score_linf": packet_score_linf,
        "score_linf_without_rate": nonrate_linf,
        "receiver_archive_sha256": getattr(
            advisory_result, "receiver_archive_sha256", None
        ),
        "chargeable_for_contest_submission": False,
    }
    payload["charged_archive_rate_accounting"] = {
        "schema": "snerv_packaged_archive_rate_accounting.v1",
        "archive_path_kind": "contest_archive_zip",
        "archive_path": archive_zip_path.as_posix(),
        "archive_bytes": archive_bytes,
        "archive_bytes_total": archive_bytes,
        "archive_sha256": archive_sha256,
        "rate_term": charged_rate,
        "score_linf": charged_score_linf,
        "score_linf_without_rate": nonrate_linf,
        "beats_frontier_rate": bool(
            frontier_bytes > 0 and archive_bytes < frontier_bytes
        ),
        "packet_bytes_preserved_as_diagnostic": packet_bytes,
        "packet_to_package_byte_delta": archive_bytes - packet_bytes,
        "chargeable_for_contest_submission": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    payload["archive_bytes_total_before_package"] = packet_bytes
    payload["rate_term_before_package"] = packet_rate
    payload["score_linf_before_package"] = packet_score_linf
    payload["charged_archive_path_kind"] = "contest_archive_zip"
    payload["charged_archive_path"] = archive_zip_path.as_posix()
    payload["charged_archive_sha256"] = archive_sha256
    payload["archive_bytes_total"] = archive_bytes
    payload["rate_term"] = charged_rate
    payload["score_linf"] = charged_score_linf
    payload["beats_frontier_rate"] = bool(
        frontier_bytes > 0 and archive_bytes < frontier_bytes
    )
    payload["score_l2_archive_path_kind"] = "receiver_snar_packet"
    payload["score_l2_package_rate_not_recomputed_reason"] = (
        "the CLI packages only the selected L-inf SNAR1 packet; L2 remains a "
        "separate packet-scoped diagnostic unless its packet is packaged too"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--bits-per-coeff", type=float, default=2.5)
    ap.add_argument("--wavelet", type=str, default="db2")
    ap.add_argument("--pair-stride", type=int, default=1)
    ap.add_argument("--start-pair", type=int, default=0)
    ap.add_argument("--pr101-frontier-bytes", type=int, default=178_493)
    ap.add_argument("--upstream-dir", type=str, default="upstream")
    ap.add_argument("--video-path", type=str, default="upstream/videos/0.mkv")
    ap.add_argument(
        "--step-map-coder-bins",
        type=int,
        default=128,
        help="Shared log2 quantizer bins for compact receiver-visible L-inf step maps.",
    )
    ap.add_argument(
        "--step-map-coder-mode",
        choices=("uniform", "adaptive", "waterfill"),
        default="uniform",
        help="Step-map packet mode: shared quantizer, saliency groups, or reverse-waterfill ladder.",
    )
    ap.add_argument(
        "--step-map-adaptive-bin-choices",
        default="128,16,4",
        help="Comma-separated bin portfolio for adaptive step-map mode.",
    )
    ap.add_argument(
        "--step-map-constant-importance-quantile",
        type=float,
        default=None,
        help="Optional low-importance quantile to encode as constant-fill maps.",
    )
    ap.add_argument(
        "--step-map-waterfill-bits-per-coeff",
        type=float,
        default=4.0,
        help="Target average bits/coefficient for waterfill step-map precision.",
    )
    ap.add_argument(
        "--hf-decoder-fit-mode",
        choices=("least_squares", "score_weighted"),
        default="least_squares",
        help="HF decoder fit mode.",
    )
    ap.add_argument(
        "--hf-decoder-saliency-gain",
        type=float,
        default=1.0,
        help="Score-saliency gain for score_weighted HF decoder fitting.",
    )
    ap.add_argument(
        "--hf-decoder-saliency-component",
        choices=("combined", "seg", "pose"),
        default="combined",
        help="Detector component used to weight score_weighted HF decoder fitting.",
    )
    ap.add_argument(
        "--snerv-fc-dim",
        type=int,
        default=None,
        help="Receiver-visible LF-context feature count for the SNeRV HF decoder.",
    )
    ap.add_argument(
        "--snerv-official-modelsize-mparams",
        type=float,
        default=None,
        help=(
            "Official SNeRV-style --modelsize value in millions of parameters. "
            "When provided, the advisory solves fc_dim from the source-bound "
            "quadratic and records official_modelsize_solution metadata."
        ),
    )
    ap.add_argument(
        "--snerv-official-enc-strds",
        default=",".join(str(v) for v in SNERV_OFFICIAL_DEFAULT_ENC_STRDS),
        help="Comma-separated official SNeRV encoder strides for --modelsize solving.",
    )
    ap.add_argument(
        "--snerv-official-dec-strds",
        default=",".join(str(v) for v in SNERV_OFFICIAL_DEFAULT_DEC_STRDS),
        help="Comma-separated official SNeRV decoder strides for --modelsize solving.",
    )
    ap.add_argument(
        "--snerv-emb-size",
        type=int,
        default=0,
        help=(
            "Receiver-generated coordinate embedding feature count for the "
            "SNeRV HF decoder."
        ),
    )
    ap.add_argument(
        "--snerv-patch-radius",
        type=int,
        default=1,
        help="Receiver-visible LF context radius used to build fc_dim features.",
    )
    ap.add_argument(
        "--snerv-spectra-preserving-adapter",
        action="store_true",
        help=(
            "Use the receiver-visible spectra-preserving MFU/HFR feature "
            "adapter instead of the historical 3x3 LF-patch adapter."
        ),
    )
    ap.add_argument(
        "--snerv-mfu-scales",
        default="1,2,4",
        help="Comma-separated deterministic MFU scales for the SNeRV adapter.",
    )
    ap.add_argument(
        "--snerv-hfr-gain",
        type=float,
        default=0.0,
        help="Deterministic HFR residual gain for the SNeRV adapter.",
    )
    ap.add_argument(
        "--snerv-temporal-context",
        type=int,
        default=0,
        help="Receiver-visible temporal context radius for SNeRV_T-style features.",
    )
    ap.add_argument(
        "--snerv-temporal-mode",
        choices=("delta", "official_haar_dwt1d_lowpass"),
        default="delta",
        help="Receiver-visible temporal basis for SNeRV_T-style features.",
    )
    ap.add_argument(
        "--decoder-payload-codec",
        choices=(
            "float32_lzma",
            "int8_symmetric",
            "int4_symmetric",
            "int2_symmetric",
            "mixed_magnitude_symmetric",
        ),
        default="float32_lzma",
        help="Receiver-visible HF decoder payload grammar.",
    )
    ap.add_argument(
        "--decoder-payload-mixed-modes",
        default="",
        help=(
            "Optional comma-separated explicit v3 mixed decoder mode list "
            "(zero,int2,int4,int8,fp16), one mode per level/subband kernel."
        ),
    )
    ap.add_argument(
        "--lf-payload-codec",
        choices=("legacy", "portfolio_auto"),
        default="portfolio_auto",
        help=(
            "Receiver-visible LF payload grammar. portfolio_auto uses the "
            "lossless SQL2 int-stream portfolio; legacy preserves raw-i64+xz."
        ),
    )
    ap.add_argument(
        "--snerv-native-mlx-decoder-train-steps",
        default=0,
        type=int,
        help=(
            "Record a requested native-MLX HF decoder training step count. "
            "This CPU advisory CLI does not execute MLX decoder training."
        ),
    )
    ap.add_argument(
        "--snerv-native-mlx-decoder-train-lr",
        default=1.0e-5,
        type=float,
        help="Recorded learning rate for --snerv-native-mlx-decoder-train-steps.",
    )
    ap.add_argument(
        "--snerv-native-mlx-decoder-train-ridge",
        default=1.0e-6,
        type=float,
        help="Recorded ridge pressure for --snerv-native-mlx-decoder-train-steps.",
    )
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--packet-out",
        type=str,
        default=None,
        help="Optional path to write the real advisory SNAR1 packet bytes.",
    )
    ap.add_argument(
        "--package-dir",
        type=str,
        default=None,
        help=(
            "Optional directory to emit archive.zip + submission runtime + "
            "receiver proof from the real advisory SNAR1 packet."
        ),
    )
    ap.add_argument(
        "--retain-package-raw",
        action="store_true",
        help="Keep generated raw proof output instead of certify-and-delete.",
    )
    ap.add_argument("--package-timeout-seconds", type=int, default=1800)
    ap.add_argument(
        "--trained-ladder-row-out",
        type=str,
        default=None,
        help=(
            "Optional path to write the false-authority trained ladder row "
            "payload emitted from the real receiver archive path."
        ),
    )
    ap.add_argument(
        "--qat-bits",
        type=int,
        default=None,
        help="Optional QAT bit count, only when this advisory was produced by a QAT run.",
    )
    args = ap.parse_args(argv)

    snerv_fc_dim_explicit = args.snerv_fc_dim is not None
    res = run_snerv_advisory(
        n_pairs=args.n_pairs,
        levels=args.levels,
        wavelet=args.wavelet,
        target_bits_per_coeff=args.bits_per_coeff,
        pair_stride=args.pair_stride,
        start_pair=args.start_pair,
        pr101_frontier_bytes=args.pr101_frontier_bytes,
        video_path=args.video_path,
        upstream_dir=args.upstream_dir,
        step_map_coder_bins=args.step_map_coder_bins,
        step_map_coder_mode=args.step_map_coder_mode,
        step_map_adaptive_bin_choices=_parse_bins(args.step_map_adaptive_bin_choices),
        step_map_constant_importance_quantile=(
            args.step_map_constant_importance_quantile
        ),
        step_map_waterfill_bits_per_coeff=args.step_map_waterfill_bits_per_coeff,
        hf_decoder_fit_mode=args.hf_decoder_fit_mode,
        hf_decoder_saliency_gain=args.hf_decoder_saliency_gain,
        hf_decoder_saliency_component=args.hf_decoder_saliency_component,
        snerv_fc_dim=9 if args.snerv_fc_dim is None else args.snerv_fc_dim,
        snerv_fc_dim_explicit=snerv_fc_dim_explicit,
        snerv_emb_size=args.snerv_emb_size,
        snerv_patch_radius=args.snerv_patch_radius,
        snerv_model_size_adapter=(
            SNERV_SPECTRA_PRESERVING_ADAPTER
            if args.snerv_spectra_preserving_adapter
            else "snerv_fc_dim_emb_size_adapter_v1"
        ),
        snerv_mfu_scales=_parse_positive_int_csv(args.snerv_mfu_scales),
        snerv_hfr_gain=args.snerv_hfr_gain,
        snerv_temporal_context=args.snerv_temporal_context,
        snerv_temporal_mode=args.snerv_temporal_mode,
        snerv_official_modelsize_mparams=args.snerv_official_modelsize_mparams,
        snerv_official_enc_strds=_parse_positive_int_csv(
            args.snerv_official_enc_strds
        ),
        snerv_official_dec_strds=_parse_positive_int_csv(
            args.snerv_official_dec_strds
        ),
        decoder_payload_codec=args.decoder_payload_codec,
        decoder_payload_mixed_modes=_parse_optional_modes(
            args.decoder_payload_mixed_modes
        ),
        lf_payload_codec=args.lf_payload_codec,
    )
    payload = res.as_jsonable()
    payload.setdefault("schema", "snerv_inverse_steg_advisory.v1")
    native_mlx_decoder_training_controls = {
        "schema": "snerv_native_mlx_decoder_training_cli_control.v1",
        "requested_steps": int(args.snerv_native_mlx_decoder_train_steps),
        "learning_rate": float(args.snerv_native_mlx_decoder_train_lr),
        "ridge": float(args.snerv_native_mlx_decoder_train_ridge),
        "consumed_by_cli": False,
        "consumed_by_archive_metadata": False,
        "native_mlx_training_executed": False,
        "blockers": [
            "snerv_native_mlx_decoder_training_controls_unreachable_from_cpu_advisory_cli"
        ]
        if int(args.snerv_native_mlx_decoder_train_steps) > 0
        else [],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    payload["native_mlx_decoder_training_controls"] = native_mlx_decoder_training_controls
    if native_mlx_decoder_training_controls["blockers"]:
        payload["blockers"] = list(
            dict.fromkeys(
                [
                    *(payload.get("blockers") or ()),
                    *native_mlx_decoder_training_controls["blockers"],
                ]
            )
        )
    out_path = Path(args.out or _default_out())
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path: Path | None = None
    package_dir: Path | None = None
    package: dict[str, object] | None = None
    if args.packet_out:
        packet_path = Path(args.packet_out)
        if not packet_path.is_absolute():
            packet_path = REPO_ROOT / packet_path
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        packet_path.write_bytes(res.receiver_archive_packet)
        payload["receiver_archive_packet_path"] = str(packet_path)
    if args.package_dir:
        package_dir = Path(args.package_dir)
        if not package_dir.is_absolute():
            package_dir = REPO_ROOT / package_dir
        package = export_snerv_archive_bound_candidate_package(
            packet=res.receiver_archive_packet,
            output_dir=package_dir,
            repo_root=REPO_ROOT,
            retain_receiver_output=bool(args.retain_package_raw),
            receiver_proof_timeout_seconds=int(args.package_timeout_seconds),
        )
        rows = package["archive_bound_candidate_adapter_package"]["candidate_rows"]
        payload["archive_byte_closure_blockers_before_package"] = list(
            payload.get("archive_byte_closure_blockers", [])
        )
        payload["archive_byte_closure_blockers"] = (
            list(rows[0].get("blockers", [])) if rows else []
        )
        payload["runtime_package_dir"] = str(package_dir)
        payload["runtime_package"] = package
        archive_zip_path = package_dir / "archive.zip"
        if archive_zip_path.is_file():
            _charge_packaged_archive_rate(
                payload=payload,
                advisory_result=res,
                archive_zip_path=archive_zip_path,
            )
        else:
            payload["packaged_archive_rate_accounting_blocker"] = (
                "contest_archive_zip_missing_after_package_export"
            )
    trained_row_archive_path: Path | None = None
    trained_row_archive_kind: str | None = None
    trained_row_receiver_proof: dict[str, object] | None = None
    if package_dir is not None:
        trained_row_archive_path = package_dir / "archive.zip"
        trained_row_archive_kind = "contest_archive_zip"
        if package is not None and isinstance(package.get("receiver_proof"), dict):
            trained_row_receiver_proof = dict(package["receiver_proof"])
    elif packet_path is not None:
        trained_row_archive_path = packet_path
        trained_row_archive_kind = "receiver_snar_packet"
    if trained_row_archive_path is not None and trained_row_archive_kind is not None:
        trained_ladder_row_payload = build_snerv_trained_ladder_row_from_advisory(
            advisory_result=res,
            archive_path=trained_row_archive_path,
            archive_path_kind=trained_row_archive_kind,
            receiver_proof=trained_row_receiver_proof,
            target_bits_per_coeff=args.bits_per_coeff,
            qat_bits=args.qat_bits,
            repo_root=REPO_ROOT,
        )
        payload["trained_ladder_row_payload"] = trained_ladder_row_payload
        payload["trained_ladder_row_archive_path"] = str(trained_row_archive_path)
        payload["trained_ladder_row_archive_path_kind"] = trained_row_archive_kind
        if args.trained_ladder_row_out:
            trained_row_out = Path(args.trained_ladder_row_out)
            if not trained_row_out.is_absolute():
                trained_row_out = REPO_ROOT / trained_row_out
            trained_row_out.parent.mkdir(parents=True, exist_ok=True)
            trained_row_out.write_text(
                json.dumps(trained_ladder_row_payload, indent=2),
                encoding="utf-8",
            )
            payload["trained_ladder_row_payload_path"] = str(trained_row_out)
    elif args.trained_ladder_row_out:
        raise SystemExit(
            "--trained-ladder-row-out requires --packet-out or --package-dir"
        )
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"[SNeRV advisory] {res.axis_tag} NON-PROMOTABLE (Catalog #341/#192/#127/#323)")
    print(f"  carrier {res.carrier_hw}  n_pairs={res.n_pairs} levels={res.levels} {res.wavelet}")
    print(f"  G3 adjoint rel-residual = {res.adjoint_rel_residual:.3e} (exact iff < 1e-12)")
    print(f"  LF stored coeffs = {res.lf_coeff_count_total}  LF payload = {res.lf_payload_bytes} B")
    print(
        "  L-inf step packet = "
        f"{res.linf_steps_payload_bytes} B ({res.linf_steps_payload_codec}, "
        f"mode={res.linf_steps_coder_mode}, bins={res.linf_steps_coder_bins}; "
        f"fp32-lzma baseline "
        f"{res.linf_steps_fp32_lzma_baseline_bytes} B; "
        f"max rel err {res.linf_steps_max_relative_error:.6f})"
    )
    if res.linf_steps_coder_groups:
        groups = [
            (group.get("bins"), len(group.get("map_indices", [])))
            for group in res.linf_steps_coder_groups
        ]
        print(f"  L-inf adaptive groups = {groups}")
    print(
        "  metadata = "
        f"{res.metadata_bytes} B  archive_header = {res.receiver_archive_header_bytes} B"
    )
    charged_archive_bytes_total = int(
        payload.get("archive_bytes_total", res.archive_bytes_total)
    )
    charged_archive_sha256 = str(
        payload.get("charged_archive_sha256", res.receiver_archive_sha256)
    )
    charged_archive_path_kind = str(
        payload.get("charged_archive_path_kind", "receiver_snar_packet")
    )
    charged_rate_term = float(payload.get("rate_term", res.rate_term))
    charged_score_linf = float(payload.get("score_linf", res.score_linf))
    charged_beats_frontier_rate = bool(
        payload.get("beats_frontier_rate", res.beats_frontier_rate)
    )
    print(
        "  decoder = "
        f"{res.decoder_bytes} B  codec={res.decoder_payload_codec} "
        f"fc_dim={res.snerv_fc_dim} "
        f"emb_size={res.snerv_emb_size} "
        f"adapter={res.snerv_model_size_adapter} "
        f"mfu_scales={list(res.snerv_mfu_scales)} "
        f"hfr_gain={res.snerv_hfr_gain:g} "
        f"temporal={res.snerv_temporal_context}:{res.snerv_temporal_mode} "
        f"features={res.decoder_feature_count} "
        f"fit={res.hf_decoder_fit_mode} "
        f"gain={res.hf_decoder_saliency_gain:g} "
        f"component={res.hf_decoder_saliency_component}  "
        f"archive_total = {charged_archive_bytes_total} B "
        f"source={charged_archive_path_kind} "
        f"sha256={charged_archive_sha256[:12]}"
    )
    official_solution = getattr(res, "official_modelsize_solution", None)
    if official_solution:
        print(
            "  modelsize control = official_snerv_modelsize "
            f"mparams={official_solution['modelsize_mparams']} "
            f"solved_fc_dim={official_solution['fc_dim']}"
        )
    else:
        print(
            "  modelsize control = "
            f"{getattr(res, 'snerv_capacity_source', 'manual_fc_dim')}"
        )
    print(
        f"  rate_term = {charged_rate_term:.5f} "
        f"(shared charged archive term; frontier {res.pr101_frontier_bytes} B = "
        f"{res.pr101_frontier_rate:.5f})"
    )
    print(f"  beats_frontier_rate = {charged_beats_frontier_rate}")
    print(f"  d_seg(linf) = {res.d_seg_mean_linf:.5f}  d_pose(linf) = {res.d_pose_mean_linf:.5f}  score_linf = {charged_score_linf:.5f}")
    print(f"  d_seg(l2)   = {res.d_seg_mean_l2:.5f}  d_pose(l2)   = {res.d_pose_mean_l2:.5f}  score_l2   = {res.score_l2:.5f}")
    print(f"  Z8 detail-store-frac = {res.z8_disease_detail_store_frac:.3f}")
    print(f"  Z8 falsification: {res.z8_falsification_verdict}")
    if args.packet_out:
        print(f"  wrote packet {packet_path}")
    if args.package_dir:
        print(f"  wrote runtime package {package_dir}")
        print(
            "  package blockers = "
            f"{', '.join(payload['archive_byte_closure_blockers'])}"
        )
    if payload.get("trained_ladder_row_payload"):
        ladder_payload = payload["trained_ladder_row_payload"]
        print(
            "  trained ladder row = "
            f"{ladder_payload['status']} blockers={len(ladder_payload['blockers'])}"
        )
        if payload.get("trained_ladder_row_payload_path"):
            print(
                f"  wrote trained ladder row {payload['trained_ladder_row_payload_path']}"
            )
    print(f"  wrote {out_path}")
    return 0


def _parse_bins(raw: str) -> tuple[int, ...]:
    out = []
    for chunk in raw.split(","):
        value = int(chunk.strip())
        if value < 2 or value > 256:
            raise ValueError("adaptive bin choices must be in [2, 256]")
        out.append(value)
    if not out:
        raise ValueError("at least one adaptive bin choice is required")
    return tuple(out)


def _parse_positive_int_csv(raw: str) -> tuple[int, ...]:
    out = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        value = int(text)
        if value < 1:
            raise ValueError("positive integer list values must be >= 1")
        out.append(value)
    if not out:
        raise ValueError("at least one positive integer is required")
    return tuple(out)


def _parse_optional_modes(raw: str) -> tuple[str, ...] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    out = tuple(chunk.strip() for chunk in text.split(",") if chunk.strip())
    if not out:
        return None
    return out


if __name__ == "__main__":
    raise SystemExit(main())
