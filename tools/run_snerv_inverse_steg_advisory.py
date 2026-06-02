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
    run_snerv_advisory,
)
from tac.substrates.snerv_inverse_steg_carrier.archive_candidate import (  # noqa: E402
    export_snerv_archive_bound_candidate_package,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (  # noqa: E402
    SNERV_SPECTRA_PRESERVING_ADAPTER,
)
from tac.substrates.snerv_inverse_steg_carrier.trained_ladder_bridge import (  # noqa: E402
    build_snerv_trained_ladder_row_from_advisory,
)


def _default_out() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f".omx/research/snerv_inverse_steg_advisory_{stamp}.json"


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
        default=9,
        help="Receiver-visible LF-context feature count for the SNeRV HF decoder.",
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
        snerv_fc_dim=args.snerv_fc_dim,
        snerv_emb_size=args.snerv_emb_size,
        snerv_patch_radius=args.snerv_patch_radius,
        snerv_model_size_adapter=(
            SNERV_SPECTRA_PRESERVING_ADAPTER
            if args.snerv_spectra_preserving_adapter
            else "snerv_fc_dim_emb_size_adapter_v1"
        ),
        snerv_mfu_scales=_parse_positive_int_csv(args.snerv_mfu_scales),
        snerv_hfr_gain=args.snerv_hfr_gain,
        decoder_payload_codec=args.decoder_payload_codec,
        decoder_payload_mixed_modes=_parse_optional_modes(
            args.decoder_payload_mixed_modes
        ),
    )
    payload = res.as_jsonable()
    payload.setdefault("schema", "snerv_inverse_steg_advisory.v1")
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
    print(
        "  decoder = "
        f"{res.decoder_bytes} B  codec={res.decoder_payload_codec} "
        f"fc_dim={res.snerv_fc_dim} "
        f"emb_size={res.snerv_emb_size} "
        f"adapter={res.snerv_model_size_adapter} "
        f"mfu_scales={list(res.snerv_mfu_scales)} "
        f"hfr_gain={res.snerv_hfr_gain:g} "
        f"features={res.decoder_feature_count} "
        f"fit={res.hf_decoder_fit_mode} "
        f"gain={res.hf_decoder_saliency_gain:g} "
        f"component={res.hf_decoder_saliency_component}  "
        f"archive_total = {res.archive_bytes_total} B "
        f"sha256={res.receiver_archive_sha256[:12]}"
    )
    print(
        f"  rate_term = {res.rate_term:.5f} "
        f"(shared charged archive term; frontier {res.pr101_frontier_bytes} B = "
        f"{res.pr101_frontier_rate:.5f})"
    )
    print(f"  beats_frontier_rate = {res.beats_frontier_rate}")
    print(f"  d_seg(linf) = {res.d_seg_mean_linf:.5f}  d_pose(linf) = {res.d_pose_mean_linf:.5f}  score_linf = {res.score_linf:.5f}")
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
