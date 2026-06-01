# SPDX-License-Identifier: MIT
"""$0 macOS-CPU advisory pipeline + conservative byte accounting for SNeRV.

Runs the COMPLETE SNeRV inverse-steganalysis stack on REAL ``upstream/videos/
0.mkv`` frames (Catalog #213) and produces an honest conservative advisory:

  1. Decode real pairs; for each frame analyze the orthonormal DWT pyramid.
  2. Fit the HF-generation decoder (ridge least squares) on real LF->HF maps.
  3. Compute the bit-exact oracle (P18 SegNet flip-risk + P19 PoseNet Fisher).
  4. Push the pixel oracle into the LF-coefficient domain via the EXACT adjoint;
     allocate the LF bit budget with the proven L-inf pose-Fisher allocator.
  5. Quantize the LF at the allocated per-coefficient steps; GENERATE the HF;
     synthesize the reconstructed frame.
  6. Charge a receiver-visible packet (entropy-coded LF + shared decoder +
     metadata + L-inf step maps) and measure ``25 * archive_bytes / N``.
  7. Re-measure ``d_seg`` / ``d_pose`` on the reconstructed pairs via the bit-exact
     mirror, and run the Z8-falsification check: does SNeRV (LF-store +
     HF-generate + L-inf) beat the PR101 frontier grammar at equal
     detector-preserving distortion, at SMALLER rate?

ALL numbers here are ``[macOS-CPU advisory]`` NON-PROMOTABLE (Catalog #341/#192/
#127/#323). NO scorer crosses the receiver boundary in the byte-closed path
(``tac.contest_eval_contract``); the scorer is used OFFLINE for the oracle and the
advisory re-measure only.

NO-FAKE: every reported number is computed live from real frames + the real
scorer + a real charged packet estimate. No advisory number is fabricated; if a
step cannot run it raises rather than emitting a placeholder. This module does
not grant exact-eval readiness until the receiver grammar can consume the charged
step maps and reproduce the same frames.
"""

from __future__ import annotations

import hashlib
import lzma
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from tac.analysis.inverse_steganalysis_linf_vs_l2_gate import (
    allocate_l2_uniform,
    measure_pair_d_seg_d_pose,
)
from tac.analysis.score_exact_saliency import (
    compute_s_pose_fisher,
    compute_s_seg_flip_risk,
    decode_real_pairs,
    load_score_exact_scorers,
)
from tac.analysis.snerv_step_map_coder import (
    AdaptiveStepMapPacket,
    StepMapPacket,
    decode_step_maps,
    encode_step_maps,
    encode_step_maps_adaptive,
    encode_step_maps_waterfill,
)
from tac.substrates.snerv_inverse_steg_carrier.allocation import (
    allocate_lf_linf,
    push_pixel_saliency_to_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    SnervArchivePacket,
    decode_snerv_archive_frames,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SnervFrameCode,
    decode_frame,
    encode_frame_lf,
    fit_hf_decoder_least_squares,
    fit_hf_decoder_weighted_least_squares,
    quantize_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.dwt import (
    WaveletPyramid,
    dwt2_native_synthesis_adjoint,
)

__all__ = [
    "CONTEST_BYTE_PRICE",
    "CONTEST_RATE_DENOM",
    "SnervAdvisoryResult",
    "run_snerv_advisory",
]

# Contest rate denominator (verified bit-exact mirror, design memo §1).
CONTEST_RATE_DENOM = 37_545_489
CONTEST_BYTE_PRICE = 25.0 / CONTEST_RATE_DENOM  # 6.659e-7 per byte
STEP_MAP_CODER_BINS = 128


@dataclass(frozen=True)
class _LfRecord:
    """One LF coefficient plane whose L-inf steps must be receiver-visible."""

    pair_index: int
    frame_index: int
    channel_index: int
    lf: np.ndarray


@dataclass(frozen=True)
class SnervAdvisoryResult:
    """Honest byte-closed advisory for the SNeRV carrier on real frames."""

    n_pairs: int
    levels: int
    wavelet: str
    carrier_hw: tuple[int, int]
    # rate
    lf_coeff_count_total: int  # stored LF coeffs across all frames+channels
    decoder_bytes: int
    hf_decoder_fit_mode: str
    hf_decoder_saliency_gain: float
    hf_decoder_saliency_component: str
    metadata_bytes: int
    receiver_archive_header_bytes: int
    receiver_archive_packet_bytes: int
    receiver_archive_sha256: str
    receiver_archive_packet: bytes
    receiver_archive_replay_verified: bool
    receiver_archive_replay_error: str
    lf_payload_bytes: int  # entropy-coded quantized LF
    linf_steps_payload_bytes: int
    linf_steps_payload_codec: str
    linf_steps_coder_mode: str
    linf_steps_coder_bins: int
    linf_steps_coder_groups: tuple[dict[str, Any], ...]
    linf_steps_fp32_lzma_baseline_bytes: int
    linf_steps_max_relative_error: float
    linf_steps_mean_relative_error: float
    archive_bytes_total: int  # charged packet byte estimate (the rate numerator)
    rate_term: float  # 25 * archive_bytes / N
    # distortion (advisory, bit-exact mirror)
    d_seg_mean_linf: float
    d_pose_mean_linf: float
    score_linf: float  # 100*d_seg + sqrt(10*d_pose) + rate
    d_seg_mean_l2: float
    d_pose_mean_l2: float
    score_l2: float
    # G3 / oracle diagnostics
    adjoint_rel_residual: float
    lf_seg_mass_mean: float
    lf_pose_mass_mean: float
    # Z8-falsification
    z8_disease_detail_store_frac: float  # frac of archive that would be detail if STORED
    z8_falsification_verdict: str
    pr101_frontier_bytes: int
    pr101_frontier_rate: float
    beats_frontier_rate: bool
    # provenance
    axis_tag: str
    promotable: bool
    score_claim: bool
    archive_byte_closure_blockers: tuple[str, ...]
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool
    wall_seconds: float
    notes: str

    def as_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["carrier_hw"] = list(self.carrier_hw)
        d["receiver_archive_packet"] = {
            "bytes": len(self.receiver_archive_packet),
            "sha256": hashlib.sha256(self.receiver_archive_packet).hexdigest(),
            "redacted": True,
        }
        return d


def _entropy_code_lf_quant(lf_quant_per_frame: list[np.ndarray]) -> bytes:
    """Byte-close the quantized LF integers with a real codec (lzma over zigzag).

    Concatenate all frames' LF integers, zigzag to non-negative, delta-code along
    the flattened raster (spatial smoothness of LF coeffs), then lzma (FORMAT_XZ).
    This is a REAL entropy-coded payload — the byte count is the measured rate, not
    a model estimate. Honest about being a baseline coder (the design memo §4 says
    the byte-format lever is saturated; the SNeRV lever is L2+L3, not L4).
    """
    parts = []
    for q in lf_quant_per_frame:
        flat = np.asarray(q, dtype=np.int64).ravel()
        # zigzag map signed->unsigned
        zz = np.where(flat >= 0, 2 * flat, -2 * flat - 1).astype(np.int64)
        # delta along raster
        delta = np.empty_like(zz)
        delta[0] = zz[0] if zz.size else 0
        if zz.size > 1:
            delta[1:] = zz[1:] - zz[:-1]
        zzd = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint64)
        parts.append(zzd.astype("<u8").tobytes())
    raw = b"".join(parts)
    return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)


def _encode_decode_linf_steps_packet(
    steps_per_frame: list[np.ndarray],
    *,
    bins: int = STEP_MAP_CODER_BINS,
    mode: str = "uniform",
    map_importance: list[float] | np.ndarray | None = None,
    adaptive_bin_choices: tuple[int, ...] = (128, 16, 4),
    constant_importance_quantile: float | None = None,
    waterfill_bits_per_coeff: float = 4.0,
) -> tuple[StepMapPacket | AdaptiveStepMapPacket, list[np.ndarray]]:
    """Charge and consume the receiver-visible compact L-inf step-map packet."""

    if mode == "uniform":
        packet = encode_step_maps(steps_per_frame, bins=bins)
    elif mode == "adaptive":
        if map_importance is None:
            raise RuntimeError("adaptive step-map coding requires map_importance")
        packet = encode_step_maps_adaptive(
            steps_per_frame,
            map_importance=map_importance,
            bin_choices=adaptive_bin_choices,
            constant_importance_quantile=constant_importance_quantile,
        )
    elif mode == "waterfill":
        if map_importance is None:
            raise RuntimeError("waterfill step-map coding requires map_importance")
        packet = encode_step_maps_waterfill(
            steps_per_frame,
            map_importance=map_importance,
            target_bits_per_coeff=waterfill_bits_per_coeff,
        )
    else:
        raise RuntimeError(f"unknown step-map coder mode: {mode!r}")
    decoded = decode_step_maps(packet.packet)
    if len(decoded) != len(steps_per_frame):
        raise RuntimeError(
            f"decoded {len(decoded)} step maps from packet, expected {len(steps_per_frame)}"
        )
    for idx, (ref, got) in enumerate(zip(steps_per_frame, decoded, strict=True)):
        if got.shape != ref.shape:
            raise RuntimeError(
                f"decoded step map {idx} shape {got.shape} != expected {ref.shape}"
            )
    return packet, decoded


def _verify_receiver_archive_roundtrip(
    receiver_archive: SnervArchivePacket,
    *,
    lf_quant_planes: list[np.ndarray],
    lf_zero_points: list[float],
    step_maps: list[np.ndarray],
    decoder,
) -> tuple[bool, str]:
    """Verify advisory bytes can be consumed into the same receiver state."""

    try:
        decoded = unpack_snerv_archive(receiver_archive.packet)
        decoded_lf = decoded.decode_lf_quant_planes()
        if len(decoded_lf) != len(lf_quant_planes):
            return False, "lf_quant_plane_count_mismatch"
        for idx, (ref, got) in enumerate(zip(lf_quant_planes, decoded_lf, strict=True)):
            if not np.array_equal(got, ref):
                return False, f"lf_quant_plane_{idx}_mismatch"

        decoded_zeros = decoded.decode_lf_zero_points()
        ref_zeros = np.asarray(lf_zero_points, dtype="<f4")
        if not np.array_equal(decoded_zeros, ref_zeros):
            return False, "lf_zero_points_mismatch"

        decoded_steps = decoded.decode_step_maps()
        if len(decoded_steps) != len(step_maps):
            return False, "step_map_count_mismatch"
        for idx, (ref, got) in enumerate(zip(step_maps, decoded_steps, strict=True)):
            if not np.array_equal(got, ref):
                return False, f"step_map_{idx}_mismatch"

        decoded_decoder = decoded.decode_decoder()
        if decoded_decoder.levels != decoder.levels:
            return False, "decoder_level_count_mismatch"
        for lvl in range(decoder.levels):
            for subband in ("LH", "HL", "HH"):
                ref = np.asarray(decoder.kernels[lvl][subband], dtype="<f4")
                got = np.asarray(decoded_decoder.kernels[lvl][subband], dtype="<f4")
                if not np.array_equal(got, ref):
                    return False, f"decoder_{lvl}_{subband}_mismatch"
    except Exception as exc:  # pragma: no cover - exercised via failure tests.
        return False, f"{type(exc).__name__}:{exc}"

    return True, ""


def _verify_receiver_archive_full_frame_replay(
    receiver_archive: SnervArchivePacket,
    *,
    reference_frames: torch.Tensor,
    atol: float = 5e-5,
) -> tuple[bool, str, np.ndarray | None]:
    """Verify SNAR1 bytes replay to the measured full-frame tensor."""

    try:
        receiver_frames = decode_snerv_archive_frames(receiver_archive.packet)
        reference = reference_frames.detach().cpu().numpy().astype(np.float32)
        if receiver_frames.shape != reference.shape:
            return (
                False,
                f"receiver_frame_shape_mismatch:{receiver_frames.shape}!={reference.shape}",
                receiver_frames,
            )
        max_abs = float(np.max(np.abs(receiver_frames - reference)))
        if not np.allclose(receiver_frames, reference, atol=atol, rtol=0.0):
            return False, f"receiver_frame_replay_mismatch:max_abs={max_abs:.8g}", receiver_frames
        return True, "", receiver_frames
    except Exception as exc:  # pragma: no cover - exercised via failure tests.
        return False, f"{type(exc).__name__}:{exc}", None


def run_snerv_advisory(
    *,
    n_pairs: int = 8,
    levels: int = 3,
    wavelet: str = "db2",
    target_bits_per_coeff: float = 5.0,
    pair_stride: int = 1,
    start_pair: int = 0,
    pr101_frontier_bytes: int = 178_493,
    video_path: str = "upstream/videos/0.mkv",
    device: str = "cpu",
    step_map_coder_bins: int = STEP_MAP_CODER_BINS,
    step_map_coder_mode: str = "uniform",
    step_map_adaptive_bin_choices: tuple[int, ...] = (128, 16, 4),
    step_map_constant_importance_quantile: float | None = None,
    step_map_waterfill_bits_per_coeff: float = 4.0,
    hf_decoder_fit_mode: str = "least_squares",
    hf_decoder_saliency_gain: float = 1.0,
    hf_decoder_saliency_component: str = "combined",
) -> SnervAdvisoryResult:
    """Run the complete byte-closed SNeRV advisory on ``n_pairs`` real pairs.

    ``target_bits_per_coeff`` is the LF bit budget per stored coefficient (the
    fairness invariant shared by the L-inf and L2 allocations). ``pr101_frontier_
    bytes`` is the canonical frontier archive size (pointer-only per Catalog #343;
    default = PR101/HNeRV 178,493 B).
    """
    t0 = time.time()
    posenet, segnet = load_score_exact_scorers(device=device)
    pairs = decode_real_pairs(
        video_path, n_pairs, pair_stride=pair_stride, start_pair=start_pair, device=device
    )  # (n_pairs, 2, 3, H, W)
    H, W = int(pairs.shape[-2]), int(pairs.shape[-1])

    # ---- 1. Analyze every frame channel; collect pyramids for decoder fit ----
    # The carrier reconstructs BOTH frames of each pair (d_seg = last frame,
    # d_pose = both frames). We process each of the 2*n_pairs frames.
    train_pyrs: list = []
    weight_pyrs: list[WaveletPyramid] | None = [] if hf_decoder_fit_mode == "score_weighted" else None
    pair_saliency_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for p in range(n_pairs):
        if hf_decoder_fit_mode == "score_weighted":
            seg = compute_s_seg_flip_risk(segnet, pairs[p])
            pose = compute_s_pose_fisher(posenet, pairs[p])
            seg_hw = seg.flip_risk.cpu().numpy()
            pose_hw = pose.s_pose.cpu().numpy()
            pair_saliency_cache[p] = (seg_hw, pose_hw)
        for fidx in range(2):
            frame = pairs[p, fidx].cpu().numpy()  # (3, H, W)
            for c in range(3):
                train_pyrs.append(encode_frame_lf(frame[c], levels=levels, wavelet=wavelet))
                if weight_pyrs is not None:
                    seg_hw, pose_hw = pair_saliency_cache[p]
                    coeff_saliency = dwt2_native_synthesis_adjoint(
                        _combine_hf_decoder_saliency(
                            seg_hw,
                            pose_hw,
                            component=hf_decoder_saliency_component,
                            target_hw=(H, W),
                        ),
                        levels=levels,
                        wavelet=wavelet,
                    )
                    weight_pyrs.append(coeff_saliency)
    if hf_decoder_fit_mode == "least_squares":
        decoder = fit_hf_decoder_least_squares(train_pyrs, levels=levels)
    elif hf_decoder_fit_mode == "score_weighted":
        decoder = fit_hf_decoder_weighted_least_squares(
            train_pyrs,
            levels=levels,
            detail_weight_pyramids=weight_pyrs,
            saliency_gain=hf_decoder_saliency_gain,
        )
    else:
        raise RuntimeError(f"unknown HF decoder fit mode: {hf_decoder_fit_mode!r}")

    # ---- 2. Oracle: per-pixel saliency on the LAST frame (seg) + both (pose) ----
    # Average the oracle across pairs to get a representative allocation map (the
    # carrier's LF bit allocation is content-adaptive but we use one shared map for
    # the advisory; per-pair would be cheaper but the shared map is conservative).
    adjoint_res = float("nan")
    from tac.substrates.snerv_inverse_steg_carrier.dwt import synthesis_adjoint_residual

    adjoint_res = synthesis_adjoint_residual((H, W), levels=levels, wavelet=wavelet)

    # ---- 3+4+5. Per-frame: oracle -> LF allocation -> quantize -> generate -> decode
    lf_seg_masses: list[float] = []
    lf_pose_masses: list[float] = []
    lf_quant_linf: list[np.ndarray] = []
    lf_quant_l2: list[np.ndarray] = []
    lf_zero_points_linf: list[float] = []
    linf_step_maps: list[np.ndarray] = []
    linf_map_importance: list[float] = []
    linf_records: list[_LfRecord] = []
    recon_pairs_linf = pairs.clone()
    recon_pairs_l2 = pairs.clone()
    lf_count_total = 0

    for p in range(n_pairs):
        # oracle on this pair (seg = last frame; pose = both frames)
        if p in pair_saliency_cache:
            seg_hw, pose_hw = pair_saliency_cache[p]
        else:
            seg = compute_s_seg_flip_risk(segnet, pairs[p])
            pose = compute_s_pose_fisher(posenet, pairs[p])
            seg_hw = seg.flip_risk.cpu().numpy()  # (384,512)
            pose_hw = pose.s_pose.cpu().numpy()  # (H?,W?) at native after preprocess backprop

        for fidx in range(2):
            frame = pairs[p, fidx].cpu().numpy()  # (3,H,W)
            for c in range(3):
                pyr = encode_frame_lf(frame[c], levels=levels, wavelet=wavelet)
                lf = pyr.lf
                lf_count_total += int(lf.size)
                target_bits = float(lf.size) * target_bits_per_coeff

                # G3: push pixel oracle into LF domain (resize pose to seg's HW
                # first so both are comparable; both resized to carrier HW inside).
                lfs = push_pixel_saliency_to_lf(
                    seg_hw,
                    _match_hw(pose_hw, seg_hw.shape),
                    carrier_hw=(H, W),
                    levels=levels,
                    wavelet=wavelet,
                )
                lf_seg_masses.append(lfs.pixel_seg_mass)
                lf_pose_masses.append(lfs.pixel_pose_mass)
                linf_map_importance.append(
                    float(
                        np.mean(
                            np.asarray(lfs.lf_saliency, dtype=np.float64)
                            * (np.abs(lf).astype(np.float64) + 1e-6)
                        )
                    )
                )

                # L-inf allocation onto LF coeffs
                alloc = allocate_lf_linf(lfs, target_bits=target_bits, min_step=0.5)
                steps = alloc.steps.reshape(lf.shape)
                linf_step_maps.append(steps)
                linf_records.append(
                    _LfRecord(
                        pair_index=p,
                        frame_index=fidx,
                        channel_index=c,
                        lf=lf,
                    )
                )

                # L2 baseline: uniform step at the SAME total bit budget
                l2alloc = allocate_l2_uniform(lf.size, target_bits=target_bits)
                l2steps = l2alloc.steps.reshape(lf.shape)
                q_l2, sc_l2, zr_l2 = quantize_lf(lf, per_element_steps=l2steps)
                lf_quant_l2.append(q_l2)
                code_l2 = SnervFrameCode(
                    lf_quant=q_l2, lf_scale=sc_l2, lf_zero=zr_l2,
                    lf_shape=lf.shape, levels=levels, wavelet=wavelet,
                    orig_hw=(H, W), per_element_steps=l2steps,
                )
                recon_l2 = decode_frame(code_l2, decoder)
                recon_pairs_l2[p, fidx, c] = torch.from_numpy(
                    np.clip(recon_l2, 0.0, 255.0)
                ).to(recon_pairs_l2)

    steps_packet, receiver_step_maps = _encode_decode_linf_steps_packet(
        linf_step_maps,
        bins=step_map_coder_bins,
        mode=step_map_coder_mode,
        map_importance=linf_map_importance,
        adaptive_bin_choices=step_map_adaptive_bin_choices,
        constant_importance_quantile=step_map_constant_importance_quantile,
        waterfill_bits_per_coeff=step_map_waterfill_bits_per_coeff,
    )
    for record, receiver_steps in zip(linf_records, receiver_step_maps, strict=True):
        q_linf, sc_linf, zr_linf = quantize_lf(
            record.lf,
            per_element_steps=receiver_steps,
        )
        lf_zero_points_linf.append(float(zr_linf))
        lf_quant_linf.append(q_linf)
        code_linf = SnervFrameCode(
            lf_quant=q_linf,
            lf_scale=sc_linf,
            lf_zero=zr_linf,
            lf_shape=record.lf.shape,
            levels=levels,
            wavelet=wavelet,
            orig_hw=(H, W),
            per_element_steps=receiver_steps,
        )
        recon_linf = decode_frame(code_linf, decoder)
        recon_pairs_linf[
            record.pair_index,
            record.frame_index,
            record.channel_index,
        ] = torch.from_numpy(np.clip(recon_linf, 0.0, 255.0)).to(recon_pairs_linf)

    # ---- 6. Charge the receiver-visible packet (the rate numerator) ----
    lf_payload = encode_lf_quant_payload(lf_quant_linf)
    decoder_bytes = encode_decoder_payload(decoder)
    metadata_payload = encode_lf_metadata_payload(
        lf_zero_points=lf_zero_points_linf,
    )
    receiver_archive = pack_snerv_archive(
        metadata_payload=metadata_payload,
        lf_payload=lf_payload,
        decoder_payload=decoder_bytes,
        step_map_packet=steps_packet.packet,
        metadata={
            "n_pairs": n_pairs,
            "frames_per_pair": 2,
            "channels": 3,
            "levels": levels,
            "wavelet": wavelet,
            "carrier_hw": [H, W],
            "lf_plane_count": len(lf_zero_points_linf),
            "lf_coeff_count_total": lf_count_total,
            "lf_zero_dtype": "float32_le",
            "lf_scale_mode": "implicit_per_element_steps_scale_1",
            "step_map_packet_schema": steps_packet.schema,
            "step_map_coder_mode": step_map_coder_mode,
            "step_map_coder_bins": step_map_coder_bins,
            "step_map_adaptive_bin_choices": list(step_map_adaptive_bin_choices),
            "step_map_constant_importance_quantile": step_map_constant_importance_quantile,
            "step_map_waterfill_bits_per_coeff": step_map_waterfill_bits_per_coeff,
            "hf_decoder_fit_mode": hf_decoder_fit_mode,
            "hf_decoder_saliency_gain": hf_decoder_saliency_gain,
            "hf_decoder_saliency_component": hf_decoder_saliency_component,
        },
    )
    metadata = len(metadata_payload)
    archive_bytes = receiver_archive.total_bytes
    rate_term = CONTEST_BYTE_PRICE * archive_bytes
    section_replay_ok, section_replay_error = _verify_receiver_archive_roundtrip(
        receiver_archive,
        lf_quant_planes=lf_quant_linf,
        lf_zero_points=lf_zero_points_linf,
        step_maps=receiver_step_maps,
        decoder=decoder,
    )
    frame_replay_ok, frame_replay_error, receiver_frames_np = (
        _verify_receiver_archive_full_frame_replay(
            receiver_archive,
            reference_frames=recon_pairs_linf,
        )
    )
    receiver_archive_replay_verified = bool(section_replay_ok and frame_replay_ok)
    receiver_archive_replay_error = (
        section_replay_error if not section_replay_ok else frame_replay_error
    )
    if receiver_frames_np is not None:
        recon_pairs_linf = torch.from_numpy(receiver_frames_np).to(pairs)

    # ---- 7. Re-measure d_seg/d_pose via bit-exact mirror ----
    dsegs_linf, dposes_linf, dsegs_l2, dposes_l2 = [], [], [], []
    for p in range(n_pairs):
        # measure expects (1, 2, 3, H, W) so preprocess's x[:, -1] slice works.
        gt = pairs[p : p + 1]
        ds_l, dp_l = measure_pair_d_seg_d_pose(
            posenet, segnet, gt, recon_pairs_linf[p : p + 1]
        )
        ds_2, dp_2 = measure_pair_d_seg_d_pose(
            posenet, segnet, gt, recon_pairs_l2[p : p + 1]
        )
        dsegs_linf.append(ds_l)
        dposes_linf.append(dp_l)
        dsegs_l2.append(ds_2)
        dposes_l2.append(dp_2)
    d_seg_linf = float(np.mean(dsegs_linf))
    d_pose_linf = float(np.mean(dposes_linf))
    d_seg_l2 = float(np.mean(dsegs_l2))
    d_pose_l2 = float(np.mean(dposes_l2))
    score_linf = 100.0 * d_seg_linf + float(np.sqrt(10.0 * max(d_pose_linf, 0.0))) + rate_term
    score_l2 = 100.0 * d_seg_l2 + float(np.sqrt(10.0 * max(d_pose_l2, 0.0))) + rate_term

    # ---- Z8-falsification: would STORING the detail blow the archive? ----
    # Estimate the bytes the HF detail WOULD cost if stored at the same coder (the
    # Z8 disease). The detail coeff count = total_coeff - lf_count.
    detail_count_total = 0
    detail_quant = []
    for pyr in train_pyrs:
        for lh, hl, hh in pyr.details:
            for d in (lh, hl, hh):
                detail_count_total += int(d.size)
                # quantize detail at a comparable step for the estimate
                dq, _, _ = quantize_lf(np.asarray(d), n_levels=256)
                detail_quant.append(dq)
    detail_payload = _entropy_code_lf_quant(detail_quant) if detail_quant else b""
    detail_store_bytes = len(detail_payload)
    would_be_archive = archive_bytes + detail_store_bytes
    z8_disease_frac = detail_store_bytes / max(would_be_archive, 1)

    pr101_rate = CONTEST_BYTE_PRICE * pr101_frontier_bytes
    beats_frontier_rate = archive_bytes < pr101_frontier_bytes

    # Z8-falsification verdict: SNeRV PASSES the gate iff its archive (LF-store +
    # generate-HF) is SMALLER than the frontier AND the detail it AVOIDED storing
    # would have dominated (proving the generate-HF cure works).
    if beats_frontier_rate and z8_disease_frac > 0.5:
        verdict = (
            "PASS: SNeRV archive < frontier AND generated (not stored) detail "
            f"would have been {z8_disease_frac:.1%} of a stored archive (Z8 disease cured)"
        )
    elif beats_frontier_rate:
        verdict = (
            "PARTIAL: SNeRV archive < frontier but detail-store fraction "
            f"{z8_disease_frac:.1%} is modest (LF itself is the cost, not avoided detail)"
        )
    else:
        verdict = (
            f"FAIL: SNeRV archive {archive_bytes} >= frontier {pr101_frontier_bytes} "
            "(LF store + decoder do not beat the frontier grammar at this operating point)"
        )

    blockers = ["paired_contest_cpu_cuda_auth_eval_missing", "not_packaged_as_contest_archive_zip"]
    if n_pairs != 600:
        blockers.insert(0, "full_600_pair_receiver_replay_missing")
    if not receiver_archive_replay_verified:
        blockers.insert(0, "receiver_archive_full_frame_replay_failed")

    return SnervAdvisoryResult(
        n_pairs=n_pairs,
        levels=levels,
        wavelet=wavelet,
        carrier_hw=(H, W),
        lf_coeff_count_total=lf_count_total,
        decoder_bytes=len(decoder_bytes),
        hf_decoder_fit_mode=hf_decoder_fit_mode,
        hf_decoder_saliency_gain=float(hf_decoder_saliency_gain),
        hf_decoder_saliency_component=hf_decoder_saliency_component,
        metadata_bytes=metadata,
        receiver_archive_header_bytes=receiver_archive.header_bytes,
        receiver_archive_packet_bytes=receiver_archive.total_bytes,
        receiver_archive_sha256=receiver_archive.as_jsonable()["packet"]["sha256"],
        receiver_archive_packet=receiver_archive.packet,
        receiver_archive_replay_verified=receiver_archive_replay_verified,
        receiver_archive_replay_error=receiver_archive_replay_error,
        lf_payload_bytes=len(lf_payload),
        linf_steps_payload_bytes=steps_packet.total_bytes,
        linf_steps_payload_codec=steps_packet.schema,
        linf_steps_coder_mode=step_map_coder_mode,
        linf_steps_coder_bins=int(getattr(steps_packet, "bins", 0)),
        linf_steps_coder_groups=tuple(getattr(steps_packet, "groups", ())),
        linf_steps_fp32_lzma_baseline_bytes=steps_packet.fp32_lzma_baseline_bytes,
        linf_steps_max_relative_error=steps_packet.max_relative_error,
        linf_steps_mean_relative_error=steps_packet.mean_relative_error,
        archive_bytes_total=archive_bytes,
        rate_term=rate_term,
        d_seg_mean_linf=d_seg_linf,
        d_pose_mean_linf=d_pose_linf,
        score_linf=score_linf,
        d_seg_mean_l2=d_seg_l2,
        d_pose_mean_l2=d_pose_l2,
        score_l2=score_l2,
        adjoint_rel_residual=adjoint_res,
        lf_seg_mass_mean=float(np.mean(lf_seg_masses)),
        lf_pose_mass_mean=float(np.mean(lf_pose_masses)),
        z8_disease_detail_store_frac=z8_disease_frac,
        z8_falsification_verdict=verdict,
        pr101_frontier_bytes=pr101_frontier_bytes,
        pr101_frontier_rate=pr101_rate,
        beats_frontier_rate=beats_frontier_rate,
        axis_tag="[macOS-CPU advisory]",
        promotable=False,
        score_claim=False,
        archive_byte_closure_blockers=tuple(blockers),
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
        wall_seconds=time.time() - t0,
        notes=(
            "NON-PROMOTABLE advisory per Catalog #341/#192/#127/#323. n_pairs subset "
            "of 600; full-video + paired CPU+CUDA reserved for operator auth. The "
            "score numbers are advisory-only; L-inf step maps are charged and "
            "consumed through the compact packet in this advisory and bundled into "
            "a receiver archive packet, but contest archive packaging and paired "
            "CPU/CUDA eval remain missing."
        ),
    )


def _match_hw(field: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """NN-resize a saliency field to ``target_hw`` (the pose surface to the seg HW)."""
    th, tw = target_hw
    fh, fw = field.shape[-2], field.shape[-1]
    f = np.asarray(field, dtype=np.float64)
    if f.ndim > 2:
        f = f.reshape(f.shape[-2], f.shape[-1])
    if (fh, fw) == (th, tw):
        return f
    ri = (np.arange(th) * fh // th).clip(0, fh - 1)
    ci = (np.arange(tw) * fw // tw).clip(0, fw - 1)
    return f[np.ix_(ri, ci)]


def _combine_hf_decoder_saliency(
    seg_hw: np.ndarray,
    pose_hw: np.ndarray,
    *,
    component: str,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Select the detector component used to weight HF decoder fitting."""

    seg = _match_hw(seg_hw, target_hw)
    pose = _match_hw(pose_hw, target_hw)
    if component == "combined":
        return seg + pose
    if component == "seg":
        return seg
    if component == "pose":
        return pose
    raise RuntimeError(
        "hf_decoder_saliency_component must be one of combined, seg, pose"
    )
