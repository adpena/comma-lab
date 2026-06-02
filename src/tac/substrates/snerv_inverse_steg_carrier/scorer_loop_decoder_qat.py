# SPDX-License-Identifier: MIT
"""Bounded SNeRV scorer-loop decoder/QAT smoke.

This is a local-only implementation smoke for the next SNeRV step. It trains the
shared HF decoder weights by trying small quantization-aware perturbations and
measuring the reconstructed SNAR1 receiver output with the real SegNet/PoseNet
mirror. It is intentionally tiny and false-authority: a passing local smoke can
only feed the pose-guarded continuation gate, never promotion or exact eval.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from tac.analysis.inverse_steganalysis_linf_vs_l2_gate import measure_pair_d_seg_d_pose
from tac.analysis.score_exact_saliency import (
    compute_s_pose_fisher,
    compute_s_seg_flip_risk,
    decode_real_pairs,
    load_score_exact_scorers,
)
from tac.analysis.snerv_step_map_coder import decode_step_maps, encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.advisory import CONTEST_BYTE_PRICE
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
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    HfGenerationDecoder,
    SnervFrameCode,
    encode_frame_lf,
    fit_hf_decoder_least_squares,
    quantize_lf,
)

SCHEMA = "snerv_scorer_loop_decoder_qat_smoke.v1"
AXIS_TAG = "[macOS-CPU advisory]"


class SnervScorerLoopDecoderQatError(ValueError):
    """Raised when the bounded scorer-loop smoke cannot run safely."""


@dataclass(frozen=True)
class QuantizedDecoderStats:
    """Summary of fake-quantizing the shared HF decoder weights."""

    bits: int
    scale_count: int
    zero_scale_count: int
    total_weights: int
    zero_weight_fraction: float
    max_abs_error: float
    mean_abs_error: float
    payload_bytes_fp32_receiver: int
    payload_sha256_fp32_receiver: str

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnervPairEval:
    """Per-pair detector response for one receiver-replayed decoder evaluation."""

    pair_index: int
    d_seg_linf: float
    d_pose_linf: float
    score_linf_without_rate: float

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnervPairDelta:
    """Per-pair delta against the baseline decoder evaluation."""

    pair_index: int
    d_seg_linf_delta: float
    d_pose_linf_delta: float
    score_linf_without_rate_delta: float

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnervDecoderEval:
    """One receiver-replayed decoder evaluation in the local scorer loop."""

    label: str
    iteration: int
    archive_bytes: int
    archive_sha256: str
    d_seg_linf: float
    d_pose_linf: float
    score_linf: float
    rate_aware_objective_linf: float
    rate_term: float
    receiver_archive_replay_verified: bool
    accepted: bool
    blockers: tuple[str, ...]
    quantized_decoder: QuantizedDecoderStats
    per_pair: tuple[SnervPairEval, ...]

    def as_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quantized_decoder"] = self.quantized_decoder.as_jsonable()
        payload["per_pair"] = [row.as_jsonable() for row in self.per_pair]
        return payload


@dataclass(frozen=True)
class SnervScorerLoopDecoderQatSmokeResult:
    """False-authority result from a bounded local SNeRV decoder/QAT smoke."""

    schema: str
    axis_tag: str
    n_pairs: int
    levels: int
    wavelet: str
    target_bits_per_coeff: float
    qat_bits: int
    max_trials: int
    search_mode: str
    perturb_scale: float
    byte_pressure_multiplier: float
    max_archive_byte_growth: int | None
    pose_slack: float
    seg_slack: float
    pair_guard_min_score_improved_fraction: float
    pair_guard_max_pose_worsened_fraction: float
    baseline: SnervDecoderEval
    best: SnervDecoderEval
    evaluations: tuple[SnervDecoderEval, ...]
    accepted_improvement: bool
    improvement_score_delta: float
    improvement_d_pose_delta: float
    improvement_d_seg_delta: float
    best_pair_deltas: tuple[SnervPairDelta, ...]
    scorer_loop_evaluations: int
    real_frame_source: str
    receiver_contract_satisfied: bool
    score_claim: bool
    promotion_eligible: bool
    rank_or_kill_eligible: bool
    ready_for_exact_eval_dispatch: bool
    ready_for_pose_guard_gate: bool
    blockers: tuple[str, ...]
    forbidden_next_actions: tuple[str, ...]
    wall_seconds: float

    def as_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["baseline"] = self.baseline.as_jsonable()
        payload["best"] = self.best.as_jsonable()
        payload["evaluations"] = [row.as_jsonable() for row in self.evaluations]
        payload["best_pair_deltas"] = [
            row.as_jsonable() for row in self.best_pair_deltas
        ]
        payload["rows"] = [_eval_as_gate_row(row) for row in self.evaluations]
        return payload


@dataclass(frozen=True)
class _CodeRecord:
    pair_index: int
    frame_index: int
    channel_index: int
    code: SnervFrameCode


@dataclass(frozen=True)
class _PreparedState:
    pairs: torch.Tensor
    codes: tuple[_CodeRecord, ...]
    lf_quant_planes: tuple[np.ndarray, ...]
    lf_zero_points: tuple[float, ...]
    step_maps: tuple[np.ndarray, ...]
    step_map_packet: bytes
    baseline_decoder: HfGenerationDecoder
    levels: int
    wavelet: str
    orig_hw: tuple[int, int]


def run_snerv_scorer_loop_decoder_qat_smoke(
    *,
    n_pairs: int = 1,
    levels: int = 2,
    wavelet: str = "db2",
    target_bits_per_coeff: float = 5.0,
    pair_stride: int = 1,
    start_pair: int = 0,
    upstream_dir: str = "upstream",
    video_path: str = "upstream/videos/0.mkv",
    device: str = "cpu",
    step_map_bins: int = 16,
    qat_bits: int = 8,
    max_trials: int = 2,
    search_mode: str = "random_signed",
    perturb_scale: float = 0.02,
    byte_pressure_multiplier: float = 1.0,
    max_archive_byte_growth: int | None = None,
    pose_slack: float = 0.0,
    seg_slack: float = 0.0,
    pair_guard_min_score_improved_fraction: float = 0.0,
    pair_guard_max_pose_worsened_fraction: float = 1.0,
    seed: int = 1337,
) -> SnervScorerLoopDecoderQatSmokeResult:
    """Run a tiny real-frame scorer-loop decoder/QAT smoke.

    The loop is deliberately bounded: it evaluates the least-squares decoder and
    up to ``2 * max_trials`` signed decoder perturbations after fake
    quantization. ``random_signed`` probes global random signs; ``top_weight_
    coordinate`` probes the largest-magnitude decoder atoms one at a time;
    ``learned_random_subspace`` runs a non-coordinate scorer-loop hill climb in
    smooth random affine subspace directions. ``nes_pair_robust`` evaluates
    symmetric probes, estimates a pair-robust objective gradient, and tests one
    synthesized update. A candidate is accepted only if it improves advisory
    score, keeps PoseNet within the hard continuation guard, and keeps SegNet
    inside the explicitly configured slack. The default SegNet slack is zero,
    preserving strict no-worse behavior unless a smoke opts into score-primary
    pose-hard exploration.
    """

    if n_pairs < 1:
        raise SnervScorerLoopDecoderQatError("n_pairs must be >= 1")
    if max_trials < 0:
        raise SnervScorerLoopDecoderQatError("max_trials must be >= 0")
    if perturb_scale < 0:
        raise SnervScorerLoopDecoderQatError("perturb_scale must be >= 0")
    if byte_pressure_multiplier < 1.0:
        raise SnervScorerLoopDecoderQatError(
            "byte_pressure_multiplier must be >= 1.0"
        )
    if max_archive_byte_growth is not None and max_archive_byte_growth < 0:
        raise SnervScorerLoopDecoderQatError(
            "max_archive_byte_growth must be >= 0 when provided"
        )
    if pose_slack < 0:
        raise SnervScorerLoopDecoderQatError("pose_slack must be >= 0")
    if seg_slack < 0:
        raise SnervScorerLoopDecoderQatError("seg_slack must be >= 0")
    if search_mode not in {
        "random_signed",
        "top_weight_coordinate",
        "learned_random_subspace",
        "nes_pair_robust",
    }:
        raise SnervScorerLoopDecoderQatError(
            "search_mode must be 'random_signed', 'top_weight_coordinate', "
            "'learned_random_subspace', or 'nes_pair_robust'"
        )
    if not 0.0 <= pair_guard_min_score_improved_fraction <= 1.0:
        raise SnervScorerLoopDecoderQatError(
            "pair_guard_min_score_improved_fraction must be in [0, 1]"
        )
    if not 0.0 <= pair_guard_max_pose_worsened_fraction <= 1.0:
        raise SnervScorerLoopDecoderQatError(
            "pair_guard_max_pose_worsened_fraction must be in [0, 1]"
        )

    t0 = time.perf_counter()
    posenet, segnet = load_score_exact_scorers(upstream_dir=upstream_dir, device=device)
    prepared = _prepare_state(
        posenet=posenet,
        segnet=segnet,
        n_pairs=n_pairs,
        levels=levels,
        wavelet=wavelet,
        target_bits_per_coeff=target_bits_per_coeff,
        pair_stride=pair_stride,
        start_pair=start_pair,
        video_path=video_path,
        device=device,
        step_map_bins=step_map_bins,
    )

    baseline = _evaluate_decoder(
        prepared.baseline_decoder,
        prepared=prepared,
        posenet=posenet,
        segnet=segnet,
        qat_bits=qat_bits,
        label="least_squares_qat_baseline",
        iteration=0,
        accepted=True,
        byte_pressure_multiplier=byte_pressure_multiplier,
    )
    best_decoder = prepared.baseline_decoder
    best_eval = baseline
    rows: list[SnervDecoderEval] = [baseline]
    rng = np.random.default_rng(seed)
    base_vec, layout = _decoder_to_vector(prepared.baseline_decoder)
    if base_vec.size == 0:
        raise SnervScorerLoopDecoderQatError("decoder has no weights")
    scale = _perturbation_scale(base_vec, perturb_scale)

    directions = _decoder_search_directions(
        base_vec,
        max_trials=max_trials,
        search_mode=search_mode,
        rng=rng,
    )
    if search_mode == "nes_pair_robust":
        current_vec = _decoder_to_vector(best_decoder)[0]
        gradient = np.zeros_like(current_vec)
        for trial, (direction_label, direction) in enumerate(directions, start=1):
            plus_row = _evaluate_decoder(
                _vector_to_decoder(current_vec + scale * direction, layout),
                prepared=prepared,
                posenet=posenet,
                segnet=segnet,
                qat_bits=qat_bits,
                label=f"{direction_label}_plus_probe",
                iteration=trial,
                accepted=False,
                byte_pressure_multiplier=byte_pressure_multiplier,
            )
            minus_row = _evaluate_decoder(
                _vector_to_decoder(current_vec - scale * direction, layout),
                prepared=prepared,
                posenet=posenet,
                segnet=segnet,
                qat_bits=qat_bits,
                label=f"{direction_label}_minus_probe",
                iteration=trial,
                accepted=False,
                byte_pressure_multiplier=byte_pressure_multiplier,
            )
            plus_objective = _nes_pair_robust_objective(
                plus_row,
                best_eval,
                pose_slack=pose_slack,
                seg_slack=seg_slack,
                byte_pressure_multiplier=byte_pressure_multiplier,
                max_archive_byte_growth=max_archive_byte_growth,
                pair_guard_min_score_improved_fraction=(
                    pair_guard_min_score_improved_fraction
                ),
                pair_guard_max_pose_worsened_fraction=(
                    pair_guard_max_pose_worsened_fraction
                ),
            )
            minus_objective = _nes_pair_robust_objective(
                minus_row,
                best_eval,
                pose_slack=pose_slack,
                seg_slack=seg_slack,
                byte_pressure_multiplier=byte_pressure_multiplier,
                max_archive_byte_growth=max_archive_byte_growth,
                pair_guard_min_score_improved_fraction=(
                    pair_guard_min_score_improved_fraction
                ),
                pair_guard_max_pose_worsened_fraction=(
                    pair_guard_max_pose_worsened_fraction
                ),
            )
            gradient += (plus_objective - minus_objective) * direction
            for row in (plus_row, minus_row):
                rows.append(
                    _replace_eval_acceptance(
                        row,
                        accepted=False,
                        blockers=tuple(
                            dict.fromkeys(
                                (
                                    "nes_probe_only_not_candidate",
                                    *_trial_blockers(
                                        row,
                                        best_eval,
                                        pose_slack=pose_slack,
                                        seg_slack=seg_slack,
                                        byte_pressure_multiplier=(
                                            byte_pressure_multiplier
                                        ),
                                        max_archive_byte_growth=(
                                            max_archive_byte_growth
                                        ),
                                        pair_guard_min_score_improved_fraction=(
                                            pair_guard_min_score_improved_fraction
                                        ),
                                        pair_guard_max_pose_worsened_fraction=(
                                            pair_guard_max_pose_worsened_fraction
                                        ),
                                    ),
                                )
                            )
                        ),
                    )
                )
        gradient_rms = float(np.sqrt(np.mean(gradient * gradient)))
        if gradient_rms > 0.0:
            update_direction = -gradient / gradient_rms
            candidate = _vector_to_decoder(current_vec + scale * update_direction, layout)
            row = _evaluate_decoder(
                candidate,
                prepared=prepared,
                posenet=posenet,
                segnet=segnet,
                qat_bits=qat_bits,
                label="nes_pair_robust_update",
                iteration=max_trials + 1,
                accepted=False,
                byte_pressure_multiplier=byte_pressure_multiplier,
            )
            accepted = decoder_trial_passes_pose_guard(
                row,
                best_eval,
                pose_slack=pose_slack,
                seg_slack=seg_slack,
                byte_pressure_multiplier=byte_pressure_multiplier,
                max_archive_byte_growth=max_archive_byte_growth,
                pair_guard_min_score_improved_fraction=(
                    pair_guard_min_score_improved_fraction
                ),
                pair_guard_max_pose_worsened_fraction=(
                    pair_guard_max_pose_worsened_fraction
                ),
            )
            if accepted:
                row = _replace_eval_acceptance(row, accepted=True, blockers=())
                best_decoder = candidate
                best_eval = row
            else:
                row = _replace_eval_acceptance(
                    row,
                    accepted=False,
                    blockers=_trial_blockers(
                        row,
                        best_eval,
                        pose_slack=pose_slack,
                        seg_slack=seg_slack,
                        byte_pressure_multiplier=byte_pressure_multiplier,
                        max_archive_byte_growth=max_archive_byte_growth,
                        pair_guard_min_score_improved_fraction=(
                            pair_guard_min_score_improved_fraction
                        ),
                        pair_guard_max_pose_worsened_fraction=(
                            pair_guard_max_pose_worsened_fraction
                        ),
                    ),
                )
            rows.append(row)
    else:
        for trial, (direction_label, direction) in enumerate(directions, start=1):
            for sign in (1.0, -1.0):
                candidate_vec = (
                    _decoder_to_vector(best_decoder)[0] + sign * scale * direction
                )
                candidate = _vector_to_decoder(candidate_vec, layout)
                row = _evaluate_decoder(
                    candidate,
                    prepared=prepared,
                    posenet=posenet,
                    segnet=segnet,
                    qat_bits=qat_bits,
                    label=f"{direction_label}_{'plus' if sign > 0 else 'minus'}",
                    iteration=trial,
                    accepted=False,
                    byte_pressure_multiplier=byte_pressure_multiplier,
                )
                accepted = decoder_trial_passes_pose_guard(
                    row,
                    best_eval,
                    pose_slack=pose_slack,
                    seg_slack=seg_slack,
                    byte_pressure_multiplier=byte_pressure_multiplier,
                    max_archive_byte_growth=max_archive_byte_growth,
                    pair_guard_min_score_improved_fraction=(
                        pair_guard_min_score_improved_fraction
                    ),
                    pair_guard_max_pose_worsened_fraction=(
                        pair_guard_max_pose_worsened_fraction
                    ),
                )
                if accepted:
                    row = _replace_eval_acceptance(row, accepted=True, blockers=())
                    best_decoder = candidate
                    best_eval = row
                else:
                    row = _replace_eval_acceptance(
                        row,
                        accepted=False,
                        blockers=_trial_blockers(
                            row,
                            best_eval,
                            pose_slack=pose_slack,
                            seg_slack=seg_slack,
                            byte_pressure_multiplier=byte_pressure_multiplier,
                            max_archive_byte_growth=max_archive_byte_growth,
                            pair_guard_min_score_improved_fraction=(
                                pair_guard_min_score_improved_fraction
                            ),
                            pair_guard_max_pose_worsened_fraction=(
                                pair_guard_max_pose_worsened_fraction
                            ),
                        ),
                    )
                rows.append(row)

    accepted_improvement = best_eval.label != baseline.label
    blockers = []
    if not accepted_improvement:
        blockers.append("no_quantized_decoder_trial_improved_score_under_pose_guard")
    if not all(row.receiver_archive_replay_verified for row in rows):
        blockers.append("receiver_archive_replay_failed_for_some_trials")
    blockers.extend(
        [
            "local_smoke_only_not_full_600_pairs",
            "paired_contest_cpu_cuda_pass_missing",
            "mixed_precision_decoder_payload_grammar_not_byte_optimized",
        ]
    )
    ready_for_gate = bool(accepted_improvement and best_eval.receiver_archive_replay_verified)
    return SnervScorerLoopDecoderQatSmokeResult(
        schema=SCHEMA,
        axis_tag=AXIS_TAG,
        n_pairs=int(n_pairs),
        levels=int(levels),
        wavelet=wavelet,
        target_bits_per_coeff=float(target_bits_per_coeff),
        qat_bits=int(qat_bits),
        max_trials=int(max_trials),
        search_mode=search_mode,
        perturb_scale=float(perturb_scale),
        byte_pressure_multiplier=float(byte_pressure_multiplier),
        max_archive_byte_growth=(
            None
            if max_archive_byte_growth is None
            else int(max_archive_byte_growth)
        ),
        pose_slack=float(pose_slack),
        seg_slack=float(seg_slack),
        pair_guard_min_score_improved_fraction=float(
            pair_guard_min_score_improved_fraction
        ),
        pair_guard_max_pose_worsened_fraction=float(
            pair_guard_max_pose_worsened_fraction
        ),
        baseline=baseline,
        best=best_eval,
        evaluations=tuple(rows),
        accepted_improvement=accepted_improvement,
        improvement_score_delta=float(best_eval.score_linf - baseline.score_linf),
        improvement_d_pose_delta=float(best_eval.d_pose_linf - baseline.d_pose_linf),
        improvement_d_seg_delta=float(best_eval.d_seg_linf - baseline.d_seg_linf),
        best_pair_deltas=decoder_eval_pair_deltas(baseline, best_eval),
        scorer_loop_evaluations=len(rows),
        real_frame_source=video_path,
        receiver_contract_satisfied=all(row.receiver_archive_replay_verified for row in rows),
        score_claim=False,
        promotion_eligible=False,
        rank_or_kill_eligible=False,
        ready_for_exact_eval_dispatch=False,
        ready_for_pose_guard_gate=ready_for_gate,
        blockers=tuple(dict.fromkeys(blockers)),
        forbidden_next_actions=(
            "claim_score_from_this_smoke",
            "dispatch_exact_eval_from_this_smoke",
            "promote_without_full600_receiver_proof",
        ),
        wall_seconds=float(time.perf_counter() - t0),
    )


def quantize_decoder_for_qat(
    decoder: HfGenerationDecoder,
    *,
    bits: int = 8,
) -> tuple[HfGenerationDecoder, QuantizedDecoderStats]:
    """Fake-quantize decoder weights and return receiver-encodable weights."""

    if bits < 2 or bits > 16:
        raise SnervScorerLoopDecoderQatError("bits must be between 2 and 16")
    qmax = (1 << (bits - 1)) - 1
    kernels: dict[int, dict[str, np.ndarray]] = {}
    errors: list[np.ndarray] = []
    zero_scales = 0
    total_weights = 0
    zero_weights = 0
    for lvl in range(int(decoder.levels)):
        kernels[lvl] = {}
        for subband, kernel in decoder.kernels[lvl].items():
            arr = np.asarray(kernel, dtype=np.float64)
            max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
            if max_abs == 0.0:
                scale = 1.0
                zero_scales += 1
            else:
                scale = max_abs / float(qmax)
            q = np.clip(np.round(arr / scale), -qmax, qmax).astype(np.int64)
            deq = q.astype(np.float64) * scale
            kernels[lvl][subband] = deq.reshape(arr.shape)
            errors.append(np.abs(deq - arr).reshape(-1))
            total_weights += int(arr.size)
            zero_weights += int(np.count_nonzero(q == 0))
    quantized = HfGenerationDecoder(kernels=kernels, levels=int(decoder.levels))
    payload = encode_decoder_payload(quantized)
    err = np.concatenate(errors) if errors else np.zeros(1, dtype=np.float64)
    return quantized, QuantizedDecoderStats(
        bits=int(bits),
        scale_count=int(decoder.levels) * 3,
        zero_scale_count=int(zero_scales),
        total_weights=int(total_weights),
        zero_weight_fraction=float(zero_weights / max(total_weights, 1)),
        max_abs_error=float(np.max(err)),
        mean_abs_error=float(np.mean(err)),
        payload_bytes_fp32_receiver=len(payload),
        payload_sha256_fp32_receiver=_sha256(payload),
    )


def decoder_trial_passes_pose_guard(
    candidate: SnervDecoderEval,
    current_best: SnervDecoderEval,
    *,
    pose_slack: float = 0.0,
    seg_slack: float = 0.0,
    byte_pressure_multiplier: float = 1.0,
    max_archive_byte_growth: int | None = None,
    pair_guard_min_score_improved_fraction: float = 0.0,
    pair_guard_max_pose_worsened_fraction: float = 1.0,
) -> bool:
    """Return whether a local decoder trial may replace the current best."""

    if pose_slack < 0:
        raise SnervScorerLoopDecoderQatError("pose_slack must be >= 0")
    if seg_slack < 0:
        raise SnervScorerLoopDecoderQatError("seg_slack must be >= 0")
    if byte_pressure_multiplier < 1.0:
        raise SnervScorerLoopDecoderQatError(
            "byte_pressure_multiplier must be >= 1.0"
        )
    if max_archive_byte_growth is not None and max_archive_byte_growth < 0:
        raise SnervScorerLoopDecoderQatError(
            "max_archive_byte_growth must be >= 0 when provided"
        )
    pair_guard_blockers = _pair_guard_blockers(
        candidate,
        current_best,
        pose_slack=pose_slack,
        min_score_improved_fraction=pair_guard_min_score_improved_fraction,
        max_pose_worsened_fraction=pair_guard_max_pose_worsened_fraction,
    )
    return bool(
        candidate.receiver_archive_replay_verified
        and candidate.d_pose_linf <= current_best.d_pose_linf + float(pose_slack)
        and candidate.d_seg_linf <= current_best.d_seg_linf + float(seg_slack)
        and candidate.score_linf < current_best.score_linf
        and _rate_aware_eval_objective(
            candidate,
            byte_pressure_multiplier=byte_pressure_multiplier,
        )
        < _rate_aware_eval_objective(
            current_best,
            byte_pressure_multiplier=byte_pressure_multiplier,
        )
        and (
            max_archive_byte_growth is None
            or candidate.archive_bytes
            <= current_best.archive_bytes + int(max_archive_byte_growth)
        )
        and not pair_guard_blockers
    )


def decoder_eval_pair_deltas(
    baseline: SnervDecoderEval,
    candidate: SnervDecoderEval,
) -> tuple[SnervPairDelta, ...]:
    """Return pair-local detector deltas for a candidate vs. baseline eval."""

    baseline_by_pair = {row.pair_index: row for row in baseline.per_pair}
    deltas = []
    for row in candidate.per_pair:
        base = baseline_by_pair.get(row.pair_index)
        if base is None:
            raise SnervScorerLoopDecoderQatError(
                f"candidate pair {row.pair_index} missing from baseline"
            )
        deltas.append(
            SnervPairDelta(
                pair_index=int(row.pair_index),
                d_seg_linf_delta=float(row.d_seg_linf - base.d_seg_linf),
                d_pose_linf_delta=float(row.d_pose_linf - base.d_pose_linf),
                score_linf_without_rate_delta=float(
                    row.score_linf_without_rate - base.score_linf_without_rate
                ),
            )
        )
    return tuple(deltas)


def decoder_search_direction_labels(
    vector: np.ndarray,
    *,
    max_trials: int,
    search_mode: str,
    seed: int = 0,
) -> tuple[str, ...]:
    """Return deterministic labels for bounded decoder perturbation directions."""

    rng = np.random.default_rng(seed)
    return tuple(
        label
        for label, _direction in _decoder_search_directions(
            np.asarray(vector, dtype=np.float64).reshape(-1),
            max_trials=max_trials,
            search_mode=search_mode,
            rng=rng,
        )
    )


def _trial_blockers(
    candidate: SnervDecoderEval,
    current_best: SnervDecoderEval,
    *,
    pose_slack: float,
    seg_slack: float,
    byte_pressure_multiplier: float = 1.0,
    max_archive_byte_growth: int | None = None,
    pair_guard_min_score_improved_fraction: float = 0.0,
    pair_guard_max_pose_worsened_fraction: float = 1.0,
) -> tuple[str, ...]:
    blockers = list(candidate.blockers)
    if not candidate.receiver_archive_replay_verified:
        blockers.append("receiver_archive_replay_failed")
    if candidate.d_pose_linf > current_best.d_pose_linf + float(pose_slack):
        blockers.append("pose_guard_failed")
    if candidate.d_seg_linf > current_best.d_seg_linf + float(seg_slack):
        blockers.append("seg_gate_failed")
    if candidate.score_linf >= current_best.score_linf:
        blockers.append("score_gate_failed")
    if _rate_aware_eval_objective(
        candidate,
        byte_pressure_multiplier=byte_pressure_multiplier,
    ) >= _rate_aware_eval_objective(
        current_best,
        byte_pressure_multiplier=byte_pressure_multiplier,
    ):
        blockers.append("rate_aware_score_gate_failed")
    if (
        max_archive_byte_growth is not None
        and candidate.archive_bytes
        > current_best.archive_bytes + int(max_archive_byte_growth)
    ):
        blockers.append("byte_growth_guard_failed")
    blockers.extend(
        _pair_guard_blockers(
            candidate,
            current_best,
            pose_slack=pose_slack,
            min_score_improved_fraction=pair_guard_min_score_improved_fraction,
            max_pose_worsened_fraction=pair_guard_max_pose_worsened_fraction,
        )
    )
    return tuple(dict.fromkeys(blockers))


def _pair_guard_blockers(
    candidate: SnervDecoderEval,
    current_best: SnervDecoderEval,
    *,
    pose_slack: float,
    min_score_improved_fraction: float,
    max_pose_worsened_fraction: float,
) -> tuple[str, ...]:
    if not 0.0 <= min_score_improved_fraction <= 1.0:
        raise SnervScorerLoopDecoderQatError(
            "pair_guard_min_score_improved_fraction must be in [0, 1]"
        )
    if not 0.0 <= max_pose_worsened_fraction <= 1.0:
        raise SnervScorerLoopDecoderQatError(
            "pair_guard_max_pose_worsened_fraction must be in [0, 1]"
        )
    if min_score_improved_fraction <= 0.0 and max_pose_worsened_fraction >= 1.0:
        return ()
    deltas = decoder_eval_pair_deltas(current_best, candidate)
    if not deltas:
        return ("pair_guard_no_pair_deltas",)
    n_pairs = float(len(deltas))
    score_improved_fraction = (
        sum(1 for row in deltas if row.score_linf_without_rate_delta < 0.0)
        / n_pairs
    )
    pose_worsened_fraction = (
        sum(1 for row in deltas if row.d_pose_linf_delta > float(pose_slack))
        / n_pairs
    )
    blockers = []
    if score_improved_fraction < float(min_score_improved_fraction):
        blockers.append("pair_score_improvement_fraction_guard_failed")
    if pose_worsened_fraction > float(max_pose_worsened_fraction):
        blockers.append("pair_pose_worsening_fraction_guard_failed")
    return tuple(blockers)


def _nes_pair_robust_objective(
    candidate: SnervDecoderEval,
    current_best: SnervDecoderEval,
    *,
    pose_slack: float,
    seg_slack: float,
    byte_pressure_multiplier: float = 1.0,
    max_archive_byte_growth: int | None = None,
    pair_guard_min_score_improved_fraction: float = 0.0,
    pair_guard_max_pose_worsened_fraction: float = 1.0,
) -> float:
    """Return a lower-is-better local objective for NES probe ranking.

    The hard promotion authority remains the regular pose gate. This objective
    only chooses the synthesized NES update direction and therefore deliberately
    over-penalizes pair-local luck and PoseNet regressions.
    """

    score = _rate_aware_eval_objective(
        candidate,
        byte_pressure_multiplier=byte_pressure_multiplier,
    )
    objective = score if np.isfinite(score) else 1.0e12
    penalty = 0.0
    if not candidate.receiver_archive_replay_verified:
        penalty += 1.0e9
    if (
        max_archive_byte_growth is not None
        and candidate.archive_bytes
        > current_best.archive_bytes + int(max_archive_byte_growth)
    ):
        penalty += 1.0e6 * float(
            candidate.archive_bytes
            - current_best.archive_bytes
            - int(max_archive_byte_growth)
        )
    penalty += 1.0e6 * max(
        0.0,
        float(candidate.d_pose_linf)
        - float(current_best.d_pose_linf)
        - float(pose_slack),
    )
    penalty += 1.0e6 * max(
        0.0,
        float(candidate.d_seg_linf) - float(current_best.d_seg_linf) - float(seg_slack),
    )

    try:
        deltas = decoder_eval_pair_deltas(current_best, candidate)
    except SnervScorerLoopDecoderQatError:
        return float(objective + penalty + 1.0e9)
    if not deltas:
        return float(objective + penalty + 1.0e6)

    n_pairs = float(len(deltas))
    score_deltas = np.asarray(
        [row.score_linf_without_rate_delta for row in deltas],
        dtype=np.float64,
    )
    pose_deltas = np.asarray(
        [row.d_pose_linf_delta for row in deltas],
        dtype=np.float64,
    )
    seg_deltas = np.asarray(
        [row.d_seg_linf_delta for row in deltas],
        dtype=np.float64,
    )
    score_improved_fraction = float(np.count_nonzero(score_deltas < 0.0) / n_pairs)
    pose_worsened_fraction = float(
        np.count_nonzero(pose_deltas > float(pose_slack)) / n_pairs
    )
    penalty += 1.0e5 * max(
        0.0,
        float(pair_guard_min_score_improved_fraction) - score_improved_fraction,
    )
    penalty += 1.0e5 * max(
        0.0,
        pose_worsened_fraction - float(pair_guard_max_pose_worsened_fraction),
    )
    penalty += 1.0e4 * float(
        np.mean(np.maximum(0.0, pose_deltas - float(pose_slack)))
    )
    penalty += 1.0e4 * float(np.mean(np.maximum(0.0, seg_deltas)))
    penalty += 1.0e3 * float(np.mean(np.maximum(0.0, score_deltas)))

    return float(
        objective
        + penalty
        + 0.25 * float(np.mean(score_deltas))
        + 0.25 * float(np.max(score_deltas))
    )


def _rate_aware_eval_objective(
    row: SnervDecoderEval,
    *,
    byte_pressure_multiplier: float,
) -> float:
    if byte_pressure_multiplier < 1.0:
        raise SnervScorerLoopDecoderQatError(
            "byte_pressure_multiplier must be >= 1.0"
        )
    return float(row.score_linf) + (
        float(byte_pressure_multiplier) - 1.0
    ) * float(row.rate_term)


def _decoder_search_directions(
    vector: np.ndarray,
    *,
    max_trials: int,
    search_mode: str,
    rng: np.random.Generator,
) -> tuple[tuple[str, np.ndarray], ...]:
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    if max_trials < 0:
        raise SnervScorerLoopDecoderQatError("max_trials must be >= 0")
    if vec.size == 0:
        return ()
    if search_mode == "random_signed":
        signs = np.array([-1.0, 1.0], dtype=np.float64)
        return tuple(
            (f"random_{idx}", rng.choice(signs, size=vec.shape))
            for idx in range(1, int(max_trials) + 1)
        )
    if search_mode == "learned_random_subspace":
        rows = []
        for idx in range(1, int(max_trials) + 1):
            raw = rng.normal(0.0, 1.0, size=vec.shape)
            mixed = raw + 0.35 * np.roll(raw, 1) - 0.15 * np.roll(raw, 2)
            direction = np.tanh(mixed)
            rms = float(np.sqrt(np.mean(direction * direction)))
            if rms > 0.0:
                direction = direction / rms
            rows.append((f"learned_subspace_{idx:03d}", direction))
        return tuple(rows)
    if search_mode == "nes_pair_robust":
        rows = []
        for idx in range(1, int(max_trials) + 1):
            raw = rng.normal(0.0, 1.0, size=vec.shape)
            mixed = raw + 0.20 * np.roll(raw, 1) - 0.10 * np.roll(raw, 3)
            direction = mixed
            rms = float(np.sqrt(np.mean(direction * direction)))
            if rms > 0.0:
                direction = direction / rms
            rows.append((f"nes_probe_{idx:03d}", direction))
        return tuple(rows)
    if search_mode == "top_weight_coordinate":
        order = np.argsort(-np.abs(vec), kind="stable")[: int(max_trials)]
        rows = []
        for idx in order:
            direction = np.zeros_like(vec)
            direction[int(idx)] = 1.0
            rows.append((f"coord_{int(idx):03d}", direction))
        return tuple(rows)
    raise SnervScorerLoopDecoderQatError(
        "search_mode must be 'random_signed', 'top_weight_coordinate', "
        "'learned_random_subspace', or 'nes_pair_robust'"
    )


def _prepare_state(
    *,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    n_pairs: int,
    levels: int,
    wavelet: str,
    target_bits_per_coeff: float,
    pair_stride: int,
    start_pair: int,
    video_path: str,
    device: str,
    step_map_bins: int,
) -> _PreparedState:
    pairs = decode_real_pairs(
        video_path,
        n_pairs,
        pair_stride=pair_stride,
        start_pair=start_pair,
        device=device,
    )
    h, w = int(pairs.shape[-2]), int(pairs.shape[-1])
    train_pyrs = []
    records: list[tuple[int, int, int, Any]] = []
    for pair_idx in range(n_pairs):
        for frame_idx in range(2):
            frame = pairs[pair_idx, frame_idx].detach().cpu().numpy()
            for channel_idx in range(3):
                pyr = encode_frame_lf(
                    frame[channel_idx],
                    levels=levels,
                    wavelet=wavelet,
                )
                train_pyrs.append(pyr)
                records.append((pair_idx, frame_idx, channel_idx, pyr))
    decoder = fit_hf_decoder_least_squares(train_pyrs, levels=levels)

    raw_step_maps: list[np.ndarray] = []
    for pair_idx in range(n_pairs):
        seg = compute_s_seg_flip_risk(segnet, pairs[pair_idx])
        pose = compute_s_pose_fisher(posenet, pairs[pair_idx])
        seg_hw = seg.flip_risk.detach().cpu().numpy()
        pose_hw = pose.s_pose.detach().cpu().numpy()
        for rec_pair, _frame_idx, _channel_idx, pyr in records:
            if rec_pair != pair_idx:
                continue
            target_bits = float(pyr.lf.size) * float(target_bits_per_coeff)
            lf_saliency = push_pixel_saliency_to_lf(
                seg_hw,
                pose_hw,
                carrier_hw=(h, w),
                levels=levels,
                wavelet=wavelet,
            )
            alloc = allocate_lf_linf(lf_saliency, target_bits=target_bits, min_step=0.5)
            raw_step_maps.append(alloc.steps.reshape(pyr.lf.shape))

    step_packet = encode_step_maps(raw_step_maps, bins=step_map_bins)
    receiver_step_maps = decode_step_maps(step_packet.packet)
    if len(receiver_step_maps) != len(records):
        raise SnervScorerLoopDecoderQatError("decoded step-map count mismatch")

    code_records: list[_CodeRecord] = []
    q_planes: list[np.ndarray] = []
    zeros: list[float] = []
    for (pair_idx, frame_idx, channel_idx, pyr), steps in zip(
        records,
        receiver_step_maps,
        strict=True,
    ):
        q, scale, zero = quantize_lf(pyr.lf, per_element_steps=steps)
        code_records.append(
            _CodeRecord(
                pair_index=pair_idx,
                frame_index=frame_idx,
                channel_index=channel_idx,
                code=SnervFrameCode(
                    lf_quant=q,
                    lf_scale=scale,
                    lf_zero=zero,
                    lf_shape=tuple(int(v) for v in q.shape),
                    levels=levels,
                    wavelet=wavelet,
                    orig_hw=(h, w),
                    per_element_steps=steps,
                ),
            )
        )
        q_planes.append(q)
        zeros.append(float(zero))

    return _PreparedState(
        pairs=pairs,
        codes=tuple(code_records),
        lf_quant_planes=tuple(q_planes),
        lf_zero_points=tuple(zeros),
        step_maps=tuple(receiver_step_maps),
        step_map_packet=step_packet.packet,
        baseline_decoder=decoder,
        levels=int(levels),
        wavelet=wavelet,
        orig_hw=(h, w),
    )


def _evaluate_decoder(
    decoder: HfGenerationDecoder,
    *,
    prepared: _PreparedState,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    qat_bits: int,
    label: str,
    iteration: int,
    accepted: bool,
    byte_pressure_multiplier: float,
) -> SnervDecoderEval:
    quantized, qstats = quantize_decoder_for_qat(decoder, bits=qat_bits)
    archive = _pack_receiver_archive(prepared, quantized)
    try:
        receiver_np = decode_snerv_archive_frames(archive.packet)
        replay_ok = receiver_np.shape == tuple(prepared.pairs.shape)
    except Exception:
        receiver_np = np.zeros(tuple(prepared.pairs.shape), dtype=np.float32)
        replay_ok = False
    receiver = torch.from_numpy(receiver_np).to(prepared.pairs)
    dsegs = []
    dposes = []
    per_pair = []
    for pair_idx in range(int(prepared.pairs.shape[0])):
        ds, dp = measure_pair_d_seg_d_pose(
            posenet,
            segnet,
            prepared.pairs[pair_idx : pair_idx + 1],
            receiver[pair_idx : pair_idx + 1],
        )
        d_seg_pair = float(ds)
        d_pose_pair = float(dp)
        dsegs.append(d_seg_pair)
        dposes.append(d_pose_pair)
        per_pair.append(
            SnervPairEval(
                pair_index=int(pair_idx),
                d_seg_linf=d_seg_pair,
                d_pose_linf=d_pose_pair,
                score_linf_without_rate=(
                    100.0 * d_seg_pair
                    + float(np.sqrt(10.0 * max(d_pose_pair, 0.0)))
                ),
            )
        )
    d_seg = float(np.mean(dsegs))
    d_pose = float(np.mean(dposes))
    rate = CONTEST_BYTE_PRICE * archive.total_bytes
    score = 100.0 * d_seg + float(np.sqrt(10.0 * max(d_pose, 0.0))) + rate
    blockers = () if replay_ok else ("receiver_archive_replay_failed",)
    rate_aware_objective = score + (
        float(byte_pressure_multiplier) - 1.0
    ) * float(rate)
    return SnervDecoderEval(
        label=label,
        iteration=int(iteration),
        archive_bytes=archive.total_bytes,
        archive_sha256=_sha256(archive.packet),
        d_seg_linf=d_seg,
        d_pose_linf=d_pose,
        score_linf=score,
        rate_aware_objective_linf=float(rate_aware_objective),
        rate_term=rate,
        receiver_archive_replay_verified=bool(replay_ok),
        accepted=bool(accepted),
        blockers=blockers,
        quantized_decoder=qstats,
        per_pair=tuple(per_pair),
    )


def _pack_receiver_archive(
    prepared: _PreparedState,
    decoder: HfGenerationDecoder,
) -> SnervArchivePacket:
    return pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(
            lf_zero_points=list(prepared.lf_zero_points),
        ),
        lf_payload=encode_lf_quant_payload(list(prepared.lf_quant_planes)),
        decoder_payload=encode_decoder_payload(decoder),
        step_map_packet=prepared.step_map_packet,
        metadata={
            "n_pairs": int(prepared.pairs.shape[0]),
            "frames_per_pair": 2,
            "channels": 3,
            "levels": prepared.levels,
            "wavelet": prepared.wavelet,
            "carrier_hw": list(prepared.orig_hw),
            "orig_hw": list(prepared.orig_hw),
            "lf_plane_count": len(prepared.lf_quant_planes),
            "hf_decoder_fit_mode": "scorer_loop_decoder_qat_smoke",
        },
    )


def _decoder_to_vector(
    decoder: HfGenerationDecoder,
) -> tuple[np.ndarray, tuple[tuple[int, str, tuple[int, ...]], ...]]:
    values = []
    layout = []
    for lvl in range(int(decoder.levels)):
        for subband in ("LH", "HL", "HH"):
            arr = np.asarray(decoder.kernels[lvl][subband], dtype=np.float64)
            values.append(arr.reshape(-1))
            layout.append((lvl, subband, tuple(arr.shape)))
    vec = np.concatenate(values) if values else np.zeros(0, dtype=np.float64)
    return vec, tuple(layout)


def _vector_to_decoder(
    vector: np.ndarray,
    layout: tuple[tuple[int, str, tuple[int, ...]], ...],
) -> HfGenerationDecoder:
    cursor = 0
    kernels: dict[int, dict[str, np.ndarray]] = {}
    for lvl, subband, shape in layout:
        count = int(np.prod(shape))
        kernels.setdefault(lvl, {})[subband] = np.asarray(
            vector[cursor : cursor + count],
            dtype=np.float64,
        ).reshape(shape)
        cursor += count
    if cursor != int(vector.size):
        raise SnervScorerLoopDecoderQatError("decoder vector/layout size mismatch")
    return HfGenerationDecoder(kernels=kernels, levels=len(kernels))


def _replace_eval_acceptance(
    row: SnervDecoderEval,
    *,
    accepted: bool,
    blockers: tuple[str, ...],
) -> SnervDecoderEval:
    return SnervDecoderEval(
        label=row.label,
        iteration=row.iteration,
        archive_bytes=row.archive_bytes,
        archive_sha256=row.archive_sha256,
        d_seg_linf=row.d_seg_linf,
        d_pose_linf=row.d_pose_linf,
        score_linf=row.score_linf,
        rate_aware_objective_linf=row.rate_aware_objective_linf,
        rate_term=row.rate_term,
        receiver_archive_replay_verified=row.receiver_archive_replay_verified,
        accepted=accepted,
        blockers=blockers,
        quantized_decoder=row.quantized_decoder,
        per_pair=row.per_pair,
    )


def _eval_as_gate_row(row: SnervDecoderEval) -> dict[str, Any]:
    if row.label == "least_squares_qat_baseline":
        sweep_label = "least_squares_baseline_existing"
        fit_mode = "least_squares"
    else:
        sweep_label = row.label
        fit_mode = "scorer_loop_qat"
    return {
        "sweep_label": sweep_label,
        "hf_decoder_fit_mode": fit_mode,
        "archive_bytes_total": row.archive_bytes,
        "receiver_archive_sha256": row.archive_sha256,
        "receiver_archive_replay_verified": row.receiver_archive_replay_verified,
        "d_seg_mean_linf": row.d_seg_linf,
        "d_pose_mean_linf": row.d_pose_linf,
        "score_linf": row.score_linf,
        "rate_aware_objective_linf": row.rate_aware_objective_linf,
        "rate_term": row.rate_term,
        "accepted": row.accepted,
        "blockers": list(row.blockers),
        "pair_count": len(row.per_pair),
        "per_pair": [pair.as_jsonable() for pair in row.per_pair],
        "source_artifact": "snerv_scorer_loop_decoder_qat_smoke",
    }


def _perturbation_scale(vector: np.ndarray, relative_scale: float) -> float:
    if relative_scale == 0.0:
        return 0.0
    magnitude = float(np.mean(np.abs(vector)))
    return max(magnitude, 1e-3) * float(relative_scale)


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


__all__ = [
    "AXIS_TAG",
    "SCHEMA",
    "QuantizedDecoderStats",
    "SnervDecoderEval",
    "SnervPairDelta",
    "SnervPairEval",
    "SnervScorerLoopDecoderQatError",
    "SnervScorerLoopDecoderQatSmokeResult",
    "decoder_eval_pair_deltas",
    "decoder_search_direction_labels",
    "decoder_trial_passes_pose_guard",
    "quantize_decoder_for_qat",
    "run_snerv_scorer_loop_decoder_qat_smoke",
]
