"""Canonical cooperative-receiver campaign queue.

The time-traveler, Bell/NSA, and Fields-medalist ledgers all converged on the
same operational frame: the contest scorer is a known cooperative receiver.
This module turns that research signal into a deterministic, planning-only
campaign queue.  It never authorizes dispatch, never claims score movement, and
never promotes a proxy result; rows must pass the proxy false-authority
contract until a byte-closed archive and exact eval artifact exist.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    validate_proxy_candidate,
)

QUEUE_SCHEMA = "tac_cooperative_receiver_campaign_queue_v1"
GENERATED_AT_STABLE = "1970-01-01T00:00:00Z"


@dataclasses.dataclass(frozen=True)
class CooperativeReceiverCandidate:
    """One falsifiable cooperative-receiver campaign row."""

    campaign_id: str
    lineage: str
    source_commit: str
    source_memo: str
    hypothesis: str
    predicted_delta_low: float
    predicted_delta_high: float
    estimated_cost_usd_low: float
    estimated_cost_usd_high: float
    timing_smoke_command: str
    byte_closed_packet_plan: str
    promotion_gate: str
    target_axis: str
    implementation_surface: str
    source_citations: tuple[str, ...]
    rank_hint: int

    def as_row(self) -> dict[str, Any]:
        """Return a proxy-safe JSON row for operator queues."""

        if self.predicted_delta_low > self.predicted_delta_high:
            raise ValueError(f"{self.campaign_id}: predicted_delta_low > high")
        midpoint = (self.predicted_delta_low + self.predicted_delta_high) / 2.0
        cost_mid = max((self.estimated_cost_usd_low + self.estimated_cost_usd_high) / 2.0, 0.10)
        row = {
            "campaign_id": self.campaign_id,
            "candidate_id": self.campaign_id,
            "lane_id": f"lane_{self.campaign_id}",
            "lineage": self.lineage,
            "source_commit": self.source_commit,
            "source_memo": self.source_memo,
            "hypothesis": self.hypothesis,
            "predicted_score_delta": midpoint,
            "predicted_delta_band": [
                self.predicted_delta_low,
                self.predicted_delta_high,
            ],
            "predicted_delta_evidence": "prediction_only_cross_domain_derivation",
            "estimated_dispatch_cost_usd": cost_mid,
            "estimated_cost_usd_band": [
                self.estimated_cost_usd_low,
                self.estimated_cost_usd_high,
            ],
            "ev_per_dollar_proxy": abs(midpoint) / cost_mid,
            "timing_smoke_command": self.timing_smoke_command,
            "byte_closed_packet_plan": self.byte_closed_packet_plan,
            "promotion_gate": self.promotion_gate,
            "target_axis": self.target_axis,
            "implementation_surface": self.implementation_surface,
            "source_citations": list(self.source_citations),
            "rank_hint": self.rank_hint,
            "planning_only": True,
            "proxy_only": True,
            "research_only": False,
            "evidence_semantics": "cooperative_receiver_campaign_planning_only",
            "score_lowering_hypothesis": self.hypothesis,
            "dispatch_blockers": [
                "byte_closed_archive_missing",
                "exact_cuda_auth_eval_missing",
                "contest_cpu_axis_pair_missing",
                "lane_dispatch_claim_missing",
            ],
        }
        return apply_proxy_evidence_boundary(row)


def default_cooperative_receiver_candidates() -> list[CooperativeReceiverCandidate]:
    """Return the hand-curated queue from the convergent 2026-05-13 ledgers."""

    return [
        CooperativeReceiverCandidate(
            campaign_id="darts_confirmed_time_traveler_config",
            lineage="darts_supernet_time_traveler",
            source_commit="local_darts_supernet",
            source_memo=".omx/research/time_traveler_architecture_reverse_engineered_20260513.md",
            hypothesis=(
                "Materialize the DARTS-confirmed time-traveler configuration "
                "(25K params, 30 B/pair, 4x4 foveation, hidden 32, int8) as the "
                "highest-EV follow-on packet candidate."
            ),
            predicted_delta_low=-0.060,
            predicted_delta_high=-0.045,
            estimated_cost_usd_low=3.00,
            estimated_cost_usd_high=8.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python "
                "tools/smoke_time_traveler_l5_autonomy_macos_cpu.py --help"
            ),
            byte_closed_packet_plan=(
                "Lower the DARTS-selected architecture into the TT5L archive grammar, "
                "prove deterministic inflate consumption, then run paired exact eval."
            ),
            promotion_gate=(
                "DARTS config materialized + TT5L archive roundtrip + paired [contest-CUDA]/[contest-CPU] exact eval"
            ),
            target_axis="representation_pose_segmentation_rate",
            implementation_surface="src/tac/composition/darts_supernet.py + src/tac/substrates/time_traveler_l5_autonomy/",
            source_citations=(
                ".omx/research/time_traveler_architecture_reverse_engineered_20260513.md",
            ),
            rank_hint=1,
        ),
        CooperativeReceiverCandidate(
            campaign_id="time_traveler_world_model_substrate",
            lineage="time_traveler_architecture",
            source_commit="1d62a114",
            source_memo=".omx/research/time_traveler_architecture_reverse_engineered_20260513.md",
            hypothesis=(
                "Encode a small ego-motion world model once, then transmit only "
                "per-pair prediction errors under a cooperative-receiver loss."
            ),
            predicted_delta_low=-0.043,
            predicted_delta_high=-0.023,
            estimated_cost_usd_low=3.00,
            estimated_cost_usd_high=8.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python "
                "tools/smoke_time_traveler_l5_autonomy_macos_cpu.py --help"
            ),
            byte_closed_packet_plan=(
                "TT5L monolithic archive: world model + per-pair side info + AC "
                "state + metadata; exact eval only after runtime closure."
            ),
            promotion_gate="TT5L archive roundtrip + macOS advisory smoke + paired exact eval",
            target_axis="representation_pose_segmentation_rate",
            implementation_surface="src/tac/substrates/time_traveler_l5_autonomy/",
            source_citations=(
                ".omx/research/time_traveler_architecture_reverse_engineered_20260513.md",
            ),
            rank_hint=2,
        ),
        CooperativeReceiverCandidate(
            campaign_id="sabor_boundary_only_renderer",
            lineage="council_f_stable_argmax_boundary_only_renderer",
            source_commit="local_sabor_substrate",
            source_memo=".omx/research/sabor_boundary_audit_20260513.md",
            hypothesis=(
                "Exploit SegNet argmax-stable interiors by charging boundary RGB "
                "and reconstructing interiors from class means plus tiny refinement."
            ),
            predicted_delta_low=-0.028,
            predicted_delta_high=-0.008,
            estimated_cost_usd_low=1.00,
            estimated_cost_usd_high=2.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -q "
                "src/tac/substrates/sabor_boundary_only_renderer/tests"
            ),
            byte_closed_packet_plan=(
                "Materialize SBO1 monolithic archive, prove boundary bytes are "
                "consumed by inflate, then run paired exact eval."
            ),
            promotion_gate="SBO1 archive custody + scorer-free inflate + paired exact eval",
            target_axis="segmentation_rate",
            implementation_surface="src/tac/substrates/sabor_boundary_only_renderer/",
            source_citations=(
                ".omx/research/sabor_boundary_audit_20260513.md",
            ),
            rank_hint=3,
        ),
        CooperativeReceiverCandidate(
            campaign_id="s2sbs_hf_byte_stuffing",
            lineage="council_f_stride2_stem_blindspot",
            source_commit="local_s2sbs_substrate",
            source_memo=".omx/research/s2sbs_blindspot_audit_20260513.md",
            hypothesis=(
                "Use high-frequency regions attenuated by the scorer stems as a "
                "charged side-information channel for useful residual bytes."
            ),
            predicted_delta_low=-0.025,
            predicted_delta_high=-0.005,
            estimated_cost_usd_low=1.00,
            estimated_cost_usd_high=3.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -q "
                "src/tac/substrates/s2sbs_byte_stuffing/tests"
            ),
            byte_closed_packet_plan=(
                "Materialize S2S1 monolithic archive, prove HF payload bytes "
                "survive decode and reduce scorer distortion net of rate."
            ),
            promotion_gate="S2S1 archive custody + byte-consumption proof + paired exact eval",
            target_axis="side_information_rate",
            implementation_surface="src/tac/substrates/s2sbs_byte_stuffing/",
            source_citations=(
                ".omx/research/s2sbs_blindspot_audit_20260513.md",
            ),
            rank_hint=4,
        ),
        CooperativeReceiverCandidate(
            campaign_id="driving_prior_pretrained_renderer_2032",
            lineage="hardware_physics_2032",
            source_commit="fdfc347f",
            source_memo=".omx/research/expert_team_hardware_physics_future_alien_tech_20260513.md",
            hypothesis=(
                "Pre-train a sub-100K dashcam renderer on public driving video "
                "distributions, then fine-tune only contest residuals using the "
                "fixed scorer as cooperative receiver and penultimate-feature saliency "
                "as the allocation signal."
            ),
            predicted_delta_low=-0.012,
            predicted_delta_high=-0.005,
            estimated_cost_usd_low=5.00,
            estimated_cost_usd_high=30.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python "
                "tools/probe_driving_prior_readiness.py --output "
                "reports/cooperative_receiver/driving_prior_readiness.json"
            ),
            byte_closed_packet_plan=(
                "Train a public-driving prior outside the contest packet, export a "
                "deterministic small renderer/codebook, then charge every consumed "
                "weight/residual byte in a monolithic archive before paired exact eval."
            ),
            promotion_gate=(
                "dataset/license manifest + scorer penultimate-hook readiness + "
                "deterministic renderer export + paired [contest-CUDA]/[contest-CPU] exact eval"
            ),
            target_axis="representation_pretraining_residual",
            implementation_surface="src/tac/analysis/driving_prior_readiness.py + future substrate",
            source_citations=(
                "Comma2k19",
                "BDD100K",
                "Waymo Open Dataset",
                ".omx/research/expert_team_hardware_physics_future_alien_tech_20260513.md",
            ),
            rank_hint=5,
        ),
        CooperativeReceiverCandidate(
            campaign_id="driving_prior_world_model_substrate",
            lineage="hardware_physics_2032",
            source_commit="local_dpw1_substrate",
            source_memo=".omx/research/expert_team_hardware_physics_future_alien_tech_20260513.md",
            hypothesis=(
                "Exercise the DPW1 charged world-model archive grammar now: a small "
                "deterministic driving prior plus residual grid that can later absorb "
                "a trained public-driving prior without changing inflate semantics."
            ),
            predicted_delta_low=-0.012,
            predicted_delta_high=-0.003,
            estimated_cost_usd_low=1.00,
            estimated_cost_usd_high=3.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -q "
                "src/tac/substrates/driving_prior_world_model/tests"
            ),
            byte_closed_packet_plan=(
                "Promote DPW1 from deterministic scaffold to trained prior by replacing "
                "the charged codebook/residual bytes, proving byte consumption, then "
                "running paired exact eval."
            ),
            promotion_gate=(
                "DPW1 byte-consumption proof + trained prior export + paired "
                "[contest-CUDA]/[contest-CPU] exact eval"
            ),
            target_axis="representation_pretraining_residual",
            implementation_surface="src/tac/substrates/driving_prior_world_model/",
            source_citations=(
                ".omx/research/expert_team_hardware_physics_future_alien_tech_20260513.md",
            ),
            rank_hint=6,
        ),
        CooperativeReceiverCandidate(
            campaign_id="h15_coord_mlp_residual_sidecar_pr103_on_pr106",
            lineage="skunkworks_cia_ach",
            source_commit="d1fb9f6a",
            source_memo=".omx/research/expert_team_aerospace_stealth_analytic_ach_matrix_20260513.md",
            hypothesis=(
                "Build a Coord-MLP residual basis as an HNeRV-external PR103-on-PR106 "
                "sidecar: small, reviewable, and directly aimed at the current "
                "frontier without another family-internal retrain."
            ),
            predicted_delta_low=-0.020,
            predicted_delta_high=-0.005,
            estimated_cost_usd_low=1.00,
            estimated_cost_usd_high=3.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python "
                "tools/probe_coord_mlp_residual_sidecar.py --output "
                "reports/cooperative_receiver/coord_mlp_residual_sidecar_probe.json"
            ),
            byte_closed_packet_plan=(
                "Fork the PR106/A1 runtime, add one charged Coord-MLP residual sidecar "
                "section consumed by inflate.py, prove no-op failure impossible, then "
                "run paired exact eval."
            ),
            promotion_gate=(
                "sidecar bytes changed and consumed + full-frame parity baseline + "
                "paired [contest-CUDA]/[contest-CPU] exact eval"
            ),
            target_axis="representation_external_sidecar",
            implementation_surface="src/tac/substrates/coord_mlp_residual_sidecar/",
            source_citations=(
                "Heuer-Pherson ACH method",
                ".omx/research/expert_team_aerospace_stealth_analytic_ach_matrix_20260513.md",
                ".omx/research/expert_team_aerospace_stealth_analytic_pre_mortem_20260513.md",
            ),
            rank_hint=7,
        ),
        CooperativeReceiverCandidate(
            campaign_id="s7_donoho_hnerv_wavelet_threshold",
            lineage="fields_medalist_statistics",
            source_commit="cbc6b48b",
            source_memo=".omx/research/expert_team_statistics_20260513.md",
            hypothesis=(
                "Apply Donoho-Johnstone universal thresholding to HNeRV-family "
                "decoder weights in a wavelet basis before entropy coding."
            ),
            predicted_delta_low=-0.020,
            predicted_delta_high=-0.010,
            estimated_cost_usd_low=0.00,
            estimated_cost_usd_high=0.10,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python "
                "tools/build_cooperative_receiver_campaign_queue.py --output "
                "reports/cooperative_receiver/campaign_queue.json"
            ),
            byte_closed_packet_plan=(
                "Patch the PR95/PR101 decoder export path to emit thresholded "
                "weight sections, prove state_dict consumption parity, then run "
                "paired [contest-CUDA]/[contest-CPU] exact eval."
            ),
            promotion_gate="state_dict_parity + full-frame inflate parity + paired exact eval",
            target_axis="rate_then_pose",
            implementation_surface="src/tac/codec/ + PR95/HNeRV export path",
            source_citations=(
                "Donoho-Johnstone 1994 Biometrika DOI:10.1093/biomet/81.3.425",
                ".omx/research/expert_team_fields_medalist_math_biology_alien_tech_20260513.md",
            ),
            rank_hint=8,
        ),
        CooperativeReceiverCandidate(
            campaign_id="l2_sar_coherent_pose_spectrum",
            lineage="bell_nsa_lincoln_lab",
            source_commit="27cd8b41",
            source_memo=".omx/research/expert_team_signal_processing_lincoln_lab_20260513.md",
            hypothesis=(
                "Pose targets may be temporally smooth enough that sparse FFT/SAR-style "
                "coherent coding beats per-pair independent pose side information."
            ),
            predicted_delta_low=-0.006,
            predicted_delta_high=-0.002,
            estimated_cost_usd_low=0.00,
            estimated_cost_usd_high=1.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python "
                "tools/probe_cooperative_receiver_pose_spectrum.py "
                "--pose-targets experiments/posenet_targets.bin "
                "--output reports/cooperative_receiver/pose_spectrum_l2.json"
            ),
            byte_closed_packet_plan=(
                "If spectrum is concentrated, add a pose-side-info section using "
                "top-K rFFT coefficients and deterministic inverse FFT in inflate."
            ),
            promotion_gate="spectral concentration smoke + pose codec roundtrip + paired exact eval",
            target_axis="pose",
            implementation_surface="src/tac/pose_delta_codec.py or src/tac/codec/pose_filler_stc_codec.py",
            source_citations=(
                "Carrara-Goodman-Majewski 1995 Spotlight Synthetic Aperture Radar",
                ".omx/research/expert_team_signal_processing_lincoln_lab_20260513.md",
            ),
            rank_hint=9,
        ),
        CooperativeReceiverCandidate(
            campaign_id="n4_video_keyed_codebook",
            lineage="bell_nsa_signal_processing",
            source_commit="27cd8b41",
            source_memo=".omx/research/expert_team_signal_processing_nsa_sigint_20260513.md",
            hypothesis=(
                "A deterministic video-derived seed can specialize latent/codebook "
                "indices to this exact video without storing a generic codebook."
            ),
            predicted_delta_low=-0.005,
            predicted_delta_high=-0.001,
            estimated_cost_usd_low=1.00,
            estimated_cost_usd_high=5.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python "
                "tools/build_cooperative_receiver_campaign_queue.py --output "
                "reports/cooperative_receiver/campaign_queue.json"
            ),
            byte_closed_packet_plan=(
                "Derive the codebook from a committed seed and fixed source bytes; "
                "store only indices plus seed metadata, with no network or hidden state."
            ),
            promotion_gate="deterministic codebook regeneration + archive-byte reduction + exact eval",
            target_axis="rate",
            implementation_surface="src/tac/codec/per_tensor_codecs.py",
            source_citations=(
                "NIST SP 800-90A deterministic DRBG",
                ".omx/research/expert_team_signal_processing_nsa_sigint_20260513.md",
            ),
            rank_hint=10,
        ),
        CooperativeReceiverCandidate(
            campaign_id="b1_atick_redlich_segmap_eigenmodes",
            lineage="fields_medalist_biology",
            source_commit="cbc6b48b",
            source_memo=".omx/research/expert_team_biology_20260513.md",
            hypothesis=(
                "SegMap signal should be encoded in whitened scorer eigenmodes, "
                "dropping low-power modes under the Atick-Redlich power constraint."
            ),
            predicted_delta_low=-0.015,
            predicted_delta_high=-0.005,
            estimated_cost_usd_low=0.50,
            estimated_cost_usd_high=1.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python "
                "tools/build_lapose_foveation_atom_manifest.py --help"
            ),
            byte_closed_packet_plan=(
                "Materialize a SegMap/eigenmode payload with deterministic basis hash; "
                "consume it from a single-file archive section before exact eval."
            ),
            promotion_gate="basis custody + decoded mask/pose coupling proof + paired exact eval",
            target_axis="segmentation",
            implementation_surface="src/tac/sensitivity_map/atick_redlich.py",
            source_citations=(
                "Atick-Redlich 1990 Neural Computation",
                ".omx/research/expert_team_fields_medalist_math_biology_alien_tech_20260513.md",
            ),
            rank_hint=11,
        ),
        CooperativeReceiverCandidate(
            campaign_id="g5_mallat_scattering_decoder",
            lineage="fields_medalist_geometry",
            source_commit="cbc6b48b",
            source_memo=".omx/research/expert_team_geometry_20260513.md",
            hypothesis=(
                "A fixed wavelet-scattering decoder may replace learned decoder bytes, "
                "moving archive budget from weights into pair-specific latents."
            ),
            predicted_delta_low=-0.040,
            predicted_delta_high=-0.020,
            estimated_cost_usd_low=3.00,
            estimated_cost_usd_high=5.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python "
                "tools/build_cooperative_receiver_campaign_queue.py --output "
                "reports/cooperative_receiver/campaign_queue.json"
            ),
            byte_closed_packet_plan=(
                "Build a substrate with fixed scattering kernels in source, train only "
                "latents/adapters, then export monolithic archive + <=200 LOC inflate."
            ),
            promotion_gate="trainer + archive grammar + runtime dependency closure + paired exact eval",
            target_axis="representation",
            implementation_surface="src/tac/substrates/mallat_scattering_decoder/",
            source_citations=(
                "Mallat 2012 CPAM DOI:10.1002/cpa.21413",
                ".omx/research/expert_team_geometry_20260513.md",
            ),
            rank_hint=12,
        ),
        CooperativeReceiverCandidate(
            campaign_id="a1_plus_lapose_composition",
            lineage="operator_lapose_pose_axis",
            source_commit="local_a1_lapose",
            source_memo=".omx/research/cross_paradigm_frontier_inventory_20260511_codex.md",
            hypothesis=(
                "Compose A1's rate-frontier renderer with LAPose foveal RGB "
                "residuals so bytes target the pose-axis marginal regime instead "
                "of another HNeRV-family internal retrain."
            ),
            predicted_delta_low=-0.013,
            predicted_delta_high=-0.003,
            estimated_cost_usd_low=0.30,
            estimated_cost_usd_high=3.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -q "
                "src/tac/substrates/a1_plus_lapose/tests"
            ),
            byte_closed_packet_plan=(
                "Lower LAPose foveal residual bytes into an A1-host sidecar, prove "
                "inflate consumption and full-frame baseline parity, then run exact eval."
            ),
            promotion_gate=(
                "sidecar byte-consumption proof + predicted packet below 0.19 target "
                "+ paired [contest-CUDA]/[contest-CPU] exact eval"
            ),
            target_axis="pose_foveation_composition",
            implementation_surface="src/tac/substrates/a1_plus_lapose/",
            source_citations=(
                ".omx/research/cross_paradigm_frontier_inventory_20260511_codex.md",
            ),
            rank_hint=13,
        ),
        CooperativeReceiverCandidate(
            campaign_id="a1_plus_wavelet_residual_retarget",
            lineage="operator_wavelet_sidecar",
            source_commit="local_a1_wavelet",
            source_memo=".omx/research/cross_paradigm_frontier_inventory_20260511_codex.md",
            hypothesis=(
                "Retarget A1+wavelet away from generic byte shaving and toward "
                "score-sensitive edge/detail bands selected by xray/saliency probes."
            ),
            predicted_delta_low=-0.006,
            predicted_delta_high=-0.001,
            estimated_cost_usd_low=0.10,
            estimated_cost_usd_high=1.00,
            timing_smoke_command=(
                "PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -q "
                "src/tac/substrates/a1_plus_wavelet_residual/tests"
            ),
            byte_closed_packet_plan=(
                "Keep wavelet residual as a sidecar compiler pass only when xray "
                "saliency predicts sub-frontier score movement; otherwise block dispatch."
            ),
            promotion_gate=(
                "score-sensitive wavelet atom selection + packet predicted below 0.19 "
                "+ paired [contest-CUDA]/[contest-CPU] exact eval"
            ),
            target_axis="segmentation_detail_composition",
            implementation_surface="src/tac/substrates/a1_plus_wavelet_residual/",
            source_citations=(
                ".omx/research/cross_paradigm_frontier_inventory_20260511_codex.md",
            ),
            rank_hint=14,
        ),
    ]


def build_campaign_queue(
    *,
    candidates: Iterable[CooperativeReceiverCandidate] | None = None,
    generated_at_utc: str = GENERATED_AT_STABLE,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Build the deterministic cooperative-receiver planning manifest."""

    rows = [candidate.as_row() for candidate in (candidates or default_cooperative_receiver_candidates())]
    rows.sort(
        key=lambda row: (
            int(row["rank_hint"]),
            -float(row["ev_per_dollar_proxy"]),
            str(row["campaign_id"]),
        )
    )
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        rows = rows[:top_k]

    violations = {
        str(row["campaign_id"]): validate_proxy_candidate(row)
        for row in rows
        if validate_proxy_candidate(row)
    }
    if violations:
        raise ValueError(f"proxy-boundary violations: {violations}")

    return {
        "schema": QUEUE_SCHEMA,
        "generated_at_utc": generated_at_utc,
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_ready_count": 0,
        "source_commits": [
            "1d62a114",
            "27cd8b41",
            "cbc6b48b",
            "d1fb9f6a",
            "fdfc347f",
            "local_dpw1_substrate",
        ],
        "shared_receiver_knowledge": [
            "contest scorer architecture",
            "contest scorer weights",
            "contest preprocessing and eval roundtrip",
            "single fixed video distribution",
        ],
        "meta_insight": (
            "Do not spend archive bytes on information the fixed scorer/decoder "
            "can infer from shared knowledge; every row remains planning-only "
            "until byte-closed exact eval exists."
        ),
        "top_k": rows,
    }


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact lab-grade markdown queue summary."""

    rows = manifest.get("top_k")
    if not isinstance(rows, list):
        raise ValueError("manifest top_k must be a list")
    lines = [
        "# Cooperative-Receiver Campaign Queue",
        "",
        f"- schema: `{manifest.get('schema')}`",
        f"- planning_only: `{str(manifest.get('planning_only') is True).lower()}`",
        f"- score_claim: `{str(manifest.get('score_claim') is True).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(manifest.get('ready_for_exact_eval_dispatch') is True).lower()}`",
        f"- meta_insight: {manifest.get('meta_insight')}",
        "",
        "| rank | campaign | predicted delta | cost | smoke | promotion gate |",
        "|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest row must be an object")
        band = row.get("predicted_delta_band")
        cost = row.get("estimated_cost_usd_band")
        lines.append(
            "| {rank} | `{campaign}` | `{band}` | `{cost}` | `{smoke}` | {gate} |".format(
                rank=row.get("rank_hint"),
                campaign=row.get("campaign_id"),
                band=band,
                cost=cost,
                smoke=row.get("timing_smoke_command"),
                gate=row.get("promotion_gate"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_campaign_queue(
    output: str | Path,
    *,
    markdown_output: str | Path | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Write JSON and optional markdown campaign artifacts."""

    manifest = build_campaign_queue(top_k=top_k)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if markdown_output is not None:
        md_path = Path(markdown_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


__all__ = [
    "QUEUE_SCHEMA",
    "CooperativeReceiverCandidate",
    "build_campaign_queue",
    "default_cooperative_receiver_candidates",
    "render_markdown",
    "write_campaign_queue",
]
