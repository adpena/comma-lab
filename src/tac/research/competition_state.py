"""Competition state -- single source of truth for all strategies and configs.

Updated after every experiment. Read by all scripts. The tripartite pact
(Yousfi, Fridrich, Contrarian) with Karpathy and Tao in skunkworks must
approve all changes to this file.

Usage::

    from tac.research.competition_state import STATE
    print(STATE.our_best_auth)
    print(STATE.cpu_lane.tricks_enabled)
    print(STATE.gpu_lane.path_a_status)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class CPULane:
    """CPU-lane state: postfilter + trick stacking on existing checkpoints."""

    current_best_proxy: float = 0.9238  # Lightning ep851
    current_auth_score: float = 1.97  # original checkpoint (pre-trick-stack)
    best_checkpoint_md5: str = ""  # populated after auth eval
    best_checkpoint_path: str = "submissions/robust_current/postfilter_int8.pt"
    training_profile: str = "proven_baseline"
    training_platform: str = "lightning_t4"
    training_epoch: int = 851

    tricks_enabled: dict = field(default_factory=lambda: {
        "crf": 34,
        "tto_steps": 0,  # not yet validated
        "multi_pass": 1,  # single pass (default)
        "deblock": False,
        "quantization_rounding": False,
        "brightness_shift": False,  # DISPROVEN -- AllNorm invariance is wrong
        "noise_shaping": True,
        "null_space_projection": True,
    })

    tricks_tested: dict = field(default_factory=lambda: {
        # Format: "trick_variant": {"rate": float, "proxy_score": float, "auth_score": float|None}
        "crf_34_baseline": {"rate": None, "proxy_score": 0.9238, "auth_score": 1.97},
        # CRF sweep not yet scored:
        "crf_32": {"rate": None, "proxy_score": None, "auth_score": None},
        "crf_33": {"rate": None, "proxy_score": None, "auth_score": None},
        "crf_35": {"rate": None, "proxy_score": None, "auth_score": None},
        "crf_36": {"rate": None, "proxy_score": None, "auth_score": None},
        "crf_37": {"rate": None, "proxy_score": None, "auth_score": None},
        "crf_38": {"rate": None, "proxy_score": None, "auth_score": None},
        # TTO variants:
        "tto_10steps_30s": {"rate": None, "proxy_score": None, "auth_score": None},
        # Multi-pass:
        "multi_pass_2": {"rate": None, "proxy_score": None, "auth_score": None},
        "multi_pass_3": {"rate": None, "proxy_score": None, "auth_score": None},
        # Deblock:
        "deblock_nlm": {"rate": None, "proxy_score": None, "auth_score": None},
    })

    next_experiments: list = field(default_factory=lambda: [
        "crf_sweep_32_to_38_independent",
        "tto_10steps_on_lightning_checkpoint",
        "multi_pass_2_on_lightning_checkpoint",
        "stack_best_crf_plus_tto_plus_multi_pass",
    ])


@dataclass
class GPULane:
    """GPU-lane state: Fridrich constrained gen vs tiny DP-SIMS."""

    path_a_name: str = "fridrich_constrained_gen"
    path_a_status: str = "smoke_tested_partial"
    # Smoke test result: seg=0.025 (boundary 0.01 violated), pose=0.078 (OK)
    # Partial success: relaxed boundary 0.03 should work.
    path_a_best_score: float | None = None  # no full score yet (smoke only)
    path_a_smoke_seg: float = 0.02484728
    path_a_smoke_pose: float = 0.07779313

    path_b_name: str = "tiny_dp_sims"
    path_b_status: str = "smoke_tested_500steps"
    path_b_best_score: float | None = None  # no auth score yet
    path_b_smoke_params: int = 159_000  # approximate
    path_b_smoke_fp4_kb: float = 78.0

    # Decision point: commit to one GPU path
    decision_date: str = "2026-04-17"

    next_experiments: list = field(default_factory=lambda: [
        "fridrich_proper_100frames_2000steps",
        "tiny_dp_sims_proper_100frames_5000steps",
        "lbfgs_refinement_on_best_gpu_output",
    ])


@dataclass
class CompetitionState:
    """Top-level competition state. THE single source of truth."""

    updated_at: str = "2026-04-12T22:00:00-05:00"
    deadline: str = "2026-05-03"
    days_remaining: int = 21

    # Scores
    quantizr_score: float = 0.60  # target to beat
    our_best_auth: float = 1.97  # best authoritative eval (robust_current)
    our_best_proxy: float = 0.9238  # best proxy eval (Lightning ep851)
    floor_auth: float = 1.33  # floor = dilated_h64 on Modal A10G (exact_current track)

    # Lanes
    cpu_lane: CPULane = field(default_factory=CPULane)
    gpu_lane: GPULane = field(default_factory=GPULane)

    # Compute budget (estimates)
    compute_budget: dict = field(default_factory=lambda: {
        "lightning_t4_hours_remaining": 63,  # free tier estimate
        "modal_a10g_hours_used": 12,
        "local_m5_max_available": True,
    })

    # Dead techniques -- NEVER retry these
    killed_techniques: list = field(default_factory=lambda: [
        "kl_distill_loss_mode",  # 2 auth evals: 1.85 and 2.05. PoseNet collapse.
        "adaptive_rebalance_weights",  # Hinton T^2 double-correction. Formula vacuous.
        "posenet_gradient_caps",  # Caused 26x PoseNet regression.
        "segnet_loss_weight_above_100",  # Overwhelms PoseNet signal.
        "brightness_shift_trick",  # AllNorm invariance disproven.
        "psd_architecture",  # Auth eval 1.49 vs dilated 1.33. Worse.
    ])

    # Active runs (populated by experiment scripts)
    active_training_runs: list = field(default_factory=lambda: [
        # Example entry:
        # {"name": "lightning_ep851_crf_sweep", "platform": "lightning_t4",
        #  "started": "2026-04-12T20:00:00Z", "status": "queued"}
    ])

    # Tripartite consensus
    tripartite_consensus: str = (
        "2026-04-12: The tripartite pact (Yousfi, Fridrich, Contrarian) with "
        "Karpathy and Tao in skunkworks consensus: (1) GPU lane Fridrich "
        "constrained gen is the high-ceiling path -- smoke test showed seg=0.025, "
        "pose=0.078, both near-feasible with relaxed boundaries. Run proper "
        "100-frame 2000-step experiment. (2) CPU lane trick stacking is the "
        "safe path -- CRF sweep + TTO + multi-pass can compound. Score each "
        "independently then stack. (3) Tiny DP-SIMS is GPU fallback if Fridrich "
        "stalls. (4) L-BFGS refinement is cheap polish on any GPU output. "
        "(5) Decision date 2026-04-17: commit to best GPU path for full 1200 frames."
    )


# Singleton instance -- import this
STATE = CompetitionState()


def refresh_days_remaining() -> int:
    """Recompute days remaining from today to deadline."""
    today = date.today()
    deadline = date.fromisoformat(STATE.deadline)
    STATE.days_remaining = (deadline - today).days
    return STATE.days_remaining


def update_trick_result(
    trick_name: str,
    *,
    rate: float | None = None,
    proxy_score: float | None = None,
    auth_score: float | None = None,
) -> None:
    """Update a trick's test result in the CPU lane."""
    entry = STATE.cpu_lane.tricks_tested.get(trick_name, {})
    if rate is not None:
        entry["rate"] = rate
    if proxy_score is not None:
        entry["proxy_score"] = proxy_score
    if auth_score is not None:
        entry["auth_score"] = auth_score
    STATE.cpu_lane.tricks_tested[trick_name] = entry


def update_gpu_path(
    path: str,  # "a" or "b"
    *,
    status: str | None = None,
    best_score: float | None = None,
) -> None:
    """Update GPU lane path status."""
    if path == "a":
        if status is not None:
            STATE.gpu_lane.path_a_status = status
        if best_score is not None:
            STATE.gpu_lane.path_a_best_score = best_score
    elif path == "b":
        if status is not None:
            STATE.gpu_lane.path_b_status = status
        if best_score is not None:
            STATE.gpu_lane.path_b_best_score = best_score


def summary() -> str:
    """One-line summary for logs."""
    return (
        f"[STATE] auth={STATE.our_best_auth} proxy={STATE.our_best_proxy} "
        f"days={STATE.days_remaining} "
        f"gpu_a={STATE.gpu_lane.path_a_status} "
        f"gpu_b={STATE.gpu_lane.path_b_status}"
    )
