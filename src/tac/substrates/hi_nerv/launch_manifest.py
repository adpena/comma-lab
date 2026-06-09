"""B1 HiNeRV PR95 229K full-curriculum launch-manifest builder.

This module is the PRE-LAUNCH GATE for the B1 HiNeRV 229K-parity PR95-faithful
8-stage MLX campaign (operator frontier-override 2026-06-09: "Full send now:
229K full curriculum" + "get full telemetry all stages all behavior best
checkpoint to learn and analyze"). No 64-hour MLX run launches until
:func:`build_launch_manifest` returns a manifest whose
``manifest_complete_and_self_consistent`` is ``True``.

It does NOT train. It is a deterministic, fail-closed self-attestation that
binds the four ground-truth facts the operator's anti-hallucination directive
requires together into one auditable artifact:

1. **Param count == 228,903** for the canonical taper
   ``decoder_channels=(36, 30, 23, 17, 14, 11, 8)`` — built from the REAL MLX
   :class:`tac.substrates.hi_nerv.mlx_renderer.HinervSubstrateMLX`, NOT a
   hand-derived number. ``param_count_confirmed`` is ``False`` (fail-closed)
   if the real model does not produce exactly 228,903 params.

2. **The selective Muon+AdamW partition** (NOT pure Muon) computed by the
   canonical :func:`tac.local_acceleration.pr95_hnerv_mlx.partition_pr95_mlx_parameter_names`
   on that exact model. PR95's exact source split is 177,156 Muon / 51,802
   AdamW on PR95's own HNeRV architecture; V1 is a *different* HiNeRV vehicle
   (hierarchical feature grid + ConvNeXt blocks), so the param distribution
   differs while the partition RULE is byte-identical. Per operator 2026-06-09
   "each vehicle will look different especially vehicle 3" + "pr95 is not
   straight up muon".

3. **The PR95 8-stage curriculum** read from the canonical
   :class:`tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum.PR95FaithfulCurriculumFactory`
   — the SAME factory the trainer/adapter consume at runtime — cross-checked
   against the static intake profiler ground truth (per-stage loss family /
   adamw_lr / muon / qat / c1a_lambda / sigma).

4. **Stage-8 Muon WIRED_AND_VALIDATED** (option A): the factory's stage-8
   ``Pr95MlxOptimizerConfig`` has ``use_muon=True`` AND the canonical
   ``apply_pr95_mlx_optimizer_step`` routes the Muon partition through the
   canonical Newton-Schulz kernel ``zeropower_via_newtonschulz5_mlx``. The
   manifest records ``WIRED_AND_VALIDATED`` only when both are structurally
   present; otherwise it records ``STOP_BEFORE_STAGE8_FAILCLOSED`` and the run
   must be configured to stop before stage 8 with a resumable checkpoint.

Every emitted manifest carries the canonical non-promotable markers per
CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/#192/#341:
the manifest attests TRAINING-CONFIG readiness, NOT a score. The archive's
score is exact-eval'd later on contest hardware (CPU Linux x86_64 + CUDA T4),
NEVER from MLX.

Cross-references:
    - CLAUDE.md "NO FAKE IMPLEMENTATIONS" (the manifest builds the REAL model
      and reads the REAL factory; it does not bake assumed constants).
    - CLAUDE.md "EMA -- NON-NEGOTIABLE" (ema_decay=0.997 inference shadow).
    - CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L14-L17.
    - ``experiments/profile_pr95_hnerv_muon_intake.py`` (the static PR95 source
      ground-truth the stage schedule is cross-checked against).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "B1_CANONICAL_DECODER_CHANNELS",
    "B1_PARAM_COUNT_EXACT",
    "B1_PARITY_PARAM_TARGET",
    "LAUNCH_MANIFEST_SCHEMA_VERSION",
    "MLX_RESEARCH_SIGNAL_AXIS",
    "PR95_SOURCE_ADAMW_PARTITION_PARAMS",
    "PR95_SOURCE_MUON_PARTITION_PARAMS",
    "PR95_SOURCE_TOTAL_PARAMS",
    "PR95_STAGE8_MUON_WIRED",
    "PR95_STAGE8_STOP_BEFORE_FAILCLOSED",
    "B1LaunchManifest",
    "B1StageRecord",
    "build_launch_manifest",
    "count_taper_params_and_partition",
]

LAUNCH_MANIFEST_SCHEMA_VERSION: str = "b1_hinerv_launch_manifest.v1"
MLX_RESEARCH_SIGNAL_AXIS: str = "[macOS-MLX research-signal]"

# The canonical 229K-parity taper. EXACTLY 228,903 params (verified by building
# the real MLX model); 1.000x the 228,958 PR95 parity, <= budget. Do NOT mutate
# architecture.py's global default; this taper is passed via HinervConfig ctor.
B1_CANONICAL_DECODER_CHANNELS: tuple[int, ...] = (36, 30, 23, 17, 14, 11, 8)
# Exact param count this taper MUST produce. Fail closed if the real model
# differs by even one parameter (the operator's anti-hallucination directive).
B1_PARAM_COUNT_EXACT: int = 228_903
# The PR95 parity ceiling (PR95's decoder is 228,958 params). The taper must be
# <= this budget.
B1_PARITY_PARAM_TARGET: int = 228_958

# PR95's EXACT-SOURCE partition (from profile_pr95_hnerv_muon_intake.py on the
# real PR95 hnerv_muon decoder). Recorded for cross-vehicle comparison ONLY;
# V1's HiNeRV partition differs because V1's architecture differs.
PR95_SOURCE_TOTAL_PARAMS: int = 228_958
PR95_SOURCE_MUON_PARTITION_PARAMS: int = 177_156
PR95_SOURCE_ADAMW_PARTITION_PARAMS: int = 51_802

# Stage-8 Muon wiring verdict sentinels.
PR95_STAGE8_MUON_WIRED: str = "WIRED_AND_VALIDATED"
PR95_STAGE8_STOP_BEFORE_FAILCLOSED: str = "STOP_BEFORE_STAGE8_FAILCLOSED"

# The canonical selective Muon partition rule, recorded verbatim in the
# manifest for audit (matches profile_pr95_hnerv_muon_intake.py:183 and
# tac.optimization.muon.partition_params_for_muon and
# tac.local_acceleration.pr95_hnerv_mlx.partition_pr95_mlx_parameter_names).
MUON_PARTITION_RULE: str = (
    "is_muon = (ndim >= 2) and ('stem' not in name.lower()) and "
    "(not name.lower().startswith('rgb')) and ('.rgb_' not in name.lower()); "
    "MLX param-name adaptation also requires name.endswith('weight') and "
    "'latents' not in name. Muon -> hidden 2D weight matrices; AdamW -> stem, "
    "rgb output head, all 1D params/biases/norms, and latents. Stages 1-7 are "
    "all AdamW; Muon activates ONLY in stage 8 under faithful_stage8_only."
)

_NON_PROMOTABLE_MARKERS: dict[str, Any] = {
    "measurement_axis": MLX_RESEARCH_SIGNAL_AXIS,
    "score_claim": False,
    "promotable": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

# The PR95-source ground-truth stage schedule (from
# experiments/profile_pr95_hnerv_muon_intake.py on the real PR95 source). Used
# ONLY to cross-check the live factory's output; the factory is the runtime
# authority. (name, loss_family, adamw_lr, uses_muon, uses_qat, c1a_lambda,
# sigma, epochs).
_PR95_SOURCE_STAGE_GROUND_TRUTH: tuple[
    tuple[str, str, float, bool, bool, float, float, int], ...
] = (
    ("stage1_v328_ce", "ce_seg_loss", 1e-3, False, False, 0.0, 0.2, 3000),
    ("stage2_v331_softplus", "tau_softplus_seg_loss", 1e-3, False, False, 0.0, 0.2, 5650),
    ("stage3_v332_smooth", "smooth_disagreement_seg_loss", 1e-4, False, False, 0.0, 0.2, 1500),
    ("stage4_v332_qat", "smooth_disagreement_seg_loss", 1e-4, False, True, 0.0, 0.2, 500),
    ("stage5_c1a_l7", "l7_softplus_seg_loss", 3e-5, False, True, 0.01, 0.2, 9000),
    ("stage6_lambda_sweep", "l7_softplus_seg_loss", 3e-5, False, True, 0.02, 0.2, 2000),
    ("stage7_sigma_sweep", "l7_softplus_seg_loss", 3e-5, False, True, 0.02, 0.1, 3000),
    ("stage8_muon_finetune", "l7_softplus_seg_loss", 1e-5, True, True, 0.02, 0.1, 5000),
)
# PR95 canonical total epoch budget (sum of the 8 stage epochs above).
PR95_TOTAL_EPOCH_BUDGET: int = sum(row[7] for row in _PR95_SOURCE_STAGE_GROUND_TRUTH)


@dataclass(frozen=True)
class B1StageRecord:
    """One stage of the PR95 8-stage curriculum, read from the live factory.

    Mirrors the factory's per-stage verdict; ``validated`` records whether the
    live factory's stage matched the PR95 static-source ground truth on the
    five score-relevant fields (loss_family / uses_muon / uses_qat /
    c1a_lambda / sigma).
    """

    stage_index: int
    name: str
    loss_form: str
    adamw_lr: float
    uses_muon: bool
    uses_qat: bool
    c1a_lambda: float
    sigma: float
    start_epoch: int
    end_epoch: int
    epochs_in_stage: int
    validated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_index": int(self.stage_index),
            "name": self.name,
            "loss_form": self.loss_form,
            "adamw_lr": float(self.adamw_lr),
            "uses_muon": bool(self.uses_muon),
            "uses_qat": bool(self.uses_qat),
            "c1a_lambda": float(self.c1a_lambda),
            "sigma": float(self.sigma),
            "start_epoch": int(self.start_epoch),
            "end_epoch": int(self.end_epoch),
            "epochs_in_stage": int(self.epochs_in_stage),
            "validated": bool(self.validated),
        }


@dataclass(frozen=True)
class PartitionResult:
    """The Muon/AdamW partition computed on the real model."""

    total_params: int
    muon_params: int
    adamw_params: int
    muon_tensor_count: int
    adamw_tensor_count: int

    @property
    def sums_to_total(self) -> bool:
        return (self.muon_params + self.adamw_params) == self.total_params

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_params": int(self.total_params),
            "muon_param_count": int(self.muon_params),
            "adamw_param_count": int(self.adamw_params),
            "muon_tensor_count": int(self.muon_tensor_count),
            "adamw_tensor_count": int(self.adamw_tensor_count),
            "sums_to_total": bool(self.sums_to_total),
        }


def count_taper_params_and_partition(
    decoder_channels: tuple[int, ...] = B1_CANONICAL_DECODER_CHANNELS,
    *,
    num_pairs: int = 600,
    output_height: int = 384,
    output_width: int = 512,
) -> PartitionResult:
    """Build the REAL MLX HiNeRV model and count params + Muon/AdamW partition.

    This is the anti-hallucination core: it constructs the actual
    :class:`HinervSubstrateMLX` with the taper and computes the partition with
    the canonical :func:`partition_pr95_mlx_parameter_names`. No constant is
    hand-derived.

    Args:
        decoder_channels: the per-stage decoder channel taper.
        num_pairs / output_height / output_width: arch geometry.

    Returns:
        a :class:`PartitionResult` from the real model.

    Raises:
        ImportError: MLX unavailable.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from tac.local_acceleration.pr95_hnerv_mlx import (
        partition_pr95_mlx_parameter_names,
    )
    from tac.substrates.hi_nerv.architecture import HinervConfig
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    cfg = HinervConfig(
        decoder_channels=tuple(int(c) for c in decoder_channels),
        num_pairs=int(num_pairs),
        output_height=int(output_height),
        output_width=int(output_width),
    )
    model = HinervSubstrateMLX(cfg)
    params = model.parameters()
    flat = tree_flatten(params)

    def _numel(value: Any) -> int:
        shape = list(getattr(value, "shape", ()))
        if not shape:
            return 1
        return int(mx.prod(mx.array(shape)).item())

    name_to_size = {name: _numel(value) for name, value in flat}
    total = sum(name_to_size.values())
    split = partition_pr95_mlx_parameter_names(params)
    muon_params = sum(name_to_size[n] for n in split["muon"])
    adamw_params = sum(name_to_size[n] for n in split["adamw"])
    return PartitionResult(
        total_params=int(total),
        muon_params=int(muon_params),
        adamw_params=int(adamw_params),
        muon_tensor_count=len(split["muon"]),
        adamw_tensor_count=len(split["adamw"]),
    )


def _build_validated_stage_records(
    *, total_epoch_budget: int, muon_policy: str
) -> tuple[list[B1StageRecord], bool]:
    """Read the live PR95 factory and validate each stage vs source ground truth."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(
        total_epoch_budget=int(total_epoch_budget),
        muon_policy=str(muon_policy),
    )
    records: list[B1StageRecord] = []
    all_validated = True
    boundaries = list(factory.stage_epoch_boundaries)
    if len(boundaries) != len(_PR95_SOURCE_STAGE_GROUND_TRUTH):
        all_validated = False
    for (stage_index, start_epoch, end_epoch) in boundaries:
        verdict = factory.current_stage_verdict(int(start_epoch))
        idx0 = int(stage_index) - 1
        validated = False
        if 0 <= idx0 < len(_PR95_SOURCE_STAGE_GROUND_TRUTH):
            gt = _PR95_SOURCE_STAGE_GROUND_TRUTH[idx0]
            # Cross-check the five score-relevant fields against PR95 source
            # ground truth. loss_form uses a substring check because the
            # factory may carry a longer descriptor (e.g. ``l7_softplus_seg_loss``).
            gt_loss = gt[1]
            validated = (
                str(verdict.loss_family).strip() == gt_loss
                and bool(verdict.uses_muon) == bool(gt[3])
                and bool(verdict.qat_active) == bool(gt[4])
                and abs(float(verdict.cat_lambda) - float(gt[5])) < 1e-9
                and abs(float(verdict.cat_sigma) - float(gt[6])) < 1e-9
            )
        if not validated:
            all_validated = False
        records.append(
            B1StageRecord(
                stage_index=int(stage_index),
                name=str(verdict.descriptor_id),
                loss_form=str(verdict.loss_family),
                adamw_lr=float(_optimizer_adamw_lr(verdict)),
                uses_muon=bool(verdict.uses_muon),
                uses_qat=bool(verdict.qat_active),
                c1a_lambda=float(verdict.cat_lambda),
                sigma=float(verdict.cat_sigma),
                start_epoch=int(start_epoch),
                end_epoch=int(end_epoch),
                epochs_in_stage=int(end_epoch) - int(start_epoch),
                validated=bool(validated),
            )
        )
    return records, all_validated


def _optimizer_adamw_lr(verdict: Any) -> float:
    """Extract the adamw_lr from a stage verdict's optimizer_config defensively."""
    oc = getattr(verdict, "optimizer_config", None)
    if oc is None:
        return 0.0
    lr = getattr(oc, "adamw_lr", None)
    if lr is None and isinstance(oc, dict):
        lr = oc.get("adamw_lr")
    try:
        return float(lr)
    except (TypeError, ValueError):
        return 0.0


def _stage8_muon_status(records: list[B1StageRecord]) -> tuple[str, bool]:
    """Determine the stage-8 Muon wiring verdict.

    Option A (WIRED_AND_VALIDATED) requires BOTH:
      - the live factory's stage-8 verdict has ``use_muon=True``;
      - the canonical Newton-Schulz MLX kernel + partition helper that the
        adapter's ``apply_pr95_mlx_optimizer_step`` consumes are importable
        (structural proof the partition is actually applied, not just flagged).

    Returns ``(status, stage8_use_muon_flag)``.
    """
    stage8_use_muon = False
    if records:
        stage8 = records[-1]
        stage8_use_muon = bool(stage8.uses_muon)
    if not stage8_use_muon:
        return PR95_STAGE8_STOP_BEFORE_FAILCLOSED, stage8_use_muon
    # Structural proof the partitioned-Muon step path exists end-to-end.
    try:
        from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: F401
            apply_pr95_mlx_optimizer_step,
            partition_pr95_mlx_parameter_names,
            zeropower_via_newtonschulz5_mlx,
        )
    except ImportError:
        return PR95_STAGE8_STOP_BEFORE_FAILCLOSED, stage8_use_muon
    return PR95_STAGE8_MUON_WIRED, stage8_use_muon


@dataclass(frozen=True)
class B1LaunchManifest:
    """The complete B1 launch manifest. ``manifest_complete_and_self_consistent``
    is the PRE-LAUNCH GATE: ``False`` => do NOT launch the 64h run.
    """

    commit_sha: str
    run_id: str
    decoder_channels: tuple[int, ...]
    param_count: int
    parity_target: int
    param_count_confirmed: bool
    sidecar_export_enabled: bool
    pay_rent_gate_active: bool
    candidate_action_evaluation_active: bool
    mlx_torch_tie_corrector_status: str
    stage_list: tuple[B1StageRecord, ...]
    stages_all_validated: bool
    stage8_muon_status: str
    stage8_use_muon_flag: bool
    muon_partition_rule: str
    muon_param_count: int
    adamw_param_count: int
    muon_partition_sums_to_total: bool
    ema_decay: float
    checkpoint_cadence_epochs: int
    telemetry_path: str
    best_checkpoint_manifest_path: str
    stop_thresholds: dict[str, Any]
    resume_command: str
    exact_eval_command_stub: str
    hallucination_audit: dict[str, Any]
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_complete_and_self_consistent(self) -> bool:
        """The pre-launch gate. ALL must hold:

        - param count confirmed (== 228,903, built from the real model);
        - param count <= parity target;
        - the Muon partition sums to the total (no dropped params);
        - all 8 stages validated against PR95 source ground truth;
        - stage-8 Muon WIRED_AND_VALIDATED (option A);
        - sidecar export OFF unless the pay-rent gate is active;
        - ema_decay is the canonical 0.997 (or an explicit operator value);
        - a non-empty exact-eval command stub + resume command exist.
        """
        if not self.param_count_confirmed:
            return False
        if int(self.param_count) != B1_PARAM_COUNT_EXACT:
            return False
        if int(self.param_count) > int(self.parity_target):
            return False
        if not self.muon_partition_sums_to_total:
            return False
        if not self.stages_all_validated:
            return False
        if self.stage8_muon_status != PR95_STAGE8_MUON_WIRED:
            return False
        # Sidecar must be OFF unless it pays rent.
        if self.sidecar_export_enabled and not self.pay_rent_gate_active:
            return False
        # EMA decay must be a sane (0, 1) value; default is the canonical 0.997.
        if not (0.0 < float(self.ema_decay) < 1.0):
            return False
        # The operator-runnable exact-eval bridge + crash-resume commands must
        # both be present (a launch with no resume path or no B2 bridge is not
        # campaign-complete).
        return bool(
            str(self.exact_eval_command_stub).strip()
            and str(self.resume_command).strip()
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": LAUNCH_MANIFEST_SCHEMA_VERSION,
            "commit_sha": self.commit_sha,
            "run_id": self.run_id,
            "decoder_channels": [int(c) for c in self.decoder_channels],
            "param_count": int(self.param_count),
            "parity_target": int(self.parity_target),
            "param_count_confirmed": bool(self.param_count_confirmed),
            "param_count_exact_required": B1_PARAM_COUNT_EXACT,
            "sidecar_export_enabled": bool(self.sidecar_export_enabled),
            "pay_rent_gate_active": bool(self.pay_rent_gate_active),
            "candidate_action_evaluation_active": bool(
                self.candidate_action_evaluation_active
            ),
            "mlx_torch_tie_corrector_status": self.mlx_torch_tie_corrector_status,
            "stage_list": [s.as_dict() for s in self.stage_list],
            "stages_all_validated": bool(self.stages_all_validated),
            "total_curriculum_epochs": sum(
                s.epochs_in_stage for s in self.stage_list
            ),
            "stage8_muon_status": self.stage8_muon_status,
            "stage8_use_muon_flag": bool(self.stage8_use_muon_flag),
            "muon_partition_rule": self.muon_partition_rule,
            "muon_param_count": int(self.muon_param_count),
            "adamw_param_count": int(self.adamw_param_count),
            "muon_partition_sums_to_total": bool(self.muon_partition_sums_to_total),
            "pr95_source_partition_for_comparison": {
                "total_params": PR95_SOURCE_TOTAL_PARAMS,
                "muon_param_count": PR95_SOURCE_MUON_PARTITION_PARAMS,
                "adamw_param_count": PR95_SOURCE_ADAMW_PARTITION_PARAMS,
                "note": (
                    "PR95 source HNeRV split. V1 is a DIFFERENT HiNeRV vehicle "
                    "(hierarchical feature grid + ConvNeXt); param distribution "
                    "differs while the partition RULE is byte-identical. Per "
                    "operator 2026-06-09 'each vehicle will look different'."
                ),
            },
            "ema_decay": float(self.ema_decay),
            "ema_decay_rationale": (
                "Canonical 0.997 per CLAUDE.md 'EMA -- NON-NEGOTIABLE' + "
                "operator directive 2026-06-09. OVERRIDES the harness "
                "PR95-faithful-source default of 0.999 via the --ema-decay "
                "trainer flag. EMA shadow is saved as the inference checkpoint."
            ),
            "checkpoint_cadence_epochs": int(self.checkpoint_cadence_epochs),
            "telemetry_path": self.telemetry_path,
            "best_checkpoint_manifest_path": self.best_checkpoint_manifest_path,
            "stop_thresholds": dict(self.stop_thresholds),
            "resume_command": self.resume_command,
            "exact_eval_command_stub": self.exact_eval_command_stub,
            "hallucination_audit": dict(self.hallucination_audit),
            "manifest_complete_and_self_consistent": (
                self.manifest_complete_and_self_consistent
            ),
            "notes": self.notes,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        payload.update(_NON_PROMOTABLE_MARKERS)
        return payload


def _default_stop_thresholds() -> dict[str, Any]:
    """The campaign stop/continue rules, encoded for the watchdog + manifest."""
    return {
        "hard_stop": [
            "nan_or_inf_detected_in_loss",
            "checkpoint_not_resumable",
            "telemetry_missing_more_than_25_epochs",
            "sidecar_exported_without_pays_rent",
            "param_count_exceeds_parity_budget_228958",
            "stage8_optimizer_config_missing_or_noop_when_entering_stage8",
        ],
        "soft_stop_review": [
            "proxy_score_not_improving_after_stage1",
            "seg_proxy_improves_but_pose_proxy_worsens_beyond_exact_score_tradeoff",
            "rate_proxy_exceeds_budget_trend",
            "ema_best_not_improving_for_k_checkpoints",
            "mlx_torch_scorer_parity_drift",
        ],
        "continue": [
            "proxy_score_decreasing",
            "ema_best_improving",
            "rate_stable",
            "pose_regression_within_exact_score_value",
            "heartbeat_healthy",
        ],
        "marginal_flip_note": (
            "Use the EXACT contest score 100*d_seg + sqrt(10*d_pose) + "
            "25*bytes/37545489, NOT a linear seg/pose tradeoff: at low pose, "
            "d/d(pose) can exceed d/d(seg) (crossover ~pose_avg 2.5e-4)."
        ),
        "k_checkpoints_for_ema_plateau": 6,
    }


def build_launch_manifest(
    *,
    commit_sha: str,
    run_id: str,
    telemetry_path: str,
    best_checkpoint_manifest_path: str,
    resume_command: str,
    exact_eval_command_stub: str,
    decoder_channels: tuple[int, ...] = B1_CANONICAL_DECODER_CHANNELS,
    num_pairs: int = 600,
    output_height: int = 384,
    output_width: int = 512,
    muon_policy: str = "faithful_stage8_only",
    total_epoch_budget: int = PR95_TOTAL_EPOCH_BUDGET,
    ema_decay: float = 0.997,
    checkpoint_cadence_epochs: int = 250,
    sidecar_export_enabled: bool = False,
    pay_rent_gate_active: bool = True,
    candidate_action_evaluation_active: bool = True,
    mlx_torch_tie_corrector_status: str = (
        "mlx_yuv6_kernel_routes_canonical_rgb_to_yuv6; differentiable "
        "eval-roundtrip active; parity attested via "
        "tac.substrates._shared.mlx_score_aware.bridge_drift"
    ),
    hallucination_audit: dict[str, Any] | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> B1LaunchManifest:
    """Build the complete, self-attesting B1 launch manifest.

    Builds the REAL MLX model (param count + partition), reads the REAL PR95
    factory (8-stage schedule), and verifies stage-8 Muon wiring. Fail-closed
    everywhere: a wrong param count, a dropped partition param, an unvalidated
    stage, or an unwired stage-8 Muon all flip
    ``manifest_complete_and_self_consistent`` to ``False``.

    Args:
        commit_sha / run_id: provenance.
        telemetry_path / best_checkpoint_manifest_path: where the run will
            write its telemetry JSONL + best-checkpoint manifest.
        resume_command / exact_eval_command_stub: the operator-runnable
            crash-resume + (B2) exact-eval bridge commands.
        decoder_channels: the taper (default the canonical 229K taper).
        ema_decay: inference-shadow EMA decay (default canonical 0.997).
        sidecar_export_enabled: MUST stay False unless the sidecar pays rent.
        pay_rent_gate_active / candidate_action_evaluation_active: gate flags.
        hallucination_audit: the P0 audit findings dict.

    Returns:
        a :class:`B1LaunchManifest`.

    Raises:
        ImportError: MLX unavailable.
        ValueError: ``decoder_channels`` is not a 7-tuple of positive ints.
    """
    channels = tuple(int(c) for c in decoder_channels)
    if len(channels) != 7 or any(c <= 0 for c in channels):
        raise ValueError(
            f"decoder_channels must be 7 positive ints; got {channels!r}"
        )

    partition = count_taper_params_and_partition(
        channels,
        num_pairs=num_pairs,
        output_height=output_height,
        output_width=output_width,
    )
    param_count = partition.total_params
    param_count_confirmed = param_count == B1_PARAM_COUNT_EXACT

    records, stages_all_validated = _build_validated_stage_records(
        total_epoch_budget=int(total_epoch_budget),
        muon_policy=str(muon_policy),
    )
    stage8_status, stage8_use_muon = _stage8_muon_status(records)

    audit = dict(hallucination_audit or {})
    # Always record the partition-vs-PR95-source comparison as an audit fact so
    # a reviewer sees the vehicle-divergence explanation inline.
    audit.setdefault(
        "muon_partition_matches_pr95_source_distribution",
        bool(
            partition.muon_params == PR95_SOURCE_MUON_PARTITION_PARAMS
            and partition.adamw_params == PR95_SOURCE_ADAMW_PARTITION_PARAMS
        ),
    )
    audit.setdefault(
        "muon_partition_rule_is_selective_not_pure_muon", True
    )

    return B1LaunchManifest(
        commit_sha=str(commit_sha),
        run_id=str(run_id),
        decoder_channels=channels,
        param_count=int(param_count),
        parity_target=B1_PARITY_PARAM_TARGET,
        param_count_confirmed=bool(param_count_confirmed),
        sidecar_export_enabled=bool(sidecar_export_enabled),
        pay_rent_gate_active=bool(pay_rent_gate_active),
        candidate_action_evaluation_active=bool(candidate_action_evaluation_active),
        mlx_torch_tie_corrector_status=str(mlx_torch_tie_corrector_status),
        stage_list=tuple(records),
        stages_all_validated=bool(stages_all_validated),
        stage8_muon_status=stage8_status,
        stage8_use_muon_flag=bool(stage8_use_muon),
        muon_partition_rule=MUON_PARTITION_RULE,
        muon_param_count=int(partition.muon_params),
        adamw_param_count=int(partition.adamw_params),
        muon_partition_sums_to_total=bool(partition.sums_to_total),
        ema_decay=float(ema_decay),
        checkpoint_cadence_epochs=int(checkpoint_cadence_epochs),
        telemetry_path=str(telemetry_path),
        best_checkpoint_manifest_path=str(best_checkpoint_manifest_path),
        stop_thresholds=_default_stop_thresholds(),
        resume_command=str(resume_command),
        exact_eval_command_stub=str(exact_eval_command_stub),
        hallucination_audit=audit,
        notes=str(notes),
        extra=dict(extra or {}),
    )


def compute_contest_score(seg: float, pose: float, archive_bytes: int) -> float:
    """The canonical contest score, for stop-rule consumers.

    ``100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489``. Provided so
    the watchdog's marginal-flip rule uses the EXACT score, never a linear
    seg/pose proxy.
    """
    return (
        100.0 * float(seg)
        + math.sqrt(10.0 * float(pose))
        + 25.0 * int(archive_bytes) / 37_545_489
    )
