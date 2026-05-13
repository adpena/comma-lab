"""Phase 3 joint scorer-renderer-codec scaffold (DESIGN-ONLY).

Implements the SCAFFOLD ONLY — no GPU dispatch path. The full trainer is
deferred until Phase 2 lands a 0.142 [contest-CUDA verified] anchor and the
operator approves the $600-$1200 GPU budget.

Mathematical core
-----------------

The Phase 3 Lagrangian extends the Phase 2 score-domain Lagrangian to the FULL
Tishby Information Bottleneck form:

  L_IB(Z; θ_enc, θ_aux) = I(X; Z)
                        - β_seg · I(Z; Y_seg)
                        - β_pose · I(Z; Y_pose)

  Substituting deterministic encoder Z = encode(X) and Berger-bounded I(X;Z):
    I(X; Z) ≥ R_iid(D) - Σ_streams 0.5·log₂(1/(1-ρ²))      (Berger 1971 §4.5)

  Substituting Hinton-distilled (T=2.0) auxiliary scorer for I(Z;Y):
    I(Z; Y_seg) ≈ E[log σ(z_aux,seg(Z) / T)]                (Hinton 2014 §3)
    I(Z; Y_pose) ≈ E[log σ(z_aux,pose(Z) / T)]              (Hinton 2014 §3)

  Final scaffold form (this is what the trainer optimizes):
    L_phase3(θ; λ; ρ) = α · B(θ) / N_REF
                     + β_seg · d_seg_aux(θ; θ_aux)
                     + γ · √(γ_p · d_pose_aux(θ; θ_aux))
                     - λ_distill · KL(σ(z_aux/T) || σ(z_contest/T))   # Hinton distillation regularizer
                     + λ_R · (B(θ) - B*) + (ρ/2) · (B(θ) - B* + u_R)²
                     + λ_S · (d_seg(θ) - d_seg*) + (ρ/2) · (d_seg(θ) - d_seg* + u_S)²
                     + λ_P · (d_pose(θ) - d_pose*) + (ρ/2) · (d_pose(θ) - d_pose* + u_P)²

Cross-paradigm composition
--------------------------

Phase 3 supports cross-PR substrate composition (PR100/101/103 + A1):
  - PR100/101 latent stream → joint encoded with A1 substrate via T17 shared VQ-VAE codebook
  - PR103 mask stream → joint encoded with T15 time-varying FiLM modulators
  - A1 base substrate → frozen reference for cross-paradigm alignment

This is the substrate-vs-codec meta-pattern's principled resolution: substrate
engineering is allowed because every component is anchored on Hinton-distilled
auxiliary scorer's gradient signal, not on score-naive proxy losses.

CLAUDE.md compliance
--------------------

  - All lambda multipliers clipped ≥ 0 (proper inequality).
  - ρ bounded; T19 adaptive-ρ band [1e-3, 1e3] strongly recommended.
  - Auxiliary scorer θ_aux is TRAINING-ONLY; replaced with frozen contest
    scorer at eval time per CLAUDE.md ``check_no_scorer_load_at_inflate``.
  - Distillation gap estimate ≤ 3% required for dispatch (Hinton 2014 §3).
  - Substrate-engineering exception applies: documented in scaffold provenance.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Phase3DispatchGateError(RuntimeError):
    """Raised when Phase 3 dispatch is attempted before all gates clear."""


# Codex round 6 HIGH 1 fix (catalog #142): path-source detection for
# ``unsafe_test_only=True``. The previous round-4 fix made the escape hatch
# ``unsafe_test_only=True`` available unconditionally, which let any
# production caller under ``tools/`` / ``experiments/`` / ``scripts/`` bypass
# every Phase 3 precondition by toggling the kwarg. The runtime guard below
# inspects the calling frame and refuses ``unsafe_test_only=True`` from
# non-test paths (production code MUST satisfy every precondition; tests
# legitimately need a permissive gate).
_TEST_PATH_TOKENS: tuple[str, ...] = (
    "/tests/",
    "/test_",
    "_test.py",
)


def _caller_path_is_test() -> tuple[bool, str]:
    """Inspect the call stack and return ``(is_test_path, first_outside_caller)``.

    Walks the call stack OUT OF the joint_scorer_renderer_codec module
    until it finds the first frame whose filename is in a different
    module. That frame is the true "construction call site" (the dataclass
    machinery interposes its own frames between ``__post_init__`` and the
    user's ``Phase3DispatchGate(...)`` call, so a fixed-depth frames-to-skip
    walk is brittle).

    Returns ``(True, filename)`` when the first-outside-caller filename
    matches a recognized test path token (``/tests/``, ``/test_``, or
    ending in ``_test.py``); ``(False, filename)`` otherwise. Returns
    ``(False, "<unknown>")`` if the frame cannot be inspected.

    Test path tokens follow the pytest convention used elsewhere in this
    repo (``check_phase3_dispatch_gate_fail_closed`` excludes the same
    set of paths from STRICT scanning).
    """
    try:
        this_file = str(Path(__file__).resolve())
        frame = inspect.currentframe()
        if frame is None:
            return (False, "<unknown>")
        # Walk past THIS helper.
        frame = frame.f_back
        # Walk past any frames in THIS file (dataclass-generated
        # __post_init__, joint scaffold __post_init__, etc.) AND past any
        # synthesized-source frames (dataclass auto-generated __init__
        # appears with co_filename like "<string>"). The first frame
        # whose filename is BOTH on disk AND not this file is the user's
        # construction site.
        while frame is not None:
            raw_fn = frame.f_code.co_filename
            # Skip dataclass-synthesized __init__ which has co_filename
            # like "<string>", "<frozen ...>", or any path starting with "<".
            if raw_fn.startswith("<"):
                frame = frame.f_back
                continue
            try:
                fn = str(Path(raw_fn).resolve())
            except (OSError, ValueError):
                # Path resolution failed (e.g. missing file); fall back to raw
                fn = raw_fn
            if fn == this_file:
                frame = frame.f_back
                continue
            # Found first outside-this-file, on-disk frame
            fn_lower = fn.replace("\\", "/")
            is_test = any(tok in fn_lower for tok in _TEST_PATH_TOKENS)
            return (is_test, fn)
        return (False, "<unknown>")
    except Exception:  # pragma: no cover - defensive
        return (False, "<unknown>")


@dataclass(frozen=True)
class Phase3DispatchGate:
    """Gate that refuses Phase 3 dispatch until all preconditions clear.

    The gate is a CONSTRUCTOR-TIME check that loud-fails any attempt to
    instantiate a Phase 3 trainer before:

      1. Phase 2 lands a 0.142 [contest-CUDA verified] anchor
      2. Hinton distillation gap estimate ≤ 3% (T10 IB co-trained scorer)
      3. Operator approval of GPU budget $600-$1200
      4. aaf68f37 adversarial-review verdict CLEAN
      5. Fresh Phase 3 dispatch-readiness council deliberation

    This is the SUBSTRATE-ENGINEERING-EXCEPTION ENFORCEMENT POINT. Phase 3 may
    engineer substrate jointly with codec ONLY because the auxiliary scorer
    is anchored on contest scorer's distillation. If the gate fires before
    distillation is verified, substrate-engineering is unprincipled.

    Codex round 4 MEDIUM 1 fix (2026-05-09, catalog #134): the gate is now
    fail-closed at construction. ``__post_init__`` calls ``self.check()``
    unless the explicit escape hatch ``unsafe_test_only=True`` is set.
    Tests that need to construct a permissive gate to exercise the API
    surface MUST set ``unsafe_test_only=True``. Production trainers
    cannot bypass; the catalog #134 STRICT preflight check refuses any
    ``Phase3DispatchGate(...)`` call site outside this module that does
    NOT set ``unsafe_test_only=True``. Memory:
    feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md.
    """

    phase2_anchor_verified: bool = False
    phase2_anchor_score: float | None = None
    phase2_anchor_evidence_path: str | None = None
    distillation_gap_estimate: float | None = None
    distillation_gap_evidence_path: str | None = None
    operator_approved_gpu_budget_usd: float | None = None
    aaf68f37_verdict_clean: bool = False
    aaf68f37_verdict_evidence_path: str | None = None
    phase3_council_deliberation_path: str | None = None
    unsafe_test_only: bool = False
    # Codex round 6 HIGH 1 (catalog #142): explicit waiver for the rare
    # non-test fixture that legitimately needs a permissive gate. Setting
    # this without explicit operator review is FORBIDDEN; preflight check
    # #142 STRICT scans for the kwarg AND refuses production callsites
    # that lack a same-line ``# PHASE3_GATE_UNSAFE_PATH_WAIVED:<reason>``
    # justification. Defaults to False so the path-audit guard is on by
    # default.
    unsafe_test_only_path_audit_waived: bool = False

    def __post_init__(self) -> None:
        """Fail-closed: enforce the gate at construction.

        Codex round 4 MEDIUM 1 (2026-05-09): the previous behaviour was
        to construct without enforcing — only ``self.check()`` enforced,
        and any future trainer/dispatcher that forgot to call it would
        silently bypass every precondition. Now construction itself
        enforces, with the explicit ``unsafe_test_only=True`` escape
        hatch for tests that need a permissive gate to exercise the
        API surface.

        Codex round 6 HIGH 1 (2026-05-09, catalog #142): the round-4 fix
        let any production caller bypass every precondition by toggling
        ``unsafe_test_only=True``. The runtime guard now detects the
        calling frame and REFUSES ``unsafe_test_only=True`` from non-test
        paths. Production code MUST satisfy every precondition; the
        escape hatch is for tests only. Same-line waiver for the rare
        legitimate non-test fixture: pass ``unsafe_test_only_path_audit_waived=True``
        AND set ``unsafe_test_only=True``; this is gated by preflight
        check #142 STRICT and requires explicit operator review.

        The class is frozen, so direct field mutation is impossible
        after construction; the only way to "downgrade" a passing gate
        is to construct a fresh one with worse fields, which would
        again fail-closed.
        """
        if self.unsafe_test_only:
            # Codex round 6 HIGH 1: refuse the escape hatch from non-test paths.
            is_test_path, caller_file = _caller_path_is_test()
            if not is_test_path and not self.unsafe_test_only_path_audit_waived:
                raise Phase3DispatchGateError(
                    "Phase 3 dispatch BLOCKED — `unsafe_test_only=True` "
                    "is only permitted from test paths "
                    "(``*/tests/*``, ``test_*.py``, ``*_test.py``); "
                    f"caller is at {caller_file}. "
                    "Production code MUST satisfy every precondition; "
                    "see Phase3DispatchGate.check() for the required "
                    "kwarg set. If this is a legitimate non-test fixture "
                    "that needs a permissive gate (e.g. an interactive "
                    "operator REPL session), set BOTH "
                    "`unsafe_test_only=True` AND "
                    "`unsafe_test_only_path_audit_waived=True` "
                    "with explicit operator review per CLAUDE.md "
                    "FORBIDDEN PATTERNS. Memory: "
                    "feedback_codex_round6_findings_fix_with_self_protection_landed_20260509.md."
                )
            # Test path or explicit operator-waiver: bypass check()
            return
        self.check()

    def check(self) -> None:
        """Raises ``Phase3DispatchGateError`` if any precondition fails."""
        errors: list[str] = []

        if not self.phase2_anchor_verified:
            errors.append("Phase 2 anchor not verified")
        elif self.phase2_anchor_score is None or self.phase2_anchor_score > 0.142:
            errors.append(
                f"Phase 2 anchor score {self.phase2_anchor_score} > 0.142 target "
                "(per Phase 2 floor REBASELINE memo)"
            )
        if not self.phase2_anchor_evidence_path:
            errors.append("Phase 2 anchor evidence path missing")

        if self.distillation_gap_estimate is None:
            errors.append("Distillation gap estimate not provided")
        elif self.distillation_gap_estimate > 0.03:
            errors.append(
                f"Distillation gap estimate {self.distillation_gap_estimate:.4f} > 3% "
                "(per Hinton 2014 §3 verified target)"
            )
        if not self.distillation_gap_evidence_path:
            errors.append("Distillation gap evidence path missing")

        if self.operator_approved_gpu_budget_usd is None:
            errors.append("Operator GPU budget approval missing")
        elif not (600.0 <= self.operator_approved_gpu_budget_usd <= 1200.0):
            errors.append(
                f"Operator GPU budget ${self.operator_approved_gpu_budget_usd} "
                "outside Phase 3 council band [$600, $1200]"
            )

        if not self.aaf68f37_verdict_clean:
            errors.append("aaf68f37 adversarial-review verdict not CLEAN")
        if not self.aaf68f37_verdict_evidence_path:
            errors.append("aaf68f37 verdict evidence path missing")

        if not self.phase3_council_deliberation_path:
            errors.append("Phase 3 dispatch-readiness council memo path missing")

        if errors:
            raise Phase3DispatchGateError(
                "Phase 3 dispatch BLOCKED — "
                + "; ".join(errors)
                + " — see fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md §Phase3"
            )


@dataclass(frozen=True)
class JointScorerRendererCodecConfig:
    """Configuration for the Phase 3 joint scorer-renderer-codec trainer.

    All fields default to scaffold-safe values. Real dispatch values are
    supplied by the Phase 3 dispatcher (FUTURE — not yet landed).

    Attributes
    ----------
    rate_target_bytes : float
        Target rate in BYTES. Phase 3 default: 100,000 (100KB-class archive
        after FP4 + Brotli + nonlinear-transform-coded entropy bottleneck).
    seg_target : float
        Target average SegNet distortion. Default: 0.0006 (slightly tighter
        than A1's 0.0007 — Phase 3 IB co-training enables it).
    pose_target : float
        Target average PoseNet distortion. Default: 0.00015 (slightly tighter
        than A1's 0.00017 — same reason).
    distillation_temperature : float
        Hinton T=2.0 (canonical). Distillation regularizer weight uses this T.
    distillation_gap_target : float
        Target gap between θ_aux and frozen contest scorer. Default: 0.03 (3%).
    rho_init : float
        Initial ρ (penalty parameter). Default: 1.0 (Boyd 2011 §3.4 default).
    rho_min, rho_max : float
        Bounds on ρ for T19 adaptive update. Default: (1e-3, 1e3) per
        coherence council §3.
    use_t19_adaptive_rho : bool
        Strongly recommended True for Phase 3 (Boyd 2011 §3.4.1 endorsed).
    use_t17_shared_vq_codebook : bool
        Strongly recommended True for cross-paradigm composition.
    use_t18_balle_nonlinear_transform : bool
        Strongly recommended True for entropy-bottleneck tightening.
    use_t13_sqrt_n_latent_budget : bool
        Strongly recommended True per Fridrich √n bound (Ker-Pevný-Fridrich 2008).
    cross_paradigm_substrate_sources : tuple[str, ...]
        Which PRs to compose substrates from. Default: ("A1",).
        Phase 3 dispatch may use ("A1", "PR100", "PR101", "PR103").
    """

    rate_target_bytes: float = 100_000.0
    seg_target: float = 0.0006
    pose_target: float = 0.00015
    distillation_temperature: float = 2.0
    distillation_gap_target: float = 0.03
    rho_init: float = 1.0
    rho_min: float = 1e-3
    rho_max: float = 1e3
    use_t19_adaptive_rho: bool = True
    use_t17_shared_vq_codebook: bool = True
    use_t18_balle_nonlinear_transform: bool = True
    use_t13_sqrt_n_latent_budget: bool = True
    cross_paradigm_substrate_sources: tuple[str, ...] = ("A1",)

    def __post_init__(self) -> None:  # noqa: D401
        # Frozen dataclass enforces immutability; basic validation:
        if self.rate_target_bytes <= 0:
            raise ValueError("rate_target_bytes must be positive")
        if self.distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be positive")
        if not (0 < self.distillation_gap_target < 1):
            raise ValueError("distillation_gap_target must be in (0, 1)")
        if not (self.rho_min > 0 and self.rho_max > self.rho_min):
            raise ValueError("rho_min must be > 0 and rho_max > rho_min")
        for src in self.cross_paradigm_substrate_sources:
            if not isinstance(src, str) or not src.strip():
                raise ValueError(
                    "cross_paradigm_substrate_sources entries must be non-empty strings"
                )


@dataclass
class JointScorerRendererCodecScaffold:
    """SCAFFOLD-ONLY orchestrator for Phase 3 joint scorer-renderer-codec.

    This is NOT a trainer. It is a design-time orchestrator that:

      1. Validates dispatch gate preconditions via :class:`Phase3DispatchGate`
      2. Records the Lagrangian form via :func:`phase3_lagrangian_form`
      3. Records the distillation-gap-estimate plan via
         :func:`phase3_distillation_gap_estimate`
      4. Emits a build-manifest stub (via :meth:`emit_build_manifest_stub`)
         that the future Phase 3 dispatcher will populate with archive
         SHA-256s, runtime tree hashes, and exact-eval custody.

    The actual trainer lives in ``experiments/train_phase3_joint_scorer_renderer_codec.py``
    (FUTURE — not yet landed; gated on dispatch decision).
    """

    config: JointScorerRendererCodecConfig
    gate: Phase3DispatchGate
    council_memo_path: str = "fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md"

    def __post_init__(self) -> None:
        """Construction-time gate enforcement.

        Codex round 4 MEDIUM 1 fix (2026-05-09, catalog #134): the gate
        is now enforced at scaffold construction. Previously the scaffold
        ``__post_init__`` was a no-op with the comment "tests need to
        construct with a permissive gate". That comment-only contract
        let any future trainer/dispatcher silently bypass every
        precondition by forgetting to call ``self.gate.check()``.

        The contract is now: scaffold construction enforces the gate
        as a defence-in-depth check. The gate ITSELF also fail-closes
        at its own construction (unless ``unsafe_test_only=True``),
        so by the time we get here the gate is either already-passing
        or explicitly opted-out for tests. We re-call ``check()`` to
        catch the (impossible-by-construction) case where a test
        injects an opted-out gate but expects scaffold dispatch.

        Tests that need to exercise scaffold API with a permissive gate
        MUST construct the gate with ``unsafe_test_only=True`` AND
        accept that ``self.gate.check()`` will then raise on first
        dispatch attempt; the scaffold's API can still be used for
        manifest emission etc., because we only re-check when the
        gate is NOT opted-out.
        """
        if not getattr(self.gate, "unsafe_test_only", False):
            self.gate.check()

    def emit_build_manifest_stub(self) -> dict[str, Any]:
        """Emit a build-manifest stub the future trainer will populate.

        Returns the dict shape that ``tools/build_phase3_archive.py`` (FUTURE)
        will expand with real archive SHA-256s, runtime tree hashes, etc.
        """
        return {
            "phase": "phase3_joint_scorer_renderer_codec",
            "lane_id": "lane_phase3_joint_scorer_renderer_codec",
            "config": {
                "rate_target_bytes": self.config.rate_target_bytes,
                "seg_target": self.config.seg_target,
                "pose_target": self.config.pose_target,
                "distillation_temperature": self.config.distillation_temperature,
                "distillation_gap_target": self.config.distillation_gap_target,
                "rho_init": self.config.rho_init,
                "rho_min": self.config.rho_min,
                "rho_max": self.config.rho_max,
                "use_t19_adaptive_rho": self.config.use_t19_adaptive_rho,
                "use_t17_shared_vq_codebook": self.config.use_t17_shared_vq_codebook,
                "use_t18_balle_nonlinear_transform": self.config.use_t18_balle_nonlinear_transform,
                "use_t13_sqrt_n_latent_budget": self.config.use_t13_sqrt_n_latent_budget,
                "cross_paradigm_substrate_sources": list(
                    self.config.cross_paradigm_substrate_sources
                ),
            },
            "lagrangian_form": phase3_lagrangian_form(),
            "distillation_gap_estimate_plan": phase3_distillation_gap_estimate(
                self.config
            ),
            "council_memo_path": self.council_memo_path,
            "dispatch_status": "DESIGN-ONLY — pending Phase 2 anchor + aaf68f37 verdict + operator approval",
            "predicted_score_band": "[predicted; Phase 3 council; multi-source aggregated; conditional on Phase 2 landing 0.142]",
            "dispatch_ready": False,
            "requires_operator_approval": True,
            "build_manifest_schema_version": 1,
        }


def phase3_lagrangian_form() -> dict[str, str]:
    """Return the Phase 3 Lagrangian form as a string dict for provenance.

    Used by the build-manifest stub and by the dispatch-readiness council to
    verify the Lagrangian form has not silently mutated between scaffold
    landing and dispatch.
    """
    return {
        "name": "Tishby IB Lagrangian (deterministic encoder + Hinton-distilled aux scorer)",
        "form": (
            "L_phase3(θ; λ; ρ) = α · B(θ) / N_REF "
            "+ β_seg · d_seg_aux(θ; θ_aux) "
            "+ γ · √(γ_p · d_pose_aux(θ; θ_aux)) "
            "- λ_distill · KL(σ(z_aux/T) || σ(z_contest/T)) "
            "+ λ_R · (B(θ) - B*) + (ρ/2) · (B(θ) - B* + u_R)² "
            "+ λ_S · (d_seg(θ) - d_seg*) + (ρ/2) · (d_seg(θ) - d_seg* + u_S)² "
            "+ λ_P · (d_pose(θ) - d_pose*) + (ρ/2) · (d_pose(θ) - d_pose* + u_P)²"
        ),
        "theorems_invoked": (
            "Tishby 1999 (IB principle); "
            "Berger 1971 §4.5 (joint-source R(D) lower bound); "
            "Hinton 2014 §3 (distillation T=2.0); "
            "Boyd 2011 §3.4 (ADMM); "
            "Boyd 2011 §3.4.1 (T19 adaptive ρ); "
            "Ballé 2018 (entropy bottleneck); "
            "He-Zheng 2024 (T18 nonlinear transform); "
            "van den Oord 2017 (T17 shared VQ-VAE codebook); "
            "Ker-Pevný-Fridrich 2008 (T13 √n latent budget)"
        ),
        "compliance_tags": (
            "ema_0p997_snapshot_restore; "
            "eval_roundtrip_true; "
            "no_mps_authoritative; "
            "scorer_at_eval_frozen_contest; "
            "substrate_engineering_exception_principled; "
            "score_tag_predicted_only"
        ),
    }


def phase3_distillation_gap_estimate(
    config: JointScorerRendererCodecConfig,
) -> dict[str, Any]:
    """Return the distillation-gap-estimate plan as a string dict for provenance.

    The plan describes HOW the distillation gap will be measured at Phase 3
    dispatch time. The actual measurement requires real GPU forward passes
    (the contest scorer at T=2.0 vs the auxiliary scorer at T=2.0 on the
    same input batch). The scaffold records the PLAN, not the measurement.
    """
    return {
        "method": "Hinton 2014 §3 KL divergence at T=2.0",
        "temperature": config.distillation_temperature,
        "target_gap": config.distillation_gap_target,
        "measurement_protocol": (
            "On a held-out validation set of N=600 pairs (Phase 1 dataset), "
            "measure per-pair KL(σ(z_aux/T) || σ(z_contest/T)) for both seg and pose "
            "outputs. Report mean ± std across pairs. Phase 3 dispatch is GATED on "
            "mean ≤ 3% (per Hinton 2014 §3 verified target)."
        ),
        "evidence_artifact_path_template": (
            "experiments/results/phase3_dispatch_<timestamp>/distillation_gap_estimate.json"
        ),
        "blocker_class_if_exceeded": "phase3_distillation_gap_too_wide_blocks_substrate_engineering_exception",
        "reactivation_criteria": [
            "tighter distillation training (more epochs)",
            "larger auxiliary scorer architecture",
            "improved Hinton distillation hyperparameters",
            "frozen contest scorer at smaller batch sizes",
        ],
    }


__all__ = [
    "JointScorerRendererCodecConfig",
    "JointScorerRendererCodecScaffold",
    "Phase3DispatchGate",
    "Phase3DispatchGateError",
    "phase3_lagrangian_form",
    "phase3_distillation_gap_estimate",
]
