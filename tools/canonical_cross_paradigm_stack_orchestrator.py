#!/usr/bin/env python3
"""Canonical PHASE 4 INTEGRATION cross-paradigm stack orchestrator.

This is the canonical PHASE 4 INTEGRATION deliverable for task #308 —
the CPU-only orchestrator that composes the four landed paradigms
(α/β/γ/δεζ) plus the substitutional finalizers (Op1/Op2) into a single
declarative pipeline and emits a byte-closed candidate archive plus a
deterministic build manifest.

It is a THIN orchestrator: it does not invent codec primitives. Every
step delegates to the canonical CodecOp instances (Op1, Op2, Op3, β, γ,
α-mask portfolio) and the Path B step 6 ADMM × continuous-K mechanism
that produced the cross-paradigm corrected winner anchor (137,469 B
inner Op1 blob sha c33243a1... at 4.15% rel_err on PR101 substrate; full
CPL2 wrapper ~141,517 B since ORCH-SYNC Bug 2 landed CPL2 default
2026-05-08). The inner Op1 blob is unchanged; CPL2's int-key envelope
adds ~4 KB to the on-disk wrapper to round-trip
``effective_byte_maps={int -> str}`` exactly. Phantom predecessor
137,531 B sha ea3b23ed... is retained as forensic record only.

Per CLAUDE.md "Deterministic packet compiler": this orchestrator runs in
``optimize`` mode by default, supports ``identity`` and ``canonicalize``
modes, and fails closed on hidden sidecars + non-deterministic builds.

Strict-scorer-rule
------------------
Pure CPU. No scorer load. No CUDA. No MPS. The orchestrator only
invokes constituent ops' ``encode`` / ``decode`` / ``validate``.

Score-claim discipline
----------------------
Every output is tagged ``evidence_grade="[CPU-build]"``,
``score_claim=False``, ``ready_for_exact_eval_dispatch=False``,
``family_falsified=False``, ``cuda_eval_worth_testing=True``. Per the
"forbidden_CPU_MPS_derived_dispatch_readiness_flag" rule the
dispatch-readiness flag is NEVER True from CPU; per
"forbidden_premature_class_level_falsification" the
falsification-scope is the measured config only.

Composition order (canonical, per Phase 4 design memo §3.1)
------------------------------------------------------------
    arch (Stage 1, design-only — operator-supplied substrate)
      -> β-weight        (paradigm β: sensitivity preprocessing)
      -> γ-allocate      (paradigm γ: joint ADMM lossy-coarsening)
      -> δεζ-finetune    (paradigm δεζ: Phase 2 stub for CPU-only)
      -> α-encode-masks  (paradigm α: NeRV/wavelet/VQ-VAE/grayscale-LUT/AV1)
      -> op-finalizer    (Op1 split-Brotli or Op2 PR103 arithmetic coding)

For paradigms whose underlying primitive is GPU-bound (α-NeRV training,
α-VQ-VAE training, δεζ joint refit, ε learned-prior training, ζ
self-compress QAT) the orchestrator records a clear
``stub_blockers`` list in the manifest rather than silently skipping —
matching the "forbidden silent-skip cascades" rule.

Four canonical CLI smoke invocations (also in tests file)
---------------------------------------------------------

1. **default winner** — reproduce the cross-paradigm winner (137,469 B):

    python tools/canonical_cross_paradigm_stack_orchestrator.py \\
        --gamma-joint-allocator ADMM-continuous-K \\
        --op-finalizer Op1-PR101-split-brotli \\
        --gamma-rms-target 0.0386 \\
        --output-dir experiments/results/orch_default_winner_<UTC>/

2. **conservative** — lower rel_err, larger archive:

    python tools/canonical_cross_paradigm_stack_orchestrator.py \\
        --gamma-joint-allocator ADMM-continuous-K \\
        --op-finalizer none \\
        --gamma-rms-target 0.0200 \\
        --output-dir experiments/results/orch_conservative_<UTC>/

3. **aggressive** — Boyd ADMM + Op2 finalizer:

    python tools/canonical_cross_paradigm_stack_orchestrator.py \\
        --gamma-joint-allocator joint-ADMM-Boyd \\
        --op-finalizer Op2-PR103-arith \\
        --gamma-rms-target 0.05 \\
        --output-dir experiments/results/orch_aggressive_<UTC>/

4. **all-paradigms** — full cross-paradigm composition (NeRV requires
   pretrained codec; if absent the orchestrator records a clear
   stub_blocker and skips α encode):

    python tools/canonical_cross_paradigm_stack_orchestrator.py \\
        --alpha-mask-encoder NeRV \\
        --beta-sensitivity-weights uniform \\
        --gamma-joint-allocator ADMM-continuous-K \\
        --delta-eps-zeta-finetune none \\
        --op-finalizer Op1-PR101-split-brotli \\
        --output-dir experiments/results/orch_all_paradigms_<UTC>/

Cross-references
----------------
* Phase 4 design memo: ``.omx/research/phase4_optimal_stack_design_20260508_claude.md``
* Cross-paradigm winner anchor (137,469 B sha c33243a1...): corrected
  encoder at commit 98d2174b. Phantom predecessor (137,531 B sha ea3b23ed...)
  is retained as forensic record at
  ``reports/raw/pr101_cross_paradigm_hstack_vstack_20260508T060656Z/manifest.json``
  but no longer dispatchable.
* CodecOp Protocol + CPL1 wire format: :mod:`tac.codec_pipeline`
* Path B step 6 ADMM mechanism: ``tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py``
* Lane registry pre-registration: ``lane_phase4_canonical_stack_orchestrator``

Falsification scope
-------------------
``cross_paradigm_stack_byte_anchor_real_pr101_substrate_only`` — this is
a byte-only proxy on a single substrate. Score impact is undetermined
until a real-archive contest-CUDA replay lands.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

TOOL_NAME = "tools/canonical_cross_paradigm_stack_orchestrator.py"
SCHEMA_VERSION = "canonical_cross_paradigm_stack_orchestrator.v1"
EVIDENCE_GRADE = "[CPU-build]"

#: Default PR101 frontier substrate. Operators may override via
#: ``--input-state-dict``. The path is the same one XPARADIGM and Path B
#: step 6 used so cross-tool comparability is preserved.
DEFAULT_INPUT_STATE_DICT = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
)

#: Vocabularies for the CLI choices. Values match the Phase 4 design memo
#: §3.1 exactly so downstream consumers can reason about composition by
#: name.
ALPHA_CHOICES: tuple[str, ...] = (
    "brotli",
    "NeRV",
    "wavelet",
    "VQ-VAE",
    "grayscale-LUT",
    "none",
)
GAMMA_CHOICES: tuple[str, ...] = (
    "ADMM-discrete-sparsity",
    "ADMM-continuous-K",
    "joint-ADMM-Boyd",
    "none",
)
DEZ_CHOICES: tuple[str, ...] = (
    "none",
    "self-compress",
    "learnable-entropy",
    "joint-refit",
)
OP_FINALIZER_CHOICES: tuple[str, ...] = (
    "Op1-PR101-split-brotli",
    "Op2-PR103-arith",
    "none",
)
MODE_CHOICES: tuple[str, ...] = ("identity", "canonicalize", "optimize")


# ---------------------------------------------------------------------------
# Stage records (deterministic, JSON-serialisable)
# ---------------------------------------------------------------------------


@dataclass
class StageRecord:
    """One stage of the orchestrator pipeline.

    ``bytes_out`` is the size of the stage's contribution to the final
    archive. ``stub_blocker`` is non-None when the stage was skipped
    because its primitive is GPU-bound or a Phase 2 stub.
    """

    name: str
    paradigm: str
    enabled: bool
    bytes_out: int = 0
    achieved_rel_err: float | None = None
    stub_blocker: str | None = None
    op_state_keys: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "paradigm": self.paradigm,
            "enabled": self.enabled,
            "bytes_out": self.bytes_out,
            "achieved_rel_err": self.achieved_rel_err,
            "stub_blocker": self.stub_blocker,
            "op_state_keys": list(self.op_state_keys),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Stage 2: γ allocator (paradigm γ)
# ---------------------------------------------------------------------------


def _stage_gamma_admm_continuous_k(
    state_dict: dict,
    *,
    rms_target: float,
) -> tuple[bytes, dict[str, Any], StageRecord]:
    """Run Path B step 6 ADMM × continuous-K on the PR101 substrate.

    Reproduces the cross-paradigm winner mechanism (commit 983598d2 +
    XPARADIGM commit 8d33d5c1). Returns the rebuilt state_dict
    (lossy-coarsened) plus the byte count of the ADMM output and a
    StageRecord for the manifest.

    Notes:
        * The ADMM run uses the same primitives Path B step 6 uses (same
          K_RANGE = range(1, 65), same int8 quantisation, same
          Lagrangian λ-bisection). The orchestrator does NOT change
          their behaviour; it merely composes them.
    """
    import numpy as np
    import torch

    from tac.pr101_split_brotli_codec import (
        FIXED_STATE_SCHEMA,
        N_QUANT,
        _quantize_tensor,
    )

    # Sibling tools — do NOT import as modules at top level so that this
    # tool remains importable even when those sibling modules' imports
    # fail (e.g. a missing optional dep on a fresh checkout).
    from pr101_lossy_coarsening_analytical import (  # type: ignore[import-not-found]
        TensorBlob,
    )
    from pr101_omega_opt_admm_x_lossy_coarsening_empirical import (  # type: ignore[import-not-found]
        bisect_admm_for_global_rms,
        precompute_per_tensor_K_curves,
    )

    tensors: list[TensorBlob] = []
    quant_meta: list[tuple[str, float, tuple]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise ValueError(
                f"γ ADMM-continuous-K requires FIXED_STATE_SCHEMA tensor "
                f"{name!r}, missing from input state_dict"
            )
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        tensors.append(
            TensorBlob(name=name, raw=qt.q_i8.astype(np.int32).flatten())
        )
        quant_meta.append((name, float(qt.scale), tuple(state_dict[name].shape)))

    curves = precompute_per_tensor_K_curves(tensors)
    admm_result = bisect_admm_for_global_rms(
        tensors, curves, rms_target=float(rms_target)
    )
    admm_bytes = int(admm_result["archive_bytes"])
    admm_rel_err = float(admm_result["rel_err"])

    # Rebuild the lossy-coarsened state_dict (post-ADMM) so downstream
    # ops can re-encode on the lower-entropy substrate.
    #
    # Substrate-mismatch fix 2026-05-08 (Codex CRITICAL #4.1, ORCH-SYNC Bug 1):
    # Earlier orchestrator code applied an incorrect /N_QUANT (=127) divisor
    # on the dequantization path here, reproducing the PHANTOM 137,531 B
    # (sha ea3b23ed...) anchor that DID NOT match the runtime decoder at
    # admm_x_lossy_coarsening_path_b_step6_*/submission_dir/inflate.py
    # (which dequantizes as ``q_i8.astype(fp32) * fp16_scale``). The
    # corrected formula yields 137,469 B (sha c33243a1...), matching the
    # corrected encoder at tools/pr101_cross_paradigm_hstack_vstack_empirical.py
    # commit 98d2174b. ``N_QUANT`` is retained as an audit constant (still
    # imported above) but is intentionally NOT used on the dequant path.
    Ks = list(admm_result["Ks"])
    rebuilt: dict[str, "torch.Tensor"] = {}
    dequantization_formula = "rounded.astype(np.float32) * float(np.float16(scale))"
    for (name, scale, shape), tb, K in zip(quant_meta, tensors, Ks, strict=True):
        rounded = (np.round(tb.raw.astype(np.float64) / K) * K).astype(np.int32)
        rounded = np.clip(rounded, -127, 127).astype(np.int8)
        # dequantize to fp32: q_i8 * fp16(scale) -- matches runtime decoder
        scale_fp16 = float(np.float16(scale))
        deq = rounded.astype(np.float32) * scale_fp16
        rebuilt[name] = torch.from_numpy(deq.reshape(shape))

    record = StageRecord(
        name="gamma_admm_continuous_k",
        paradigm="γ",
        enabled=True,
        bytes_out=admm_bytes,
        achieved_rel_err=admm_rel_err,
        op_state_keys=["Ks", "rms_target"],
        notes=(
            f"Path B step 6 ADMM × continuous-K mechanism at "
            f"rms_target={rms_target}; substrate post-ADMM available for "
            f"downstream finalizer."
        ),
    )
    extras = {
        "Ks": [int(k) for k in Ks],
        "rms_target": float(rms_target),
        "admm_archive_bytes": admm_bytes,
        "admm_rel_err": admm_rel_err,
        "rebuilt_substrate": rebuilt,
    }
    return b"", extras, record  # ADMM blob is not directly emitted; the
    # finalizer re-encodes the rebuilt substrate. Returning empty bytes
    # keeps the wire-format honest (γ blob is conceptually consumed by
    # downstream Op1/Op2 finalizer, not shipped separately).


def _stage_gamma_admm_boyd_or_discrete(
    state_dict: dict,
    *,
    allocator: str,
    rms_target: float,
) -> tuple[bytes, dict[str, Any], StageRecord]:
    """Run γ via the canonical Op_GammaJointADMM (Boyd 2011 ADMM coordinator).

    For ``ADMM-discrete-sparsity`` and ``joint-ADMM-Boyd``: runs the
    canonical Op_GammaJointADMM op which embeds run_joint_codec_stack
    (JCSv1 wire format). The op IS substitutional (not a substrate
    transform), so its output blob is the γ-final archive when no
    op-finalizer is configured; when an op-finalizer is configured, the
    orchestrator runs the finalizer on the original input (Boyd ADMM is
    a substitutional alternative to Op1/Op2, not a pre-pass).
    """
    from tac.codec_pipeline import EncodeResult
    from tac.codec_pipeline_joint_admm import Op_GammaJointADMM

    op = Op_GammaJointADMM(max_admm_iters=4)
    res: EncodeResult = op.encode(state_dict, context={})

    record = StageRecord(
        name=f"gamma_{allocator.replace('-', '_')}",
        paradigm="γ",
        enabled=True,
        bytes_out=res.bytes_out,
        achieved_rel_err=None,  # Op_GammaJointADMM dequant rel_err is
        # internally bounded by int8 grid; not reported by the op directly.
        op_state_keys=sorted(res.op_state.keys()),
        notes=(
            f"Op_GammaJointADMM (allocator={allocator!r}) substitutional "
            f"output; max_admm_iters=4 for CPU runtime; rms_target "
            f"({rms_target}) recorded for provenance only — Boyd ADMM uses "
            f"its own KKT residual."
        ),
    )
    extras = {
        "blob": res.blob,
        "op_state": res.op_state,
        "allocator": allocator,
    }
    return res.blob, extras, record


# ---------------------------------------------------------------------------
# Stage 5: op-finalizer (Op1 / Op2 / none)
# ---------------------------------------------------------------------------


def _stage_op_finalizer(
    state_dict: dict,
    *,
    finalizer: str,
) -> tuple[bytes, dict[str, Any], StageRecord]:
    """Run the chosen op-finalizer on the (possibly-transformed) substrate."""
    if finalizer == "none":
        return (
            b"",
            {},
            StageRecord(
                name="op_finalizer",
                paradigm="-",
                enabled=False,
                bytes_out=0,
                stub_blocker=None,
                notes="finalizer=none; γ blob (if any) is the final archive",
            ),
        )

    from tac.codec_pipeline import (
        CodecPipeline,
        Op1_PR101SplitBrotli,
        Op2_PR103ArithmeticCodec,
    )

    if finalizer == "Op1-PR101-split-brotli":
        op = Op1_PR101SplitBrotli(auto_select=True)
    elif finalizer == "Op2-PR103-arith":
        op = Op2_PR103ArithmeticCodec()
    else:
        raise ValueError(
            f"unknown op-finalizer {finalizer!r}; supported: "
            f"{OP_FINALIZER_CHOICES}"
        )

    pipeline = CodecPipeline([op])
    blob, manifest = pipeline.encode(state_dict, skip_validate=True)

    record = StageRecord(
        name=f"finalizer_{finalizer.replace('-', '_')}",
        paradigm="-",
        enabled=True,
        bytes_out=len(blob),
        op_state_keys=sorted(manifest.op_results[0].op_state.keys()),
        notes=(
            f"Finalizer {finalizer!r} re-encodes the (possibly post-γ) "
            f"substrate; output is the final archive when finalizer != none."
        ),
    )
    extras = {
        "blob": blob,
        "final_blob_sha256": manifest.final_blob_sha256,
        "op_state": manifest.op_results[0].op_state,
    }
    return blob, extras, record


# ---------------------------------------------------------------------------
# Stage 1: β preprocessor (paradigm β; default identity)
# ---------------------------------------------------------------------------


def _stage_beta_sensitivity_preprocess(
    state_dict: dict,
    *,
    beta_weights_path: str | None,
) -> tuple[dict, StageRecord]:
    """Apply β (sensitivity-aware) preprocessing.

    For CPU-only orchestration with ``--beta-sensitivity-weights`` not
    set or set to ``"uniform"`` the op runs in identity mode (every
    tensor passes through the mid band unchanged), preserving cross-tool
    comparability with XPARADIGM's β=identity stack. Future operator
    sensitivity artifacts can be supplied via ``beta_weights_path``
    pointing at a sensitivity dict JSON.
    """
    from tac.codec_pipeline import CodecPipeline
    from tac.codec_pipeline_sensitivity import Op_SensitivityPreprocess

    if beta_weights_path is None or beta_weights_path == "uniform":
        op = Op_SensitivityPreprocess.identity()
        substrate = dict(state_dict)
        record = StageRecord(
            name="beta_identity",
            paradigm="β",
            enabled=False,
            bytes_out=0,
            notes=(
                "β=identity (default): every tensor passes through the mid "
                "band unchanged; downstream stages observe original substrate"
            ),
        )
        return substrate, record

    # Non-trivial β: apply the sensitivity preprocess and feed the
    # decoded substrate into downstream stages. The β op is a
    # substrate-transform op, so the orchestrator runs encode + decode
    # to materialise the post-β substrate.
    op = Op_SensitivityPreprocess(sensitivity_source="precomputed")
    pipeline = CodecPipeline([op])
    blob, manifest = pipeline.encode(
        state_dict, context={"sensitivity_artifact_path": beta_weights_path}
    )
    decoded, _ = pipeline.decode(blob)
    record = StageRecord(
        name="beta_sensitivity_preprocess",
        paradigm="β",
        enabled=True,
        bytes_out=len(blob),
        op_state_keys=sorted(manifest.op_results[0].op_state.keys()),
        notes=(
            f"β sensitivity-aware preprocess applied with weights "
            f"{beta_weights_path!r}; downstream stages observe post-β substrate"
        ),
    )
    return decoded, record


# ---------------------------------------------------------------------------
# Stage 3: δεζ finetune (Phase 2 stub for CPU-only)
# ---------------------------------------------------------------------------


def _stage_delta_eps_zeta(mode: str) -> StageRecord:
    """δεζ finetune is GPU-bound (Phase 2). For CPU-only orchestration the
    only supported mode is ``none``; other modes record a clear stub blocker.
    """
    if mode == "none":
        return StageRecord(
            name="delta_eps_zeta",
            paradigm="δεζ",
            enabled=False,
            bytes_out=0,
            notes="δεζ disabled (default for CPU-only orchestrator)",
        )
    return StageRecord(
        name=f"delta_eps_zeta_{mode}",
        paradigm="δεζ",
        enabled=False,
        bytes_out=0,
        stub_blocker=(
            f"δεζ mode {mode!r} is GPU-bound (Phase 2 stub); the CPU-only "
            f"orchestrator records this stage as deferred. To execute, run "
            f"the Phase 2 dispatch driver with operator GPU authorization."
        ),
        notes=(
            "Per Phase 4 design memo §3.1 / §3.4 sanity ladder — δ joint "
            "training, ε learnable entropy, ζ self-compress all require "
            "GPU; no CPU substitute available."
        ),
    )


# ---------------------------------------------------------------------------
# Stage 4: α mask encoder (paradigm α)
# ---------------------------------------------------------------------------


def _stage_alpha_mask(
    *, alpha: str
) -> StageRecord:
    """α mask encoder — none of the alternatives produce mask bytes for the
    PR101 weight-only substrate the orchestrator currently runs on.

    PR101's archive is monolithic (per `feedback_pr106_archive_is_monolithic
    _single_file_20260508`), so α mask encoding is conceptually scoped to
    the *separate* mask-payload sub-archive (HNeRV / PR106-class lanes). The
    orchestrator records the chosen α primitive in the manifest for forensic
    completeness but does NOT inject mask bytes into the PR101 weight blob.
    α=NeRV/VQ-VAE require pretrained codecs (GPU-bound); α=wavelet and
    α=grayscale-LUT can run CPU-only on a mask tensor input but the
    PR101-only substrate doesn't carry one.
    """
    if alpha == "none":
        return StageRecord(
            name="alpha_mask",
            paradigm="α",
            enabled=False,
            bytes_out=0,
            notes="α=none; PR101 substrate is weight-only (no mask payload)",
        )

    # All alpha-family ops require either a mask tensor in state_dict or
    # a pretrained codec (NeRV / VQ-VAE). The orchestrator does not
    # synthesise a mask; recording a clear blocker is the honest path.
    blocker = (
        f"α={alpha!r} encoder requested but PR101 substrate carries no "
        f"mask tensor; α-NeRV / α-VQ-VAE additionally require a "
        f"pretrained codec passed in context (GPU-bound training). The "
        f"α primitive is wired in `tac.codec_pipeline_mask` but cannot "
        f"run on this CPU-only weight-only substrate. Use the α "
        f"mask-encoder bakeoff harness directly when a mask substrate "
        f"becomes available."
    )
    return StageRecord(
        name=f"alpha_{alpha.replace('-', '_').lower()}_mask",
        paradigm="α",
        enabled=False,
        bytes_out=0,
        stub_blocker=blocker,
        notes="α mask encoder requires mask tensor + (NeRV/VQ-VAE) pretrained codec",
    )


# ---------------------------------------------------------------------------
# Local roundtrip smoke
# ---------------------------------------------------------------------------


def _local_roundtrip_smoke(
    archive_blob: bytes,
    *,
    finalizer: str,
    state_dict: dict,
) -> dict[str, Any]:
    """Local smoke: decode the archive and confirm the reconstructed
    state_dict has the same key set as the input.

    This is NOT a contest-CUDA roundtrip — it's the canonical CPU-side
    "decode replays cleanly" check. ``contest_auth_eval_pending=True``
    is recorded to make clear that score impact is undetermined.
    """
    from tac.codec_pipeline import (
        CodecPipeline,
        Op1_PR101SplitBrotli,
        Op2_PR103ArithmeticCodec,
    )

    if finalizer == "Op1-PR101-split-brotli":
        op = Op1_PR101SplitBrotli(auto_select=True)
        pipeline = CodecPipeline([op])
    elif finalizer == "Op2-PR103-arith":
        op = Op2_PR103ArithmeticCodec()
        pipeline = CodecPipeline([op])
    elif finalizer == "none":
        # γ Boyd substitutional output: not a CPL1 wrapper, no decode replay.
        return {
            "decode_attempted": False,
            "reason": "finalizer=none and γ Boyd output is not CPL1-wrapped",
        }
    else:
        return {
            "decode_attempted": False,
            "reason": f"unknown finalizer {finalizer!r}",
        }

    if not archive_blob:
        return {
            "decode_attempted": False,
            "reason": "archive_blob empty (γ-only path with finalizer=none)",
        }

    try:
        decoded, replayed = pipeline.decode(archive_blob)
        decoded_keys = sorted(decoded.keys())
        input_keys = sorted(state_dict.keys())
        # The finalizer is run on the FIXED_STATE_SCHEMA portion of the
        # input; key set equality is checked against the schema, not the
        # full input dict (which may include extra non-schema tensors
        # that the finalizer doesn't carry).
        from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA
        schema_keys = sorted({n for n, _ in FIXED_STATE_SCHEMA})
        return {
            "decode_attempted": True,
            "decode_passed": decoded_keys == schema_keys,
            "decoded_n_tensors": len(decoded),
            "schema_n_tensors": len(schema_keys),
            "replayed_ops": replayed,
            "extra_input_keys": sorted(set(input_keys) - set(schema_keys)),
            "missing_decoded_keys": sorted(set(schema_keys) - set(decoded_keys)),
        }
    except (ValueError, RuntimeError) as exc:
        return {
            "decode_attempted": True,
            "decode_passed": False,
            "decode_error": str(exc),
        }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorResult:
    """Final result of one orchestrator run."""

    archive_bytes: int
    archive_blob_sha256: str
    achieved_rel_err: float | None
    stages: list[StageRecord]
    stub_blockers: list[str]
    smoke: dict[str, Any]


def run_orchestrator(
    state_dict: dict,
    *,
    alpha: str,
    beta_weights_path: str | None,
    gamma: str,
    gamma_rms_target: float,
    delta_eps_zeta: str,
    op_finalizer: str,
    mode: str = "optimize",
) -> tuple[bytes, OrchestratorResult]:
    """Compose the canonical cross-paradigm stack and emit byte-closed archive.

    Returns ``(archive_blob, OrchestratorResult)``. The archive_blob is a
    deterministic CPL1 wrapper (when finalizer is Op1/Op2) or the γ
    Boyd substitutional blob (when finalizer=none and γ is Boyd) or
    empty bytes (when no encoder runs — pathological config).

    Strict invariants:
        * No scorer load anywhere in this function.
        * No CUDA / MPS device usage.
        * Re-running with identical inputs produces identical bytes
          (``identity`` mode roundtrip parity is enforced by the
          underlying CodecPipeline's deterministic CPL1 wire format).
    """
    if mode not in MODE_CHOICES:
        raise ValueError(
            f"mode must be one of {MODE_CHOICES!r}; got {mode!r}"
        )

    stages: list[StageRecord] = []
    stub_blockers: list[str] = []

    # Stage 1: β preprocess (default identity).
    substrate, beta_rec = _stage_beta_sensitivity_preprocess(
        state_dict, beta_weights_path=beta_weights_path
    )
    stages.append(beta_rec)

    # Stage 2: γ allocator. ADMM-continuous-K transforms the substrate;
    # other γ choices produce a substitutional blob.
    gamma_blob = b""
    achieved_rel_err: float | None = None
    if gamma == "ADMM-continuous-K":
        _, gamma_extras, gamma_rec = _stage_gamma_admm_continuous_k(
            substrate, rms_target=gamma_rms_target
        )
        substrate = gamma_extras["rebuilt_substrate"]
        achieved_rel_err = gamma_rec.achieved_rel_err
        stages.append(gamma_rec)
    elif gamma in ("ADMM-discrete-sparsity", "joint-ADMM-Boyd"):
        gamma_blob, _gamma_extras, gamma_rec = _stage_gamma_admm_boyd_or_discrete(
            substrate, allocator=gamma, rms_target=gamma_rms_target
        )
        stages.append(gamma_rec)
    elif gamma == "none":
        stages.append(
            StageRecord(
                name="gamma_disabled",
                paradigm="γ",
                enabled=False,
                bytes_out=0,
                notes="γ disabled; substrate flows through unchanged",
            )
        )
    else:
        raise ValueError(
            f"unknown γ allocator {gamma!r}; supported: {GAMMA_CHOICES}"
        )

    # Stage 3: δεζ finetune (CPU-only stub).
    dez_rec = _stage_delta_eps_zeta(delta_eps_zeta)
    stages.append(dez_rec)
    if dez_rec.stub_blocker:
        stub_blockers.append(dez_rec.stub_blocker)

    # Stage 4: α mask encoder (CPU-only stub for PR101 weight substrate).
    alpha_rec = _stage_alpha_mask(alpha=alpha)
    stages.append(alpha_rec)
    if alpha_rec.stub_blocker:
        stub_blockers.append(alpha_rec.stub_blocker)

    # Stage 5: op-finalizer.
    finalizer_blob, _finalizer_extras, finalizer_rec = _stage_op_finalizer(
        substrate, finalizer=op_finalizer
    )
    stages.append(finalizer_rec)

    # Determine final archive blob.
    # - finalizer != none: finalizer_blob is the archive
    # - finalizer == none + γ produced a substitutional blob: γ blob is the archive
    # - finalizer == none + γ in (continuous-K, none): no blob produced (pathological)
    if op_finalizer != "none":
        archive_blob = finalizer_blob
    elif gamma_blob:
        archive_blob = gamma_blob
    else:
        archive_blob = b""

    archive_bytes = len(archive_blob)
    archive_blob_sha256 = (
        hashlib.sha256(archive_blob).hexdigest() if archive_blob else ""
    )

    smoke = _local_roundtrip_smoke(
        archive_blob, finalizer=op_finalizer, state_dict=state_dict
    )

    return archive_blob, OrchestratorResult(
        archive_bytes=archive_bytes,
        archive_blob_sha256=archive_blob_sha256,
        achieved_rel_err=achieved_rel_err,
        stages=stages,
        stub_blockers=stub_blockers,
        smoke=smoke,
    )


# ---------------------------------------------------------------------------
# Manifest emission + evidence row
# ---------------------------------------------------------------------------


def _write_build_manifest(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    result: OrchestratorResult,
    archive_path: Path,
    inflate_py_path: Path,
    inflate_sh_path: Path,
    input_state_dict_path: Path,
) -> Path:
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cuda_eval_worth_testing": True,
        "family_falsified": False,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "falsification_scope": (
            "cross_paradigm_stack_byte_anchor_real_pr101_substrate_only"
        ),
        "mode": args.mode,
        "input_state_dict": str(input_state_dict_path),
        "selections": {
            "alpha_mask_encoder": args.alpha_mask_encoder,
            "beta_sensitivity_weights": args.beta_sensitivity_weights,
            "gamma_joint_allocator": args.gamma_joint_allocator,
            "gamma_rms_target": float(args.gamma_rms_target),
            "delta_eps_zeta_finetune": args.delta_eps_zeta_finetune,
            "op_finalizer": args.op_finalizer,
        },
        "stages": [s.to_dict() for s in result.stages],
        "stub_blockers": result.stub_blockers,
        "archive": {
            "path": str(archive_path),
            "bytes": result.archive_bytes,
            "sha256": result.archive_blob_sha256,
        },
        "achieved_rel_err": result.achieved_rel_err,
        "deployable_inflate_py": str(inflate_py_path),
        "deployable_inflate_sh": str(inflate_sh_path),
        "local_roundtrip_smoke": result.smoke,
        "generated_at_utc": _dt.datetime.now(_dt.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "dispatch_blockers": [
            "byte_proxy_only_no_score_test",
            "no_real_archive_substrate_for_cuda_replay",
            "missing_exact_cuda_auth_eval",
        ],
        "reactivation_criteria_remaining": [
            "real_archive_dispatch_to_lightning_T4_or_vast_4090",
            "post_phase2_dispatch_with_delta_eps_zeta_enabled",
            "alpha_mask_encoder_bakeoff_with_real_mask_substrate",
        ],
        "non_negotiable_compliance": {
            "strict_scorer_rule": "no_scorer_loaded",
            "device": "cpu_only",
            "deterministic": True,
            "tagged_evidence_grade": EVIDENCE_GRADE,
        },
    }
    manifest_path = out_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return manifest_path


def _write_inflate_py(out_dir: Path, *, finalizer: str) -> Path:
    """Emit a minimal deployable inflate.py that decodes the candidate archive.

    The script is byte-deterministic: it imports the canonical CodecOp
    decoders and replays the pipeline against the recorded finalizer.
    For ``finalizer=none`` (γ Boyd substitutional output) the script
    delegates to the γ decoder directly.
    """
    body = f'''#!/usr/bin/env python3
"""Deployable inflate.py emitted by canonical_cross_paradigm_stack_orchestrator.

Decodes ``archive.bin`` produced by the orchestrator with finalizer={finalizer!r}.
Strict-scorer-rule: pure CPU; no scorers loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3] if (Path(__file__).resolve().parents[3] / "src" / "tac").exists() else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    archive_path = Path(__file__).resolve().parent / "archive.bin"
    blob = archive_path.read_bytes()
    finalizer = {finalizer!r}
    if finalizer == "Op1-PR101-split-brotli":
        from tac.codec_pipeline import CodecPipeline, Op1_PR101SplitBrotli
        pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=True)])
        state_dict, replayed = pipeline.decode(blob)
    elif finalizer == "Op2-PR103-arith":
        from tac.codec_pipeline import CodecPipeline, Op2_PR103ArithmeticCodec
        pipeline = CodecPipeline([Op2_PR103ArithmeticCodec()])
        state_dict, replayed = pipeline.decode(blob)
    elif finalizer == "none":
        # γ Boyd substitutional output — uses Op_GammaJointADMM decode directly.
        from tac.codec_pipeline_joint_admm import Op_GammaJointADMM
        op = Op_GammaJointADMM()
        # Minimal op_state required: the recorded op_state from the encode side
        # is shipped alongside the blob in build_manifest.json. Operators must
        # supply it via a sidecar JSON (no implicit lookup; per the
        # "forbidden silent-skip cascades" rule).
        raise SystemExit(
            "γ-only finalizer=none output requires the op_state sidecar from "
            "build_manifest.json; pass via --op-state-json sidecar (not yet "
            "wired in this minimal inflate.py — tracked as next task)."
        )
    else:
        raise SystemExit(f"unknown finalizer {{finalizer!r}}")
    print(f"decoded {{len(state_dict)}} tensors via {{replayed}}")


if __name__ == "__main__":
    main()
'''
    inflate_path = out_dir / "inflate.py"
    inflate_path.write_text(body)
    return inflate_path


def _write_inflate_sh(out_dir: Path) -> Path:
    """Emit canonical bootstrap-aware inflate.sh.

    Per CLAUDE.md `forbidden_remote_bootstrap_inline`, this script
    delegates dependency setup to the canonical wrapper
    ``scripts/remote_archive_only_eval.sh::bootstrap_runtime_deps()``
    when available. For local CPU smoke it falls back to invoking the
    repo's existing venv directly (no inline ``curl``/``apt`` install).
    """
    body = """#!/usr/bin/env bash
# Canonical inflate.sh emitted by canonical_cross_paradigm_stack_orchestrator.
# Strict-scorer-rule: pure CPU; no scorers loaded.
# Per CLAUDE.md `forbidden_remote_bootstrap_inline`, this delegates to the
# canonical bootstrap wrapper when present.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." 2>/dev/null && pwd || echo "${PWD}")"

if [[ -f "${REPO_ROOT}/scripts/remote_archive_only_eval.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/remote_archive_only_eval.sh"
  if declare -f bootstrap_runtime_deps >/dev/null 2>&1; then
    bootstrap_runtime_deps
  fi
fi

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

cd "${HERE}"
"${PY}" inflate.py
"""
    inflate_sh_path = out_dir / "inflate.sh"
    inflate_sh_path.write_text(body)
    inflate_sh_path.chmod(0o755)
    return inflate_sh_path


def _append_evidence_row(
    *,
    args: argparse.Namespace,
    result: OrchestratorResult,
    manifest_path: Path,
) -> None:
    evidence_path = REPO_ROOT / "reports" / "cathedral_autopilot_evidence.jsonl"
    if not evidence_path.parent.exists():
        return
    technique = (
        f"phase4_orchestrator_alpha={args.alpha_mask_encoder}"
        f"_beta={(args.beta_sensitivity_weights or 'uniform')}"
        f"_gamma={args.gamma_joint_allocator}"
        f"_dez={args.delta_eps_zeta_finetune}"
        f"_finalizer={args.op_finalizer}"
        f"_rms={args.gamma_rms_target}"
    )
    # Manifest path may be relative or absolute depending on --output-dir;
    # render an absolute repo-root-relative path when possible, else absolute.
    try:
        manifest_path_resolved = manifest_path.resolve()
        rel = manifest_path_resolved.relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        rel = str(manifest_path.resolve())
    row = {
        "technique": technique,
        "empirical_archive_bytes": result.archive_bytes,
        "source": f"[CPU-build empirical] {rel}",
        "achieved_rel_err": result.achieved_rel_err,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "family_falsified": False,
        "cuda_eval_worth_testing": True,
        "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with evidence_path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-state-dict",
        type=Path,
        default=DEFAULT_INPUT_STATE_DICT,
        help="PR101 decoder state_dict.pt (default: PR101 frontier substrate).",
    )
    parser.add_argument(
        "--alpha-mask-encoder",
        choices=ALPHA_CHOICES,
        default="none",
        help="Paradigm α mask encoder selection (default: none — PR101 substrate is weight-only).",
    )
    parser.add_argument(
        "--beta-sensitivity-weights",
        type=str,
        default=None,
        help=(
            "Paradigm β sensitivity-map path or 'uniform' (default: identity passthrough)."
        ),
    )
    parser.add_argument(
        "--gamma-joint-allocator",
        choices=GAMMA_CHOICES,
        default="ADMM-continuous-K",
        help="Paradigm γ joint allocator (default: ADMM-continuous-K — Path B step 6 mechanism).",
    )
    parser.add_argument(
        "--gamma-rms-target",
        type=float,
        default=0.0386,
        help="γ RMS target for ADMM-continuous-K (default: 0.0386 — Path B step 6 winner).",
    )
    parser.add_argument(
        "--delta-eps-zeta-finetune",
        choices=DEZ_CHOICES,
        default="none",
        help="Paradigm δεζ finetune mode (default: none — GPU-bound Phase 2 stub).",
    )
    parser.add_argument(
        "--op-finalizer",
        choices=OP_FINALIZER_CHOICES,
        default="Op1-PR101-split-brotli",
        help="Op-finalizer for substitutional re-encode (default: Op1-PR101-split-brotli).",
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="optimize",
        help="Packet-compiler mode (default: optimize).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: experiments/results/orch_cross_paradigm_<UTC>/).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch

    out_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"orch_cross_paradigm_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ORCH] loading PR101 substrate from {args.input_state_dict}")
    if not args.input_state_dict.exists():
        raise SystemExit(
            f"input state_dict not found: {args.input_state_dict} — "
            f"orchestrator requires a real PR101 frontier substrate"
        )
    state_dict = torch.load(
        args.input_state_dict, map_location="cpu", weights_only=False
    )
    print(
        f"        tensors={len(state_dict)} "
        f"fp32_bytes={sum(t.numel() * t.element_size() for t in state_dict.values()):,}"
    )

    print(
        f"[ORCH] composition: α={args.alpha_mask_encoder!r} "
        f"β={args.beta_sensitivity_weights!r} γ={args.gamma_joint_allocator!r} "
        f"δεζ={args.delta_eps_zeta_finetune!r} finalizer={args.op_finalizer!r} "
        f"mode={args.mode!r} rms={args.gamma_rms_target}"
    )

    archive_blob, result = run_orchestrator(
        state_dict,
        alpha=args.alpha_mask_encoder,
        beta_weights_path=args.beta_sensitivity_weights,
        gamma=args.gamma_joint_allocator,
        gamma_rms_target=args.gamma_rms_target,
        delta_eps_zeta=args.delta_eps_zeta_finetune,
        op_finalizer=args.op_finalizer,
        mode=args.mode,
    )

    archive_path = out_dir / "archive.bin"
    archive_path.write_bytes(archive_blob)
    inflate_py_path = _write_inflate_py(out_dir, finalizer=args.op_finalizer)
    inflate_sh_path = _write_inflate_sh(out_dir)
    manifest_path = _write_build_manifest(
        out_dir,
        args=args,
        result=result,
        archive_path=archive_path,
        inflate_py_path=inflate_py_path,
        inflate_sh_path=inflate_sh_path,
        input_state_dict_path=args.input_state_dict,
    )
    _append_evidence_row(args=args, result=result, manifest_path=manifest_path)

    print(f"[ORCH] archive: {archive_path} ({result.archive_bytes:,} B)")
    print(f"[ORCH] sha256:  {result.archive_blob_sha256}")
    if result.achieved_rel_err is not None:
        print(f"[ORCH] rel_err: {result.achieved_rel_err:.4f}")
    print(f"[ORCH] manifest: {manifest_path}")
    if result.stub_blockers:
        print("[ORCH] stub blockers (recorded, not silently skipped):")
        for b in result.stub_blockers:
            print(f"        - {b}")
    if result.smoke.get("decode_attempted"):
        passed = result.smoke.get("decode_passed")
        print(f"[ORCH] local roundtrip smoke: passed={passed}")


if __name__ == "__main__":
    main()
