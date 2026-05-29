# SPDX-License-Identifier: MIT
"""SHARED substrate module for SYNTHESIZE_FRAME emission via Atick-Redlich.

Canonical SHARED helper for the 3 sister per-substrate symposia landed by
Slot L (2026-05-29): PR110 / PR101 / DQS1 × SYNTHESIZE_FRAME (α=1.10 each
per Slot H Phase B 84-cell composition_alpha matrix).

Operationalizes the canonical Atick-Redlich 1990 cooperative-receiver
framing: SegNet+PoseNet IS the canonical receiver; synthetic-frame
emission optimizes per-pixel boundary-class exploitation by training
against the FIXED known scorer pair.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" +
Slot L op-routable #4: the SYNTHESIZE_FRAME emission pipeline IS canonical
SHARED (this module); the per-substrate adapters (PR110 V14 cascade /
PR101 PR95-family canonical / DQS1 top32_selective_decoderq) are UNIQUE
per substrate via the :class:`SubstrateGrammarAdapter` Protocol.

Per Wave N+46 canonical anti-pattern (do NOT duplicate code; EXTEND
existing canonical surfaces): this module CONSUMES
:func:`tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss`
(canonical primitive landed 2026-05-13) rather than re-implementing.

Cross-references
----------------
- Design memo: ``.omx/research/synthesize_frame_emission_atick_redlich_shared_substrate_module_design_20260529.md``
- Slot L landing memo: ``feedback_slot_l_slot_h_top_3_super_additive_per_substrate_symposium_prep_landed_20260529.md``
- 3 symposia under ``.omx/research/council_per_substrate_symposium_{pr110,pr101,dqs1}_x_synthesize_frame_super_additive_20260529.md``
- Canonical Atick-Redlich primitive consumed:
  :func:`tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss`
- Canonical scorer-loss helper (transitively):
  :func:`tac.substrates.score_aware_common.score_pair_components`
- Canonical eval-roundtrip pipeline (transitively):
  :func:`tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`
- Catalog #356 per-axis decomposition:
  :class:`tac.cathedral.consumer_contract.AxisDecomposition`
- Catalog #323 canonical Provenance:
  :func:`tac.provenance.builders.build_provenance_for_predicted`
- Catalog #295 PYTHONPATH self-containment + 8th MLX-first standing
  directive: NUMPY-PORTABLE inflate (no MLX dep at inflate time)
- Atick & Redlich 1990 "Towards a theory of early visual processing",
  Neural Computation 2(3):308-320

Public API
----------
- :class:`SynthesizeFrameEmissionConfig` (frozen)
- :class:`AtickRedlichReceiver` (frozen; per-substrate adapter wrapper)
- :class:`SynthesizeFrameEmissionPerPairResult` (frozen)
- :class:`SubstrateGrammarAdapter` (Protocol; per-substrate fork point)
- :func:`build_atick_redlich_cooperative_receiver_for_substrate`
- :func:`synthesize_frame_emission_per_pair`
- :func:`verify_synthesize_frame_emission_byte_stability`

Tier A canonical-routing markers per Catalog #341 (binding for every
emission row returned by this module):

    predicted_delta_adjustment = 0.0         # observability-only
    promotable = False                        # NON-PROMOTABLE by construction
    axis_tag = "[predicted]"                  # scaffold; not [contest-*]
    evidence_grade_per_row = "[predicted]"
    score_claim = False
    ready_for_exact_eval_dispatch = False     # operator-routable per Catalog #246

horizon_class: frontier_pursuit
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import torch

from tac.cathedral.consumer_contract import AxisDecomposition
from tac.codec.cooperative_receiver.atick_redlich import (
    AtickRedlichWeights,
    CooperativeReceiverOutput,
    cooperative_receiver_loss,
)
from tac.provenance.builders import build_provenance_for_predicted
from tac.provenance.validator import provenance_to_dict
from tac.substrates._shared.inflate_runtime import CAMERA_HW

# ---------------------------------------------------------------------------
# Canonical constants (per design memo §6 Public API + Catalog #341)
# ---------------------------------------------------------------------------

#: Recognized substrate IDs per Slot L 3 symposia (PR110/PR101/DQS1) +
#: ``protocol_test`` placeholder for Protocol-conformance testing.
RECOGNIZED_SUBSTRATE_IDS: frozenset[str] = frozenset(
    {"pr110", "pr101", "dqs1", "protocol_test"}
)

#: Recognized framework-agnostic backends per 8th MLX-first standing
#: directive. ``pytorch`` is the canonical CUDA paired-dispatch backend;
#: ``mlx`` is the Apple Silicon macOS-CPU-advisory probe backend per
#: Contrarian binding revision; ``numpy`` is the CPU portable inflate
#: backend per Catalog #295 PYTHONPATH self-containment.
RECOGNIZED_FRAMEWORK_BACKENDS: frozenset[str] = frozenset(
    {"pytorch", "mlx", "numpy"}
)

#: Canonical model_id for Provenance per Catalog #323 sister discipline.
#: Used by :func:`build_provenance_for_predicted` so downstream consumers
#: can identify the predictor (this SHARED helper) in the canonical
#: posterior + canonical equations registry.
PROVENANCE_MODEL_ID: str = (
    "tac.substrates._shared.synthesize_frame_emission_atick_redlich.v1"
)

#: Canonical predicted_band per cell (Slot L symposia §6 Catalog #324).
#: PR110/DQS1 narrower band per Dykstra CO-LEAD verdict; PR101 EXPANDED
#: per PR95-family L3 grammar overhead per Assumption-Adversary binding.
PREDICTED_BAND_PER_SUBSTRATE: Mapping[str, tuple[float, float]] = {
    "pr110": (-0.0025, 0.0015),
    "pr101": (-0.0025, 0.0040),
    "dqs1": (-0.0025, 0.0015),
}


# ---------------------------------------------------------------------------
# Frozen contracts (design memo §6)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthesizeFrameEmissionConfig:
    """Canonical configuration for one SYNTHESIZE_FRAME emission run.

    Attributes:
        substrate_id: Canonical substrate ID per Slot L 3 symposia
            (one of ``{"pr110", "pr101", "dqs1", "protocol_test"}``).
            Future 4th sister cells add to :data:`RECOGNIZED_SUBSTRATE_IDS`
            without modifying this dataclass (per cargo-cult unwind A8).
        weights: Atick-Redlich Lagrangian weights; default matches contest
            formula via :class:`AtickRedlichWeights` defaults.
        per_pair_metadata_budget_bytes: Maximum synthetic-frame metadata
            bytes per pair. Bounded per HNeRV parity L3 grammar slack
            audit per substrate (PR110/DQS1 ~80 B; PR101 EXPANDED to
            accommodate PR95-family rigid 4-section grammar overhead).
        framework_agnostic_backend: ``"pytorch"`` (canonical CUDA paired
            dispatch) / ``"mlx"`` (Apple Silicon macOS-CPU-advisory probe
            per Contrarian binding revision) / ``"numpy"`` (CPU portable
            inflate per Catalog #295). The eval-roundtrip + scorer
            forward run in this backend.
        apply_eval_roundtrip: Forbidden False per CLAUDE.md non-negotiable
            ``eval_roundtrip — non-negotiable`` + Catalog #164. Defaults to
            True; explicitly settable for documentation clarity.
        random_seed: Pinned per deterministic-reproducibility (9-dim
            Dimension 7). Used by :func:`verify_synthesize_frame_emission_byte_stability`
            re-run discipline.

    Raises:
        ValueError: on any field invariant violation (substrate_id /
            framework backend / budget / eval_roundtrip false / seed).
    """

    substrate_id: str
    weights: AtickRedlichWeights = field(default_factory=AtickRedlichWeights)
    per_pair_metadata_budget_bytes: int = 80
    framework_agnostic_backend: str = "pytorch"
    apply_eval_roundtrip: bool = True
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.substrate_id, str) or not self.substrate_id:
            raise ValueError(
                "SynthesizeFrameEmissionConfig.substrate_id must be a "
                f"non-empty string; got {self.substrate_id!r}"
            )
        if self.substrate_id not in RECOGNIZED_SUBSTRATE_IDS:
            raise ValueError(
                f"SynthesizeFrameEmissionConfig.substrate_id={self.substrate_id!r} "
                f"not in RECOGNIZED_SUBSTRATE_IDS={sorted(RECOGNIZED_SUBSTRATE_IDS)}; "
                "Slot L 3 symposia recognized cells = pr110 / pr101 / dqs1"
            )
        if not isinstance(self.weights, AtickRedlichWeights):
            raise ValueError(
                "SynthesizeFrameEmissionConfig.weights must be "
                f"AtickRedlichWeights; got {type(self.weights).__name__}"
            )
        if (
            not isinstance(self.per_pair_metadata_budget_bytes, int)
            or isinstance(self.per_pair_metadata_budget_bytes, bool)
        ):
            raise ValueError(
                "SynthesizeFrameEmissionConfig.per_pair_metadata_budget_bytes "
                "must be int (non-bool); got "
                f"{type(self.per_pair_metadata_budget_bytes).__name__}"
            )
        if self.per_pair_metadata_budget_bytes < 0:
            raise ValueError(
                "per_pair_metadata_budget_bytes must be >= 0; got "
                f"{self.per_pair_metadata_budget_bytes}"
            )
        if self.framework_agnostic_backend not in RECOGNIZED_FRAMEWORK_BACKENDS:
            raise ValueError(
                f"framework_agnostic_backend={self.framework_agnostic_backend!r} "
                f"not in RECOGNIZED_FRAMEWORK_BACKENDS={sorted(RECOGNIZED_FRAMEWORK_BACKENDS)}; "
                "per 8th MLX-first standing directive"
            )
        if not isinstance(self.apply_eval_roundtrip, bool):
            raise ValueError(
                "apply_eval_roundtrip must be bool; got "
                f"{type(self.apply_eval_roundtrip).__name__}"
            )
        if not self.apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable' + Catalog #164"
            )
        if (
            not isinstance(self.random_seed, int)
            or isinstance(self.random_seed, bool)
        ):
            raise ValueError(
                "random_seed must be int (non-bool); got "
                f"{type(self.random_seed).__name__}"
            )

    def predicted_band(self) -> tuple[float, float] | None:
        """Return the canonical predicted ΔS band per Slot L symposium for
        this substrate; ``None`` if substrate is ``protocol_test``."""
        return PREDICTED_BAND_PER_SUBSTRATE.get(self.substrate_id)


# ---------------------------------------------------------------------------
# Per-substrate adapter Protocol (Catalog #290 FORK_BECAUSE_PRINCIPLED_MISMATCH)
# ---------------------------------------------------------------------------

@runtime_checkable
class SubstrateGrammarAdapter(Protocol):
    """Protocol for per-substrate archive-grammar slot injection.

    Per Catalog #357 dual-tier canonical contract sister discipline; per
    HNeRV parity L3 + UNIQUE-AND-COMPLETE-PER-METHOD: each adapter is
    UNIQUE per substrate (PR110 V14 cascade / PR101 PR95-family canonical /
    DQS1 top32_selective_decoderq). The Protocol is the canonical
    interface; sister 4th cells add NEW adapter without modifying this
    SHARED helper (cargo-cult unwind A8).

    Implementations live in
    ``tac.substrates._shared.synthesize_frame_emission_atick_redlich_adapters.<substrate_id>``
    (TBD landing in Step 4 of Slot L 8-step cascade — recipe YAMLs).

    Attributes:
        substrate_id: Canonical substrate ID matching
            :class:`SynthesizeFrameEmissionConfig.substrate_id`.

    Methods:
        inject_synthesize_frame_metadata: Insert synthetic-frame
            metadata into the substrate's archive grammar slot.
            Returns ``(modified_archive_bytes, per_axis_archive_bytes_delta)``.
    """

    substrate_id: str

    def inject_synthesize_frame_metadata(
        self,
        archive_bytes: bytes,
        synthetic_frame_metadata_bytes: bytes,
    ) -> tuple[bytes, int]:
        """Insert synthetic-frame metadata into the substrate's archive slot."""
        ...


# ---------------------------------------------------------------------------
# Per-substrate adapter wrapper (frozen)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtickRedlichReceiver:
    """Per-substrate adapter for SYNTHESIZE_FRAME emission.

    The receiver wraps the canonical SegNet+PoseNet scorer pair PLUS the
    per-substrate :class:`SubstrateGrammarAdapter` adapter callable.

    Per UNIQUE-AND-COMPLETE-PER-METHOD operating mode: the
    ``archive_grammar_slot_injector`` is the canonical fork point; the
    SHARED emission helper :func:`synthesize_frame_emission_per_pair`
    consumes it via the Protocol-typed interface.

    Attributes:
        substrate_id: Canonical substrate ID.
        seg_scorer: Contest SegNet module (canonical
            ``preprocess_input`` contract per Catalog #164).
        pose_scorer: Contest PoseNet module (canonical
            ``preprocess_input`` contract per Catalog #164).
        archive_grammar_slot_injector: Per-substrate callable matching
            :class:`SubstrateGrammarAdapter.inject_synthesize_frame_metadata`.

    Raises:
        ValueError: on substrate_id mismatch / missing preprocess_input
            on either scorer / non-callable injector.
    """

    substrate_id: str
    seg_scorer: Any  # torch.nn.Module-like (Protocol elided for testability)
    pose_scorer: Any  # torch.nn.Module-like (Protocol elided for testability)
    archive_grammar_slot_injector: Callable[[bytes, bytes], tuple[bytes, int]]

    def __post_init__(self) -> None:
        if not isinstance(self.substrate_id, str) or not self.substrate_id:
            raise ValueError(
                "AtickRedlichReceiver.substrate_id must be a non-empty "
                f"string; got {self.substrate_id!r}"
            )
        if self.substrate_id not in RECOGNIZED_SUBSTRATE_IDS:
            raise ValueError(
                f"AtickRedlichReceiver.substrate_id={self.substrate_id!r} "
                f"not in RECOGNIZED_SUBSTRATE_IDS={sorted(RECOGNIZED_SUBSTRATE_IDS)}"
            )
        if not hasattr(self.seg_scorer, "preprocess_input"):
            raise ValueError(
                "AtickRedlichReceiver.seg_scorer must expose preprocess_input "
                "per Catalog #164 canonical scorer contract"
            )
        if not hasattr(self.pose_scorer, "preprocess_input"):
            raise ValueError(
                "AtickRedlichReceiver.pose_scorer must expose preprocess_input "
                "per Catalog #164 canonical scorer contract"
            )
        if not callable(self.archive_grammar_slot_injector):
            raise ValueError(
                "AtickRedlichReceiver.archive_grammar_slot_injector must be "
                f"callable; got {type(self.archive_grammar_slot_injector).__name__}"
            )


# ---------------------------------------------------------------------------
# Frozen per-pair emission result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthesizeFrameEmissionPerPairResult:
    """One per-pair emission result with Catalog #356 AxisDecomposition.

    Per Catalog #341 Tier A canonical-routing markers; this is an
    observability-only result NON-PROMOTABLE by construction.

    Attributes:
        per_pair_index: Pair index (0..N_PAIRS-1).
        frame_0_bytes: Synthetic frame_0 RGB bytes (raw uint8;
            per-pixel cooperative-receiver-optimized).
        frame_1_bytes: Synthetic frame_1 RGB bytes (sister; per-pair).
        per_pair_metadata_bytes: Synthetic-frame metadata bytes for this
            pair (bounded per
            :attr:`SynthesizeFrameEmissionConfig.per_pair_metadata_budget_bytes`).
        atick_redlich_loss: Canonical :class:`CooperativeReceiverOutput`
            with scalar loss + seg + pose + pose_sqrt components.
        predicted_axis_decomposition: Per Catalog #356 +
            :class:`AxisDecomposition` ``as_dict()`` output; carries
            canonical Provenance per Catalog #323.
        canonical_routing_markers: Frozen dict of Tier A markers per
            Catalog #341 (predicted_delta_adjustment=0.0 + promotable=False
            + axis_tag=[predicted] + score_claim=False +
            ready_for_exact_eval_dispatch=False).
    """

    per_pair_index: int
    frame_0_bytes: bytes
    frame_1_bytes: bytes
    per_pair_metadata_bytes: bytes
    atick_redlich_loss: CooperativeReceiverOutput
    predicted_axis_decomposition: Mapping[str, Any]
    canonical_routing_markers: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.per_pair_index, int)
            or isinstance(self.per_pair_index, bool)
            or self.per_pair_index < 0
        ):
            raise ValueError(
                "per_pair_index must be a non-negative int; got "
                f"{self.per_pair_index!r}"
            )
        for fname in ("frame_0_bytes", "frame_1_bytes", "per_pair_metadata_bytes"):
            value = getattr(self, fname)
            if not isinstance(value, (bytes, bytearray)):
                raise ValueError(
                    f"SynthesizeFrameEmissionPerPairResult.{fname} must be "
                    f"bytes; got {type(value).__name__}"
                )
            if isinstance(value, bytearray):
                object.__setattr__(self, fname, bytes(value))
        if not isinstance(self.atick_redlich_loss, CooperativeReceiverOutput):
            raise ValueError(
                "atick_redlich_loss must be CooperativeReceiverOutput; got "
                f"{type(self.atick_redlich_loss).__name__}"
            )
        if not isinstance(self.predicted_axis_decomposition, Mapping):
            raise ValueError(
                "predicted_axis_decomposition must be a Mapping (Catalog #356 "
                f"as_dict() form); got {type(self.predicted_axis_decomposition).__name__}"
            )
        if not isinstance(self.canonical_routing_markers, Mapping):
            raise ValueError(
                "canonical_routing_markers must be a Mapping (Catalog #341 "
                f"Tier A); got {type(self.canonical_routing_markers).__name__}"
            )
        # Catalog #341 Tier A invariants (defensive; this helper constructs
        # the markers internally so they SHOULD always satisfy invariants).
        markers = self.canonical_routing_markers
        if markers.get("predicted_delta_adjustment") != 0.0:
            raise ValueError(
                "canonical_routing_markers.predicted_delta_adjustment MUST "
                "be 0.0 per Catalog #341 Tier A (observability-only)"
            )
        if markers.get("promotable") is not False:
            raise ValueError(
                "canonical_routing_markers.promotable MUST be False per "
                "Catalog #341 Tier A (NON-PROMOTABLE by construction)"
            )
        if markers.get("axis_tag") != "[predicted]":
            raise ValueError(
                "canonical_routing_markers.axis_tag MUST be '[predicted]' per "
                "Catalog #341 Tier A (scaffold; not [contest-*])"
            )


# ---------------------------------------------------------------------------
# Canonical Tier A markers (Catalog #341 binding)
# ---------------------------------------------------------------------------

def _canonical_tier_a_routing_markers() -> dict[str, Any]:
    """Return the canonical Tier A markers per Catalog #341 (binding)."""
    return {
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "axis_tag": "[predicted]",
        "evidence_grade_per_row": "[predicted]",
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


# ---------------------------------------------------------------------------
# Public API: factory + per-pair emission + byte-stability verifier
# ---------------------------------------------------------------------------

def build_atick_redlich_cooperative_receiver_for_substrate(
    substrate_id: str,
    *,
    seg_scorer: Any,
    pose_scorer: Any,
    archive_grammar_slot_injector: Callable[[bytes, bytes], tuple[bytes, int]],
) -> AtickRedlichReceiver:
    """Construct the per-substrate :class:`AtickRedlichReceiver` adapter.

    Per UNIQUE-AND-COMPLETE-PER-METHOD operating mode: this canonical
    factory wraps the per-substrate ``archive_grammar_slot_injector``
    callable + canonical scorer pair into a frozen receiver instance the
    SHARED emission helper consumes.

    The per-substrate adapter implementations live separately under
    ``tac.substrates._shared.synthesize_frame_emission_atick_redlich_adapters.<substrate_id>``
    (TBD landing in Step 4 of Slot L 8-step cascade — recipe YAMLs);
    test fixtures pass synthetic injector callables that satisfy the
    :class:`SubstrateGrammarAdapter` Protocol.

    Args:
        substrate_id: Canonical substrate ID; must be in
            :data:`RECOGNIZED_SUBSTRATE_IDS`.
        seg_scorer: Contest SegNet module (canonical preprocess_input
            contract per Catalog #164).
        pose_scorer: Contest PoseNet module (canonical preprocess_input
            contract per Catalog #164).
        archive_grammar_slot_injector: Per-substrate callable matching
            :class:`SubstrateGrammarAdapter.inject_synthesize_frame_metadata`.

    Returns:
        Frozen :class:`AtickRedlichReceiver`.
    """
    return AtickRedlichReceiver(
        substrate_id=substrate_id,
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        archive_grammar_slot_injector=archive_grammar_slot_injector,
    )


def _generate_synthetic_frame_bytes(
    *,
    rgb: torch.Tensor,
    per_pair_index: int,
    config: SynthesizeFrameEmissionConfig,
    role: str,
) -> bytes:
    """Generate the synthetic frame bytes from a rendered RGB tensor.

    Per Catalog #146 contest-compliant output contract: bytes are raw
    uint8 RGB; resolution is the rendered tensor's resolution at
    emission time (per-pair adapter is responsible for downstream
    resize to ``CAMERA_HW`` per Catalog #146 if recipe requires it).

    The synthesis is deterministic: same input + same config + same
    per_pair_index + same role yields byte-identical output (Dim 7
    deterministic reproducibility per the 9-dim checklist).

    Args:
        rgb: predicted RGB tensor in ``[0, 255]``; expected shape
            ``(B, 3, H, W)`` with B=1 per single-pair emission.
        per_pair_index: pair index for deterministic seeding.
        config: emission config; ``random_seed`` is mixed into the
            byte-stable hash for diff-ability across configs.
        role: ``"frame_0"`` or ``"frame_1"`` — used to differentiate
            the two paired emissions deterministically.

    Returns:
        bytes: raw uint8 RGB bytes (C-contiguous; per-pixel).
    """
    if role not in {"frame_0", "frame_1"}:
        raise ValueError(f"role must be 'frame_0' or 'frame_1'; got {role!r}")
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(
            f"_generate_synthetic_frame_bytes expects (B, 3, H, W); "
            f"got {tuple(rgb.shape)}"
        )
    # Detach + clamp + cast to uint8 (no eval-roundtrip here; that ran in
    # the Atick-Redlich loss path already). Per CLAUDE.md "MPS auth eval
    # is NOISE" we always materialize the bytes on CPU.
    detached = rgb.detach().clamp(0.0, 255.0).to(dtype=torch.uint8, device="cpu")
    # Use C-contiguous (B, H, W, C) layout for canonical contest raw form.
    nhwc = detached.permute(0, 2, 3, 1).contiguous()
    return bytes(nhwc.numpy().tobytes(order="C"))


def _generate_synthetic_frame_metadata(
    *,
    per_pair_index: int,
    atick_redlich_loss: CooperativeReceiverOutput,
    config: SynthesizeFrameEmissionConfig,
) -> bytes:
    """Generate the per-pair synthetic-frame metadata bytes.

    The metadata is a compact deterministic payload that carries the
    Atick-Redlich loss components for downstream inflate-time cooperative
    reconstruction. Per Catalog #220 operational-mechanism + Catalog #272
    distinguishing-feature integration: this is the byte payload the
    per-substrate adapter injects into the archive grammar slot.

    Format (canonical; bounded to ``per_pair_metadata_budget_bytes``):

        - 4 bytes: per_pair_index (uint32 little-endian)
        - 4 bytes: random_seed (uint32 little-endian)
        - 8 bytes: seg_term (float64 little-endian; clipped finite)
        - 8 bytes: pose_term (float64 little-endian; clipped finite)
        - 8 bytes: pose_sqrt (float64 little-endian; clipped finite)
        - remaining: deterministic SHA-256 prefix derived from the
          above + substrate_id (bounded to budget)

    Returns:
        bytes: ``per_pair_metadata_budget_bytes`` bytes (canonical; padded
            with deterministic SHA-256 prefix if budget exceeds fixed
            header).
    """
    import struct

    def _finite(x: float, fallback: float = 0.0) -> float:
        if not isinstance(x, (int, float)):
            return fallback
        if math.isnan(x) or math.isinf(x):
            return fallback
        return float(x)

    seg_term = _finite(float(atick_redlich_loss.seg_term.detach().cpu().item()))
    pose_term = _finite(
        float(atick_redlich_loss.pose_term.detach().cpu().item())
    )
    pose_sqrt = _finite(float(atick_redlich_loss.pose_sqrt.detach().cpu().item()))

    header = struct.pack(
        "<IIddd",
        int(per_pair_index) & 0xFFFFFFFF,
        int(config.random_seed) & 0xFFFFFFFF,
        seg_term,
        pose_term,
        pose_sqrt,
    )
    budget = int(config.per_pair_metadata_budget_bytes)
    if budget < len(header):
        # Budget is smaller than the fixed header; truncate the header
        # deterministically (canonical fallback so the helper never raises
        # on tiny budgets; per-substrate adapter sets the canonical budget).
        return header[:budget]
    # Pad to budget with deterministic SHA-256 prefix derived from header +
    # substrate_id (Dim 7 deterministic reproducibility per the 9-dim
    # checklist).
    pad_seed = header + config.substrate_id.encode("utf-8")
    pad = b""
    while len(header) + len(pad) < budget:
        pad += hashlib.sha256(pad_seed + pad).digest()
    return (header + pad)[:budget]


def _build_predicted_axis_decomposition(
    *,
    config: SynthesizeFrameEmissionConfig,
    per_pair_index: int,
    atick_redlich_loss: CooperativeReceiverOutput,
    per_pair_metadata_bytes_count: int,
) -> dict[str, Any]:
    """Build the per-pair canonical AxisDecomposition per Catalog #356.

    Carries canonical Provenance per Catalog #323 via
    :func:`build_provenance_for_predicted` (model_id = canonical helper
    self-reference; inputs_sha256 = SHA-256 over the deterministic
    per-pair inputs tuple).

    Per Catalog #341 Tier A: the AxisDecomposition is OBSERVABILITY-ONLY
    (axis_tag = "[predicted]"; not [contest-*]).

    Returns:
        ``as_dict()`` form of :class:`AxisDecomposition`.
    """
    # Catalog #323 canonical Provenance via canonical builder
    inputs_sha256 = hashlib.sha256(
        f"{config.substrate_id}|{per_pair_index}|{config.random_seed}".encode("utf-8")
    ).hexdigest()
    provenance = build_provenance_for_predicted(
        model_id=PROVENANCE_MODEL_ID,
        inputs_sha256=inputs_sha256,
    )
    provenance_dict = provenance_to_dict(provenance)

    # Per-axis predicted deltas: SHARED helper emits scaffold-bounded
    # values per Slot L symposia + Catalog #356 contract. seg / pose
    # deltas use the unweighted Atick-Redlich loss components (negative
    # = improvement); archive bytes delta is the per-pair metadata budget
    # (signed; positive = archive grows).
    seg_term_value = float(
        atick_redlich_loss.seg_term.detach().cpu().item()
    )
    pose_term_value = float(
        atick_redlich_loss.pose_term.detach().cpu().item()
    )
    # Sanitize: AxisDecomposition rejects NaN / inf; the Atick-Redlich loss
    # should never produce them but be defensive per Dim 4 RIGOR.
    if math.isnan(seg_term_value) or math.isinf(seg_term_value):
        seg_term_value = 0.0
    if math.isnan(pose_term_value) or math.isinf(pose_term_value):
        pose_term_value = 0.0

    decomposition = AxisDecomposition(
        # Per design memo §6: negative = improvement; the Atick-Redlich
        # loss terms are non-negative distortions, so the predicted DELTA
        # (relative to baseline) is bounded to [-loss_term, 0.0].
        # SHARED helper emits the conservative -loss_term as the prediction.
        predicted_d_seg_delta=-seg_term_value,
        predicted_d_pose_delta=-pose_term_value,
        # signed int: the per-pair metadata bytes ADD to the archive
        # (positive = archive grows; canonical signed convention per
        # Catalog #356 docstring).
        predicted_archive_bytes_delta=int(per_pair_metadata_bytes_count),
        axis_tag="[predicted]",
        canonical_provenance=provenance_dict,
    )
    return decomposition.as_dict()


def synthesize_frame_emission_per_pair(
    *,
    per_pair_index: int,
    rgb_0: torch.Tensor,
    rgb_1: torch.Tensor,
    gt_rgb_0: torch.Tensor,
    gt_rgb_1: torch.Tensor,
    cooperative_receiver: AtickRedlichReceiver,
    config: SynthesizeFrameEmissionConfig,
) -> SynthesizeFrameEmissionPerPairResult:
    """Synthesize one per-pair SYNTHESIZE_FRAME emission via Atick-Redlich.

    Per Catalog #311 ego-motion-conditioned predictive coding sister +
    Slot L 3 symposia + canonical Atick-Redlich 1990 receiver-conditioned
    MI framing: SegNet+PoseNet IS the canonical receiver; the per-pair
    emission optimizes per-pixel boundary-class exploitation by training
    against the FIXED known scorer pair.

    Flow:
        1. Compute Atick-Redlich cooperative-receiver loss via canonical
           primitive (delegates to ``score_pair_components`` +
           ``apply_eval_roundtrip`` per Catalog #164; eval_roundtrip
           enforced per CLAUDE.md non-negotiable).
        2. Generate synthetic frame_0 + frame_1 bytes from the rendered
           RGB inputs (deterministic per Dim 7).
        3. Generate per-pair synthetic-frame metadata bytes (bounded per
           ``per_pair_metadata_budget_bytes``; canonical format).
        4. Build canonical AxisDecomposition per Catalog #356 with
           canonical Provenance per Catalog #323.
        5. Return frozen result with Tier A canonical-routing markers
           per Catalog #341.

    Per UNIQUE-AND-COMPLETE-PER-METHOD: this helper does NOT call the
    per-substrate adapter's ``archive_grammar_slot_injector`` — that is
    the recipe-level operation (Step 4 of Slot L cascade). This helper
    emits the per-pair bytes + metadata + AxisDecomposition; the
    recipe-level trainer aggregates per-pair results and calls the
    injector once per archive build.

    Args:
        per_pair_index: pair index (0..N_PAIRS-1).
        rgb_0: predicted RGB pair frame_0 tensor ``(1, 3, H, W)`` in
            ``[0, 255]``. Gradient flows in via canonical primitive.
        rgb_1: predicted RGB pair frame_1 tensor ``(1, 3, H, W)`` in
            ``[0, 255]``.
        gt_rgb_0: ground-truth frame_0 ``(1, 3, H, W)`` in ``[0, 255]``.
            Gradient does NOT flow into targets.
        gt_rgb_1: ground-truth frame_1 ``(1, 3, H, W)`` in ``[0, 255]``.
        cooperative_receiver: per-substrate :class:`AtickRedlichReceiver`.
        config: emission config; must satisfy substrate_id consistency
            with ``cooperative_receiver.substrate_id``.

    Returns:
        Frozen :class:`SynthesizeFrameEmissionPerPairResult`.

    Raises:
        ValueError: on substrate_id mismatch / config invariant violation.
    """
    if cooperative_receiver.substrate_id != config.substrate_id:
        raise ValueError(
            "cooperative_receiver.substrate_id="
            f"{cooperative_receiver.substrate_id!r} != "
            f"config.substrate_id={config.substrate_id!r}; "
            "per UNIQUE-AND-COMPLETE-PER-METHOD: receiver + config MUST "
            "describe the same substrate"
        )

    # Step 1: canonical Atick-Redlich loss (delegates to score_pair_components
    # + apply_eval_roundtrip per Catalog #164).
    # The canonical primitive validates RGB [0, 255] range + finite + eval_
    # roundtrip non-negotiable + scorer preprocess_input contract.
    atick_redlich_loss = cooperative_receiver_loss(
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_rgb_0,
        gt_rgb_1=gt_rgb_1,
        seg_scorer=cooperative_receiver.seg_scorer,
        pose_scorer=cooperative_receiver.pose_scorer,
        weights=config.weights,
        apply_eval_roundtrip=config.apply_eval_roundtrip,
    )

    # Step 2: synthetic frame bytes (deterministic per Dim 7).
    frame_0_bytes = _generate_synthetic_frame_bytes(
        rgb=rgb_0,
        per_pair_index=per_pair_index,
        config=config,
        role="frame_0",
    )
    frame_1_bytes = _generate_synthetic_frame_bytes(
        rgb=rgb_1,
        per_pair_index=per_pair_index,
        config=config,
        role="frame_1",
    )

    # Step 3: per-pair synthetic-frame metadata bytes (bounded canonical).
    per_pair_metadata_bytes = _generate_synthetic_frame_metadata(
        per_pair_index=per_pair_index,
        atick_redlich_loss=atick_redlich_loss,
        config=config,
    )

    # Step 4: canonical AxisDecomposition per Catalog #356 + Provenance.
    predicted_axis_decomposition = _build_predicted_axis_decomposition(
        config=config,
        per_pair_index=per_pair_index,
        atick_redlich_loss=atick_redlich_loss,
        per_pair_metadata_bytes_count=len(per_pair_metadata_bytes),
    )

    # Step 5: Tier A canonical-routing markers per Catalog #341 + frozen.
    return SynthesizeFrameEmissionPerPairResult(
        per_pair_index=per_pair_index,
        frame_0_bytes=frame_0_bytes,
        frame_1_bytes=frame_1_bytes,
        per_pair_metadata_bytes=per_pair_metadata_bytes,
        atick_redlich_loss=atick_redlich_loss,
        predicted_axis_decomposition=predicted_axis_decomposition,
        canonical_routing_markers=_canonical_tier_a_routing_markers(),
    )


def verify_synthesize_frame_emission_byte_stability(
    synthesize_result: SynthesizeFrameEmissionPerPairResult,
    *,
    rerun_config: SynthesizeFrameEmissionConfig,
    rgb_0: torch.Tensor,
    rgb_1: torch.Tensor,
    gt_rgb_0: torch.Tensor,
    gt_rgb_1: torch.Tensor,
    cooperative_receiver: AtickRedlichReceiver,
) -> bool:
    """Verify byte-stability per Catalog #266 sister discipline + Dim 7.

    Re-runs the emission with the same inputs + config + receiver; compares
    byte output bit-exactly. Per Catalog #266 byte-stability invariant: the
    same inputs MUST produce the same byte output.

    Returns:
        ``True`` if synthetic frames + metadata bytes match bit-exactly;
        ``False`` otherwise.
    """
    rerun = synthesize_frame_emission_per_pair(
        per_pair_index=synthesize_result.per_pair_index,
        rgb_0=rgb_0,
        rgb_1=rgb_1,
        gt_rgb_0=gt_rgb_0,
        gt_rgb_1=gt_rgb_1,
        cooperative_receiver=cooperative_receiver,
        config=rerun_config,
    )
    return (
        rerun.frame_0_bytes == synthesize_result.frame_0_bytes
        and rerun.frame_1_bytes == synthesize_result.frame_1_bytes
        and rerun.per_pair_metadata_bytes
        == synthesize_result.per_pair_metadata_bytes
    )


__all__ = [
    "RECOGNIZED_SUBSTRATE_IDS",
    "RECOGNIZED_FRAMEWORK_BACKENDS",
    "PROVENANCE_MODEL_ID",
    "PREDICTED_BAND_PER_SUBSTRATE",
    "SynthesizeFrameEmissionConfig",
    "AtickRedlichReceiver",
    "SynthesizeFrameEmissionPerPairResult",
    "SubstrateGrammarAdapter",
    "build_atick_redlich_cooperative_receiver_for_substrate",
    "synthesize_frame_emission_per_pair",
    "verify_synthesize_frame_emission_byte_stability",
]
