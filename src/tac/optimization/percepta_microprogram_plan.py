# SPDX-License-Identifier: MIT
"""Contest-faithful planning guards for programs-in-weights prototypes.

This module translates the Percepta/WebAssembly-in-weights idea into a Pact
planning surface.  It deliberately does not execute bytecode, load scorers, or
rewrite archives.  It only answers whether a tiny deterministic circuit is
worth materializing before the existing FEC6 decoder-q candidate pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

RATE_WEIGHT = 25.0
RATE_DENOMINATOR_BYTES = 37_545_489
DEFAULT_BYTE_POLISH_FLOOR_BYTES = 78
DEFAULT_MAX_MICRO_OPS = 16
DEFAULT_MAX_PROGRAM_BYTES = 64
DEFAULT_MAX_RUNTIME_PATCH_BYTES = 512
DEFAULT_PR110_RUNTIME_DIR = (
    "experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1"
)
DEFAULT_PR110_SOURCE_MEMBER = (
    "experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/"
    "contest_oracle_search/op3v3_decoder_q_inflate_controls_20260520_codex/"
    "baseline/data_dir/x"
)
DEFAULT_PR110_BASELINE_RAW = (
    "experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/"
    "contest_oracle_search/op3v3_decoder_q_inflate_controls_20260520_codex/"
    "baseline/inflated/0.raw"
)

PrototypeSurface = Literal[
    "weight_embedded_circuit",
    "decoder_side_microprogram",
    "general_wasm_interpreter",
]

Verdict = Literal[
    "NO_GO",
    "PROTOTYPE_GO_PROMOTION_BLOCKED",
    "PROMOTION_GO",
]

ALLOWED_TINY_OPCODES = frozenset(
    {
        "add_i8_saturating",
        "affine_i8",
        "branch_on_selector_bit",
        "clamp_u8",
        "const_i8",
        "lookup_const4",
        "mul_pow2",
        "select_masked",
        "xor_selector_bit",
    }
)


@dataclass(frozen=True)
class ExactEvalCustody:
    """Minimum custody evidence before a microprogram candidate can promote."""

    candidate_archive_sha256: str | None = None
    runtime_tree_sha256: str | None = None
    inflated_outputs_manifest_sha256: str | None = None
    terminal_dispatch_claim: bool = False
    axis_tag: str | None = None

    def ready(self) -> bool:
        return (
            bool(self.candidate_archive_sha256)
            and bool(self.runtime_tree_sha256)
            and bool(self.inflated_outputs_manifest_sha256)
            and bool(self.terminal_dispatch_claim)
            and self.axis_tag in {"[contest-CPU]", "[contest-CUDA]"}
        )

    def missing_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.candidate_archive_sha256:
            missing.append("candidate_archive_sha256")
        if not self.runtime_tree_sha256:
            missing.append("runtime_tree_sha256")
        if not self.inflated_outputs_manifest_sha256:
            missing.append("inflated_outputs_manifest_sha256")
        if not self.terminal_dispatch_claim:
            missing.append("terminal_dispatch_claim")
        if self.axis_tag not in {"[contest-CPU]", "[contest-CUDA]"}:
            missing.append("axis_tag")
        return tuple(missing)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_archive_sha256": self.candidate_archive_sha256,
            "runtime_tree_sha256": self.runtime_tree_sha256,
            "inflated_outputs_manifest_sha256": self.inflated_outputs_manifest_sha256,
            "terminal_dispatch_claim": self.terminal_dispatch_claim,
            "axis_tag": self.axis_tag,
            "ready": self.ready(),
            "missing_fields": list(self.missing_fields()),
        }


@dataclass(frozen=True)
class MicroprogramPrototypeSpec:
    """A proposed tiny correction circuit before archive materialization.

    Score deltas follow the challenge convention: negative is better.
    ``expected_component_delta_score`` must exclude the byte-rate term; this
    planner adds the charged archive-byte delta separately.
    """

    prototype_id: str
    surface: PrototypeSurface
    opcodes: tuple[str, ...]
    expected_component_delta_score: float
    recompressed_archive_byte_delta: int = 0
    encoded_program_bytes: int = 0
    runtime_patch_bytes: int = 0
    best_simple_edit_delta_score: float | None = None
    deterministic_inflate: bool = True
    scorer_free_inflate: bool = True
    network_free_inflate: bool = True
    touches_pr110_live_files: bool = False
    custody: ExactEvalCustody = field(default_factory=ExactEvalCustody)
    notes: str = ""

    @property
    def operation_count(self) -> int:
        return len(self.opcodes)

    @property
    def charged_byte_delta(self) -> int:
        return int(self.recompressed_archive_byte_delta) + int(self.encoded_program_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prototype_id": self.prototype_id,
            "surface": self.surface,
            "opcodes": list(self.opcodes),
            "operation_count": self.operation_count,
            "expected_component_delta_score": self.expected_component_delta_score,
            "recompressed_archive_byte_delta": self.recompressed_archive_byte_delta,
            "encoded_program_bytes": self.encoded_program_bytes,
            "runtime_patch_bytes": self.runtime_patch_bytes,
            "charged_byte_delta": self.charged_byte_delta,
            "best_simple_edit_delta_score": self.best_simple_edit_delta_score,
            "deterministic_inflate": self.deterministic_inflate,
            "scorer_free_inflate": self.scorer_free_inflate,
            "network_free_inflate": self.network_free_inflate,
            "touches_pr110_live_files": self.touches_pr110_live_files,
            "custody": self.custody.as_dict(),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MicroprogramPlan:
    """Planner verdict with rate math, blockers, and smoke commands."""

    spec: MicroprogramPrototypeSpec
    verdict: Verdict
    prototype_blockers: tuple[str, ...]
    promotion_blockers: tuple[str, ...]
    charged_rate_delta_score: float
    projected_total_delta_score: float
    simple_edit_hurdle_delta_score: float
    cheapest_empirical_smoke: tuple[str, ...]
    risk_register: tuple[str, ...]

    @property
    def plan_id(self) -> str:
        seed = (
            f"{self.spec.prototype_id}:{self.spec.surface}:"
            f"{self.spec.opcodes}:{self.spec.charged_byte_delta}:"
            f"{self.projected_total_delta_score:.12g}:{self.verdict}"
        )
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "percepta_microprogram_plan.v1",
            "plan_id": self.plan_id,
            "spec": self.spec.as_dict(),
            "verdict": self.verdict,
            "prototype_blockers": list(self.prototype_blockers),
            "promotion_blockers": list(self.promotion_blockers),
            "charged_rate_delta_score": self.charged_rate_delta_score,
            "projected_total_delta_score": self.projected_total_delta_score,
            "simple_edit_hurdle_delta_score": self.simple_edit_hurdle_delta_score,
            "cheapest_empirical_smoke": list(self.cheapest_empirical_smoke),
            "risk_register": list(self.risk_register),
        }


def rate_delta_for_bytes(byte_delta: int) -> float:
    """Return the score delta from a signed archive-byte delta."""

    return RATE_WEIGHT * float(byte_delta) / float(RATE_DENOMINATOR_BYTES)


def simple_edit_hurdle_delta(
    *,
    best_simple_edit_delta_score: float | None,
    byte_polish_floor_bytes: int = DEFAULT_BYTE_POLISH_FLOOR_BYTES,
) -> float:
    """Return the improvement a microprogram must beat before promotion."""

    byte_floor_delta = -rate_delta_for_bytes(abs(int(byte_polish_floor_bytes)))
    if best_simple_edit_delta_score is None:
        return byte_floor_delta
    return min(float(best_simple_edit_delta_score), byte_floor_delta)


def build_microprogram_plan(
    spec: MicroprogramPrototypeSpec,
    *,
    byte_polish_floor_bytes: int = DEFAULT_BYTE_POLISH_FLOOR_BYTES,
    max_micro_ops: int = DEFAULT_MAX_MICRO_OPS,
    max_program_bytes: int = DEFAULT_MAX_PROGRAM_BYTES,
    max_runtime_patch_bytes: int = DEFAULT_MAX_RUNTIME_PATCH_BYTES,
) -> MicroprogramPlan:
    """Classify a proposed Percepta-style tiny circuit for Pact use."""

    prototype_blockers = _prototype_blockers(
        spec,
        max_micro_ops=max_micro_ops,
        max_program_bytes=max_program_bytes,
        max_runtime_patch_bytes=max_runtime_patch_bytes,
    )
    charged_rate_delta = rate_delta_for_bytes(spec.charged_byte_delta)
    total_delta = float(spec.expected_component_delta_score) + charged_rate_delta
    hurdle = simple_edit_hurdle_delta(
        best_simple_edit_delta_score=spec.best_simple_edit_delta_score,
        byte_polish_floor_bytes=byte_polish_floor_bytes,
    )

    promotion_blockers: list[str] = []
    if prototype_blockers:
        promotion_blockers.append("prototype_blocked")
    if not spec.custody.ready():
        promotion_blockers.append("exact_eval_custody_missing")
    if total_delta >= hurdle:
        promotion_blockers.append("does_not_beat_simple_q_or_byte_edit_hurdle")

    if prototype_blockers:
        verdict: Verdict = "NO_GO"
    elif promotion_blockers:
        verdict = "PROTOTYPE_GO_PROMOTION_BLOCKED"
    else:
        verdict = "PROMOTION_GO"

    return MicroprogramPlan(
        spec=spec,
        verdict=verdict,
        prototype_blockers=tuple(prototype_blockers),
        promotion_blockers=tuple(promotion_blockers),
        charged_rate_delta_score=charged_rate_delta,
        projected_total_delta_score=total_delta,
        simple_edit_hurdle_delta_score=hurdle,
        cheapest_empirical_smoke=_cheapest_empirical_smoke(spec),
        risk_register=_risk_register(spec),
    )


def _prototype_blockers(
    spec: MicroprogramPrototypeSpec,
    *,
    max_micro_ops: int,
    max_program_bytes: int,
    max_runtime_patch_bytes: int,
) -> list[str]:
    blockers: list[str] = []
    if spec.surface == "general_wasm_interpreter":
        blockers.append("full_wasm_interpreter_not_byte_faithful")
    forbidden = sorted(set(spec.opcodes) - ALLOWED_TINY_OPCODES)
    if forbidden:
        blockers.append("forbidden_or_unbounded_opcode:" + ",".join(forbidden))
    if spec.operation_count > int(max_micro_ops):
        blockers.append("too_many_micro_ops")
    if spec.encoded_program_bytes > int(max_program_bytes):
        blockers.append("encoded_program_too_large")
    if spec.runtime_patch_bytes > int(max_runtime_patch_bytes):
        blockers.append("runtime_patch_too_large")
    if not spec.deterministic_inflate:
        blockers.append("inflate_not_deterministic")
    if not spec.scorer_free_inflate:
        blockers.append("inflate_loads_scorer")
    if not spec.network_free_inflate:
        blockers.append("inflate_uses_network_or_external_io")
    if spec.touches_pr110_live_files:
        blockers.append("touches_pr110_live_submission_files")
    return blockers


def _cheapest_empirical_smoke(spec: MicroprogramPrototypeSpec) -> tuple[str, ...]:
    """Return existing Pact commands for the first byte-closed smoke."""

    base = "experiments/results/percepta_microprogram_smoke"
    return (
        ".venv/bin/python tools/probe_op3v3_decoder_mutation_feasibility.py "
        f"--archive-bin {DEFAULT_PR110_SOURCE_MEMBER} "
        "--max-offsets-per-tensor 4 --deltas -1,1 "
        f"--output {base}/{spec.prototype_id}_feasibility.json",
        ".venv/bin/python tools/materialize_decoder_q_candidates.py "
        f"--archive-bin {DEFAULT_PR110_SOURCE_MEMBER} "
        f"--feasibility {base}/{spec.prototype_id}_feasibility.json "
        f"--limit 2 --output-dir {base}/{spec.prototype_id}_archives "
        f"--manifest-output {base}/{spec.prototype_id}_manifest.json",
        ".venv/bin/python tools/run_decoder_q_candidate_inflate_controls.py "
        f"--runtime-dir {DEFAULT_PR110_RUNTIME_DIR} "
        f"--candidate-root {base}/{spec.prototype_id}_archives "
        f"--baseline-raw {DEFAULT_PR110_BASELINE_RAW} "
        f"--output-root {base}/{spec.prototype_id}_inflate "
        f"--output {base}/{spec.prototype_id}_inflate_controls.json",
        ".venv/bin/python tools/run_decoder_q_candidate_advisory_batch.py "
        f"--runtime-dir {DEFAULT_PR110_RUNTIME_DIR} "
        f"--candidate-root {base}/{spec.prototype_id}_archives "
        f"--baseline-raw {DEFAULT_PR110_BASELINE_RAW} "
        f"--output-root {base}/{spec.prototype_id}_advisory "
        f"--axis-label '[macOS-CPU advisory]' --max-candidates 2 "
        f"--output {base}/{spec.prototype_id}_advisory_summary.json",
    )


def _risk_register(spec: MicroprogramPrototypeSpec) -> tuple[str, ...]:
    risks = [
        "Percepta supports program-as-weights plausibility, not any Pact score claim.",
        "Promotion requires byte-closed archive rebuild plus same-runtime exact eval custody.",
        "A decoder-side microprogram adds charged bytes and must outperform q-symbol or byte-only edits.",
        "Inflate must stay deterministic, scorer-free, and network-free.",
    ]
    if spec.surface == "decoder_side_microprogram":
        risks.append("Runtime patch LOC and byte overhead can dominate tiny correction benefits.")
    if spec.surface == "weight_embedded_circuit":
        risks.append("No runtime patch is safest, but the circuit must be proven consumed through changed raw output.")
    return tuple(risks)


def default_weight_embedded_probe_spec() -> MicroprogramPrototypeSpec:
    """Return a conservative no-claim starting spec for the FEC6 q lane."""

    return MicroprogramPrototypeSpec(
        prototype_id="percepta_weight_embedded_rgb_bias_gate_smoke",
        surface="weight_embedded_circuit",
        opcodes=("select_masked", "add_i8_saturating"),
        expected_component_delta_score=0.0,
        recompressed_archive_byte_delta=0,
        encoded_program_bytes=0,
        runtime_patch_bytes=0,
        notes="Zero-runtime first smoke: compile the 'program' into existing q-symbol choices.",
    )
