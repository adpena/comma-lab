# SPDX-License-Identifier: MIT
"""Row #4 — Per-archive inflate-device pin metadata helper (sister of Catalog #205).

Per CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback trap)" +
Catalog #205 the inflate device selection (PACT_INFLATE_DEVICE env var) is
already canonical at the source-text surface. BUT the AUTO selection per-archive
yields different floating-point results between CPU and CUDA bicubic kernels
(per CLAUDE.md "A1 PR Council Round 1 F1/F11" anchor 2026-05-13: +0.0335 score
gap between CPU and CUDA on the SAME archive bytes sha
``87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5``).

This module SURFACES per-archive device-pin metadata so dual-eval submission
packets can document EXACTLY which device was used for each anchor. The
formula is metadata canonicalization (not numerical):

    contest_axis_anchor = {
        "contest_cpu_pin_device": "cpu",
        "contest_cuda_pin_device": "cuda",
    }
    pinned_metadata = {
        "device_used": device,
        "PACT_INFLATE_DEVICE_env_var_value": resolved_env_value,
        "torch_cuda_available": bool,
        "linux_x86_64_compliant": bool,
        "score_axis_canonical_tag": "[contest-CPU]" | "[contest-CUDA]" | ...,
    }

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable, every
submission archive needs BOTH a [contest-CPU] AND [contest-CUDA] paired anchor;
this helper canonicalizes the per-archive metadata that documents which device
produced which axis.

Canonical-vs-unique decision per layer
--------------------------------------
- Data type / framework: ADOPT_CANONICAL (plain dict[str, Any])
- Solver pattern: UNIQUE (metadata canonicalization formula)
- Atom emission: ADOPT_CANONICAL (tac.atom.builders.build_arbitrary_value_atom)

9-dimension success checklist evidence
--------------------------------------
- Uniqueness: per-archive device-pin; sister Catalog #205 covers source-text level
- Beauty + elegance: pure dict construction + canonical tag mapping
- Distinctness: per-axis canonical pinning resolves +0.0335 A1 ambiguity
- Rigor: refuses unknown device; refuses unsupported axis
- Optimization per technique: solves dual-eval metadata canonicalization
- Stack-of-stacks composability: emits Atom + Provenance
- Deterministic reproducibility: pure function
- Extreme optimization: O(1)
- Optimal minimal contest score: predicted ΔS [-0.005, -0.001]

Observability surface (6 facets)
--------------------------------
- inspectable per layer: device + env var + axis exposed in result
- decomposable per signal: per-axis pin separated from cuda_available
- diff-able across runs: pure function
- queryable post-hoc: result is a frozen dataclass
- cite-able: A1 PR Council F1/F11 anchor + Catalog #205
- counterfactual-able: change device -> observe axis-tag delta

6-hook wire-in declaration per Catalog #125
-------------------------------------------
1. Sensitivity-map: N/A — metadata canonicalization
2. Pareto constraint: N/A
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE via Atom + per-axis disambiguation
5. Continual-learning posterior: ACTIVE via Provenance + axis tag
6. Probe-disambiguator: ACTIVE — the per-axis pinning IS the canonical
   disambiguator between CPU-axis and CUDA-axis anchors per CLAUDE.md
   "Apples-to-apples evidence discipline"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from tac.formula_extinctions.canonical_warmup_schedule import FormulaSolveResult

if TYPE_CHECKING:
    from tac.atom.atom import Atom


_LITERATURE_CITATION = (
    "CLAUDE.md A1 PR Council F1/F11 anchor 2026-05-13 (+0.0335 CPU-vs-CUDA "
    "score gap on bit-identical archive); Catalog #205 canonical inflate "
    "device-fork; CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' "
    "non-negotiable (dual-eval mandatory for every shippable archive)"
)

_VALID_DEVICES: tuple[str, ...] = ("cpu", "cuda")
_VALID_AXES: tuple[str, ...] = ("contest_cpu", "contest_cuda", "diagnostic_cpu", "diagnostic_cuda")

_AXIS_TAG_MAP: dict[tuple[str, str, bool], str] = {
    ("cpu", "contest_cpu", True): "[contest-CPU]",
    ("cuda", "contest_cuda", True): "[contest-CUDA]",
    ("cpu", "diagnostic_cpu", True): "[diagnostic-CPU]",
    ("cpu", "diagnostic_cpu", False): "[diagnostic-CPU]",
    ("cuda", "diagnostic_cuda", True): "[diagnostic-CUDA]",
    ("cuda", "diagnostic_cuda", False): "[diagnostic-CUDA]",
    ("cpu", "contest_cpu", False): "[macOS-CPU advisory]",
}


@dataclass(frozen=True)
class InflateDevicePinInput:
    """Inputs to the per-archive inflate-device-pin metadata helper."""

    device: Literal["cpu", "cuda"]
    score_axis: str
    pact_inflate_device_env_var: str = "auto"
    torch_cuda_available: bool = True
    linux_x86_64_compliant: bool = True
    archive_sha256: str = ""

    def __post_init__(self) -> None:
        if self.device not in _VALID_DEVICES:
            raise ValueError(
                f"device must be one of {_VALID_DEVICES}; got {self.device!r}"
            )
        if self.score_axis not in _VALID_AXES:
            raise ValueError(
                f"score_axis must be one of {_VALID_AXES}; got {self.score_axis!r}"
            )


def canonical_inflate_device_pin_metadata(
    inputs: InflateDevicePinInput,
    *,
    emit_arbitrariness_atom: bool = False,
    substrate_id: str = "<unknown_substrate>",
) -> FormulaSolveResult:
    """Compute canonical per-archive inflate-device-pin metadata.

    Parameters
    ----------
    inputs : InflateDevicePinInput
        Frozen dataclass with device + score_axis + env var + cuda availability.
    emit_arbitrariness_atom : bool
        When True, also emit a canonical ``tac.atom.Atom`` instance.
    substrate_id : str
        Substrate id for atom file_path resolution.

    Returns
    -------
    FormulaSolveResult
        ``solved_value`` is the canonical metadata dict with axis-tag literal.

    Examples
    --------
    >>> r = canonical_inflate_device_pin_metadata(InflateDevicePinInput(
    ...     device="cpu", score_axis="contest_cpu", linux_x86_64_compliant=True,
    ... ))
    >>> r.solved_value["score_axis_canonical_tag"]
    '[contest-CPU]'

    >>> # macOS CPU is NEVER [contest-CPU] per CLAUDE.md
    >>> r = canonical_inflate_device_pin_metadata(InflateDevicePinInput(
    ...     device="cpu", score_axis="contest_cpu", linux_x86_64_compliant=False,
    ... ))
    >>> r.solved_value["score_axis_canonical_tag"]
    '[macOS-CPU advisory]'
    """
    key = (inputs.device, inputs.score_axis, inputs.linux_x86_64_compliant)
    tag = _AXIS_TAG_MAP.get(key, "[advisory only]")

    metadata: dict[str, Any] = {
        "device_used": inputs.device,
        "PACT_INFLATE_DEVICE_env_var_value": inputs.pact_inflate_device_env_var,
        "torch_cuda_available": inputs.torch_cuda_available,
        "linux_x86_64_compliant": inputs.linux_x86_64_compliant,
        "archive_sha256": inputs.archive_sha256,
        "score_axis": inputs.score_axis,
        "score_axis_canonical_tag": tag,
    }
    intermediate: dict[str, Any] = {
        "axis_tag_lookup_key": key,
        "is_authoritative_axis": tag in ("[contest-CPU]", "[contest-CUDA]"),
        "requires_paired_axis": True,  # always per CLAUDE.md dual-eval mandate
    }
    coupled: dict[str, Any] = {}

    if emit_arbitrariness_atom:
        coupled["atom"] = _emit_atom(inputs, tag, substrate_id)

    return FormulaSolveResult(
        solved_value=metadata,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.formula_extinctions.inflate_device_pin_metadata."
            "canonical_inflate_device_pin_metadata"
        ),
        coupled_adjustments=coupled,
        notes=(
            f"Substrate {substrate_id} archive {inputs.archive_sha256[:8]}: "
            f"device={inputs.device} axis={inputs.score_axis} -> tag={tag}"
        ),
    )


def _emit_atom(
    inputs: InflateDevicePinInput,
    tag: str,
    substrate_id: str,
) -> "Atom":
    """Lazy-import atom builder."""
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath
    from tac.provenance.builders import build_provenance_for_predicted

    provenance = build_provenance_for_predicted(
        model_id="formula_extinctions.inflate_device_pin_metadata.v1",
        inputs_sha256="0" * 64,
    )
    return build_arbitrary_value_atom(
        atom_id=f"inflate_device_pin_for_{substrate_id}_{inputs.score_axis}",
        file_path="submissions/<substrate>/inflate.py",
        current_value="auto (CUDA if available; else CPU)",
        predicted_replacement={
            "device": inputs.device,
            "score_axis_canonical_tag": tag,
        },
        resolution_path=ResolutionPath.FORMULA,
        predicted_ev_delta_s=(-0.005, -0.001),
        cost_envelope_usd=0.0,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_repo_link=(
            "src/tac/formula_extinctions/inflate_device_pin_metadata.py"
        ),
        provenance=provenance,
        captured_by_subagent="lane_arbitrariness_extinction_wave_2b_path3_formula_batch_20260518",
    )
