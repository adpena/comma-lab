"""Op 4: full-stack composition orchestrator.

Per the canonical four-way-stack composition contract
(``.omx/research/four_way_stack_composition_contract_20260507_claude.md``),
the four-way stack {Op 1 split-Brotli, Op 2 PR103 arithmetic codec,
Op 3 apogee_intN substrate, Op_GammaJointADMM, Op_SensitivityPreprocess (beta)}
landed as five independent :class:`CodecOp` instances.

Op 4 is the **composition orchestrator** that runs the canonical
composition matrix on a single input ``state_dict`` and reports per-stack
byte impact. It is NOT a new codec - it is a thin convenience wrapper
that builds N pipelines, encodes each one, and returns the
``EncodeResult`` for the final wrapper of every stack so operators can
reason about composability empirically.

Composition modes covered (canonical taxonomy from the contract memo):

  * ``substitutional`` - Op 1 alone, Op 2 alone, Op_GammaJointADMM alone:
    each emits an independent blob; comparing their ``bytes_out`` answers
    "which encoder is smallest on this state_dict?".
  * ``substrate-transform`` - ``[Op3, Op1]``, ``[Op3, Op2]``,
    ``[Op_SensitivityPreprocess, Op1]``: the upstream op decodes its
    blob immediately, and its reconstructed state_dict becomes the
    input to the downstream encoder. Per composition rule #2, the ops
    that opt in via ``transforms_state_dict=True`` (Op 3 and beta) make
    the substrate-transform composition genuinely stackable - the
    downstream encoder sees the lower-entropy substrate, not the
    original fp32 weights.
  * ``decorator`` (decode-side only) - Op 2.5 (PR102 inference tuning)
    runs at inflate time and is NOT part of the encode matrix. It is
    excluded from this orchestrator.

The matrix this op runs is fixed (per the parent task spec):

    Stack 0: [Op1]                                        substitutional
    Stack 1: [Op2]                                        substitutional alt
    Stack 2: [Op_GammaJointADMM]                          gamma alone
    Stack 3: [Op_SensitivityPreprocess.identity, Op1]     beta=identity (control)
    Stack 4: [Op3(int6), Op1]                             int6 -> split-Brotli
    Stack 5: [Op3(int6), Op2]                             int6 -> AC
    Stack 6: [Op3(int7), Op1]                             int7 (basin-parity PASS) variant
    Stack 7: [Op_SensitivityPreprocess.identity, Op3(int6), Op2]  beta + Op3 + Op2

Each stack roundtrips within the quantization grid of its constituent ops:
  * Op 1 / Op 2 alone: lossy at Op 1's / Op 2's internal fixed-grid
    representation (~int7-band; per-tensor rel-err <= ~1% on synthetic
    fp32 input).
  * Stacks containing Op 3: lossy at the int-N grid; recovery is exact
    on the dequantized substrate that Op 1 / Op 2 then re-encodes on
    its own grid.
  * Stacks containing Op_GammaJointADMM: lossy at the orchestrator's
    internal int8 grid; recovery is exact on the int8-rounded substrate.
  * Stacks containing beta=identity: passthrough (every tensor goes
    through the mid band unchanged before Op 1 / Op 2 / Op 3 encode it).

Strict-scorer-rule
------------------
Pure CPU. No scorer load. No CUDA / MPS. The orchestrator only invokes
the constituent ops' ``encode`` / ``decode`` / ``validate`` methods.

Score-claim discipline
----------------------
The orchestrator NEVER emits a ``[contest-CUDA]`` score. The manifest
JSON is tagged ``[empirical:<manifest path>]`` per CLAUDE.md - a record
of byte deltas measured on a synthetic state_dict. Any score derived
from these deltas would have to come from a contest-CUDA replay on a
real archive built from a real trained checkpoint.

Empirical receipts
------------------
Running ``run_composition_matrix`` on a synthetic state_dict (28-tensor
:data:`tac.pr101_split_brotli_codec.FIXED_STATE_SCHEMA`, seed=0,
scale=0.1) writes::

    experiments/results/lane_codec_pipeline_full_stack_<UTC>/
        composition_matrix.json

The JSON records each stack's name, op chain, ``bytes_out`` (final CPL1
wrapper bytes), and per-op byte deltas. Tagged
``[empirical:experiments/results/lane_codec_pipeline_full_stack_<UTC>/composition_matrix.json]``
with ``score_claim=False``.

References
----------
* Composition contract memo:
  ``.omx/research/four_way_stack_composition_contract_20260507_claude.md``
* CodecOp Protocol + CPL1 wire format: :mod:`tac.codec_pipeline`
* Op 1 / Op 2: :mod:`tac.codec_pipeline`
* Op 3: :mod:`tac.codec_pipeline_apogee_int`
* beta: :mod:`tac.codec_pipeline_sensitivity`
* gamma: :mod:`tac.codec_pipeline_joint_admm`
* alpha mask portfolio (separate pipeline, not in matrix):
  :mod:`tac.codec_pipeline_mask`
"""
from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tac.codec_pipeline import (
    CodecPipeline,
    EncodeResult,
    Op1_PR101SplitBrotli,
    Op2_PR103ArithmeticCodec,
    PipelineManifest,
    ValidationReport,
)

if TYPE_CHECKING:
    import torch

from tac.codec_pipeline_apogee_int import Op3_ApogeeIntN_Substrate
from tac.codec_pipeline_joint_admm import Op_GammaJointADMM
from tac.codec_pipeline_sensitivity import Op_SensitivityPreprocess

# ---------------------------------------------------------------------------
# Canonical matrix definition
# ---------------------------------------------------------------------------

#: Names of the eight canonical stacks - fixed per the parent task spec.
#: Order matches the task table; the run helper preserves this order so
#: downstream consumers can rely on positional access.
CANONICAL_STACK_NAMES: tuple[str, ...] = (
    "Op1_alone",
    "Op2_alone",
    "Op_GammaJointADMM_alone",
    "beta_identity_then_Op1",
    "Op3_int6_then_Op1",
    "Op3_int6_then_Op2",
    "Op3_int7_then_Op1",
    "beta_identity_then_Op3_int6_then_Op2",
)


def _build_canonical_pipelines() -> dict[str, CodecPipeline]:
    """Build the eight canonical stacks per the composition matrix.

    Each pipeline is a fresh instance - the orchestrator does not share
    op instances across pipelines (so per-pipeline ``op_state`` cannot
    leak across stacks). Op_GammaJointADMM is built with a small
    ``max_admm_iters`` so the matrix runs in seconds on CPU; the
    orchestrator is for empirical comparison, not production gamma
    encodes.

    Notes:
        * ``Op_SensitivityPreprocess.identity()`` is the true passthrough
          (every tensor lands in the mid band). The plain
          ``Op_SensitivityPreprocess()`` constructor (default
          ``low_threshold=0.1``) classifies every uniform-zero score as
          LOW which would int4-pre-quantize the substrate; that's not
          the "control" condition the task spec requires.
    """
    return {
        "Op1_alone": CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)]),
        "Op2_alone": CodecPipeline([Op2_PR103ArithmeticCodec()]),
        "Op_GammaJointADMM_alone": CodecPipeline(
            [Op_GammaJointADMM(max_admm_iters=2)]
        ),
        "beta_identity_then_Op1": CodecPipeline(
            [
                Op_SensitivityPreprocess.identity(),
                Op1_PR101SplitBrotli(auto_select=False),
            ]
        ),
        "Op3_int6_then_Op1": CodecPipeline(
            [
                Op3_ApogeeIntN_Substrate(bits=6),
                Op1_PR101SplitBrotli(auto_select=False),
            ]
        ),
        "Op3_int6_then_Op2": CodecPipeline(
            [
                Op3_ApogeeIntN_Substrate(bits=6),
                Op2_PR103ArithmeticCodec(),
            ]
        ),
        "Op3_int7_then_Op1": CodecPipeline(
            [
                Op3_ApogeeIntN_Substrate(bits=7),
                Op1_PR101SplitBrotli(auto_select=False),
            ]
        ),
        "beta_identity_then_Op3_int6_then_Op2": CodecPipeline(
            [
                Op_SensitivityPreprocess.identity(),
                Op3_ApogeeIntN_Substrate(bits=6),
                Op2_PR103ArithmeticCodec(),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Per-stack run record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StackRunRecord:
    """One row in the composition matrix.

    Attributes:
        stack_name: canonical name (one of :data:`CANONICAL_STACK_NAMES`).
        op_names: list of op ``name`` strings in pipeline order.
        bytes_out: length of the final CPL1 wrapper.
        per_op_bytes: list of ``(op_name, bytes_out)`` for each op's
            contribution (the op's own blob length, not the wrapper).
        manifest: the underlying :class:`PipelineManifest` (preserved so
            callers can introspect ``op_state`` / SHAs without re-running
            encode).
    """
    stack_name: str
    op_names: tuple[str, ...]
    bytes_out: int
    per_op_bytes: tuple[tuple[str, int], ...]
    manifest: PipelineManifest

    @property
    def final_blob_sha256(self) -> str:
        return self.manifest.final_blob_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack_name": self.stack_name,
            "op_names": list(self.op_names),
            "bytes_out": self.bytes_out,
            "per_op_bytes": [
                {"op": op_name, "bytes_out": n}
                for op_name, n in self.per_op_bytes
            ],
            "final_blob_sha256": self.final_blob_sha256,
        }


# ---------------------------------------------------------------------------
# Op 4 - composition orchestrator (CodecOp-shaped façade)
# ---------------------------------------------------------------------------

@dataclass
class Op4_FullStackOrchestrator:
    """Op 4: convenience wrapper that runs the canonical composition matrix.

    Op 4 is itself a :class:`CodecOp` so it can be dropped into a
    :class:`CodecPipeline` for forensic / reporting purposes. Its
    ``encode`` runs the canonical stack ``Op1_alone`` (the smallest
    stable substitutional baseline) and records the full matrix's
    composition record in ``op_state`` so the manifest carries the
    audit trail.

    Use :func:`run_composition_matrix` for the canonical "compare every
    stack on one state_dict" entry-point. Use Op4 inside a pipeline only
    when you want a single op identifier in the manifest that points at
    the matrix audit.

    Strict-scorer-rule: pure CPU; no scorer load; no CUDA / MPS.

    Notes:
        * Op 4 does NOT itself implement a substrate transform. Its
          ``encode`` delegates to ``Op1_PR101SplitBrotli`` so the
          ``decode`` path replays Op 1's bit-faithful inverse. The
          matrix audit (per-stack records) lives in ``op_state`` for
          observability only - decoders ignore it.
    """

    name: str = "full_stack_orchestrator"

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any] | None = None,
    ) -> EncodeResult:
        ctx = dict(context) if context is not None else {}
        # Delegate to Op1_alone for the "carry-through" blob (so decode
        # is bit-faithful). The matrix audit goes into op_state.
        delegate = Op1_PR101SplitBrotli(auto_select=False)
        delegate_result = delegate.encode(state_dict, context=ctx)
        records = run_composition_matrix(state_dict, write_manifest=False)
        op_state: dict[str, Any] = {
            "delegate_op_name": delegate.name,
            "delegate_op_state": dict(delegate_result.op_state),
            "matrix_records": [r.to_dict() for r in records.values()],
            "matrix_winner_stack_name": pick_smallest_stack(records),
        }
        return EncodeResult(
            blob=delegate_result.blob,
            bytes_in=delegate_result.bytes_in,
            bytes_out=delegate_result.bytes_out,
            op_name=self.name,
            op_state=op_state,
        )

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        # Replay the delegate (Op 1) bit-faithful decode. The matrix
        # audit in op_state is observability only; decoders ignore it.
        delegate = Op1_PR101SplitBrotli(auto_select=False)
        delegate_state = op_state.get("delegate_op_state", {}) or {}
        ctx = dict(context) if context is not None else {}
        return delegate.decode(blob, op_state=delegate_state, context=ctx)

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any] | None = None,
    ) -> ValidationReport:
        # Delegate to Op 1's validate - Op 4 is a Op 1 façade plus a
        # matrix audit, so its validation surface is Op 1's surface.
        delegate = Op1_PR101SplitBrotli(auto_select=False)
        ctx = dict(context) if context is not None else {}
        delegate_rep = delegate.validate(state_dict, context=ctx)
        return ValidationReport(
            passed=delegate_rep.passed,
            op_name=self.name,
            findings=list(delegate_rep.findings),
        )


# ---------------------------------------------------------------------------
# Top-level orchestrator helpers
# ---------------------------------------------------------------------------

def _per_op_bytes(manifest: PipelineManifest) -> tuple[tuple[str, int], ...]:
    return tuple((r.op_name, r.bytes_out) for r in manifest.op_results)


def run_composition_matrix(
    state_dict: dict[str, torch.Tensor],
    *,
    write_manifest: bool = True,
    output_dir: Path | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, StackRunRecord]:
    """Run all eight canonical stacks on ``state_dict`` and record byte impact.

    Args:
        state_dict: input HNeRV-shape state_dict (must satisfy the
            ``FIXED_STATE_SCHEMA`` shape contract for Op 1 / Op 2 / Op 3
            stacks). Op_GammaJointADMM accepts any non-empty tensor
            dict, so the gamma-alone stack runs even on partial inputs;
            the canonical matrix uses the same FIXED_STATE_SCHEMA input
            for cross-stack comparability.
        write_manifest: if ``True`` (default), persist a JSON manifest
            under ``experiments/results/lane_codec_pipeline_full_stack_<UTC>/``.
        output_dir: explicit output directory (overrides the default
            ``experiments/results/lane_codec_pipeline_full_stack_<UTC>/``).
            Useful for tests that want to write into a tmp_path.
        context: forwarded to every op's ``encode`` (e.g. for
            ``sensitivity_artifact``). Defaults to ``{}``.

    Returns:
        Dict keyed by stack name (one of :data:`CANONICAL_STACK_NAMES`)
        with :class:`StackRunRecord` values.

    Strict-scorer-rule: pure CPU; this helper never loads a scorer.

    Score-claim discipline: the manifest JSON is tagged
    ``[empirical:<path>]`` with ``score_claim=False``. Per CLAUDE.md
    "Forbidden score claims", no ``[contest-CUDA]`` score is emitted.
    """
    ctx = dict(context) if context is not None else {}
    pipelines = _build_canonical_pipelines()
    records: dict[str, StackRunRecord] = {}
    for stack_name in CANONICAL_STACK_NAMES:
        pipeline = pipelines[stack_name]
        blob, manifest = pipeline.encode(state_dict, context=ctx)
        records[stack_name] = StackRunRecord(
            stack_name=stack_name,
            op_names=tuple(pipeline.op_names),
            bytes_out=len(blob),
            per_op_bytes=_per_op_bytes(manifest),
            manifest=manifest,
        )

    if write_manifest:
        out_dir = output_dir or _default_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_composition_manifest(out_dir / "composition_matrix.json", records)

    return records


def pick_smallest_stack(records: Mapping[str, StackRunRecord]) -> str:
    """Return the canonical name of the stack with the smallest ``bytes_out``.

    Ties are broken by ``CANONICAL_STACK_NAMES`` order (i.e. the first
    stack with the minimal byte count wins). Returns the empty string
    iff ``records`` is empty.
    """
    best_name = ""
    best_bytes = -1
    for stack_name in CANONICAL_STACK_NAMES:
        if stack_name not in records:
            continue
        n = records[stack_name].bytes_out
        if best_bytes < 0 or n < best_bytes:
            best_bytes = n
            best_name = stack_name
    return best_name


def _default_output_dir() -> Path:
    """Default output dir under ``experiments/results/`` (NEVER ``/tmp``)."""
    repo_root = Path(__file__).resolve().parents[2]
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "experiments" / "results" / f"lane_codec_pipeline_full_stack_{ts}"


def _write_composition_manifest(
    path: Path,
    records: Mapping[str, StackRunRecord],
) -> None:
    """Persist the matrix as JSON with the canonical evidence-grade tag."""
    payload: dict[str, Any] = {
        "evidence_tag": f"[empirical:{path.as_posix()}]",
        "evidence_grade": "predicted",
        "score_claim": False,
        "note": (
            "Synthetic state_dict; this is a per-stack byte-impact record on a "
            "FIXED_STATE_SCHEMA-shaped fp32 input. NOT a contest-CUDA score "
            "claim. Per CLAUDE.md any score derived from these deltas requires "
            "a contest-CUDA replay on a real archive."
        ),
        "stacks": [records[name].to_dict() for name in CANONICAL_STACK_NAMES if name in records],
        "winner_stack_name": pick_smallest_stack(records),
        "winner_bytes_out": (
            records[pick_smallest_stack(records)].bytes_out
            if records else None
        ),
        "generated_at_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


__all__ = [
    "CANONICAL_STACK_NAMES",
    "Op4_FullStackOrchestrator",
    "StackRunRecord",
    "pick_smallest_stack",
    "run_composition_matrix",
]
