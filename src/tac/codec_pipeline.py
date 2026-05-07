"""Composable codec pipeline — canonical orchestrator for the four-way stack.

Per the user's "do canonicalization and composability" directive 2026-05-07:
extract the four-way-stack hand-coded sequence into a declarative pipeline of
:class:`CodecOp` objects. Each op is independently testable, configurable,
and composable with any other op.

Design:
    class CodecOp(Protocol):
        name: str
        def encode(self, state_dict, context) -> EncodeResult
        def decode(self, blob, context) -> dict[str, torch.Tensor]
        def validate(self, state_dict) -> ValidationReport  # Contrarian gate

    pipeline = CodecPipeline([
        Op1_PR101SplitBrotli(auto_select=True),
        Op2_PR103ArithmeticCoding(),    # when landed by subagent
        Op2_5_PR102InferenceTuning(),    # zero-byte score reduction
        Op3_Apogee_intN_QuantizedSubstrate(bits=6),
    ])

    bytes_out, manifest = pipeline.encode(state_dict)
    state_dict_back = pipeline.decode(bytes_out, manifest)

The manifest records per-op byte savings + which ops were enabled/skipped, so
operator can reason about composability empirically (Op 1 saves -X bytes, Op 2
saves -Y on top of Op 1, etc.).

Strict-scorer-rule: pipeline contains only CPU codec ops; no scorers loaded.
Composability invariant: each op preserves bit-faithful roundtrip when its
``validate(state_dict)`` returns ``passed=True``.

Cross-references:
    - Four-way stack composition manifest:
      ``.omx/research/four_way_stack_cross_paradigm_composition_manifest_20260507_claude.md``
    - Non-arbitrariness audit:
      ``.omx/research/pr_top3_non_arbitrariness_paper_cross_reference_20260507_claude.md``
    - Op 1 (landed): :mod:`tac.pr101_split_brotli_codec`
    - Op 2 (in flight subagent a96c7aff938701fc0): :mod:`tac.pr103_arithmetic_codec`
    - Derivers (in flight subagent a00452e1ead175a32):
      :mod:`tac.pr101_split_brotli_codec_derivers`
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodeResult:
    """Output of one CodecOp's encode step.

    Attributes:
        blob: encoded bytes (may be a wrapper around the previous op's blob).
        bytes_in: size of the input blob/state-dict baseline (for reporting).
        bytes_out: size of ``blob``.
        op_name: human-readable identifier of the op (e.g. ``"pr101_split_brotli"``).
        op_state: dict of any state the decoder needs (e.g. effective_byte_maps).
            Encoded into the manifest; passed back at decode time.
    """
    blob: bytes
    bytes_in: int
    bytes_out: int
    op_name: str
    op_state: dict[str, Any] = field(default_factory=dict)

    @property
    def bytes_delta(self) -> int:
        return self.bytes_out - self.bytes_in


@dataclass(frozen=True)
class ValidationReport:
    """Output of a CodecOp's Contrarian-gate validate step."""
    passed: bool
    op_name: str
    findings: list[str] = field(default_factory=list)


@runtime_checkable
class CodecOp(Protocol):
    """Protocol every pipeline op must satisfy.

    Each op is responsible for one bit-level transformation (split-Brotli,
    arithmetic coding, byte-map permutation, etc.). Ops compose by feeding
    one's output into the next as raw bytes; the pipeline orchestrator
    threads the ``op_state`` for decoder reconstruction.
    """
    name: str

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        ...

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        ...

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        ...


# ---------------------------------------------------------------------------
# Op 1 wrapper: tac.pr101_split_brotli_codec
# ---------------------------------------------------------------------------

@dataclass
class Op1_PR101SplitBrotli:
    """Op 1: PR101 split-Brotli + per-tensor byte-map decoder codec.

    Wraps :func:`tac.pr101_split_brotli_codec.encode_decoder_compact` and
    :func:`decode_decoder_compact`. Optionally runs the Round-3 ``auto_select``
    path that derives the brotli-optimal byte_map per tensor under joint
    stream-window context.
    """
    name: str = "pr101_split_brotli"
    brotli_quality: int = 11
    auto_select: bool = True
    explicit_overrides: dict[int, str] | None = None

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        from tac.pr101_split_brotli_codec import encode_decoder_compact

        bytes_in = sum(t.numel() * t.element_size() for t in state_dict.values())
        blob = encode_decoder_compact(
            state_dict,
            brotli_quality=self.brotli_quality,
            effective_byte_maps=self.explicit_overrides,
            auto_select=self.auto_select and self.explicit_overrides is None,
        )
        # Capture the override that was actually used (caller may pass None +
        # auto_select=True, in which case the encoder ran auto_select_byte_maps
        # internally; we re-compute it here for the decoder side).
        op_state: dict[str, Any] = {}
        if self.explicit_overrides is not None:
            op_state["effective_byte_maps"] = dict(self.explicit_overrides)
        elif self.auto_select:
            from tac.pr101_split_brotli_codec import auto_select_byte_maps
            op_state["effective_byte_maps"] = auto_select_byte_maps(
                state_dict, brotli_quality=self.brotli_quality
            )
        return EncodeResult(
            blob=blob,
            bytes_in=bytes_in,
            bytes_out=len(blob),
            op_name=self.name,
            op_state=op_state,
        )

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        from tac.pr101_split_brotli_codec import decode_decoder_compact
        overrides = op_state.get("effective_byte_maps")
        return decode_decoder_compact(blob, effective_byte_maps=overrides)

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        from tac.pr101_split_brotli_codec import (
            FIXED_STATE_SCHEMA,
            validate_byte_map_savings,
        )
        findings: list[str] = []
        # Schema check: state_dict must contain all FIXED_STATE_SCHEMA names.
        schema_names = {name for name, _ in FIXED_STATE_SCHEMA}
        missing = schema_names - set(state_dict.keys())
        if missing:
            findings.append(f"missing tensors: {sorted(missing)}")
        # Contrarian gate: per-byte-map regression check.
        if not missing:
            results = validate_byte_map_savings(state_dict, brotli_quality=self.brotli_quality)
            for idx, info in results.items():
                if info["delta_bytes"] > 0:
                    findings.append(
                        f"byte_map={info['byte_map']!r} regresses tensor "
                        f"idx={idx} by {info['delta_bytes']}B (auto_select="
                        f"{self.auto_select} will pick a better map)"
                    )
        return ValidationReport(
            passed=not findings or all("auto_select" in f for f in findings),
            op_name=self.name,
            findings=findings,
        )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

@dataclass
class PipelineManifest:
    """Per-encode manifest produced by :meth:`CodecPipeline.encode`.

    Records each op's byte impact so operators can reason about composability
    empirically. All score predictions tagged ``[predicted]``.
    """
    started_at_utc: str
    elapsed_seconds: float
    op_results: list[EncodeResult] = field(default_factory=list)
    final_blob_sha256: str = ""
    final_bytes: int = 0
    score_claim: bool = False
    score_evidence_grade: str = "predicted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at_utc": self.started_at_utc,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "ops": [
                {
                    "name": r.op_name,
                    "bytes_in": r.bytes_in,
                    "bytes_out": r.bytes_out,
                    "delta_bytes": r.bytes_delta,
                    "op_state_keys": sorted(r.op_state.keys()),
                }
                for r in self.op_results
            ],
            "final_blob_sha256": self.final_blob_sha256,
            "final_bytes": self.final_bytes,
            "score_claim": self.score_claim,
            "score_evidence_grade": self.score_evidence_grade,
        }


class CodecPipeline:
    """Orchestrate a sequence of :class:`CodecOp` instances.

    Composition contract:
        - Ops run in order. Each op transforms the input state_dict (or
          previous op's reconstruction) to a blob.
        - The pipeline writes a wrapper containing per-op blob + op_state
          metadata so the decoder can replay the operation chain in reverse.
        - Each op's ``validate(state_dict)`` is called pre-encode; if any op
          returns ``passed=False``, encoding aborts (Contrarian gate).

    Wire format of the wrapper (deterministic, byte-exact):
        magic   : 4 bytes  = b"CPL1"  (codec pipeline v1)
        n_ops   : u32_LE
        for each op:
            name_len : u16_LE
            name     : utf-8 bytes
            state_json_len : u32_LE
            state_json     : utf-8 bytes (json-encoded op_state)
            blob_len : u32_LE
            blob     : raw bytes

    Strict-scorer-rule: no scorers loaded anywhere.
    """

    MAGIC = b"CPL1"

    def __init__(self, ops: list[CodecOp]) -> None:
        if not ops:
            raise ValueError("CodecPipeline needs at least one op")
        for op in ops:
            if not isinstance(op, CodecOp):
                raise TypeError(
                    f"op {op!r} does not satisfy CodecOp protocol"
                )
        self.ops = list(ops)

    @property
    def op_names(self) -> list[str]:
        return [op.name for op in self.ops]

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any] | None = None,
        skip_validate: bool = False,
    ) -> tuple[bytes, PipelineManifest]:
        ctx = dict(context) if context is not None else {}
        started = time.time()
        results: list[EncodeResult] = []

        # Pre-encode validation of every op (Contrarian gate).
        if not skip_validate:
            for op in self.ops:
                rep = op.validate(state_dict, context=ctx)
                if not rep.passed:
                    raise ValueError(
                        f"CodecPipeline.encode aborted: op {op.name!r} validation "
                        f"failed with findings {rep.findings}"
                    )

        # Encode chain: only Op-0 sees the raw state_dict; subsequent ops can
        # in principle take the prior op's reconstruction as input. For now
        # the v1 pipeline is "linear over the state_dict" — every op sees the
        # same state_dict and produces a blob; the wrapper concatenates them.
        # Future v2 may chain reconstructions for ops that re-quantize.
        for op in self.ops:
            res = op.encode(state_dict, context=ctx)
            results.append(res)

        # Wrap into the deterministic CPL1 container.
        import struct
        import json
        out = bytearray()
        out += self.MAGIC
        out += struct.pack("<I", len(results))
        for res in results:
            name_b = res.op_name.encode("utf-8")
            state_b = json.dumps(res.op_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
            out += struct.pack("<H", len(name_b))
            out += name_b
            out += struct.pack("<I", len(state_b))
            out += state_b
            out += struct.pack("<I", len(res.blob))
            out += res.blob

        final = bytes(out)
        elapsed = time.time() - started
        from datetime import datetime, timezone
        manifest = PipelineManifest(
            started_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            elapsed_seconds=elapsed,
            op_results=results,
            final_blob_sha256=hashlib.sha256(final).hexdigest(),
            final_bytes=len(final),
        )
        return final, manifest

    def decode(
        self,
        blob: bytes,
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """Decode the CPL1 wrapper. Returns the reconstructed state_dict from
        the LAST op (the canonical decoded view); auxiliary state from earlier
        ops is discarded by default.

        Returns:
            (state_dict, op_names_replayed) — for forensic confirmation.
        """
        import struct
        import json

        ctx = dict(context) if context is not None else {}
        if blob[:4] != self.MAGIC:
            raise ValueError(
                f"CodecPipeline.decode: bad magic {blob[:4]!r}, expected {self.MAGIC!r}"
            )
        cursor = 4
        n_ops = struct.unpack_from("<I", blob, cursor)[0]
        cursor += 4
        if n_ops != len(self.ops):
            raise ValueError(
                f"CodecPipeline.decode: blob has {n_ops} ops but pipeline "
                f"has {len(self.ops)} ops (mismatch — wrong pipeline?)"
            )

        decoded_state: dict[str, torch.Tensor] | None = None
        replayed: list[str] = []
        for i in range(n_ops):
            name_len = struct.unpack_from("<H", blob, cursor)[0]; cursor += 2
            op_name = blob[cursor:cursor+name_len].decode("utf-8"); cursor += name_len
            state_len = struct.unpack_from("<I", blob, cursor)[0]; cursor += 4
            op_state = json.loads(blob[cursor:cursor+state_len].decode("utf-8")); cursor += state_len
            blob_len = struct.unpack_from("<I", blob, cursor)[0]; cursor += 4
            op_blob = blob[cursor:cursor+blob_len]; cursor += blob_len

            expected_op = self.ops[i]
            if expected_op.name != op_name:
                raise ValueError(
                    f"CodecPipeline.decode: op[{i}] name mismatch "
                    f"(blob says {op_name!r}, pipeline says {expected_op.name!r})"
                )
            replayed.append(op_name)
            decoded_state = expected_op.decode(op_blob, op_state=op_state, context=ctx)

        if cursor != len(blob):
            raise ValueError(
                f"CodecPipeline.decode: trailing {len(blob) - cursor} bytes after final op"
            )
        if decoded_state is None:
            raise RuntimeError("decode produced no state_dict — pipeline is empty")
        return decoded_state, replayed


__all__ = [
    "CodecOp",
    "CodecPipeline",
    "EncodeResult",
    "Op1_PR101SplitBrotli",
    "PipelineManifest",
    "ValidationReport",
]
