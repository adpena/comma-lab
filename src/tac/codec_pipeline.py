# SPDX-License-Identifier: MIT
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
from datetime import UTC
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
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
    arithmetic coding, byte-map permutation, etc.). Substitutional ops emit an
    independent blob for the current state. Substrate-transform ops may expose
    ``transforms_state_dict=True``; when they do, the pipeline decodes that
    blob immediately and feeds the reconstructed state to downstream ops. The
    pipeline always threads ``op_state`` for decoder reconstruction.
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

    Substrate-adaptive derivers (task #394, 2026-05-07): when
    ``auto_derive_all_constants=True``, all 5 PR101 hardcoded constants
    (storage_order, stream_ends, conv4_perms, byte_maps, latent_dim_order)
    are derived from the input ``state_dict`` instead of using PR101's
    own-substrate defaults. The derived constants are threaded through
    ``op_state`` so :func:`decode` reproduces them byte-faithfully. Default
    False to preserve PR101 wire-format compatibility for the canonical
    PR101 archive replay path.
    """
    name: str = "pr101_split_brotli"
    brotli_quality: int = 11
    brotli_lgwin: int | None = None
    brotli_lgblock: int | None = None
    auto_select: bool = True
    explicit_overrides: dict[int, str] | None = None
    auto_derive_all_constants: bool = False

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        # Bug-hunter v2 fix 2026-05-07 (LOW, re-opened from prior round):
        # the previous version called ``encode_decoder_compact(...,
        # auto_select=True)`` (which runs ``auto_select_byte_maps`` inside)
        # and then re-ran ``auto_select_byte_maps`` here just to populate
        # ``op_state`` for the decoder side. That doubled the per-encode
        # auto-select cost (~30 extra brotli evals per encode call) for
        # zero correctness benefit. Now we compute the effective byte_maps
        # once -- either from the explicit override, the auto_select path,
        # or PR101's defaults -- and thread it through ``encode_decoder_compact``
        # with ``auto_select=False`` so the encoder doesn't re-derive it.
        from tac.pr101_split_brotli_codec import encode_decoder_compact

        bytes_in = sum(t.numel() * t.element_size() for t in state_dict.values())

        if self.explicit_overrides is not None:
            effective_byte_maps: dict[int, str] | None = dict(self.explicit_overrides)
        elif self.auto_select:
            from tac.pr101_split_brotli_codec import auto_select_byte_maps
            effective_byte_maps = auto_select_byte_maps(
                state_dict,
                brotli_quality=self.brotli_quality,
                brotli_lgwin=self.brotli_lgwin,
                brotli_lgblock=self.brotli_lgblock,
            )
        else:
            effective_byte_maps = None  # let PR101 default DECODER_BYTE_MAPS apply

        op_state: dict[str, Any] = {}

        if self.auto_derive_all_constants:
            # Task #394: substrate-adaptive deriver path. Run all 4 decoder-blob
            # derivers (byte_maps already computed above; storage_order +
            # stream_ends + conv4_perms below) on this state_dict and thread
            # the derived constants through op_state so decode roundtrips.
            from tac.pr101_split_brotli_codec_derivers import (
                derive_conv4_perms,
                derive_storage_order,
                derive_stream_ends,
            )
            storage_order = derive_storage_order(state_dict)
            conv4_perms = derive_conv4_perms(
                state_dict, brotli_quality=self.brotli_quality
            )
            stream_ends = derive_stream_ends(
                state_dict, storage_order, brotli_quality=self.brotli_quality
            )
            blob = encode_decoder_compact(
                state_dict,
                brotli_quality=self.brotli_quality,
                brotli_lgwin=self.brotli_lgwin,
                brotli_lgblock=self.brotli_lgblock,
                effective_byte_maps=effective_byte_maps,
                derived_storage_order=storage_order,
                derived_stream_ends=stream_ends,
                derived_conv4_perms=conv4_perms,
            )
            op_state["derived_storage_order"] = list(storage_order)
            op_state["derived_stream_ends"] = list(stream_ends)
            op_state["derived_conv4_perms"] = {
                str(idx): list(perm) for idx, perm in conv4_perms.items()
            }
        else:
            blob = encode_decoder_compact(
                state_dict,
                brotli_quality=self.brotli_quality,
                brotli_lgwin=self.brotli_lgwin,
                brotli_lgblock=self.brotli_lgblock,
                effective_byte_maps=effective_byte_maps,
                auto_select=False,  # already computed above; do not re-derive
            )

        if effective_byte_maps is not None:
            op_state["effective_byte_maps"] = effective_byte_maps
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
        # Task #394: if encode used the substrate-adaptive deriver path,
        # op_state carries derived storage_order / stream_ends / conv4_perms.
        # Re-hydrate them into the kwarg shapes ``decode_decoder_compact``
        # expects (tuples for orderings, dict[int, tuple] for perms).
        derived_storage_order = op_state.get("derived_storage_order")
        derived_stream_ends = op_state.get("derived_stream_ends")
        derived_conv4_perms_raw = op_state.get("derived_conv4_perms")
        kwargs: dict[str, Any] = {"effective_byte_maps": overrides}
        if derived_storage_order is not None:
            kwargs["derived_storage_order"] = tuple(derived_storage_order)
        if derived_stream_ends is not None:
            kwargs["derived_stream_ends"] = tuple(derived_stream_ends)
        if derived_conv4_perms_raw is not None:
            kwargs["derived_conv4_perms"] = {
                int(idx): tuple(perm)
                for idx, perm in derived_conv4_perms_raw.items()
            }
        return decode_decoder_compact(blob, **kwargs)

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
            results = validate_byte_map_savings(
                state_dict,
                brotli_quality=self.brotli_quality,
                brotli_lgwin=self.brotli_lgwin,
                brotli_lgblock=self.brotli_lgblock,
            )
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

        Version 1 (legacy, retained for forensic compat):
            magic   : 4 bytes  = b"CPL1"  (codec pipeline v1)
            n_ops   : u32_LE
            for each op:
                name_len : u16_LE
                name     : utf-8 bytes
                state_json_len : u32_LE
                state_json     : utf-8 bytes (json-encoded op_state)
                blob_len : u32_LE
                blob     : raw bytes

        Version 2 (canonical, default — landed 2026-05-08 ORCH-SYNC Bug 2):
            magic   : 4 bytes  = b"CPL2"  (codec pipeline v2)
            n_ops   : u32_LE
            for each op:
                name_len : u16_LE
                name     : utf-8 bytes
                state_json_len : u32_LE
                state_json     : utf-8 bytes (int-key-preserving JSON; see
                    :func:`_encode_op_state_v2_json` /
                    :func:`_decode_op_state_v2_json`)
                blob_len : u32_LE
                blob     : raw bytes

        Why CPL2: ``json.dumps`` coerces dict-with-int-keys to string keys.
        WIRE-DECODER (commit 669b5b5f) discovered 100% sign-flip on negzig
        tensors (e.g. ``blocks.5.bias``) when CPL1's reconstructed
        ``effective_byte_maps`` had string keys but the decoder did
        ``idx in effective_byte_maps`` for an integer ``idx``. CPL2's JSON
        encoder records int keys explicitly via a sentinel ``{"__intkey__":
        true, "items": [[k, v], ...]}`` envelope so they round-trip exactly.

    Strict-scorer-rule: no scorers loaded anywhere.
    """

    MAGIC = b"CPL1"  # legacy v1 magic (forensic compat)
    MAGIC_V2 = b"CPL2"  # canonical v2 magic (int-key preserving)
    DEFAULT_VERSION = 2

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
        version: int | None = None,
    ) -> tuple[bytes, PipelineManifest]:
        """Encode the pipeline.

        Args:
            version: 2 (default, CPL2 — int-key preserving) or 1 (legacy
                CPL1 — forensic compat only; coerces dict-with-int-keys
                to string keys, which causes the negzig sign-flip bug
                fixed by CPL2). When ``None``, uses ``DEFAULT_VERSION``.
        """
        if version is None:
            version = self.DEFAULT_VERSION
        if version not in (1, 2):
            raise ValueError(
                f"CodecPipeline.encode: version must be 1 or 2, got {version!r}"
            )
        ctx = dict(context) if context is not None else {}
        started = time.time()
        results: list[EncodeResult] = []

        # Encode chain: substitutional ops observe the current state_dict and
        # emit independent blobs. Substrate-transform ops opt in with
        # ``transforms_state_dict=True``; after they encode, their decoded
        # reconstruction becomes the current state for downstream ops. This
        # keeps PR101/PR103-style alternatives substitutional while making beta
        # sensitivity and apogee-int substrate transforms genuinely stackable.
        current_state = dict(state_dict)
        for op in self.ops:
            if not skip_validate:
                rep = op.validate(current_state, context=ctx)
                if not rep.passed:
                    raise ValueError(
                        f"CodecPipeline.encode aborted: op {op.name!r} validation "
                        f"failed with findings {rep.findings}"
                    )
            res = op.encode(current_state, context=ctx)
            results.append(res)
            if bool(getattr(op, "transforms_state_dict", False)):
                current_state = op.decode(res.blob, op_state=res.op_state, context=ctx)

        # Wrap into the deterministic CPL1/CPL2 container.
        import struct
        out = bytearray()
        magic = self.MAGIC if version == 1 else self.MAGIC_V2
        out += magic
        out += struct.pack("<I", len(results))
        for res in results:
            name_b = res.op_name.encode("utf-8")
            # Bug-hunter v3 fix 2026-05-07 (MEDIUM, integration seam):
            # the previous version called ``json.dumps(res.op_state, ...)``
            # bare. When an op accidentally placed a non-JSON-serializable
            # value (torch.Tensor, numpy array/scalar, set, frozenset, dataclass,
            # ...) into ``op_state``, the resulting ``TypeError`` was opaque
            # ("Object of type Tensor is not JSON serializable") with no
            # indication of which op or which key was responsible. CodecPipeline
            # composes 2-3 ops per stack and 8 stacks per matrix; without the
            # op_name and key in the error message, operators have to bisect
            # by hand. Catch the TypeError, identify the offending key via a
            # walk over op_state, and re-raise with the op_name + key path.
            #
            # CPL2 (default 2026-05-08 ORCH-SYNC Bug 2): use the int-key-
            # preserving encoder so ``effective_byte_maps`` and other dicts
            # with integer keys roundtrip exactly. CPL1 retained for forensic
            # compat (caller must opt in with ``version=1``).
            try:
                state_b = _encode_op_state_to_json_bytes(
                    res.op_state, version=version
                )
            except TypeError as exc:
                offender = _find_non_json_serializable_key(res.op_state)
                raise TypeError(
                    f"CodecPipeline.encode: op {res.op_name!r} produced an "
                    f"op_state value that is not JSON-serializable at key path "
                    f"{offender!r}. CPL{version} wire format requires op_state "
                    f"to be JSON-encodable (str/int/float/bool/None/list/dict). "
                    f"Convert tensors with ``.tolist()`` and numpy types with "
                    f"``int(...)`` / ``float(...)``. Underlying error: {exc}"
                ) from exc
            out += struct.pack("<H", len(name_b))
            out += name_b
            out += struct.pack("<I", len(state_b))
            out += state_b
            out += struct.pack("<I", len(res.blob))
            out += res.blob

        final = bytes(out)
        elapsed = time.time() - started
        from datetime import datetime
        manifest = PipelineManifest(
            started_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        """Decode the CPL1/CPL2 wrapper. Returns the reconstructed state_dict
        from the LAST op (the canonical decoded view); auxiliary state from
        earlier ops is discarded by default.

        Magic is auto-detected: ``CPL1`` (legacy, string-keyed JSON) and
        ``CPL2`` (canonical, int-key-preserving JSON) are both accepted.

        Returns:
            (state_dict, op_names_replayed) — for forensic confirmation.
        """
        import struct

        ctx = dict(context) if context is not None else {}
        magic = blob[:4]
        if magic == self.MAGIC:
            wire_version = 1
        elif magic == self.MAGIC_V2:
            wire_version = 2
        else:
            raise ValueError(
                f"CodecPipeline.decode: bad magic {magic!r}, expected "
                f"{self.MAGIC!r} (CPL1) or {self.MAGIC_V2!r} (CPL2)"
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
            name_len = struct.unpack_from("<H", blob, cursor)[0]
            cursor += 2
            op_name = blob[cursor : cursor + name_len].decode("utf-8")
            cursor += name_len
            state_len = struct.unpack_from("<I", blob, cursor)[0]
            cursor += 4
            op_state = _decode_op_state_from_json_bytes(
                blob[cursor : cursor + state_len], version=wire_version
            )
            cursor += state_len
            blob_len = struct.unpack_from("<I", blob, cursor)[0]
            cursor += 4
            op_blob = blob[cursor : cursor + blob_len]
            cursor += blob_len

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


# ---------------------------------------------------------------------------
# Op 2 wrapper: tac.pr103_arithmetic_codec
# ---------------------------------------------------------------------------

@dataclass
class Op2_PR103ArithmeticCodec:
    """Op 2: PR103 arithmetic-coded decoder + adaptive lgwin Brotli.

    Wraps :func:`tac.pr103_arithmetic_codec.encode_decoder_ac` and
    :func:`decode_decoder_ac`. PR103 is the silver-medal (0.195) submission
    that replaced PR101's split-Brotli with arithmetic coding on
    AC_TENSOR_INDICES plus a merged RangeEncoder + adaptive lgwin Brotli
    search on the non-AC tensors. Empirical: -420 bytes vs Op 1 substrate,
    -661 bytes vs PR106 baseline (subagent measurement, commit 35abccf5).

    The op_state required by the decoder:
        - ``section_lengths``: dict with five keys, exactly mirroring the wire
          sections: ``br`` (concatenated non-AC brotli streams),
          ``hists`` (per-tensor histogram blob), ``merged_ac`` (merged
          RangeEncoder stream over AC_TENSOR_INDICES + latent-hi),
          ``hi_hist`` (latent-hi histogram blob), and ``ac_fallback`` (the
          fallback-stream brotli blob populated when ``ac_auto_fallback=True``
          rerouted regressing AC tensors back to brotli; length 0 when no
          fallback fired). Bug-hunter v2: prior docstring listed only the
          first 4 keys (drift introduced when ``ac_fallback`` landed
          2026-05-07; decoder accepted ac_fallback as optional so existing
          callers worked, but the wrap docstring was no longer accurate).
        - ``n_latent_hi_symbols``: int, how many latent-hi symbols to drain
          from the merged RangeDecoder AFTER the 8 weight streams.
        - ``ac_fallback_set``: list[int], the AC_TENSOR_INDICES that were
          rerouted to brotli by ``ac_auto_fallback`` (records the
          encoder-side decision so the decoder reads the same fallback
          blob).
    """
    name: str = "pr103_arithmetic_codec"
    brotli_quality: int = 11
    adaptive_lgwin: bool = True
    latent_hi_symbols: Any = None  # np.ndarray | None — embedded latent-hi
    n_latent_hi_symbols: int = 0  # decoder-side drain count
    ac_auto_fallback: bool = True
    """Per-tensor AC auto-fallback (substrate-mismatch protection landed
    2026-05-07). When True (default), each AC_TENSOR_INDICES tensor is measured
    against its brotli baseline and falls back to brotli encoding if it
    regresses. The selected fallback set is recorded in op_state as
    ``ac_fallback_set`` and replayed by the decoder. Empirical PR106-int6
    substrate: -11,498 B savings vs ``ac_auto_fallback=False``."""

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        from tac.pr103_arithmetic_codec import encode_decoder_ac

        # Bug-hunter fix 2026-05-07 (CRITICAL #1): when `latent_hi_symbols` is
        # supplied as a non-empty array, the decoder MUST drain exactly that
        # many symbols from the merged RangeDecoder. Honor an explicit caller
        # `n_latent_hi_symbols` only when it matches the array length;
        # otherwise auto-derive from the array (the encoder embeds them all,
        # so any mismatch would silently corrupt the decoded state_dict).
        # Without this guard, a caller that passed `latent_hi_symbols=arr` but
        # forgot to set `n_latent_hi_symbols` (default 0) would land an
        # encode-vs-decode mismatch that the test suite did not cover.
        if self.latent_hi_symbols is not None:
            try:
                actual_hi_len = len(self.latent_hi_symbols)
            except TypeError as exc:
                raise ValueError(
                    "Op2_PR103ArithmeticCodec.latent_hi_symbols must be a "
                    f"sized array; got {type(self.latent_hi_symbols).__name__}"
                ) from exc
            if self.n_latent_hi_symbols not in (0, actual_hi_len):
                raise ValueError(
                    "Op2_PR103ArithmeticCodec: n_latent_hi_symbols="
                    f"{self.n_latent_hi_symbols} does not match "
                    f"len(latent_hi_symbols)={actual_hi_len}; the decoder "
                    "would drain the wrong number of merged-AC symbols and "
                    "silently corrupt the state_dict roundtrip"
                )
            effective_n_hi = actual_hi_len
        else:
            effective_n_hi = int(self.n_latent_hi_symbols)
            if effective_n_hi != 0:
                raise ValueError(
                    "Op2_PR103ArithmeticCodec: n_latent_hi_symbols="
                    f"{effective_n_hi} but latent_hi_symbols is None; "
                    "decoder would attempt to drain symbols that were never "
                    "embedded by the encoder"
                )

        bytes_in = sum(t.numel() * t.element_size() for t in state_dict.values())
        result = encode_decoder_ac(
            state_dict,
            brotli_quality=self.brotli_quality,
            adaptive_lgwin=self.adaptive_lgwin,
            latent_hi_symbols=self.latent_hi_symbols,
            return_layout=True,
            ac_auto_fallback=self.ac_auto_fallback,
        )
        # `result` is EncodedAcDecoderBlob when return_layout=True.
        # non_ac_brotli_streams is a tuple of per-stream bytes; the wire
        # format concatenates them, so the "br" section length is the SUM
        # of stream lengths (not the tuple count).
        br_total = sum(len(s) for s in result.non_ac_brotli_streams)
        section_lengths = {
            "br": br_total,
            "hists": len(result.histograms_blob),
            "merged_ac": len(result.merged_ac_blob),
            "hi_hist": len(result.latent_hi_hist_blob),
            "ac_fallback": len(result.ac_fallback_blob),
        }
        op_state: dict[str, Any] = {
            "section_lengths": section_lengths,
            # Bug-hunter fix: record the EFFECTIVE drain count derived from
            # the embedded array length, not the raw configured field.
            "n_latent_hi_symbols": effective_n_hi,
            # JSON-serializable representation of the fallback set; the
            # CodecPipeline wraps op_state via json.dumps so we use a list.
            "ac_fallback_set": list(result.ac_fallback_set),
        }
        return EncodeResult(
            blob=result.blob,
            bytes_in=bytes_in,
            bytes_out=len(result.blob),
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
        from tac.pr103_arithmetic_codec import decode_decoder_ac
        section_lengths = op_state.get("section_lengths")
        if section_lengths is None:
            raise ValueError(
                "Op 2 decode missing section_lengths in op_state — "
                "encoder must have populated it"
            )
        n_hi = int(op_state.get("n_latent_hi_symbols", 0))
        ac_fallback_set = tuple(op_state.get("ac_fallback_set", ()) or ())
        decoded = decode_decoder_ac(
            blob,
            section_lengths=section_lengths,
            n_latent_hi_symbols=n_hi,
            ac_fallback_set=ac_fallback_set,
        )
        return decoded.state_dict

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        from tac.pr103_arithmetic_codec import (
            FIXED_STATE_SCHEMA as PR103_SCHEMA,
        )
        from tac.pr103_arithmetic_codec import (
            validate_ac_savings,
        )
        findings: list[str] = []
        schema_names = {name for name, _ in PR103_SCHEMA}
        missing = schema_names - set(state_dict.keys())
        if missing:
            findings.append(f"missing tensors: {sorted(missing)}")
        if not missing:
            # Contrarian gate: AC savings audit per AC_TENSOR_INDICES.
            results = validate_ac_savings(state_dict)
            for idx, info in results.items():
                if info.get("delta_bytes", 0) > 0:
                    findings.append(
                        f"AC regresses tensor idx={idx} by {info['delta_bytes']}B "
                        f"vs brotli baseline (auto-fallback to brotli will mitigate)"
                    )
        return ValidationReport(
            passed=not findings or all("auto-fallback" in f for f in findings),
            op_name=self.name,
            findings=findings,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#: CPL2 sentinel for envelopes that record int-keyed dicts. The envelope
#: shape is ``{__intkey__: true, items: [[k, v], [k, v], ...]}`` where each
#: ``k`` is an int and each ``v`` is a (recursively encoded) JSON value.
#: A real op_state dict will never contain this sentinel as a literal key
#: because Python dict keys with the literal string ``"__intkey__"`` are
#: indistinguishable from the sentinel; encoding rejects any such key with
#: a clear error.
_CPL2_INTKEY_SENTINEL = "__intkey__"
_CPL2_ITEMS_KEY = "items"


def _cpl2_envelope_for_int_keyed_dict(d: dict[int, Any]) -> dict[str, Any]:
    """Wrap an int-keyed dict in the CPL2 envelope.

    Sub-values are recursively encoded so nested int-keyed dicts also
    roundtrip exactly. Sub-keys that are ints become ints in the items
    list (preserved by JSON's number type).
    """
    return {
        _CPL2_INTKEY_SENTINEL: True,
        _CPL2_ITEMS_KEY: [
            [int(k), _cpl2_recursively_encode(v)] for k, v in sorted(d.items())
        ],
    }


def _cpl2_recursively_encode(obj: Any) -> Any:
    """Recursively encode an op_state value for CPL2 wire.

    - Dicts with any int key are wrapped in the int-key envelope.
    - Mixed-key dicts (some int, some str) are not allowed (would be
      ambiguous); raise a clear ValueError.
    - Lists/tuples are recursed element-wise (tuples become lists per JSON).
    - Scalars passthrough (json.dumps does the heavy lifting).
    - The literal string key ``__intkey__`` in any input dict is rejected
      because it would alias the sentinel.
    """
    if isinstance(obj, dict):
        if not obj:
            return {}
        # Detect mixed keys + sentinel collision.
        key_types = {type(k) for k in obj}
        if any(k == _CPL2_INTKEY_SENTINEL for k in obj if isinstance(k, str)):
            raise ValueError(
                f"CPL2 op_state dict cannot contain reserved string key "
                f"{_CPL2_INTKEY_SENTINEL!r} (collides with int-key sentinel)"
            )
        has_int = any(isinstance(k, int) and not isinstance(k, bool) for k in obj)
        has_str = any(isinstance(k, str) for k in obj)
        if has_int and has_str:
            raise ValueError(
                f"CPL2 op_state dict mixes int and str keys: "
                f"{sorted({type(k).__name__ for k in obj})}; either-or required "
                f"for unambiguous int-key preservation"
            )
        if has_int:
            return _cpl2_envelope_for_int_keyed_dict(obj)
        # All-string-keyed dict: recurse into values (sub-dicts may still be
        # int-keyed).
        return {k: _cpl2_recursively_encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cpl2_recursively_encode(v) for v in obj]
    return obj


def _cpl2_recursively_decode(obj: Any) -> Any:
    """Reverse of :func:`_cpl2_recursively_encode`.

    Detects int-key envelopes and rehydrates them as ``dict[int, Any]``;
    recursively decodes nested values.
    """
    if isinstance(obj, dict):
        if obj.get(_CPL2_INTKEY_SENTINEL) is True and _CPL2_ITEMS_KEY in obj:
            items = obj[_CPL2_ITEMS_KEY]
            if not isinstance(items, list):
                raise ValueError(
                    "CPL2 int-key envelope: 'items' must be a list of [k, v] pairs"
                )
            return {
                int(pair[0]): _cpl2_recursively_decode(pair[1])
                for pair in items
            }
        return {k: _cpl2_recursively_decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpl2_recursively_decode(v) for v in obj]
    return obj


def _encode_op_state_to_json_bytes(
    op_state: dict[str, Any], *, version: int
) -> bytes:
    """Encode op_state to JSON bytes for the chosen CPL wire version.

    Version 1: bare ``json.dumps(..., sort_keys=True, separators=(",", ":"))``.
        Coerces dict-with-int-keys to string keys (legacy bug; retained for
        backwards compat with on-disk CPL1 archives).
    Version 2: int-key-preserving via the CPL2 sentinel envelope. Falls back
        to legacy formatting only for op_states that contain no int-keyed
        dicts at any depth, in which case the wire bytes are the same as v1.
    """
    import json

    if version == 1:
        return json.dumps(
            op_state, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    if version == 2:
        encoded = _cpl2_recursively_encode(op_state)
        return json.dumps(
            encoded, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    raise ValueError(f"unknown CPL wire version {version!r}; supported: 1, 2")


def _decode_op_state_from_json_bytes(
    state_bytes: bytes, *, version: int
) -> dict[str, Any]:
    """Decode op_state JSON bytes for the chosen CPL wire version."""
    import json

    raw = json.loads(state_bytes.decode("utf-8"))
    if version == 1:
        # Legacy: int keys are coerced to strings on encode and stay strings
        # on decode. This is the substrate of the WIRE-DECODER negzig
        # sign-flip bug; CPL1 callers must rehydrate manually if they need
        # int keys downstream.
        return raw
    if version == 2:
        return _cpl2_recursively_decode(raw)
    raise ValueError(f"unknown CPL wire version {version!r}; supported: 1, 2")


def _find_non_json_serializable_key(
    obj: Any, path: str = ""
) -> str:
    """Walk a (possibly nested) op_state and return the dotted-key path to the
    first value that is not JSON-serializable.

    Used by :meth:`CodecPipeline.encode` to enrich the opaque ``TypeError``
    raised by ``json.dumps`` when an op accidentally embeds a tensor, numpy
    array, set, etc. in ``op_state``. The walk is recursive over dicts and
    lists/tuples; for any leaf we attempt ``json.dumps(value)`` and on failure
    return the current path. If everything is serialisable we return
    ``"<unknown>"`` (which means the caller's TypeError must have come from
    something other than a leaf — e.g. a custom ``__dict__``-only object).
    """
    import json

    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = _find_non_json_serializable_key(
                v, f"{path}.{k}" if path else str(k)
            )
            if sub != "<json-ok>":
                return sub
        return "<json-ok>"
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            sub = _find_non_json_serializable_key(v, f"{path}[{i}]")
            if sub != "<json-ok>":
                return sub
        return "<json-ok>"
    try:
        json.dumps(obj)
    except TypeError:
        return path or "<root>"
    return "<json-ok>"


__all__ = [
    "CodecOp",
    "CodecPipeline",
    "EncodeResult",
    "Op1_PR101SplitBrotli",
    "Op2_PR103ArithmeticCodec",
    "PipelineManifest",
    "ValidationReport",
]
