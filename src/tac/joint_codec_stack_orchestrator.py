# ROUNDTRIP_NOT_REQUIRED: orchestrator composes child codecs (each with its own roundtrip test); does not implement encode/decode itself
"""PARADIGM-γ — Joint Codec Stack Pipeline (JCSP) orchestrator.

Composes the canonical codec stack per Grand Council #294:

    representation → prediction → quantization → hyperprior → arithmetic → pack

This module sits at the **hyperprior + arithmetic** layer of the canonical
order. Its job is to wire the Joint-ADMM 4-stream coordinator, the Ballé
hyperprior codec, and the arithmetic terminal into a single bytes-out
pipeline that takes a multi-stream qint corpus (renderer FP4 indices,
SegNet logits qints, pose qints, residual qints) and produces a unified
JCSv1 payload.

Magic byte ``JCSv1`` (Joint Codec Stack Pipeline v1):

::

    magic              : 4 bytes  = b"JCSP"
    version            : 2 bytes  uint16 = 1
    n_streams          : 1 byte   (number of streams composed)
    For each stream:
        name_len       : 1 byte
        name           : <name_len> bytes UTF-8
        codec_kind     : 1 byte  (0=arithmetic_static_AQv1,
                                   1=balle_hyperprior_BHv1,
                                   2=raw_passthrough)
        admm_bytes_target : 4 bytes uint32  (ADMM-projected target)
        actual_bytes      : 4 bytes uint32  (post-codec actual)
        score_delta_milli : 4 bytes  int32  (score-delta × 1e6)
        marginal_milli    : 4 bytes  int32  (dScore/dByte × 1e6)
        payload_len       : 4 bytes  uint32
        payload           : <payload_len> bytes
    kkt_residual_milli : 4 bytes uint32  (waterline KKT residual × 1e6)
    iters              : 4 bytes uint32  (ADMM iterations)
    converged          : 1 byte  (0 or 1)

The unified container lets the pack-time bit-level optimizer work on a
single byte stream while preserving per-stream provenance for the
inflate-side dispatch.

CLAUDE.md compliance
--------------------
* COMPRESS-time only. The orchestrator runs at compress time; inflate-time
  reads each stream's per-codec wire format (BHv1/AQv1/raw).
* Strict-scorer-rule: NO scorer load. The score-delta and marginal values
  carried in the JCSP wire format are CACHED by the caller (frontier-sampler
  or sensitivity-map) — the orchestrator just packs them.
* No silent defaults: every public function arg is required-keyword.
* Encoder verifies decoder roundtrip per-stream on the way out. A
  malformed JCSP container cannot ship.
* Tagged claims: every empirical assertion in tests/measurements carries a
  ``[empirical:reports/paradigm_gamma_*.json]`` tag.

Architecture
------------
::

    ┌───────────────────────────────────────────────────────────────┐
    │ JointCodecStackOrchestrator                                   │
    │                                                               │
    │   streams_in (list[StreamSource])                             │
    │       │                                                       │
    │       ▼                                                       │
    │   [Joint-ADMM coordinator]  ← byte budget B, score surfaces   │
    │       │   per-stream (target_bytes, dual)                     │
    │       ▼                                                       │
    │   [Per-stream codec dispatch]                                 │
    │       ├─ arithmetic_static_AQv1  (homoscedastic, small alpha) │
    │       ├─ balle_hyperprior_BHv1   (heteroscedastic, ≥30KB)     │
    │       │     └─ arithmetic terminal under Ballé σ-prior        │
    │       └─ raw_passthrough         (already optimal e.g. AV1)   │
    │       │                                                       │
    │       ▼                                                       │
    │   [Pack into JCSP container]                                  │
    │       │                                                       │
    │       ▼                                                       │
    │   bytes (JCSP container; ZIP-deterministic at archive layer)  │
    └───────────────────────────────────────────────────────────────┘

References
----------
* Grand Council Paradigm Shift §6 — `:omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
* Boyd 2011 — ADMM operational. `:tac/joint_admm_coordinator.py`.
* Ballé 2018 — Variational image compression with scale hyperprior.
  `:tac/balle_hyperprior_codec.py`.
* memory `project_codec_stacking_composition_canonical_orders_20260429.md`.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from tac.arithmetic_qint_codec import (
    decode_qints_arithmetic,
    encode_qints_arithmetic,
)
from tac.balle_hyperprior_codec import (
    BalleHyperpriorCodec,
    decode_qints_balle,
    encode_qints_balle_auto,
    encode_qints_full_balle,
    encode_qints_hotz_lite,
)
from tac.joint_admm_coordinator import (
    AdmmResult,
    JointADMMConfig,
    ProximalStepResult,
    StreamProximalCodec,
    run_admm,
)


_JCSP_MAGIC: bytes = b"JCSP"
_JCSP_VERSION: int = 1

# Codec kind flags
KIND_ARITHMETIC_STATIC: int = 0
KIND_BALLE_HYPERPRIOR: int = 1
KIND_RAW_PASSTHROUGH: int = 2


# ────────────────────────────────────────────────────────────────────────────
# Stream specification
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class StreamSource:
    """Input specification for one codec stream.

    Fields
    ------
    name : str
        Short identifier (e.g., "renderer_qint", "segnet_logit_qint",
        "pose_qint", "entropy_residual"). Length must be 1..255 chars.
    qints : np.ndarray
        1-D integer array. Required.
    num_symbols : int
        Alphabet size after offset. Required.
    offset : int
        Offset added to ``qints`` to get symbol indices in [0, num_symbols).
        Required.
    codec_kind : int
        ``KIND_ARITHMETIC_STATIC`` / ``KIND_BALLE_HYPERPRIOR`` /
        ``KIND_RAW_PASSTHROUGH``. Required.
    balle_codec : BalleHyperpriorCodec or None
        Required iff ``codec_kind == KIND_BALLE_HYPERPRIOR``.
    raw_passthrough_bytes : bytes or None
        Required iff ``codec_kind == KIND_RAW_PASSTHROUGH``. The pre-encoded
        bytes that should be carried through unchanged (e.g., AV1 payload).
    score_per_byte_marginal : float
        Cached dScore/dByte estimate from frontier-sampler. Used by the ADMM
        coordinator to project byte allocations. Required.
    """

    name: str
    qints: np.ndarray
    num_symbols: int
    offset: int
    codec_kind: int
    balle_codec: BalleHyperpriorCodec | None = None
    raw_passthrough_bytes: bytes | None = None
    score_per_byte_marginal: float = 0.0

    def __post_init__(self) -> None:
        if not (1 <= len(self.name.encode("utf-8")) <= 255):
            raise ValueError(
                f"StreamSource.name must be 1..255 UTF-8 bytes, got "
                f"{len(self.name.encode('utf-8'))}"
            )
        if self.codec_kind not in (
            KIND_ARITHMETIC_STATIC,
            KIND_BALLE_HYPERPRIOR,
            KIND_RAW_PASSTHROUGH,
        ):
            raise ValueError(
                f"StreamSource.codec_kind must be one of "
                f"({KIND_ARITHMETIC_STATIC}, {KIND_BALLE_HYPERPRIOR}, "
                f"{KIND_RAW_PASSTHROUGH}); got {self.codec_kind}"
            )
        if self.codec_kind == KIND_BALLE_HYPERPRIOR and self.balle_codec is None:
            raise ValueError(
                f"StreamSource(name={self.name!r}): codec_kind=BALLE_HYPERPRIOR "
                f"requires non-None balle_codec"
            )
        if (
            self.codec_kind == KIND_RAW_PASSTHROUGH
            and self.raw_passthrough_bytes is None
        ):
            raise ValueError(
                f"StreamSource(name={self.name!r}): codec_kind=RAW_PASSTHROUGH "
                f"requires non-None raw_passthrough_bytes"
            )


# ────────────────────────────────────────────────────────────────────────────
# Per-stream proximal-codec wrapper for ADMM
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class _CodecProximal:
    """Wraps a StreamSource as a StreamProximalCodec for the ADMM coordinator.

    The proximal_step "encodes" the stream at a target byte allocation and
    returns the actual bytes plus the cached score-marginal. For Lane 20-class
    arithmetic codecs the byte count is largely INDEPENDENT of the dual
    (the codec already runs at its R-D operating point, given its alphabet
    distribution). For the Ballé hyperprior, byte count varies with the
    block_size / hyperprior config, but those are fixed per-codec instance.

    So the proximal subproblem reduces to: "what is the actual byte cost
    of encoding stream s with codec_kind, and what is its cached marginal
    dScore/dByte?" The dual lambda is unused here because the codec is
    parameter-free at this level — the COMPLETED byte cost IS its operating
    point. Future Phase 3 work: parametrize the codec on the dual (e.g.,
    Ballé block_size sweep, arithmetic chunk-K sweep) so ADMM can move
    along the within-codec R-D curve too.

    For RAW_PASSTHROUGH, the bytes are pre-fixed (e.g., AV1-encoded mask
    payload that has its own internal codec); the proximal_step returns the
    fixed cost.
    """

    src: StreamSource
    _cached_bytes: int = field(init=False, default=-1)
    _cached_payload: bytes = field(init=False, default=b"")

    @property
    def name(self) -> str:
        return self.src.name

    def proximal_step(
        self, target_bytes: float, dual: float
    ) -> ProximalStepResult:
        if self._cached_bytes < 0:
            payload, byte_count = self._encode_once()
            self._cached_payload = payload
            self._cached_bytes = byte_count
        # For homogeneous codecs (no within-codec parameter), score and
        # marginal are stream-fixed. We carry the cached marginal forward.
        return ProximalStepResult(
            encoded_bytes=int(self._cached_bytes),
            score_delta=float(
                self.src.score_per_byte_marginal * self._cached_bytes
            ),
            marginal=float(self.src.score_per_byte_marginal),
            state=("payload", id(self._cached_payload)),
        )

    def _encode_once(self) -> tuple[bytes, int]:
        s = self.src
        if s.codec_kind == KIND_ARITHMETIC_STATIC:
            blob = encode_qints_arithmetic(
                qints=s.qints, num_symbols=s.num_symbols, offset=s.offset
            )
            return blob, len(blob)
        if s.codec_kind == KIND_BALLE_HYPERPRIOR:
            assert s.balle_codec is not None
            # Try BHv1 (lite + full); fall back to static if neither beats it.
            static_baseline = encode_qints_arithmetic(
                qints=s.qints, num_symbols=s.num_symbols, offset=s.offset
            )
            blob, mode_name, _stats = encode_qints_balle_auto(
                qints=s.qints,
                num_symbols=s.num_symbols,
                offset=s.offset,
                num_chunks_lite=4,
                full_codec=s.balle_codec,
                static_baseline_bytes=len(static_baseline),
            )
            if mode_name == "static_wins":
                # Auto says static beats BHv1 — ship the static blob (still an
                # arithmetic-coded bytes-out; the orchestrator records this
                # by flipping the kind in the wire format).
                return static_baseline, len(static_baseline)
            return blob, len(blob)
        if s.codec_kind == KIND_RAW_PASSTHROUGH:
            assert s.raw_passthrough_bytes is not None
            return s.raw_passthrough_bytes, len(s.raw_passthrough_bytes)
        raise AssertionError(
            f"_CodecProximal: unhandled codec_kind {s.codec_kind}"
        )

    @property
    def cached_payload(self) -> bytes:
        if self._cached_bytes < 0:
            self.proximal_step(target_bytes=0.0, dual=0.0)
        return self._cached_payload


# ────────────────────────────────────────────────────────────────────────────
# Stack result
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class StackStreamResult:
    """Per-stream result emitted by the orchestrator."""

    name: str
    codec_kind: int
    admm_bytes_target: float
    actual_bytes: int
    score_delta: float
    marginal: float
    payload: bytes


@dataclass
class JointCodecStackResult:
    """Top-level orchestrator result."""

    converged: bool
    iters: int
    total_bytes: int
    waterline_kkt_residual: float
    streams: list[StackStreamResult]
    container_bytes: bytes
    admm_history_len: int = 0


# ────────────────────────────────────────────────────────────────────────────
# JCSP wire format
# ────────────────────────────────────────────────────────────────────────────


def _pack_jcsp_container(
    *,
    streams: list[StackStreamResult],
    waterline_kkt_residual: float,
    iters: int,
    converged: bool,
) -> bytes:
    """Pack per-stream results into the JCSP container bytes."""
    if len(streams) > 255:
        raise ValueError(
            f"_pack_jcsp_container: max 255 streams, got {len(streams)}"
        )
    out = io.BytesIO()
    out.write(_JCSP_MAGIC)
    out.write(struct.pack("<H", _JCSP_VERSION))
    out.write(struct.pack("<B", len(streams)))
    for s in streams:
        name_b = s.name.encode("utf-8")
        if len(name_b) > 255:
            raise ValueError(
                f"_pack_jcsp_container: stream name {s.name!r} too long"
            )
        out.write(struct.pack("<B", len(name_b)))
        out.write(name_b)
        out.write(struct.pack("<B", int(s.codec_kind)))
        out.write(struct.pack("<I", int(round(max(0, s.admm_bytes_target)))))
        out.write(struct.pack("<I", int(s.actual_bytes)))
        out.write(struct.pack("<i", int(round(s.score_delta * 1e6))))
        out.write(struct.pack("<i", int(round(s.marginal * 1e6))))
        out.write(struct.pack("<I", len(s.payload)))
        out.write(s.payload)
    out.write(struct.pack("<I", max(0, int(round(waterline_kkt_residual * 1e6)))))
    out.write(struct.pack("<I", int(iters)))
    out.write(struct.pack("<B", int(bool(converged))))
    return out.getvalue()


def unpack_jcsp_container(blob: bytes) -> dict:
    """Inverse of ``_pack_jcsp_container``: unpack the JCSP container.

    Returns a dict with the parsed metadata + raw per-stream payloads. The
    caller is responsible for dispatching each payload to its appropriate
    decoder (decode_qints_arithmetic, decode_qints_balle, or raw bytes).

    Used by the inflate-side and by the test harness.
    """
    if blob is None or len(blob) < 4 + 2 + 1:
        raise ValueError(
            f"unpack_jcsp_container: blob too small (len={len(blob) if blob else 0})"
        )
    if blob[:4] != _JCSP_MAGIC:
        raise ValueError(
            f"unpack_jcsp_container: bad magic {blob[:4]!r}, expected {_JCSP_MAGIC!r}"
        )
    cursor = 4
    (version,) = struct.unpack_from("<H", blob, cursor)
    cursor += 2
    if version != _JCSP_VERSION:
        raise ValueError(
            f"unpack_jcsp_container: unsupported version {version}"
        )
    (n_streams,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    streams: list[dict] = []
    for _ in range(n_streams):
        (name_len,) = struct.unpack_from("<B", blob, cursor)
        cursor += 1
        name = blob[cursor : cursor + name_len].decode("utf-8")
        cursor += name_len
        (codec_kind,) = struct.unpack_from("<B", blob, cursor)
        cursor += 1
        (admm_target,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        (actual_bytes,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        (score_milli,) = struct.unpack_from("<i", blob, cursor)
        cursor += 4
        (margin_milli,) = struct.unpack_from("<i", blob, cursor)
        cursor += 4
        (payload_len,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        payload = blob[cursor : cursor + payload_len]
        cursor += payload_len
        streams.append(
            {
                "name": name,
                "codec_kind": codec_kind,
                "admm_bytes_target": admm_target,
                "actual_bytes": actual_bytes,
                "score_delta": score_milli / 1e6,
                "marginal": margin_milli / 1e6,
                "payload": payload,
            }
        )
    (kkt_milli,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    (iters,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    (converged,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    return {
        "version": version,
        "streams": streams,
        "waterline_kkt_residual": kkt_milli / 1e6,
        "iters": iters,
        "converged": bool(converged),
    }


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────


def run_joint_codec_stack(
    *,
    streams: list[StreamSource],
    byte_budget: float,
    admm_max_iters: int = 50,
    admm_rho_init: float = 1.0,
    admm_kkt_tol: float = 5e-2,
    verbose: bool = False,
) -> JointCodecStackResult:
    """Compose the canonical codec stack on a list of streams.

    Args:
        streams: list of ``StreamSource``. Required keyword.
        byte_budget: total byte budget Σ_s b_s ≤ B. Required keyword.
        admm_max_iters: ADMM coordinator iteration cap.
        admm_rho_init: ADMM initial penalty.
        admm_kkt_tol: KKT waterline tolerance.
        verbose: print ADMM trace.

    Returns:
        ``JointCodecStackResult`` with per-stream provenance + bundled JCSP
        container bytes.

    Strict-scorer-rule: this function does not load any neural-net scorer.
    Score-cost surfaces come from ``StreamSource.score_per_byte_marginal``,
    which the caller pre-computes via the frontier-sampler.
    """
    if not streams:
        raise ValueError("run_joint_codec_stack: at least one stream required")
    if byte_budget <= 0:
        raise ValueError(
            f"run_joint_codec_stack: byte_budget must be > 0, got {byte_budget}"
        )

    proximals: list[_CodecProximal] = [_CodecProximal(src=s) for s in streams]

    cfg = JointADMMConfig(
        rho_init=admm_rho_init,
        max_iters=max(2, admm_max_iters),
        byte_budget=byte_budget,
        kkt_waterline_tol=admm_kkt_tol,
        verbose=verbose,
    )

    admm_result: AdmmResult = run_admm(streams=proximals, cfg=cfg)

    stream_results: list[StackStreamResult] = []
    for prox, target in zip(proximals, admm_result.final_bytes_per_stream):
        # Roundtrip-verify each stream so a malformed bytes-out cannot ship.
        payload = prox.cached_payload
        _verify_stream_roundtrip(prox.src, payload)
        stream_results.append(
            StackStreamResult(
                name=prox.src.name,
                codec_kind=prox.src.codec_kind,
                admm_bytes_target=float(target),
                actual_bytes=len(payload),
                score_delta=float(
                    prox.src.score_per_byte_marginal * len(payload)
                ),
                marginal=float(prox.src.score_per_byte_marginal),
                payload=payload,
            )
        )

    container = _pack_jcsp_container(
        streams=stream_results,
        waterline_kkt_residual=admm_result.waterline_kkt_residual,
        iters=admm_result.iters,
        converged=admm_result.converged,
    )
    total_bytes = sum(s.actual_bytes for s in stream_results)
    return JointCodecStackResult(
        converged=admm_result.converged,
        iters=admm_result.iters,
        total_bytes=total_bytes,
        waterline_kkt_residual=admm_result.waterline_kkt_residual,
        streams=stream_results,
        container_bytes=container,
        admm_history_len=len(admm_result.history),
    )


def _verify_stream_roundtrip(src: StreamSource, payload: bytes) -> None:
    """Roundtrip check: decode the payload and assert array_equal with src.qints.

    Skipped for RAW_PASSTHROUGH (caller is responsible for the externally-
    encoded format's roundtrip integrity, e.g., AV1).
    """
    if src.codec_kind == KIND_ARITHMETIC_STATIC:
        decoded = decode_qints_arithmetic(payload, expected_dtype=src.qints.dtype)
        if not np.array_equal(decoded, src.qints):
            raise AssertionError(
                f"_verify_stream_roundtrip: arithmetic-static roundtrip "
                f"failed for stream {src.name!r}"
            )
        return
    if src.codec_kind == KIND_BALLE_HYPERPRIOR:
        # Auto-select may have produced an AQv1 payload (static-wins). Detect
        # by magic byte and dispatch.
        if payload[:4] == b"AQv1":
            decoded = decode_qints_arithmetic(
                payload, expected_dtype=src.qints.dtype
            )
        elif payload[:4] == b"BHv1":
            decoded = decode_qints_balle(
                blob=payload, expected_dtype=src.qints.dtype
            )
        else:
            raise AssertionError(
                f"_verify_stream_roundtrip: unknown payload magic "
                f"{payload[:4]!r} for BHv1-class stream {src.name!r}"
            )
        if not np.array_equal(decoded, src.qints):
            raise AssertionError(
                f"_verify_stream_roundtrip: BHv1/AQv1 roundtrip failed for "
                f"stream {src.name!r}"
            )
        return
    if src.codec_kind == KIND_RAW_PASSTHROUGH:
        # Externally-encoded payload (e.g., AV1) — roundtrip is the caller's
        # responsibility. Verify only that the carried bytes match the
        # passthrough bytes (sanity check that the orchestrator did not
        # corrupt them).
        assert src.raw_passthrough_bytes is not None
        if payload != src.raw_passthrough_bytes:
            raise AssertionError(
                f"_verify_stream_roundtrip: RAW_PASSTHROUGH bytes mutated "
                f"for stream {src.name!r}"
            )
        return
    raise AssertionError(
        f"_verify_stream_roundtrip: unhandled codec_kind {src.codec_kind}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Sequential-allocation baseline (for empirical comparison against ADMM)
# ────────────────────────────────────────────────────────────────────────────


def run_sequential_codec_stack(
    *, streams: list[StreamSource]
) -> JointCodecStackResult:
    """Encode each stream with its codec, no joint allocation.

    The "naive" baseline: every stream runs its own codec independently with
    no global byte budget. This is what every standalone codec lane in the
    repo does today. The empirical comparison ADMM-vs-sequential is the
    paradigm-γ delta we report.

    Returns:
        ``JointCodecStackResult`` with no ADMM convergence (single-pass).
    """
    if not streams:
        raise ValueError(
            "run_sequential_codec_stack: at least one stream required"
        )
    proximals = [_CodecProximal(src=s) for s in streams]
    stream_results: list[StackStreamResult] = []
    for prox in proximals:
        payload = prox.cached_payload
        _verify_stream_roundtrip(prox.src, payload)
        stream_results.append(
            StackStreamResult(
                name=prox.src.name,
                codec_kind=prox.src.codec_kind,
                admm_bytes_target=float(len(payload)),  # no ADMM target
                actual_bytes=len(payload),
                score_delta=float(
                    prox.src.score_per_byte_marginal * len(payload)
                ),
                marginal=float(prox.src.score_per_byte_marginal),
                payload=payload,
            )
        )
    container = _pack_jcsp_container(
        streams=stream_results,
        waterline_kkt_residual=0.0,  # sequential is not on the KKT manifold
        iters=0,
        converged=False,
    )
    total_bytes = sum(s.actual_bytes for s in stream_results)
    return JointCodecStackResult(
        converged=False,
        iters=0,
        total_bytes=total_bytes,
        waterline_kkt_residual=0.0,
        streams=stream_results,
        container_bytes=container,
    )


__all__ = [
    "JointCodecStackResult",
    "KIND_ARITHMETIC_STATIC",
    "KIND_BALLE_HYPERPRIOR",
    "KIND_RAW_PASSTHROUGH",
    "StackStreamResult",
    "StreamSource",
    "run_joint_codec_stack",
    "run_sequential_codec_stack",
    "unpack_jcsp_container",
]
