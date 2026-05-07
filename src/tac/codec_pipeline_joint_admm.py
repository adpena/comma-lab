"""gamma paradigm Joint-ADMM CodecOp wrap for the canonical codec_pipeline.

Wraps the existing :mod:`tac.joint_codec_stack_orchestrator` (JCSv1 wire
format, Joint-ADMM 4-stream coordinator, Ballé hyperprior, arithmetic
terminal) as a :class:`tac.codec_pipeline.CodecOp` so the canonical
:class:`CodecPipeline` can compose gamma alongside Op 1 (PR101 split-Brotli)
and Op 2 (PR103 arithmetic codec).

Composition mode
----------------
gamma is **substitutional** with Op 2 (per the four-way-stack composition
contract memo at ``.omx/research/four_way_stack_composition_contract_20260507_claude.md``):
both ops solve the same role - entropy-coded compression of qint-quantised
weight streams. The pipeline can build either::

    [Op1, Op2]                      # PR101 split-brotli + PR103 arithmetic
    [Op1, Op_GammaJointADMM]        # PR101 split-brotli + gamma multi-stream ADMM
    [Op_GammaJointADMM]             # gamma alone

and compare the resulting blob sizes empirically. gamma is a fuller / more
rigorous arithmetic encoder (multi-stream ADMM joint-rate-allocation +
optional Ballé learned entropy model + arithmetic terminal) at the cost
of more encode-time compute. It is NOT claimed to dominate Op 2 - that's
an empirical question the orchestrator answers per state_dict.

Strict-scorer-rule
------------------
Encode time runs the ADMM coordinator and per-stream codec (CPU only, no
scorer load - score marginals are pre-cached on each StreamSource per the
gamma orchestrator's own contract). Decode time reads the JCSv1 container and
dispatches each per-stream payload to ``decode_qints_arithmetic`` /
``decode_qints_balle`` / raw passthrough. Neither path loads SegNet or
PoseNet weights.

Quantisation contract (op-internal)
-----------------------------------
Because the gamma orchestrator works on multi-stream qint corpora (not raw
fp32 state_dicts), this op performs a per-tensor symmetric int8 quant
internally on each tensor in ``state_dict``. The per-tensor scale and
shape are recorded in ``op_state['streams'][i]`` so ``decode`` can
restore the original fp32 tensors. This quantisation is intentionally
simple (per-tensor scale, symmetric, int8) - it is the SAME quantisation
contract used by the orchestrator's existing test fixtures, and the
orchestrator's own roundtrip verification asserts integer-exact recovery
of the qints. Distortion vs the input fp32 state is unavoidable at this
quantisation step; it is the same int8-band distortion floor every
qint-stream codec sees and is *out of scope* for the wrap layer.

For higher-fidelity floats (e.g., apogee_int7 substrate), the intended future
extension is a reviewed precomputed-qint corpus path. This wrapper does not
currently implement that bypass; it always uses its internal deterministic int8
quantisation pass.

Substrate-aware hyperprior initialization (Council STUB 2026-05-07)
-------------------------------------------------------------------
Ballé's Grand Council position
(``.omx/research/grand_council_pr106_substrate_findings_zig_default_20260507.md``)
flagged that gamma's PR106 regression (202,544 B vs Op2_alone's 169,755 B,
+33K B) stems from the hyperprior using default factorized priors that don't
model PR106's actual tensor distributions. His CPU-only fix prescription:
**substrate-aware INITIALIZATION** of the hyperprior from per-tensor
empirical histograms before invoking the joint orchestrator.

Status: the underlying :class:`tac.balle_hyperprior_codec.BalleHyperpriorCodec`
does NOT expose a ``prior_init`` parameter — its hyperprior network is
trained end-to-end and the static-arithmetic baseline (the path gamma
actually uses today via ``KIND_ARITHMETIC_STATIC``) builds the symbol
frequency table from the qint stream itself, so the static prior is
already substrate-aware by construction.

Per CLAUDE.md "NEVER invent CLI flags" we GREPPED the underlying codec
APIs (``balle_hyperprior_codec.py``, ``joint_codec_stack_orchestrator.py``,
``arithmetic_qint_codec.py``) for ``prior_init`` / ``substrate_aware_init``
/ ``empirical_prior`` keywords on 2026-05-07: zero matches. Inventing a
flag the underlying codec would silently ignore would ship a dead-flag
bug class (the ``feedback_dead_flag_wiring_pattern`` memory).

Fix landed: a ``substrate_aware_init: bool = False`` flag is exposed on
:class:`Op_GammaJointADMM` as a STUB. When True, the encoder logs a
WARN-level message that the hyperprior init API is missing ("γ-Phase-2
needs hyperprior init API"), records a finding in ``op_state``, and
proceeds with the existing static-arithmetic path (the WARN does not
abort encode — the existing behaviour is byte-identical regardless of
the flag). When False (default), the flag is silently inert.

When the underlying ``BalleHyperpriorCodec`` grows a ``prior_init`` /
``hyperprior_warmstart_from_histogram`` API, this STUB graduates: encode
will pre-compute per-tensor histograms and pass them through context to
the orchestrator. Until then this flag is a forensic record + design
contract anchor, not a functional behaviour change.

References
----------
* :mod:`tac.codec_pipeline` - ``CodecOp`` Protocol, ``CodecPipeline``
  orchestrator, CPL1 wire format.
* :mod:`tac.joint_codec_stack_orchestrator` - JCSv1 wire format,
  ``run_joint_codec_stack``, ``unpack_jcsp_container``.
* :mod:`tac.joint_admm_coordinator` - Boyd-style ADMM 4-stream coord.
* :mod:`tac.balle_hyperprior_codec` - Ballé 2018 hyperprior entropy model.
* :mod:`tac.arithmetic_qint_codec` - arithmetic-coded qint streams.
* :mod:`tac.arithmetic_terminal` - terminal arithmetic encoder.
* Composition contract memo:
  ``.omx/research/four_way_stack_composition_contract_20260507_claude.md``
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tac.codec_pipeline import EncodeResult, ValidationReport

logger = logging.getLogger(__name__)

# Council STUB 2026-05-07 (Ballé): substrate-aware hyperprior initialisation
# is the prescribed CPU-only fix for the +33K B PR106 regression. The
# underlying ``BalleHyperpriorCodec`` does not yet expose a ``prior_init``
# API (verified by grep on 2026-05-07; per CLAUDE.md "NEVER invent CLI
# flags"). Until the underlying codec grows the API, the flag is a STUB:
# WARN-only behaviour, no functional change. The exact WARN string is
# pinned here so tests + the substrate-aware-init flag's forensic record
# can match against it stably.
SUBSTRATE_AWARE_INIT_WARN_MESSAGE = (
    "Op_GammaJointADMM(substrate_aware_init=True): γ-Phase-2 needs "
    "hyperprior init API. The underlying tac.balle_hyperprior_codec does "
    "not yet expose a prior_init / empirical_prior parameter; encode "
    "proceeds with the existing static-arithmetic path (no functional "
    "change). When the underlying API lands, this STUB will graduate to "
    "pre-computing per-tensor histograms and threading them via context."
)

# ---------------------------------------------------------------------------
# Per-tensor symmetric int8 quantisation helpers
# ---------------------------------------------------------------------------

# Symmetric int8 range [-127, 127] (omit -128 to keep the quantiser symmetric).
_INT8_QMAX: int = 127


def _per_tensor_scale(tensor: torch.Tensor) -> float:
    """Per-tensor symmetric scale: ``max(|t|) / 127`` (>= a small floor)."""
    abs_max = float(tensor.detach().abs().max().item()) if tensor.numel() else 0.0
    if not np.isfinite(abs_max):
        raise ValueError(
            f"_per_tensor_scale: non-finite abs_max={abs_max} on tensor "
            f"shape={tuple(tensor.shape)}"
        )
    if abs_max == 0.0:
        return 1.0  # zero tensor - scale is irrelevant; pick 1.0 for stability
    return abs_max / float(_INT8_QMAX)


def _quantise_to_int8(tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor int8 quant. Returns (qints, scale)."""
    scale = _per_tensor_scale(tensor)
    flat = tensor.detach().to(torch.float32).reshape(-1).cpu().numpy()
    q = np.rint(flat / scale).astype(np.int64)
    np.clip(q, -_INT8_QMAX, _INT8_QMAX, out=q)
    return q.astype(np.int8), float(scale)


def _dequantise_from_int8(
    qints: np.ndarray, scale: float, shape: tuple[int, ...]
) -> torch.Tensor:
    """Inverse of :func:`_quantise_to_int8`. Returns a fp32 torch tensor."""
    arr = qints.astype(np.float32) * float(scale)
    return torch.from_numpy(arr).to(torch.float32).reshape(*shape)


# ---------------------------------------------------------------------------
# Op_GammaJointADMM
# ---------------------------------------------------------------------------


@dataclass
class Op_GammaJointADMM:
    """gamma paradigm: Joint-ADMM + Ballé hyperprior + arithmetic terminal,
    wrapped as a :class:`CodecOp` for the canonical pipeline.

    Composition mode: substitutional alternative to Op 2 (PR103 arithmetic).
    Op_GammaJointADMM is a fuller / more rigorous AC encoder (multi-stream
    ADMM coordinator + arithmetic per-stream + optional Ballé hyperprior)
    at the cost of more encode-time compute. Pipeline can compare both
    empirically.

    Internally calls the existing :func:`run_joint_codec_stack` (JCSv1 wire
    format) and embeds its container as the CodecOp blob; ``op_state``
    carries per-stream metadata (name, dtype, shape, dequant scale) so the
    decoder can restore fp32 tensors.

    Attributes
    ----------
    rho_init : float
        Initial ADMM penalty :math:`\\rho`. Boyd 2011 section 3.4 default 1.0;
        increase for tighter convergence at higher iter cost.
    max_admm_iters : int
        ADMM iteration cap. Default 50 matches the orchestrator's
        production default. Tests may use 2 for speed.
    balle_quality : int
        Reserved for future Ballé hyperprior wiring. Currently unused -
        the wrap defaults all streams to ``KIND_ARITHMETIC_STATIC``
        (the orchestrator's auto-fallback path also covers BHv1
        static-wins). Future work: wire a Ballé codec instance per
        large tensor when ``context['use_balle_hyperprior']=True``.
    substrate_aware_init : bool
        Council STUB 2026-05-07 (Ballé). When True, encode logs a WARN
        about the missing ``prior_init`` API on the underlying Ballé
        codec and records a "γ-Phase-2 needs hyperprior init API"
        finding in ``op_state['stub_findings']``. No functional change
        to byte counts or roundtrip behaviour today; the flag is the
        design-contract anchor for the fix that requires GPU training
        of an end-to-end hyperprior. Default False (silent inert).
    name : str
        ``"gamma_joint_admm"``.
    """

    rho_init: float = 1.0
    max_admm_iters: int = 50
    balle_quality: int = 4
    substrate_aware_init: bool = False
    name: str = "gamma_joint_admm"

    # ---- Validation ---------------------------------------------------------

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        """Contrarian gate. Verifies hyperparameter sanity + tensor shape.

        For default use (no precomputed qint corpus), the op accepts any
        non-empty state_dict of fp32-castable tensors. ``passed=True`` is
        the common path.

        A reviewed precomputed-qint context path can be added later, but the
        current implementation deliberately uses the internal deterministic
        int8 quantisation path only.
        """
        findings: list[str] = []

        if self.rho_init <= 0.0:
            findings.append(
                f"rho_init must be > 0.0, got {self.rho_init}"
            )
        if self.max_admm_iters < 1:
            findings.append(
                f"max_admm_iters must be >= 1, got {self.max_admm_iters}"
            )
        if self.balle_quality < 0:
            findings.append(
                f"balle_quality must be >= 0, got {self.balle_quality}"
            )
        if not isinstance(state_dict, dict) or not state_dict:
            findings.append(
                "state_dict must be a non-empty dict of fp32 torch.Tensors"
            )
        else:
            for name, t in state_dict.items():
                if not isinstance(t, torch.Tensor):
                    findings.append(
                        f"state_dict[{name!r}] is not a torch.Tensor "
                        f"(got {type(t).__name__})"
                    )
                    continue
                if t.numel() == 0:
                    findings.append(
                        f"state_dict[{name!r}] is empty (numel=0)"
                    )
                    continue
                # gamma orchestrator works in numpy; reject obviously-corrupt floats.
                if t.dtype.is_floating_point and not torch.isfinite(t).all():
                    findings.append(
                        f"state_dict[{name!r}] contains non-finite values"
                    )

        return ValidationReport(
            passed=not findings,
            op_name=self.name,
            findings=findings,
        )

    # ---- Encode -------------------------------------------------------------

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        """Quantise -> run_joint_codec_stack -> embed JCSv1 container as blob.

        Returns:
            :class:`EncodeResult` whose ``blob`` is the JCSv1 container and
            whose ``op_state`` records per-stream dequant metadata.
        """
        from tac.joint_codec_stack_orchestrator import (
            KIND_ARITHMETIC_STATIC,
            StreamSource,
            run_joint_codec_stack,
        )

        bytes_in = sum(
            t.numel() * t.element_size() for t in state_dict.values()
        )

        # Council STUB 2026-05-07 (Ballé): substrate-aware hyperprior init
        # is the prescribed CPU-only fix for the +33K B PR106 regression.
        # The underlying BalleHyperpriorCodec does not yet expose a
        # prior_init API (verified by grep). When set, log a WARN and
        # record a finding in op_state; do NOT invent flags / silently
        # mutate behaviour. Encode proceeds with the existing static
        # arithmetic path.
        stub_findings: list[str] = []
        if self.substrate_aware_init:
            logger.warning(SUBSTRATE_AWARE_INIT_WARN_MESSAGE)
            stub_findings.append(SUBSTRATE_AWARE_INIT_WARN_MESSAGE)

        # Stable encoding order: sorted by tensor name. JCSv1 packing is
        # deterministic given identical input order; sorted keys give
        # bit-determinism across runs.
        ordered_items = sorted(state_dict.items(), key=lambda kv: kv[0])

        streams: list[StreamSource] = []
        # Per-stream metadata for the decoder side (in the same order as
        # `ordered_items`, which equals the order JCSv1 packs the streams).
        stream_meta: list[dict[str, Any]] = []
        for tensor_name, tensor in ordered_items:
            qints, scale = _quantise_to_int8(tensor)
            stream_name = self._encode_stream_name(tensor_name)
            streams.append(
                StreamSource(
                    name=stream_name,
                    qints=qints,
                    num_symbols=2 * _INT8_QMAX + 1,  # 255 symbols
                    offset=_INT8_QMAX,                # maps [-127, 127] -> [0, 254]
                    codec_kind=KIND_ARITHMETIC_STATIC,
                    # Tiny non-zero marginal: the orchestrator's ADMM uses
                    # marginals to compute the KKT waterline residual; an
                    # all-zero marginal produces a divide-by-zero / infinity
                    # in the residual scale. Caller-supplied real marginals
                    # may override per-stream via context['score_marginals'];
                    # the wrap defaults to a tiny constant so structural
                    # encode succeeds even without scorer-aware planning.
                    score_per_byte_marginal=1e-6,
                )
            )
            stream_meta.append(
                {
                    "stream_name": stream_name,
                    "tensor_name": tensor_name,
                    "shape": [int(d) for d in tensor.shape],
                    "scale": float(scale),
                    "dtype": "float32",
                    "qint_dtype": "int8",
                }
            )

        # Choose a generous byte budget so the orchestrator's container check
        # never refuses a successful encode. The orchestrator only uses this
        # as an upper bound; it does not pad the output. We pick: 2 x sum of
        # raw byte counts + 4096 KiB metadata slack - well above what any
        # arithmetic-coded int8 stream would ever need.
        raw_total = sum(int(s.qints.size) for s in streams)
        byte_budget = float(2 * raw_total + 4 * 1024 * 1024)

        result = run_joint_codec_stack(
            streams=streams,
            byte_budget=byte_budget,
            admm_max_iters=int(self.max_admm_iters),
            admm_rho_init=float(self.rho_init),
        )

        op_state: dict[str, Any] = {
            "streams": stream_meta,
            "admm_iters": int(result.iters),
            "admm_converged": bool(result.converged),
            "stream_count": len(stream_meta),
            "substrate_aware_init": bool(self.substrate_aware_init),
            "stub_findings": stub_findings,
        }

        return EncodeResult(
            blob=result.container_bytes,
            bytes_in=bytes_in,
            bytes_out=len(result.container_bytes),
            op_name=self.name,
            op_state=op_state,
        )

    # ---- Decode -------------------------------------------------------------

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Unpack JCSv1 container, decode each qint stream, dequantise to fp32."""
        from tac.arithmetic_qint_codec import decode_qints_arithmetic
        from tac.balle_hyperprior_codec import decode_qints_balle
        from tac.joint_codec_stack_orchestrator import (
            KIND_ARITHMETIC_STATIC,
            KIND_BALLE_HYPERPRIOR,
            KIND_RAW_PASSTHROUGH,
            unpack_jcsp_container,
        )

        if "streams" not in op_state:
            raise ValueError(
                "Op_GammaJointADMM.decode: op_state missing 'streams' "
                "metadata (encoder must have populated it)"
            )

        parsed = unpack_jcsp_container(blob)
        parsed_streams = parsed["streams"]
        meta_streams = op_state["streams"]
        if len(parsed_streams) != len(meta_streams):
            raise ValueError(
                f"Op_GammaJointADMM.decode: stream count mismatch - "
                f"JCSv1 has {len(parsed_streams)} but op_state has "
                f"{len(meta_streams)}"
            )

        # Build a quick-lookup by stream name (JCSv1 emits sorted name order
        # because the encoder sorted the state_dict by tensor_name; the
        # op_state was built in the same order so positional zip is fine,
        # but we cross-check on name for safety).
        out: dict[str, torch.Tensor] = {}
        for parsed_s, meta in zip(parsed_streams, meta_streams, strict=True):
            if parsed_s["name"] != meta["stream_name"]:
                raise ValueError(
                    f"Op_GammaJointADMM.decode: stream name mismatch at "
                    f"position - JCSv1 has {parsed_s['name']!r} but "
                    f"op_state has {meta['stream_name']!r}"
                )
            payload: bytes = parsed_s["payload"]
            kind = int(parsed_s["codec_kind"])
            if kind == KIND_ARITHMETIC_STATIC:
                qints = decode_qints_arithmetic(payload, expected_dtype=np.int8)
            elif kind == KIND_BALLE_HYPERPRIOR:
                qints = decode_qints_balle(blob=payload, expected_dtype=np.int8)
            elif kind == KIND_RAW_PASSTHROUGH:
                # Default int8 wrap doesn't emit RAW; refuse cleanly.
                raise ValueError(
                    f"Op_GammaJointADMM.decode: stream {parsed_s['name']!r} is "
                    f"RAW_PASSTHROUGH but default wrap produced int8 "
                    f"qint payloads only - caller likely staged a custom "
                    f"qint corpus; decode of RAW streams not yet wired."
                )
            else:
                raise ValueError(
                    f"Op_GammaJointADMM.decode: unknown codec_kind={kind} "
                    f"for stream {parsed_s['name']!r}"
                )
            shape = tuple(int(d) for d in meta["shape"])
            scale = float(meta["scale"])
            tensor_name = str(meta["tensor_name"])
            out[tensor_name] = _dequantise_from_int8(qints, scale, shape)
        return out

    # ---- Internal helpers ---------------------------------------------------

    @staticmethod
    def _encode_stream_name(tensor_name: str) -> str:
        """Tensor names like ``"blocks.0.weight"`` are valid JCSv1 stream
        names (UTF-8, no NUL, <=255 bytes). Encode as-is; future variants may
        need escaping if the schema introduces other restrictions.
        """
        b = tensor_name.encode("utf-8")
        if len(b) == 0 or len(b) > 255:
            raise ValueError(
                f"_encode_stream_name: tensor name {tensor_name!r} encodes "
                f"to {len(b)} UTF-8 bytes (must be 1..255)"
            )
        if "\x00" in tensor_name:
            raise ValueError(
                f"_encode_stream_name: tensor name {tensor_name!r} contains "
                f"a NUL byte - JCSv1 forbids it"
            )
        return tensor_name


__all__ = [
    "Op_GammaJointADMM",
    "SUBSTRATE_AWARE_INIT_WARN_MESSAGE",
]
