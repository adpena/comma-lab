# SPDX-License-Identifier: MIT
"""beta paradigm - sensitivity-aware substrate-transform :class:`CodecOp`.

Composition mode (per the four-way-stack composition contract,
``.omx/research/four_way_stack_composition_contract_20260507_claude.md``):
**substrate-transform / preprocessing**.

The op runs *upstream* of Op 1 (PR101 split-Brotli) / Op 2 (PR103 arithmetic
codec). It modifies the input ``state_dict`` based on a per-tensor sensitivity
score and emits a TRANSFORMED state_dict where:

* high-sensitivity tensors are bit-protected (kept fp16-quantized-and-back so
  downstream ops re-quantize with minimum additional error),
* low-sensitivity tensors are pre-quantized to ``low_bits`` (default int4) and
  dequantized back to fp32 (the int4 grid bakes the bit-allocation decision
  into the substrate),
* mid-band tensors pass through unchanged.

Downstream Op 1 / Op 2 then encode the beta-transformed state_dict. beta is the
*last* op the original weights see; downstream ops see the beta-transformed view
and roundtrip *that* view bit-faithfully.

Sensitivity sources
-------------------

``sensitivity_source`` selects how per-tensor scalar sensitivities are
obtained:

* ``"uniform"`` (default) - every tensor gets sensitivity 0.0; combined
  with the default ``high_threshold=0.5`` / ``low_threshold=0.1`` this
  classifies every tensor as "low" (below ``low_threshold``). To make
  ``"uniform"`` an *identity* transform for testing, callers pass
  ``low_threshold=-1.0`` so no tensor is below it and every tensor lands
  in the "mid" band (passthrough). The op's ``Op_SensitivityPreprocess.identity()``
  classmethod constructs that exact configuration.

* ``"fisher"`` - requires a real Fisher artifact in
  ``context["sensitivity_artifact"]``. The artifact must follow the
  :mod:`tac.sensitivity_map` contract (``{"<module>.weight": Tensor[O]}``).
  Per-tensor sensitivity is the *max* over output channels; a tensor not
  named in the artifact is treated as protected (bit-faithful). On a
  synthetic state_dict you can stub a per-tensor float dict via
  ``context["sensitivity_scores"]`` and use ``sensitivity_source="stub"``.

* ``"imp"`` - uses :mod:`tac.imp_sensitivity_weighted`-style classification.
  Requires either ``context["sensitivity_artifact"]`` (per-channel) or a
  pre-computed ``context["sensitivity_scores"]`` (per-tensor scalar).

* ``"stub"`` - consumes ``context["sensitivity_scores"]`` (per-tensor
  ``{name: float}``) directly. Used by tests + smoke runs where a full
  Fisher pass is not warranted.

Wire format
-----------

The blob is a deterministic concatenation of a small text header + each
tensor's raw bytes, brotli-compressed. The wire format is INTENTIONALLY
custom (rather than ``torch.save``) because ``torch.save`` is not byte
deterministic on PyTorch's tar/zipfile serializer for arbitrarily-keyed
state_dicts (storage IDs leak run-to-run state). Format::

    magic   : 8 bytes = b"BetaPSD\x00"  (beta preprocess state-dict v0)
    n_tensors : u32 LE
    for each tensor (sorted by name):
        name_len  : u16 LE
        name      : utf-8 bytes
        dtype_id  : u8  (0=float32, 1=float16)
        ndim      : u8
        shape     : u32 LE * ndim
        nbytes    : u32 LE
        raw       : nbytes (little-endian native)

The decoder un-brotlis and parses the header, returning the transformed
state_dict ready for the next op. Because we record the per-tensor
classification in ``op_state``, an auditor can reconstruct which tensors
were bit-protected vs pre-quantized. The op currently only emits fp32
tensors (high-sens tensors are fp16-roundtripped *back* to fp32 to match
the substrate that downstream Op 1/Op 2 expect), so dtype_id is 0 in all
current configurations; the byte is reserved for a future fp16-on-the-wire
optimization.

Identity short-circuit (Council fix 2026-05-07)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When :attr:`Op_SensitivityPreprocess.optimize_identity` is True (the default)
**and** the configuration is identity (every tensor lands in the *mid*
band, no transformation applied), encode emits a TINY 12-byte marker
blob (``magic + 4-byte zero "n_tensors"``) instead of serialising the
full fp32 substrate. The op_state records ``is_identity=True``.

The decoder, when it sees ``is_identity=True``, looks up the original
substrate from the shared ``context["_beta_identity_substrate"]`` slot
that encode populated. Within a single :class:`CodecPipeline.encode`
call, the orchestrator threads the same ``ctx`` dict through every op's
encode + the immediate substrate-transform decode, so the next op
(Op 1 / Op 2 / etc.) sees the original substrate without any byte cost
beyond the 12-byte marker.

For standalone (non-pipeline) encode + decode the caller MUST pass the
SAME context dict to both calls (encode mutates it, decode reads it).
Tests cover both the pipeline-threaded path and the same-context path.

At decompress time the pipeline does NOT have the original substrate in
context (we only ship the BLOB), so β-identity decode without context
returns an empty state_dict. This is intentional and safe: the pipeline
loop in :meth:`CodecPipeline.decode` overrides ``decoded_state`` on every
iteration and returns only the LAST op's decoded state, so the empty
β output is discarded. Operators using β identity *as the only op*
without context would receive an empty state_dict — that mode is
explicitly degenerate and not part of the supported configuration
(β identity is meant to compose, not to ship alone).

This optimisation closes a 274,411 B substrate ballooning on PR106
(``[empirical:experiments/results/lane_codec_pipeline_full_stack_pr106_20260507T172731Z/composition_matrix_pr106.json]``)
where the ``beta_identity_then_Op1`` stack measured 445,840 B vs
Op1_alone's 170,037 B — beta was paying full state_dict bytes for a
no-op transform. Selfcomp + Boyd flagged this in the Grand Council
review at ``.omx/research/grand_council_pr106_substrate_findings_zig_default_20260507.md``.

Strict-scorer-rule: this op never loads scorers. Sensitivity must be supplied
upstream (via the artifact or stub dict).

References
----------

* :mod:`tac.codec_pipeline` - :class:`CodecOp` Protocol + :class:`CodecPipeline`
  orchestrator + :class:`Op1_PR101SplitBrotli` reference impl.
* :mod:`tac.sensitivity_map` - Fisher / score-Jacobian per-channel artifact contract.
* :mod:`tac.imp_sensitivity_weighted` - IMP-style classification of layers by
  sensitivity (bit-protect / aggressive / standard).
* :mod:`tac.balle_sensitivity_weighted` - Ballé hyperprior with sensitivity weighting.
* :mod:`tac.pr101_split_brotli_codec` - ``FIXED_STATE_SCHEMA`` (28-tensor HNeRV
  state-dict shape used for synthetic tests).

Strategic note
--------------

Per CLAUDE.md "Forbidden score claims": this module never claims a
``[contest-CUDA]`` score. Empirical byte savings claims must be tagged
``[empirical:<manifest path>]``; predicted gains must be tagged
``[predicted]``.
"""
from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import brotli
import torch

from tac.codec_pipeline import EncodeResult, ValidationReport

# ---------------------------------------------------------------------------
# Deterministic state-dict wire format (BetaPSD v0)
# ---------------------------------------------------------------------------

_BETA_PSD_MAGIC = b"BetaPSD\x00"
_DTYPE_TO_ID: dict[torch.dtype, int] = {torch.float32: 0, torch.float16: 1}
_ID_TO_DTYPE: dict[int, torch.dtype] = {0: torch.float32, 1: torch.float16}


def _serialize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bytes:
    """Pack a state_dict into the deterministic BetaPSD wire format.

    Tensors are written in **sorted name order** so two encodes of
    equivalent state_dicts produce byte-identical output regardless of
    the input dict's iteration order.
    """
    parts: list[bytes] = [_BETA_PSD_MAGIC]
    items = sorted(state_dict.items(), key=lambda kv: kv[0])
    parts.append(struct.pack("<I", len(items)))
    for name, tensor in items:
        if not isinstance(tensor, torch.Tensor):
            raise SensitivityPreprocessError(
                f"state_dict[{name!r}] must be a torch.Tensor; "
                f"got {type(tensor).__name__}"
            )
        t = tensor.detach().contiguous().cpu()
        if t.dtype not in _DTYPE_TO_ID:
            raise SensitivityPreprocessError(
                f"state_dict[{name!r}] has unsupported dtype {t.dtype}; "
                f"BetaPSD v0 supports {sorted(d.__str__() for d in _DTYPE_TO_ID)}"
            )
        name_b = name.encode("utf-8")
        if len(name_b) > 0xFFFF:
            raise SensitivityPreprocessError(
                f"state_dict tensor name too long ({len(name_b)} bytes); "
                "BetaPSD v0 limits names to 65535 bytes"
            )
        parts.append(struct.pack("<H", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<B", _DTYPE_TO_ID[t.dtype]))
        if t.ndim > 0xFF:
            raise SensitivityPreprocessError(
                f"tensor {name!r} has too many dims ({t.ndim}); BetaPSD v0 limits to 255"
            )
        parts.append(struct.pack("<B", t.ndim))
        for dim in t.shape:
            if int(dim) > 0xFFFFFFFF:
                raise SensitivityPreprocessError(
                    f"tensor {name!r} dim {dim} exceeds u32 limit"
                )
            parts.append(struct.pack("<I", int(dim)))
        raw = t.numpy().tobytes()
        if len(raw) > 0xFFFFFFFF:
            raise SensitivityPreprocessError(
                f"tensor {name!r} byte length {len(raw)} exceeds u32 limit"
            )
        parts.append(struct.pack("<I", len(raw)))
        parts.append(raw)
    return b"".join(parts)


def _deserialize_state_dict(buf: bytes) -> dict[str, torch.Tensor]:
    """Inverse of :func:`_serialize_state_dict`."""
    if len(buf) < len(_BETA_PSD_MAGIC) + 4:
        raise SensitivityPreprocessError("BetaPSD payload truncated at header")
    if buf[: len(_BETA_PSD_MAGIC)] != _BETA_PSD_MAGIC:
        raise SensitivityPreprocessError(
            f"BetaPSD payload bad magic {buf[:8]!r}; expected {_BETA_PSD_MAGIC!r}"
        )
    cursor = len(_BETA_PSD_MAGIC)
    (n_tensors,) = struct.unpack_from("<I", buf, cursor)
    cursor += 4
    out: dict[str, torch.Tensor] = {}
    import numpy as np  # local - keep top-level import surface narrow

    for _ in range(n_tensors):
        if cursor + 2 > len(buf):
            raise SensitivityPreprocessError("BetaPSD payload truncated at name_len")
        (name_len,) = struct.unpack_from("<H", buf, cursor)
        cursor += 2
        if cursor + name_len > len(buf):
            raise SensitivityPreprocessError("BetaPSD payload truncated at name")
        name = buf[cursor : cursor + name_len].decode("utf-8")
        cursor += name_len
        if cursor + 2 > len(buf):
            raise SensitivityPreprocessError("BetaPSD payload truncated at dtype/ndim")
        (dtype_id,) = struct.unpack_from("<B", buf, cursor)
        cursor += 1
        (ndim,) = struct.unpack_from("<B", buf, cursor)
        cursor += 1
        if dtype_id not in _ID_TO_DTYPE:
            raise SensitivityPreprocessError(
                f"BetaPSD unknown dtype_id {dtype_id} for {name!r}"
            )
        if cursor + 4 * ndim > len(buf):
            raise SensitivityPreprocessError("BetaPSD payload truncated at shape")
        shape = struct.unpack_from(f"<{ndim}I", buf, cursor)
        cursor += 4 * ndim
        if cursor + 4 > len(buf):
            raise SensitivityPreprocessError("BetaPSD payload truncated at nbytes")
        (nbytes,) = struct.unpack_from("<I", buf, cursor)
        cursor += 4
        if cursor + nbytes > len(buf):
            raise SensitivityPreprocessError(
                f"BetaPSD payload truncated at raw ({name!r})"
            )
        raw = buf[cursor : cursor + nbytes]
        cursor += nbytes
        dtype = _ID_TO_DTYPE[dtype_id]
        np_dtype = np.float32 if dtype == torch.float32 else np.float16
        arr = (
            np.frombuffer(raw, dtype=np_dtype).reshape(shape)
            if shape
            else np.frombuffer(raw, dtype=np_dtype)
        )
        # Copy so the resulting tensor owns its memory.
        out[name] = torch.from_numpy(arr.copy())
    if cursor != len(buf):
        raise SensitivityPreprocessError(
            f"BetaPSD payload has {len(buf) - cursor} trailing bytes"
        )
    return out

# Supported sensitivity sources - exposed as a constant for the
# validate-rejects-unknown-source test.
SUPPORTED_SENSITIVITY_SOURCES: frozenset[str] = frozenset(
    {"uniform", "fisher", "imp", "stub"}
)

# Brotli quality used for the inter-op blob. Default 9 (a good
# size/time tradeoff for transformed state_dict blobs that downstream ops
# will re-encode anyway). Quality 11 is overkill since beta is not the final
# encoding stage.
DEFAULT_BROTLI_QUALITY = 9

# Identity short-circuit (Council fix 2026-05-07): when beta detects that
# every tensor lands in the mid band (i.e. the configured op IS the identity
# transform), encode emits a tiny 12-byte marker blob instead of the full
# BetaPSD payload. The shared ``context`` dict is used to thread the original
# substrate through to the in-pipeline decode call. Constants keep the magic
# bytes and the context key DRY across encode/decode + tests.
_BETA_IDENTITY_MARKER_MAGIC = b"BetaIDv0"  # 8 bytes, distinct from BetaPSD magic.
_BETA_IDENTITY_MARKER_BLOB = _BETA_IDENTITY_MARKER_MAGIC + struct.pack("<I", 0)
_BETA_IDENTITY_CONTEXT_KEY = "_beta_identity_substrate"


class SensitivityPreprocessError(ValueError):
    """Raised when beta preprocessing inputs are malformed or unsupported."""


# ---------------------------------------------------------------------------
# Sensitivity resolution
# ---------------------------------------------------------------------------

def _per_tensor_scores_from_artifact(
    state_dict: Mapping[str, torch.Tensor],
    artifact: Mapping[str, Any],
) -> dict[str, float]:
    """Reduce a per-channel sensitivity artifact to a per-tensor scalar dict.

    The :mod:`tac.sensitivity_map` contract is
    ``{"<module>.weight": Tensor[O]}`` (one scalar per output channel). We
    reduce via ``max`` (matches :func:`tac.imp_sensitivity_weighted.classify_layers_by_sensitivity`'s
    "any high-sensitivity channel protects the whole tensor" rule). Tensors
    not named in the artifact map default to 0.0 (will be classified as
    low / aggressive unless the caller raised the threshold).
    """
    scores: dict[str, float] = {}
    for name in state_dict:
        sens = artifact.get(name)
        if sens is None:
            scores[name] = 0.0
            continue
        if isinstance(sens, torch.Tensor):
            if sens.numel() == 0:
                scores[name] = 0.0
            else:
                scores[name] = float(sens.detach().abs().max().item())
        elif isinstance(sens, (int, float)):
            scores[name] = float(abs(sens))
        else:
            raise SensitivityPreprocessError(
                f"sensitivity artifact entry for {name!r} has unsupported "
                f"type {type(sens).__name__}; expected Tensor or scalar"
            )
    return scores


def _resolve_per_tensor_scores(
    state_dict: Mapping[str, torch.Tensor],
    sensitivity_source: str,
    context: Mapping[str, Any],
) -> dict[str, float]:
    """Resolve a ``{tensor_name: float}`` sensitivity dict from the context."""
    if sensitivity_source == "uniform":
        return dict.fromkeys(state_dict.keys(), 0.0)
    if sensitivity_source == "stub":
        scores = context.get("sensitivity_scores")
        if not isinstance(scores, Mapping):
            raise SensitivityPreprocessError(
                "sensitivity_source='stub' requires "
                "context['sensitivity_scores']: Mapping[str, float]"
            )
        out: dict[str, float] = {}
        for name in state_dict:
            v = scores.get(name, 0.0)
            if not isinstance(v, (int, float)):
                raise SensitivityPreprocessError(
                    f"context['sensitivity_scores'][{name!r}] must be scalar"
                )
            out[name] = float(v)
        return out
    if sensitivity_source in ("fisher", "imp"):
        artifact = context.get("sensitivity_artifact")
        if artifact is None:
            scores = context.get("sensitivity_scores")
            if isinstance(scores, Mapping):
                # Allow per-tensor scalar fallback so test fixtures don't
                # need to materialize a full per-channel artifact for
                # source='imp'. This is Council-approved per the
                # sensitivity_source='imp' usage in IMP-side training.
                return _resolve_per_tensor_scores(state_dict, "stub", context)
            raise SensitivityPreprocessError(
                f"sensitivity_source={sensitivity_source!r} requires "
                "context['sensitivity_artifact'] (per-channel sensitivity map) "
                "or context['sensitivity_scores'] (per-tensor scalar)"
            )
        if not isinstance(artifact, Mapping):
            raise SensitivityPreprocessError(
                "context['sensitivity_artifact'] must be a Mapping[str, Tensor]"
            )
        return _per_tensor_scores_from_artifact(state_dict, artifact)
    # Should be unreachable - validate() enforces the supported set first.
    raise SensitivityPreprocessError(
        f"unknown sensitivity_source {sensitivity_source!r}; "
        f"supported: {sorted(SUPPORTED_SENSITIVITY_SOURCES)}"
    )


# ---------------------------------------------------------------------------
# Bit-allocation decisions
# ---------------------------------------------------------------------------

# Classification labels - ``high`` = bit-protect (round-trip via fp16),
# ``low`` = pre-quantize to ``low_bits``, ``mid`` = passthrough.
_CLASS_HIGH = "high"
_CLASS_MID = "mid"
_CLASS_LOW = "low"
_VALID_CLASSES = (_CLASS_HIGH, _CLASS_MID, _CLASS_LOW)


def _classify(score: float, *, high_threshold: float, low_threshold: float) -> str:
    if score >= high_threshold:
        return _CLASS_HIGH
    if score < low_threshold:
        return _CLASS_LOW
    return _CLASS_MID


def _bit_protect_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """Round-trip the tensor through fp16 (bit-protect: tiny + bounded loss).

    fp16 represents fp32 weights with ~3 decimal digits of precision; for
    HNeRV weights in [-2, 2] this is a ~1e-3 relative perturbation. The
    downstream Op 1 / Op 2 codec re-quantizes to int8 anyway, so the fp16
    pass is genuinely "protective" - we keep more of the original signal
    than we'd lose to int4 pre-quantization.
    """
    return tensor.detach().to(torch.float16).to(torch.float32)


def _quantize_low_bits(tensor: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Per-tensor symmetric ``bits``-bit quantize-then-dequantize.

    Uses absmax scaling: ``q = round(x * (qmax / absmax)).clamp(-qmax, qmax)``;
    dequant: ``x_hat = q * absmax / qmax``. Returns fp32 tensor on the same
    quantization grid as the int-N substrate (i.e. lossy, but the *grid*
    is preserved - downstream ops see far fewer unique values to encode).
    """
    if bits < 2 or bits > 8:
        raise SensitivityPreprocessError(
            f"low_bits={bits} out of range; expected 2..8"
        )
    t = tensor.detach().to(torch.float32)
    absmax = t.abs().max().item()
    if absmax <= 0.0:
        return t.clone()
    qmax = (1 << (bits - 1)) - 1  # 2..127
    scale = absmax / qmax
    q = (t / scale).round().clamp(-qmax, qmax)
    return q * scale


def _apply_classification(
    state_dict: Mapping[str, torch.Tensor],
    classes: Mapping[str, str],
    *,
    low_bits: int,
) -> dict[str, torch.Tensor]:
    """Apply the bit-allocation decision per-tensor."""
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        cls = classes.get(name, _CLASS_MID)
        if cls == _CLASS_HIGH:
            out[name] = _bit_protect_fp16(tensor)
        elif cls == _CLASS_LOW:
            out[name] = _quantize_low_bits(tensor, bits=low_bits)
        else:
            # Passthrough - but clone-detach so callers can't mutate inputs.
            out[name] = tensor.detach().clone()
    return out


# ---------------------------------------------------------------------------
# The op
# ---------------------------------------------------------------------------

@dataclass
class Op_SensitivityPreprocess:
    """beta preprocessing op - substrate-transform CodecOp.

    Per the four-way-stack composition contract this op runs as the first
    stage (or the only stage, when used as a smoke-test identity transform).
    Its output is a *transformed* state_dict that the next CodecOp sees as
    its input.

    Args:
        sensitivity_source: one of ``"uniform"`` / ``"fisher"`` / ``"imp"`` /
            ``"stub"``. ``"uniform"`` is the default and is intentionally
            non-identity (every tensor lands below ``low_threshold``); use
            :meth:`Op_SensitivityPreprocess.identity` for a true passthrough.
        high_threshold: tensors with sensitivity >= this are bit-protected
            (fp16 round-trip). Default 0.5 (calibrated for IMP-style scores
            normalized in [0, 1]).
        low_threshold: tensors with sensitivity < this are pre-quantized to
            ``low_bits``. Default 0.1.
        low_bits: bit-depth for low-sensitivity tensors. Default 4 (int4
            substrate). Range 2..8.
        brotli_quality: brotli quality for the inter-op blob. Default 9.
        optimize_identity: when True (default) and every tensor classifies
            to the *mid* band (no transform applied), emit a 12-byte marker
            blob and thread the original substrate through ``context``
            instead of serialising the full fp32 state_dict. Closes the
            274,411 B substrate ballooning observed on PR106 in the
            Council review (zig_default findings 2026-05-07). Set to
            False to force the full BetaPSD payload (legacy behaviour
            useful for forensic / comparative measurements).

    Invariants:
        * Bit-faithful roundtrip *of the transformed state_dict* - encode
          followed by decode produces the same fp32 tensors that downstream
          ops would have encoded.
        * Identity behavior when constructed via
          :meth:`Op_SensitivityPreprocess.identity`.
        * Identity short-circuit: when configured for identity AND
          ``optimize_identity=True``, the emitted blob is the 12-byte
          marker; encode + decode share the input substrate via the
          shared ``context`` dict ``_beta_identity_substrate`` slot.
        * Validation rejects unknown ``sensitivity_source``.
    """

    name: str = "sensitivity_preprocess"
    transforms_state_dict: bool = True
    sensitivity_source: str = "uniform"
    high_threshold: float = 0.5
    low_threshold: float = 0.1
    low_bits: int = 4
    brotli_quality: int = DEFAULT_BROTLI_QUALITY
    optimize_identity: bool = True

    @classmethod
    def identity(cls, **kwargs: Any) -> Op_SensitivityPreprocess:
        """Configure the op as a true passthrough (every tensor -> mid band).

        Equivalent to ``Op_SensitivityPreprocess(sensitivity_source='uniform',
        low_threshold=-1.0)``: no tensor falls below ``low_threshold=-1.0``,
        no tensor reaches ``high_threshold=0.5`` (uniform scores are 0.0),
        so every tensor passes through unchanged.
        """
        defaults: dict[str, Any] = {
            "sensitivity_source": "uniform",
            "low_threshold": -1.0,
            "high_threshold": 0.5,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    # ------------------------------------------------------------------
    # Validation gate
    # ------------------------------------------------------------------

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any] | None = None,
    ) -> ValidationReport:
        ctx = context or {}
        findings: list[str] = []

        if self.sensitivity_source not in SUPPORTED_SENSITIVITY_SOURCES:
            findings.append(
                f"unknown sensitivity_source {self.sensitivity_source!r}; "
                f"supported: {sorted(SUPPORTED_SENSITIVITY_SOURCES)}"
            )

        # Threshold + bits sanity.
        if not (0 <= self.low_threshold <= self.high_threshold) and not (
            self.low_threshold < 0  # identity escape hatch is allowed
        ):
            findings.append(
                f"low_threshold={self.low_threshold} must be <= "
                f"high_threshold={self.high_threshold}"
            )
        if self.low_bits < 2 or self.low_bits > 8:
            findings.append(
                f"low_bits={self.low_bits} out of range (expected 2..8)"
            )

        # Importability of named beta source modules - validate that the
        # dependency surface exists, even if we don't invoke the
        # per-channel branches in this op.
        try:  # pragma: no cover - defensive
            import tac.imp_sensitivity_weighted
            import tac.sensitivity_map  # noqa: F401
        except ImportError as exc:
            findings.append(
                f"beta source modules not importable: {exc}; "
                "Op_SensitivityPreprocess requires tac.sensitivity_map + "
                "tac.imp_sensitivity_weighted to be installed"
            )

        # Source-specific input availability checks. ``"uniform"`` and
        # ``"stub"`` are checked at encode time (so empty state_dict +
        # uniform yields a clean validate); ``"fisher"``/``"imp"`` need
        # context to be present.
        if self.sensitivity_source in ("fisher", "imp") and not findings:
            artifact = ctx.get("sensitivity_artifact")
            scores = ctx.get("sensitivity_scores")
            if artifact is None and scores is None:
                findings.append(
                    f"sensitivity_source={self.sensitivity_source!r} requires "
                    "context['sensitivity_artifact'] (per-channel) or "
                    "context['sensitivity_scores'] (per-tensor scalar)"
                )

        return ValidationReport(
            passed=not findings,
            op_name=self.name,
            findings=findings,
        )

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def _classify_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        context: Mapping[str, Any],
    ) -> tuple[dict[str, str], dict[str, float]]:
        scores = _resolve_per_tensor_scores(
            state_dict, self.sensitivity_source, context
        )
        classes = {
            name: _classify(
                scores[name],
                high_threshold=self.high_threshold,
                low_threshold=self.low_threshold,
            )
            for name in state_dict
        }
        return classes, scores

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any] | None = None,
    ) -> EncodeResult:
        ctx = context if context is not None else {}
        # Reject unknown sensitivity_source eagerly even if the caller
        # passed skip_validate=True at the pipeline level - encoding with
        # an unknown source would silently be uniform, which is a
        # confusing failure mode.
        if self.sensitivity_source not in SUPPORTED_SENSITIVITY_SOURCES:
            raise SensitivityPreprocessError(
                f"unknown sensitivity_source {self.sensitivity_source!r}; "
                f"supported: {sorted(SUPPORTED_SENSITIVITY_SOURCES)}"
            )

        bytes_in = sum(t.numel() * t.element_size() for t in state_dict.values())
        classes, scores = self._classify_state_dict(state_dict, ctx)

        n_high = sum(1 for c in classes.values() if c == _CLASS_HIGH)
        n_mid = sum(1 for c in classes.values() if c == _CLASS_MID)
        n_low = sum(1 for c in classes.values() if c == _CLASS_LOW)

        # Identity short-circuit: when every tensor lands in the mid band
        # the op is a no-op transform. Emit a 12-byte marker blob and stash
        # the input substrate in the shared context dict so the immediate
        # in-pipeline decode (or a same-context standalone decode) can
        # carry the substrate through without paying the full fp32 byte
        # cost. Council fix 2026-05-07 (Selfcomp + Boyd) — closes the
        # 274,411 B PR106 ballooning measured at
        # ``experiments/results/lane_codec_pipeline_full_stack_pr106_20260507T172731Z/composition_matrix_pr106.json``.
        is_identity = (
            self.optimize_identity
            and self.sensitivity_source == "uniform"
            and n_high == 0
            and n_low == 0
            and n_mid == len(classes)
        )

        op_state: dict[str, Any] = {
            "sensitivity_source": self.sensitivity_source,
            "high_threshold": self.high_threshold,
            "low_threshold": self.low_threshold,
            "low_bits": self.low_bits,
            "classes": classes,
            "scores": scores,
            "n_high": n_high,
            "n_mid": n_mid,
            "n_low": n_low,
            "is_identity": is_identity,
        }

        if is_identity:
            # Stash the substrate so decode can return it without the
            # serialised payload. ``ctx`` is the same dict the pipeline
            # threads through every op for this encode + the immediate
            # substrate-transform decode (see CodecPipeline.encode).
            ctx[_BETA_IDENTITY_CONTEXT_KEY] = dict(state_dict)
            blob = _BETA_IDENTITY_MARKER_BLOB
            return EncodeResult(
                blob=blob,
                bytes_in=bytes_in,
                bytes_out=len(blob),
                op_name=self.name,
                op_state=op_state,
            )

        # Non-identity path: full BetaPSD wire format.
        transformed = _apply_classification(state_dict, classes, low_bits=self.low_bits)
        raw_bytes = _serialize_state_dict(transformed)
        blob = brotli.compress(raw_bytes, quality=self.brotli_quality)

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
        context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        ctx = context if context is not None else {}

        # Identity short-circuit: when op_state flags is_identity, the
        # blob is a 12-byte marker. Pull the substrate from the shared
        # context dict that encode populated. If the context slot is
        # missing (e.g. CPL1 wrapper decode at decompress time, where we
        # only have the wire bytes), return an empty dict — the pipeline
        # decode loop overrides decoded_state on every op iteration and
        # returns the LAST op's state, so this empty short-circuit is
        # correctly discarded for ``[β_identity, Op1, ...]`` configs.
        # Operators using β_identity as the only op with no context get
        # an empty dict; that mode is degenerate (β-identity is meant to
        # compose with downstream encoders, not to ship alone).
        if op_state.get("is_identity"):
            if blob[: len(_BETA_IDENTITY_MARKER_MAGIC)] != _BETA_IDENTITY_MARKER_MAGIC:
                raise SensitivityPreprocessError(
                    f"is_identity=True but blob lacks marker magic "
                    f"{_BETA_IDENTITY_MARKER_MAGIC!r} (got {blob[:8]!r})"
                )
            stashed = ctx.get(_BETA_IDENTITY_CONTEXT_KEY)
            if stashed is None:
                return {}
            return dict(stashed)

        raw_bytes = brotli.decompress(blob)
        return _deserialize_state_dict(raw_bytes)


__all__ = [
    "DEFAULT_BROTLI_QUALITY",
    "SUPPORTED_SENSITIVITY_SOURCES",
    "_BETA_IDENTITY_CONTEXT_KEY",
    "_BETA_IDENTITY_MARKER_BLOB",
    "_BETA_IDENTITY_MARKER_MAGIC",
    "Op_SensitivityPreprocess",
    "SensitivityPreprocessError",
]
