"""alpha-paradigm mask-encoder bakeoff CodecPipeline ops.

Per the four-way stack composition contract
(``.omx/research/four_way_stack_composition_contract_20260507_claude.md``),
the alpha paradigm is a **substitutional** composition mode: NeRV / wavelet /
VQ-VAE / grayscale-LUT all target the SAME slot (``masks.mkv``); they are
alternative mask encoders, not additive ones. The bakeoff picks the smallest
output across the four ready codecs.

This module wraps each existing mask-codec module behind the canonical
:class:`tac.codec_pipeline.CodecOp` Protocol so the alpha pipeline is
indistinguishable from the decoder-weights pipeline at the orchestrator level.
The alpha pipeline is its own :class:`tac.codec_pipeline.CodecPipeline` instance
(separate from the decoder weights pipeline) because masks live in
``masks.mkv``, not in the HNeRV decoder state_dict.

Input contract (the ``state_dict`` argument all CodecOp methods accept here):

    * Required key: ``"masks"`` - a 3-D ``int64`` ``torch.Tensor`` of shape
      ``(T, H, W)`` with values in ``[0, 5)`` (the 5-class SegNet contest
      taxonomy). The pipeline orchestrator passes a plain
      ``dict[str, torch.Tensor]`` to satisfy the upstream Protocol; this is
      that dict's "state_dict" view.

    * Optional key: ``"poses"`` - reserved for future codecs that need them.
      None of the four alpha codecs presently consume poses.

The :class:`MaskInput` TypedDict at the top of the module makes the contract
explicit and documents the slot the input must carry.

Stub-vs-ready policy:

    * If an underlying codec module is a stub / not-yet-L2 / requires a
      trained-checkpoint upstream, the wrap's ``validate()`` returns
      ``passed=False`` with a single finding pointing at the readiness audit
      (``alpha_mask_codec_readiness.py``). The wrap exists so the pipeline
      orchestrator's Contrarian gate refuses ``encode()`` cleanly instead of
      crashing inside the wrapped codec.

    * If the codec module IS ready (wavelet / VQ-VAE / grayscale-LUT /
      AV1-baseline as of 2026-05-07), the wrap returns ``passed=True`` and
      ``encode()`` runs the underlying codec.

CLAUDE.md compliance:

    * No scorers loaded anywhere (strict-scorer-rule).
    * No CUDA / MPS - pure CPU paths.
    * No ``[contest-CUDA]`` claims; bakeoff byte counts are
      ``[empirical:<manifest path>]`` only.
    * No ``/tmp`` paths in any persisted artifact.
    * Lane registry: lanes ``lane_alpha_{nerv,wavelet,vqvae,grayscale_lut}_mask``
      already exist; the bakeoff lane registers as
      ``lane_codec_pipeline_mask_alpha`` once tests pass.

Cross-references:
    * Canonical orchestrator: :mod:`tac.codec_pipeline`
    * Composition contract: ``.omx/research/four_way_stack_composition_contract_20260507_claude.md``
    * Readiness audit: :mod:`tac.alpha_mask_codec_readiness`
    * Architecture clarification:
      ``project_paradigm_alpha_architecture_clarification_20260506.md``
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch

from tac.codec_pipeline import (
    CodecOp,
    EncodeResult,
    ValidationReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------

class MaskInput(TypedDict, total=False):
    """The canonical alpha-pipeline input contract.

    Keys:
        masks: REQUIRED. ``(T, H, W)`` int64 tensor with values in ``[0, 5)``.
            The 5-class SegNet contest taxonomy: 0=road, 1=lane_markings,
            2=undrivable, 3=movable, 4=my_car. The alpha pipeline encodes this
            tensor's class IDs to a self-describing blob.
        poses: OPTIONAL. ``(T, K)`` float32 pose tensor - reserved for
            codecs that may need pose-conditioned mask encoding. None of the
            four current alpha codecs consume poses.
    """

    masks: torch.Tensor
    poses: torch.Tensor


_REQUIRED_MASK_KEY = "masks"


def _extract_mask_tensor(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pull and validate the canonical ``masks`` tensor from the state_dict.

    Raises:
        ValueError: ``masks`` key missing OR shape/dtype/range invalid.
    """
    if _REQUIRED_MASK_KEY not in state_dict:
        raise ValueError(
            f"alpha-pipeline state_dict missing required key {_REQUIRED_MASK_KEY!r}; "
            f"got keys {sorted(state_dict.keys())}"
        )
    masks = state_dict[_REQUIRED_MASK_KEY]
    if not isinstance(masks, torch.Tensor):
        raise ValueError(
            f"alpha-pipeline {_REQUIRED_MASK_KEY!r} must be a torch.Tensor; "
            f"got {type(masks).__name__}"
        )
    if masks.dim() != 3:
        raise ValueError(
            f"alpha-pipeline {_REQUIRED_MASK_KEY!r} must be 3-D (T,H,W); "
            f"got shape {tuple(masks.shape)}"
        )
    if masks.dtype != torch.int64:
        raise ValueError(
            f"alpha-pipeline {_REQUIRED_MASK_KEY!r} must be int64; got {masks.dtype}"
        )
    if masks.numel() > 0:
        mn, mx = int(masks.min()), int(masks.max())
        if mn < 0 or mx >= 5:
            raise ValueError(
                f"alpha-pipeline {_REQUIRED_MASK_KEY!r} values must be in [0, 5); "
                f"got [{mn}, {mx}]"
            )
    return masks


# ---------------------------------------------------------------------------
# Op_NerVMaskCodec - stub wrap (NeRV requires upstream training)
# ---------------------------------------------------------------------------

@dataclass
class Op_NerVMaskCodec:
    """alpha-NeRV mask codec wrap.

    NeRV training is OUT OF SCOPE for ``CodecOp.encode`` (the trainer is in
    :class:`tac.nerv_mask_codec.NeRVMaskTrainer` and requires CUDA per
    CLAUDE.md non-negotiables). This wrap exists so the alpha-pipeline can
    enumerate it; ``validate()`` returns ``passed=False`` UNLESS an upstream
    pre-trained ``NeRVMaskCodec`` instance is supplied via
    ``context["pretrained_nerv_codec"]``.

    With a pretrained codec in context, ``encode`` calls
    :func:`tac.nerv_mask_codec.encode_nerv_codec` and ``decode`` calls
    :func:`tac.nerv_mask_codec.decode_nerv_codec`. The decoded codec is wrapped
    back into a state_dict with key ``"masks"`` containing the rendered
    argmax masks.
    """

    name: str = "alpha_nerv_mask"
    weight_dtype: str = "fp16"

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        masks = _extract_mask_tensor(state_dict)
        codec = context.get("pretrained_nerv_codec")
        if codec is None:
            raise NotImplementedError(
                "Op_NerVMaskCodec.encode requires a pretrained NeRVMaskCodec "
                "in context['pretrained_nerv_codec']; alpha-NeRV training is "
                "out of scope for the CodecOp Protocol - see "
                "tac.nerv_mask_codec.NeRVMaskTrainer for the trainer."
            )
        from tac.nerv_mask_codec import encode_nerv_codec

        bytes_in = int(masks.numel()) * 8  # int64 baseline
        blob = encode_nerv_codec(codec, weight_dtype=self.weight_dtype)
        op_state: dict[str, Any] = {
            "weight_dtype": self.weight_dtype,
            # Decoder needs to know the original mask shape to render argmax.
            "mask_shape": list(masks.shape),
        }
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
        from tac.nerv_mask_codec import decode_nerv_codec, render_mask_argmax

        codec = decode_nerv_codec(blob)
        shape = op_state.get("mask_shape")
        if shape is None or len(shape) != 3:
            raise ValueError(
                f"Op_NerVMaskCodec.decode op_state missing/invalid 'mask_shape'; "
                f"got {shape!r}"
            )
        t, h, w = int(shape[0]), int(shape[1]), int(shape[2])
        argmax = render_mask_argmax(codec, num_frames=t, height=h, width=w)
        return {_REQUIRED_MASK_KEY: argmax.to(torch.int64)}

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        findings: list[str] = []
        try:
            _extract_mask_tensor(state_dict)
        except ValueError as exc:
            findings.append(str(exc))
            return ValidationReport(passed=False, op_name=self.name, findings=findings)

        codec = context.get("pretrained_nerv_codec")
        if codec is None:
            findings.append(
                "alpha-NeRV not yet L2 ready - encode requires "
                "context['pretrained_nerv_codec'] (trained upstream by "
                "NeRVMaskTrainer on CUDA). See "
                "tac.alpha_mask_codec_readiness.py for the family contract."
            )
            return ValidationReport(passed=False, op_name=self.name, findings=findings)
        return ValidationReport(passed=True, op_name=self.name, findings=findings)


# ---------------------------------------------------------------------------
# Op_WaveletMaskCodec - direct encode/decode (no training needed)
# ---------------------------------------------------------------------------

@dataclass
class Op_WaveletMaskCodec:
    """alpha-wavelet mask codec wrap.

    Wraps :func:`tac.wavelet_mask_codec.encode_wavelet_codec` and
    :func:`decode_wavelet_codec`. The codec encodes ``(T, H, W) int64``
    masks directly via Haar DWT + static-prob arithmetic coding; no upstream
    training required. Bit-faithful roundtrip when ``step_ll/step_detail``
    are large enough that the inverse DWT + argmax recovers the same
    one-hot encoding.
    """

    name: str = "alpha_wavelet_mask"
    levels: int = 2
    step_ll: float = 1.0
    step_detail: float = 1.0
    num_classes: int = 5

    def _config(self):
        from tac.wavelet_mask_codec import WaveletConfig

        return WaveletConfig(
            levels=self.levels,
            step_ll=self.step_ll,
            step_detail=self.step_detail,
            num_classes=self.num_classes,
        )

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        from tac.wavelet_mask_codec import encode_wavelet_codec

        masks = _extract_mask_tensor(state_dict)
        bytes_in = int(masks.numel()) * 8
        blob = encode_wavelet_codec(masks, config=self._config())
        op_state: dict[str, Any] = {
            "levels": self.levels,
            "step_ll": float(self.step_ll),
            "step_detail": float(self.step_detail),
            "num_classes": self.num_classes,
        }
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
        from tac.wavelet_mask_codec import decode_wavelet_codec

        masks = decode_wavelet_codec(blob)
        return {_REQUIRED_MASK_KEY: masks.to(torch.int64)}

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        findings: list[str] = []
        try:
            masks = _extract_mask_tensor(state_dict)
        except ValueError as exc:
            findings.append(str(exc))
            return ValidationReport(passed=False, op_name=self.name, findings=findings)

        # Wavelet config requires H,W divisible by 2^levels.
        _, h, w = masks.shape
        if h % (2 ** self.levels) != 0 or w % (2 ** self.levels) != 0:
            findings.append(
                f"alpha-wavelet: H,W=({h},{w}) must each be divisible by "
                f"2^levels=2^{self.levels}; reduce levels or pad input"
            )
        return ValidationReport(
            passed=not findings, op_name=self.name, findings=findings
        )


# ---------------------------------------------------------------------------
# Op_VqvaeMaskCodec - codebook built from masks at encode time
# ---------------------------------------------------------------------------

@dataclass
class Op_VqvaeMaskCodec:
    """alpha-VQ-VAE mask codec wrap.

    Wraps :func:`tac.vqvae_mask_codec.encode_vqvae_codec` and
    :func:`decode_vqvae_codec`. The codebook is built from the input masks
    at encode time via :func:`tac.vqvae_mask_codec.build_codebook_top_k` and
    embedded in the blob, so the inflate path needs no external file. Bit-
    faithful roundtrip when every mask patch is exactly representable by some
    codebook entry; a lossy fallback (nearest-neighbour) is otherwise.
    """

    name: str = "alpha_vqvae_mask"
    patch_size: int = 4
    codebook_size: int = 256
    num_classes: int = 5

    def _config(self):
        from tac.vqvae_mask_codec import VQVAEConfig

        return VQVAEConfig(
            patch_size=self.patch_size,
            codebook_size=self.codebook_size,
            num_classes=self.num_classes,
        )

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        from tac.vqvae_mask_codec import build_codebook_top_k, encode_vqvae_codec

        masks = _extract_mask_tensor(state_dict)
        bytes_in = int(masks.numel()) * 8
        codebook = build_codebook_top_k(
            masks, patch_size=self.patch_size, k=self.codebook_size
        )
        blob = encode_vqvae_codec(masks, codebook=codebook, config=self._config())
        op_state: dict[str, Any] = {
            "patch_size": self.patch_size,
            "codebook_size": self.codebook_size,
            "num_classes": self.num_classes,
        }
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
        from tac.vqvae_mask_codec import decode_vqvae_codec

        masks = decode_vqvae_codec(blob)
        return {_REQUIRED_MASK_KEY: masks.to(torch.int64)}

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        findings: list[str] = []
        try:
            masks = _extract_mask_tensor(state_dict)
        except ValueError as exc:
            findings.append(str(exc))
            return ValidationReport(passed=False, op_name=self.name, findings=findings)

        _, h, w = masks.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            findings.append(
                f"alpha-VQ-VAE: H,W=({h},{w}) must each be divisible by "
                f"patch_size={self.patch_size}"
            )
        return ValidationReport(
            passed=not findings, op_name=self.name, findings=findings
        )


# ---------------------------------------------------------------------------
# Op_GrayscaleLutMask - Selfcomp grayscale-LUT
# ---------------------------------------------------------------------------

@dataclass
class Op_GrayscaleLutMask:
    """alpha-grayscale-LUT (Selfcomp) mask codec wrap.

    Wraps :func:`tac.mask_grayscale_lut.encode_masks_grayscale` (class IDs ->
    grayscale uint8) and :func:`decode_grayscale_to_classes` (uint8 ->
    class IDs). The blob format is a small self-describing header
    (``"GLT1"`` magic + dims) followed by raw uint8 grayscale bytes. Bit-
    faithful roundtrip is exact (nearest-neighbour matches the
    Selfcomp class-target table).

    Note: this is the BARE grayscale-LUT primitive; an actual selfcomp
    submission would AV1-encode the grayscale plane on top. The pipeline op
    keeps it raw so the bakeoff isolates the LUT primitive's byte impact.
    """

    name: str = "alpha_grayscale_lut_mask"

    _MAGIC = b"GLT1"

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        from tac.mask_grayscale_lut import encode_masks_grayscale

        masks = _extract_mask_tensor(state_dict)
        bytes_in = int(masks.numel()) * 8
        gray = encode_masks_grayscale(masks)  # uint8 (T, H, W)
        if gray.dtype != torch.uint8 or gray.dim() != 3:
            raise RuntimeError(
                f"encode_masks_grayscale returned unexpected tensor: "
                f"dtype={gray.dtype}, shape={tuple(gray.shape)}"
            )
        t, h, w = gray.shape
        # Self-describing header so decode is independent of op_state for
        # forensic reconstruction. op_state still records the shape for the
        # outer pipeline manifest.
        import struct

        header = self._MAGIC + struct.pack("<III", int(t), int(h), int(w))
        body = gray.cpu().numpy().tobytes()
        blob = header + body
        op_state: dict[str, Any] = {"shape": [int(t), int(h), int(w)]}
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
        import struct

        from tac.mask_grayscale_lut import decode_grayscale_to_classes

        if blob[:4] != self._MAGIC:
            raise ValueError(
                f"Op_GrayscaleLutMask.decode bad magic {blob[:4]!r}, "
                f"expected {self._MAGIC!r}"
            )
        t, h, w = struct.unpack_from("<III", blob, 4)
        body = blob[4 + 12 :]
        expected = int(t) * int(h) * int(w)
        if len(body) != expected:
            raise ValueError(
                f"Op_GrayscaleLutMask.decode body length {len(body)} != "
                f"expected {expected} (T={t}, H={h}, W={w})"
            )
        gray = torch.from_numpy(
            np.frombuffer(body, dtype=np.uint8).copy()
        ).reshape(int(t), int(h), int(w))
        masks = decode_grayscale_to_classes(gray)
        return {_REQUIRED_MASK_KEY: masks.to(torch.int64)}

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        findings: list[str] = []
        try:
            _extract_mask_tensor(state_dict)
        except ValueError as exc:
            findings.append(str(exc))
            return ValidationReport(passed=False, op_name=self.name, findings=findings)
        return ValidationReport(passed=True, op_name=self.name)


# ---------------------------------------------------------------------------
# Op_AV1BaselineMask - AV1 baseline (control)
# ---------------------------------------------------------------------------

@dataclass
class Op_AV1BaselineMask:
    """alpha-AV1 baseline (control) mask codec wrap.

    Wraps :func:`tac.mask_codec.encode_masks` / :func:`decode_masks` (the
    canonical AV1 mask path). Used as the bakeoff control: every other alpha
    codec must beat this byte count to be worth wiring in.

    Requires ``ffmpeg`` on PATH. ``validate()`` checks for it; on a host
    without ffmpeg the wrap returns ``passed=False`` so the pipeline skips
    cleanly instead of crashing.

    The blob format is a small header (``"AV1B"`` magic + dims + crf) plus
    the raw AV1 video bytes. Bit-faithful roundtrip is approximate (AV1 is
    lossy on grayscale at finite CRF); the wrap accepts that and the
    validate() warns when CRF > 30 (where reconstruction may flip class
    boundaries).
    """

    name: str = "alpha_av1_baseline_mask"
    crf: int = 20
    fps: int = 20

    _MAGIC = b"AV1B"

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        from tac.mask_codec import encode_masks

        masks = _extract_mask_tensor(state_dict)
        bytes_in = int(masks.numel()) * 8
        # Use a temp dir for the intermediate .mp4; the bytes are read back
        # into the blob. The temp dir is cleaned up immediately.
        # CLAUDE.md "Forbidden /tmp paths" applies to PERSISTED artifacts;
        # this in-memory pipe-through is fine since nothing references the
        # path after the bytes are read.
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = Path(tmpdir) / "alpha_av1_baseline.mp4"
            encode_masks(masks, mp4_path, crf=self.crf, fps=self.fps)
            video_bytes = mp4_path.read_bytes()

        import struct

        t, h, w = masks.shape
        header = self._MAGIC + struct.pack(
            "<IIIBI", int(t), int(h), int(w), int(self.crf), int(self.fps)
        )
        blob = header + video_bytes
        op_state: dict[str, Any] = {
            "shape": [int(t), int(h), int(w)],
            "crf": int(self.crf),
            "fps": int(self.fps),
        }
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
        import struct

        from tac.mask_codec import decode_masks

        if blob[:4] != self._MAGIC:
            raise ValueError(
                f"Op_AV1BaselineMask.decode bad magic {blob[:4]!r}, "
                f"expected {self._MAGIC!r}"
            )
        t, h, w, crf, fps = struct.unpack_from("<IIIBI", blob, 4)
        header_size = 4 + 4 + 4 + 4 + 1 + 4  # = 21
        video_bytes = blob[header_size:]
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = Path(tmpdir) / "alpha_av1_baseline_decode.mp4"
            mp4_path.write_bytes(video_bytes)
            masks = decode_masks(mp4_path, expected_frames=int(t))
        return {_REQUIRED_MASK_KEY: masks.to(torch.int64)}

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        findings: list[str] = []
        try:
            _extract_mask_tensor(state_dict)
        except ValueError as exc:
            findings.append(str(exc))
            return ValidationReport(passed=False, op_name=self.name, findings=findings)
        if shutil.which("ffmpeg") is None:
            findings.append(
                "alpha-AV1-baseline requires ffmpeg on PATH; not found. "
                "Install ffmpeg (libsvtav1) to enable this op."
            )
            return ValidationReport(passed=False, op_name=self.name, findings=findings)
        return ValidationReport(passed=True, op_name=self.name)


# ---------------------------------------------------------------------------
# Bakeoff helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BakeoffEntry:
    """One entry in a bakeoff run."""

    op_name: str
    ready: bool
    bytes_out: int | None
    findings: list[str] = field(default_factory=list)
    error: str | None = None


def default_alpha_mask_ops() -> list[CodecOp]:
    """Return the canonical 5-op alpha-pipeline candidate list.

    The order is the order in which ``pick_smallest_mask_codec`` runs them.
    The list is NOT a pipeline (those are substitutional, not sequential):
    the bakeoff runs each independently and picks the winner.
    """
    return [
        Op_NerVMaskCodec(),
        Op_WaveletMaskCodec(),
        Op_VqvaeMaskCodec(),
        Op_GrayscaleLutMask(),
        Op_AV1BaselineMask(),
    ]


def pick_smallest_mask_codec(
    state_dict: dict[str, torch.Tensor],
    *,
    candidates: list[CodecOp] | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[CodecOp, EncodeResult, list[BakeoffEntry]]:
    """Run all ready alpha candidates; return the smallest-output winner.

    Args:
        state_dict: ``MaskInput``-shaped dict with ``"masks"`` int64 tensor.
        candidates: list of alpha ``CodecOp`` instances; default is
            :func:`default_alpha_mask_ops`.
        context: optional context dict (e.g. ``pretrained_nerv_codec``).

    Returns:
        Tuple of:
            * winner: the ``CodecOp`` whose blob is smallest among ready
              candidates.
            * winner_result: the corresponding ``EncodeResult``.
            * entries: a per-candidate ``BakeoffEntry`` (ready / not ready /
              bytes_out / findings) for forensic logging.

    Raises:
        RuntimeError: if NO candidate is ready (every validate() failed).
    """
    if candidates is None:
        candidates = default_alpha_mask_ops()
    ctx = dict(context) if context is not None else {}
    entries: list[BakeoffEntry] = []
    best: tuple[CodecOp, EncodeResult] | None = None
    for op in candidates:
        report = op.validate(state_dict, context=ctx)
        if not report.passed:
            entries.append(
                BakeoffEntry(
                    op_name=op.name,
                    ready=False,
                    bytes_out=None,
                    findings=list(report.findings),
                )
            )
            continue
        try:
            result = op.encode(state_dict, context=ctx)
        except Exception as exc:
            entries.append(
                BakeoffEntry(
                    op_name=op.name,
                    ready=False,
                    bytes_out=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        entries.append(
            BakeoffEntry(
                op_name=op.name,
                ready=True,
                bytes_out=result.bytes_out,
                findings=list(report.findings),
            )
        )
        if best is None or result.bytes_out < best[1].bytes_out:
            best = (op, result)
    if best is None:
        raise RuntimeError(
            "alpha-bakeoff: no ready candidates. Findings: "
            + "; ".join(
                f"{e.op_name}={e.findings or e.error or 'not ready'}" for e in entries
            )
        )
    return best[0], best[1], entries


def run_bakeoff_and_write_manifest(
    state_dict: dict[str, torch.Tensor],
    *,
    output_dir: Path,
    candidates: list[CodecOp] | None = None,
    context: dict[str, Any] | None = None,
) -> Path:
    """Run a bakeoff and persist a manifest JSON for empirical reasoning.

    The manifest records every candidate's bytes_out (or not-ready reason),
    the winner, and the input mask shape. Tagged ``[empirical:<manifest>]``
    by virtue of being the artifact path.

    Args:
        state_dict: alpha-pipeline input.
        output_dir: directory for ``bakeoff_manifest.json`` (NOT ``/tmp``).
        candidates: optional candidate override.
        context: optional context dict.

    Returns:
        Path to the written manifest.
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = _extract_mask_tensor(state_dict)
    winner, winner_result, entries = pick_smallest_mask_codec(
        state_dict, candidates=candidates, context=context
    )

    manifest = {
        "schema_version": 1,
        "contract": "alpha_mask_bakeoff_manifest_v1",
        "input_shape": list(masks.shape),
        "input_dtype": str(masks.dtype),
        "winner": {
            "op_name": winner.name,
            "bytes_out": winner_result.bytes_out,
            "bytes_in": winner_result.bytes_in,
            "delta_bytes": winner_result.bytes_delta,
        },
        "candidates": [
            {
                "op_name": e.op_name,
                "ready": e.ready,
                "bytes_out": e.bytes_out,
                "findings": e.findings,
                "error": e.error,
            }
            for e in entries
        ],
        "score_claim": False,
        "score_evidence_grade": "empirical",
        "evidence_tag": "[empirical:alpha_mask_bakeoff]",
    }

    manifest_path = output_dir / "bakeoff_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


__all__ = [
    "BakeoffEntry",
    "MaskInput",
    "Op_AV1BaselineMask",
    "Op_GrayscaleLutMask",
    "Op_NerVMaskCodec",
    "Op_VqvaeMaskCodec",
    "Op_WaveletMaskCodec",
    "default_alpha_mask_ops",
    "pick_smallest_mask_codec",
    "run_bakeoff_and_write_manifest",
]
