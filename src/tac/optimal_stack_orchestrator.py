"""Phase 4 final integration — Optimal Stack Orchestrator.

This module is the **integration deliverable** of the 6-month plan
compressed into the time remaining to deadline (per
``feedback_full_six_month_plan_aggressive_no_shortcuts_20260430.md``). It
composes the paradigm-shift outputs (α/β/γ/δ/ε/ζ) into a single
``archive.zip`` with a top-level ``OPTSTACK`` manifest header.

Canonical composition order
---------------------------

Per Grand Council #294 (`grand_council_paradigm_shift_to_shannon_floor_20260430.md`)
Section 4 + ``project_codec_stacking_composition_canonical_orders_20260429.md``::

    representation → prediction → quantization → hyperprior →
    arithmetic → pack

That maps onto the per-stream codec pipeline:

==================  ============================================================
Stream              Pipeline (paradigm shift letters in brackets)
==================  ============================================================
masks.mkv           NeRV / wavelet / STC clean-source [α]
                      → arithmetic terminal [γ]
renderer.bin        Sensitivity map [β] → IMP [ε] → block-FP / Self-Compress NN
                      [β + ε] → Ballé hyperprior [γ] → arithmetic terminal [γ]
                      → wraps as OWV3 / OWV2 / IMPS payload (existing magic
                      bytes preserved)
optimized_poses.pt  PFP16 cast (Lane PFP16 ε-class).
manifest            OPTSTACK header pointing to per-stream codec choices and
                      provenance. Decoder dispatches on each stream's existing
                      magic bytes (no manifest dependency at decode time).
==================  ============================================================

Why a manifest at decode time?
------------------------------

The contest inflate.sh enumerates ``archive.zip`` and dispatches per file
name:

* ``renderer.bin`` → ``submissions/robust_current/inflate_renderer.py``
* ``masks.mkv`` → AV1/codec-aware decoder
* ``optimized_poses.pt`` → torch.load

Each paradigm codec already ships with a 4-byte magic prefix (``ASYM``,
``OWV2``, ``OWV3``, ``IMPS``, …) registered in
``tac.codec_magic_registry``. Decoder dispatch is *purely* magic-byte
driven; the OPTSTACK manifest is **provenance only** — it lets us audit
which paradigm shift contributed which bytes without changing the inflate
contract. This is critical for the strict-scorer-rule (no scorer at
inflate) and for reproducing builds exactly.

Strategic Secrecy
-----------------

The OPTSTACK manifest writes layer-level provenance (which codec, which
bit-budget, which sensitivity-map sha) into ``optstack_manifest.json``
inside the archive. That JSON is **NEVER** consumed at decode time. It
exists for our internal forensics and can be stripped at submission time
via ``OptimalStackOrchestrator.build(strip_manifest=True)`` to honor the
Strategic Secrecy Rule (`CLAUDE.md`).

Tagging discipline
------------------

Every score this module derives is ``[derivation]`` until contest-CUDA
verifies the bytes through ``inflate.sh → upstream/evaluate.py``.
``OptimalStackOrchestrator.build()`` returns a ``BuildResult`` whose
``predicted_score`` is tagged ``[derivation]`` and the caller is required
to log a follow-up ``[contest-CUDA]`` measurement.

This module does NOT touch the upstream snapshot, does NOT load scorers
at decode time, and does NOT emit any score claim that hasn't been
contest-CUDA-validated upstream. Its only job is **composition**.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import zipfile
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Mapping, Sequence

logger = logging.getLogger(__name__)

# Manifest magic. Eight bytes — registry already covers 4-byte codec
# magic. The manifest sits inside ``optstack_manifest.json`` (UTF-8
# text); the magic prefix is purely advisory and lets forensics tools
# spot an OptStack archive at a glance without unpacking. Following the
# Boyd-ADMM contract, the manifest is NOT a codec — it is a sidecar.
OPTSTACK_MANIFEST_MAGIC = b"OPTSTACK"
OPTSTACK_MANIFEST_VERSION = 1

# Canonical contest constants (mirrored from upstream/evaluate.py to
# avoid scorer load at decode time). Score formula:
#   score = 100*seg + sqrt(10*pose) + 25 * archive_bytes / FRAME_BYTES
FRAME_BYTES_DENOMINATOR = 37_545_489
LANE_G_V3_BASELINE_SCORE = 1.05  # [contest-CUDA] reference anchor

# Recognized stream names. Decoder dispatches on file name first, then
# 4-byte magic per stream (existing contract).
RENDERER_FILENAME = "renderer.bin"
MASKS_FILENAME = "masks.mkv"
POSES_FILENAME = "optimized_poses.pt"
MANIFEST_FILENAME = "optstack_manifest.json"


@dataclass(frozen=True)
class StreamProvenance:
    """Provenance record for one stream of the optimal stack.

    The orchestrator never re-encodes — it consumes already-encoded
    bytes from the per-paradigm builders and records what produced
    them. This keeps the orchestrator side-effect free at decode time
    and makes the per-stream codec choices auditable from the
    manifest.
    """

    name: str
    bytes_size: int
    sha256: str
    codec_magic: str
    """Either a registered 4-byte magic (e.g. ``OWV3``) or a sentinel
    like ``AV1`` / ``RAW`` for streams that don't go through the codec
    registry."""
    paradigm_shifts: tuple[str, ...]
    """Letters from the Grand Council paradigm taxonomy
    (``α``, ``β``, ``γ``, ``δ``, ``ε``, ``ζ``). May be empty for
    untouched streams (e.g. Lane G v3 baseline mask)."""
    source_lane: str
    """Lane id whose builder produced these bytes (for cross-ref to
    the lane maturity registry)."""


@dataclass(frozen=True)
class BuildResult:
    """Returned by :func:`OptimalStackOrchestrator.build`."""

    archive_path: Path
    archive_bytes: int
    archive_sha256: str
    predicted_score: float
    """Predicted score from rate term only (component distortions are
    inherited from the source streams). Tagged ``[derivation]``;
    contest-CUDA is required for any promotion."""
    predicted_score_tag: str
    streams: tuple[StreamProvenance, ...]
    manifest_stripped: bool
    elapsed_seconds: float


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _zinfo(name: str) -> zipfile.ZipInfo:
    """Deterministic ZipInfo. Matches existing builders
    (e.g. ``experiments/build_lane_g_v3_owv3_stack.py``) so archive
    bytes are reproducible across hosts.

    See :ref:`feedback_zip_dep_bootstrap_trap` for why we never call
    the shell ``zip`` binary or use Python ``zipfile.ZipFile.write``
    (whose mtime is non-deterministic).
    """

    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o644 & 0xFFFF) << 16
    return info


def _classify_codec_magic(payload: bytes) -> str:
    """Inspect the first 4 bytes; return a canonical magic string.

    Falls back to a sentinel for non-registry payloads (e.g. AV1
    masks.mkv whose first 4 bytes are ``\\x1aE\\xdf\\xa3`` — the
    Matroska EBML head).
    """

    if len(payload) < 4:
        return "RAW"
    head = payload[:4]
    # Canonical codec magic registry first
    try:
        from tac.codec_magic_registry import find_by_magic

        entry = find_by_magic(head)
        if entry is not None:
            return entry.magic.decode("ascii")
    except Exception:  # registry import failure is a deeper bug; fail loud
        raise
    # Recognized non-registry sentinels for human-readable provenance
    if head == b"\x1aE\xdf\xa3":
        return "MKV"  # Matroska EBML head — masks.mkv (AV1 monochrome)
    if head == b"ASYM":
        return "ASYM"  # Lane G v3 baseline renderer
    if head[:2] == b"PK":
        return "ZIP"
    return f"UNKNOWN:{head.hex()}"


def _validate_score_components(seg: float, pose: float) -> None:
    """Defensive sanity check — score component distortions must be
    finite, non-negative, and within plausible bounds.

    These are NOT thresholds against the score target; they catch
    upstream corruption (NaN or sign-flipped distortions). Score
    arithmetic decisions live in ``CLAUDE.md`` non-negotiables.
    """

    for name, value in (("seg", seg), ("pose", pose)):
        if value is None:
            return  # baseline-inherit mode; no validation
        if not (value == value):  # NaN check
            raise ValueError(f"{name}_distortion is NaN")
        if value < 0:
            raise ValueError(f"{name}_distortion={value!r} must be >= 0")
        if value > 1e3:  # plausible upper bound
            raise ValueError(
                f"{name}_distortion={value!r} unreasonably large; "
                "check upstream measurement"
            )


@dataclass
class StreamInput:
    """Caller-provided input to the orchestrator. Each stream is
    already encoded bytes (paradigm-shift agents produce these);
    orchestrator just composes."""

    name: str
    payload: bytes
    paradigm_shifts: Sequence[str] = ()
    source_lane: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.payload, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"StreamInput.payload must be bytes-like, got {type(self.payload).__name__}"
            )
        self.payload = bytes(self.payload)
        if self.name not in (RENDERER_FILENAME, MASKS_FILENAME, POSES_FILENAME):
            raise ValueError(
                f"Unknown stream name {self.name!r}; expected one of "
                f"{RENDERER_FILENAME!r}, {MASKS_FILENAME!r}, {POSES_FILENAME!r}"
            )
        for letter in self.paradigm_shifts:
            if letter not in {"α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι"}:
                raise ValueError(
                    f"unknown paradigm shift letter {letter!r}; valid set "
                    "α/β/γ/δ/ε/ζ/η/θ/ι (Grand Council taxonomy)"
                )


class OptimalStackOrchestrator:
    """Compose paradigm-shift outputs into a single archive.

    Usage::

        orch = OptimalStackOrchestrator(
            renderer_stream=StreamInput(
                name=RENDERER_FILENAME,
                payload=owv3_blob,            # produced by paradigm β
                paradigm_shifts=("β", "ε"),
                source_lane="lane_g_v3_owv3",
            ),
            masks_stream=StreamInput(
                name=MASKS_FILENAME,
                payload=nerv_blob,             # produced by paradigm α
                paradigm_shifts=("α",),
                source_lane="lane_12_nerv_mask_codec",
            ),
            poses_stream=StreamInput(
                name=POSES_FILENAME,
                payload=pfp16_blob,            # produced by paradigm ε
                paradigm_shifts=("ε",),
                source_lane="lane_pfp16",
            ),
            baseline_score=1.05,
            baseline_archive_bytes=694_074,
        )
        result = orch.build(output=Path("archive_optstack_v1.zip"))

    The orchestrator NEVER computes or modifies stream payloads.
    Caller is responsible for the contest-CUDA test of each input
    stream (per the auth-eval-everywhere CLAUDE.md non-negotiable).

    Predicted score
    ---------------

    Returned ``predicted_score`` covers ONLY the rate term delta (per
    score formula). Distortion deltas are inherited from the source
    streams, which the caller has already auth-eval-tested. The label
    ``[derivation]`` is preserved until ``contest_auth_eval.py``
    confirms the FINAL stacked bytes; tag-promotion is a strict policy
    enforced upstream.
    """

    def __init__(
        self,
        renderer_stream: StreamInput,
        masks_stream: StreamInput,
        poses_stream: StreamInput,
        *,
        baseline_score: float = LANE_G_V3_BASELINE_SCORE,
        baseline_archive_bytes: int = 694_074,
        component_seg_dist: float | None = None,
        component_pose_dist: float | None = None,
    ) -> None:
        if renderer_stream.name != RENDERER_FILENAME:
            raise ValueError(
                f"renderer_stream.name must be {RENDERER_FILENAME!r}"
            )
        if masks_stream.name != MASKS_FILENAME:
            raise ValueError(f"masks_stream.name must be {MASKS_FILENAME!r}")
        if poses_stream.name != POSES_FILENAME:
            raise ValueError(f"poses_stream.name must be {POSES_FILENAME!r}")
        if baseline_archive_bytes <= 0:
            raise ValueError(
                f"baseline_archive_bytes must be positive, got {baseline_archive_bytes}"
            )
        if not (0 < baseline_score < 200):
            raise ValueError(
                f"baseline_score={baseline_score!r} outside plausible range "
                "(0, 200)"
            )
        _validate_score_components(component_seg_dist, component_pose_dist)

        self.renderer_stream = renderer_stream
        self.masks_stream = masks_stream
        self.poses_stream = poses_stream
        self.baseline_score = baseline_score
        self.baseline_archive_bytes = baseline_archive_bytes
        self.component_seg_dist = component_seg_dist
        self.component_pose_dist = component_pose_dist

    # ── public API ────────────────────────────────────────────────

    def build(
        self,
        output: Path,
        *,
        strip_manifest: bool = False,
        provenance_json: Path | None = None,
    ) -> BuildResult:
        """Compose the three streams into a deterministic ZIP archive.

        Parameters
        ----------
        output:
            Destination archive path. Parent directory created if
            missing.
        strip_manifest:
            If True (Strategic Secrecy mode for public submission),
            do NOT include ``optstack_manifest.json`` in the archive.
            Decoder doesn't need it; it exists for internal forensics
            only. CLAUDE.md "Strategic Secrecy Rule" governs the
            decision; default is False (manifest IN) for our internal
            builds.
        provenance_json:
            Optional sidecar provenance file (always written, never
            shipped inside the archive). Mirrors the manifest content
            with extra build metadata. Used by Phase 4 reproducibility
            harness.

        Returns
        -------
        BuildResult
            Archive metadata + predicted score (rate term only,
            tagged ``[derivation]``).
        """

        t0 = time.monotonic()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        streams = self._collect_provenance()
        manifest = self._build_manifest(streams)
        manifest_bytes = json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")

        with zipfile.ZipFile(
            output, "w", zipfile.ZIP_DEFLATED, compresslevel=9
        ) as zf:
            # Canonical order: renderer first, masks second, poses
            # third, manifest last. This matches the inflate.sh
            # enumeration order for forensic-clean diffing.
            zf.writestr(_zinfo(RENDERER_FILENAME), self.renderer_stream.payload)
            zf.writestr(_zinfo(MASKS_FILENAME), self.masks_stream.payload)
            zf.writestr(_zinfo(POSES_FILENAME), self.poses_stream.payload)
            if not strip_manifest:
                zf.writestr(_zinfo(MANIFEST_FILENAME), manifest_bytes)

        archive_bytes = output.stat().st_size
        archive_sha256 = _sha256_path(output)
        predicted_score = self._predicted_score(archive_bytes)

        result = BuildResult(
            archive_path=output,
            archive_bytes=archive_bytes,
            archive_sha256=archive_sha256,
            predicted_score=predicted_score,
            predicted_score_tag="[derivation]",
            streams=streams,
            manifest_stripped=strip_manifest,
            elapsed_seconds=time.monotonic() - t0,
        )
        if provenance_json is not None:
            self._write_provenance(provenance_json, result, manifest)
        return result

    # ── helpers ───────────────────────────────────────────────────

    def _collect_provenance(self) -> tuple[StreamProvenance, ...]:
        prov = []
        for stream in (
            self.renderer_stream,
            self.masks_stream,
            self.poses_stream,
        ):
            prov.append(
                StreamProvenance(
                    name=stream.name,
                    bytes_size=len(stream.payload),
                    sha256=_sha256_bytes(stream.payload),
                    codec_magic=_classify_codec_magic(stream.payload),
                    paradigm_shifts=tuple(stream.paradigm_shifts),
                    source_lane=stream.source_lane,
                )
            )
        return tuple(prov)

    def _build_manifest(
        self, streams: tuple[StreamProvenance, ...]
    ) -> dict[str, object]:
        return {
            "format": "optstack_manifest_v1",
            "magic": OPTSTACK_MANIFEST_MAGIC.decode("ascii"),
            "version": OPTSTACK_MANIFEST_VERSION,
            "council_reference": (
                "grand_council_paradigm_shift_to_shannon_floor_20260430.md"
            ),
            "composition_order": [
                "representation",
                "prediction",
                "quantization",
                "hyperprior",
                "arithmetic",
                "pack",
            ],
            "baseline_score": self.baseline_score,
            "baseline_score_tag": "[contest-CUDA]",
            "baseline_archive_bytes": self.baseline_archive_bytes,
            "frame_bytes_denominator": FRAME_BYTES_DENOMINATOR,
            "streams": [asdict(s) for s in streams],
            "decode_contract": (
                "manifest is provenance only; decoder dispatches on per-stream "
                "magic bytes via tac.codec_magic_registry. NO scorer load at "
                "decode time. Strict-scorer-rule compliant."
            ),
            "score_validation_required": (
                "contest-CUDA via experiments/contest_auth_eval.py — predicted "
                "score below is [derivation] from rate term only."
            ),
        }

    def _predicted_score(self, archive_bytes: int) -> float:
        rate_delta = (
            25.0
            * (archive_bytes - self.baseline_archive_bytes)
            / FRAME_BYTES_DENOMINATOR
        )
        # If the caller furnished component distortions, replace the
        # baseline contribution with measurement; otherwise inherit.
        score = self.baseline_score + rate_delta
        if self.component_seg_dist is not None:
            # Reconstitute the score from components (delta from
            # baseline = (new_seg - baseline_seg)*100 already absorbed
            # by predicting from absolute components when supplied).
            #
            # Without baseline component distortions, we can only
            # report the rate delta. The caller is responsible for the
            # full math when measurements exist; this is a derivation
            # from rate alone otherwise.
            pass
        return float(score)

    def _write_provenance(
        self,
        provenance_json: Path,
        result: BuildResult,
        manifest: Mapping[str, object],
    ) -> None:
        provenance_json.parent.mkdir(parents=True, exist_ok=True)
        prov = {
            "format": "optstack_provenance_v1",
            "manifest": manifest,
            "build": {
                "archive_path": str(result.archive_path),
                "archive_bytes": result.archive_bytes,
                "archive_sha256": result.archive_sha256,
                "predicted_score": result.predicted_score,
                "predicted_score_tag": result.predicted_score_tag,
                "manifest_stripped": result.manifest_stripped,
                "elapsed_seconds": result.elapsed_seconds,
                "iso_completed": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            "strict_scorer_rule": (
                "compliant — manifest is provenance only; decode dispatch is "
                "magic-byte driven via tac.codec_magic_registry."
            ),
            "strategic_secrecy_rule": (
                "manifest_stripped=True is the public-submission mode; "
                "default False is for internal builds."
            ),
        }
        provenance_json.write_text(json.dumps(prov, indent=2))


__all__ = [
    "BuildResult",
    "OptimalStackOrchestrator",
    "StreamInput",
    "StreamProvenance",
    "OPTSTACK_MANIFEST_MAGIC",
    "OPTSTACK_MANIFEST_VERSION",
    "MANIFEST_FILENAME",
    "RENDERER_FILENAME",
    "MASKS_FILENAME",
    "POSES_FILENAME",
    "FRAME_BYTES_DENOMINATOR",
    "LANE_G_V3_BASELINE_SCORE",
]
