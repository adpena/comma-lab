"""ddm_iv1 — the semantic tensor as a POSE ACTUATOR (the sa1 inversion).

sa1 measured the rr4/cp135 semantic block as nearly seg-INERT and strongly
pose-LOAD-BEARING: three mechanistically distinct *lossy* edits each paid
68-512x their rate credit in pose while d_seg moved <= +4.8e-6.  sa1 read that
as a refusal.  Read as an *actuator datasheet* it says the opposite thing: this
block is a knob that moves d_pose with almost no d_seg cross-talk.

This arm inverts it.  Instead of perturbing the tensor to BUY BYTES (which
costs pose), it re-solves the tensor's VALUES in the d_pose-DESCENT direction
at ~zero byte cost.

Structural facts established by this module's own preflight, not assumed:

* ``cpr1/inflate.py::render_video`` writes ``output[2p+1]`` (frame_1) from
  ``SemanticTokenRenderer`` and ``output[2p]`` (frame_0) from the carrier.  The
  semantic actuator therefore moves frame_1 ONLY.
* ``frame_embed.weight`` is ``nn.Embedding(600, 8)`` -- one row PER PAIR.  In
  the shipped WANS1 stream it is ``int8`` codes times 8 per-column scales, so a
  pair's actuator is exactly 8 small integers on a fixed grid.
* ``WANS_BODY_BYTES == 36_040`` is a FIXED-LENGTH body.  Re-solving codes
  cannot change the body length; only the brotli-q11 stream over it moves.
  The byte delta is therefore small but NOT assumed -- it is measured by
  rebuilding the real archive.

WHY THIS IS NOT pk4.  pk4 fit a *model* (linear overlays from Jacobians on
train pairs) and applied it to unseen pairs; 23/23 in-sample winners were 0/23
LOO.  Here the actuator is a per-pair DECISION VARIABLE that the encoder sets
using that pair's own ground truth, and every accepted move is measured on the
real objective through the real decode.  There is no fitted model to
generalize.  The pk4 hazard is nonetheless honoured two ways: (1) a HELDOUT
block of pairs is left untouched as a null control, proving the edit does not
leak across pairs; (2) the GLOBAL-tensor arm, which *does* pose a genuine
generalization question, is measured solve-vs-heldout separately.

Axis: ``[macOS-CPU advisory frozen CPU-torch PoseNet]`` -- score_claim=false,
promotion_eligible=false.  No Modal, no GPU, no MLX (pk4 measured MLX PoseNet
drifting 0.55% rel; torch-CPU is the pose authority).

SCOPE REDUCTION (declared, per the charter-time optimal-form law): the solve
runs on a seeded RANDOM subset of pairs (never a prefix -- m88/m96 prefix-bias
law, and pose prefixes measure 2.5-4.2x HARDER than the population).  This
reduces SCOPE (n), never MECHANISM: every render, uint8 rounding, PoseNet call
and integer move is the exact shipping realization at full resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path("/Users/adpena/Projects/pact")
UPSTREAM: Final = REPO / "upstream"
RUNTIME: Final = Path(
    "/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime"
)
SA1: Final = Path("/Volumes/APDataStore/pact/ddm_sa1")
BASE_ATTEMPT: Final = SA1 / "advisory_n600_cpu/rr4_base/attempt_0002"
BASE_RAW: Final = BASE_ATTEMPT / "work/inflated/0.raw"
BASE_TOKENS: Final = (
    BASE_ATTEMPT / "work/inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
OUTPUT: Final = Path("/Volumes/APDataStore/pact/ddm_iv1")
EXPERIMENT_BOOK: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book"
)

PAIR_COUNT: Final = 600
POSE_DIMENSIONS: Final = 6
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
EVAL_H: Final = 384
EVAL_W: Final = 512
FRAME_DIM: Final = 8
# WANS1 stores W4 codes as 4-bit signed nibbles (renderer_weight_codec
# ``pack_signed(codes, 4)``), so the actuator grid is bounded.  MEASURED on the
# shipped block: codes occupy [-7, +7]; 8 of 4,800 entries sit at the +7 edge.
CODE_MIN: Final = -7  # -8 is a RESERVED WANS1 symbol (renderer_weight_codec:294)
CODE_MAX: Final = 7
WANS1_STRATEGY: Final = "per_tensor"

UNCOMPRESSED_BYTES: Final = 37_545_489
RATE_S_PER_BYTE: Final = 25.0 / UNCOMPRESSED_BYTES

# The bought same-instrument base leg (sa1 advisory adjudication attempt_0002).
BASE_ARCHIVE_SHA: Final = (
    "35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956"
)
BASE_ARCHIVE_BYTES: Final = 181_161
BASE_D_SEG: Final = 0.00042714
BASE_D_POSE: Final = 0.00014747
ADMIT_BAR_S: Final = -3.5e-6

# torch CPU thread configuration recorded in the base decode checkpoint.
TORCH_THREADS: Final = 4
TORCH_INTEROP_THREADS: Final = 1

AXIS: Final = (
    "[macOS-CPU advisory frozen CPU-torch PoseNet; seeded-random pair subset] "
    "NON-PROMOTABLE score_claim=false"
)


class IV1Error(RuntimeError):
    """An iv1 precondition, instrument control, or geometry check failed."""


# ---------------------------------------------------------------------------
# retention (ALWAYS KEEP THE PAYLOAD)
# ---------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return file_record(path)


def retain_json(path: Path, value: Any) -> dict[str, Any]:
    return _atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode())


def retain_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    import io

    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return _atomic_write(path, buffer.getvalue())


# ---------------------------------------------------------------------------
# the shipped semantic block
# ---------------------------------------------------------------------------


def _runtime_paths() -> None:
    for entry in (str(RUNTIME), str(RUNTIME / "cpr1"), str(REPO / "experiments")):
        if entry not in sys.path:
            sys.path.insert(0, entry)


@dataclass(frozen=True)
class SemanticBlock:
    """The decoded WANS1 semantic block plus its integer actuator view."""

    state: OrderedDict
    frame_codes: np.ndarray  # (600, 8) int8 -- the actuator
    frame_scales: np.ndarray  # (8,) float32 -- per-column grid step
    parts: Any
    semantic_blob: bytes

    @classmethod
    def load(cls) -> SemanticBlock:
        import torch

        _runtime_paths()
        import runtime.residual_archive as residual_archive

        archive = RUNTIME / "archive.zip"
        if sha256_file(archive) != BASE_ARCHIVE_SHA:
            raise IV1Error("rr4 base archive sha differs from the sealed value")
        parts = residual_archive.read_residual_archive(archive)
        records = residual_archive.decode_wans1(parts.semantic_blob)
        state: OrderedDict[str, Any] = OrderedDict()
        frame_codes = None
        frame_scales = None
        for record in records:
            values = np.ascontiguousarray(record.values, dtype=np.float32)
            state[record.schema.name] = torch.from_numpy(values)
            if record.schema.name == "frame_embed.weight":
                frame_codes = np.array(record.codes, dtype=np.int16)
                frame_scales = np.array(record.scales, dtype=np.float32)
        if frame_codes is None or frame_scales is None:
            raise IV1Error("frame_embed.weight is absent from the semantic block")
        if frame_codes.shape != (PAIR_COUNT, FRAME_DIM):
            raise IV1Error(f"actuator geometry differs: {frame_codes.shape}")
        if frame_scales.shape != (FRAME_DIM,):
            raise IV1Error(f"actuator scale geometry differs: {frame_scales.shape}")
        # control: values must be exactly codes * per-column scale
        reconstructed = frame_codes.astype(np.float32) * frame_scales[None, :]
        shipped = state["frame_embed.weight"].numpy()
        deviation = float(np.abs(reconstructed - shipped).max())
        if deviation != 0.0:
            raise IV1Error(f"actuator grid control failed: max dev {deviation}")
        return cls(
            state=state,
            frame_codes=frame_codes,
            frame_scales=frame_scales,
            parts=parts,
            semantic_blob=bytes(parts.semantic_blob),
        )

    def renderer(self, frame_codes: np.ndarray | None = None) -> Any:
        """Build the SHIPPING renderer, optionally with a re-solved actuator."""
        import torch

        _runtime_paths()
        import inflate as cpr1_inflate

        network = cpr1_inflate.SemanticTokenRenderer()
        state = OrderedDict((key, value.clone()) for key, value in self.state.items())
        if frame_codes is not None:
            codes = np.asarray(frame_codes, dtype=np.int16)
            if codes.shape != (PAIR_COUNT, FRAME_DIM):
                raise IV1Error(f"actuator geometry differs: {codes.shape}")
            values = codes.astype(np.float32) * self.frame_scales[None, :]
            state["frame_embed.weight"] = torch.from_numpy(values)
        network.load_state_dict(state)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        return network


# ---------------------------------------------------------------------------
# the shipping render path (exactly cpr1/inflate.py::render_video, master leg)
# ---------------------------------------------------------------------------


def render_master(network: Any, tokens: Any, pairs: Sequence[int]) -> Any:
    """frame_1 uint8 at camera resolution for the given pairs.

    Mirrors ``render_video``'s master leg byte for byte: renderer -> bilinear
    to (874, 1164) -> clamp -> round -> uint8.
    """
    import torch
    import torch.nn.functional as functional

    index = torch.as_tensor(list(pairs), dtype=torch.long)
    field = tokens[index].long()
    rendered = network(field, index)
    upsampled = functional.interpolate(
        rendered, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
    )
    return upsampled.clamp(0.0, 255.0).round().to(torch.uint8).permute(0, 2, 3, 1)


def load_tokens() -> Any:
    import torch

    raw = np.memmap(
        BASE_TOKENS, dtype=np.uint8, mode="r", shape=(PAIR_COUNT, EVAL_H, EVAL_W)
    )
    return torch.from_numpy(np.ascontiguousarray(raw))


# ---------------------------------------------------------------------------
# the frozen CPU-torch pose authority
# ---------------------------------------------------------------------------


def load_posenet() -> Any:
    from safetensors.torch import load_file

    sys.path.insert(0, str(UPSTREAM))
    try:
        from modules import PoseNet, posenet_sd_path
    finally:
        sys.path.pop(0)
    network = PoseNet().eval().cpu()
    network.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def pose_vectors(posenet: Any, pairs: Any) -> np.ndarray:
    """PoseNet first-six on (B, 2, H, W, 3) uint8 -- the exact evaluator path."""
    import torch

    if pairs.ndim != 5 or tuple(pairs.shape[1:]) != (2, CAMERA_H, CAMERA_W, 3):
        raise IV1Error(f"PoseNet input geometry differs: {tuple(pairs.shape)}")
    with torch.inference_mode():
        tensor = pairs.permute(0, 1, 4, 2, 3).float()
        output = posenet(posenet.preprocess_input(tensor))["pose"][
            ..., :POSE_DIMENSIONS
        ]
    return output.cpu().numpy().astype(np.float64, copy=False)


def decode_gt_pairs(pairs: Sequence[int]) -> np.ndarray:
    """Canonical GT decode (frame_utils.yuv420_to_rgb); never PyAV rgb24."""
    import av

    sys.path.insert(0, str(UPSTREAM))
    try:
        from frame_utils import yuv420_to_rgb
    finally:
        sys.path.pop(0)

    wanted: dict[int, int] = {}
    for slot, pair in enumerate(pairs):
        wanted[2 * int(pair)] = slot
        wanted[2 * int(pair) + 1] = slot
    result = np.zeros((len(pairs), 2, CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    seen = 0
    container = av.open(str(UPSTREAM / "videos/0.mkv"))
    try:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            slot = wanted.get(index)
            if slot is None:
                continue
            array = np.asarray(yuv420_to_rgb(frame), dtype=np.uint8)
            if array.shape != (CAMERA_H, CAMERA_W, 3):
                raise IV1Error(f"GT frame geometry differs: {array.shape}")
            result[slot, index % 2] = array
            seen += 1
            if seen == 2 * len(pairs):
                break
    finally:
        container.close()
    if seen != 2 * len(pairs):
        raise IV1Error("GT decode did not reach every requested frame")
    return result


# ---------------------------------------------------------------------------
# byte cost of a re-solved actuator (measured on the real archive)
# ---------------------------------------------------------------------------


def rebuild_archive(block: SemanticBlock, frame_codes: np.ndarray) -> tuple[bytes, int]:
    """Rebuild the real archive with a re-solved actuator; return (zip, bytes).

    The WANS1 body is fixed length, so only the brotli-q11 stream moves.  Every
    other section is asserted byte-identical to the base.
    """
    _runtime_paths()
    import ddm_rx1_rate_representation_attack as rx1
    import runtime.residual_archive as residual_archive

    blob = encode_actuator_into_blob(block, frame_codes)
    outer = zipfile.ZipFile(RUNTIME / "archive.zip").read("p")
    header = rx1.RX1_HEADER
    _, _, codec, table_mode, _, hpac_bytes, semantic_bytes, carrier_bytes = (
        header.unpack_from(outer)
    )
    offset = header.size
    hpac_stream = outer[offset : offset + hpac_bytes]
    offset += hpac_bytes
    offset += semantic_bytes
    carrier_stream = outer[offset : offset + carrier_bytes]
    offset += carrier_bytes
    tail = outer[offset:]

    book_src = EXPERIMENT_BOOK / "src"
    sys.path.insert(0, str(book_src))
    try:
        from cpr1_sub4.entropy.renderer_weight_codec import encode_f12_wans_body
    finally:
        sys.path.pop(0)
    body = encode_f12_wans_body(blob, residual_archive.WANS_STREAM_ORDER)
    if len(body) != residual_archive.WANS_BODY_BYTES:
        raise IV1Error(f"WANS body length moved: {len(body)}")
    stream = subprocess.run(
        ["brotli", "-q", "11", "-c"], input=body, capture_output=True, check=True
    ).stdout
    model = rx1.pack_rx1_model(
        hpac_stream, stream, carrier_stream, codec_id=codec, table_mode=table_mode
    )
    archive = rx1.deterministic_zip(model + tail)
    return archive, len(archive)


def encode_actuator_into_blob(block: SemanticBlock, frame_codes: np.ndarray) -> bytes:
    """Substitute new int8 actuator codes into the canonical WANS1 blob.

    The record layout is preserved exactly: same tensor order, same per-column
    scales, same bit widths.  Only the 4,800 int8 code bytes move.
    """
    _runtime_paths()
    import dataclasses

    import runtime.residual_archive as residual_archive

    codes = np.asarray(frame_codes, dtype=np.int16)
    if codes.shape != (PAIR_COUNT, FRAME_DIM):
        raise IV1Error(f"actuator geometry differs: {codes.shape}")
    if int(codes.min()) < CODE_MIN or int(codes.max()) > CODE_MAX:
        raise IV1Error(
            f"actuator codes leave the 4-bit grid [{CODE_MIN}, {CODE_MAX}]: "
            f"[{int(codes.min())}, {int(codes.max())}]"
        )

    book_src = EXPERIMENT_BOOK / "src"
    sys.path.insert(0, str(book_src))
    try:
        from cpr1_sub4.entropy.renderer_weight_codec import decode_wans1, encode_wans1
    finally:
        sys.path.pop(0)

    records = decode_wans1(block.semantic_blob)
    rebuilt = []
    seen = False
    for record in records:
        if record.schema.name == "frame_embed.weight":
            new_codes = codes.astype(np.int8)
            new_values = new_codes.astype(np.float32) * block.frame_scales[None, :]
            record = dataclasses.replace(
                record, codes=new_codes, values=new_values
            )
            seen = True
        rebuilt.append(record)
    if not seen:
        raise IV1Error("actuator record missing from the WANS1 blob")

    blob, _ = encode_wans1(tuple(rebuilt), strategy=WANS1_STRATEGY)
    # control: the blob must decode back to exactly these codes
    for record in residual_archive.decode_wans1(blob):
        if record.schema.name == "frame_embed.weight":
            back = np.array(record.codes, dtype=np.int16)
            if not np.array_equal(back, codes):
                raise IV1Error("actuator parse-back differs from the written codes")
            break
    else:
        raise IV1Error("actuator record missing after re-encode")
    return blob


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def configure_torch() -> dict[str, Any]:
    import torch

    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    torch.manual_seed(20260818)
    return {
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }


def render_variant(
    network: Any, tokens: Any, pair: int, row_values: np.ndarray
) -> Any:
    """Render frame_1 for `pair` with a substituted frame_embed row."""
    import torch

    with torch.no_grad():
        saved = network.frame_embed.weight[pair].clone()
        network.frame_embed.weight[pair] = torch.from_numpy(
            np.asarray(row_values, dtype=np.float32)
        )
        try:
            return render_master(network, tokens, [pair])[0]
        finally:
            network.frame_embed.weight[pair] = saved


def stage_probe(args: argparse.Namespace) -> int:
    """Measure the actuator's REALIZED 6x8 pose response, per pair, on the grid.

    No model is fitted.  Each column of the response matrix is a real render at
    a real integer code, through the real uint8 R operator and the frozen
    CPU-torch PoseNet.
    """
    import torch

    environment = configure_torch()
    started = time.time()
    block = SemanticBlock.load()
    tokens = load_tokens()
    network = block.renderer()
    posenet = load_posenet()

    rng = np.random.default_rng(args.seed)
    pairs = np.sort(rng.choice(PAIR_COUNT, size=args.pairs, replace=False))
    raw = np.memmap(
        BASE_RAW,
        dtype=np.uint8,
        mode="r",
        shape=(PAIR_COUNT * 2, CAMERA_H, CAMERA_W, 3),
    )
    ground_truth = decode_gt_pairs([int(value) for value in pairs])

    rows: list[dict[str, Any]] = []
    responses = np.zeros((len(pairs), FRAME_DIM, 2, POSE_DIMENSIONS))
    residuals = np.zeros((len(pairs), POSE_DIMENSIONS))
    base_errors = np.zeros(len(pairs))

    for slot, pair in enumerate(pairs):
        begin = time.time()
        pair = int(pair)
        frame0 = torch.from_numpy(np.ascontiguousarray(raw[2 * pair]))
        base_frame1 = torch.from_numpy(np.ascontiguousarray(raw[2 * pair + 1]))
        gt_pair = torch.from_numpy(np.ascontiguousarray(ground_truth[slot]))

        stack = [torch.stack([frame0, base_frame1]), gt_pair]
        variants: list[tuple[int, int]] = []
        for dim in range(FRAME_DIM):
            for delta in (1, -1):
                codes = block.frame_codes[pair].astype(np.int32).copy()
                codes[dim] += delta
                values = codes.astype(np.float32) * block.frame_scales
                frame1 = render_variant(network, tokens, pair, values)
                stack.append(torch.stack([frame0, frame1]))
                variants.append((dim, delta))

        vectors = []
        for start in range(0, len(stack), 8):
            chunk = torch.stack(stack[start : start + 8])
            vectors.append(pose_vectors(posenet, chunk))
        vectors = np.concatenate(vectors, axis=0)

        base_vector, gt_vector = vectors[0], vectors[1]
        residual = base_vector - gt_vector
        residuals[slot] = residual
        base_errors[slot] = float(np.mean(np.square(residual)))
        for index, (dim, delta) in enumerate(variants):
            responses[slot, dim, 0 if delta == 1 else 1] = (
                vectors[2 + index] - base_vector
            )

        plus = responses[slot, :, 0, :]
        minus = responses[slot, :, 1, :]
        jacobian = 0.5 * (plus - minus)
        asymmetry = float(
            np.linalg.norm(plus + minus) / max(np.linalg.norm(jacobian), 1e-30)
        )
        quantum = float(np.median(np.linalg.norm(plus, axis=1)))
        rows.append(
            {
                "pair": pair,
                "base_d_pose": base_errors[slot],
                "residual_norm": float(np.linalg.norm(residual)),
                "response_quantum_median": quantum,
                "response_quantum_min": float(
                    np.min(np.linalg.norm(plus, axis=1))
                ),
                "jacobian_singular_values": [
                    float(value) for value in np.linalg.svd(jacobian.T, compute_uv=False)
                ],
                "central_difference_asymmetry": asymmetry,
                "seconds": round(time.time() - begin, 2),
            }
        )
        print(
            f"pair {pair:3d} d_pose {base_errors[slot]:.6e} "
            f"|r| {rows[-1]['residual_norm']:.5f} "
            f"quantum {quantum:.5f} asym {asymmetry:.3f} "
            f"{rows[-1]['seconds']:.1f}s",
            flush=True,
        )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    retained = {
        "responses": retain_npy(output / "retained/probe_responses.npy", responses),
        "residuals": retain_npy(output / "retained/probe_residuals.npy", residuals),
        "pairs": retain_npy(output / "retained/probe_pairs.npy", pairs),
        "frame_codes": retain_npy(
            output / "retained/base_frame_codes.npy", block.frame_codes
        ),
        "frame_scales": retain_npy(
            output / "retained/base_frame_scales.npy", block.frame_scales
        ),
    }

    t4_residual_norm = float(np.sqrt(POSE_DIMENSIONS * 6.88e-6))
    cpu_residual_norm = float(np.sqrt(POSE_DIMENSIONS * BASE_D_POSE))
    receipt = {
        "schema": "ddm_iv1_actuator_probe.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "environment": environment,
        "pair_count": len(pairs),
        "pairs": [int(value) for value in pairs],
        "sample_mean_base_d_pose": float(base_errors.mean()),
        "harness_n600_base_d_pose": BASE_D_POSE,
        "median_response_quantum": float(
            np.median([row["response_quantum_median"] for row in rows])
        ),
        "cpu_base_residual_norm": cpu_residual_norm,
        "t4_base_residual_norm": t4_residual_norm,
        "rows": rows,
        "retained": retained,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    retain_json(output / "receipts/actuator_probe.json", receipt)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "sample_mean_base_d_pose",
                    "harness_n600_base_d_pose",
                    "median_response_quantum",
                    "cpu_base_residual_norm",
                    "t4_base_residual_norm",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    return 0


def stage_solve(args: argparse.Namespace) -> int:
    """Re-solve the per-pair actuator in the d_pose-DESCENT direction.

    Predictions only PROPOSE; REALIZED measurements DECIDE.  A candidate code
    vector is accepted only when its realized d_pose -- rendered through the
    shipping path, rounded to uint8, scored by the frozen CPU-torch PoseNet --
    is strictly below the base.  Nothing is fitted and nothing is extrapolated.

    HELDOUT pairs are deliberately left untouched as a null control: because
    ``frame_embed`` is indexed by pair, an untouched row must leave its pair
    bit-identical.  That is asserted, not assumed.
    """
    import itertools

    import torch

    environment = configure_torch()
    started = time.time()
    block = SemanticBlock.load()
    tokens = load_tokens()
    network = block.renderer()
    posenet = load_posenet()

    rng = np.random.default_rng(args.seed)
    pairs = np.sort(rng.choice(PAIR_COUNT, size=args.pairs, replace=False))
    raw = np.memmap(
        BASE_RAW,
        dtype=np.uint8,
        mode="r",
        shape=(PAIR_COUNT * 2, CAMERA_H, CAMERA_W, 3),
    )
    ground_truth = decode_gt_pairs([int(value) for value in pairs])
    combos = np.array(list(itertools.product([0, 1, -1], repeat=FRAME_DIM)))

    def pose_batch(stack: list[Any]) -> np.ndarray:
        out = []
        for start in range(0, len(stack), 8):
            out.append(pose_vectors(posenet, torch.stack(stack[start : start + 8])))
        return np.concatenate(out, axis=0)

    solved_codes = block.frame_codes.astype(np.int32).copy()
    rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []

    # stratify the split by base pose error using a cheap first pass
    for slot, pair in enumerate(pairs):
        begin = time.time()
        pair = int(pair)
        frame0 = torch.from_numpy(np.ascontiguousarray(raw[2 * pair]))
        base_frame1 = torch.from_numpy(np.ascontiguousarray(raw[2 * pair + 1]))
        gt_pair = torch.from_numpy(np.ascontiguousarray(ground_truth[slot]))
        base_codes = block.frame_codes[pair].astype(np.int32)

        is_heldout = (slot % args.heldout_stride) == (args.seed % args.heldout_stride)

        stack = [torch.stack([frame0, base_frame1]), gt_pair]
        probes: list[tuple[int, int]] = []
        if not is_heldout:
            for dim in range(FRAME_DIM):
                for delta in (1, -1):
                    codes = base_codes.copy()
                    codes[dim] += delta
                    frame1 = render_variant(
                        network, tokens, pair, codes.astype(np.float32) * block.frame_scales
                    )
                    stack.append(torch.stack([frame0, frame1]))
                    probes.append((dim, delta))
        vectors = pose_batch(stack)
        base_vector, gt_vector = vectors[0], vectors[1]
        residual = base_vector - gt_vector
        base_error = float(np.mean(np.square(residual)))

        if is_heldout:
            # control: an untouched row must leave the pair bit-identical
            rerender = render_variant(
                network, tokens, pair, base_codes.astype(np.float32) * block.frame_scales
            )
            deviation = int(
                (rerender.numpy().astype(np.int16) - base_frame1.numpy().astype(np.int16))
                .__abs__()
                .max()
            )
            heldout_rows.append(
                {
                    "pair": pair,
                    "base_d_pose": base_error,
                    "untouched_render_max_abs_deviation": deviation,
                    "seconds": round(time.time() - begin, 2),
                }
            )
            print(
                f"pair {pair:3d} HELDOUT d_pose {base_error:.6e} "
                f"untouched_dev {deviation}",
                flush=True,
            )
            continue

        plus = np.zeros((FRAME_DIM, POSE_DIMENSIONS))
        minus = np.zeros((FRAME_DIM, POSE_DIMENSIONS))
        for index, (dim, delta) in enumerate(probes):
            target = plus if delta == 1 else minus
            target[dim] = vectors[2 + index] - base_vector

        # PROPOSE: additive superposition over the measured per-sign responses
        displacement = np.zeros((len(combos), POSE_DIMENSIONS))
        for dim in range(FRAME_DIM):
            column = combos[:, dim]
            displacement += (column == 1)[:, None] * plus[dim][None, :]
            displacement += (column == -1)[:, None] * minus[dim][None, :]
        predicted = np.mean(np.square(residual[None, :] + displacement), axis=1)
        order = np.argsort(predicted)[: args.verify]
        proposals = [combos[index] for index in order]
        # always carry the exactly-measured best single move as a fallback
        single = np.mean(
            np.square(residual[None, :] + np.concatenate([plus, minus], axis=0)), axis=1
        )
        best_single_index = int(single.argmin())
        fallback = np.zeros(FRAME_DIM, dtype=int)
        fallback[best_single_index % FRAME_DIM] = (
            1 if best_single_index < FRAME_DIM else -1
        )
        proposals.append(fallback)

        # DECIDE: realized measurement of every proposal
        verify_stack = []
        for combo in proposals:
            codes = base_codes + np.asarray(combo, dtype=np.int32)
            frame1 = render_variant(
                network, tokens, pair, codes.astype(np.float32) * block.frame_scales
            )
            verify_stack.append(torch.stack([frame0, frame1]))
        realized_vectors = pose_batch(verify_stack)
        realized = np.mean(
            np.square(realized_vectors - gt_vector[None, :]), axis=1
        )
        winner = int(realized.argmin())
        accepted = bool(realized[winner] < base_error)
        chosen = np.asarray(proposals[winner], dtype=np.int32) if accepted else np.zeros(
            FRAME_DIM, dtype=np.int32
        )
        solved_codes[pair] = base_codes + chosen

        rows.append(
            {
                "pair": pair,
                "base_d_pose": base_error,
                "realized_d_pose": float(realized[winner]) if accepted else base_error,
                "accepted": accepted,
                "chosen_code_delta": [int(value) for value in chosen],
                "l1_step": int(np.abs(chosen).sum()),
                "predicted_d_pose_of_winner": float(predicted[order[winner]])
                if winner < len(order)
                else None,
                "best_predicted_d_pose": float(predicted[order[0]]),
                "best_single_move_d_pose": float(single[best_single_index]),
                "winner_was_fallback": winner == len(proposals) - 1,
                "seconds": round(time.time() - begin, 2),
            }
        )
        print(
            f"pair {pair:3d} SOLVE  {base_error:.6e} -> {rows[-1]['realized_d_pose']:.6e} "
            f"({rows[-1]['realized_d_pose'] / base_error:.4f}) k={list(chosen)} "
            f"{rows[-1]['seconds']:.1f}s",
            flush=True,
        )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    retained = {
        "solved_frame_codes": retain_npy(
            output / "retained/solved_frame_codes.npy", solved_codes
        ),
    }

    base_sum = sum(row["base_d_pose"] for row in rows)
    realized_sum = sum(row["realized_d_pose"] for row in rows)
    n_solved = len(rows)
    receipt = {
        "schema": "ddm_iv1_actuator_solve.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "environment": environment,
        "seed": args.seed,
        "solve_pairs": n_solved,
        "heldout_pairs": len(heldout_rows),
        "solve_mean_base_d_pose": base_sum / max(n_solved, 1),
        "solve_mean_realized_d_pose": realized_sum / max(n_solved, 1),
        "solve_relative_change": (realized_sum / base_sum - 1.0) if base_sum else None,
        "accepted_count": sum(1 for row in rows if row["accepted"]),
        "fallback_winner_count": sum(1 for row in rows if row["winner_was_fallback"]),
        "heldout_mean_base_d_pose": (
            sum(row["base_d_pose"] for row in heldout_rows) / len(heldout_rows)
            if heldout_rows
            else None
        ),
        "heldout_untouched_control_passed": all(
            row["untouched_render_max_abs_deviation"] == 0 for row in heldout_rows
        ),
        "all_pairs_mean_base_d_pose": (
            (base_sum + sum(row["base_d_pose"] for row in heldout_rows))
            / (n_solved + len(heldout_rows))
        ),
        "harness_n600_base_d_pose": BASE_D_POSE,
        "rows": rows,
        "heldout_rows": heldout_rows,
        "retained": retained,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    retain_json(output / "receipts/actuator_solve.json", receipt)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "solve_pairs",
                    "heldout_pairs",
                    "solve_mean_base_d_pose",
                    "solve_mean_realized_d_pose",
                    "solve_relative_change",
                    "accepted_count",
                    "fallback_winner_count",
                    "heldout_untouched_control_passed",
                    "all_pairs_mean_base_d_pose",
                    "harness_n600_base_d_pose",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    return 0


def load_segnet() -> Any:
    from safetensors.torch import load_file

    sys.path.insert(0, str(UPSTREAM))
    try:
        from modules import SegNet, segnet_sd_path
    finally:
        sys.path.pop(0)
    network = SegNet().eval().cpu()
    network.load_state_dict(load_file(segnet_sd_path, device="cpu"))
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def seg_argmax(segnet: Any, pairs: Any) -> Any:
    """SegNet argmax on the LAST frame -- the exact evaluator path."""
    import torch

    with torch.inference_mode():
        tensor = pairs.permute(0, 1, 4, 2, 3).float()
        return segnet(segnet.preprocess_input(tensor)).argmax(dim=1)


def stage_finalize(args: argparse.Namespace) -> int:
    """Repair the grid bound, then measure the REAL archive bytes and d_seg."""
    import torch

    environment = configure_torch()
    started = time.time()
    block = SemanticBlock.load()
    tokens = load_tokens()
    network = block.renderer()
    posenet = load_posenet()
    segnet = load_segnet()

    solved = np.load(args.output / "retained/solved_frame_codes.npy").astype(np.int32)
    solve_receipt = json.loads(
        (args.output / "receipts/actuator_solve.json").read_text()
    )
    raw = np.memmap(
        BASE_RAW,
        dtype=np.uint8,
        mode="r",
        shape=(PAIR_COUNT * 2, CAMERA_H, CAMERA_W, 3),
    )

    # --- repair: -8 is a reserved WANS1 symbol; re-verify any clamped pair ----
    clamped = np.clip(solved, CODE_MIN, CODE_MAX)
    repaired = [int(p) for p in np.where((clamped != solved).any(axis=1))[0]]
    solved = clamped

    solve_rows = {int(row["pair"]): row for row in solve_receipt["rows"]}
    pairs = sorted(solve_rows)
    ground_truth = decode_gt_pairs(pairs)
    gt_slot = {pair: slot for slot, pair in enumerate(pairs)}

    repair_rows: list[dict[str, Any]] = []
    for pair in repaired:
        slot = gt_slot[pair]
        frame0 = torch.from_numpy(np.ascontiguousarray(raw[2 * pair]))
        gt_pair = torch.from_numpy(np.ascontiguousarray(ground_truth[slot]))
        frame1 = render_variant(
            network, tokens, pair, solved[pair].astype(np.float32) * block.frame_scales
        )
        vectors = pose_vectors(
            posenet, torch.stack([torch.stack([frame0, frame1]), gt_pair])
        )
        realized = float(np.mean(np.square(vectors[0] - vectors[1])))
        base_error = float(solve_rows[pair]["base_d_pose"])
        if realized >= base_error:  # clamping destroyed the gain -> revert
            solved[pair] = block.frame_codes[pair]
            realized = base_error
        solve_rows[pair]["realized_d_pose"] = realized
        solve_rows[pair]["clamped_repair"] = True
        repair_rows.append(
            {"pair": pair, "repaired_realized_d_pose": realized, "base": base_error}
        )
        print(f"repaired pair {pair}: realized {realized:.6e} (base {base_error:.6e})")

    # --- the REAL archive -----------------------------------------------------
    _runtime_paths()
    import runtime.residual_archive as residual_archive

    blob = encode_actuator_into_blob(block, solved)
    book_src = EXPERIMENT_BOOK / "src"
    sys.path.insert(0, str(book_src))
    try:
        from cpr1_sub4.entropy.renderer_weight_codec import encode_f12_wans_body
    finally:
        sys.path.pop(0)
    body = encode_f12_wans_body(blob, residual_archive.WANS_STREAM_ORDER)
    body_delta = len(body) - residual_archive.WANS_BODY_BYTES
    archive_bytes = None
    archive_sha = None
    parse_back = None
    if body_delta == 0:
        archive, archive_bytes = rebuild_archive(block, solved)
        archive_sha = sha256_bytes(archive)
        _atomic_write(args.output / "retained/candidate_archive.zip", archive)
        reparsed = residual_archive.read_residual_archive(
            args.output / "retained/candidate_archive.zip"
        )
        parse_back = bool(reparsed.semantic_blob == blob)

    # --- the seg leg (F4) -----------------------------------------------------
    seg_pairs = pairs[: args.seg_pairs]
    base_flips = 0
    candidate_flips = 0
    pixels = 0
    for pair in seg_pairs:
        slot = gt_slot[pair]
        frame0 = torch.from_numpy(np.ascontiguousarray(raw[2 * pair]))
        base_frame1 = torch.from_numpy(np.ascontiguousarray(raw[2 * pair + 1]))
        gt_pair = torch.from_numpy(np.ascontiguousarray(ground_truth[slot]))
        frame1 = render_variant(
            network, tokens, pair, solved[pair].astype(np.float32) * block.frame_scales
        )
        stack = torch.stack(
            [torch.stack([frame0, base_frame1]), torch.stack([frame0, frame1]), gt_pair]
        )
        masks = seg_argmax(segnet, stack)
        base_flips += int((masks[0] != masks[2]).sum())
        candidate_flips += int((masks[1] != masks[2]).sum())
        pixels += int(masks[0].numel())
    base_d_seg = base_flips / max(pixels, 1)
    candidate_d_seg = candidate_flips / max(pixels, 1)

    # --- the arithmetic -------------------------------------------------------
    rows = [solve_rows[pair] for pair in pairs]
    base_mean = float(np.mean([row["base_d_pose"] for row in rows]))
    realized_mean = float(np.mean([row["realized_d_pose"] for row in rows]))
    relative = realized_mean / base_mean - 1.0
    projected_d_pose = BASE_D_POSE * (1.0 + relative)
    pose_term = lambda value: float(np.sqrt(10.0 * value))  # noqa: E731
    receipt = {
        "schema": "ddm_iv1_finalize.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "environment": environment,
        "repaired_pairs": repair_rows,
        "solve_pairs": len(rows),
        "rows_changed": int((solved != block.frame_codes).any(axis=1).sum()),
        "solve_mean_base_d_pose": base_mean,
        "solve_mean_realized_d_pose": realized_mean,
        "solve_relative_change": relative,
        "wans_body_bytes": len(body),
        "wans_body_required": residual_archive.WANS_BODY_BYTES,
        "wans_body_delta": body_delta,
        "archive_bytes": archive_bytes,
        "archive_delta_bytes": (
            archive_bytes - BASE_ARCHIVE_BYTES if archive_bytes else None
        ),
        "archive_sha256": archive_sha,
        "archive_parse_back": parse_back,
        "seg": {
            "pairs": len(seg_pairs),
            "base_d_seg_sample": base_d_seg,
            "candidate_d_seg_sample": candidate_d_seg,
            "delta_d_seg_sample": candidate_d_seg - base_d_seg,
        },
        "advisory_projection_if_all_600_pairs_matched_this_relative_change": {
            "note": (
                "DERIVED projection on the CPU advisory instrument only. The "
                "solved subset is a seeded random sample, so its relative change "
                "is an unbiased estimator of the population relative change ONLY "
                "under the assumption that the per-pair reduction factor is "
                "independent of the pair. NOT a score."
            ),
            "base_d_pose": BASE_D_POSE,
            "projected_d_pose": projected_d_pose,
            "base_pose_term": pose_term(BASE_D_POSE),
            "projected_pose_term": pose_term(projected_d_pose),
            "delta_pose_term_S": pose_term(projected_d_pose) - pose_term(BASE_D_POSE),
        },
        "t4_transfer_gate": {
            "t4_sealed_base_d_pose": 6.88e-6,
            "t4_pose_term": pose_term(6.88e-6),
            "cpu_over_t4_d_pose_ratio": BASE_D_POSE / 6.88e-6,
            "binding_risk": (
                "The CPU advisory instrument reads d_pose 21.4x higher than T4 on "
                "IDENTICAL frames. If that gap is a smooth per-image instrument "
                "bias it cancels and the fix transfers; if it is a GT-decode "
                "difference (DALI/NVDEC vs PyAV -- the named unresolved mechanism) "
                "then minimising the CPU residual moves the render toward the WRONG "
                "target and T4 pose could REGRESS. These two hypotheses predict "
                "opposite signs. Only a T4 leg resolves it."
            ),
            "t4_maximum_possible_pose_credit_S": pose_term(6.88e-6),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    retain_json(args.output / "receipts/finalize.json", receipt)
    retain_npy(args.output / "retained/final_frame_codes.npy", solved)
    print(json.dumps(receipt, indent=2, default=str)[:2600])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["control", "probe", "solve", "finalize"],
    )
    parser.add_argument("--verify", type=int, default=8)
    parser.add_argument("--seg-pairs", type=int, default=12)
    parser.add_argument("--heldout-stride", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    if args.stage == "control":
        return stage_control(args)
    if args.stage == "probe":
        return stage_probe(args)
    if args.stage == "solve":
        return stage_solve(args)
    if args.stage == "finalize":
        return stage_finalize(args)
    raise IV1Error(f"unknown stage {args.stage}")


def stage_control(args: argparse.Namespace) -> int:
    """Instrument control: the torch render must reproduce the shipped bytes."""
    import torch

    environment = configure_torch()
    started = time.time()
    block = SemanticBlock.load()
    tokens = load_tokens()
    network = block.renderer()

    rng = np.random.default_rng(args.seed)
    pairs = np.sort(rng.choice(PAIR_COUNT, size=args.pairs, replace=False))
    raw = np.memmap(
        BASE_RAW,
        dtype=np.uint8,
        mode="r",
        shape=(PAIR_COUNT * 2, CAMERA_H, CAMERA_W, 3),
    )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for pair in pairs:
            begin = time.time()
            rendered = render_master(network, tokens, [int(pair)])[0].numpy()
            shipped = np.asarray(raw[2 * int(pair) + 1])
            difference = np.abs(
                rendered.astype(np.int16) - shipped.astype(np.int16)
            )
            rows.append(
                {
                    "pair": int(pair),
                    "max_abs_deviation": int(difference.max()),
                    "mismatched_pixels": int((difference > 0).sum()),
                    "render_seconds": round(time.time() - begin, 3),
                }
            )
            print(
                f"pair {int(pair):3d} maxdev {rows[-1]['max_abs_deviation']} "
                f"mismatch {rows[-1]['mismatched_pixels']} "
                f"{rows[-1]['render_seconds']:.2f}s",
                flush=True,
            )

    worst = max(row["max_abs_deviation"] for row in rows)
    receipt = {
        "schema": "ddm_iv1_render_control.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "environment": environment,
        "base_archive_sha256": BASE_ARCHIVE_SHA,
        "base_raw": file_record(BASE_RAW),
        "tokens": {"path": str(BASE_TOKENS), "bytes": BASE_TOKENS.stat().st_size},
        "actuator": {
            "name": "frame_embed.weight",
            "shape": list(block.frame_codes.shape),
            "dtype": "int8 codes x per-column float32 scale",
            "scales": [float(value) for value in block.frame_scales],
            "code_min": int(block.frame_codes.min()),
            "code_max": int(block.frame_codes.max()),
        },
        "rows": rows,
        "worst_max_abs_deviation": worst,
        "control_passed": worst == 0,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    retain_json(args.output / "receipts/render_control.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("worst_max_abs_deviation", "control_passed", "elapsed_seconds")}, indent=2))
    return 0 if worst == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
