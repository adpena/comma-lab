#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-SF1: the semantic/FiLM row-group POSE map on the hv1 ep0634 vehicle.

WHY THIS ARM EXISTS.  ``ra2crr`` NEXT_IF_RESUMED row 1b named the live blocker for the banked
mp2 rate candidates: they are held on pose collateral in the SEMANTIC renderer section, and the
map that would say which row groups are pose-safe had never been measured.  This tool measures
it.

WHAT THE INSTRUMENT IS, verified at source before any code was written:

* ``cpr1/inflate.py::render_video`` writes ``output[2*i + 1]`` from the SEMANTIC renderer and
  ``output[2*i]`` from the CARRIER.  The semantic renderer therefore paints **frame_1 only**.
  ``runtime/f26_inflate.py::_apply_frame0_selector`` touches even indices only.  So a semantic
  weight edit cannot move frame_0 -- a claim this tool proves at n600 rather than asserts.
* ``upstream/modules.py::PoseNet.preprocess_input`` keeps BOTH frames (channels 0-5 = frame_0,
  6-11 = frame_1), so a frame_1-only edit does move pose.  ``compute_distortion`` is the MSE of
  the first 6 pose dims.
* ``upstream/modules.py::SegNet.preprocess_input`` slices ``x[:, -1, ...]`` = frame_1, so the
  same edit is fully visible to seg.
* ``experiments/ddm_sm3_semantic_representation.py::pack_prune_candidate`` ranks FiLM rows by
  ``flat.square().sum(dim=1)`` -- the row L2 norm -- and prunes the smallest.  There is no pose
  term, no output-gain term and no score term in that selector despite the candidate id
  ``score_gated_film_row_prune``.
* ``experiments/ddm_mp2_semantic_receiver.py::_decode_row_prune`` reconstructs a pruned row as
  ``torch.zeros``.  So "prune a row" IS "zero a row": the finite difference this tool takes with
  step ``-W_row`` is the exact actuator, not a surrogate for it.

CONSEQUENCE THAT MAKES THE MAP ACTIONABLE.  The SM3R packet spends ``6`` bytes per retained row
(one fp16 scale + eight int4 codes) plus a fixed 24-byte row mask per tensor.  The byte credit
of a prune therefore depends only on HOW MANY rows are dropped, never on WHICH.  If a pose-quiet
subset of equal cardinality exists, it buys the same bytes for less pose.

AXIS.  ``[macOS-CPU advisory]`` throughout -- never a score.  ``d_pose`` is reported against the
authority-tracking DALI GT cache (MEASURED by ``pi2`` at 1.00081x the contest axis) and, for
commensurability with mp2's own table, against the PyAV lineage those rows were drawn on.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import platform
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"

#: The hv1 ep0634 frontier generation: archive + receiver runtime + renderer, custody-pinned.
GEN_ROOT = Path(
    "/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815"
    "/generations/hv1_base_control"
)
MP2_ROOT = Path(
    "/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815")
HV1_ADVISORY = Path("/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2")
RA2CRR_RETAINED = Path("/Volumes/APDataStore/pact/ddm_ra2crr/retained")

#: MEASURED by pi2 to track the contest authority at 1.00081x.  Read-only tier: never written.
GT_CACHE_DALI = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt")

OUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_sf1_semantic_film_pose_map_20260817")

ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
ARCHIVE_BYTES = 182_759
GT_CACHE_SHA_PIN = "a91d98252fe377c5"

FRAMES, POSE_DIMS = 600, 6
CAMERA_H, CAMERA_W = 874, 1164
FRAME_BYTES = CAMERA_H * CAMERA_W * 3
EVAL_H, EVAL_W = 384, 512

S_PER_BYTE = 25.0 / 37_545_489.0
FRONTIER_S = 0.15959729295498598
TARGET_S = 0.15
#: 100 / (600 * 384 * 512) -- one flipped scorer pixel, in S.
SEG_S_PER_FLIP = 100.0 / (FRAMES * EVAL_H * EVAL_W)

#: The three FiLM weights the SM3R row-prune format is allowed to touch.
PRUNE_NAMES = ("blocks.1.film.weight", "blocks.2.film.weight", "blocks.3.film.weight")
FILM_ROWS = 192
#: rows [0, 96) drive the FiLM scale, rows [96, 192) drive the FiLM shift.
FILM_SCALE_ROWS = 96

THREADS = 8


class SF1Refusal(RuntimeError):
    """Fail-closed refusal: missing custody, wrong shape, or a broken control."""


# ---------------------------------------------------------------- io + custody


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    _atomic_bytes(path, payload)
    return {"path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def retain_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    """Persist a measured array and return its custody record.

    ALWAYS KEEP THE PAYLOAD is P0.  Every array this tool measures goes through here.
    """
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array))
    payload = buffer.getvalue()
    _atomic_bytes(path, payload)
    return {"path": str(path), "bytes": len(payload), "dtype": str(array.dtype),
            "shape": list(array.shape), "sha256": hashlib.sha256(payload).hexdigest()}


def retain_raw(path: Path, array: np.ndarray) -> dict[str, Any]:
    """Persist a large uint8 frame stack as flat bytes (npy header would hide the geometry)."""
    payload = np.ascontiguousarray(array, dtype=np.uint8).tobytes()
    _atomic_bytes(path, payload)
    return {"path": str(path), "bytes": len(payload), "dtype": "uint8",
            "shape": list(array.shape), "sha256": hashlib.sha256(payload).hexdigest()}


def require_file(path: Path, *, expect_sha: str | None = None,
                 expect_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise SF1Refusal(f"missing required input: {path}")
    fact = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    if expect_bytes is not None and fact["bytes"] != expect_bytes:
        raise SF1Refusal(f"{path} is {fact['bytes']} B, expected {expect_bytes} B")
    if expect_sha is not None and not fact["sha256"].startswith(expect_sha):
        raise SF1Refusal(f"{path} sha256 {fact['sha256']} does not match pin {expect_sha}")
    return fact


# ------------------------------------------------------------------- renderer


def load_renderer():
    """Load the shipped hv1 renderer module and its receiver runtime, unmodified."""
    renderer_path = GEN_ROOT / "cpr1" / "inflate.py"
    if not renderer_path.is_file():
        raise SF1Refusal(f"renderer not found: {renderer_path}")
    for entry in (str(GEN_ROOT), str(GEN_ROOT / "cpr1")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location("_sf1_renderer", renderer_path)
    if spec is None or spec.loader is None:
        raise SF1Refusal(f"cannot load renderer from {renderer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_sf1_renderer"] = module
    spec.loader.exec_module(module)
    return module


def semantic_base_state(renderer) -> dict[str, np.ndarray]:
    """Decode the shipped semantic weights exactly as ``f26_inflate`` does."""
    import torch
    from runtime.entropy.renderer_weight_codec import decode_wans1
    from runtime.residual_archive import read_residual_archive

    parts = read_residual_archive(GEN_ROOT / "archive.zip")
    template = renderer.SemanticTokenRenderer(renderer.SEMANTIC_WIDTH)
    state = renderer.unpack_variant_semantic_or_none(parts.semantic_blob,
                                                     template.state_dict())
    if state is None:
        state = {
            record.schema.name: torch.from_numpy(
                np.ascontiguousarray(record.values, dtype=np.float32))
            for record in decode_wans1(parts.semantic_blob)
        }
    if len(state) != 38:
        raise SF1Refusal(f"semantic state has {len(state)} tensors, expected 38")
    return state


def build_semantic(renderer, state):
    import torch

    module = renderer.SemanticTokenRenderer(renderer.SEMANTIC_WIDTH)
    module.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()}, strict=True)
    return module.eval()


class Renderer:
    """frame_1 re-render, bit-identical to the shipped raw at the zero perturbation."""

    def __init__(self, renderer_module, tokens: np.ndarray) -> None:
        import torch

        self._torch = torch
        self._mod = renderer_module
        self._tokens = tokens
        torch.set_num_threads(THREADS)

    def frame1(self, semantic, pair: int) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            raw = semantic(
                torch.from_numpy(self._tokens[pair:pair + 1]).long(),
                torch.arange(pair, pair + 1),
            )
            master = (
                torch.nn.functional.interpolate(
                    raw, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
                .clamp(0.0, 255.0)
                .round()
            )
            return master.to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]


class PoseInstrument:
    """Frozen CPU-torch PoseNet, batch = 1 pair, upstream preprocessing verbatim."""

    def __init__(self) -> None:
        import torch

        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0, str(UPSTREAM))
        from modules import PoseNet
        from safetensors.torch import load_file

        torch.set_num_threads(THREADS)
        self._torch = torch
        self.net = PoseNet().eval()
        self.net.load_state_dict(
            load_file(str(UPSTREAM / "models" / "posenet.safetensors")), strict=True)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    def pose6(self, frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            stacked = torch.from_numpy(np.stack([frame0, frame1]))
            batch = stacked.permute(0, 3, 1, 2).float().unsqueeze(0)
            out = self.net(self.net.preprocess_input(batch))
            return out["pose"][0, :POSE_DIMS].numpy().astype(np.float64)


def read_frame(path: Path, index: int) -> np.ndarray:
    array = np.fromfile(path, dtype=np.uint8, count=FRAME_BYTES,
                        offset=index * FRAME_BYTES)
    if array.size != FRAME_BYTES:
        raise SF1Refusal(f"short read of frame {index} from {path}")
    return array.reshape(CAMERA_H, CAMERA_W, 3)


def d_pose(generated: np.ndarray, target: np.ndarray) -> float:
    """upstream ``PoseNet.compute_distortion``: MSE over the first 6 pose dims, then the mean."""
    return float(((generated - target) ** 2).mean(axis=1).mean())


def score_pose(value: float) -> float:
    return float(np.sqrt(10.0 * value))


# --------------------------------------------------------------- row grouping


def film_row_norms(state) -> dict[str, np.ndarray]:
    norms = {}
    for name in PRUNE_NAMES:
        flat = np.asarray(state[name], dtype=np.float64).reshape(FILM_ROWS, -1)
        norms[name] = (flat ** 2).sum(axis=1)
    return norms


def prune_order(norms: np.ndarray) -> list[int]:
    """The shipped selector's own ordering: descending norm, index as the tie-break."""
    return sorted(range(len(norms)), key=lambda index: (-float(norms[index]), index))


def mechanism_groups(norms: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """3 tensors x {scale, shift} x 3 norm terciles = 18 disjoint groups of 32 rows.

    Equal cardinality means equal byte credit, so the groups are directly comparable as
    alternative 32-row prunes at the same rate.
    """
    groups: list[dict[str, Any]] = []
    for name in PRUNE_NAMES:
        block = name.split(".")[1]
        for half, lo, hi in (("scale", 0, FILM_SCALE_ROWS),
                             ("shift", FILM_SCALE_ROWS, FILM_ROWS)):
            rows = list(range(lo, hi))
            ordered = sorted(rows, key=lambda index: (-float(norms[name][index]), index))
            third = len(ordered) // 3
            for tercile, tag in enumerate(("hi", "mid", "lo")):
                chunk = ordered[tercile * third:(tercile + 1) * third]
                groups.append({
                    "group_id": f"b{block}_{half}_{tag}",
                    "family": "mechanism_partition",
                    "rows": {name: sorted(chunk)},
                    "row_count": len(chunk),
                })
    return groups


def keep87_selection(norms: dict[str, np.ndarray]) -> dict[str, list[int]]:
    """Exactly what ``pack_prune_candidate(state, 87)`` drops: the 25 smallest-norm rows."""
    selection: dict[str, list[int]] = {}
    for name in PRUNE_NAMES:
        keep = max(1, round(FILM_ROWS * 87 / 100.0))
        order = prune_order(norms[name])
        selection[name] = sorted(order[keep:])
    return selection


def selector_groups(norms: dict[str, np.ndarray], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    """Same-cardinality alternatives to keep87's choice: random, anti-selected, and mp2's own."""
    incumbent = keep87_selection(norms)
    count = len(incumbent[PRUNE_NAMES[0]])
    groups = [{
        "group_id": "sel_mp2_keep87_lowest_norm",
        "family": "selector_alternative",
        "rows": incumbent,
        "row_count": sum(len(v) for v in incumbent.values()),
    }]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rows = {name: sorted(int(x) for x in rng.choice(FILM_ROWS, size=count, replace=False))
                for name in PRUNE_NAMES}
        groups.append({
            "group_id": f"sel_random_seed{seed}",
            "family": "selector_alternative",
            "rows": rows,
            "row_count": sum(len(v) for v in rows.values()),
        })
    anti = {name: sorted(prune_order(norms[name])[:count]) for name in PRUNE_NAMES}
    groups.append({
        "group_id": "sel_highest_norm_anticontrol",
        "family": "selector_alternative",
        "rows": anti,
        "row_count": sum(len(v) for v in anti.values()),
    })
    return groups


def global_lowest_norm_selection(norms: dict[str, np.ndarray],
                                 count: int) -> dict[str, list[int]]:
    """The ``count`` smallest-norm rows ranked ACROSS all three tensors at once.

    The shipped packer ranks WITHIN each tensor, so it is forced to take rows from
    ``blocks.1.film.weight`` -- whose total row energy is ~150x that of blocks 2 and 3 -- even
    though every row there is larger than every row in the other two.  The SM3R byte credit
    depends only on the COUNT of dropped rows, so re-ranking globally is free.
    """
    ranked = sorted(((float(norms[name][row]), name, row)
                     for name in PRUNE_NAMES for row in range(FILM_ROWS)),
                    key=lambda item: (item[0], item[1], item[2]))
    chosen: dict[str, list[int]] = {name: [] for name in PRUNE_NAMES}
    for _, name, row in ranked[:count]:
        chosen[name].append(row)
    return {name: sorted(rows) for name, rows in chosen.items() if rows}


def global_selection_groups(norms: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Byte-matched global re-rankings of the two mp2 prune counts."""
    groups = []
    for count, label in ((75, "keep87count"), (432, "keep25count")):
        rows = global_lowest_norm_selection(norms, count)
        groups.append({
            "group_id": f"glob_lowest_norm_{label}",
            "family": "global_reselection",
            "rows": rows,
            "row_count": sum(len(v) for v in rows.values()),
        })
    return groups


def null_group() -> dict[str, Any]:
    return {"group_id": "null_zero_perturbation_control", "family": "control",
            "rows": {}, "row_count": 0}


def sm3r_roundtrip_state(state, dropped: dict[str, list[int]]):
    """The weights an SM3R candidate actually decodes to: q4 re-encode PLUS the row zeroing.

    Mirrors ``pack_prune_candidate``'s ``expected`` map with an explicit mask.  This exists to
    SEPARATE two things mp2's candidates conflate: the shipped base is a WANS1 packet, so every
    SM3R candidate re-encodes all 38 tensors at q4 *and* drops rows.  Zeroing rows in the base
    weights measures the prune alone; this measures what the receiver really reconstructs.
    """
    import torch

    sm3 = _load_sm3()
    restored = {}
    for name, raw in state.items():
        value = torch.as_tensor(np.array(raw, copy=True))
        if value.ndim < 2:
            restored[name] = value.detach().cpu().to(torch.float16).float()
        elif name not in sm3.PRUNE_NAMES:
            _, back = sm3.standard_q4_payload(name, value)
            restored[name] = back
        else:
            drop = set(dropped.get(name, ()))
            selected = np.ones(FILM_ROWS, dtype=np.uint8)
            for row in drop:
                selected[row] = 0
            flat = value.detach().cpu().float().reshape(FILM_ROWS, -1)
            keep_mask = torch.from_numpy(selected.astype(bool))
            _, compact = sm3.standard_q4_payload("pruned.rows", flat[keep_mask])
            dense = torch.zeros_like(flat)
            dense[keep_mask] = compact
            restored[name] = dense.reshape(value.shape)
    return restored


def attribution_groups(norms: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Split an SM3R candidate's pose damage into its q4 re-encode half and its row-prune half."""
    return [
        {"group_id": "attr_sm3r_q4_reencode_only", "family": "format_attribution",
         "rows": {}, "row_count": 0, "builder": "sm3r_roundtrip"},
        {"group_id": "attr_sm3r_q4_plus_keep87_rows", "family": "format_attribution",
         "rows": keep87_selection(norms), "row_count": 75, "builder": "sm3r_roundtrip"},
        {"group_id": "attr_zero_all_film_rows", "family": "format_attribution",
         "rows": {name: list(range(FILM_ROWS)) for name in PRUNE_NAMES},
         "row_count": 3 * FILM_ROWS},
    ]


# --------------------------------------------------------------- byte pricing


def _load_sm3():
    """Import the committed SM3R packer so the byte pricing mirrors the shipped format exactly."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return importlib.import_module("experiments.ddm_sm3_semantic_representation")


def sm3r_packet_bytes(state, dropped: dict[str, list[int]]) -> tuple[int, int]:
    """Return (raw packet bytes, brotli-q11 bytes) for a semantic packet with ``dropped`` zeroed.

    The archive stores its single member ``p`` without deflation (182,659 B payload inside a
    182,759 B zip), so a change in the brotli'd semantic section moves ``archive.zip`` by exactly
    the same amount.  Mirrors ``ddm_sm3_semantic_representation.pack_prune_candidate`` byte for
    byte, but takes an explicit mask instead of a keep-percentage.
    """
    import brotli
    import torch

    sm3 = _load_sm3()
    tensors = {k: torch.as_tensor(v) for k, v in state.items()}
    kept_first = next(iter(dropped.values())) if dropped else []
    keep_count = FILM_ROWS - len(kept_first)
    keep_percent = max(1, min(99, round(100.0 * keep_count / FILM_ROWS)))
    payload = bytearray(sm3.representation_header(sm3.MODE_ROW_PRUNE, keep_percent))
    qnames = sm3.sd1.quantized_names(tensors)
    payload.extend(struct.pack("<H", sm3.mask_for_names(qnames, sm3.PRUNE_NAMES)))
    for name, value in tensors.items():
        if value.ndim < 2:
            payload.extend(
                value.detach().cpu().to(torch.float16).numpy().astype("<f2").tobytes())
        elif name not in sm3.PRUNE_NAMES:
            encoded, _ = sm3.standard_q4_payload(name, value)
            payload.extend(encoded)
        else:
            drop = set(dropped.get(name, ()))
            selected = np.ones(FILM_ROWS, dtype=np.uint8)
            for row in drop:
                selected[row] = 0
            payload.extend(np.packbits(selected, bitorder="little").tobytes())
            flat = value.detach().cpu().float().reshape(FILM_ROWS, -1)
            compact = flat[torch.from_numpy(selected.astype(bool))]
            encoded, _ = sm3.standard_q4_payload("pruned.rows", compact)
            payload.extend(encoded)
    raw = bytes(payload)
    return len(raw), len(brotli.compress(raw, quality=11))


# ------------------------------------------------------------------- stages


def _pair_subset(seed: int, count: int) -> np.ndarray:
    """A seeded RANDOM pair subset.  NEVER a contiguous prefix.

    m96 / ddm_na2 MEASURED that pose prefixes read 2.54-4.21x HARDER than the population while
    seg prefixes read 0.95-0.97x easier, so a prefix-based pose verdict is the false-negative
    shape.
    """
    return np.sort(np.random.default_rng(seed).choice(FRAMES, size=count, replace=False))


def stage_controls(args) -> dict[str, Any]:
    import torch

    out = OUT_ROOT / "retained"
    out.mkdir(parents=True, exist_ok=True)
    base_raw = HV1_ADVISORY / "inflated" / "0.raw"
    tokens_path = (HV1_ADVISORY / "inflated" / ".f26_decode_checkpoints"
                   / "tokens_cpu_stage_complete.u8")

    inputs = {
        "archive": require_file(GEN_ROOT / "archive.zip", expect_sha=ARCHIVE_SHA256,
                                expect_bytes=ARCHIVE_BYTES),
        "base_raw": require_file(base_raw, expect_bytes=FRAMES * 2 * FRAME_BYTES),
        "tokens": require_file(tokens_path, expect_bytes=FRAMES * EVAL_H * EVAL_W),
        "gt_cache_dali": require_file(GT_CACHE_DALI, expect_sha=GT_CACHE_SHA_PIN),
        "ra2crr_base_residual": require_file(
            RA2CRR_RETAINED / "ra2crr_base_residual_authority.float64.npy"),
        "ra2crr_authority_gt_pose6": require_file(
            RA2CRR_RETAINED / "ra2crr_authority_gt_pose6.float64.npy"),
    }

    gt_dali = torch.load(io.BytesIO(GT_CACHE_DALI.read_bytes()),
                         map_location="cpu")["pose"].double().numpy()
    if gt_dali.shape != (FRAMES, POSE_DIMS):
        raise SF1Refusal(f"authority GT shape {gt_dali.shape}")

    renderer = load_renderer()
    state = semantic_base_state(renderer)
    semantic = build_semantic(renderer, state)
    tokens = np.fromfile(tokens_path, dtype=np.uint8).reshape(FRAMES, EVAL_H, EVAL_W)
    render = Renderer(renderer, tokens)
    pose = PoseInstrument()

    # Control 1 -- the zero perturbation reproduces the shipped frame_1 BIT-IDENTICALLY.
    identity_pairs = _pair_subset(args.control_seed, args.control_pairs).tolist()
    identical = 0
    for pair in identity_pairs:
        rebuilt = render.frame1(semantic, pair)
        if np.array_equal(rebuilt, read_frame(base_raw, 2 * pair + 1)):
            identical += 1
    if identical != len(identity_pairs):
        raise SF1Refusal(
            f"zero-perturbation render reproduced only {identical}/{len(identity_pairs)} frames")

    # Control 2 -- frame_0 is invariant under every measured semantic edit, at n600.
    frame0_invariance = {}
    for candidate, raw_path in _candidate_raws().items():
        equal = _even_frames_identical(base_raw, raw_path)
        frame0_invariance[candidate] = equal
    if not all(v["frame0_identical_pairs"] == FRAMES for v in frame0_invariance.values()):
        raise SF1Refusal("frame_0 moved under a semantic-only edit; the region model is wrong")

    # Control 3 -- an independent n600 base pose6 must reproduce ra2crr's authority base d_pose.
    started = time.time()
    base_pose6 = np.zeros((FRAMES, POSE_DIMS), dtype=np.float64)
    for pair in range(FRAMES):
        base_pose6[pair] = pose.pose6(read_frame(base_raw, 2 * pair),
                                      read_frame(base_raw, 2 * pair + 1))
    base_d_pose = d_pose(base_pose6, gt_dali)
    ra2crr_base = 6.885595058208011e-06
    relative = abs(base_d_pose - ra2crr_base) / ra2crr_base

    payloads = {
        "base_pose6_authority": retain_array(out / "sf1_base_pose6.float64.npy", base_pose6),
        "authority_gt_pose6": retain_array(out / "sf1_authority_gt_pose6.float64.npy", gt_dali),
        "film_row_norms": retain_array(
            out / "sf1_film_row_norms.float64.npy",
            np.stack([film_row_norms(state)[n] for n in PRUNE_NAMES])),
    }

    record = {
        "schema": "ddm_sf1_controls.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "inputs": inputs,
        "instrument": {
            "renderer": "shipped hv1 cpr1/inflate.py SemanticTokenRenderer, batch = 1 pair",
            "pose": "frozen CPU-torch PoseNet, upstream/models/posenet.safetensors, batch = 1",
            "threads": THREADS,
            "torch_version": torch.__version__,
            "platform": f"{platform.system()}-{platform.machine()}",
        },
        "controls": {
            "zero_perturbation_frame1_bit_identical_pairs": identical,
            "zero_perturbation_pairs_tested": len(identity_pairs),
            "zero_perturbation_pair_seed": args.control_seed,
            "frame0_invariance_n600": frame0_invariance,
            "base_d_pose_authority_gt": base_d_pose,
            "base_d_pose_authority_gt_ra2crr_reference": ra2crr_base,
            "base_d_pose_relative_agreement": relative,
            "base_pose_score_contribution": score_pose(base_d_pose),
            "pose_forward_seconds": (time.time() - started) / FRAMES,
        },
        "retained_payload": payloads,
    }
    record["receipt"] = atomic_json(OUT_ROOT / "SF1_CONTROLS.json", record)
    return record


def _candidate_raws() -> dict[str, Path]:
    root = MP2_ROOT / "advisory_n600_cpu"
    return {
        "score_gated_selected_mixed_q3q4":
            root / "score_gated_selected_mixed_q3q4/attempt_0000/work/inflated/0.raw",
        "score_gated_film_row_prune_keep87":
            root / "score_gated_film_row_prune_keep87/attempt_0000/work/inflated/0.raw",
        "score_gated_film_row_prune_keep75":
            root / "score_gated_film_row_prune_keep75/attempt_0000/work/inflated/0.raw",
    }


def _even_frames_identical(base: Path, other: Path) -> dict[str, Any]:
    same = 0
    max_abs = 0
    for pair in range(FRAMES):
        a = read_frame(base, 2 * pair)
        b = read_frame(other, 2 * pair)
        if np.array_equal(a, b):
            same += 1
        else:
            max_abs = max(max_abs,
                          int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()))
    return {"frame0_identical_pairs": same, "frame0_max_abs_diff": max_abs}


def stage_reprice(args) -> dict[str, Any]:
    """Re-price mp2's three measured candidates on the authority GT.  No render: $0 of decode."""
    controls = json.loads((OUT_ROOT / "SF1_CONTROLS.json").read_text())
    gt_dali = np.load(controls["retained_payload"]["authority_gt_pose6"]["path"])
    base_pose6 = np.load(controls["retained_payload"]["base_pose6_authority"]["path"])
    base_authority = d_pose(base_pose6, gt_dali)

    pose = PoseInstrument()
    out = OUT_ROOT / "retained"
    rows = []
    for candidate, raw_path in _candidate_raws().items():
        require_file(raw_path, expect_bytes=FRAMES * 2 * FRAME_BYTES)
        generated = np.zeros((FRAMES, POSE_DIMS), dtype=np.float64)
        for pair in range(FRAMES):
            generated[pair] = pose.pose6(read_frame(raw_path, 2 * pair),
                                         read_frame(raw_path, 2 * pair + 1))
        authority = d_pose(generated, gt_dali)
        eval_json = json.loads(
            (raw_path.parents[1] / "contest_auth_eval.json").read_text())
        archive_bytes = int(eval_json["archive_size_bytes"])
        delta_bytes = archive_bytes - ARCHIVE_BYTES
        drift = generated - base_pose6
        rows.append({
            "candidate_id": candidate,
            "archive_bytes": archive_bytes,
            "delta_bytes": delta_bytes,
            "delta_S_rate": delta_bytes * S_PER_BYTE,
            "d_pose_authority": authority,
            "delta_d_pose_authority": authority - base_authority,
            "delta_S_pose_authority": score_pose(authority) - score_pose(base_authority),
            "d_pose_pyav_advisory_mp2": float(eval_json["avg_posenet_dist"]),
            "d_seg_pyav_advisory_mp2": float(eval_json["avg_segnet_dist"]),
            "pose_drift_rms": float(np.sqrt((drift ** 2).mean())),
            "retained": retain_array(out / f"sf1_pose6_{candidate}.float64.npy", generated),
        })
        rows[-1]["net_delta_S_authority_pose_and_rate"] = (
            rows[-1]["delta_S_pose_authority"] + rows[-1]["delta_S_rate"])

    record = {
        "schema": "ddm_sf1_reprice.v1",
        "axis": "[macOS-CPU advisory; authority-tracking DALI GT, 1.00081x]",
        "score_claim": False,
        "promotable": False,
        "base_d_pose_authority": base_authority,
        "base_d_pose_pyav_advisory": 0.0001474678204472775,
        "pyav_over_authority_ratio": 0.0001474678204472775 / base_authority,
        "rows": rows,
    }
    record["receipt"] = atomic_json(OUT_ROOT / "SF1_REPRICE.json", record)
    return record


#: Every banked mp2 generation, with the candidate byte figures the -2,874 B claim was built on.
BANKED_GENERATIONS = (
    "hv1_base_control",
    "score_gated_selected_mixed_q3q4",
    "score_gated_film_row_prune_keep87",
    "score_gated_film_row_prune_keep75",
    "score_gated_film_row_prune_keep62",
    "score_gated_film_row_prune_keep50",
    "score_gated_film_row_prune_keep37",
    "score_gated_film_row_prune_keep25",
    "score_gated_film_row_prune_keep75_minus_keep87",
)


def stage_bytes(args) -> dict[str, Any]:
    """Re-derive every byte figure the -2,874 B claim rests on, from the banked archives.

    Two things get MEASURED here rather than inherited:

    1. ``archive.zip`` stores its single member ``p`` without deflation, so an archive delta
       equals the semantic-section delta exactly.  That is what licenses this tool to price an
       arbitrary row mask by brotli-ing one section instead of rebuilding a whole archive.
    2. The receiver dispatches on ONE magic: ``SD1M`` (mixed per-tensor bit depth) or ``SM3R``
       (row prune, every depth hardcoded to 4).  Neither format can express the other, so the
       mixed-q3/q4 and FiLM-row credits are not merely non-additive -- their sum is not
       decodable by the shipped receiver at all.
    """
    root = MP2_ROOT / "generations"
    census = []
    base_semantic = None
    for name in BANKED_GENERATIONS:
        gen = root / name
        sections = {}
        for label, relative in (("archive", "archive.zip"), ("p", "retained/p"),
                                ("semantic_br", "retained/semantic.br"),
                                ("semantic_raw", "retained/semantic.raw.bin"),
                                ("carrier_br", "retained/carrier.br"),
                                ("hpac_br", "retained/hpac.br"),
                                ("member_tail", "retained/member_tail.bin")):
            path = gen / relative
            sections[label] = path.stat().st_size if path.is_file() else None
        magic = (gen / "retained/semantic.raw.bin").read_bytes()[:4].decode(
            "latin-1") if (gen / "retained/semantic.raw.bin").is_file() else None
        if name == "hv1_base_control":
            base_semantic = sections["semantic_br"]
        census.append({
            "generation": name,
            "sections": sections,
            "semantic_format_magic": magic,
            "zip_overhead_bytes": (sections["archive"] - sections["p"])
            if sections["archive"] and sections["p"] else None,
            "delta_archive_vs_base": (sections["archive"] - ARCHIVE_BYTES)
            if sections["archive"] else None,
            "delta_semantic_br_vs_base": (sections["semantic_br"] - base_semantic)
            if sections["semantic_br"] and base_semantic else None,
        })

    zip_overheads = {row["zip_overhead_bytes"] for row in census}
    archive_equals_semantic = all(
        row["delta_archive_vs_base"] == row["delta_semantic_br_vs_base"] for row in census)

    controls_path = OUT_ROOT / "SF1_CONTROLS.json"
    base_d = None
    if controls_path.is_file():
        base_d = json.loads(controls_path.read_text())["controls"]["base_d_pose_authority_gt"]

    def break_even(delta_bytes: int, base: float) -> dict[str, Any]:
        credit = -delta_bytes * S_PER_BYTE
        allowed_root = score_pose(base) + credit
        allowed = (allowed_root ** 2) / 10.0
        return {"rate_credit_S": credit, "allowed_d_pose": allowed,
                "allowed_delta_d_pose": allowed - base,
                "allowed_ratio_over_base": allowed / base}

    gap_S = FRONTIER_S - TARGET_S
    record = {
        "schema": "ddm_sf1_bytes.v1",
        "axis": "[macOS-CPU advisory] -- byte figures are exact file measurements",
        "score_claim": False,
        "promotable": False,
        "census": census,
        "controls": {
            "zip_overhead_constant_bytes": sorted(zip_overheads),
            "archive_delta_equals_semantic_br_delta": archive_equals_semantic,
        },
        "format_exclusivity": {
            "sd1m_expresses": "per-tensor bit depth 2..8, no row prune",
            "sm3r_expresses": "row prune on 3 FiLM weights, every depth HARDCODED to 4",
            "receiver_dispatch": "ddm_mp2_semantic_receiver.unpack_variant_semantic_or_none "
                                 "branches on ONE magic; there is no combined format",
            "consequence": "the mixed-q3/q4 credit and the FiLM-row credit cannot be summed: "
                           "no shipped receiver decodes a candidate carrying both",
        },
        "claimed_minus_2874_B": {
            "arithmetic": "823 (mixed q3/q4) + 2051 (FiLM keep25)",
            "status": "UNREACHABLE by the shipped receiver, and already forbidden as "
                      "double-counting by RFO2 per mp2 line 115",
            "honest_maximum_from_banked_candidates_bytes": 2051,
            "honest_maximum_credit_S": 2051 * S_PER_BYTE,
            "honest_maximum_share_of_gap": 2051 * S_PER_BYTE / gap_S,
            "claimed_share_of_gap": 2874 * S_PER_BYTE / gap_S,
        },
        "gap": {"S": gap_S, "bytes": gap_S / S_PER_BYTE, "S_per_byte": S_PER_BYTE},
        "base_d_pose_authority": base_d,
        "break_even": {
            str(delta): break_even(delta, base_d)
            for delta in (-130, -471, -823, -2051, -2874)
        } if base_d else None,
    }
    record["receipt"] = atomic_json(OUT_ROOT / "SF1_BYTES.json", record)
    return record


def _group_dir(group_id: str, tag: str = "") -> Path:
    """Group receipts live under their own directory; ``tag`` separates repeat samplings."""
    name = group_id if not tag else f"{group_id}__{tag}"
    return OUT_ROOT / "groups" / name


def stage_map(args) -> dict[str, Any]:
    import torch

    controls = json.loads((OUT_ROOT / "SF1_CONTROLS.json").read_text())
    gt_dali = np.load(controls["retained_payload"]["authority_gt_pose6"]["path"])
    base_pose6_full = np.load(controls["retained_payload"]["base_pose6_authority"]["path"])

    renderer = load_renderer()
    state = semantic_base_state(renderer)
    norms = film_row_norms(state)
    tokens = np.fromfile(
        HV1_ADVISORY / "inflated" / ".f26_decode_checkpoints"
        / "tokens_cpu_stage_complete.u8",
        dtype=np.uint8).reshape(FRAMES, EVAL_H, EVAL_W)
    render = Renderer(renderer, tokens)
    pose = PoseInstrument()
    base_raw = HV1_ADVISORY / "inflated" / "0.raw"

    groups = [null_group()]
    groups.extend(mechanism_groups(norms))
    groups.extend(selector_groups(norms, tuple(args.selector_seeds)))
    groups.extend(attribution_groups(norms))
    groups.extend(global_selection_groups(norms))
    if args.only_family:
        groups = [g for g in groups if g["family"] == args.only_family]
    if args.only_groups:
        wanted = set(args.only_groups)
        groups = [g for g in groups if g["group_id"] in wanted]
        missing = wanted - {g["group_id"] for g in groups}
        if missing:
            raise SF1Refusal(f"unknown group ids: {sorted(missing)}")

    pairs = _pair_subset(args.pair_seed, args.pairs)
    base_subset = base_pose6_full[pairs]
    gt_subset = gt_dali[pairs]
    base_d_subset = d_pose(base_subset, gt_subset)

    # frame_0 and the base frame_1 are fixed for the whole map: read them once.
    frame0 = {int(p): read_frame(base_raw, 2 * int(p)) for p in pairs}
    frame1_base = {int(p): read_frame(base_raw, 2 * int(p) + 1) for p in pairs}

    results = []
    for group in groups:
        gid = group["group_id"]
        gdir = _group_dir(gid, args.tag)
        receipt_path = gdir / "GROUP.json"
        if receipt_path.is_file() and not args.force:
            results.append(json.loads(receipt_path.read_text()))
            print(f"[resume] {gid}", flush=True)
            continue
        gdir.mkdir(parents=True, exist_ok=True)
        started = time.time()

        if group.get("builder") == "sm3r_roundtrip":
            perturbed = sm3r_roundtrip_state(state, group["rows"])
            weight_energy = sum(
                float(((torch.as_tensor(np.array(state[n], copy=True)).double()
                        - perturbed[n].double()) ** 2).sum())
                for n in state)
        else:
            perturbed = {k: torch.as_tensor(np.array(v, copy=True)) for k, v in state.items()}
            weight_energy = 0.0
            for name, rows in group["rows"].items():
                tensor = perturbed[name]
                flat = tensor.reshape(FILM_ROWS, -1)
                for row in rows:
                    weight_energy += float((flat[row].double() ** 2).sum())
                    flat[row] = 0.0
        semantic = build_semantic(renderer, perturbed)

        generated = np.zeros((len(pairs), POSE_DIMS), dtype=np.float64)
        output_energy = np.zeros(len(pairs), dtype=np.float64)
        stack = np.zeros((len(pairs), CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
        for slot, pair in enumerate(pairs):
            pair = int(pair)
            new_frame1 = render.frame1(semantic, pair)
            stack[slot] = new_frame1
            delta = new_frame1.astype(np.float64) - frame1_base[pair].astype(np.float64)
            output_energy[slot] = float((delta ** 2).sum())
            generated[slot] = pose.pose6(frame0[pair], new_frame1)

        value = d_pose(generated, gt_subset)
        drift = generated - base_subset
        raw_bytes, brotli_bytes = sm3r_packet_bytes(state, group["rows"])
        base_raw_bytes, base_brotli = sm3r_packet_bytes(state, {})

        record = {
            "schema": "ddm_sf1_group.v1",
            "group_id": gid,
            "family": group["family"],
            "builder": group.get("builder", "zero_rows"),
            "row_count": group["row_count"],
            "rows": {k: list(v) for k, v in group["rows"].items()},
            "pairs": {"seed": args.pair_seed, "count": len(pairs),
                      "sampling": "seeded random, never a prefix"},
            "d_pose_authority": value,
            "base_d_pose_authority_same_subset": base_d_subset,
            "delta_d_pose": value - base_d_subset,
            "delta_S_pose": score_pose(value) - score_pose(base_d_subset),
            "pose_drift_rms": float(np.sqrt((drift ** 2).mean())),
            "weight_perturbation_energy": weight_energy,
            "output_perturbation_energy_total": float(output_energy.sum()),
            "output_perturbation_energy_mean_per_pair": float(output_energy.mean()),
            "bytes": {
                "semantic_packet_raw": raw_bytes,
                "semantic_packet_brotli_q11": brotli_bytes,
                "base_semantic_packet_brotli_q11": base_brotli,
                "delta_bytes_vs_base_packet": brotli_bytes - base_brotli,
            },
            "wall_seconds": time.time() - started,
            "retained": {
                "pose6": retain_array(gdir / "pose6.float64.npy", generated),
                "output_energy": retain_array(gdir / "output_energy.float64.npy",
                                              output_energy),
                "frame1_stack": retain_raw(gdir / "frame1_stack.u8", stack),
            },
        }
        if group["rows"]:
            record["retained"]["perturbed_film"] = retain_array(
                gdir / "perturbed_film.float32.npy",
                np.stack([np.asarray(perturbed[n], dtype=np.float32) for n in PRUNE_NAMES]))
        atomic_json(receipt_path, record)
        results.append(record)
        print(f"[{gid}] d_pose={value:.6e} dd={value - base_d_subset:+.3e} "
              f"E_w={weight_energy:.4e} E_out={output_energy.sum():.4e} "
              f"{record['wall_seconds']:.0f}s", flush=True)

    summary = {
        "schema": "ddm_sf1_map.v1",
        "axis": "[macOS-CPU advisory; authority-tracking DALI GT, 1.00081x]",
        "score_claim": False,
        "promotable": False,
        "pair_subset": {"seed": args.pair_seed, "count": len(pairs),
                        "indices": [int(p) for p in pairs]},
        "base_d_pose_authority_same_subset": base_d_subset,
        "groups": results,
    }
    map_name = "SF1_MAP.json" if not args.tag else f"SF1_MAP__{args.tag}.json"
    summary["receipt"] = atomic_json(OUT_ROOT / map_name, summary)
    return summary


def stage_verdict(args) -> dict[str, Any]:
    map_name = "SF1_MAP.json" if not args.tag else f"SF1_MAP__{args.tag}.json"
    summary = json.loads((OUT_ROOT / map_name).read_text())
    groups = [g for g in summary["groups"] if g["family"] == "mechanism_partition"]
    if not groups:
        raise SF1Refusal("no mechanism-partition groups measured")

    per_weight = {g["group_id"]: g["delta_d_pose"] / g["weight_perturbation_energy"]
                  for g in groups}
    per_output = {g["group_id"]: g["delta_d_pose"] / g["output_perturbation_energy_total"]
                  for g in groups}
    gain = {g["group_id"]: g["output_perturbation_energy_total"] / g["weight_perturbation_energy"]
            for g in groups}

    def spread(values: dict[str, float]) -> dict[str, Any]:
        positive = {k: v for k, v in values.items() if v > 0}
        if not positive:
            return {"ratio": None, "note": "no positive entries"}
        lo_id = min(positive, key=positive.get)
        hi_id = max(positive, key=positive.get)
        return {"ratio": positive[hi_id] / positive[lo_id],
                "quietest": lo_id, "quietest_value": positive[lo_id],
                "loudest": hi_id, "loudest_value": positive[hi_id],
                "negative_entries": sorted(set(values) - set(positive))}

    weight_spread = spread(per_weight)
    output_spread = spread(per_output)

    # Control (a): the zero perturbation re-renders bit-identically, so its pose must not move
    # by one ulp.  A non-zero here means the instrument drifted between the base pass and the map.
    null = next((g for g in summary["groups"]
                 if g["group_id"] == "null_zero_perturbation_control"), None)
    if null is None:
        raise SF1Refusal("the null zero-perturbation control was not measured")
    if null["delta_d_pose"] != 0.0:
        raise SF1Refusal(
            f"null control moved d_pose by {null['delta_d_pose']!r}; instrument is not stable")

    # Control (b): the instrument's own repeat floor.  Deterministic forward + deterministic
    # render means the floor is exactly zero, so the honest variability to report is the
    # SUBSET-SAMPLING spread, measured by a second seed in stage_map_repeat.
    base_subset = summary["base_d_pose_authority_same_subset"]
    selectors = [g for g in summary["groups"] if g["family"] == "selector_alternative"]
    selector_rows = []
    for group in sorted(selectors, key=lambda g: g["delta_d_pose"]):
        delta_bytes = group["bytes"]["delta_bytes_vs_base_packet"]
        credit = -delta_bytes * S_PER_BYTE
        allowed = ((score_pose(base_subset) + credit) ** 2) / 10.0 - base_subset
        selector_rows.append({
            "group_id": group["group_id"],
            "row_count": group["row_count"],
            "delta_bytes_vs_base_packet": delta_bytes,
            "delta_d_pose": group["delta_d_pose"],
            "delta_S_pose": group["delta_S_pose"],
            "allowed_delta_d_pose_at_this_rate": allowed,
            "miss_factor_over_break_even": (group["delta_d_pose"] / allowed)
            if allowed > 0 else None,
        })
    for group in sorted((g for g in summary["groups"]
                         if g["family"] in ("global_reselection", "format_attribution")),
                        key=lambda g: g["delta_d_pose"]):
        delta_bytes = group["bytes"]["delta_bytes_vs_base_packet"]
        credit = -delta_bytes * S_PER_BYTE
        allowed = ((score_pose(base_subset) + credit) ** 2) / 10.0 - base_subset
        selector_rows.append({
            "group_id": group["group_id"],
            "family": group["family"],
            "row_count": group["row_count"],
            "delta_bytes_vs_base_packet": delta_bytes,
            "delta_d_pose": group["delta_d_pose"],
            "delta_S_pose": group["delta_S_pose"],
            "allowed_delta_d_pose_at_this_rate": allowed,
            "miss_factor_over_break_even": (group["delta_d_pose"] / allowed)
            if allowed > 0 and group["delta_d_pose"] > 0 else None,
        })

    # Additivity: is the damage of the whole equal to the sum of the 18 disjoint parts?
    # This is the one linearization-shaped claim in the unit, so it gets realised, not assumed.
    whole = next((g for g in summary["groups"]
                  if g["group_id"] == "attr_zero_all_film_rows"), None)
    additivity = None
    if whole is not None:
        predicted = sum(g["delta_d_pose"] for g in groups)
        additivity = {
            "predicted_sum_of_18_disjoint_groups": predicted,
            "realised_all_576_rows_zeroed": whole["delta_d_pose"],
            "realised_over_predicted": whole["delta_d_pose"] / predicted
            if predicted else None,
        }

    incumbent = next((r for r in selector_rows
                      if r["group_id"] == "sel_mp2_keep87_lowest_norm"), None)
    best = selector_rows[0] if selector_rows else None
    selector_verdict = None
    if incumbent and best:
        selector_verdict = {
            "incumbent_is_best": incumbent["group_id"] == best["group_id"],
            "best_group_id": best["group_id"],
            "incumbent_delta_d_pose": incumbent["delta_d_pose"],
            "best_delta_d_pose": best["delta_d_pose"],
            "reselection_gain_factor": (incumbent["delta_d_pose"] / best["delta_d_pose"])
            if best["delta_d_pose"] > 0 else None,
        }

    def bucket(ratio: float | None) -> str:
        if ratio is None:
            return "INDETERMINATE"
        if ratio >= 10.0:
            return "STRUCTURED"
        if ratio <= 2.0:
            return "ENERGY_LIKE"
        return "INDETERMINATE"

    gap_S = FRONTIER_S - TARGET_S
    record = {
        "schema": "ddm_sf1_verdict.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "gap_to_target_S": gap_S,
        "gap_to_target_bytes": gap_S / S_PER_BYTE,
        "per_unit_weight_energy": per_weight,
        "per_unit_output_energy": per_output,
        "output_energy_per_unit_weight_energy_gain": gain,
        "weight_energy_spread": weight_spread,
        "output_energy_spread": output_spread,
        "branch_on_weight_energy": bucket(weight_spread.get("ratio")),
        "branch_on_output_energy": bucket(output_spread.get("ratio")),
        "pre_registered_bars": {"STRUCTURED": ">= 10.0", "ENERGY_LIKE": "<= 2.0"},
        "null_control_delta_d_pose": null["delta_d_pose"],
        "additivity": additivity,
        "selector_alternatives": selector_rows,
        "selector_verdict": selector_verdict,
    }
    record["receipt"] = atomic_json(OUT_ROOT / "SF1_VERDICT.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("controls", "bytes", "reprice", "map", "verdict"))
    parser.add_argument("--pairs", type=int, default=120,
                        help="seeded RANDOM screening pair count (never a prefix)")
    parser.add_argument("--pair-seed", type=int, default=20260817)
    parser.add_argument("--control-pairs", type=int, default=12)
    parser.add_argument("--control-seed", type=int, default=8170001)
    parser.add_argument("--selector-seeds", type=int, nargs="*",
                        default=[101, 202, 303, 404, 505])
    parser.add_argument("--only-family", default="",
                        help="restrict the map to one group family")
    parser.add_argument("--only-groups", nargs="*", default=[],
                        help="restrict the map to named group ids (for the n600 promotion)")
    parser.add_argument("--tag", default="",
                        help="separate a repeat sampling from the primary map")
    parser.add_argument("--force", action="store_true",
                        help="re-measure groups that already have a receipt")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.UTC).isoformat()
    stage = {"controls": stage_controls, "bytes": stage_bytes,
             "reprice": stage_reprice, "map": stage_map,
             "verdict": stage_verdict}[args.stage]
    result = stage(args)
    print(json.dumps({"stage": args.stage, "started_utc": started,
                      "receipt": result.get("receipt")}, indent=2))


if __name__ == "__main__":
    main()
