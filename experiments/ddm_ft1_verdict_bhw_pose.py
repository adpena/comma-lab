#!/usr/bin/env python3
"""DDM-FT1 candidate verdict: realized d_seg, d_pose, and the B/H/W split.

A headline ``d_seg`` cannot tell anyone whether a renderer fine-tune beat the
measured collateral bound or merely re-derived it.  ``d_seg`` sees only ``B-H``;
two epochs with the same ``d_seg`` can have wildly different ``B`` and ``H``
(``ddm_qs3_saturation_compose_20260813.md``: ``157 = 2H + R``, not ``H + R``).
This module therefore reports the decomposition every prior arm on this object
reported, using the same definitions:

* ``B`` benefit  -- incumbent wrong, candidate right   ("sites fixed")
* ``H`` harm     -- incumbent right, candidate wrong   ("newly broken")
* ``W`` wash     -- both wrong, wrong to different wrong ("unchanged")

taken verbatim from ``experiments/ddm_fcd1_field_for_coder_diagonal.py``.  Note
``W`` is *correctness*-unchanged, not *untouched*: the population is
``incumbent != candidate`` only, so the untouched count is reported separately
against the full 117,964,800-position denominator.

Both scorers run through the evaluator's own ``upstream/modules.py`` classes on
the composed camera-resolution pair, so the numbers are the evaluator's
arithmetic rather than a re-implementation:

* frame ``2p``   -- the shipped pose carrier, ``127.5 + 64*basis@coeff``, bicubic
  to camera.  It is FIXED: a renderer change cannot move it.
* frame ``2p+1`` -- the semantic render, bilinear to camera.  This is the only
  frame the renderer writes, and it is the frame SegNet reads
  (``upstream/modules.py`` ``x[:, -1, ...]``) *and* half of what PoseNet reads.

That asymmetry is the whole reason a seg-only renderer move costs pose: the
shipped carrier coefficients were solved to convergence against the ORIGINAL
frame ``2p+1``, so moving that frame walks out from under a converged solve.

Axis: ``[macOS-CPU advisory]``.  torch-CPU is the pose authority (MLX drifts
0.55% relative, MPS 23x), but a CPU advisory row is still not a contest score.
No output of this module is a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ddm_ft1_identity_gate_and_caches import (
    EVAL_H,
    EVAL_W,
    FRONTIER_ARCHIVE,
    GT_CACHE_DALI,
    N_PAIRS,
    SEMANTIC_WIDTH,
    SHIPPED_TOKENS,
    SUBMISSION_ROOT,
    load_shipped_renderer_module,
    read_semantic_section,
    recover_sm3r_allocation,
    sha256_bytes,
)

CAMERA_H, CAMERA_W = 874, 1164
NUM_CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
FIELD_POSITIONS = N_PAIRS * EVAL_H * EVAL_W

DEFAULT_OUT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/retained"
)


def load_gt_tables(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the GT argmax + pose tables from either supported container.

    The DALI container (``{"seg", "pose"}``) is the lineage the contest T4
    axis scores against; the npz container (``{"lstars", "gt_poses"}``) is
    the PyAV control.  They are NOT interchangeable: they disagree on 18,954
    argmax positions and by 1.4061e-04 in pose MSE.
    """

    if path.suffix == ".npz":
        cache = np.load(path)
        return (
            cache["lstars"].astype(np.uint8),
            cache["gt_poses"].astype(np.float64),
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        payload["seg"].numpy().astype(np.uint8),
        payload["pose"].numpy().astype(np.float64),
    )


def build_semantic_model(state: Mapping[str, torch.Tensor], device: torch.device):
    shipped = load_shipped_renderer_module()
    model = shipped.SemanticTokenRenderer(SEMANTIC_WIDTH)
    model.load_state_dict(dict(state), strict=True)
    return model.eval().to(device)


def load_carrier(archive_path: Path, device: torch.device):
    """Return the shipped pose carrier exactly as ``f26_inflate`` materializes it."""

    if str(SUBMISSION_ROOT) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_ROOT))
    import struct

    residual_archive = importlib.import_module("runtime.residual_archive")
    carrier_repack = importlib.import_module("runtime.carrier_repack")

    parts = residual_archive.read_residual_archive(archive_path)
    carrier_blob, _selector = carrier_repack.split_frame0_selector_carrier(
        parts.carrier_blob
    )
    renderer = load_shipped_renderer_module()
    canonical = carrier_repack.materialize_cpr1(carrier_blob, renderer)
    marker = bytes(40_252)
    payload = struct.pack("<II", len(marker), len(canonical)) + marker + canonical
    _semantic, basis, coefficients = renderer.unpack_semantic_pose(payload)
    basis = renderer.normalized_basis(basis.to(device))
    return basis, coefficients.to(device), renderer


def carrier_frames(
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    renderer,
    pair_ids: torch.Tensor,
) -> torch.Tensor:
    """Frame ``2p`` at camera resolution, uint8, per the shipped renderer."""

    carrier = torch.einsum("bk,kchw->bchw", coefficients[pair_ids], basis)
    carrier = carrier / math.sqrt(renderer.CARRIER_DIM)
    frame = (127.5 + renderer.CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round()
    frame = torch.nn.functional.interpolate(
        frame, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
    )
    return frame.clamp(0.0, 255.0).round()


def master_frames(
    model, tokens: torch.Tensor, pair_ids: torch.Tensor
) -> torch.Tensor:
    """Frame ``2p+1`` at camera resolution, uint8, per the shipped renderer."""

    frame = model(tokens, pair_ids)
    frame = torch.nn.functional.interpolate(
        frame, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
    )
    return frame.clamp(0.0, 255.0).round()


def classify_pool(
    incumbent: np.ndarray, candidate: np.ndarray, gt: np.ndarray
) -> dict[str, int]:
    """The fcd1 B/H/W law, verbatim in definition.

    ``W`` counts wrong-to-different-wrong, so it is a subset of the CHANGED
    population, never the untouched one.
    """

    changed = incumbent != candidate
    benefit = changed & (incumbent != gt) & (candidate == gt)
    harm = changed & (incumbent == gt) & (candidate != gt)
    wash = changed & (incumbent != gt) & (candidate != gt)
    return {
        "B_benefit": int(benefit.sum()),
        "H_harm": int(harm.sum()),
        "W_wash": int(wash.sum()),
        "changed": int(changed.sum()),
    }


def per_class_pool(
    incumbent: np.ndarray, candidate: np.ndarray, gt: np.ndarray
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for index, name in enumerate(CLASS_NAMES):
        selected = gt == index
        out[name] = classify_pool(
            incumbent[selected], candidate[selected], gt[selected]
        )
        out[name]["gt_pixels"] = int(selected.sum())
    return out


def score_components(d_seg: float, d_pose: float, archive_bytes: int) -> dict[str, float]:
    rate = 25.0 * archive_bytes / 37_545_489
    seg_term = 100.0 * d_seg
    pose_term = math.sqrt(10.0 * d_pose)
    return {
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate,
        "S": seg_term + pose_term + rate,
    }


def evaluate_candidate(
    *,
    state: Mapping[str, torch.Tensor],
    tokens: np.ndarray,
    labels: np.ndarray,
    gt_poses: np.ndarray,
    archive_path: Path,
    device: torch.device,
    batch_size: int,
    pair_ids: list[int],
    modules,
) -> dict[str, Any]:
    """One full pass: argmax field, d_seg, and d_pose on the composed pair."""

    model = build_semantic_model(state, device)
    basis, coefficients, renderer = load_carrier(archive_path, device)

    segnet = modules.SegNet().eval().to(device)
    from safetensors.torch import load_file

    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    posenet = modules.PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(modules.posenet_sd_path, device=str(device)))
    for parameter in list(segnet.parameters()) + list(posenet.parameters()):
        parameter.requires_grad_(False)

    # The evaluator scores only the first half of the pose head's outputs
    # (``upstream/modules.py`` PoseNet.compute_distortion: ``[..., : h.out // 2]``).
    # Derive that width instead of assuming 6, and refuse if the GT table
    # disagrees -- a silent width mismatch would compare different quantities.
    pose_heads = [h for h in posenet.hydra.heads if h.name == "pose"]
    if len(pose_heads) != 1:
        raise ValueError(f"expected exactly one pose head, found {len(pose_heads)}")
    pose_width = pose_heads[0].out // 2
    if pose_width != gt_poses.shape[1]:
        raise ValueError(
            f"pose head scores {pose_width} dims but the GT table has "
            f"{gt_poses.shape[1]}"
        )

    argmax = np.empty((len(pair_ids), EVAL_H, EVAL_W), dtype=np.uint8)
    pose_out = np.empty((len(pair_ids), pose_width), dtype=np.float64)
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(pair_ids), batch_size):
            selected = pair_ids[start : start + batch_size]
            idx = torch.tensor(selected, dtype=torch.long, device=device)
            token_batch = torch.from_numpy(tokens[selected].astype(np.int64)).to(device)
            frame1 = master_frames(model, token_batch, idx)
            frame0 = carrier_frames(basis, coefficients, renderer, idx)
            pair = torch.stack([frame0, frame1], dim=1)
            seg_logits = segnet(segnet.preprocess_input(pair))
            argmax[start : start + len(selected)] = (
                seg_logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
            )
            pose = posenet(posenet.preprocess_input(pair))["pose"][..., :pose_width]
            pose_out[start : start + len(selected)] = pose.double().cpu().numpy()

    gt_labels = labels[pair_ids]
    d_seg = float((argmax != gt_labels).mean())
    d_pose = float(((pose_out - gt_poses[pair_ids]) ** 2).mean(axis=1).mean())
    return {
        "argmax": argmax,
        "pose": pose_out,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "pairs": len(pair_ids),
        "elapsed_seconds": time.time() - started,
    }


def export_section(
    state: Mapping[str, torch.Tensor], archive_path: Path
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Re-encode the candidate through the deployed SM3R encoder.

    Returns the export record AND the state the shipped receiver would actually
    load back.  The two are NOT the same object: the deployed encoder quantizes
    to the per-tensor depth table and keeps only ``keep_percent`` of each pruned
    tensor's rows, while the trainer's QAT models a uniform int4 grid and no
    pruning at all.  Scoring the trained weights would therefore report a d_seg
    for a model that never ships -- the realization gap this arm has to measure,
    not hide.
    """

    sm3 = importlib.import_module("experiments.ddm_sm3_semantic_representation")
    shipped = load_shipped_renderer_module()
    template = shipped.SemanticTokenRenderer(SEMANTIC_WIDTH).state_dict()
    blob = read_semantic_section(archive_path)
    keep_percent, allocation = recover_sm3r_allocation(blob, template)
    encoded, expected, meta = sm3.pack_prune_mixed_candidate(
        dict(state), keep_percent=keep_percent, depths=allocation
    )
    # Decode through the SHIPPED receiver, not through ``expected``: agreement
    # between the two is itself the parse-back proof.
    parsed = shipped.unpack_variant_semantic_or_none(encoded, template)
    if parsed is None:
        raise ValueError("shipped receiver rejected the exported semantic section")
    parse_back_max_delta = max(
        float((parsed[name] - expected[name]).abs().max()) for name in expected
    )
    if parse_back_max_delta != 0.0:
        raise ValueError(
            f"encoder/receiver disagree by {parse_back_max_delta} on the exported section"
        )
    quantization_max_delta = max(
        float((parsed[name] - state[name]).abs().max()) for name in state
    )
    record = {
        "bytes": len(encoded),
        "sha256": sha256_bytes(encoded),
        "shipped_bytes": len(blob),
        "shipped_sha256": sha256_bytes(bytes(blob)),
        "size_preserved": len(encoded) == len(blob),
        "kept_rows": meta["kept_rows"],
        "keep_percent": keep_percent,
        "bit_allocation": dict(allocation),
        "parse_back_max_abs_delta": parse_back_max_delta,
        "trained_vs_realized_max_abs_delta": quantization_max_delta,
        "payload": encoded,
    }
    return record, {k: v.clone() for k, v in parsed.items()}


def load_state_from_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], str]:
    """Load the weights that DEPLOY -- the EMA shadow, never the live weights.

    The PR130 resumable trainer declares which tensors deploy in
    ``deployment_weights`` and puts them at the top-level ``state_dict``; the
    live weights sit at ``training_state.model_state_dict``.  Honour that
    declaration explicitly rather than falling through a key-preference list:
    a silent pick of the live weights would violate the EMA non-negotiable and
    would be invisible in the receipt.
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    declared = payload.get("deployment_weights")
    if declared is not None:
        if declared != "ema_shadow":
            raise ValueError(
                f"checkpoint declares deployment_weights={declared!r}; this arm "
                "only scores an EMA-shadow deployment"
            )
        block = payload.get("state_dict")
        if not isinstance(block, Mapping):
            raise ValueError(
                "checkpoint declares an ema_shadow deployment but carries no state_dict"
            )
        return (
            {k: v.detach().cpu().clone() for k, v in block.items()},
            "state_dict (declared deployment_weights=ema_shadow)",
        )
    for key in ("ema", "ema_state"):
        block = payload.get(key)
        if isinstance(block, Mapping) and isinstance(block.get("shadow"), Mapping):
            return {k: v.detach().cpu().clone() for k, v in block["shadow"].items()}, (
                f"{key}.shadow"
            )
    for key in ("ema_shadow", "state_dict"):
        block = payload.get(key)
        if isinstance(block, Mapping):
            return {k: v.detach().cpu().clone() for k, v in block.items()}, key
    raise ValueError(f"no recognizable weights in {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--base", type=Path, help="incumbent checkpoint (default: shipped)")
    parser.add_argument("--archive", type=Path, default=FRONTIER_ARCHIVE)
    parser.add_argument("--tokens", type=Path, default=SHIPPED_TOKENS)
    parser.add_argument("--gt-cache", type=Path, default=GT_CACHE_DALI)
    parser.add_argument("--challenge-root", type=Path, default=Path("upstream"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    parser.add_argument("--archive-bytes", type=int, default=180_002)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "verdict.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")

    root = args.challenge_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    modules = importlib.import_module("modules")

    tokens = np.fromfile(args.tokens, dtype=np.uint8).reshape(N_PAIRS, EVAL_H, EVAL_W)
    labels, gt_poses = load_gt_tables(args.gt_cache)
    pair_ids = list(range(min(args.pairs, N_PAIRS)))

    shipped_blob = read_semantic_section(args.archive)
    shipped = load_shipped_renderer_module()
    template = shipped.SemanticTokenRenderer(SEMANTIC_WIDTH).state_dict()
    if args.base is None:
        base_state = shipped.unpack_variant_semantic_or_none(shipped_blob, template)
        base_source = "frontier archive semantic section"
    else:
        base_state, base_source = load_state_from_checkpoint(args.base)
    candidate_state, candidate_source = load_state_from_checkpoint(args.candidate)

    common = {
        "tokens": tokens,
        "labels": labels,
        "gt_poses": gt_poses,
        "archive_path": args.archive,
        "device": device,
        "batch_size": args.batch_size,
        "pair_ids": pair_ids,
        "modules": modules,
    }
    # The bytes decide the score, so the CANDIDATE of record is the state the
    # shipped receiver loads back out of the exported section -- never the
    # trained weights.  The trained state is evaluated too, and the difference
    # between them is the realization gap.
    export, realized_state = export_section(candidate_state, args.archive)
    payload = export.pop("payload")

    base = evaluate_candidate(state=base_state, **common)
    trained = evaluate_candidate(state=candidate_state, **common)
    candidate = evaluate_candidate(state=realized_state, **common)

    gt_labels = labels[pair_ids]
    pool = classify_pool(base["argmax"], candidate["argmax"], gt_labels)
    per_class = per_class_pool(base["argmax"], candidate["argmax"], gt_labels)

    positions = len(pair_ids) * EVAL_H * EVAL_W
    delta_seg = candidate["d_seg"] - base["d_seg"]
    identity_residual = pool["B_benefit"] - pool["H_harm"] + delta_seg * positions
    base_scores = score_components(base["d_seg"], base["d_pose"], args.archive_bytes)
    candidate_scores = score_components(
        candidate["d_seg"], candidate["d_pose"], args.archive_bytes
    )
    trained_scores = score_components(
        trained["d_seg"], trained["d_pose"], args.archive_bytes
    )

    receipt = {
        "schema": "tac.ddm_ft1.candidate_verdict.v1",
        "label": args.label,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "pairs": len(pair_ids),
        "positions": positions,
        "gt_lineage": str(args.gt_cache),
        "base": {
            "source": base_source,
            "d_seg": base["d_seg"],
            "d_pose": base["d_pose"],
            "elapsed_seconds": base["elapsed_seconds"],
            **base_scores,
        },
        "candidate": {
            "source": str(args.candidate),
            "weights_key": candidate_source,
            "weights_scored": "export -> shipped-receiver parse-back (the bytes that ship)",
            "d_seg": candidate["d_seg"],
            "d_pose": candidate["d_pose"],
            "elapsed_seconds": candidate["elapsed_seconds"],
            **candidate_scores,
        },
        "trained_weights_diagnostic": {
            "note": (
                "the trainer's own object, BEFORE the deployed encoder's "
                "per-tensor depths and row prune; never the score"
            ),
            "d_seg": trained["d_seg"],
            "d_pose": trained["d_pose"],
            **trained_scores,
        },
        "realization_gap": {
            "d_seg_trained_minus_realized": trained["d_seg"] - candidate["d_seg"],
            "d_pose_trained_minus_realized": trained["d_pose"] - candidate["d_pose"],
            "S_trained_minus_realized": trained_scores["S"] - candidate_scores["S"],
            "note": (
                "nonzero means the export discarded part of what training bought; "
                "the trainer models a uniform int4 grid with no row prune, the "
                "deployed encoder keeps keep_percent of each pruned tensor's rows"
            ),
        },
        "delta": {
            "d_seg": delta_seg,
            "d_seg_relative": delta_seg / base["d_seg"] if base["d_seg"] else None,
            "d_pose": candidate["d_pose"] - base["d_pose"],
            "d_pose_ratio": (
                candidate["d_pose"] / base["d_pose"] if base["d_pose"] else None
            ),
            "S": candidate_scores["S"] - base_scores["S"],
            "coupling_dpose_over_dseg": (
                abs(candidate["d_pose"] - base["d_pose"]) / abs(delta_seg)
                if delta_seg
                else None
            ),
        },
        "bhw": {
            **pool,
            "untouched": positions - pool["changed"],
            "B_share_of_changed": (
                pool["B_benefit"] / pool["changed"] if pool["changed"] else None
            ),
            "selectivity_B_over_H": (
                pool["B_benefit"] / pool["H_harm"] if pool["H_harm"] else None
            ),
            "identity_residual_B_minus_H_plus_dseg_positions": identity_residual,
            "law": "fcd1: B=inc!=gt & cand==gt; H=inc==gt & cand!=gt; W=both wrong",
            "note": "W is correctness-unchanged within CHANGED, never the untouched count",
        },
        "bhw_per_gt_class": per_class,
        "export": export,
    }
    # ALWAYS KEEP THE PAYLOAD: the exported section is written BEFORE the
    # receipt that describes it, so a crash can never leave a receipt citing
    # bytes that were never persisted.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    section_path = args.out.with_name(f"{args.label}_semantic_section.bin")
    section_path.write_bytes(payload)
    receipt["export"]["retained_path"] = str(section_path)
    receipt["export"]["retained_sha256"] = hashlib.sha256(payload).hexdigest()
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
