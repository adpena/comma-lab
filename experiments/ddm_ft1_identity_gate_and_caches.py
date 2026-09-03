#!/usr/bin/env python3
"""DDM-FT1 identity gate + aligned training caches for the SHIPPED renderer.

The fine-tune this arm charters is only meaningful if the object it starts from
is *the* object that holds the frontier.  This module proves that, byte for
byte, before any training step is taken:

1.  **Weights leg.**  Read the frontier ``archive.zip`` with the shipped
    receiver's own parser, decode the semantic-renderer section into a state
    dict, then re-encode it through the deployed SM3R encoder.  The section must
    come back byte-identical.  Because the SM3R ``MODE_ROW_PRUNE_MIXED`` payload
    is laid out from shapes, ``keep_percent`` and the per-tensor depth table --
    never from the weight *values* -- this also establishes that a fine-tuned
    state dict exports at exactly the same size.
2.  **Render leg.**  Load the same state dict into the shipped receiver's
    renderer and into the lifted trainer's renderer and compare the raw forward
    and the exact ``R`` (bilinear up to camera, uint8, bilinear down to eval)
    output on a fixed pair sample.  Both must be bit-identical.

A failure in either leg STOPS the arm: the fine-tune would be on a different
object.  Nothing here is a score; the caches it writes are training inputs.

The arm's aligned objective needs two caches that differ only in their target:

* ``--input-cache``  : the SHIPPED decoded token field (the renderer's
  conditioning), taken from the retained ``tokens.u8`` payload.
* ``--target-cache`` : the DALI GT argmax table ``gt_cache_dali.pt["seg"]`` --
  the quantity ``d_seg`` is actually measured against.  Training against the
  tokens instead (the PR130 curriculum) leaves the token-vs-GT disagreement
  entirely unaddressed; that control cache is written too.

Axis: this module performs no scorer evaluation and makes no score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import struct
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ROOT = REPO_ROOT / "submissions" / "semantic_joint_ctxmix"
RENDERER_DIR = SUBMISSION_ROOT / "cpr1"

#: The frontier archive (canonical_frontier_pointer.effective_frontier).
FRONTIER_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/store_v2/retained/archive.zip"
)
FRONTIER_ARCHIVE_SHA256 = (
    "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
)
FRONTIER_ARCHIVE_BYTES = 180_002

#: The decoded shipped token field retained beside that archive.
SHIPPED_TOKENS = Path(
    "/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/store_v2/"
    "retained/inputs/tokens.u8"
)
SHIPPED_TOKENS_SHA256 = (
    "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
)

#: The DALI ground-truth tables -- the lineage the contest T4 axis actually
#: scores against.  Built on a Tesla T4 with CUDA (``result_summary.json``:
#: ``device_name: "Tesla T4"``), schema ``{"seg": uint8[600,384,512],
#: "pose": f32[600,6]}``.
GT_CACHE_DALI = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt"
)
GT_CACHE_DALI_SHA256 = (
    "a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994"
)

#: The PyAV-lineage sister of the same tables, kept as the fork control.
GT_CACHE_PYAV = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_av.pt"
)

#: MEASURED, not assumed: ``experiments/results/mlx_fleet_gt_cache/gt_n600.npz``
#: is the PYAV lineage, not DALI.  Its ``lstars`` differ from ``gt_cache_av.pt``
#: at 2 of 117,964,800 positions and its ``gt_poses`` differ by MSE 3.6e-12,
#: while DALI-vs-PyAV differ by 20,671 argmax positions and pose MSE
#: 1.4061e-04 (exactly the additive fork rf1 records).  Training against it
#: would aim the renderer at a table that disagrees with the scored table at
#: 87% of d_seg's whole 23,757-flip budget, so it is the CONTROL here,
#: never the target.
GT_CACHE_MLX_FLEET_PYAV = (
    REPO_ROOT / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
)

DEFAULT_OUT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/retained"
)

N_PAIRS = 600
EVAL_H, EVAL_W = 384, 512
SEMANTIC_WIDTH = 96
SEMANTIC_BLOCKS = 4
SEMANTIC_FRAME_DIM = 8

#: Fixed, spread pair sample for the render-identity leg.  Endpoints plus
#: interior pairs, so a frame-embedding indexing error cannot hide.
IDENTITY_PAIR_IDS = (0, 1, 2, 137, 299, 300, 450, 599)

SEMANTIC_CHECKPOINT_SCHEMA = "tac.pr130.semantic_checkpoint.v2"


class IdentityGateError(RuntimeError):
    """Raised when the shipped object cannot be reproduced exactly."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_shipped_renderer_module():
    """Import the shipped receiver exactly as ``inflate.sh`` does."""

    if str(SUBMISSION_ROOT) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_ROOT))
    f26 = importlib.import_module("runtime.f26_inflate")
    return f26._load_renderer(RENDERER_DIR)


def read_semantic_section(archive_path: Path) -> bytes:
    if str(SUBMISSION_ROOT) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_ROOT))
    residual_archive = importlib.import_module("runtime.residual_archive")
    return residual_archive.read_residual_archive(archive_path).semantic_blob


def recover_sm3r_allocation(
    blob: bytes, template: Mapping[str, torch.Tensor]
) -> tuple[int, dict[str, int]]:
    """Read ``keep_percent`` and the per-tensor depth table out of the payload.

    The deployed encoder needs both to reproduce the section, and neither is
    recorded anywhere outside the bytes themselves.  Reading them back from the
    shipped blob is what makes the re-encode a *reproduction* rather than a
    guess that happens to be the same length.
    """

    sd1 = importlib.import_module("experiments.ddm_sd1_semantic_rd_curve")
    if not blob.startswith(b"SM3R") or len(blob) < 10:
        raise IdentityGateError("semantic section is not an SM3R payload")
    version, mode, keep_percent, reserved = blob[4:8]
    if version != 1 or mode != 6 or reserved != 0:
        raise IdentityGateError(
            f"unsupported SM3R header: version={version} mode={mode} reserved={reserved}"
        )
    remaining = memoryview(blob)[8:]
    mask = struct.unpack_from("<H", remaining)[0]
    remaining = remaining[2:]
    names = sd1.quantized_names(template)
    # The deployed encoder recomputes the row-prune selection mask from its own
    # PRUNE_NAMES set.  If a future section ever prunes a different tensor set,
    # re-encoding would silently produce a DIFFERENT structure at the same size,
    # so refuse here rather than export something the receiver never shipped.
    sm3 = importlib.import_module("experiments.ddm_sm3_semantic_representation")
    try:
        expected_mask = sm3.mask_for_names(names, sm3.PRUNE_NAMES)
    except ValueError as error:
        raise IdentityGateError(
            "template does not carry the deployed encoder's pruned tensors"
        ) from error
    if mask != expected_mask:
        raise IdentityGateError(
            f"SM3R prune mask {mask} differs from the deployed encoder's {expected_mask}"
        )
    depth_bytes = (len(names) + 1) // 2
    depths = sd1._unpack_depth_nibbles(
        bytes(remaining[:depth_bytes]), len(names)
    )
    return int(keep_percent), dict(zip(names, depths, strict=True))


def exact_r(frame: torch.Tensor, qat) -> torch.Tensor:
    """The evaluator's realized path: up to camera, uint8, down to eval."""

    camera = qat.ste_uint8(
        torch.nn.functional.interpolate(
            frame,
            size=(qat.CAMERA_H, qat.CAMERA_W),
            mode="bilinear",
            align_corners=False,
        )
    )
    return torch.nn.functional.interpolate(
        camera, size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False
    )


def run_identity_gate(
    archive_path: Path, tokens: np.ndarray
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Prove the shipped section round-trips and both renderers agree."""

    from tac.pr130_lift.train_semantic_quantized_resumable import _load_lifted_qat

    sm3 = importlib.import_module("experiments.ddm_sm3_semantic_representation")

    blob = read_semantic_section(archive_path)
    shipped = load_shipped_renderer_module()
    shipped_model = shipped.SemanticTokenRenderer(SEMANTIC_WIDTH).eval()
    template = shipped_model.state_dict()

    state = shipped.unpack_variant_semantic_or_none(blob, template)
    if state is None:
        raise IdentityGateError("shipped receiver did not recognize the semantic section")
    shipped_model.load_state_dict(state, strict=True)

    keep_percent, allocation = recover_sm3r_allocation(blob, template)
    reencoded, expected, meta = sm3.pack_prune_mixed_candidate(
        state, keep_percent=keep_percent, depths=allocation
    )
    weights_identical = bytes(blob) == reencoded
    state_max_delta = max(
        float((state[name] - expected[name]).abs().max()) for name in state
    )

    qat = _load_lifted_qat()
    lifted_model = qat.SemanticTokenRenderer(
        width=SEMANTIC_WIDTH,
        blocks=SEMANTIC_BLOCKS,
        frame_dim=SEMANTIC_FRAME_DIM,
        num_pairs=N_PAIRS,
        num_tokens=5,
        phase_y=1,
        phase_x=1,
        temporal_radius=0,
    ).eval()
    if set(lifted_model.state_dict()) != set(state):
        raise IdentityGateError("lifted and shipped renderer key sets differ")
    lifted_model.load_state_dict(state, strict=True)

    pair_ids = list(IDENTITY_PAIR_IDS)
    idx = torch.tensor(pair_ids, dtype=torch.long)
    token_field = torch.from_numpy(tokens.astype(np.int64))
    sample = token_field[idx]
    conditioning = qat.gather_conditioning(token_field, idx, 0)
    with torch.no_grad():
        shipped_raw = shipped_model(sample, idx)
        lifted_raw = lifted_model(conditioning, idx)
        shipped_r = exact_r(shipped_raw, qat)
        lifted_r = qat.render_float(lifted_model, conditioning, idx, exact_path=True)

    raw_equal = bool(torch.equal(shipped_raw, lifted_raw))
    r_equal = bool(torch.equal(shipped_r, lifted_r))
    receipt = {
        "schema": "tac.ddm_ft1.identity_gate.v1",
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
        "semantic_section": {
            "bytes": len(blob),
            "sha256": sha256_bytes(bytes(blob)),
            "format": "SM3R.v1.MODE_ROW_PRUNE_MIXED",
            "keep_percent": keep_percent,
            "bit_allocation": allocation,
            "kept_rows": meta["kept_rows"],
            "width": SEMANTIC_WIDTH,
            "parameters": int(sum(v.numel() for v in state.values())),
            "zero_parameters": int(sum(int((v == 0).sum()) for v in state.values())),
        },
        "weights_leg": {
            "reencoded_bytes": len(reencoded),
            "reencoded_sha256": sha256_bytes(reencoded),
            "byte_identical": weights_identical,
            "state_roundtrip_max_abs_delta": state_max_delta,
            "size_is_value_independent": True,
        },
        "render_leg": {
            "pair_ids": pair_ids,
            "raw_forward_bit_identical": raw_equal,
            "raw_forward_max_abs_delta": float((shipped_raw - lifted_raw).abs().max()),
            "exact_r_bit_identical": r_equal,
            "exact_r_max_abs_delta": float((shipped_r - lifted_r).abs().max()),
        },
        "passed": bool(
            weights_identical and state_max_delta == 0.0 and raw_equal and r_equal
        ),
        "score_claim": False,
    }
    if not receipt["passed"]:
        raise IdentityGateError(json.dumps(receipt, indent=2, sort_keys=True))
    return state, receipt


def write_token_cache(field: np.ndarray, path: Path) -> dict[str, Any]:
    """Persist a ``{"seg": uint8[600,384,512]}`` cache the trainer can load.

    uint8 is deliberate: the trainer promotes with ``.long()`` on load, so the
    on-disk form is 8x smaller with no change in the values consumed.
    """

    if field.shape != (N_PAIRS, EVAL_H, EVAL_W):
        raise ValueError(f"token field must be {(N_PAIRS, EVAL_H, EVAL_W)}, got {field.shape}")
    if field.min() < 0 or field.max() > 4:
        raise ValueError("token field must hold class indices in [0, 4]")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seg": torch.from_numpy(np.ascontiguousarray(field, dtype=np.uint8))}, path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_init_checkpoint(state: Mapping[str, torch.Tensor], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SEMANTIC_CHECKPOINT_SCHEMA,
        "state_dict": {k: v.detach().cpu().clone() for k, v in state.items()},
        "architecture_config": {
            "width": SEMANTIC_WIDTH,
            "blocks": SEMANTIC_BLOCKS,
            "frame_dim": SEMANTIC_FRAME_DIM,
            "num_pairs": N_PAIRS,
            "num_tokens": 5,
            "phase_y": 1,
            "phase_x": 1,
            "temporal_radius": 0,
        },
        "quant_bits": 4,
        "provenance": {
            "source": "frontier archive semantic_renderer section",
            "archive_sha256": FRONTIER_ARCHIVE_SHA256,
            "decoded_by": "submissions/semantic_joint_ctxmix/cpr1 (shipped receiver)",
        },
    }
    torch.save(payload, path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_gt_seg(path: Path) -> np.ndarray:
    """Read a GT argmax table from either supported container."""

    if path.suffix == ".npz":
        return np.load(path)["lstars"].astype(np.uint8)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["seg"].numpy().astype(np.uint8)


def token_disagreement(
    tokens: np.ndarray, labels: np.ndarray, lineage: str
) -> dict[str, Any]:
    """How far the shipped tokens already sit from the quantity d_seg scores."""

    mismatch = tokens != labels
    total = int(mismatch.size)
    count = int(mismatch.sum())
    per_class = {}
    for klass in range(5):
        selected = mismatch & (labels == klass)
        per_class[str(klass)] = {
            "mismatches": int(selected.sum()),
            "share_of_mismatches": float(selected.sum()) / max(count, 1),
            "class_pixels": int((labels == klass).sum()),
        }
    return {
        "positions": total,
        "mismatches": count,
        "rate": count / total,
        "per_gt_class": per_class,
        "lineage": lineage,
        "score_claim": False,
    }


def gt_lineage_fork(dali: np.ndarray, pyav: np.ndarray) -> dict[str, Any]:
    """Measure the fork between the two GT argmax tables.

    The two lineages are not interchangeable targets: a renderer trained to
    reproduce the PyAV table learns the positions where PyAV and DALI disagree
    as if they were signal, and the contest scores against DALI.
    """

    disagree = dali != pyav
    return {
        "positions": int(disagree.size),
        "argmax_disagreements": int(disagree.sum()),
        "rate": float(disagree.sum()) / int(disagree.size),
        "note": (
            "DALI is the scored lineage; PyAV is the control. Training against "
            "the PyAV table aims the renderer at these positions as if signal."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=FRONTIER_ARCHIVE)
    parser.add_argument("--tokens", type=Path, default=SHIPPED_TOKENS)
    parser.add_argument("--gt-cache", type=Path, default=GT_CACHE_DALI)
    parser.add_argument("--gt-cache-control", type=Path, default=GT_CACHE_PYAV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--skip-caches",
        action="store_true",
        help="run the identity gate only; do not materialize the training caches",
    )
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    started = time.time()

    tokens = np.fromfile(args.tokens, dtype=np.uint8).reshape(N_PAIRS, EVAL_H, EVAL_W)
    state, receipt = run_identity_gate(args.archive, tokens)
    receipt["shipped_tokens"] = {
        "path": str(args.tokens),
        "bytes": args.tokens.stat().st_size,
        "sha256": sha256_file(args.tokens),
        "sha256_expected": SHIPPED_TOKENS_SHA256,
    }

    if not args.skip_caches:
        labels = load_gt_seg(args.gt_cache)
        control = load_gt_seg(args.gt_cache_control)
        receipt["gt_lineage"] = {
            "target": {"path": str(args.gt_cache), "sha256": sha256_file(args.gt_cache)},
            "control": {
                "path": str(args.gt_cache_control),
                "sha256": sha256_file(args.gt_cache_control),
            },
            "fork": gt_lineage_fork(labels, control),
        }
        receipt["token_vs_gt"] = token_disagreement(
            tokens, labels, "DALI (scored lineage) vs shipped decoded tokens"
        )
        receipt["token_vs_gt_control"] = token_disagreement(
            tokens, control, "PyAV (control lineage) vs shipped decoded tokens"
        )
        receipt["caches"] = {
            "input_shipped_tokens": write_token_cache(
                tokens, args.out / "cache_input_shipped_tokens.pt"
            ),
            "target_aligned_dali": write_token_cache(
                labels, args.out / "cache_target_dali_seg.pt"
            ),
            "control_pyav": write_token_cache(
                control, args.out / "cache_control_pyav_seg.pt"
            ),
            "init_checkpoint": write_init_checkpoint(
                state, args.out / "init_shipped_semantic_renderer.pt"
            ),
        }
    receipt["elapsed_seconds"] = time.time() - started

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "identity_gate_receipt.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
