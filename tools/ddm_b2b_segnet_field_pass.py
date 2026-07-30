#!/usr/bin/env python
"""ddm_b2b — QA75/QA80 SegNet field-pass harness (READY-TO-RUN, post-burn).

The two burn-2 blockers whose scorer step was deliberately deferred (b2p §2/§3):

* **QA75** (ph3 §10.1): the LOGIT/MARGIN DISTILL FIELD over the materialized EXACT-solve
  frames (``tac.witness_dsl.qa75_solve_frame_targets.SolveFrameTargets``). Soft targets that
  encode the boundary-annulus structure exactly where 100%% of flips live, with margins
  FEASIBLE by construction (solve margins are realized). The burn-2 distill stage consumes it.
* **QA80** (ph3 §10.2): the EXACT per-pixel flip-distance field ``d=|m|/||dw||`` over the burn
  frames — needs the RUNNER-UP class (2nd-argmax) the gt cache does not carry, hence a SegNet
  pass. Feeds the margin-slack photometric budget (band lemma).

BOTH derive from ONE SegNet forward (logits -> top2 -> argmax/runner/margin), so this harness
runs it ONCE per frame and emits either/both fields. The scorer is INJECTABLE: the real run
loads the frozen CPU-torch SegNet (NEVER MPS); the smoke uses a deterministic STUB so the
plumbing (loader -> derived chunking -> field compute -> SSD manifest+sha) is validated with
NO Metal / NO real SegNet / NO n600 pass (the scorer pass is POST-BURN).

Chunking honors the charter ``<=120 pairs/chunk`` law (the #205 verdict-batch OOM lesson: a
full-P batched scorer forward spikes RSS; chunking is bit-identical because eval-mode BatchNorm
uses running stats).

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED. Advisory fields; score_claim=False.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

SEG_H, SEG_W, N_CLASSES = 384, 512, 5
MAX_CHUNK_PAIRS = 120  # charter: <=120 pairs/chunk (verdict-batch OOM law, #205)

#: One chunk's SegNet fields (all at the 384x512 scorer resolution).
SegNetFields = dict  # {"argmax":(n,H,W) int64, "runner":(n,H,W) int64, "margin":(n,H,W) f64,
#                       "logits":(n,5,H,W) f32}
SegNetFieldFn = Callable[[np.ndarray], SegNetFields]  # (frames (n,H,W,3) uint8) -> fields


def derive_chunk_size(n_pairs: int, requested: int | None) -> int:
    """Derived (never a bare constant): the chunk is min(requested-or-cap, n_pairs), capped at
    the charter MAX_CHUNK_PAIRS. Fewer pairs than the cap => one chunk."""
    cap = MAX_CHUNK_PAIRS if requested is None else min(int(requested), MAX_CHUNK_PAIRS)
    return max(1, min(cap, int(n_pairs)))


def real_segnet_field_fn(device: str = "cpu") -> SegNetFieldFn:
    """The REAL frozen CPU-torch SegNet forward (NEVER MPS). Replicates the canonical
    ``measure_segnet_argmax`` preprocess and additionally returns the runner-up + logits."""
    import torch

    from tac.boundary_math.seg_core import load_real_segnet

    if device not in ("cpu", "cuda"):
        raise ValueError("SegNet authority device is cpu|cuda, NEVER mps")
    segnet = load_real_segnet(device=device)

    def _fn(frames: np.ndarray) -> SegNetFields:
        out_a, out_r, out_m, out_l = [], [], [], []
        for fr in frames:
            r = np.asarray(fr, dtype=np.float64)
            pair = torch.from_numpy(np.stack([r, r], axis=0)[None]).float()  # (1,2,H,W,3)
            xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
            with torch.inference_mode():
                seg_in = segnet.preprocess_input(xp)          # -> (1,3,384,512)
                logits = segnet(seg_in)                        # (1,5,384,512)
                top2 = torch.topk(logits, k=2, dim=1)
                margin = (top2.values[:, 0] - top2.values[:, 1]).clamp_min(0.0)[0]
                out_a.append(top2.indices[:, 0][0].cpu().numpy().astype(np.int64))
                out_r.append(top2.indices[:, 1][0].cpu().numpy().astype(np.int64))
                out_m.append(margin.cpu().numpy().astype(np.float64))
                out_l.append(logits[0].cpu().numpy().astype(np.float32))
        return {"argmax": np.stack(out_a), "runner": np.stack(out_r),
                "margin": np.stack(out_m), "logits": np.stack(out_l)}

    return _fn


def stub_segnet_field_fn(seed: int = 0) -> SegNetFieldFn:
    """Deterministic STUB (smoke only): fabricates well-formed fields (winner != runner) of the
    exact real shapes so the harness plumbing is validated with NO real SegNet / NO Metal."""

    def _fn(frames: np.ndarray) -> SegNetFields:
        n = int(frames.shape[0])
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal((n, N_CLASSES, SEG_H, SEG_W)).astype(np.float32)
        order = np.argsort(-logits, axis=1)          # descending per pixel
        argmax = order[:, 0].astype(np.int64)
        runner = order[:, 1].astype(np.int64)
        top1 = np.take_along_axis(logits, order[:, 0:1], axis=1)[:, 0]
        top2 = np.take_along_axis(logits, order[:, 1:2], axis=1)[:, 0]
        margin = np.clip(top1 - top2, 0.0, None).astype(np.float64)
        return {"argmax": argmax, "runner": runner, "margin": margin, "logits": logits}

    return _fn


class FramePairSource:
    """Uniform frame source: pair_id -> the frame1 (H,W,3) uint8 the SegNet scores.

    Plain class (NOT a dataclass): a ``Callable`` field on a frozen dataclass trips
    ``dataclasses._is_type`` under importlib spec-load (module not yet in sys.modules)."""

    def __init__(self, name: str, count: int, get: Callable[[int], np.ndarray]) -> None:
        self.name = name
        self.count = int(count)
        self._get = get

    def frame1(self, pair_id: int) -> np.ndarray:
        fr = np.ascontiguousarray(self._get(pair_id))
        if fr.shape[-1] != 3 or fr.ndim != 3:
            raise ValueError(f"{self.name} pair {pair_id} frame must be (H,W,3); got {fr.shape}")
        return fr.astype(np.uint8)


def qa75_solve_frame_source(frames_root: str | Path) -> FramePairSource:
    """QA75: the materialized EXACT-solve frames (b2p SolveFrameTargets)."""
    from tac.witness_dsl.qa75_solve_frame_targets import SolveFrameTargets

    tgt = SolveFrameTargets.load(frames_root)
    return FramePairSource("qa75_solve", tgt.pair_count, lambda i: np.asarray(tgt.frame1(i)))


def npy_frame_source(name: str, frames_npy: str | Path) -> FramePairSource:
    """QA80: burn frame1s as an (N,H,W,3) uint8 .npy (post-burn decode artifact)."""
    arr = np.load(frames_npy, mmap_mode="r", allow_pickle=False)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"{name} npy must be (N,H,W,3) uint8; got {arr.shape}")
    return FramePairSource(name, int(arr.shape[0]), lambda i: np.asarray(arr[i]))


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_field_pass(source: FramePairSource, out_dir: str | Path, field_fn: SegNetFieldFn, *,
                   field_kind: str = "both", chunk: int | None = None,
                   limit: int | None = None) -> dict:
    """Run the SegNet field pass over ``source``, chunked, emitting per-pair fields + a manifest.

    field_kind: "distill_logit_margin" (QA75) | "exact_flip_distance" (QA80) | "both".
    Writes ``pair-NNNNNN.npz`` (the requested fields) + ``field_pass_manifest.json`` (per-pair
    sha256 + schema). Determinism: same source + same field_fn => bit-identical .npz.
    """
    from tac.boundary_math.margin_budget_field import exact_flip_distance_field

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = source.count if limit is None else min(int(limit), source.count)
    csize = derive_chunk_size(n, chunk)
    rows: list[dict] = []
    for lo in range(0, n, csize):
        hi = min(lo + csize, n)
        frames = np.stack([source.frame1(i) for i in range(lo, hi)])
        fields = field_fn(frames)
        for j, pid in enumerate(range(lo, hi)):
            payload: dict[str, np.ndarray] = {}
            if field_kind in ("distill_logit_margin", "both"):
                payload["distill_logits"] = fields["logits"][j].astype(np.float16)
                payload["distill_margin"] = fields["margin"][j].astype(np.float32)
                payload["argmax"] = fields["argmax"][j].astype(np.uint8)
            if field_kind in ("exact_flip_distance", "both"):
                d = exact_flip_distance_field(
                    fields["margin"][j], fields["argmax"][j], fields["runner"][j])
                payload["exact_flip_distance"] = d.astype(np.float32)
                payload["winner"] = fields["argmax"][j].astype(np.uint8)
                payload["runner"] = fields["runner"][j].astype(np.uint8)
            buf = _npz_bytes(payload)
            path = out / f"pair-{pid:06d}.npz"
            path.write_bytes(buf)
            rows.append({"pair_id": pid, "path": path.name, "sha256": _sha256_bytes(buf)})
    manifest = {
        "schema": "ddm_b2b_segnet_field_pass.v1", "source": source.name,
        "field_kind": field_kind, "pair_count": n, "chunk_size": csize,
        "geometry": {"seg_h": SEG_H, "seg_w": SEG_W, "n_classes": N_CLASSES},
        "authority": "local-CPU-torch-advisory (frozen scorer; NEVER MPS)",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "pairs": rows,
    }
    (out / "field_pass_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _npz_bytes(payload: dict[str, np.ndarray]) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez(buf, **payload)
    return buf.getvalue()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-source", choices=("qa75_solve", "qa80_burn"), required=True)
    ap.add_argument("--frames-root", type=str, default=None,
                    help="QA75: SolveFrameTargets root (with manifest.json)")
    ap.add_argument("--frames-npy", type=str, default=None,
                    help="QA80: (N,H,W,3) uint8 burn frame1s .npy")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--field-kind", default="both",
                    choices=("distill_logit_margin", "exact_flip_distance", "both"))
    ap.add_argument("--chunk", type=int, default=None,
                    help=f"pairs/chunk (derived; capped at charter {MAX_CHUNK_PAIRS})")
    ap.add_argument("--limit", type=int, default=None, help="cap pairs (smoke)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="SegNet authority device (NEVER mps)")
    ap.add_argument("--smoke-stub-scorer", action="store_true",
                    help="use the deterministic STUB scorer (plumbing smoke; NO real SegNet/Metal)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.frame_source == "qa75_solve":
        if not args.frames_root:
            raise SystemExit("--frames-root required for qa75_solve")
        source = qa75_solve_frame_source(args.frames_root)
    else:
        if not args.frames_npy:
            raise SystemExit("--frames-npy required for qa80_burn")
        source = npy_frame_source("qa80_burn", args.frames_npy)
    field_fn = (stub_segnet_field_fn() if args.smoke_stub_scorer
                else real_segnet_field_fn(args.device))
    manifest = run_field_pass(source, args.out_dir, field_fn, field_kind=args.field_kind,
                              chunk=args.chunk, limit=args.limit)
    print(json.dumps({k: manifest[k] for k in
                      ("source", "field_kind", "pair_count", "chunk_size")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
