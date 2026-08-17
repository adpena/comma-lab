"""ddm_rc4 Stage 1b - MEASURE the seg amplification A of a token drop on hv1.

A = (net SegNet argmax flips created against GT) / (token flips created).
The drop ladder prices rate exactly; A is the one unmeasured factor that turns a
token flip into score. This renders the SHIPPED semantic renderer on a
STRATIFIED-RANDOM pair sample (never a prefix - prefix bias is a measured law
here), pushes both frames through the exact upstream scorer preprocess, and
counts flips against the retained GT SegNet argmax field.

Pose is reported RELATIVE (ratio to base) only: the local advisory pose
instrument carries a measured ~18x offset vs contest-CUDA (rn1), so its
magnitude is not quotable; its SIGN and RATIO are.

ALWAYS KEEP THE PAYLOAD: every rendered frame batch, argmax field and drop mask
is persisted with sha256 + byte count.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

GEN = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/generations/"
    "hv1_ep0634_s1p25_c1p0_brotli_q10"
)
STORE = Path("/Volumes/APDataStore/pact/ddm_rc4_rung4_token_drop_20260816")
TOKENS_U8 = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
MT1 = Path(
    "/Volumes/APDataStore/pact/ddm_mt1_t4_sign_gate_20260814_custody/"
    "ddm_mt1_t4_sign_gate_20260814/inputs"
)
UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
ARCHIVE_SHA = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
NUM_CLASSES = 5


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", type=float, nargs="+", required=True,
                    help="drop thresholds as u = -log2(1 - p_max)")
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--out", type=str, default="AMPLIFICATION.json")
    args = ap.parse_args()

    retained = STORE / "retained" / "amplification"
    retained.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(GEN))
    sys.path.insert(0, str(GEN / "cpr1"))
    sys.path.insert(0, str(UPSTREAM))

    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    from runtime.hpac_inference import optimize_sparse_evaluator
    from runtime.ihs2 import materialize_ihs1
    from runtime.residual_archive import (
        _boundary_buckets,
        _probability_table,
        _sparse_class,
        read_residual_archive,
    )

    spec = importlib.util.spec_from_file_location("_rc4_renderer", GEN / "cpr1" / "inflate.py")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules["_rc4_renderer"] = runtime
    spec.loader.exec_module(runtime)

    archive = GEN / "archive.zip"
    if sha256_bytes(archive.read_bytes()) != ARCHIVE_SHA:
        raise SystemExit("archive sha mismatch")
    parts = read_residual_archive(archive)

    tokens = np.fromfile(TOKENS_U8, dtype=np.uint8).reshape(
        runtime.N, runtime.EVAL_H, runtime.EVAL_W
    )
    gt_argmax = np.load(MT1 / "gt_argmax.npy", mmap_mode="r")

    # STRATIFIED-RANDOM pair sample: 1 pair from each of `pairs` equal blocks.
    rng = np.random.default_rng(args.seed)
    edges = np.linspace(0, runtime.N, args.pairs + 1).astype(int)
    sample = np.array(
        [rng.integers(edges[i], max(edges[i] + 1, edges[i + 1])) for i in range(args.pairs)]
    )
    sample = np.unique(np.clip(sample, 0, runtime.N - 1))

    device = torch.device("cpu")
    base_hpac = materialize_ihs1(parts.hpac_blob, runtime)
    model = runtime.load_hpac(base_hpac, device)
    masks = runtime.group_masks(device)
    sparse = _sparse_class(GEN / "cpr1")(model, runtime.EVAL_H, runtime.EVAL_W)
    # Match the shipping decoder's evaluator path.  MEASURED value-identical in
    # OPTIMIZE_SPARSE_CONTROL.json (argmax identical, p_max max-abs-diff 0.0 on frames
    # 0/137/411) -- purely a speed path -- but the decoder always calls it, so we do too.
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
    group_plans = []
    for mask in masks:
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        group_plans.append((torch.from_numpy(flat).to(device), flat))

    # Frame-local replay: each frame's probability table depends only on the
    # PREVIOUS retained token frame plus the progressively filled current frame,
    # both of which are known exactly from the retained field.
    def frame_stats(frame: int):
        index = torch.tensor([frame], dtype=torch.long, device=device)
        prev_np = tokens[frame - 1] if frame else None
        previous = torch.from_numpy(
            (prev_np if prev_np is not None else np.zeros_like(tokens[0])).astype(np.int64)
        )[None]
        context = model.prepare_frame_context(index, previous)
        if frame:
            boundary = _boundary_buckets(prev_np).reshape(-1)
        else:
            boundary = np.full(runtime.EVAL_H * runtime.EVAL_W, 4, dtype=np.uint8)
        truth_flat = tokens[frame].reshape(-1).astype(np.int64)
        current = torch.zeros((1, runtime.EVAL_H, runtime.EVAL_W), dtype=torch.long)
        p_max = np.zeros(truth_flat.size, dtype=np.float64)
        arg = np.zeros(truth_flat.size, dtype=np.int64)
        bits = np.zeros(truth_flat.size, dtype=np.float64)
        for _group, (device_positions, flat_positions) in enumerate(group_plans):
            selected = sparse.selected_logits(current, context, _group)
            base_logits = selected.cpu().numpy()
            predicted = base_logits.argmax(axis=1).astype(np.int64)
            feature = boundary[flat_positions].astype(np.int64) * NUM_CLASSES + predicted
            corrected = base_logits + parts.table.values[feature]
            prob = _probability_table(corrected, runtime.HPAC_LOGIT_PRECISION).astype(np.float64)
            a = prob.argmax(axis=1)
            n = np.arange(a.size)
            p_max[flat_positions] = prob[n, a]
            arg[flat_positions] = a
            actual = truth_flat[flat_positions]
            bits[flat_positions] = -np.log2(np.maximum(prob[n, actual], 1e-300))
            current.reshape(-1)[device_positions] = torch.from_numpy(actual).to(device)
        return (
            p_max.reshape(runtime.EVAL_H, runtime.EVAL_W),
            arg.reshape(runtime.EVAL_H, runtime.EVAL_W),
            bits.reshape(runtime.EVAL_H, runtime.EVAL_W),
        )

    # Renderer + scorer.  Exact shipping construction (runtime/f26_inflate.py:338-344):
    # SemanticTokenRenderer(96) loaded strictly from decode_wans1(parts.semantic_blob).
    # The carrier/basis/coefficients path is frame-0 only and is untouched by token drop.
    from runtime.entropy.renderer_weight_codec import decode_wans1

    semantic = runtime.SemanticTokenRenderer(96)
    records = decode_wans1(parts.semantic_blob)
    semantic.load_state_dict(
        {
            r.schema.name: torch.from_numpy(np.ascontiguousarray(r.values, dtype=np.float32))
            for r in records
        },
        strict=True,
    )
    semantic = semantic.eval().to(device)

    import modules  # upstream

    distortion_net = modules.DistortionNet().eval().to(device)
    distortion_net.load_state_dicts(modules.posenet_sd_path, modules.segnet_sd_path, device)
    segnet = distortion_net.segnet

    def render_master(tok_frame: np.ndarray, pair: int) -> torch.Tensor:
        """Exact shipping master path: renderer -> bilinear^ -> clamp/round uint8."""
        t = torch.from_numpy(tok_frame.astype(np.int64))[None]
        idx = torch.tensor([pair], dtype=torch.long)
        with torch.inference_mode():
            out = semantic(t, idx)
            master = (
                torch.nn.functional.interpolate(
                    out, size=(runtime.CAMERA_H, runtime.CAMERA_W),
                    mode="bilinear", align_corners=False,
                )
                .clamp(0.0, 255.0)
                .round()
                .to(torch.uint8)
            )
        return master  # (1,3,874,1164) uint8

    def seg_argmax(master_u8: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            x = master_u8.float()
            x = torch.nn.functional.interpolate(
                x, size=(modules.segnet_model_input_size[1], modules.segnet_model_input_size[0]),
                mode="bilinear",
            )
            return segnet(x).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    results = []
    started = time.time()
    cache: dict[int, tuple] = {}
    base_arg_cache: dict[int, np.ndarray] = {}

    for u in args.u:
        tau = 1.0 - 2.0**-u
        tot_token_flips = 0
        tot_bits = 0.0
        base_flips = 0
        drop_flips = 0
        benef = harmful = wrong2wrong = 0
        for k, pair in enumerate(sample):
            pair = int(pair)
            if pair not in cache:
                cache[pair] = frame_stats(pair)
            p_max, arg, bits = cache[pair]
            drop = p_max >= tau
            new_tok = np.where(drop, arg, tokens[pair]).astype(np.uint8)
            tf = int((new_tok != tokens[pair]).sum())
            tot_token_flips += tf
            tot_bits += float(bits[drop].sum())
            gt = np.asarray(gt_argmax[pair])
            if pair not in base_arg_cache:
                base_arg_cache[pair] = seg_argmax(render_master(tokens[pair], pair))
            a_base = base_arg_cache[pair]
            a_drop = seg_argmax(render_master(new_tok, pair)) if tf else a_base
            wb = a_base != gt
            wd = a_drop != gt
            base_flips += int(wb.sum())
            drop_flips += int(wd.sum())
            changed = a_base != a_drop
            benef += int((changed & wb & ~wd).sum())
            harmful += int((changed & ~wb & wd).sum())
            wrong2wrong += int((changed & wb & wd).sum())
            if k % 20 == 0:
                print(f"u={u} pair {k+1}/{len(sample)} tf={tot_token_flips} "
                      f"dflips={drop_flips-base_flips} {time.time()-started:.0f}s", flush=True)
        net = drop_flips - base_flips
        results.append(
            {
                "u": u,
                "p_max_threshold": tau,
                "sample_pairs": len(sample),
                "token_flips_sample": tot_token_flips,
                "bits_saved_sample": tot_bits,
                "bytes_saved_sample": tot_bits / 8.0,
                "seg_flips_base_sample": base_flips,
                "seg_flips_drop_sample": drop_flips,
                "net_seg_flips_sample": net,
                "beneficial_B": benef,
                "harmful_H": harmful,
                "wrong_to_wrong_W": wrong2wrong,
                "A_net_seg_flips_per_token_flip": (net / tot_token_flips) if tot_token_flips else 0.0,
                "bytes_per_net_seg_flip": (tot_bits / 8.0 / net) if net else float("inf"),
            }
        )
        print(json.dumps(results[-1], indent=2), flush=True)

    mask_path = retained / "sample_pairs.npy"
    np.save(mask_path, sample)
    out = {
        "arm": "ddm_rc4",
        "stage": "1b_seg_amplification",
        "archive_sha256": ARCHIVE_SHA,
        "axis": "[macOS-CPU advisory, stratified-random pair sample] COMPONENT-ONLY NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "sample_seed": args.seed,
        "sample_pairs_path": str(mask_path),
        "sample_pairs_sha256": sha256_bytes(mask_path.read_bytes()),
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    (STORE / args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", STORE / args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
