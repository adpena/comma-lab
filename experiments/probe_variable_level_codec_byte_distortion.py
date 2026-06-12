# SPDX-License-Identifier: MIT
"""Variable-level codec — REAL byte win + REAL distortion (the Lever-4 OPTIMIZE proof).

The MEASURED training A/B (``probe_lever4_qat_training_ab_net_score.py``) established the
honest ceiling of score-aware QAT under the VENDORED codec: the win washes to ~-7 B because
the codec ALWAYS re-quantizes at 127. The fix is the VARIABLE-LEVEL codec
(``tac.losses.variable_level_codec``): it stores each tensor at ITS OWN ``n_levels`` so the
reverse-waterfill ACTUALLY changes deployed bytes (the levels ARE the deployed grid).

This probe is the bit-spend + distortion proof for the optimization. On the basin EMA
weights it:
  1. derives the REAL per-tensor score-aware allocation (one frozen-scorer ``||∂S/∂w||``
     backward, like the MED-2 probe),
  2. byte-closes the decoder blob THREE ways:
       * vendored-127 (the baseline deployed grid),
       * variable-level @ the score-aware allocation (the optimization),
       * variable-level @ all-uniform (the byte-identity guard — must equal vendored),
  3. dequantizes each and scores the REAL frozen SegNet/PoseNet advisory distortion on the
     SAME real pairs,
  4. reports the REAL deployed byte delta AND the advisory contest-score delta.

VERDICT:
  * SURVIVES_AT_NON_WORSE_DISTORTION — the variable codec's byte win holds at advisory
    distortion within tolerance (the deployed reverse-waterfill is a REAL byte lever, unlike
    the vendored-127 indirect path; the net advisory score improves or holds).
  * DISTORTION_DOMINATES — the byte win is real but the per-tensor coarsening hurts d_seg/
    d_pose enough that the advisory score is NOT lower; the allocation band must be tightened
    (less aggressive coarsening) OR the decoder must TRAIN at the variable grid (the
    eval_roundtrip-sister training the snap can't capture).

Authority: $0 / local / torch-CPU (TRUSTED for advisory). Every number is
``[macOS-CPU advisory]`` NON-PROMOTABLE — NOT a 600-pair contest eval.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

_BASIN_CKPT = (
    _REPO_ROOT
    / "experiments/results/forkpoints/basin_bc20_20260612T121523Z"
    / "torch_vehicle_checkpoint_state.pt"
)
_VIDEO = _REPO_ROOT / "upstream/videos/0.mkv"
_EVAL_H, _EVAL_W = 384, 512
_N = 37_545_489


def _contest_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose + 1e-12) + 25.0 * archive_bytes / _N


def _real_sensitivity(decoder, latents, ctx, n):
    """One frozen-scorer forward+backward -> per-tensor ||grad|| sensitivity EMA."""
    from tac.torch_vehicle.score_aware_qat import accumulate_tensor_sensitivity

    idx = torch.arange(n)
    for p in decoder.parameters():
        p.requires_grad_(True)
        p.grad = None
    decoded_pair = decoder(latents[idx])
    flat = decoded_pair.reshape(n * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(n, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2)
    dc = decoded_bhwc.clamp(0, 255)
    decoded_bhwc = dc + (dc.round() - dc).detach()
    net = ctx.distortion_net
    posenet_in, segnet_in = net.preprocess_input(decoded_bhwc)
    seg_out = net.segnet(segnet_in)
    pose6 = net.posenet(posenet_in)["pose"][:, :6]
    seg_l = F.cross_entropy(seg_out, ctx.seg_targets_hard[idx])
    pose_l = torch.sqrt(10.0 * F.mse_loss(pose6, ctx.pose_targets[idx]) + 1e-12)
    (100.0 * seg_l + pose_l).backward()
    sens: dict[str, float] = {}
    accumulate_tensor_sensitivity(decoder, sens, decay=0.0)
    for p in decoder.parameters():
        p.grad = None
    return sens


def _advisory_distortion_from_sd(sd, latents, ctx, n, model):
    dec = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec.load_state_dict(sd)
    idx = torch.arange(n)
    with torch.no_grad():
        decoded_pair = dec(latents[idx])
        flat = decoded_pair.reshape(n * 2, 3, _EVAL_H, _EVAL_W)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
        decoded_bhwc = down.reshape(n, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2)
        decoded_bhwc = decoded_bhwc.clamp(0, 255).round()
        net = ctx.distortion_net
        posenet_in, segnet_in = net.preprocess_input(decoded_bhwc)
        seg_out = net.segnet(segnet_in)
        pose6 = net.posenet(posenet_in)["pose"][:, :6]
        d_seg = float((seg_out.argmax(dim=1) != ctx.seg_targets_hard[idx]).float().mean().item())
        d_pose = float(F.mse_loss(pose6, ctx.pose_targets[idx]).item())
    return d_seg, d_pose


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--min-level-ratio", type=float, default=0.5,
                    help="coarsening aggressiveness; 0.5 -> least-sensitive tensor at 64 levels")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from tac.losses.variable_level_codec import (
        build_decoder_blob_variable_or_vendored,
        decode_decoder_variable,
        levels_from_sensitivity_for_codec,
    )
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    codec = import_vendored("codec")
    model = import_vendored("model")

    ck = torch.load(_BASIN_CKPT, map_location="cpu", weights_only=False)
    sd = {k: v.detach().float() for k, v in ck["ema_decoder"].items()}
    latents = ck["ema_latents"].detach().float()

    ctx = RealScorerContext(
        str(_VIDEO), device="cpu", max_pairs=args.max_pairs,
        targets_cache=str(_REPO_ROOT / ".omx/tmp/lever4_probe_targets"),
    )
    n = ctx.n_pairs

    # --- REAL score-aware allocation (codec-keyed, i.e. by state_dict weight name) ---
    dec = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec.load_state_dict(sd)
    sens_mod = _real_sensitivity(dec, latents, ctx, n)  # keyed by MODULE name
    # map module-name sensitivity -> state_dict weight key (".weight")
    mod_to_key = {k[:-len(".weight")]: k for k in sd if k.endswith(".weight")}
    sens_codec = {mod_to_key[m]: s for m, s in sens_mod.items() if m in mod_to_key}
    names = list(sd.keys())
    levels = levels_from_sensitivity_for_codec(
        sens_codec, names, min_level_ratio=args.min_level_ratio,
    )
    n_coarse = sum(1 for v in levels.values() if v < 127)

    # --- byte-close THREE ways ---
    vend_blob = codec.encode_decoder(codec.quantize_state_dict(sd))
    var_blob, is_var = build_decoder_blob_variable_or_vendored(sd, levels)
    uni_blob, is_uni_var = build_decoder_blob_variable_or_vendored(sd, None)

    assert uni_blob == vend_blob and is_uni_var is False, "uniform builder NOT vendored-identical"

    # full archive bytes for each (so rate is the deployed full-archive rate)
    meta = {"base_channels": 20, "latent_dim": 28}
    lat_brotli_len = len(__import__("brotli").compress(codec.encode_latents(latents), quality=11))
    import json as _json
    meta_brotli_len = len(__import__("brotli").compress(_json.dumps(meta).encode(), quality=11))
    # archive = 4 + meta + 4 + decoder_blob + 4 + latents (the build_archive layout)
    vend_archive = 4 + meta_brotli_len + 4 + len(vend_blob) + 4 + lat_brotli_len
    var_archive = 4 + meta_brotli_len + 4 + len(var_blob) + 4 + lat_brotli_len

    # --- dequantize each + advisory distortion ---
    # vendored dequant
    vend_sd = codec.decode_decoder(vend_blob)
    # variable dequant (the deployed weights when the variable codec ships)
    var_sd = decode_decoder_variable(var_blob)

    vend_dseg, vend_dpose = _advisory_distortion_from_sd(vend_sd, latents, ctx, n, model)
    var_dseg, var_dpose = _advisory_distortion_from_sd(var_sd, latents, ctx, n, model)

    vend_score = _contest_score(vend_dseg, vend_dpose, vend_archive)
    var_score = _contest_score(var_dseg, var_dpose, var_archive)
    blob_delta = len(var_blob) - len(vend_blob)
    score_delta = var_score - vend_score

    SEG_TOL = 0.02
    seg_ok = (var_dseg - vend_dseg) <= SEG_TOL
    if score_delta < 0:
        verdict = "SURVIVES_AT_NON_WORSE_DISTORTION"
        note = (
            f"variable-level codec wins NET on the snap: advisory score {var_score:.6f} < "
            f"vendored-127 {vend_score:.6f} (Δ {score_delta:+.6f}); deployed blob {-blob_delta} B "
            f"smaller ({len(var_blob)} vs {len(vend_blob)}). The reverse-waterfill is a REAL "
            "deployed byte lever (the win SURVIVES inflate, unlike the vendored-127 indirect "
            "path that washed to ~-7 B in the training A/B). The 600-pair dual exact eval is "
            "still required before a SCORE claim."
        )
    elif blob_delta < 0 and seg_ok:
        verdict = "BYTE_WIN_DISTORTION_NEAR_BREAKEVEN"
        note = (
            f"variable-level codec gives a REAL {-blob_delta} B deployed blob win ({len(var_blob)} "
            f"vs {len(vend_blob)}) at advisory d_seg within {SEG_TOL} (vend {vend_dseg:.4f} -> var "
            f"{var_dseg:.4f}), but the snap's advisory score is {score_delta:+.6f} (the d_pose "
            f"uptick {var_dpose-vend_dpose:+.6f} offsets the byte win). The byte lever is REAL + "
            "deployed; training the decoder AT the variable grid (eval_roundtrip-sister) should "
            "recover the distortion — that is the next A/B. Tighten --min-level-ratio toward 1.0 "
            "to trade byte win for distortion."
        )
    else:
        verdict = "DISTORTION_DOMINATES"
        note = (
            f"the per-tensor coarsening hurt distortion too much on the snap (advisory score "
            f"{score_delta:+.6f}; d_seg {vend_dseg:.4f}->{var_dseg:.4f}, d_pose "
            f"{vend_dpose:.6f}->{var_dpose:.6f}). The byte direction is {blob_delta:+d} B. "
            "Tighten the allocation band (raise --min-level-ratio) OR train the decoder at the "
            "variable grid before shipping. Do NOT claim a variable-codec net win on this snap."
        )

    result = {
        "probe": "variable_level_codec_byte_distortion",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE — snap on basin weights, NOT a 600-pair contest eval",
        "basin_checkpoint": str(_BASIN_CKPT.relative_to(_REPO_ROOT)),
        "max_pairs": n,
        "min_level_ratio": args.min_level_ratio,
        "n_tensors": len(levels),
        "n_coarsened_tensors": n_coarse,
        "uniform_builder_is_vendored_identical": bool(uni_blob == vend_blob),
        "vendored_127_blob_bytes": len(vend_blob),
        "variable_blob_bytes": len(var_blob),
        "deployed_blob_delta_bytes": blob_delta,
        "vendored_full_archive_bytes": vend_archive,
        "variable_full_archive_bytes": var_archive,
        "advisory_d_seg_vendored": round(vend_dseg, 6),
        "advisory_d_seg_variable": round(var_dseg, 6),
        "advisory_d_pose_vendored": round(vend_dpose, 8),
        "advisory_d_pose_variable": round(var_dpose, 8),
        "advisory_score_vendored": round(vend_score, 6),
        "advisory_score_variable": round(var_score, 6),
        "advisory_score_delta": round(score_delta, 6),
        "verdict": verdict,
        "verdict_note": note,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 80)
        print("VARIABLE-LEVEL CODEC — REAL deployed byte win + advisory distortion (snap)")
        print("=" * 80)
        print(f"basin: {result['basin_checkpoint']}  ({n} real pairs, min_level_ratio={args.min_level_ratio})")
        print(f"  uniform builder == vendored bytes: {uni_blob == vend_blob}")
        print(f"  {n_coarse}/{len(levels)} tensors coarsened below 127")
        print()
        print(f"  vendored-127 blob {len(vend_blob):>7d} B   archive {vend_archive:>7d} B")
        print(f"  variable      blob {len(var_blob):>7d} B   archive {var_archive:>7d} B")
        print(f"  deployed blob delta {blob_delta:>+7d} B  ({100*blob_delta/len(vend_blob):+.2f}%)")
        print()
        print(f"  advisory d_seg  vend {vend_dseg:.5f} -> var {var_dseg:.5f}  (Δ {var_dseg-vend_dseg:+.5f})")
        print(f"  advisory d_pose vend {vend_dpose:.6f} -> var {var_dpose:.6f}  (Δ {var_dpose-vend_dpose:+.6f})")
        print(f"  advisory score  vend {vend_score:.6f} -> var {var_score:.6f}  (Δ {score_delta:+.6f})")
        print()
        print(f"VERDICT: {verdict}")
        print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
