# SPDX-License-Identifier: MIT
"""BLIND-SPOT PROBE C (measurement-trust): is the d_seg "wall" REAL or an
EMA-shadow-lag artifact (the capstone bug class)?

$0 CPU-ONLY, READ-ONLY on the running trajectory + basin. Loads the LIVE decoder
and the EMA-SHADOW decoder from a checkpoint and computes the EXACT contest
d_seg/d_pose for BOTH on the SAME frames via the vendored authority path
(``score.evaluate_decoder`` streaming GT through ``frame_utils.yuv420_to_rgb``,
the canonical GT decode — PyAV rgb24 is FORBIDDEN).

AUDIT 1: live vs EMA-shadow d_seg gap (the capstone bug: shadow lags → fakes a
         plateau). Quantify per checkpoint.
AUDIT 2: pose eval-path variance root-cause: live d_pose vs shadow d_pose vs
         byte-closed-archive d_pose on the SAME state.
AUDIT 3: CE power-law trust: does d_seg(LIVE) follow 0.0367*ep^-0.351, or is the
         flattening tail a shadow artifact? (uses AUDIT-1 live points + the
         trajectory shadow points.)

NON-PROMOTABLE [contest-CPU advisory]. NOT a score claim; a measurement-trust
diagnostic. The basin/ARM-B checkpoints are NEVER mutated (read-only torch.load).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _build_decoder(base_channels: int, latent_dim: int, *, taper_channels=None,
                   pose_film_v2=False, n_pairs=600, film_hidden=8, state_dict=None):
    """Build the decoder matching the CHECKPOINT architecture (faithful per arm).

    * default            -> plain vendored HNeRVDecoder (the basin).
    * taper_channels set  -> ConfigurableTaperHNeRVDecoder with that schedule.
    * pose_film_v2        -> wrap in PoseFiLMHNeRVWrapperV2 (ARM B's FiLM carrier);
                             the ``stored_pose`` buffer is restored from the
                             state_dict so the FiLM pose path matches the run.
    Built on CPU (the authority device). Returns (module, score_mod)."""
    from tac.torch_vehicle.vendored_imports import import_vendored

    model_mod = import_vendored("model")
    score_mod = import_vendored("score")
    EVAL_H = getattr(score_mod, "EVAL_H", 384)
    EVAL_W = getattr(score_mod, "EVAL_W", 512)

    if taper_channels is not None:
        from tac.torch_vehicle.configurable_taper_decoder import (
            ConfigurableTaperHNeRVDecoder,
        )
        core = ConfigurableTaperHNeRVDecoder(
            latent_dim=latent_dim, base_channels=base_channels,
            eval_size=(EVAL_H, EVAL_W), channels=list(taper_channels),
        ).to("cpu")
    else:
        core = model_mod.HNeRVDecoder(
            latent_dim=latent_dim, base_channels=base_channels,
            eval_size=(EVAL_H, EVAL_W),
        ).to("cpu")

    if pose_film_v2:
        from tac.torch_vehicle.pose_film_v2 import PoseFiLMHNeRVWrapperV2
        mod = PoseFiLMHNeRVWrapperV2(
            core, n_pairs=n_pairs, film_hidden=film_hidden,
        ).to("cpu")
    else:
        mod = core
    mod.eval()
    return mod, score_mod


def _arch_from_checkpoint(state_dict, manifest):
    """Infer (taper_channels, pose_film_v2, film_hidden) from manifest+ckpt keys."""
    taper = manifest.get("taper_channels")
    pose_film_v2 = "stored_pose" in state_dict and any(
        k.startswith("decoder.") for k in state_dict)
    film_hidden = 8
    if "pose_mlp.fc1.weight" in state_dict:
        film_hidden = int(state_dict["pose_mlp.fc1.weight"].shape[0])
    return taper, pose_film_v2, film_hidden


def _exact_metrics(decoder, latents, score_mod, distortion_net, video_path,
                   n_pairs, archive_bytes, *, film_idx=False):
    """Exact d_seg/d_pose/score via the vendored authority evaluate_decoder.

    ``film_idx``: when True AND the decoder is a FiLM wrapper, wrap it so the
    eval ``decoder(z)`` call routes through the FiLM-conditioned path (idx looks
    up stored_pose). When False the wrapper's ``forward(z)`` runs FiLM-FREE (the
    vendored rgb_0 path) — this isolates the FiLM carrier's contribution to
    d_pose (AUDIT 2 root-cause)."""
    latents = latents[:n_pairs].to("cpu")
    eval_mod = decoder
    if film_idx and hasattr(decoder, "stored_pose"):
        # shim so evaluate_decoder's decoder(z) passes the matching idx (the
        # eval batches pairs in-order from 0, batch_pairs=8 → idx = arange).
        class _FilmIdxShim(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self._cursor = 0

            def eval(self):
                self.inner.eval()
                return self

            def forward(self, z):
                b = z.shape[0]
                idx = torch.arange(self._cursor, self._cursor + b,
                                   device=z.device)
                self._cursor += b
                return self.inner(z, idx)

        eval_mod = _FilmIdxShim(decoder)
    with torch.inference_mode():
        dist = score_mod.evaluate_decoder(
            eval_mod, latents, distortion_net, str(video_path),
            batch_pairs=8, device="cpu",
        )
    d_seg = float(dist["seg_distortion"])
    d_pose = float(dist["pose_distortion"])
    rate = archive_bytes / 37_545_489.0
    score = 100.0 * d_seg + (10.0 * d_pose + 1e-12) ** 0.5 + 25.0 * rate
    return {"d_seg": d_seg, "d_pose": d_pose, "rate": rate, "score": score}


def _load_state(ckpt_path: Path):
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _restore(mod, sd):
    """Load a state dict; tolerate the FiLM ``decoder.`` prefix + buffers."""
    mod.load_state_dict(sd, strict=True)
    return mod


def audit_checkpoint(label, result_dir: Path, distortion_net, score_mod_holder,
                     video_path, n_pairs, base_channels, latent_dim):
    ckpt = result_dir / "torch_vehicle_checkpoint_state.pt"
    summ = json.loads((result_dir / "torch_vehicle_summary.json").read_text())
    manifest = json.loads(
        (result_dir / "torch_vehicle_checkpoint_manifest.json").read_text())
    st = _load_state(ckpt)
    archive_bytes = int(summ.get("last_eval", {}).get("archive_bytes", 90000))

    taper, pose_film_v2, film_hidden = _arch_from_checkpoint(st["decoder"], manifest)

    def build():
        return _build_decoder(
            base_channels, latent_dim, taper_channels=taper,
            pose_film_v2=pose_film_v2, n_pairs=st["latents"].shape[0],
            film_hidden=film_hidden)

    dec_live, score_mod = build()
    score_mod_holder["m"] = score_mod
    dec_shadow, _ = build()
    _restore(dec_live, st["decoder"])
    _restore(dec_shadow, st["ema_decoder"])
    live_lat = st["latents"]
    shadow_lat = st["ema_latents"]

    # For FiLM arms the wrapper's forward(z) is FiLM-FREE; pass film_idx=True to
    # route the CONDITIONED path. Measure BOTH to isolate the carrier (AUDIT 2).
    live = _exact_metrics(dec_live, live_lat, score_mod, distortion_net,
                          video_path, n_pairs, archive_bytes,
                          film_idx=pose_film_v2)
    shadow = _exact_metrics(dec_shadow, shadow_lat, score_mod, distortion_net,
                            video_path, n_pairs, archive_bytes,
                            film_idx=pose_film_v2)

    out = {
        "label": label,
        "result_dir": str(result_dir),
        "arch": {"taper_channels": taper, "pose_film_v2": pose_film_v2,
                 "film_hidden": film_hidden},
        "best_ep": summ.get("best_ep"),
        "global_epoch_at_ckpt": st.get("global_epoch"),
        "archive_bytes": archive_bytes,
        "n_pairs_evaluated": n_pairs,
        "ema_decay": summ.get("run_meta", {}).get("ema_decay"),
        "LIVE": live,
        "EMA_SHADOW": shadow,
        "live_vs_shadow_d_seg_gap": live["d_seg"] - shadow["d_seg"],
        "live_vs_shadow_d_pose_gap": live["d_pose"] - shadow["d_pose"],
        "live_vs_shadow_score_gap": live["score"] - shadow["score"],
        "summary_last_eval": summ.get("last_eval"),
        "summary_best_score": summ.get("best_score"),
    }

    if pose_film_v2:
        # AUDIT 2 carrier isolation: FiLM-FREE pose vs FiLM-conditioned pose on
        # the SAME (live) state. d_seg is invariant (rgb_1 is FiLM-clean) so any
        # d_pose difference IS the carrier's contribution + variance.
        live_filmfree = _exact_metrics(
            dec_live, live_lat, score_mod, distortion_net, video_path,
            n_pairs, archive_bytes, film_idx=False)
        shadow_filmfree = _exact_metrics(
            dec_shadow, shadow_lat, score_mod, distortion_net, video_path,
            n_pairs, archive_bytes, film_idx=False)
        out["FILM_ISOLATION"] = {
            "live_film_on_d_pose": live["d_pose"],
            "live_film_off_d_pose": live_filmfree["d_pose"],
            "live_carrier_d_pose_delta": live["d_pose"] - live_filmfree["d_pose"],
            "shadow_film_on_d_pose": shadow["d_pose"],
            "shadow_film_off_d_pose": shadow_filmfree["d_pose"],
            "live_film_on_d_seg": live["d_seg"],
            "live_film_off_d_seg": live_filmfree["d_seg"],
            "d_seg_film_invariant": abs(live["d_seg"] - live_filmfree["d_seg"]),
        }
    else:
        # cross-pairing (basin / non-FiLM): localize pose variance to weights vs
        # latents — shadow DECODER + LIVE latents and vice-versa.
        out["CROSS_shadowdec_livelat"] = _exact_metrics(
            dec_shadow, live_lat, score_mod, distortion_net, video_path,
            n_pairs, archive_bytes)
        out["CROSS_livedec_shadowlat"] = _exact_metrics(
            dec_live, shadow_lat, score_mod, distortion_net, video_path,
            n_pairs, archive_bytes)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=64,
                    help="pairs to evaluate (full=600; 64 is a faithful sub-sample "
                         "for the live-vs-shadow GAP which is the audit signal)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", nargs="+", default=["basin", "armb"])
    args = ap.parse_args()

    video_path = REPO / "upstream/videos/0.mkv"
    dirs = {
        "basin": REPO / "experiments/results/torch_vehicle_full_mps_basin_bc20_n600",
        "armb": REPO / "experiments/results/bindall_arm_b_canonical50k_n600",
    }

    from tac.score_aware_loop.targets import load_frozen_distortion_net
    print("[probe-c] loading frozen distortion net on CPU (authority)...",
          file=sys.stderr)
    distortion_net = load_frozen_distortion_net(device="cpu")

    score_mod_holder = {}
    out = {"audit": "blindspot_probe_c_measurement_trust",
           "authority": "[contest-CPU advisory] NON-PROMOTABLE", "results": []}
    for arm in args.arms:
        d = dirs[arm]
        print(f"[probe-c] === {arm}: {d.name} (n_pairs={args.n_pairs}) ===",
              file=sys.stderr)
        res = audit_checkpoint(
            arm, d, distortion_net, score_mod_holder, video_path,
            args.n_pairs, base_channels=20, latent_dim=28)
        out["results"].append(res)
        print(json.dumps(res, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[probe-c] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
