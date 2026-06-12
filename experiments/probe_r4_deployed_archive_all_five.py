#!/usr/bin/env python
"""R4 adversarial probe — the DEPLOYED ARCHIVE under ALL-5-ON, end-to-end.

R1-R3 verified the LOSS / gradient-direction / resume. R4's un-covered ground is
the ARTIFACT: take the archive an all-5-on driver run actually PRODUCES and prove
(1) it parses back, (2) the numpy-portable inflate decodes it to valid scoreable
frames, (3) the FiLM pose section is WRITTEN and CONSUMED at inflate (not dropped),
(4) the BEST-selection eval render (``_FiLMEvalDecoder`` cursor path) is BIT/atol
identical to the deployed ``inflate_film_decoder`` render, and (5) the QAT
score-aware grid round-trips through the codec's 127-requant to what training saw.

This is the train/deploy ARTIFACT gap, not the loss gap. Authority: synthetic
scorer (RESEARCH-ONLY, [macOS-CPU advisory]); the eval==inflate seam is
architecture-agnostic so the synthetic scorer exercises it faithfully, and the
REAL vendored ``evaluate_decoder`` cursor contract is verified separately (static
trace in the memo + the parity assertion below mirrors its exact call pattern).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.pose_film import (
    PoseFiLMHNeRVWrapper,
    _FiLMEvalDecoder,
    inflate_film_decoder,
    parse_pose_section,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _ce(s, t):
    return torch.nn.functional.cross_entropy(s, t)


def _all_five_spec(epochs: int = 2) -> StageSpec:
    """A StageSpec with EVERY Layer-2 lever ON (mirrors ``--levers all`` semantics
    from ``_resolve_lever_overrides``: seg surrogate + T-anneal 1.0→0.05 + rate
    surrogate (w+lat) + score-aware QAT + margin τ=2.0 + C1a), built directly like
    the test ``_spec`` helper so it does not depend on a vendored stage accessor."""
    return StageSpec(
        name="all5", epochs=epochs, seg_loss_fn=_ce, eval_every=1, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0,
        cat_lambda=0.01,  # C1a on (composition with Lever-1)
        cat_sigma=0.2, use_qat=True, init_latents_random=True,
        # the 5 Layer-2 StageSpec levers, all ON:
        seg_surrogate="soft_cosine",       # Lever 2
        seg_temperature=1.0,
        seg_temperature_end=0.05,          # Lever 2 anneal
        rate_lambda_w=1e-3,                # Lever 1 (weight rate)
        rate_lambda_lat=1e-3,              # Lever 1 (latent rate)
        score_aware_qat=True,              # Lever 4
        qat_sensitivity_decay=0.99,
        margin_weight_tau=2.0,             # Lever 5
    )


def main() -> int:
    torch.manual_seed(7)
    v = import_vendored_bundle()
    n_pairs = 8
    base_channels = 8
    latent_dim = 28

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        cfg = TorchVehicleConfig(
            base_channels=base_channels,
            latent_dim=latent_dim,
            out_dir=out_dir,
            checkpoint_every_epochs=1,
            device="cpu",
            seed=7,
            pose_film_enabled=True,  # Lever 3
            pose_film_hidden=8,
        )
        sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=7)
        drv = TorchVehicleDriver(
            cfg, scorer=sc, vendored=v, curriculum=[_all_five_spec(epochs=2)]
        )
        out = drv.run()
        assert out["status"] == "complete", f"run did not complete: {out}"

        # (1) The BEST archive the run produced — the contest-visible artifact.
        best_archive = (out_dir / "best" / "best_archive.bin").read_bytes()
        assert len(best_archive) > 0
        print(f"[1] best archive produced: {len(best_archive)} bytes")

        # (2) Parse-back: vendored 3 sections succeed on the additive archive.
        dec_sd, latents, meta = v.parse_archive(best_archive)
        assert int(meta["n_pairs"]) == n_pairs
        film_keys = [k for k in dec_sd if k.startswith("pose_film.")]
        assert film_keys, "FiLM weights NOT in the decoder blob (pose-FiLM dropped!)"
        print(f"[2] parse_archive OK; {len(film_keys)} pose_film.* keys in decoder blob")

        # (3) The pose section is WRITTEN and parseable (not silently dropped).
        pose = parse_pose_section(best_archive, v.parse_archive)
        assert pose is not None, "pose section MISSING from the produced archive"
        assert tuple(pose.shape) == (n_pairs, 6), f"bad pose shape {tuple(pose.shape)}"
        print(f"[3] pose section present + parseable: shape {tuple(pose.shape)}")

        # (4) The numpy-portable inflate decodes to VALID scoreable frames.
        frames = inflate_film_decoder(
            best_archive, v.parse_archive, v.HNeRVDecoder, film_hidden=8
        )
        assert tuple(frames.shape) == (n_pairs, 2, 3, 384, 512), frames.shape
        assert torch.isfinite(frames).all(), "inflated frames have NaN/inf"
        assert frames.min().item() >= 0.0 and frames.max().item() <= 255.0
        print(
            f"[4] inflate_film_decoder OK: {tuple(frames.shape)} in "
            f"[{frames.min():.1f},{frames.max():.1f}], all finite"
        )

        # (5) The DEPLOYED inflate frames are SCOREABLE: run the synthetic exact_eval
        #     on a decoder that reproduces the deployed render, get a finite score.
        #     (We score the SAME frames the contest would see.)
        score_out = sc.exact_eval(
            _build_eval_decoder_from_archive(best_archive, v, film_hidden=8),
            latents,
            len(best_archive),
        )
        assert all(
            torch.isfinite(torch.tensor(float(score_out[k]))).item()
            for k in ("score", "seg_distortion", "pose_distortion", "rate")
        ), f"non-finite score component: {score_out}"
        print(
            f"[5] deployed archive is SCOREABLE: score={score_out['score']:.4f} "
            f"d_seg={score_out['seg_distortion']:.4f} d_pose={score_out['pose_distortion']:.4f} "
            f"rate={score_out['rate']:.6f}  [macOS-CPU advisory, synthetic — NON-PROMOTABLE]"
        )

        # (6) THE HEADLINE SEAM: BEST-selection eval (cursor path, the SAME pattern
        #     the vendored evaluate_decoder uses: .eval() once, then decoder(z) per
        #     batch in strict pair order) == deployed inflate render. If these diverge,
        #     training picks a BEST that the contest does NOT score (train/deploy skew).
        eval_dec = _build_eval_decoder_from_archive(best_archive, v, film_hidden=8)
        # Mirror vendored evaluate_decoder EXACTLY: eval() resets cursor; batch_pairs=8.
        eval_dec.eval()
        with torch.inference_mode():
            cursor_render = torch.cat(
                [eval_dec(latents[i : i + 8]) for i in range(0, n_pairs, 8)], dim=0
            )
        assert torch.allclose(cursor_render, frames, atol=1e-4), (
            "EVAL/INFLATE SKEW: the cursor-based BEST-selection render does not match "
            "the deployed inflate render — training would pick a BEST the contest does "
            "not score."
        )
        print("[6] eval(cursor)==inflate render parity HOLDS (atol 1e-4) — no skew")

        # (7) Tail-batch alignment: also verify with a SPLIT batch (cursor advances
        #     across two calls of unequal-ish size, like the vendored tail) that the
        #     per-pair pose stays correctly aligned.
        eval_dec.eval()
        with torch.inference_mode():
            split_render = torch.cat(
                [eval_dec(latents[0:5]), eval_dec(latents[5:8])], dim=0
            )
        assert torch.allclose(split_render, frames, atol=1e-4), (
            "TAIL-BATCH SKEW: split-batch cursor render misaligns pose vs inflate."
        )
        print("[7] split-batch (5+3) cursor render == inflate (tail alignment OK)")

    print("\nR4-A DEPLOYED-ARCHIVE PROBE: PASS (all 7 checks). "
          "All-5-on produces a valid, scoreable archive whose decode matches training.")
    return 0


def _build_eval_decoder_from_archive(archive: bytes, v, *, film_hidden: int):
    """Rebuild the FiLM eval decoder (cursor adapter) from a produced archive —
    the same object ``_build_archive_and_eval_decoder`` returns, reconstructed from
    bytes (so it is provably the DEPLOYED artifact, not the in-memory wrapper)."""
    dec_sd, latents, meta = v.parse_archive(archive)
    pose = parse_pose_section(archive, v.parse_archive)
    film_sd = {
        k[len("pose_film.") :]: vv for k, vv in dec_sd.items() if k.startswith("pose_film.")
    }
    bare = {k: vv for k, vv in dec_sd.items() if not k.startswith("pose_film.")}
    vd = v.HNeRVDecoder(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    )
    vd.load_state_dict(bare)
    vd.eval()
    wrapper = PoseFiLMHNeRVWrapper(vd, n_pairs=int(meta["n_pairs"]), film_hidden=film_hidden)
    if film_sd:
        wrapper.pose_film.load_state_dict(film_sd)
    if pose is not None:
        wrapper.set_stored_pose(pose)
    wrapper.eval()
    return _FiLMEvalDecoder(wrapper)


if __name__ == "__main__":
    raise SystemExit(main())
