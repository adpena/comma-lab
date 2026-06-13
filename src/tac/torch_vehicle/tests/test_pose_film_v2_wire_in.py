# SPDX-License-Identifier: MIT
"""NO-FAKE driver wire-in tests for Lever-3 v2 (``pose_film_version=2``).

These assert the DRIVER touchpoints (not the v2 module, which has its own tests):
  * §A critical fix — the FiLM params (``pose_mlp`` + ``film_resid``) are EXCLUDED from
    Muon AND given a DEDICATED capped-LR AdamW group, while STILL being in the clip set
    (so a naive prefix-swap wire-in that routed FiLM through Muon at ``muon_lr`` — the
    bug the full-stack review caught — cannot regress in);
  * byte-identity DEFAULT-OFF — ``pose_film_version=2`` with ``pose_film_enabled=False``
    produces a BYTE-IDENTICAL archive to the vendored path (the version is inert when
    FiLM is off → the basin/control arm is unaffected);
  * the v2 export round-trips — a tiny v2 run builds an archive whose additive pose
    section parses to ``(n_pairs, 6)`` and whose ``inflate_film_decoder_v2`` renders the
    full pair tensor (export-first faithfulness);
  * the optimizer is byte-identical when FiLM is OFF (2 groups, no FiLM group).

Behavior, not constants (Slot EEE Class 2): every assertion would FAIL if the wire-in
reverted to routing FiLM through Muon / dropping the capped group / breaking byte-identity.
Authority: torch-CPU; synthetic scorer (the wire-in mechanics are scorer-agnostic).
"""

from __future__ import annotations

import copy

import pytest
import torch

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    _FILM_LR_CAP,
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _ce(s, t):
    return torch.nn.functional.cross_entropy(s, t)


def _spec(*, use_muon: bool, adamw_lr: float = 1e-2, epochs: int = 4) -> StageSpec:
    return StageSpec(
        name="t", epochs=epochs, seg_loss_fn=_ce, eval_every=1, batch_size=2,
        ema_decay=0.999, use_muon=use_muon, adamw_lr=adamw_lr, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0, cat_lambda=0.0,
        cat_sigma=0.2, use_qat=False, init_latents_random=True,
    )


def _driver(*, film: bool, version: int, out, n_pairs: int = 6, curriculum=None):
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=out, checkpoint_every_epochs=1,
        device="cpu", seed=0, pose_film_enabled=film, pose_film_hidden=8,
        pose_film_version=version,
    )
    return TorchVehicleDriver(
        cfg, scorer=SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=0),
        vendored=import_vendored_bundle(),
        curriculum=curriculum if curriculum is not None else [],
    )


# ---------------------------------------------------------------------------
# §A critical fix — FiLM excluded from Muon, capped AdamW group, in clip set
# ---------------------------------------------------------------------------
def _build_v2_runtime(use_muon: bool, adamw_lr: float = 1e-2):
    d = _driver(film=True, version=2, out="/tmp/_v2_wirein_unused")
    dec = d._new_decoder(torch.device("cpu"))
    film_ids = d._film_param_ids(dec)
    lat = torch.nn.Parameter(torch.randn(6, 28))
    rt = d._build_stage_runtime(
        _spec(use_muon=use_muon, adamw_lr=adamw_lr),
        decoder=dec, latents=lat, ema_decoder=None, ema_latents=None,
    )
    return d, dec, film_ids, rt


def test_v2_decoder_is_the_v2_wrapper():
    d = _driver(film=True, version=2, out="/tmp/_v2_wirein_unused")
    dec = d._new_decoder(torch.device("cpu"))
    assert type(dec).__name__ == "PoseFiLMHNeRVWrapperV2"
    # v2 FiLM submodules are pose_mlp + film_resid (NOT v1's single pose_film).
    names = {n.split(".")[0] for n, _ in dec.named_parameters()}
    assert "pose_mlp" in names and "film_resid" in names
    assert "pose_film" not in names


@pytest.mark.parametrize("use_muon", [True, False])
def test_v2_film_excluded_from_muon_and_capped_adamw_group(use_muon):
    """THE §A FIX: FiLM never trains under Muon; it has a dedicated capped AdamW group;
    and it is STILL in the clip set."""
    d, dec, film_ids, rt = _build_v2_runtime(use_muon=use_muon, adamw_lr=1e-2)
    assert len(film_ids) > 0
    # (a) FiLM NOT in the Muon partition.
    muon_ids = {id(p) for p in rt.muon_params}
    assert not (film_ids & muon_ids), "FiLM must NOT be optimized by Muon (the §A bug)"
    # (b) a dedicated AdamW group at the capped LR contains EXACTLY the FiLM params.
    cap = min(1e-2, _FILM_LR_CAP)
    film_groups = [g for g in rt.adamw_opt.param_groups if abs(g["lr"] - cap) < 1e-12]
    assert len(film_groups) == 1, "exactly one capped FiLM AdamW group expected"
    fg_ids = {id(p) for p in film_groups[0]["params"]}
    assert fg_ids == film_ids, "the capped group must be EXACTLY the FiLM params"
    # (c) FiLM is in the clip set (grad-clip covers the capped group too).
    clip_ids = {id(p) for p in rt.adamw_clip_params}
    assert film_ids <= clip_ids, "FiLM must be in the clip set"


def test_v2_film_group_lr_is_min_of_adamw_and_cap():
    """When the stage's adamw_lr is BELOW the cap, the FiLM group uses adamw_lr (no
    spurious raise); when ABOVE, it is capped to _FILM_LR_CAP."""
    # adamw_lr above cap -> capped
    _, _, _, rt_hi = _build_v2_runtime(use_muon=False, adamw_lr=1e-2)
    lrs_hi = sorted(g["lr"] for g in rt_hi.adamw_opt.param_groups)
    assert min(lrs_hi) == pytest.approx(_FILM_LR_CAP)
    # adamw_lr below cap -> FiLM group == adamw_lr (the smallest non-latent lr equals it)
    _, _, film_ids, rt_lo = _build_v2_runtime(use_muon=False, adamw_lr=5e-4)
    film_groups = [
        g for g in rt_lo.adamw_opt.param_groups
        if {id(p) for p in g["params"]} == film_ids
    ]
    assert len(film_groups) == 1
    assert film_groups[0]["lr"] == pytest.approx(5e-4)


# ---------------------------------------------------------------------------
# byte-identity DEFAULT-OFF + optimizer byte-identity OFF
# ---------------------------------------------------------------------------
def test_v2_film_off_optimizer_is_two_groups_byte_identical():
    """``pose_film_enabled=False`` (regardless of version) → NO FiLM group; the optimizer
    is the vendored 2-group structure (decoder + latents)."""
    d = _driver(film=False, version=2, out="/tmp/_v2_wirein_unused")
    dec = d._new_decoder(torch.device("cpu"))
    assert d._film_param_ids(dec) == set()  # no FiLM params when off
    lat = torch.nn.Parameter(torch.randn(6, 28))
    rt = d._build_stage_runtime(
        _spec(use_muon=False), decoder=dec, latents=lat, ema_decoder=None, ema_latents=None
    )
    assert len(rt.adamw_opt.param_groups) == 2  # decoder + latents, NO film group


def test_v2_archive_byte_identical_to_v1_when_film_off(tmp_path):
    """A run with ``pose_film_version=2`` + ``pose_film_enabled=False`` produces a
    BYTE-IDENTICAL best archive to ``pose_film_version=1`` + off (the version is inert
    when FiLM is off → the basin/control arm is unaffected by the v2 wire-in)."""
    cur = [_spec(use_muon=False, epochs=3)]
    d1 = _driver(film=False, version=1, out=tmp_path / "v1off", curriculum=copy.deepcopy(cur))
    d2 = _driver(film=False, version=2, out=tmp_path / "v2off", curriculum=copy.deepcopy(cur))
    d1.run()
    d2.run()
    a1 = (tmp_path / "v1off" / "best" / "best_archive.bin").read_bytes()
    a2 = (tmp_path / "v2off" / "best" / "best_archive.bin").read_bytes()
    assert a1 == a2, "version must be inert when FiLM is OFF (byte-identity broken)"


# ---------------------------------------------------------------------------
# v2 export round-trip (export-first faithfulness)
# ---------------------------------------------------------------------------
def test_v2_run_builds_archive_with_parseable_pose_section_and_inflates(tmp_path):
    """A tiny v2 run builds an archive whose additive pose section parses to (n_pairs,6)
    and whose v2 inflate renders the full pair tensor (the export contract)."""
    from tac.torch_vehicle.pose_film_v2 import inflate_film_decoder_v2, parse_pose_section

    n_pairs = 6
    d = _driver(
        film=True, version=2, out=tmp_path / "v2run", n_pairs=n_pairs,
        curriculum=[_spec(use_muon=False, epochs=3)],
    )
    d.run()
    arch = (tmp_path / "v2run" / "best" / "best_archive.bin").read_bytes()
    pose = parse_pose_section(arch, d.v.parse_archive)
    assert pose is not None and tuple(pose.shape) == (n_pairs, 6)
    frames = inflate_film_decoder_v2(arch, d.v.parse_archive, d.v.HNeRVDecoder, device="cpu")
    assert tuple(frames.shape) == (n_pairs, 2, 3, 384, 512)
    assert torch.isfinite(frames).all() and float(frames.min()) >= 0.0 and float(frames.max()) <= 255.0
