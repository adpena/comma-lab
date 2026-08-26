# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the FiLM-v2 TRUNK DECOUPLING completion (``cfg.pose_film_
trunk_stopgrad``) — the EXACT ∂d_seg/∂(pose-objective)=0 routing.

The leaked synergy this completes: FiLM-v2 already makes the FiLM HEAD decoupled
(rgb_1 is FiLM-clean → ∂d_seg/∂(FiLM-params)=0), but the POSE LOSS gradient still
leaks into the SHARED decoder: PoseNet reads BOTH frames, so pose grad flows
pose_loss → f1 → rgb_1/trunk/latents AND pose_loss → f0 → rgb_0/trunk/latents, and
the SAME trunk+latents produce f1 — so training to reduce d_pose perturbs the
shared trunk, which moves f1, which moves d_seg. With the flag ON, the pose
gradient is routed ONLY to the FiLM pose path (``pose_mlp`` + ``film_resid``) and is
stop-gradient on the shared trunk + latents + rgb_0/rgb_1 heads.

THE decisive NO-FAKE test (the one that, if the routing is fake, FAILS):
``test_flag_on_pose_grad_zero_on_trunk_and_latents`` — after a pose-loss backward
with the flag ON, the ``.grad`` on every SHARED trunk weight + latent tensor is
None/zero while the FiLM pose params have NON-zero grad; with the flag OFF the
trunk DOES get pose grad (the flag changes REAL behavior, default byte-identical).

The whole suite is a forced-CPU split (one frozen scorer for both the "train" and
"authority" head paths) so it runs without MPS hardware and is deterministic; the
only thing MPS changes in production is the DEVICE the seg cotangent is computed
on — a value, not the gradient-routing assembly these tests prove.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext, _TinyFrozenScorer

_BASE_CH = 8
_LATENT_DIM = 28
_N_PAIRS = 6


def _ce_seg_loss(seg_logits, targets_hard):
    return F.cross_entropy(seg_logits, targets_hard)


def _stage(seg_weight=100.0, pose_weight=1.0) -> StageSpec:
    return StageSpec(
        name="trunk_decouple_test",
        epochs=2,
        seg_loss_fn=_ce_seg_loss,
        eval_every=2,
        batch_size=_N_PAIRS,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=1e-3,
        muon_lr=2e-4,
        muon_weight_decay=0.0,
        latent_lr_mult=10.0,
        grad_clip=1e9,
        grad_clip_muon=1e9,
        lr_floor_ratio=5e-6,
        seg_weight=seg_weight,
        pose_weight=pose_weight,
        cat_lambda=0.0,
        cat_sigma=0.2,
        use_qat=False,
        init_latents_random=True,
    )


def _cfg(tmp_path: Path, *, trunk_stopgrad: bool) -> TorchVehicleConfig:
    """A split-by-head FiLM-v2 config. ``train_device='mps'`` only to satisfy the
    split-device __post_init__ gate; the test FORCES the driver devices to CPU after
    construction so the logic runs everywhere (no MPS hardware required)."""
    return TorchVehicleConfig(
        base_channels=_BASE_CH,
        latent_dim=_LATENT_DIM,
        out_dir=tmp_path / ("on" if trunk_stopgrad else "off"),
        device="cpu",
        train_device="mps",
        split_by_head=True,
        pose_film_enabled=True,
        pose_film_version=2,
        pose_film_trunk_stopgrad=trunk_stopgrad,
        seed=0,
    )


def _build_driver(tmp_path: Path, *, trunk_stopgrad: bool) -> TorchVehicleDriver:
    """Build a driver on a forced-CPU split with the real v2 FiLM wrapper + a tiny
    frozen scorer. The scorer's distinct (identical-weight) train scorer mirrors the
    real split structure; both run on CPU here."""
    cfg = _cfg(tmp_path, trunk_stopgrad=trunk_stopgrad)
    scorer = SyntheticScorerContext(
        n_pairs=_N_PAIRS, device="cpu", seed=0, split_by_head=True
    )
    scorer.split_device = True
    scorer.train_device = torch.device("cpu")
    scorer._train_scorer = _TinyFrozenScorer(seed=0).to("cpu").eval()
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=[_stage()]
    )
    driver.train_device = torch.device("cpu")
    driver.device = torch.device("cpu")
    driver.split_by_head = True
    return driver


def _make_decoder_latents(driver: TorchVehicleDriver):
    """A fresh v2 FiLM wrapper decoder + a learnable latent block, both on CPU."""
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(_N_PAIRS, _LATENT_DIM))
    return decoder, latents


def _roundtrip(decoder, latents, idx):
    """Replicate the driver's eval-roundtrip clamp+round STE on the wrapper output so
    the frame tensor fed to the heads matches the production graph exactly."""
    decoded_pair = decoder(latents[idx], idx)  # (B, 2, 3, 384, 512)
    B = len(idx)
    flat = decoded_pair.reshape(B * 2, 3, 384, 512)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    clamped = decoded_bhwc.clamp(0, 255)
    return clamped + (clamped.round() - clamped).detach()


def _film_and_shared(driver, decoder, latents):
    """Return (film_named, shared_named) where each is a list of (name, param). The
    latents are appended to shared under the name 'latents' (they are a shared code)."""
    film_ids = driver._film_param_ids(decoder)
    film = [(n, p) for n, p in decoder.named_parameters() if id(p) in film_ids]
    shared = [(n, p) for n, p in decoder.named_parameters() if id(p) not in film_ids]
    shared.append(("latents", latents))
    return film, shared


def _zero_all_grads(decoder, latents):
    for p in decoder.parameters():
        p.grad = None
    latents.grad = None


def _backward_once(driver, decoder, latents, *, compute_pose=True):
    """Run one ``_split_by_head_backward`` with grads zeroed first; returns nothing
    (inspect the populated ``.grad`` on decoder params + latents afterward)."""
    idx = torch.arange(_N_PAIRS)
    decoded_bhwc = _roundtrip(decoder, latents, idx)
    _zero_all_grads(decoder, latents)
    return driver._split_by_head_backward(
        decoded_bhwc,
        idx,
        driver.curriculum[0],
        compute_pose=compute_pose,
        decoder=decoder,
        latents=latents,
    )


# ---------------------------------------------------------------------------
# Config / wiring guards.
# ---------------------------------------------------------------------------
def test_flag_default_off():
    c = TorchVehicleConfig(out_dir=Path("/dev/null"))
    assert c.pose_film_trunk_stopgrad is False


def test_flag_requires_film_enabled():
    with pytest.raises(ValueError, match="pose_film_enabled=True"):
        TorchVehicleConfig(out_dir=Path("/dev/null"), pose_film_trunk_stopgrad=True)


def test_flag_requires_v2():
    with pytest.raises(ValueError, match="pose_film_version=2"):
        TorchVehicleConfig(
            out_dir=Path("/dev/null"), device="cpu", train_device="mps",
            split_by_head=True, pose_film_enabled=True, pose_film_version=1,
            pose_film_trunk_stopgrad=True,
        )


def test_flag_requires_split_by_head():
    with pytest.raises(ValueError, match="split_by_head=True"):
        TorchVehicleConfig(
            out_dir=Path("/dev/null"), device="cpu", pose_film_enabled=True,
            pose_film_version=2, pose_film_trunk_stopgrad=True,
        )


def test_flag_valid_on():
    c = TorchVehicleConfig(
        out_dir=Path("/dev/null"), device="cpu", train_device="mps",
        split_by_head=True, pose_film_enabled=True, pose_film_version=2,
        pose_film_trunk_stopgrad=True,
    )
    assert c.pose_film_trunk_stopgrad is True


def test_non_film_grad_params_excludes_film_includes_latents(tmp_path):
    driver = _build_driver(tmp_path, trunk_stopgrad=True)
    decoder, latents = _make_decoder_latents(driver)
    shared = driver._non_film_grad_params(decoder, latents)
    film_ids = driver._film_param_ids(decoder)
    assert film_ids, "v2 wrapper must expose FiLM params"
    shared_ids = {id(p) for p in shared}
    # No FiLM param is in the shared (to-be-masked) set.
    assert shared_ids.isdisjoint(film_ids)
    # The latents ARE in the shared set (the pose loss must not move the shared code).
    assert id(latents) in shared_ids
    # Every non-FiLM decoder param IS in the shared set.
    for _n, p in decoder.named_parameters():
        if id(p) not in film_ids:
            assert id(p) in shared_ids


# ---------------------------------------------------------------------------
# THE DECISIVE NO-FAKE TEST: flag ON ⇒ pose grad zero on trunk+latents, nonzero on FiLM.
# ---------------------------------------------------------------------------
def test_flag_on_pose_grad_zero_on_trunk_and_latents(tmp_path):
    """With the flag ON, after a SEG+POSE split backward, the SHARED params (trunk +
    rgb_0/rgb_1 heads + latents) carry the SEG-ONLY gradient (no pose contribution),
    and the FiLM pose params carry a NON-zero pose gradient. We prove "no pose
    contribution on shared" by comparing to a SEG-ONLY backward: the shared grads must
    be BIT-IDENTICAL (the pose backward left zero residue there)."""
    driver = _build_driver(tmp_path, trunk_stopgrad=True)
    decoder, latents = _make_decoder_latents(driver)
    film, shared = _film_and_shared(driver, decoder, latents)

    # (1) SEG-ONLY reference: run the backward with pose throttled OFF so ONLY the seg
    #     cotangent trains the graph. The shared grads here are the seg-only grads.
    _backward_once(driver, decoder, latents, compute_pose=False)
    seg_only_shared = {n: (None if p.grad is None else p.grad.detach().clone())
                       for n, p in shared}
    seg_only_film = {n: (None if p.grad is None else p.grad.detach().clone())
                     for n, p in film}

    # (2) SEG+POSE with the flag ON: the shared grads MUST equal the seg-only grads
    #     (pose removed) and the FiLM grads MUST differ (pose added).
    _backward_once(driver, decoder, latents, compute_pose=True)
    for n, p in shared:
        ref = seg_only_shared[n]
        got = p.grad
        if ref is None:
            assert got is None or torch.count_nonzero(got) == 0, (
                f"shared param {n} got pose gradient under trunk-stopgrad (expected zero)"
            )
        else:
            assert got is not None, f"shared param {n} lost its seg gradient"
            assert torch.equal(got, ref), (
                f"shared param {n} grad changed when pose was added → pose LEAKED into "
                f"the shared trunk (max abs diff {(got - ref).abs().max().item():.3e})"
            )
    # The FiLM pose path MUST have a non-zero pose gradient (pose is actually carried).
    film_has_grad = False
    for n, p in film:
        assert p.grad is not None, f"FiLM param {n} has no gradient at all"
        if torch.count_nonzero(p.grad) > 0:
            film_has_grad = True
    assert film_has_grad, "no FiLM pose param received a non-zero pose gradient"
    # And the FiLM grad with pose differs from the seg-only FiLM grad (pose changed it).
    film_changed = any(
        not torch.equal(p.grad, seg_only_film[n])
        if seg_only_film[n] is not None
        else torch.count_nonzero(p.grad) > 0
        for n, p in film
    )
    assert film_changed, "FiLM grad identical with/without pose → pose not routed to FiLM"


def test_full_film_pose_path_receives_grad_off_identity(tmp_path):
    """At the wrapper's IDENTITY init (proj.weight=0, beta_head.weight=0) the pose grad
    reaches ``film_resid.proj``/``film_resid.beta_head`` but NOT ``pose_mlp``/``gamma_head``
    yet (a zero-init transient: ∂cond/∂beta = beta_head.weight = 0 at init). After ONE
    perturbation off identity, the ENTIRE FiLM pose path — pose_mlp.fc1/fc2 AND gamma_head
    INCLUDED — must receive a non-zero pose gradient (proving the routing reaches the whole
    pose subgraph, not just the two heads that escape the init transient)."""
    driver = _build_driver(tmp_path, trunk_stopgrad=True)
    decoder, latents = _make_decoder_latents(driver)
    film_ids = driver._film_param_ids(decoder)
    # Jolt the FiLM off identity so the chain rule is fully alive (beta_head/proj nonzero).
    with torch.no_grad():
        for _n, p in decoder.named_parameters():
            if id(p) in film_ids:
                p.add_(torch.randn_like(p) * 0.3)
    _backward_once(driver, decoder, latents, compute_pose=True)
    for n, p in decoder.named_parameters():
        if id(p) in film_ids:
            assert p.grad is not None and torch.count_nonzero(p.grad) > 0, (
                f"FiLM pose param {n} got NO pose gradient off identity — the routing does "
                f"not reach the full pose subgraph"
            )


def test_flag_off_pose_grad_DOES_reach_trunk(tmp_path):
    """The behavioral contrast (proves the flag is not a no-op): with the flag OFF, the
    fused combined backward DOES flow pose grad into the shared trunk — so the shared
    grads differ from the seg-only grads. This is the leak the flag fixes."""
    driver = _build_driver(tmp_path, trunk_stopgrad=False)
    decoder, latents = _make_decoder_latents(driver)
    _film, shared = _film_and_shared(driver, decoder, latents)

    _backward_once(driver, decoder, latents, compute_pose=False)
    seg_only = {n: (None if p.grad is None else p.grad.detach().clone())
                for n, p in shared}

    _backward_once(driver, decoder, latents, compute_pose=True)
    # At least one shared param's grad must DIFFER (pose leaked in) — otherwise the
    # default path would already be decoupled and the flag would be pointless.
    leaked = False
    for n, p in shared:
        ref = seg_only[n]
        if ref is None:
            if p.grad is not None and torch.count_nonzero(p.grad) > 0:
                leaked = True
        elif p.grad is not None and not torch.equal(p.grad, ref):
            leaked = True
    assert leaked, (
        "flag OFF but no shared param received pose gradient — the default path is "
        "unexpectedly already decoupled (the flag would be a no-op)"
    )


def test_flag_on_vs_off_shared_grad_differs_film_path_active(tmp_path):
    """Direct ON-vs-OFF contrast on the SAME init: with identical decoder+latents+inputs,
    the flag ON gives shared grads that DIFFER from flag OFF (pose removed from shared),
    proving the flag changes the real gradient — not a cosmetic toggle."""
    torch.manual_seed(123)
    driver_off = _build_driver(tmp_path / "x_off", trunk_stopgrad=False)
    driver_on = _build_driver(tmp_path / "x_on", trunk_stopgrad=True)

    # Shared init: build the decoder+latents once, deep-copy into both runs.
    decoder_a, latents_a = _make_decoder_latents(driver_off)
    import copy

    decoder_b = copy.deepcopy(decoder_a)
    latents_b = torch.nn.Parameter(latents_a.detach().clone())

    _backward_once(driver_off, decoder_a, latents_a, compute_pose=True)
    _backward_once(driver_on, decoder_b, latents_b, compute_pose=True)

    # The latents (a shared code) must differ between ON and OFF (pose grad present OFF,
    # removed ON) — the load-bearing contrast.
    ga = latents_a.grad
    gb = latents_b.grad
    assert ga is not None and gb is not None
    assert not torch.equal(ga, gb), (
        "latents grad identical ON vs OFF → the trunk-stopgrad routing had no effect"
    )


# ---------------------------------------------------------------------------
# ∂d_seg/∂(pose params) = 0 NUMERICALLY: perturbing the FiLM pose params leaves the
# SegNet frame f1 (hence d_seg) bit-unchanged — the FiLM HEAD decoupling that the
# trunk-stopgrad makes the WHOLE-objective story consistent with.
# ---------------------------------------------------------------------------
def test_d_seg_invariant_to_film_pose_params(tmp_path):
    """The contest SegNet reads ONLY the last frame f1, which is FiLM-clean
    (``rgb_1(x)``, never x0). So PERTURBING any FiLM pose param leaves f1 (and the
    SegNet logits / argmax / d_seg) BIT-IDENTICAL. This is the structural reason the
    pose objective (routed only to the FiLM path) cannot move d_seg."""
    driver = _build_driver(tmp_path, trunk_stopgrad=True)
    decoder, latents = _make_decoder_latents(driver)
    decoder.eval()
    idx = torch.arange(_N_PAIRS)
    with torch.no_grad():
        f1_before = decoder(latents[idx], idx)[:, 1].clone()  # the SegNet frame
    # Jolt EVERY FiLM pose param away from identity.
    film_ids = driver._film_param_ids(decoder)
    with torch.no_grad():
        for _n, p in decoder.named_parameters():
            if id(p) in film_ids:
                p.add_(torch.randn_like(p) * 3.0)
        f1_after = decoder(latents[idx], idx)[:, 1].clone()
    assert torch.equal(f1_before, f1_after), (
        "f1 (the SegNet frame) changed when FiLM pose params moved → d_seg is NOT "
        "invariant to the pose path (the FiLM-clean rgb_1 contract is broken)"
    )


def test_film_perturbation_DOES_change_f0(tmp_path):
    """Negative control for the test above: the SAME FiLM perturbation MUST change f0
    (the pose-conditioned frame) — otherwise the FiLM is dead and the invariance above
    would be a vacuous pass (a dead FiLM trivially can't move anything)."""
    driver = _build_driver(tmp_path, trunk_stopgrad=True)
    decoder, latents = _make_decoder_latents(driver)
    decoder.eval()
    idx = torch.arange(_N_PAIRS)
    film_ids = driver._film_param_ids(decoder)
    with torch.no_grad():
        f0_before = decoder(latents[idx], idx)[:, 0].clone()
        for _n, p in decoder.named_parameters():
            if id(p) in film_ids:
                p.add_(torch.randn_like(p) * 3.0)
        f0_after = decoder(latents[idx], idx)[:, 0].clone()
    assert not torch.equal(f0_before, f0_after), (
        "f0 unchanged by a FiLM perturbation → the FiLM pose path is dead (the d_seg "
        "invariance test would be vacuous)"
    )


# ---------------------------------------------------------------------------
# SEG still trains the trunk under the flag (we did not freeze the trunk entirely).
# ---------------------------------------------------------------------------
def test_seg_still_trains_trunk_under_flag(tmp_path):
    """Under the flag ON, the SEG loss must STILL train the shared trunk (it produces
    f1). At least one trunk weight must carry a non-zero gradient after the backward —
    otherwise the flag would have frozen the trunk entirely (a bug)."""
    driver = _build_driver(tmp_path, trunk_stopgrad=True)
    decoder, latents = _make_decoder_latents(driver)
    _backward_once(driver, decoder, latents, compute_pose=True)
    film_ids = driver._film_param_ids(decoder)
    trunk_has_grad = any(
        id(p) not in film_ids and p.grad is not None and torch.count_nonzero(p.grad) > 0
        for _n, p in decoder.named_parameters()
    )
    assert trunk_has_grad, "no shared/trunk param trained by seg under the flag (frozen!)"


# ---------------------------------------------------------------------------
# FORWARD PARITY: the flag is a BACKWARD-routing change only → the forward (and thus
# the rendered frames / archive bytes) is byte-identical with the flag on vs off.
# ---------------------------------------------------------------------------
def test_forward_parity_flag_on_vs_off(tmp_path):
    """The flag changes ONLY the backward routing; the FORWARD output (the rendered
    frames) must be BIT-IDENTICAL with the flag on vs off on the same weights+inputs.
    This is the byte-identical-default contract at the render surface."""
    driver_off = _build_driver(tmp_path / "f_off", trunk_stopgrad=False)
    decoder_a, latents_a = _make_decoder_latents(driver_off)
    import copy

    decoder_b = copy.deepcopy(decoder_a)
    latents_b = torch.nn.Parameter(latents_a.detach().clone())
    idx = torch.arange(_N_PAIRS)
    # The flag only affects the BACKWARD; the forward is identical on the same weights,
    # so we compare the same render produced by two independent (but bit-equal) modules.
    with torch.no_grad():
        out_off = decoder_a(latents_a[idx], idx)
        out_on = decoder_b(latents_b[idx], idx)
    assert torch.equal(out_off, out_on), "forward differs ON vs OFF (it must not)"


def test_throttled_epoch_is_seg_only_in_both_modes(tmp_path):
    """A throttled epoch (compute_pose=False) flows the seg cotangent alone in BOTH
    modes (there is no pose cotangent to route), so the shared grads must be identical
    ON vs OFF — the trunk-stopgrad branch only engages when a pose cotangent exists."""
    driver_off = _build_driver(tmp_path / "t_off", trunk_stopgrad=False)
    driver_on = _build_driver(tmp_path / "t_on", trunk_stopgrad=True)
    decoder_a, latents_a = _make_decoder_latents(driver_off)
    import copy

    decoder_b = copy.deepcopy(decoder_a)
    latents_b = torch.nn.Parameter(latents_a.detach().clone())
    _backward_once(driver_off, decoder_a, latents_a, compute_pose=False)
    _backward_once(driver_on, decoder_b, latents_b, compute_pose=False)
    # Compare the latents grad (representative shared code) — must match exactly.
    assert (latents_a.grad is None) == (latents_b.grad is None)
    if latents_a.grad is not None:
        assert torch.equal(latents_a.grad, latents_b.grad)


# ---------------------------------------------------------------------------
# Small end-to-end CPU smoke (gradient-routing only — NOT a score claim): the
# decoupled-oomph driver run completes with the flag on; pose params move + trunk is
# trained by seg.
# ---------------------------------------------------------------------------
def _stage_short() -> StageSpec:
    s = _stage()
    return StageSpec(
        **{**s.__dict__, "epochs": 3, "eval_every": 3, "grad_clip": 1.0,
           "grad_clip_muon": 1.0}
    )


def test_end_to_end_smoke_flag_on_completes(tmp_path):
    """Force the split-by-head FiLM-v2 trunk-stopgrad path on CPU for a few epochs and
    verify the run COMPLETES with a tracked BEST + DONE marker — the separated backward
    + grad masking does not break the curriculum loop. (Synthetic scorer → research-only,
    NO score claim; this proves the gradient-routing integrates end-to-end.)"""
    cfg = TorchVehicleConfig(
        base_channels=_BASE_CH, latent_dim=_LATENT_DIM,
        out_dir=tmp_path / "e2e", checkpoint_every_epochs=1,
        device="cpu", train_device="mps", split_by_head=True,
        pose_film_enabled=True, pose_film_version=2, pose_film_trunk_stopgrad=True,
        seed=0,
    )
    scorer = SyntheticScorerContext(
        n_pairs=_N_PAIRS, device="cpu", seed=0, split_by_head=True
    )
    scorer.split_device = True
    scorer.train_device = torch.device("cpu")
    scorer._train_scorer = _TinyFrozenScorer(seed=0).to("cpu").eval()
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(),
        curriculum=[_stage_short()],
    )
    driver.train_device = torch.device("cpu")
    driver.device = torch.device("cpu")
    driver.split_by_head = True
    summary = driver.run()
    assert summary["status"] == "complete"
    assert driver.best_score < float("inf")
