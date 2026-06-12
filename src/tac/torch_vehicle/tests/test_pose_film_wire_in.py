# SPDX-License-Identifier: MIT
"""NO-FAKE tests for Lever 3 — the pose-FiLM (Quantizr STORE-pose) wire-in on the
torch-vehicle driver (``tac.torch_vehicle.pose_film`` + ``driver``).

The load-bearing claims (each, if wrong, breaks the lever A/B or — worse —
silently changes the LIVE base_ch=20 basin if it crash-resumes onto this code):

1. **DEFAULT byte-identity (basin-resume safety).** ``cfg.pose_film_enabled=False``
   (the default) makes the driver build the VENDORED decoder unchanged and add NO
   pose section — the archive bytes + the decoder output are bit-for-bit what the
   legacy vendored path produced. Proved by comparing the driver's default
   archive-build against a DIRECT ``v.build_archive`` call (the legacy path) →
   identical bytes, and by FiLM-off run-to-run determinism.

2. **IDENTITY-AT-INIT.** With FiLM ON, the FIRST-step FiLM render (zero-init fc2,
   gamma=1/beta=0) equals the no-FiLM decoder bit-for-bit — so enabling FiLM does
   not perturb the basin until the FiLM weights actually train.

3. **TRAINED FiLM changes the d_pose-relevant output (NO-FAKE behavior).** A
   non-identity FiLM actually modulates the rendered frames (differs from the
   no-FiLM render) and is differentiable to the FiLM weights — so it is a real
   operational mechanism, not a metadata stub.

4. **POSE-SECTION ROUND-TRIP.** ``build_archive_with_pose`` → ``parse_pose_section``
   reconstructs the stored pose (within the uint8 min/max quant step); a FiLM-off
   archive returns ``None`` (no pose section). The vendored ``parse_archive`` still
   reads its 3 sections on the additive archive (the additive grammar is safe).

5. **BYTE-CLOSED INFLATE.** ``inflate_film_decoder`` parses the additive archive,
   rebuilds the FiLM wrapper from the int8-dequantized weights + the parsed pose,
   and renders FiLM-conditioned frames deterministically — and those frames DIFFER
   from the no-FiLM inflate (FiLM is active through the byte-closed round-trip).

All asserts are real behavioral checks (no constant-checking).
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
    build_archive_with_pose,
    decode_pose_section,
    encode_pose_section,
    inflate_film_decoder,
    parse_pose_section,
    stored_pose_bytes,
    wrapper_sd_to_archive_decoder_sd,
)


def _ce(s, t):
    return torch.nn.functional.cross_entropy(s, t)


def _spec(epochs: int = 2) -> StageSpec:
    return StageSpec(
        name="ce", epochs=epochs, seg_loss_fn=_ce, eval_every=1, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0, cat_lambda=0.0,
        cat_sigma=0.2, use_qat=False, init_latents_random=True,
    )


def _wrapper(base_ch: int = 8, n_pairs: int = 6, *, train_film: bool = False, seed: int = 0):
    v = import_vendored_bundle()
    dec = v.HNeRVDecoder(latent_dim=28, base_channels=base_ch).eval()
    w = PoseFiLMHNeRVWrapper(dec, n_pairs=n_pairs, film_hidden=8)
    if train_film:
        g = torch.Generator().manual_seed(seed)
        for p in w.pose_film.parameters():
            p.data = p.data + torch.randn(p.shape, generator=g) * 0.2
    g2 = torch.Generator().manual_seed(seed + 1)
    pose = torch.randn(n_pairs, 6, generator=g2) * 0.5
    w.set_stored_pose(pose)
    return v, dec, w, pose


# ---------------------------------------------------------------------------
# CLAIM 2 — IDENTITY AT INIT.
# ---------------------------------------------------------------------------
def test_film_identity_at_init_matches_no_film_render():
    """Zero-init FiLM (gamma=1/beta=0) => first-step FiLM render == no-FiLM render,
    bit-for-bit. This is the basin-non-perturbation contract for enabling FiLM."""
    _v, dec, w, _pose = _wrapper(train_film=False)
    z = torch.randn(3, 28) * 0.1
    idx = torch.arange(3)
    out_film = w(z, idx)
    out_nofilm = w(z, None)
    assert torch.equal(out_film, out_nofilm), "FiLM is not identity at init"
    # And equals the bare vendored decoder forward exactly.
    out_vendored = dec(z)
    assert torch.equal(out_film, out_vendored), "FiLM-on @ init != vendored decoder"


def test_film_wrapper_does_not_mutate_vendored_decoder_params():
    """The wrapper holds the vendored decoder by reference but adds only the FiLM
    sub-module; the vendored decoder params are unchanged objects (no fork)."""
    _v, dec, w, _pose = _wrapper()
    # The vendored decoder's params are reachable via the wrapper unchanged.
    assert w.decoder is dec
    film_param_ids = {id(p) for p in w.pose_film.parameters()}
    dec_param_ids = {id(p) for p in dec.parameters()}
    assert film_param_ids.isdisjoint(dec_param_ids)


# ---------------------------------------------------------------------------
# CLAIM 3 — TRAINED FiLM is a REAL operational mechanism (NO-FAKE).
# ---------------------------------------------------------------------------
def test_trained_film_changes_the_rendered_output():
    """A non-identity FiLM actually modulates the rendered frames (differs from the
    no-FiLM render) — proving FiLM is operational, not a metadata stub."""
    _v, _dec, w, _pose = _wrapper(train_film=True, seed=1)
    z = torch.randn(4, 28) * 0.1
    idx = torch.arange(4)
    out_film = w(z, idx)
    out_nofilm = w(z, None)
    assert not torch.allclose(out_film, out_nofilm, atol=1e-3), (
        "trained FiLM did not change the render (FAKE-mechanism guard)"
    )


def test_film_is_differentiable_to_the_film_weights():
    """FiLM gradient flows to fc1/fc2 (so training can descend it) — a constant /
    no-op FiLM would give zero/None grad on its own weights."""
    _v, _dec, w, _pose = _wrapper(train_film=True, seed=2)
    z = torch.randn(2, 28) * 0.1
    idx = torch.arange(2)
    out = w(z, idx)
    out.sum().backward()
    grads = [p.grad for p in w.pose_film.parameters() if p.grad is not None]
    assert grads, "FiLM weights received no gradient"
    total = sum(g.abs().sum().item() for g in grads)
    assert total > 0, "FiLM gradient is identically zero"


def test_film_conditions_on_the_per_pair_stored_pose():
    """Different stored poses for the SAME latents/index produce different frame1
    renders — proving the pose (not just the latents) drives the FiLM."""
    _v, _dec, w, _pose = _wrapper(train_film=True, seed=3)
    z = torch.randn(2, 28) * 0.1
    idx = torch.arange(2)
    out_a = w(z, idx).clone()
    # Swap the stored pose for these indices to a clearly different pose.
    new_pose = w.stored_pose.clone()
    new_pose[idx] = new_pose[idx] + 1.0
    w.set_stored_pose(new_pose)
    out_b = w(z, idx)
    assert not torch.allclose(out_a, out_b, atol=1e-3), (
        "render is invariant to the stored pose (FiLM is not pose-conditioned)"
    )


# ---------------------------------------------------------------------------
# CLAIM 4 — POSE-SECTION ROUND-TRIP + additive-grammar safety.
# ---------------------------------------------------------------------------
def test_pose_section_round_trip_within_quant_step():
    """encode_pose_section -> decode_pose_section reconstructs the pose within the
    uint8 per-dim min/max quant step (the codec is the latent codec, applied to
    pose)."""
    g = torch.Generator().manual_seed(7)
    pose = torch.randn(50, 6, generator=g) * 0.7
    sec = encode_pose_section(pose)
    pose_rt = decode_pose_section(sec)
    assert pose_rt.shape == pose.shape
    # quant step = (max-min)/254 per dim; the round-trip is within one step.
    step = (pose.max(dim=0).values - pose.min(dim=0).values) / 254.0
    max_err = (pose - pose_rt).abs().max(dim=0).values
    assert torch.all(max_err <= step + 1e-6), (
        f"pose round-trip error {max_err} exceeds quant step {step}"
    )


def test_additive_pose_archive_is_vendored_readable_and_pose_parseable():
    """The additive archive: (a) the vendored parse_archive still reads its 3
    sections (additive grammar is safe), (b) parse_pose_section reconstructs the
    stored pose, (c) a FiLM-OFF archive has NO pose section (None)."""
    v, _dec, w, pose = _wrapper(train_film=True, seed=4)
    latents = torch.randn(6, 28) * 0.1
    meta = {"n_pairs": 6, "latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]}
    dec_sd = wrapper_sd_to_archive_decoder_sd(w.state_dict())
    arch = build_archive_with_pose(v.build_archive, dec_sd, latents, meta, pose)

    # (a) vendored parse still reads 3 sections + the FiLM weights ride in the blob.
    dsd, _dlat, dmeta = v.parse_archive(arch)
    assert dmeta["base_channels"] == 8
    assert any(k.startswith("pose_film.") for k in dsd)

    # (b) pose section reconstructs (within quant step).
    pose_parsed = parse_pose_section(arch, v.parse_archive)
    assert pose_parsed is not None
    step = (pose.max(dim=0).values - pose.min(dim=0).values) / 254.0
    assert torch.all((pose - pose_parsed).abs().max(dim=0).values <= step + 1e-6)

    # (c) a vendored-only (FiLM-off) archive has no pose section.
    bare = {k[len("decoder."):]: vv for k, vv in w.state_dict().items()
            if k.startswith("decoder.")}
    arch_off = v.build_archive(bare, latents, meta)
    assert parse_pose_section(arch_off, v.parse_archive) is None


def test_stored_pose_bytes_is_the_real_encoded_size():
    """stored_pose_bytes equals len(encode_pose_section) — the REAL byte cost the
    archive pays (Catalog #304: the encode IS the empirical bit-spend proof)."""
    g = torch.Generator().manual_seed(9)
    pose = torch.randn(600, 6, generator=g) * 0.4
    assert stored_pose_bytes(pose) == len(encode_pose_section(pose))
    # The pose payload is small (~the memo's ~1 KB at 600 pairs).
    assert stored_pose_bytes(pose) < 5000


# ---------------------------------------------------------------------------
# CLAIM 5 — BYTE-CLOSED INFLATE round-trip.
# ---------------------------------------------------------------------------
def test_inflate_film_decoder_round_trip_is_deterministic_and_film_active():
    """The additive archive inflates to FiLM-conditioned frames deterministically,
    and those frames DIFFER from the no-FiLM inflate — FiLM survives the byte-closed
    int8 round-trip (the export-first proof)."""
    v, _dec, w, pose = _wrapper(train_film=True, seed=5)
    latents = torch.randn(6, 28) * 0.1
    meta = {"n_pairs": 6, "latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]}
    dec_sd = wrapper_sd_to_archive_decoder_sd(w.state_dict())
    arch = build_archive_with_pose(v.build_archive, dec_sd, latents, meta, pose)

    frames = inflate_film_decoder(arch, v.parse_archive, v.HNeRVDecoder, film_hidden=8)
    assert tuple(frames.shape) == (6, 2, 3, 384, 512)
    assert frames.min().item() >= 0.0 and frames.max().item() <= 255.0
    # Deterministic re-inflate.
    frames2 = inflate_film_decoder(arch, v.parse_archive, v.HNeRVDecoder, film_hidden=8)
    assert torch.equal(frames, frames2)

    # FiLM active through the round-trip: differs from inflating the SAME parse-back
    # decoder WITHOUT FiLM.
    dsd, dlat, _m = v.parse_archive(arch)
    dec_only = {k: vv for k, vv in dsd.items() if not k.startswith("pose_film.")}
    vd = v.HNeRVDecoder(latent_dim=28, base_channels=8)
    vd.load_state_dict(dec_only)
    vd.eval()
    with torch.inference_mode():
        nofilm = torch.cat([vd(dlat[i:i + 16]) for i in range(0, 6, 16)], dim=0)
    assert not torch.allclose(frames, nofilm, atol=1e-2), (
        "FiLM had no effect through the byte-closed inflate (round-trip lost the pose)"
    )


# ---------------------------------------------------------------------------
# CLAIM 1 — DEFAULT byte-identity (the basin-resume safety proof).
# ---------------------------------------------------------------------------
def test_driver_pose_film_off_builds_byte_identical_vendored_archive():
    """With cfg.pose_film_enabled=False the driver's archive-build path is the
    LEGACY vendored path bit-for-bit: ``_build_archive_and_eval_decoder`` returns
    EXACTLY ``v.build_archive(ema_sd, latents, meta)`` — no pose section, no FiLM
    keys. A basin resuming onto this code produces the IDENTICAL archive."""
    from tac.torch_vehicle.scorer_context import SyntheticScorerContext

    v = import_vendored_bundle()
    with tempfile.TemporaryDirectory() as td:
        cfg = TorchVehicleConfig(
            base_channels=8, latent_dim=28, out_dir=Path(td), checkpoint_every_epochs=1,
            device="cpu", seed=5, pose_film_enabled=False,
        )
        sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=5)
        drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_spec()])

        # Build a representative EMA-shadow state dict (a fresh vendored decoder).
        dec = v.HNeRVDecoder(latent_dim=28, base_channels=8)
        ema_sd = {k: t.detach().cpu().clone() for k, t in dec.state_dict().items()}
        latents = torch.randn(6, 28) * 0.1
        meta = {"n_pairs": 6, "latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]}

        archive, eval_dec, eval_latents = drv._build_archive_and_eval_decoder(
            ema_sd, latents, meta
        )
        legacy = v.build_archive(ema_sd, latents, meta_dict=meta)
        assert archive == legacy, "FiLM-off archive bytes differ from the legacy path"
        # The eval decoder is the plain vendored decoder (no FiLM wrapper).
        assert type(eval_dec).__name__ == "HNeRVDecoder"
        # parse-back has no pose section.
        assert parse_pose_section(archive, v.parse_archive) is None


def test_driver_pose_film_off_run_is_deterministic_and_unchanged():
    """Two FiLM-off runs from the same seed produce identical best-score AND
    identical best-archive bytes — the determinism the basin relies on."""
    from tac.torch_vehicle.scorer_context import SyntheticScorerContext

    v = import_vendored_bundle()

    def _run(td: str):
        cfg = TorchVehicleConfig(
            base_channels=8, latent_dim=28, out_dir=Path(td), checkpoint_every_epochs=1,
            device="cpu", seed=5, pose_film_enabled=False,
        )
        sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=5)
        drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_spec()])
        out = drv.run()
        arch = (Path(td) / "best" / "best_archive.bin").read_bytes()
        return out["best_score"], arch

    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        s1, a1 = _run(t1)
        s2, a2 = _run(t2)
    assert s1 == s2
    assert a1 == a2


def test_driver_pose_film_on_run_completes_and_writes_pose_section():
    """A FiLM-ON run trains end-to-end, the best archive carries the additive pose
    section, and it parses back — the operational mechanism through the full driver
    (build -> parse-back FiLM eval -> best archive with pose)."""
    from tac.torch_vehicle.scorer_context import SyntheticScorerContext

    v = import_vendored_bundle()
    with tempfile.TemporaryDirectory() as td:
        cfg = TorchVehicleConfig(
            base_channels=8, latent_dim=28, out_dir=Path(td), checkpoint_every_epochs=1,
            device="cpu", seed=5, pose_film_enabled=True,
        )
        sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=5)
        drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_spec()])
        out = drv.run()
        assert out["status"] == "complete"
        arch = (Path(td) / "best" / "best_archive.bin").read_bytes()
        pose = parse_pose_section(arch, v.parse_archive)
        assert pose is not None
        assert tuple(pose.shape) == (6, 6)


def test_driver_pose_film_on_eval_uses_film_conditioned_render():
    """The FiLM-on eval path (``_build_archive_and_eval_decoder``) returns a
    cursor-based FiLM eval adapter whose render is FiLM-conditioned — so the BEST
    selection scores the FiLM-conditioned archive (faithful to inflate), NOT the
    no-FiLM decoder."""
    from tac.torch_vehicle.pose_film import _FiLMEvalDecoder
    from tac.torch_vehicle.scorer_context import SyntheticScorerContext

    v = import_vendored_bundle()
    with tempfile.TemporaryDirectory() as td:
        cfg = TorchVehicleConfig(
            base_channels=8, latent_dim=28, out_dir=Path(td), checkpoint_every_epochs=1,
            device="cpu", seed=5, pose_film_enabled=True, pose_film_hidden=8,
        )
        sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=5)
        drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_spec()])

        # Build a NON-identity FiLM wrapper state dict as the EMA shadow.
        dec = v.HNeRVDecoder(latent_dim=28, base_channels=8)
        w = PoseFiLMHNeRVWrapper(dec, n_pairs=6, film_hidden=8)
        g = torch.Generator().manual_seed(11)
        for p in w.pose_film.parameters():
            p.data = p.data + torch.randn(p.shape, generator=g) * 0.2
        pose = torch.randn(6, 6, generator=torch.Generator().manual_seed(12)) * 0.5
        w.set_stored_pose(pose)
        ema_sd = {k: t.detach().cpu().clone() for k, t in w.state_dict().items()}
        latents = torch.randn(6, 28) * 0.1
        meta = {"n_pairs": 6, "latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]}

        archive, eval_dec, eval_latents = drv._build_archive_and_eval_decoder(
            ema_sd, latents, meta
        )
        assert isinstance(eval_dec, _FiLMEvalDecoder)
        # The cursor-based eval renders FiLM-conditioned frames matching inflate.
        eval_dec.eval()  # reset cursor to pair 0
        with torch.inference_mode():
            film_render = torch.cat(
                [eval_dec(eval_latents[i:i + 4]) for i in range(0, 6, 4)], dim=0
            )
        ref = inflate_film_decoder(archive, v.parse_archive, v.HNeRVDecoder, film_hidden=8)
        assert torch.allclose(film_render, ref, atol=1e-4), (
            "the FiLM eval render does not match the inflate render (eval/inflate skew)"
        )
