#!/usr/bin/env python
"""NO-FAKE behavior tests for the PR95 reverse-engineer/prune executor.

These verify BEHAVIOR (not constants): the inflate actually reconstructs the
229K-param bc36 decoder, the structured prune actually shrinks params AND loads
into a smaller decoder, the KD step actually lowers frame-MSE (distillation
runs), and the score math matches upstream/evaluate.py. Every test would FAIL
if the executor were replaced by canonical-marker stubs.

Run: .venv/bin/python -m pytest experiments/test_reverse_engineer_pr95_prune.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reverse_engineer_pr95_prune_executor as X  # noqa: E402


def _inflate():
    return X.inflate_pr95(X.PR95_ARCHIVE)


def test_inflate_reproduces_bc36_229k_decoder():
    decoder_sd, latents, meta = _inflate()
    assert meta["base_channels"] == 36
    dec = X.build_decoder(meta)
    dec.load_state_dict(decoder_sd)  # strict load => keys+shapes match exactly
    n = X.count_params(dec.state_dict())
    # converged PR95 hnerv_muon is ~229K params
    assert 220_000 < n < 235_000, n
    # latents are (n_pairs, latent_dim)
    assert latents.ndim == 2 and latents.shape[1] == meta["latent_dim"]
    assert latents.shape[0] == meta["n_pairs"]


def test_prune_actually_shrinks_and_loads():
    decoder_sd, latents, meta = _inflate()
    full_n = X.count_params(decoder_sd)
    for bc in (32, 28, 24, 20, 16):
        pruned_sd, kept = X.prune_decoder_state(decoder_sd, meta, bc)
        # strict load into the smaller decoder => prune produced consistent shapes
        pdec = X.build_decoder(meta, base_channels=bc)
        pdec.load_state_dict(pruned_sd)
        pn = X.count_params(pruned_sd)
        assert pn < full_n, (bc, pn, full_n)
        # monotone: smaller bc => fewer params
        if bc < 36:
            assert pn < full_n


def test_prune_is_monotone_in_params():
    decoder_sd, _, meta = _inflate()
    sizes = []
    for bc in (32, 28, 24, 20, 16):
        pruned_sd, _ = X.prune_decoder_state(decoder_sd, meta, bc)
        sizes.append(X.count_params(pruned_sd))
    assert sizes == sorted(sizes, reverse=True), sizes


def test_kd_frame_mse_actually_decreases():
    """The distillation must REALLY run: a few KD steps must lower the
    student-vs-teacher frame-MSE (not a constant)."""
    decoder_sd, latents_full, meta = _inflate()
    n_pairs = 6
    latents = latents_full[:n_pairs].clone().detach()
    teacher = X.build_decoder(meta).eval()
    teacher.load_state_dict(decoder_sd)
    for p in teacher.parameters():
        p.requires_grad_(False)
    pruned_sd, _ = X.prune_decoder_state(decoder_sd, meta, 28)
    student = X.build_decoder(meta, base_channels=28)
    student.load_state_dict(pruned_sd)
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3)

    losses = []
    for _ in range(8):
        with torch.no_grad():
            tf = teacher(latents)
        sf = student(latents)
        loss = X.kd_frame_mse(sf, tf)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0], losses  # distillation actually ran


def test_score_math_matches_upstream():
    from tac.contest_score import compute_contest_score

    d_seg, d_pose, b = 5.6e-4, 2.4e-5, 178_417
    expected = 100 * d_seg + math.sqrt(10 * d_pose) + 25 * b / 37_545_489
    got = compute_contest_score(d_seg, d_pose, b)
    assert abs(got - expected) < 1e-9, (got, expected)


def test_seg_aware_ce_term_is_differentiable_and_decreases():
    """The score-aware seg KD term must REALLY flow gradients: SegNet-CE between the
    student's last-frame seg-logits and the teacher's seg-argmax must decrease over a
    few steps (a real distillation of the argmax signal, not a constant)."""
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    decoder_sd, latents_full, meta = _inflate()
    n_pairs = 4
    latents = latents_full[:n_pairs].clone().detach()
    teacher = X.build_decoder(meta).eval()
    teacher.load_state_dict(decoder_sd)
    for p in teacher.parameters():
        p.requires_grad_(False)
    pruned_sd, _ = X.prune_decoder_state(decoder_sd, meta, 28)
    student = X.build_decoder(meta, base_channels=28)
    student.load_state_dict(pruned_sd)
    student.train()
    seg = SegNet().eval()
    seg.load_state_dict(load_file(segnet_sd_path, device="cpu"))
    for p in seg.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3)

    ces = []
    for _ in range(6):
        with torch.no_grad():
            tf = teacher(latents)
            t_seg = X._seg_logits_from_frames(seg, tf).argmax(dim=1)
        sf = student(latents)
        s_logits = X._seg_logits_from_frames(seg, sf)
        ce = torch.nn.functional.cross_entropy(s_logits, t_seg)
        # grad must reach the student through SegNet + render
        assert ce.requires_grad
        opt.zero_grad()
        ce.backward()
        opt.step()
        ces.append(float(ce.item()))
    assert ces[-1] < ces[0], ces  # seg-CE distillation actually ran


def test_render_uint8_roundtrip_shape():
    decoder_sd, latents_full, meta = _inflate()
    dec = X.build_decoder(meta).eval()
    dec.load_state_dict(decoder_sd)
    recon = X.render_recon_pairs(dec, latents_full[:2], meta, torch.device("cpu"))
    assert recon.shape == (2, X.SEQ_LEN, X.CAMERA_H, X.CAMERA_W, 3)
    assert recon.dtype == torch.uint8


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
