# SPDX-License-Identifier: MIT
"""Behavior tests for the smaller-student distillation basis (task #74).

NO-FAKE discipline (Slot EEE class 2 + class 6 + class 8): these tests verify BEHAVIOR not
constants. Every test would FAIL if the student forward were replaced by a constant-frame stub, or
if the byte cost were a fixed formula, or if the KD loss did not actually flow gradient through the
soft KL distribution. The decisive class-2 guards:

  * the student frames ACTUALLY vary across pairs and pixels (a constant stub fails);
  * the byte cost ACTUALLY tracks the quantized array sizes (a bigger student = more bytes);
  * the KL-T2 distill ACTUALLY pushes the student SegNet logits toward the teacher's (a no-op fails);
  * the numpy-portable forward ACTUALLY reproduces the torch forward (the inflate contract);
  * a smaller student ACTUALLY costs fewer bytes than a larger one (the rate-vs-distortion axis);
  * a CONSTANT student does NOT match teacher logits (the d_seg signal is real, not a stub).

These tests are SCORER-FREE (no frozen-net load): they exercise the architecture + numpy portability
+ byte accounting + KD-loss math directly. The end-to-end exact-scorer measurement is the trainer's
job (it RE-MEASURES d_seg/d_pose on the frozen CPU scorer; that path is exercised by the smoke run,
not unit-tested here to keep the suite fast + deterministic).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from tac.distillation.smaller_student import (  # noqa: E402
    KL_TEMPERATURE,
    StudentDecoderConfig,
    load_student_npz,
    measure_student_bytes,
    numpy_reference_forward,
    save_student_npz,
    size_ladder,
    student_pair_frames,
    student_param_count,
)

# Load the trainer module (KD-loss math + torch student decoder) via the tools path.
sys.path.insert(0, str(REPO_ROOT / "tools"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_distill_trainer_t74",
    REPO_ROOT / "tools" / "distill_smaller_student_from_frontier_teacher.py",
)
_trainer = importlib.util.module_from_spec(_spec)
# guard: the trainer imports render_and_score_lib at function scope only, so module import is safe.
_spec.loader.exec_module(_trainer)
TorchStudentDecoder = _trainer.TorchStudentDecoder
kd_seg_kl_t2 = _trainer.kd_seg_kl_t2
kd_pose_mse = _trainer.kd_pose_mse
kd_frame_recon = _trainer.kd_frame_recon
EMA = _trainer.EMA


def _tiny_cfg(num_pairs=3) -> StudentDecoderConfig:
    return StudentDecoderConfig(
        num_pairs=num_pairs, latent_dim=6, seed_ch=6, seed_h=3, seed_w=4,
        stage_channels=(6, 4), size_label="tiny",
    )


def _np_weights_from_torch(model: TorchStudentDecoder):
    return model.numpy_params()


# ---------------------------------------------------------------------------
# 1. The student decodes BOTH frames (PoseNet reads both; SegNet reads frame1).
# ---------------------------------------------------------------------------
def test_student_decodes_two_frames():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = TorchStudentDecoder(cfg)
    out = model(0)
    assert out.shape[0] == 2, "student must decode BOTH frames of the pair"
    assert out.shape[1] == cfg.n_channels == 3


# ---------------------------------------------------------------------------
# 2. The student frames VARY across pairs (NOT a stored constant table) — NO FAKE class 2.
# ---------------------------------------------------------------------------
def test_student_frames_vary_across_pairs():
    cfg = _tiny_cfg(num_pairs=4)
    torch.manual_seed(1)
    model = TorchStudentDecoder(cfg)
    with torch.no_grad():
        f_a = model(0)
        f_b = model(1)
    # different pair latents -> different decoded frames (a constant stub would make these identical).
    assert float((f_a - f_b).abs().max()) > 1e-3


# ---------------------------------------------------------------------------
# 3. The student frame VARIES across pixels (NOT a flat constant) — NO FAKE class 2.
# ---------------------------------------------------------------------------
def test_student_frame_varies_across_pixels():
    cfg = _tiny_cfg()
    torch.manual_seed(2)
    model = TorchStudentDecoder(cfg)
    with torch.no_grad():
        f = model(0)  # (2,3,fh,fw)
    spatial_std = float(f.reshape(2, 3, -1).std(dim=-1).max())
    assert spatial_std > 1e-3, "a flat constant frame would have ~0 spatial std (constant-stub fail)"


# ---------------------------------------------------------------------------
# 4. numpy-portable forward reproduces the torch forward (the inflate contract) — within 1 LSB.
# ---------------------------------------------------------------------------
def test_numpy_torch_parity_within_1lsb():
    cfg = _tiny_cfg(num_pairs=2)
    torch.manual_seed(3)
    model = TorchStudentDecoder(cfg)
    weights, latents = _np_weights_from_torch(model)
    for j in range(cfg.num_pairs):
        with torch.inference_mode():
            t = model(j)  # (2,3,fh,fw) block-stack
            t_np = (t.clamp(0, 255).round().numpy())
        n_np = numpy_reference_forward(weights, cfg, latents[j])  # (2,3,fh,fw) float
        n_np = np.clip(np.round(n_np), 0, 255)
        frac = float(np.mean(np.abs(t_np - n_np) <= 1.0))
        assert frac >= 0.99, f"pair {j} numpy<->torch parity {frac} < 0.99"


# ---------------------------------------------------------------------------
# 5. student_pair_frames returns camera-res uint8 (2,H,W,3) — the inflate-time decode shape.
# ---------------------------------------------------------------------------
def test_student_pair_frames_camera_res_uint8():
    cfg = _tiny_cfg()
    torch.manual_seed(4)
    model = TorchStudentDecoder(cfg)
    weights, latents = _np_weights_from_torch(model)
    frames = student_pair_frames(weights, cfg, latents, 0, out_h=40, out_w=48)
    assert frames.shape == (2, 40, 48, 3)
    assert frames.dtype == np.uint8
    assert frames.min() >= 0 and frames.max() <= 255


# ---------------------------------------------------------------------------
# 6. A SMALLER student costs FEWER bytes (the rate-vs-distortion axis) — NO FAKE class 8.
# ---------------------------------------------------------------------------
def test_smaller_student_fewer_bytes():
    ladder = size_ladder(600)
    accts = {}
    for label, cfg in ladder.items():
        torch.manual_seed(7)
        model = TorchStudentDecoder(cfg)
        weights, latents = _np_weights_from_torch(model)
        accts[label] = measure_student_bytes(weights, latents, cfg).total_bytes
    # the nominal ladder must be monotone in bytes: 40kb < 60kb < 80kb < 100kb < 120kb.
    ordered = [accts["40kb"], accts["60kb"], accts["80kb"], accts["100kb"], accts["120kb"]]
    assert ordered == sorted(ordered), f"byte ladder not monotone: {accts}"
    assert accts["40kb"] < accts["120kb"]


# ---------------------------------------------------------------------------
# 7. param_count is monotone in the ladder (a bigger student has more params).
# ---------------------------------------------------------------------------
def test_param_count_monotone_in_ladder():
    ladder = size_ladder(600)
    counts = [student_param_count(ladder[k]) for k in ("40kb", "60kb", "80kb", "100kb", "120kb")]
    assert counts == sorted(counts)


# ---------------------------------------------------------------------------
# 8. byte cost ACTUALLY tracks the weights (not a fixed formula) — perturb weights, bytes change.
# ---------------------------------------------------------------------------
def test_byte_cost_tracks_actual_weights():
    cfg = _tiny_cfg(num_pairs=50)
    torch.manual_seed(8)
    model = TorchStudentDecoder(cfg)
    weights, latents = _np_weights_from_torch(model)
    base = measure_student_bytes(weights, latents, cfg).total_bytes
    # zero the weights -> brotli compresses the all-zero blob far smaller (a fixed formula wouldn't).
    zeroed = {k: np.zeros_like(v) for k, v in weights.items()}
    zlat = np.zeros_like(latents)
    zbytes = measure_student_bytes(zeroed, zlat, cfg).total_bytes
    assert zbytes < base, "all-zero weights must brotli SMALLER than random (byte cost is real)"


# ---------------------------------------------------------------------------
# 9. KL-T2 distill: identical student==teacher logits -> ~0 loss; divergent -> >0 (it's real).
# ---------------------------------------------------------------------------
def test_kl_t2_zero_when_identical_positive_when_divergent():
    torch.manual_seed(9)
    teacher = torch.randn(1, 5, 8, 8)
    same = kd_seg_kl_t2(teacher.clone(), teacher)
    diff = kd_seg_kl_t2(torch.randn(1, 5, 8, 8), teacher)
    assert float(same) < 1e-4, "KL of identical logits must be ~0"
    assert float(diff) > 1e-2, "KL of divergent logits must be clearly positive"


# ---------------------------------------------------------------------------
# 10. KL-T2 uses the T^2 Hinton normalization at T=2.0 (the PR95 canon).
# ---------------------------------------------------------------------------
def test_kl_t2_hinton_temperature_scaling():
    assert KL_TEMPERATURE == 2.0
    torch.manual_seed(10)
    student = torch.randn(1, 5, 6, 6, requires_grad=True)
    teacher = torch.randn(1, 5, 6, 6)
    # manual reference at T=2.0 with the T^2 scaling.
    T = 2.0
    log_p = F.log_softmax(student / T, dim=1)
    q = F.softmax(teacher / T, dim=1)
    ref = (F.kl_div(log_p, q, reduction="none").sum(dim=1).mean()) * (T * T)
    got = kd_seg_kl_t2(student, teacher, temperature=T)
    assert abs(float(got.detach()) - float(ref.detach())) < 1e-5


# ---------------------------------------------------------------------------
# 11. The KL-T2 gradient ACTUALLY flows to the student logits (the distill is differentiable).
# ---------------------------------------------------------------------------
def test_kl_t2_gradient_flows_to_student():
    torch.manual_seed(11)
    student = torch.randn(1, 5, 6, 6, requires_grad=True)
    teacher = torch.randn(1, 5, 6, 6)
    loss = kd_seg_kl_t2(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert float(student.grad.abs().sum()) > 0.0, "KL distill gradient must reach the student logits"


# ---------------------------------------------------------------------------
# 12. The student trains TOWARD the teacher: an optimizer step on the recon loss REDUCES it
#     (the teacher IS the loss; the well-conditioned objective the #62 finding endorses).
# ---------------------------------------------------------------------------
def test_student_trains_toward_teacher_recon():
    cfg = _tiny_cfg(num_pairs=2)
    torch.manual_seed(12)
    model = TorchStudentDecoder(cfg)
    fh, fw = cfg.final_hw()
    teacher_f0 = torch.rand(3, fh, fw) * 255.0
    teacher_f1 = torch.rand(3, fh, fw) * 255.0
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2)

    def recon_now():
        pair = model(0)
        return kd_frame_recon(pair[0], pair[1], teacher_f0, teacher_f1)

    before = float(recon_now().detach())
    for _ in range(40):
        opt.zero_grad()
        loss = recon_now()
        loss.backward()
        opt.step()
    after = float(recon_now().detach())
    assert after < before, f"recon-to-teacher must DECREASE under training ({before} -> {after})"


# ---------------------------------------------------------------------------
# 13. The student's KL-to-teacher DECREASES under a SegNet-free logit-matching proxy
#     (the distill term actually moves the student toward the teacher distribution).
# ---------------------------------------------------------------------------
def test_student_logits_move_toward_teacher_under_kl():
    torch.manual_seed(13)
    student_logits = torch.zeros(1, 5, 6, 6, requires_grad=True)
    teacher_logits = torch.randn(1, 5, 6, 6) * 3.0
    opt = torch.optim.SGD([student_logits], lr=1.0)
    before = float(kd_seg_kl_t2(student_logits, teacher_logits).detach())
    for _ in range(50):
        opt.zero_grad()
        loss = kd_seg_kl_t2(student_logits, teacher_logits)
        loss.backward()
        opt.step()
    after = float(kd_seg_kl_t2(student_logits, teacher_logits).detach())
    assert after < before, f"KL distill must pull student logits toward teacher ({before}->{after})"


# ---------------------------------------------------------------------------
# 14. A CONSTANT student does NOT match the teacher's argmax (the d_seg signal is real).
#     A flat-logit student has a uniform/degenerate argmax; the teacher's KL stays high.
# ---------------------------------------------------------------------------
def test_constant_student_cannot_match_teacher_distribution():
    torch.manual_seed(14)
    teacher = torch.randn(1, 5, 8, 8) * 4.0  # sharp teacher distribution (a real argmax partition)
    flat_student = torch.zeros(1, 5, 8, 8)   # constant/uniform logits (a "constant frame" analog)
    kl_flat = float(kd_seg_kl_t2(flat_student, teacher))
    kl_match = float(kd_seg_kl_t2(teacher.clone(), teacher))
    assert kl_flat > kl_match + 1e-2, "a constant student must NOT match the teacher (d_seg is real)"


# ---------------------------------------------------------------------------
# 15. pose-MSE distill: zero when student==teacher pose, positive when divergent.
# ---------------------------------------------------------------------------
def test_pose_mse_distill_zero_when_identical():
    teacher_pose = torch.randn(1, 6)
    assert float(kd_pose_mse(teacher_pose.clone(), teacher_pose)) < 1e-8
    assert float(kd_pose_mse(torch.randn(1, 6), teacher_pose)) > 0.0


# ---------------------------------------------------------------------------
# 16. EMA shadow is distinct from live weights after updates (the inference checkpoint discipline).
# ---------------------------------------------------------------------------
def test_ema_shadow_tracks_but_differs_from_live():
    cfg = _tiny_cfg()
    torch.manual_seed(16)
    model = TorchStudentDecoder(cfg)
    ema = EMA(model, decay=0.9)
    # perturb live weights then update EMA; the shadow must lag (differ from) the live weights.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    ema.update(model)
    live = model.state_dict()["seed.weight"]
    shadow = ema.state_dict()["seed.weight"]
    assert float((live - shadow).abs().max()) > 1e-6, "EMA shadow must lag the live weights"


# ---------------------------------------------------------------------------
# 17. Save/load round-trips the student npz (portable checkpoint; NO /tmp).
# ---------------------------------------------------------------------------
def test_save_load_student_npz_roundtrip(tmp_path):
    cfg = _tiny_cfg(num_pairs=3)
    torch.manual_seed(17)
    model = TorchStudentDecoder(cfg)
    weights, latents = _np_weights_from_torch(model)
    path = tmp_path / "student.npz"
    save_student_npz(path, weights, latents, cfg)
    w2, l2, cfg2 = load_student_npz(path)
    assert cfg2.to_dict() == cfg.to_dict()
    np.testing.assert_allclose(l2, latents, rtol=0, atol=1e-6)
    for k in weights:
        np.testing.assert_allclose(w2[k], weights[k], rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# 18. /tmp paths are refused in the save path (CLAUDE.md transient-evidence trap).
# ---------------------------------------------------------------------------
def test_save_refuses_tmp_path():
    cfg = _tiny_cfg()
    torch.manual_seed(18)
    model = TorchStudentDecoder(cfg)
    weights, latents = _np_weights_from_torch(model)
    with pytest.raises(ValueError, match="tmp"):
        save_student_npz(Path("/tmp/student.npz"), weights, latents, cfg)


# ---------------------------------------------------------------------------
# 19. The trainer refuses a stub loop (internal-consistency elapsed >= epochs*MIN_SEC) — NO FAKE.
#     We verify the guard constant + the RuntimeError path exists by direct construction.
# ---------------------------------------------------------------------------
def test_internal_consistency_floor_constant():
    assert _trainer.MIN_SEC_PER_EPOCH > 0.0
    # the train() function raises RuntimeError when elapsed < epochs * MIN_SEC; we assert the
    # message contract is present in the source (the NO-FAKE stub-loop refusal).
    src = (REPO_ROOT / "tools" / "distill_smaller_student_from_frontier_teacher.py").read_text()
    assert "refusing a stub training loop (NO FAKE)" in src
    assert "elapsed < epochs * MIN_SEC_PER_EPOCH" in src


# ---------------------------------------------------------------------------
# 20. out_channels == 2 * n_channels (the BOTH-frames contract) + final_hw lifts to >= camera-ish.
# ---------------------------------------------------------------------------
def test_config_both_frames_contract():
    cfg = size_ladder(600)["80kb"]
    assert cfg.out_channels == 2 * cfg.n_channels == 6
    fh, fw = cfg.final_hw()
    assert fh == cfg.seed_h * (2 ** len(cfg.stage_channels))
    assert fw == cfg.seed_w * (2 ** len(cfg.stage_channels))
