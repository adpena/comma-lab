# SPDX-License-Identifier: MIT
"""ddm_tt1 — differentiable torch twin of the v4c continuous decode (QA71).

THE REVIVED FORM (ph3 doctrine): the frozen-scorer gradient is the PROPOSAL
ENGINE, never the acceptor.  This module builds a torch (MPS-capable, fp32)
twin of the v4c decode path for the CONTINUOUS DOF — the per-pair pose 6-vector
and the photometric gains (a, b) — so that ONE analytic backward yields the
joint gradient d(d_pose)/d(pose,a,b), replacing the v4c numerical-Jacobian GN
(6-8 PoseNet forwards / relin).  Acceptance is NOT this module's business: every
proposed step is realized through the REAL numpy decode + CPU PoseNet (the
ddm_tt1_joint_tto harness) and accepted iff realized d_pose descends.

STRUCTURAL CRUX (upstream/modules.py:108, MEASURED-confirmed): SegNet reads
``x[:,-1]`` = frame_1 ONLY.  pose + (a,b) touch ONLY frame_0, so d_seg is
INVARIANT to them.  On these DOF the joint objective 100*d_seg + sqrt(10*d_pose)
+ 25*rate reduces to d_pose (seg + rate frozen).  The genuine 3-way d_seg
coupling lives in the token stream (frame_1) — the stage-2 stretch.

Twin geometry mirrors experiments/pfs1_warp_receiver.py + inflate_runner_v4c.py
exactly: pose6 -> H = K(R(s_r*w) - t(s_t*tau) n^T / h) Kinv ; inverse-warp
bilinear with identity fill; two-plane static compose (rows < 437 = far H_inf) ;
f0 := a*warp + b ; STE-uint8 ; PoseNet6 ; MSE vs target.

Axis: [macOS advisory].  MPS = gradient/proposal device ONLY (fp32; NEVER a
score).  Realized authority is CPU numpy decode + CPU PoseNet.  Pointer
0.1910828242 [contest-CPU] UNMOVED.  score_claim=false.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path("/Users/adpena/Projects/pact")
_PFS1_SUB = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1")
for _p in (_REPO / "src", _REPO / "experiments", _REPO / "src/tac/optimization", _PFS1_SUB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import tac

if tac.__file__ != str(_REPO / "src/tac/__init__.py"):
    raise SystemExit(f"tac HIJACK: {tac.__file__} (export PYTHONPATH={_REPO}/src first)")

import ddm_p3v2_optimal_form_pose_resolve as p3v2
from inflate_runner_v4c import Decoder

CAMERA_H, CAMERA_W = 874, 1164
CAMERA_HEIGHT_M = 1.22


def _skew_torch(w):
    import torch
    z = torch.zeros((), dtype=w.dtype, device=w.device)
    return torch.stack([
        torch.stack([z, -w[2], w[1]]),
        torch.stack([w[2], z, -w[0]]),
        torch.stack([-w[1], w[0], z]),
    ])


def _expmap_so3_torch(omega):
    """Differentiable Rodrigues; matches pfs1_warp_receiver._expmap_so3 for
    theta>0 (poses are never exactly 0).  Small-theta-safe via +eps in theta."""
    import torch
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    Kk = _skew_torch(omega)
    theta = torch.linalg.vector_norm(omega) + 1e-20
    A = torch.sin(theta) / theta
    B = (1.0 - torch.cos(theta)) / (theta * theta)
    return eye + A * Kk + B * (Kk @ Kk)


class WarpTwin:
    """Differentiable d_pose(pose6, a, b) twin + the numpy acceptor primitives.

    The numpy Decoder (self.dec) is the ground truth for frame_1 rendering AND
    the realized acceptor (self.acceptor_f0 / self.acceptor_d_pose).  The torch
    path (self.d_pose_diff) is the PROPOSAL gradient only.
    """

    def __init__(self, archive_dir: Path, device: str = "mps", s_r: float = 1.0) -> None:
        import torch

        self.torch = torch
        self.dec = Decoder(Path(archive_dir))
        self.n_pairs = self.dec.n_pairs
        self.s_r = float(s_r)
        # CPU posenet (patched yuv6, frozen) — used by BOTH the acceptor
        # (p3v2.d_pose_u8, inference_mode) and, on device, the twin forward.
        self.posenet, _ = p3v2.load_posenet()
        self.targets = p3v2.load_targets(self.n_pairs)
        self.dev = torch.device(device)
        # device copy of posenet for the differentiable forward (same weights)
        if self.dev.type == "cpu":
            self.net = self.posenet
        else:
            self.net = p3v2.load_posenet()[0].to(self.dev)
            for p in self.net.parameters():
                p.requires_grad = False
        # torch geometry constants (float32 on device)
        f32 = torch.float32
        self.K = torch.tensor(self.dec.K, dtype=f32, device=self.dev)
        self.Kinv = torch.tensor(self.dec.Kinv, dtype=f32, device=self.dev)
        self.grid = torch.tensor(self.dec.grid, dtype=f32, device=self.dev)  # (3, HW)
        self.n_plane = torch.tensor([0.0, -1.0, 0.0], dtype=f32, device=self.dev)
        far = torch.tensor(self.dec._far.reshape(-1), device=self.dev)  # (HW,) bool
        self.far_flat = far
        self._f1_cache: dict[int, np.ndarray] = {}
        self._f1cam: dict[int, object] = {}
        self._tgt_t: dict[int, object] = {}

    # ---- numpy side (frozen f1 + the realized acceptor) ------------------- #
    def f1(self, i: int) -> np.ndarray:
        if i not in self._f1_cache:
            self._f1_cache[i] = self.dec.f1(i)
        return self._f1_cache[i]

    def acceptor_f0(self, i: int, pose: np.ndarray, a: float, b: float) -> np.ndarray:
        """The EXACT v4c decode f0 at candidate (pose, a, b) — numpy authority.
        Mirrors inflate_runner_v4c.Decoder.f0 for rs_beta==0 (v4c ships 0)."""
        f1_f = self.f1(i).astype(np.float64)
        s_t = float(self.dec.st_vals[self.dec.st_idx[i]])
        sel = int(self.dec.sel[i])
        f0f = self.dec._warp_pair(f1_f, np.asarray(pose, np.float64), s_t, sel, 1.0)
        if a != 1.0 or b != 0.0:
            f0f = a * f0f + b
        from pfs1_warp_receiver import _to_uint8
        return _to_uint8(f0f)

    def acceptor_d_pose(self, i: int, pose: np.ndarray, a: float, b: float) -> float:
        """Realized d_pose at the SHIPPED quantization (pose f16, a/b f16)."""
        q = np.asarray(pose, np.float64).astype(np.float16).astype(np.float64)
        qa = float(np.float16(a))
        qb = float(np.float16(b))
        f0 = self.acceptor_f0(i, q, qa, qb)
        f1 = self.f1(i)
        return float(p3v2.d_pose_u8(self.posenet, f0, f1, self.targets[i]))

    # ---- torch side (the differentiable proposal) ------------------------- #
    def _f1_cam_tensor(self, i: int):
        if i not in self._f1cam:
            f1 = self.torch.tensor(self.f1(i).astype(np.float32), device=self.dev)
            self._f1cam[i] = f1.permute(2, 0, 1).contiguous()  # (3,H,W)
        return self._f1cam[i]

    def _target_t(self, i: int):
        if i not in self._tgt_t:
            self._tgt_t[i] = self.torch.tensor(
                self.targets[i].astype(np.float32), device=self.dev)
        return self._tgt_t[i]

    def _pose_to_H(self, pose6, s_t: float, s_r: float):
        torch = self.torch
        t = s_t * torch.stack([pose6[2], pose6[1], pose6[0]])
        R = _expmap_so3_torch(s_r * torch.stack([pose6[3], pose6[4], pose6[5]]))
        M = R - torch.outer(t, self.n_plane) / CAMERA_HEIGHT_M
        return self.K @ M @ self.Kinv

    def _warp(self, f1_flat, H):
        """Inverse-warp bilinear with identity fill; matches warp_rgb numpy."""
        torch = self.torch
        Hinv = torch.linalg.inv(H)
        src = Hinv @ self.grid            # (3, HW)
        z = src[2]
        su = src[0] / z
        sv = src[1] / z
        W1 = CAMERA_W - 1
        H1 = CAMERA_H - 1
        valid = (torch.isfinite(su) & torch.isfinite(sv) & (z > 0)
                 & (su >= 0) & (su <= W1) & (sv >= 0) & (sv <= H1))
        su_c = su.clamp(0.0, float(W1))
        sv_c = sv.clamp(0.0, float(H1))
        x0 = torch.floor(su_c).long()
        y0 = torch.floor(sv_c).long()
        x1 = torch.clamp(x0 + 1, max=W1)
        y1 = torch.clamp(y0 + 1, max=H1)
        wx = (su_c - x0.to(su_c.dtype)).unsqueeze(1)
        wy = (sv_c - y0.to(sv_c.dtype)).unsqueeze(1)
        Ia = f1_flat[y0 * CAMERA_W + x0]
        Ib = f1_flat[y0 * CAMERA_W + x1]
        Ic = f1_flat[y1 * CAMERA_W + x0]
        Id = f1_flat[y1 * CAMERA_W + x1]
        top = Ia * (1.0 - wx) + Ib * wx
        bot = Ic * (1.0 - wx) + Id * wx
        sampled = top * (1.0 - wy) + bot * wy
        return torch.where(valid.unsqueeze(1), sampled, f1_flat)  # (HW,3)

    def _f0_cam(self, i: int, pose6, a, b):
        """Differentiable camera-res f0 (HW,3) float, PRE-uint8."""
        torch = self.torch
        f1_flat = self._f1_cam_tensor(i).permute(1, 2, 0).reshape(-1, 3)  # (HW,3)
        s_t = float(self.dec.st_vals[self.dec.st_idx[i]])
        sel = int(self.dec.sel[i])
        warp_g = self._warp(f1_flat, self._pose_to_H(pose6, s_t, self.s_r))
        if sel == 1:
            warp_f = self._warp(f1_flat, self._pose_to_H(pose6, 0.0, self.s_r))
            f0 = torch.where(self.far_flat.unsqueeze(1), warp_f, warp_g)
        else:
            f0 = warp_g
        return a * f0 + b  # (HW,3)

    def d_pose_diff(self, i: int, pose6, a, b):
        """Differentiable realized-path d_pose for pair i at (pose6, a, b).
        pose6 (6,), a (), b () torch tensors (requires_grad as desired)."""
        torch = self.torch
        f0 = self._f0_cam(i, pose6, a, b).reshape(CAMERA_H, CAMERA_W, 3)
        f0c = f0.clamp(0.0, 255.0)
        f0c = f0c + (torch.round(f0c) - f0c).detach()          # STE uint8
        f0_chw = f0c.permute(2, 0, 1)
        x = torch.stack([f0_chw, self._f1_cam_tensor(i)]).unsqueeze(0)  # (1,2,3,H,W)
        out = self.net(self.net.preprocess_input(x))
        pose = out["pose"] if isinstance(out, dict) else out
        p6 = pose[0, :6]
        return ((p6 - self._target_t(i)) ** 2).mean()
