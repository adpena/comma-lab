# SPDX-License-Identifier: MIT
"""Scorer contexts for the torch-vehicle driver.

Two implementations of the :class:`tac.torch_vehicle.driver.ScorerContext`
protocol:

* :class:`RealScorerContext` — production. Binds the REAL frozen contest
  SegNet/PoseNet via the vendored ``data.precompute_targets`` (GT decoded ONLY
  via ``frame_utils.yuv420_to_rgb`` — the canonical GT path; PyAV rgb24 is
  FORBIDDEN per CLAUDE.md as it manufactures ~100× phantom pose) and the
  vendored ``score.evaluate_decoder`` / ``score.compute_score`` for the BEST-by-
  canonical-score exact eval. The seg/pose forward matches ``common.py`` 1:1.
* :class:`SyntheticScorerContext` — test. A tiny deterministic frozen scorer +
  random GT targets so the resume round-trip (architecture-AGNOSTIC) is fast.
  This mirrors the MLX ``test_capstone_vq_nerv._build_capstone_setup`` synthetic-
  scorer fixture; it is RESEARCH-ONLY (``research_only=True``) and NEVER a score
  claim — its only job is to exercise the driver's STATE serialization.

Authority: real context = torch-CPU TRUSTED (NO MPS). The in-loop d_seg/d_pose
are ``[contest-CPU advisory]`` NON-PROMOTABLE until the byte-closed archive is
run through ``upstream/evaluate.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class RealScorerContext:
    """Production scorer context: real frozen SegNet/PoseNet + canonical GT.

    Precomputes the per-pair ``seg_targets_hard`` + ``pose_targets`` once (the
    vendored ``precompute_targets``), holds the frozen ``distortion_net``, and
    routes the per-step forward + the BEST-tracking exact eval through the
    vendored primitives unchanged.

    Split-device (the 104x MPS lever): pass ``train_device='mps'`` to run the
    per-step ``seg_pose_forward`` (the gradient backend) on the Apple GPU while
    ``exact_eval`` ALWAYS runs on the AUTHORITY device (``device``, CPU-TRUSTED /
    CUDA) — the exact d_seg/d_pose that pick BEST are NEVER read off MPS
    (CLAUDE.md "MPS auth eval is NOISE": 23x pose drift). Two frozen scorer
    instances are held (authority on ``device``, train on ``train_device``); the
    weights are identical (frozen), only the device differs. The targets are held
    on the train device (where the per-step loss is computed); ``exact_eval``
    streams fresh GT through the authority net so it needs no target copy.
    """

    research_only = False

    def __init__(
        self,
        video_path: str | Path,
        device: str = "cpu",
        *,
        train_device: str | None = None,
        split_by_head: bool = False,
        max_pairs: int | None = None,
        targets_cache: str | Path | None = None,
    ):
        if str(device).lower().startswith("mps"):
            raise ValueError(
                "MPS is NEVER trusted as the AUTHORITY device (CLAUDE.md). Use 'cpu' "
                "(TRUSTED) or 'cuda' for device; pass train_device='mps' to use the "
                "Apple GPU as the per-step GRADIENT backend only (split-device)."
            )
        self.device = torch.device(device)  # AUTHORITY/eval device
        self.train_device = torch.device(train_device) if train_device else self.device
        self.split_device = self.train_device != self.device
        # Split-by-HEAD (the pose-axis salvage): run the SegNet-path forward+backward
        # on the FAST train device (MPS — bit-identical on d_seg per the gate) and the
        # PoseNet-path forward+backward on the CPU AUTHORITY (zero pose drift). The
        # two per-head frame cotangents are summed at the frame tensor → a combined
        # gradient that is descent-equivalent on BOTH terms by construction (the pose
        # part IS the authority gradient). Requires split_device (train != authority).
        self.split_by_head = bool(split_by_head)
        if self.split_by_head and not self.split_device:
            raise ValueError(
                "split_by_head requires a split device (train_device != device): "
                "the SegNet path runs on train_device (MPS) and the PoseNet path on "
                "the CPU authority. Pass train_device='mps' (or another fast device) "
                "distinct from device='cpu'."
            )
        self.video_path = Path(video_path)
        from tac.torch_vehicle.vendored_imports import import_vendored

        self._data = import_vendored("data")  # applies the differentiable-yuv6 patch
        self._score = import_vendored("score")
        # The AUTHORITY scorer (eval) is always built on ``device`` (CPU-TRUSTED).
        from tac.score_aware_loop.targets import load_frozen_distortion_net

        self.distortion_net = load_frozen_distortion_net(device=str(self.device))
        # Targets are the frozen scorer's OWN output on GT (the d_seg/d_pose
        # reference) — ALWAYS computed on the CPU AUTHORITY (never MPS). For the
        # full basis we use the uncapped vendored ``precompute_targets``; for a
        # cheap-n A/B / smoke (max_pairs set) we use the CAPPED, cached
        # ``build_gt_targets`` so the per-step target precompute does not pay the
        # full 600-pair CPU cost (~2.5h) — only the n we actually train on.
        if max_pairs is None:
            (
                _net,
                seg_targets_hard,
                pose_targets,
                _gt_half,
                self.n_pairs,
            ) = self._data.precompute_targets(self.video_path, self.device)
        else:
            seg_targets_hard, pose_targets = self._build_capped_targets(
                int(max_pairs), targets_cache
            )
            self.n_pairs = int(seg_targets_hard.shape[0])
        # The per-step loss runs on the train device; hold the targets there.
        self.seg_targets_hard = seg_targets_hard.to(self.train_device)
        self.pose_targets = pose_targets.to(self.train_device)
        # The TRAIN-side frozen scorer (gradient backend). Identical frozen weights
        # to the authority net; only the device differs. When not split-device this
        # IS the authority net (no second copy).
        if self.split_device:
            from tac.score_aware_loop.targets import load_frozen_distortion_net

            self.train_distortion_net = load_frozen_distortion_net(
                device=str(self.train_device)
            )
        else:
            self.train_distortion_net = self.distortion_net

    def _build_capped_targets(
        self, max_pairs: int, targets_cache: str | Path | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (or load from cache) the first ``max_pairs`` CPU-authority GT
        targets via the capped ``build_gt_targets`` path. Cached to disk so a
        repeated A/B / smoke does not re-pay the per-pair CPU scorer cost."""
        from tac.score_aware_loop.targets import build_gt_targets

        cache_file: Path | None = None
        if targets_cache is not None:
            cache_dir = Path(targets_cache)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"gt_targets_n{max_pairs}.pt"
            if cache_file.exists():
                blob = torch.load(cache_file, map_location="cpu", weights_only=False)
                return blob["seg"][:max_pairs], blob["pose"][:max_pairs]
        seg_t, pose_t, _n = build_gt_targets(
            self.distortion_net,
            video_path=str(self.video_path),
            max_pairs=max_pairs,
            device=str(self.device),  # CPU AUTHORITY
        )
        if cache_file is not None:
            torch.save({"seg": seg_t.cpu(), "pose": pose_t.cpu()}, cache_file)
        return seg_t, pose_t

    def maybe_compile_scorers(self, *, mode: str = "default") -> None:
        """SCORE-NEUTRAL launch-overhead lever: ``torch.compile`` the FROZEN train-side
        SegNet + PoseNet forwards in place. The scorers are frozen (requires_grad=False,
        eval-mode) so compiling their forward is purely a runtime placement of the SAME
        ops — it does NOT change the gradient/score MATH (modulo inductor numerics, which
        the caller MUST clear with a paired d_seg/d_pose neutrality check at ~1e-4 rel
        before a real run trusts it). Wraps ONLY the two frozen scorer submodules of the
        TRAIN distortion net (the gradient backend); the AUTHORITY eval net is untouched
        so the exact d_seg/d_pose that pick BEST stay eager + CPU-trusted."""
        net = self.train_distortion_net
        seg = getattr(net, "segnet", None)
        pose = getattr(net, "posenet", None)
        if seg is None or pose is None:
            print(
                "[scorer-context] maybe_compile_scorers: net has no segnet/posenet "
                "submodule; skipping compile.",
                flush=True,
            )
            return
        try:
            net.segnet = torch.compile(seg, mode=mode)
            net.posenet = torch.compile(pose, mode=mode)
            print(
                f"[scorer-context] compiled train-side SegNet+PoseNet forwards "
                f"(mode={mode}) on {self.train_device}.",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - defensive (compile env varies)
            print(
                f"[scorer-context] torch.compile(mode={mode}) FAILED ({exc!r}); "
                "falling back to eager scorers.",
                flush=True,
            )
            net.segnet = seg
            net.posenet = pose

    def seg_pose_forward(self, decoded_bhwc: torch.Tensor):
        """Run the real frozen scorer on roundtripped decoder output, 1:1 with
        ``common.py:187-189`` — on the TRAIN device (the gradient backend, possibly
        the Apple GPU). This is the only path that uses MPS; its gradient is
        research-signal until the descent-equivalence gate clears it."""
        net = self.train_distortion_net
        posenet_in, segnet_in = net.preprocess_input(decoded_bhwc)
        seg_out = net.segnet(segnet_in)
        pose_out = net.posenet(posenet_in)
        return seg_out, pose_out["pose"][:, :6]

    def seg_forward_train(self, decoded_bhwc_train: torch.Tensor) -> torch.Tensor:
        """SegNet-path forward on the TRAIN device (MPS — the 90x lever; validated
        bit-identical on d_seg by the descent-equivalence gate). ``decoded_bhwc_train``
        is on ``train_device``; returns seg logits on ``train_device``."""
        net = self.train_distortion_net
        _posenet_in_unused, segnet_in = net.preprocess_input(decoded_bhwc_train)
        return net.segnet(segnet_in)

    def pose_forward_authority(self, decoded_bhwc_cpu: torch.Tensor) -> torch.Tensor:
        """PoseNet-path forward on the CPU AUTHORITY net (zero MPS pose drift — the
        whole point of the split-by-head salvage). ``decoded_bhwc_cpu`` is on the
        authority ``device``; returns pose6 on the authority ``device``. Because this
        IS the authority PoseNet, its backward gives the AUTHORITY pose gradient."""
        net = self.distortion_net
        posenet_in, _segnet_in_unused = net.preprocess_input(decoded_bhwc_cpu)
        pose_out = net.posenet(posenet_in)
        return pose_out["pose"][:, :6]

    def pose_forward_train(self, decoded_bhwc_train: torch.Tensor) -> torch.Tensor:
        """PoseNet-path forward on the TRAIN device net (the FULL-MPS unbundle — MPS as
        the GRADIENT backend for the pose head too). ``decoded_bhwc_train`` is on
        ``train_device``; returns pose6 on ``train_device``. Uses ``train_distortion_net``
        (the SAME frozen net the SegNet head uses; patch_scorer_for_mps + differentiable
        rgb_to_yuv6 are already applied when that net is built on MPS via
        ``load_frozen_distortion_net``). This is a GRADIENT path ONLY — the exact d_pose
        that picks BEST ALWAYS runs through ``exact_eval`` on the CPU authority net. The
        MPS pose gradient is per-step faithful (the full-MPS basin reached CPU-authority
        d_pose=0.00034); it must still pass the descent-equivalence chaos-check before a
        real run trusts it."""
        net = self.train_distortion_net
        posenet_in, _segnet_in_unused = net.preprocess_input(decoded_bhwc_train)
        pose_out = net.posenet(posenet_in)
        return pose_out["pose"][:, :6]

    def exact_eval(self, ema_decoder: nn.Module, ema_latents: torch.Tensor, archive_bytes: int):
        """Canonical d_seg/d_pose/rate/score on the EMA shadow (BEST tracker).

        ALWAYS on the AUTHORITY device (``self.device``), NEVER the train device:
        the score that picks BEST is the CPU-TRUSTED exact metric (CLAUDE.md "MPS
        auth eval is NOISE"). Routes through the vendored ``evaluate_decoder``
        (streams GT via ``yuv420_to_rgb``) + ``compute_score`` (the official
        metric). The archive_bytes is the BYTE-CLOSED size from the build_archive
        the driver already produced (so the rate term is the real archive bytes)."""
        tvb = self._score.total_video_bytes(self.video_path)
        dist = self._score.evaluate_decoder(
            ema_decoder,
            ema_latents.to(self.device),
            self.distortion_net,  # the AUTHORITY net (device=self.device)
            self.video_path,
            batch_pairs=8,
            device=self.device,
        )
        result = self._score.compute_score(
            dist["seg_distortion"], dist["pose_distortion"], archive_bytes, tvb
        )
        return {
            "seg_distortion": result["seg_distortion"],
            "pose_distortion": result["pose_distortion"],
            "rate": result["rate"],
            "score": result["score"],
        }


class _TinyFrozenScorer(nn.Module):
    """A tiny deterministic 'frozen scorer' for the resume round-trip test.

    NOT the contest scorer — a fixed-weight conv that maps roundtripped frames to
    5-class seg logits + a 6-dim pose, just enough that the driver's loss has a
    real gradient to the decoder. Frozen (requires_grad=False). Deterministic
    given a seed.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # seg head: 3->5 over the 384x512 last-frame (downsampled by the driver's
        # preprocess is skipped here; we operate on the raw 384x512x3).
        self.seg = nn.Conv2d(3, 5, 3, padding=1)
        self.pose = nn.Linear(3, 6)
        for p in self.parameters():
            p.data = torch.empty_like(p).uniform_(-0.05, 0.05, generator=g)
            p.requires_grad_(False)


class SyntheticScorerContext:
    """Test scorer context — fast, deterministic, RESEARCH-ONLY (no score claim).

    Provides the same interface the driver needs; the resume round-trip is
    architecture-agnostic so this exercises STATE serialization without the real
    EfficientNet scorer. Mirrors the MLX synthetic-scorer fixture.
    """

    research_only = True

    def __init__(
        self, n_pairs: int = 6, device: str = "cpu", seed: int = 0, *,
        train_device: str | None = None, split_by_head: bool = False,
    ):
        self.device = torch.device(device)  # authority/eval device
        self.train_device = torch.device(train_device) if train_device else self.device
        self.split_device = self.train_device != self.device
        self.split_by_head = bool(split_by_head)
        self.n_pairs = int(n_pairs)
        # Authority scorer (eval) + train scorer (forward). Same frozen weights;
        # only the device differs (mirrors RealScorerContext split-device).
        self._scorer = _TinyFrozenScorer(seed=seed).to(self.device).eval()
        self._train_scorer = (
            _TinyFrozenScorer(seed=seed).to(self.train_device).eval()
            if self.split_device
            else self._scorer
        )
        g = torch.Generator().manual_seed(seed + 1)
        # GT targets: random-but-fixed seg labels (0..4) over 384x512, pose 6-d.
        # Held on the TRAIN device (where the per-step loss is computed).
        self.seg_targets_hard = torch.randint(
            0, 5, (self.n_pairs, 384, 512), generator=g
        ).to(self.train_device)
        self.pose_targets = (
            torch.empty(self.n_pairs, 6).uniform_(-1, 1, generator=g).to(self.train_device)
        )

    def seg_pose_forward(self, decoded_bhwc: torch.Tensor):
        """decoded_bhwc: (B, 2, 384, 512, 3) float [0,255]. Use last frame for seg
        (matching the contest's last-frame seg), both-frame mean for pose. Runs on
        the TRAIN scorer (gradient backend; mirrors RealScorerContext split)."""
        last = decoded_bhwc[:, -1].permute(0, 3, 1, 2) / 255.0  # (B,3,384,512)
        seg_out = self._train_scorer.seg(last)  # (B,5,384,512)
        # pose from a cheap global pool of the pair mean.
        pooled = decoded_bhwc.mean(dim=(1, 2, 3)) / 255.0  # (B,3)
        pose_pred6 = self._train_scorer.pose(pooled)  # (B,6)
        return seg_out, pose_pred6

    def seg_forward_train(self, decoded_bhwc_train: torch.Tensor) -> torch.Tensor:
        """SegNet-path forward on the TRAIN scorer (mirrors RealScorerContext)."""
        last = decoded_bhwc_train[:, -1].permute(0, 3, 1, 2) / 255.0
        return self._train_scorer.seg(last)

    def pose_forward_authority(self, decoded_bhwc_cpu: torch.Tensor) -> torch.Tensor:
        """PoseNet-path forward on the AUTHORITY scorer (mirrors RealScorerContext)."""
        pooled = decoded_bhwc_cpu.mean(dim=(1, 2, 3)) / 255.0
        return self._scorer.pose(pooled)

    def pose_forward_train(self, decoded_bhwc_train: torch.Tensor) -> torch.Tensor:
        """PoseNet-path forward on the TRAIN scorer (the FULL-MPS unbundle; mirrors
        RealScorerContext). Same frozen weights as the authority pose head; only the
        device differs (so on a same-device synthetic control the gradient is identical
        to the authority path — exactly what the device-routing test asserts)."""
        pooled = decoded_bhwc_train.mean(dim=(1, 2, 3)) / 255.0
        return self._train_scorer.pose(pooled)

    def exact_eval(self, ema_decoder: nn.Module, ema_latents: torch.Tensor, archive_bytes: int):
        """A deterministic synthetic 'score' from the EMA shadow argmax-flip vs
        targets + a byte rate. RESEARCH-ONLY — not a contest score. ALWAYS on the
        AUTHORITY device/scorer (mirrors RealScorerContext: eval is never on the
        train device)."""
        seg_targets = self.seg_targets_hard.to(self.device)
        pose_targets = self.pose_targets.to(self.device)
        with torch.inference_mode():
            n = min(self.n_pairs, ema_latents.shape[0])
            z = ema_latents[:n].to(self.device)
            decoded = ema_decoder(z)  # (n,2,3,384,512)
            last = decoded[:, -1] / 255.0  # (n,3,384,512)
            seg_logits = self._scorer.seg(last)
            d_seg = (
                (seg_logits.argmax(dim=1) != seg_targets[:n]).float().mean().item()
            )
            pooled = decoded.mean(dim=(1, 3, 4)) / 255.0  # (n,3)... careful dims
            # decoded is (n,2,3,384,512); mean over frames+spatial -> (n,3)
            pooled = decoded.mean(dim=(1, 3, 4))[:, :3] / 255.0
            pose_pred = self._scorer.pose(pooled)
            d_pose = torch.nn.functional.mse_loss(pose_pred, pose_targets[:n]).item()
        rate = archive_bytes / 37_545_489.0
        score = 100.0 * d_seg + (10.0 * d_pose + 1e-12) ** 0.5 + 25.0 * rate
        return {
            "seg_distortion": d_seg,
            "pose_distortion": d_pose,
            "rate": rate,
            "score": score,
        }


__all__ = ["RealScorerContext", "SyntheticScorerContext"]
