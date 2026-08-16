#!/usr/bin/env python3
"""JC1: the per-pair carrier->PoseNet Jacobian producer, and the two consumers it serves.

WHAT THIS BUILDS (the object nobody had). For every pair i in 0..599:

    J_i  =  d( PoseNet(pair_i)['pose'][:6] ) / d( carrier_coeff_i )      in R^{6x12}

through the REAL shipped chain -- fx1 carrier render (einsum -> /sqrt(12) -> 127.5 +
64*carrier -> clamp -> round -> bicubic to 874x1164 -> clamp -> round -> uint8), then
the exact upstream PoseNet preprocess (bilinear to 384x512, then rgb_to_yuv6), then the
frozen CPU-torch PoseNet.

FORWARD IS EXACT, BACKWARD IS RELAXED -- and the distinction is load-bearing:

  * FORWARD. The two ``.round()`` calls are evaluated exactly. The rendered frame_0 is
    required to be BYTE-IDENTICAL to the retained base render (`0.raw` even frames).
    That control is the instrument check: if it passes, the value this Jacobian is
    linearized AT is the shipped chain's own operating point, not a nearby relaxation.
  * BACKWARD. ``round`` has zero derivative almost everywhere, so the EXACT Jacobian of
    the quantized chain is identically zero and carries no information. Every gradient
    here is therefore taken through a straight-through estimator on both rounds
    (`x + (x.round() - x).detach()`), the standard and only informative linearization.
    The two ``clamp`` calls keep their REAL gradient mask -- a clamped pixel genuinely
    has zero sensitivity, and that zero is signal, not an artifact.

  ==> Every J_i, and every number derived from it, is MEASURED-ON-THE-STE-RELAXED-CHAIN.
      A first-order claim from J_i is a NECESSARY screen for a real byte-closed edit, not
      a sufficient one. That caveat travels with the number.

CONSUMER (a) -- POSE-METRIC RE-FIT. `ddm_ra2a_carrier_fidelity_pose_ladder.py:138` states
its own scope: "Exhaustive-optimal keep set in the EUCLIDEAN metric, least-squares refit"
(plain `lstsq`, :151), and `ddm_mp2_carrier_exact_byte_race.py:132` records
`rank_refit_status = NOT_MEASURED_BY_THIS_EXACT_RACE`. The re-fit exists only in the
Euclidean metric and was never measured. d_pose reads the PoseNet-Jacobian metric, where
Eckart-Young gives no theorem. With J_i and the per-pair residual r_i in hand the
pose-optimal re-fit is a CLOSED FORM, not a search:

    d_pose_i(new)  ~=  (1/6) || r_i + J_i . dc_i ||^2 ,     r_i = pose6_gen_i - pose6_gt_i

CONSUMER (b) -- COMMON POSE-NULL DIMENSION. d_pose reads 6 scalars per pair and the
carrier is 12-dimensional, so rank(G_i) = rank(J_i^T J_i) <= 6 and EVERY pair has a
>=6-dim pose-null subspace in carrier coordinates. The decisive number is what survives
the intersection:

    K  =  dim( intersect_{i=1..600} null(G_i) )  =  dim( null(J_stack) )  =  12 - rank(J_stack)

where J_stack is the (3600 x 12) vertical stack. That identity is why one SVD answers it:
v is in every null(G_i) iff J_i v = 0 for all i iff J_stack v = 0. K is reported as a
function of tolerance -- the singular-value spectrum IS the curve -- in two NAMED
coordinates (raw stored, and coefficient-RMS-whitened), never a single arbitrary cutoff.

Axis: [macOS-CPU advisory]. score_claim=false, promotable=false. No Modal, no paid GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RA2B_SOURCE = REPO / "experiments/ddm_ra2b_carrier_chain_control.py"

DEFAULT_ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/archive.zip"
)
DEFAULT_BASE_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/0.raw"
)
DEFAULT_UPSTREAM = Path("/Volumes/APDataStore/pact/upstream_eval_mirror_20260815")
DEFAULT_OUT = Path("/Volumes/APDataStore/pact/ddm_jc1/retained")

ARCHIVE_SHA = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
ARCHIVE_BYTES = 182_759

N_PAIRS = 600
CAMERA_H, CAMERA_W = 874, 1164
FRAME_BYTES = CAMERA_H * CAMERA_W * 3
POSE_ROWS = 6
CARRIER_DIM = 12

#: The retained base advisory row (work_r2/report.txt) -- the operating point J is taken at.
BASE_D_POSE = 0.00014747
BASE_D_SEG = 0.00042714

AXIS = "[macOS-CPU advisory]"
SEED = 1234


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def load_ra2b():
    """Import the PROVEN decode chain wholesale -- same custody pins, same decode."""
    spec = importlib.util.spec_from_file_location("ra2b_chain", RA2B_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ra2b_chain"] = module
    spec.loader.exec_module(module)
    return module


def ste_round(x):
    """Straight-through round: exact forward value, identity gradient.

    The exact chain's derivative through ``round`` is 0 a.e., so it carries no
    information. This is the standard linearization and it is labelled as such
    everywhere the resulting Jacobian is used.
    """
    return x + (x.round() - x).detach()


def render_frame0_differentiable(torch, F, normalized_basis, coeff_row, *, ste: bool):
    """The carrier half of fx1 render_video (inflate.py:659-676), verbatim.

    ``ste=False`` reproduces the shipped forward EXACTLY (hard round, no grad path);
    ``ste=True`` keeps the identical forward VALUE and relaxes only the backward.
    Both clamps keep their real gradient mask in either mode.
    """
    rnd = ste_round if ste else (lambda t: t.round())
    carrier = torch.einsum("bk,kchw->bchw", coeff_row, normalized_basis)
    carrier = carrier / math.sqrt(CARRIER_DIM)
    slave_eval = rnd((127.5 + 64.0 * carrier).clamp(0.0, 255.0))
    slave = F.interpolate(
        slave_eval, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
    ).clamp(0.0, 255.0)
    return rnd(slave)


def apply_pixel_mode_differentiable(torch, frame_bchw, mode, selector_module):
    """Differentiable mirror of runtime/frame0_selector.apply_pixel_mode, BCHW float.

    The shipped receiver applies ONE integer pixel operation to a sparse subset of
    frame_0 frames AFTER the carrier render (ra2c:207-210). Omitting it renders the
    wrong frame for those pairs -- caught here by the byte-identity control, which is
    exactly what the control is for.

    Every mode is differentiable a.e.: LUMA/CHANNEL/TILE are ``clip(x + delta, 0, 255)``
    (gradient 1 unclipped, a REAL 0 where clipped); ROLL is a permutation (exact
    gradient). Forward values are identical to the integer path because the incoming
    frame is already integer-valued in [0, 255].
    """
    sel = selector_module
    if mode.kind == sel.IDENTITY:
        return frame_bchw
    if mode.kind == sel.ROLL:
        # numpy applies shift=(b, a) over BHWC axes (1, 2) = (H, W); BCHW is (2, 3).
        return torch.roll(frame_bchw, shifts=(int(mode.b), int(mode.a)), dims=(2, 3))
    if mode.kind == sel.LUMA:
        delta = torch.full((1, 1, 1, 1), float(mode.a), dtype=frame_bchw.dtype)
    elif mode.kind == sel.CHANNEL:
        delta = torch.tensor(
            [float(mode.a), float(mode.b), float(mode.c)], dtype=frame_bchw.dtype
        ).reshape(1, 3, 1, 1)
    elif mode.kind == sel.TILE:
        height, width = frame_bchw.shape[-2], frame_bchw.shape[-1]
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.int32),
            torch.arange(width, dtype=torch.int32),
            indexing="ij",
        )
        if mode.a == 0:
            signs = ((yy + xx) & 1) * 2 - 1
        elif mode.a == 1:
            signs = (yy & 1) * 2 - 1
        elif mode.a == 2:
            signs = (xx & 1) * 2 - 1
        elif mode.a == 3:
            signs = (((yy >> 2) + (xx >> 2)) & 1) * 2 - 1
        else:
            raise SystemExit(f"invalid tile selector mode a={mode.a}")
        delta = (signs.to(frame_bchw.dtype) * float(mode.b))[None, None]
    else:
        raise SystemExit(f"unsupported F26 selector mode kind={mode.kind}")
    return (frame_bchw + delta).clamp(0.0, 255.0)


def posenet_pose6(torch, F, posenet, differentiable_rgb_to_yuv6, frame0_bchw, frame1_bchw):
    """upstream PoseNet.preprocess_input + forward, differentiably, first 6 pose dims.

    upstream/modules.py:71-75 -- rearrange (b t c h w) -> (b*t) c h w, interpolate to
    (384, 512) bilinear, rgb_to_yuv6, rearrange back to b (t c) h w. Verified at source
    2026-08-16: the interpolate happens BEFORE rgb_to_yuv6, and its target is literally
    ``segnet_model_input_size``.
    """
    stacked = torch.cat((frame0_bchw, frame1_bchw), dim=0)          # (2, 3, 874, 1164)
    resized = F.interpolate(stacked, size=(384, 512), mode="bilinear")
    yuv = differentiable_rgb_to_yuv6(resized)                        # (2, 6, 192, 256)
    posenet_in = yuv.reshape(1, 2 * 6, yuv.shape[-2], yuv.shape[-1])
    return posenet(posenet_in)["pose"][..., :POSE_ROWS]


def build_posenet(torch, upstream: Path):
    """Load the frozen CPU-torch PoseNet -- the authority instrument."""
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import modules as upstream_modules

    posenet = upstream_modules.PoseNet().eval()
    from safetensors.torch import load_file

    posenet.load_state_dict(load_file(str(upstream / "models" / "posenet.safetensors"), device="cpu"))
    for param in posenet.parameters():
        param.requires_grad_(False)
    return posenet


def gt_pair_iterator(upstream: Path, video: Path):
    """Stream GT pairs through the UPSTREAM decode path (yuv420_to_rgb, never PyAV rgb24).

    CLAUDE.md: GT decodes ONLY via frame_utils.yuv420_to_rgb -- PyAV rgb24 manufactures
    ~100x phantom pose. This uses upstream's own helper, unmodified.
    """
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import av
    from frame_utils import yuv420_to_rgb

    container = av.open(str(video))
    stream = container.streams.video[0]
    buf = []
    try:
        for frame in container.decode(stream):
            buf.append(yuv420_to_rgb(frame))
            if len(buf) == 2:
                yield buf[0], buf[1]
                buf = []
    finally:
        container.close()


def run_producer(args) -> int:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---- custody -------------------------------------------------------------
    archive = Path(args.archive)
    archive_sha = sha256_file(archive)
    archive_bytes = archive.stat().st_size
    if archive_sha != ARCHIVE_SHA or archive_bytes != ARCHIVE_BYTES:
        raise SystemExit(
            "CUSTODY REFUSED: archive sha/bytes differ from the pinned frontier\n"
            f"  got  {archive_sha} / {archive_bytes}\n"
            f"  want {ARCHIVE_SHA} / {ARCHIVE_BYTES}"
        )
    base_raw = Path(args.base_raw)
    expected_raw = N_PAIRS * 2 * FRAME_BYTES
    if base_raw.stat().st_size != expected_raw:
        raise SystemExit(f"base render is {base_raw.stat().st_size} B, expected {expected_raw} B")

    # ---- decode the carrier through the PROVEN chain --------------------------
    ra2b = load_ra2b()
    chain = ra2b.load_chain()
    renderer = chain[0]
    basis, coeff, _selector, provenance = ra2b.decode_carrier(archive, chain)
    if tuple(coeff.shape) != (N_PAIRS, CARRIER_DIM):
        raise SystemExit(f"unexpected coeff shape {tuple(coeff.shape)}")
    normalized_basis = renderer.normalized_basis(basis).detach()
    print(f"carrier decoded: basis {tuple(basis.shape)} coeff {tuple(coeff.shape)}", flush=True)

    # The sparse frame-0 selector runs AFTER the carrier render (ra2c:207-210). It is
    # part of the shipped frame_0, so it is part of the Jacobian's chain.
    import runtime.frame0_selector as selector_module

    decode_selector = chain[5]
    modes = sel_choices = None
    if _selector is not None:
        modes, sel_choices = decode_selector(_selector)
        if sel_choices.size != N_PAIRS:
            raise SystemExit(f"selector covers {sel_choices.size} frames, expected {N_PAIRS}")
        nontrivial = int(np.count_nonzero(sel_choices))
        print(f"selector: {nontrivial}/{N_PAIRS} frames carry a non-identity pixel mode", flush=True)

    posenet = build_posenet(torch, Path(args.upstream))
    raw = np.memmap(base_raw, dtype=np.uint8, mode="r", shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3))

    pairs = list(range(N_PAIRS)) if args.pairs == "all" else [int(p) for p in args.pairs.split(",")]
    n = len(pairs)

    if args.fd_check:
        jacs = np.load(Path(args.output) / "jacobian_pose6_x_coeff12.float64.npy")
        rows = [
            _fd_check(torch, F, differentiable_rgb_to_yuv6, posenet, normalized_basis,
                      coeff, raw, modes, sel_choices, selector_module, jacs, pid, step)
            for pid in [int(v) for v in args.fd_pairs.split(",")]
            for step in [float(v) for v in args.fd_steps.split(",")]
        ]
        receipt = {"schema": "ddm_jc1_fd_check.v1", "axis": AXIS, "score_claim": False,
                   "note": "central finite difference vs the STE Jacobian; see _fd_check",
                   "rows": rows}
        (Path(args.output) / "JC1_FD_CHECK.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2))
        return 0

    if args.eval_coeff:
        return _exact_eval(
            torch, F, differentiable_rgb_to_yuv6, posenet, renderer, normalized_basis,
            raw, pairs, modes, sel_choices, selector_module, args,
        )

    gt_iter = None
    gt_cache: dict[int, np.ndarray] = {}
    if args.with_gt:
        gt_iter = gt_pair_iterator(Path(args.upstream), Path(args.upstream) / "videos" / "0.mkv")

    jac = np.zeros((n, POSE_ROWS, CARRIER_DIM), dtype=np.float64)
    pose_gen = np.zeros((n, POSE_ROWS), dtype=np.float64)
    pose_gt = np.zeros((n, POSE_ROWS), dtype=np.float64)
    control_identical = np.zeros(n, dtype=bool)
    have_gt = np.zeros(n, dtype=bool)
    selector_mode = np.zeros((n, 4), dtype=np.int32)   # (kind, a, b, c) per pair

    started = time.time()
    gt_pos = 0
    for slot, pair_id in enumerate(pairs):
        frame1_np = np.asarray(raw[2 * pair_id + 1])
        frame1 = torch.from_numpy(frame1_np.copy()).permute(2, 0, 1)[None].float()

        mode = None if modes is None else modes[int(sel_choices[pair_id])]
        if mode is not None:
            selector_mode[slot] = (mode.kind, mode.a, mode.b, mode.c)

        # --- CONTROL: the hard-round forward must reproduce the shipped bytes ---
        with torch.no_grad():
            hard = render_frame0_differentiable(
                torch, F, normalized_basis, coeff[pair_id : pair_id + 1], ste=False
            )
            if mode is not None:
                hard = apply_pixel_mode_differentiable(torch, hard, mode, selector_module)
            hard_np = hard.to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]
        control_identical[slot] = bool(np.array_equal(hard_np, np.asarray(raw[2 * pair_id])))

        # --- the Jacobian, STE backward, exact forward -------------------------
        c = coeff[pair_id : pair_id + 1].clone().detach().requires_grad_(True)
        frame0 = render_frame0_differentiable(torch, F, normalized_basis, c, ste=True)
        if mode is not None:
            frame0 = apply_pixel_mode_differentiable(torch, frame0, mode, selector_module)
        # forward-value equality with the hard render is asserted, not assumed
        if not torch.equal(frame0.detach(), hard):
            raise SystemExit(f"STE forward value differs from hard forward at pair {pair_id}")
        pose6 = posenet_pose6(torch, F, posenet, differentiable_rgb_to_yuv6, frame0, frame1)
        pose_gen[slot] = pose6.detach().numpy()[0].astype(np.float64)
        for dim in range(POSE_ROWS):
            grad = torch.autograd.grad(
                pose6[0, dim], c, retain_graph=dim < POSE_ROWS - 1, create_graph=False
            )[0]
            jac[slot, dim] = grad.detach().numpy()[0].astype(np.float64)

        # --- GT pose for the per-pair residual r_i ------------------------------
        if gt_iter is not None:
            while gt_pos <= pair_id:
                g0, g1 = next(gt_iter)
                if gt_pos == pair_id:
                    gt_cache[pair_id] = (g0, g1)
                gt_pos += 1
            g0, g1 = gt_cache.pop(pair_id)
            # upstream frame_utils.yuv420_to_rgb returns a torch uint8 (H, W, 3) tensor.
            with torch.no_grad():
                gp = posenet_pose6(
                    torch,
                    F,
                    posenet,
                    differentiable_rgb_to_yuv6,
                    g0.permute(2, 0, 1)[None].float(),
                    g1.permute(2, 0, 1)[None].float(),
                )
            pose_gt[slot] = gp.numpy()[0].astype(np.float64)
            have_gt[slot] = True

        # Fail LOUD and EARLY: a control failure means the chain is wrong, and every
        # later pair would only add wrong rows.
        if slot + 1 >= args.control_abort_after and not control_identical[: slot + 1].all():
            bad = [pairs[i] for i in np.flatnonzero(~control_identical[: slot + 1])]
            raise SystemExit(
                f"CONTROL FAILED early: rendered frame_0 differs from the shipped 0.raw at "
                f"pairs {bad[:10]} (of first {slot + 1}). The chain is not the shipped chain."
            )

        if (slot + 1) % args.report_every == 0 or slot + 1 == n:
            elapsed = time.time() - started
            print(
                f"  pair {slot + 1}/{n} (id {pair_id})  {elapsed:.1f}s  "
                f"{elapsed / (slot + 1):.2f}s/pair  ctrl_ok={int(control_identical[: slot + 1].sum())}",
                flush=True,
            )
            _persist(out, pairs[: slot + 1], jac[: slot + 1], pose_gen[: slot + 1],
                     pose_gt[: slot + 1], have_gt[: slot + 1], control_identical[: slot + 1],
                     selector_mode[: slot + 1], provenance, archive_sha, archive_bytes, elapsed, partial=slot + 1 < n)

    elapsed = time.time() - started
    receipt = _persist(out, pairs, jac, pose_gen, pose_gt, have_gt, control_identical,
                       selector_mode, provenance, archive_sha, archive_bytes, elapsed, partial=False)
    print(json.dumps(receipt, indent=2))
    if not control_identical.all():
        bad = [pairs[i] for i in np.flatnonzero(~control_identical)]
        raise SystemExit(f"CONTROL FAILED: render differs from shipped bytes at pairs {bad[:10]}")
    return 0


def _fd_check(torch, F, differentiable_rgb_to_yuv6, posenet, normalized_basis, coeff,
              raw, modes, sel_choices, selector_module, jac, pair_id: int, step: float):
    """Central finite difference vs the STE Jacobian, one pair, all 12 coefficients.

    THE ONLY DIRECT CHECK ON THE GRADIENT ITSELF. The byte-identity and base-d_pose
    controls prove the FORWARD; nothing else proves the BACKWARD.

    The step must be large enough to move the quantized render. At a step below one
    quantization LSB the rendered frame is byte-identical and the true finite difference
    is EXACTLY ZERO while the STE Jacobian is not -- that is the quantization, not an
    error, and it is why the STE relaxation is needed at all. So this check reports
    agreement AT A FINITE STEP and is a statement about the relaxation's fidelity there,
    not a proof of a derivative that does not exist.
    """
    frame1 = torch.from_numpy(np.asarray(raw[2 * pair_id + 1]).copy())
    frame1 = frame1.permute(2, 0, 1)[None].float()
    mode = None if modes is None else modes[int(sel_choices[pair_id])]

    def pose_at(vec):
        with torch.no_grad():
            f0 = render_frame0_differentiable(torch, F, normalized_basis, vec, ste=False)
            if mode is not None:
                f0 = apply_pixel_mode_differentiable(torch, f0, mode, selector_module)
            return posenet_pose6(
                torch, F, posenet, differentiable_rgb_to_yuv6, f0.to(torch.uint8).float(),
                frame1,
            ).numpy()[0].astype(np.float64)

    base = coeff[pair_id : pair_id + 1].clone()
    fd = np.zeros((POSE_ROWS, CARRIER_DIM))
    for k in range(CARRIER_DIM):
        plus, minus = base.clone(), base.clone()
        plus[0, k] += step
        minus[0, k] -= step
        fd[:, k] = (pose_at(plus) - pose_at(minus)) / (2.0 * step)
    ste = jac[pair_id]
    denom = max(float(np.abs(ste).max()), 1e-30)
    return {
        "pair_id": int(pair_id),
        "step": float(step),
        "cosine_similarity": float(
            (fd * ste).sum() / (np.linalg.norm(fd) * np.linalg.norm(ste))
        ),
        "relative_frobenius_error": float(np.linalg.norm(fd - ste) / np.linalg.norm(ste)),
        "max_abs_error_over_max_abs_ste": float(np.abs(fd - ste).max() / denom),
        "norm_ratio_fd_over_ste": float(np.linalg.norm(fd) / np.linalg.norm(ste)),
    }


def _exact_eval(torch, F, differentiable_rgb_to_yuv6, posenet, renderer, normalized_basis,
                raw, pairs, modes, sel_choices, selector_module, args) -> int:
    """TRUE n600 d_pose for candidate coefficients -- no linearization anywhere.

    The producer's own positive control established that this forward reproduces
    upstream evaluate.py's pose pipeline to 1.5e-5 relative, so the number this emits is
    the scored quantity, not a proxy. d_seg needs no re-measurement: it is invariant to
    frame_0 carrier edits, MEASURED identically to 8 s.f. on three treatments.

    Uses the RETAINED pose6_groundtruth, so no GT decode is repeated.
    """
    payload = Path(args.output)
    pose_gt = np.load(payload / "pose6_groundtruth.float64.npy")
    if pose_gt.shape[0] != len(pairs):
        raise SystemExit("retained GT pose does not cover the requested pairs")

    results = []
    for spec in args.eval_coeff:
        path = Path(spec)
        cand = np.load(path).astype(np.float64)
        if cand.shape != (len(pairs), CARRIER_DIM):
            raise SystemExit(f"candidate {path.name} has shape {cand.shape}")
        cand_t = torch.from_numpy(cand).float()
        pose = np.zeros((len(pairs), POSE_ROWS), dtype=np.float64)
        started = time.time()
        with torch.no_grad():
            for slot, pair_id in enumerate(pairs):
                frame1 = torch.from_numpy(np.asarray(raw[2 * pair_id + 1]).copy())
                frame1 = frame1.permute(2, 0, 1)[None].float()
                frame0 = render_frame0_differentiable(
                    torch, F, normalized_basis, cand_t[pair_id : pair_id + 1], ste=False
                )
                if modes is not None:
                    frame0 = apply_pixel_mode_differentiable(
                        torch, frame0, modes[int(sel_choices[pair_id])], selector_module
                    )
                # The receiver writes uint8; make the quantization explicit, not implied.
                frame0 = frame0.to(torch.uint8).float()
                pose[slot] = posenet_pose6(
                    torch, F, posenet, differentiable_rgb_to_yuv6, frame0, frame1
                ).numpy()[0]
                if (slot + 1) % 100 == 0:
                    print(f"  {path.name} {slot + 1}/{len(pairs)} "
                          f"{time.time() - started:.0f}s", flush=True)
        d_pose = float(((pose - pose_gt) ** 2).sum(axis=1).mean() / POSE_ROWS)
        row = {
            "candidate": path.name,
            "candidate_sha256": sha256_file(path),
            "d_pose_measured": d_pose,
            "ratio_vs_base": d_pose / BASE_D_POSE,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    receipt = {
        "schema": "ddm_jc1_exact_candidate_eval.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "n_pairs": len(pairs),
        "base_d_pose": BASE_D_POSE,
        "base_d_seg": BASE_D_SEG,
        "d_seg_note": "invariant to frame_0 carrier edits; MEASURED identically on 3 treatments",
        "measurement_status": "EXACT_FORWARD_NO_LINEARIZATION",
        "rows": results,
    }
    target = payload / args.eval_output
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, target)
    print(json.dumps(receipt, indent=2))
    return 0


def _persist(out, pairs, jac, pose_gen, pose_gt, have_gt, control, selector_mode, provenance,
             archive_sha, archive_bytes, elapsed, *, partial):
    """ALWAYS KEEP THE PAYLOAD (P0). The arrays land on the SSD tier with their shas."""
    paths = {
        "jacobian_pose6_x_coeff12.float64.npy": np.ascontiguousarray(jac),
        "pose6_generated.float64.npy": np.ascontiguousarray(pose_gen),
        "pose6_groundtruth.float64.npy": np.ascontiguousarray(pose_gt),
        "have_gt.bool.npy": np.ascontiguousarray(have_gt),
        "control_byte_identical.bool.npy": np.ascontiguousarray(control),
        "selector_mode_kind_a_b_c.int32.npy": np.ascontiguousarray(selector_mode),
        "pair_ids.int32.npy": np.asarray(pairs, dtype=np.int32),
    }
    payload = {}
    for name, arr in paths.items():
        target = Path(out) / name
        tmp = target.with_name(target.name + ".tmp")
        # np.save appends '.npy' unless handed an open file object -- hand it one.
        with tmp.open("wb") as fh:
            np.save(fh, arr, allow_pickle=False)
        os.replace(tmp, target)
        payload[name] = {"sha256": sha256_file(target), "bytes": target.stat().st_size,
                         "shape": list(arr.shape), "dtype": str(arr.dtype)}
    receipt = {
        "schema": "ddm_jc1_carrier_pose_jacobian.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "partial": bool(partial),
        "seed": SEED,
        "n_pairs": len(pairs),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "carrier_provenance": {k: (str(v) if not isinstance(v, (int, float, bool)) else v)
                               for k, v in provenance.items()},
        "base_d_pose": BASE_D_POSE,
        "base_d_seg": BASE_D_SEG,
        "gradient_status": "MEASURED_ON_STE_RELAXED_CHAIN",
        "gradient_note": (
            "forward exact (hard round, byte-identical control vs shipped 0.raw); backward "
            "straight-through on both rounds; clamp gradient masks kept real. The exact "
            "quantized chain's Jacobian is 0 a.e. and carries no information."
        ),
        "control_byte_identical_count": int(np.asarray(control).sum()),
        "elapsed_seconds": round(elapsed, 3),
        "payload": payload,
        "payload_root": str(out),
    }
    tmp = Path(out) / "JC1_PRODUCER.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, Path(out) / "JC1_PRODUCER.json")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--base-raw", type=Path, default=DEFAULT_BASE_RAW)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pairs", type=str, default="all",
                        help="'all' for 0..599, or a comma-separated list (scope reduction only)")
    parser.add_argument("--with-gt", action="store_true",
                        help="also decode GT and record pose6_gt (needed for the re-fit residual)")
    parser.add_argument("--threads", type=int, default=0, help="torch threads; 0 = leave default")
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--eval-coeff", type=Path, nargs="*", default=None,
                        help="EXACT-eval mode: measure true n600 d_pose for these candidate "
                             "coefficient .npy files (no Jacobian, no linearization)")
    parser.add_argument("--eval-output", type=str, default="JC1_EXACT_CANDIDATES.json")
    parser.add_argument("--fd-check", action="store_true",
                        help="central finite difference vs the STE Jacobian (gradient check)")
    parser.add_argument("--fd-pairs", type=str, default="0,299,599")
    parser.add_argument("--fd-steps", type=str, default="0.05,0.02,0.01")
    parser.add_argument("--control-abort-after", type=int, default=20,
                        help="abort as soon as any byte-identity control fails past this many pairs")
    args = parser.parse_args()
    return run_producer(args)


if __name__ == "__main__":
    raise SystemExit(main())
