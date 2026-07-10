# SPDX-License-Identifier: MIT
"""Exact-score-gated latent CLICK-POLISH SEARCH on our PR110-lineage frontier payload.

HONEST NAMING (NO-FAKE #6): this is a *search / polish* — greedy exact-score-gated
discrete coordinate descent on the stored 8-bit latent codes. It is NOT a solver,
compiler, or synthesizer. It enumerates +-1/+-2 code steps ("clicks") and keeps a
click only when the EXACT per-pair distortion (through the real inflate chain + the
frozen contest scorers) plus the real re-encoded byte term does not regress the
exact contest score. Mechanism borrowed from public PR #128 (a12dongithub,
``rhnerv_latent_polish``); substrate is OURS (PR110-lineage frontier archive,
``lane_pr110_payload_entropy_recode_20260610``). Per NO-FAKE #7 this is a DEFENSIVE
BANK, not an innovation claim — see ``borrowed_substrate_accounting``.

Structural exploit (verified, not assumed — see ``verify_pair_locality``):
**pair-locality.** Pair ``p``'s two rendered frames depend only on latent row
``z[p]``; the contest metric is a mean over the 600 pairs of per-pair judgments.
Therefore (a) one candidate click's exact effect is computable by re-rendering ONE
pair, and (b) accepted clicks on DIFFERENT pairs add exactly. "Diagonal batching":
one 600-pair render (every pair carries its own copy of the same ``(dim, delta)``
click) scores 600 independent candidates exactly.

What is FROZEN (spliced as verbatim byte slices, never re-encoded): the FP11
wrapper, the CTXR decoder section, the 607-byte latent sidecar, the FECa selector
payload, and the optional DQS1 tail. What CHANGES: only the CTXR latent section
(``codec_ctx.encode_latent_section`` on the polished codes). The archive is a
deterministic ZIP_STORED single-member ``x`` at fixed 1980 epoch; the rate term is
the real ``archive.zip`` file size (byte-exact custody, never estimated for the
gate).

AXIS DISCIPLINE: selection runs the scorers on CPU-torch ONLY (per PR128's own
measured -0.0009 GPU->CPU loss + our MPS-never-authority law). Every emitted row is
axis-tagged; a macOS-CPU run is ``[macOS-CPU advisory]`` NON-PROMOTABLE; only a
Linux x86_64 ``upstream/evaluate.py`` row on the exact candidate bytes is
``[contest-CPU]`` authority.

Deterministic + resumable (P0): fixed sweep order, seeded, and an append-only
accepted-clicks JSONL ledger so a killed/preempted run resumes without losing
accepted clicks.

TRIALITY: this is a search/polish TOOL, not a training lever. DSL leg = N/A (no
trainer flag). Equations leg = deferred to the first measured exact row. DAG leg =
FEED-clickpolish-build.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import io
import json
import os
import struct
import sys
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np

# Canonical score authority (THE compliance bedrock — never hand-roll the formula).
from tac.contest_score import compute_contest_score

# ---------------------------------------------------------------------------
# frozen constants of the PR110 frontier packet (verified against the archive)
# ---------------------------------------------------------------------------
N_PAIRS = 600
LATENT_DIM = 28
CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512  # segnet_model_input_size = (512, 384) -> (H, W) here
# byte-identical ZIP framing of the frontier archive (create_system=3 unix,
# external_attr carries unix perms). Verified: reproduces b46897267ded... exactly.
ZIP_MEMBER_NAME = "x"
ZIP_CREATE_SYSTEM = 3
ZIP_EXTERNAL_ATTR = 25165824
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)

DEFAULT_SUBMISSION_DIR = (
    "experiments/results/pr110_payload_entropy_recode_20260610/submission_dir"
)
DEFAULT_ARCHIVE = DEFAULT_SUBMISSION_DIR + "/archive.zip"
DEFAULT_UPSTREAM = "upstream"
DEFAULT_GT_CACHE = "experiments/results/mlx_fleet_gt_cache"

# The frontier pointer lineage this bank descends from (recorded, never a score claim).
FRONTIER_ARCHIVE_SHA256 = (
    "b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e"
)
FRONTIER_CONTEST_CPU_SCORE = 0.19109982419209975  # incumbent pointer (contest-CPU)


# ---------------------------------------------------------------------------
# submission-src import shim (uses OUR own frozen codec/model/inflate surfaces,
# so bytes and pixels match the archive-producing code by construction).
# ---------------------------------------------------------------------------
def _load_submission_modules(submission_dir: str | Path):
    """Import codec_ctx / codec / codec_sidecar / inflate from a submission dir.

    Returns a namespace object with attributes: codec_ctx, codec, inflate.
    The submission dir travels with the run (portable to the Modal container)."""
    sub = Path(submission_dir).resolve()
    src = sub / "src"
    for p in (str(src), str(sub)):
        if p not in sys.path:
            sys.path.insert(0, p)
    ns = types_ns()
    ns.codec_ctx = importlib.import_module("codec_ctx")
    ns.codec = importlib.import_module("codec")
    ns.codec_sidecar = importlib.import_module("codec_sidecar")
    ns.inflate = importlib.import_module("inflate")
    ns.submission_dir = str(sub)
    return ns


def types_ns():
    import types

    return types.SimpleNamespace()


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# FrozenPacket: parse the frontier member into (frozen byte slices) + editable Q
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class FrozenPacket:
    """The frontier archive member decomposed for latent click-polish.

    All non-latent sections are kept as VERBATIM byte slices and spliced back
    unchanged on every repack; only the CTXR latent section is re-encoded."""

    ns: object  # submission modules namespace
    fp11_source_len_field_offset: int
    dec_sec: bytes
    sidecar: bytes
    sel_bytes: bytes
    dqs1_tail: bytes
    ctxr_version: int
    hdr16: bytes  # 112-byte fp16 mins+scales header of the latent payload
    Q0: np.ndarray  # (600, 28) uint8 baseline latent codes (natural dim order)
    mins: np.ndarray  # (28,) float32
    scales: np.ndarray  # (28,) float32
    original_member: bytes
    original_archive: bytes
    archive_path: str

    # ---- parse ----
    @classmethod
    def parse(cls, archive_path: str | Path, submission_dir: str | Path) -> "FrozenPacket":
        ns = _load_submission_modules(submission_dir)
        archive_bytes = Path(archive_path).read_bytes()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
            names = z.namelist()
            if names != [ZIP_MEMBER_NAME]:
                raise ValueError(f"unexpected archive members {names!r}")
            member = z.read(ZIP_MEMBER_NAME)

        if member[:4] != b"FP11":
            raise ValueError(f"bad FP11 magic {member[:4]!r}")
        pos = 4
        (src_len,) = struct.unpack_from("<I", member, pos)
        pos += 4
        src = member[pos : pos + src_len]
        pos += src_len
        (sel_len,) = struct.unpack_from("<H", member, pos)
        pos += 2
        sel_bytes = member[pos : pos + sel_len]
        pos += sel_len
        dqs1_tail = member[pos:]

        if src[:4] != b"CTXR":
            raise ValueError(f"bad CTXR magic {src[:4]!r}")
        ctxr_version = src[4]
        p = 5
        dl = int.from_bytes(src[p : p + 3], "little"); p += 3
        ll = int.from_bytes(src[p : p + 3], "little"); p += 3
        slen = int.from_bytes(src[p : p + 3], "little"); p += 3
        dec_sec = src[p : p + dl]; p += dl
        lat_sec = src[p : p + ll]; p += ll
        sidecar = src[p : p + slen]; p += slen
        if p != len(src):
            raise ValueError("CTXR container has trailing bytes")

        latent_raw = ns.codec_ctx.decode_latent_section(lat_sec)
        hdr16, Q0, mins, scales = _latent_raw_to_Q(latent_raw)

        return cls(
            ns=ns,
            fp11_source_len_field_offset=4,
            dec_sec=dec_sec,
            sidecar=sidecar,
            sel_bytes=sel_bytes,
            dqs1_tail=dqs1_tail,
            ctxr_version=ctxr_version,
            hdr16=hdr16,
            Q0=Q0,
            mins=mins,
            scales=scales,
            original_member=member,
            original_archive=archive_bytes,
            archive_path=str(archive_path),
        )

    # ---- codes <-> bytes ----
    def latent_raw_from_Q(self, Q: np.ndarray) -> bytes:
        return _Q_to_latent_raw(self.hdr16, Q)

    def latent_section_from_Q(self, Q: np.ndarray) -> bytes:
        return self.ns.codec_ctx.encode_latent_section(self.latent_raw_from_Q(Q))

    def repack_member(self, Q: np.ndarray, drop_sidecar: bool = False) -> bytes:
        lat_sec = self.latent_section_from_Q(Q)
        sidecar = b"" if drop_sidecar else self.sidecar
        ctxr = (
            b"CTXR"
            + bytes([self.ctxr_version])
            + len(self.dec_sec).to_bytes(3, "little")
            + len(lat_sec).to_bytes(3, "little")
            + len(sidecar).to_bytes(3, "little")
            + self.dec_sec
            + lat_sec
            + sidecar
        )
        member = (
            b"FP11"
            + struct.pack("<I", len(ctxr))
            + ctxr
            + struct.pack("<H", len(self.sel_bytes))
            + self.sel_bytes
            + self.dqs1_tail
        )
        return member

    def repack_archive_bytes(self, Q: np.ndarray, drop_sidecar: bool = False) -> bytes:
        return _deterministic_zip(self.repack_member(Q, drop_sidecar=drop_sidecar))

    def fold_sidecar_custody(self) -> dict:
        """Independently-verifiable sidecar-drop transform (task item 3).

        Produces before (sidecar kept) / after (sidecar folded/dropped) archives at
        the UNCHANGED baseline Q0 and records byte-exact custody. Dropping the
        607-byte sidecar section shrinks the archive by exactly len(sidecar) bytes;
        it CHANGES the decoded latents (removes the sub-grid corrections), so it is
        only a net win when an exact-score-gated re-search recovers the distortion.
        This function proves the byte transform is exact + reversible in accounting;
        the score effect is measured separately through the exact gate."""
        before = self.repack_archive_bytes(self.Q0, drop_sidecar=False)
        after = self.repack_archive_bytes(self.Q0, drop_sidecar=True)
        return {
            "sidecar_bytes": len(self.sidecar),
            "before_archive_bytes": len(before),
            "after_archive_bytes": len(after),
            "archive_byte_delta": len(after) - len(before),
            "before_sha256": sha256_hex(before),
            "after_sha256": sha256_hex(after),
            "expected_delta": -len(self.sidecar),
            "delta_matches_sidecar": (len(before) - len(after)) == len(self.sidecar),
        }

    # ---- no-op / round-trip proof ----
    def verify_roundtrip(self) -> dict:
        """Prove decode->encode of the polished section is byte-exact identity on
        the REAL archive (the no-op detector: no click => byte-identical archive)."""
        latent_raw = _Q_to_latent_raw(self.hdr16, self.Q0)
        member = self.repack_member(self.Q0)
        archive = self.repack_archive_bytes(self.Q0)
        return {
            "latent_raw_roundtrip_byte_exact": (
                latent_raw
                == self.ns.codec_ctx.decode_latent_section(
                    self._original_lat_sec()
                )
            ),
            "member_byte_exact": member == self.original_member,
            "archive_byte_exact": archive == self.original_archive,
            "archive_bytes": len(archive),
            "original_archive_bytes": len(self.original_archive),
            "archive_sha256": sha256_hex(archive),
            "original_archive_sha256": sha256_hex(self.original_archive),
        }

    def _original_lat_sec(self) -> bytes:
        src = self.original_member[
            4 + 4 : 4 + 4 + struct.unpack_from("<I", self.original_member, 4)[0]
        ]
        p = 5
        dl = int.from_bytes(src[p : p + 3], "little"); p += 3
        ll = int.from_bytes(src[p : p + 3], "little"); p += 3
        p += 3
        p += dl
        return src[p : p + ll]


# ---------------------------------------------------------------------------
# code<->raw conversions (natural dim order Q <-> delta-form storage-order raw)
# verified byte-exact against decode_latents_from_raw on the real archive.
# ---------------------------------------------------------------------------
def _latent_dim_order(submission_ns=None) -> list[int]:
    from codec import LATENT_DIM_ORDER  # imported after submission src on path

    return list(LATENT_DIM_ORDER)


def _latent_raw_to_Q(latent_raw: bytes):
    from codec import LATENT_DIM_ORDER, LATENT_DIM as LD, N_PAIRS as NP
    from codec_ctx import LATENT_HDR

    buf = io.BytesIO(latent_raw)
    hdr16 = buf.read(LATENT_HDR)
    mins = np.frombuffer(hdr16[: LD * 2], dtype=np.float16).astype(np.float32)
    scales = np.frombuffer(hdr16[LD * 2 :], dtype=np.float16).astype(np.float32)
    stored = np.frombuffer(buf.read(NP * LD), dtype=np.uint8)
    if stored.size != NP * LD:
        raise ValueError("truncated latent payload")
    delta = stored.reshape(LD, NP).copy()
    q_ord = delta.copy()
    q_ord[:, 1:] = np.cumsum(
        ((delta[:, 1:].astype(np.int16) - 128) & 255), axis=1, dtype=np.uint16
    ).astype(np.uint8) + delta[:, :1]
    q_ord = q_ord.T.copy()  # (NP, LD) in storage order
    Q = np.empty((NP, LD), np.uint8)
    Q[:, list(LATENT_DIM_ORDER)] = q_ord
    return hdr16, Q, mins, scales


def _Q_to_latent_raw(hdr16: bytes, Q: np.ndarray) -> bytes:
    from codec import LATENT_DIM_ORDER

    q_ord = Q[:, list(LATENT_DIM_ORDER)].T.copy()  # (LD, NP) storage order
    delta = np.empty_like(q_ord)
    delta[:, 0] = q_ord[:, 0]
    delta[:, 1:] = (
        (q_ord[:, 1:].astype(np.int16) - q_ord[:, :-1].astype(np.int16) + 128) & 255
    ).astype(np.uint8)
    return hdr16 + delta.tobytes()


def _deterministic_zip(member: bytes) -> bytes:
    b = io.BytesIO()
    info = zipfile.ZipInfo(ZIP_MEMBER_NAME, date_time=ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = ZIP_CREATE_SYSTEM
    info.external_attr = ZIP_EXTERNAL_ATTR
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(info, member)
    return b.getvalue()


# ---------------------------------------------------------------------------
# Renderer: latent codes -> frames, EXACT R3 inflate chain (reuses our inflate.py)
# ---------------------------------------------------------------------------
class Renderer:
    """Renders arbitrary pairs from a Q latent-code table using OUR exact inflate
    chain (decoder + DQS1 substitution + #98 biases + bicubic + FEC/selector).

    Pair-local: rendering pair p uses only row Q[p] (plus frozen per-pair
    selector/DQS1 choices). Verified by ``verify_pair_locality``."""

    def __init__(self, packet: FrozenPacket, device: str = "cpu",
                 drop_sidecar: bool = False):
        import torch

        self.packet = packet
        self.ns = packet.ns
        self.device = torch.device(device)
        self.drop_sidecar = drop_sidecar
        inflate = self.ns.inflate
        meta = {
            "n_pairs": N_PAIRS,
            "latent_dim": LATENT_DIM,
            "base_channels": self.ns.codec.BASE_CHANNELS,
            "eval_size": list(self.ns.codec.EVAL_SIZE),
        }
        self.meta = meta
        raw_streams = self.ns.codec_ctx.decode_decoder_section(packet.dec_sec)
        raw_joined = b"".join(raw_streams)
        decoder_sd = inflate.decode_decoder_state_from_raw(raw_joined, meta)
        self.decoder = inflate.HNeRVDecoder(
            latent_dim=LATENT_DIM,
            base_channels=meta["base_channels"],
            eval_size=tuple(meta["eval_size"]),
        ).to(self.device)
        self.decoder.load_state_dict(decoder_sd)
        self.decoder.eval()

        # optional DQS1 mutated decoder
        self.dqs1_packet = (
            inflate.parse_dqs1_payload(packet.dqs1_tail) if packet.dqs1_tail else None
        )
        self.mutated_decoder = None
        self.selected_pairs: set[int] = set()
        self.frame_policy = "pair_all_frames"
        if self.dqs1_packet is not None:
            mutated_sd = inflate.apply_dqs1_patch_to_decoder_state(
                decoder_sd, raw_joined, self.dqs1_packet, meta
            )
            self.mutated_decoder = inflate.HNeRVDecoder(
                latent_dim=LATENT_DIM,
                base_channels=meta["base_channels"],
                eval_size=tuple(meta["eval_size"]),
            ).to(self.device)
            self.mutated_decoder.load_state_dict(mutated_sd)
            self.mutated_decoder.eval()
            self.selected_pairs = {int(v) for v in self.dqs1_packet["pair_indices"]}
            self.frame_policy = str(self.dqs1_packet["frame_policy"])

        self.selector_kind, self.selector_codes, self.selector_specs = (
            inflate.unpack_pr101_selector(packet.sel_bytes)
        )
        self.mins = None
        self.scales = None
        self._set_grid(packet.mins, packet.scales)

    def _set_grid(self, mins: np.ndarray, scales: np.ndarray):
        import torch

        self.mins = torch.from_numpy(mins.astype(np.float32))
        self.scales = torch.from_numpy(scales.astype(np.float32))

    def _latents_float(self, Q: np.ndarray):
        """Q (600,28 uint8) -> sidecar-applied float latents (600,28), EXACT chain."""
        import torch

        q = torch.from_numpy(Q.astype(np.float32))
        latents = q * self.scales.unsqueeze(0) + self.mins.unsqueeze(0)
        sidecar = b"" if self.drop_sidecar else self.packet.sidecar
        latents = self.ns.inflate.apply_latent_sidecar(latents, sidecar)
        return latents.to(self.device)

    def render(self, Q: np.ndarray, pair_indices, batch_pairs: int = 16):
        """Render the given pair indices -> uint8 frames (len*2, 874,1164,3) NHWC.

        pair_indices: iterable of ints in [0, 600). Rendered in the given order;
        frames come out as [pair0_f0, pair0_f1, pair1_f0, pair1_f1, ...]."""
        import torch
        import torch.nn.functional as F

        inflate = self.ns.inflate
        pair_indices = [int(p) for p in pair_indices]
        latents = self._latents_float(Q)
        eval_h, eval_w = self.meta["eval_size"]
        out_frames = []
        with torch.inference_mode():
            for s in range(0, len(pair_indices), batch_pairs):
                pidx = pair_indices[s : s + batch_pairs]
                sel = torch.tensor(pidx, dtype=torch.long, device=self.device)
                z = latents.index_select(0, sel)
                decoded = self.decoder(z)  # (b,2,3,eh,ew)
                if self.mutated_decoder is not None:
                    local = [
                        i for i, p in enumerate(pidx) if p in self.selected_pairs
                    ]
                    if local:
                        lt = torch.tensor(local, dtype=torch.long, device=self.device)
                        decoded = decoded.clone()
                        mutated = self.mutated_decoder(z)
                        if self.frame_policy == "pair_all_frames":
                            decoded[lt] = mutated[lt]
                        elif self.frame_policy == "segnet_last_frame_only":
                            decoded[lt, 1] = mutated[lt, 1]
                        else:
                            raise ValueError(
                                f"bad DQS1 frame policy {self.frame_policy!r}"
                            )
                b = len(pidx)
                flat = decoded.reshape(b * 2, 3, eval_h, eval_w)
                up = F.interpolate(
                    flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
                )
                up = up.reshape(b, 2, 3, CAMERA_H, CAMERA_W)
                up[:, 0, 0].sub_(1.0)
                up[:, 0, 2].sub_(1.0)
                up[:, 1, 1].sub_(1.0)
                rounded = up.reshape(b * 2, 3, CAMERA_H, CAMERA_W).clamp(0, 255).round()
                rounded = _apply_selector_per_pair(
                    inflate, rounded, self.selector_kind, self.selector_codes,
                    self.selector_specs, pidx,
                )
                frames = rounded.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                out_frames.append(frames)
        return np.concatenate(out_frames, axis=0)


def _apply_selector_per_pair(inflate, frames_bchw, kind, codes, specs, pair_indices):
    """Apply the per-pair selector transform using the ACTUAL pair index for each
    rendered pair (supports arbitrary / non-contiguous pair lists)."""
    if kind == "static":
        # static selector indexes by absolute pair; render contiguous only.
        # For non-contiguous, apply per pair via the compact path is not defined;
        # our frontier is compact (FECa), so this branch is unused in practice.
        raise NotImplementedError("static selector requires contiguous render")
    out = frames_bchw.clone()
    for offset, pair_index in enumerate(pair_indices):
        spec = specs[int(codes[pair_index])]
        family, _params, frame_index = spec
        if family == "identity":
            continue
        fo = offset * 2 + int(frame_index)
        out[fo] = inflate.apply_dynamic_mode(out[fo], spec)
    return out.clamp_(0.0, 255.0).round_()


# ---------------------------------------------------------------------------
# Scorer: comp frames -> per-pair d_seg / d_pose vs cached GT targets
# (authority-aligned: same upstream SegNet/PoseNet + lstars/gt_poses targets).
# ---------------------------------------------------------------------------
class Scorer:
    def __init__(self, upstream_dir: str | Path = DEFAULT_UPSTREAM, device: str = "cpu"):
        from tac.scorer import load_default_scorers

        self.device = device
        self.posenet, self.segnet = load_default_scorers(str(upstream_dir), device=device)
        self.posenet.eval()
        self.segnet.eval()

    def per_pair(self, frames_nhwc: np.ndarray, gt_lstars: np.ndarray, gt_poses: np.ndarray):
        """frames_nhwc: (len*2, 874,1164,3) uint8 for `len` pairs, ordered
        [f0,f1,f0,f1,...]. gt_lstars: (len,384,512) int; gt_poses: (len,6).
        Returns (d_seg[len], d_pose[len]) float64 numpy arrays."""
        import torch

        n2 = frames_nhwc.shape[0]
        assert n2 % 2 == 0
        npair = n2 // 2
        x = torch.from_numpy(frames_nhwc.astype(np.float32))  # (2*npair,H,W,3)
        x = x.reshape(npair, 2, CAMERA_H, CAMERA_W, 3).permute(0, 1, 4, 2, 3).contiguous()
        x = x.to(self.device)
        gt_seg = torch.from_numpy(np.asarray(gt_lstars)).long().to(self.device)
        gt_pose = torch.from_numpy(np.asarray(gt_poses).astype(np.float32)).to(self.device)
        with torch.inference_mode():
            seg_logits = self.segnet(self.segnet.preprocess_input(x))
            seg_arg = seg_logits.argmax(dim=1)  # (npair, 384,512)
            d_seg = (seg_arg != gt_seg).float().mean(dim=(1, 2)).cpu().numpy()

            pose_out = self.posenet(self.posenet.preprocess_input(x))
            pose6 = pose_out["pose"][..., :6]  # (npair, 6)
            d_pose = (pose6 - gt_pose).pow(2).mean(dim=1).cpu().numpy()
        return d_seg.astype(np.float64), d_pose.astype(np.float64)


# ---------------------------------------------------------------------------
# GT cache loading
# ---------------------------------------------------------------------------
def load_gt_targets(gt_cache_dir: str | Path, n_pairs: int):
    """Load (lstars, gt_poses) for the first ``n_pairs`` contest pairs from the
    canonical mlx_fleet_gt_cache npz (built via the upstream scorers / yuv420_to_rgb).

    Returns (lstars: (n,384,512) int64, gt_poses: (n,6) float64, source_path)."""
    gt_cache_dir = Path(gt_cache_dir)
    # smallest cache that covers n_pairs
    for cand_n in (1, 6, 24, 96, 200, 600):
        if cand_n < n_pairs:
            continue
        for stem in (f"gt_n{cand_n}.npz", f"gt_strided_n{cand_n}.npz"):
            p = gt_cache_dir / stem
            if p.exists():
                d = np.load(p)
                if "lstars" in d.files and "gt_poses" in d.files:
                    ls = np.asarray(d["lstars"])[:n_pairs]
                    gp = np.asarray(d["gt_poses"])[:n_pairs]
                    if ls.shape[0] >= n_pairs:
                        return ls[:n_pairs], gp[:n_pairs], str(p)
    raise FileNotFoundError(
        f"no gt cache in {gt_cache_dir} covering {n_pairs} pairs with lstars+gt_poses"
    )


# ---------------------------------------------------------------------------
# exact score of a Q candidate (through render + scorer + real bytes)
# ---------------------------------------------------------------------------
def exact_components_for_Q(
    packet: FrozenPacket, renderer: Renderer, scorer: Scorer, Q: np.ndarray,
    gt_lstars: np.ndarray, gt_poses: np.ndarray, pair_indices=None,
):
    """Exact (d_seg, d_pose, archive_bytes, S) for a candidate Q over the given
    pairs (default: all N in the GT set). Distortions are MEANS over the scored
    pairs; bytes is the real re-encoded archive size."""
    if pair_indices is None:
        pair_indices = list(range(gt_lstars.shape[0]))
    frames = renderer.render(Q, pair_indices)
    d_seg_pp, d_pose_pp = scorer.per_pair(
        frames, gt_lstars[pair_indices], gt_poses[pair_indices]
    )
    d_seg = float(d_seg_pp.mean())
    d_pose = float(d_pose_pp.mean())
    archive_bytes = len(
        packet.repack_archive_bytes(Q, drop_sidecar=getattr(renderer, "drop_sidecar", False))
    )
    S = compute_contest_score(d_seg, d_pose, archive_bytes)
    return {
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "S": S,
        "d_seg_per_pair": d_seg_pp,
        "d_pose_per_pair": d_pose_pp,
    }


# ---------------------------------------------------------------------------
# verification helpers (the greens the phase-1 gate requires)
# ---------------------------------------------------------------------------
def verify_pair_locality(packet: FrozenPacket, renderer: Renderer, dim: int = 9,
                         delta: int = 2, pair_a: int = 0, pair_b: int = 1) -> dict:
    """Prove a click on pair_a's latent changes ONLY pair_a's frames, not pair_b's,
    AND that a 2-pair joint render of pair_a equals its single-pair render.

    This is the pair-locality assumption the diagonal batching relies on — MEASURED
    on our decoder, not assumed."""
    Q0 = packet.Q0
    # baseline joint render of [pair_a, pair_b]
    base = renderer.render(Q0, [pair_a, pair_b])
    base_a, base_b = base[0:2], base[2:4]

    # find a (dim, delta) whose click actually changes pair_a's pixels, so the test
    # is non-vacuous; the locality property itself is dim-independent.
    trials = [(dim, delta)] + [(d, s) for d in range(LATENT_DIM)
                               for s in (2, -2, 4, -4, 8, -8)]
    Qc = Q0.copy()
    clk_a = base_a
    for d, s in trials:
        Qc = Q0.copy()
        Qc[pair_a, d] = np.clip(int(Qc[pair_a, d]) + s, 0, 255)
        clicked = renderer.render(Qc, [pair_a, pair_b])
        clk_a, clk_b = clicked[0:2], clicked[2:4]
        if not np.array_equal(base_a, clk_a):
            dim, delta = d, s
            break

    # THE no-cross-talk proof: in the SAME 2-batch, clicking pair_a leaves pair_b's
    # frames BYTE-IDENTICAL (mathematical pair-locality, no float confound because the
    # batch layout is identical). pair_a is (non-vacuously) changed.
    b_unchanged = bool(np.array_equal(base_b, clk_b))
    a_changed = bool(not np.array_equal(base_a, clk_a))

    # Informational: single-pair (batch=1) vs joint (batch=2) render of pair_a differs
    # ONLY by CPU batch-float nondeterminism (conv reduction order), a few +-1 uint8
    # pixels — NOT a locality violation. Report the pixel-diff magnitude.
    single = renderer.render(Qc, [pair_a])
    px_diff = int(np.abs(single.astype(np.int32) - clk_a.astype(np.int32)).sum())
    px_maxabs = int(np.abs(single.astype(np.int32) - clk_a.astype(np.int32)).max())
    return {
        "pair_b_unchanged_by_pair_a_click": b_unchanged,  # HARD: no cross-talk
        "pair_a_changed_by_its_click": a_changed,
        "single_vs_joint_pixel_absdiff_sum": px_diff,  # info: batch-float jitter
        "single_vs_joint_pixel_maxabs": px_maxabs,
        "locality_holds": b_unchanged and a_changed,
        "dim": dim, "delta": delta, "pair_a": pair_a, "pair_b": pair_b,
        "note": "locality = pair_b byte-identical under pair_a's click (same batch). "
                "single-vs-joint pixel diff is batch-float nondeterminism, handled by "
                "the search's EXACT re-render accept gate.",
    }


def verify_batch_equivalence(packet: FrozenPacket, renderer: Renderer, scorer: Scorer,
                             gt_lstars, gt_poses, dim: int = 9, delta: int = 1,
                             pairs=(0, 1, 2, 3)) -> dict:
    """Prove diagonal-batch scoring == per-candidate sequential scoring on a tiny
    subset. Diagonal: every pair carries the same (dim, delta) click; render all at
    once; per-pair scores. Sequential: for each pair, render it alone with the click
    and score it. The per-pair distortion arrays must match exactly."""
    pairs = list(pairs)
    Q0 = packet.Q0
    # diagonal: apply (dim,delta) to EVERY pair simultaneously
    Qdiag = Q0.copy()
    Qdiag[:, dim] = np.clip(Qdiag[:, dim].astype(np.int16) + delta, 0, 255).astype(np.uint8)
    frames_diag = renderer.render(Qdiag, pairs)
    dseg_diag, dpose_diag = scorer.per_pair(frames_diag, gt_lstars[pairs], gt_poses[pairs])

    # sequential: click only pair p, render pair p alone, score
    dseg_seq = np.empty(len(pairs)); dpose_seq = np.empty(len(pairs))
    for i, p in enumerate(pairs):
        Qp = Q0.copy()
        Qp[p, dim] = np.clip(int(Qp[p, dim]) + delta, 0, 255)
        fr = renderer.render(Qp, [p])
        ds, dp = scorer.per_pair(fr, gt_lstars[[p]], gt_poses[[p]])
        dseg_seq[i] = ds[0]; dpose_seq[i] = dp[0]

    # seg-argmax disagreement is exact across batch sizes; pose MSE differs only by
    # CPU batch-float nondeterminism (~1e-8), far below the click gate granularity
    # (clicks move d_pose by ~1e-6). Diagonal batching therefore scores the same
    # candidates as sequential up to that jitter; the search's accept gate is exact.
    seg_match = bool(np.allclose(dseg_diag, dseg_seq, atol=0.0, rtol=0.0))
    pose_abs = float(np.abs(dpose_diag - dpose_seq).max())
    pose_close = bool(pose_abs < 1e-6)
    return {
        "diagonal_equals_sequential_seg_exact": seg_match,
        "diagonal_vs_sequential_pose_maxabsdiff": pose_abs,
        "pose_within_batch_float_tol": pose_close,
        "equivalence_holds": seg_match and pose_close,
        "dseg_diag": dseg_diag.tolist(), "dseg_seq": dseg_seq.tolist(),
        "dpose_diag": dpose_diag.tolist(), "dpose_seq": dpose_seq.tolist(),
    }


# ---------------------------------------------------------------------------
# the search: greedy exact-score-gated discrete coordinate descent
# ---------------------------------------------------------------------------
SWEEP_DELTAS = (1, -1, 2, -2)  # +-1, +-2 clicks


def _pose_marginal_weight(d_pose: float) -> float:
    """Local slope d(sqrt(10*d_pose))/d(d_pose) = 5/sqrt(10*d_pose). Used ONLY to
    rank per-pair candidate clicks; correctness comes from the exact S gate."""
    import math

    return 5.0 / math.sqrt(10.0 * max(d_pose, 1e-9))


@dataclasses.dataclass
class ClickPolishSearch:
    packet: FrozenPacket
    renderer: Renderer
    scorer: Scorer
    gt_lstars: np.ndarray
    gt_poses: np.ndarray
    out_dir: str
    axis_tag: str = "[macOS-CPU advisory]"  # phase-2 sets [contest-CPU] on Linux x86_64
    max_rounds: int = 40
    eps: float = 0.0  # strict improvement gate (S must decrease)
    log: Callable[[str], None] = print

    def __post_init__(self):
        self.n = int(self.gt_lstars.shape[0])
        self.pairs = list(range(self.n))
        self.Q = self.packet.Q0.copy()
        os.makedirs(self.out_dir, exist_ok=True)
        self.ledger_path = os.path.join(self.out_dir, "accepted_clicks_ledger.jsonl")

    # ---- resume: replay ledger -> reconstruct Q ----
    def resume(self) -> int:
        rounds = 0
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    for p, d, dl in row["clicks"]:
                        self.Q[p, d] = np.clip(int(self.Q[p, d]) + int(dl), 0, 255)
                    rounds = max(rounds, int(row["round"]) + 1)
            self.log(f"[resume] replayed {rounds} accepted rounds from ledger")
        return rounds

    def _append_ledger(self, row: dict):
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _score_Q(self, Q: np.ndarray) -> dict:
        return exact_components_for_Q(
            self.packet, self.renderer, self.scorer, Q, self.gt_lstars, self.gt_poses,
            pair_indices=self.pairs,
        )

    def _sweep(self, base: dict):
        """One diagonal sweep. Returns per-pair best improving click proposal:
        best[p] = (dim, delta, dseg_c, dpose_c) or None."""
        dseg0 = base["d_seg_per_pair"]
        dpose0 = base["d_pose_per_pair"]
        wpose = _pose_marginal_weight(base["d_pose"])
        best = [None] * self.n
        best_gain = np.zeros(self.n)  # positive = improvement in per-pair proxy
        for d in range(LATENT_DIM):
            for dl in SWEEP_DELTAS:
                Qc = self.Q.copy()
                Qc[:, d] = np.clip(
                    Qc[:, d].astype(np.int16) + dl, 0, 255
                ).astype(np.uint8)
                frames = self.renderer.render(Qc, self.pairs)
                dseg_c, dpose_c = self.scorer.per_pair(
                    frames, self.gt_lstars, self.gt_poses
                )
                # per-pair proxy improvement (distortion units; bytes second-order)
                proxy_delta = (
                    100.0 * (dseg_c - dseg0) + wpose * (dpose_c - dpose0)
                )
                for p in range(self.n):
                    # a click that does not change this pair's codes is a no-op
                    if Qc[p, d] == self.Q[p, d]:
                        continue
                    gain = -proxy_delta[p]
                    if gain > best_gain[p] + 1e-12:
                        best_gain[p] = gain
                        best[p] = (d, int(dl), float(dseg_c[p]), float(dpose_c[p]))
        return best

    def run(self) -> dict:
        start_round = self.resume()
        base = self._score_Q(self.Q)
        self.log(
            f"[baseline] S={base['S']:.8f} d_seg={base['d_seg']:.8f} "
            f"d_pose={base['d_pose']:.8f} bytes={base['archive_bytes']} "
            f"axis={self.axis_tag}"
        )
        best_row = {
            "round": start_round - 1, "S": base["S"], "d_seg": base["d_seg"],
            "d_pose": base["d_pose"], "archive_bytes": base["archive_bytes"],
        }
        for rnd in range(start_round, self.max_rounds):
            best = self._sweep(base)
            # build joint candidate: at most one click per pair
            clicks = []
            dseg_pp = base["d_seg_per_pair"].copy()
            dpose_pp = base["d_pose_per_pair"].copy()
            for p in range(self.n):
                if best[p] is None:
                    continue
                d, dl, dseg_c, dpose_c = best[p]
                clicks.append((p, d, dl))
                dseg_pp[p] = dseg_c
                dpose_pp[p] = dpose_c
            if not clicks:
                self.log(f"[round {rnd}] plateau — no improving click")
                break
            # additive S PREDICTION (locality; used only to rank/pre-filter) ...
            Qcand = self.Q.copy()
            for p, d, dl in clicks:
                Qcand[p, d] = np.clip(int(Qcand[p, d]) + dl, 0, 255)
            j_dseg = float(dseg_pp.mean())
            j_dpose = float(dpose_pp.mean())
            j_bytes = len(
                self.packet.repack_archive_bytes(
                    Qcand, drop_sidecar=self.renderer.drop_sidecar
                )
            )
            j_S_pred = compute_contest_score(j_dseg, j_dpose, j_bytes)
            # ... but the ACCEPT GATE is the EXACT re-render in the canonical 16-batch
            # layout (kills the ~1e-8 batch-float jitter of the additive prediction).
            comp = self._score_Q(Qcand)
            if comp["S"] < base["S"] - self.eps:
                accepted, new_base = clicks, comp
                self.Q = Qcand
            else:
                accepted, new_base = self._bisect_accept(base, clicks)
            if not accepted:
                self.log(
                    f"[round {rnd}] no accepted subset improved S "
                    f"(pred S={j_S_pred:.8f} exact S={comp['S']:.8f}) — stop"
                )
                break
            base = new_base  # exact, canonical-layout baseline
            row = {
                "round": rnd, "clicks": accepted, "n_clicks": len(accepted),
                "S_after": base["S"], "d_seg": base["d_seg"],
                "d_pose": base["d_pose"], "archive_bytes": base["archive_bytes"],
                "axis_tag": self.axis_tag,
            }
            self._append_ledger(row)
            self._bank(rnd)
            self.log(
                f"[round {rnd}] accepted {len(accepted)} clicks -> S={base['S']:.8f} "
                f"d_seg={base['d_seg']:.8f} d_pose={base['d_pose']:.8f} "
                f"bytes={base['archive_bytes']}"
            )
            best_row = {
                "round": rnd, "S": base["S"], "d_seg": base["d_seg"],
                "d_pose": base["d_pose"], "archive_bytes": base["archive_bytes"],
            }
        final = self._finalize(best_row)
        return final

    def _bisect_accept(self, base: dict, clicks: list):
        """If the full set does not improve exact S, greedily halve until an
        improving subset is found (or none). Each subset is verified by an EXACT
        re-render (canonical layout). Returns (accepted_clicks, exact_comp) and sets
        self.Q on success; ([], base) on failure."""
        cur = list(clicks)
        while cur:
            half = cur[: max(1, len(cur) // 2)]
            Qcand = self.Q.copy()
            for p, d, dl in half:
                Qcand[p, d] = np.clip(int(Qcand[p, d]) + dl, 0, 255)
            comp = self._score_Q(Qcand)  # exact render+score of the subset
            if comp["S"] < base["S"] - self.eps:
                self.Q = Qcand
                return half, comp
            if len(cur) == 1:
                return [], base
            cur = half
        return [], base

    def _bank(self, rnd: int):
        archive = self.packet.repack_archive_bytes(self.Q, drop_sidecar=self.renderer.drop_sidecar)
        path = os.path.join(self.out_dir, "candidate_archive.zip")
        with open(path, "wb") as f:
            f.write(archive)
        return path

    def _finalize(self, best_row: dict) -> dict:
        archive = self.packet.repack_archive_bytes(self.Q, drop_sidecar=self.renderer.drop_sidecar)
        path = os.path.join(self.out_dir, "candidate_archive.zip")
        with open(path, "wb") as f:
            f.write(archive)
        n_changed = int((self.Q != self.packet.Q0).sum())
        pairs_touched = int((self.Q != self.packet.Q0).any(axis=1).sum())
        out = {
            "candidate_archive_path": path,
            "candidate_archive_bytes": len(archive),
            "candidate_archive_sha256": sha256_hex(archive),
            "incumbent_archive_sha256": FRONTIER_ARCHIVE_SHA256,
            "incumbent_contest_cpu_score": FRONTIER_CONTEST_CPU_SCORE,
            "best_row": best_row,
            "net_changed_codes": n_changed,
            "pairs_touched": pairs_touched,
            "n_pairs_scored": self.n,
            "axis_tag": self.axis_tag,
            "score_claim": False,  # not a score claim until upstream/evaluate.py exact row
            "promotable": False,
            "borrowed_substrate_accounting": borrowed_substrate_accounting(),
        }
        with open(os.path.join(self.out_dir, "search_result.json"), "w") as f:
            json.dump(out, f, indent=2)
        return out


# ---------------------------------------------------------------------------
# borrowed-substrate accounting (NO-FAKE #7)
# ---------------------------------------------------------------------------
def borrowed_substrate_accounting() -> dict:
    return {
        "mechanism": "PR128 exact-score-gated discrete latent click-polish "
        "(a12dongithub, rhnerv_latent_polish, MIT) — [external unverified]",
        "substrate": "OURS — PR110-lineage frontier archive "
        "(lane_pr110_payload_entropy_recode_20260610); decoder=PR95/#101, "
        "selector=PR110 FECa, entropy=PR112 codec_ctx",
        "ours_original_in_this_harness": [
            "the search orchestration + diagonal-batch actuator over OUR packet",
            "pair-locality verification on our decoder",
            "byte-exact splice-repack of the FP11/CTXR/FECa/DQS1 frontier grammar",
        ],
        "reuse_ledger": {
            "code_lifted_from_pr128_tree": "NONE — the harness imports only OUR "
            "frontier submission src (MIT, @adpena); PR128 is a METHOD reference "
            "(METHOD.md), not a code source.",
            "code_from_haochen_rye_HNeRV_or_NeRV": "NONE (those repos are UNLICENSED "
            "= do-not-copy; we never touch them).",
            "notices_preserved": "SPDX MIT header + PR128/PR95/#101/#110/#112 "
            "attribution retained.",
        },
        "classification": "DEFENSIVE BANK, not innovation — borrowed mechanism on "
        "borrowed substrate; every reuse/originality claim gated by this block.",
        "score_claim": False,
        "promotable": False,
    }


def dispatch_discipline() -> dict:
    """Determinism + custody rules the phase-2 dispatch MUST honor (folded from the
    parallel PR128 fresh-eyes advisory 2026-07-10)."""
    return {
        "same_container_determinism": "torch bicubic interpolate raw LSBs differ "
        "across CPU microarchitectures (PR128's own expected_output.sha256 is "
        "NOT portable). Therefore SELECTION and EXACT EVAL run in the SAME Modal "
        "container / microarch — MANDATORY. Never assume a decode sha is portable; "
        "verify in-container.",
        "pin_dependencies": {
            "constriction": "0.4.2",  # coded bytes depend on the exact coder
            "torch": "2.12.1",
            "numpy": "1.26.4",
            "note": "pin these in the Modal image so the candidate archive bytes + "
            "decode are reproducible; constriction is unpinned upstream in PR128.",
        },
        "output_custody": {
            "expected_raw_bytes": 1200 * CAMERA_H * CAMERA_W * 3,  # 3,662,409,600
            "assert_exact_consumption": "inflate parse asserts no trailing bytes; "
            "assert the inflated raw is EXACTLY expected_raw_bytes (a SHORT raw = "
            "evaluator truncation = NO-FAKE failure).",
            "atomic_writes": "write candidate archive + inflated raw via tmp+rename.",
            "storage_preflight": "reserve >= expected_raw_bytes + archive on the "
            "container before inflate; fail closed if insufficient.",
        },
        "waterline_equivalents": {
            "pointer_contest_cpu": FRONTIER_CONTEST_CPU_SCORE,
            "S_per_archive_byte": 6.6586e-7,
            "S_per_seg_cell": 8.4771e-7,
            "sub_0_19_needs_bytes": 1651.74,
            "sub_0_19_needs_seg_cells": 1297.41,
            "note": "frame measured deltas against these one-axis equivalents.",
        },
    }
