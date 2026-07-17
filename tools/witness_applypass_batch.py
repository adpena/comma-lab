#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""#406 WITNESS APPLY-PASS BATCH — registry-driven per-lever ΔS orchestrator.

ONE orchestrator that, given a FROZEN witness checkpoint npz, runs a REGISTERED
SEQUENCE of post-hoc rate/pose levers. Each lever produces (a) a transformed
checkpoint or codec, (b) its REAL byte-close blob byte count, and (in fire mode)
(c) the n600 chunked frozen-CPU-scorer d_seg / d_pose through the REAL decode —
emitted as canonical :class:`tac.verdicts.MeasurementRow` objects, then folded
into a per-lever ΔS attribution table (advisory axis, NON-PROMOTABLE, pointer
UNMOVED — this is the compress-half MEANS).

WHY A SECOND TOOL (and why it is NOT a duplicate). ``tools/witness_apply_pass.py``
is the canonical home of five levers (#336 sensitivity bit-alloc, #140 low-rank
pose, sidecar-fold, #311 TropNNC, #401 blind-coordinate). It predates the #519
gauge/palette canonicalization result (``.omx/research/null_subspace_rate_measure_
20260717.md``) and has no compose-best mode. This batch tool ADDS exactly those
two things on top of a thin lever REGISTRY:

  * The #519 param-space levers (gauge / palette / both) are measured HERE through
    the #519 byte-close+score path — reusing ``null_subspace_rate_measure``'s
    ``project_gauge`` / ``project_palette_gauge`` / ``build_blob_for`` /
    ``decode_frames`` / ``score_frames`` (NO codec math re-derived).
  * The five pre-built levers are registered as fail-closed DELEGATES to
    ``witness_apply_pass.py`` (the canonical home) — this tool never re-implements
    their codecs; it validates their deps and, in fire mode, delegates.
  * ``--compose-best`` chains the individually-winning param-transform levers and
    RE-measures (composition != sum — it is measured, never summed).

THE REGISTERED SEQUENCE (ordered):
  1. gauge_519         — out_sdf class-mean gauge projected out (param transform).
  2. palette_canon_519 — palette channel-mean folded into out_tex.bias (transform).
  3. both_canon_519    — gauge ∘ palette (both, composed then measured).
  4. bit_alloc_336     — DELEGATE -> apply_sensitivity_bitalloc_witness.py.
  5. low_rank_pose_140 — DELEGATE -> witness_apply_pass.py (needs a (600,6) target).
  6. tropnnc_311       — DELEGATE -> tac.boundary_math.tropnnc_witness_reduction.
  7. blind_coord_401   — DELEGATE -> tac.through_r.blind_coordinate (receiver-side).
  8. compose_best      — (mode) chain Δd_seg-favorable param-transform levers.

CONTAINMENT (this harness may run while the live v9 run owns 76+ GiB):
  * ``--dry-run`` (structural): validate the registry + checkpoint param keys, and
    for param-transform levers build the REAL byte-close blob (LIGHT — no torch, no
    frames, no gt cache; ~70 MB / ~1 s) to emit EXACT byte rows + Δbytes. NO decode,
    NO scorer, NO gt-cache load. A lever whose deps are missing emits an HONEST
    ``skipped`` row — never a fake pass.
  * Fire mode (default: ``--pairs 600``) decodes + scores through R. It is guarded
    by ``--min-free-gib`` (default 24): if free memory is below the floor it
    REFUSES (rc=4) rather than risk OOM-ing the live P0 trainer (the #205 lesson).
  * NO Modal / paid dispatch, EVER. A promotable ΔS still needs the n600 byte-close
    + exact eval on contest-compliant hardware.

AUTHORITY: every measured row is ``[macOS-CPU advisory]`` (or ``through_R`` for the
scorer scalars) NON-PROMOTABLE. The pointer does NOT move here.

Usage (dry-run against the FROZEN donor while the trainer is live — SAFE):
  OMP_NUM_THREADS=1 .venv/bin/python tools/witness_applypass_batch.py --dry-run \
      --ckpt-dir experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z \
      --npz-name levelset_witness_ema_BEST.npz \
      --out-dir experiments/results/applypass_batch_dryrun_<utc>

Fire the full per-lever ΔS table (ONLY when the machine is free):
  OMP_NUM_THREADS=1 .venv/bin/python tools/witness_applypass_batch.py \
      --ckpt-dir <v9_run_dir> --npz-name levelset_witness_ema_BEST.npz \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
      --pairs 600 --compose-best --out-dir experiments/results/applypass_batch_<utc>
"""
from __future__ import annotations

import os

# 1-thread standard for any measurement code path (must precede numpy import).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# REUSE the #519 measurement path (byte-close + decode + score) and the byte-close
# module — NO codec math is re-derived in this file.
import levelset_byte_close_and_eval as bc  # noqa: E402
import null_subspace_rate_measure as ns  # noqa: E402

from tac.verdicts.measurement_row import (  # noqa: E402
    AxisTag,
    MeasurementRow,
    Provenance,
    ReviewStatus,
)

_PY = sys.executable
_WITNESS_APPLY_PASS = _REPO / "tools" / "witness_apply_pass.py"
_BITALLOC = _REPO / "tools" / "apply_sensitivity_bitalloc_witness.py"
_GT_CACHE_DEFAULT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
_RATE_DENOM = 37_545_489  # contest rate denominator (25 * bytes / denom)
AXIS_ADVISORY = "[macOS-CPU advisory] NON-PROMOTABLE"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _vm_stat_free_gib() -> float:
    """macOS vm_stat free+speculative in GiB (EXCLUDES inactive — the CONSERVATIVE
    truly-free floor). Inactive/purgeable pages are reclaimable-but-in-use; counting
    them as free is the admission-gate confound (MEMORY: admission_gate_naive_counts_
    reclaimable...). NaN when unavailable (non-macOS)."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        ps = 4096
        free = spec = 0
        for line in out.splitlines():
            if "page size of" in line:
                ps = int("".join(c for c in line.split("page size of")[1] if c.isdigit()))
            elif line.startswith("Pages free:"):
                free = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages speculative:"):
                spec = int(line.split(":")[1].strip().rstrip("."))
        return (free + spec) * ps / (1024 ** 3)
    except Exception:
        return float("nan")


def _free_gib() -> float:
    """CONSERVATIVE free memory in GiB = MIN(psutil.available, vm_stat free+spec).

    A SAFETY REFUSE guard must err toward refusing: psutil.available on macOS
    over-counts reclaimable/inactive memory (measured 58.7 vs a truly-free 14.9 next
    to a 74.6 GiB trainer), so trusting it alone would run the OOM-heavy scorer path
    against the live P0 trainer. Take the smaller reading. NaN only if BOTH fail."""
    readings: list[float] = []
    try:
        import psutil  # type: ignore
        readings.append(float(psutil.virtual_memory().available) / (1024 ** 3))
    except Exception:
        pass
    vm = _vm_stat_free_gib()
    if vm == vm:  # not NaN
        readings.append(vm)
    finite = [r for r in readings if r == r]
    return min(finite) if finite else float("nan")


def _score_delta(d_seg: float, d_pose: float, bytes_: float,
                 base: dict[str, float]) -> float:
    """Advisory ΔS vs baseline in contest score units (100*d_seg + sqrt(10*d_pose)
    + 25*bytes/denom). Composition is NEVER summed — this is only used on MEASURED
    (d_seg, d_pose, bytes) triples that already came through the real decode."""
    def _s(ds: float, dp: float, b: float) -> float:
        return 100.0 * ds + math.sqrt(10.0 * max(dp, 0.0)) + 25.0 * b / _RATE_DENOM
    return _s(d_seg, d_pose, bytes_) - _s(base["d_seg"], base["d_pose"], base["bytes"])


# ---------------------------------------------------------------------------
# lever registry
# ---------------------------------------------------------------------------
ParamTransform = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def _compose(*fns: ParamTransform) -> ParamTransform:
    """Left-to-right function composition on the param dict (fns[0] applied first)."""
    def _f(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for fn in fns:
            p = fn(p)
        return p
    return _f


@dataclass
class Lever:
    """One registered lever. ``kind`` selects how it is measured.

    param_transform : ``transform`` maps params->params; measured HERE via the #519
                      byte-close+score path. ``required_keys`` must exist in the npz.
    delegate        : this tool does NOT re-implement it; it validates ``deps`` and,
                      in fire mode, delegates to the canonical tool (``delegate_note``
                      + ``fire_argv_fn`` describe/produce the delegated invocation).
    """

    lever_id: str
    human: str
    kind: str  # "param_transform" | "delegate"
    transform: ParamTransform | None = None
    required_keys: tuple[str, ...] = ()
    deps: Callable[[], tuple[bool, str]] | None = None
    delegate_note: str = ""
    fire_argv_fn: Callable[["ApplyPassBatch"], list[str]] | None = None


def _dep_module(modname: str) -> Callable[[], tuple[bool, str]]:
    def _check() -> tuple[bool, str]:
        import importlib.util
        ok = importlib.util.find_spec(modname) is not None
        return ok, ("" if ok else f"module {modname!r} not importable")
    return _check


def _dep_file(path: Path, *, also_import: str | None = None) -> Callable[[], tuple[bool, str]]:
    def _check() -> tuple[bool, str]:
        if not path.exists():
            return False, f"delegate tool {path} missing"
        if also_import is not None:
            import importlib.util
            if importlib.util.find_spec(also_import) is None:
                return False, f"dep {also_import!r} not importable"
        return True, ""
    return _check


def build_registry() -> list[Lever]:
    """The ordered registered sequence. Compose-best is a MODE, not a registry row."""
    return [
        Lever(
            lever_id="gauge_519", kind="param_transform",
            human="#519 out_sdf class-mean gauge projected out (render-invariant in fp32)",
            transform=ns.project_gauge,
            required_keys=("out_sdf.weight", "out_sdf.bias"),
        ),
        Lever(
            lever_id="palette_canon_519", kind="param_transform",
            human="#519 palette channel-mean folded into out_tex.bias (render-invariant)",
            transform=ns.project_palette_gauge,
            required_keys=("palette", "out_tex.bias"),
        ),
        Lever(
            lever_id="both_canon_519", kind="param_transform",
            human="#519 gauge ∘ palette canonicalization (composed, then measured)",
            transform=_compose(ns.project_gauge, ns.project_palette_gauge),
            required_keys=("out_sdf.weight", "out_sdf.bias", "palette", "out_tex.bias"),
        ),
        Lever(
            lever_id="bit_alloc_336", kind="delegate",
            human="#336 sensitivity KKT reverse-water-fill bit allocation",
            deps=_dep_file(_BITALLOC),
            delegate_note=("re-quantizes the npz per MEASURED d_seg sensitivity; delegates to "
                           "apply_sensitivity_bitalloc_witness.py (via witness_apply_pass.py "
                           "--fire-scorer-stages). Scorer-heavy: fire only when the machine is free."),
            fire_argv_fn=lambda h: h._witness_apply_pass_argv(),
        ),
        Lever(
            lever_id="low_rank_pose_140", kind="delegate",
            human="#140 rank-k SVD codec on the (600,6) stored PoseNet target",
            deps=_dep_module("brotli"),
            delegate_note=("needs a (600,6) pose target (--pose-target). Delegates to "
                           "witness_apply_pass.py:_low_rank_pose (real brotli byte codec + "
                           "recon-MSE d_pose floor). OWED if no target is supplied."),
            fire_argv_fn=lambda h: h._witness_apply_pass_argv(),
        ),
        Lever(
            lever_id="tropnnc_311", kind="delegate",
            human="#311 Laguerre-cell-aware structured trunk reduction (exact-Δd_seg=0 accept)",
            deps=_dep_module("tac.boundary_math.tropnnc_witness_reduction"),
            delegate_note=("mean-compensated neuron prune ranked by tropical dominance; exact "
                           "trunk-byte savings + n600 SegNet-argmax-equality accept. Delegates "
                           "to witness_apply_pass.py:_tropnnc (--fire-scorer-stages for the screen)."),
            fire_argv_fn=lambda h: h._witness_apply_pass_argv(),
        ),
        Lever(
            lever_id="blind_coord_401", kind="delegate",
            human="#401 blind-coordinate generic fill (camera px read by NO scorer resize)",
            deps=_dep_module("tac.through_r.blind_coordinate"),
            delegate_note=("RECEIVER-side / decode-time lever: 22.70% camera px are blind to both "
                           "scorers -> generic rule-118 fill. DIRECT rate saving is 0 on a "
                           "pure-generator archive (no camera-res section); it is a capacity "
                           "observation until a camera-res residual/sidecar is carried. Delegates "
                           "to witness_apply_pass.py:_blind_coord + blind_coordinate_proof.py."),
            fire_argv_fn=lambda h: h._witness_apply_pass_argv(),
        ),
    ]


# ---------------------------------------------------------------------------
# stage result container
# ---------------------------------------------------------------------------
@dataclass
class LeverResult:
    lever_id: str
    kind: str
    status: str = "pending"  # measured | dryrun | staged | delegated | owed | skipped | error
    rows: list[MeasurementRow] = field(default_factory=list)
    delta: dict[str, Any] = field(default_factory=dict)
    staged_commands: list[dict] = field(default_factory=list)
    note: str = ""
    raw: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "lever_id": self.lever_id, "kind": self.kind, "status": self.status,
            "rows": [r.to_json_dict() for r in self.rows], "delta": self.delta,
            "staged_commands": self.staged_commands, "note": self.note, "raw": self.raw,
        }


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
class ApplyPassBatch:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.git_sha = _git_sha()
        self.ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        src_ckpt = Path(args.ckpt_dir)
        if not src_ckpt.is_dir():
            raise FileNotFoundError(f"--ckpt-dir not found: {src_ckpt}")
        if src_ckpt.resolve() == self.out_dir.resolve():
            raise ValueError("--out-dir must not be the checkpoint dir (no in-place ops)")
        self.npz_name = args.npz_name
        # COPY the frozen checkpoint in — NEVER operate in-place on any run dir.
        self.frozen_dir = self.out_dir / "frozen_ckpt"
        self.frozen_dir.mkdir(exist_ok=True)
        src_npz = src_ckpt / self.npz_name
        if not src_npz.exists():
            raise FileNotFoundError(f"npz not found: {src_npz}")
        dst_npz = self.frozen_dir / self.npz_name
        if not dst_npz.exists():
            shutil.copy2(src_npz, dst_npz)
        self.npz_sha = _sha256_file(dst_npz)
        self.config_ref = f"{src_ckpt.name}/{self.npz_name}"
        # load params/cfg + self-orient (LIGHT; no torch / frames / scorer).
        self.params, self.cfg = bc._load_levelset_ckpt(self.frozen_dir, self.npz_name)
        z = np.load(dst_npz, allow_pickle=False)
        fa = float(z["__cfg_freq_across"]) if "__cfg_freq_across" in z.files else 32.0
        fl = float(z["__cfg_freq_along"]) if "__cfg_freq_along" in z.files else 4.0
        self.epoch = int(z["__epoch"]) if "__epoch" in z.files else None
        self.so = bc.detect_self_orient(
            self.cfg, {"freq_across": fa, "freq_along": fl,
                       "tau": args.so_tau, "iters": args.so_iters})
        self.registry = build_registry()

    # -- provenance / row helpers (mirror witness_apply_pass envelope discipline) --
    def _prov(self, tool: str, config_ref: str | None = None) -> Provenance:
        return Provenance(git_sha=self.git_sha, tool=tool, seed=self.args.seed,
                          config_ref=config_ref or self.config_ref, inputs_sha256=self.npz_sha)

    def _bytes_row(self, tool: str, value: int, quantity: str,
                   n_samples: int, reason: str | None) -> MeasurementRow:
        return MeasurementRow(
            value=float(value), units="bytes", axis_tag=AxisTag.MACOS_CPU_ADVISORY,
            provenance=self._prov(tool), n_samples=n_samples, n_samples_reason=reason,
            review_status=ReviewStatus.PROVISIONAL, noise_floor=0.0,
            floor_provenance="levelset 0.bin byte count is exact/deterministic",
            quantity=quantity)

    def _scalar_row(self, tool: str, value: float, quantity: str, units: str,
                    axis: AxisTag, n_samples: int, reason: str | None) -> MeasurementRow:
        return MeasurementRow(
            value=float(value), units=units, axis_tag=axis, provenance=self._prov(tool),
            n_samples=n_samples, n_samples_reason=reason,
            review_status=ReviewStatus.PROVISIONAL, noise_floor=None, floor_provenance=None,
            quantity=quantity)

    # -- staged delegate argv to the canonical home -------------------------
    def _witness_apply_pass_argv(self) -> list[str]:
        argv = [_PY, str(_WITNESS_APPLY_PASS),
                "--ckpt-dir", str(self.args.ckpt_dir), "--npz-name", self.npz_name,
                "--out-dir", str(self.out_dir / "delegate_witness_apply_pass"),
                "--gt-cache", self.args.gt_cache, "--fire-scorer-stages",
                "--eval-pairs", str(self.args.pairs)]
        if self.args.pose_target:
            argv += ["--pose-target", str(self.args.pose_target)]
        return argv

    # -- validate the required param keys are present -----------------------
    def _keys_present(self, lever: Lever) -> tuple[bool, list[str]]:
        missing = [k for k in lever.required_keys if k not in self.params]
        return (len(missing) == 0), missing

    # ======================================================================
    # baseline byte-close (LIGHT) + optional baseline decode/score (fire)
    # ======================================================================
    def _byte_close_bytes(self, params: dict[str, np.ndarray]) -> tuple[bytes, int]:
        blob, bd = ns.build_blob_for(params, self.cfg, self.so)
        return blob, int(bd["total_0bin_bytes"])

    # ======================================================================
    # measure ONE param-transform lever
    # ======================================================================
    def _measure_param_transform(self, lever: Lever, *, fire: bool,
                                  base_bytes: int, base_score: dict | None,
                                  scorers, gt) -> LeverResult:
        res = LeverResult(lever_id=lever.lever_id, kind=lever.kind)
        ok, missing = self._keys_present(lever)
        if not ok:
            res.status = "skipped"
            res.note = (f"required param key(s) absent: {missing} — checkpoint head does not "
                        "match the #519 shared-head form; honest SKIP (no fake apply).")
            return res
        assert lever.transform is not None
        tp = lever.transform(self.params)
        blob, nbytes = self._byte_close_bytes(tp)
        n_pairs = int(self.args.pairs)
        reason_bytes = (None if n_pairs == 600
                        else f"byte term is exact regardless of pairs; scorer subset at {n_pairs}")
        res.rows.append(self._bytes_row(
            "witness_applypass_batch:param_transform", nbytes,
            f"archive_0bin_bytes[{lever.lever_id}]", n_pairs, reason_bytes))
        res.raw = {"total_0bin_bytes": nbytes, "delta_bytes_vs_baseline": nbytes - base_bytes}
        res.delta = {"delta_bytes": nbytes - base_bytes}
        if not fire:
            res.status = "dryrun"
            res.note = ("byte-close blob built (LIGHT, no scorer). Δd_seg/Δd_pose need fire mode "
                        "(--pairs 600, machine free).")
            return res
        # fire: decode + n600 chunked frozen-CPU scorer
        seg_cpu, posenet_cpu = scorers
        lstars, gt_poses = gt
        frames, _argmax = ns.decode_frames(blob, n_pairs)
        score = ns.score_frames(frames, lstars, gt_poses, seg_cpu, posenet_cpu,
                                batch=self.args.score_batch)
        reason_scorer = (None if n_pairs == 600
                         else f"advisory scorer subset at {n_pairs} pairs (n600 for a promotable ΔS)")
        res.rows.append(self._scalar_row(
            "witness_applypass_batch:param_transform", score["d_seg"],
            f"d_seg[{lever.lever_id}]", "argmax_disagreement", AxisTag.THROUGH_R,
            n_pairs, reason_scorer))
        res.rows.append(self._scalar_row(
            "witness_applypass_batch:param_transform", score["d_pose"],
            f"d_pose[{lever.lever_id}]", "mse_pose6", AxisTag.THROUGH_R,
            n_pairs, reason_scorer))
        res.status = "measured"
        res.raw.update({"d_seg": score["d_seg"], "d_pose": score["d_pose"]})
        if base_score is not None:
            ds = score["d_seg"] - base_score["d_seg"]
            dp = score["d_pose"] - base_score["d_pose"]
            dS = _score_delta(score["d_seg"], score["d_pose"], nbytes, base_score)
            res.delta = {"delta_bytes": nbytes - base_bytes, "delta_d_seg": ds,
                         "delta_d_pose": dp, "delta_S_advisory": dS}
        return res

    # ======================================================================
    # a delegate lever (validate deps; stage / delegate the canonical tool)
    # ======================================================================
    def _delegate_lever(self, lever: Lever, *, fire: bool) -> LeverResult:
        res = LeverResult(lever_id=lever.lever_id, kind=lever.kind)
        ok, reason = (lever.deps() if lever.deps else (True, ""))
        if not ok:
            res.status = "skipped"
            res.note = f"delegate deps unmet: {reason} — honest SKIP (no fake apply)."
            return res
        argv = lever.fire_argv_fn(self) if lever.fire_argv_fn else []
        res.staged_commands.append({
            "lever": lever.lever_id,
            "measures": "byte term + (fire) d_seg/d_pose through R via the canonical home",
            "delegates_to": str(_WITNESS_APPLY_PASS.relative_to(_REPO)),
            "authority": "[macOS-CPU advisory] -> stage to [contest-CPU/CUDA] for a score",
            "argv": argv,
            "fire_when": "v9 first good checkpoint; machine free; operator GO for any paid eval",
        })
        if lever.lever_id == "low_rank_pose_140" and not self.args.pose_target:
            res.status = "owed"
            res.note = (f"{lever.human}: {lever.delegate_note} No --pose-target supplied -> OWED.")
            return res
        if not fire:
            res.status = "staged"
            res.note = f"{lever.human}: {lever.delegate_note}"
            return res
        # FIRE: delegate to the canonical harness once (it runs all its own stages);
        # harvest its per-lever rows and re-tag under this lever id namespace.
        res.status = "delegated"
        res.note = ("delegated to witness_apply_pass.py --fire-scorer-stages; harvest its "
                    "apply_pass_rows.jsonl (see staged argv). This batch tool does not "
                    "re-implement the delegate codec.")
        return res

    # ======================================================================
    # compose-best: chain Δd_seg-favorable param-transform levers, re-measure
    # ======================================================================
    def _compose_best(self, per_lever: list[LeverResult], *, fire: bool,
                      base_bytes: int, base_score: dict | None, scorers, gt) -> LeverResult:
        res = LeverResult(lever_id="compose_best", kind="param_transform")
        # candidate param-transform levers that applied cleanly (not skipped)
        pt = {lv.lever_id: lv for lv in self.registry if lv.kind == "param_transform"}
        applied = [r for r in per_lever
                   if r.lever_id in pt and r.status in ("measured", "dryrun")]
        # exclude the pre-composed 'both' (avoid double-applying gauge/palette);
        # compose the ATOMIC winners.
        atomic = [r for r in applied if r.lever_id in ("gauge_519", "palette_canon_519")]
        if not atomic:
            res.status = "skipped"
            res.note = "no atomic param-transform lever applied cleanly -> nothing to compose."
            return res
        if fire and base_score is not None:
            winners = [r for r in atomic if r.delta.get("delta_d_seg", 1.0) <= 0.0]
            selection = "delta_d_seg<=0 (favorable through-R)"
        else:
            winners = atomic  # dry-run: compose ALL atomic (preview); note selection is fire-only
            selection = "ALL atomic (dry-run preview; Δd_seg winner-selection needs fire mode)"
        if not winners:
            res.status = "skipped"
            res.note = f"no favorable atomic lever ({selection}) -> compose-best empty."
            return res
        fns = [pt[r.lever_id].transform for r in winners]
        composed = _compose(*[f for f in fns if f is not None])
        tp = composed(self.params)
        blob, nbytes = self._byte_close_bytes(tp)
        n_pairs = int(self.args.pairs)
        res.raw = {"composed_of": [r.lever_id for r in winners], "selection": selection,
                   "total_0bin_bytes": nbytes, "delta_bytes_vs_baseline": nbytes - base_bytes}
        res.rows.append(self._bytes_row(
            "witness_applypass_batch:compose_best", nbytes, "archive_0bin_bytes[compose_best]",
            n_pairs, (None if n_pairs == 600 else "byte term exact; scorer subset")))
        res.delta = {"delta_bytes": nbytes - base_bytes}
        if not fire:
            res.status = "dryrun"
            res.note = (f"composed {[r.lever_id for r in winners]} ({selection}); byte-close only. "
                        "ΔS of the composition needs fire mode (composition != sum — measured).")
            return res
        seg_cpu, posenet_cpu = scorers
        lstars, gt_poses = gt
        frames, _am = ns.decode_frames(blob, n_pairs)
        score = ns.score_frames(frames, lstars, gt_poses, seg_cpu, posenet_cpu,
                                batch=self.args.score_batch)
        reason = (None if n_pairs == 600 else f"advisory scorer subset at {n_pairs} pairs")
        res.rows.append(self._scalar_row("witness_applypass_batch:compose_best", score["d_seg"],
                                         "d_seg[compose_best]", "argmax_disagreement",
                                         AxisTag.THROUGH_R, n_pairs, reason))
        res.rows.append(self._scalar_row("witness_applypass_batch:compose_best", score["d_pose"],
                                         "d_pose[compose_best]", "mse_pose6",
                                         AxisTag.THROUGH_R, n_pairs, reason))
        res.status = "measured"
        res.raw.update({"d_seg": score["d_seg"], "d_pose": score["d_pose"]})
        if base_score is not None:
            res.delta = {
                "delta_bytes": nbytes - base_bytes,
                "delta_d_seg": score["d_seg"] - base_score["d_seg"],
                "delta_d_pose": score["d_pose"] - base_score["d_pose"],
                "delta_S_advisory": _score_delta(score["d_seg"], score["d_pose"], nbytes, base_score),
                "note": "composition measured through the real decode, NOT summed from atomics",
            }
        return res

    # ======================================================================
    def run(self) -> dict:
        fire = not self.args.dry_run
        if fire:
            free = _free_gib()
            if not (free >= self.args.min_free_gib):
                raise SystemExit(
                    f"[applypass-batch] REFUSE fire mode: free≈{free:.1f} GiB < "
                    f"--min-free-gib {self.args.min_free_gib}. The scorer path loads the "
                    f"{Path(self.args.gt_cache).name} gt cache + SegNet/PoseNet and could OOM the "
                    "live P0 trainer (the #205 lesson). Use --dry-run, or wait for the machine to "
                    "be free / lower --min-free-gib deliberately (rc=4).")
        # baseline byte-close (LIGHT)
        _blob0, base_bytes = self._byte_close_bytes(self.params)
        base_score: dict | None = None
        scorers = gt = None
        if fire:
            n_pairs = int(self.args.pairs)
            scorers = ns.load_scorers()
            gt = ns.load_gt_subset(Path(self.args.gt_cache), n_pairs)
            frames0, _a0 = ns.decode_frames(_blob0, n_pairs)
            s0 = ns.score_frames(frames0, gt[0], gt[1], scorers[0], scorers[1],
                                 batch=self.args.score_batch)
            base_score = {"bytes": float(base_bytes), "d_seg": s0["d_seg"], "d_pose": s0["d_pose"]}
            del frames0

        results: list[LeverResult] = []
        for lever in self.registry:
            if lever.kind == "param_transform":
                results.append(self._measure_param_transform(
                    lever, fire=fire, base_bytes=base_bytes, base_score=base_score,
                    scorers=scorers, gt=gt))
            else:
                results.append(self._delegate_lever(lever, fire=fire))

        compose_res: LeverResult | None = None
        if self.args.compose_best:
            compose_res = self._compose_best(
                results, fire=fire, base_bytes=base_bytes, base_score=base_score,
                scorers=scorers, gt=gt)
            results.append(compose_res)

        summary = {
            "task": "#406 apply-pass batch (registry-driven per-lever ΔS)",
            "ts": self.ts, "git_sha": self.git_sha, "epoch": self.epoch,
            "mode": "dry-run" if self.args.dry_run else "fire",
            "frozen_ckpt": str(self.frozen_dir), "npz_name": self.npz_name,
            "npz_sha256": self.npz_sha, "config_ref": self.config_ref,
            "authority": AXIS_ADVISORY + "; pointer UNMOVED (compress-half MEANS)",
            "pairs": self.args.pairs, "compose_best": bool(self.args.compose_best),
            "baseline_0bin_bytes": base_bytes,
            "baseline_score": base_score,
            "registry_order": [lv.lever_id for lv in self.registry] + (
                ["compose_best"] if self.args.compose_best else []),
            "levers": [r.to_json_dict() for r in results],
            "skipped": [r.lever_id for r in results if r.status == "skipped"],
            "owed": [r.lever_id for r in results if r.status == "owed"],
            "errors": [{"lever": r.lever_id, "note": r.note} for r in results if r.status == "error"],
        }
        rows_path = self.out_dir / "applypass_batch_rows.jsonl"
        with rows_path.open("w") as fh:
            for r in results:
                for row in r.rows:
                    fh.write(json.dumps({"lever": r.lever_id, **row.to_json_dict()}) + "\n")
        summary_path = self.out_dir / "applypass_batch_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        self._print_table(results, base_bytes, base_score, summary_path, rows_path)
        return summary

    def _print_table(self, results: list[LeverResult], base_bytes: int,
                     base_score: dict | None, summary_path: Path, rows_path: Path) -> None:
        print("\n" + "=" * 92)
        print(f"#406 APPLY-PASS BATCH — per-lever ΔS  [{'dry-run' if self.args.dry_run else 'fire'}]  "
              f"{AXIS_ADVISORY}")
        print(f"baseline 0.bin bytes = {base_bytes}"
              + (f" · d_seg={base_score['d_seg']:.6g} d_pose={base_score['d_pose']:.6g}"
                 if base_score else " · (scorer baseline: fire mode only)"))
        print("=" * 92)
        for r in results:
            d = r.delta
            db = d.get("delta_bytes")
            dseg = d.get("delta_d_seg")
            dS = d.get("delta_S_advisory")
            seg = f"Δd_seg={dseg:+.3e}" if isinstance(dseg, float) else "Δd_seg=—"
            byt = f"Δbytes={db:+d}" if isinstance(db, int) else "Δbytes=—"
            sc = f"ΔS={dS:+.5f}" if isinstance(dS, float) else "ΔS=—"
            print(f"  {r.lever_id:<20} {r.status:<10} {byt:<14} {seg:<18} {sc}")
            if r.status in ("skipped", "owed", "error", "staged") and r.note:
                print(f"      note: {r.note[:150]}")
        print("-" * 92)
        print(f"summary: {summary_path}")
        print(f"rows:    {rows_path}")
        print("-" * 92)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="FROZEN run dir (copied in; NEVER the live run)")
    ap.add_argument("--npz-name", default="levelset_witness_ema_BEST.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-cache", default=_GT_CACHE_DEFAULT)
    ap.add_argument("--pose-target", default=None,
                    help="(600,6) stored PoseNet target .pt for the #140 delegate lever")
    ap.add_argument("--pairs", type=int, default=600,
                    help="scorer pairs (n600 default; the non-negotiable). Byte term is exact regardless.")
    ap.add_argument("--dry-run", action="store_true",
                    help="structural validation + LIGHT byte-close blobs only (no torch/frames/scorer).")
    ap.add_argument("--compose-best", action="store_true",
                    help="chain the Δd_seg-favorable atomic param-transform levers and RE-measure.")
    ap.add_argument("--min-free-gib", type=float, default=24.0,
                    help="refuse fire mode below this free-memory floor (OOM guard for the live trainer).")
    ap.add_argument("--score-batch", type=int, default=8,
                    help="chunked verdict batch (bit-identical; running-stat BN) for the scorer path.")
    ap.add_argument("--so-iters", type=int, default=4)
    ap.add_argument("--so-tau", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ApplyPassBatch(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
