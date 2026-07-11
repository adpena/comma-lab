#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""#406 WITNESS APPLY-PASS — the one-command rate/pose apply-pass batch.

ONE harness that turns a FROZEN witness checkpoint into a per-lever ΔS attribution
table: for each rate/pose lever it byte-closes + (optionally) scores through R and
emits ONE canonical :class:`tac.verdicts.MeasurementRow` per measured scalar
(bytes + advisory d_seg / d_pose), each carried with its full honesty envelope
(axis tag, provenance, noise-floor-unknown, n_samples + reason, review_status).

WHY THIS EXISTS (task #406). The compress-half levers were each built + verified in
ISOLATION (the #336 sensitivity bit-alloc apply, the #140 low-rank pose codec, the
sidecar-fold in the byte-close orbit) but never composed into ONE pass that fires the
moment the live v9 run produces its first good checkpoint. This is that pass. It is a
pure ORCHESTRATOR: every stage delegates to the canonical existing tool (no re-derived
codec math), and every number it emits is a REAL measurement of a REAL byte-closed
blob — NO-FAKE. A lever that is NOT built (#311 TropNNC, #401 blind-coordinate) is
registered as an EXPLICIT stub stage that FAILS LOUDLY if invoked and otherwise reports
itself OWED — it never pretend-applies.

THE STAGES (ordered; rate then pose):
  1. baseline_byte_close  — the reference archive.zip rate term (levelset_byte_close).
  2. bit_alloc (#336)     — #157 KKT reverse-water-fill applied to the witness npz
                            (apply_sensitivity_bitalloc_witness): fewer int8 symbols ->
                            smaller brotli, calibrated by MEASURED d_seg sensitivity.
  3. low_rank_pose (#140) — rank-k SVD codec on the 600x6 stored PoseNet target matrix:
                            store U,S,V factors quantized+brotli vs the scalar sidecar;
                            reconstruction MSE == the advisory d_pose floor it ADDS.
  4. sidecar_fold         — fold the stored-pose sidecar section into the packet and
                            re-measure the archive rate term (byte-neutral recode orbit).
  5. STUB #311 TropNNC        — NOT built. Loud NotImplementedError if --run-stub.
  6. STUB #401 blind_coord    — NOT built. Loud NotImplementedError if --run-stub.

CONTAINMENT (this harness runs while the live v9 run owns the GPU):
  * CPU-light + SERIAL. Each stage is a separate subprocess that exits (frees memory).
  * SCORER-heavy stages (any through-R d_seg/d_pose calibration) DEFAULT to STAGED:
    the exact argv is captured + emitted, NOT fired, unless --fire-scorer-stages.
  * NO Modal / paid dispatch, EVER (operator HOLD). Exact-eval commands are staged text.
  * The frozen checkpoint is COPIED into the out-dir; we NEVER operate in-place on any
    run dir (least of all the live one).

AUTHORITY: every measured row is ``[macOS-CPU advisory]`` NON-PROMOTABLE and (in the
dry-run) sub-n600 with a stated reason. A promotable ΔS attribution needs the full n600
byte-close + exact eval on contest-compliant hardware — emitted as the staged command
per lever. The pointer does NOT move here; this is the compress-half MEANS.

Usage (dry-run against a FROZEN old checkpoint):
  .venv/bin/python tools/witness_apply_pass.py \
      --ckpt-dir experiments/results/levelset_v752_baseline_20260710T185913Z \
      --npz-name levelset_witness_ema_BEST.npz \
      --pose-target experiments/results/lane_a_landed/gt_pose_targets.pt \
      --max-pairs 6 --out-dir experiments/results/apply_pass_dryrun_<utc>

Fire the scorer stages too (heavier; only when the machine is free):
      ... --fire-scorer-stages --probe-pairs 2 --eval-pairs 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tac.verdicts.measurement_row import (  # noqa: E402
    AxisTag,
    MeasurementRow,
    Provenance,
    ReviewStatus,
)

_PY = sys.executable
_LEVELSET_BYTE_CLOSE = _REPO / "tools" / "levelset_byte_close_and_eval.py"
_BITALLOC = _REPO / "tools" / "apply_sensitivity_bitalloc_witness.py"
_GT_CACHE_DEFAULT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# stage result container
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    """One stage's output: measured rows + staged commands + owed notes + raw."""

    lever: str
    kind: str  # "byte_measured" | "scorer" | "stub"
    status: str = "pending"  # "measured" | "staged" | "owed" | "error"
    rows: list[MeasurementRow] = field(default_factory=list)
    staged_commands: list[dict] = field(default_factory=list)
    note: str = ""
    raw: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "lever": self.lever,
            "kind": self.kind,
            "status": self.status,
            "rows": [r.to_json_dict() for r in self.rows],
            "staged_commands": self.staged_commands,
            "note": self.note,
            "raw": self.raw,
        }


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
class ApplyPass:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.git_sha = _git_sha()
        self.ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # COPY the frozen checkpoint into our own dir — never operate in-place.
        src_ckpt = Path(args.ckpt_dir)
        if not src_ckpt.is_dir():
            raise FileNotFoundError(f"--ckpt-dir not found: {src_ckpt}")
        if src_ckpt.resolve() == self.out_dir.resolve():
            raise ValueError("--out-dir must not be the checkpoint dir (no in-place ops)")
        self.frozen_dir = self.out_dir / "frozen_ckpt"
        self.frozen_dir.mkdir(exist_ok=True)
        self.npz_name = args.npz_name
        src_npz = src_ckpt / self.npz_name
        if not src_npz.exists():
            raise FileNotFoundError(f"npz not found: {src_npz}")
        dst_npz = self.frozen_dir / self.npz_name
        if not dst_npz.exists():
            shutil.copy2(src_npz, dst_npz)
        # copy the train_result.json / best.json / manifest siblings if present (dims/seed)
        for sib in ("train_result.json", "levelset_train_result.json",
                    "levelset_best.json", "constants_manifest.json"):
            sp = src_ckpt / sib
            if sp.exists():
                shutil.copy2(sp, self.frozen_dir / sib)
        self.npz_sha = _sha256_file(dst_npz)
        self.config_ref = f"{src_ckpt.name}/{self.npz_name}"

    # -- provenance / row helpers ------------------------------------------
    def _prov(self, tool: str) -> Provenance:
        return Provenance(
            git_sha=self.git_sha,
            tool=tool,
            seed=self.args.seed,
            config_ref=self.config_ref,
            inputs_sha256=self.npz_sha,
        )

    def _bytes_row(self, tool: str, value: int, quantity: str) -> MeasurementRow:
        # archive bytes are an EXACT deterministic count -> known 0 noise floor.
        return MeasurementRow(
            value=float(value),
            units="bytes",
            axis_tag=AxisTag.MACOS_CPU_ADVISORY,
            provenance=self._prov(tool),
            n_samples=self.args.max_pairs if self.args.max_pairs else 600,
            n_samples_reason=(
                None if (self.args.max_pairs in (None, 600))
                else f"dry-run byte-close capped at --max-pairs {self.args.max_pairs} "
                     "(byte term is exact regardless of pair count; d_seg/d_pose need n600)"
            ),
            review_status=ReviewStatus.PROVISIONAL,
            noise_floor=0.0,
            floor_provenance="archive.zip st_size is an exact deterministic byte count",
            quantity=quantity,
        )

    def _scalar_row(
        self, tool: str, value: float, quantity: str, units: str,
        axis: AxisTag, n_samples: int, reason: str | None,
    ) -> MeasurementRow:
        return MeasurementRow(
            value=float(value),
            units=units,
            axis_tag=axis,
            provenance=self._prov(tool),
            n_samples=n_samples,
            n_samples_reason=reason,
            review_status=ReviewStatus.PROVISIONAL,
            noise_floor=None,  # UNKNOWN (honest): through-R scorer floor not composed here
            floor_provenance=None,
            quantity=quantity,
        )

    # -- subprocess runner --------------------------------------------------
    def _run(self, argv: list[str], timeout: float) -> tuple[int, str, str]:
        print(f"[apply-pass] FIRE: {' '.join(argv)}", flush=True)
        t0 = time.time()
        try:
            p = subprocess.run(
                argv, cwd=_REPO, capture_output=True, text=True, timeout=timeout
            )
            print(f"[apply-pass]   -> rc={p.returncode} ({time.time()-t0:.1f}s)", flush=True)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            print(f"[apply-pass]   -> TIMEOUT after {timeout}s", flush=True)
            return 124, "", f"timeout after {timeout}s"

    # ======================================================================
    # STAGE 1 + 4: baseline byte-close + sidecar fold (levelset_byte_close)
    # ======================================================================
    def _byte_close(self, lever: str, extra_argv: list[str]) -> StageResult:
        out_json = self.out_dir / f"{lever}_byte_close.json"
        argv = [
            _PY, str(_LEVELSET_BYTE_CLOSE),
            "--ckpt-dir", str(self.frozen_dir),
            "--npz-name", self.npz_name,
            "--skip-parity",  # dry-run: byte term only (no scorer); parity is staged
            "--out", str(out_json),
            *extra_argv,
        ]
        if self.args.max_pairs:
            argv += ["--max-pairs", str(self.args.max_pairs)]
        rc, so, se = self._run(argv, timeout=self.args.byte_timeout)
        res = StageResult(lever=lever, kind="byte_measured", status="measured")
        if rc != 0 or not out_json.exists():
            res.status = "error"
            res.note = f"byte-close rc={rc}; stderr tail: {se[-600:]}"
            return res
        rep = json.loads(out_json.read_text())
        bc = rep.get("byte_close", {}) or {}
        zb = bc.get("archive_zip_bytes")
        if not isinstance(zb, int):
            res.status = "error"
            res.note = f"no archive_zip_bytes in report {out_json}"
            res.raw = {"report_keys": list(rep.keys())}
            return res
        res.rows.append(self._bytes_row(
            "levelset_byte_close_and_eval", zb, f"archive_zip_bytes[{lever}]"))
        res.raw = {"archive_zip_bytes": zb, "rate_term": bc.get("rate_term"),
                   "report": str(out_json)}
        # stage the through-R parity + n600 exact-eval command per lever
        res.staged_commands.append(self._staged_parity_cmd(lever, extra_argv))
        return res

    def _staged_parity_cmd(self, lever: str, extra_argv: list[str]) -> dict:
        argv = [
            _PY, str(_LEVELSET_BYTE_CLOSE),
            "--ckpt-dir", "<v9_run_dir>", "--npz-name", "levelset_witness_ema_BEST.npz",
            "--gt-cache", _GT_CACHE_DEFAULT, "--keep-packet",
            "--out", f"reports/apply_pass_{lever}_n600.json",
            *extra_argv,
        ]
        return {
            "lever": lever,
            "measures": "realized d_seg + d_pose through R on n600 inflated frames + archive bytes",
            "authority": "[macOS-CPU advisory] -> stage to [contest-CPU] Linux x86_64 for a score",
            "argv": argv,
            "fire_when": "v9 first good checkpoint lands; machine free; operator GO for any paid eval",
        }

    # ======================================================================
    # STAGE 2: #336 sensitivity bit-alloc
    # ======================================================================
    def _bit_alloc(self) -> StageResult:
        lever = "bit_alloc_336"
        out_json = self.out_dir / f"{lever}.json"
        argv = [
            _PY, str(_BITALLOC),
            "--ckpt-dir", str(self.frozen_dir),
            "--npz-name", self.npz_name,
            "--gt-cache", self.args.gt_cache,
            "--probe-pairs", str(self.args.probe_pairs),
            "--eval-pairs", str(self.args.eval_pairs),
            "--mean-bits", *[str(b) for b in self.args.mean_bits],
            "--out", str(out_json),
            "--ckpt-provenance", self.config_ref,
        ]
        res = StageResult(lever=lever, kind="scorer")
        if not self.args.fire_scorer_stages:
            res.status = "staged"
            res.note = ("SCORER stage staged (default): calibrates bit allocation with MEASURED "
                        "d_seg through R -> loads SegNet + gt cache. Fire with --fire-scorer-stages "
                        "when the machine is free.")
            res.staged_commands.append({
                "lever": lever,
                "measures": "bytes_before/after (real brotli) + d_seg_eval before/after per mean_bits",
                "authority": "[macOS-CPU advisory] NON-PROMOTABLE (sub-n600 probe/eval)",
                "argv": [*argv, "# n600: --probe-pairs 16 --eval-pairs 600"],
                "fire_when": "v9 first good checkpoint; machine free",
            })
            return res
        rc, so, se = self._run(argv, timeout=self.args.scorer_timeout)
        if rc != 0 or not out_json.exists():
            res.status = "error"
            res.note = f"bit-alloc rc={rc}; stderr tail: {se[-800:]}"
            return res
        rep = json.loads(out_json.read_text())
        res.status = "measured"
        base = rep.get("baseline_int8", {}) or {}
        allocs = rep.get("allocations", {}) or {}
        res.raw = {"report": str(out_json), "baseline_int8": base,
                   "what_n600_needs": rep.get("what_n600_needs")}
        reason = (f"dry-run bit-alloc probe={self.args.probe_pairs}/eval={self.args.eval_pairs} "
                  "(ADVISORY subset; n600 needed for a promotable ΔS)")
        if isinstance(base.get("bytes"), int):
            res.rows.append(self._bytes_row(
                "apply_sensitivity_bitalloc_witness", base["bytes"], "archive_bytes[bit_alloc_baseline_int8]"))
        if isinstance(base.get("d_seg_eval"), (int, float)):
            res.rows.append(self._scalar_row(
                "apply_sensitivity_bitalloc_witness", base["d_seg_eval"],
                "d_seg[bit_alloc_baseline_int8]", "argmax_disagreement",
                AxisTag.THROUGH_R, self.args.eval_pairs, reason))
        for mb, entry in allocs.items():
            wf = (entry or {}).get("waterfill", {}) or {}
            if isinstance(wf.get("bytes"), int):
                res.rows.append(self._bytes_row(
                    "apply_sensitivity_bitalloc_witness", wf["bytes"],
                    f"archive_bytes[bit_alloc_waterfill_mean{mb}]"))
            if isinstance(wf.get("d_seg_eval"), (int, float)):
                res.rows.append(self._scalar_row(
                    "apply_sensitivity_bitalloc_witness", wf["d_seg_eval"],
                    f"d_seg[bit_alloc_waterfill_mean{mb}]", "argmax_disagreement",
                    AxisTag.THROUGH_R, self.args.eval_pairs, reason))
        res.staged_commands.append({
            "lever": lever, "measures": "n600 promotable bit-alloc",
            "authority": "stage to full #202 byte-close + exact eval",
            "argv": [*argv[:-2], "--probe-pairs", "16", "--eval-pairs", "600",
                     "--out", "reports/apply_pass_bit_alloc_n600.json"],
            "fire_when": "v9 first good checkpoint; machine free",
        })
        return res

    # ======================================================================
    # STAGE 3: #140 low-rank pose codec (rank-k SVD on 600x6 target matrix)
    # ======================================================================
    def _low_rank_pose(self) -> StageResult:
        import numpy as np

        lever = "low_rank_pose_140"
        res = StageResult(lever=lever, kind="byte_measured")
        tgt_path = Path(self.args.pose_target) if self.args.pose_target else None
        if tgt_path is None or not tgt_path.exists():
            res.status = "owed"
            res.note = ("no --pose-target 600x6 matrix provided; #140 low-rank pose codec needs the "
                        "stored PoseNet target matrix (scorer_targets). Extract via "
                        "tac.scorer_targets.extract_and_save then re-run.")
            return res
        try:
            import brotli  # type: ignore
        except Exception:
            res.status = "error"
            res.note = "brotli not importable (needed for the real byte measurement)"
            return res
        try:
            import torch
            t = torch.load(tgt_path, map_location="cpu", weights_only=False)
            targ = np.asarray(t.numpy() if hasattr(t, "numpy") else t, dtype=np.float64)
        except Exception as e:
            res.status = "error"
            res.note = f"could not load pose target {tgt_path}: {e}"
            return res
        if targ.ndim != 2 or targ.shape[1] != 6:
            res.status = "error"
            res.note = f"pose target must be (N,6); got {targ.shape}"
            return res
        n = targ.shape[0]
        target_sha = _sha256_file(tgt_path)

        # --- baseline: scalar-store sidecar (fp16, zlib) — the reference bytes ---
        import zlib
        fp16 = targ.astype(np.float16).tobytes()
        baseline_bytes = len(zlib.compress(fp16, 9))
        res.rows.append(MeasurementRow(
            value=float(baseline_bytes), units="bytes", axis_tag=AxisTag.MACOS_CPU_ADVISORY,
            provenance=Provenance(git_sha=self.git_sha, tool="witness_apply_pass:low_rank_pose",
                                  seed=self.args.seed, config_ref=str(tgt_path.name),
                                  inputs_sha256=target_sha),
            n_samples=n, n_samples_reason=(None if n == 600 else f"pose target has {n} pairs"),
            review_status=ReviewStatus.PROVISIONAL, noise_floor=0.0,
            floor_provenance="zlib byte count is exact", quantity="archive_bytes[pose_scalar_store_baseline]"))

        # --- rank-k SVD codec: store U(N,k)*S, V(k,6) quantized+brotli ---
        rank = int(self.args.pose_rank)
        col_mean = targ.mean(axis=0, keepdims=True)
        centered = targ - col_mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        Uk = U[:, :rank] * S[:rank]           # (N, k) absorbs singular values
        Vk = Vt[:rank, :]                      # (k, 6)
        qb = int(self.args.pose_qbits)

        def _q_dq(arr: np.ndarray, q_bits: int) -> tuple[bytes, np.ndarray]:
            amax = float(np.max(np.abs(arr))) or 1.0
            scale = amax / (2 ** (q_bits - 1) - 1)
            q = np.round(arr / scale).astype(np.int16)
            blob = brotli.compress(q.tobytes(), quality=11)
            return blob, q.astype(np.float64) * scale  # dequantized (what decode sees)

        blob_u, Uk_dq = _q_dq(Uk, qb)
        blob_v, Vk_dq = _q_dq(Vk, qb)
        # recon MSE from the DEQUANTIZED factors (truncation + quantization = the real
        # d_pose floor the codec adds). errstate: macOS Accelerate BLAS emits spurious
        # matmul warnings; guard finiteness so a NaN can never form a MeasurementRow.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            recon = Uk_dq @ Vk_dq + col_mean
        recon_mse = float(np.mean((recon - targ) ** 2))
        if not np.isfinite(recon_mse):
            res.status = "error"
            res.note = "rank-k recon MSE non-finite (extreme pose-target values?)"
            return res
        blob_mean = brotli.compress(col_mean.astype(np.float16).tobytes(), quality=11)
        lowrank_bytes = len(blob_u) + len(blob_v) + len(blob_mean) + 12  # +section length prefixes
        res.rows.append(MeasurementRow(
            value=float(lowrank_bytes), units="bytes", axis_tag=AxisTag.MACOS_CPU_ADVISORY,
            provenance=Provenance(git_sha=self.git_sha, tool="witness_apply_pass:low_rank_pose",
                                  seed=self.args.seed, config_ref=f"rank{rank}_q{qb}/{tgt_path.name}",
                                  inputs_sha256=target_sha),
            n_samples=n, n_samples_reason=(None if n == 600 else f"pose target has {n} pairs"),
            review_status=ReviewStatus.PROVISIONAL, noise_floor=0.0,
            floor_provenance="brotli stream lengths are exact",
            quantity=f"archive_bytes[pose_low_rank_r{rank}_q{qb}]"))

        # reconstruction MSE == the advisory d_pose FLOOR the lossy codec adds (prediction,
        # NOT byte-closed through the real render). Contribution = sqrt(10 * d_pose).
        res.rows.append(MeasurementRow(
            value=recon_mse, units="mse_pose6", axis_tag=AxisTag.PREDICTION,
            provenance=Provenance(git_sha=self.git_sha, tool="witness_apply_pass:low_rank_pose",
                                  seed=self.args.seed, config_ref=f"rank{rank}/{tgt_path.name}",
                                  inputs_sha256=target_sha),
            n_samples=n, n_samples_reason=(None if n == 600 else f"pose target has {n} pairs"),
            review_status=ReviewStatus.PROVISIONAL, noise_floor=None, floor_provenance=None,
            quantity=f"d_pose_recon_floor[pose_low_rank_r{rank}]"))

        res.status = "measured"
        res.raw = {
            "pose_target": str(tgt_path), "n_pairs": n, "rank": rank, "qbits": qb,
            "baseline_scalar_store_bytes": baseline_bytes,
            "low_rank_bytes": lowrank_bytes,
            "byte_ratio_vs_scalar": round(baseline_bytes / max(lowrank_bytes, 1), 3),
            "recon_mse_d_pose_floor": recon_mse,
            "recon_sqrt10_contribution": float((10.0 * recon_mse) ** 0.5),
            "singular_values": S.tolist(),
        }
        res.staged_commands.append({
            "lever": lever,
            "measures": "n600 d_pose through the REAL render conditioned on the rank-k target",
            "authority": "recon MSE is a PREDICTION; real d_pose needs the byte-closed render",
            "argv": ["# fold rank-k pose section into levelset_byte_close --fold-pose-sidecar "
                     "--pose-sidecar-path <rank_k.npz> ; then parity on n600"],
            "fire_when": "v9 first good checkpoint; machine free",
        })
        return res

    # ======================================================================
    # STAGE 4: sidecar fold (fold stored-pose section, re-measure rate term)
    # ======================================================================
    def _sidecar_fold(self) -> StageResult:
        lever = "sidecar_fold"
        res = StageResult(lever=lever, kind="byte_measured")
        # levelset_byte_close --fold-pose-sidecar is fail-closed NO-FAKE: it REQUIRES a
        # built posenet_targets.bin (no fabrication). Without it, stage the command.
        sidecar = getattr(self.args, "pose_sidecar_path", None)
        fold_argv = [
            _PY, str(_LEVELSET_BYTE_CLOSE),
            "--ckpt-dir", str(self.frozen_dir), "--npz-name", self.npz_name,
            "--skip-parity", "--fold-pose-sidecar",
            "--pose-sidecar-path", str(sidecar) if sidecar else "<posenet_targets.bin>",
            "--out", str(self.out_dir / "sidecar_fold_byte_close.json"),
        ]
        if self.args.max_pairs:
            fold_argv += ["--max-pairs", str(self.args.max_pairs)]
        if not sidecar or not Path(sidecar).exists():
            res.status = "staged"
            res.note = ("--fold-pose-sidecar is fail-closed (NO-FAKE): needs a built "
                        "posenet_targets.bin. Build it (tac.scorer_targets.extract_and_save "
                        "-> pose sidecar bin) then pass --pose-sidecar-path. Staged.")
            res.staged_commands.append({
                "lever": lever,
                "measures": "archive rate term WITH the stored-pose section folded in (delta bytes "
                            "vs baseline); a code-pose witness pays DEAD bytes (loud honesty)",
                "authority": "[macOS-CPU advisory] byte term is exact",
                "argv": fold_argv, "fire_when": "pose sidecar bin built; machine free",
            })
            return res
        rc, so, se = self._run(fold_argv, timeout=self.args.byte_timeout)
        out_json = self.out_dir / "sidecar_fold_byte_close.json"
        if rc != 0 or not out_json.exists():
            res.status = "error"
            res.note = f"sidecar-fold rc={rc}; stderr tail: {se[-500:]}"
            return res
        rep = json.loads(out_json.read_text())
        zb = (rep.get("byte_close", {}) or {}).get("archive_zip_bytes")
        if isinstance(zb, int):
            res.status = "measured"
            res.rows.append(self._bytes_row(
                "levelset_byte_close_and_eval", zb, "archive_zip_bytes[sidecar_fold]"))
            res.raw = {"archive_zip_bytes": zb, "report": str(out_json)}
        else:
            res.status = "error"
            res.note = "no archive_zip_bytes in folded report"
        return res

    # ======================================================================
    # STUB STAGES: #311 TropNNC, #401 blind-coordinate (NOT built)
    # ======================================================================
    def _stub(self, lever: str, human: str, what: str) -> StageResult:
        res = StageResult(lever=lever, kind="stub", status="owed")
        if self.args.run_stub and lever in self.args.run_stub:
            raise NotImplementedError(
                f"STUB '{lever}' ({human}) is NOT BUILT. #406 registers it as an explicit "
                f"fail-loud stub per NO-FAKE — a stub must say it is one, never pretend-apply. "
                f"What it WOULD do: {what}. Build it, then remove it from the stub registry.")
        res.note = (f"OWED — {human} is NOT built. {what} "
                    f"Invoking --run-stub {lever} raises NotImplementedError (no fake apply).")
        return res

    # ======================================================================
    def run(self) -> dict:
        stages: list[StageResult] = []
        # 1. baseline byte-close (rate reference)
        stages.append(self._byte_close("baseline", extra_argv=[]))
        # 2. #336 sensitivity bit-alloc
        stages.append(self._bit_alloc())
        # 3. #140 low-rank pose codec
        stages.append(self._low_rank_pose())
        # 4. sidecar fold (byte-neutral recode orbit)
        stages.append(self._sidecar_fold())
        # 5 + 6. stubs
        stages.append(self._stub(
            "tropnnc_311", "#311 TropNNC (tropical-geometry NN compression)",
            "tropical-argmax-aware structured pruning of the witness trunk before byte-close."))
        stages.append(self._stub(
            "blind_coord_401", "#401 blind-coordinate transform",
            "a blind (data-independent) coordinate rotation of the code/latent to flatten "
            "the symbol distribution before int8+brotli."))

        summary = {
            "task": "#406 apply-pass readiness",
            "ts": self.ts,
            "git_sha": self.git_sha,
            "frozen_ckpt": str(self.frozen_dir),
            "npz_name": self.npz_name,
            "npz_sha256": self.npz_sha,
            "config_ref": self.config_ref,
            "authority": "[macOS-CPU advisory] NON-PROMOTABLE; pointer UNMOVED (MEANS)",
            "fire_scorer_stages": bool(self.args.fire_scorer_stages),
            "max_pairs": self.args.max_pairs,
            "stages": [s.to_json_dict() for s in stages],
            "owed": [s.lever for s in stages if s.status == "owed"],
            "errors": [{"lever": s.lever, "note": s.note} for s in stages if s.status == "error"],
        }
        # write rows JSONL + summary
        rows_path = self.out_dir / "apply_pass_rows.jsonl"
        with rows_path.open("w") as fh:
            for s in stages:
                for r in s.rows:
                    fh.write(json.dumps({"lever": s.lever, **r.to_json_dict()}) + "\n")
        summary_path = self.out_dir / "apply_pass_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        self._print_table(stages, summary_path, rows_path)
        return summary

    def _print_table(self, stages: list[StageResult], summary_path: Path, rows_path: Path) -> None:
        print("\n" + "=" * 78)
        print("#406 APPLY-PASS — per-lever dry-run rows  [macOS-CPU advisory] NON-PROMOTABLE")
        print("=" * 78)
        for s in stages:
            tag = {"measured": "MEASURED", "staged": "STAGED", "owed": "OWED (stub)",
                   "error": "ERROR"}.get(s.status, s.status)
            print(f"\n[{s.lever}]  kind={s.kind}  status={tag}")
            for r in s.rows:
                d = r.to_json_dict()
                print(f"    {d['quantity']:<44} = {d['value']:>14.6g} {d['units']:<18} "
                      f"{d['axis_tag']} n={d['n_samples']}")
            if s.raw.get("byte_ratio_vs_scalar"):
                print(f"    -> byte ratio vs scalar-store: {s.raw['byte_ratio_vs_scalar']}x ; "
                      f"recon sqrt(10*d_pose) contribution: {s.raw['recon_sqrt10_contribution']:.4f}")
            if s.status in ("staged", "owed", "error") and s.note:
                print(f"    note: {s.note[:200]}")
            for sc in s.staged_commands:
                print(f"    STAGED[{sc['lever']}]: {sc['measures']}")
        print("\n" + "-" * 78)
        print(f"summary: {summary_path}")
        print(f"rows:    {rows_path}")
        print("-" * 78)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True,
                    help="FROZEN run dir (copied in; NEVER the live run).")
    ap.add_argument("--npz-name", default="levelset_witness_ema_BEST.npz")
    ap.add_argument("--pose-target", default=None,
                    help="path to a (600,6) stored PoseNet target .pt for the #140 low-rank codec")
    ap.add_argument("--pose-sidecar-path", default=None,
                    help="built posenet_targets.bin for the sidecar-fold stage (fail-closed without it)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=6,
                    help="dry-run byte-close pair cap (byte term is exact regardless)")
    ap.add_argument("--gt-cache", default=_GT_CACHE_DEFAULT)
    ap.add_argument("--byte-timeout", type=float, default=900.0)
    ap.add_argument("--scorer-timeout", type=float, default=1800.0)
    # #336 bit-alloc knobs (only used with --fire-scorer-stages)
    ap.add_argument("--fire-scorer-stages", action="store_true",
                    help="actually FIRE the scorer-heavy stages (default: STAGE them). "
                         "Only when the machine is free — the live run owns the GPU.")
    ap.add_argument("--probe-pairs", type=int, default=2)
    ap.add_argument("--eval-pairs", type=int, default=4)
    ap.add_argument("--mean-bits", type=float, nargs="+", default=[6.0, 5.0])
    # #140 low-rank pose knobs
    ap.add_argument("--pose-rank", type=int, default=2)
    ap.add_argument("--pose-qbits", type=int, default=10)
    # stubs
    ap.add_argument("--run-stub", nargs="*", default=None,
                    help="fail LOUDLY on the named stub lever(s) (tropnnc_311 / blind_coord_401)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    passer = ApplyPass(args)
    passer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
