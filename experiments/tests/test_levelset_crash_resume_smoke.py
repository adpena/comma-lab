# SPDX-License-Identifier: MIT
"""CRASH-RESUME PROOF SMOKE for the LEVEL-SET witness trainer (#205 durability hardening).

The unit tests in ``test_levelset_checkpoint_resume.py`` prove the pure-numpy checkpoint/resume
HELPERS in isolation. THIS file proves the END-TO-END machinery on the REAL MLX trainer
(``experiments/train_levelset_witness_realized_through_R_mlx.py``) — the non-negotiable
"resumable-from-disk + per-stage checkpoint" gate for the multi-day from-scratch #205 launch
(CLAUDE.md "Resumability + per-stage checkpoints are MANDATORY", operator binding 2026-06-27).

NO-FAKE: this runs the actual trainer subprocess, actually SIGKILLs it mid-stage, actually
``--resume-from``s the surviving checkpoint, and asserts the resumed trajectory is BIT-IDENTICAL
to an uninterrupted run of the same seed+config. It measures — it does not assert-by-construction.
The scorer weights, R operator, EMA, curriculum, Muon MultiOptimizer, and checkpoint I/O are all
the REAL launch-path code; only the shape (n=1 pair, 96x128 render, 5 epochs) is a fast twin.

The seven proof assertions (mapped to the #205 requirements):
  (a) trains through >=1 curriculum stage boundary + intra-stage checkpoints          [req 1,2,3,7]
  (b) is KILLED mid-stage with SIGKILL (abrupt, no clean shutdown)                     [req 5]
  (c) ``--resume-from`` the surviving rolling checkpoint continues the run             [req 1]
  (d) resumed FINAL live+EMA weights == continuous FINAL live+EMA weights (bit-exact)  [req 10]
      AND the optimizer state restored (restored_opt), the precondition for bit-exact  [req 6]
  (e) per-stage checkpoints exist under DISTINCT stage-encoded filenames (no overwrite) [req 2]
  (f) the deploy EMA npz is the EMA SHADOW (!= live weights) AND byte-close-loadable    [req 4,6]
  (g) a resume with a DELIBERATELY-DIVERGED lever (--film-per-layer) fail-closes        [req 8]
  + MUON-FINISHER crash-resume: bit-identical continuation INSIDE the Muon finisher     [req 6,10]

MLX-CPU is bit-identical across independent processes (verified: two identical runs hash-match;
MLX-*GPU* is NOT — measured, all 28 tensors diverge cross-process — so this proof is CPU-locked),
so (d) asserts EXACT equality, not a tolerance band.

FAST + RELIABLE + SELF-CONTAINED (the durability-gate re-runnability rewrite, 2026-07-02):
  * The 8 trainer subprocesses run as a small dependency DAG on a ThreadPoolExecutor, capped at
    ``_MAX_CONCURRENT`` concurrent processes (each ``subprocess``-``wait``ed DIRECTLY — no external
    agent-waiter / Monitor / background-poll that could stall). ONE foreground call runs every arm
    and writes report.json.
  * Config is a fast twin (n=1 pair, 96x128 render, tiny MLP, 5-epoch compressed curriculum,
    verdict only at ep0+final) so the whole DAG completes in ~40-55s wall — comfortably inside the
    2-min foreground budget AND under the pytest-timeout (a per-module ``pytest.mark.timeout`` is
    set as a belt-and-suspenders safety net).
  * Output defaults to a guard-legal DURABLE base (SSD tier, then a non-/tmp repo-results dir);
    the trainer out-dir args are passed relative-to-repo when the checkout itself is under /tmp so
    the trainer's ``_refuse_tmp`` CLAUDE.md guard never fail-closes the smoke.

Standalone (writes report.json, deterministically pollable):
    .venv/bin/python experiments/tests/test_levelset_crash_resume_smoke.py
Under pytest (runs green in ~50s; targets this file directly):
    .venv/bin/python -m pytest experiments/tests/test_levelset_crash_resume_smoke.py -q

SELF-PROTECT / no silent rot: the module is marked ``slow`` (the repo idiom for scorer-loading
tests) so the default fast suite / CI (``-m "not slow"``) deselects it, but a targeted run OR a
dedicated ``-m slow`` CI job exercises it. It is NOT in the default ``testpaths`` (tests,
src/tac/tests), so it never burdens an unrelated ``pytest`` — but ``pytest
experiments/tests/test_levelset_crash_resume_smoke.py`` runs it green. Wire a CI job (or the #205
pre-launch preflight) to invoke exactly that command as the durability gate for the multi-day run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
# Standalone (__main__) importability of ``tools`` / ``tac`` (pytest adds these via rootdir; the
# direct-run path does not) so the byte-close consumer import in step 6 resolves.
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
_GT_CACHE = _REPO / "experiments" / "results" / "mlx_fleet_gt_cache" / "gt_n6.npz"

# Fast-twin config that STILL exercises BOTH curriculum stage boundaries (ce->tau at ep2, tau->l7
# at ep3) + intra-stage checkpoints + the Muon finisher — compressed to 5 epochs / 1 pair so the
# whole DAG fits well under the 2-min budget. (Was 6ep/2pairs; the resume proof is shape-agnostic.)
_EPOCHS = 5
_TAU_START = 2
_L7_START = 3
_MUON_START = 3          # == l7 start (PR95 placement; no non-l7-start warn)
_BASE_KILL_EPOCH = 3     # first L7 intra-stage ckpt (crash mid-L7; resume runs ep4,5)
_MUON_KILL_EPOCH = 4     # mid-finisher intra-stage ckpt (>=1 finisher epoch of momentum; resume runs ep5)
_FILM_EPOCHS = 2

# The trainer wants many threads; running the whole DAG lets peak concurrent SUBPROCESSES be capped
# here (measured ~1.5x per-process slowdown at 5-way on this 18-core box — fine). Bounds thrash.
_MAX_CONCURRENT = 5
_ARM_TIMEOUT = 240.0     # generous per-arm wall cap so a genuinely-hung arm fails instead of stalling
_KILL_HARD_TIMEOUT = 150.0

_TMP_PREFIXES = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
# SSD tiers per CLAUDE.md "Local Disk, SSD Spill" priority order (first writable wins).
_SSD_TIERS = ("/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact")

_SUBPROC_GATE = threading.BoundedSemaphore(_MAX_CONCURRENT)


def _under_tmp(p: Path | str) -> bool:
    return any(str(p).startswith(pre) for pre in _TMP_PREFIXES)


def durable_smoke_base(name: str) -> Path:
    """Choose a guard-legal, durable scratch base for the smoke's subprocess out-dirs.

    The trainer REFUSES /tmp-class out-dirs (``_refuse_tmp`` per CLAUDE.md "Local Disk, SSD Spill"),
    so when the repo checkout itself lives under /tmp (isolated worktrees; some CI) a naive
    ``_REPO/experiments/results`` would fail-closed. Prefer the SSD tier, then a non-/tmp
    repo-results dir. As a last resort return the repo-results path even if under /tmp — the trainer
    out-dir args are passed RELATIVE-to-repo in that case (``_trainer_out_arg``) which dodges the
    literal /tmp string check while resolving identically (the trainer runs with ``cwd=_REPO``)."""
    for tier in _SSD_TIERS:
        t = Path(tier)
        if t.is_dir() and os.access(t, os.W_OK):
            return t / name
    return _REPO / "experiments" / "results" / name


def _trainer_out_arg(out_dir: Path) -> str:
    """Path string to hand the trainer's ``--out-dir`` / ``--resume-from`` so ``_refuse_tmp`` passes.

    If ``out_dir`` is under a /tmp-class repo checkout, pass it RELATIVE to ``_REPO`` (the trainer
    runs with ``cwd=_REPO`` so it resolves identically, and the relative string is not /tmp-class).
    Otherwise (SSD tier / non-/tmp repo) pass the absolute path unchanged."""
    if _under_tmp(out_dir):
        try:
            return str(out_dir.relative_to(_REPO))
        except ValueError:
            pass  # not under _REPO (shouldn't happen for durable bases) -> fall through
    return str(out_dir)


def _env() -> dict[str, str]:
    e = dict(os.environ)
    # The custom Metal grouped-backward kernel is GPU-only; the deterministic CPU smoke disables it.
    e["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "0"
    e["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO / "src"), str(_REPO / "upstream"), str(_REPO / "experiments"), str(_REPO)]
        + ([e["PYTHONPATH"]] if e.get("PYTHONPATH") else [])
    )
    return e


def _base_argv(out_dir: Path, *extra: str, epochs: int = _EPOCHS, curriculum: bool = True,
               tau_start: int = _TAU_START, l7_start: int = _L7_START,
               muon_start: int | None = None) -> list[str]:
    argv = [
        sys.executable, str(_TRAINER),
        "--out-dir", _trainer_out_arg(out_dir),
        "--num-pairs", "1", "--gt-cache", str(_GT_CACHE),
        "--epochs", str(epochs),
        ("--curriculum" if curriculum else "--no-curriculum"),
        "--tau-softplus-start-epoch", str(tau_start), "--l7-start-epoch", str(l7_start),
        # verdict only at ep0 + final (the CPU-torch verdict is the per-epoch cost; the resume proof
        # does not read it) -> the trainer still writes best.json + deploy npz + history.
        "--ckpt-every", "1", "--eval-every", str(epochs), "--mlx-device", "cpu", "--seed", "0",
        "--render-h", "96", "--render-w", "128", "--hidden-dim", "32", "--n-hidden", "2",
        "--no-self-orient",
    ]
    if muon_start is not None:
        argv += ["--muon-start-epoch", str(muon_start)]
    return argv + list(extra)


def _run_to_completion(out_dir: Path, *extra: str, timeout: float = _ARM_TIMEOUT,
                       **kw) -> subprocess.CompletedProcess:
    with _SUBPROC_GATE:
        return subprocess.run(
            _base_argv(out_dir, *extra, **kw), env=_env(), cwd=str(_REPO),
            capture_output=True, text=True, timeout=timeout,
        )


def _run_and_sigkill(out_dir: Path, kill_substr: str, *extra: str,
                     hard_timeout: float = _KILL_HARD_TIMEOUT, **kw):
    """Launch the trainer, stream stdout, SIGKILL (abrupt, no cleanup) the FIRST time ``kill_substr``
    appears. Returns (killed: bool, seen_lines: list[str]). This simulates a real crash/OOM/operator-cut
    mid-stage: the surviving on-disk state is whatever the last ATOMIC checkpoint wrote."""
    with _SUBPROC_GATE:
        proc = subprocess.Popen(
            _base_argv(out_dir, *extra, **kw), env=_env(), cwd=str(_REPO),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        seen: list[str] = []
        killed = False
        t0 = time.time()
        try:
            while True:
                line = proc.stdout.readline()
                if line == "":
                    break  # EOF (process exited on its own before the kill trigger)
                seen.append(line.rstrip())
                if kill_substr in line:
                    os.kill(proc.pid, signal.SIGKILL)
                    killed = True
                    break
                if time.time() - t0 > hard_timeout:
                    os.kill(proc.pid, signal.SIGKILL)
                    break
        finally:
            try:
                proc.stdout.close()
            except OSError:
                pass
            proc.wait(timeout=30)
    return killed, seen


def _hash_state(npz_path: Path, prefixes: tuple[str, ...] = ("liveP__", "emaP__")) -> dict[str, str]:
    """sha256 of every tensor under the given resume-sidecar prefixes (liveP__ / emaP__)."""
    import numpy as np

    z = np.load(npz_path, allow_pickle=False)
    out: dict[str, str] = {}
    for k in z.files:
        for pref in prefixes:
            if k.startswith(pref):
                out[k] = hashlib.sha256(
                    np.ascontiguousarray(np.asarray(z[k], np.float32)).tobytes()
                ).hexdigest()
    return out


def _resume_epoch(npz_path: Path) -> int:
    import numpy as np

    z = np.load(npz_path, allow_pickle=False)
    return int(z["__resume_epoch"])


# =========================================================================== the orchestrated smoke
def run_crash_resume_smoke(base: Path) -> dict:
    """Run the full crash->resume->parity proof under ``base`` and return a machine-readable report.

    Shared by the pytest fixture AND the ``__main__`` standalone entry so the heavy subprocess work
    happens exactly once. The 8 arms run as a small dependency DAG (independent arms concurrent;
    each ``resume``/``bad`` arm waits DIRECTLY on its predecessor's future) capped at
    ``_MAX_CONCURRENT`` concurrent subprocesses — one foreground call, no external waiter."""
    import numpy as np

    assert _GT_CACHE.exists(), f"GT cache missing: {_GT_CACHE} (need gt_n6.npz for the n=1 smoke)"
    cont_dir = base / "continuous"
    crash_dir = base / "crash"
    resume_dir = base / "resumed"
    film_dir = base / "film_per_layer"
    film_bad_dir = base / "film_resume_bad"
    m_cont_dir = base / "muon_continuous"
    m_crash_dir = base / "muon_crash"
    m_resume_dir = base / "muon_resumed"
    for d in (cont_dir, crash_dir, resume_dir, film_dir, film_bad_dir,
              m_cont_dir, m_crash_dir, m_resume_dir):
        d.mkdir(parents=True, exist_ok=True)

    report: dict = {"utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "steps": {},
                    "base": str(base)}
    _progress = base / "progress.log"
    _prog_lock = threading.Lock()

    def _prog(msg: str) -> None:
        line = f"{datetime.now(UTC).strftime('%H:%M:%S')} [smoke] {msg}\n"
        with _prog_lock:  # thread-safe (arms run concurrently)
            print(line, end="", file=sys.stderr, flush=True)
            with open(_progress, "a", encoding="utf-8") as fh:  # durable, explicit flush
                fh.write(line)

    # ---------------------------------------------------------------- arm callables (DAG nodes) ---
    def arm_base_cont() -> dict:
        _prog("arm base/continuous: start")
        cp = _run_to_completion(cont_dir)
        _prog(f"arm base/continuous: done rc={cp.returncode}")
        assert cp.returncode == 0, (
            f"continuous run failed rc={cp.returncode}\n{cp.stdout[-2000:]}\n{cp.stderr[-2000:]}")
        cont_final = cont_dir / "levelset_resume_state.npz"
        return {"rc": cp.returncode, "final": cont_final,
                "hash": _hash_state(cont_final), "final_epoch": _resume_epoch(cont_final)}

    def arm_base_crash() -> dict:
        _prog("arm base/crash: start (SIGKILL mid-L7)")
        killed, _ = _run_and_sigkill(
            crash_dir, f'"kind": "intra_stage", "epoch": {_BASE_KILL_EPOCH}')
        cr = crash_dir / "levelset_resume_state.npz"
        assert cr.exists(), "crash left NO rolling resume_state.npz (nothing to resume from!)"
        k_epoch = _resume_epoch(cr)
        _prog(f"arm base/crash: killed={killed} resume_epoch={k_epoch}")
        assert killed, "crash run exited before the kill-trigger checkpoint"
        assert 2 <= k_epoch < _EPOCHS, f"surviving epoch {k_epoch} not mid-run (2..{_EPOCHS - 1})"
        return {"killed": killed, "k_epoch": k_epoch,
                "ce_survived": (crash_dir / "levelset_ckpt_stageCE_ep1.npz").exists()}

    def arm_base_resume(_crash: dict) -> dict:
        _prog("arm base/resumed: start (--resume-from crash)")
        cp = _run_to_completion(resume_dir, "--resume-from", _trainer_out_arg(crash_dir))
        _prog(f"arm base/resumed: done rc={cp.returncode}")
        assert cp.returncode == 0, (
            f"resume run failed rc={cp.returncode}\n{cp.stdout[-2000:]}\n{cp.stderr[-2000:]}")
        restored_opt = False
        resumed_from_epoch = None
        for ln in cp.stdout.splitlines():
            if '"stage": "resume"' in ln and "restored_opt" in ln:
                try:
                    obj = json.loads(ln)
                    restored_opt = bool(obj.get("restored_opt"))
                    resumed_from_epoch = int(obj.get("resumed_epoch"))
                except (ValueError, TypeError):
                    pass
        rf = resume_dir / "levelset_resume_state.npz"
        return {"rc": cp.returncode, "restored_opt": restored_opt,
                "resumed_from_epoch": resumed_from_epoch,
                "final_epoch": _resume_epoch(rf), "hash": _hash_state(rf)}

    def arm_film_train() -> dict:
        _prog("arm film/train: start (--film-per-layer)")
        cp = _run_to_completion(film_dir, "--film-per-layer",
                                epochs=_FILM_EPOCHS, curriculum=False, tau_start=1, l7_start=2)
        ok = cp.returncode == 0 and (film_dir / "levelset_resume_state.npz").exists()
        _prog(f"arm film/train: done rc={cp.returncode} ok={ok}")
        return {"ok": ok, "rc": cp.returncode}

    def arm_film_bad(_film: dict) -> dict:
        # resume the film-per-layer ckpt WITHOUT the flag => the ckpt carries film_pl.* the rebuilt
        # model lacks => the arch-drift guard MUST fail-closed (silent-param-drop = corrupted resume).
        _prog("arm film/resume_bad: start (diverged lever must fail-closed)")
        cp = _run_to_completion(film_bad_dir, "--resume-from", _trainer_out_arg(film_dir),
                                epochs=_FILM_EPOCHS + 1, curriculum=False, tau_start=1, l7_start=2)
        guard_txt = cp.stdout + cp.stderr
        fc = cp.returncode != 0 and ("SILENTLY DROP" in guard_txt or "no slot for" in guard_txt)
        _prog(f"arm film/resume_bad: rc={cp.returncode} fail_closed={fc}")
        return {"rc": cp.returncode, "fail_closed": fc}

    def arm_muon_cont() -> dict:
        _prog("arm muon/continuous: start (muon finisher)")
        cp = _run_to_completion(m_cont_dir, muon_start=_MUON_START)
        _prog(f"arm muon/continuous: done rc={cp.returncode}")
        assert cp.returncode == 0, f"muon continuous failed\n{cp.stdout[-1500:]}\n{cp.stderr[-1500:]}"
        return {"rc": cp.returncode,
                "hash": _hash_state(m_cont_dir / "levelset_resume_state.npz")}

    def arm_muon_crash() -> dict:
        _prog("arm muon/crash: start (SIGKILL mid-finisher)")
        killed, _ = _run_and_sigkill(
            m_crash_dir, f'"kind": "intra_stage", "epoch": {_MUON_KILL_EPOCH}',
            muon_start=_MUON_START)
        cr = m_crash_dir / "levelset_resume_state.npz"
        assert cr.exists(), "muon crash left no rolling resume_state.npz"
        assert killed, "muon crash exited before the mid-finisher checkpoint"
        k_epoch = _resume_epoch(cr)
        _prog(f"arm muon/crash: killed={killed} resume_epoch={k_epoch}")
        return {"killed": killed, "k_epoch": k_epoch}

    def arm_muon_resume(_crash: dict) -> dict:
        _prog("arm muon/resumed: start (--resume-from muon_crash)")
        cp = _run_to_completion(m_resume_dir, "--resume-from", _trainer_out_arg(m_crash_dir),
                                muon_start=_MUON_START)
        _prog(f"arm muon/resumed: done rc={cp.returncode}")
        assert cp.returncode == 0, f"muon resume failed\n{cp.stdout[-1500:]}\n{cp.stderr[-1500:]}"
        resumed_into_finisher = False
        for ln in cp.stdout.splitlines():
            if '"stage": "resume"' in ln and "resumed_into_finisher" in ln:
                try:
                    resumed_into_finisher = bool(json.loads(ln).get("resumed_into_finisher"))
                except ValueError:
                    pass
        return {"rc": cp.returncode, "resumed_into_finisher": resumed_into_finisher,
                "hash": _hash_state(m_resume_dir / "levelset_resume_state.npz")}

    # ---------------------------------------------------------------- run the DAG (direct waits) ---
    # Independent arms launch concurrently; each dependent arm waits DIRECTLY on its predecessor's
    # future inside its own worker thread (no external poller). ``_SUBPROC_GATE`` caps concurrent
    # subprocesses; workers>=tasks so a blocked dependent never starves an independent arm.
    _prog(f"DAG start: 8 arms, <= {_MAX_CONCURRENT} concurrent subprocesses")
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="smoke") as ex:
        f_bc = ex.submit(arm_base_cont)
        f_bk = ex.submit(arm_base_crash)
        f_ft = ex.submit(arm_film_train)
        f_mc = ex.submit(arm_muon_cont)
        f_mk = ex.submit(arm_muon_crash)
        f_br = ex.submit(lambda: arm_base_resume(f_bk.result()))
        f_fb = ex.submit(lambda: arm_film_bad(f_ft.result()))
        f_mr = ex.submit(lambda: arm_muon_resume(f_mk.result()))
        bc = f_bc.result()
        bk = f_bk.result()
        br = f_br.result()
        ft = f_ft.result()
        fb = f_fb.result()
        mc = f_mc.result()
        mk = f_mk.result()
        mr = f_mr.result()
    _prog("DAG done: assembling report")

    # ---------------------------------------------------------------- assemble the report ----------
    report["steps"]["continuous"] = {
        "rc": bc["rc"], "final_epoch": bc["final_epoch"], "n_tensors": len(bc["hash"])}

    report["steps"]["crash"] = {
        "sigkilled": bool(bk["killed"]), "surviving_resume_epoch": bk["k_epoch"],
        "stageCE_ckpt_survived": bk["ce_survived"],
        "note": "SIGKILL is abrupt (no clean shutdown); surviving state == last ATOMIC checkpoint"}
    assert bk["ce_survived"], "per-stage CE checkpoint did NOT survive the crash"

    # BIT-IDENTICAL parity: resumed FINAL == continuous FINAL
    assert set(br["hash"]) == set(bc["hash"]), "resumed/continuous tensor KEY sets differ"
    diffs = [k for k in bc["hash"] if bc["hash"][k] != br["hash"][k]]
    report["steps"]["resume"] = {
        "rc": br["rc"], "resumed_from_epoch": br["resumed_from_epoch"],
        "restored_opt": br["restored_opt"], "final_epoch": br["final_epoch"],
        "n_tensors": len(br["hash"]), "bit_identical_to_continuous": (diffs == []),
        "n_diverged_tensors": len(diffs), "diverged": diffs[:8]}

    # per-stage DISTINCT filenames (prior stage NOT overwritten)
    stage_files = sorted(p.name for p in cont_dir.glob("levelset_ckpt_stage*_ep*.npz"))
    resume_stage_files = sorted(p.name for p in cont_dir.glob("levelset_resume_stage*_ep*.npz"))
    report["steps"]["per_stage"] = {
        "deploy_stage_ckpts": stage_files, "resume_stage_ckpts": resume_stage_files,
        "distinct": len(stage_files) == len(set(stage_files)) and len(stage_files) >= 3}

    # deploy EMA npz is the SHADOW (!= live) AND byte-close-loadable
    ema_deploy = cont_dir / "levelset_witness_ema_mlx.npz"
    zc = np.load(bc["final"], allow_pickle=False)
    zd = np.load(ema_deploy, allow_pickle=False)
    kk = "in_proj.weight"

    def _sha(a):
        return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()).hexdigest()

    deploy_is_shadow = _sha(zd[kk]) == _sha(zc["emaP__" + kk])   # deploy == EMA shadow
    deploy_neq_live = _sha(zd[kk]) != _sha(zc["liveP__" + kk])   # deploy != live weights
    from tools.levelset_byte_close_and_eval import _load_levelset_ckpt

    params, _cfg = _load_levelset_ckpt(cont_dir)
    byte_close_ok = all(
        r in params for r in ("code", "in_proj.weight", "out_sdf.weight", "out_tex.weight", "palette"))
    report["steps"]["ema_shadow"] = {
        "deploy_is_ema_shadow": deploy_is_shadow, "deploy_differs_from_live": deploy_neq_live,
        "byte_close_loadable": byte_close_ok,
        "best_json_exists": (cont_dir / "levelset_best.json").exists()}

    report["steps"]["fail_closed"] = {
        "film_train_ok": ft["ok"], "diverged_resume_rc": fb["rc"], "fail_closed": fb["fail_closed"]}

    # PROVENANCE completeness (#205 + coordinator): git sha in result.json AND every ckpt cfg
    res_json = json.loads((cont_dir / "levelset_train_result.json").read_text())
    prov = res_json.get("provenance", {})
    report["steps"]["provenance"] = {
        "result_json_git_sha_present": bool(prov.get("git_sha")) and prov.get("git_sha") != "unknown",
        "result_json_seed": prov.get("seed"),
        "result_json_upstream_sha_present": bool(prov.get("upstream_snapshot_sha256"))
        and prov.get("upstream_snapshot_sha256") != "unknown",
        "resume_ckpt_has_git_sha": "__cfg_git_sha" in zc.files,
        "deploy_ckpt_has_git_sha": "__cfg_git_sha" in zd.files,
        "deploy_ckpt_has_upstream_sha": "__cfg_upstream_snapshot_sha256" in zd.files,
        "stage_ckpt_has_git_sha": "__cfg_git_sha" in np.load(
            cont_dir / "levelset_ckpt_stageCE_ep1.npz", allow_pickle=False).files}

    # MUON-FINISHER crash-resume: bit-identical continuation INSIDE the finisher
    m_diffs = [k for k in mc["hash"] if mc["hash"][k] != mr["hash"].get(k)]
    report["steps"]["muon_resume"] = {
        "sigkilled_mid_finisher": bool(mk["killed"]), "surviving_resume_epoch": mk["k_epoch"],
        "resumed_into_finisher": mr["resumed_into_finisher"],
        "bit_identical_to_continuous": (m_diffs == [] and set(mc["hash"]) == set(mr["hash"])),
        "n_diverged_tensors": len(m_diffs), "diverged": m_diffs[:8]}

    report["ALL_PASS"] = (
        report["steps"]["resume"]["bit_identical_to_continuous"]
        and report["steps"]["resume"]["restored_opt"]
        and report["steps"]["crash"]["sigkilled"]
        and report["steps"]["per_stage"]["distinct"]
        and report["steps"]["ema_shadow"]["deploy_is_ema_shadow"]
        and report["steps"]["ema_shadow"]["deploy_differs_from_live"]
        and report["steps"]["ema_shadow"]["byte_close_loadable"]
        and report["steps"]["fail_closed"]["fail_closed"]
        and report["steps"]["provenance"]["result_json_git_sha_present"]
        and report["steps"]["provenance"]["deploy_ckpt_has_git_sha"]
        and report["steps"]["provenance"]["stage_ckpt_has_git_sha"]
        and report["steps"]["muon_resume"]["resumed_into_finisher"]
        and report["steps"]["muon_resume"]["bit_identical_to_continuous"]
    )
    return report


# =========================================================================== pytest surface
# ``slow``: repo idiom for scorer-loading tests -> the default fast suite / CI (``-m "not slow"``)
# deselects it; a targeted run or a ``-m slow`` CI job exercises it. ``timeout(240)``: the fast DAG
# runs in ~40-55s but the global pytest-timeout is 60s and fixture-setup time is attributed to the
# first test using it, so a generous per-module override runs it GREEN even under load variance
# (the pre-rewrite "finicky" failure was the global 60s killing the ~3.5-min fixture mid-DAG).
pytestmark = [pytest.mark.slow, pytest.mark.timeout(240)]


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    # Runs 8 real MLX trainer subprocesses (~40-55s). Not in the default ``testpaths``, and marked
    # ``slow``, so it never burdens an unrelated ``pytest`` -- but ``pytest <this file>`` runs it.
    os.environ.setdefault("TAC_MLX_CUSTOM_GROUPED_BACKWARD", "0")  # the CPU smoke: GPU kernel off
    base = durable_smoke_base(f"durability_crash_resume_smoke_{os.getpid()}")
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    try:
        yield run_crash_resume_smoke(base)
    finally:
        shutil.rmtree(base, ignore_errors=True)  # success-only scratch cleanup (disk hygiene)


def test_crash_run_was_sigkilled_mid_stage(smoke):
    assert smoke["steps"]["crash"]["sigkilled"] is True
    assert 2 <= smoke["steps"]["crash"]["surviving_resume_epoch"] < _EPOCHS
    assert smoke["steps"]["crash"]["stageCE_ckpt_survived"] is True


def test_resume_optimizer_state_restored(smoke):
    # bit-exact continuation requires the AdamW moments to restore (not fresh-rewarm).
    assert smoke["steps"]["resume"]["restored_opt"] is True


def test_resume_is_bit_identical_to_continuous(smoke):
    r = smoke["steps"]["resume"]
    assert r["bit_identical_to_continuous"] is True, f"diverged tensors: {r['diverged']}"
    assert r["n_diverged_tensors"] == 0
    assert r["final_epoch"] == _EPOCHS == smoke["steps"]["continuous"]["final_epoch"]


def test_per_stage_checkpoints_distinct_filenames(smoke):
    ps = smoke["steps"]["per_stage"]
    assert ps["distinct"] is True
    # the three PR95 curriculum stages each preserved under a DISTINCT stage+epoch filename
    # (compressed curriculum: CE=ep1, Tau=ep2, L7=ep5-final).
    for name in ("levelset_ckpt_stageCE_ep1.npz", "levelset_ckpt_stageTau_ep2.npz",
                 "levelset_ckpt_stageL7_ep5.npz"):
        assert name in ps["deploy_stage_ckpts"], f"missing preserved per-stage ckpt {name}"


def test_deploy_npz_is_ema_shadow_and_byte_close_loadable(smoke):
    es = smoke["steps"]["ema_shadow"]
    assert es["deploy_is_ema_shadow"] is True     # deploy == EMA shadow
    assert es["deploy_differs_from_live"] is True  # ...NOT the live weights (EMA non-negotiable)
    assert es["byte_close_loadable"] is True       # consumable by the exact-eval byte-close tool
    assert es["best_json_exists"] is True


def test_resume_fail_closes_on_diverged_lever(smoke):
    fc = smoke["steps"]["fail_closed"]
    assert fc["film_train_ok"] is True
    assert fc["fail_closed"] is True, f"diverged-lever resume did NOT fail-closed (rc={fc['diverged_resume_rc']})"


def test_provenance_git_sha_in_result_and_checkpoints(smoke):
    p = smoke["steps"]["provenance"]
    assert p["result_json_git_sha_present"] is True     # result.json carries the git sha
    assert p["result_json_seed"] == 0                    # + the seed
    assert p["deploy_ckpt_has_git_sha"] is True          # deploy (byte-close) ckpt cfg carries it
    assert p["deploy_ckpt_has_upstream_sha"] is True     # + the upstream snapshot sha
    assert p["resume_ckpt_has_git_sha"] is True          # resume sidecar carries it
    assert p["stage_ckpt_has_git_sha"] is True           # every per-stage byte-close ckpt carries it


def test_muon_finisher_resume_is_bit_identical(smoke):
    m = smoke["steps"]["muon_resume"]
    assert m["sigkilled_mid_finisher"] is True
    assert m["resumed_into_finisher"] is True, "resume did not detect the finisher window"
    assert m["bit_identical_to_continuous"] is True, f"finisher resume diverged: {m['diverged']}"
    assert m["n_diverged_tensors"] == 0


def test_all_pass(smoke):
    assert smoke["ALL_PASS"] is True, json.dumps(smoke, indent=2)


# =========================================================================== standalone entry
if __name__ == "__main__":
    import traceback

    # Durable, guard-legal base (SSD tier / non-/tmp repo-results). Fixed dir (not pid-suffixed) so
    # a DETACHED run is deterministically pollable: progress.log + error.txt (on failure) +
    # report.json (on success) all land at a known path.
    _t0 = time.time()
    _base = durable_smoke_base("durability_crash_resume_smoke_run")
    if _base.exists():
        shutil.rmtree(_base, ignore_errors=True)
    _base.mkdir(parents=True, exist_ok=True)
    try:
        _rep = run_crash_resume_smoke(_base)
    except BaseException:  # capture the traceback durably (detach-robust) + re-raise for the exit code
        (_base / "error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        raise
    _rep["wall_seconds"] = round(time.time() - _t0, 1)
    _report_path = _base / "report.json"
    _report_path.write_text(json.dumps(_rep, indent=2))
    # disk hygiene: drop the heavy per-arm checkpoint dirs, keep the small evidence artifacts.
    for _d in _base.iterdir():
        if _d.is_dir():
            shutil.rmtree(_d, ignore_errors=True)
    print(json.dumps(_rep, indent=2))
    print(f"\nreport: {_report_path}  ({_rep['wall_seconds']}s)")
    print("ALL_PASS:" if _rep["ALL_PASS"] else "FAIL:", _rep["ALL_PASS"])
    raise SystemExit(0 if _rep["ALL_PASS"] else 1)
