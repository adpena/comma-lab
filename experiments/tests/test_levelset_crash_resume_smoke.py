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

The seven proof assertions (mapped to the #205 requirements):
  (a) trains through >=1 curriculum stage boundary + intra-stage checkpoints          [req 1,2,3,7]
  (b) is KILLED mid-stage with SIGKILL (abrupt, no clean shutdown)                     [req 5]
  (c) ``--resume-from`` the surviving rolling checkpoint continues the run             [req 1]
  (d) resumed FINAL live+EMA weights == continuous FINAL live+EMA weights (bit-exact)  [req 10]
      AND the optimizer state restored (restored_opt), the precondition for bit-exact  [req 6]
  (e) per-stage checkpoints exist under DISTINCT stage-encoded filenames (no overwrite) [req 2]
  (f) the deploy EMA npz is the EMA SHADOW (!= live weights) AND byte-close-loadable    [req 4,6]
  (g) a resume with a DELIBERATELY-DIVERGED lever (--film-per-layer) fail-closes        [req 8]

MLX-CPU is bit-identical across independent processes (verified: two identical runs hash-match),
so (d) asserts EXACT equality, not a tolerance band.

Runtime ~40-60s (tiny n=2, 6-epoch, 96x128-render CPU runs). Standalone:
    TAC_MLX_CUSTOM_GROUPED_BACKWARD=0 .venv/bin/python experiments/tests/test_levelset_crash_resume_smoke.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
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

# Tiny, fast, deterministic config that still exercises BOTH curriculum stage boundaries
# (ce->tau at ep3, tau->l7 at ep5) plus per-epoch intra-stage checkpoints.
_EPOCHS = 6
_TAU_START = 3
_L7_START = 5


def _env() -> dict[str, str]:
    e = dict(os.environ)
    # The custom Metal grouped-backward kernel is GPU-only; the deterministic CPU smoke disables it.
    e["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "0"
    e["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO / "src"), str(_REPO / "upstream"), str(_REPO / "experiments"), str(_REPO)]
        + ([e["PYTHONPATH"]] if e.get("PYTHONPATH") else [])
    )
    return e


def _base_argv(out_dir: Path, *extra: str, epochs: int = _EPOCHS,
               muon_start: int | None = None) -> list[str]:
    argv = [
        sys.executable, str(_TRAINER),
        "--out-dir", str(out_dir),
        "--num-pairs", "2", "--gt-cache", str(_GT_CACHE),
        "--epochs", str(epochs), "--curriculum",
        "--tau-softplus-start-epoch", str(_TAU_START), "--l7-start-epoch", str(_L7_START),
        "--ckpt-every", "1", "--eval-every", "2", "--mlx-device", "cpu", "--seed", "0",
        "--render-h", "96", "--render-w", "128", "--hidden-dim", "32", "--n-hidden", "2",
        "--no-self-orient",
    ]
    if muon_start is not None:
        argv += ["--muon-start-epoch", str(muon_start)]
    return argv + list(extra)


def _run_to_completion(out_dir: Path, *extra: str, timeout: float = 300.0,
                       epochs: int = _EPOCHS, muon_start: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        _base_argv(out_dir, *extra, epochs=epochs, muon_start=muon_start), env=_env(), cwd=str(_REPO),
        capture_output=True, text=True, timeout=timeout,
    )


def _run_and_sigkill(out_dir: Path, kill_substr: str, *extra: str, hard_timeout: float = 200.0,
                     epochs: int = _EPOCHS, muon_start: int | None = None):
    """Launch the trainer, stream stdout, SIGKILL (abrupt, no cleanup) the FIRST time ``kill_substr``
    appears. Returns (killed: bool, seen_lines: list[str]). This simulates a real crash/OOM/operator-cut
    mid-stage: the surviving on-disk state is whatever the last ATOMIC checkpoint wrote."""
    proc = subprocess.Popen(
        _base_argv(out_dir, *extra, epochs=epochs, muon_start=muon_start), env=_env(), cwd=str(_REPO),
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


def _hash_state(npz_path: Path, prefixes: tuple[str, ...]) -> dict[str, str]:
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
def run_crash_resume_smoke(base: Path, *, include_muon: bool = True) -> dict:
    """Run the full crash->resume->parity proof under ``base`` and return a machine-readable report.

    Shared by the pytest fixture AND the ``__main__`` standalone entry so the heavy subprocess work
    happens exactly once. ``include_muon=False`` runs only the base arm (steps 1-8) so each arm fits
    a <2-min budget (the muon arm is then run separately via ``run_muon_finisher_arm``)."""
    import numpy as np

    assert _GT_CACHE.exists(), f"GT cache missing: {_GT_CACHE} (need gt_n6.npz for the n=2 smoke)"
    cont_dir = base / "continuous"
    crash_dir = base / "crash"
    resume_dir = base / "resumed"
    film_dir = base / "film_per_layer"
    for d in (cont_dir, crash_dir, resume_dir, film_dir):
        d.mkdir(parents=True, exist_ok=True)

    report: dict = {"utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "steps": {}}
    _progress = base / "progress.log"

    def _prog(msg: str) -> None:
        line = f"{datetime.now(UTC).strftime('%H:%M:%S')} [smoke] {msg}\n"
        print(line, end="", file=sys.stderr, flush=True)
        with open(_progress, "a", encoding="utf-8") as fh:  # durable, explicit flush (detach-robust)
            fh.write(line)

    # --- (1) CONTINUOUS reference run (uninterrupted, epochs 1..6) ---
    _prog("step 1/9: continuous reference run")
    cont = _run_to_completion(cont_dir)
    assert cont.returncode == 0, f"continuous run failed rc={cont.returncode}\n{cont.stdout[-2000:]}\n{cont.stderr[-2000:]}"
    cont_final = cont_dir / "levelset_resume_state.npz"
    cont_hash = _hash_state(cont_final, ("liveP__", "emaP__"))
    report["steps"]["continuous"] = {
        "rc": cont.returncode, "final_epoch": _resume_epoch(cont_final), "n_tensors": len(cont_hash),
    }

    _prog("step 2/9: crash run + SIGKILL mid-stage")
    # --- (2) CRASH run: SIGKILL right after the epoch-3 intra-stage checkpoint (mid tau stage) ---
    killed, seen = _run_and_sigkill(crash_dir, '"kind": "intra_stage", "epoch": 3')
    crash_resume = crash_dir / "levelset_resume_state.npz"
    assert crash_resume.exists(), "crash left NO rolling resume_state.npz (nothing to resume from!)"
    k_epoch = _resume_epoch(crash_resume)
    # per-stage preserved ckpt for the CE stage MUST survive the crash (it was written at ep2).
    ce_pres = crash_dir / "levelset_ckpt_stageCE_ep2.npz"
    report["steps"]["crash"] = {
        "sigkilled": bool(killed), "surviving_resume_epoch": k_epoch,
        "stageCE_ckpt_survived": ce_pres.exists(),
        "note": "SIGKILL is abrupt (no clean shutdown); surviving state == last ATOMIC checkpoint",
    }
    assert killed, "crash run exited before the epoch-3 checkpoint (kill trigger never fired)"
    assert 2 <= k_epoch < _EPOCHS, f"surviving resume epoch {k_epoch} not mid-run (expected 2..{_EPOCHS-1})"
    assert ce_pres.exists(), "per-stage CE checkpoint did NOT survive the crash"

    _prog("step 3/9: resume from crash checkpoint")
    # --- (3) RESUME from the surviving rolling checkpoint into a FRESH out-dir ---
    res = _run_to_completion(resume_dir, "--resume-from", str(crash_dir))
    assert res.returncode == 0, f"resume run failed rc={res.returncode}\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}"
    # observability: confirm the optimizer state was restored (the bit-exact precondition).
    restored_opt = False
    resumed_from_epoch = None
    for ln in res.stdout.splitlines():
        if '"stage": "resume"' in ln and "restored_opt" in ln:
            try:
                obj = json.loads(ln)
                restored_opt = bool(obj.get("restored_opt"))
                resumed_from_epoch = int(obj.get("resumed_epoch"))
            except (ValueError, TypeError):
                pass
    resume_final = resume_dir / "levelset_resume_state.npz"
    resume_hash = _hash_state(resume_final, ("liveP__", "emaP__"))

    # --- (4) BIT-IDENTICAL parity: resumed FINAL == continuous FINAL ---
    assert set(resume_hash) == set(cont_hash), "resumed/continuous tensor KEY sets differ"
    diffs = [k for k in cont_hash if cont_hash[k] != resume_hash[k]]
    report["steps"]["resume"] = {
        "rc": res.returncode, "resumed_from_epoch": resumed_from_epoch, "restored_opt": restored_opt,
        "final_epoch": _resume_epoch(resume_final), "n_tensors": len(resume_hash),
        "bit_identical_to_continuous": (diffs == []), "n_diverged_tensors": len(diffs),
        "diverged": diffs[:8],
    }

    _prog("step 5/9: per-stage distinct filenames")
    # --- (5) per-stage DISTINCT filenames (prior stage NOT overwritten) ---
    stage_files = sorted(p.name for p in cont_dir.glob("levelset_ckpt_stage*_ep*.npz"))
    resume_stage_files = sorted(p.name for p in cont_dir.glob("levelset_resume_stage*_ep*.npz"))
    report["steps"]["per_stage"] = {
        "deploy_stage_ckpts": stage_files, "resume_stage_ckpts": resume_stage_files,
        "distinct": len(stage_files) == len(set(stage_files)) and len(stage_files) >= 3,
    }

    _prog("step 6/9: deploy npz is EMA shadow + byte-close-loadable")
    # --- (6) deploy EMA npz is the SHADOW (!= live) AND byte-close-loadable ---
    ema_deploy = cont_dir / "levelset_witness_ema_mlx.npz"
    zc = np.load(cont_final, allow_pickle=False)
    zd = np.load(ema_deploy, allow_pickle=False)
    k = "in_proj.weight"

    def _sha(a):
        return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()).hexdigest()

    deploy_is_shadow = _sha(zd[k]) == _sha(zc["emaP__" + k])       # deploy == EMA shadow
    deploy_neq_live = _sha(zd[k]) != _sha(zc["liveP__" + k])       # deploy != live weights
    from tools.levelset_byte_close_and_eval import _load_levelset_ckpt

    params, cfg = _load_levelset_ckpt(cont_dir)
    byte_close_ok = all(r in params for r in ("code", "in_proj.weight", "out_sdf.weight", "out_tex.weight", "palette"))
    best_json = cont_dir / "levelset_best.json"
    report["steps"]["ema_shadow"] = {
        "deploy_is_ema_shadow": deploy_is_shadow, "deploy_differs_from_live": deploy_neq_live,
        "byte_close_loadable": byte_close_ok, "best_json_exists": best_json.exists(),
    }

    _prog("step 7/9: fail-closed on diverged lever")
    # --- (7) fail-closed on a DELIBERATELY-DIVERGED lever (--film-per-layer) ---
    film = _run_to_completion(film_dir, "--film-per-layer", "--epochs", "2",
                              "--no-curriculum", "--tau-softplus-start-epoch", "1",
                              "--l7-start-epoch", "2")
    film_ok = film.returncode == 0 and (film_dir / "levelset_resume_state.npz").exists()
    # resume the film-per-layer ckpt WITHOUT the flag => the ckpt carries film_pl.* the rebuilt model
    # lacks => the arch-drift guard MUST fail-closed (silent-param-drop = corrupted resume).
    bad = _run_to_completion(base / "film_resume_bad", "--resume-from", str(film_dir),
                             "--epochs", "3", "--no-curriculum",
                             "--tau-softplus-start-epoch", "1", "--l7-start-epoch", "2")
    guard_txt = (bad.stdout + bad.stderr)
    report["steps"]["fail_closed"] = {
        "film_train_ok": film_ok,
        "diverged_resume_rc": bad.returncode,
        "fail_closed": bad.returncode != 0 and ("SILENTLY DROP" in guard_txt or "no slot for" in guard_txt),
    }

    _prog("step 8/9: provenance completeness")
    # --- (8) PROVENANCE completeness (#205 + coordinator): git sha in result.json AND every ckpt cfg ---
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
            cont_dir / "levelset_ckpt_stageCE_ep2.npz", allow_pickle=False).files,
    }

    if include_muon:
        _prog("step 9/9: muon-finisher crash-resume")
        report["steps"]["muon_resume"] = run_muon_finisher_arm(base)

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
        and (not include_muon or (
            report["steps"]["muon_resume"]["resumed_into_finisher"]
            and report["steps"]["muon_resume"]["bit_identical_to_continuous"]))
    )
    return report


def run_muon_finisher_arm(base: Path) -> dict:
    """(step 9) MUON-FINISHER crash-resume: bit-identical continuation INSIDE the finisher (the #205
    launch uses muon@726). Crash mid-finisher, resume -> the Muon MultiOptimizer momentum must
    RESTORE (not re-warm), else the finisher tail diverges. Self-contained (own dirs) so it can run
    as its own <2-min unit. Returns the ``muon_resume`` step dict."""
    m_cont = base / "muon_continuous"
    m_crash = base / "muon_crash"
    m_resume = base / "muon_resumed"
    for d in (m_cont, m_crash, m_resume):
        d.mkdir(parents=True, exist_ok=True)
    _M_EPOCHS, _M_START = 7, 5  # muon@5 == first l7 epoch (PR95 placement, no warn); finisher ep5..7
    mc = _run_to_completion(m_cont, epochs=_M_EPOCHS, muon_start=_M_START)
    assert mc.returncode == 0, f"muon continuous failed\n{mc.stdout[-1500:]}\n{mc.stderr[-1500:]}"
    m_cont_hash = _hash_state(m_cont / "levelset_resume_state.npz", ("liveP__", "emaP__"))
    # crash AFTER the epoch-6 checkpoint => surviving state is mid-finisher (ep6, momentum accumulated
    # since the ep5 switch); resume must run ep7 with that RESTORED momentum.
    m_killed, _ = _run_and_sigkill(m_crash, '"kind": "intra_stage", "epoch": 6',
                                   epochs=_M_EPOCHS, muon_start=_M_START)
    m_crash_resume = m_crash / "levelset_resume_state.npz"
    assert m_crash_resume.exists(), "muon crash left no rolling resume_state.npz"
    assert m_killed, "muon crash exited before the epoch-6 finisher checkpoint"
    m_k_epoch = _resume_epoch(m_crash_resume)
    mr = _run_to_completion(m_resume, "--resume-from", str(m_crash), epochs=_M_EPOCHS, muon_start=_M_START)
    assert mr.returncode == 0, f"muon resume failed\n{mr.stdout[-1500:]}\n{mr.stderr[-1500:]}"
    resumed_into_finisher = False
    for ln in mr.stdout.splitlines():
        if '"stage": "resume"' in ln and "resumed_into_finisher" in ln:
            try:
                resumed_into_finisher = bool(json.loads(ln).get("resumed_into_finisher"))
            except ValueError:
                pass
    m_resume_hash = _hash_state(m_resume / "levelset_resume_state.npz", ("liveP__", "emaP__"))
    m_diffs = [k for k in m_cont_hash if m_cont_hash[k] != m_resume_hash.get(k)]
    return {
        "sigkilled_mid_finisher": bool(m_killed), "surviving_resume_epoch": m_k_epoch,
        "resumed_into_finisher": resumed_into_finisher,
        "bit_identical_to_continuous": (m_diffs == [] and set(m_cont_hash) == set(m_resume_hash)),
        "n_diverged_tensors": len(m_diffs), "diverged": m_diffs[:8],
    }


# =========================================================================== pytest surface
@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    # This runs 8 real MLX trainer subprocesses (~3-4 min) so it is OPT-IN: set
    # RUN_CRASH_RESUME_SMOKE=1 to execute. The default suite skips it (avoids a multi-minute hang);
    # the standalone ``__main__`` path (which writes report.json) is the primary evidence surface.
    if os.environ.get("RUN_CRASH_RESUME_SMOKE") != "1":
        pytest.skip("set RUN_CRASH_RESUME_SMOKE=1 to run the ~4-min end-to-end crash-resume proof")
    os.environ.setdefault("TAC_MLX_CUSTOM_GROUPED_BACKWARD", "0")  # the CPU smoke: GPU kernel off
    # Portable base (Linux CI /tmp would trip the trainer's _refuse_tmp): use a repo-results dir.
    base = _REPO / "experiments" / "results" / f"durability_crash_resume_smoke_{os.getpid()}"
    if base.exists():
        shutil.rmtree(base)
    try:
        rep = run_crash_resume_smoke(base)
        yield rep
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
    # the three PR95 curriculum stages each preserved under a DISTINCT stage+epoch filename.
    for name in ("levelset_ckpt_stageCE_ep2.npz", "levelset_ckpt_stageTau_ep4.npz",
                 "levelset_ckpt_stageL7_ep6.npz"):
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

    # Fixed dir (not pid-suffixed) so a DETACHED run is deterministically pollable: progress.log +
    # error.txt (on failure) + report.json (on success) all land at a known path.
    _base = _REPO / "experiments" / "results" / "durability_crash_resume_smoke_run"
    if _base.exists():
        shutil.rmtree(_base)
    _base.mkdir(parents=True, exist_ok=True)
    try:
        _rep = run_crash_resume_smoke(_base)
    except BaseException:  # capture the traceback durably (detach-robust) + re-raise for the exit code
        (_base / "error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        raise
    _report_path = _base / "report.json"
    _report_path.write_text(json.dumps(_rep, indent=2))
    print(json.dumps(_rep, indent=2))
    print(f"\nreport: {_report_path}")
    print("ALL_PASS:" if _rep["ALL_PASS"] else "FAIL:", _rep["ALL_PASS"])
    raise SystemExit(0 if _rep["ALL_PASS"] else 1)
