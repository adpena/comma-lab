#!/usr/bin/env python
"""$0 lambda_pre HVP probe AT THE 0.025 GOLD BASIN (DE #3 binding pre-condition).

The DE #3 clean-warm-start gate (witness_config_differential_equations_derivation_20260705.md
§DE#3, lines 207-209) asks ONE measured question before any warm-start from the preserved
`v5_dseg0026_preserved_20260705` gold is worth it:

    is the Adam-preconditioned sharpness lambda_pre at the 0.025 GOLD weights HIGHER (basin
    SHARPER => warm-start RISKIER) or LOWER (FLATTER => SAFER) than at ep100 (=3.66e6, measured
    by the eik-stab build's sibling probe)?

METHOD (reuses the trainer's OWN `--lambda-pre-probe-iters` mode + the eik-stab stepping-probe
helpers verbatim — nothing about the HVP / preconditioner / power-iteration is reimplemented):
  * SNAPSHOT-DOCTORING is the ONLY new step. We copy the ep100 resume snapshot and overwrite its
    SHADOW (emaP__) + LIVE (liveP__) model-param arrays with the GOLD ema_BEST weights, keeping
    the ep100 optimizer moments (optP__), step, and cfg IDENTICAL. Resume seeds live<-shadow, so
    the model the probe evaluates H at is the GOLD 0.025 basin, while the Adam preconditioner P is
    BIT-IDENTICAL to the one the 3.66e6 measurement used. The ONLY variable across the A/B is the
    WEIGHTS => the lambda_pre delta is PURE basin geometry (single-variable control).
  * A second, moment-INDEPENDENT cross-check: run BOTH states with moments dropped (drop_opt=True
    => eps-floor preconditioner, uniform 1/sqrt(1e-8)). Because the preconditioner is then a
    constant diagonal, the gold/ep100 ratio of the reported lambda_pre EQUALS the ratio of the RAW
    Hessian sharpness lambda_max(H) — a preconditioner-free geometry read that must AGREE in
    direction with the restored-moment A/B for the verdict to be robust.

Four probes:
  E_restored : ep100 weights + ep100 moments   (reproduces 3.66e6 under this exact code path =>
               env / doctoring sanity)
  G_restored : GOLD  weights + ep100 moments    (the money A/B vs E_restored)
  E_fresh    : ep100 weights + zero moments      (raw-H proxy, ep100)
  G_fresh    : GOLD  weights + zero moments       (raw-H proxy, gold)

AXIS / NO-FAKE: every number is `[n24 advisory -- mechanism probe, NOT n600 evidence]`; the
verdict is a MEASUREMENT (or an honest UNMEASURABLE), never a hope. NOTHING trains; the trainer
EXITS before any step. Pointer contest-CPU 0.19110 UNMOVED (means/apparatus).

Foreground-only; no daemons. Usage:
    .venv/bin/python experiments/probe_lambda_pre_at_gold.py \
        [--iters 12] [--mlx-device gpu|cpu] [--only E_restored,G_restored]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EP100_SNAP = REPO / "experiments/results/bd_calib_20260705/snap/resume_state_ep100.npz"
GOLD_BEST = REPO / "experiments/results/v5_dseg0026_preserved_20260705/levelset_witness_ema_BEST.npz"
OUT_DIR = REPO / "experiments/results/de3_lambda_pre_at_gold_20260705"
EP100_LAMBDA_PRE = 3.663e6  # the sibling eik-stab measurement (ep100, restored moments, n24, GPU)


def _load_stepping_probe():
    p = REPO / "experiments" / "probe_resume_stepping_instability.py"
    spec = importlib.util.spec_from_file_location("_stepping_probe_for_gold_lambda", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stepping_probe_for_gold_lambda"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_gold_doctored_snapshot(dst: Path) -> dict:
    """Copy ep100 snapshot, overwrite emaP__/liveP__ model-param arrays with GOLD ema_BEST.

    Keeps optP__ moments, step, __resume_epoch, __cfg_* IDENTICAL to ep100. Only the 20
    model-param arrays (those with an emaP__ slot) are swapped. Fail-closed on any shape mismatch
    or missing slot (a silent partial-swap would be a NO-FAKE violation)."""
    snap = np.load(EP100_SNAP, allow_pickle=False)
    best = np.load(GOLD_BEST, allow_pickle=False)
    gold_params = [k for k in best.files if not k.startswith(("__", "__bank", "__render"))]
    # the model-param keys are exactly those with an emaP__ slot in the snapshot.
    gold_params = [k for k in gold_params if f"emaP__{k}" in snap.files]
    out: dict[str, np.ndarray] = {}
    swapped = 0
    for k in snap.files:
        out[k] = snap[k]
    for k in gold_params:
        gv = best[k]
        for pref in ("emaP__", "liveP__"):
            slot = pref + k
            if slot not in snap.files:
                continue
            if snap[slot].shape != gv.shape:
                raise SystemExit(
                    f"FATAL shape mismatch {slot}: snap {snap[slot].shape} vs gold {gv.shape}")
            out[slot] = gv.astype(snap[slot].dtype, copy=True)
            swapped += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(dst)
    stats = {"n_gold_params": len(gold_params), "slots_swapped": swapped,
             "resume_epoch": int(np.asarray(out["__resume_epoch"])) if "__resume_epoch" in out else None,
             "has_opt": int(np.asarray(out.get("__resume_has_opt", 0)))}
    return stats


def _run_probe(sp, tag: str, full_snapshot: Path, *, drop_opt: bool, iters: int,
               fd_eps: float, device: str, accum: int, timeout_s: int) -> dict:
    """Slice `full_snapshot` to n24 (drop_opt selects restored vs fresh moments) and run the
    trainer's lambda_pre probe. Returns the parsed final row."""
    work = OUT_DIR / tag
    work.mkdir(parents=True, exist_ok=True)
    snap24 = work / "resume_state_n24.npz"
    slice_stats = sp.slice_snapshot(full_snapshot, snap24, 24, drop_opt=drop_opt)
    cfg = sp.ARMS["baseline_v3"]
    argv_t = sp._base_argv(
        work / "trainer_out", snap24, epochs=103, accum=accum,
        seed_anneal_epochs=cfg["seed_anneal"], bd_weight=cfg["bd"], lr=cfg["lr"],
        lr_end=cfg["lr_end"], tau_start=cfg["tau_start"], band_start=cfg["band_start"],
        persist_warmup=cfg["persist_warmup"],
        extra=["--lambda-pre-probe-iters", str(iters),
               "--lambda-pre-probe-fd-eps", f"{fd_eps:g}"])
    di = argv_t.index("--mlx-device")
    argv_t[di + 1] = device
    (work / "probe.argv.json").write_text(json.dumps(argv_t, indent=1))
    log_path = work / "probe.log"
    env = dict(os.environ)
    env.setdefault("TAC_MLX_CUSTOM_GROUPED_BACKWARD", "1")
    print(json.dumps({"stage": "probe_start", "tag": tag, "drop_opt": drop_opt,
                      "slice": slice_stats, "device": device, "iters": iters}), flush=True)
    with log_path.open("w") as lf:
        try:
            rc = subprocess.run(argv_t, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=timeout_s, env=env, cwd=str(REPO)).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    final, start = None, None
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("stage") == "lambda_pre":
            final = row
        elif row.get("stage") == "lambda_pre_probe_start":
            start = row
    return {"tag": tag, "rc": rc, "drop_opt": drop_opt, "slice": slice_stats,
            "start": start, "final": final}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--fd-eps", type=float, default=1e-3)
    ap.add_argument("--mlx-device", type=str, default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--accum-pairs", type=int, default=8)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--only", type=str, default="E_restored,G_restored,E_fresh,G_fresh",
                    help="comma list of probes to run")
    args = ap.parse_args(argv)

    if not EP100_SNAP.exists():
        print(f"FATAL: ep100 snapshot missing at {EP100_SNAP}", file=sys.stderr)
        return 2
    if not GOLD_BEST.exists():
        print(f"FATAL: gold ema_BEST missing at {GOLD_BEST}", file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp = _load_stepping_probe()

    gold_full = OUT_DIR / "gold_doctored_full_snapshot.npz"
    doc_stats = _build_gold_doctored_snapshot(gold_full)
    print(json.dumps({"stage": "gold_doctored", "stats": doc_stats}), flush=True)

    want = set(s.strip() for s in args.only.split(",") if s.strip())
    plan = [
        ("E_restored", EP100_SNAP, False),
        ("G_restored", gold_full, False),
        ("E_fresh", EP100_SNAP, True),
        ("G_fresh", gold_full, True),
    ]
    results: dict[str, dict] = {}
    for tag, snap_src, drop_opt in plan:
        if tag not in want:
            continue
        results[tag] = _run_probe(sp, tag, snap_src, drop_opt=drop_opt, iters=args.iters,
                                  fd_eps=args.fd_eps, device=args.mlx_device,
                                  accum=args.accum_pairs, timeout_s=args.timeout_s)

    _CONV_REL = 0.30  # fwd-vs-central rel must be below this for the power iteration to be TRUSTED.

    def _final(tag: str):
        return (results.get(tag) or {}).get("final") or {}

    def _converged(f: dict) -> bool:
        rel = f.get("fwd_vs_central_rel")
        return bool(f) and rel is not None and abs(float(rel)) < _CONV_REL

    def _verdict_pair(e_tag: str, g_tag: str, label: str, unit_note: str) -> dict:
        ef, gf = _final(e_tag), _final(g_tag)
        e, g = ef.get("lambda_pre"), gf.get("lambda_pre")
        e_ok, g_ok = _converged(ef), _converged(gf)
        out = {"ep100_lambda_pre": e, "ep100_converged": e_ok, "ep100_rel": ef.get("fwd_vs_central_rel"),
               "gold_lambda_pre": g, "gold_converged": g_ok, "gold_rel": gf.get("fwd_vs_central_rel"),
               "note": unit_note}
        if not (e and g):
            out["verdict"] = "INCOMPLETE (a probe did not produce a lambda_pre row)"
            return out
        if not g_ok:
            # NO-FAKE: a non-converged power iteration is NOT a measurement. Do NOT emit sharper/flatter.
            out["verdict"] = ("UNMEASURABLE — the FD-HVP power iteration did NOT converge at the GOLD "
                              f"basin (rel={gf.get('fwd_vs_central_rel')}); ep100 converged "
                              f"(rel={ef.get('fwd_vs_central_rel')}). The gold/ep100 comparison is "
                              "INCONCLUSIVE; the 'flatter=>safer' DE#3 precondition is UNCONFIRMED.")
            return out
        ratio = abs(g) / abs(e)  # compare MAGNITUDES (Hessians may be indefinite)
        out["gold_over_ep100_abs"] = ratio
        out["verdict"] = ("gold SHARPER (warm-start RISKIER)" if ratio > 1.15
                          else "gold FLATTER (warm-start SAFER)" if ratio < 0.87
                          else "COMPARABLE (within +-15%)")
        return out

    verdict = {
        "restored_moments": _verdict_pair(
            "E_restored", "G_restored", "restored-moments",
            "SAME ep100 preconditioner for both states; the ONLY variable is the WEIGHTS => isolates "
            "basin geometry. ep100 must reproduce 3.66e6 for the A/B to be trusted."),
        "fresh_moments_raw_H_proxy": _verdict_pair(
            "E_fresh", "G_fresh", "fresh-moments",
            "eps-floor preconditioner is uniform => the gold/ep100 lambda_pre RATIO == ratio of raw "
            "lambda_max(H); the ABSOLUTE eps-floor lambda_pre is EoS-meaningless (trainer WARN)."),
    }
    report = {
        "axis": "[n24 advisory -- mechanism probe, NOT n600 evidence]",
        "de": "DE#3 clean-warm-start binding pre-condition (lambda_pre at 0.025 gold vs ep100)",
        "ep100_lambda_pre_reference_3663e6": EP100_LAMBDA_PRE,
        "gold_doctoring": doc_stats,
        "results": {k: {"rc": v["rc"], "drop_opt": v["drop_opt"], "final": v["final"],
                        "start": v.get("start")} for k, v in results.items()},
        "verdict": verdict,
        "pointer": "0.19110 UNMOVED",
    }
    rp = OUT_DIR / "de3_lambda_pre_at_gold_report.json"
    rp.write_text(json.dumps(report, indent=1))
    print(json.dumps({"stage": "report_written", "path": str(rp)}), flush=True)
    print("\n==== DE#3 lambda_pre-at-gold VERDICT ====")
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
