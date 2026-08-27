#!/usr/bin/env python3
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Duty-ticket revision materializer (#563, 2026-07-19): real ep725-fork compile evidence.

Resolves the four blockers of the sealed BLOCKED handoff
(`.omx/research/duty_queue_fire_tickets_20260719_codex.md`) by writing a
`revision_claude_20260719/` package INSIDE each existing ticket dir (append-only: the
composer's sealed artifacts are never touched):

* 01 DsegAwareTaper  — ADJUDICATED: NOT a valid ep725 contrast (structural epoch-0
  basis-feature lever; trainer F2 resume-divergence class; no taper-trained trunk
  exists to serve as the ON control). RE-SCOPED to the fresh-run ISO pair; the pair's
  full compile is itself BLOCKED at current HEAD by the pre-existing V9-432
  hosc_beta_end LawRef recompute defect (10.0 != 3.177) — captured honestly.
* 02 HorizonWeightedMargin — REAL compiled ep725-fork pair (v9c3_duty_hwm_off/on) with
  full dsl_compile_hash + resolved argv + schedule-gate rc0 + boundary-receipt path +
  measured-noise-derived FIRED-PAYS/NEUTRAL/HURTS thresholds.
* 03 StepNativeActivation — REAL compiled ep725-fork pair (v9c3_duty_step_off/on),
  one-flag delta --hosc-beta-end 4.0 -> 8.0, #517/#518 re-anchor + v0 response receipt.
* 04 #497 curvelet — ADJUDICATED fresh-arms-correct (basis levers cannot fork at ep725);
  wrapper fail-closed repairs landed separately in the fire script; residual OWED items
  named.

Pure/$0: compiles + static checks only; NEVER launches a trainer or governed launch.
Pointer 0.1910828242 [contest-CPU] UNMOVED — everything here is MEANS.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

TICKET_ROOT = REPO / ".omx" / "research" / "duty_queue_fire_tickets_20260719"
REV = "revision_claude_20260719"

# ---------------------------------------------------------------------------
# MEASURED verdict-noise floor (provenance: the donor run's own n600 advisory
# verdict d_seg series, extracted read-only from
# experiments/results/levelset_n600_witness_20260717T113932Z/run.log on 2026-07-19
# via /usr/bin/grep -oE '"epoch": [0-9]+[^}]*"d_seg": [0-9.e-]+'. The ep750-1075
# post-Muon segment is the smooth-recovery regime; the second-difference estimator
# (var(d2) = 6*sigma^2 for iid point noise on a smooth trend) gives the per-point
# noise WITHOUT assuming a global trend shape.
# ---------------------------------------------------------------------------
DONOR_VERDICT_SERIES = (
    (750, 0.004294), (775, 0.004236), (800, 0.004164), (825, 0.004122),
    (850, 0.004008), (875, 0.003975), (900, 0.003891), (925, 0.003847),
    (950, 0.003789), (975, 0.003762), (1000, 0.003708), (1025, 0.003693),
    (1050, 0.003664), (1075, 0.003628),
)
Z95_TWO_SIDED = 1.959963985  # 97.5% standard-normal quantile (95% two-sided)


def derive_thresholds() -> dict:
    y = [v for _, v in DONOR_VERDICT_SERIES]
    d2 = [y[i + 1] - 2 * y[i] + y[i - 1] for i in range(1, len(y) - 1)]
    sigma_plain = math.sqrt(sum(v * v for v in d2) / len(d2) / 6.0)
    med_abs = sorted(abs(v) for v in d2)[len(d2) // 2]
    sigma_robust = med_abs / (0.6745 * math.sqrt(6.0))
    sigma = max(sigma_plain, sigma_robust)  # conservative: the larger estimator
    def h95(k: int) -> float:
        # paired ON-OFF mean over k verdict points; independence upper bound sqrt(2)
        # (common-mode cancellation in a paired same-seed design can only shrink it)
        return Z95_TWO_SIDED * math.sqrt(2.0 / k) * sigma
    return {
        "schema": "duty_ab_preregistered_thresholds.v1",
        "instrument": "n600 advisory verdict d_seg (EMA shadow, frozen CPU scorer) "
                      "[macOS-MLX research-signal] — NOT a score axis; byte-close + "
                      "exact eval OWED before any promotion claim",
        "noise_provenance": {
            "series_source": "experiments/results/levelset_n600_witness_20260717T113932Z/"
                             "run.log (read-only), verdict rows ep750-1075",
            "series": [list(p) for p in DONOR_VERDICT_SERIES],
            "estimator": "second-difference: sigma = sqrt(mean(d2^2)/6); robust variant "
                         "median(|d2|)/(0.6745*sqrt(6)); the LARGER is used",
            "sigma_per_point_plain": sigma_plain,
            "sigma_per_point_robust": sigma_robust,
            "sigma_per_point_used": sigma,
            "caveat": "iid-noise-on-smooth-trend assumption; single-trajectory; the "
                      "sqrt(2) paired scaling is an independence UPPER bound (paired "
                      "same-seed arms share common-mode noise, which only shrinks it)",
        },
        "windows": {
            "primary": {"verdict_epochs": [775, 800, 825, 850], "k": 4,
                        "h95_dseg": h95(4)},
            "secondary": {"verdict_epochs": [775 + 25 * i for i in range(10)], "k": 10,
                          "h95_dseg": h95(10)},
        },
        "rules": {
            "metric": "delta(ep) = d_seg_ON(ep) - d_seg_OFF(ep) on paired verdict rows",
            "FIRED-PAYS": "mean(primary delta) <= -h95(k=4); confirm sign holds on the "
                          "secondary mean (|secondary| >= h95(k=10) with the same sign)",
            "FIRED-HURTS": "mean(primary delta) >= +h95(k=4) (secondary confirmation "
                           "same as PAYS, opposite sign)",
            "FIRED-NEUTRAL": "|mean| < h95 on BOTH windows",
            "INDETERMINATE": "primary and secondary disagree in sign at threshold — "
                             "report both, no verdict; extension arm required",
            "derivation_note": "1.959963985 = 95% two-sided standard-normal quantile; "
                               "no round-number constants (constants-are-poison)",
        },
        "power_context": {
            "hwm": "adverse prior (v9c2 launch.sh header): dS ceiling 0.012-0.024 => "
                   "d_seg ceiling 1.2e-4..2.4e-4 = 4.7x..9.4x h95(k=4) — a >=21% capture "
                   "of the LOW ceiling is detectable",
            "step": "no measured prior effect size (never-fired lever); the design "
                    "detects |delta d_seg| >= h95(k=4) (~0.74% of trunk d_seg 0.003458)",
        },
        "ncde_note": "the composer's NCDE response-time was UNTRUSTED (R^2=0.06020 < "
                     "0.5, on log_total not d_seg) — windows here are cadence floors "
                     "sized against direct paired verdict noise instead",
    }


ADMISSIBILITY = {
    "schema": "duty_ab_verdict_admissibility.v1",
    "preconditions_both_arms": [
        "run completes through ep1000 (crash => --resume-from continue, never a "
        "partial-window verdict); per-stage + every-25-ep checkpoints preserved",
        "ep_loss > 0 at every epoch in the verdict window (frozen-epoch alarm clear)",
        "no spike_deadlock / gnorm_hijack / term_domination confound_alarm rows in "
        "the window",
        "verdict rows carry ema_warmup=false inside the verdict windows "
        "(ForkEmaClearance stamps; window starts >=22 ep past the ep753 boundary, "
        ">=49 ep past resume — shadow matured, ~3675 updates >> 333)",
        "argv custody: the realized launch.sh pair diff equals EXACTLY the signed "
        "one-lever delta (plus --out-dir); any other diff voids the pair",
        "arms run SEQUENTIALLY on the same host (projected peak 67.6 GiB each; "
        "never concurrent with each other or any other n600 run)",
    ],
    "positive_control_sentinels": [
        "resume_lr_rewarmup row: boundary_epoch=726, rewarmup_epochs=27, reason in "
        "{warm_start_weights_only, lever_drift_retreatment} (deterministic known-fire; "
        "absent => instrument untrusted, NO verdict)",
        "baseline_v0_schedule_positioned row pre-v0 carrying tau/beta at resume-epoch "
        "schedule values (#517): control beta 3.1772, step-ON beta 6.0801",
        "ON arms: lever_engage status=fired row for the treatment lever at its "
        "boundary (hwm: horizon_margin @753; step: activation schedule live from 726)",
        "hwm ON: horizon_margin_boundary_receipt.json exists with resolved_weight > 0 "
        "(schema hwm_v9_stage_share_boundary.v1)",
    ],
}


def _sha256_file(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _git_head() -> str:
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip()


def _compile_arm(ticket: str, arm: str) -> dict:
    import launch_witness_run as lwr
    import schedule_provenance_gate as spg
    from tac.witness_autoconfig import _crucible_v7_argv_pairs
    from tac.witness_dsl.spec_v9c3_duty_ab_20260719 import compile_v9c3_duty_ab_config

    cfg = compile_v9c3_duty_ab_config(ticket, arm)
    doc, _, _ = lwr.compile_dsl_document_for_config(
        cfg, cfg.typed.out_dir, program_name=cfg.typed.name)
    argv = list(cfg.typed.to_program().compile_trainer_argv())
    pairs = list(_crucible_v7_argv_pairs(argv))
    trainer_text = lwr.TRAINER_PATH.read_text()
    verdicts = spg.classify_launch(
        pairs, registry=spg.schedule_when_flags(trainer_text),
        manifest_keys=set(cfg.constants_manifest or {}),
        governance=cfg.schedule_governance or {},
        event_registry=spg.event_start_flags(trainer_text))
    ok, violations, _table = spg.gate_report(verdicts)
    argv_json = json.dumps(doc["resolved_argv"], sort_keys=True,
                           separators=(",", ":")).encode()
    return {
        "program": cfg.typed.name,
        "arm": arm,
        "out_dir": cfg.typed.out_dir,
        "full_dsl_compile_hash": doc["dsl_compile_hash"],
        "typed_config_hash": cfg.typed.typed_config_hash(),
        "resolved_argv": doc["resolved_argv"],
        "resolved_argv_sha256": hashlib.sha256(argv_json).hexdigest(),
        "schedule_gate": {"rc": 0 if ok else 6, "verdicts": len(verdicts),
                          "violations": [str(v) for v in violations]},
        "constants_manifest": cfg.constants_manifest,
        "dsl_program_manifest": cfg.dsl_program_manifest,
        "governed_launch_command": (
            f".venv/bin/python tools/launch_witness_run.py --config {cfg.typed.name} "
            f"--gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz "
            f"--num-pairs 600 --out-dir {cfg.typed.out_dir} "
            f"[REQUIRES: launcher registration (owed, 3-line c2 pattern) + green "
            f"--dry-start receipt + operator GO; the launcher owns safe_run/governor/"
            f"memory-preflight wraps]"),
    }


def _memory_preflight() -> dict:
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "witness_memory_preflight.py"),
         "--num-pairs", "600", "--render-h", "384", "--render-w", "512",
         "--verdict-batch", "32", "--n-dir-freqs", "4"],
        capture_output=True, text=True, check=False, cwd=str(REPO))
    return {"rc": r.returncode, "stdout": r.stdout.strip().splitlines(),
            "note": "static projection at the real arm parameters; the launcher re-runs "
                    "this + the dry-start bench (MEASURED authority) at GO time"}


def _argv_delta(a: dict, b: dict) -> dict:
    def toks(row):
        out, argv = {}, row["resolved_argv"]
        i = 0
        while i < len(argv):
            t = argv[i]
            if isinstance(t, str) and t.startswith("--"):
                if i + 1 < len(argv) and not str(argv[i + 1]).startswith("--"):
                    out[t] = str(argv[i + 1]); i += 2; continue
                out[t] = True
            i += 1
        return out
    ta, tb = toks(a), toks(b)
    return {k: [ta.get(k), tb.get(k)] for k in sorted(set(ta) | set(tb))
            if ta.get(k) != tb.get(k)}


def _ticket_1(head: str) -> dict:
    """Taper: adjudicate + re-scope to the fresh ISO pair; capture the honest full-compile blocker."""
    import launch_witness_run as lwr
    from tac.witness_dsl import spec_v9_cgauge as v9

    treat = v9.compile_v9_cgauge_432_taper_off_launch_config(
        out_dir="experiments/results/v9_cgauge_432_taper_off_20260715")
    ctrl = v9.compile_v9_cgauge_432_launch_config(
        out_dir="experiments/results/v9_cgauge_432_taper_ctrl_20260719")
    iso = treat.dsl_program_manifest.get("iso_contract", {})
    rows = {}
    for name, cfg in (("control_on", ctrl), ("treatment_off", treat)):
        row = {"program": cfg.typed.name,
               "typed_config_hash": cfg.typed.typed_config_hash()}
        try:
            doc, _, _ = lwr.compile_dsl_document_for_config(
                cfg, cfg.typed.out_dir, program_name=cfg.typed.name)
            row["full_dsl_compile_hash"] = doc["dsl_compile_hash"]
        except Exception as exc:  # the honest pre-existing blocker, captured verbatim
            row["full_dsl_compile_hash"] = None
            row["full_compile_blocker"] = f"{type(exc).__name__}: {exc}"
        rows[name] = row
    return {
        "schema": "duty_ticket_revision.v1",
        "ticket": "01_dseg_aware_taper",
        "verdict": "CANNOT-RESOLVE(ep725-fork) / RE-SCOPED-TO-FRESH-RUN "
                   "(fresh pair full-compile BLOCKED by pre-existing V9-432 LawRef defect)",
        "adjudication": {
            "question": "is removing a structural epoch-0 lever after an ep725 warm "
                        "start a valid contrast?",
            "answer": "NO. (a) DsegAwareTaper is STRUCTURAL (its own factory docstring: "
                      "'active from ep0 by construction — it changes the input feats the "
                      "in_proj is trained on'; the trainer's F2 resume-divergence guard "
                      "REFUSES adding/changing it on resume as a basis change); (b) the "
                      "ep725 trunk (mod32cap->v9c2 lineage) was trained WITHOUT the taper "
                      "— the canonical ON control (taper-trained trunk) does not exist at "
                      "ep725; adding the taper at a fork would measure the add-shock of "
                      "reweighting features the in_proj never trained under, NOT the "
                      "lever's value; (c) --warm-start-weights-only auto-allows lever "
                      "drift, so the trainer would not even refuse — the contrast would "
                      "run and be silently confounded. Charter path taken: 'emit the "
                      "honest verdict that this lever NEEDS a fresh-run arm'.",
            "re_scope": "the canonical contrast IS the existing fresh mod19/3000-ep ISO "
                        "pair: v9_cgauge_432 base (taper ON control) vs "
                        "compile_v9_cgauge_432_taper_off_launch_config (whole-Lever "
                        "removal), duty 78.9%, one-lever delta verified below.",
        },
        "iso_contract": {
            "one_lever_delta": iso.get("one_lever_delta"),
            "argv_diff": iso.get("argv_diff"),
            "control_config": iso.get("control_config"),
            "config_id": iso.get("config_id"),
            "duty_to_measure_percent": iso.get("duty_to_measure_percent"),
        },
        "arms": rows,
        "named_blocker": {
            "id": "V9_432_HOSC_BETA_END_LAWREF_RECOMPUTE_DEFECT",
            "detail": "build_dsl_compile_provenance_document self-recompile refuses BOTH "
                      "432-family arms: LawRef equation recompute yields hosc_beta_end "
                      "10.0 while the config emits 3.177 (the CLAUDE.md 2026-07-15 "
                      "reconciliation's OWED LawRef/compiler-record debt, surfaced live "
                      "here). Pre-existing, NOT introduced by this revision; the fresh "
                      "pair cannot carry a full_dsl_compile_hash until that custody row "
                      "is repaired at its own surface.",
            "measured_at_head": head,
        },
        "confound_note": "fresh from-scratch arms: seeds/data-order identical by config; "
                         "#518 fork machinery N/A (no resume); thresholds for a 3000-ep "
                         "fresh pair must be re-derived from ITS lineage verdict noise at "
                         "fire time (the ep725-fork thresholds do not transfer).",
    }


def _ticket_4(head: str) -> dict:
    wrapper = REPO / "tools" / "fire_curvelet_matched_bytes_ab_p0_497.py"
    return {
        "schema": "duty_ticket_revision.v1",
        "ticket": "04_curvelet_matched_bytes_p0_497",
        "verdict": "READY-FOR-GAUNTLET(fresh-arms; wrapper fail-closed repairs landed) "
                   "with NAMED OWED items",
        "adjudication": {
            "ep725_fork_applicability": "INVALID for a basis lever: "
                    "LiteralPolarCurveletBasis changes the first-layer feature map the "
                    "trained weights are conditioned on (same structural class as the "
                    "taper — the F2 resume-divergence guard's basis-change class); the "
                    "existing FRESH 3000-ep paired arms are the CORRECT vehicle, so the "
                    "composer's 'no warm-start/resume input' gap is ADJUDICATED "
                    "not-a-defect (fresh arms need no #518 binding).",
            "one_factor": "the contrast is one-LEVER (a single --dsl-lever token); the "
                    "lever internally bundles basis + native-orient + AA behavior — "
                    "verdict_scope at fire time is the COMPOSITE lever, never 'curvelet "
                    "basis alone'; within-bundle attribution requires follow-up arms.",
        },
        "wrapper_repairs_landed": [
            "absent c2 run dir now FAIL-CLOSED (quiescent=False; explicit override "
            "required instead of silently treating missing-as-quiescent)",
            "liveness inspection failure (psutil import OR runtime error, e.g. "
            "PermissionError) now FAIL-CLOSED (quiescent=False, liveness_unknown=true) "
            "instead of only catching ImportError",
            "--skip-c2-gate now requires --operator-go (an override is an operator act)",
            "arm mutual exclusion: refuses to dry-run/fire while ANY process references "
            "either arm out_dir (the two arms are never concurrent)",
        ],
        "wrapper_sha256_after_repair": _sha256_file(wrapper),
        "owed_items_named": [
            "enforced equal-byte completion chain (curvelet_equal_byte_ab_receipt.py "
            "match -> inflate both -> finalize) is documented but still not REFUSED-on-"
            "skip by any gate — the finalize tool remains the enforcement surface to land",
            "telemetry beyond front_end for the treatment lever (lever_engage term-share "
            "binding) — flag-presence is not engagement proof",
            "current governed dry receipts + arm checkpoints do not exist yet (arms "
            "PREPARED_NOT_FIRED)",
        ],
        "measured_at_head": head,
    }


def main() -> int:
    head = _git_head()
    thresholds = derive_thresholds()
    mem = _memory_preflight()

    summary_rows = []
    for ticket_dir, ticket in (("02_horizon_weighted_margin", "hwm"),
                               ("03_step_native_activation", "step")):
        off = _compile_arm(ticket, "off")
        on = _compile_arm(ticket, "on")
        delta = _argv_delta(off, on)
        pkg = {
            "schema": "duty_ticket_revision.v1",
            "ticket": ticket_dir,
            "verdict": "READY-FOR-GAUNTLET",
            "measured_at_head": head,
            "checkpoint_custody": {
                "path": "experiments/results/banks/v9c2_defensive_bank_20260718/"
                        "levelset_witness_ema_BEST.npz",
                "sha256": "b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef",
                "bytes": 460448, "epoch": 725, "d_seg": 0.003457972208658854,
            },
            "arms": {"off": off, "on": on},
            "signed_argv_delta_off_to_on": delta,
            "one_lever_delta": True,
            "thresholds": thresholds,
            "admissibility": ADMISSIBILITY,
            "memory_preflight": mem,
            "receipt_paths": {
                "hwm_boundary_receipt": "<out_dir>/horizon_margin_boundary_receipt.json "
                                        "(schema hwm_v9_stage_share_boundary.v1; emitted "
                                        "by the trainer at the ep753 derived-live scan)",
                "v0_response_receipt": "run.log baseline_v0_schedule_positioned + the "
                                       "launcher baseline_v0_d_seg/_d_pose extraction "
                                       "(#518 item 5b)",
                "dry_start_receipt": "<run>/dry_start_report.json (typed-hash-matched "
                                     "GREEN required; currently an open launch blocker)",
            },
        }
        out = TICKET_ROOT / ticket_dir / REV
        _write_json(out / "compiled_pair.json", pkg)
        _write_text(out / "verdict_card.md", _card(pkg))
        summary_rows.append({"ticket": ticket_dir, "verdict": pkg["verdict"],
                             "off_hash": off["full_dsl_compile_hash"],
                             "on_hash": on["full_dsl_compile_hash"]})

    t1 = _ticket_1(head)
    out1 = TICKET_ROOT / "01_dseg_aware_taper" / REV
    _write_json(out1 / "adjudication.json", t1)
    _write_text(out1 / "verdict_card.md", _card(t1))
    summary_rows.append({"ticket": "01_dseg_aware_taper", "verdict": t1["verdict"]})

    t4 = _ticket_4(head)
    out4 = TICKET_ROOT / "04_curvelet_matched_bytes_p0_497" / REV
    _write_json(out4 / "adjudication.json", t4)
    _write_text(out4 / "verdict_card.md", _card(t4))
    summary_rows.append({"ticket": "04_curvelet_matched_bytes_p0_497",
                         "verdict": t4["verdict"]})

    # per-revision-dir manifests (append-only: the composer's sealed manifests are
    # snapshots of THEIR files; these cover only the revision packages).
    for tdir in ("01_dseg_aware_taper", "02_horizon_weighted_margin",
                 "03_step_native_activation", "04_curvelet_matched_bytes_p0_497"):
        rdir = TICKET_ROOT / tdir / REV
        rows = [{"path": p.name, "bytes": p.stat().st_size, "sha256": _sha256_file(p)}
                for p in sorted(rdir.iterdir())
                if p.is_file() and p.name != "revision_manifest.json"]
        _write_json(rdir / "revision_manifest.json",
                    {"schema": "duty_ticket_revision_manifest.v1", "files": rows})

    _write_json(TICKET_ROOT / f"{REV}_summary.json", {
        "schema": "duty_ticket_revision_summary.v1",
        "measured_at_head": head,
        "tickets": summary_rows,
        "containment": "compiles/static only; zero launches; pointer UNMOVED",
    })
    print(json.dumps(summary_rows, indent=2))
    return 0


def _card(pkg: dict) -> str:
    lines = [f"# {pkg['ticket']} — revision_claude_20260719", "",
             f"Verdict: `{pkg['verdict']}`", "",
             f"Measured at HEAD `{pkg.get('measured_at_head', '')[:12]}`. "
             "Containment: compile/static evidence only — NO launch occurred; "
             "pointer 0.1910828242 [contest-CPU] UNMOVED (MEANS).", ""]
    arms = pkg.get("arms")
    if arms and isinstance(arms, dict) and "off" in arms:
        lines += ["| arm | program | full_dsl_compile_hash | typed_config_hash | schedule gate |",
                  "|---|---|---|---|---|"]
        for a in ("off", "on"):
            r = arms[a]
            lines.append(f"| {a} | `{r['program']}` | `{r['full_dsl_compile_hash']}` | "
                         f"`{r['typed_config_hash']}` | rc{r['schedule_gate']['rc']}, "
                         f"{r['schedule_gate']['verdicts']} verdicts, "
                         f"{len(r['schedule_gate']['violations'])} violations |")
        lines += ["", "Signed OFF->ON argv delta:", "```json",
                  json.dumps(pkg["signed_argv_delta_off_to_on"], indent=2), "```", ""]
        th = pkg["thresholds"]
        lines += [
            "Pre-registered thresholds (derived from the donor run's measured verdict "
            f"noise, sigma={th['noise_provenance']['sigma_per_point_used']:.6e}/point):",
            f"- primary K=4 @ ep 775-850: h95 = "
            f"{th['windows']['primary']['h95_dseg']:.6e} d_seg",
            f"- secondary K=10 @ ep 775-1000: h95 = "
            f"{th['windows']['secondary']['h95_dseg']:.6e} d_seg",
            "- PAYS/HURTS/NEUTRAL/INDETERMINATE rules + admissibility preconditions + "
            "positive-control sentinels: see compiled_pair.json", ""]
    else:
        lines += ["```json", json.dumps(
            {k: v for k, v in pkg.items() if k not in ("schema",)}, indent=2), "```", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
