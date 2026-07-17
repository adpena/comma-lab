# c2_surgical_warm PRE-FIRE FRESH-EYES REVIEW — 2026-07-17

**Reviewer:** independent fresh-eyes subagent (p0_515 fire-condition gate).
**Scope:** the four surfaces named in the review charter; every claim EXECUTED, not read.
**Pointer:** contest-CPU 0.19108 UNMOVED — this review gates the run that attempts to move it.

## OVERALL VERDICT: **PROCEED-WITH-NOTES**

No launch-blocking finding. Four findings (2 cosmetic, 2 operational notes), all quantified below;
none invalidates the green receipt, the projection, the confound fixes, or the launch path.

---

## Surface 1 — the green decomposed bench receipt: **PASS (load-bearing)**

Receipt: `experiments/results/levelset_n600_witness_20260717T031229Z/dry_start_report.json`.

Executed verifications:
- **Hash match (re-derived, not trusted):** live `compile_c2_surgical_warm_launch_config()` →
  `dsl_program_manifest.typed_config_hash = 2d486e3bff935949c4018a9b998b621cb439e8000640772aa8abd23dd21d5a8d`
  — EXACT match to the receipt. `launch_blockers: []`; `composed_bench_receipt.status: MEASURED_GREEN`
  pointing at this receipt; `warm_start_ckpt_custody: CLEAR` (458,622 bytes). The hash-tightened
  blocker clear (spec_c2_surgical_20260716.py:453-475) refuses receipts without a recorded hash
  (stale-receipt laundering closed) — verified in source.
- **Decomposition re-computed from raw JSONLs:** fresh
  `dry_start/witness_component_wallclock.jsonl` (n=16): median `epoch_total_s` = **77.55**,
  median `span_epoch_tail_s` = **0.01**. Observer-ON run
  (`…20260716T211713Z/dry_start/…`, n=3): tail median **1539.76**.
  ckpt_epoch_extra = 1539.76 − 0.01 = **1539.75** ✓; amortized = 77.55 + 1539.75/25 = **139.14** ✓;
  projected 750 × 139.14 / 3600 = **28.99 h** ✓. All arithmetic reproduces exactly.
- **F4 single-knob attack:** full flag-diff of the two launch.sh files — the ONLY structural delta
  between observer-ON (211713Z) and fresh bench (031229Z) is `--no-mod-dim-ablation` (+ out-dir).
  `--skip-boot-baseline-verdict` is bench-pass-argv-only and verified BOOT-ONLY in the trainer
  (train_levelset…mlx.py:10180-10196 — gates only the pre-loop v0 verdict; consumer-safety audit in
  the comment; `test_real_launch_argv_never_carries_skip_flag` green). The tail A/B is genuinely
  single-knob.
- **Amortization model structurally verified:** `_mdd_ablation_checkpoint` fires ONLY on
  `is_transition or do_periodic` checkpoint epochs (trainer:13379-13381); the real run pays it every
  25 epochs → the amortized law is the right shape, and the observer is governor-gated (cost can only
  go DOWN under pressure).
- **F5 exclusion empirically bounded:** the bench window (ep651-691) contains exactly ONE
  eval/verdict-cadence epoch (675). Its `profile_timing` row (dry_start_resume/run.log:82) measures
  **t_epoch_s = 71.30 s** with t_verdict_s = 0.41 s (async verdict) and the R-microbench INCLUDED —
  i.e. the excluded verdict/eval-cadence extras are, on the main thread, SMALLER than a typical epoch.
  Unmodeled residue: stage-transition ckpts (muon engage ep726, surgical engage ep700) each add one
  extra ~1540 s ablation firing — a handful over 750 epochs ≈ +1–2 h. Even a 2× amortized error
  (≈58 h = 2.4 d) stays under the DERIVED wall-clock budget **3.88 d**; the actual projection is
  1.21 d (3.2× headroom). **The budget gate cannot plausibly break on F5.**
- **Crash-resume chain verified from the receipt + artifacts:** pass1 resumed ckpt ep650 → start 651
  → completed 667 (last ckpt 666); pass2 resumed ckpt ep666 → start 667 → completed 691 (last ckpt
  690). Resume epochs internally consistent; the bench is itself the execution proof that the
  mod32cap ep650 EMA checkpoint loads and trains under this config.
- **Warm-start checkpoint custody:** npz metadata read directly — `__epoch = 650`, self_orient=1,
  hosc, curriculum flags match the config-of-record deviations (l7 parked at 1001).

**Finding F-A (cosmetic):** receipt top-level `peak_rss_gib: 63.651` is pass1's peak; pass2 peaked
**75.013 GiB**. The headline field understates the observed cross-pass peak. Envelope check
(63.651 vs 90.224 limit) also passes at 75.013; the launcher's admission uses its OWN projection
(71.54), not this field — no decision corrupted. Fix-behind: headline `peak_rss_gib = max(passes)`.

**Finding F-B (cosmetic):** epoch 675 — the only verdict-cadence epoch benched — emitted NO
`witness_component_wallclock` row (rows jump 674→676). Its cost is captured by `profile_timing`
(71.3 s), so nothing load-bearing is lost, but the row-emission gap on exactly the most
interesting epoch class is a telemetry hole worth a fix-behind.

## Surface 2 — launcher durability + confound-fix wave: **PASS**

- **Test suites EXECUTED:** `test_dry_start_delta_bench.py` + `test_witness_chain_watchdog.py` →
  **68 passed**; `test_spawn_durable_daemon_lifecycle.py` + `_memguard.py` → **30 passed**.
- **F1 read in source** (launch_witness_run.py:1590-1634): `_launch_pass_child` = Popen +
  `communicate(timeout)`; TimeoutExpired AND BaseException (incl. SystemExit from the SIGTERM
  handler) route through `_graceful_kill_child` = SIGTERM → wait(10) → SIGKILL-last-resort; pid held
  in `_ACTIVE_BENCH_CHILD` for the handler cascade. The bare-SIGKILL-orphans-trainer class is closed.
- **F3/D1:** live manifest `.omx/state/witness_chain_manifest.jsonl` contains exactly **1 row** (the
  delta bench) — no scratch/execution-proof pollution (D1a env-redirect verified in source at
  :1575-1577 with the pytest guard). Watchdog EXECUTED read-only: reports
  `CHAIN_DEAD_RECEIPTED … receipt=True rc=0` for the finished bench — correct non-alarm behavior on
  the exact phantom-death class.
- **Bench lever layer:** `test_bench_passes_disable_mod_dim_ablation_full_and_delta`,
  `test_real_launch_argv_never_carries_no_mod_dim_ablation`,
  `test_no_other_default_on_observer_rides_checkpoint_cadence` (the cadence-rider sweep), and the
  end-to-end compiled-launch.sh test all green — the poison-5 confound cannot silently recur via a
  sibling default-ON checkpoint-cadence observer.

## Surface 3 — admission-gate fix (6f96a94a30): **PASS**

- **Tests EXECUTED:** `test_system_memory_governor.py` → **77 passed**.
- **The one question that matters — can anything spuriously KILL the 29 h run mid-flight?**
  Executed + source-verified inventory of kill paths:
  1. `decide_governor_action` **NEVER kills** — pause(SIGSTOP)/resume(SIGCONT)/alert only
     (system_memory_governor.py:1825-1876), and no governor watch loop is auto-started anyway.
  2. `memory_guard --watch` (which CAN kill custody arms) is **not running** (live ps: only
     `memory_blackbox.py --daemon`, an observer) and is not auto-started by the launch path.
  3. sdd `--min-free-gb` is **launch-preflight-only** (spawn_durable_daemon.py:450-491) — no
     mid-flight actuation.
  4. The ONLY mid-flight kill is safe_run's own group-RSS cap → Finding F-C below.
  Legacy-basis `classify_pressure` (WARN <15 / CRITICAL <8 GiB available) at the run's 71-82 GiB
  profile on a 128 GiB box leaves ~40+ GiB legacy-available — pause pressure is remote, and pause is
  reversible regardless.
- **Reclaimable-aware admission EXECUTED live:** `live_admission_decision` at the current ambient
  (~29.6 GiB committed): projected-peak 63.7 → ADMIT (headroom +14.1); 75.0 → ADMIT (+2.8);
  82.0 → REFUSE (−4.2). The launcher passes its OWN projection **71.54 GiB**
  (`project_from_launch_sh` on the c2 launch.sh, executed: cf_cache 47.13 + gt 3.41 + verdict 6.0 +
  fixed 15.0), which ADMITS with ~6 GiB ceiling headroom at this ambient.

**Finding F-C (note, not blocking):** safe_run cap `--rss-cap-mb 90000` (87.9 GiB) sits **5.9 GiB**
above the inherited full-bench peak (82.02 GiB). If a real-run peak repeats 82 and coincides with a
~6 GiB verdict transient NOT already inside that 82, layer-3 SIGKILLs the trainer group. Bounded
loss: ckpt-every 25 → ≤25 epochs ≈ 58 min, fully resumable; the failure is loud and diagnosable.
Operator choice: accept (recommended — the cap is the crash-insurance) or raise the cap modestly
with an explicit GO.

**Finding F-D (note):** admission is ambient-sensitive — fire-time ambient committed must stay
≲32 GiB for the 71.54 projection to admit (measured: 29.6 ambient → +6 GiB headroom). Matches the
stated retry-loop condition; the retry loop is the right mechanism.

## Surface 4 — the launch command itself: **PASS**

- **Argv:** the real launch is the governed `tools/launch_witness_run.py` path → compiles the DSL
  spec (blocker gate re-evaluated at compile time against the hash-matched receipt) → writes
  launch.sh → `spawn_durable_daemon` (`start_new_session=True` verified, :920 — own session/pgroup,
  not a harness child) wrapping `bash launch.sh` under safe_run `--rss-mb 90000`. The B3
  sandboxed-harness warning path exists (:1897-1906); the fire must come from an unsandboxed shell —
  operator procedure.
- **No hidden auto-fire:** there is no fire-on-green daemon; "self-fires on green receipt" =
  the compile-time `C2_COMPOSED_BENCH_NOT_MEASURED` blocker clears on the hash-matched green
  receipt, and `operator_go_required: True` stands. CONTAINMENT preserved.
- **Resume-from:** launch.sh carries
  `--resume-from experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz`
  + `--warm-start-weights-only`; checkpoint verified on disk with `__epoch = 650`; the bench
  execution-proved the load (651 continuation). Real out_dir
  `experiments/results/c2_surgical_warm_20260716` does not exist yet — no stale-state collision;
  same-outdir spawn guard (p0_512) additionally active.
- **Resumability + per-stage checkpoints (CLAUDE.md non-negotiable): ON** — `--ckpt-every 25`,
  intra-stage + stage-transition preserved checkpoints (trainer:13360-13375), EMA-shadow saved,
  resume sidecar embeds config; proven twice in this very bench.
- **Dashboard/telemetry:** `is_run_dir` is the structural launch.sh+run.log fingerprint
  (witness_run_artifacts.py:112-131, name-agnostic) — the non-`levelset_`-named out_dir WILL be
  discovered by mtime; shadow observer auto-starts (ensure_shadow_observer, SENSE-only).

## Findings (one-line each)
1. **F-A cosmetic** — receipt headline `peak_rss_gib` = pass1 (63.651), omits pass2's 75.013; no decision consumed it wrongly.
2. **F-B cosmetic** — missing wallclock row for the sole verdict-cadence epoch (675); profile_timing covers it (71.3 s).
3. **F-C note** — safe_run 87.9 GiB cap vs 82.0 GiB inherited peak = 5.9 GiB margin ≈ one verdict transient; loss bounded to ≤25 epochs by design.
4. **F-D note** — admission admits 71.54 GiB projection only at ambient ≲32 GiB committed (live-executed: 29.6 → ADMIT +6 GiB); the retry loop owns this.

## Certification
I looked for a reason not to fire and did not find one. The receipt is genuine and load-bearing
(hash re-derived, arithmetic re-computed from raw rows, single-knob A/B verified at the argv level,
F5 empirically bounded by the in-bench verdict epoch); the durability wave and governor fix are
tested (175 tests executed green) and source-verified; no mid-flight path can silently kill the run;
the launch command is durable, correctly resumed, gated, and observable. **PROCEED-WITH-NOTES.**
Pointer 0.19108 UNMOVED until this run's byte-closed exact row says otherwise.
