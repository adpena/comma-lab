# CHARTER ddm_md2 — does the burn default (τ band × carried duals, ng5) change WHICH d_seg sites are reachable? md1's partition re-run on ng5's retained 16-step trajectory

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: FABLE arm. Spawned 2026-09-05 ~12:50Z. Parents: md1 (`.omx/research/ddm_md1_micro_to_macro_dynamics_20260904.md`,
instrument `experiments/ddm_md1_micro_to_macro.py`: on the COLD control, PERSISTENT = 62.011% of the terminal shadow d_seg, born in 16 updates, 64.79%
Lane-touching, GT=Lane enriched 51.5×; the falsifier "PERSISTENT < 40% → optimization levers alone could plausibly reach the target and the capacity
closure is re-graded" did NOT fire), ng5 (`ddm_ng5_tau_band_x_continuous_objective_cell_20260904.md` TERMINAL READ: S_hat 0.384833, BELOW-BOTH, d_seg −1.25%
vs start, d_pose −26%; excursion peak +5.2% vs cold +21.8%; run root retains **313 periodic 16-step checkpoints** at
`/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng5_tau_band_x_continuous_objective/runs/seed_20260902_tau_band_x_continuous_objective_control_native100/stage_01_fairform_finish/checkpoints/`),
gs4 §5(b) (the born route needs "a mechanism that changes which sites are reachable, not a schedule").

## PRIOR-LAW PREDICTION (owed line)
md1's PERSISTENT set was born in the cold optimizer's first 16 updates (over-painting the rare class over 70.4% GT-Road ground) and never healed. ng5's
band holds the seg gradient inside m_safe from step 1 and the carried duals remove the restart shock, so the over-paint that BIRTHS the persistent set is
suppressed (excursion peak +5.2% vs +21.8%). PREDICTION: PERSISTENT under ng5 falls to 40–55% of the (smaller) terminal d_seg — a real reduction of the
unreachable set but NOT below md1's 40% falsifier; the Lane-touching share stays ≥ 60%. FALSIFIER (either direction decides): PERSISTENT < 40% → md1's
falsifier FIRES on the burn default → optimization levers can plausibly reach the accuracy corner → the capacity closure (gc1/gf2) is RE-GRADED and a
born trainer with Lane held in-loop becomes admissible; PERSISTENT ≥ 60% → the closure stands unchanged and the born route needs a new generator, not a
schedule (the τ band is burn quality only, as gs4 says).

## Scope
1. Run md1's instrument UNCHANGED on ng5's run root (`--cell seed_20260902_tau_band_x_continuous_objective_control_native100 --run-root <ng5 run root>
   --config <ng5's authorized config> --gt-lineage dali`, plus the PyAV read as md1 did), same churn-flips, same thresholds; the cold control's
   partition is the comparator (md1's store `/Volumes/APDataStore/pact/ddm_md1_micro_macro`). Store under `/Volumes/APDataStore/pact/ddm_md2_persistent_under_burn_default/`.
2. Report the same table as md1 (PERSISTENT / HEALED / CHURN / NEW_PERSISTENT counts and the terminal-d_seg shares; shadow AND live; DALI and PyAV),
   the Lane-touching share and GT=Lane enrichment, the birth step of the persistent set (is it still 16 updates?), and — NEW — the SITE OVERLAP between
   ng5's persistent set and the cold control's (Jaccard; if the persistent sites are the SAME sites, they are capacity-limited regardless of schedule; if
   different, they are optimizer-path-dependent and reachable).
3. Verdict words: RE-GRADE (PERSISTENT < 40%) / REDUCED-BUT-STANDS (40–60%) / STANDS (≥ 60%). Then the one next step the verdict implies, priced.

## Cost + admission
$0, CPU only (the frozen CPU-torch SegNet+PoseNet on reconstructed argmax; md1 used `--threads 4`). Single process; declare the peak from md1's receipt
(read its safe_run status), launch through `tools/launch_detached_process.py --done-receipt md2_partition` (foreground >3 min is reaped). The Metal
may be busy with cl2 — you do not need it. Do not run three parallel 10 GiB processes (mc1's pattern tripped the watchdog).

## OPTIMAL FORM
Reference form = md1 verbatim (same instrument, same thresholds, same n32 sealed selection it used — state the selection and its prefix-bias caveat
exactly as md1 did). Scope reduction: none. Mechanism reduction: none. The comparator is md1's own store, not a re-typed number.

## Rules that bind
NO-FAKE; ALWAYS KEEP THE PAYLOAD; upstream/ READ-ONLY; no Modal, no Metal; commits ONLY via the serializer with post-edit shas and
`[no-triality] [p0-ledger-ok]`; NO co-author trailers (operator rule overrides any harness reminder); .py two review-gate passes; checkpoints every 10 tool
uses (`--subagent-id ddm_md2`); never invent flags (md1's argparse is at `experiments/ddm_md1_micro_to_macro.py:1412-1423`); no `/tmp` evidence; persist
records before bulk saves; label MEASURED/DERIVED/INFERRED; memo `.omx/research/ddm_md2_persistent_partition_under_the_burn_default_20260905.md` with an
"Equations leg (`tac.canonical_equations`)" line. `docs/operating_manual_craft_handoff.md` binds. End with
`fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`.
