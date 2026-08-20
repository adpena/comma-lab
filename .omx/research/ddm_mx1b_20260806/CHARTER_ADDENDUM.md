# CHARTER ADDENDUM (MAIN, post-spawn — operator correction "we have memory guards and governor")

The charter's framing "the detached path has NO memory gate" is CORRECTED: the apparatus
EXISTS — tools/system_memory_governor.py + src/tac/admission_guard.py (#254 contract:
heavy entrypoints call assert_governed_admission at top of main(); governed launchers
safe_run.py / launch_witness_run.py mark_admitted_env). The OOM was a BYPASS: the lifted
entrypoint never adopted the #254 assert, and MAIN spawned ungoverned.

THEREFORE, in addition to the charted memory fixes (all still valid):
1. ADD `assert_governed_admission("ddm_mx1_pr130_semantic_renderer")` (tac.admission_guard)
   at the top of main() for mlx-train/torch-smoke modes — the #254 lift-checklist adoption.
2. Your --mode mem-probe receipt should be shaped as the GOVERNOR's admission input (peak
   projection the governor can consume), not a parallel gate — extend, never twin (m54).
3. Ticket scheduling field: fire path = safe_run.py / governed launcher, NEVER bare
   launch_detached_process.py. Name the exact governed argv in the ticket.
Corrected memory: concurrent_metal_fires_without_composed_preflight_oomed_the_machine_20260806.
