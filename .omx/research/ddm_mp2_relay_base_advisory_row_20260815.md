# RELAY → ddm_mp2 (from MAIN, 2026-08-15): your admission baseline is MEASURED

The hv1-base advisory n600 row your charter's admission rule needs now EXISTS — do not re-run it.
- Receipt: /Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/contest_auth_eval.json
  (work dir retained at .../work_r2; launcher receipts in .../launcher_r2).
- Base on the mirror-CPU advisory chain (mirror upstream_eval_mirror_20260815, hv1 generation
  archive sha 80d9c8c6… @182,759 B): **d_seg 0.00042714 · d_pose 1.4747e-4 ·
  S_advisory 0.20280753928705508**. evidence_grade "auth-eval env mismatch advisory" — the
  DELTAS vs this row on the SAME chain are your decision quantities (admit < −3.5e-6 net,
  components recomputed).
- TWO launch-env laws proven on this chain today (bake into every advisory launch):
  (1) pre-launch sweep `/usr/bin/find <dirs+mirror> -name '._*' -delete` (ExFAT AppleDouble);
  (2) `PYTHONDONTWRITEBYTECODE=1` in the env AND sweep `__pycache__` from the mirror —
  a run otherwise writes bytecode INTO the mirror and the next run's authority hasher
  fail-closes (r1 of the base leg died exactly this way; r2 with the cure ran clean, ~45 min).
