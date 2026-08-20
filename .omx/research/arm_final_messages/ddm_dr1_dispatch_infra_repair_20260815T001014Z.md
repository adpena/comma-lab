Implemented and committed the DR1 repair as `10c81bd3c26e5b99c799a81586c038e301940fca`.

- R1: resealed fire order SHA `e7527b7c…`; request remains byte-identical at `c9d6d62c…`.
- R2: fixed the partial package-mount/import chain. Local isolated topology simulation passes. The required Modal CPU smoke remains externally blocked after two pre-entrypoint connectivity failures.
- R3: fixed CPU/CUDA call-ID registration and executed the F26R closer control: `CLOSED`, `rc=0`, two canonical ledger rows, verified retained payload, idempotent rerun.
- Verification: 51 tests passed, Ruff/compile/diff checks passed. No T4 or scorer work ran.

Full handoff: [ddm_dr1_dispatch_infra_repair_20260814.md](/Users/adpena/Projects/pact/.omx/research/ddm_dr1_dispatch_infra_repair_20260814.md).

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_dr1_dispatch_infra_repair_20260814/retained/mt1_import_smoke/IMPORT_SEAL_PARSE_SMOKE.json`; fire trigger: Modal connectivity is restored; run the exact CPU-only smoke argv recorded in the handoff and require `passed=true`.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN sole Modal scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1`; fire trigger: CPU container smoke is green, the T4 lane is free, and all sealed hashes match; consume `SEALED_FIRE_ORDER.json` unchanged.

## LIVE-HYPOTHESES

- The repaired package-path image will pass once Modal is reachable; the exact former topology failed locally and the corrected isolated topology passes.
- Immediate call-ID registration should eliminate the F26R closer failure class on the next detached CPU or CUDA auth dispatch.
- The unchanged MT1 experiment may reproduce its local negative sign on T4; this remains untested.

## DEAD-ENDS

- `experiments/__init__.py` is not the root cause; the partial mount and transitive exception capture were.
- Extending the closer timeout is not a cure; it received terminal `rc=0` before refusing the missing ledger row.
- Further Modal retries without a connectivity change are closed after two identical pre-entrypoint failures.
- Firing the real T4 gate in this arm was forbidden and was not attempted.
- Own-vehicle frontier: `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`; unchanged.