Implemented and committed the root-cause fix.

- Root cause: XI1 loaded `hpac_integer.py` under two module names, creating incompatible class identities. Self-compression therefore registered zero `bit_depth` parameters.
- Fix: canonical imports plus fail-closed validation of all nine tensors and 517 scalar depths throughout trainer, optimizer, EMA, checkpoint, resume, and pack paths. See [runner](/Users/adpena/Projects/pact/tools/run_ddm_xi1_screw_conditioned_learned_prior.py:559).
- Fix commit: `39e2ad7eac5ae40fef4142ca08c372211c3db8ef`
- Handoff commit: `5096755e633cb9e4bbb60b6bbcda17b78751ef64`
- Full receipt: [ddm_xi1f_leg_a_pack_schema_fix_20260812.md](/Users/adpena/Projects/pact/.omx/research/ddm_xi1f_leg_a_pack_schema_fix_20260812.md:1)

CPU verification passed: `2 passed`; Ruff, `py_compile`, inline self-test, two review passes, and the targeted payload-retention gate all passed. The retained synthetic EMA checkpoint packed and decoded with `0.0` maximum logit difference; its 14,662-byte payload repeated byte-identically. See [CPU_PACK_ROUNDTRIP.json](/Volumes/APDataStore/pact/ddm_xi1_20260812/fix/tests/CPU_PACK_ROUNDTRIP.json) and the [unit test](/Users/adpena/Projects/pact/tools/tests/test_run_ddm_xi1_screw_conditioned_learned_prior.py:27).

Resume verdict: `FRESH_RERUN_REQUIRED`. The old checkpoint contains none of the nine depth tensors and its depth optimizer group is empty. Reconstructing 8-bit defaults would invent missing rate-gradient, QAT, optimizer, and EMA history. The old payload remains preserved. Audit: [LEGACY_RESUME_AUDIT.json](/Volumes/APDataStore/pact/ddm_xi1_20260812/fix/LEGACY_RESUME_AUDIT.json).

Exact MAIN re-fire command:

```bash
PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/safe_run.py --rss-mb 12288 --projected-gib 12 --timeout 2390 --label ddm_xi1f_leg_a_n120 --status-receipt /Volumes/APDataStore/pact/ddm_xi1_20260812/fix/leg_a.safe_run.json -- .venv/bin/python tools/run_ddm_xi1_screw_conditioned_learned_prior.py --leg a --epochs 20
```

No MPS or scorer run occurred, and no corrected real-byte row exists yet. Own-vehicle frontier remains `S = 0.16959899569230852 @ 187,226 B [macOS-CPU advisory, n600]`.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN Metal executor; consumer store: `/Volumes/APDataStore/pact/ddm_xi1_20260812/fix/LEG_A_RESULT.json`; fire trigger: governed MPS availability and commit `39e2ad7eac` present; execute the pinned fresh-run command above.

## LIVE-HYPOTHESES

- The corrected run will learn nonuniform bit depths, plausibly following CL1’s proven 0–8-bit distribution on the same nine-module mechanism.
- ξ conditioning may reduce real Range bytes against the matched spatial control because it changes the causal `conv_past` signal without changing capacity or schedule.

## DEAD-ENDS

- `strict=False` packing is closed because it silently invents quantization state.
- Reconstructing the old checkpoint with default depths is closed because its optimization and EMA history never existed.
- Treating the old empty-histogram bpp as a learned-self-compression result is closed because the rate mechanism was inactive.