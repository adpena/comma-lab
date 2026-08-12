# ddm_xi1f — Leg-A learned bit-depth schema fix

**Status:** root cause fixed and CPU pack round-trip verified in commit `39e2ad7eac`; a fresh Metal rerun is queued. No MPS run, scorer run, archive build, or score measurement occurred. Frontier **UNMOVED**.

## Conclusions

1. XI1 loaded `hpac_integer.py` as private module `ddm_xi1_hpac_integer`, while `hpac_self_compress.py` imported it as canonical module `hpac_integer`. Python created distinct `IntegerConv2d` and `IntegerLinear` class objects. All self-compression `isinstance` checks therefore returned false, so the trainer registered zero `bit_depth` parameters, the rate term saw zero variable-weight bits, the histogram stayed empty, the optimizer's bit-depth group stayed empty, and EMA checkpoints could not satisfy the packer.
2. The fix imports the three mutually dependent PR130 modules through their canonical names, validates their source paths and shared class identities, and fails closed unless all nine expected bit-depth tensors are present in the trainer, optimizer, live checkpoint, EMA shadow, resume path, and pack load path.
3. The checkpoint schema is now `ddm_xi1_hpac_checkpoint.v2`. Corrected Leg-A payloads go under `/Volumes/APDataStore/pact/ddm_xi1_20260812/fix/`, preserving the invalid historical run unchanged.
4. The existing epoch-20 checkpoint is not reconstructibly resumable. Its live and EMA states contain zero of nine bit-depth tensors; its dedicated optimizer group has 0 parameters (`[19, 0, 9]`) versus CL1's 9 (`[19, 9, 9]`). Initializing nine tensors to 8 now would invent their missing optimizer, rate-gradient, QAT, and EMA history. Disposition: `FRESH_RERUN_REQUIRED`.

## CPU verification

- Command: `.venv/bin/pytest -q tools/tests/test_run_ddm_xi1_screw_conditioned_learned_prior.py`
- Result: `2 passed`.
- Axis: `[macOS-CPU unit test; synthetic checkpoint; scorer-free]`.
- Fresh trainer registration: 9 bit-depth parameter tensors, 517 scalar depths; initial histogram `{"8": 517}`; optimizer group counts `[19, 9, 9]`.
- Synthetic EMA checkpoint → production pack load → `IHS1` serialize → plain receiver deserialize: max absolute logit difference `0.0` on the seeded CPU fixture.
- Packed payload: 14,662 B, SHA-256 `fdc0b53fd4d22d4ebdfa970621e66d37a03fe40782a9d7a8b7a42a6939db9e9f`; deterministic repeat byte-identical.
- Evidence: `/Volumes/APDataStore/pact/ddm_xi1_20260812/fix/tests/CPU_PACK_ROUNDTRIP.json`.
- Legacy audit: `/Volumes/APDataStore/pact/ddm_xi1_20260812/fix/LEGACY_RESUME_AUDIT.json`.
- Consolidated fix result: `/Volumes/APDataStore/pact/ddm_xi1_20260812/fix/FIX_RESULT.json`.
- Static checks: Ruff clean, `py_compile` clean, inline self-test PASS, diff check clean, targeted payload-retention gate 0 findings.
- Review tracker: two consecutive clean post-edit passes covered 49 runner entities and 3 test entities. No override was used.

## RECALL EVIDENCE

Searched `.omx/research/`, canonical research indexes, `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`, `.omx/state/main_hot_state.md`, task/queue ledgers, and the canonical equation registry using `ddm_xi1|screw-conditioned`, `HPAC|IntegerHPAC`, `self-compress|self_compression`, `bit_depth`, `EMA|ema_shadow`, and `pack|round-trip`.

Beyond the charter seeds:

- RR3 established the real PR130 schema: 517 learned scalar depths across nine integer affine modules, 259 B of depth descriptors, and checkpoint-to-packer max logit difference 0. This changed the guard from a nonempty check to an exact nine-name schema check.
- HB2 established that HPAC pack acceptance requires source/consumer logit equality and real payload retention, not merely a successful `load_state_dict`. This changed the CPU regression into a complete synthetic EMA load, serialize, deserialize, exact-logit round trip with retained raw bytes and repeat.
- The canonical equations `weight_entropy_rate_in_loss_lever_v1` and `weights_at_order0_entropy_floor_v1` confirmed that trained bit depth is part of the live rate mechanism, so default reconstruction or `strict=False` would be mechanism-invalid.
- CL1's retained checkpoint and Gate-3 receipts proved the expected full optimizer/EMA/pack shape and provided the resume comparison. This made the missing rate-gradient history, not just the missing state keys, decisive for the fresh-rerun verdict.
- Did not find a prior diagnosis of the duplicate-module-class-identity failure in the bounded HPAC/self-compression research, index, DAG, and task-ledger scope.

## Exact MAIN re-fire

Fire only when a governed MAIN process reports `torch.backends.mps.is_available() == True` and commit `39e2ad7eac` is present:

```bash
PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/safe_run.py --rss-mb 12288 --projected-gib 12 --timeout 2390 --label ddm_xi1f_leg_a_n120 --status-receipt /Volumes/APDataStore/pact/ddm_xi1_20260812/fix/leg_a.safe_run.json -- .venv/bin/python tools/run_ddm_xi1_screw_conditioned_learned_prior.py --leg a --epochs 20
```

Queue disposition: `QUEUED-WITH-A-FIRE-ORDER`. Owner: `MAIN Metal executor`. Consumer store: `/Volumes/APDataStore/pact/ddm_xi1_20260812/fix/LEG_A_RESULT.json`. Fire trigger: governed MPS availability plus commit `39e2ad7eac`. Machine-readable order: `/Volumes/APDataStore/pact/ddm_xi1_20260812/fix/queue/leg_a_mps.json`.

## Boundaries

- No MPS execution was performed by this arm.
- No real Leg-A coded-byte row exists after the fix; the prior spatial bpp `0.120087...` is invalid as a learned-bit-depth rate result because its rate mechanism was inactive.
- No scorer or exact evaluator was loaded. No score or frontier improvement is claimed.
- Current own-vehicle frontier remains LC2 `S = 0.16959899569230852 @ 187,226 B [macOS-CPU advisory, n600]`.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN Metal executor; consumer store: `/Volumes/APDataStore/pact/ddm_xi1_20260812/fix/LEG_A_RESULT.json`; fire trigger: governed `torch.backends.mps.is_available() == True` and fix commit `39e2ad7eac` present; execute the pinned fresh 20-epoch command above.

## LIVE-HYPOTHESES

- Learned bit depths will become nonuniform during the corrected run, because the proven CL1 chain trained the same nine-module mechanism to a broad 0–8-bit histogram; this is plausible but untested for XI1's n120 cells.
- The ξ-conditioned cell may reduce real Range token bytes relative to the matched spatial control once the rate mechanism is actually active, because carried screw context changes `conv_past` input while topology, capacity, seed, and schedule remain matched; no corrected byte row exists yet.

## DEAD-ENDS

- `strict=False` at pack load is closed: it would silently substitute default depths that drive real quantization and create a fake pack.
- Reconstructing the old epoch-20 checkpoint with 8-bit defaults is closed: all rate-gradient, optimizer, QAT, and EMA history for the nine missing tensors is absent.
- Treating the prior empty histogram and bpp as a learned-self-compression result is closed: self-compression never registered on that run.
