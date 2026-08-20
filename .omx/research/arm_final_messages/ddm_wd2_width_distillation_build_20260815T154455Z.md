WD2 is built, receiver-closed, source-pinned, dry-run validated, and committed as `706a8f9d9680989aa5d0c1ff67d2950ffa88df12`.

No training, Modal dispatch, or scorer ran. The live e960 process and its controls were untouched.

Key results:

- Primary design: flattened depth-4/width-64 student, exact raw packet 19,465 B.
- Strict rate-only sub-0.15 requirement: save at least 15,157 B, reaching archive ≤168,345 B. The charter’s 15,153 B rung is four bytes short.
- Random-init receiver-closed apparatus: 166,169 B, but non-promotable because distortion is unmeasured.
- Inactive receiver: 7/7 container fields identical; pair-0 camera output byte-identical.
- Memory probe: 2,455,371,776 B peak RSS at full geometry, zero optimizer steps.
- Verification: 6 tests passed, Ruff and compilation clean, payload-retention census 0/3 findings, two review passes complete, watcher configs valid, and both final source-pinned launch commands dry-run successfully.
- The repo-wide developer preflight remains red on eight existing global gate classes; an explicit path audit found zero WD2 matches in all eight.

Full evidence is in the [WD2 memo](/Users/adpena/Projects/pact/.omx/research/ddm_wd2_width_distillation_build_20260815.md:3) and [sealed launch ticket](/Users/adpena/Projects/pact/.omx/research/ddm_wd2_width_distillation_build_20260815/launch_ticket.json:1).

## NEXT_IF_RESUMED

- Disposition: `QUEUED-WITH-FIRE-ORDER`; owner: `MAIN local-Metal executor`; consumer store: `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/teacher_cache_e480b`; fire trigger: e960 has a durable terminal/approved early-stop receipt, terminal controls, and explicit Metal-lane handoff. Action: materialize the retained n600 teacher cache.
- Disposition: `QUEUED-WITH-FIRE-ORDER`; owner: `MAIN local-Metal executor`; consumer store: `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/primary_flattened_d4_w64`; fire trigger: teacher-cache receipt verifies and its watcher has no alert. Action: run the 60-epoch flattened d4/w64 QAT stage.
- Disposition: `QUEUED-WITH-FIRE-ORDER`; owner: `MAIN scorer-slot owner`; consumer store: the primary student store above; fire trigger: a receiver-closed candidate saves more than 2,051 B, holds the fidelity band, and the sole n600 scorer slot is claimed. Action: evaluate that exact retained archive.
- Disposition: `QUEUED-WITH-FIRE-ORDER`; owner: `WD2 semantic/carrier successor`; consumer store: `/Volumes/APDataStore/pact/ddm_wd2_width_distillation`; fire trigger: semantic distillation holds exact distortion but saves fewer than 15,157 B. Action: open carrier ownership as stage 2 while preserving the admitted semantic archive.

## LIVE-HYPOTHESES

- Flattened d4/w64 should fire first because it keeps four receptive blocks and full-rank mixing while removing repeated FiLM storage.
- Factorized d4/w64/r19 may win if per-block temporal conditioning matters more than full-rank pointwise mixing.
- Dense d4/w56 may preserve teacher behavior best because it changes the inherited computation least.
- Precision waterfill may remove further bytes once trained student sensitivity is measured through the actual receiver.
- Carrier ownership remains plausible if semantic fidelity holds but the archive misses 168,345 B, because 22,161 inherited carrier bytes remain untouched.

## DEAD-ENDS

- Exact recoding of the frozen teacher is closed: mz2 found all 38 tensors receiver-required and tested alternatives grew bytes.
- Weight-space MSE is closed because it does not measure realized output through resize and uint8.
- Choosing one guessed width is closed because no current-renderer distortion law supports it.
- Calling a 15,153 B saving sub-0.15 is closed by exact arithmetic.
- Promoting the 166,169 B random-init archive is closed because it has no trained or scorer fidelity.
- Vertigo is closed for the future teacher cache and n600 renders at current free space; those payloads route to APDataStore.

Vehicle frontier unchanged: S=0.1600920261571558 @ 183,502 B `[contest-CUDA T4, n600]`.

