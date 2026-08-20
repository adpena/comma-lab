Implemented and committed the continuation-launch determinizer as `de518e67e9`.

Key artifacts:

- [fire_watched_continuation.py](/Users/adpena/Projects/pact/tools/fire_watched_continuation.py) derives environment gates, resource limits, checkpoint selection, watcher configs, and launch arguments from authoritative sources.
- [local_endpoint_close.py](/Users/adpena/Projects/pact/tools/local_endpoint_close.py) waits for the canonical done receipt, retains payload evidence, runs the scorer-free fitter, and emits the typed downstream fire order.
- [evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_lh2_continuation_launch_determinizer_20260815.md) records controls, provenance, census, and boundaries.

Verification:

- Focused tests: 22 passed.
- Existing watched-launch compatibility: 17 passed.
- Ruff, compilation, diff checks, payload-retention scan, watcher validation, and two review passes passed.
- Real-parent no-fire composition retained at `/Volumes/VertigoDataTier/pact/ddm_lh2_20260815/positive_control/composed_e960_next/`.
- Five manual launch attempts were deduplicated into one closed incident in the canonical harness ledger.
- The broader Modal-closer suite had 54 passes and 3 unrelated failures caused by now-stale fixed timestamps.

The live read-only closer remains `ARMED_WAITING` under recorded supervisor PID 96666. Its source done receipt is still absent. The sacred trainer remained untouched and had reached epoch 518 at the final read. The reference trainer remains byte-identical at SHA-256 `8392a9b9f2d303698de59e627fa489a792ab0b0b38170cebd425f9310162059e`.

LH2 launched no trainer, scorer, Modal/provider work, archive build, or exact evaluation. No score was measured and neither frontier moved. Own-vehicle frontier remains `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **Disposition: FIRED. Owner: recorded local endpoint-closer supervisor PID 96666, then MAIN. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_lh2_20260815/dogfood_live_e960_v2/closure/`. Fire trigger: canonical `.omx/tmp/codex_runs/rx2_wc2_full_mps_e960.done` appears with `rc=0`; retain it, refit the endpoint, hash both final checkpoints, and emit `NEXT_FIRE_ORDER.json` for #1058.**

## LIVE-HYPOTHESES

- Source-parsed environment gates should prevent another sequential gate refusal because both the live three-gate case and a synthetic fourth gate passed the same AST-derived control.
- The armed closer should complete without intervention because every post-trigger input is local and its full closure path passed the positive control.
- The composer should cover later RX2/WC2 continuation modes whose behavior remains expressible through sealed `PORT_MODES` and the parent manifest.

## DEAD-ENDS

- Hand-written launcher/trainer arguments and shell polling are closed for this continuation class; they caused the five-attempt incident.
- Hardcoding the current three environment gates is closed; the reference trainer is now the authority.
- Treating a three-second launch window or bounded silence as process death is closed; only the canonical done receipt produces `DEAD(rc)`.
- Adding another endpoint poller is closed; the closer consumes the existing canonical receipt.
- Explicit `--nice 10` dogfood retries are closed in this sandbox because `setpriority` was denied; the successful arming retained the same enforced resource envelope without mutating niceness.