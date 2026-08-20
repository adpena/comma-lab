`READY_TO_FIRE` at commit `c063f85ae10a4f418d5f096bd16b2944fde64fa7`.

The worker now uses the adapted runtime’s canonical F26 receiver, bootstraps its exact `Brotli==1.2.0` dependency into retained custody, passes the correct `cpr1/` decoder path, and fails closed against the CP135 token golden.

Measured proof:

- Exact 186,252-byte CP135 archive decoded across all 600 frames.
- 117,964,800 token bytes matched the canonical plane byte-for-byte.
- SHA-256: `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.
- Scorer ran: false. Modal dispatched: false.
- Focused tests: 11 passed. Ruff, compilation, diff check, payload-retention gate, and two review passes passed.
- Developer preflight remains 17/25 due the same eight repository-wide findings documented by the parent; this is not release-preflight green.
- Transient runtime `__pycache__` files were removed; archive/runtime source bytes remain unchanged.

Artifacts: [handoff memo](/Users/adpena/Projects/pact/.omx/research/ddm_vd1b_worker_adapted_decode_fix_20260812.md), [full n600 success receipt](/Volumes/VertigoDataTier/pact/ddm_vd1b_20260812/fix_no_brotli_detached/FIXED_WORKER_LOAD_SUCCESS.json), [pre-fix reproduction](/Volumes/VertigoDataTier/pact/ddm_vd1b_20260812/repro_before_no_brotli/WORKER_LOAD_RECEIVER_STATE_FAILURE.json), [adapted-reader control](/Volumes/VertigoDataTier/pact/ddm_vd1b_20260812/adapted_reader/ADAPTED_READER_SUCCESS.json).

The unchanged fire command is:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/ddm_vd1_modal_batch_event_validator.py::main \
  --archive /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip \
  --runtime /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime \
  --event-store /Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200 \
  --jo1-analysis /Volumes/VertigoDataTier/pact/ddm_jo1_20260812/10_ANALYSIS.json \
  --output-dir .omx/state/ddm_vd1_modal_batch_event_validator/ddm_vd1_20260812 \
  --k 200 --run-id ddm_vd1_20260812 --resume-from ddm_vd1_20260812 \
  --lane-id ddm_vd1_modal_batch_event_validator \
  --instance-job-id modal:ddm_vd1_20260812 --claim-agent codex:ddm_vd1 \
  --detach --provider-detach-ack
```

Effective frontier remains CP135 `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`. Own-vehicle frontier remains LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN scorer-lane router. Consumer store: Modal volume `comma-ddm-vd1-event-validator-retained/ddm_vd1_20260812/` and local harvest `.omx/state/ddm_vd1_modal_batch_event_validator/ddm_vd1_20260812/`. Fire trigger: ps135 and every other Modal/scorer claim are terminal, single-flight reconciliation passes, and release-preflight findings are adjudicated.

## LIVE-HYPOTHESES

- The repaired worker should clear the remote T4 receiver stage. This is plausible because the exact archive passed the Brotli-absent local seam and full n600 token-identity gate, but the actual Modal installation/network environment remains untested.
- The K=200 census may produce an interaction-safe selection whose optimistic Seg gain clears `0.000216`. This remains plausible because it evaluates the complete generation-1 event alphabet, but no scorer ran here.

## DEAD-ENDS

- Treating the archive as corrupt is closed: the adapted reader parsed the same bytes and the fixed worker reproduced the canonical token plane.
- Calling the residual parser directly without the adapted dependency bootstrap is closed: absent Brotli silently selected the invalid LZMA compatibility path.
- Adding Brotli while retaining `runtime/` as the HPAC code directory is closed: it exposes the missing `hpac_integer_sparse` failure; canonical code lives under `cpr1/`.
- Silent long-running sandbox launches are closed: they were terminated without receipts. The retained-heartbeat PTY gate completed and produced the decisive receipt.

