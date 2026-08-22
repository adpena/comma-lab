Fork B is confirmed: `RX1M` is the intended outer shipping container. The generated F26 receiver incorrectly validated the wrapped semantic body before unwrapping it.

The fix in [ddm_jo2_receiver_close.py](/Users/adpena/Projects/pact/experiments/ddm_jo2_receiver_close.py:975) now:

- Unwraps `J2R1` before checking the inner `WANS1`/`SD1M`/`SM3R` tag.
- Executes the exact staged shipping receiver before closure.
- Refuses the receiver-close pointer unless `0.raw` is exactly 3,662,409,600 bytes.

Measured control: the exact 181,484-byte r8 archive decoded successfully in 722.48 seconds through the fixed shipping tree. The retained raw SHA-256 is `98eb8cb2…`; both n600 identity censuses reported zero mismatches. [Control receipt](/Volumes/VertigoDataTier/pact/ddm_jo6_receiver_container_compat/real_receiver_control_0000/CONTROL_RESULT.json) · [identity receipt](/Volumes/VertigoDataTier/pact/ddm_jo6_receiver_container_compat/real_receiver_control_0000/IDENTITY_CONTROL_RESULT.json).

Tests: 37 passed, plus Ruff, `py_compile`, diff checks, and two review passes. The isolated serializer commit is `0f018e8216ee9fc7d7049656d7c2d6303a6856ae`. Canonical landing remains blocked because the shared Git object store rejected writes; the index is empty and the durable [bundle](/Users/adpena/Projects/pact/.omx/research/ddm_jo6_receiver_container_compat_20260822/SOURCE_FIX_0f018e821.bundle) is ready.

The source-pinned r9 seal is active after a byte-identical checkpoint migration. At handoff it had completed 21/600 fresh-Schur pairs under PID 37877 at 4.49 GiB RSS. `RECEIVER_EXECUTION_POINTER.json` is not yet written, so no runtime-completion or score claim is made. Full evidence and GESTALT-DELTA are in the [JO6 memo](/Users/adpena/Projects/pact/.omx/research/ddm_jo6_receiver_container_compat_20260822.md).

GESTALT-DELTA: receiver closure previously proved only Python parse-back. It now requires both parse-back and a complete decode through the receiver that will actually ship.

## NEXT_IF_RESUMED

- **FIRED** — owner: MAIN; consumer store: `ddm_jo6_receiver_container_compat_20260822_r9_final/stages/01_target_birth`; fire trigger: PID 37877 writes `RECEIVER_EXECUTION_POINTER.json` or a terminal receipt; action: harvest the pointer and subsequent stage disposition.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN/operator; consumer store: canonical Git `main`; fire trigger: canonical Git object writes become available; action: land bundle commit `0f018e821` through the serializer and verify HEAD content hashes.

## LIVE-HYPOTHESES

- R9 should reproduce r8’s candidate codes and archive bytes because its migrated training state, residual payload, and candidate master are byte-identical.
- The later `run_receiver` step should pass because the same shipping execution is now mandatory inside receiver closure.

## DEAD-ENDS

- Rewrapping RX1M: closed because RX1M is the outer container; the other magics identify its inner semantic body.
- Using a non-shipping receiver: closed by the exact shipping-tree control.
- Directly resuming the old r8 seal: closed because the live inventory SHA-pins the receiver source.
- Transplanting r8 pointers into r9: closed because their workload identities differ.
- Treating failed `kill -0` as proof of daemon death: closed because the governor and advancing receipts refuted it.

Own-vehicle frontier remains dx2: S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4 n600]`.