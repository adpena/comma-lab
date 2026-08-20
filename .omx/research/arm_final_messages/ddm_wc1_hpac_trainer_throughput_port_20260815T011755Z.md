The charter is blocked at its mandatory profile-first gate. No throughput port, ≥3× claim, parity trajectory, or serialized endpoint was produced.

The managed sandbox refused `nice -n 10` with `Operation not permitted`. The resulting child was terminated after 42.06 seconds; PID 87994 is gone, its governed receipt says `status=killed`, and no profile/checkpoint payload was discarded. `taskpolicy` and launchd handoff were also refused. Live RX2 PID 63183 remains healthy and untouched.

The complete evidence and recall handoff is in [ddm_wc1_hpac_throughput_port_20260814.md](/Users/adpena/Projects/pact/.omx/research/ddm_wc1_hpac_throughput_port_20260814.md). The primary Git database is read-only, so that file remains untracked there. Fallback custody was committed through the required serializer:

- Commit: `49ce553238dbcf5824ae6fa25db247db2fe8b9d4`
- Memo SHA-256: `45315e8c49a59a52f4d19f4305f61d2bb567286d006d52c43d2f2b32da39e158`
- Transfer patch: [0001-ddm-wc1-record-blocked-HPAC-profile-gate-no-triality.patch](/Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/fallback_patch/0001-ddm-wc1-record-blocked-HPAC-profile-gate-no-triality.patch)
- Patch SHA-256: `b8ba4dad921b67ccd550d8f012a27206297ed103f1936bb7153bb7fd75ae6c2b`

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER / BLOCKED-ENVIRONMENT** — owner: MAIN or unsandboxed WC1 executor; consumer store: `/Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/`; fire trigger: prove `nice -n 10` sets priority 10, confirm live RX2 cadence, then run the memo’s exact two-epoch profile command in a fresh directory.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: successor WC1 builder; consumer store: the WC1 SSD root and receipt memo; fire trigger: the compliant profile identifies its hot stage; implement only the first ladder rung that measures ≥3×.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: successor WC1 verifier; consumer store: the same WC1 root; fire trigger: a ≥3× rung exists; run identical-seed six-epoch parity including at least two QAT epochs and retain both real IHS1-packed endpoints.

## LIVE-HYPOTHESES

- Convolution forward/backward is likely dominant because every batch performs multiple full-resolution masked/depthwise convolutions plus backward. This remains source-derived, not profiled.
- Torch MPS may exceed 3× on this convolution-heavy model, but STE operations, deterministic support, one-hot expansion, and synchronization could erase the gain.
- A CPU optimization may exist around repeated one-hot/layout materialization or batching, but batch shape changes the instrument and cannot be selected before profiling.

## DEAD-ENDS

- Literal niceness inside this managed sandbox is closed: `nice`, self-`setpriority`, and `taskpolicy` were denied.
- Launchd handoff is closed as a verified niceness route here; the bounded control returned rc 1.
- Borrowing prior WC1/WC2 speedups is closed because those receipts concern a different MLX renderer and training instrument.
- Building from source inspection alone is closed by the charter’s profile-first rule.

Own-vehicle frontier: **S = 0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]**.