Completed SR3. Two closed trees were losslessly compressed, fully reconstructed and hash-verified before exact-source removal.

| Tree | Retained archive | Compression | Certified free-space gain |
|---|---:|---:|---:|
| `ddm_sa1` | 9,616,134,370 B | 2.557689× | 16,824,270,848 B |
| `ddm_b2e_f2_alone_run` | 11,606,530,604 B | 2.568771× | 18,373,279,744 B |

AP free space is now **48,109,191,168 B (44.805 GiB)**:

- W96B trigger exceeded by 14,539,812,864 B.
- SR3’s 36 GiB stop condition exceeded by 9,454,485,504 B.
- Protected `ddm_bs3_born_small_resolved` and `ddm_w96a_aligned_window` stores remain intact.
- No extraction scratch remains.
- No scorer, evaluator, training, Modal job, or upstream mutation occurred. GB1 remains `S=0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]`.

Evidence:

- [SR3 result memo](/Users/adpena/Projects/pact/.omx/research/ddm_sr3_ap_certify_compress_reclaim_20260826.md)
- [SA1 reclaim certificate](/Volumes/APDataStore/pact/ddm_sa1/SR3_RECLAIM_CERTIFICATE.json)
- [B2E reclaim certificate](/Volumes/APDataStore/pact/ddm_b2e_f2_alone_run/SR3_RECLAIM_CERTIFICATE.json)
- [Certifier](/Users/adpena/Projects/pact/experiments/ddm_sr3_ap_certify_compress_reclaim.py)
- [Tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_sr3_ap_certify_compress_reclaim.py)

Landed as commit `e12c213d8e`. Five focused tests, Ruff, diff checks, post-commit hashes, and the 709-row canonical-ledger validation passed; the worktree is clean. Prior memory informed only the fail-closed/no-signal-loss boundary—current storage and certificate facts were verified live.

## NEXT_IF_RESUMED

- Disposition: **QUEUED-WITH-A-FIRE-ORDER**. Owner: **MAIN/operator**. Consumer store: `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`. Fire trigger: AP free ≥33,569,378,304 B—currently green—and sealed W96B implementation/config gates remain green; execute `SEALED_FIRE_ORDER_W96B.json` sequentially for both seeds.

## LIVE-HYPOTHESES

- The aligned W96B seeds may produce a lower exact row because their implementation/config and storage gates are now green; this remains untested until MAIN fires them.
- Other terminal raw-array trees may compress near the measured 2.56× regime. This is plausible from two independent real trees, but each still requires explicit terminal-owner adjudication.

## DEAD-ENDS

- The predicted ≥3× compression ratio is refuted: SA1 measured 2.557689× and B2E measured 2.568771×.
- Compressing B2E first with only 12.054 GiB free is closed: it hit the 2 GiB fail-closed floor. B2E succeeded only after SA1 created headroom.
- Delete-only, symlink-only, unverified deduplication, custody-tree mutation, and live-tree mutation are inadmissible because they cannot satisfy the lossless reconstruction contract.
- SR3 itself is not a score-moving path; it removed the storage blocker but left the exact frontier unchanged.

