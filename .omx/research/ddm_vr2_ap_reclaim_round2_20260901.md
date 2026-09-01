# ddm_vr2 AP reclaim round 2 — pfs1 returned to Vertigo; rxc1 READY

**Disposition:** `COMPLETE_MOVE__RXC1_READY__50GB_DECIMAL_TARGET_MET`  
**Axis:** `[byte/custody apparatus, scorer-free]`  
**Frontier effect:** none. No scorer, Modal call, training, archive mutation, evaluator, or pointer write ran.

## Result

The closed pfs1 cold-store tree was certified and moved from APDataStore back to VertigoDataTier.
The current source and destination each contained **234 regular files** and **51,417,783,170 logical
bytes**. A detached full-checksum `rsync -a -c --delete --dry-run --itemize-changes` completed rc=0
after 2,078.412852 s with an exactly empty difference log. Only then was the immutable MOVE_CERT
written, the Vertigo staging directory renamed into place, the old AP path replaced with a symlink,
read-through proven on a 3,662,409,600-byte payload, and the verified AP duplicate retired.

AP available space rose by **51,450,216,448 bytes**, from **1,263,665,152** to a stable
**52,713,881,600 bytes**. Three samples ten seconds apart were identical. This meets the charter's
50 GB decimal loop target and the rxc1 1,400,000,000-byte fire trigger. It is **49.09 GiB**, not
50 GiB; no GiB claim is made.

## Certified move

| src | dst | Files | Logical bytes | Verify | MOVE_CERT | Old-path disposition |
|---|---|---:|---:|---|---|---|
| `/Volumes/APDataStore/pact/vertigo_coldstore/pact/ddm_pfs1_20260729` | `/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729` | 234 | 51,417,783,170 | rc=0; 0 itemized-difference bytes; log SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `/Volumes/VertigoDataTier/pact/ddm_vr2_ap_reclaim_round2/MOVE_CERT_pfs1_20260901.json`; SHA-256 `6a5173ff326e667e952f5642efa06c3bfd6037e2de49a337d0b88aa8044fa7df` | symlink to the destination; 3,662,409,600-byte read-through probe equal |

The completion receipt is
`/Volumes/VertigoDataTier/pact/ddm_vr2_ap_reclaim_round2/MOVE_COMPLETE_pfs1_20260901.json`
(SHA-256 `a5fdeb3ebfde3dd855bac364cb303e765be554ec56c1f54481156a1dac0d40ae`).
Copy and checksum work used the canonical detached launcher with counters 714 and 715. Their launch
manifests, logs, and done receipts are pinned inside the MOVE_CERT.

The 2026-08-20 pfs1 manifest had 307 rows and SHA-256
`b70760cd1c486cb0e2af03edc41c511788b5e1d1437b0afa34228301f8136ce2`. The current tree has 73
fewer paths, all `__pycache__/*.pyc`, and no extra paths. This is rebuildable-cache drift that
predates this move, not a payload-equality claim against the old manifest. The proof used for this
move is current-source versus current-destination full-byte checksum equality. Both current trees
had zero internal symlinks and zero `._*` AppleDouble sidecars.

## Measured tier map

| Tier | Before used | Before available | Stable post-reclaim available | Available delta |
|---|---:|---:|---:|---:|
| `/Volumes/APDataStore` | 1,952,216,576 KiB | 1,234,048 KiB (1,263,665,152 B) | **51,478,400 KiB (52,713,881,600 B)** | **+50,244,352 KiB (+51,450,216,448 B)** |
| `/Volumes/VertigoDataTier` | 1,701,907,416 KiB | 251,109,508 KiB (257,136,136,192 B) | **200,721,712 KiB (205,539,033,088 B)** | **-50,387,796 KiB (-51,597,103,104 B)** |

The AP fleet-wide writer hazard is relieved but not eliminated: the volume remains 98% full.
Vertigo retained more than 20 GiB of post-copy reserve throughout the operation.

## rxc1 gen-3 READY

`/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/BLOCKER.json` was read only and matched
SHA-256 `581a076846dfdba0164ff5b6ab4c4818258eaa61b2591ab010e27e97885d839b`. It records 26/32 sealed
screen rows and a resumable incomplete stride-200 leg at frame 400. The exact fire trigger is now
satisfied:

- **Disposition:** `READY`.
- **Owner:** MAIN.
- **Consumer store:** `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`.
- **Fire command:** `.venv/bin/python experiments/ddm_rxc1_restartable_exact_coder.py --stage screen`.
- **Measured gate:** 52,713,881,600 AP free bytes in each of three samples; required
  1,400,000,000; no concurrent decline.

This arm did not touch the rxc1 experiment or store beyond reading `BLOCKER.json`, and it did not
fire the scorer-owned continuation.

## #1165 residual dispositions and denominator

All **3/3 named charter items** were adjudicated. The current eligible AP-to-Vertigo copy population
was **1/1 moved, 0 refused**. The other two items were already terminal under the later #1364 grant,
so they are folded completions rather than new moves or candidates exhausted by this arm.

| Item | Typed disposition | Evidence | Consequence |
|---|---|---|---|
| pfs1 | `MOVED_SYMLINKED_COMPLETE` | current MOVE_CERT and completion receipt above | AP target met; loop stops |
| wwc1 | `FOLDED_ALREADY_COMPLETE` | rows 1-4 of `.omx/research/vr2_local_coldstore_move_ledger_20260831.jsonl`: 1,663 files, 2,094,598,664 B, manifest `65abedc6…e894990`, terminal `MOVED_SYMLINKED`; current original path points to `/Users/adpena/pact_cold_store/pact/pact/ddm_wwc1_winwin_cone_sweep` | do not copy or count again |
| pk4 `retained/jacobian_bank` | `FOLDED_ALREADY_COMPLETE` | rows 5-8 of the same ledger: 6,603 files, 60,907,736,706 B, manifest `146ef469…ff4a1e`, terminal `MOVED_SYMLINKED`; current original path points to `/Users/adpena/pact_cold_store/pact/pact/ddm_pk4_20260813/retained/jacobian_bank` | do not copy or count again |

The cited #1364 ledger SHA-256 is
`4b31927bb8b56a03cf6e44b1ab9353bed0ebc1ad062a1e144457b3449c36302d`.

## Tool-debt dispositions

- `FOLDED_PAID` — symlink-target and mode manifest equality now exists in
  `tools/vertigo_certify_move.py` (current SHA-256
  `874b8f897c98836245ead5e56ea98dbd9de32b4094943a69b25e64f628aa75c2`). This move's payload
  contained zero symlinks, so that code path was not exercised here.
- `QUEUED-WITH-A-FIRE-ORDER` — durable partial-retirement cleanup failure row remains owed. The
  certifier still calls bare `shutil.rmtree(tmp_old)` after cutover, so a cleanup exception can
  escape without a typed durable ledger row. Owner: storage apparatus maintainer; consumer:
  `tools/vertigo_certify_move.py`; trigger: before that tool performs its next source retirement.

## RECALL EVIDENCE

**Sources searched:** `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, the common arm contract, craft handoff,
live hot state and active-lane claims; pfs1/vr1, vr2, sr2/sr3, #1165, #1364, wwc1, pk4, rxc1, and
crossing-ledger memos/receipts; `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, design/SPEC surfaces,
task ledger, executable source, the canonical equations registry, and current SSD path state.

**Queries included:** `pfs1|ddm_pfs1_20260729|#772|#1078|#1165|#1364|sr2|sr3|wwc1|pk4|jacobian_bank`,
`certified_cold_move|MOVE_CERT|symlink_target|tmp_old|rmtree`, the exact AP and Vertigo paths,
the rxc1 blocker SHA and fire command, and bounded literal-reference scans across state, research,
source, ledger, index, and DAG surfaces.

**Beyond the charter seeds:** recall found that wwc1 and the exact pk4 subtree were already
hash-verified and retired-with-symlink by #1364; found that symlink manifest fidelity is paid in the
current certifier while durable partial-retirement failure reporting remains owed; recovered pfs1's
original 307-row manifest and classified its 73-row cache-only drift; and confirmed that the
historical Vertigo path is the path used by surviving references. This changed the plan by avoiding
two duplicate moves, keeping the original Vertigo path canonical, recording the prior-manifest drift
without pretending equality to it, and stopping after the one live candidate crossed 50 GB decimal.
The canonical equation registry supplied no storage equation that supersedes the custody contract.

## Boundaries

- Exact `lsof +D` checks found no open descriptor on the source before copy or before retirement.
  Global `ps` inspection was denied by the managed sandbox, so no broader process-census claim is made.
- No unverified byte was deleted. The source survived until full-checksum rc=0, zero differences, an
  immutable MOVE_CERT, final-path installation, old-path symlink installation, and read-through proof.
- The only retired bytes were the verified AP duplicate covered by that certificate. The payload is
  retained at the destination and accessible through both historical path surfaces.
- `/Volumes/APDataStore/pact/ddm_sg2b_*`, `/Volumes/APDataStore/pact/ddm_xov1_crossover_pass/`, and
  all rxc1 payloads were untouched. `upstream/` was untouched.
- No scorer, GT decode, MPS authority, CUDA job, Modal call, or archive evaluation ran. This is a
  byte/custody result, not a score result.

## NEXT_IF_RESUMED

- `READY-TO-FIRE` — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`; fire trigger: immediately after rechecking AP free bytes remain at least 1,400,000,000 with no concurrent decline; run `.venv/bin/python experiments/ddm_rxc1_restartable_exact_coder.py --stage screen` through the required resumable launcher path.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: storage apparatus maintainer; consumer store: `tools/vertigo_certify_move.py`; fire trigger: before the certifier's next source retirement; add a typed durable failure row around partial-retirement cleanup and preserve the existing rollback behavior.

## LIVE-HYPOTHESES

- rxc1 gen-3 can finish the remaining six screen rows without repeating the 26 sealed rows. This is plausible because its blocker records byte-close receipts, a physical frame-400 restart point, and storage reserve as the sole interruption reason; it remains untested until MAIN fires it.
- The reclaimed AP headroom will be sufficient for rxc1's projected remainder plus reserve. This is plausible because the blocker projected 1,360,789,504 bytes and the stable measurement is 52,713,881,600 bytes, but concurrent fleet writers remain an external variable.
- A typed cleanup-failure row can close the remaining certifier ambiguity without changing successful-move semantics. This is plausible because the gap is localized to the unguarded final `shutil.rmtree(tmp_old)` call after verified cutover.

## DEAD-ENDS

- Recopying pfs1 is closed: the current destination passed full-byte checksum equality, the original AP path resolves to it, and the AP duplicate is retired under an immutable certificate.
- Re-moving wwc1 or the pk4 Jacobian bank is closed: both already have terminal hash-verified `MOVED_SYMLINKED` rows and live symlink probes under #1364.
- Calling the result 50 GiB is closed: the exact stable AP value is 52,713,881,600 bytes, or 49.09 GiB. The charter's 50 GB decimal target is met.
- Treating the old 307-row pfs1 manifest as the current move's equality proof is closed: 73 rebuildable `.pyc` paths are absent. Equality was proven between the two current 234-file trees.
- Running rxc1 or a scorer from this arm is closed: the charter assigns the fire to MAIN and grants this arm no scorer lane.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; vr2 ran no scorer and did not move the pointer.`
