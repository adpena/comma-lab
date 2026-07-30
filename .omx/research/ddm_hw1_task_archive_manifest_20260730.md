# ddm_hw1 — TASK-LEDGER ARCHIVE CUSTODY MANIFEST (Deliverable 3, task #785 / QA88)

**Pointer honesty first:** `0.1910828242 [contest-CPU custody]` UNMOVED. This is apparatus
(context-pollution cure), a MEANS, not a score-mover.

**Purpose.** Cure the context-pollution tax: the harness dumps ~780 mostly-completed task rows
into MAIN's context every few turns. This manifest gives MAIN a **custody-verified**
SAFE-TO-ARCHIVE candidate set + a DO-NOT-ARCHIVE guardrail, so MAIN can prune the live task list
harness-side without losing signal.

**STORES CONSULTED:** git commit-subject log (last 6000 commits, the durable research timeline
per CLAUDE.md "git history IS our research timeline"); `.omx/research/*.md` (memo mentions);
`.omx/research/ddm_deferral_queue_ledger_20260729.md` (the live QA/QB/QC/QD/QE task-tracking
surface); grep across the number-space. Reconstruction script:
`scratchpad/reconstruct_task_custody.py` (one-shot, not landed).

---

## PRIMARY FINDING (honest scope bound — read before trusting any number below)

**`#NNN` is a SHARED number-space in this repo** — task#, Catalog#, PR#, and canonical-equation#
all use `#NNN`. A bare-`#NNN` custody claim is therefore **unreliable** (confidence-laundering,
operating-manual §5): e.g. bare `#26` has **1480** `.omx/research` mentions because it is
*canonical-equation #26 / Catalog #26*, **not** a task. A naive bare-`#NNN` sweep reports 980
"SAFE" numbers — most of which are catalog/equation/PR collisions, not tasks. **That list is NOT
trustworthy and is deliberately NOT delivered as the SAFE set.**

**Two signals ARE defensible as task-scoped custody:**
1. **Explicit `task #NNN`** in a git commit subject — a commit that literally says "task #494" is
   durable, verifiable custody for task 494 (the commit sha is the pointer).
2. **The deferral-queue ledger row-IDs** (`QA##`/`QD##`/`QE##`) — the repo's actual live
   task-tracking surface, each with a status column.

**Hard limit (stated, not hidden):** the ~780-row completed-task list lives **harness-side** and
**cannot be queried from this repo**. This manifest therefore **cannot enumerate completed-status
authoritatively** and **cannot invent it**. It delivers what IS verifiable — a custody index MAIN
cross-references against the harness list to make the actual archive decision. NO-CUSTODY-FOUND
(a harness "completed" row with zero repo evidence) is **only determinable by MAIN** doing that
cross-reference; it is not greppable here.

---

## COUNTS (defensible signal only)

| set | count | meaning |
|---|---:|---|
| task#s with explicit `task #NNN` git-commit custody | **67** | durable, verifiable; commit sha = pointer |
| ...of those, in an OPEN/DUE/FOLD ledger row (KEEP-LIVE) | **0** | none of the 67 are still open |
| **SAFE-TO-ARCHIVE candidates** (custody ∧ not open) | **67** | MAIN may archive from the live list after cross-ref |
| OPEN-class ledger rows (DO-NOT-ARCHIVE guardrail) | **60** | `QA##`/`QD##`/`QE##`, status OPEN/DUE/FOLD/rider/LIVE |
| ledger rows already settled (CLOSED/DONE/BUILT/MEASURED) | **6** | `QA71 QA78 QA86 QD01 QD02 QD03` — archivable from the queue's own tracking |

Cross-check note: `QA86` shows in both the OPEN and settled scans (a duplicate/status-transition
row) — the ledger's own **status column is the authority**; MAIN resolves it on prune.

---

## SAFE-TO-ARCHIVE (67 task#s — explicit `task #NNN` commit custody, none in an open ledger row)

Format `task# : latest_commit_sha : n_commits`. Each sha is the verifiable custody pointer.

```
#36:a026a95487:1   #48:da92030c50:1   #52:752a30cdb9:1   #54:f9dfc2f155:1   #55:090a3cf6b1:2
#58:34133cd04e:1   #59:5fd6ab8ea7:1   #60:61c438fe03:1   #63:887651cfb0:1   #66:c7eae734ca:1
#67:316460d6ff:1   #68:89e8829c63:1   #69:242fa9c42f:3   #72:8b3a95d3d3:4   #73:f458cc14c0:1
#76:eb449c50c4:1   #77:a22029b241:1   #78:89f6c3bc6a:1   #82:6eba63c887:2   #84:9a8e2db3e6:1
#136:2b227d27ad:1  #137:5ca1a85c60:1  #140:89c6926924:1  #170:28a0a1203f:1  #172:2e2eb0f747:1
#189:97c50d4195:2  #225:32df344c52:2  #243:e3d57c5a35:1  #284:bef5a95d45:1  #333:e826f1be94:1
#348:6175362f53:2  #350:7982b65583:5  #351:67eb1503fa:5  #356:c86104570f:1  #357:dc65252dd3:1
#358:d6f59c99a8:1  #365:4e2a118553:1  #366:46b680f986:1  #381:d2cb4ede30:1  #383:44533b203f:1
#388:d6df9cbfc8:1  #399:21c2091503:1  #400:044e988cb9:1  #412:6a34b66d69:1  #413:4e695e860b:1
#432:524dea5d97:1  #434:8d706e4b84:1  #494:89b970ff60:1  #500:59198ca485:1  #504:28176da827:2
#513:73bc2f0e93:1  #514:c5aafccee1:1  #525:74ac1dc2b1:4  #548:bfc17b00a5:3  #567:49a500b9eb:1
#574:2b1aee4185:1  #575:b6e67c7b8c:3  #578:e4092c1b5f:1  #597:2053cf7db7:1  #603:173d56e4a8:5
#613:7b047d8582:1  #615:c9ab7b4ee9:1  #626:2d51055b88:1  #628:0efe5ea789:1  #630:fecbefe5a5:1
#636:095b0822e1:1  #766:7309c3a748:1
```

**Caveat that travels with the list (manual §5):** `#766` also appears as "waterfill rung #766"
(a rate concept) — it earned its slot from a commit that literally said `task #766`, but MAIN
should confirm it is the *task* before archiving. This is the residual of the shared-number-space.

---

## DO-NOT-ARCHIVE guardrail (60 OPEN-class ledger row-IDs — the live task surface)

These `QA##`/`QD##`/`QE##` rows are OPEN/DUE/FOLD/rider/LIVE in the deferral-queue ledger. MAIN
must **not** archive the tasks they track (they are pending/in_progress = KEEP-LIVE by the
charter's rule):

```
QA03 QA04 QA08 QA12 QA13 QA14 QA15 QA18 QA25 QA31 QA33 QA34 QA36 QA38 QA40 QA44 QA47 QA48 QA49
QA50 QA51 QA52 QA53 QA54 QA55 QA58 QA60 QA61 QA65 QA66 QA68 QA69 QA70 QA72 QA73 QA75 QA76 QA77
QA79 QA80 QA81 QA82 QA83 QA84 QA86 QA87 QA88 QD04 QD05 QD06 QD07 QD08 QD09 QD10 QD11 QD12 QD13
QD14 QD15 QE09
```

(QA87/QA88 are this arm's rows — QA87 flips to MEASURED, QA88 to BUILT in this same landing.)

---

## HOW MAIN APPLIES THIS (the archive decision stays harness-side)

1. For each row the harness lists as **completed** and dumps into context: look up its number in
   the **SAFE-TO-ARCHIVE** set above. If present → archive it out of the live list (custody is a
   named commit sha).
2. If the harness-completed row's number is **NOT** in the SAFE set: it is either (a) a
   catalog/equation/PR-collision number MAIN can still archive if the harness marks it done, or
   (b) a genuine NO-CUSTODY-FOUND — MAIN decides using the harness status (the authority I lack).
3. **Never** archive a number in the DO-NOT-ARCHIVE guardrail without checking the ledger status
   column.
4. Re-run `scratchpad/reconstruct_task_custody.py` (strict `task #NNN` mode) after future landings
   to refresh custody; it is deterministic and $0.

**Verdict scope:** this is an INSTANCE-level custody snapshot (one repo state, one grep pass),
not a FAMILY claim about the whole task list. Its value is the guardrail + the 67 verified
custody pointers, not a completeness claim.
