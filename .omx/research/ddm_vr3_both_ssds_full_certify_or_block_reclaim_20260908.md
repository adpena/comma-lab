# DDM VR3 — certified raw reclaim freed 61.395786 GiB on Vertigo

Date completed: 2026-09-09  
Owner: `ddm_vr3`  
Authority: filesystem custody, scorer-free  
Disposition: **TARGET MET**  
Tokens: `[no-triality] [p0-ledger-ok]`

## Result first

Eighteen exact, rebuildable `0.raw` files were deleted only after their full SHA-256,
retained archive, retained runtime manifest, decode-identity receipt, repository-reference scan,
and immediate `lsof +D` check all passed. The executor stopped when both the measured filesystem
gain and this ledger's attributable logical deletion count exceeded 60 GiB.

| quantity | measured result `[filesystem custody / scorer-free]` |
|---|---:|
| Vertigo available at plan baseline | 8,023,265,280 B (7.472248 GiB) |
| Vertigo available at executor completion | 73,946,488,832 B (68.868034 GiB) |
| measured available-byte gain | **65,923,223,552 B (61.395786 GiB)** |
| attributable logical bytes deleted | **65,923,372,800 B (61.395925 GiB)** |
| charter target | 64,424,509,440 B (60 GiB) |
| measured margin over target | **1,498,714,112 B (1.395786 GiB)** |
| deleted / compressed / moved | **18 / 0 / 0** |
| retained census rows | **141/159**, all identity-current after apply |

The 149,248 B difference between logical deleted bytes and measured available-byte gain is bounded
filesystem activity while the live frontier arm continued writing. The conclusion uses the smaller
measured `df` gain, and the independent logical-attribution gate also passed. A later audit sample
showed 73,946,513,408 B available; the executor-completion denominator above is the frozen result.

At arm start Vertigo had 18,581,644 KiB available. It fell to 7,835,216 KiB before plan/apply while
the protected live arm wrote. APDataStore had 51,012,992 KiB available at arm start and 39,981,184
KiB at the post-apply audit. No AP or local-disk payload was moved, compressed, or deleted.

`score_claim=false`. No scorer, evaluator, Modal job, training job, archive builder, or submission
path ran. `upstream/` was read-only. This arm did not move a frontier pointer. MAIN independently
advanced the global pointer during this storage run; that external result is not attributed here.

## Exact census and denominator

The bounded census covered every regular file at least 1 GiB under
`/Volumes/VertigoDataTier/pact/` whose basename ends in `.raw` or whose path has an `inflated`,
`parseback`, or `renders` component.

| census fact | measured value |
|---|---:|
| candidate rows / unique paths | 159 / 159 |
| bytes | 582,323,126,400 B (542.330673 GiB) |
| stable current-device SHA rows | 159/159 |
| exact row bytes | 3,662,409,600 B each |
| `lsof +D` plans completed / empty | 159/159 / 159 |
| zero-reference / referenced rows | 62 / 97 |
| scan errors | 0 |

The two sj1 raws appear in the denominator at ranks 87 and 88. They were not silently excluded:
both carry `LIVE_POINTER_TREE_PROTECTED`, remained present, and retained byte/mtime/device/inode
identity after apply.

Durable census evidence:

- inventory: `/Volumes/VertigoDataTier/pact/ddm_vr3_reclaim_20260908/census_run/inventory.json`,
  SHA-256 `ab2eff20af82209af42e92bae71e2ee0f31a4b2d4e1e7acc606d233cb686cd3c`;
- resumable hash ledger: `/Volumes/VertigoDataTier/pact/ddm_vr3_reclaim_20260908/census_run/hash_progress.jsonl`,
  180 attempts resolving to 159 latest unique paths, SHA-256
  `dd3e3345dbee974ccac0b87da10f20eca50f23a23307e8ca33c452849982d585`;
- first detached receipt: `.omx/tmp/codex_runs/ddm_vr3_census.done.done`, `rc=120` after
  48,836.532 s of host sleep/inactivity, with 21 stable rows already durable, SHA-256
  `7b90e3c472e26a53835caabae555d5f2e6024e0d1849d98b347abe7cc34028d0`;
- completion resume: `.omx/tmp/codex_runs/ddm_vr3_census_resume1.done.done`, `rc=0` after
  14,985.129 s, SHA-256 `fad77e377d8d68edf13f31336477e0642c99f93619cb68f0c4cc1b665587be20`;
- current-device repair: `.omx/tmp/codex_runs/ddm_vr3_census_resume2.done.done`, `rc=0` after
  2,150.474 s, SHA-256 `9c8dd4e6190cf10fa624cea3348db3994987631fe9afd86ee68e7184f9d46f47`.

The full plan correctly refused before writing a ledger when a remount changed 21 rows from device
`16777242` to `16777240`. Bytes, mtime, inode, and SHA ultimately agreed on all 21, but the old
device-bound receipts were not reused. Resume 2 re-read every affected payload and left 159/159
latest rows on device `16777240` with no identity issue.

## Certification and applied rows

The pre-apply ledger had 159 rows and SHA-256
`ea4c00ce8b17aadf839572fb317b5802bcdeb075be04d76482228294a40fc002`.
Exactly 20 rows were `DELETABLE`, totaling 73,248,192,000 B (68.217695 GiB): 16 AP1 terminal
advisory raws and four JF2 terminal advisory raws. Every one carried:

- a current full raw SHA-256 equal to the exact `0.raw` field in its decode receipt;
- a retained, current archive SHA-256;
- a retained 38-file runtime manifest with exact current path-set, byte, and file-SHA equality plus
  the recorded runtime tree/content/file digests;
- the exact decode argv, inflate script SHA-256, and upstream snapshot SHA-256;
- zero hits for current and recorded pre-relocation path aliases in the full `.omx`, code, docs,
  config, report, submission, and live-sj1 reference scopes;
- an empty plan-time `lsof +D` result.

The 18 applied rows were the 16 AP1 tags `carrier_l1`, `carrier_l1_fixed_coder`, `carrier_l2`,
`carrier_l2_fixed_coder`, `carrier_l3`, `carrier_l3_fixed_coder`, `control_r2`, `hpac_l1`, `hpac_l2`,
`hpac_l3`, `residual_l1`, `residual_l2`, `residual_l3`, `semantic_l1_r2`, `semantic_l2`, and
`semantic_l3`, followed by JF2 `k002500_r2` and `k040000_r2`. JF2 `k060000_r2` and `null_r2`
remained present as `BLOCKED:TARGET_MET_BEFORE_ROW`.

Before each unlink, the executor rechecked exact stat identity, the fixed AP1/JF2 allowlist, every
reproducer component, repository references, the full raw SHA, and `lsof +D`. It fsynced a
`PRE_DELETE` journal row, unlinked only that raw, fsynced the parent, verified absence, fsynced a
`DELETED` row, and atomically updated the repo ledger. All 18 immediate blocker lists were empty and
all 18 full raw rehashes matched.

Post-apply, all 18 deleted paths were absent; all 141 retained census rows matched their recorded
byte/mtime/device/inode identity; both surplus certified raws remained present; and all 20 retained
archive/runtime/decode chains revalidated. No compatibility symlink was installed because each
deleted row had zero current or historical consumer-reference hits.

Final custody artifacts:

- machine-readable ledger: `.omx/research/ddm_vr3_reclaim_20260908.jsonl`, 159 rows, final SHA-256
  `9cc82907d690cfd8d5fb61a7019427b21c3c04655ccfa12a1f45ea6fcaa029e1`;
- apply journal: `/Volumes/VertigoDataTier/pact/ddm_vr3_reclaim_20260908/apply_journal.jsonl`,
  38 rows (`APPLY_START=1`, `PRE_DELETE=18`, `DELETED=18`, `APPLY_COMPLETE=1`), SHA-256
  `cd92cf3f98ee8231b7207ecd2251569d33d65f3f902ae7a0d35926d6d75da5b5`;
- launch manifest: `/Volumes/VertigoDataTier/pact/ddm_vr3_reclaim_20260908/apply_run/launch_manifest.json`,
  SHA-256 `ae3335d16ab90513d53e164d60e9414d764ee175905784792587266121c848a8`;
- done receipt: `.omx/tmp/codex_runs/ddm_vr3_apply.done.done`, `rc=0`, SHA-256
  `1b39c5b7286a38f407a2ae9ce09beea9b51474685bae5240400f007131c22923`.

## Blocked rows and non-delete routes

The 159-row final denominator is exhaustive at the charter boundary:

| final verdict class | rows | retained state |
|---|---:|---|
| `DELETED` | 18 | exact raw absent; reproducer retained |
| `BLOCKED:TARGET_MET_BEFORE_ROW` | 2 | exact raw retained and identity-current |
| no full certificate plus repository reference | 95 | retained |
| no full certificate, no observed repository reference | 26 | retained |
| submission-tree protection plus no full certificate | 16 | retained |
| live-pointer protection plus no full certificate and references | 2 | retained |

Reason counts overlap where one row carries multiple fail-closed reasons: 139 lacked a complete
per-file certificate in this batch, 97 had repository references, 16 were under `submissions`, and
two were under live sj1. The target was met, so the charter's conditional top-10 unblock table was
not activated.

Compression applied to 0 rows. Verified-at-source:
`.omx/research/ddm_sr3_ap_certify_compress_reclaim_20260826.md:34-44` defers cold-store namespaces
without a complete reconstruction contract, and
`experiments/ddm_sr3_ap_certify_compress_reclaim.py:40-46` binds the existing compressor to
APDataStore, protects two named trees, and identifies `cold_store`/`vertigo_coldstore` as custody
prefixes. No selected Vertigo tree had its own keep-uncompressed protection lift, so compression
was independently `BLOCKED:SR3_CUSTODY_NAMESPACE_OR_NO_PER_TREE_KEEP_UNCOMPRESSED_PROTECTION_LIFT`.

Move applied to 0 rows. Verified-at-source: the binding 10 GiB AP floor appears in
`.omx/research/ddm_vr2_vertigo_reclaim_round2_20260831.md:10,55,129-133`. At the post-apply audit,
AP's 38.129028 GiB available left only 28.129028 GiB above that floor, below the 60 GiB target; the
certified deletion path met the target without ExFAT tree custody risk. The exact raw size was
re-measured as 3,662,409,600 B for all 159 current census rows rather than inherited from prose.

## RECALL EVIDENCE

Sources searched before adjudication:

- `tools/vertigo_certify_move.py --help`, `tools/local_disk_reclaim.py --help`, vr1, the 142-row vr2
  memo/ledger, and SR3's memo/implementation;
- full-content searches across `.omx/research/`, arm receipts, `CANONICAL_RESEARCH_INDEX*`,
  `sub015_DAG_*`, `main_hot_state.md`, lane/task registries, and the canonical task ledger using
  `storage|compress|reclaim|content-address|dedup|zstd`,
  `ddm_ap1|ddm_jf2|scorer|advisory|MOVED|VERIFIED`, and exact candidate paths;
- `.venv/bin/python tools/list_canonical_equations.py --json`: 476 current equations were examined
  for a filesystem-custody law.

Beyond the charter seeds, DK2 supplied source/destination manifest equality for the JF2 scorer tree
at `.omx/research/ddm_dk2_disk_reclaim_certs_20260904.jsonl:21,29,32`; AP1/JF2 terminal receipts
supplied exact archive/runtime/decode chains for 20 rows; and the current live board required
protecting the whole sj1 component rather than only the two named subpaths. No canonical equation in
the searched 476-row registry superseded manifest + archive + runtime + round-trip identity for this
storage operation. These findings changed the plan from a generic retired-tree sweep to a fixed
20-path AP1/JF2 allowlist, kept all other rows blocked, and made current-device identity a hard gate
after the remount.

# FORMALIZATION_PENDING:storage custody — no measured score law

## Verification

- `ruff check` passed for the certifier and focused tests.
- `pytest -q experiments/tests/test_ddm_vr3_certified_raw_reclaim.py`: **6 passed**.
- Two genuine `review_tracker.py scan` + `mark-file` passes covered all 22 certifier entities and all
  seven test-file entities after the final edit.
- The reference-scan regression proves this arm's observation transcript is not mistaken for a
  downstream consumer while a substantive `.omx/research` reference still blocks.
- Catalog #344 strict passed after memo finalization.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — disposition: certify the 26 currently unreferenced but
  certificate-incomplete rows in a new successor ledger; owner: next Vertigo storage custodian;
  consumer store: `.omx/research/ddm_vr3_reclaim_20260908.jsonl` plus
  `/Volumes/VertigoDataTier/pact/ddm_vr3_reclaim_20260908/`; fire trigger: Vertigo falls below 50 GiB
  available again, then refresh device/inode/mtime/full SHA, owner/liveness references, retained
  archive/runtime digests, and exact decode identity before admitting any row.

## LIVE-HYPOTHESES

- Some of the 26 unreferenced, certificate-incomplete rows may form the next certified deletion
  batch. This is plausible because they are the same 3,662,409,600 B decoded-raw shape and have no
  current repository path hit, but it remains untested whether each has an intact retained archive,
  runtime tree, and exact decode receipt.
- The two surplus certified JF2 raws can supply 6.821769 GiB of rapid future headroom. This is
  plausible because their full certificate chains passed and stayed current after apply, but they
  must still receive fresh stat, raw-SHA, reference, and `lsof` checks when a new fire trigger occurs.

## DEAD-ENDS

- Reusing the first 21 hashes across a volume remount is closed: device identity changed, so a full
  current-device rehash was required even though bytes, mtime, inode, and eventual SHA agreed.
- Moving enough bulk to AP is closed at the measured denominator: only 28.129028 GiB remained above
  the verified 10 GiB floor, and ExFAT is not a safe runtime-tree destination.
- Local-disk cleanup is closed for this objective because it cannot increase Vertigo free space.
- SR3 compression of these cold-store custody namespaces is closed without a per-tree protection
  lift and complete reconstruction contract.
- Any deletion outside the 20-path allowlist is closed for this run: 139 rows lacked a complete
  certificate, 97 also had repository references, 16 were submission paths, and two were protected
  live sj1 raws. This is an instance-scoped refusal, not a claim that all 139 can never be certified.
- Path/size digests, hand `rm`/`mv`, citation rewrites, archive/runtime mutation, and sj1 mutation
  remain forbidden custody shortcuts.

sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]
