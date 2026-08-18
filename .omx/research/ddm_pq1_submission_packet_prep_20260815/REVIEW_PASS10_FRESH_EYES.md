# Review pass 10 — fresh eyes (reviewer #6)

**Round:** 10 · **Reviewer:** FRESH-EYES Opus arm #6 (no part in any generation,
any fix batch, or any prior review round) · **Run:** 2026-08-18

**Candidate:** `gen3_sz1_composed_split`, archive
`debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a` / 179,930 B
**Canonical receipt:** `pre_submission_compliance.gen3.r5.json`
(sha `6f4f6dc8e3648eb09ec19e2aefb8f28ceea7d4d8e24c9211e16ca21bc25cf741`), 82/86

**VERDICT: 3 FINDINGS. Counter stays `0/5`; reset to `0/5` after the fixes land.**

**Mid-review mutation, disclosed and adjudicated:** MAIN removed the F1
contamination *during* this round (13:27:21Z) and disclosed it with receipts. I
verified the receipts rather than accepting them (§ "MAIN's mid-review
remediation"). **My judgement: the mutation does NOT invalidate the round.** The
scored object never moved — I re-hashed all 34 manifest files *during* the
contaminated window and got 34/34 identical, and the archive sha is unchanged
before and after. Only non-manifest litter appeared and was removed, and both
the addition and the removal are now timestamped and certified. F3 below is a
finding *against one claim inside that remediation receipt*, not against the
remediation, which was correct and well-executed.

The archive bytes, the score, the r5 receipt, the 34-file runtime tree, the
public text, and **both round-9 fixes** are all CLEAN — verified from disk, not
from the prompt. Round 9's F1 fix landed at *more* sites than the finding named,
and its F2 fix cured the identity defect properly.

**F1 was caught by measurement, not by reading.** For about eleven minutes
during this review, the staged submission directory held **33 files that were
not there when r5 was bought** — 15 `.pyc` embedding the packet's own absolute
path, 18 AppleDouble `._*` siblings, in 3 `__pycache__` directories. A
concurrent sibling arm's ad-hoc probe had imported the packet's runtime modules
*in place*. This is simultaneously the round-5-F5 class and the round-1-F2
class, both previously "CLOSED".

**The directory is clean again as I write this** — MAIN removed the contamination
at `13:27:21Z` under a certify-or-block manifest, and the file count is back to
exactly the round-9 baseline. (My first reading attributed the cleanup to the
originating arm; MAIN's disclosure corrected that, and I have re-verified the
correct attribution from the landed receipt.) So
F1 is **not** "stray files are present now." F1 is: **the packet has no guard
against this, and today it demonstrably happened, 25 minutes after the canonical
receipt was bought.** Rounds 5 and 9 each verified "zero `.pyc`" as proof of
cleanliness; that reading is now known to be a *point-in-time sample of an
unstable quantity*. The round-5 cure deleted the files; it never made the
directory unable to re-acquire them. That is the instance-versus-class root
cause this series has named five times, caught this time in the act.

---

## Findings

| # | Class | Finding | Evidence | Severity |
|---|---|---|---|---|
| **F1** | round-5-F5 + round-1-F2 classes — **instance cured, class uncured; measured firing** | **The staged submission directory is a live, writable, importable surface, and a concurrent arm contaminated it post-r5.** Between `13:16:25Z` and `13:27:50Z` it contained 33 undeclared files (15 `.pyc` + 18 AppleDouble `._*`) in 3 `__pycache__` directories under `cpr1/`, `runtime/`, `runtime/entropy/`. Each `.pyc` embedded `co_filename = /Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen3_sz1_composed_split/…` — the exact signature round-5 F5 was raised on. **The state is now clean; the missing guard is the finding.** | **Measured, timed, and causally traced.** All 33 files + 3 dirs shared one mtime `1787058985` = `2026-08-18T13:16:25Z` (r5 = `12:51:30Z`; round-9 fix commit `cea1dbe3ce` = `13:11:14Z`). Cause at source: `.omx/tmp/sa3/probe_identity.py` (mtime `13:16:23Z`, **two seconds earlier**) sets `SZ1 = Path("…/gen3_sz1_composed_split")` at `:19`, then `:34-41` `sys.path.insert(0, str(root))`, `sys.path.insert(0, str(root / "cpr1"))`, `import carrier_codec`, `import runtime.residual_archive`, `from runtime.carrier_repack import materialize_cpr1`. CPython 3.13 (`.venv` = 3.13.12) matches the `cpython-313` tags; ExFAT added one `._*` per new file. **My own counts:** at ~`13:21Z` the directory held **72 files / 34 manifest / 38 outside** (5 declared exclusions + 33 undeclared); at `13:28:46Z` it holds **39 files** = 34 + 5, and `find` for `*.pyc`, `._*`, `__pycache__` returns **0/0/0**. Removal bracketed by the three subdirectory mtimes, all `1787059670` = `13:27:50Z`, and attributed by MAIN's landed manifest `removed_utc = 2026-08-18T13:27:21Z`. | **MEDIUM** |
| **F3** | claim-not-honored-by-the-code, in a **landed custody artifact** | **`gen3_receipts/PYCACHE_CONTAMINATION_REMOVED_20260818.json` states a mechanism that did not occur.** Its `reason` field reads: the `.pyc` "…embed absolute local paths (public-hygiene violation) **and break the pinned runtime-tree sha `0d0fc008d6a37bd5cfa804073e617a8ea30a7c6b6e6c4a1022e2c5d7ce6f9513`**". The public-hygiene half is correct and I confirmed it. **The tree-sha half is false:** these files could not break that sha. | **The tree sha is manifest-derived, not walk-derived.** `scripts/pre_submission_compliance_check.py` computes it in `_runtime_tree_sha_from_manifest` (`:624`) and `_portable_runtime_tree_sha_from_manifest` (`:712`), over a file list whose `basis` is `contest_auth_eval_runtime_dependency_manifest_v1_without_custody_files` — the dependency list recorded by the T4 auth-eval run, not a live directory walk. The 34-entry manifest contains no `.pyc` (`any pyc/pycache in manifest: NONE`). **Direct measurement:** I re-hashed all 34 manifest files *while the 33 contaminant files were present* → **34/34 byte-identical**, so the tree sha was never perturbed. The packet's own history agrees: round-5 F5 described this exact class as "`.pyc` w/ local paths **outside the hashed walk**". Consequently MAIN's restoration receipt returning the pinned sha is consistent with the sha never having moved — it confirms the tree is intact (true, and I verified it) but does not, as claimed, prove the `.pyc` were "the ONLY delta". | **LOW** |
| **F2** | counter-authority completeness | **`ADVERSARIAL_REVIEW_SCAFFOLD.md` has no round-9 row.** Its round table (`:8-17`) ends at round 8. The document is the counter authority and says so at `:100-101` ("For the live counter, read ONLY the header and table of this file"), and at `:31-32` states "Counting rounds resume at **round 9**". A reader of the authority cannot tell that round 9 ran, what it found, or that its fixes landed — the file reads as though round 9 never happened. | **Precedent unambiguous; the omission is one commit deep.** Every prior round's row was added in the *same commit* as its fix batch — `71e6d0a076` row 5, `1a74d8f2d1` row 6, `cfe778f18b` row 7, `91fb329359` row 8 (each verified via `git show <sha> -- ADVERSARIAL_REVIEW_SCAFFOLD.md`). The round-9 batch `cea1dbe3ce` **did edit this same file**, but only to re-point two r4 citations to r5 (`:21`, `:98`); it added no row. `REVIEW_PASS9_FRESH_EYES.md` exists (170 lines, 2 findings) and its fixes verify as landed, so the round is real and unrecorded. The counter *value* `0/5` is correct; the ledger behind it is not. | **MEDIUM** |

### F1 — blast radius, scoped honestly

I checked the reach rather than assuming it, and I am deliberately **not**
inflating this:

- **Archive bytes, member, score: UNAFFECTED.** Re-verified after the event.
- **The 34-file runtime tree hashes: UNAFFECTED, and r5 is NOT invalidated.**
  The receipt's file list is dependency-derived, not a directory walk —
  `submission_runtime.basis =
  contest_auth_eval_runtime_dependency_manifest_v1_without_custody_files`. I
  re-hashed all 34 during the contaminated window: **34/34 byte-identical, 0
  mismatch, 0 missing.** This is also why round-5 F5 correctly called the `.pyc`
  "outside the hashed walk".
- **A receipt re-buy during such a window: at genuine risk.** Round-1 F2 records
  that AppleDouble files inside this submission directory **crashed the
  compliance checker mid-run and left a stale receipt on disk**. That is the
  realistic harm — not shipped bytes, but a corrupted or stale receipt if the
  window overlaps a re-buy. Publication during a window carries the round-5-F5
  harm (internal absolute paths in published files).
- **I did not re-run the checker.** Buying a receipt is MAIN's action, not a
  reviewer's. The reasoning above is read from
  `scripts/pre_submission_compliance_check.py` (recursive branches at `:2216`
  and `:2428` fire only when a scan path is a directory; today
  `public_hygiene.scan_paths` is a single `.md` file) and is labelled as
  reasoning, not measurement.

### F1 — what the class cure is, and what it is not

The files are already gone, so "delete them" is not the fix — that is precisely
the instance move that left the class open after round 5. What is owed is a
guard:

1. **Make the invariant a gate.** "Files in the submission dir outside
   {34-file manifest ∪ 5 declared custody exclusions} == 0" is a one-line query;
   it returned 38 during the window and 5 now. Run it immediately before any
   receipt re-buy and again immediately before publication — the same place the
   receipt-freshness law already lives in `SWAP_PROCEDURE` step 5. A per-round
   reading is not enough, because the quantity moves between rounds.
2. **Remove the write path.** Sibling arms must not import the staged packet in
   place: import from the sealed source tree
   (`/Volumes/APDataStore/pact/ddm_sz1/runtime/tuned`), or set
   `PYTHONDONTWRITEBYTECODE=1` / `sys.dont_write_bytecode = True`. **The sa3
   arm's production driver already does this correctly** —
   `experiments/ddm_sa3_rebase_sz1.py:228,384` set `PYTHONDONTWRITEBYTECODE=1`
   and `:538,594` copy with `shutil.ignore_patterns("__pycache__", "._*")`. It
   was the **ad-hoc probe** in `.omx/tmp/sa3/` that carried neither. The pattern
   to enforce is the one that arm already uses in its mature path.
3. **Keep `COPYFILE_DISABLE=1`** (round-1 F2's cure) and re-run the non-UTF-8
   scan before the next receipt, since `._*` files proved able to return.

**This is not a criticism of the round-9 fixers.** The event came from outside
the packet chain, after their batch, and MAIN removed it within about eleven
minutes under a certify-or-block manifest. It is a criticism of the *guard*,
which is what a class cure is for.

### MAIN's mid-review remediation — verified, and adjudicated

MAIN disclosed, mid-round, that it removed the contamination and asked me to
adjudicate rather than take its word. I verified every claim from disk:

- **`gen3_receipts/PYCACHE_CONTAMINATION_REMOVED_20260818.json`** — exists, 6,633 B, schema `pycache_contamination_removal.v1`, `count = 33`, `removed_utc = 2026-08-18T13:27:21Z`. It records **33 file entries, every one carrying `path` + `bytes` + `sha256`** (verified programmatically: `all have sha256: True`, `all have bytes: True`). Its decomposition is **15 `.pyc` + 18 `._*`**, which matches my own independent census exactly. (MAIN's prose to me said "15 twins + 3 dirs"; the *artifact* is the accurate one — the 18 AppleDouble entries are 15 file-twins plus 3 directory-twins such as `cpr1/.___pycache__`. No discrepancy in the record.) **Nothing was deleted un-recorded** — the certify-or-block rule was honoured.
- **`gen3_receipts/TREE_RESTORE_VERIFY_20260818.json`** — exists, 74,181 B. **86 checks / 82 passed / 4 red**, and the reds are the *identical* adjudicated set. All five tree checks GREEN: `auth_eval_runtime_tree_recorded`, `auth_eval_runtime_tree_expected_match`, `submission_runtime_tree_recorded`, `submission_runtime_tree_matches_auth_eval`, `dispatch_claim_terminal_runtime_tree_sha_bound`. `runtime_file_count = 34`, `runtime_tree_sha256 = 67059c1d…`, portable `= 994f8aaa…` — identical to r5.
- **Is the object I reviewed byte-identical to what r5 verified? YES**, verified by me on both sides of the event: 34/34 manifest files re-hashed identical *during* the window, and archive sha `debb025f…` / 179,930 B identical before and after.

**Adjudication — the mutation does not invalidate round 10.** The mutation never
touched the scored surface: not the archive, not the member, not any of the 34
manifest files, not the receipt, not a document. It added and then removed litter
that sits outside the dependency-derived manifest, and both transitions are now
timestamped and hash-recorded. Every verification in this report was performed
against a stable object, and where timing could matter I have stated it
explicitly. MAIN's conduct here — disclose, certify before deleting, re-run the
checker, hand the adjudication to the reviewer — is the correct handling and I
am recording it as such. **The one thing I do not accept is a single sentence
inside the removal receipt: see F3.**

---

## What I verified, with denominators

Every value below was read from disk. Nothing was taken from the prompt.

### Bytes, grammar, score — CLEAN

- **Archive:** `shasum -a 256` → `debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a`; `stat` → **179,930 B**. Re-verified again after the F1 event: unchanged.
- **Grammar:** `zipfile` walk → **exactly 1 member**, name `p`, `compress_type=0` (stored), `file_size == compress_size == 179,830`, `CRC32 = 3747474564`, member sha `be6db33bce471fe38b3d32cf6b421368721b1ea2ddd3f77b577f2bd27d06b7a8`. All match the prompt.
- **Score re-derived from components** — not from the `final_score` field, which is the rounded `0.16` that CLAUDE.md warns lies. Source `contest_auth_eval.json`, call `fc-01M09EHX5MMTJACMRADQPN9P7Z`, `evidence_grade = contest-CUDA`, `n_samples = 600`, `gpu_model = Tesla T4`, `gpu_t4_match = True`:
  - seg `100 × 0.00029611` = `0.029610999999999998`, Python `==` `0.029611` → **True**
  - pose `sqrt(10 × 6.88e-06)` = `0.008294576541331089` → **True**
  - rate `25 × 179930 / 37545489` = `0.11980800143527229` → **True**
  - **S = `0.15771357797660338`**, `==` claimed → **True**. `canonical_score` agrees; `canonical_score_source = report_8dp_components_plus_exact_archive_bytes`.
- **Inflate budget:** `inflate_elapsed_seconds = 1143.270127967` vs 1,800 s → 1.574× headroom, matching the PR body.

### Reproduction and the determinism claim — CLEAN (pursued hard, not a finding)

I chased a suspected mis-scoping and it did not hold, so I record the negative
result rather than a finding.

`RESULT_pq2_e2e.json` records a determinism repeat only in `.verification` — the
**pre-split** archive `9de0f6db…` / 180,450 B. The `split_verification` block has
**no** determinism field, while the public text (`PR_BODY_DRAFT.md:73-74`,
`REPORT_PUBLIC.txt:49-50`, `README_PUBLIC.md:119-120`) places a byte-identical
determinism claim beside the *final* archive. I resolved it on disk:
`split_stage/archives/fx2_a__tuned/archive.zip` **and** `archive.repeat.zip` both
hash to `debb025f…` at **179,930 B**. The public claim is **true of the shipped
bytes**; only the receipt's field coverage is thinner than the artifacts on disk.
**Not a finding.**

Reproduction otherwise verifies: all 4 pinned inputs `sha256_matches = True`,
3 stages rc=0, `split_verification.sha256_matches = True`, `bytes_match = True`.

### r5 receipt and the RECEIPT-FRESHNESS LAW — CLEAN

- Receipt sha `6f4f6dc8e3648eb09ec19e2aefb8f28ceea7d4d8e24c9211e16ca21bc25cf741` ✓, schema `pre_submission_compliance_check_v1`, **86 checks / 82 pass / 4 red**, per-check field `passed`. Reds are exactly the adjudicated set: `auth_eval_raw_promotion_policy_blockers_absent`, `contest_cpu_auth_eval_exists`, `submission_runtime_has_no_network_install_or_local_paths`, `hosted_archive_manifest_supplied`.
- **Freshness re-derived, not trusted.** I read the scanned-surface list out of the receipt itself (`public_hygiene.scan_paths`, `public_template_placeholders.sources`, `public_evidence_axis_labels.sources`, `report.path`, `archive_manifest.path`, `archive.path`, `submission_dir.required_files`) and added the two inputs round 9's table did not list separately — the dispatch-claims ledger and the auth-eval JSON. **Denominator: 8 named inputs + 34 runtime files.**

  | scanned input | mtime (UTC) | vs r5 `12:51:30Z` |
  |---|---|---|
  | packet `README.md` | 12:33:15Z | −1,095 s ✓ |
  | packet `report.txt` | 09:53:13Z | −10,698 s ✓ |
  | packet `archive_manifest.json` | 09:53:31Z | −10,680 s ✓ |
  | packet `archive.zip` | 03:56:39Z | −32,092 s ✓ |
  | packet `inflate.sh` | 03:56:38Z | −32,092 s ✓ |
  | repo `PR_BODY_DRAFT.md` | 12:32:42Z | −1,129 s ✓ |
  | repo `active_lane_dispatch_claims.md` | 08:18:37Z | −16,373 s ✓ |
  | `contest_auth_eval.json` | 03:48:59Z | −32,552 s ✓ |

  **Zero of 8 post-date r5.** The round-9 claim is **verified, not trusted**: batch `cea1dbe3ce` touched `ADVERSARIAL_REVIEW_SCAFFOLD.md`, `COMPLIANCE_RUNBOOK.md`, `GAP_REPORT.md`, `PACKET_TARGET.json`, `REVIEW_PASS9_FRESH_EYES.md` — **none is a scanned surface**. r5 stands.
  *(F1's transient files post-dated r5 but sat outside the dependency-derived manifest, so no tree hash moved — see the F1 scoping note.)*

### Runtime tree — CLEAN

- `runtime_file_count = 34`; I re-hashed **all 34** from `submission_runtime.files`: **34/34 byte-identical, 0 mismatch, 0 missing** (performed during the contaminated window, which is exactly why it proves the contamination did not reach the manifest).
- `runtime_tree_sha256` = `67059c1db9ded5d45904d3018d8d1612e4d4e24f5bd4ba3fc0c36c92885a6043` (submission) ✓ · `portable_runtime_tree_sha256_without_custody_files` = `994f8aaab28ec1ffbaeedd1075b7de73a1ca411773edf5112efe309f64230b35` ✓ · auth-eval side `0d0fc008d6a37bd5cfa804073e617a8ea30a7c6b6e6c4a1022e2c5d7ce6f9513` ✓. All three match the prompt and their declared roles.
- `external_dependency_roots = []`, `scorer_import_hits = []`, `disallowed_runtime_imports = []`. Exactly **one** `forbidden_side_effect_hit`: `inflate.sh:27`, the adjudicated Brotli pinned-wheel bootstrap. Runtime imports resolve to `brotli`, `numpy`, `torch` + stdlib + local modules — **no `constriction`** (round-5 F3 cure holding).
- `inflate.sh` = `e1b3df4d…62` ✓, `inflate.py` = `5c5baf88…6d` ✓ vs `PACKET_TARGET.json:41-42`.
- **Final directory census (13:28:46Z):** 39 files = 34 manifest + 5 declared custody exclusions (`README.md`, `report.txt`, `archive_manifest.json`, `BORROWED_SUBSTRATE_ACCOUNTING.md`, `archive.zip`). Zero surprise files. Matches round 9's baseline exactly.

### Class 1 — receipt citations (the round-9 F1 class) — CLEAN, fix over-delivered

Population query `grep -rn "pre_submission_compliance"` across the prep dir and
the packet dir. **Denominator: 34 hits total; 16 in live governance documents**
(the other 18 sit inside `REVIEW_PASS5-9`, correct as review history).
Packet-side documents cite **no** receipt at all.

All 16 adjudicated with one consistent test — *does this line assert a superseded
receipt is the current one, without an adjacent supersession label?*

- **6 name r5 as canonical/current** ✓ — `ADVERSARIAL_REVIEW_SCAFFOLD.md:32`, `COMPLIANCE_RUNBOOK.md:12`, `COMPLIANCE_RUNBOOK.md:160`, `GAP_REPORT.md:6`, `SWAP_PROCEDURE.md:97`, `GAP_REPORT.json:7`.
- **6 are labelled history or explicitly superseded** ✓ — `ADVERSARIAL_REVIEW_SCAFFOLD.md:14` (round-5 table row); `COMPLIANCE_RUNBOOK.md:34` and `:104` (gen-0 `final.json`, under the `:6` HISTORICAL heading); `:93` (r2, the gen-3 first run, corrected in place by the r3/r4/r5 blocks below it); `:101` (the r3 correction block); `:148` (r4, reading "superseded by r5").
- **4 are not receipt citations** — the script name at `:17`, and command/check-name strings at `GAP_REPORT.json:22, :38, :41`.

**Zero stale live citations.** Round-9 F1 is fully cured, and the fixer's claim to
have swept five sites rather than the three named is confirmed by the diff:
`COMPLIANCE_RUNBOOK.md:12` (+ a new r5 sha line), `:48-49` ("per the r5 paragraph
below"), and `GAP_REPORT.md:6` (+ a new superseded-r4 parenthetical).

### Class 2 — path-valued fields (round-7-F4 / round-8-F1 class) — CLEAN

Machine sweep over **8 JSON custody documents** (5 repo-side: `ARCHIVE_MANIFEST`,
`CPU_AXIS_SEALED_FIRE_ORDER`, `GAP_REPORT`, `PACKET_TARGET`,
`RECIPE_sz1_composed`; 3 packet-side: `GENERATION_RECEIPT`, `RECEIVER_PARSEBACK`,
`archive_manifest`). **476 string values walked, 118 path-like values extracted,
107 resolve.**

The 11 non-resolving split cleanly:

- **8 are regex false positives** on prose fragments, hand-checked: `debb025f…/179`, `35ac2b9b…/181`, `9de0f6db…/180`, `operator/MAIN`, `report/label`, `sha/bytes`, `experiments/ddm_sz1_*` (a `git log` glob inside a note), and `.reproduction_entry_point`'s inline sha text.
- **3 are real and correctly absent** — `CPU_AXIS_SEALED_FIRE_ORDER.json` `command_argv[17]`, `close.poller_argv_template[5]`, `close.recovery_argv[3]`, all `experiments/results/ddm_pq2_rr4_exact_contest_cpu_20260817`. That file is `status: SUPERSEDED_BY_EXECUTED_GEN3_FIRE` / `disposition: SUPERSEDED`; the order **never fired**, so these are output directories a run would have created, not custody pointers. Same adjudication as round 9, reached independently.

Spot-checks of the two fields prior rounds fixed: `PACKET_TARGET.json:16`
`generation_manifest` → `generations/gen3_receipts/GENERATION_MANIFEST.json`
**exists** (round-7 F4 holding). `:36` `posterior_anchor` →
`.omx/state/continual_learning_posterior.json accepted_anchor_history[191]`,
resolved by hand: the array has **192** entries and index 191 carries
`archive_sha256 = debb025f…`, `archive_bytes = 179930`,
`score_value = 0.15771357797660338`, `[contest-CUDA]`, `linux_x86_64_t4`.

### Class 3 — identity-vs-existence (round-9 F2's cure) — CLEAN, cure is correct

`PACKET_TARGET.json:111` now reads as follows, and every claim verifies on disk:

- **Primary (gen-2, matching the block it lives in):** `/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/CUSTODY_SUPERSEDED.json` — measured sha `8d65ac3f5b270a90183c1c3c5c5467a909900d1d27c053c10843f9a874d0b1fa` ✓, **2,912 B** ✓, schema `pact.custody_supersession.v1` ✓, `archive_actually_in_this_directory` = `35ac2b9b…` / **181,161 B** ✓.
- **Explicitly distinguished (gen-3):** `generations/gen3_receipts/CUSTODY_SUPERSEDED.json` — sha `f1f3eebff5ba90c84689ab016ae7de2cf7285a50f9f8b7350db2b243fb2830d8` ✓, **1,535 B** ✓, schema `pq1_custody_superseded.v1` ✓, `real_candidate.archive_sha256` = `debb025f…` / **179,930 B** ✓.
- Both exist, both shas match, the two records are correctly asserted to be different files, and the field closes with **"verify by sha, never by existence"** — the class cure stated inside the artifact, which is the right place for it.

### Class 4 — hand-typed values — CLEAN

**14 `*_utc` fields across 15 custody JSONs** enumerated. The two round-8
replaced are exact against git, re-derived with `TZ=UTC git show -s`:

- `GAP_REPORT.json` `superseded_utc = 2026-08-18T12:34:45Z` ⇄ `cfe778f18b` = **12:34:45Z** ✓
- `CPU_AXIS_SEALED_FIRE_ORDER.json` `superseded_utc = 2026-08-18T05:00:51Z` ⇄ `a3ab650ce9` = **05:00:51Z** ✓

No timestamp is in the future. `PACKET_TARGET.json`
`reproduction.verified_at_utc = 05:12:00Z` sits ~50 s after the e2e receipt's
`built_at_utc = 2026-08-18T05:11:10.312277+00:00` — consistent with a wall-clock
read at recording time; **I am not raising it**, matching round 9, as nothing
shows it invented. Machine-derived values cross-check:
`GENERATION_MANIFEST.authority.observed_at_utc` and the posterior anchor's
`observed_at_utc` are the identical `2026-08-18T03:49:41.073665+00:00`.

### Class 5 — prose vs settled adjudications — CLEAN

**Denominator: 10 documents read in full, 1,199 lines** (`CONTRIBUTION_ETIQUETTE`
47, `COMPLIANCE_RUNBOOK` 169, `SWAP_PROCEDURE` 108, `PR_BODY_DRAFT` 301,
`README_PUBLIC` 159, `REPORT_PUBLIC` 64, `GAP_REPORT.md` 43, `GENERATION_LOG` 85,
packet `README.md` 159, packet `report.txt` 64). **Zero contradictions** against
all five settled adjudications. Six borderline sentences were individually
adjudicated as banner-covered or conditional-rule, and are explicitly **not**
findings:

- **Compression source** — `PR_BODY_DRAFT.md:123` "Yes, and it is offered for merge" stands, backed by `COMPLIANCE_RUNBOOK.md:72-81`. The two "answer no" strings are a conditional rule (`COMPLIANCE_RUNBOOK.md:67-70`, discharged two lines later at `:72`) and a banner-covered line (`GAP_REPORT.md:42`, under the `:3-14` SUPERSEDED banner that names this exact item as ADJUDICATED SATISFIED).
- **CPU axis** — **no CPU score number appears anywhere** in the 10 files. Every live surface states MEASURED-INFEASIBLE with 3,422.7 s vs 1,800 s. The three "pending"-adjacent lines are `SWAP_PROCEDURE.md:26-29` (step 3 of a generic swap procedure, adjudicated at `:89-95`), `COMPLIANCE_RUNBOOK.md:37-40` (under the `:6` HISTORICAL gen-0 heading), and `GAP_REPORT.md:26-28` (under the `:3-14` banner).
- **The 4 reds** — typed terminally at `COMPLIANCE_RUNBOOK.md:129-145`, each with explicit non-convertibility language. Hosting language correctly treats red 5 as convertible only by operator authorization.
- **Submission name** — `sz1_composed_reencode` live at `PR_BODY_DRAFT.md:1`, `README_PUBLIC.md:1,3`, packet `README.md:1,3`, `COMPLIANCE_RUNBOOK.md:163`. The old form appears **only** in `REVIEW_PASS7/8/9`, which is review history. Round-7 F5 cure holding.
- **Reproduction** — no `PENDING_REBIND`, no "re-bind in progress". `Status: VERIFIED` in 6 live locations.
- **Stale numbers** — 57 lines carrying a byte literal or a `0.1xx` decimal checked; every non-current figure is labelled (183,502 under the HISTORICAL heading; 182,759 inside the custody note that says "That is not the archive in this packet."; 181,161 / `0.15853325034789678` labelled "Vs generation 2"; 180,450 as the pre-split intermediate; PR130/133/135/138 figures attributed at `PR_BODY_DRAFT.md:255-269`).

### Public hygiene and document sync — CLEAN

- **5 public surfaces, 747 lines** scanned case-insensitively for `/Volumes/`, `/Users/`, `APDataStore`, `VertigoDataTier`, `modal.com`, `api_key`, `Claude`, `Anthropic`, `fc-01`, `100.x` addresses: **zero hits on all five.** A supplementary sweep (`vast.ai`, `lightning`, `tailscale`, `ssh`, `/home/`, bearer/token/secret, `codex`, `subagent`) returned one hit — `PR_BODY_DRAFT.md:259` "Shreyan Mohanty (`codexblack`)", the real GitHub handle of the PR #135 author. Correct attribution, not a leak.
- **Repo ⇄ packet sync:** `README_PUBLIC.md` ≡ packet `README.md` (`ea08053dd4fcc584…`) and `REPORT_PUBLIC.txt` ≡ packet `report.txt` (`6c41f7faa5a951d9…`) — byte-identical. No split-brain.

### Observation (not a finding)

An untracked `.omx/state/` tree has appeared **inside** the prep directory
(`…/ddm_pq1_submission_packet_prep_20260815/.omx/state/{triality_drift_marker.json,
magnitude_dismissal_marker.json}`) — hook markers written by tools whose working
directory was the prep dir. Untracked, outside the packet, unable to reach the
submission or the PR. Not a finding; repo litter worth sweeping when convenient.

---

## Counter recommendation

**Counter: `0/5`.** Three findings, so round 10 is not a clean pass. Per the
scaffold, the counter resets to `0/5` after the fixes land.

I record explicitly, because MAIN offered to burn the round over it: **the
mid-review mutation is not my reason for withholding a clean pass.** The scored
object was stable throughout and I verified that on both sides of the event. The
counter stays at `0/5` because of F1, F2 and F3 on their own merits.

Recommended, at class level:

1. **F1** — the files are already gone, so do **not** treat "delete them" as the
   fix; that is the instance move that left this class open after round 5.
   Install the guard: (a) the outside-the-manifest invariant as a gate run
   immediately before any receipt re-buy **and** immediately before publication,
   beside the receipt-freshness law in `SWAP_PROCEDURE` step 5; (b) sibling arms
   import the sealed source tree rather than the staged packet, or set
   `PYTHONDONTWRITEBYTECODE=1` — the pattern `experiments/ddm_sa3_rebase_sz1.py`
   already uses correctly at `:228,384,538,594`; (c) keep `COPYFILE_DISABLE=1`
   and re-run the non-UTF-8 scan before the next receipt.
   **r5 does not need re-buying** — the transient files sat outside the
   dependency-derived manifest and all 34 tree files re-hash identical.
2. **F2** — add the round-9 row to the scaffold table, and round 10's when these
   fixes land. The durable cure is to make "append the round row" the same step
   as "write the review file", so the ledger cannot lag the reviews again.
3. **F3** — correct the `reason` field in
   `PYCACHE_CONTAMINATION_REMOVED_20260818.json`: keep the public-hygiene
   justification (it is true and sufficient on its own) and drop or restate the
   "break the pinned runtime-tree sha" clause, since the sha is manifest-derived
   and was measurably never perturbed. The honest replacement is the framing the
   packet already used at round-5 F5 — *outside the hashed walk, inside the
   published directory* — which is exactly why the guard in item 1 is the thing
   that matters. Removal itself needs no re-doing.
4. **Receipt freshness for this batch:** these fixes touch the scaffold,
   procedure text, and a `gen3_receipts/` JSON — none is a checker-scanned
   surface. If a fix batch ends up editing packet `README.md`, `report.txt`,
   `archive_manifest.json`, or `PR_BODY_DRAFT.md`, re-run the checker in the same
   batch per the standing law. **r5 remains valid** and needs no re-buy for any
   of F1/F2/F3.

**Nothing in this round licenses a push, a hosting action, or a PR opening.** The
refusal condition in `SWAP_PROCEDURE.md:80-81` is unchanged and binding: no such
action without explicit operator authorization.
