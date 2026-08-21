# Review pass 9 — fresh eyes (reviewer #5)

**Round:** 9 · **Reviewer:** FRESH-EYES Opus arm #5 (no part in any generation,
any fix batch, or any prior review round) · **Run:** 2026-08-18

**Candidate:** `gen3_sz1_composed_split`, archive
`debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a` / 179,930 B
**Canonical receipt:** `pre_submission_compliance.gen3.r5.json`
(sha `6f4f6dc8e3648eb09ec19e2aefb8f28ceea7d4d8e24c9211e16ca21bc25cf741`), 82/86

**VERDICT: 2 FINDINGS. Counter stays `0/5`; reset to `0/5` after the fixes land.**

The bytes, the score, the receipt, the runtime tree, the public text, and the
round-8 fix batch are all **CLEAN** — verified independently from disk, not
from the prompt. Both findings are the *same recurring root cause* rounds 6, 7
and 8 each named: **a fix lands on the instance the finding cited, not on the
class it described.** Round 8 re-bought the receipt as r5 and re-pointed four
documents; two more documents that cite the receipt were not in that batch.

---

## Findings

| # | Class | Finding | Evidence | Severity |
|---|---|---|---|---|
| **F1** | round-8-F2 class, un-swept population | **Two live documents still name the SUPERSEDED r4 receipt as the current/terminal receipt for the live candidate.** `COMPLIANCE_RUNBOOK.md:10-12` reads "The ACTIVE generation-3 values: archive `debb025f…`/179,930 B on APDataStore, submission tree `67059c1d…`, portable content tree `994f8aaa…`, **receipt r4**" — while the *same document* at `:145` says "**r4** … **superseded by r5**" and at `:157` says "**r5** … the **CANONICAL** terminal receipt". `GAP_REPORT.md:5-7` reads "The live candidate is generation-3 sz1 composed (`debb025f…`/179,930 B) with **terminal compliance receipt** `gen3_receipts/pre_submission_compliance.gen3.r4.json`". Secondary instance: `COMPLIANCE_RUNBOOK.md:46-47` routes the reader to "the **r4** paragraph below" for the gen-3 tree values. | **Load-bearing, not cosmetic:** r4 is exactly the receipt round-8 F2 proved stale. I re-opened it — `pre_submission_compliance.gen3.r4.json` → `post_deadline_submission_policy.statement_preview` = `"# submission name: sz1 composed re-encode"`, the pre-fix name that round-7 F5 ruled non-compliant. A verifier following either pointer lands on a receipt that contradicts the settled submission name `sz1_composed_reencode`. Git proves the split: `GAP_REPORT.md` last touched `1a74d8f2d1` (round 6, 07:16:01-05:00) and was in **neither** the round-7 (`cfe778f18b`) nor round-8 (`91fb329359`) batch, while its sibling `GAP_REPORT.json:7` **was** re-pointed to r5. `COMPLIANCE_RUNBOOK.md` **was** in the round-8 batch — the r5 paragraph was appended at `:157` and the document's own header at `:12` was left behind. | **MEDIUM** |
| **F2** | round-8-F1 class, second-order | **`PACKET_TARGET.json:111` `supersession_record` resolves to the wrong generation's record and asserts a file identity that does not hold.** The field lives inside `generation_2_superseded.custody_note` — whose `explanation` (`:110`) is a **generation-2** statement ("describe a 182,759 B archive, not the **181,161 B** archive they sit beside") — but its primary pointer is `generations/gen3_receipts/CUSTODY_SUPERSEDED.json`, a **generation-3** record. The parenthetical then claims the record is "**also at** the gen-2 store root `/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/CUSTODY_SUPERSEDED.json`". | Both files exist, and they are **not** the same record: gen3_receipts copy = sha `f1f3eebff5ba90c8…`, 1,535 B, schema `pq1_custody_superseded.v1`, `real_candidate.archive_sha256` = `debb025f…` / **179,930 B**; gen-2 store-root copy = sha `8d65ac3f5b270a90…`, 2,912 B, schema `pact.custody_supersession.v1`, `archive_actually_in_this_directory` = `35ac2b9b…` / **181,161 B**. The gen-2 block's corroborating record is the *parenthetical* one, not the primary. The round-8 fix note calls the target a "gen3_receipts **copy** (both copies exist; sha-checked)" — the existence check passed, the identity claim did not. | **LOW** |

Neither finding touches the archive, the score, the runtime tree, or any
public-facing shipped text. Both are custody-pointer precision defects on
repo-side documents.

### Same class, checked and adjudicated OK (not findings)

Applying one consistent test — *does the sentence assert r4 is the current
receipt without an adjacent supersession label?* — five of the eight r4
citation sites pass:

- `ADVERSARIAL_REVIEW_SCAFFOLD.md:21` — "(receipt r4 after the pass-5 fixes)"
  is corrected inside the same paragraph at `:31-34` ("the **r5 receipt** …
  r4 is superseded"). Self-labelling; OK. (A tidy-up is optional, not owed.)
- `ADVERSARIAL_REVIEW_SCAFFOLD.md:98` — inside the section banner-labelled
  `HISTORICAL` at `:89` with "read ONLY the header and table" at `:100-101`.
- `ADVERSARIAL_REVIEW_SCAFFOLD.md:14, 15, 17` — round-history table rows
  recording what each past round reviewed against. Correct as history.
- `COMPLIANCE_RUNBOOK.md:145-162` — explicitly "superseded by r5", and `:162`
  correctly records that r5's expected tree sha was *derived from* r4.
- `SWAP_PROCEDURE.md:97-99` — cites r5 as CANONICAL, r3/r4 as superseded.

---

## What I verified, with denominators

Every value below was read from disk. Nothing was taken from the prompt.

### Bytes and score — CLEAN

- **Archive:** `shasum -a 256` → `debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a`; `stat` → **179,930 B**. Both match.
- **Grammar:** `zipfile` walk → **exactly 1 member**, name `p`, `compress_type=0` (stored), `file_size = compress_size = 179,830`, `CRC32 = 3747474564`, member sha `be6db33bce471fe38b3d32cf6b421368721b1ea2ddd3f77b577f2bd27d06b7a8`. Local header agrees with the central directory on all six fields; `unsafe_reason = null`.
- **Score re-derived from components, independently, to all 17 digits:**
  - seg `100 × 0.00029611` = `0.029611`
  - pose `sqrt(10 × 6.88e-06)` = `0.008294576541331089`
  - rate `25 × 179930 / 37545489` = `0.11980800143527229`
  - **S = `0.15771357797660338`** — Python `==` equality against the claimed value on all three components **and** the total. Source: `contest_auth_eval.json`, call `fc-01M09EHX5MMTJACMRADQPN9P7Z`, `evidence_grade = contest-CUDA`, `n_samples = 600`, `gpu_model = Tesla T4`, `gpu_t4_match = True`.
- **Byte lineage** (4 generations): 183,502 → 182,759 (−743) → 181,161 (−1,598) → 179,930 (−1,231). Reconciles with the PR body's "2,829 bytes … (1,598 + 711 … 520)": 711 + 520 = 1,231 ✓, 1,598 + 711 + 520 = 2,829 ✓.
- **Generation-2 delta re-derived:** gen-2 S = `0.15853325034789678` (recomputed exactly from the same seg/pose plus 181,161 B), Δ = `−0.0008196723712934` — matches the report's "delta S -0.00081967237" ✓.

### r5 receipt and the RECEIPT-FRESHNESS LAW — CLEAN

- Receipt sha `6f4f6dc8e3648eb09ec19e2aefb8f28ceea7d4d8e24c9211e16ca21bc25cf741` ✓ (prompt prefix matches).
- Parses; schema `pre_submission_compliance_check_v1`; **86 checks**, per-check field is `passed`; **82 pass / 4 red**; reds are exactly the 4 adjudicated ones: `auth_eval_raw_promotion_policy_blockers_absent`, `contest_cpu_auth_eval_exists`, `submission_runtime_has_no_network_install_or_local_paths`, `hosted_archive_manifest_supplied`.
- `statement_preview` carries **`# submission name: sz1_composed_reencode`** ✓ — the current name, so r5 post-dates the round-7 rename.
- **Freshness proved by mtime against every surface the receipt itself records as scanned** (I did not assume the list — I read `public_hygiene.scan_paths` and the four `sources` arrays from the receipt):

  | scanned surface | mtime (epoch) | vs r5 `1787057490` |
  |---|---:|---|
  | packet `README.md` | 1787056395 | −1,095 s ✓ |
  | packet `report.txt` | 1787046793 | −10,697 s ✓ |
  | packet `archive_manifest.json` | 1787046811 | −10,679 s ✓ |
  | repo `PR_BODY_DRAFT.md` (statement file **and** public-scan path) | 1787056362 | −1,128 s ✓ |
  | `archive.zip` | 1787025399 | −32,091 s ✓ |
  | newest of all 34 runtime files | ≤ 1787025399 | ✓ |

  **Zero surfaces post-date r5.** r5 is fresh under the law in `SWAP_PROCEDURE.md:45-58`.
- The four reds' adjudications were checked against `COMPLIANCE_RUNBOOK.md:127-143` and the `SWAP_PROCEDURE.md:83-108` note; none is converted by an edited receipt or check.

### Runtime tree — CLEAN

- `submission_runtime.runtime_file_count = 34`. I re-hashed **all 34** recorded files on disk: **34/34 byte-identical, 0 mismatches, 0 missing.**
- Submission tree `67059c1db9ded5d45904d3018d8d1612e4d4e24f5bd4ba3fc0c36c92885a6043` ✓ · portable content tree `994f8aaab28ec1ffbaeedd1075b7de73a1ca411773edf5112efe309f64230b35` ✓ · auth-eval side `0d0fc008d6a37bd5cfa804073e617a8ea30a7c6b6e6c4a1022e2c5d7ce6f9513` ✓ (from `contest_auth_eval.json` provenance). All three match the prompt and each other's declared role.
- **Tree-sha vocabulary is consistent, not mislabelled** (I specifically tested the round-5-F7 class one field over): the packet convention is "runtime tree" = `0d0fc008` (the tree the T4 row validated) and "submission tree" = `67059c1d` — used that way at `COMPLIANCE_RUNBOOK.md:11-12, 150`, and the shipped `README.md:87` discloses the basis explicitly ("computed over the sealed tree containing those exact bytes"). The **shipped** manifest carries only the cross-side-invariant `portable_runtime_content_tree_sha256 = 994f8aaa` — the round-5-F7 cure is holding.
- **Round-5 F5 holding:** `/usr/bin/find` for `__pycache__` and `*.pyc` over the whole packet → **zero hits**.
- Files on disk outside the evaluated manifest: **exactly 5**, all declared custody exclusions (`README.md`, `report.txt`, `archive_manifest.json`, `BORROWED_SUBSTRATE_ACCOUNTING.md`, `archive.zip`). No surprise files.
- `external_dependency_roots = []`, `scorer_import_hits = []`, `disallowed_runtime_imports = []`. Single `forbidden_side_effect_hit` = `inflate.sh:27` (the adjudicated Brotli pinned-wheel bootstrap). Runtime imports resolve to `brotli`, `numpy`, `torch` + stdlib + local modules — **no `constriction`** (round-5 F3 cure holding).
- `inflate.sh` = `e1b3df4d…62` ✓ and `inflate.py` = `5c5baf88…6d` ✓ vs `PACKET_TARGET.json:41-42`.

### F1 class — path-valued fields (the round-7-F4 / round-8-F1 class)

Machine sweep over **8 JSON custody documents** (5 repo-side: `ARCHIVE_MANIFEST`, `CPU_AXIS_SEALED_FIRE_ORDER`, `GAP_REPORT`, `PACKET_TARGET`, `RECIPE_sz1_composed`; 3 packet-side: `GENERATION_RECEIPT`, `RECEIVER_PARSEBACK`, `archive_manifest`). **476 string values** walked, **91 path-like values** resolved against the store root, the prep dir, and the repo root.

- **88 / 91 resolve.** `PACKET_TARGET.json:16` `generation_manifest` → `generations/gen3_receipts/GENERATION_MANIFEST.json` **exists** (round-7 F4 fix holding).
- **3 unresolved — all adjudicated NOT findings:** `CPU_AXIS_SEALED_FIRE_ORDER.json` `command_argv[17]`, `close.poller_argv_template[5]`, `close.recovery_argv[3]`, all the same `--output-dir experiments/results/ddm_pq2_rr4_exact_contest_cpu_20260817`. That file is `status: SUPERSEDED_BY_EXECUTED_GEN3_FIRE` / `disposition: SUPERSEDED` (`:6-7`) with a complete supersession block (`:149-164`); the order **never fired**, so these are output directories a run would have *created*, not custody pointers to artifacts. Correctly not on disk.
- Two space-containing path strings the regex skips were resolved by hand: `PACKET_TARGET.json:111` `supersession_record` (→ **F2** above) and `:36` `posterior_anchor`.

### F2 class — receipt citations

**8 citation sites** enumerated across the live governance set: 2 live-stale (**F1** above), 1 weak pointer (folded into F1), 5 correctly labelled as history or superseded (listed in the "adjudicated OK" section).

### F3 class — prose vs settled adjudications — CLEAN

Read **in full**, not grepped: `CONTRIBUTION_ETIQUETTE.md` (48 lines), `COMPLIANCE_RUNBOOK.md` (168), `SWAP_PROCEDURE.md` (109), `PR_BODY_DRAFT.md` (302), `README_PUBLIC.md` (168), `ADVERSARIAL_REVIEW_SCAFFOLD.md` (102). **No sentence contradicts any settled adjudication.**

- **Compression-source gate:** etiquette row 19 states VERIFIED + "offered for merge" ✓; runbook `:65-68` keeps the conditional rule and `:70-82` records the adjudication that satisfies it ✓; PR body `:123` "Yes, and it is offered for merge" with Stage A explicitly "documented, not re-run" ✓. Round-6-F6 and round-8-F3 cures both holding; no residual "answer no" contradiction.
- **CPU axis:** etiquette row 21, PR body `:109-119` and `:212-218`, README `:15-24`, report `:76-88`, `PACKET_TARGET.json:47-61`, `SWAP_PROCEDURE.md:90-95` — all state MEASURED-INFEASIBLE with the 3,422.7 s / 1,800 s numbers, and all state that no CPU score exists or is claimed. **No CPU score appears anywhere.** Round-8-F3 cure holding.
- **`pending` sweep** across all live docs: 7 hits, all legitimate — hosting (a true operator gate), the generic `RESET_AUTHORITY` step text, and three that explicitly say the CPU axis is *not* pending.
- **Submission name:** 6 live occurrences of `sz1_composed_reencode` (PR body `:1`, `README_PUBLIC.md:1,3`, packet `README.md:1,3`, runbook `:161`), **zero** occurrences of the old `sz1 composed re-encode` outside review-history documents. Round-7-F5 cure holding.
- **PR-body report block is verbatim, not re-authored** (round-5 F2): extracted the fenced block and compared to the shipped `report.txt` — `block + "\n" == report.txt` exactly, sha `6c41f7faa5a951d905e23651…` on both sides. 3,715 / 3,716 chars.
- **Internal arithmetic in the PR body checks out:** headroom `1800 / 1143.270127967 = 1.574×` ✓; 0.15771 < PR #135's 0.162 ✓; the PR #138 comparison is explicitly labelled measured-vs-unverified with no priority claim ✓.

### F4 class — invented / hand-typed values — CLEAN

**13 `*_utc` fields** enumerated (9 repo-side, 4 packet-side). The two that round 8 fixed were spot-checked against git and are **exact**:

- `GAP_REPORT.json:5` `superseded_utc = 2026-08-18T12:34:45Z` ⇄ `cfe778f18b` committed `2026-08-18T07:34:45-05:00` = **12:34:45Z** ✓
- `CPU_AXIS_SEALED_FIRE_ORDER.json:150` `superseded_utc = 2026-08-18T05:00:51Z` ⇄ `a3ab650ce9` committed `2026-08-18T00:00:51-05:00` = **05:00:51Z** ✓

No timestamp is in the future. `PACKET_TARGET.json:64` `verified_at_utc = 05:12:00Z` sits ~50 s after the e2e receipt's own `built_at_utc = 2026-08-18T05:11:10.312277+00:00` — consistent with a wall-clock read at recording time; **I am not raising it**, as no evidence shows it invented. **Hand-typed sha check:** the two encoder pins in the public text resolve to real commits — `31c64e4ce0` → `31c64e4ce0601e4e2368945a410dbb2cbd5925a4` (ddm_sz1 semantic byte-split) and `85880c77a6` → `85880c77a6eec4cb529b80bf137bddb0a8c64323` (ddm_fx2 SHIPPED_CONFIG freeze) ✓.

### Public hygiene and custody — CLEAN

- Scanned the 3 shipped public files and the 3 repo public sources (6 files) for 11 patterns — `/Volumes/`, `/Users/`, `APDataStore`, `VertigoDataTier`, tailscale-range IPs, `modal.com`, `api_key`, `token=`, `Claude`, `Anthropic`, `fc-01` call ids: **zero hits.** No local paths, no provider identifiers, no machine attribution in any public surface.
- **Repo ⇄ packet document sync:** `README_PUBLIC.md` ≡ packet `README.md` (`ea08053dd4fcc584…`), `REPORT_PUBLIC.txt` ≡ packet `report.txt` (`6c41f7faa5a951d9…`), `BORROWED_SUBSTRATE_ACCOUNTING.md` ≡ packet copy (`ffe913e20f5b00c5…`) — all three **byte-identical**. The two `archive_manifest` files differ by design (repo-side is a documented superset); every shared field agrees.
- **Terminal lane row present and conforming:** `.omx/state/active_lane_dispatch_claims.md:9` — `ddm_sz1_composed_t4_n600` / `sz1_composed_r3` / `completed_contest_cuda_exact_eval_harvested`, sha-bound to archive `debb025f…`, bytes 179,930, runtime tree `0d0fc008…`, call `fc-01M09EHX5MMTJACMRADQPN9P7Z`. The CPU row `:12` records `completed_measured_infeasible_cpu_inflate_3423s_vs_1800s_budget` with no score.
- **Accounting cross-check:** 7 of the 8 section shas the PR body's borrowed-substrate table cites are present in `BORROWED_SUBSTRATE_ACCOUNTING.md`, including the honest `05839d14…` ≠ `5c75e2c7…` disclosure that the shipped RC64 receiver is a *modified* PR135 descendant. The 8th (`5b09fd78`, token stream) is carried by `archive_manifest.json:36` and `CUSTODY_SUPERSEDED.json`.
- `GENERATION_MANIFEST.json` gen-3 core fields agree with `PACKET_TARGET.json` and the receipt on archive sha/bytes, all three score components, `n_samples`, hardware, and both call ids.

---

## Counter recommendation

**Counter: `0/5`.** Two findings, so round 9 is not a clean pass. Per the
scaffold, the counter resets to `0/5` after the fixes land.

**Recommended fix, at CLASS level — the root cause rounds 6/7/8 all named:**

1. **F1** — re-point `COMPLIANCE_RUNBOOK.md:12`, `COMPLIANCE_RUNBOOK.md:47`, and
   `GAP_REPORT.md:6` to r5. Do not fix only the two the finding names: the
   durable cure is to treat "receipt citation" as a **queryable population**
   (`grep -rn "pre_submission_compliance" .` returns it in one line) and make
   the *last step of every receipt re-buy* the re-pointing of every hit, exactly
   as `SWAP_PROCEDURE.md:48-53` already requires. That law was written in round
   8 and this round shows its first execution missed two sites — the law is
   sound, the sweep was not run to closure.
2. **F2** — point `PACKET_TARGET.json:111` at the **gen-2** record
   (`/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/CUSTODY_SUPERSEDED.json`,
   sha `8d65ac3f…`), since the block it lives in is the gen-2 custody note, and
   drop the "also at" identity claim — name the gen-3 record separately as the
   live candidate's own supersession record (sha `f1f3eebf…`). Both shas should
   be written into the field so the next reviewer can check identity, not just
   existence.
3. **Receipt freshness:** both fixes touch only repo-side `.md`/`.json`
   surfaces. `GAP_REPORT.md`, `COMPLIANCE_RUNBOOK.md`, and `PACKET_TARGET.json`
   are **not** in the r5 scanned set (which is packet `README.md`, `report.txt`,
   `archive_manifest.json`, and `PR_BODY_DRAFT.md`). **r5 therefore stays valid
   and does not need re-buying** — provided the fix batch touches none of those
   four files. If it does, re-run the checker in the same batch.

**Nothing in this round licenses a push, a hosting action, or a PR opening.**
The final refusal condition in `SWAP_PROCEDURE.md:80-81` is unchanged and
binding: no such action without explicit operator authorization.

## Erratum — the accounting cross-check's manifest citation is stale (rv17 round 14, R14-F1 derived-set find)

The accounting cross-check cites `archive_manifest.json:36` as the carrier of the `5b09fd78…`
token-stream sha. That citation was generation-correct against the then-live manifest and is
stale against the SHIPPED gen-6 packet: the archive manifest is regenerated per candidate and
the shipped copy is 20 lines. The sha itself remains carried by `CUSTODY_SUPERSEDED.json` as
the same sentence records; only the manifest line reference dangles.

covered-citation: `archive_manifest.json:36`
