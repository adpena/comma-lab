# Review pass 7 — fresh eyes, adversarial

**Verdict: 6 FINDINGS. The counter stays at `0/5`.**

Reviewer: fresh-eyes arm #3. I had no part in building any generation of this
packet, no part in any prior fix batch, and no part in any prior review. Date:
2026-08-18. Candidate: gen-3 `fx2_a__tuned` (sz1 composed),
`debb025f45bb42e3…`/179,930 B. Read-only: I edited, staged, committed, and
deleted nothing.

**The bytes are sound and the packet itself is clean.** I recomputed the whole
hash chain from disk, re-derived the score to the last digit, hashed all 34
runtime files against the r4 manifest, and proved the PR body's report block is
string-equal to the shipped file. Every one of those checks passed.

**Four of the six round-6 fixes landed. Two landed only halfway.** Round 6 named
`COMPLIANCE_RUNBOOK.md` and `GAP_REPORT.md/.json`; the fix commit `1a74d8f2d1`
cured one surface in each document and left the other. F1 and F2 below are the
two remainders, and git proves it: `GAP_REPORT.json` was never touched by
`1a74d8f2d1` at all.

**Three findings are new classes nobody has checked.** F4 is a broken on-disk
path in `PACKET_TARGET.json` — no prior round resolved a single path reference.
F5 is a template answer that violates the template's own stated constraint —
rounds 5 and 6 both checked that the heading was present and answered, never
that the answer satisfies the rule. F6 is a stale forward-looking instruction
inside the counter-authority file.

The root cause round 6 named is real and still firing: documents in this
directory drift out of the active generation. Round 6 cured the *rule*
(SWAP_PROCEDURE step 4 is now a directory, and un-refreshed documents must carry
a banner). Nobody then applied the cured rule to the whole directory. F1, F2 and
F6 are three documents that the new rule already forbids.

---

## Findings

| # | Axis | Finding | Measured truth | Severity |
|---|---|---|---|---|
| **F1** | 1 — round-6 F4 half-landed | **The runbook's entire pre-"Generation 3" block is generation-0, unlabelled, and framed as current.** Line 3 states "This runbook targets the active candidate in `PACKET_TARGET.json`", then the command block pins generation-0 values on a different volume. Lines 32-39 then assert in the **present tense** that "the packet's executable runtime **still matches** the measured CUDA authority tree: full tree `77b94b5c…` and portable content tree `26c7d418…`". Round-6 F4 named this exact text and its fix list said to label it generation-0. The r4 paragraph landed at lines 136-143; the labelling did not. | Command pins `e3e6f440…`/183,502 B / tree `77b94b5c…` (gen-0, `/Volumes/VertigoDataTier/…`). Gen-3's measured values are submission tree `67059c1db9ded5d4…`, portable `994f8aaab28ec1ff…` — stated 100 lines later at line 141 **of the same file**. A reader who applies lines 32-39 to this packet computes a mismatch. | **MEDIUM-HIGH** |
| **F2** | 1 — round-6 F5 half-landed | **`GAP_REPORT.json` carries no banner and is generation-2 content.** Round-6 F5 named `GAP_REPORT.md` **and** `.json`. The `.md` received the supersession banner. The `.json` was **not touched by the fix commit** — `git log -1` returns `a411f612aa` (2026-08-17, the gen-2 re-target). It declares `"packet_disposition": "READY_EXCEPT_OPERATOR_GO"` and `"submission_authority": "OPERATOR_GO_REQUIRED_NOT_GRANTED"` for a retired candidate, with no supersession field of any kind. | `.json` says `swap_generation: 2`, archive `35ac2b9beb7e6fa8…`, 181,161 B, strict `81/85`, receipt `ba596648…`. The `.md` banner beside it says the report is **generation-0** (`e3e6f440…`). Two halves of one report, two different superseded generations, and the machine-readable half is unlabelled. `CPU_AXIS_SEALED_FIRE_ORDER.json` shows the correct pattern (`"status": "SUPERSEDED_BY_EXECUTED_GEN3_FIRE"` plus a full `supersession` block). | **MEDIUM-HIGH** |
| **F3** | 1 — round-6 F4 class at a sibling | **`SWAP_PROCEDURE.md`'s generation-3 adjudication note binds the packet to the superseded r3 receipt.** Lines 86-88 read: "the gen-3 terminal state is 82/86 (`gen3_receipts/pre_submission_compliance.gen3.r3.json`)". r3 does **not** describe the packet on disk. The document names r4 nowhere. Round 6 edited this very note (it added the counter-authority bullet) and left the receipt pointer stale — the document that carries the cure was itself missed. | mtimes: r3 `03:19:38`; round-5 fixes rewrote `report.txt` `04:53:13` and `archive_manifest.json` `04:53:31`; r4 `04:56:23`. r3 was run against the `PENDING_REBIND` report text and the pre-fix manifest key. The runbook calls r4 "the CANONICAL terminal receipt"; the scaffold says "All review passes from round 6 onward count against r4". | **MEDIUM** |
| **F4** | 7 — new class: path resolution | **`PACKET_TARGET.json` points at a file that does not exist.** `active_candidate.generation_manifest` = `"generations/gen3_sz1_composed_split/GENERATION_MANIFEST.json"`. Nothing is there. `auth_runtime.file_inventory` compounds it: "GENERATION_MANIFEST.json files{} — per-file sha256 for the full tree" — so the packet's per-file custody inventory is unreachable from the canonical target file. The move was deliberate (the runbook records "non-standard metadata moved to `gen3_receipts/` so the runtime-tree hash matches the sealed value"); `PACKET_TARGET.json` was never updated and round 6 did not touch it. | `/usr/bin/find` over `generations/`: the file is at **`generations/gen3_receipts/GENERATION_MANIFEST.json`** (52 `files{}` entries, carries the `cpu_axis` block). `CPU_AXIS_SEALED_FIRE_ORDER.json` cites it correctly as "GENERATION_MANIFEST.json cpu_axis (gen3)". Every other path in `PACKET_TARGET.json` resolves — I checked all eight. | **MEDIUM** |
| **F5** | 6 — new class: template answer validity | **The PR body's submission name cannot be the directory name the template demands.** The body answers `# submission name: sz1 composed re-encode`. The upstream template says: "the directory name of your submission … **please make sure it matches exactly**". Upstream's convention is `submissions/<name>/`. The answer contains spaces, matches no directory, and **no document in the packet declares a submission directory name** (grep over the whole prep directory returns only this line). `README_PUBLIC.md:1` uses a third string, "sz1 composed candidate". Our own `CONTRIBUTION_ETIQUETTE.md` row 1 requires following the template literally, citing PR #110. | `upstream/.github/pull_request_template.md` lines 1-3; `upstream/README.md:59-62,96-104`. Peer submissions all use valid directory names: `semantic-pose-HPAC_CPR1_polished`, `opal_v1`, `hnerv_lc_v2`. Rounds 5 and 6 verified the heading was present and answered; neither checked the answer against the constraint. | **LOW-MEDIUM** |
| **F6** | 1 — stale instruction in the counter authority | **Two round-1 findings are still marked OPEN in the scaffold, and both are resolved.** F4 reads "**OPEN** — cure the tool before firing any CPU row"; F5 reads "**OPEN** — operator/MAIN economic call; fire order prepared, $1.38 headroom". F4 is a forward-looking instruction telling a future actor to do work already done. The table carries no as-of date, and F1-F3 were updated to CLOSED, so it reads as live status rather than frozen history. | `tools/fire_modal_auth_eval.py:405-411` declares `--axis cuda\|cpu`, "added 2026-08-18 for task #1105". The CPU row **was** fired through it: `CPU_AXIS_SEALED_FIRE_ORDER.json.supersession` records "bought on the GEN-3 bytes via the canonical tool (`tools/fire_modal_auth_eval.py --axis cpu`)", call `fc-01M09G62A7SZ7HZYE5Q28YS7VP`. F6 (hosted archive) is correctly still open. *Scoping note: if MAIN reads the round-1 table as frozen-at-round-1 history, this is not a defect — but then it owes a HISTORICAL banner under the rule round 6 just added.* | **LOW** |

**Considered and NOT filed.**

- *The PR body's determinism-repeat claim.* Line 142 says the final archive was
  "reproduced at `debb025f…`/179,930 bytes with a byte-identical fresh-process
  repeat". `RESULT_pq2_e2e.json`'s top-level `verification` block records the
  repeat only for the **pre-split** archive (`9de0f6db…`/180,450 B), which looked
  like a named-object-vs-measured-object error. It is not. The split stage
  carries its own proof: `split_verification.build_report.archive.repeat_identical
  = true`, and I hashed the artifact — `split_stage/archives/fx2_a__tuned/archive.repeat.zip`
  is `debb025f45bb42e3…`. The claim is exact. `PACKET_TARGET.json`'s wording is
  the most precise of all the surfaces.
- *The public source URL carries the operator's GitHub handle* (3 hits across the
  7 public files). That is the deliberate, contest-required source pin naming the
  submitter. Not a leak.
- *`ARCHIVE_MANIFEST.json` (repo) and `archive_manifest.json` (packet) share the
  schema id `ddm_pq1_archive_manifest.v1` with different field sets.* The repo copy
  is a superset. All ten `archive_manifest_*` checks pass at r4. Sloppy, not wrong.
- *The runbook's "Generation 3" section still headlines r2's "80 GREEN / 6 RED".*
  The r3 correction block and the r4 canonical paragraph follow it in order. It
  reads as a chronological ledger, and r4 is now named. Round 5 reached the same
  conclusion; I reached it independently.

---

## Per-axis verification

### Axis 2 — hash chain from disk: **CLEAN**

Recomputed on this machine with `.venv/bin/python`. Nothing taken on trust.

```
debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a  archive.zip   179930 B
e1b3df4d9178a1572cf04bc2dd9d2ddcc8f6157deac8ec1c40e89b2114522d62  inflate.sh      2203 B
5c5baf88bd3e0b9860c80d496333bba902ee986ef5ee1dbc0a7724ba948ad6bd  inflate.py      2282 B
6c41f7faa5a951d905e23651d615b88fe8f1cbbeef2f330627b22d227885203f  report.txt      3716 B
cf9884f36caa6639a3e19f9743bf3d73b4ef8c66659e870783856da1228aa36e  archive_manifest.json  665 B
```

ZIP internals through `zipfile`:

```
name='p'  compress_type=0 (stored)  file_size=179830  compress_size=179830
crc=3747474564 (0xdf5dec84)
member sha256: be6db33bce471fe38b3d32cf6b421368721b1ea2ddd3f77b577f2bd27d06b7a8
members total: 1
```

Every pinned value matches `PACKET_TARGET.json`, both manifests, the PR body,
and `README_PUBLIC.md`. Single member, stored, no duplicates.

Score re-derived from the r3 auth-eval JSON's 8-dp components and the exact
archive bytes:

```
seg  = 100 * 0.00029611      = 0.029611
pose = sqrt(10 * 0.00000688) = 0.008294576541331089
rate = 25 * 179930/37545489  = 0.11980800143527229
S                             = 0.15771357797660338   (float-equal, ==)
```

Equal to the receipt's `canonical_score`, to `PACKET_TARGET.json`, to both
manifests, to `report.txt`, and to the PR body. `canonical_score_source =
report_8dp_components_plus_exact_archive_bytes`.

Byte lineage across all four generations reconciles:
`183,502 → 182,759 → 181,161 → 179,930`; `−1,598` and `−1,231` as logged;
token `112,110 → 110,512 → 109,801`; cumulative `1,598 + 711 + 520 = 2,829`
as the PR body claims. ΔS gen2→gen3 `= 0.00081967237`; gen1→gen2
`= 0.0010640426070892`. Both match the log. Headroom `1800/1143.270127967 =
1.5744` ("1.574x"/"1.57x"); CPU `3422.711146813/1800 = 1.9015` ("1.90x").
All seven cited git commits resolve.

### Axis 1 — round-6 fix regression: **F1, F2, F3, F6**

| Round-6 fix | Verdict | Proof |
|---|---|---|
| F1 — repo `ARCHIVE_MANIFEST.json` VERIFIED/true | **LANDED** | `reproduction_verified: true`; entry point now reads "…(VERIFIED 2026-08-18: rebuilt debb025f45bb42e3…/179,930 B exactly, all stages rc=0, determinism repeat byte-identical; receipt RESULT_pq2_e2e.json)". No `PENDING_REBIND` anywhere. |
| F2 — scaffold swap note self-contradiction | **LANDED** | The note is retitled "HISTORICAL; the table above is the counter authority", states the re-bind landed and the chain reached its terminal state, drops the three superseded commit pins, and closes "For the live counter, read ONLY the header and table of this file." |
| F3 — compression-source governance | **LANDED** | The runbook carries the "Compression-source gate adjudication (2026-08-18, MAIN): SATISFIED under the pinned-inputs reading" note; `GAP_REPORT.md`'s "answer no" line now sits under a supersession banner that names the adjudication. I verified the underlying receipt rather than the prose — see Axis 4. |
| F4 — runbook points at r4 | **PARTIAL → F1** | The r4 paragraph landed at lines 136-143 with correct values. The generation-0 tree hashes at lines 32-39 were not labelled. |
| F5 — gap report generation label | **PARTIAL → F2** | `.md` banner landed. `.json` untouched by the fix commit. |
| F6 — etiquette CPU row | **LANDED** | Line 21 now reads "the exact-byte CPU axis is ADJUDICATED MEASURED-INFEASIBLE (inflation 3,422.7 s vs the 1,800 s budget on 4-thread x86_64; no CPU score exists or is claimed — a measured boundary, not a pending item)". |
| Root cause — step 4 is a directory | **LANDED** | `SWAP_PROCEDURE.md:30-44` makes the consumer "**EVERY** document in … the list is a DIRECTORY, never a closed enumeration" and adds "**Any document deliberately NOT refreshed must carry an explicit HISTORICAL/SUPERSEDED banner in the same swap**". The rule is right. F1, F2 and F6 are documents it already forbids. |
| Runbook r4 facts vs the receipt | **CORRECT** | Verified below. |

### Axis 4 — receipt integrity: **CLEAN**

`pre_submission_compliance.gen3.r4.json` parses. Receipt sha
`f13030171df65100e35d2c83739b1755f673f669697759ca63983e5e2fe6883f`, 74,181 B.
The field name is **`passed`** (top-level `passed = False`; per-check `passed`).
**86 checks, 82 `passed: true`, 4 false.** The four failures are exactly the four
the runbook routes, one-to-one, no red without a route and no route without a red:

```
auth_eval_raw_promotion_policy_blockers_absent          → runbook red 2 (documented structural)
contest_cpu_auth_eval_exists                            → runbook red 3 (measured infeasible)
submission_runtime_has_no_network_install_or_local_paths→ runbook red 4 (pinned-wheel Brotli, e4 precedent)
hosted_archive_manifest_supplied                        → runbook red 5 (operator-gated)
```

It binds this packet: `archive.sha256 = debb025f…`, `bytes = 179930`,
`runtime_file_count = 34`, `runtime_tree_sha256 = 67059c1db9ded5d4…`,
`portable_runtime_tree_sha256_without_custody_files = 994f8aaab28ec1ff…`.

The runbook's r4 paragraph claims those three values are "byte-identical to r3".
**True** — I compared the receipts directly: r3 and r4 both report `67059c1d…`,
`994f8aaa…`, and 34. Diffing all 86 checks between r3 and r4, **zero** differ on
`passed` and exactly **one** differs on detail
(`post_deadline_policy_statement_substantive`, statement length 14,562 → 17,961,
which is the PR body growing). The r4 paragraph is accurate.

I resolved the three tree hashes rather than trusting the labels.
`0d0fc008d6a37bd5…` is the **auth-eval side** value
(`provenance.inflate_runtime_manifest.runtime_tree_sha256` in the T4 receipt);
`67059c1d…` is the **submission-dir side**; `994f8aaa…` is the portable content
tree, which both sides agree on — which is why the round-5 manifest-key fix put
that value in the packet manifest. All three usages in the documents are
correctly named.

**Independent custody check.** I hashed every file in the r4 runtime manifest
against disk: **all 34 byte-identical, zero mismatches, zero missing.** The 5
files on disk outside the manifest are exactly `archive.zip`,
`archive_manifest.json`, `report.txt`, `README.md`, and
`BORROWED_SUBSTRATE_ACCOUNTING.md` — matching the receipt's declared
`excluded_custody_filenames`. 34 + 5 = 39 = the disk count.

**Reproduction receipt.** `RESULT_pq2_e2e.json` is real and supports every claim
made about it: 3 stages, all `returncode = 0` (encode 210.1 s, build 3.2 s,
split 6.8 s); 4 inputs all `sha256_matches: true`; pre-split asserted at
`9de0f6db…`/180,450 B; final `debb025f…`/179,930 B with `sha256_matches: true`;
`seed = 1234`; `score_claim: False`, `promotable: False`; Stage A honestly
stamped `DOCUMENTED_NOT_RE_RUN`. All eight paths cited by `PACKET_TARGET.json`
resolve on disk **except** the one in F4.

### Axis 3 — cross-document consistency: **F1, F2 (plus F3, F4 as pointer defects)**

I built an occurrence matrix of nine identity values across all 15 governing
documents, using 8-hex prefixes so that `…`-truncated citations are not missed.
(My first pass used 16-hex patterns and produced false negatives; I caught and
redid it.) Result: no document states a *wrong* value for the active candidate.
Every gen-3 number agrees everywhere it appears. The defects are documents
stating *superseded* values without a label — F1 and F2.

`REPORT_PUBLIC.txt` and the packet `report.txt` are **byte-identical**:
`6c41f7fa…`, 3,716 B, `diff` silent. Both staged packet documents are
byte-identical to their repo copies (`README.md`/`README_PUBLIC.md`
`eabc8564…`; accounting `ffe913e2…`).

The PR body's report block is a **literal copy**. I extracted the ```` ```text ````
fence programmatically: 3,716 bytes, sha `6c41f7fa…`, and
`fence == shipped_file` is `True` as a string comparison.

### Axis 5 — packet hygiene: **CLEAN**

`/usr/bin/find` over the gen-3 directory for `*.pyc`, `__pycache__`, `._*`,
`.DS_Store`: **zero matches**. 39 files. Every file except `archive.zip` decodes
as UTF-8.

Public-text scan over the 7 public files, **with the denominator reported**
(7 of 7 read, 78,309 bytes total): `/Users/`, `/Volumes/`, `/root/`,
`/workspace`, `/tmp/`, Tailscale, RFC-1918 and CGNAT addresses, Modal call ids
(`fc-01…`), `modal`/`vast.ai`/`lightning`, bearer tokens, api keys, secrets,
co-author trailers, `Claude`, `Anthropic`, "Generated with", operator email.
**Total hits: 3**, all the same deliberate line — the public source repository
URL, which the contest requires. No leak, no machine attribution.

I flag a method note against myself: my first scan of this axis was **vacuous**.
zsh does not word-split unquoted variables, so `grep … $FILES` passed one mashed
argument and matched nothing, while `2>/dev/null` hid the error. Zero hits looked
like a pass. I found it because the file list I could see contained a string my
scan reported absent, and I re-ran with an explicit per-file denominator. A
scanner that cannot state how many files it read is not evidence.

### Axis 6 — PR template and governance: **F5**

All six upstream headings are present, in order, and answered in plain language.
Four extra headings (eval host, build cost, changes from upstream, competitive or
innovative) come from the newer template the etiquette document requires. A
superset is fine. The "copy the report.txt content here" instruction is honoured
literally — proved byte-for-byte above.

Governance coherence holds elsewhere. `SWAP_PROCEDURE.md`'s refusal conditions
are unweakened and its adjudication note maps the two literal conditions
("both exact axes complete", "strict checker green") onto documented
adjudications without editing a receipt or a check; the final refusal — no push,
hosting, or PR without explicit operator authorization — is restated verbatim.
No hosted URL is claimed anywhere and `hosted_archive_manifest_supplied` is red
by design. CPU and CUDA are labelled separately in every surface, and the CPU
entry reports a measured boundary rather than a promise. No document now
instructs a contradictory answer to the compression-source question: the
runbook's adjudication and the gap report's banner both point the same way as
the PR body's scoped "Yes".

The one failure is F5: the submission name cannot be the directory name the
template demands, and no directory name exists to match.

### Axis 7 — other

The borrowed-substrate accounting still holds up under a fresh read. It is
repeatedly unflattering to its author — it withdraws the `ours-original` label
on the residual payload for want of a receipt, flags the compressed-model
container as "PR-level equality not independently verified", records that the
shipped RC64 receiver backend `05839d14…` **differs** from PR135's source
`5c75e2c7…`, and declines to call a standard shuffle filter original. No
whole-vehicle originality claim exists; both public documents say the opposite
in plain words. The PR #138 concurrency disclosure concedes that PR #138
published the mechanism class first, gives the timeline to the minute, and makes
no priority claim. The offset alignment-noise qualification appears in all three
required places, and I confirmed it against the build receipt's own `profile`
block, which states it more bluntly than the public text does.

---

## What I would fix before pass 8

1. Label the runbook's lines 3-39 as the **generation-0** run — a heading and a
   past-tense verb are enough — and add the gen-3 values beside them (F1).
2. Add a supersession block to `GAP_REPORT.json` in the shape
   `CPU_AXIS_SEALED_FIRE_ORDER.json` already uses (F2).
3. Repoint `SWAP_PROCEDURE.md`'s adjudication note at r4 (F3).
4. Correct `PACKET_TARGET.json`'s `generation_manifest` path to
   `generations/gen3_receipts/GENERATION_MANIFEST.json` (F4).
5. Decide the submission directory name, use it verbatim in the PR body, and
   make `README_PUBLIC.md`'s title agree (F5).
6. Close scaffold round-1 F4 and F5, or date the table (F6).

Then close the gap the last two rounds have both fallen into. Round 6 wrote the
right rule and did not run it over the directory. **Before pass 8, walk every
file in the prep directory and the packet and answer one question per file: does
this describe generation 3, or does it carry a banner saying it does not?**
Six of the fifteen fail that question today. It is a ten-minute sweep and it is
mechanical — which is exactly why three consecutive rounds have found new
instances of it instead.

None of this touches `archive.zip`, the runtime tree, the r4 receipt, or the
measured row. The score, the bytes, the custody, and the CUDA authority are
sound, and I verified each of them from disk rather than from the documents.

---

*Read-only review. No file in the packet, the receipts, or the repository was
modified by this pass, and nothing was committed. MAIN records the scaffold row.*
