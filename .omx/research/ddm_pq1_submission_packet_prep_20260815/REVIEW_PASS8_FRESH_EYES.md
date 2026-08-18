# Review pass 8 — fresh eyes, adversarial

**Verdict: 4 FINDINGS. The counter stays at `0/5`.**

Reviewer: fresh-eyes arm #4. I built no generation of this packet, ran no fix
batch, and wrote no prior review. Date: 2026-08-18 (12:43Z at write time).
Candidate: gen-3 `fx2_a__tuned` (sz1 composed), `debb025f45bb42e3…`/179,930 B.
Read-only: I edited, staged, committed, and deleted nothing outside this file.

**The bytes are sound.** I recomputed the whole hash chain from disk, re-derived
`S` to the last digit from the T4 receipt's own 8-dp components, hashed all 34
runtime files against the r4 manifest, reconciled the byte lineage across four
generations, resolved every cited commit and its UTC timestamp, and proved the
PR body's report block string-equal to the shipped `report.txt`. All of that
passed.

**All six round-7 fixes landed in full.** I checked each one against the file on
disk, not against the fix commit message. This is the first round where nothing
half-landed.

**Two of the four findings are second instances of classes round 7 named.** F1 is
a broken path in `PACKET_TARGET.json` — round-7 F4 corrected the one key it
named and nobody swept the file. F3 is a stale row in
`CONTRIBUTION_ETIQUETTE.md` — round-6 F6 corrected one row of that table and
nobody read the row above it.

**F2 is new and it is the one that matters.** The round-7 fix batch edited two of
the four files the canonical compliance receipt reads. `r4` was produced before
those edits and still carries the superseded text inside itself. Four documents
assert `r4` binds the packet as it stands. It does not.

---

## Findings

| # | Axis | Finding | Measured truth | Severity |
|---|---|---|---|---|
| **F1** | 4, 7 — round-7 F4 class, second key | **`PACKET_TARGET.json` still contains a path that does not resolve.** `generation_2_superseded.custody_note.supersession_record` = `"candidate_runtime/CUSTODY_SUPERSEDED.json"` (line 111). Nothing is there. Round-7 F4 fixed `active_candidate.generation_manifest` and stopped; the file was never swept. The value is not a superseded *number* under the standing adjudication — it is a pointer that names the one location the packet's own hazard record says the file must **not** occupy. | `/usr/bin/find` over the gen-2 store: the record is at **`/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/CUSTODY_SUPERSEDED.json`** (store root). `GAP_REPORT.json:104-107` states the reason in its own words: *"custody file inside the hashed tree … writing a `CUSTODY_SUPERSEDED.json` beside them would have changed the pinned tree hash and broken replay … cure: moved the supersession record to the store root."* `candidate_runtime/` holds exactly 7 entries: `archive.zip`, `cpr1`, `GENERATION_RECEIPT.json`, `inflate.py`, `inflate.sh`, `RECEIVER_PARSEBACK.json`, `runtime`. I resolved all 21 paths cited across `PACKET_TARGET.json` and the public docs; this is the only miss. | **MEDIUM** |
| **F2** | 4, 7 — NEW class: the canonical receipt predates its own inputs | **`pre_submission_compliance.gen3.r4.json` was produced before the round-7 fix batch changed two of the four files it scans, and no re-run was performed.** `r4` reads `README.md` (packet), `report.txt`, `archive_manifest.json`, and `PR_BODY_DRAFT.md`. Round 7 rewrote `PR_BODY_DRAFT.md:1` and `README.md:1-5`. The receipt proves its own staleness: `post_deadline_submission_policy.statement_preview` still reads **`"# submission name: sz1 composed re-encode"`** — the exact string round-7 F5 ruled non-compliant. Six checks consume those two files (`public_hygiene`, `post_deadline_submission_policy`, `public_template_placeholders`, `public_source_reproducibility`, `hosted_archive`, `public_evidence_axis_labels`). The packet's own precedent is the opposite: round 5's fixes were *also* on custody-excluded surfaces (`report.txt` text, manifest key, staged docs) and `r4` exists **because** those fixes triggered a re-run plus a measured hash-neutrality proof. Round 7's batch got neither. | UTC times: `r4` written `09:56:23Z`; `PR_BODY_DRAFT.md` `12:32:31Z`; packet `README.md` `12:33Z`; fix commit `cfe778f18b` `12:34:45Z`. `git show 1a74d8f2d1:…/PR_BODY_DRAFT.md \| head -1` → `# submission name: sz1 composed re-encode`; current line 1 → `# submission name: sz1_composed_reencode`. Four documents assert r4 is current: `COMPLIANCE_RUNBOOK.md:145-152` ("the CANONICAL terminal receipt … All review passes from round 6 onward count against r4"), `SWAP_PROCEDURE.md:86-89`, `ADVERSARIAL_REVIEW_SCAFFOLD.md:18-21`, `GAP_REPORT.json` supersession note. The scaffold's round-7 row says "Packet bytes untouched (README.md is custody-excluded)" — true for the runtime-tree hash, and irrelevant to whether the checker read the file. | **MEDIUM-HIGH** |
| **F3** | 3, 6 — round-6 F6 class, adjacent row | **`CONTRIBUTION_ETIQUETTE.md:19` states a present-tense fact about the compression chain that three current surfaces contradict.** The row reads: *"The draft credits the lineage and exposes the exact compression-side script inventory, **while refusing to call the current path-saturated scripts publication-ready**."* Round 6 edited exactly one row of this table (the CPU row, line 21) and left this one. The runbook binds the document: *"Before recommending a PR, apply `CONTRIBUTION_ETIQUETTE.md`."* So the packet's governance rubric now instructs the reviewer to withhold the answer the PR body gives. | `git show 1a74d8f2d1 -- …/CONTRIBUTION_ETIQUETTE.md` — a one-line diff, the CPU row only. Contradicted by: `COMPLIANCE_RUNBOOK.md:70-79` ("Compression-source gate adjudication … **SATISFIED** … The bundle exists and is real"); `GAP_REPORT.json:110-114` (`friendly_compression_source_bundle`, `"disposition": "CLOSED"`, *"No private paths; inputs arrive via `--inputs-json` or environment variables and are sha256-verified"*); `BORROWED_SUBSTRATE_ACCOUNTING.md:98` ("stage scripts carry no local filesystem defaults"); `PR_BODY_DRAFT.md:123` ("**Yes, and it is offered for merge.**"). "Path-saturated" is the falsified half; "not publication-ready" is the contradicted half. | **MEDIUM** |
| **F4** | 7 — NEW class: hand-typed timestamps in machine-readable custody fields | **Two `_utc` supersession timestamps are invented, and one of them is in the future.** `GAP_REPORT.json.supersession.superseded_utc = "2026-08-18T13:30:00Z"` — the block was committed at `12:34:45Z` and it is `12:43Z` as I write. The file asserts an event that has not happened. `CPU_AXIS_SEALED_FIRE_ORDER.json.supersession.superseded_utc = "2026-08-18T05:30:00Z"`, committed `05:00:51Z` — 29 minutes ahead of its own write. Both are round-numbered to the half hour; every machine-emitted timestamp in this packet carries sub-second precision (`GENERATION_MANIFEST.created_utc = 2026-08-18T04:01:04.454883+00:00`). These are the fields a future custody audit joins on. | `TZ=UTC git log -S'SUPERSEDED_BY_GENERATION_3' -- …/GAP_REPORT.json` → `cfe778f18b 2026-08-18T12:34:45Z`. `TZ=UTC git log -S'SUPERSEDED_BY_EXECUTED_GEN3_FIRE' -- …/CPU_AXIS_SEALED_FIRE_ORDER.json` → `a3ab650ce9 2026-08-18T05:00:51Z`. `date -u` at review time → `2026-08-18T12:43:00Z`. Host TZ is `-0500`; I confirmed it against `archive.zip` mtime `Aug 17 22:56` local = `2026-08-18T03:56Z`, which matches the CPU fire's `measured_at_utc`. | **LOW** |

**Considered and NOT filed.**

- *`BORROWED_SUBSTRATE_ACCOUNTING.md:1` titles itself "generation 3 (rr4
  re-encode candidate)" while `GENERATION_LOG.md` calls generation 3
  `fx2_a__tuned`.* Two shipped documents use "generation 3" for two candidates.
  I nearly filed it. The document disambiguates itself twice — in the title's own
  parenthetical, and at `:186-192` with an explicit numbering note ("this
  document's internal generations count its own revisions; the packet's
  `GENERATION_LOG.md` counts candidates"). No reader takes away a wrong fact. A
  top-of-file banner would still be an improvement and I would take it.
- *`.omx/research/…` receipt citations inside the shipped public accounting* (20
  hits, all in that one file). I checked every one with `git ls-files
  --error-unmatch`: all 15 distinct memos are tracked and present, so a reader at
  `github.com/adpena/comma-lab` can follow them. Not a leak, not a dead citation.
- *The staging directory is `gen3_sz1_composed_split` while the submission
  directory name is `sz1_composed_reencode`.* Custody path versus submission
  name; `README_PUBLIC.md:3-5` states the distinction explicitly.
- *`README_PUBLIC.md:131` shows `bash evaluate.sh --submission-dir . --device
  cuda` after telling the reader to work inside the submission directory, where
  `evaluate.sh` does not sit.* A convenience block, not a claim; upstream's own
  README runs it from the repo root. Cosmetic.
- *`report.txt` prints `Runtime tree SHA-256: 0d0fc008…` unqualified* while the
  submission directory hashes to `67059c1d…`. I resolved all three tree hashes
  independently: `0d0fc008…` is the auth-eval side
  (`provenance.inflate_runtime_manifest.runtime_tree_sha256` in the T4 receipt),
  `67059c1d…` is the submission-dir side, `994f8aaa…` is the portable content
  tree both sides agree on. The line sits inside the "Exact result identity"
  block, which describes the authority run. Correctly named.

---

## Per-axis verification

### Axis 1 — round-7 fix regression: **all six LANDED**

| Round-7 fix | Verdict | Proof from disk |
|---|---|---|
| (a) runbook gen-0 block labelled | **LANDED** | `COMPLIANCE_RUNBOOK.md:6` — `## Generation 0 run (2026-08-15) — HISTORICAL, retired e480b candidate`. `:8-12` past tense ("WERE the generation-0 invocation") plus the gen-3 values beside. `:42-47` now reads "The GENERATION-0 final strict run **confirmed** (2026-08-15, historical) that **that** packet's executable runtime **matched** its measured CUDA authority tree", with `77b94b5c…`/`26c7d418…` explicitly tagged GENERATION-0 and `67059c1d…`/`994f8aaa…` named as the gen-3 counterparts. |
| (b) `GAP_REPORT.json` supersession | **LANDED** (stamped time wrong → F4) | Parses. `status = SUPERSEDED_BY_GENERATION_3`. Supersession facts all verified true: gen-3 archive `debb025f…`/179,930 B ✓, canonical receipt `gen3_receipts/pre_submission_compliance.gen3.r4.json` (82/86) ✓, and the `--axis cpu` resolution ✓ — `tools/fire_modal_auth_eval.py` declares `--axis cuda\|cpu` and the CPU row fired through it (call `fc-01M09G62A7SZ7HZYE5Q28YS7VP`, terminal lane row present). `live_counter_authority` points at the scaffold. |
| (c) `SWAP_PROCEDURE` cites r4 | **LANDED** | `:86-89` — "the gen-3 terminal state is 82/86 (`gen3_receipts/pre_submission_compliance.gen3.r4.json` — the CANONICAL terminal receipt, re-run after the round-5 fixes; r3 predates those edits and is superseded)". |
| (d) `PACKET_TARGET` manifest path | **LANDED** | `generation_manifest = "generations/gen3_receipts/GENERATION_MANIFEST.json"` resolves; the file is the real one — `schema pq1_generation_manifest.v1`, `generation: 3`, `candidate_id fx2_a__tuned`, **52 `files{}` entries**, and it carries the `cpu_axis` block. (A different path in the same file is still broken → F1.) |
| (e) submission name | **LANDED** | `PR_BODY_DRAFT.md:1` = `# submission name: sz1_composed_reencode`. `README_PUBLIC.md:1` = `# sz1_composed_reencode — submission packet`, with `:3-5` naming `submissions/sz1_composed_reencode/`. Packet `README.md` **byte-identical**: both `ea08053dd4fcc584bae5ae756999c3cbbdf5a3dc8e5538732b2cb22589e233f1`, 8,187 B, `diff` silent. The string is a legal directory name and satisfies the template's match-exactly rule. (The edit invalidated r4's scan → F2.) |
| (f) scaffold round-1 F4/F5 | **LANDED** | Both read **CLOSED 2026-08-18** with receipts: F4 cites the `--axis cuda\|cpu` landing (task #1105) and the call id; F5 cites the measured-infeasible adjudication. F6 (hosted archive) correctly stays OPEN. |

### Axis 2 — hash chain from disk: **CLEAN**

Recomputed with `shasum -a 256`, `unzip -lv`, and `.venv/bin/python`. Nothing
taken on trust.

```
debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a  archive.zip   179930 B
be6db33bce471fe38b3d32cf6b421368721b1ea2ddd3f77b577f2bd27d06b7a8  member p      179830 B  stored  crc 3747474564 (0xdf5dec84)
e1b3df4d9178a1572cf04bc2dd9d2ddcc8f6157deac8ec1c40e89b2114522d62  inflate.sh      2203 B
5c5baf88bd3e0b9860c80d496333bba902ee986ef5ee1dbc0a7724ba948ad6bd  inflate.py      2282 B
6c41f7faa5a951d905e23651d615b88fe8f1cbbeef2f330627b22d227885203f  report.txt      3716 B
ea08053dd4fcc584bae5ae756999c3cbbdf5a3dc8e5538732b2cb22589e233f1  README.md       8187 B
f13030171df65100e35d2c83739b1755f673f669697759ca63983e5e2fe6883f  r4 receipt     74181 B
7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b  upstream/evaluate.py
```

One member, stored, no duplicates. Every value matches `PACKET_TARGET.json`, both
manifests, `report.txt`, `README_PUBLIC.md`, and the PR body.

Score re-derived from the r3 auth-eval JSON's own 8-dp fields
(`avg_segnet_dist = 0.00029611`, `avg_posenet_dist = 6.88e-06`) and the exact
archive bytes:

```
seg  = 100 * 0.00029611        = 0.029611
pose = sqrt(10 * 6.88e-06)     = 0.008294576541331089
rate = 25 * 179930 / 37545489  = 0.11980800143527229
S                              = 0.15771357797660338   (float-equal, ==)
```

Equal to `canonical_score` in the receipt, to `PACKET_TARGET.json`, to both
manifests, to `report.txt`, to `README_PUBLIC.md`, to the PR body, and to
`accepted_anchor_history[191]` in the posterior. The raw T4 `report.txt` gives
`Compression Rate: 0.00479232` → `×25 = 0.11980800` and `Final score … = 0.16`,
matching the packet report's `Reported (2 dp display): 0.16`.

Whole lineage reconciles, each score recomputed from its own byte count against
the same seg+pose:

```
gen0 183,502 B → 0.1600920261571558   gen1 182,759 B → 0.15959729295498598
gen2 181,161 B → 0.15853325034789678  gen3 179,930 B → 0.15771357797660338   (all ==)
ΔS gen1→gen2 = 0.0010640426070892     ΔS gen2→gen3 = 0.000819672371293  ("-0.00081967237")
bytes  181,161 − 179,930 = 1,231 = 711 (token) + 520 (split)
token  112,110 → 110,512 → 109,801;  cumulative 1,598 + 711 + 520 = 2,829
headroom 1800 / 1143.270127967 = 1.5744 ("1.574x" / "1.57x")
CPU      3422.711146813 / 1800 = 1.9015 ("1.90x")
```

Timing fields cross-check against the receipt: `inflate 1143.270127967`,
`evaluate 38.30728400299995`, `wrapper 1191.703496111`. The seal holds —
`FIRE_MANIFEST.stage3b_seal` is `CONSISTENT`, `pinned == measured == debb025f…`
at 179,930 B, `problems: []`.

**Commits and the concurrency timeline.** Every commit cited in the public
accounting resolves, and each UTC timestamp matches the claim to the minute
(host TZ `-0500`): `915e87dce3` 2026-07-22 21:21Z; `fdf3298801` 14:41:58Z;
`c8e6ee416c` 14:49:00Z; `f7e29a124c` 16:04:57Z; `f1de91eb46` 19:32:52Z. "About
twenty-six days" from 2026-07-22 to PR #138's 2026-08-17 08:31Z is exact. Source
pins resolve: `2e0af5996…` matches `provenance.pact_commit` in the T4 receipt as
the body claims; `31c64e4ce0` and `85880c77a6` both exist with the described
subjects. Leaderboard figures in the credits match
`.omx/state/canonical_frontier_pointer.json` exactly (PR #135 0.162 rank 1, PR
#133 0.166 rank 2, PR #130 0.172 rank 3).

### Axis 3 — cross-document consistency: **F3**

`REPORT_PUBLIC.txt` and the packet `report.txt` are **byte-identical**
(`6c41f7fa…`, 3,716 B, `diff` silent). So are `README_PUBLIC.md` ↔ packet
`README.md` (`ea08053d…`) and the repo accounting ↔ packet accounting
(`ffe913e2…`).

The PR body's report block is a **literal copy**, not a re-authoring. I extracted
the ```` ```text ```` fence programmatically: 3,716 bytes, sha `6c41f7fa…`, and
`fence == shipped_file` is `True` as a string comparison.

Submission-name sweep over the whole prep directory and the packet:
`sz1_composed_reencode` appears in exactly the three intended places (PR body
line 1, `README_PUBLIC.md` title, packet `README.md` title). No stale
`sz1 composed re-encode` or `sz1 composed candidate` title survives anywhere
outside the review reports. The remaining `gen3_sz1_composed_split` hits are all
custody paths.

The only cross-document contradiction I found is F3.

### Axis 4 — receipt integrity: **F1, F2**

`pre_submission_compliance.gen3.r4.json` parses. Schema
`pre_submission_compliance_check_v1`, top-level `passed = False`, per-check field
`passed`. **86 checks, 82 true, 4 false.** The four map one-to-one onto the
runbook's routing — no red without a route, no route without a red:

```
auth_eval_raw_promotion_policy_blockers_absent           → runbook red 2 (documented structural)
contest_cpu_auth_eval_exists                             → runbook red 3 (measured infeasible)
submission_runtime_has_no_network_install_or_local_paths → runbook red 4 (pinned-wheel Brotli, e4 precedent)
hosted_archive_manifest_supplied                         → runbook red 5 (operator-gated)
```

It binds this packet: `archive.sha256 = debb025f…`, `bytes = 179930`, member `p`
with `crc32 = 3747474564` and a local header that agrees with the central
directory, `runtime_file_count = 34`, `runtime_tree_sha256 = 67059c1db9ded5d4…`,
`portable_… = 994f8aaab28ec1ff…`.

**Independent custody check.** I hashed every file in the r4 manifest against
disk: **all 34 byte-identical, zero mismatches, zero missing**, byte counts
equal. The 5 files on disk outside the manifest are `archive.zip`,
`archive_manifest.json`, `report.txt`, `README.md`,
`BORROWED_SUBSTRATE_ACCOUNTING.md`. 34 + 5 = 39 = the disk count.

**The runbook's r4 paragraph is accurate.** I diffed r3 against r4 directly:
identical `67059c1d…`, identical `994f8aaa…`, identical 34-file manifest
(set-equal on `(path, sha256)`), and **zero** checks differ on `passed`. The
claim "byte-identical to r3 — measured proof the fixes were hash-neutral" holds.

**Path resolution.** I resolved all 21 paths cited by `PACKET_TARGET.json`, the
custody records, and the public docs. Twenty resolve. One does not — F1.

**Reproduction receipt.** `RESULT_pq2_e2e.json` supports every claim made about
it: `seed = 1234`, `score_claim: false`, `promotable: false`, pre-split asserted
at `9de0f6db…`/180,450 B with `determinism_repeat_byte_identical: true`, and
`split_verification` landing `debb025f…`/179,930 B with
`build_report.archive.repeat_identical: true`, `delta_bytes_vs_base: -520`, and
`receiver.bit_exact_vs_base: true` / `base_still_decodes_unchanged: true`. The
build report's own `profile` block states the offset qualification more bluntly
than the public text: `offset 49`, `length 8284`, *"The +22 B over DERIVED is
Brotli alignment noise fitted to this frozen payload, NOT mechanism."* That
qualification appears in all three required public places.

**Inherited-label check.** I opened the two stale receipts rather than trusting
the README. `GENERATION_RECEIPT.json` → `candidate_id hv1_base_control`,
`archive.bytes 182759`, `archive.sha256 80d9c8c6…`. `RECEIVER_PARSEBACK.json` →
same candidate, `token_stream 112,110 B`. Exactly what the README, the
`CUSTODY_SUPERSEDED.json`, and `GENERATION_MANIFEST.custody_note` all say. Their
per-section shas match the accounting's receipts (semantic `b489c735…`/36,051 B,
carrier `196f0e51…`/22,242 B, compensation `38792b49…`/36 B, model
`e35d1237…`/70,453 B, HPAC `e8c0cfd7…`/17,952 B, residual `74775aab…`/100 B). The
README's claim that they are inert is true: `grep -rn 'GENERATION_RECEIPT\|RECEIVER_PARSEBACK'`
over `inflate.sh`, `inflate.py`, `cpr1/`, `runtime/` returns **nothing**.

**Lane rows.** The terminal CUDA row exists, is sha-bound, and uses the
conforming status prefix (`completed_contest_cuda_exact_eval_harvested`, archive
+ runtime tree + call id in its notes), on lane `ddm_sz1_composed_t4_n600` / job
`sz1_composed_r3` as the runbook says. The CPU terminal row matches
`CPU_AXIS_SEALED_FIRE_ORDER.supersession.lane_terminal_row` word for word.

### Axis 5 — packet hygiene: **CLEAN**

`/usr/bin/find` over the gen-3 directory for `*.pyc`, `__pycache__`, `._*`,
`.DS_Store`, and any dotfile: **zero matches**, exit 0. 39 files total. Every
file except `archive.zip` decodes as UTF-8 (denominator: 39 scanned, 38
decoded, 0 failures). AppleDouble litter exists one level up in
`generations/` and in `gen3_receipts/`, which is outside the submission tree —
worth remembering at staging time, not a packet defect.

Public-text scan over the 7 public files, **with the denominator stated**: 7 of 7
read, **78,688 bytes**, 15 pattern families. Results:

```
/Users/ 0   /Volumes/ 0   /root/ 0   /workspace 0   /tmp/ 0   /home/ 0
modal call ids (fc-01…) 0   providers (modal|vast.ai|lightning|kaggle|runpod) 0
tailscale 0   RFC-1918 0   CGNAT 0   bearer/api-key/secret/token/password 0
email 0   claude|anthropic|co-authored-by|"generated with" 0
machine names (m5 max|macbook|bat00|molt|mac-mini) 0
```

Three families returned non-zero and I read every hit: `.omx/research/…` (20,
all in the accounting — tracked memos, checked with `git ls-files`),
`experiments/…` (10 — the deliberate chain-script inventory), and `arm64 smoke`
(5 — the cross-microarchitecture disclosure, which is the honest content, not
machine attribution). **No credential, no infrastructure address, no provider
name, no local path, no machine attribution, no Claude attribution.**

I state the failure mode I guarded against: round 7 recorded that its first scan
of this axis was vacuous because zsh did not word-split the file list. I ran mine
inside Python with an explicit per-file loop and printed the byte denominator, so
a zero here is a measured zero.

### Axis 6 — PR template and governance: **F3**

All six upstream headings are present, in order, and answered. I checked the
answers against the template's own constraints, not just their presence:

- `# submission name:` — `<!-- the directory name … matches exactly -->` →
  `sz1_composed_reencode`. Legal directory name, no spaces, and
  `README_PUBLIC.md:3-5` pins it as `submissions/sz1_composed_reencode/`.
  Round-7 F5 is closed.
- `# upload zipped archive.zip` — hold-state, no URL claimed. Consistent with
  `hosted_archive_manifest_supplied` being red by design; not a placeholder.
- `# report.txt` — `<!-- copy the report.txt content here -->` honoured
  literally, proved byte-for-byte above.
- `# does your submission require gpu…?` — `<!-- yes|no -->` → "**Yes**", with
  the measurement behind it.
- `# did you include the compression script? and want it to be merged?` —
  `<!-- yes|no -->` → "**Yes, and it is offered for merge**", scoped to Stage A
  documented / Stage B exact / Stage C decode.
- `# additional comments` — answered.

Four extra headings (eval host, build cost, changes from upstream, competitive or
innovative) come from the newer template `CONTRIBUTION_ETIQUETTE.md` requires. A
superset is fine.

`SWAP_PROCEDURE.md`'s refusal conditions are unweakened. Its gen-3 note maps two
literal conditions onto documented adjudications without editing a receipt or a
check, names the scaffold as the single counter authority, and restates the final
refusal verbatim: no push, hosting, or PR without explicit operator
authorization. CPU and CUDA are labelled separately on every public surface, and
the CPU entry reports a measured boundary rather than a promise.

One document instructs a contradictory answer, and it is the one the runbook
tells the reviewer to apply before recommending a PR — F3.

### Axis 7 — other

I went at the surfaces nobody has resolved.

- **`CPU_AXIS_SEALED_FIRE_ORDER.json` supersession facts: all TRUE except the
  timestamp.** `executed_call_id` matches `PACKET_TARGET.cpu_axis.modal_call_id`;
  the outcome numbers match to the digit (3,422.7 s vs 1,800 s; token decode
  3,108.7 s; decoded field `9ba2e52b…`); `lane_terminal_row` matches the live
  claims file; `artifacts_dir` resolves; all five `outcome_recorded_in` surfaces
  do carry the outcome. Only `superseded_utc` is invented — F4.
- **`GENERATION_MANIFEST.json` (52 entries)** is internally consistent with the
  packet: archive block, authority block (`record_source` cites
  `accepted_anchor_history[191]`, which exists — history length 192 — and whose
  archive sha, bytes, score, runtime tree, portable tree, inflate.sh sha and
  `pact_commit` all match disk), `cpu_axis`, `supersedes`, `custody_note`,
  `reproduction`.
- **`GENERATION_LOG.md`'s latest entries** are correct: the gen-3 row carries the
  right bytes and score, "What changed at generation 3" matches the mechanism
  and both byte deltas, and the reproduction section matches
  `RESULT_pq2_e2e.json`.
- **Accounting arithmetic** re-checked at section level: §7.4's "2,829 bytes
  (1,598 + 711 + 520)" is exact, and §7.1/§7.2 classifications match the PR
  body's inline table row for row. The document remains unflattering to its
  author where the evidence demands — it withdraws `ours-original` on the
  residual payload for want of a receipt, flags the model container as
  "PR-level equality not independently verified", records that the shipped RC64
  receiver backend `05839d14…` **differs** from PR135's source `5c75e2c7…`, and
  declines to call a standard shuffle filter original. No whole-vehicle
  originality claim exists.
- **`PACKET_TARGET.json`'s `cpu_axis` and `reproduction` blocks** were checked
  against their receipts on disk rather than against each other. Every `cpu_axis`
  number appears in the CPU artifacts and the lane row; every `reproduction`
  number appears in `RESULT_pq2_e2e.json`. The `reproduction.note` is the most
  precisely scoped wording of the determinism claim anywhere in the packet.
- **Runtime source claims** verified against the shipped scripts: the Brotli
  bootstrap really is `inflate.sh:27` (as red 4 and `GAP_REPORT.json` say); the
  C compile really is `inflate.sh:32` with `${CC:-cc}` (as the accounting says);
  `F26_TOKEN_DECODER` really defaults to `python`, so the `Darwin` /
  `brew --prefix libomp` branch really is unreachable (as the README says);
  `exit 69` really is fail-closed on missing `uv`; `inflate.py` pins
  `ARCHIVE_SHA256 = debb025f…` and `ARCHIVE_BYTES = 179_930` and refuses on
  mismatch.

---

## What I would fix before pass 9

1. **Re-run the strict checker and produce `r5`** on the current bytes and the
   current public text, then repoint the runbook, `SWAP_PROCEDURE`, the scaffold
   clause, and `GAP_REPORT.json` at it — and state, as round 5 did, whether the
   runtime tree and 34-file manifest are byte-identical to r4 (F2). Then add the
   missing rule to `SWAP_PROCEDURE` step 4: *any edit to a file the compliance
   checker reads requires a re-run, whether or not the file is custody-excluded.*
   `report.txt` and `README.md` are both custody-excluded and both are checker
   inputs; "custody-excluded" has now been mistaken for "not an input" once.
2. Correct `PACKET_TARGET.json`'s `supersession_record` to the store-root path,
   and **resolve every path in the file in the same pass** — the round-7 fix
   corrected one key and left its sibling (F1).
3. Refresh `CONTRIBUTION_ETIQUETTE.md:19` to the adjudicated state, or banner the
   row (F3).
4. Replace both hand-typed `superseded_utc` values with the real write times, or
   drop the field (F4).

The pattern that produced F1 and F3 is now four rounds old and it is the same
one every time: **a fix is applied to the instance the finding named, not to the
class the finding described.** Round 7 fixed one path in a file with two;
round 6 fixed one row in a table with two. The cheapest cure is a rule the
fixer must satisfy before closing: *for each finding, name the class, enumerate
every instance of that class in the repository, and fix or banner all of them in
the same commit.* That is stronger than the mechanical per-file sweep round 7
ran, because a per-file sweep asks "does this file describe gen-3?" and neither
F1 nor F3 fails that question.

None of this touches `archive.zip`, the runtime tree, the measured row, or the
score. The bytes, the custody, and the CUDA authority are sound, and I verified
each of them from disk rather than from the documents.

---

*Read-only review. No file in the packet, the receipts, or the repository was
modified by this pass except this report, and nothing was committed. MAIN
records the scaffold row.*
