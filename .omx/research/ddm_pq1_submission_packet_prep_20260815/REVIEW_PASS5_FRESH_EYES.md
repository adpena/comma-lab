# Review pass 5 — fresh eyes, adversarial

**Verdict: 8 FINDINGS. The counter must reset to `0/5`.**

Reviewer: fresh-eyes arm, no part in building any generation of this packet.
Date: 2026-08-18. Candidate: gen-3 `fx2_a__tuned` (sz1 composed),
`debb025f45bb42e3…`/179,930 B. Read-only: nothing was edited, fixed, staged,
committed, or deleted.

The hash chain and the score are clean. I re-derived both from disk and they
match to the last digit. Every finding below is in the **documentation and
packet-staging layer**, not in the bytes. Three of them are public-facing
statements a maintainer would read and act on.

---

## Findings

| # | Axis | Finding | Evidence path | Severity |
|---|---|---|---|---|
| F1 | 2 — PR body reconciliation | The **shipped `report.txt` says reproduction is `PENDING_REBIND`** while every other surface says VERIFIED. Round 2 fixed the stale re-bind wording in the PR body but the fix never reached `report.txt`. | `…/gen3_sz1_composed_split/report.txt:43-49` · `REPORT_PUBLIC.txt:43-49` (byte-identical, sha `7ff89058…`) vs `PACKET_TARGET.json:63` · `README_PUBLIC.md:106` · `GENERATION_LOG.md` "Reproduction at generation 3" · `PR_BODY_DRAFT.md:249-251` | **HIGH** |
| F2 | 7 — template conformity | The PR body's `# report.txt` block is a **re-authored subset, not the real `report.txt`**. Upstream template says "copy the report.txt content here"; `CONTRIBUTION_ETIQUETTE.md` says "copy the real `report.txt`". The re-author silently drops the whole `=== Reproduction ===` section — which is how F1 stayed hidden through four passes. | `PR_BODY_DRAFT.md:20-39` vs `…/gen3_sz1_composed_split/report.txt` (full text) · `upstream/.github/pull_request_template.md` · `COMPLIANCE_RUNBOOK.md:52` | **HIGH** |
| F3 | 2 — unverifiable public claim | Three public documents declare **`constriction`** a runtime dependency. It appears **nowhere** in the packet — zero matches across the whole tree, and the AST import inventory of all 34 evaluated files lists only `brotli`, `numpy`, `torch`. Round-1 finding F1 already established this and is marked **CLOSED**; the correction never reached the public text. `README_PUBLIC.md` also calls its list "four things" and **omits `numpy`**, which the receiver really does use. | `PR_BODY_DRAFT.md:225-226` · `README_PUBLIC.md:145` · `BORROWED_SUBSTRATE_ACCOUNTING.md:70` vs `pre_submission_compliance.gen3.r3.json` → `submission_runtime.runtime_imports` · `ADVERSARIAL_REVIEW_SCAFFOLD.md:39` | **HIGH** |
| F4 | 5 — packet staging | **`README.md` and `BORROWED_SUBSTRATE_ACCOUNTING.md` are not in the packet directory.** Two public documents state the accounting file ships beside the archive. `SWAP_PROCEDURE.md` step 4 names `README.md` as a packet consumer, and `COMPLIANCE_RUNBOOK.md` refers to "the public `README.md`". Gen-2 staged a README; gen-3 dropped it. | packet dir listing (no `README*`, no `BORROWED*`) vs `PR_BODY_DRAFT.md:198-200` · `README_PUBLIC.md:61-62` · `SWAP_PROCEDURE.md:30-32` · `COMPLIANCE_RUNBOOK.md:32` | **MEDIUM-HIGH** |
| F5 | 5 + 3 — hygiene, checker blind spot | **14 stale `__pycache__/*.pyc` files sit in the packet runtime tree**, each embedding the internal absolute path `/Volumes/APDataStore/pact/ddm_sz1/runtime/tuned/…` in its code object. They are **invisible to the strict checker** — its runtime manifest is 34 files and excludes them, so `submission_runtime_has_no_network_install_or_local_paths` cannot see them. They were **not** in the T4-evaluated tree. All 14 are stale against their sources. | `/usr/bin/find` output below · `pre_submission_compliance.gen3.r3.json` → `submission_runtime.runtime_file_count = 34` · `contest_auth_eval.json` → `inflate_runtime_manifest.runtime_file_count = 34` | **MEDIUM-HIGH** |
| F6 | 7 — governance | **Three different clean-pass counts across two governing documents.** The scaffold header says `0/5`; its own table rows 4-5 say `1/5`; `SWAP_PROCEDURE.md` says "passes 1-4 complete … pass 5 pending", which reads as 4 of 5. A reader of the swap procedure would believe one pass remains when the scaffold says four do. | `ADVERSARIAL_REVIEW_SCAFFOLD.md:3` vs `:13-14` vs `:24` vs `SWAP_PROCEDURE.md:84-86` | **MEDIUM** |
| F7 | 1 — manifest labelling | `archive_manifest.json` key **`portable_runtime_content_tree_sha256` carries `0d0fc008…`, which is the non-portable `runtime_tree_sha256`.** The real portable content tree is `994f8aaab28ec1ff…` on both the auth-eval side and the submission side. A verifier who computes the portable content tree gets `994f8aaa` and reads a mismatch. | `…/gen3_sz1_composed_split/archive_manifest.json` vs `contest_auth_eval.json` → `portable_runtime_tree_sha256_without_submission_custody_files` · `pre_submission_compliance.gen3.r3.json` → `submission_runtime.portable_runtime_tree_sha256_without_custody_files` | **LOW** |
| F8 | 6 — accounting completeness | The sz1 split offset (49) was chosen by **argmax over offsets 0-400**, and the build receipt states plainly that "+22 B over DERIVED is Brotli alignment noise fitted to this frozen payload, **NOT mechanism**". No public document carries that qualification; all state −520 B flat. The repo made exactly this correction for `fx2` in commit `31c64e4ce0` ("−515 B corrected to −498 B mechanism plus 17 B alignment noise"). The same disclosure is owed for sz1. Judgment call, filed per "when in doubt, file it". | `RESULT_pq2_e2e.json` → `split_verification.build_report.profile.rationale` vs `PR_BODY_DRAFT.md:191` · `README_PUBLIC.md:48-54` · `BORROWED_SUBSTRATE_ACCOUNTING.md:205-214` | **LOW** |

**Considered and NOT filed.** `BORROWED_SUBSTRATE_ACCOUNTING.md` titles itself
"generation 3 (rr4 …)" while the packet calls rr4 generation 2. I nearly filed
this. §7 discloses the two numbering schemes explicitly ("this document's
internal generations count its own revisions; the packet's `GENERATION_LOG.md`
counts candidates"). It is self-consistent and disclosed. Not a finding.

---

## Per-axis verification notes

### Axis 1 — hash chain from disk: **CLEAN**

Recomputed on this machine, nothing taken on trust:

```
debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a  archive.zip   179930 B
e1b3df4d9178a1572cf04bc2dd9d2ddcc8f6157deac8ec1c40e89b2114522d62  inflate.sh      2203 B
5c5baf88bd3e0b9860c80d496333bba902ee986ef5ee1dbc0a7724ba948ad6bd  inflate.py      2282 B
```

ZIP internals, read through `zipfile`:

```
name='p'  compress_type=0 (stored)  size=179830  csize=179830
crc=0xdf5dec84 = 3747474564   flag_bits=0   extra_len=0   comment=b''
member sha256: be6db33bce471fe38b3d32cf6b421368721b1ea2ddd3f77b577f2bd27d06b7a8
```

All four values match `archive_manifest.json`, `PACKET_TARGET.json`
`active_candidate.member`, and `PR_BODY_DRAFT.md:15-18`. Single member, stored,
no duplicates, no extra fields.

Score recomputed independently from the report's 8-dp components and the exact
archive bytes:

```
seg  = 100 * 0.00029611              = 0.029611
pose = sqrt(10 * 0.00000688)         = 0.008294576541331089
rate = 25 * 179930 / 37545489        = 0.11980800143527229
S                                     = 0.15771357797660338
```

All three components and the 17-digit total reproduce **exactly** — string-equal
to `contest_auth_eval.json` `canonical_score` and to the value in
`PACKET_TARGET.json`, `archive_manifest.json`, `report.txt`, and the PR body.
Derived figures also check: `1800/1143.270127967 = 1.5744` (claimed "1.574x" /
"1.57x"), `3422.711146813/1800 = 1.9015` (claimed "1.90x over budget"),
`181161 − 179930 = 1231` B and `ΔS = 0.00081967237` vs generation 2.

Byte arithmetic across the lineage is fully coherent: inherited base 182,759 −
1,598 (rr4) = 181,161 − 711 (fx2) − 520 (sz1) = **179,930**, and the claimed
cumulative 2,829 B = 1,598 + 711 + 520.

### Axis 2 — PR body reconciliation: **F1, F3**

Everything numeric traces to a receipt. All five cited commits resolve in git
(`2e0af599`, `31c64e4ce0`, `85880c77a6`, `6449c7cdd5`, `e7ca8575`), and all six
cited chain scripts exist. The e2e receipt `RESULT_pq2_e2e.json` is real and
green: 3 stages `rc=0`, pre-split archive asserted at `9de0f6db…`/180,450 B,
final archive rebuilt to `debb025f…`/179,930 B, determinism repeat
byte-identical, and it carries the correct non-authority labels
(`score_claim = false`, axis `[macOS-CPU advisory …]`).

Two claims do not survive: the reproduction status (F1) and the `constriction`
dependency (F3). F3 is the worse of the two — it is not a stale status but a
statement about the runtime contract that was measured false in round 1, closed,
and then left standing in three public documents.

### Axis 3 — public-text hygiene: **CLEAN in prose, F5 in the tree**

Scanned `PR_BODY_DRAFT.md`, `README_PUBLIC.md`, `REPORT_PUBLIC.txt`, the packet
`report.txt`, and `BORROWED_SUBSTRATE_ACCOUNTING.md` for `/Users/`, `/Volumes/`,
`/tmp/`, `/root/`, `/workspace/`, Tailscale and RFC-1918 addresses, credentials,
Modal call ids, provider transcripts, and any Claude/Anthropic/AI attribution.
**Zero hits in all five.** The 34 evaluated runtime source files are also clean
of local paths.

The leak is in the compiled bytes only (F5), for example:

```
runtime/entropy/__pycache__/rc64.cpython-313.pyc
  → /Volumes/APDataStore/pact/ddm_sz1/runtime/tuned/runtime/entropy/rc64.py
```

`GENERATION_RECEIPT.json` and `RECEIVER_PARSEBACK.json` also carry generator
absolute paths, but that is **disclosed and adjudicated** — `README_PUBLIC.md`
names it, `CUSTODY_SUPERSEDED.json` (verified: binds `debb025f…`/179,930 B and
the right receipt chain) supersedes them, and they cannot be edited without
breaking the pinned tree hash. Not a finding. The `.pyc` files have no such
justification: they are not in the pinned tree at all.

### Axis 4 — compliance receipt: **CLEAN**

`pre_submission_compliance.gen3.r3.json` parses to **86 checks, 82 passed, 4
failed** — exactly as claimed. Receipt sha `11fb93d563d6c12a…`, matching the
scaffold's citation. The four reds and their documented routes:

| Failed check | `COMPLIANCE_RUNBOOK.md` route |
|---|---|
| `auth_eval_raw_promotion_policy_blockers_absent` | red 2 — documented structural, unsatisfiable for any raw-emitter payload |
| `contest_cpu_auth_eval_exists` | red 3 — CPU axis measured infeasible, structural |
| `submission_runtime_has_no_network_install_or_local_paths` | red 4 — `inflate.sh:27` pinned-wheel Brotli bootstrap, by design, e4 precedent |
| `hosted_archive_manifest_supplied` | red 5 — operator-gated hosting |

One-to-one. No red lacks a route, and no route lacks a red. I read the checker's
own detail strings rather than the prose: the `inflate.sh:27` hit is the literal
`uv pip install` line, and the structural red is the unconditional blocker trio
stamped by `contest_auth_eval.py`.

The runbook's §"Generation 3" header still points at the **r2** receipt
(80/6) and only the correction block names r3 (82/4). Cosmetic; the correction
is unambiguous and immediately below. Not filed.

### Axis 5 — packet hygiene: **F4, F5**

`/usr/bin/find` over the packet directory: **51 files**. No AppleDouble `._*`
and no `.DS_Store` inside the packet — the round-1 F2 cure held. But:

- **2 `__pycache__` directories, 14 `.pyc` files.**
- Disk (51) − `.pyc` (14) − `archive.zip` − `archive_manifest.json` −
  `report.txt` = **34**, which is exactly the file count in both the
  auth-eval manifest and the compliance manifest. The `.pyc` are outside
  every hashed and scanned set.
- All 14 are **stale**: header source-mtimes (1786819963-1787019426) differ
  from current source mtime (1787025398). They are `cpython-313` while the
  evaluated environment is Python 3.11.12, so a contest run would ignore them —
  the decode risk is nil. The defect is the path leak and the unevaluated,
  unreproducible bytes riding in a submission directory.

`AppleDouble ._*` files do exist in `gen3_receipts/` (7 of them), outside the
packet. Reported, not touched.

### Axis 6 — borrowed-substrate accounting: **strong, F8 only**

This is the best part of the packet. The accounting is section-level, carries a
closed 4-value class set, and is repeatedly unflattering to its author: it
**withdraws** an `ours-original` label on the residual payload for lack of a
receipt, flags the compressed-model container as "PR-level equality not
independently verified", and states that the shipped RC64 receiver backend
(`05839d14…`) **differs** from PR135's source (`5c75e2c7…`). Lineage credit is
explicit and byte-bound for PR #130, #133, #135, #136, and #138, including the
`rc64_backend.c` row the review charter asks for.

No whole-vehicle originality claim exists anywhere. Both the PR body and the
accounting state the opposite in plain words: "This remains a lossless re-encode
program on a PR130/PR135 learned substrate, not a claim that the learned vehicle
is original." The PR #138 concurrency disclosure is unusually honest — it
concedes PR #138 published the mechanism class first, gives the timeline to the
minute, and makes no priority claim.

The one gap is F8: the argmax-selected offset and its admitted alignment-noise
component are not disclosed publicly, though the same disclosure was made for
`fx2` in the repo's own commit history.

### Axis 7 — template conformity: **F2, F6**

All six upstream template headings are present and answered in plain language:
submission name, upload, `report.txt`, GPU-required, compression-script, and
additional comments. Four extra headings (eval host, build cost, changes from
upstream, competitive or innovative) come from the newer template that
`CONTRIBUTION_ETIQUETTE.md` requires; the pinned upstream snapshot carries only
the older six. A superset is fine — not a finding.

Other etiquette gates hold:

- **One PR, one archive URL.** No URL is claimed. The draft says "pending
  operator-authorized public hosting", and the hosted-manifest check is red by
  design. No archive is checked into the repo tree (verified).
- **CPU and CUDA labeled separately.** Both `[contest-CUDA]` and `[contest-CPU]`
  appear with their own axis tags, and the CPU entry reports a **measured
  boundary**, not a promise. No score is transferred across bytes or axes.
- **No dependency-file mutation, no provider transcripts, no machine
  attribution.** Verified by scan.

The failures are F2 (the `report.txt` section is re-authored rather than copied,
against both the upstream instruction and the packet's own gate) and F6 (the
counter contradiction between the two governing documents).

---

## What I would fix before pass 6

In priority order, and each is small:

1. Re-stage `report.txt` with the VERIFIED reproduction text (F1) and correct
   `REPORT_PUBLIC.txt` with it. Note that `report.txt` is a custody-excluded
   file, so re-staging does not disturb the pinned runtime tree.
2. Replace the PR body's `# report.txt` block with the literal file contents
   (F2). That change alone would have caught F1.
3. Remove `constriction` from all three public documents and add `numpy` to the
   README dependency list (F3). Re-read round-1 F1 before rewriting.
4. Stage `README.md` and `BORROWED_SUBSTRATE_ACCOUNTING.md` into the packet, or
   change the two documents that claim they ship there (F4).
5. Delete the two `__pycache__` directories and add a staging-time guard (F5).
   Consider whether the checker's runtime manifest should refuse `.pyc` in a
   submission dir rather than silently skip it — a gate that cannot see a class
   of file cannot clear it.
6. Reconcile the counter in one place and have the other document cite it (F6).
7. Fix the `archive_manifest.json` key name (F7) and add the offset-selection
   qualification (F8).

None of these touch `archive.zip`, the runtime tree, or the measured row. The
score, the bytes, and the CUDA authority are sound.

---

*Read-only review. No file in the packet, the receipts, or the repository was
modified by this pass. MAIN records the scaffold row.*
