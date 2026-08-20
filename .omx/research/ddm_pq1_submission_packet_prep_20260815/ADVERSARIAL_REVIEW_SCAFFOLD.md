# Submission PR adversarial review scaffold

Status: `HOLD`, consecutive clean passes: `0/5`.

> **CANDIDATE CHANGED AGAIN 2026-08-20 (second change that day) — rounds 1–13
> reviewed SUPERSEDED BYTES.** The packet was re-staged to generation 6, the
> composed candidate (`df7fd266e1b7488c…` / 180,456 B, 36-row runtime tree
> `fdd57749…`, `[contest-CUDA]` **0.14827847122030852**) at the sixteenth pointer
> move. Every row below reviewed generation 2, 3, 4 or 5. Their findings and class
> cures carry — the apparatus is candidate-agnostic — but **no clean pass on
> superseded bytes counts toward this candidate**, and the counter was already
> `0/5`. **Round 14 is the first review of these bytes**; see "Round 14 — what to
> examine" at the end of this file.
>
> Note for the next reviewer: generation 5's body decodes to the *same state* as
> this one (identical n600 raw output on the shipping axis), so generation 5's
> DISTORTION reasoning transfers verbatim. Nothing else does — archive, runtime
> tree, manifest row count, timings, hosted URL and compliance receipt are all
> different objects.
>
> Prior banners preserved verbatim:
>
> **CANDIDATE CHANGED 2026-08-20 — rounds 1–12 reviewed SUPERSEDED BYTES.** The
> packet was re-staged to generation 5 (`f3bce5d259a08183…` / 180,625 B,
> `[contest-CUDA]` **0.14839100138338618** — the first sub-0.15 row in this
> packet's history) at the fifteenth pointer move. Every row below reviewed
> generation 2, 3 or 4. Their findings and class cures carry — the apparatus is
> candidate-agnostic — but **no clean pass on superseded bytes counts toward this
> candidate**, and the counter was already `0/5`. **Round 13 is the first review
> of these bytes**; see "Round 13 — what to examine" at the end of this file.
> 
> Prior banner preserved verbatim:
> 
> **CANDIDATE CHANGED 2026-08-19 — rounds 1–11 reviewed SUPERSEDED BYTES.** The
> packet was re-staged to generation 4 (`35c318d541d70370…` / 177,182 B,
> `[contest-CUDA]` 0.15710198138050818) at the ck1 tenth-move boundary. Every
> row below reviewed generation 2 or generation 3. Their findings and class
> cures carry — the apparatus is candidate-agnostic — but **no clean pass on
> superseded bytes counts toward this candidate**, and the counter was already
> `0/5`. Round 12 is the first review of these bytes; see "Round 12 — what to
> examine" at the end of this file.

Any finding, however small, resets the counter to `0/5` after the finding is
fixed. A pass cannot be counted while the strict compliance chain is red.

| Round | State | Reviewer | Candidate archive SHA-256 | Compliance receipt SHA-256 | Findings | Counter after round |
|---:|---|---|---|---|---|---:|
| 1 | RUN 2026-08-17 | ddm_pq2 | `35ac2b9beb7e6fa8…9618956` (rr4, SUPERSEDED by gen-3) | `ba596648e1e1f6a5…c43656` | 6 (3 closed, 3 open — candidate superseded; open items re-checked below vs gen-3) | 0/5 |
| 2 | RUN 2026-08-18 | MAIN | `debb025f45bb42e3…4b037a` | `11fb93d563d6c12a…` (gen3.r3) | 2 — stale e2e "re-bind in progress" claims in PR body (accounting row + operator checklist); e2e was VERIFIED per `RESULT_pq2_e2e.json`. FIXED `412763e63f` | 0/5 |
| 3 | RUN 2026-08-18 | MAIN | `debb025f45bb42e3…4b037a` | `11fb93d563d6c12a…` | 1 — swap-procedure step-7/refusal literal conditions unsatisfiable for gen-3 (CPU axis measured-infeasible; 82/86 terminal). FIXED `378752e9e2` (adjudication note, refusals unchanged) | 0/5 |
| 4 | RUN 2026-08-18 | MAIN | `debb025f45bb42e3…4b037a` | `11fb93d563d6c12a…` | 0 — hash chain re-verified from disk: archive `debb025f`/179,930 · member `p` `be6db33b`/179,830 · inflate.sh `e1b3df4d` · inflate.py `5c5baf88` all match PACKET_TARGET + manifest; PR-body numbers reconcile with receipts | 1/5 |
| 5 | RUN 2026-08-18 | FRESH-EYES Opus arm (no part in any generation) | `debb025f45bb42e3…4b037a` | `11fb93d563d6c12a…` (reviewed against r3) | **8** — F1 report.txt PENDING_REBIND vs VERIFIED elsewhere · F2 PR-body report block re-authored not copied · F3 `constriction` falsely declared in 3 public docs + numpy omitted · F4 README/accounting absent from packet · F5 14 stale `.pyc` w/ local paths outside the hashed walk · F6 three conflicting counter statements · F7 manifest portable-tree key held the non-portable value · F8 offset alignment-noise qualification undisclosed. Full report: `REVIEW_PASS5_FRESH_EYES.md`. ALL 8 FIXED same-day; packet re-verified `pre_submission_compliance.gen3.r4.json` = 82/86, same 4 typed reds, runtime tree + 34-file manifest byte-identical to r3 (all fixes on custody-excluded surfaces) | **0/5** |
| 6 | RUN 2026-08-18 | FRESH-EYES Opus arm #2 (no part in any generation or prior fix) | `debb025f45bb42e3…4b037a` | `f1303017…` (r4) | **6** — bytes/score/all-8-round-5-fixes verified CLEAN (PR-body block hashes string-equal to shipped report.txt; zero `.pyc`; S re-derived exactly); all 6 findings = stale satellite docs sharing ONE root cause (SWAP_PROCEDURE step-4 consumer list was a CLOSED enumeration omitting 5 siblings): repo `ARCHIVE_MANIFEST.json` PENDING_REBIND/false · scaffold swap note self-contradiction · runbook named r4 nowhere · `GAP_REPORT.md` gen-0 gaps in present tense · etiquette CPU row "pending" vs adjudicated infeasible · compression-source governance contradiction (runbook "answer no" vs PR body "Yes"). Full report: `REVIEW_PASS6_FRESH_EYES.md`. ALL 6 FIXED same-day + ROOT-CAUSE CURED (step-4 list is now a DIRECTORY, never an enumeration; un-refreshed docs must carry HISTORICAL banners) + the governance contradiction ADJUDICATED (compression-source gate SATISFIED under the pinned-inputs reading, receipt `RESULT_pq2_e2e.json`; PR body's scoped "Yes" stands). All fixes on custody-excluded `.md`/repo-side surfaces — packet bytes untouched | **0/5** |
| 7 | RUN 2026-08-18 | FRESH-EYES Opus arm #3 (no part in any generation, fix, or prior review) | `debb025f45bb42e3…4b037a` | `f1303017…` (r4) | **6** — bytes/score/custody again CLEAN (full hash chain + S re-derived + all 34 runtime files byte-identical to r4 + byte lineage across 4 generations reconciled + borrowed-substrate accounting re-verified); 4 findings = round-6 fixes that landed HALFWAY or the rule un-applied (F1 runbook gen-0 block present-tense · F2 `GAP_REPORT.json` untouched by the fix commit · F3 swap-procedure note pinned superseded r3 · F6 scaffold round-1 F4/F5 stale-OPEN) + 2 NEW classes (F4 `PACKET_TARGET.json` generation_manifest path broken — file lives in `gen3_receipts/` · F5 submission name "sz1 composed re-encode" cannot be the directory name the template demands). Full report: `REVIEW_PASS7_FRESH_EYES.md`. ALL 6 FIXED same-day: runbook gen-0 block LABELED + gen-3 values beside · GAP_REPORT.json supersession block in the CPU_AXIS shape · swap note → r4 (canonical, post-round-5-fix) · path corrected · submission name → `sz1_composed_reencode` (PR body + README title + packet README re-synced byte-identical) · round-1 F4/F5 → CLOSED w/ receipts. PLUS the reviewer-prescribed MECHANICAL SWEEP executed: 27 files (18 prep + 9 packet, denominator stated), per-file question "describes gen-3 OR carries a banner" — 5 stragglers adjudicated OK (byte-lineage tables ×3, self-labeling `rr4_generation_2` + `generation_2_superseded` JSON keys), 0 unresolved. Packet bytes untouched (README.md is custody-excluded) | **0/5** |
| 8 | RUN 2026-08-18 | FRESH-EYES Opus arm #4 (no part in any generation, fix, or prior review) | `debb025f45bb42e3…4b037a` | `f1303017…` (r4, found STALE) | **4** — bytes CLEAN + ALL SIX round-7 fixes landed IN FULL (first time a fix batch fully landed); F2 (MED-HIGH, new class): r4 receipt PREDATES round-7's edits to two files it scans — proved by its own statement_preview carrying the old submission name · F1 (MED): PACKET_TARGET supersession_record path pointed at the one location the packet's own hazard record forbids (round-7-F4 class, second key) · F3 (MED): etiquette row 19 "path-saturated not publication-ready" contradicted the compression-source adjudication (round-6-F6 class, adjacent row) · F4 (LOW, new class): two INVENTED hand-typed superseded_utc timestamps, one in the future. Full report: `REVIEW_PASS8_FRESH_EYES.md`. ALL 4 FIXED same-day: path → gen3_receipts copy (both copies exist; sha-checked) · etiquette row → adjudicated state · timestamps → the REAL git commit times (cfe778f18b 12:34:45Z / a3ab650ce9 05:00:51Z) · **the receipt RE-BOUGHT: r5 = 82/86, same 4 reds, statement_preview carries the new name** — and the first r5 attempt was itself REFUSED by the checker because MAIN hand-completed a truncated tree-sha prefix (two tree-mismatch reds), an executed proof of the no-hand-typed-values law; rerun with the receipt-derived sha went clean. TWO CLASS CURES encoded in SWAP_PROCEDURE step 5: the RECEIPT-FRESHNESS LAW (any edit to a scanned surface invalidates the receipt; re-run in the same batch) + NO HAND-TYPED VALUES (shas/timestamps always derived). Root cause the arm named — "each fix lands on the instance, not the class" — now has both structural halves | **0/5** |
| 9 | RUN 2026-08-18 | FRESH-EYES Opus arm #5 (no part in any generation, fix, or prior review) | `debb025f45bb42e3…4b037a` | `6f4f6dc8e3648eb0…` (r5) | **2** — bytes/score/custody CLEAN again; F1 (MED): three docs still cited the SUPERSEDED r4 as canonical after r5 was bought (receipt-citation class, round-8-F2's sister) · F2 (MED): `PACKET_TARGET.json`'s gen-2 block cited the GEN-3 custody record and asserted they were the same file — identity-vs-existence class. Full report: `REVIEW_PASS9_FRESH_EYES.md`. BOTH FIXED same-day in `cea1dbe3ce`, and F1 was fixed AT CLASS LEVEL: the reviewer named 3 sites, the population query `grep -rn "pre_submission_compliance"` found **5**, all 5 re-pointed — the first fix batch of this campaign to run the population query the root cause demands. F2 now cites BOTH shas so identity is checkable, never existence. r5 freshness preserved: no checker-scanned surface touched (verified, not assumed) | **0/5** |
| 10 | RUN 2026-08-18 | FRESH-EYES Opus arm #6 (no part in any generation, fix, or prior review) | `debb025f45bb42e3…4b037a` | `6f4f6dc8e3648eb0…` (r5) | **3** — S re-derived bit-exact `0.15771357797660338` on all three terms (not the rounded field) · 34/34 runtime files · r5 freshness re-derived from the receipt's OWN scanned list across 8 named inputs · 16 live receipt citations of 34 · 118 path-like values of 476 strings across 8 JSON docs (107 resolve, 11 adjudicated) · 14 `*_utc` fields · 1,199 lines/10 docs zero prose-vs-adjudication contradictions · 747 lines/5 public surfaces zero hygiene hits — every class with a DENOMINATOR. Findings: F1 (MED) the staged dir held 33 undeclared files (15 `.pyc` + 18 AppleDouble + 3 `__pycache__`) between 13:16:25Z and 13:27:21Z, cause traced to source (`.omx/tmp/sa3/probe_identity.py:34-41` sys.path.insert + imports); the finding is the MISSING GUARD — rounds 5 and 9 read "zero .pyc" as proof of cleanliness and that quantity is now known unstable between rounds · F2 (MED) this scaffold had no round-9 row while declaring itself the counter authority · F3 (LOW) MAIN's removal receipt asserted the .pyc broke the pinned tree sha; the sha is MANIFEST-DERIVED and measurably never moved (34/34 identical DURING contamination). MAIN's mid-review remediation was verified from disk and adjudicated as NOT invalidating the round — the scored object never moved. Full report: `REVIEW_PASS10_FRESH_EYES.md`. ALL 3 FIXED same-day: F3 receipt corrected + false claim withdrawn in-record · F2 rows 9+10 added · F1 CLASS-CURED by `tools/packet_census_guard.py` (directory census: manifest ∪ declared exclusions, any extra file REFUSES) | **0/5** |

| 11 | RUN 2026-08-18 | FRESH-EYES Opus arm #7 (`ddm_rv8`) | `debb025f45bb42e3…4b037a` | `6f4f6dc8e3648eb0…` (r5, found STALE by INSTRUMENT and by WORLD) | **6** — F1 (MED-HIGH, new class) the canonical receipt r5 is stale on two axes the round-8 freshness law did not cover: the CHECKER changed 1h46m after r5 was bought (86 checks then, 87 live) and our own frontier moved past the candidate twice that day, so `frontier_no_regression_on_submitted_axis` was RED at contest-final severity — re-run measured 82/87 with 5 reds · F2 (MED, new class) **63 absolute local `/Volumes/…` paths shipping in `GENERATION_RECEIPT.json` (16) + `RECEIVER_PARSEBACK.json` (47)**, invisible to the leak scanner for two independent reasons · F3 (MED) the round-10 census guard had no consumer, no tests, and printed `39 declared (34 + 7)` · F4 (MED) the swap trigger still named `e480b`/`e960` in the present tense and could not admit the live pointer · F5 (LOW) accounting title named the wrong candidate · F6 two undeclared marker files staged in the git index inside the PREP tree. Full report: `.omx/research/p0_1111_review_round11_20260818.md`. Fixes: `.omx/research/p0_1111_round11_fixes_20260818.md` — F2(b)/F3/F4/F5/F6 + two class cures (receipts now carry `instrument_and_world`; harvests write a scanner-visible anchor mirror) all FIXED; **F2(a) REFUSED with a measurement** (sanitizing manifest-pinned bytes in place would ship bytes the T4 row never evaluated under an unchanged green tree hash) and deferred to the next re-stage | **0/5** |
| 12 | RUN 2026-08-19 | FRESH-EYES Opus arm #8 (`ddm_rv12`) | `35c318d541d70370…677e3` | `587af0cf78b67858…` (gen4.r1, 83/87) | **6** — F1 (MED) sign-determinacy margin divided by ONE row's 8dp bound for a TWO-row delta; "18.0x" should be **8.97x**, overstated by exactly 2.00x, and flagged as a CLASS across sister docs · F2 (MED) decode-time **network** dependency (`uv` + egress) undisclosed and the bootstrap branch never exercised on the authority run · F3 (MED) **19.08x** advisory-vs-authority pose drift on identical bytes undisclosed (d_seg drifts only 1.43x) · F4 (LOW) shipped leg split asserted "sums to the net" while printed 5-sig-fig values do not · F5 (LOW) AppleDouble cure applied to the staged tree but not to `gen4_receipts/` or `generations/` · F6 (LOW) the superseded **79.40216174747616** contest-CUDA row on the IDENTICAL archive sha undisclosed. Full report: `.omx/research/p0_1111_round12_review_20260819.md`. ALL 6 FIXED (commits `d39cd384b3`, `6e976eeafd`) | **0/5** |
| 13 | RUN 2026-08-20 | FRESH-EYES codex cross-family arm (`ddm_pq10`) | `f3bce5d259a08183…8acb7e` (gen-5, SUPERSEDED same day by gen-6) | none re-bought this round | **4** — F1 (BLOCKING) the selected-object swap never reached the packet: the old archive SHA in 17 files, old byte count in 16, old score in 14, old runtime pin in 11, and ZERO files naming the selected composed object · F2 (HIGH) `GATED-ON-RC2` in 33 places after both fresh receipts already existed, plus two stale method statements (183 B rider "declined", native port "does not ship") · F3 (MED-HIGH) repo-side `ARCHIVE_MANIFEST.json` stale at generation 4 even on its own named object · F4 (BLOCKING for a counted pass) the reviewer appendix executes against the wrong object. The round left every packet byte untouched and routed the indivisible typed swap rather than manufacturing a partial fix. Full report: `.omx/research/ddm_pq10_codex_packet_review_round_20260820.md`. ALL 4 FIXED by the generation-6 swap (`ddm_pq11`), which is why round 14 reviews different bytes | **0/5** |
| — | ROW ADDED LATE 2026-08-20 by `ddm_pq3` | — | — | — | Round 12 was RUN and FIXED on 2026-08-19 but no scaffold row was written — the round-10 F2 class ("this scaffold had no round-N row while declaring itself the counter authority") recurring for the third time. Recorded here by the re-stage arm rather than left for a reviewer to rediscover. **The recurrence is itself the finding**: two class-fixes have failed to stop it, because both cured the missing ROW and neither cured the missing STEP — nothing in `SWAP_PROCEDURE.md` step 6 requires the scaffold row as part of a fix batch. | **0/5** |

**Strict-chain-red clause, gen-4 adjudication (2026-08-19):** the clause is read
against the generation-4 terminal state **83 GREEN / 4 RED of 87** (receipt
`gen4_receipts/pre_submission_compliance.gen4.r1.json`, sha `587af0cf78b67858…`).
The red SET is identical in kind to generation 3's, and two of generation 3's
hardest-won greens survived the swap (schema metric consistency, dispatch
terminal row) while round-11 F1's fifth red is now green because this candidate
IS the pointer. One red changed its JUSTIFICATION and the change is a
weakening we state rather than hide: generation 3 could call the CPU axis
MEASURED INFEASIBLE on its own bytes; **generation 4 has no CPU row at all**,
and the infeasibility is an inherited expectation. Passes count against this
adjudicated state.

**HISTORICAL — the gen-3 clause, preserved:** the clause "a
pass cannot be counted while the strict compliance chain is red" is read
against the gen-3 TERMINAL state 82/86 (receipt **r5**, re-bought after the
round-7/8 fixes; r3/r4 superseded):
the 4 residual reds are each typed and documented (2 structural-by-
construction, 1 by-design Brotli bootstrap per the e4 precedent, 1 operator-
gated hosted manifest) per `COMPLIANCE_RUNBOOK.md` and the SWAP_PROCEDURE
gen-3 adjudication note. Passes count against this adjudicated terminal
state; none of the 4 reds is convertible by further work short of the
operator's hosting authorization. Remaining: **5 consecutive clean passes**
(fresh-eyes reviewers preferred for at least two; passes 5, 6, 7 AND 8 all
proved why — each fresh arm found a class no prior pass enumerated) before
SELECT_ACTIVE_GENERATION. Counting rounds resume at round 9 against the
post-round-8-fix docs + the **r5 receipt** (`pre_submission_compliance.gen3.r5.json`,
sha `6f4f6dc8e3648eb0…`, 82/86, same 4 reds — re-bought after the round-7/8
fix batches per the new receipt-freshness law; r4 is superseded, it predated
round-7's edits to two files it scans). Packet archive bytes unchanged
throughout.

## Round 1 — findings

**The counter stays at `0/5`, and would stay there even with zero findings:** the
strict chain is red at 81/85, and the scaffold forbids counting a clean pass
while it is red. Round 1 is therefore a real review, not a countable pass.

Round 1 was run by the same arm that re-targeted the packet. **That is a
conflict** and is recorded as such: rounds 2 through 5 must be run by reviewers
who did not build generation 2.

| # | Axis | Finding | State |
|---|---|---|---|
| F1 | dependency behavior | The charter's premise that this runtime carries a `constriction` declared-dep is **false**. An AST scan of all 32 runtime files returns `brotli`, `numpy`, `torch`; `constriction` appears nowhere. The real self-installed dep is `Brotli==1.2.0`, and it was smoked instead. | **CLOSED** — premise corrected, correct dep verified |
| F2 | public-text / packet hygiene | The ExFAT store creates AppleDouble `._*` files on every copy. They are non-UTF-8, they **crashed the compliance checker mid-run** leaving a stale receipt on disk, and they would have shipped inside a submission directory. | **CLOSED** — `COPYFILE_DISABLE=1`, explicit strip, non-UTF-8 scan before every run |
| F3 | stale receipts across the byte boundary | The two inherited receipts are **inside** the hashed 32-file runtime tree, so the prescribed cure of writing a supersession file beside them would have changed the pinned tree hash `7acedb07…`. | **CLOSED** — supersession moved to the store root; correction placed in `README.md`, which the evaluated manifest excludes; pinned hash verified to reproduce |
| F4 | exact score axes | `tools/fire_modal_auth_eval.py`, the one canonical deterministic firing path, is **CUDA-only**. A contest-CPU row would have to be hand-assembled, which is the hazard the standing law names an error factory. | **CLOSED 2026-08-18** — `--axis cuda\|cpu` landed (task #1105); the gen-3 CPU row was fired THROUGH the canonical tool (call `fc-01M09G62A7SZ7HZYE5Q28YS7VP`) |
| F5 | both exact score axes on the same bytes | Only one axis exists. `[contest-CPU]` is unmeasured on these bytes and the packet says so everywhere it matters. | **CLOSED 2026-08-18** — the CPU axis was BOUGHT on the gen-3 bytes and adjudicated MEASURED-INFEASIBLE (inflate 3,422.7 s vs the 1,800 s budget; tokens decoded byte-exact; receipt in `PACKET_TARGET.json` `cpu_axis`) |
| F6 | hosted archive URL and manifest | No public URL exists; hosting is unauthorized. The packet claims none. | **OPEN** — blocked on operator authority |

## Axes covered in round 1

- **archive/member SHA, size, grammar, deterministic repeat** — verified
  independently: `35ac2b9b…`, 181,161 B, single stored member `p`, 181,061 B,
  CRC32 885609521; rebuild reproduced the sha and the determinism repeat was
  byte-identical.
- **runtime closure, dependency behavior, budget, hidden payloads** — 32-file
  tree reproduces the pinned hash; bootstrap smoked in a bare venv and exits 69
  fail-closed without `uv`; only a pinned wheel crosses the network; inflation
  used 476.6 s of 1,800 s (3.78x headroom).
- **score recomputation, labels, snapshot, runtime binding** — S re-derived from
  components to all 17 digits; upstream snapshot, `evaluate.py` sha, and runtime
  tree sha all bound in the terminal lane row.
- **borrowed-substrate accounting** — re-run at section level against the shipped
  sections; two lineage items added that the prior generation did not disclose
  (the encoder compiles PR135's `rc64_backend.c` unmodified; the shipped receiver
  coder is a **modified** descendant, sha `05839d14…` ≠ `5c75e2c7…`).
- **public source pin, hosted URL, report linkage** — pin present, visibility
  unverified; no hosted URL claimed.
- **public-text hygiene** — scanned `README_PUBLIC.md`, `REPORT_PUBLIC.txt`,
  `PR_BODY_DRAFT.md` and the staged `README.md` / `report.txt` for local paths,
  infrastructure addresses, credentials, provider records, and machine
  attribution: **clean**.
- **swap delta vs prior generation** — recorded in `GENERATION_LOG.md`; no stale
  receipt crosses the byte boundary, and the two that live in the tree are named
  explicitly in the public README.
- **PR template conformity and scope of pending numbers** — every pending or
  advisory number is labelled; the competitive claim explicitly declines to
  assert a win over PR138's unverified figure.

The fifth clean pass authorizes only a recommendation to MAIN. It does not
authorize submission, push, or hosting.

## Generation-3 swap note (2026-08-18) — HISTORICAL; the table above is the counter authority

The candidate hot-swapped to the sz1 composed archive
(`debb025f45bb42e3…`, 179,930 bytes, measured `[contest-CUDA]`
0.15771357797660338). Round 1's findings were reviewed against the
generation-2 (rr4) bytes; generation-independent items carried forward.
Statements below written at swap time and SINCE RESOLVED: the reproduction
re-bind landed (PACKET_TARGET `reproduction.status = VERIFIED`, receipt
`RESULT_pq2_e2e.json`, rebuilt `debb025f…` exactly); the compliance chain
reached its adjudicated terminal state (r5, 82/86). The CPU axis is closed
for review purposes: measured infeasible within the 1,800 s budget, receipt
in `PACKET_TARGET.json` `cpu_axis`. For the live counter, read ONLY the
header and table of this file.

---

## Round 12 — what to examine

Written by the re-stage arm (`ddm_pq3`), which therefore **cannot run round 12**.
This list names where I believe a fresh reviewer is most likely to find something,
including the places where I know my own work is thinnest.

**The claim classes that are genuinely new at this generation.** Everything the
first eleven rounds hardened was built around a LOSSLESS candidate: "decoded
state is identical, therefore `d_seg` and `d_pose` carry over, therefore every
byte is pure rate." That sentence is false here. The reviewer should assume the
packet's prose has stale reflexes from that era and hunt for them specifically:

1. **Every surviving "byte-identical", "decoded values unchanged", "pure rate",
   or "lossless" sentence.** I rewrote the ones I found across README, report.txt,
   PR body, accounting §8, generation log, and both manifests. A population query
   (not a spot check) is the right instrument. The two §2 accounting rows that
   changed class are the load-bearing ones; §§1–7 of that document are append-only
   and still say `byte-identical` in the old rows by design, under a banner.
2. **The borrowed-substrate re-classification itself.** I moved rows 1 and 2 from
   `inherited-substrate` to `mechanism-adopt-with-attribution`. Argue the other
   side: is a lossy re-quantization of someone else's trained tensors better
   described as still-inherited (we changed only the representation) or as
   something we should not be claiming any share of at all? The classification is
   defensible, not obvious, and I chose the cell that puts more attribution on
   PR #135 rather than less. Check the pose-compensation row especially — the
   solver form is adapted from PR #135's published experiment book and the
   edit-then-recompensate pattern is theirs, which I disclose in three places;
   verify that disclosure is not softened anywhere.
3. **The CPU-axis downgrade.** Generation 3 shipped a MEASURED infeasibility
   receipt on its own bytes. Generation 4 has none, and I replaced the claim with
   an inherited expectation everywhere I found it. Grep for any surviving sentence
   that reads as if the CPU axis were measured HERE. The env-mismatch advisory is
   a decode-correctness proof and must never read as a score; its own receipt
   stamps `score_claim=False`, and the packet should never contradict that stamp.
4. **The reproduction downgrade.** Generation 3's end-to-end VERIFIED label is
   generation 3's. I set `reproduction_verified: false` and wrote the honest
   note in five places. Check that no document still implies these exact bytes
   were rebuilt end-to-end — including `RECIPE_sz1_composed.json`, which describes
   generation 3's chain and is now banner-labelled rather than deleted.

**Where my own process was weakest, named rather than left to be found.**

5. **My staging script's census was VACUOUS on the class that actually bit it.**
   It filtered `._*` out of BOTH sides of its comparison, so it reported "33
   files, 0 undeclared" while 38 AppleDouble sidecars sat in the staged tree. The
   census guard caught them on the next run. I removed them and re-verified all
   33 files byte-identical, but the lesson is that a check which excludes a file
   class cannot certify that class — the round-11 F2(b) blindness in miniature,
   authored by the arm that had just read about it. Re-run the census yourself
   rather than trusting mine.
6. **I appended the terminal dispatch-claim row that turned one red green.** The
   dispatch genuinely completed (rc=0, harvested, `poller.done` present) and
   CLAUDE.md requires closing completed dispatches, so I hold this is hygiene and
   not a converted red — but the arm that benefits from a green should not be the
   only one who says so. Verify the row's contents against the receipt.
7. **Section-level numbers I did not measure myself.** The semantic/carrier/tail
   byte splits, the changed-coordinate count, and the cancellation fraction are
   READ from `SA3_REBASE.json`, the compile receipt. I re-derived the score, the
   leg split, both tree hashes, the member record, and the archive hash
   independently; I did not re-derive the section split. If a number in the README
   or accounting has no independent path, it is one of those.
8. **`GENERATION_RECEIPT.json` / `RECEIVER_PARSEBACK.json` are MISSING, and that
   is the fix.** The census prints them as missing-declared because they remain in
   the guard's `DECLARED_NON_RUNTIME` allowlist. Confirm that this reads as
   round-11 F2 closing by construction and not as a packet that lost two receipts.
   Related: the guard's double-declaration overlap is now `0 in both`, so the
   masking F3 identified is gone for this generation — check that the guard's
   arithmetic still reconciles.

**The standing invariants, unchanged.** Re-derive the score from components
rather than reading `final_score`. Re-hash every manifest row from disk. Check
receipt freshness on all three axes (surfaces, instrument, world) using the
receipt's own `instrument_and_world` block. Confirm no push, hosting action, or
PR opening has occurred.


---

## Round 14 — what to examine (written by `ddm_pq11`, who therefore cannot run it)

This list is written by the arm that executed the generation-6 swap. It names the
thin spots I know about; a fresh reviewer should not treat it as exhaustive. The
generation-5 list is preserved below it, because most of its items are about the
body, and the body did not change.

1. **Re-derive the score from the receipt yourself**, from all three components,
   and confirm `0.14827847122030852` rather than the evaluator's printed `0.15`.
   Check that no surface anywhere states `0.15` as the claim.
2. **Re-run the staging proof on 36 rows.** Verify independently that the staged
   runtime files hash to `MANIFEST.sha256`, and that `runtime_tree_sha256`
   re-derived from freshly measured staged bytes — not from the manifest's own
   claimed digests — equals `fdd57749…`. The receipt asserts it; do not take its
   word, and do not take mine.
3. **Attack the identity claim I lean on hardest.** I claim the generation-5 and
   generation-6 objects emit byte-identical n600 output on the shipping axis, and
   I use that to call the score delta EXACT rather than bound-limited. Both raw
   SHA-256 values are in the two T4 receipts. If they are not equal, the exactness
   claim collapses back to a bounded delta and several sentences are wrong.
4. **Check what I did NOT fill from the receipt.** The `GATED-ON-RC2` markers are
   gone, but the CI residual window (822–1302 s) is still a projection and must
   still read as one everywhere. Grep for any sentence that has quietly promoted
   it to a measurement.
5. **The compliance receipt does not exist for these bytes.** I did not re-buy it,
   deliberately — bytes, surfaces and pointer all moved, so the generation-5
   receipt is stale on all three inputs of the freshness law. Confirm no surface
   cites a generation-5 compliance number as current.
6. **The hosted URL is blank on purpose.** Verify that no leg of any verification
   command resolves to the superseded archive, and that the blank is disclosed as
   a held step rather than an oversight.
7. **Population-query the byte-direction sentences again.** Generation 5 grew;
   generation 6 shrank by 169 B against it while still being larger than
   generation 4. Any sentence that says "spends bytes" or "gives bytes back" now
   needs its reference row named.
8. **The 33-of-36 version-control disclosure.** I measured it; re-measure it. The
   three unpublished files are the two entry points and one receiver module, and
   that set will change the moment the operator publishes.
9. **The AppleDouble ordering law.** Confirm the packet, prep and receipts trees
   are clean AT THE MOMENT YOU LOOK — any write to the ExFAT volume re-creates the
   sidecars, and I wrote to that volume repeatedly during the swap.

---

## Round 13 — what to examine (written by `ddm_pq3`, who staged generation 5)

Preserved for the body-level items, which still apply. Items naming 33 rows,
`2103073d…` or `0.14839100138338618` describe the SUPERSEDED generation-5 object.

1. **Re-derive the score from the receipt yourself**, from all three components,
   and confirm `0.14839100138338618` rather than the evaluator's printed `0.15`.
   The display rounds UP across the exact boundary this packet claims. Check that
   no surface anywhere states `0.15` as the claim.
2. **Re-run the staging proof.** `tools/stage_contest_submission_packet.py` is
   NEW and unreviewed by anyone but its author. Verify independently that the 33
   staged runtime files hash to the manifest, and that
   `runtime_tree_sha256` re-derived from the staged rows equals `2103073d…`. My
   tool asserts this; do not take its word.
3. **The tool has no tests.** That is owed and named. Its census logic is the same
   class that produced the round-10 F1 and round-12 F5 findings, and I wrote it
   in the same session in which I then hit the AppleDouble recurrence myself.
4. **Argue the other side of the accounting.** §9 classifies the joint admission
   waterfill as `ours-original` at 0 counted bytes. Is a decision rule over
   someone else's representation really ours to claim, and does the §9.3 "what is
   not ours" paragraph do enough work?
5. **The terminal dispatch row I appended.** Three compliance reds went green
   because I appended a claim row binding both full shas. The dispatch genuinely
   completed and was harvested — but the arm that benefits from a green should not
   be the only one who says so. Verify the row against the receipt.
6. **Population-query the byte-direction sentences.** This is the first candidate
   whose archive GREW. Grep every surviving sentence that assumes bytes fall
   ("smaller", "byte reduction", "rate credit") and confirm none of them now
   describes these bytes.
7. **The wall-clock disclosure.** I graded the T4 path WARN on a derived residual
   window, not a measured CI run. Check whether the packet ever states the WARN as
   though it were measured, and whether 10.7 s of margin is disclosed everywhere it
   should be.
8. **The GPU-routing document.** I measured that the one-line flip moves the tree
   sha and therefore costs a new T4 row. Verify that measurement independently, and
   check that no surface implies variant (a) is free.
9. **The AppleDouble ordering law.** I purged 51 sidecars mid-session and re-ran
   the census. Confirm the packet, prep and receipts trees are all clean AT THE
   MOMENT YOU LOOK, because any write to the ExFAT volume re-creates them.
