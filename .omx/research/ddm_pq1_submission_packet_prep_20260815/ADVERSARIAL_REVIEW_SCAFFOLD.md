# Submission PR adversarial review scaffold

Status: `HOLD`, consecutive clean passes: `0/5`.

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

**Strict-chain-red clause, gen-3 adjudication (2026-08-18):** the clause "a
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
