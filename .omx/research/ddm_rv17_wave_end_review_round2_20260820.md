# ddm_rv17 — wave-end adversarial review, ROUND 2: two findings, counter stays 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [receipt + source review, scorer-free]` ·
`score_claim: false` · cost $0 · sibling of `ddm_rv17_wave_end_review_round1_20260820.md`.
Counter authority for the #1157 wave-end 3-pass cycle (the packet's own
`ADVERSARIAL_REVIEW_SCAFFOLD.md` 0/5 counter is a SEPARATE authority, untouched here).

## THE ANSWER, FIRST

**Counter 0/3 — two findings, both MED, both in the same genus: a corrected body under a
heading that still carries the superseded claim.**

Round 2 is mostly good news, and the good news is load-bearing:

- **SEC-F1 CURED and verified by my own control**, not the three claimed. All five branches of
  `run_staged_secrets_scan` behave: absence → 1, timeout → 1, tool-error → 1, leak → 1, clean → 0.
- **RV17-F1's body correction is complete and correct** — it names the false clauses, records all
  four measured values, and correctly upgrades identity to TOKEN level. **But the section heading
  above it still reads "confirmed at the component level."**
- **The swap landing is sound.** 36/36 manifest rows verify against the frozen tree; the refusal
  registry genuinely refuses the composed sha; the zero-stale census's exempt classes are not
  hiding a live stale claim; and pq11's strongest claim — **"the delta is EXACT, not
  bound-limited" — VERIFIES**: jg5 and rc2 produce *byte-identical* raw fields.
- **Item 3 found no regression and no silent fix.** Every sampled round-1 finding is either
  correctly cured or correctly still open with its routing intact. Sealed custody was respected.

---

## ITEM 1 — did my two HIGH cures hold?

| # | checked | method EXECUTED | MEASURED result | verdict |
|---|---|---|---|---|
| 1a | RV17-F1 memo correction (`5ea3ccbc0c`) | grep every residual clause incl. headings; diff fills vs the receipt values I measured in round 1 | body correction complete + accurate; **heading still "component level"** | **FINDING R2-F1** |
| 1b | SEC-F1 fail-closed (`5ea3ccbc0c`) | source read of all branches + **my own 5-branch execution** | absence/timeout/tool-error/leak → 1; clean → 0 | **CLEAN — cure held** |

### 1a — the body is right; the heading is stale

The appended correction is exemplary: it quotes the two false clauses, marks them FALSE, and
records `raw_sha256`, the full `stage_seconds` split, `decoded_token_sha256` (`cc10a7b0…`, noted
as byte-matching the CPU leg), `inflate_elapsed_seconds = 458.752594349`, and the harness verdict
`498.476 s charged ≤ 822 s cold-cache ceiling, PASS`. Every figure matches what I measured
independently in round 1. §THE DECODE WALL was rewritten to lead with the measured term and
confine wc2's PROJECTION to the GitHub-Actions setup terms only. That is a complete cure of the
claim.

### RV17-R2-F1 — MED — the corrected section still carries the superseded claim in its heading

`.omx/research/ddm_rc2_t4_row_sixteenth_move_20260820.md:27`

```
## DECODE IDENTITY — confirmed at the component level; the rr2 desync did NOT occur
```

The body four lines below now states identity is proven **at TOKEN level, "stronger than the
component-level form."** The heading was not rewritten. A reader scanning headings — which is how
these memos are actually read — takes away the weaker, superseded claim, and the memo contradicts
itself between its own heading and its own body.

This is the campaign's named genus, and my round-1 memo invoked the very law this cure tripped
over: *corrections land in bodies, headlines keep the stale number*
(`corrections_land_in_bodies_headlines_keep_the_stale_number_20260805`).
**CURE:** rewrite the heading to `## DECODE IDENTITY — proven at TOKEN level (decoded-token sha
matches across both axes); the rr2 desync did NOT occur`.

### 1b — SEC-F1 held, on my own instrument

I did not run the three claimed controls. I drove the **real** `run_staged_secrets_scan` with
`subprocess.run` monkeypatched per branch, mutating no git index (concurrent arms were committing):

```
  nobin    rc=1 expect=1 PASS      timeout  rc=1 expect=1 PASS
  toolerr  rc=1 expect=1 PASS      leak     rc=1 expect=1 PASS
  clean    rc=0 expect=0 PASS
```

Source confirms: `return 1` at `:1644` (FileNotFoundError), `:1652` (TimeoutExpired), `:1674`
(rc=1), `:1683` (any other rc); `return 0` only at `:1659` (rc=0) and `:1609` (explicit waiver).
The fail-open vacuity class is gone. A bonus the cure earned: the tool-error message now names its
own migration path (*"if `protect` was dropped by a gitleaks upgrade, migrate this step to the
supported staged-scan command"*), which partly answers SEC-F6's misleading-headline complaint.

---

## ITEM 2 — the swap landing `afdca8b5ac` as new material

| # | checked | method EXECUTED | MEASURED result | verdict |
|---|---|---|---|---|
| 2a | 36-row tree from staged bytes | `shasum -a 256 -c` against the frozen gen6 tree | **36 OK** | CLEAN |
| 2b | tree pin re-derived myself | 2 canonical derivations + source read of the producer | **neither reproduces** `fdd57749…`; 3 non-row inputs enter | **FINDING R2-F2** |
| 2c | zero-stale census, 4 exempt classes | independent census + context read of every public-doc hit | 40 lines; every hit genuinely exempt | CLEAN |
| 2d | compress.py refusal registry | executed `refuse_if_not_expressible` on 4 shas | composed REFUSED, jg5 REFUSED, unknowns pass | CLEAN |
| 2e | "delta is EXACT not bound-limited" | both receipts' `raw_sha256` | **both `6bf8acf8d4412e43…`** | CLEAN — claim VERIFIED |

**2a.** The repo does not carry the composed runtime, and pq11 does not pretend otherwise — the
frozen generation lives at `/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_rc2_composed/`.
Run there, the manifest verifies **36/36 OK**. Run against the jg5 tree it is 30 OK / 3 changed
(`inflate.py`, `inflate.sh`, `runtime/residual_archive.py`) / 3 absent
(`runtime/f26_corrector_native.c`, `runtime/native_free_corrector.py`, `runtime/rr5_arith_basis.py`)
— exactly the port×rider delta, and 33 + 3 = 36 closes.

**2e is the sharpest result of this round.** pq11 claims the ΔS is exact rather than limited by
8dp component rounding. It is: `raw_sha256` is **identical** across jg5 and rc2
(`6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883`), as is `decoded_token_sha256`
(`cc10a7b0…`). The two decodes produce the same 3,662,409,600 bytes, so `d_seg` and `d_pose` are
identical *pixel-for-pixel*, not merely equal to eight decimals. The receipt's own
`report_8dp_score_worst_case_abs_error_bound` of 3.633e-06 therefore does not apply to the delta;
the −1.125302e-4 is exactly the rate term. Combined with round 1's CPU-leg match, the same token
sha now holds three ways: **jg5-CUDA = rc2-CUDA = rc2-CPU.**

### RV17-R2-F2 — MED — "these 36 rows ARE the pin" is false as written

`.omx/research/ddm_pq1_submission_packet_prep_20260815/MANIFEST.sha256:3-5`

> `These 36 rows ARE the pin. runtime_tree_sha256 fdd5774921319a…` *"is derived from exactly this
> enumerated row set, NOT from a recursive walk of this directory."*

MEASURED — I could not reproduce it from the rows by either canonical route:

```
mode=ondisk      416f4cfbc22a6c8c…      mode=normalized  a9b29313e94eb2a6…
DECLARED         fdd5774921319a31…
```

Source explains why (`experiments/contest_auth_eval.py:505-514`): `tree_sha` hashes a payload of
`runtime_root_name` + `files` (richer rows than the manifest's two fields) +
`external_dependency_roots` + `repo_local_tac_import_manifest` + `upstream_evaluate_py`. Verified
non-row inputs for this pin: `runtime_root_name = "submission_dir"`, a populated
`repo_local_tac_import_manifest`, and `upstream_evaluate_py` (`evaluate.py`, 6,005 B). The row set
alone cannot determine the digest, so a public reviewer following the header cannot reproduce it —
and the header is the packet's own instruction to try.

The pin itself is **not in doubt**: `STAGING_RECEIPT.runtime_only.json` records
`runtime_tree_sha256_rederived_from_measured_staged_bytes = fdd57749…` with 36 declared / 36
verified and verdict `STAGED_TREE_PROVED_IDENTICAL_TO_EVALUATED_TREE`, and it equals the rc2
authority receipt's `expected_runtime_tree_sha256`. The defect is the *claim about what determines
it*. This is round-1 RV17-F7 sharpened: F7 said the verify command omits its working directory;
F2 says the pin sentence is false for every published digest.

**CURE:** publish the env-decoupled `runtime_content_tree_sha256` (`ccd9f7ab…`, already computed
and sitting unpublished in the receipt) as the reviewer-reproducible pin, and restate
`runtime_tree_sha256` as the harness's binding that additionally covers root name, repo-local tac
imports, and `upstream/evaluate.py`.

**2c, for the record.** My independent census found 40 old-identity lines. Every public-surface
hit is a genuine exemption: `README_PUBLIC.md:800` and `PR_BODY_DRAFT.md:52` are explicit
both-rows-named comparisons; `REPORT_PUBLIC.txt:36` and `PR_BODY_DRAFT.md:109` say "Prior packet
generation 5 measured…"; `GAP_REPORT.md:11` is under a "superseded object" marker; and
`GPU_ROUTING_VARIANTS.md:88` now carries the "33-row tree" scope pq11 added — that fix held. No
exemption class is hiding a live stale claim.

---

## ITEM 3 — did the routed owners regress, or silently fix without routing?

Eight round-1 findings sampled (≥3 required). **No regression. No silent fix. No sealed-custody
edit.** Every row is either correctly cured or correctly still open with routing intact.

| finding | sev | status now | evidence |
|---|---|---|---|
| RV17-F6 receipts as bytes-reprs | MED | **did NOT propagate** — all 10 gen6 receipts are proper JSON/text; jg5 tree still `b'` and **not silently edited** (last commit `2d61b51988`, the original custody landing) | `head -c 2` across both generations |
| SW1-F3/F4/F5 strict-flip blockers | MED→HIGH | **correctly NOT strict-flipped** — `preflight.py:6132` still `strict=False` | source read |
| SW1-F1 `RuntimeError` on `~user/…` | MED | still raises — open, routed | `portable_path_form('~notauser/x')` → RuntimeError |
| SEC-F5 `gitleaks protect` deprecated | MED | still at `:1622`; **acknowledged** in the new rc=2 message rather than silently dropped | source read |
| SEC-F7 waiver needs no rationale | MED | unchanged at `:1603` — open | source read |
| RV17-F2 "2 ULP" | LOW | unchanged at `:13` — open | grep |
| RV17-F7 manifest verify cwd | LOW | unchanged, and **sharpened** into R2-F2 | grep + re-derivation |
| SW1-F6 "35 tests" vs 25 | MED | no erratum commit exists — open | `git log --all --since` → 0 |

The healthy signal here is the *shape*: the two HIGH findings were cured immediately, the MED/LOW
rows were left honestly open rather than quietly patched, and the one place an edit would have
been a violation (sealed jg5 custody) was left alone even though the defect is real.

---

## COUNTER

**0 / 3.** Two findings (R2-F1, R2-F2), both MED, neither manufactured. Round 1's prediction was
that every round this week finds ≥1 defect; round 2 sustains it, and the falsifier — a genuinely
clean pass — remains unclaimed and available.

Worth naming plainly: both findings are the *same failure at two surfaces*. A body was corrected
while the heading above it kept the superseded claim (R2-F1), and a manifest's rows were made
correct while the sentence describing what they determine stayed wrong (R2-F2). The cures were
real; the *labels on top of them* were not re-read. Round 3 should check headings, section titles,
and one-line summaries first — that is where this wave's residual defect lives.

No code fix landed this round: R2-F1 is a one-line heading rewrite owned by the rc2 memo owner,
R2-F2 a header restatement inside the packet, which is read-only to me.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round2_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
