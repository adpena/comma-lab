# ddm_rv2 — frontier adversarial review, ROUND 1 of 3 (2026-08-17)

**FINDINGS: 10. Counter resets to 0/3.**

Arm: `ddm_rv2_frontier_adversarial_review_r1` (Opus, review-only) · harness task #823
Axis: `[review — re-derivation from primary artifacts]` · `score_claim: false` ·
Pointer moved: **false** (this is a review; it does not move the pointer and did not try to)
Modal dispatched: **false** · scorer run: **false** · spend: **$0**

Scope: the FRONTIER CHAIN only — the row that moved the pointer at 19:17Z, its composability,
the next-mover ranking, and the two contradictions that touch routing. Not a hygiene sweep.

The row itself is **sound**. Every finding below is in the *custody*, the *narration*, or the
*apparatus that was supposed to catch the narration* — not in the score. I re-derived
`S = 0.15853325034789678` from components and it is exact to 17 digits. That number stands.

---

## OBJECT 1 — THE ROW'S CUSTODY

### 1.1 The score: **CONFIRMED**

Re-derived independently from `avg_segnet_dist` and `avg_posenet_dist` in
`/Users/adpena/Projects/pact/experiments/results/ddm_rr4_cuda_exact_contest_cuda_20260817_r1/MODAL_REMOTE_RESULT.json`:

```
100 * 0.00029611                = 0.02961100000000000   (seg)
sqrt(10 * 6.88e-06)             = 0.00829457654133109   (pose)
25 * 181161 / 37545489          = 0.12062767380656568   (rate)
                          S     = 0.15853325034789678   <- matches to 17 digits
```

The rounded `final_score: 0.16` in the same file is the field CLAUDE.md says lies. The
recomputation is authority and it agrees. `passed: true`, `returncode: 0`,
`validation_errors: []`, `gpu_t4_match: true`, `n_samples: 600`, `evidence_grade: contest-CUDA`.

### 1.2 The archive byte chain: **CONFIRMED**

Hashed the fired bytes myself:

| object | measured | matches |
|---|---|---|
| `candidate_runtime/archive.zip` | sha `35ac2b9b…9618956`, 181,161 B | `expected_archive_sha256` in the T4 result ✓ |
| its single ZIP member `p` | sha `1a6b40cc…9ba9da47a`, 181,061 B, stored | `RESULT_build.json.member` ✓ |
| ZIP overhead | 100 B | consistent |

`RESULT_build.json` proves the mechanism claim: **7 of 8 parsed sections are byte-identical to
the frontier base** (carrier, compensation, models, hpac, residual, semantic, table_codes) and
only `token_stream` changed (112,110 → 110,512 B). The decode proof closes it:
`RESULT_parseback_v2.json` carries `token_checkpoint.binding.archive_sha256 = 35ac2b9b…`,
`decoded_field_bit_identical: true`, `decoded_token_sha256 = 9ba2e52b…` — which is exactly the
value `RESULT_build.json` pre-registered as `falsifier.decoded_field_target_sha256`. The
falsifier that was `PENDING` at build time **resolved, and it resolved in favour of the claim.**

### 1.3 FINDING 1 — **CUSTODY-GAP (HIGH).** The eval payload was measured and discarded.

`MODAL_REMOTE_RESULT.json` lists 8 returned artifacts. Every one of them reads:

```json
"contest_auth_eval.json": { "embedded_value_type": "str" }
```

That is a **type label, not a payload**. No bytes, no sha, no length. `returned_artifacts/`
does not exist in the run dir. The cause is one branch in
`/Users/adpena/Projects/pact/tools/modal_endpoint_close.py:385-388`:

```python
for raw_name, payload in artifacts.items():
    name = _safe_artifact_name(str(raw_name))
    if not isinstance(payload, bytes):
        artifact_records[name] = {"embedded_value_type": type(payload).__name__}
        continue                                   # <- the payload is dropped here
    destination = output_dir / "returned_artifacts" / name
    atomic_bytes(destination, payload)
```

The function's own docstring says *"Materialize embedded bytes"* — and materializes bytes only.
A `str` payload is silently converted to its type name. This is the exact detector signature in
the CLAUDE.md ALWAYS-KEEP-THE-PAYLOAD banner: *the only persisted artifact is scalars, while a
payload existed in memory.*

**It is a regression with a date.** Census over every `MODAL_REMOTE_RESULT.json` in the repo:

| run dir | n artifacts | record kind | `returned_artifacts/` |
|---|---:|---|---|
| `ddm_f26p_mc36_contest_cpu_20260814` | 5 | FILE-RECORD (sha + bytes) | yes |
| `ddm_f26r_mc36_contest_cpu_20260814` | 9 | FILE-RECORD (sha + bytes) | yes |
| `ddm_rr4_cuda_exact_contest_cuda_20260817_r1` | 8 | **TYPE-ONLY** | **no** |

Same closer, same code path. What changed between 08-14 and 08-17 is that the new canonical fire
tool returns the artifacts as `str`, and the bytes-only branch drops them.

**This is not cosmetic. It blocks the compliance gate on the current frontier row.**
`scripts/pre_submission_compliance_check.py` requires `report.txt` (`:2013`) and
`contest_auth_eval.json` (`:2965`) for `--contest-final`. Both are gone. So is `provenance.json`
(pact/upstream commit) and `inflated_outputs_manifest.json` (inflated-output aggregate sha).

**Same root cause, second surface.** The anchor row's null custody fields are these missing
artifacts. Null-field census over the last 8 accepted anchors:

```
ddm_js2_cp135 … ddm_f26r_mc36  (08-12 .. 08-14)   nulls 2/7
lane_ddm_hv1_ep0634 (08-15)                       nulls 7/7
lane_ddm_rr4 (08-17)                              nulls 7/7
```

The rr4 verdict memo justifies exactly one of the seven nulls (`upstream_snapshot_sha256`, the
standing #836 symlink-hasher refusal, real and precedented). The other six are not drift — they
are the discarded payload. `runtime_tree_sha256` is the sharpest loss: the T4 receipt *has* it
(`7acedb07…`, verified, `validation_errors: []`), and it is precisely the field that would have
caught rr2's own failure mode (staging infidelity — the wrong tree fired).

**Recovery window is open and closing.** Modal `.spawn()` result cache TTL is ~24 h; the harvest
was `2026-08-17T19:10:34Z`. Re-reading `FunctionCall.from_id('fc-01M08HHS64QJNDV7M34E6AG96T').get()`
is a cache read, not a dispatch — **$0 of GPU**. After ~19:10Z on 08-18 the artifacts are gone and
recovering them costs a re-fire ($0.16 against $1.38 of remaining headroom, see FINDING 5).

Sealed fire-order **FO-1** below.

### 1.4 FINDING 2 — **CUSTODY-GAP (MEDIUM).** The receipts inside the fired tree describe a different archive.

`candidate_runtime/GENERATION_RECEIPT.json` is `candidate_id: hv1_base_control`. It declares:

```
archive.bytes  = 182759          archive.sha256 = 80d9c8c6…1420178e
outer_inflate  = fdd2d19a…610e60f1
```

The archive sitting beside it — the one that fired — is **181,161 B, sha `35ac2b9b…`**, and its
`inflate.py` is sha `3ba93237…`. Same for `RECEIVER_PARSEBACK.json`: sha `e79a61ba…`, which is
byte-identical to the hv1_base_control receipt the GENERATION_RECEIPT itself points at, and whose
`candidate_id` is `hv1_base_control`. Both files were inherited from the base and never
regenerated. Neither mentions the fired sha.

**This is a labeling gap, not a proof gap** — and the distinction matters. The genuine proofs
exist one directory up (`RESULT_build.json`, `RESULT_receiver_build.json`,
`RESULT_parseback_v2.json`), all three bound to `35ac2b9b…`. So the chain PROVES the fired bytes
decode to the claimed tokens. But a future agent who opens the runtime tree — the natural place
to look — reads a receipt for a 182,759 B archive and either believes it or has to reconstruct
the real chain from scratch. The producing arm flagged the file as stale and shipped it anyway.

The residual hazard is concrete: the runtime-tree sha is over this directory, so regenerating the
receipts would change `7acedb07…` and break replay against the pinned value. The cure is to write
a `CUSTODY_SUPERSEDED.json` beside them naming the fired sha and pointing at the three real
receipts — additive, outside the hashed set if the tree hash excludes it, and honest either way.

**STORES CONSULTED (Object 1):** `MODAL_REMOTE_RESULT.json` · `ENDPOINT_CLOSURE.{done,receipt}.json` ·
`/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/{RESULT_build,RESULT_encode,RESULT_receiver_build,RESULT_parseback_v2,RESULT_receiver_parseback}.json` ·
`candidate_runtime/{GENERATION_RECEIPT,RECEIVER_PARSEBACK}.json` + the archive bytes themselves ·
`.omx/state/continual_learning_posterior.json` (189 accepted anchors) ·
`.omx/state/canonical_frontier_pointer.json` · `tools/modal_endpoint_close.py` ·
`scripts/pre_submission_compliance_check.py`.

### 1.5 The anchor row: **CONFIRMED on what it carries, CUSTODY-GAP on what it drops**

Field-for-field against the T4 result — `archive_sha256`, `archive_bytes` (181161),
`score_value` (0.15853325034789678), `axis` (cuda), `evidence_tag` (`[contest-CUDA]`),
`hardware_substrate` (linux_x86_64_t4), `n_samples` (600): **all faithful, no drift.** The
pointer's `effective_frontier` correctly applies `min(ours-CPU 0.188044, ours-CUDA 0.158533,
upstream 0.162)` = 0.158533. The dropped fields are FINDING 1.

---

## OBJECT 2 — COMPOSABILITY

### 2.1 Is the −1,598 B recode a standing operator? **CONFIRMED, WITH A CORRECTION.**

**The MECHANISM is standing. The MAGNITUDE is not.** Both halves are load-bearing and the rr4
memo states only the first.

Read at source, `/Users/adpena/Projects/pact/experiments/ddm_rr4_free_corrector_v2.py`:

* `FreeCorrector.__init__` allocates `counts`, `hits`, `phat_q`, `prev1`, `prev2`, `run` **all
  zeros**, `have_prev = False`. There is no fitted state, no trained table, no per-checkpoint
  constant anywhere in the object.
* Every constant is first-principles and round: `KT_ALPHA 0.5`, `U_STEP 0.5`, `U_BINS 64`,
  `MIN_COUNT 32`, `DELTA_CLIP 4.0`, thresholds `2^(-k/2)`. The docstring asserts none was swept
  against the clip; the values are consistent with that (they are the shapes you would write
  down, not the shapes a sweep would return).
* The estimator is **online**: the shift is derived from already-decoded symbols by a fixed rule.
  Nothing is transmitted. That is what makes it free under rule 118, and it is also what makes it
  checkpoint-agnostic.

So: **any hv1-lineage checkpoint should admit the same re-encode.** The operator transfers.

**What does NOT transfer is −1,598 B.** That number is 1.425 % of *this* token stream, and it is
the amount by which the online context model beat the HPAC base probabilities on *ep0634's*
symbol statistics. A different checkpoint has different statistics and therefore a different win.
`RESULT_encode.json` records `warm_contexts: 9613` — a stream-dependent count, computed online,
different on every stream. Quoting −1,598 B as the expected gain on a future checkpoint is the
constants-are-poison move. Quote the OPERATOR; re-measure the MAGNITUDE. The re-measure is cheap:
encode-only, scorer-free, ~48 s (`RESULT_encode.json.encoded.elapsed_seconds = 47.75`).

One residual, low severity: the "none was swept against the clip" claim is an assertion, not a
receipt. `MIN_COUNT 32` and `DELTA_CLIP 4.0` are the two that could hide a tune. They look
generic and I have no evidence against them. Flagged, not charged.

### 2.2 FINDING 3 — **REFUTED (HIGH).** The banked-micro-edit hazard is a registered falsified premise.

The rr4 verdict memo says, verbatim:

> *"Sister hazard: the banked micro-edit offsets (qs2 −4.375e-6 @ +34 B, re1 −1.207e-6 @ 0 B)
> were compiled against the OLD coder — per the cross-regime constant-transfer genus they need
> RECOMPILE against the new stream before any union fire."*

**There is no bank. There is no pending union. Those offsets are already inside the row that
just fired.**

`/Users/adpena/Projects/pact/.omx/research/ddm_bu1_bank_union_compile_20260817.md` establishes it
at the event id, not the pair index: mc36 Variant C's runtime parse-back recovers compensation
pairs `[7, 96, 105, 176, 178, 517, 523]` = (qs2's six minus the measured-harmful 532) ∪ (re1's 96
and 7). The union was built as mc35, repaired as mc36 Variant C, fired on T4, and promoted at net
realized **ΔS −2.068040e-5**. hv1 ep0634 then carried mc36's frames at 3,510 fewer bytes.

I verified the carry arithmetically rather than taking it on cite. mc36 measured
`d_seg 0.00029611 / d_pose 6.88e-6`
(`ddm_mc36_promotion_complete_s_verdict_20260814.md:19-20`). hv1's anchor S is 0.15959729295498598
at 182,759 B, and `100·0.00029611 + √(10·6.88e-6) + 25·182759/37545489` reproduces it to 17
digits. rr4 fired the identical distortion pair. **mc36 → hv1 → rr4 is one unbroken distortion
lineage; the micro-edits rode all the way in.**

Consequence for routing: treating −4.375e-6 and −1.207e-6 as available future gain
**double-counts value the frontier already holds**, and the recompile it prescribes is work on an
object that does not exist.

The timeline is what makes this a finding rather than an old error:

```
09:57:59  bu1 memo commits — union already fired, already won
10:00:54  falsified_premise_registry.jsonl gains the row, registered_by ddm_bu1 via MAIN
14:20:01  rr4 verdict memo commits — restates "banked ... before any union fire"
```

The premise was falsified, written down, and committed **4 h 20 m before** the memo that restated
it. And the same stale premise then propagated into this arm's own charter (Object 2, second
half), so I was dispatched to adjudicate a hazard that had been closed that morning. That is the
charter-recall-is-apparatus-not-volition genus firing on us twice in one day.

### 2.3 FINDING 4 — **CORRECTED (HIGH).** The detector built to catch FINDING 3 cannot fire. Three reasons.

I ran the registry's own matcher against the offending text. It is silent:

```
_lint_falsified_premises(rr4_verdict_memo_text)  ->  []   (no warnings)
_lint_falsified_premises(rv2_charter_object_2)   ->  []   (no warnings)
```

Applying the detector-zeroes-on-the-cure test at source
(`/Users/adpena/Projects/pact/tools/codex_arm_queue.py:1611-1690`):

**(a) U+2212 MINUS SIGN is not normalized.** The matcher normalizes en dash and em dash only:

```python
haystack = text.replace("–", "-").replace("—", "-").lower()
```

The repo writes minus as U+2212 (`−`). The rr4 memo contains 6 of them. So a registered pattern
`qs2 -4.375e-6` (ASCII hyphen) cannot match the memo's `qs2 −4.375e-6`. Measured:

```
'qs2 -4.375e-6' in haystack (current)          -> False
'qs2 -4.375e-6' in haystack (+ U+2212 mapped)  -> True
```

**Census: 16 of the 65 registered `claim_patterns` (25 %) contain an ASCII hyphen adjacent to a
digit.** A quarter of the whole registry is structurally blind to memos written in the repo's own
typographic house style. One `.replace("−", "-")` re-arms all of them.

**(b) The patterns are composite, with the connector baked in.** The registered pattern is
`"qs2 -4.375e-6 + re1 -1.207e-6"`. The memo writes `qs2 −4.375e-6 @ +34 B, re1 −1.207e-6 @ 0 B`.
Even with U+2212 fixed, the composite still misses — a paraphrased connector defeats a substring
match. Decisive test:

```
with U+2212 fix, registered composite  -> NONE (still silent)
with U+2212 fix, atomic 'qs2 -4.375e-6'-> True
with U+2212 fix, atomic 're1 -1.207e-6'-> True
```

**Both cures are required.** Register the smallest distinctive literal, never a phrasing.

**(c) The gate runs at codex-arm spawn only, and codex is walled.** `_lint_falsified_premises`
lives in `tools/codex_arm_queue.py` and fires from `lint_charter_recall_advisories`. MAIN-written
memos never pass through it, and neither do Agent-tool Opus charters like this one. With codex
walled to 08-20, the gate's denominator is approximately zero — a silent instrument reads as a
clean one. This is the vacuity-equals-pass genus: the detector did not *fail*, it never *ran*.

**STORES CONSULTED (Object 2):** `experiments/ddm_rr4_free_corrector_v2.py` ·
`RESULT_encode.json` · `RESULT_build.json` · `.omx/research/ddm_bu1_bank_union_compile_20260817.md` ·
`.omx/research/ddm_mc36_promotion_complete_s_verdict_20260814.md` ·
`.omx/research/ddm_mc35_micro35_union_build_20260814.md` (via bu1) ·
`.omx/research/falsified_premise_registry.jsonl` (65 patterns) · `tools/codex_arm_queue.py:1611-1690` ·
git log for both memos.

---

## OBJECT 3 — NEXT-MOVER RANKING

### 3.0 The exchange rates, re-derived at the live operating point

```
dS/d(d_seg)  = 100.0000
dS/d(d_pose) = 602.8035        = 5 / sqrt(10 * 6.88e-6)   -> pose marginal is 6.028x seg's
dS/d(byte)   = 6.6586e-07

gap to 0.15  = 0.00853325
```

Single-axis sufficiency — the fact that should govern routing:

| axis | best case alone | closes the gap? |
|---|---|---|
| **pose** | d_pose → 0 saves **0.00829458** | **NO — mathematically insufficient, short by 0.00023867** |
| seg | d_seg → 0 saves 0.02961100 | yes; needs only Δd_seg = −8.5333e-5, i.e. **−28.8 %** of d_seg |
| rate | −12,815 B | yes; **−7.07 %** of the archive, ceiling **≤ 168,346 B** |

Equivalences: 1,000 B = 7.80 % of the gap · 1e-5 of d_seg = 11.72 % · 1e-6 of d_pose = 7.06 % ·
1e-5 of d_seg is worth 1,502 B.

### 3.1 FINDING 5 — **CORRECTED (HIGH).** Modal spend is misstated 2.66×; headroom 9.4×.

The rr4 verdict memo closes with *"Modal spend ≈ $7.0/$20."* The measured billing authority,
written the day before by the tool built for exactly this
(`/Users/adpena/Projects/pact/.omx/state/modal_spend_receipts.jsonl`, four agreeing reads over
60 min, `tools/modal_spend_report.py`):

```
total_usd  = 18.61972873   of cap_usd 20.00
headroom   =  1.38027127   cap_consumed_fraction = 0.9309864
```

**$18.62, not $7.0.** Headroom is **$1.38, not ~$13**. At $0.16 a T4 row that is **~8 rows left,
not ~80** — an order of magnitude, and it is the single number that decides whether the next
plan can afford to measure or must reason. The receipt's own `ledger_semantics` warns that the
call-id ledger is a LOWER BOUND (`ledger_sum_usd 1.74`, 55 of 63 calls unpriced) and that billing
is authority; the memo's $7.0 looks like an incremented running tally never rebased on billing
(mc36 on 08-14 quoted $5.85 the same way). The tool that measures it landed 2026-08-16 and was
not consulted 2026-08-17. **An operator call on the cap is OWED before the next paid row.**

### 3.2 FINDING 6 — **CORRECTED (HIGH).** Every route memo argues against a stale gap.

All five routes were argued at the pre-rr4 gap **0.0095973** and a byte bar of **−14,414 B**
(itself already a rebase of rfo2's **−15,157 B**, which was computed against a base three pointer
moves back). Live values are **gap 0.00853325, bar −12,815 B**. The −15,157 B figure is not a
rate *win* — it is a required *cut*, and a stale one. Any route's "miss factor" must be
re-divided by the live bar before it is trusted; a candidate can pass its own stale bar while
being worse than what we ship. The structural cure is already written and should be enforced:
**quote the ceiling (`archive ≤ 168,346 B`), never the delta** — the delta's base is archive size
and rots on every rate move.

### 3.3 FINDING 7 — **CORRECTED (MEDIUM-HIGH).** The aligned-objective claim: arithmetic verified, headline over-claims, vehicle does not transfer by citation.

Verified at source (`.omx/research/ddm_ce1_allocation_ladder_verdict_20260817.md`,
`src/tac/witness_dsl/cw1_semantic_curriculum_levers_20260817.py`, raw payloads under
`/Volumes/APDataStore/pact/ddm_ce1/` and `/Volumes/APDataStore/pact/ddm_jr1/A2_repeat/`):

| number | verdict | what it actually is |
|---|---|---|
| **81.19 %** | VERIFIED-AT-SOURCE | a **closed form**, not a fit — re-derived 81.201 % by integrating `CosineAnnealingLR` over the phase fractions. Its companion `cos(sign g) = 0.20872` IS measured, **n=120 seeded random over all 600 pairs, not a prefix**. "Worst-aligned" means worst of **three** measured objectives, not worst possible. |
| **92.7 %** | VERIFIED-AT-SOURCE, **headline over-claims** | 92.6508 % of the control's **8,654-flip training SURCHARGE**, matched A/B/C off a shared 33,757-flip init. As a share of the seg wall it is **23.8 %** of the init floor / 18.9 % of total error. The memo body says so; the title does not. Cite the body. |
| **13.6×** | VERIFIED-AT-SOURCE, **denominator in the noise** | 8654/636 exactly, a ratio of surcharges. Numerator is 14.3× the measured A/A floor (605 flips, from the byte-identical twin `ddm_lr1/A2`) — solid. Denominator 636 is **1.05× the A/A floor, n=1** — indistinguishable from parity. The 9.4×-of-band *ordering* is solid; the *ratio* is not. |

Two further corrections that matter more than the grading:

* **Rows #1089 / #1091 do not exist in the repo ledger.** `.omx/state/canonical_task_status.jsonl`
  (563 rows via the strict loader) has no such ids; its numeric prefixes stop at 1085. They are
  harness TaskList ids — the m89 split-ledger. No arm can resolve them. Cite content, never a bare id.
* **Vehicle:** the measurements are on the **semantic renderer** (38 tensors, 66,339 params,
  FiLM), which `cw1_semantic_curriculum_levers_20260817.py:43-46` states is a different
  architecture from the hv1 checkpoint (37 tensors, 39,375 params, no FiLM). So it does **not**
  transfer to the frontier by citation.

But do not read that caveat as "irrelevant." The shipping archive's own base receipt reports
`semantic_tensor_denominator: 38`, and `semantic.br` is **34,763 B = 19.19 %** of the 181,161 B
archive. The semantic renderer is a **shipping section that hv1 did not train** (ep0634 holds the
cl1 token model — b2e established this when it caught its charter pinning the wrong object). So
the honest read is: this is a real lever on a real shipping section, whose reach to the frontier
is **UNMEASURED** and requires a byte-closed re-pack. That is a strong open route, not a closed one.

The strongest thing on the board is independent of the disputed 636: **EF3000 already fired.**
`--ce-fraction 0.0 --softplus-fraction 0.0`, 3,000 steps, `best_step = 3000`,
`improved_over_init = True`, endpoint **−2,286 flips BELOW init** — where ten prior runs, sweeping
three decades of learning rate, could not descend at all. The curriculum shape is a *fraction of
the run*, hence invariant to exactly the axis those ten runs swept. They explored the axis that
did not matter.

### 3.4 The ranked table

Ranked by **probability of producing a lower exact score soonest**, not by elegance.

| # | route | strongest MEASURED receipt | expected ΔS (arithmetic shown) | cost | falsifier |
|---|---|---|---|---|---|
| **1** | **hg1 arm_b margin-hinge** — LIVE now (pid 5749, 40 m elapsed, lands ~23:20Z) | target `m_safe = 0.0391803` = 2×`delta_R` 0.0195902, p95 of uint8-induced margin perturbation, `reports/delta_R_noise_floor.json`; trainer default 1.0 is **25.5×** too big, so **97.65 %** of hinge gradient pulls already-correct pixels | seg route. Closing alone needs **28.8 %** seg recovery (= 0.00853325/0.029611) — its own §4.1 says 32.41 % at the stale gap, so **the case is stronger than the memo claims** | $0, already spent | pre-registered: realized recovery <25 % of the re-derived ladder ⇒ report honestly, **do not re-tune into a result**. Second: hinge active fraction ≈0 ⇒ INERT and the run is confounded |
| **2** | **ce1/cw1 aligned objective on the semantic section** | EF3000 `improved_over_init=True`, **−2,286 flips below init** after ten runs at `False`; `.omx/research/ddm_ef3000_first_descent_verdict_20260817.md` | unpriced on the frontier axis. Section is 19.19 % of archive; effect reaches S only through a **byte-closed re-pack**, which has never been run | 14 min $0 for the decisive EF0 seed repeat; re-pack is a build | **the EF0 repeat**: 636 is n=1 at 1.05× the A/A floor. Lands near 8,654 ⇒ the ladder is noise. Lands near 636 ⇒ it hardens |
| **3** | **ra2 CPR1 inner coder + ra1 basis_scales** | ~278 B measured (263 B raw + 48 B, ~230 B realized), **zero pose risk** | 25·278/37545489 = **−1.851e-4** = **2.2 %** of the gap | **$0, local** | blocked only by a self-imposed gate ("fire only when a ≥2 KB rung is also in flight") that qw1 measured **VACUOUS** — by its own gate it can never fire. Retire the gate |
| 4 | wd3 width distillation (D56/F64, #1070) | n120 seeded stratified, non-prefix: W0_warm 0.0012388 vs D56 0.0024939 (2.01×), F64 0.0029885 (2.41×) | admission stop is **PROJECTED** (~6× over a byte-derived bar of Δd_seg ≈1.07e-4), not measured | ~45 min Metal for the n600 row that was trajectory-stopped | reactivation already written: any arm projecting within ~1.5× of its bar fires n600 first. Two arms measured **one defect twice** (fresh-init regime, not capacity form) — a third W96 arm would measure it a third time |
| 5 | mp2/ra2/rfo2 post-hoc byte surgery | rank-4 saves **14,709 B** measured exact coded bytes = 114.8 % of the live bar — **the rate side passes outright** | killed on distortion: ra2crr minimized over the **entire sphere** (292/292 descents within 1 %); cheapest droppable direction costs Δd_pose 3.2824e-3 vs break-even ~1.05e-6 = **1,498–3,139× miss**. mp2's own three candidates: −823 B but d_pose 4.96× ⇒ NET **+4.67e-2** | — | family CLOSED at family scope by six treatments. Reopening needs **joint descent** (carrier retrained with pose in the loop), not another post-hoc cut |
| 6 | ns1/b2e train-for-editability burn-2 | `REGIME_THESIS_INSTANCE_REFUTED`, n600 600/600; required ≥50× collapse, measured **0.945 / 1.059 / 0.748** | the training window did nothing: Δd_seg +1.4e-7, Δd_pose +7.9e-7, **Δbytes +330**, ΔS_adv +0.000336 | correct object identified (`tools/train_ddm_cl1_hpac_capacity.py`; semantic/carrier with pose in-loop), unbuilt, multi-day Metal | the edit-replay harness is reusable at $0 against any future burn checkpoint |
| 7 | pose | the **uncapped exact GN solve converges to d_pose 1.285917e-05 — 1.87× WORSE than the shipping decode's 6.88e-06** (`ddm_pv1_pose_floor_and_admission_bar_20260816.md`, n=50 from raw shards) | **arithmetically insufficient alone** (§3.0): d_pose→0 saves 0.00829458 < gap 0.00853325. And the available movement is measured **up, not down** | — | CLOSED as a post-hoc axis. Reopening requires a different object: joint descent with pose in the training loop |

Two live hazards inside the table, both flagged by receipts and neither yet acted on:

* **hg1's binding constraint may be pose, not seg.** rn1 measured the pose marginal at 6.03× seg's
  at this operating point (I re-derived 6.028×), and the advisory instrument is **18.2× optimistic
  on pose** while sound to 2.5 % on seg. A hinge that recovers 1.5× of the gap on seg can still
  lose. Arm B owes a measured pose leg that **may not** be quoted from the advisory instrument.
* **ps1u's sealed r2 admission rule has a false-admit window.** pv1 §4 finds a T4 row landing in
  `[6.245822e-06, 6.251199e-06)` **ADMITS on the sealed rule while RAISING S**. Related: the
  16-digit `6.885642960696714e-06` pinned in 13+ modules is the **CP135 base at 186,252 B**,
  carried onto a different archive — a borrowed constant under a live admission bar.

**STORES CONSULTED (Object 3):** `.omx/state/canonical_frontier_pointer.json` ·
`.omx/state/modal_spend_receipts.jsonl` + `tools/modal_spend_report.py` ·
`ddm_ce1_allocation_ladder_verdict_20260817.md` · `ddm_cw1_corrected_window_20260817.md` ·
`src/tac/witness_dsl/cw1_semantic_curriculum_levers_20260817.py` ·
`ddm_ef3000_first_descent_verdict_20260817.md` · `ddm_rg1b_band_objective_build_20260816.md` +
`/Volumes/APDataStore/pact/ddm_rg1/grad_cosine/RG1B_WEIGHT_SPACE_COSINE.json` ·
`ddm_hg1_ring0_margin_hinge_20260816.md` + `reports/delta_R_noise_floor.json` ·
`ddm_wd3_n120_family_disposition_20260816.md` · `ddm_ra1_carrier_rank_refit_preproof_20260816.md` ·
`ddm_ra2crr_priced_pose_null_and_pool_census_20260816.md` ·
`ddm_ra2_charter_stale_family_closed_and_lossless_axis_20260817.md` ·
`ddm_mp2_relay_base_advisory_row_20260815.md` · `ddm_b2e_edit_replay_admission_verdict_20260816.md` ·
`ddm_b2e_landing_and_charter_repin_20260816.md` · `ddm_pv1_pose_floor_and_admission_bar_20260816.md` ·
`ddm_rfo2_fresh_eyes_gestalt_synergy_20260815.md` · `ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md` ·
`ddm_gx1_gap_closure_composition_table_20260816.md` · live `ps -p 5749`.

---

## OBJECT 4 — THE CONTRADICTION SET

### 4.1 FINDING 8 — **CORRECTED (MEDIUM).** This arm's own charter inverted the polarity.

The charter states *"`boundary_gated_token_code_width` 'wired' (tc1's claim) or 'DECLARED-ONLY' (oq1's claim)."*
It is the reverse. Verbatim:

* `ddm_oq1_orphan_queue_drain_20260817.md:229-231` — *"is **wired**, registered in the TR1
  trainer's own lever table at `experiments/train_tr1_partition_renderer_mlx.py:248`"*
* commit `7763583f6d` (tc1, 14:11) — *"boundary-gated code-width is DECLARED-ONLY not wired
  (verified at source)"*

Same genus as FINDING 3: a charter written from a headline rather than the source. Recorded
against myself as much as anyone — I would have inherited the inversion had the retrieval been
shallower.

### 4.2 oq1 vs the 51-rows arm on #882: **BOTH PARTIALLY.** Frontier-neutral.

A row numbered 882 **is** present in `.omx/state/canonical_task_status.jsonl` — 4 events at both
HEAD and working tree, served by the strict loader. oq1's `51 absent / 1 present` arithmetic
reproduces exactly (214 distinct task_ids, 44 numeric, spanning 383–1029; intersecting the 52
charter ids yields exactly `882`).

But the repo's #882 is a **different object**: registered 2026-08-03 by `ddm_pj2`,
`owner: ddm_pj2`, `status: completed`, with commit shas and a closing note. The charter's
population was *"unowned pending task rows from the 08-16/17 audit waves."* Not unowned, not
pending, not from those waves. oq1's own disposition reason — *"owner must append a
CLOSING-ARTIFACT or typed blocker"* — is contradicted by the row it cites.

**FINDING 9 — CORRECTED (MEDIUM).** Root cause: **bare-integer join across two identifier spaces**
(harness TaskList vs the repo ledger, overlapping integer ranges), with no content, owner, date,
or status check — which oq1's own charter explicitly forbade (*"never bare ids"*). The 51-rows arm
has landed nothing (no memo, no commit, no charter file), so its reasoning is not inspectable and
I do not infer it; if it reports the charter's #882 absent *as a matching object*, it is right in
substance.

Routing impact: **none.** oq1's own live-vehicle verdict is *"LIVE-FRONTIER (hv1/HPAC): none"*,
and #882 is a completed 08-03 pose row already re-priced to 0.0000000 with a "DO NOT SUM THESE
THREE" guard on the ledger row itself.

### 4.3 oq1 vs tc1 on 22-vs-35: **BOTH RIGHT — different scopes.** Frontier-neutral.

Measured directly from `.omx/research/ddm_oq1_drain_dispositions_20260817.json` (437 rows):
`vehicle_scope_counts.live_achiever_tr1 = 35`; of those, `oq1_disposition == QUEUED` = **22**
(the remainder 5 DEFERRED / 5 FIRED / 3 FOLDED), and exactly those 22 carry a non-null
`oq1_tr1_phase` (COMPOSITION 12, PRE-seeding 4, POST-solving 4, DURING-conditioning 2). Both
numbers are inside oq1's own JSON; the defect is that the memo never states their relation, so a
reader of the headline and a reader of §3 diverge. Neither count moves a row across tc1's §9.5
split (4 PRE-seeding d_seg-touching / 2 pose-only / ~29 rate rows DOMINATED).

Secondary: tc1 §9.2 quotes oq1 as *"298 rows (68.2 %) dead mass"*; the landed oq1 says **296
(67.7 %)** — tc1 read oq1 commit `02f3f1851b` (14:02), oq1 revised in `f9cf240434` (14:08), tc1
committed 14:11. A **9-minute** staleness window. The m37 freshness-at-consumption law, live.

### 4.4 FINDING 10 — **REFUTED (HIGH, routing-relevant).** `boundary_gated_token_code_width` is not wired. tc1 is right.

Repo-wide, `rg -n "boundary_gated_token_code_width"` returns **exactly one code hit**:
`experiments/train_tr1_partition_renderer_mlx.py:248`. It sits inside `DUTY_TO_MEASURE`, a
string-valued documentation tuple whose own preamble at `:186-188` reads *"named levers DESIGNED
here but NOT half-wired — each carries its receipt and its activation state (never-fired)."* The
entry's `state` is literally `"never-fired"`.

And `DUTY_TO_MEASURE` itself has **zero readers**: `rg -n "DUTY_TO_MEASURE"` returns the
definition at `:189` and two docstring mentions at `:56`/`:62`. Nothing consumes it. So the grade
is **weaker than "parsed flag with no consumer"** — there is no argparse flag at all. The only
real flag is `--code-width` (`:3001`, `choices=(2,4,6)`), a **uniform global scalar** threaded to
`TR1Config.code_width` and read at the coder as a bare `c = cfg.code_width` over all kept cells
(`:5200`, `:5225`). There is no interior/boundary cell partition anywhere in the trainer.

**This is the one contradiction that moves routing.** oq1 §6 called it *"the cleanest single item
in the TR1 handoff"* — a registered lever, stated falsifier, zero dollars, never fired — and FO-3
routed tc1 to start there. It is dead on two independent counts: (1) the "$0 gate" premise is
false — adopting requires building variable-width tokens *and* a coder that prices them, a build
budgeted as a read; (2) its own adoption rule (*"adopt iff ≥15 % token-stream saving vs uniform
c"*) is a pure **rate** criterion, and tc1 measured that a **free archive** still leaves the TR1
class at 2.49× the frontier — the gate could pass at 100 % and the class verdict would not move.
Compounding it, oq1 §8a logs *"confirmed wired in the TR1 trainer"* — a verification claim for a
check that was not performed.

**tc1's re-route to the 4 PRE-seeding candidates stands and should be honored.**

**STORES CONSULTED (Object 4):** `.omx/state/canonical_task_status.jsonl` (HEAD + working tree,
strict loader) · `ddm_oq1_orphan_queue_drain_20260817.md` + `ddm_oq1_drain_dispositions_20260817.json` ·
`.omx/research/charters/ddm_oq1_orphan_queue_drain_20260817.md` ·
`ddm_tc1_tr1_lifecycle_spec_20260817.md` · `experiments/train_tr1_partition_renderer_mlx.py`
(`:186-253`, `:3001`, `:4195`, `:5200`, `:5225`) · `ddm_hv2_task_rows_pending_ledger_unblock_20260816.json` ·
`ddm_pu1_pose_underpricing_and_tail_20260803.md` · `.omx/state/main_hot_state.md` · git log.

---

## SEALED FIRE-ORDERS FOR MAIN

No launches were made by this arm. These are sealed, not fired.

**FO-1 — HIGHEST EV. Recover the rr4 eval payload. Deadline ~2026-08-18T19:10Z. $0 GPU.**
The Modal result cache still holds all 8 discarded artifacts. Re-read the completed call and
materialize them:

```
FunctionCall.from_id('fc-01M08HHS64QJNDV7M34E6AG96T').get()
```

This is a cache read on a finished call — no dispatch, no compute charge, no draw on the $1.38.
Write each artifact to
`experiments/results/ddm_rr4_cuda_exact_contest_cuda_20260817_r1/returned_artifacts/` with sha +
byte count, then backfill the anchor's `runtime_tree_sha256` (`7acedb07…`, already verified by the
eval), `inflate_script_sha256` (`e1b3df4d…`), `inflated_output_manifest_sha256`, `pact_commit`
and `upstream_commit` from `provenance.json`. **Until this runs, the frontier row cannot pass
`pre_submission_compliance_check.py --contest-final`** — it requires `report.txt` and
`contest_auth_eval.json`, both currently discarded. After the window closes the same recovery
costs a re-fire ($0.16 of $1.38). Do this before anything else in this list.

**FO-2 — the permanent fix for FO-1's cause (two landings, per the bugs-must-self-protect rule).**
(a) `tools/modal_endpoint_close.py:385-388`: encode `str` payloads to UTF-8 and write them through
the same `atomic_bytes` + `file_record` path as `bytes`, so both branches persist sha + length.
(b) A gate that refuses a closure whose `materialized_artifacts` contains any `embedded_value_type`
record while the result declared artifacts — the scalar-only signature, caught at the closer.

**FO-3 — re-arm the falsified-premise detector (three one-line cures, $0).**
(a) add `.replace("−", "-")` to the `haystack` normalizer at `tools/codex_arm_queue.py:1639`
— re-arms 16 of 65 patterns (25 %); (b) split composite `claim_patterns` into atomic distinctive
literals (`qs2 -4.375e-6`, `re1 -1.207e-6`), and make that the registration convention;
(c) run the lint over new `.omx/research/*.md` at commit time, not only at codex-arm spawn — the
gate's current denominator is ~zero while codex is walled. Regression test: the rr4 verdict memo
text must produce ≥1 warning after the fix. It produces 0 today.

**FO-4 — correct the record, additively (APPEND-ONLY, no body mutation).**
Append a correction banner to `ddm_rr4_t4_verdict_pointer_move_20260817.md` for (i) the qs2/re1
bank — already fired, already inside this row, quoting it as future gain double-counts; (ii)
Modal spend $18.62/$20 measured, not $7.0, headroom $1.38 ≈ 8 T4 rows; (iii) composability —
operator standing, magnitude not. And write `CUSTODY_SUPERSEDED.json` into
`/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime/` naming the fired sha
`35ac2b9b…` and pointing at the three real receipts, without regenerating the inherited files
(regenerating them would break the pinned runtime-tree sha `7acedb07…`).

**FO-5 — the cheapest live byte, $0, local.** Retire the vacuous gate on the ra2 CPR1 inner-coder
row (qw1 measured that by its own condition it can never fire) and land the ~278 B / ΔS −1.851e-4
rung. 2.2 % of the gap, zero pose risk. Small, but it is the only measured, pose-safe, unfired
byte in the corpus, and orphaning it is the failure this program keeps paying for.

**FO-6 — the decisive $0 experiment on the #2 route.** One EF0 seed repeat (~14 min, local).
The 13.6× ladder's denominator is n=1 sitting at 1.05× the measured A/A floor. One repeat converts
the strongest open lever from suggestive to hard, or kills it. Pair it with a matched-cadence
control (rerun A2 at `eval_every=25`) to remove the last asymmetry by measurement rather than
argument.

**Owed to the operator, not fireable by an arm:** a call on the Modal cap. $1.38 of $20 remains.

---

## FINDINGS SUMMARY

| # | object | verdict | severity | new this round? |
|---|---|---|---|---|
| 1 | 1 | CUSTODY-GAP — 8 eval artifacts measured and discarded; blocks `--contest-final`; ~24 h recovery window | HIGH | yes |
| 2 | 1 | CUSTODY-GAP — in-tree receipts describe the base archive, not the fired one (labeling, not proof) | MED | yes |
| 3 | 2 | REFUTED — "banked micro-edits need recompile" is a registered falsified premise; they already shipped | HIGH | yes (new violation of a same-day registration) |
| 4 | 2 | CORRECTED — the detector for #3 cannot fire: U+2212 unnormalized (25 % of registry), composite patterns, spawn-only path | HIGH | yes |
| 5 | 3 | CORRECTED — Modal spend $18.62/$20 measured vs $7.0 quoted; headroom $1.38 ≈ 8 rows, not ~80 | HIGH | yes |
| 6 | 3 | CORRECTED — every route memo argues against the stale gap 0.0095973 / bar −14,414 B; live is 0.00853325 / −12,815 B | HIGH | extends fb1/gx1 |
| 7 | 3 | CORRECTED — 92.7 % is of the training surcharge not the seg wall; 13.6× denominator n=1 at 1.05× noise; #1089/#1091 absent from the repo ledger | MED-HIGH | yes |
| 8 | 4 | CORRECTED — this arm's charter inverted the oq1/tc1 polarity | MED | yes |
| 9 | 4 | CORRECTED — oq1 joined #882 on a bare integer across two ledgers, against its own charter | MED | yes |
| 10 | 4 | REFUTED — `boundary_gated_token_code_width` is DECLARED-ONLY with zero readers; oq1 routed the next unit to a non-existent $0 gate on a dominated term | HIGH (routing) | ratifies tc1 |

**Round 1 of the 3-clean-pass counter: 10 findings. Counter 0/3.** Round 2 must re-derive, not
re-read, and must open with FO-1's outcome.

Nine of ten findings are the same shape: **a number or a claim survived past the artifact that
falsified it.** Three separate detectors existed for exactly that (`falsified_premise_registry`,
`modal_spend_report`, the closure manifest), all three landed within the last 48 hours, and none
of them fired — one because of a character encoding, one because nobody called it, one because it
refused and was overridden by prose. The apparatus is not missing. It is **built and unconsumed**,
which reads identically to clean.

---

## WHAT I DID NOT DO

No launches. No Modal. No scorer runs. No `upstream/` edits. This unit produced no exact row and
**did not move the pointer** — it is a review, and that is the correct outcome for a review. The
next lower score comes from route 1 (hg1, landing tonight) or route 2 (the semantic-section
aligned objective), not from this memo.

**Own-vehicle frontier: S 0.15853325034789678 @ 181,161 B [contest-CUDA T4, n600].
Unmoved by this unit.**
