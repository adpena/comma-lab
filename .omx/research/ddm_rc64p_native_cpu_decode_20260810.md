# ddm_rc64p native CPU decode

**Status:** COMPLETE for the chartered scorer-free host arm. Route A is
bit-exact but refused as an entropy-speed cure. Route B is a receiver-closed
4-byte rate win. Contest-CPU authority and an exact score on the Route-B
archive were not measured.

## Result first

All timing rows below are full n600, uninterrupted, scorer-free observations on
this Mac. Every row decoded all 117,964,800 symbols to SHA-256
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`
and passed its terminal-state check. There is one run per cell and no measured
noise floor; cell order showed material host drift, so the table does not
support a stable cross-cell speed ranking.

| token wire / HPAC runtime | threads | wall seconds | process seconds | entropy-call seconds | entropy fraction |
|---|---:|---:|---:|---:|---:|
| lc2 constriction ANS control | 1 | 1,383.084 | 1,280.543 | 1.113 | 0.0805% |
| lc2 native C ANS, identical bytes | 1 | 1,477.077 | 1,282.000 | 1.783 | 0.1207% |
| Route-B RC64, settled sparse HPAC | 1 | 1,461.825 | 1,370.129 | 3.066 | 0.2097% |
| Route-B RC64, cached-plan HPAC | 1 | 1,437.420 | 1,345.842 | 3.003 | 0.2089% |
| lc2 constriction ANS control | 4 | 635.936 | 1,565.197 | 1.111 | 0.1746% |
| lc2 native C ANS, identical bytes | 4 | 647.083 | 1,581.896 | 1.793 | 0.2771% |
| Route-B RC64, settled sparse HPAC | 4 | 812.561 | 1,747.975 | 3.128 | 0.3850% |
| Route-B RC64, cached-plan HPAC | 4 | 716.937 | 1,677.427 | 2.900 | 0.4046% |

**Mechanism conclusion:** #998's label “constriction-Python loop” was wrong for
this receiver. Direct entropy recovery takes only 1.11–3.13 seconds. More than
99.5% of every full token cell is outside the entropy coder, principally the
per-group causal sparse integer-HPAC probability generation. Route A's
single-thread falsifier (>900 seconds) fired exactly as pre-registered.

**Route A verdict:** `refuse_native_entropy_as_cpu_cure`,
`verdict_scope=INSTANCE(lc2 receiver on this host)`. The original C decoder is
correct: pinned constriction 0.5.0 and native output matched on a 10,003-symbol
golden vector, including the midstream ANS snapshot, and on all n600 symbols.
It did not produce a measured wall-clock win in either thread cell.

**Route B verdict:** `adopt_rate_only_receiver_closed_candidate`. The retained
tagged/aligned RC64 field is **114,524 B**, SHA-256
`df52c86682363073bdaf2d654b6c12459063ee29a01d4881f91e58e05ad08ce4`,
versus lc2 ANS at 114,528 B. The deterministic archive is **187,222 B**, SHA-256
`b3365410a423fa6ae4d53e9a86fc2bd38bc59793ea2b437fc161bdcca11712b0`,
four bytes below lc2. Literal `inflate.sh` parse-back retained 3,662,409,600 raw
bytes at SHA-256
`a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353`
in 265.674 seconds using a proven full-token cache. Thus the rate win is real;
it is not evidence of a stable decode-speed win.

**HPAC addendum verdict:** caching immutable rounded weights, compacted kernels,
exponent powers, and conv-a gather indices was bit-identical on all 983,040
frame-0 logits and all n600 decoded symbols. It did not win the measured timing
gate. Refuse this exact implementation with `verdict_scope=INSTANCE`; do not
generalize the negative to native C/Rust HPAC lowering, direct gathered-one-hot
construction, or cross-context parallel restructuring, which remain untested.

## Budget interpretation

Every clean 1-thread and 4-thread token cell happened to fit the charter's
1,500-second token-margin gate on this Mac. For Route B, adding the separately
measured 4-thread token cell (812.561 seconds) and cached-token literal render
(265.674 seconds) gives **1,078.235 seconds**, a **721.765-second** margin to
1,800. This is a component-sum projection, not one uncached contest run. It is
not contest-CPU authority, and the visible timing drift makes the margin a host
signal rather than a promotion receipt.

The exact lc2 score anchor is 0.16959899569230852 at 187,226 bytes. Raw identity
makes a rate-only Route-B projection well-defined:

`0.16959899569230852 - 25*4/37_545_489 = 0.16959633225649604`.

That number is **NOT measured** because `upstream/evaluate.py` was not run on
the Route-B archive. The canonical pointer therefore does not move in this arm.

## Native runtime and payload audits

- `ans_backend.c` is original generic receiver code for constriction's public
  24-bit, five-class ANS grammar. It contains no learned/video-derived values.
- Route B borrows PR135's granted `rc64_backend.c` verbatim (12,222 B, SHA-256
  `5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6`).
  This arm adds only ctypes glue, `R6D1`/`R6C1` framing, and snapshot/resume.
  The RC64 recurrence is not claimed as original work.
- The lc2 ANS format, temporal-reversion model, retained-code corpus, inflate
  line, and Rust audit patterns are Pact substrate. The underlying
  semantic/pose/HPAC vehicle remains the granted PR130 lineage; this is not an
  original-vehicle claim.
- `archive_payload_manifest.json`, `binary_source_audit.md`,
  `embedded_constants_audit.txt`, `rebuild_instructions.md`, and
  `python_reference_equivalence_test.py` are the audit bundle. Both native
  sources report `VIDEO_DERIVED_CONSTANTS=NONE`; learned HPAC parameters remain
  counted inside `archive.zip`.
- A fresh build/load smoke using `/opt/homebrew/bin/python3` (no constriction)
  compiled both dylibs in 0.591 seconds. It checked `cc`, ANS precision/alphabet,
  and RC64 total frequency. Required mode has typed exits and cannot silently
  fall back for an explicit RC64 wire.

## Retention, reproducibility, and parse-back

Control metadata and counted payloads live at
`/Volumes/VertigoDataTier/pact/ddm_rc64p_20260810/`. Bulky decoded fields,
caches, and raw outputs live at
`/Volumes/APDataStore/pact/ddm_rc64p_20260810/`. Principal receipts:

- `receipts/prepare.json`: exact source pins plus byte-identical repeat builds
  for ANS (`231b5bf5…`, 33,936 B) and RC64 (`4f4b72a8…`, 34,688 B).
- `receipts/golden.json`: pinned Python/native ANS parity.
- `route_b/encode_receipt.json`: n600 RC64 encode from the deep-hashed DT1
  retained corpus; distinct encoder checkpoints every 25 frames; archive and
  repeat archive byte-identical.
- `runs/*/timing_receipt.json`: full per-cell timings, terminal state, exact
  decoded field, cache, and progress checkpoint.
- `receipts/bootstrap_smoke.json`: fresh native compiler/load proof.
- `receipts/route_b_end_to_end.json`: literal Route-B raw parse-back.
- `receipts/final_summary.json`: machine-readable final verdict and boundaries.

The first RC64 1-thread correctness run encountered an APDataStore open stall
while writing growing 10-frame checkpoints. It was interrupted and resumed
from the retained frame-370/380 state after copying the checkpoint to the
preferred Vertigo tier. Its decoded field and receiver cache receipt are valid
correctness evidence, but its continuation wall was excluded. The later
`rc64_clean_t1` and `rc64_clean_t4` cells started from frame 0 and completed
uninterrupted on Vertigo checkpoints. Nothing was deleted; both the stalled
checkpoint copies and all complete payloads remain retained.

## RECALL EVIDENCE

Full corpus recall ran before implementation with:

`tools/corpus_query.py --stores research,equations,memory,dag,council,tasks,docs --top 30 --json 'lc2 native ANS decoder constriction RC64 token decode wallclock #998'`

Stores consulted were research 8,339, equations 878, memory 2,102, DAG 915,
council 297, tasks 531, and docs 96. The canonical equations registry was also
listed with `tools/list_canonical_equations.py --json`; the relevant registered
coder equation was `pr95_family_l30_range_arithmetic_coding_categorical_v1`.
Content searches covered `.omx/research/`, canonical research index/DAG FEED
surfaces, runtime-rs audit patterns, design/spec docs, hot state, and the task
ledger.

Findings beyond the charter's named seeds changed the plan:

1. `ddm_dt1_ans_decode_wallclock_gate_20260809.md` had already measured that
   entropy coding itself was about 1.5 seconds while shared model work was
   96.56% of its receiver. That forced explicit entropy-call instrumentation
   and prevented a false “native decoder killed 1,777 seconds” claim.
2. The landed cp135 store measured RC64 beating same-state ANS by 6 bytes on
   control and 9 bytes on hp3. That fired conditional Route B only after Route
   A's falsifier, while preserving the lc2 byte race as empirical.
3. DT1 retained exact n600 symbols/logit codes, and TM1 supplied the exact lc2
   temporal-reversion transform. Those retained payloads made Route-B encoding
   resumable and avoided regenerating/throwing away HPAC outputs.
4. No pre-existing native lowering of this exact sparse integer-HPAC receiver
   was found in the searched runtime-rs/research/task scopes. The tested
   cached-plan implementation therefore remained an original narrow apparatus
   change, not a rediscovery claim.

## Boundaries and dispositions

**MEASURED:** full-n600 symbol identity; terminal coder state; old/native/RC64
1-thread and 4-thread host timings; direct entropy time; Route-B bytes; fresh
native bootstrap; deterministic archive repeat; literal raw parse-back.

**NOT measured:** contest-CPU runtime; one literal uncached full-run total;
repeat/noise floor for timing cells; exact evaluator score on Route B; Seg/Pose
components on Route B; a true native C/Rust HPAC kernel; scorer or Modal row.

- **FOLDED:** further entropy-coder port work as a CPU cure. Consumer:
  `receipts/final_summary.json`. Fire condition to reopen: a different receiver
  where direct entropy timing is a material wall fraction.
- **QUEUED:** exact Route-B contest-CPU replay. Owner: MAIN. Consumer:
  contest-CPU receipt / CPU feasibility ledger. Fire trigger: MAIN claims the
  single-flight lane and authorizes the Modal spend after this landing.
- **QUEUED:** exact evaluator replay of archive SHA `b3365410…`. Owner: MAIN.
  Consumer: canonical frontier pointer. Fire trigger: scorer lane is free and
  the exact archive/runtime bundle is admitted; do not promote the projected
  score before then.
- **QUEUED:** native C/Rust lowering of the true sparse-HPAC hot path. Owner:
  a successor runtime arm. Consumer: this store's timing matrix and runtime
  audit bundle. Fire trigger: contest CPU misses 1,800 seconds, margin is below
  the operator threshold, or MAIN explicitly prioritizes more CPU headroom.

Own-vehicle frontier remains **lc2 S 0.16959899569230852 @ 187,226 B
[contest-CUDA T4, n600]**. The effective frontier is cp135 composed, but it is
not this arm's own-vehicle row; Route B's 0.16959633225649604 is projection-only.

## ADDENDUM 1 (MAIN, 2026-08-11T00:15Z) — contest-CPU authority receipt: inflate DOES NOT fit 1,800 s as shipped

MAIN fired the queued contest-CPU replay (Modal call `fc-01KZPZSXSP2RZ0XSW0FHHEQVZ0`, lane
`lane_ddm_rc64p_lc2_contest_cpu_authority_20260810`, archive f154f0ab… 187,226 B, locked-env
proven). Result: the shipped single-thread `inflate.sh` was killed at exactly the 1,800 s
contest cap. Measured on Modal contest-CPU hardware `[contest-CPU-class wall; no score]`:
tokens 2,156.8 s + render 250.5 s ≈ 2,407 s total (the orphaned child completed the raw after
the harness kill; local byte-identity sha a18eb42a… gives confidence in output identity, not
re-verified on host). This is 2.8× the local M5 shipped-path wall (856.0 s, LC2 READY receipt).
The eval step never ran; no CPU score exists. Cost ≈ 40 min CPU, < $1 of #381.

Consequences: (1) this memo's component-sum margin (721.8 s on this Mac) does NOT transfer to
contest-CPU hardware — host drift is ~2.8–3.0× on the HPAC term; (2) the QUEUED native/threading
successor's fire trigger ("contest CPU misses 1,800 seconds") has FIRED with an exact receipt —
projected cure: 4-thread decode (2.0× measured locally, t1 1,437.4→t4 716.9) puts Modal-class
decode ≈ 1,329 s < 1,800; native C/Rust HPAC adds headroom beyond that; (3) per the
first-attempt-wall-clock law (2026-08-10), this negative is IMPLEMENTATION-scoped — the CPU
axis stays OPEN pending the threaded/native inflate; (4) the CPU-vs-CUDA SCORE delta remains
unmeasured — a diagnostic re-dispatch with `--inflate-timeout 7200` (tool-suggested dev mode,
score_claim=false, no-promotion) buys that delta on identical bytes without claiming compliance.
Receipts: `.omx/research/ddm_rc64p_native_cpu_decode_20260810_modal/contest_cpu/`.
